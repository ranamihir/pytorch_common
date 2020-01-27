import logging

from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from vrdscommon.dsp_pipeline import DspPipeline
from vrdscommon.model_metric_tracker import ModelType
from vrdscommon import dsprunner
from vrdscommon.dsprunner import TrainingResult
from .models import create_model, create_estimator
import numpy as np
import pandas as pd


class ModelTrainer(dsprunner.ModelTrainer):

    def __init__(self, config, args, job_reporter, dbrunner):
        super().__init__()
        self.config = config
        self.args = args
        self.job_reporter = job_reporter
        self._target_column = config.target
        self.dbrunner = dbrunner

    def train(self, feature_matrix, model_name):
        model = create_model(model_name, self.config)
        X_train, y_train, X_test, y_test = self.split(feature_matrix, test_split=0.7, target=self._target_column)
        model.fit(X_train, y_train)
        p_train = model.predict_proba(X_train)
        p_test = model.predict_proba(X_test)

        # Parameter search
        self.parameter_search(feature_matrix, model_name)

        # Save model metrics
        self.job_reporter.record_model_metrics(ModelType(self.config.model_type), y_train, p_train, p_test, y_test)

        # Calculate additional model metrics
        metric_data = self.evaluate(y_train, p_train, y_test, p_test)

        # Save additional evaluation metrics
        self.job_reporter.metrics('evaluation', metric_data)
        # use this to validate metric values
        self.job_reporter.metric('data_size', name='training_size', value=len(X_train),
                                 min_val=self.config.min_rows, max_val=self.config.max_rows)

        # save the model, or models, which will go to artifacts automatically on save
        training_result = TrainingResult()
        training_result.add_model('model', model)

        """
        # example of adding data needed for prediction - do NOT include PIIs / User Personal Information to be saved in artifacts!

        geo_data = [{'geo_id': '186338', 'feature_multiplier': 0.22},
                    {'geo_id': '60763', 'feature_multiplier': 0.01},
                    {'geo_id': '187147', 'feature_multiplier': 0.5}]
        geos_df = pd.DataFrame(geo_data)
        training_result.add_data('geos_features', geos_df, for_prediction=True)
        """

        # example of extra data to be saved for evaluation but not needed for prediction
        extra_df = pd.DataFrame([{'eval': 1}])
        training_result.add_data('extra_data', extra_df, for_prediction=False)

        return training_result

    def train_transform(self, feature_matrix, model_name):
        model = create_model(model_name, self.config)
        X_train, y_train, X_test, y_test = self.split(feature_matrix, test_split=0.7, target=self._target_column)
        X_train_transformed = model.fit_transform(X_train, y_train)
        X_test_transformed = model.transform(X_test)
        datadict = {
            'X_train': X_train_transformed,
            'X_test': X_test_transformed,
            'y_train': y_train,
            'y_test': y_test
        }
        return datadict, model

    def train_fit(self, feature_matrix_transformed, model_name):
        X_train, X_test = feature_matrix_transformed['X_train'], feature_matrix_transformed['X_test']
        y_train, y_test = feature_matrix_transformed['y_train'], feature_matrix_transformed['y_test']
        estimator = create_estimator(model_name)
        estimator.fit(X_train, y_train)
        p_train = estimator.predict_proba(X_train)
        p_test = estimator.predict_proba(X_test)

        metric_data = self.evaluate(y_train, p_train, y_test, p_test)
        self.job_reporter.metrics('evaluation', metric_data)
        # use this to validate metric values
        self.job_reporter.metric('data_size', name='training_size', value=len(X_train),
                                 min_val=self.config.min_rows, max_val=self.config.max_rows)
        return estimator

    def predict(self, feature_matrix, training_result):
        identifiers = feature_matrix[['memberid', 'productcode', 'tageoid']]
        prediction = training_result.get_model('model').predict(feature_matrix)
        return pd.DataFrame(pd.np.column_stack([identifiers, prediction]),
                            columns=['memberid', 'productcode', 'tageoid', 'bought'])

    def save_model(self, name, model, directory):
        # we know this should be a DspPipeline, which has save_model implemented
        assert isinstance(model, DspPipeline)
        model.save_model(name, model, directory)

    def load_model(self, name, directory):
        # create a dummy DspPipeline which knows how to load saved DspPipelines
        dp = DspPipeline("loader")
        model = dp.load_model(name, directory)
        assert isinstance(model, DspPipeline)
        return model

    def split(self, df, target, test_split):
        df_train, df_test = train_test_split(df, train_size=test_split)
        return df_train.drop(columns=[target]), df_train[target], df_test.drop(columns=[target]), df_test[target]

    def evaluate(self, y_train, p_train, y_test, p_test):
        metrics = {}
        loss_train = np.mean(log_loss(y_train, p_train))
        loss_test = np.mean(log_loss(y_test, p_test))
        auc_train = np.mean(roc_auc_score(y_train, p_train))
        auc_test = np.mean(roc_auc_score(y_test, p_test))
        metrics['log_loss_train'] = loss_train
        metrics['log_loss_test'] = loss_test
        metrics['auc_train'] = auc_train
        metrics['auc_test'] = auc_test
        logging.info('logloss train: {:.4g} -- logloss test: {:.4g} -- AUC train {:.3g} -- '
              'AUC: test {:.3g}'.format(loss_train, loss_test, auc_train, auc_test))
        return metrics

    def save_predictions(self, predict_df):
        if self.config.bigquery:
            self.dbrunner.push_to_bq(predict_df,
                                     'tmp',
                                     self.config.output_table,
                                     if_exists='replace')

    def parameter_search(self, feature_matrix, model_name):

        # split train/test
        X_train, y_train, X_test, y_test = self.split(feature_matrix, test_split=0.7, target=self._target_column)

        # create CV folds
        X = pd.concat([X_train, X_test])
        y = pd.concat([y_train, y_test])
        test_fold = [-1] * len(X_train) + [1] * len(X_test)

        # create grid search object
        model = create_model(model_name, self.config).pipeline

        # define parameters to search over
        penalty = ['l2', 'l1']
        C = [0.01, 0.1, 1.0]
        max_iter = [100, 200, 300]
        param_grid = {
            'model__penalty': penalty,
            'model__C': C,
            'model__max_iter': max_iter
        }

        grid_search = GridSearchCV(model, param_grid=param_grid,
                                   scoring='f1_weighted',
                                   verbose=0,
                                   cv=PredefinedSplit(test_fold),
                                   return_train_score=True)
        # begin parameter search
        grid_search.fit(X, y)
        logging.info(grid_search.cv_results_)
        logging.info('mean train score: {}'.format(grid_search.cv_results_['mean_train_score']))
        logging.info('mean test score:{}'.format(grid_search.cv_results_['mean_test_score']))
        logging.info('rank test score:{}'.format(grid_search.cv_results_['rank_test_score']))
        logging.info('best params:{}'.format(grid_search.best_params_))

    def compare_models(self, feature_matrix, model_list):
        metrics_list = []
        X_train, y_train, X_test, y_test = self.split(feature_matrix, test_split=0.7, target=self._target_column)
        for model_name in model_list:
            logging.info('Training {} model.'.format(model_name))
            model = create_model(model_name, self.config)

            p_train = model.fit(X_train, y_train).predict_proba(X_train)
            p_test = model.predict_proba(X_test)
            metrics = self.evaluate(y_train, p_train, y_test, p_test)
            metrics_list.append(pd.DataFrame(metrics, index=[model_name]))
        return pd.concat(metrics_list)

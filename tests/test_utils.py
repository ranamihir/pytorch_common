import unittest
import numpy as np
import os
import torch

from pytorch_common.additional_configs import BaseModelConfig
from pytorch_common.models import create_model
from pytorch_common import utils


class TestUtils(unittest.TestCase):
    def test_pickle_file_handling(self):
        def _test_file_handling(data, primary_path, file_name=None, module='pickle'):
            utils.save_object(data, primary_path, file_name, module=module)
            loaded_data = utils.load_object(primary_path, file_name, module=module)
            result = data == loaded_data
            if not isinstance(result, bool):
                result = result.all()
            self.assertTrue(result)

            if file_name is None:
                utils.remove_object(primary_path)
                self.assertFalse(os.path.isfile(primary_path))
            else:
                utils.remove_object(primary_path, file_name)
                self.assertFalse(os.path.isfile(os.path.join(primary_path, file_name)))

        primary_path = 'dummy_dir'

        dummy_np_data = np.random.randn(10,10)
        dummy_yaml_data = {'x': 1, 'y': 2, 'z': 3}

        for data, file_name, module in zip([dummy_np_data, dummy_np_data, dummy_yaml_data],
                                           ['dummy_data.pkl', 'dummy_data.pkl', 'dummy_data.yaml'],
                                           ['pickle', 'dill', 'yaml']):
            _test_file_handling(data, file_name, module=module)

            utils.make_dirs(primary_path)
            _test_file_handling(data, primary_path, file_name=file_name, module=module)
            utils.remove_dir(primary_path)
            self.assertFalse(os.path.isdir(primary_path))

    def test_send_model_to_device(self):
        config = BaseModelConfig({'in_dim': 1, 'num_classes': 1})
        model = create_model('single_layer_classifier', config)

        model_cpu = utils.send_model_to_device(model.copy(), 'cpu')
        self.assertFalse(utils.is_model_on_gpu(model_cpu))
        self.assertFalse(utils.is_model_on_gpu(model))

        if torch.cuda.is_available():
            model_cuda = utils.send_model_to_device(model_cpu, 'cuda:0')
            self.assertTrue(utils.is_model_on_gpu(model_cuda))
            self.assertTrue(utils.is_model_on_gpu(model_cpu))
            self.assertFalse(utils.is_model_on_gpu(model))

            n_gpu = torch.cuda.device_count()
            if n_gpu > 1:
                model_parallel = utils.send_model_to_device(model_cuda, 'cuda:0', range(n_gpu))
                self.assertTrue(utils.is_model_on_gpu(model_parallel))
                self.assertTrue(utils.is_model_parallelized(model_parallel))
                self.assertNotIsInstance(model_cuda, utils.DataParallel)
                self.assertFalse(utils.is_model_on_gpu(model))
                self.assertFalse(utils.is_model_parallelized(model))

                model_cpu = utils.send_model_to_device(model_parallel, 'cpu')
                self.assertFalse(utils.is_model_on_gpu(model_cpu))

            model_cpu = utils.send_model_to_device(model_cuda, 'cpu')
            self.assertFalse(utils.is_model_on_gpu(model_cpu))

    def test_send_batch_to_device(self):
        a = torch.tensor([1,2,3], device='cpu')
        b = torch.tensor([4,5,6], device='cpu')
        c = torch.tensor([7,8,9], device='cpu')
        batch = ((a, b), c)

        if torch.cuda.is_available():
            batch_cuda = utils.send_batch_to_device(batch, 'cuda:0')
            self.assertTrue(utils.compare_tensors_or_arrays(batch_cuda, batch))
            self.assertFalse(utils.is_batch_on_gpu(batch))
            self.assertTrue(utils.is_batch_on_gpu(batch_cuda))

            batch_cpu = utils.send_batch_to_device(batch, 'cpu')
            self.assertTrue(utils.compare_tensors_or_arrays(batch_cpu, batch))
            self.assertFalse(utils.is_batch_on_gpu(batch_cpu))
            self.assertTrue(utils.is_batch_on_gpu(batch_cuda))

    def test_convert_tensor_to_numpy(self):
        a, b, c = [1.5,2,3], [4,-5,6], [7,0,9]

        a_np = np.array(a)
        b_np = np.array(b)
        c_np = np.array(c)
        batch_np = ((a_np, b_np), c_np)

        a_torch = torch.tensor(a, device='cpu')
        b_torch = torch.tensor(b, device='cpu')
        c_torch = torch.tensor(c, device='cpu')
        batch_torch = ((a_torch, b_torch), c_torch)

        self.assertTrue(utils.compare_tensors_or_arrays(batch_np,
                        utils.convert_tensor_to_numpy(batch_torch)))
        if torch.cuda.is_available():
            batch_torch_cuda = utils.send_batch_to_device(batch_torch, 'cuda:0')
            self.assertTrue(utils.compare_tensors_or_arrays(batch_np,
                            utils.convert_tensor_to_numpy(batch_torch_cuda)))

    def test_convert_numpy_to_tensor(self):
        a, b, c = [1.5,2,3], [4,-5,6], [7,0,9]

        a_np = np.array(a)
        b_np = np.array(b)
        c_np = np.array(c)
        batch_np = ((a_np, b_np), c_np)

        a_torch = torch.tensor(a, device='cpu')
        b_torch = torch.tensor(b, device='cpu')
        c_torch = torch.tensor(c, device='cpu')
        batch_torch = ((a_torch, b_torch), c_torch)

        self.assertTrue(utils.compare_tensors_or_arrays(batch_torch,
                        utils.convert_numpy_to_tensor(batch_np)))
        if torch.cuda.is_available():
            batch_torch_cuda = utils.convert_numpy_to_tensor(batch_np, 'cuda:0')
            self.assertTrue(utils.compare_tensors_or_arrays(batch_torch, batch_torch_cuda))
            self.assertTrue(utils.is_batch_on_gpu(batch_torch_cuda))


if __name__ == '__main__':
    unittest.main()

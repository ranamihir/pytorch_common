import os
import unittest

import numpy as np
import torch

from pytorch_common import utils
from pytorch_common.additional_configs import BaseModelConfig
from pytorch_common.models import create_model
from pytorch_common.types import Callable, Dict, List, Optional, Union, _Batch


class TestUtils(unittest.TestCase):
    def test_file_handling(self):
        """
        Test saving / loading of different
        files (pickle, dill, yaml).
        """

        def _test_module_file_handling(
            data: Union[np.ndarray, Dict],
            primary_path: str,
            file_name: Optional[str] = None,
            module: Optional[str] = "pickle",
        ) -> None:
            """
            Test saving / loading of a
            file with a given `module`.
            """
            # Save the file
            utils.save_object(data, primary_path, file_name, module=module)

            # Load saved file
            loaded_data = utils.load_object(primary_path, file_name, module=module)

            # Ensure results match
            result = data == loaded_data
            if not isinstance(result, bool):
                result = result.all()
            self.assertTrue(result)

            # Delete saved file
            if file_name is None:
                utils.remove_object(primary_path)
                self.assertFalse(os.path.isfile(primary_path))
            else:
                utils.remove_object(primary_path, file_name)
                self.assertFalse(os.path.isfile(utils.get_file_path(primary_path, file_name)))

        # Initialize dummy directory and data
        primary_path = "dummy_dir"
        dummy_np_data = np.random.randn(10, 10)
        dummy_yaml_data = {"x": 1, "y": 2, "z": 3}

        # Test file handling for all file types
        for data, file_name, module in zip(
            [dummy_np_data, dummy_np_data, dummy_yaml_data],
            ["dummy_data.pkl", "dummy_data.pkl", "dummy_data.yaml"],
            ["pickle", "dill", "yaml"],
        ):
            # Test directly with `file_name`
            _test_module_file_handling(data, file_name, module=module)

            # Test with `file_name` inside `primary_path`
            utils.make_dirs(primary_path)
            _test_module_file_handling(data, primary_path, file_name=file_name, module=module)

            # Delete created directories
            utils.remove_dir(primary_path)
            self.assertFalse(os.path.isdir(primary_path))

    def test_get_string_from_dict(self):
        """
        Test correct generation of string
        from config dictionary.
        """
        # Test output for empty inputs
        self.assertEqual(utils.get_string_from_dict(), "")
        self.assertEqual(utils.get_string_from_dict({}), "")

        # Test output
        dictionary = {"size": 100, "lr": 1e-3}
        self.assertEqual(utils.get_string_from_dict(dictionary), "lr_0.001-size_100")

        # Test same output regardless of order
        dictionary = {"lr": 1e-3, "size": 100}
        self.assertEqual(utils.get_string_from_dict(dictionary), "lr_0.001-size_100")

    def test_get_string_from_dict(self):
        """
        Test correct generation of unique
        string from config dictionary.
        """
        primary_name = "dummy"

        # Test unique string
        dictionary = {"size": 100, "lr": 1e-3}
        unique_str1 = utils.get_unique_config_name(primary_name, dictionary)
        self.assertTrue(unique_str1.startswith("dummy-"))

        # Test same string regardless of order
        dictionary = {"lr": 1e-3, "size": 100}
        unique_str2 = utils.get_unique_config_name(primary_name, dictionary)
        self.assertEqual(unique_str1, unique_str2)

    def test_send_model_to_device(self):
        """
        Test sending of model to different devices.
        """
        # Create model
        config = BaseModelConfig({"in_dim": 1, "num_classes": 1})
        model = create_model("single_layer_classifier", config)

        # Test sending model to CPU
        model_cpu = utils.send_model_to_device(model.copy(), "cpu")
        self.assertFalse(utils.is_model_on_gpu(model_cpu))
        self.assertFalse(utils.is_model_on_gpu(model))
        self.assertFalse(model_cpu.is_cuda)
        self.assertFalse(model.is_cuda)

        if torch.cuda.is_available():
            # Test sending model to GPU
            model_cuda = utils.send_model_to_device(model_cpu.copy(), "cuda")
            self.assertTrue(utils.is_model_on_gpu(model_cuda))
            self.assertTrue(model_cuda.is_cuda)

            # Ensure original models unchanged
            self.assertFalse(utils.is_model_on_gpu(model_cpu))
            self.assertFalse(utils.is_model_on_gpu(model))
            self.assertFalse(model_cpu.is_cuda)
            self.assertFalse(model.is_cuda)

            n_gpu = torch.cuda.device_count()
            if n_gpu > 1:
                # Test sending model to multiple GPUs
                model_parallel = utils.send_model_to_device(model_cuda.copy(), "cuda", range(n_gpu))
                self.assertTrue(utils.is_model_on_gpu(model_parallel))
                self.assertTrue(utils.is_model_parallelized(model_parallel))
                self.assertTrue(model_parallel.is_cuda)
                self.assertTrue(model_parallel.is_parallelized)

                # Ensure original single-GPU model unchanged
                self.assertTrue(utils.is_model_on_gpu(model_cuda))
                self.assertFalse(utils.is_model_parallelized(model_cuda))
                self.assertTrue(model_cuda.is_cuda)
                self.assertFalse(model_cuda.is_parallelized)

                # Ensure original model unchanged
                self.assertFalse(utils.is_model_on_gpu(model))
                self.assertFalse(utils.is_model_parallelized(model))
                self.assertFalse(model.is_cuda)
                self.assertFalse(model.is_parallelized)

                # Test sending of multi-GPU model to CPU
                model_cpu = utils.send_model_to_device(model_parallel, "cpu")
                self.assertFalse(utils.is_model_on_gpu(model_cpu))
                self.assertFalse(utils.is_model_parallelized(model_cpu))
                self.assertFalse(model_cpu.is_cuda)
                self.assertFalse(model_cpu.is_parallelized)

            # Test sending of single-GPU model to CPU
            model_cpu = utils.send_model_to_device(model_cuda, "cpu")
            self.assertFalse(utils.is_model_on_gpu(model_cpu))
            self.assertFalse(utils.is_model_parallelized(model_cpu))
            self.assertFalse(model_cpu.is_cuda)
            self.assertFalse(model_cpu.is_parallelized)

    def test_send_batch_to_device(self):
        """
        Test sending of batch to different devices.
        """
        # Define batch
        a, b, c = [1, 2, 3], [4, 5, 6], [7, 8, 9]
        batch = self._get_batch(a, b, c, batch_type=torch.tensor, device="cpu")

        if torch.cuda.is_available():
            # Test sending batch to GPU
            batch_cuda = utils.send_batch_to_device(batch, "cuda")
            self.assertTrue(utils.compare_tensors_or_arrays(batch_cuda, batch))
            self.assertFalse(utils.is_batch_on_gpu(batch))
            self.assertTrue(utils.is_batch_on_gpu(batch_cuda))

            # Test sending batch to CPU
            batch_cpu = utils.send_batch_to_device(batch, "cpu")
            self.assertTrue(utils.compare_tensors_or_arrays(batch_cpu, batch))
            self.assertFalse(utils.is_batch_on_gpu(batch_cpu))
            self.assertTrue(utils.is_batch_on_gpu(batch_cuda))

    def test_convert_tensor_to_numpy(self):
        """
        Test converting a batch of torch
        tensor(s) to numpy array(s).
        """
        # Define numpy and torch batches
        a, b, c = [1.5, 2, 3], [4, -5, 6], [7, 0, 9]
        batch_np = self._get_batch(a, b, c, batch_type=np.array)
        batch_torch = self._get_batch(a, b, c, batch_type=torch.tensor, device="cpu")

        # Compare contents of both batches
        self.assertTrue(utils.compare_tensors_or_arrays(batch_np, utils.convert_tensor_to_numpy(batch_torch)))

        if torch.cuda.is_available():
            # Compare contents of both batches when tensor is on GPU
            batch_torch_cuda = utils.send_batch_to_device(batch_torch, "cuda")
            self.assertTrue(utils.compare_tensors_or_arrays(batch_np, utils.convert_tensor_to_numpy(batch_torch_cuda)))

    def test_convert_numpy_to_tensor(self):
        """
        Test converting a batch of numpy
        array(s) to torch tensor(s).
        """
        # Define numpy and torch batches
        a, b, c = [1.5, 2, 3], [4, -5, 6], [7, 0, 9]
        batch_np = self._get_batch(a, b, c, batch_type=np.array)
        batch_torch = self._get_batch(a, b, c, batch_type=torch.tensor, device="cpu")

        # Compare contents of both batches
        self.assertTrue(utils.compare_tensors_or_arrays(batch_torch, utils.convert_numpy_to_tensor(batch_np)))

        if torch.cuda.is_available():
            # Compare contents of both batches when tensor is on GPU
            batch_torch_cuda = utils.convert_numpy_to_tensor(batch_np, "cuda")
            self.assertTrue(utils.compare_tensors_or_arrays(batch_torch, batch_torch_cuda))
            self.assertTrue(utils.is_batch_on_gpu(batch_torch_cuda))

    def _get_batch(
        self,
        a: List[float],
        b: List[float],
        c: List[float],
        batch_type: Callable[[List[float]], Union[np.ndarray, torch.Tensor]],
        **kwargs,
    ) -> _Batch:
        """
        Construct a numpy / torch batch of shape
        which forces recursion in type conversion.
        """
        a_ = batch_type(a, **kwargs)
        b_ = batch_type(b, **kwargs)
        c_ = batch_type(c, **kwargs)
        return ((a_, b_), c_)


if __name__ == "__main__":
    unittest.main()

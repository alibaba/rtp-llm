import os
import unittest
from pathlib import Path
from maga_transformer.utils.database import CkptDatabase


class CkptDataBaseTest(unittest.TestCase):

    def __init__(self, methodName: str = "Run CkptDataBaseTest") -> None:
        super().__init__(methodName)

    @staticmethod
    def _testdata_path():
        return os.path.join(os.getcwd(), "maga_transformer/utils/test/testdata/ckpt_database_testdata/")

    def test_collect_ckpt_file(self):
        path = os.path.join(self._testdata_path(), "bin_testdata")
        database = CkptDatabase(path)
        self.assertEqual(1, len(database.PretrainFileList))
        self.assertEqual(path + '/pytorch_model.bin', database.PretrainFileList[0].file_name)
        self.assertEqual(12, len(database.PretrainFileList[0].get_tensor_names()))

        path = os.path.join(self._testdata_path(), "pt_testdata")
        database = CkptDatabase(path)
        self.assertEqual(1, len(database.PretrainFileList))
        self.assertEqual(path + '/test.pt', database.PretrainFileList[0].file_name)
        self.assertEqual(36, len(database.PretrainFileList[0].get_tensor_names()))

        path = os.path.join(self._testdata_path(), "safetensor_testdata")
        database = CkptDatabase(path)
        self.assertEqual(1, len(database.PretrainFileList))
        self.assertEqual(path + '/test.safetensors', database.PretrainFileList[0].file_name)
        self.assertEqual(28, len(database.PretrainFileList[0].get_tensor_names()))

        path = os.path.join(self._testdata_path(), "bin_testdata")
        lora_path = os.path.join(self._testdata_path(), "lora_testdata")
        database = CkptDatabase(path)
        database.load_lora("test", lora_path)
        self.assertEqual(1, len(database.PretrainFileList))
        self.assertEqual(path + '/pytorch_model.bin', database.PretrainFileList[0].file_name)
        self.assertEqual(12, len(database.PretrainFileList[0].get_tensor_names()))
        self.assertEqual(1, len(database.LoraCkpt.LoraFileList))
        self.assertEqual(8, list(database.LoraCkpt.LoraFileList)[0].rank)
        self.assertEqual(8, list(database.LoraCkpt.LoraFileList)[0].lora_alpha)
        self.assertEqual(0.0, list(database.LoraCkpt.LoraFileList)[0].lora_dropout)
        self.assertEqual(['c_proj', 'w2', 'c_attn', 'w1'], list(database.LoraCkpt.LoraFileList)[0].target_modules)
        self.assertEqual(1, len(list(database.LoraCkpt.LoraFileList.values())[0]))
        self.assertEqual(12, len(list(database.LoraCkpt.LoraFileList.values())[0][0].get_tensor_names()))

    def test_mix_ckpt_file(self):
        path = os.path.join(self._testdata_path(), "mixture_testdata")
        database = CkptDatabase(path)
        self.assertEqual(1, len(database.PretrainFileList))
        self.assertEqual(path + '/test.safetensors', database.PretrainFileList[0].file_name)
        self.assertEqual(28, len(database.PretrainFileList[0].get_tensor_names()))


class LoraTest(unittest.TestCase):

    def __init__(self, methodName: str = "Run CkptDataBaseTest") -> None:
        super().__init__(methodName)
    
    @staticmethod
    def _testdata_path():
        return os.path.join(os.getcwd(), "maga_transformer/utils/test/testdata/ckpt_database_testdata/")

    def test_collect_ckpt_file(self):
        path = os.path.join(self._testdata_path(), "bin_testdata")
        database = CkptDatabase(path)
        self.assertEqual(1, len(database.PretrainFileList))
        self.assertEqual(path + '/pytorch_model.bin', database.PretrainFileList[0].file_name)
        self.assertEqual(12, len(database.PretrainFileList[0].get_tensor_names()))

        lora_path = os.path.join(self._testdata_path(), "lora_testdata")
        database.load_lora("test_name", lora_path)
        self.assertEqual(1, len(database.LoraCkpt.LoraFileList))
        lora_config = database.get_lora_config("test_name")
        self.assertEqual(8, lora_config.rank)
        self.assertEqual(8, lora_config.lora_alpha)
        self.assertEqual(0.0, lora_config.lora_dropout)
        self.assertEqual(['c_proj', 'w2', 'c_attn', 'w1'], lora_config.target_modules)
        self.assertEqual(1, len(database.LoraCkpt.get_lora("test_name")))
        self.assertEqual(12, len(database.get_lora_tensor_names("test_name")))

        self.assertTrue(database.remove_lora("test_name"))
        lora_config = database.get_lora_config("test_name")
        self.assertEqual(0, lora_config.rank)
        self.assertEqual(0, lora_config.lora_alpha)
        self.assertEqual(0.0, lora_config.lora_dropout)
        self.assertEqual([], lora_config.target_modules)
        self.assertEqual(0, len(database.LoraCkpt.get_lora("test_name")))
        self.assertEqual(0, len(database.get_lora_tensor_names("test_name")))


        lora_path = os.path.join(self._testdata_path(), "lora_testdata_safetensor")
        database.load_lora("test_name", lora_path)
        self.assertEqual(1, len(database.LoraCkpt.LoraFileList))
        lora_config = database.get_lora_config("test_name")
        self.assertEqual(8, lora_config.rank)
        self.assertEqual(8, lora_config.lora_alpha)
        self.assertEqual(0.0, lora_config.lora_dropout)
        self.assertEqual(['c_proj', 'w2', 'c_attn', 'w1'], lora_config.target_modules)
        self.assertEqual(1, len(database.LoraCkpt.get_lora("test_name")))
        self.assertEqual(12, len(database.get_lora_tensor_names("test_name")))

if __name__ == '__main__':
    unittest.main()

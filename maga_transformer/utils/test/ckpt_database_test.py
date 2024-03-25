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
        self.assertEqual(1, len(database.LoraFileList))
        self.assertEqual(8, list(database.LoraFileList)[0].rank)
        self.assertEqual(8, list(database.LoraFileList)[0].lora_alpha)
        self.assertEqual(0.0, list(database.LoraFileList)[0].lora_dropout)
        self.assertEqual(['c_proj', 'w2', 'c_attn', 'w1'], list(database.LoraFileList)[0].target_modules)
        self.assertEqual(1, len(list(database.LoraFileList.values())[0]))
        self.assertEqual(12, len(list(database.LoraFileList.values())[0][0].get_tensor_names()))

    def test_mix_ckpt_file(self):
        path = os.path.join(self._testdata_path(), "mixture_testdata")
        database = CkptDatabase(path)
        self.assertEqual(1, len(database.PretrainFileList))
        self.assertEqual(path + '/test.safetensors', database.PretrainFileList[0].file_name)
        self.assertEqual(28, len(database.PretrainFileList[0].get_tensor_names()))

class MegatronCheckpointingTest(unittest.TestCase):

    @staticmethod
    def _testdata_path():
        return os.path.join(os.getcwd(), 'maga_transformer/utils/test/testdata/megatron_checkpointing_test/')

    def test_is_megatron_ckpt(self):    
        ckpt_path = os.path.join(self._testdata_path(), "hf_bin")
        CkptLoader = CkptDatabase(None)
        self.assertFalse(CkptLoader.is_megatron_ckpt(Path(ckpt_path)))
        ckpt_path = os.path.join(self._testdata_path(), "hf_pt")
        self.assertFalse(CkptLoader.is_megatron_ckpt(Path(ckpt_path)))
        ckpt_path = os.path.join(self._testdata_path(), "hf_pt")
        self.assertFalse(CkptLoader.is_megatron_ckpt(Path(ckpt_path)))
        ckpt_path = os.path.join(self._testdata_path(), "deep_m6")
        self.assertFalse(CkptLoader.is_megatron_ckpt(Path(ckpt_path))) 
        ckpt_path = os.path.join(self._testdata_path(), "dm_rank_xx")
        self.assertTrue(CkptLoader.is_megatron_ckpt(Path(ckpt_path)))         
        ckpt_path = os.path.join(self._testdata_path(), "dm_rank_xx_xx")
        self.assertTrue(CkptLoader.is_megatron_ckpt(Path(ckpt_path)))    
        ckpt_path = os.path.join(self._testdata_path(), "m6")
        self.assertTrue(CkptLoader.is_megatron_ckpt(Path(ckpt_path))) 
        

    def test_get_megatron_ckpt_files(self):    
        ckpt_path = os.path.join(self._testdata_path(), "dm_rank_xx")
        CkptLoader = CkptDatabase(None)
        ckpts = CkptLoader.get_megatron_ckpt_files(Path(ckpt_path))
        expect_files = ['dm_rank_xx/iter_0008000/mp_rank_00/model_rng.pt', 'dm_rank_xx/iter_0008000/mp_rank_01/model_rng.pt']
        expect_files: list[str] = [str(Path(os.path.join(self._testdata_path(), x)).resolve()) for x in expect_files]
        self.assertEqual(expect_files, [f.file_name for f in ckpts], f"{expect_files}, {[f.file_name for f in ckpts]}" )

        ckpt_path = os.path.join(self._testdata_path(), "dm_rank_xx_xx")
        ckpts = CkptLoader.get_megatron_ckpt_files(Path(ckpt_path))
        expect_files = ['dm_rank_xx_xx/iter_0008000/mp_rank_00_000/model_rng.pt', 
                                   'dm_rank_xx_xx/iter_0008000/mp_rank_01_000/model_rng.pt', 
                                   'dm_rank_xx_xx/iter_0008000/mp_rank_00_001/model_rng.pt', 
                                   'dm_rank_xx_xx/iter_0008000/mp_rank_01_001/model_rng.pt']
        expect_files = [str(Path(os.path.join(self._testdata_path(), x)).resolve()) for x in expect_files]
        self.assertEqual(expect_files, [f.file_name for f in ckpts], f"{expect_files} vs {[f.file_name for f in ckpts]}" )

        ckpt_path = os.path.join(self._testdata_path(), "m6")
        ckpts = CkptLoader.get_megatron_ckpt_files(Path(ckpt_path))
        expect_files = ['m6/global_step200/global_step200/mp_rank_00_model_states.pt']
        expect_files = [str(Path(os.path.join(self._testdata_path(), x)).resolve()) for x in expect_files]
        self.assertEqual(expect_files, [f.file_name for f in ckpts], f"{expect_files} vs {[f.file_name for f in ckpts]}" )

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
        self.assertEqual(1, len(database.LoraFileList))
        lora_config = database.get_lora_config("test_name")
        self.assertEqual(8, lora_config.rank)
        self.assertEqual(8, lora_config.lora_alpha)
        self.assertEqual(0.0, lora_config.lora_dropout)
        self.assertEqual(['c_proj', 'w2', 'c_attn', 'w1'], lora_config.target_modules)
        self.assertEqual(1, len(database.get_lora("test_name")))
        self.assertEqual(12, len(database.get_lora_tensor_names("test_name")))

        self.assertTrue(database.remove_lora("test_name"))
        lora_config = database.get_lora_config("test_name")
        self.assertEqual(0, lora_config.rank)
        self.assertEqual(0, lora_config.lora_alpha)
        self.assertEqual(0.0, lora_config.lora_dropout)
        self.assertEqual([], lora_config.target_modules)
        self.assertEqual(0, len(database.get_lora("test_name")))
        self.assertEqual(0, len(database.get_lora_tensor_names("test_name")))


        lora_path = os.path.join(self._testdata_path(), "lora_testdata_safetensor")
        database.load_lora("test_name", lora_path)
        self.assertEqual(1, len(database.LoraFileList))
        lora_config = database.get_lora_config("test_name")
        self.assertEqual(8, lora_config.rank)
        self.assertEqual(8, lora_config.lora_alpha)
        self.assertEqual(0.0, lora_config.lora_dropout)
        self.assertEqual(['c_proj', 'w2', 'c_attn', 'w1'], lora_config.target_modules)
        self.assertEqual(1, len(database.get_lora("test_name")))
        self.assertEqual(12, len(database.get_lora_tensor_names("test_name")))

    def test_lora_tensor_name_check(self):
        database = CkptDatabase(None)
        test_valid_names = ["base_model.model.x.lora_A.weight",
                            "base_model.model.x.lora_B.weight",]

        test_invalid_names = ["base_model.model.x.x.weight",
                              "base_model.x.lora_B.weight",
                              "base_model.model.x.lora_A.default.weight",
                              "base_model.model.x.lora_A.default.weight.x"]

        for name in test_valid_names:
            self.assertEqual(None, database.lora_tensor_check(name))

        for name in test_invalid_names:
            self.assertRaises(Exception, database.lora_tensor_check, name)
            

if __name__ == '__main__':
    unittest.main()

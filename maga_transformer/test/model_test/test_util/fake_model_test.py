from maga_transformer.async_decoder_engine.async_model import AsyncModel
from maga_transformer.model_factory import ModelFactory
from typing import Dict, Any
import torch
from maga_transformer.test.model_test.test_util.model_test_base import ModelTestBase
from maga_transformer.test.model_test.test_util.fake_model_loader import FakeModelLoader

class FakeModelTest(ModelTestBase):
    def __init__(self, methodName: str = "runTest",
                        model_type: str = "",
                        tokenizer_path: str = "",
                        ckpt_path: str = "",
                        weight_type: torch.dtype = torch.float16,
                        test_loss: bool = False,
                        fake_name: str = "") -> None:
        super().__init__(methodName, model_type, tokenizer_path, ckpt_path, weight_type, test_loss, fake_name)

    def _load_model(self):
        fake_model_loader = FakeModelLoader(self.model_type, self.tokenizer_path, self.ckpt_path, self.weight_type)
        return fake_model_loader.load_model()

def single_fake_test(name: str, fake_name: str, model_config: Dict[str, Any], test_loss: bool):
    model_test = FakeModelTest("runTest",
                                model_config["model_type"],
                                model_config["tokenizer_path"],
                                model_config["ckpt_path"],
                                model_config["weight_type"],                                
                                test_loss=test_loss,
                                fake_name=fake_name)
    model_test.simple_test(is_fake=True)
    del model_test

import os
import sys
import time
import copy
import torch
import logging
import pynvml
from typing import Any, List, Union, Dict
from maga_transformer.models.llama import Llama
from maga_transformer.pipeline.pipeline import Pipeline
from maga_transformer.model_factory import ModelFactory, ModelConfig
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.utils.ft_plugin import plguin_loader
from maga_transformer.async_decoder_engine.async_model import AsyncModel
from unittest import TestCase, main

class ModelTestBase(TestCase):
    '''
    using model to test synax of pipeline, do not check result
    '''
    def __init__(self, methodName: str = "runTest",
                        model_type: str = "",
                        tokenizer_path: str = "",
                        ckpt_path: str = "",
                        weight_type: torch.dtype = torch.float16,
                        test_loss: bool = False,
                        fake_name: str = ""):
        super().__init__(methodName)

        self.model_type = model_type
        self.tokenizer_path = tokenizer_path
        self.ckpt_path = ckpt_path
        self.weight_type = weight_type        
        self.test_loss = test_loss
        self.fake_name = fake_name

        os.environ['FT_PLUGIN_PATH'] = os.path.join(os.getcwd(), "maga_transformer/plugins/ret_hidden_states.py")
        plguin_loader.reload()

        logging.info(f"model_type: {self.model_type}")
        logging.info(f"tokenizer path: {self.tokenizer_path}")
        logging.info(f"check point path: {self.ckpt_path}")
        logging.info(f"weight_type: {weight_type}")
        logging.info(f"test_loss: {test_loss}")

    def flat(self, x: Union[List[bool], bool]) -> List[bool]:
        ret: List[bool] = []
        if isinstance(x, list):
            for ele in x:
                ret.extend(self.flat(ele))
        else:
            ret.append(x)
        return ret

    def close(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # set higher accuracy for CI test
        if os.environ.get('CI_TEST', None) == '1':
            return torch.isclose(a.cpu(), b.cpu(), rtol=1e-03, atol=1e-05)
        else:
            return torch.isclose(a.cpu(), b.cpu(), rtol=1e-03, atol=1e-02)

    def _test_score(self, pipeline: Pipeline, generate_config: Dict[str, Any], expect_result_file: str):
        test_input = ["hello?", "what's your name"]
        # this returns list[torch.Tensor]
        responses = [response[1] for response in pipeline.pipeline(test_input, [[], []], generate_config=generate_config)]

        print("expect_result_file = ", expect_result_file)
        # torch.save(
        #     {"context_decoder_result": torch.stack(responses[0]), "decoder_result": torch.stack(responses[1])}, expect_result_file
        # )

        expect_result = torch.load(expect_result_file)

        print("expect_result['context_decoder_result'] = ", expect_result['context_decoder_result'])
        print("responses[0].hidden_states = ", responses[0])

        print("expect_result['decoder_result'] = ", expect_result['decoder_result'])
        print("responses[1].hidden_states = ", responses[1])

        expect_result['context_decoder_result'] = expect_result['context_decoder_result'].reshape(-1)
        expect_result['decoder_result'] = expect_result['decoder_result'].reshape(-1)

        responses0 = torch.concat(responses[0]).reshape(-1)
        responses1 = torch.concat(responses[1]).reshape(-1)
        sys.stdout.flush()

        self.assertTrue(all(self.flat(self.close(expect_result['context_decoder_result'], responses0).tolist())))
        self.assertTrue(all(self.flat(self.close(expect_result['decoder_result'], responses1).tolist())))

    # 由于async请求时候跑的顺序不一定一致，所以需要修改max_new_tokens并取最后一个，来获取确定的结果
    def _test_async_score(self, pipeline: Pipeline, generate_config: Dict[str, Any], expect_result_file: str):
        test_input = ["hello?", "what's your name"]
        expect_result = torch.load(expect_result_file)
        expect_result['context_decoder_result'] = expect_result['context_decoder_result'].reshape(-1)
        expect_result['decoder_result'] = expect_result['decoder_result'].reshape(-1)

        generate_config['max_new_tokens'] = 1
        responses = [response[1] for response in pipeline.pipeline(test_input, [[], []], generate_config=generate_config)]

        print("expect_result[0]:", expect_result['context_decoder_result'])
        print("hidden_states: ", responses[-1])

        responses0 = torch.concat(responses[-1]).reshape(-1)
        self.assertTrue(all(self.flat(self.close(expect_result['context_decoder_result'], responses0).tolist())))

        generate_config['max_new_tokens'] = 2
        responses = [response[1] for response in pipeline.pipeline(test_input, [[], []], generate_config=generate_config)]

        print("expect_result[1]:", expect_result['decoder_result'])
        print("hidden_states: ", responses[-1])

        responses1 = torch.concat(responses[-1]).reshape(-1)
        self.assertTrue(all(self.flat(self.close(expect_result['decoder_result'], responses1).tolist())))

    def _test_ft_score(self, pipeline: Pipeline, expect_result_file: str):
        generate_config = {
            "top_k": 1,
            "max_new_tokens": 10
        }
        self._test_score(pipeline, generate_config, expect_result_file)

    def _test_ft_async_score(self, pipeline: Pipeline, expect_result_file: str):
        generate_config = {
            "top_k": 1,
            "max_new_tokens": 1
        }
        self._test_async_score(pipeline, generate_config, expect_result_file)


    def _test_ft_config_score(self, pipeline: Pipeline, expect_result_file: str):
        generate_config = {
            "max_new_tokens": 16,
            "random_seed": None,
            "top_k": 1,
            "do_sample": False,
            "top_p": 0.95,
            "temperature": 1
        }
        self._test_score(pipeline, generate_config, expect_result_file)

    def _test_hf_score(self, pipeline: Pipeline, expect_result_file: str):
        generate_config = {
            "top_k": 1,
            "temperature": None,
            "max_new_tokens": 10,
            "repetition_penalty": 1,
            "using_hf_sampling": True
        }
        self._test_score(pipeline, generate_config, expect_result_file)

    def _test_loss(self, pipeline: Pipeline, expect_result_file: str):
        generate_config = {
            "top_k": 1,
            "max_new_tokens": 10,
            "repetition_penalty": 1,
            "calculate_loss": True
        }

        test_input = "hello?"
        response = pipeline.pipeline(test_input, [[], []], generate_config=generate_config)
        batch_loss = torch.Tensor(response.generate_output.loss)
        self.assertTrue(all(self.flat(self.close(expect_result[0], batch_loss).tolist())))

        test_input = "what's your name"
        response = pipeline.pipeline(test_input, [[], []], generate_config=generate_config)
        batch_loss = torch.Tensor(response.generate_output.loss)
        self.assertTrue(all(self.flat(self.close(expect_result[1], batch_loss).tolist())))

    def _load_model(self):
        model = ModelFactory.from_model_config(ModelConfig(
            model_type=self.model_type,
            tokenizer_path=self.tokenizer_path,
            ckpt_path=self.ckpt_path,
            weight_type=self.weight_type,
            max_seq_len=8192))
        return model

    def simple_test(self, is_fake: bool):
        model = self._load_model()
        try:
            pipeline = Pipeline(model, model.config, model.tokenizer)
            if model.config.pre_seq_len > 0:
                model_str = "/ptuning"
            else:
                model_str = ""
            if is_fake:
                model_str += "/fake"
            else:
                model_str +=  "/real"
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_name = pynvml.nvmlDeviceGetName(handle)
                pynvml.nvmlShutdown()
                if 'A100' in gpu_name:
                    model_str += '/A100'
                elif 'V100' in gpu_name:
                    model_str += '/V100'
                else:
                    print("No suitable GPU related output file, use V100 output")
                    model_str += '/V100'

            expected_path = "maga_transformer/test/model_test/" + self.fake_name + "/testdata/" + self.model_type + model_str
            if not os.path.exists(expected_path):
                expected_path = "internal_source/maga_transformer/test/" + self.fake_name + "/testdata/" + self.model_type + model_str
            if self.test_loss:
                expected_path += "/expect.loss"
                self._test_loss(pipeline, expected_path)
            else:
                expected_path += "/expect.pt"
                self._test_ft_async_score(pipeline, expected_path)
        finally:
            if isinstance(model, AsyncModel):
                model.stop()

if __name__ == '__main__':
    main()

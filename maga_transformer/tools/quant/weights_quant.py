import json
import logging
import logging.config
import argparse
import torch
import os
import datasets
import time
import hashlib
import numpy as np
import random
import shutil
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Union, NamedTuple
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizer
#from maga_transformer.tokenizer.tokenizer_base import TokenizerBase
from maga_transformer.utils.fuser import fetch_remote_file_to_local, MountRwMode
from maga_transformer.utils.weight_type import WEIGHT_TYPE
from maga_transformer.utils.time_util import Timer, timer_wrapper
from maga_transformer.model_factory import ModelFactory
from maga_transformer.config.generate_config import GenerateConfig, RequestFormat
from maga_transformer.tools.quant.base_quanter import QUANT_TYPE, BaseQuanter, QuanterFactory
from maga_transformer.models.base_model import ModelConfig
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.tools.quant.datasets_adapter import DatasetsAdapter
from maga_transformer.tools.quant.datasets_adapter import DatasetParams, DatasetsAdapter, DatasetType
from maga_transformer.tools.api.model_basic_info_analyzer import parse_ft_model_type
from maga_transformer.openai.api_datatype import ChatCompletionRequest
from maga_transformer.openai.renderer_factory import ChatRendererFactory
from maga_transformer.openai.renderers.custom_renderer import RendererParams
from maga_transformer.structure.request_extractor import RequestExtractor
from maga_transformer.pipeline.default_plugin import DefaultPlugin

CUR_PATH: str = os.path.dirname(os.path.abspath(__file__))


from auto_gptq.modeling._base import BaseGPTQForCausalLM
def register_gptq_models(hf_model_type: str, model_gptq_cls: Type[BaseGPTQForCausalLM]):
    from auto_gptq.modeling.auto import GPTQ_CAUSAL_LM_MODEL_MAP
    registed_model_gptq_cls = GPTQ_CAUSAL_LM_MODEL_MAP.get(hf_model_type)
    if registed_model_gptq_cls and registed_model_gptq_cls != model_gptq_cls:
        raise Exception(f"try register {hf_model_type}'s gtpq_cls {model_gptq_cls} confict with registed_cls: {registed_model_gptq_cls}")
    GPTQ_CAUSAL_LM_MODEL_MAP.update({hf_model_type: model_gptq_cls})
    logging.info(f"register {hf_model_type}'s gtpq_cls {model_gptq_cls}")

    from auto_gptq.modeling._const import SUPPORTED_MODELS
    if hf_model_type not in SUPPORTED_MODELS:
        SUPPORTED_MODELS.append(hf_model_type)
        logging.info("append {hf_model_type} to gptq supported_models")
    else:
        logging.info(f"hf_model_type:[{hf_model_type}] have been registed")

from awq.models.base import BaseAWQForCausalLM
def register_awq_models(hf_model_type: str, model_gptq_cls: Type[BaseAWQForCausalLM]):
    from awq.models.auto import AWQ_CAUSAL_LM_MODEL_MAP
    registed_model_awq_cls = AWQ_CAUSAL_LM_MODEL_MAP.get(hf_model_type)
    if registed_model_awq_cls and registed_model_awq_cls != model_gptq_cls:
        raise Exception(f"try register {hf_model_type}'s awq_cls {model_gptq_cls} confict with registed_cls: {registed_model_awq_cls}")
    AWQ_CAUSAL_LM_MODEL_MAP.update({hf_model_type: model_gptq_cls})
    logging.info(f"register {hf_model_type}'s awq_cls {model_gptq_cls}")


    from awq.models.base import TRANSFORMERS_AUTO_MAPPING_DICT
    if hf_model_type not in TRANSFORMERS_AUTO_MAPPING_DICT:
        TRANSFORMERS_AUTO_MAPPING_DICT[hf_model_type]='AutoModelForCausalLM'
        logging.info("append {hf_model_type} to gptq supported_models")
    else:
        logging.info(f"hf_model_type:[{hf_model_type}] have been registed")

class SampleStrategy(NamedTuple):
    distinct_attrs: List[str] = []
    sample_size: int = 128


class WeightsQuantizer:
    def __init__(self, model_path: str, model_type:Optional[str] = None, offload_folder:Optional[str] = None):
        self.model_path = fetch_remote_file_to_local(model_path)
        if not offload_folder:
            temp_offload_folder = f'{hashlib.md5(self.model_path.encode("utf-8")).hexdigest()}_{time.time()}'
            self.offload_folder = os.path.join(CUR_PATH, temp_offload_folder)
            os.makedirs(self.offload_folder, exist_ok=True)
        else:
            self.offload_folder = offload_folder

        assert self.model_path
        if not model_type:
            model_type = parse_ft_model_type(self.model_path).get('ft_model_type', None)
            assert model_type
        self.model_type = model_type

        self.model_cls = ModelFactory.get_model_cls(self.model_type)
        model_config = ModelConfig(
            model_type=self.model_type,
            ckpt_path=self.model_path,
            weight_type=WEIGHT_TYPE.FP16,
            ptuning_path=None,
            max_seq_len=0,
            tokenizer_path=self.model_path
        )
        self.config: GptInitModelParameters = self.model_cls.create_config(model_config)
        self.tokenizer = self.create_tokenizer(self.model_cls, self.config)
        self.special_tokens = self.config.special_tokens
        # self.config.max_seq_len = self.tokenizer.model_max_length
        logging.info(f'max_seq_len:{self.tokenizer.model_max_length}')

        self.eos_token_id = None
        if (isinstance(self.tokenizer, PreTrainedTokenizer)):
            self.eos_token_id = self.tokenizer.eos_token_id
        if (self.eos_token_id == None):
            self.eos_token_id = self.config.special_tokens.eos_token_id

        self.stop_words_id_list = self.config.special_tokens.stop_words_id_list

        render_params = RendererParams(
            model_type=model_type,
            max_seq_len=self.config.max_seq_len,
            eos_token_id=self.eos_token_id,
            stop_word_ids_list=self.stop_words_id_list,
        )

        self._open_ai_request_render = self.chat_renderer = ChatRendererFactory.get_renderer(self.tokenizer, render_params)



    def quantize(self, quant_type_str: str, quantize_config: Dict[str, str], dataset_params: DatasetParams, sample_strategy: SampleStrategy, output_path: str):
        ret_code = 0
        try:
            dataset = DatasetsAdapter.load_dataset(dataset_params)
            if isinstance(dataset, datasets.dataset_dict.DatasetDict):
                dataset = dataset['train']

            dataset = self.stratified_sampling_by_attributes(dataset, sample_strategy)

            quant_type = QUANT_TYPE.from_str(quant_type_str)
            quanter = self.create_quanter(quant_type, quantize_config)

            examples = self.create_tokenized_samples(dataset, dataset_params)

            with Timer() as t:
                # quanter.quantize(examples)
                examples_for_quant: List[Dict[str, torch.Tensor]] = [{'input_ids': input_ids, 'attention_mask':torch.ones_like(input_ids)} for input_ids in examples]
                quanter.quant(examples_for_quant)
            logging.info(f'quantize model use:{t.cost_ms()/1000:.0f}s')
            save_ret = False
            try:
                self._save_quantized_model(quanter, quantize_config, output_path)
                save_ret = True
            except BaseException as e:
                logging.warn(f"save to {output_path} failed, e: {str(e)}")

            if not save_ret:
                raise Exception(f"save to {output_path} failed")

        except BaseException as e:
            logging.info(f'run failed : {e}')
            logging.exception(e)
            ret_code = -1
        finally:
            self.release_temp_resource()
            return ret_code

    @staticmethod
    @timer_wrapper('dataset sample')
    def stratified_sampling_by_attributes(dataset, sample_strategy: SampleStrategy):
        attributes = sample_strategy.distinct_attrs
        if sample_strategy.sample_size > dataset.num_rows:
            return dataset

        sample_size = sample_strategy.sample_size
        if not attributes:
            indices = random.sample(range(dataset.num_rows), sample_size)
            return dataset.select(indices)

        # 1. 创建一个组合键，它将属性列表中的所有属性结合起来
        def group_key(example):
            return {'group_key': '+'.join([str(example[attr]) for attr in attributes])}

        # 2. 为每个样本添加组合键
        dataset = dataset.map(group_key)

        # 3. 获取每个组合键的计数
        group_counts =  Counter(dataset['group_key'])
        total_count = dataset.num_rows

        # 4. 计算每个组合键的采样比例，并确定应该采样的数量
        proportional_sample_sizes = {
            group_key: math.ceil(sample_size * count / total_count) for group_key, count in group_counts.items()
        }

        # 5. 补偿四舍五入造成的样本数量差异
        while sum(proportional_sample_sizes.values()) > sample_size:
            # 从计数最多的组开始减少
            group_key_to_reduce = max(proportional_sample_sizes, key=lambda k: (proportional_sample_sizes[k], group_counts[k]))
            proportional_sample_sizes[group_key_to_reduce] -= 1

        # 6. 从每个分组采样
        sampled_indices = []
        for group_key, size in proportional_sample_sizes.items():
            group_indices = [i for i, example in enumerate(dataset['group_key']) if example == group_key]
            if len(group_indices) <= size:
                sampled_indices.extend(group_indices)
            else:
                sampled_indices.extend(np.random.choice(group_indices, size=size, replace=False))

        # 7. 从数据集中选择采样的样本
        sampled_dataset = dataset.select(sampled_indices)
        return sampled_dataset

    @timer_wrapper('create tokenizer')
    def create_tokenizer(self, model_cls, params: GptInitModelParameters) -> Union[PreTrainedTokenizerBase]:
        tokenizer = model_cls.get_tokenizer(params)
        return tokenizer

    @timer_wrapper('pretrain load model')
    def create_quanter(self, quant_type: QUANT_TYPE, quantize_config: Dict[str, str]) -> BaseQuanter:
        return QuanterFactory.create_quanter(quant_type, quantize_config, self.model_path, self.offload_folder)


    @timer_wrapper('encode samples')
    def create_tokenized_samples(self, dataset: datasets.Dataset, dataset_params: DatasetParams) -> List[torch.Tensor]:
        # 根据dataset 的类型获取
        dataset_type:Optional[DatasetType] = DatasetsAdapter.parse_dataset_type(dataset, dataset_params)
        if not dataset_type:
            raise TypeError(f'not support this dataset:{dataset}]')

        if dataset_type == DatasetType.RTP_LLM_ACCESS_LOG:
            return self._encode_ft_access_log(dataset)
        elif dataset_type == DatasetType.RTP_LLM_ACCESS_LOG_JSON_STR:
            return self._encode_ft_acccess_log_json_str(dataset)
        elif dataset_type == DatasetType.TEXT:
            return self._encode_text(dataset)

    @timer_wrapper('release resource')
    def release_temp_resource(self):
        def rm_path(directory):
            if os.path.exists(directory):
                shutil.rmtree(directory)
                logging.info(f"The directory {directory} has been removed successfully")
            else:
                logging.info(f"The directory {directory} does not exist, nothing to remove")
        rm_path(self.offload_folder)

    @timer_wrapper('save quantized model')
    def _save_quantized_model(self, quanter: BaseQuanter, quantize_config: Dict[str, str],  output_path: str):
        output_path = fetch_remote_file_to_local(output_path, MountRwMode.RWMODE_RW)
        quanter.save_quantized_model(output_path)
        self.tokenizer.save_pretrained(output_path)
        # rewrite config.json
        quantize_config.update({'quant_method': quanter.quant_type().to_str()})
        # load config.json
        config_file = os.path.join(output_path, "config.json")
        logging.info(f'rewrite config_file:{config_file}')
        with open(config_file) as f:
            config = json.load(f)
        quantize_config.update(config.get('quantization_config', {}))
        config.update({'quantization_config': quantize_config})
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4, sort_keys=True, ensure_ascii=False)
        # cp tokenizer and model

        # touch done
        done_file = os.path.join(output_path, 'done')
        with open(done_file, 'w') as f:
            pass

    def _encode_ft_access_log(self, dataset: datasets.Dataset) -> List[torch.Tensor]:
        samples = []
        for data in dataset:
            request_json_str:str = data['request.request_json']
            try:
                request = json.loads(request_json_str)
            except Exception:
                continue

            if not request:
                continue

            response_json_str: str = data['response.responses']
            if not response_json_str:
                continue
            try:
                responses = json.loads(response_json_str)
            except Exception:
                continue
            if not responses:
                continue

            response = responses[0]

            if request.get('messages'):
                token_ids = self._encode_openai_request(request, response)
            else:
                token_ids = self._encode_pipeline_request(request, response)
            if token_ids is not None:
                samples.append(token_ids)

        return samples

    def _encode_ft_acccess_log_json_str(self, dataset: datasets.Dataset) -> List[torch.Tensor]:
        # DOTO(luoli.hn) 某些任务需要根据指定字段进行采样
        samples = []
        for data in dataset:
            raw_data_str = data['text']
            try:
                raw_data = json.loads(raw_data_str)
            except Exception:
                continue
            request = raw_data.get('request', {}).get('request_json', None)
            response = raw_data.get('response', {}).get('responses', None)
            if not request or not response:
                continue

            if 'messages' in request:
                token_ids = self._encode_openai_request(request, response)
            else:
                token_ids = self._encode_pipeline_request(response, request)
            if token_ids is not None:
                samples.append(token_ids)
        return samples


    def _encode_text(self, dataset: datasets.Dataset) -> List[torch.Tensor]:
        samples = []
        for data in dataset:
            text_data = data['text']
            is_chatml = False
            try:
                text_data = json.loads(text_data)
                is_chatml = 'type' in text_data and text_data['type'] == 'chatml'
            except Exception:
                pass
            if is_chatml:
                text_data.update({'request_format':'chatapi'})
                inference_request, remain_args = RequestExtractor(GenerateConfig()).extract_request(text_data)
                token_ids = DefaultPlugin.process_encode_func(inference_request.input_texts[0], inference_request.generate_configs[0].dict(), self.special_tokens, self.tokenizer)
            else:
                assert isinstance(text_data, str)
                token_ids = self.tokenizer.encode(text_data)
            if token_ids and len(token_ids) < self.config.max_seq_len:
                samples.append(torch.tensor(token_ids, dtype=torch.int))
                logging.info(f'sample length: {len(samples[-1])}')
        return samples

    def _encode_openai_request(self, request: Dict[str, Any], response: Dict[str, Any]) -> torch.Tensor:
        # concat request & response
        response_choices = response[0].get('choices')
        request['messages'].extend([choice['message'] for choice in response_choices])
        chat_request = ChatCompletionRequest.model_validate(request)

        rendered_input = self.chat_renderer.render_chat(chat_request)
        input_ids = torch.tensor(rendered_input.input_ids, dtype=torch.int)
        return input_ids

    def _encode_pipeline_request(self, request: Dict[str, Any], response: Dict[str, Any]) -> None:
        inference_request, remain_args = RequestExtractor(GenerateConfig()).extract_request(request)
        if inference_request.batch_infer or inference_request.num_return_sequences > 1:
            return None

        if inference_request.generate_configs[0].request_format == RequestFormat.CHAT_API:
            response_str = response.get('response')[0]
            if isinstance(inference_request.input_texts[0], str):
                inference_request.input_texts[0] = json.loads(inference_request.input_texts[0])
            inference_request.input_texts[0].append({'role': 'assistant', 'content': response_str})
        else:
            inference_request.input_texts[0] = inference_request.input_texts[0] + response.get('response')[0]
        # 调用DefaultPlugin encode
        token_ids = DefaultPlugin.process_encode_func(inference_request.input_texts[0], inference_request.generate_configs[0].dict(), self.special_tokens, self.tokenizer)
        return torch.tensor(token_ids, dtype=torch.int)

def main():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Quantize model weights.')

    # 添加参数
    parser.add_argument('--pretrained_model_dir', type=str, help='Pretrained model path')
    parser.add_argument('--output_dir_base', type=str, help='Output base folder')
    parser.add_argument('--dataset', type=str, help='Output base folder')
    parser.add_argument('--quant_type', type=str, help='Quant type.')
    parser.add_argument('--dataset_format', type=str, default='', help='[Optinal] dataset_format for load_dataset, when dataset is file.')
    parser.add_argument('--dataset_load_args', type=str, default='{}', help='[Optinal] Json desc the Args : used for load dataset')
    parser.add_argument('--model_type', type=str, default= '', help='[Optinal] the model_type to be quantized.')
    parser.add_argument('--quant_config', type=str, help='Json desc the quantization config')
    parser.add_argument('--workdir_path', type=str, default = None, help='Json desc the quantization config')
    parser.add_argument('--sample_strategy', type=str, default='{}', help='Strategy for sample dateset will be used to quantize model')

    # 解析参数
    args = parser.parse_args()
    weights_quantizer = WeightsQuantizer(args.pretrained_model_dir, args.model_type, args.workdir_path)

    dataset_params = DatasetParams(source=args.dataset,
                                   data_format=args.dataset_format,
                                   load_args=json.loads(args.dataset_load_args))
    sample_strategy = SampleStrategy(**json.loads(args.sample_strategy))
    ret_code = weights_quantizer.quantize(args.quant_type, json.loads(args.quant_config), dataset_params, sample_strategy, args.output_dir_base)
    exit(ret_code)

if __name__ == '__main__':
    # logging.config.dictConfig(LOGGING_CONFIG)
    main()

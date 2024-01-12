import os
import json
import logging
from typing import Any, Dict, Type, Union,  Optional

import sys
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))

from maga_transformer.models.base_model import BaseModel, ModelConfig
from maga_transformer.async_decoder_engine.async_model import AsyncModel
from maga_transformer.tools.api.hf_model_helper import get_model_info_from_hf
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.utils.dump_config_utils import dump_model_to_table
from maga_transformer.utils.fuser import fetch_remote_file_to_local
from maga_transformer.utils.util import WEIGHT_TYPE, get_weight_type_from_env, get_sp_weight_type_from_env

from maga_transformer.model_factory_register import _model_factory

class ModelFactory:
    @staticmethod
    def get_config_json(ckpt_path: str):
        assert os.path.isdir(ckpt_path)
        config_json_path = os.path.join(ckpt_path, 'config.json')
        assert os.path.isfile(config_json_path)
        with open(config_json_path, "r", encoding="utf-8") as reader:
            text = reader.read()
            return json.loads(text)

    @staticmethod
    def get_weight_cls(model_type: str):
        global _model_factory
        model_cls = _model_factory[model_type]
        return model_cls.get_weight_cls()

    @staticmethod
    def get_model_cls(model_type: str):
        global _model_factory
        model_cls = _model_factory[model_type]
        return model_cls

    @staticmethod
    def _create_model(model_config: ModelConfig):
        global _model_factory
        if model_config.model_type not in _model_factory:
            raise Exception(f"model {model_config.model_type} not registered!")
        model_cls = _model_factory[model_config.model_type]
        config: GptInitModelParameters = model_cls.create_config(model_config)
        model = model_cls.from_config(config)
        dump_model_to_table(ModelFactory.model_config_json(model_cls, model_config, config))
        return model
    
    #TODO: remove model_config, get all info from gpt_config
    @staticmethod
    def model_config_json(model_cls: Type[Any], model_config: ModelConfig, config: GptInitModelParameters) -> Dict[str, Any]:
        config_json = {
            "model_type": model_cls.__name__,
            "act_type": str(model_config.act_type),
            "weight_type": str(model_config.weight_type),
            "max_seq_len": config.max_seq_len,
            "use_sparse_head": config.is_sparse_head,
            "use_multi_task_prompt": config.multi_task_prompt,
            "use_medusa": config.use_medusa,
            "lora_infos": config.lora_infos
        }
        return config_json

    @staticmethod
    def from_model_type(model_config: ModelConfig, sp_model_config: Optional[ModelConfig] = None) -> Union[AsyncModel, BaseModel]:
        model = ModelFactory._create_model(model_config)
        if model_config.async_mode:
            sp_model = None if sp_model_config is None else ModelFactory._create_model(sp_model_config)
            model = AsyncModel(model, sp_model)
        return model

    @staticmethod
    def from_huggingface(model_path_or_name: str, revision: Optional[str] = None, model_config: ModelConfig = ModelConfig()):
        model_path, model_type = get_model_info_from_hf(model_path_or_name, revision)
        new_model_config = ModelConfig(
            model_type=model_type,
            ckpt_path=model_path,
            tokenizer_path=model_path,
            async_mode=model_config.async_mode,
            weight_type=model_config.weight_type,
            act_type=model_config.act_type,
            max_seq_len=model_config.max_seq_len,
            seq_size_per_block=model_config.seq_size_per_block,
            gen_num_per_circle=model_config.gen_num_per_circle,
            ptuning_path=model_config.ptuning_path,
            lora_infos=model_config.lora_infos
        )
        return ModelFactory.from_model_type(new_model_config)

    @staticmethod
    def create_from_env():
        tokenizer_path = os.environ["TOKENIZER_PATH"]
        ckpt_path = os.environ["CHECKPOINT_PATH"]
        lora_infos = os.environ.get("LORA_INFO", None)

        extra_data_path = os.environ.get('EXTRA_DATA_PATH', "")
        if extra_data_path:
            extra_data_path = fetch_remote_file_to_local(extra_data_path)
            os.environ['LOCAL_EXTRA_DATA_PATH'] = extra_data_path

        tokenizer_path = fetch_remote_file_to_local(tokenizer_path)

        ckpt_path = fetch_remote_file_to_local(ckpt_path)
        if lora_infos is not None:
            logging.info(f"lora_infos is {lora_infos}")
            lora_infos = json.loads(lora_infos)
            for lora_name, lora_path in lora_infos.items():
                lora_infos[lora_name] = fetch_remote_file_to_local(lora_path)

        logging.info(f"load model from tokenizer_path: {tokenizer_path}, ckpt_path: {ckpt_path}")
        model_type = os.environ["MODEL_TYPE"]
        async_mode = bool(int(os.environ.get("ASYNC_MODE", "0")))

        weight_type: WEIGHT_TYPE = get_weight_type_from_env(os.environ)
        act_type = weight_type if weight_type in [ WEIGHT_TYPE.FP16, WEIGHT_TYPE.BF16] else WEIGHT_TYPE.FP16
        ACT_TYPE = "ACT_TYPE"
        if os.environ.get(ACT_TYPE, None):
            act_type = WEIGHT_TYPE.from_str(os.environ.get(ACT_TYPE))

        max_seq_len = int(os.environ.get("MAX_SEQ_LEN", "0"))

        ptuning_path = None
        if 'PTUNING_PATH' in os.environ:
            ptuning_path = os.environ['PTUNING_PATH']
            ptuning_path = fetch_remote_file_to_local(ptuning_path)

        model_config = ModelConfig(model_type=model_type,
                                   ckpt_path=ckpt_path,
                                   tokenizer_path=tokenizer_path,
                                   async_mode=async_mode,
                                   weight_type=weight_type,
                                   act_type=act_type,
                                   max_seq_len=max_seq_len,
                                   lora_infos=lora_infos,
                                   ptuning_path=ptuning_path)
        # speculative model params
        sp_model_config = None
        sp_model_type = os.environ.get("SP_MODEL_TYPE", None)
        if sp_model_type is not None:
            if not async_mode:
                raise Exception("SP_CONFIG should only be used in aysnc mode")
            logging.info("use sp model")
            sp_ckpt_path = fetch_remote_file_to_local(os.environ['SP_CHECKPOINT_PATH'])
            logging.info(f"load sp model from ckpt_path: {sp_ckpt_path}")

            gen_num_per_circle = int(os.environ.get('GEN_NUM_PER_CIRCLE', '5'))

            sp_weight_type = get_sp_weight_type_from_env(os.environ)
            sp_act_type = sp_weight_type if sp_weight_type in [ WEIGHT_TYPE.FP16, WEIGHT_TYPE.BF16] else weight_type
            SP_ACT_TYPE = "SP_ACT_TYPE"
            if os.environ.get(SP_ACT_TYPE, None):
                sp_act_type = WEIGHT_TYPE.from_str(os.environ.get(SP_ACT_TYPE))

            sp_model_config = ModelConfig(model_type=sp_model_type,
                                          ckpt_path=sp_ckpt_path,
                                          tokenizer_path=tokenizer_path,
                                          lora_infos=None,
                                          async_mode=False,
                                          weight_type=sp_weight_type,
                                          act_type=sp_act_type,
                                          max_seq_len=max_seq_len,
                                          gen_num_per_circle=gen_num_per_circle)

        model = ModelFactory.from_model_type(model_config, sp_model_config)

        if 'GENERATION_CONFIG_PATH' in os.environ:
            model.default_generate_config.update(
                json.load(open(os.path.join(os.environ['GENERATION_CONFIG_PATH'], 'generation_config.json')))
            )
            logging.info(f"load generate config:{os.environ['GENERATION_CONFIG_PATH']}/generation_config.json: \n\
                         {json.dumps(model.default_generate_config, indent=4)}"
            )
        return model

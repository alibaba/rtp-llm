import os
import json
import logging
import torch
from typing import Any, Dict, Type, Union,  Optional

import sys
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))

from maga_transformer.models.base_model import BaseModel, ModelConfig
from maga_transformer.models.propose_model.propose_model import ProposeModel
from maga_transformer.async_decoder_engine.async_model import AsyncModel
from maga_transformer.tools.api.hf_model_helper import get_model_info_from_hf
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters, ConfigMode
from maga_transformer.utils.dump_config_utils import dump_model_to_table
from maga_transformer.utils.fuser import fetch_remote_file_to_local
from maga_transformer.utils.util import check_with_info
from maga_transformer.models.multimodal.multimodal_mixin import MultiModalMixin
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.utils.weight_type import WEIGHT_TYPE, get_weight_type_from_env, get_propose_weight_type_from_env

from maga_transformer.model_factory_register import _model_factory

class ModelFactory:
    @staticmethod
    def get_config_json(ckpt_path: str):
        check_with_info(os.path.isdir(ckpt_path), f"{ckpt_path} check os.isdir failed")
        config_json_path = os.path.join(ckpt_path, 'config.json')
        check_with_info(os.path.isfile(config_json_path), f"{config_json_path} check os.isdir failed")
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
    def create_gpt_init_config(model_config: ModelConfig):
        global _model_factory
        if model_config.model_type not in _model_factory:
            raise Exception(f"model type {model_config.model_type} not registered!")
        model_cls = _model_factory[model_config.model_type]
        config: GptInitModelParameters = model_cls.create_config(
            model_config, parallel_info=g_parallel_info, config_mode=ConfigMode.SimpleMode)
        config.model_name = model_cls.__name__
        if issubclass(model_cls, MultiModalMixin):
            config.is_multimodal = True

        return model_cls, config

    @staticmethod
    def _create_model(model_config: ModelConfig):
        global _model_factory
        if model_config.model_type not in _model_factory:
            raise Exception(f"model type {model_config.model_type} not registered!")
        model_cls = _model_factory[model_config.model_type]
        config: GptInitModelParameters = model_cls.create_config(model_config)
        model = model_cls.from_config(config)
        dump_model_to_table(ModelFactory.model_config_json(model_cls, model_config, config))
        return model

    @staticmethod
    def _create_sp_model(score_model_gpt_config: GptInitModelParameters, model_config: ModelConfig):
        model = None
        global _model_factory
        if model_config.sp_type == "vanilla" or model_config.sp_type == "mtp":
            if model_config.model_type not in _model_factory:
                raise Exception(f"model type {model_config.model_type} not registered!")
            if model_config.model_type == "deepseek-v3-mtp" or model_config.model_type == "mixtbstars-mtp":
                logging.warning(f"create sp model type is {model_config.model_type}, so change the sp type to mtp")
                model_config.sp_type = "mtp"
            model_cls = _model_factory[model_config.model_type]
            # propose model's max seq len must be equal to score model's max seq len
            model_config.max_seq_len = score_model_gpt_config.max_seq_len
            config: GptInitModelParameters = model_cls.create_config(model_config)
            gpt_model = model_cls.from_config(config)
            dump_model_to_table(ModelFactory.model_config_json(model_cls, model_config, config))
            model = ProposeModel(model_config.sp_type, model_config.gen_num_per_circle, gpt_model)
        elif model_config.sp_type == "deterministic":
            model = ProposeModel(model_config.sp_type, model_config.gen_num_per_circle)
        elif model_config.sp_type == "eagle":
            raise NotImplementedError
        return model

    #TODO: remove model_config, get all info from gpt_config
    @staticmethod
    def model_config_json(model_cls: Type[Any], model_config: ModelConfig, config: GptInitModelParameters) -> Dict[str, Any]:
        config_json = {
            "model_type": model_cls.__name__,
            "act_type": str(model_config.act_type),
            "max_seq_len": config.max_seq_len,
            "use_sparse_head": config.is_sparse_head,
            "use_multi_task_prompt": config.multi_task_prompt,
            "lora_infos": config.lora_infos
        }
        return config_json

    @staticmethod
    def from_model_config(model_config: ModelConfig, propose_model_config: Optional[ModelConfig] = None) -> AsyncModel:
        model = ModelFactory._create_model(model_config)
        if model_config.model_type == 'fake_model' or model.config.vit_separation == 1:
            return model
        propose_model = None if propose_model_config is None else ModelFactory._create_sp_model(model.config, propose_model_config)
        if propose_model:
            logging.info("set enable_speculative_decoding")
            model.config.enable_speculative_decoding = True
        model = AsyncModel(model, propose_model)
        if propose_model:
            logging.info("create propose model done")
        logging.info("create rpc model done")
        return model

    @staticmethod
    def from_huggingface(model_path_or_name: str, revision: Optional[str] = None, model_config: ModelConfig = ModelConfig()):
        model_path, model_type = get_model_info_from_hf(model_path_or_name, revision)
        new_model_config = model_config
        new_model_config = new_model_config._replace(model_type=model_type, ckpt_path=model_path, tokenizer_path=model_path)
        return ModelFactory.from_model_config(new_model_config)

    @staticmethod
    def create_normal_model_config():
        model_type = os.environ["MODEL_TYPE"]
        ckpt_path = os.environ["CHECKPOINT_PATH"]
        tokenizer_path = os.environ.get("TOKENIZER_PATH", ckpt_path)
        lora_infos = os.environ.get("LORA_INFO", "{}")
        max_seq_len = int(os.environ.get("MAX_SEQ_LEN", "0"))
        seq_size_per_block = int(os.environ.get("SEQ_SIZE_PER_BLOCK", "8"))

        tokenizer_path = fetch_remote_file_to_local(tokenizer_path)
        ckpt_path = fetch_remote_file_to_local(ckpt_path)

        extra_data_path = os.environ.get('EXTRA_DATA_PATH', "")
        if extra_data_path:
            extra_data_path = fetch_remote_file_to_local(extra_data_path)
            os.environ['LOCAL_EXTRA_DATA_PATH'] = extra_data_path

        ptuning_path = None
        if 'PTUNING_PATH' in os.environ:
            ptuning_path = os.environ['PTUNING_PATH']
            ptuning_path = fetch_remote_file_to_local(ptuning_path)

        lora_infos = json.loads(lora_infos)
        for lora_name, lora_path in lora_infos.items():
            lora_infos[lora_name] = fetch_remote_file_to_local(lora_path)

        logging.info(f"load model from tokenizer_path: {tokenizer_path}, ckpt_path: {ckpt_path}, lora_infos: {lora_infos}, ptuning_path: {ptuning_path}")

        weight_type: WEIGHT_TYPE = get_weight_type_from_env(os.environ)
        act_type = weight_type if weight_type in [ WEIGHT_TYPE.FP16, WEIGHT_TYPE.BF16] else WEIGHT_TYPE.FP16

        # TODO(xinfei.sxf) fix this
        ACT_TYPE = "ACT_TYPE"
        if os.environ.get(ACT_TYPE, None):
            act_type = WEIGHT_TYPE.from_str(os.environ.get(ACT_TYPE))
        model_config = ModelConfig(model_type=model_type,
                                   ckpt_path=ckpt_path,
                                   tokenizer_path=tokenizer_path,
                                   weight_type=weight_type,
                                   act_type=act_type,
                                   max_seq_len=max_seq_len,
                                   seq_size_per_block=seq_size_per_block,
                                   lora_infos=lora_infos,
                                   ptuning_path=ptuning_path)

        return model_config

    @staticmethod
    def create_propose_model_config(normal_model_config: ModelConfig):
        propose_model_config = None

        sp_type = os.environ.get("SP_TYPE", None)
        if sp_type == "vanilla" or sp_type == "mtp":
            logging.info("use vanilla speculative model")
            propose_model_type = os.environ.get("SP_MODEL_TYPE", None)
            gen_num_per_circle = int(os.environ.get('GEN_NUM_PER_CIRCLE', '5'))
            propose_ckpt_path = fetch_remote_file_to_local(os.environ['SP_CHECKPOINT_PATH'])
            logging.info(f"load propose model from ckpt_path: {propose_ckpt_path}")

            propose_weight_type = get_propose_weight_type_from_env(os.environ)
            propose_act_type = propose_weight_type if propose_weight_type in [WEIGHT_TYPE.FP16, WEIGHT_TYPE.BF16] else WEIGHT_TYPE.FP16
            SP_ACT_TYPE = "SP_ACT_TYPE"
            if os.environ.get(SP_ACT_TYPE, None):
                propose_act_type = WEIGHT_TYPE.from_str(os.environ.get(SP_ACT_TYPE))

            propose_model_config = ModelConfig(model_type=propose_model_type,
                                          ckpt_path=propose_ckpt_path,
                                          tokenizer_path=normal_model_config.tokenizer_path,
                                          lora_infos=None,
                                          weight_type=propose_weight_type,
                                          act_type=propose_act_type,
                                          max_seq_len=normal_model_config.max_seq_len,
                                          gen_num_per_circle=gen_num_per_circle,
                                          sp_type=sp_type)
        elif sp_type == "deterministic":
            gen_num_per_circle = int(os.environ.get('GEN_NUM_PER_CIRCLE', '5'))
            propose_model_config = ModelConfig(sp_type=sp_type,
                                               gen_num_per_circle=gen_num_per_circle)
            logging.info("use deterministic speculative model")
        elif sp_type == "eagle":
            logging.info("use eagle speculative model")
            raise NotImplementedError

        return propose_model_config

    @staticmethod
    def load_default_generate_config(model: Union[BaseModel, AsyncModel]):
        if 'GENERATION_CONFIG_PATH' in os.environ:
            model.default_generate_config.update(
                json.load(open(os.path.join(os.environ['GENERATION_CONFIG_PATH'], 'generation_config.json')))
            )
            logging.info(f"load generate config:{os.environ['GENERATION_CONFIG_PATH']}/generation_config.json: \n\
                         {json.dumps(model.default_generate_config.model_dump(), indent=4)}"
            )

    @staticmethod
    def create_from_env():
        normal_model_config = ModelFactory.create_normal_model_config()
        propose_model_config = ModelFactory.create_propose_model_config(normal_model_config)
        model = ModelFactory.from_model_config(normal_model_config, propose_model_config)
        ModelFactory.load_default_generate_config(model)

        return model

    @staticmethod
    def create_from_module(ref_module: torch.nn.Module):
        normal_model_config = ModelFactory.create_normal_model_config()
        normal_model_config.add_ref_module(ref_module)
        model = ModelFactory.from_model_config(normal_model_config)
        ModelFactory.load_default_generate_config(model)

        return model

    @staticmethod
    def create_from_dict(ref_dict: Dict[str, torch.Tensor]):
        normal_model_config = ModelFactory.create_normal_model_config()
        normal_model_config.add_ref_dict(ref_dict)
        model = ModelFactory.from_model_config(normal_model_config)
        ModelFactory.load_default_generate_config(model)

        return model

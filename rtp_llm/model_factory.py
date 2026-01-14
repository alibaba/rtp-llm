import json
import logging
import os
import sys
from typing import Any, Dict, Optional, Type, Union

import torch

from rtp_llm.async_decoder_engine.base_engine import BaseEngine
from rtp_llm.config.py_config_modules import StaticConfig

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))

from rtp_llm.config.gpt_init_model_parameters import ConfigMode, GptInitModelParameters
from rtp_llm.distribute.gang_info import get_gang_info
from rtp_llm.distribute.worker_info import g_parallel_info
from rtp_llm.model_factory_register import _model_factory
from rtp_llm.models.multimodal.multimodal_mixin import MultiModalMixin
from rtp_llm.tools.api.hf_model_helper import get_model_info_from_hf
from rtp_llm.utils.base_model_datatypes import ModelConfig
from rtp_llm.utils.dump_config_utils import dump_model_to_table
from rtp_llm.utils.fuser import fetch_remote_file_to_local
from rtp_llm.utils.util import check_with_info
from rtp_llm.utils.weight_type import WEIGHT_TYPE


class ModelFactory:
    @staticmethod
    def get_config_json(ckpt_path: str):
        check_with_info(os.path.isdir(ckpt_path), f"{ckpt_path} check os.isdir failed")
        config_json_path = os.path.join(ckpt_path, "config.json")
        check_with_info(
            os.path.isfile(config_json_path),
            f"{config_json_path} check os.isdir failed",
        )
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
    def create_frontend_config(model_config: ModelConfig):
        config = GptInitModelParameters(0, 0, 0, 0, 0)
        config.update_common(
            ckpt_path=model_config.ckpt_path,
            tokenizer_path=model_config.tokenizer_path,
            quantization=model_config.quantization,
            data_type=model_config.act_type,
            kv_cache_type=model_config.kv_cache_type,
            max_seq_len=model_config.max_seq_len,
            seq_size_per_block=model_config.seq_size_per_block,
            gen_num_per_circle=model_config.gen_num_per_circle,
            lora_infos=model_config.lora_infos,
            ptuning_path=model_config.ptuning_path,
            ref_module=model_config.ref_module,
            ref_dict=model_config.ref_dict,
            parallel_info=g_parallel_info,
            gang_info=get_gang_info(),
            config_mode=ConfigMode.SimpleMode,
        )
        config.seq_size_per_block = model_config.seq_size_per_block
        config.update_config_with_custom_modal(model_config.ckpt_path)

        return config

    @staticmethod
    def _create_model(model_config: ModelConfig):
        global _model_factory
        if model_config.model_type not in _model_factory:
            raise Exception(f"model type {model_config.model_type} not registered!")
        model_cls = _model_factory[model_config.model_type]
        config: GptInitModelParameters = model_cls.create_config(model_config)
        
        # Dynamic mixin for custom_modal
        if getattr(config, "custom_modal", None) and not issubclass(model_cls, MultiModalMixin):
            class CustomModalWrapper(model_cls, MultiModalMixin):
                def _init_multimodal(self, config):
                    pass
            
            # Disguise the wrapper as the original class to maintain compatibility
            # with logging, config dumping, and registry lookups by name.
            CustomModalWrapper.__name__ = model_cls.__name__
            CustomModalWrapper.__qualname__ = model_cls.__qualname__
            
            model_cls = CustomModalWrapper
            logging.info(f"Dynamically mixed in MultiModalMixin for custom_modal support on {config.model_name}")

        config.model_name = model_cls.__name__
        model = model_cls.from_config(config)
        dump_model_to_table(
            ModelFactory.model_config_json(model_cls, model_config, config)
        )
        return model

    @staticmethod
    def _create_sp_model(
        score_model_gpt_config: GptInitModelParameters, model_config: ModelConfig
    ):
        from rtp_llm.models.propose_model.propose_model import ProposeModel

        model = None
        global _model_factory
        if (
            model_config.sp_type == "vanilla"
            or model_config.sp_type == "mtp"
            or model_config.sp_type == "eagle3"
            or model_config.sp_type == "eagle"
        ):
            if model_config.model_type not in _model_factory:
                raise Exception(f"model type {model_config.model_type} not registered!")
            if (
                model_config.model_type == "deepseek-v3-mtp"
                or model_config.model_type == "mixtbstars-mtp"
            ):
                logging.warning(
                    f"create sp model type is {model_config.model_type}, so change the sp type to mtp"
                )
                model_config.sp_type = "mtp"
            if model_config.model_type == "qwen_3_moe-mtp":
                logging.warning(
                    f"create sp model type is {model_config.model_type}, so change the sp type to eagle3"
                )
                model_config.sp_type = "eagle3"
            model_cls = _model_factory[model_config.model_type]
            # propose model's max seq len must be equal to score model's max seq len
            model_config.max_seq_len = score_model_gpt_config.max_seq_len
            config: GptInitModelParameters = model_cls.create_config(model_config)
            gpt_model = model_cls.from_config(config)
            dump_model_to_table(
                ModelFactory.model_config_json(model_cls, model_config, config)
            )
            model = ProposeModel(
                model_config.sp_type, model_config.gen_num_per_circle, gpt_model
            )
        elif model_config.sp_type == "deterministic":
            model = ProposeModel(model_config.sp_type, model_config.gen_num_per_circle)
        return model

    # TODO: remove model_config, get all info from gpt_config
    @staticmethod
    def model_config_json(
        model_cls: Type[Any], model_config: ModelConfig, config: GptInitModelParameters
    ) -> Dict[str, Any]:
        config_json = {
            "model_type": model_cls.__name__,
            "act_type": str(model_config.act_type),
            "max_seq_len": config.max_seq_len,
            "use_sparse_head": config.is_sparse_head,
            "use_multi_task_prompt": config.multi_task_prompt,
            "lora_infos": config.lora_infos,
        }
        return config_json

    @staticmethod
    def from_model_config(
        model_config: ModelConfig,
        propose_model_config: Optional[ModelConfig] = None,
        gang_info=None,
    ) -> BaseEngine:
        from rtp_llm.async_decoder_engine.engine_creator import create_engine

        model = ModelFactory._create_model(model_config)
        if model_config.model_type == "fake_model" or model.config.vit_separation == 1:
            return model
        propose_model = (
            None
            if propose_model_config is None
            else ModelFactory._create_sp_model(model.config, propose_model_config)
        )
        if propose_model:
            logging.info("set enable_speculative_decoding")
            model.config.enable_speculative_decoding = True
        engine = create_engine(model, propose_model, gang_info)
        engine.start()
        if propose_model:
            logging.info("create propose model done")
        logging.info("create engine done")
        return engine

    @staticmethod
    def from_huggingface(
        model_path_or_name: str,
        revision: Optional[str] = None,
        model_config: ModelConfig = ModelConfig(),
    ):
        model_path, model_type = get_model_info_from_hf(model_path_or_name, revision)
        new_model_config = model_config
        new_model_config = new_model_config._replace(
            model_type=model_type, ckpt_path=model_path, tokenizer_path=model_path
        )
        return ModelFactory.from_model_config(new_model_config)

    @staticmethod
    def creat_standalone_py_model_from_huggingface(
        model_path_or_name: str,
        revision: Optional[str] = None,
        model_config: ModelConfig = ModelConfig(),
    ):
        assert os.environ["LOAD_PYTHON_MODEL"] == "1"
        model_path, model_type = get_model_info_from_hf(model_path_or_name, revision)
        new_model_config = model_config
        new_model_config = new_model_config._replace(
            model_type=model_type, ckpt_path=model_path, tokenizer_path=model_path
        )
        model = ModelFactory._create_model(new_model_config)
        return model

    @staticmethod
    def create_normal_model_config():
        model_type = StaticConfig.model_config.model_type
        ckpt_path = StaticConfig.model_config.checkpoint_path
        tokenizer_path = StaticConfig.model_config.tokenizer_path
        if tokenizer_path == "":
            tokenizer_path = ckpt_path
        lora_infos = StaticConfig.lora_config.lora_info
        max_seq_len = StaticConfig.engine_config.max_seq_len
        seq_size_per_block = StaticConfig.py_kv_cache_config.seq_size_per_block

        tokenizer_path = fetch_remote_file_to_local(tokenizer_path)
        ckpt_path = fetch_remote_file_to_local(ckpt_path)

        extra_data_path = StaticConfig.model_config.extra_data_path
        if extra_data_path:
            extra_data_path = fetch_remote_file_to_local(extra_data_path)
            StaticConfig.model_config.local_extra_data_path = extra_data_path

        ptuning_path = StaticConfig.model_config.ptuning_path
        if ptuning_path is not None:
            ptuning_path = fetch_remote_file_to_local(ptuning_path)

        lora_infos = json.loads(lora_infos)
        for lora_name, lora_path in lora_infos.items():
            lora_infos[lora_name] = fetch_remote_file_to_local(lora_path)

        logging.info(
            f"load model from tokenizer_path: {tokenizer_path}, ckpt_path: {ckpt_path}, lora_infos: {lora_infos}, ptuning_path: {ptuning_path}"
        )

        act_type = None
        # TODO(xinfei.sxf) fix this
        act_type = StaticConfig.model_config.act_type
        kv_cache_type = StaticConfig.py_kv_cache_config.kv_cache_dtype
        quantization = StaticConfig.quantization_config.quantization
        model_config = ModelConfig(
            model_type=model_type,
            ckpt_path=ckpt_path,
            tokenizer_path=tokenizer_path,
            act_type=act_type,
            kv_cache_type=kv_cache_type,
            max_seq_len=max_seq_len,
            seq_size_per_block=seq_size_per_block,
            lora_infos=lora_infos,
            ptuning_path=ptuning_path,
            quantization=quantization,
        )

        return model_config

    @staticmethod
    def create_propose_model_config(normal_model_config: ModelConfig):
        propose_model_config = None

        sp_type = StaticConfig.py_speculative_execution_config.sp_type
        if (
            sp_type == "vanilla"
            or sp_type == "mtp"
            or sp_type == "eagle3"
            or sp_type == "eagle"
        ):
            logging.info("use vanilla speculative model")
            propose_model_type = (
                StaticConfig.py_speculative_execution_config.sp_model_type
            )
            gen_num_per_circle = (
                StaticConfig.py_speculative_execution_config.gen_num_per_circle
            )
            origin_ckpt_path = (
                StaticConfig.py_speculative_execution_config.sp_checkpoint_path
            )
            if origin_ckpt_path is None:
                logging.error("sp is disabled since SP_CHECKPOINT_PATH is not set")
                return None
            propose_ckpt_path = fetch_remote_file_to_local(origin_ckpt_path)
            logging.info(f"load propose model from ckpt_path: {propose_ckpt_path}")

            propose_act_type = WEIGHT_TYPE.from_str(
                StaticConfig.model_config.act_type
            ).to_str()
            quantization = StaticConfig.py_speculative_execution_config.sp_quantization
            kv_cache_type = (
                StaticConfig.py_speculative_execution_config.sp_kv_cache_dtype
            )

            propose_model_config = ModelConfig(
                model_type=propose_model_type,
                ckpt_path=propose_ckpt_path,
                tokenizer_path=normal_model_config.tokenizer_path,
                lora_infos=None,
                act_type=propose_act_type,
                kv_cache_type=kv_cache_type,
                max_seq_len=normal_model_config.max_seq_len,
                gen_num_per_circle=gen_num_per_circle,
                sp_type=sp_type,
                quantization=quantization,
            )
        elif sp_type == "deterministic":
            gen_num_per_circle = (
                StaticConfig.py_speculative_execution_config.gen_num_per_circle
            )
            propose_model_config = ModelConfig(
                sp_type=sp_type, gen_num_per_circle=gen_num_per_circle
            )
            logging.info("use deterministic speculative model")

        return propose_model_config

    @staticmethod
    def load_default_generate_config(engine):
        generation_config_path = StaticConfig.generate_env_config.generation_config_path
        if generation_config_path:
            engine.default_generate_config.update(
                json.load(
                    open(os.path.join(generation_config_path, "generation_config.json"))
                )
            )
            logging.info(
                f"load generate config:{generation_config_path}/generation_config.json: \n\
                         {json.dumps(engine.default_generate_config.model_dump(), indent=4)}"
            )

    @staticmethod
    def create_from_env(gang_info=None) -> BaseEngine:
        from rtp_llm.distribute.gang_info import get_gang_info
        from rtp_llm.utils import aot_compiler

        normal_model_config = ModelFactory.create_normal_model_config()

        # Try to auto-compile AOT model if configured
        try:
            # We need a temporary config object to access GptInitModelParameters logic
            # to resolve custom_modal config properly.
            temp_gpt_config = ModelFactory.create_frontend_config(normal_model_config)
            aot_compiler.try_auto_compile(normal_model_config.ckpt_path, temp_gpt_config)
        except Exception as e:
            logging.warning(f"Auto-compile step failed: {e}. Proceeding with existing artifacts or python fallback.")

        propose_model_config = ModelFactory.create_propose_model_config(
            normal_model_config
        )
        if gang_info is None:
            gang_info = get_gang_info()
        engine = ModelFactory.from_model_config(
            normal_model_config, propose_model_config, gang_info
        )
        ModelFactory.load_default_generate_config(engine)

        return engine

    @staticmethod
    def create_from_module(ref_module: torch.nn.Module) -> BaseEngine:
        normal_model_config = ModelFactory.create_normal_model_config()
        normal_model_config.add_ref_module(ref_module)
        engine = ModelFactory.from_model_config(normal_model_config)
        ModelFactory.load_default_generate_config(engine)

        return engine

    @staticmethod
    def create_from_dict(ref_dict: Dict[str, torch.Tensor]):
        normal_model_config = ModelFactory.create_normal_model_config()
        normal_model_config.add_ref_dict(ref_dict)
        engine = ModelFactory.from_model_config(normal_model_config)
        ModelFactory.load_default_generate_config(engine)

        return engine

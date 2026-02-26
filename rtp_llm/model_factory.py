import json
import logging
import os
import sys
from typing import Any, Dict, Optional, Type, Union

import torch

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))

import rtp_llm.models
from rtp_llm.config.engine_config import EngineConfig, finalize_scheduler_config
from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.model_args import ModelArgs
from rtp_llm.config.model_config import ModelConfig, build_model_config
from rtp_llm.config.py_config_modules import (
    EmbeddingConfig,
    GenerateEnvConfig,
    LoraConfig,
    QuantizationConfig,
    RenderConfig,
    VitConfig,
)
from rtp_llm.model_factory_register import _model_factory
from rtp_llm.ops import ProfilingDebugLoggingConfig, SpeculativeType, VitSeparation
from rtp_llm.utils.util import check_with_info


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
    def _create_model(
        model_config: ModelConfig,
        engine_config: EngineConfig,
        vit_config: Optional[VitConfig] = None,
        merge_lora: bool = False,
    ):
        """Create model from independent config objects.

        All model metadata (template_type, model_name, lora_infos, mm_model_config) is now stored in model_config.

        Args:
            model_config: Model configuration (contains mm_model_config)
            engine_config: Engine configuration
            vit_config: Optional VitConfig (needed for multimodal models)
            merge_lora: Whether to merge LoRA weights
        """
        model_type = model_config.model_type
        model_cls = ModelFactory.get_model_cls(model_type)

        # Get model_name from model_config (default to model class name if not set)
        model_name = model_config.model_name or model_cls.__name__
        model_config.model_name = model_name
        engine_config.runtime_config.model_name = model_name

        model = model_cls.from_config(
            model_config=model_config,
            parallelism_config=engine_config.parallelism_config,
            hw_kernel_config=engine_config.hw_kernel_config,
            kv_cache_config=engine_config.kv_cache_config,
            fmha_config=engine_config.fmha_config,
            moe_config=engine_config.moe_config,
            load_python_model=engine_config.model_specific_config.load_python_model,
            load_method=engine_config.load_config.load_method,
            max_generate_batch_size=engine_config.runtime_config.max_generate_batch_size,
            vit_config=vit_config,
            merge_lora=merge_lora,
            device_resource_config=engine_config.device_resource_config,
        )
        return model

    @staticmethod
    def get_sp_model(
        model_config: ModelConfig,
        propose_model_config: Optional[ModelConfig],
        engine_config: EngineConfig,
    ) -> Optional[Any]:
        """Get and create ProposeModel from engine_config and propose_model_config.

        This function handles sp_type determination and ProposeModel creation logic.

        Args:
            model_config: Main ModelConfig (for max_seq_len alignment)
            propose_model_config: Optional propose ModelConfig
            engine_config: EngineConfig containing sp_config

        Returns:
            ProposeModel instance or None if no propose model needed
        """
        sp_type = engine_config.sp_config.type  # Get SpeculativeType enum value
        if sp_type == SpeculativeType.NONE:
            return None

        gen_num_per_circle = engine_config.sp_config.gen_num_per_cycle

        # Adjust sp_type based on propose model type if needed
        if (
            sp_type == SpeculativeType.VANILLA
            or sp_type == SpeculativeType.MTP
            or sp_type == SpeculativeType.EAGLE3
            or sp_type == SpeculativeType.EAGLE
        ):
            model_type = propose_model_config.model_type
            if model_type == "deepseek-v3-mtp" or model_type == "mixtbstars-mtp":
                logging.warning(
                    f"create sp model type is {model_type}, so change the sp type to mtp"
                )
                engine_config.sp_config.type = SpeculativeType.MTP
                sp_type = SpeculativeType.MTP
            elif model_type == "qwen_3_moe-mtp":
                logging.warning(
                    f"create sp model type is {model_type}, so change the sp type to eagle3"
                )
                engine_config.sp_config.type = SpeculativeType.EAGLE3
                sp_type = SpeculativeType.EAGLE3

            # Need to create GPT model for propose model
            model_cls = ModelFactory.get_model_cls(propose_model_config.model_type)
            # propose model's max seq len must be equal to score model's max seq len
            propose_model_config.max_seq_len = model_config.max_seq_len
            from rtp_llm.models.propose_model.propose_model import ProposeModel

            gpt_model = model_cls.from_config(
                model_config=propose_model_config,
                parallelism_config=engine_config.parallelism_config,
                hw_kernel_config=engine_config.hw_kernel_config,
                kv_cache_config=engine_config.kv_cache_config,
                fmha_config=engine_config.fmha_config,
                moe_config=engine_config.moe_config,
                load_python_model=engine_config.model_specific_config.load_python_model,
                load_method=engine_config.load_config.load_method,
                max_generate_batch_size=engine_config.runtime_config.max_generate_batch_size,
                device_resource_config=engine_config.device_resource_config,
                vit_config=None,  # Propose model doesn't need vit_config
                merge_lora=False,  # Propose model doesn't need merge_lora
            )
            logging.info(f"create propose model {engine_config.sp_config.type}")
            return ProposeModel(sp_type, gen_num_per_circle, gpt_model)
        elif sp_type == SpeculativeType.DETERMINISTIC:
            logging.info(f"create propose model {engine_config.sp_config.type}")
            return ProposeModel(sp_type, gen_num_per_circle)
        else:
            raise ValueError(f"unknown sp_type: {str(sp_type)}")

        return None

    @staticmethod
    def from_model_configs(
        model_config: ModelConfig,
        engine_config: EngineConfig,
        world_info,
        vit_config: Optional[VitConfig] = None,
        merge_lora: bool = False,
        propose_model_config: Optional[ModelConfig] = None,
    ):
        """Create engine from independent config objects, with optional propose model.

        All model metadata (template_type, model_name, lora_infos, mm_model_config) should be set in model_config before calling this method.

        This replaces from_gpt_config() and returns BaseEngine instead of AsyncModel.

        Args:
            model_config: Model configuration (contains mm_model_config)
            engine_config: Engine configuration
            world_info: WorldInfo instance from DistributedServer
            vit_config: Optional VitConfig (needed for multimodal models)
            merge_lora: Whether to merge LoRA weights
            propose_model_config: Optional propose model configuration
            generate_env_config: Optional GenerateEnvConfig for loading default generate config

        Returns:
            BaseEngine instance (RPCEngine or EmbeddingCppEngine)
        """
        model = ModelFactory._create_model(
            model_config=model_config,
            engine_config=engine_config,
            vit_config=vit_config,
            merge_lora=merge_lora,
        )

        model_type = model_config.model_type
        if model_type == "fake_model":
            logging.info("create fake_model")

        if (
            vit_config is not None
            and vit_config.vit_separation == VitSeparation.VIT_SEPARATION_ROLE
        ):
            logging.info("vit role, continue")
            return model

        # Create propose model if provided
        propose_model = ModelFactory.get_sp_model(
            model_config=model_config,
            propose_model_config=propose_model_config,
            engine_config=engine_config,
        )

        # Create engine using create_engine function (replaces AsyncModel)
        alog_conf_path = engine_config.profiling_debug_logging_config.ft_alog_conf_path

        from rtp_llm.async_decoder_engine.engine_creator import create_engine

        engine = create_engine(
            model=model,
            engine_config=engine_config,
            alog_conf_path=alog_conf_path,
            world_info=world_info,
            propose_model=propose_model,
        )
        engine.start()
        if propose_model:
            logging.info("create propose model done")
        logging.info("create engine done")
        return engine

    @staticmethod
    def create_model_config(
        model_args: ModelArgs,
        lora_config: LoraConfig,
        kv_cache_config: KVCacheConfig,
        profiling_debug_logging_config: ProfilingDebugLoggingConfig,
        generate_env_config: Optional[GenerateEnvConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        quantization_config: Optional[QuantizationConfig] = None,
        render_config: Optional[Any] = None,
        eplb_config: Optional[Any] = None,
    ) -> ModelConfig:
        """Create ModelConfig from configuration objects.

        This method handles ModelConfig construction and initialization logic for the main model.

        The flow is:
        1. Call model's _create_config to create ModelConfig with model architecture
        2. Apply ModelArgs to ModelConfig (overwrite with user-provided values)
        3. Build ModelConfig with build_model_config

        Args:
            model_args: ModelArgs containing model configuration
            lora_config: LoraConfig containing LoRA configuration
            kv_cache_config: KVCacheConfig for model config building
            profiling_debug_logging_config: ProfilingDebugLoggingConfig for model config building
            generate_env_config: Optional GenerateEnvConfig for generation settings
            embedding_config: Optional EmbeddingConfig for embedding settings
            quantization_config: Optional QuantizationConfig for quantization settings
            render_config: Optional RenderConfig for renderer factory settings
            eplb_config: Optional EPLBConfig for EPLB settings

        Returns:
            ModelConfig instance for the main model
        """
        model_cls = ModelFactory.get_model_cls(model_args.model_type)
        model_config = model_cls._create_config(model_args.ckpt_path)
        build_model_config(
            model_config=model_config,
            model_args=model_args,
            kv_cache_config=kv_cache_config,
            profiling_debug_logging_config=profiling_debug_logging_config,
            embedding_config=embedding_config,
            quantization_config=quantization_config,
        )

        # Set model metadata fields
        # Set lora_infos from lora_config (direct assignment)
        if lora_config.lora_info:
            lora_infos = json.loads(lora_config.lora_info)
            model_config.lora_infos = lora_infos if lora_infos else {}

        # Set model_name (default to model class name)
        model_config.model_name = model_cls.__name__

        # Set renderer configuration fields
        model_config.generate_env_config = (
            generate_env_config
            if generate_env_config is not None
            else GenerateEnvConfig()
        )
        model_config.render_config = (
            render_config if render_config is not None else RenderConfig()
        )

        # Set eplb_config
        if eplb_config is not None:
            model_config.eplb_config = eplb_config

        logging.info("model_config: %s", model_config.to_string())

        return model_config

    @staticmethod
    def update_engine_config_from_model_config(
        engine_config: EngineConfig,
        model_config: ModelConfig,
    ) -> None:
        """Update EngineConfig based on ModelConfig.

        This method finalizes scheduler config and sets model_name in engine_config.

        Args:
            engine_config: EngineConfig to update
            model_config: ModelConfig containing model information
        """
        # Finalize scheduler config based on ModelConfig (only once, for main model)
        finalize_scheduler_config(
            fifo_scheduler_config=engine_config.runtime_config.fifo_scheduler_config,
            max_seq_len=model_config.max_seq_len,
        )

        # Set model_name to engine_config.runtime_config.model_name (for backward compatibility)
        engine_config.runtime_config.model_name = model_config.model_name

    @staticmethod
    def create_propose_model_config(
        engine_config: EngineConfig,
        model_config: ModelConfig,
        model_args: ModelArgs,
    ) -> Optional[ModelConfig]:
        """Create propose ModelConfig from configuration objects.

        This method handles ModelConfig construction and initialization logic for the propose model.
        The main model_config must be created first, as propose model's max_seq_len must match main model.

        Args:
            engine_config: Already built EngineConfig
            model_config: Main ModelConfig (used for max_seq_len alignment)
            model_args: ModelArgs containing model configuration (used for tokenizer_path, act_type, etc.)

        Returns:
            ModelConfig instance for propose model, or None if not needed
        """
        sp_config = engine_config.sp_config
        if not sp_config.type or sp_config.type == SpeculativeType.NONE:
            return None

        if not sp_config.checkpoint_path:
            return None

        if sp_config.use_new_sp_engine:
            # only support mtp and eagle
            if sp_config.type not in [SpeculativeType.MTP, SpeculativeType.EAGLE]:
                logging.error(
                    f"use_new_sp_engine only support mtp and eagle, but got {sp_config.type.name}"
                )
                raise ValueError(
                    f"use_new_sp_engine only support mtp and eagle, but got {sp_config.type.name}"
                )

        # Create ModelArgs for propose model (reuse main model args, but override ckpt_path)
        propose_model_args = ModelArgs()
        propose_model_args.ckpt_path = sp_config.checkpoint_path
        propose_model_args.tokenizer_path = model_args.tokenizer_path
        propose_model_args.model_type = sp_config.model_type
        propose_model_args.act_type = model_args.act_type
        propose_model_args.mla_ops_type = model_args.mla_ops_type

        # Create propose ModelConfig using _create_config
        propose_model_cls = ModelFactory.get_model_cls(sp_config.model_type)
        propose_model_config = propose_model_cls._create_config(
            sp_config.checkpoint_path
        )
        # Ensure max_seq_len matches main model
        propose_model_config.max_seq_len = model_config.max_seq_len
        propose_model_config.quantization = sp_config.quantization

        logging.info(
            f"load propose model from tokenizer_path: {propose_model_config.tokenizer_path}, "
            f"ckpt_path: {propose_model_config.ckpt_path}, quantization: {propose_model_config.quantization}"
        )

        # Build propose model config (no finalize_scheduler_config for propose model)
        build_model_config(
            model_config=propose_model_config,
            model_args=propose_model_args,
            kv_cache_config=engine_config.kv_cache_config,
            profiling_debug_logging_config=engine_config.profiling_debug_logging_config,
            embedding_config=None,  # Propose model doesn't need embedding_config
        )

        return propose_model_config

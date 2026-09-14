import copy
import json
import logging
import os
import sys
from typing import Any, Dict, Optional, Type, Union

import torch

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))

from rtp_llm.config.engine_config import EngineConfig, finalize_scheduler_config
from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.model_args import ModelArgs
from rtp_llm.config.model_config import ModelConfig, build_model_config
from rtp_llm.config.py_config_modules import (
    EmbeddingConfig,
    GenerateEnvConfig,
    LoraConfig,
    PyEnvConfigs,
    QuantizationConfig,
    RenderConfig,
    VitConfig,
)
from rtp_llm.device.device_type import is_hip
from rtp_llm.model_factory_register import _model_factory, ensure_model_registered
from rtp_llm.ops import (
    DataType,
    KvCacheDataType,
    ProfilingDebugLoggingConfig,
    SpeculativeType,
    TaskType,
    VitSeparation,
)
from rtp_llm.utils.util import check_with_info
from rtp_llm.utils.warmup import configure_warmup


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
        if not ensure_model_registered(model_type):
            raise KeyError(f"model type [{model_type}] is not registered")
        model_cls = _model_factory[model_type]
        return model_cls.get_weight_cls()

    @staticmethod
    def get_model_cls(model_type: str):
        global _model_factory
        if not ensure_model_registered(model_type):
            raise KeyError(f"model type [{model_type}] is not registered")
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
        configure_warmup(
            engine_config.runtime_config.warm_up,
            engine_config.runtime_config.model_warm_up,
        )
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
            load_method=engine_config.load_config.load_method,
            max_generate_batch_size=engine_config.runtime_config.max_generate_batch_size,
            vit_config=vit_config,
            merge_lora=merge_lora,
            device_resource_config=engine_config.device_resource_config,
            force_cpu_load_weights=engine_config.load_config.force_cpu_load_weights,
            loader_recycle_handles=engine_config.load_config.loader_recycle_handles,
            moe_pure_tp_preshard=engine_config.load_config.moe_pure_tp_preshard,
        )
        return model

    @staticmethod
    def get_sp_model(
        model_config: ModelConfig,
        propose_model_config: Optional[ModelConfig],
        engine_config: EngineConfig,
        target_model: Optional[Any] = None,
    ) -> Optional[Any]:
        """Get and create ProposeModel from engine_config and propose_model_config.

        This function handles sp_type determination and ProposeModel creation logic.

        Args:
            model_config: Main ModelConfig (for max_seq_len alignment)
            propose_model_config: Optional propose ModelConfig
            engine_config: EngineConfig containing sp_config
            target_model: Optional already-loaded target model, used as the owner
                of any global weights the draft model borrows

        Returns:
            ProposeModel instance or None if no propose model needed
        """
        configure_warmup(
            engine_config.runtime_config.warm_up,
            engine_config.runtime_config.model_warm_up,
        )
        sp_type = engine_config.sp_config.type  # Get SpeculativeType enum value
        if sp_type == SpeculativeType.NONE:
            return None

        from rtp_llm.models.propose_model.propose_model import ProposeModel

        gen_num_per_circle = engine_config.sp_config.gen_num_per_cycle

        # Adjust sp_type based on propose model type if needed
        if (
            sp_type == SpeculativeType.VANILLA
            or sp_type == SpeculativeType.MTP
            or sp_type == SpeculativeType.EAGLE3
            or sp_type == SpeculativeType.EAGLE
            or sp_type == SpeculativeType.DSPARK
            or sp_type == SpeculativeType.DFLASH
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
            propose_model_config.gen_num_per_cycle = model_config.gen_num_per_cycle

            alias_names = ()
            if target_model is not None:
                alias_names = tuple(
                    model_cls.speculative_weight_alias_names(
                        target_model, propose_model_config
                    )
                )
            if alias_names and target_model.weight is None:
                raise RuntimeError("speculative shared-weight owner is not loaded")

            propose_hw_kernel_config = engine_config.hw_kernel_config
            if (
                sp_type in (SpeculativeType.DSPARK, SpeculativeType.DFLASH)
                and propose_model_config.model_type
                in ("qwen_3_dspark", "qwen_3_dflash")
                and is_hip()
                and propose_hw_kernel_config.use_swizzleA
            ):
                # The target's FP8 PTPC path benefits from ROCm swizzle, but the
                # Qwen3 DSpark checkpoint is BF16.  hipBLASLt on MI308X has no
                # preshuffled BF16 solution for the draft's 5120x5120 GEMMs.
                # Keep the target config unchanged and give the draft an
                # independent raw-layout config for both loading and dispatch.
                propose_hw_kernel_config = copy.deepcopy(propose_hw_kernel_config)
                propose_hw_kernel_config.use_swizzleA = False
                logging.info(
                    "disable ROCm swizzleA for BF16 Qwen3 block-draft propose model"
                )

            gpt_model = model_cls.from_config(
                model_config=propose_model_config,
                parallelism_config=engine_config.parallelism_config,
                hw_kernel_config=propose_hw_kernel_config,
                kv_cache_config=engine_config.kv_cache_config,
                fmha_config=engine_config.fmha_config,
                moe_config=engine_config.moe_config,
                load_method=engine_config.load_config.load_method,
                max_generate_batch_size=engine_config.runtime_config.max_generate_batch_size,
                device_resource_config=engine_config.device_resource_config,
                vit_config=None,  # Propose model doesn't need vit_config
                merge_lora=False,  # Propose model doesn't need merge_lora
                loader_recycle_handles=engine_config.load_config.loader_recycle_handles,
                moe_pure_tp_preshard=engine_config.load_config.moe_pure_tp_preshard,
                weight_alias_owner=target_model if alias_names else None,
                weight_alias_names=alias_names,
            )
            aliased_local_bytes = 0
            for name in alias_names:
                owner_tensor = target_model.weight.get_global_weight(name)
                alias_tensor = gpt_model.weight.get_global_weight(name)
                if (
                    alias_tensor is not owner_tensor
                    or alias_tensor.data_ptr() != owner_tensor.data_ptr()
                ):
                    raise RuntimeError(
                        f"speculative global weight alias {name!r} did not preserve storage identity"
                    )
                aliased_local_bytes += (
                    owner_tensor.numel() * owner_tensor.element_size()
                )
            if alias_names:
                logging.info(
                    "speculative model aliases owner weights: %s; local HBM not duplicated: %.3f GiB",
                    alias_names,
                    aliased_local_bytes / (1024**3),
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
            mm_process_engine: Optional MMProcessEngine instance for multimodal processing in EmbeddingCppEngine

        Returns:
            BaseEngine instance (RPCEngine or EmbeddingCppEngine)
        """
        # Set gen_num_per_cycle on model_config so it flows to AttentionConfigs
        # for RoPE cache sizing in speculative decoding
        model_config.gen_num_per_cycle = engine_config.sp_config.gen_num_per_cycle

        model = ModelFactory._create_model(
            model_config=model_config,
            engine_config=engine_config,
            vit_config=vit_config,
            merge_lora=merge_lora,
        )
        if model_config.task_type == TaskType.LANGUAGE_MODEL:
            engine_config.grammar_config.tokenizer_info_json = (
                model.build_grammar_tokenizer_info()
            )

        model_type = model_config.model_type
        if model_type == "fake_model":
            logging.info("create fake_model")

        logging.info(f"create model finish")

        # Create propose model if provided
        propose_model = ModelFactory.get_sp_model(
            model_config=model_config,
            propose_model_config=propose_model_config,
            engine_config=engine_config,
            target_model=model,
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
        vit_config: Optional[VitConfig] = None,
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
            vit_config=vit_config,
        )
        model_cls._apply_kv_cache_config(model_config, kv_cache_config)
        model_cls._post_build_model_config(model_config)

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
        scheduler_config = engine_config.runtime_config.fifo_scheduler_config
        # Generic MoE executors allocate their fixed-capacity communication
        # buffers while the Python model is constructed. Preserve the finalized
        # scheduler prefill bound on the model config so those buffers cover a
        # full admitted context batch, not just one maximum-length request.
        model_config.moe_prefill_max_tokens_per_rank = min(
            int(scheduler_config.max_context_batch_size)
            * int(model_config.max_seq_len),
            int(scheduler_config.max_batch_tokens_size),
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

        # Current learned-draft SP engine supports MTP, EAGLE, DSpARK and DFlash.
        if sp_config.type not in [
            SpeculativeType.MTP,
            SpeculativeType.EAGLE,
            SpeculativeType.DSPARK,
            SpeculativeType.DFLASH,
        ]:
            logging.error(
                "Speculative engine only supports MTP, EAGLE, DSpARK and DFlash, but got %s",
                sp_config.type.name,
            )
            raise ValueError(
                "Speculative engine only supports MTP, EAGLE, DSpARK and DFlash, but got %s"
                % sp_config.type.name
            )

        # Create ModelArgs for propose model (reuse main model args, but override ckpt_path)
        propose_model_args = ModelArgs()
        propose_model_args.ckpt_path = sp_config.checkpoint_path
        propose_model_args.tokenizer_path = model_args.tokenizer_path
        propose_model_args.model_type = sp_config.model_type
        propose_model_args.act_type = model_args.act_type
        propose_model_args.mla_ops_type = model_args.mla_ops_type
        propose_model_args.enable_fp32_lm_head = model_args.enable_fp32_lm_head

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
        propose_model_cls._apply_kv_cache_config(
            propose_model_config, engine_config.kv_cache_config
        )
        propose_model_cls._post_build_model_config(propose_model_config)

        if sp_config.type == SpeculativeType.DSPARK:
            ModelFactory._setup_dspark_configs(
                sp_config, model_config, propose_model_config
            )
        elif sp_config.type == SpeculativeType.DFLASH:
            ModelFactory._setup_dflash_configs(
                sp_config, model_config, propose_model_config
            )

        return propose_model_config

    @staticmethod
    def _setup_dspark_configs(
        sp_config, model_config: ModelConfig, propose_model_config: ModelConfig
    ) -> None:
        """Validate fixed-width DSpARK and wire target aux-state capture.

        DeepSeek-V4 DSpARK uses a draft query block of exactly ``gamma`` rows
        (one anchor plus ``gamma - 1`` noise tokens). The target verifies
        ``gamma + 1`` rows. ``gen_num_per_cycle`` therefore remains the single
        source of truth for the fixed proposal width in the engine.
        """
        required = {
            "dspark_noise_token_id": propose_model_config.dspark_noise_token_id,
            "dspark_target_layer_ids": propose_model_config.dspark_target_layer_ids,
            "dspark_markov_rank": propose_model_config.dspark_markov_rank,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "sp_type dspark requires draft checkpoint metadata: "
                + ", ".join(missing)
            )

        gamma = int(sp_config.gen_num_per_cycle)
        if gamma <= 0:
            raise ValueError(
                f"dspark requires a positive gen_num_per_cycle, got {gamma}"
            )

        noise_token_id = int(propose_model_config.dspark_noise_token_id)
        # The noise token is consumed by the draft backbone embedding, not by
        # the reduced Markov output head.  Speculators checkpoints may expose
        # a 20K draft output vocabulary while retaining the target-sized input
        # embedding, so validate in the input-token id space.
        # ModelConfig uses zero when the input vocabulary was not set
        # separately; in that case the embedding spans the model vocabulary.
        configured_input_vocab_size = getattr(
            propose_model_config, "input_vocab_size", 0
        )
        input_vocab_size = int(
            configured_input_vocab_size or propose_model_config.vocab_size
        )
        if noise_token_id < 0 or noise_token_id >= input_vocab_size:
            raise ValueError(
                f"invalid dspark_noise_token_id {noise_token_id} for "
                f"input_vocab_size {input_vocab_size}"
            )

        target_layer_ids = [
            int(layer_id) for layer_id in propose_model_config.dspark_target_layer_ids
        ]
        if not target_layer_ids:
            raise ValueError("dspark_target_layer_ids must not be empty")
        if target_layer_ids != sorted(set(target_layer_ids)):
            raise ValueError(
                "dspark_target_layer_ids must be unique and ordered by target "
                f"layer boundary, got {target_layer_ids}"
            )
        invalid_layer_ids = [
            layer_id
            for layer_id in target_layer_ids
            if layer_id < 0 or layer_id >= model_config.num_layers
        ]
        if invalid_layer_ids:
            raise ValueError(
                f"dspark_target_layer_ids {invalid_layer_ids} are out of range "
                f"for target with {model_config.num_layers} layers"
            )

        markov_rank = int(propose_model_config.dspark_markov_rank)
        if markov_rank <= 0:
            raise ValueError(f"invalid dspark_markov_rank: {markov_rank}")

        sp_config.sp_dspark_mask_token_id = noise_token_id
        sp_config.sp_dspark_sample_from_anchor = bool(
            getattr(propose_model_config, "dspark_sample_from_anchor", True)
        )
        # Both models carry the capture ids: the target uses them to capture
        # and to size the shared MTP hidden buffer rows; the draft only needs
        # them for the same row-width derivation (it never captures).
        model_config.capture_aux_hidden_layer_ids = target_layer_ids
        propose_model_config.capture_aux_hidden_layer_ids = target_layer_ids
        logging.info(
            "DSpARK fixed-width wiring: gamma=%d, noise_token_id=%d, "
            "target capture layer ids=%s, markov_rank=%d",
            gamma,
            noise_token_id,
            target_layer_ids,
            markov_rank,
        )

    @staticmethod
    def _setup_dflash_configs(
        sp_config, model_config: ModelConfig, propose_model_config: ModelConfig
    ) -> None:
        """Validate DFlash V1 metadata and wire target hidden capture.

        DFlash uses the existing fixed-block ABI during the initial rollout,
        but has independent checkpoint metadata and no Markov state.
        """
        gamma = int(sp_config.gen_num_per_cycle)
        if gamma not in tuple(range(1, 8)) + (15,):
            raise ValueError(
                "DFlash V1 supports proposal gamma=1..7 or native gamma=15 "
                f"(query width gamma+1), got {gamma}"
            )
        noise_token_id = getattr(propose_model_config, "dflash_mask_token_id", None)
        layer_ids = getattr(propose_model_config, "dflash_target_layer_ids", None)
        layer_types = getattr(propose_model_config, "dflash_layer_types", None)
        window = getattr(propose_model_config, "dflash_sliding_window", None)
        native_block_size = getattr(
            propose_model_config, "dflash_native_block_size", None
        )
        if noise_token_id is None or layer_ids is None or layer_types is None:
            raise ValueError(
                "sp_type dflash requires mask token, target layer ids, and layer types"
            )
        input_vocab = int(
            getattr(propose_model_config, "input_vocab_size", 0)
            or propose_model_config.vocab_size
        )
        noise_token_id = int(noise_token_id)
        if noise_token_id < 0 or noise_token_id >= input_vocab:
            raise ValueError(
                f"invalid dflash_mask_token_id {noise_token_id} for input_vocab_size {input_vocab}"
            )
        layer_ids = [int(layer_id) for layer_id in layer_ids]
        if not layer_ids or layer_ids != sorted(set(layer_ids)):
            raise ValueError("dflash target layer ids must be unique and ordered")
        if any(layer_id < 0 or layer_id >= model_config.num_layers for layer_id in layer_ids):
            raise ValueError(
                f"dflash target layer ids {layer_ids} are invalid for target layers={model_config.num_layers}"
            )
        layer_types = [str(layer_type) for layer_type in layer_types]
        if len(layer_types) != propose_model_config.num_layers:
            raise ValueError(
                "dflash layer_types count must equal draft num_layers: "
                f"{len(layer_types)} != {propose_model_config.num_layers}"
            )
        if any(layer_type not in ("sliding_attention", "full_attention") for layer_type in layer_types):
            raise ValueError(f"unsupported dflash layer types: {layer_types}")
        if "sliding_attention" in layer_types and (window is None or int(window) <= 0):
            raise ValueError("dflash sliding_attention requires a positive sliding_window")
        if native_block_size != 16:
            raise ValueError(
                "DFlash V1 requires checkpoint native block_size=16, got "
                f"{native_block_size}"
            )
        if int(propose_model_config.hidden_size) != int(model_config.hidden_size):
            raise ValueError(
                "dflash target/draft hidden sizes must match for shared target features: "
                f"{model_config.hidden_size} != {propose_model_config.hidden_size}"
            )
        if int(propose_model_config.vocab_size) != int(model_config.vocab_size):
            raise ValueError(
                "dflash requires the target full vocabulary for the shared lm head: "
                f"{propose_model_config.vocab_size} != {model_config.vocab_size}"
            )
        target_input_vocab = int(
            getattr(model_config, "input_vocab_size", 0) or model_config.vocab_size
        )
        if input_vocab != target_input_vocab:
            raise ValueError(
                "dflash requires input_vocab_size to match the target shared embedding: "
                f"{input_vocab} != {target_input_vocab}"
            )
        config_dtype = str(getattr(propose_model_config, "config_dtype", "")).lower()
        effective_dtype = getattr(propose_model_config, "data_type", None)
        if (
            config_dtype not in ("bfloat16", "bf16")
            or effective_dtype != DataType.TYPE_BF16
        ):
            raise ValueError(
                "DFlash V1 supports BF16 draft weights and activations only, got "
                f"config_dtype={config_dtype!r}, data_type={effective_dtype!r}"
            )
        if getattr(propose_model_config, "quantization", None) not in (None, "", "none"):
            raise ValueError("DFlash V1 does not support quantized draft weights")
        if getattr(propose_model_config, "quant_config", None) is not None:
            raise ValueError("DFlash V1 does not support a draft quant_config")
        quant_algo = getattr(propose_model_config, "quant_algo", None)
        if quant_algo is not None and quant_algo.isQuant():
            raise ValueError("DFlash V1 does not support a quantized draft quant_algo")
        if not bool(getattr(propose_model_config, "qk_norm", False)):
            raise ValueError("DFlash V1 requires Qwen3 Q/K RMSNorm and RoPE")
        # The draft has its own cache allocation.  Never inherit the target's
        # FP8 cache request into the BF16-only DFlash writer/attention kernels.
        propose_model_config.attn_config.kv_cache_dtype = KvCacheDataType.BASE

        # Reuse the current fixed block input fields until the C++ ABI grows
        # DFlash-specific names.  This is geometry only, never a claim that
        # DFlash has DSpARK's Markov sampler.
        sp_config.sp_dspark_mask_token_id = noise_token_id
        sp_config.sp_dspark_sample_from_anchor = False
        model_config.capture_aux_hidden_layer_ids = layer_ids
        propose_model_config.capture_aux_hidden_layer_ids = layer_ids
        logging.info(
            "DFlash fixed-block wiring: gamma=%d, mask_token_id=%d, "
            "target capture layer ids=%s, layer_types=%s",
            gamma,
            noise_token_id,
            layer_ids,
            layer_types,
        )

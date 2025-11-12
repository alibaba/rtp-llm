import json
import logging
import os
import sys
from typing import Any, Optional

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))

from rtp_llm.async_decoder_engine.async_model import AsyncModel
from rtp_llm.config.engine_config import EngineConfig, finalize_scheduler_config
from rtp_llm.config.model_config import ModelConfig, build_py_model_config
from rtp_llm.config.model_args import ModelArgs
from rtp_llm.config.py_config_modules import (
    VitConfig,
    LoraConfig,
    GenerateEnvConfig,
    EmbeddingConfig,
)
from rtp_llm.model_factory_register import ModelDict, _model_factory
from rtp_llm.models.propose_model.propose_model import ProposeModel
from rtp_llm.ops import MMModelConfig
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
        mm_model_config: MMModelConfig,
        engine_config: EngineConfig,
        vit_config: Optional[VitConfig] = None,
        merge_lora: bool = False,
    ):
        """Create model from independent config objects.
        
        All model metadata (template_type, model_name, lora_infos) is now stored in model_config.
        
        Args:
            model_config: Model configuration
            mm_model_config: Multimodal model configuration
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
            py_model_config=model_config,
            mm_model_config=mm_model_config,
            engine_config=engine_config,
            vit_config=vit_config,
            merge_lora=merge_lora,
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
        if not propose_model_config:
            return None
            
        sp_type = engine_config.sp_config.sp_type
        if sp_type == "none" or not sp_type:
            return None
        
        # Adjust sp_type based on propose model type if needed
        if (
            sp_type == "vanilla"
            or sp_type == "mtp"
            or sp_type == "eagle3"
            or sp_type == "eagle"
        ):
            model_type = propose_model_config.model_type
            if (
                model_type == "deepseek-v3-mtp"
                or model_type == "mixtbstars-mtp"
            ):
                logging.warning(
                    f"create sp model type is {model_type}, so change the sp type to mtp"
                )
                engine_config.sp_config.sp_type = "mtp"
                sp_type = "mtp"
            elif model_type == "qwen_3_moe-mtp":
                logging.warning(
                    f"create sp model type is {model_type}, so change the sp type to eagle3"
                )
                engine_config.sp_config.sp_type = "eagle3"
                sp_type = "eagle3"
        
        gen_num_per_circle = engine_config.sp_config.gen_num_per_cycle
        
        if (
            sp_type == "vanilla"
            or sp_type == "mtp"
            or sp_type == "eagle3"
            or sp_type == "eagle"
        ):
            # Need to create GPT model for propose model
            model_cls = ModelFactory.get_model_cls(propose_model_config.model_type)
            # propose model's max seq len must be equal to score model's max seq len
            propose_model_config.max_seq_len = model_config.max_seq_len
            # Create MMModelConfig for propose model
            propose_mm_model_config = MMModelConfig()
            gpt_model = model_cls.from_config(
                py_model_config=propose_model_config,
                mm_model_config=propose_mm_model_config,
                engine_config=engine_config,
                vit_config=None,  # Propose model doesn't need vit_config
                merge_lora=False,  # Propose model doesn't need merge_lora
            )
            return ProposeModel(sp_type, gen_num_per_circle, gpt_model)
        elif sp_type == "deterministic":
            return ProposeModel(sp_type, gen_num_per_circle)
        
        return None

    @staticmethod
    def from_model_configs(
        model_config: ModelConfig,
        mm_model_config: MMModelConfig,
        engine_config: EngineConfig,
        gang_info,
        vit_config: Optional[VitConfig] = None,
        merge_lora: bool = False,
        propose_model_config: Optional[ModelConfig] = None,
    ):
        """Create model from independent config objects, with optional propose model.
        
        All model metadata (template_type, model_name, lora_infos) should be set in model_config before calling this method.
        
        This replaces from_gpt_config().
        
        Args:
            model_config: Model configuration
            mm_model_config: Multimodal model configuration
            engine_config: Engine configuration
            gang_info: GangInfo instance from GangServer
            vit_config: Optional VitConfig (needed for multimodal models)
            merge_lora: Whether to merge LoRA weights
            propose_model_config: Optional propose model configuration
        """
        model = ModelFactory._create_model(
            model_config=model_config,
            mm_model_config=mm_model_config,
            engine_config=engine_config,
            vit_config=vit_config,
            merge_lora=merge_lora,
        )
        from rtp_llm.ops import VitSeparation
        model_type = model_config.model_type
        if model_type == "fake_model" or (vit_config is not None and vit_config.vit_separation == VitSeparation.VIT_SEPARATION_ROLE):
            return model
        
        # Create propose model if provided
        propose_model = ModelFactory.get_sp_model(
            model_config=model_config,
            propose_model_config=propose_model_config,
            engine_config=engine_config,
        )

        model = AsyncModel(model, gang_info, propose_model)
        logging.info("create rpc model done")
        return model

    @staticmethod
    def load_default_generate_config(model, generate_env_config: Optional[Any] = None):
        """Load default generate config from GenerateEnvConfig.
        
        Args:
            model: Model instance to update
            generate_env_config: Optional GenerateEnvConfig object
        """
        if generate_env_config is None:
            return
        generation_config_path = generate_env_config.generation_config_path
        if generation_config_path:
            model.default_generate_config.update(
                json.load(
                    open(os.path.join(generation_config_path, "generation_config.json"))
                )
            )
            logging.info(
                f"load generate config:{generation_config_path}/generation_config.json: \n\
                         {json.dumps(model.default_generate_config.model_dump(), indent=4)}"
            )


    @staticmethod
    def create_model_configs(
        engine_config: EngineConfig,
        model_args: ModelArgs,
        lora_config: LoraConfig,
        generate_env_config: Optional[GenerateEnvConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
    ) -> tuple[ModelConfig, Optional[ModelConfig]]:
        """Create ModelConfig and optional propose ModelConfig from configuration objects.
        
        This method handles all ModelConfig construction and initialization logic.
        EngineConfig should already be built before calling this method.
        
        The flow is:
        1. Use provided ModelArgs (already populated by server_args)
        2. Call model's _create_config to create ModelConfig with model architecture
        3. Apply ModelArgs to ModelConfig (overwrite with user-provided values)
        4. Build ModelConfig with build_py_model_config
        
        Args:
            engine_config: Already built EngineConfig
            model_args: ModelArgs containing model configuration
            lora_config: LoraConfig containing LoRA configuration
            generate_env_config: Optional GenerateEnvConfig for generation settings
            embedding_config: Optional EmbeddingConfig for embedding settings
            
        Returns:
            Tuple of (model_config, propose_model_config)
            propose_py_model_config will be None if not needed
        """
        # Step 1: Use provided ModelArgs (already populated by server_args)
        
        # Step 1.5: Infer model_type from checkpoint if not provided
        if not model_args.model_type and model_args.ckpt_path:
            try:
                config_json = ModelFactory.get_config_json(model_args.ckpt_path)
                inferred_model_type = ModelDict.get_ft_model_type_by_config(config_json)
                if inferred_model_type:
                    model_args.model_type = inferred_model_type
                    logging.info(f"Inferred model_type '{inferred_model_type}' from checkpoint path: {model_args.ckpt_path}")
                else:
                    # Try to get from architectures field directly
                    if "architectures" in config_json and config_json["architectures"]:
                        architecture = config_json["architectures"][0]
                        # Try to map architecture to model_type
                        inferred_model_type = ModelDict.get_ft_model_type_by_hf_architectures(architecture)
                        if inferred_model_type:
                            model_args.model_type = inferred_model_type
                            logging.info(f"Inferred model_type '{inferred_model_type}' from architecture '{architecture}'")
            except Exception as e:
                logging.warning(f"Failed to infer model_type from checkpoint: {e}")
        
        # Check if model_type is still empty
        if not model_args.model_type:
            raise ValueError(
                f"model_type is not set and could not be inferred from checkpoint path: {model_args.ckpt_path}. "
                f"Please provide --model_type or MODEL_TYPE environment variable."
            )
        
        # Step 2: Get model class and create ModelConfig with model architecture
        model_cls = ModelFactory.get_model_cls(model_args.model_type)

        # Call _create_config to get model architecture config
        model_config = model_cls._create_config(model_args.ckpt_path)
        # Step 4: Build ModelConfig (setup paths, quantization, etc.)
        build_py_model_config(
            py_model_config=model_config,
            model_args=model_args,
            kv_cache_config=engine_config.kv_cache_config,
            py_hw_kernel_config=engine_config.hw_kernel_config,
            profiling_debug_logging_config=engine_config.profiling_debug_logging_config,
            parallelism_config=engine_config.parallelism_config,
            embedding_config=embedding_config,
        )
         
        # Set model metadata fields
        # Set lora_infos from lora_config (direct assignment)
        if lora_config.lora_info:
            lora_infos = json.loads(lora_config.lora_info)
            model_config.lora_infos = lora_infos if lora_infos else {}
        
        # Set model_name (default to model class name)
        model_config.model_name = model_cls.__name__
        
        # Finalize scheduler config based on ModelConfig (only once, for main model)
        finalize_scheduler_config(
            fifo_scheduler_config=engine_config.runtime_config.fifo_scheduler_config,
            max_seq_len=model_config.max_seq_len,
        )
        
        # Set model_name to engine_config.runtime_config.model_name (for backward compatibility)
        engine_config.runtime_config.model_name = model_config.model_name
        
        # Create propose model config if needed
        propose_model_config = None
        sp_type = engine_config.sp_config.sp_type
        if sp_type and sp_type != "none":
            propose_model_type = engine_config.sp_config.sp_model_type
            propose_ckpt_path = engine_config.sp_config.sp_checkpoint_path
            if propose_ckpt_path:
                # Create ModelArgs for propose model (reuse main model args, but override ckpt_path)
                propose_model_args = ModelArgs()
                propose_model_args.ckpt_path = propose_ckpt_path
                propose_model_args.tokenizer_path = model_args.tokenizer_path
                propose_model_args.model_type = propose_model_type
                propose_model_args.act_type = model_args.act_type
                propose_model_args.use_float32 = model_args.use_float32
                propose_model_args.mla_ops_type = model_args.mla_ops_type
                
                # Create propose ModelConfig using _create_config
                propose_model_cls = ModelFactory.get_model_cls(propose_model_type)
                propose_model_config = propose_model_cls._create_config(propose_ckpt_path)
                # Ensure max_seq_len matches main model
                propose_model_config.max_seq_len = model_config.max_seq_len
                propose_model_config.quantization = engine_config.sp_config.sp_quantization
                
                logging.info(
                    f"load propose model from tokenizer_path: {propose_model_config.tokenizer_path}, "
                    f"ckpt_path: {propose_model_config.ckpt_path}, quantization: {propose_model_config.quantization}"
                )
                
                # Build propose model config (no finalize_scheduler_config for propose model)
                build_py_model_config(
                    py_model_config=propose_model_config,
                    model_args=propose_model_args,
                    kv_cache_config=engine_config.kv_cache_config,
                    py_hw_kernel_config=engine_config.hw_kernel_config,
                    profiling_debug_logging_config=engine_config.profiling_debug_logging_config,
                    parallelism_config=engine_config.parallelism_config,
                    embedding_config=None,  # Propose model doesn't need embedding_config
                )
        
        return model_config, propose_model_config


import json
import logging
import math
import os
from typing import Any, Dict, Optional

import torch

from rtp_llm.config.quant_config import (
    Fp8BlockWiseQuantConfig,
    QuantizationConfig,
    init_quant_config,
)
from rtp_llm.ops import KVCacheConfig, KvCacheDataType
from rtp_llm.ops import ModelConfig as CppModelConfig
from rtp_llm.ops import TaskType
from rtp_llm.utils.gemm_utils.cutlass_config import load_cutlass_gemm_config
from rtp_llm.utils.util import get_config_from_path, to_torch_dtype
from rtp_llm.utils.weight_type import WEIGHT_TYPE


def kv_cache_dtype_to_torch_dtype(
    kv_cache_dtype: KvCacheDataType, data_type: WEIGHT_TYPE
) -> torch.dtype:
    """Convert KvCacheDataType enum to torch.dtype.

    Args:
        kv_cache_dtype: KvCacheDataType enum value
        data_type: WEIGHT_TYPE enum value to use for BASE case

    Returns:
        torch.dtype value
    """
    if kv_cache_dtype == KvCacheDataType.INT8:
        return torch.int8
    elif kv_cache_dtype == KvCacheDataType.FP8:
        return torch.float8_e4m3fn
    else:  # BASE
        return data_type.to_torch_dtype()


class VitParameters:
    """Vit parameters for multimodal models."""

    # config includes origin vit config in ckpt/config.json
    config: Dict[str, Any] = {}
    special_token_ids: Dict[str, Any] = {}
    special_tokens: Dict[str, Any] = {}
    vit_weights: Any = None
    preprocess_batch_size: int = 1
    eval_param_count = None
    eval_model_size = None


class ModelConfig(CppModelConfig):
    # Python-only fields that are allowed to be set
    _python_fields = {
        "is_mtp",
        "normalize_lm_head_weight",
        "has_lm_head_bias",
        "tie_word_embeddings",
        "quantization",
        "mm_related_params",
        "src_quantization_bit",
        "config_dtype",
        "template_type",
        "model_name",
        "quant_config",
        "inter_size",
        "moe_inter_size",
        "generate_env_config",
        "render_config",
        "phy2log_path",
    }

    # Known C++ ModelConfig members (from ModelConfig.h)
    # These may not be explicitly exposed via pybind11 but exist in C++
    _cpp_members = {
        "num_layers",
        "max_seq_len",
        "vocab_size",
        "hidden_size",
        "attn_config",
        "special_tokens",
        "quant_algo",
        "eplb_config",
        "ckpt_path",
        "tokenizer_path",
        "lora_infos",
        "position_ids_style",
        "pre_seq_len",
        "use_kvcache",
        "logit_scale",
        "qk_norm",
        "expert_num",
        "moe_n_group",
        "moe_k",
        "moe_style",
        "moe_layer_index",
        "data_type",
        "activation_type",
        "norm_type",
        "layernorm_type",
        "task_type",
        "mla_ops_type",
        "extra_data_path",
        "local_extra_data_path",
        "model_type",
        "ptuning_path",
        "mm_model_config",
        "deepseek_rope_mscale",
        "deepseek_mscale_all_dim",
        "moe_topk_group",
        "routed_scaling_factor",
        "layernorm_eps",
        "partial_rotary_factor",
        "input_embedding_scalar",
        "residual_scalar",
        "use_norm_input_residual",
        "use_norm_attn_out_residual",
        "input_vocab_size",
        "type_vocab_size",
        "embedding_size",
        "moe_normalize_expert_scale",
        "scoring_func",
        "has_positional_encoding",
        "has_pre_decoder_layernorm",
        "has_post_decoder_layernorm",
        "has_lm_head",
        "use_attention_linear_bias",
        "use_fp32_to_compute_logit",
        "add_bias_linear",
        "has_moe_norm",
        "prefix_projection",
        "reverse_e_h_norm",
    }

    def __setattr__(self, name: str, value: Any) -> None:
        """Override __setattr__ to prevent assignment of undefined Python attributes.

        This allows C++ binding attributes (managed by pybind11) to work normally,
        while preventing assignment of undefined Python attributes.
        """
        # Allow setting attributes that are:
        # 1. In _python_fields (Python-only fields)
        # 2. Start with underscore (private attributes)
        # 3. In _cpp_members (known C++ members)
        if (
            name.startswith("_")
            or name in self._python_fields
            or name in self._cpp_members
        ):
            super().__setattr__(name, value)
        else:
            # For other attributes, check if they exist (C++ binding attributes)
            # Try hasattr first - it catches AttributeError internally
            # If hasattr returns True or raises TypeError, the attribute exists
            try:
                if hasattr(self, name):
                    # Attribute exists (likely a C++ binding attribute), allow setting
                    super().__setattr__(name, value)
                else:
                    # hasattr returned False, but try accessing anyway in case it's a C++ member
                    # that pybind11 hasn't exposed yet but exists in C++
                    try:
                        # Try to get the attribute - this will work if it exists
                        getattr(self, name)
                        # If we get here, attribute exists, allow setting
                        super().__setattr__(name, value)
                    except AttributeError:
                        # Attribute doesn't exist, raise error
                        raise AttributeError(
                            f"'{self.__class__.__name__}' object has no attribute '{name}'. "
                            f"Valid Python-only attributes: {', '.join(sorted(self._python_fields))}"
                        )
            except TypeError:
                # TypeError can occur when property getter returns unregistered C++ type
                # This means the attribute exists (as a property), so allow setting
                super().__setattr__(name, value)

    @property
    def compute_dtype(self) -> torch.dtype:
        """Get compute dtype as torch.dtype from model_config.data_type.

        Returns:
            torch.dtype: The compute dtype converted from data_type
        """
        return to_torch_dtype(self.data_type)

    def eval_model_weight_size(self) -> float:
        """
        Evaluate total model size including weights, KV cache, and runtime buffers.
        All required parameters (quant_algo, task_type, vocab_size) are obtained from self.

        Returns:
            Total model size in bytes
        """
        quant_algo = self.quant_algo
        vocab_size = self.vocab_size

        layer_param_bytes = 2
        if quant_algo.getWeightBits() == 8:
            layer_param_bytes = 1
        elif quant_algo.getWeightBits() == 4:
            layer_param_bytes = 0.54

        model_size = (
            self.word_emb_param_count(vocab_size) * 2
            + self.layer_weight_param_count() * layer_param_bytes
            + self.word_emb_param_count(vocab_size) * 2
        )  # maybe some model donot have lm_head

        if self.mm_related_params.eval_model_size:
            model_size += self.mm_related_params.eval_model_size(self)

        return model_size

    def eval_model_size(self) -> float:
        model_size = self.eval_model_weight_size()
        kv_cache_mem_size = self._eval_kv_cache_mem_size()
        runtime_buffer = self._eval_runtime_buffer_mem_size()
        total_size = model_size + kv_cache_mem_size + runtime_buffer
        logging.info(
            f"total_size: {total_size/1000**3:.2f}GB, model_size:{model_size/1000**3:.2f}GB, kv_cache_mem_size:{kv_cache_mem_size/1000**3:.2f}GB, runtime_buffer:{runtime_buffer/1000**3:.2f}GB"
        )
        return total_size

    def _eval_kv_cache_mem_size(self) -> float:
        """Evaluate KV cache memory size."""
        if self.task_type != TaskType.LANGUAGE_MODEL:
            return 0
        # Get kv_cache_dtype from attn_config
        kv_cache_dtype_enum = self.attn_config.kv_cache_dtype
        kv_cache_bytes = (
            1
            if kv_cache_dtype_enum in [KvCacheDataType.FP8, KvCacheDataType.INT8]
            else 2
        )
        kv_cache_size = (
            2
            * self.num_layers
            * self.attn_config.kv_head_num
            * self.attn_config.size_per_head
            * kv_cache_bytes
            * self.max_seq_len
        )
        return kv_cache_size

    def _eval_runtime_buffer_mem_size(self) -> float:
        """Evaluate runtime buffer memory size."""
        input_buffer = self.max_seq_len * self.hidden_size
        qkv_gemm_buffer_size = (
            self.max_seq_len
            * (self.attn_config.kv_head_num * 2 + self.attn_config.kv_head_num)
            * self.attn_config.size_per_head
        )
        attn_buffer_size = self.max_seq_len * self.hidden_size
        ffn_expert_num = self.expert_num if self.moe_k else 1
        # Use isGatedActivation() to determine if we need 2 weights (gated) or 1 weight (non-gated like GELU)
        ffn_w_count = 2 if self.isGatedActivation() else 1

        # Calculate FFN buffer size based on MOE configuration
        if self.moe_style == 1:
            # Pure MOE: all layers use routed experts with moe_inter_size
            ffn_buffer = (
                self.max_seq_len * self.hidden_size
                + ffn_w_count * self.max_seq_len * self.moe_inter_size
            ) * ffn_expert_num
        elif self.moe_style == 2:
            # Hybrid MOE: shared experts (inter_size) + routed experts (moe_inter_size)
            # Need buffer for both shared and routed experts
            shared_buffer = (
                self.max_seq_len * self.hidden_size
                + ffn_w_count * self.max_seq_len * self.inter_size
            )
            routed_buffer = (
                self.max_seq_len * self.hidden_size
                + ffn_w_count * self.max_seq_len * self.moe_inter_size
            ) * ffn_expert_num
            # Total buffer is sum of shared and routed (they run in sequence)
            ffn_buffer = shared_buffer + routed_buffer
        else:
            # No MOE: all layers use regular FFN with inter_size
            ffn_buffer = (
                self.max_seq_len * self.hidden_size
                + ffn_w_count * self.max_seq_len * self.inter_size
            )

        return input_buffer + qkv_gemm_buffer_size + attn_buffer_size + ffn_buffer

    def model_param_count(self) -> int:
        """
        Calculate total model parameter count.
        vocab_size is obtained from self.vocab_size.

        Returns:
            Total parameter count
        """
        vocab_size = self.vocab_size
        param_count = (
            self.word_emb_param_count(vocab_size) * 2
            + self.layer_weight_param_count()
            + self.hidden_size
        )

        if self.mm_related_params.eval_param_count:
            param_count += self.mm_related_params.eval_param_count(self)
        return param_count

    def word_emb_param_count(self, vocab_size: int) -> int:
        """
        Calculate word embedding parameter count.

        Args:
            vocab_size: Vocabulary size

        Returns:
            Word embedding parameter count
        """
        return vocab_size * self.hidden_size

    def layer_weight_param_count(self) -> int:
        """
        Calculate layer weight parameter count.

        Returns:
            Layer weight parameter count
        """
        hidden_size = self.hidden_size

        layer_weight_param_count = 0

        # qkv
        layer_weight_param_count = (
            layer_weight_param_count
            + self.num_layers * hidden_size * hidden_size
            + self.num_layers
            * hidden_size
            * (self.attn_config.kv_head_num * self.attn_config.size_per_head)
            * 2
        )

        # attn_o_w
        layer_weight_param_count = (
            layer_weight_param_count + self.num_layers * hidden_size * hidden_size
        )

        # ffn w1, w2, w3
        ffn_expert_num = self.expert_num if self.expert_num > 0 else 1
        # Use isGatedActivation() to determine if we need 3 weights (gated) or 2 weights (non-gated like GELU)
        ffn_w_count = 3 if self.isGatedActivation() else 2
        if self.moe_style == 1:
            # Pure MOE: all layers use routed experts with moe_inter_size
            layer_weight_param_count = (
                layer_weight_param_count
                + self.num_layers
                * self.moe_inter_size
                * hidden_size
                * ffn_w_count
                * ffn_expert_num
            )
            # Gate weights for MOE layers
            layer_weight_param_count = (
                layer_weight_param_count
                + self.num_layers * hidden_size * ffn_expert_num
            )
        elif self.moe_style == 2:
            # Hybrid MOE: shared experts + routed experts
            # Shared experts use inter_size
            layer_weight_param_count = (
                layer_weight_param_count
                + len(self.moe_layer_index)
                * self.inter_size
                * hidden_size
                * ffn_w_count
            )
            # Routed experts use moe_inter_size
            layer_weight_param_count = (
                layer_weight_param_count
                + len(self.moe_layer_index)
                * self.moe_inter_size
                * hidden_size
                * ffn_w_count
                * ffn_expert_num
            )
        else:
            # No MOE: all layers use regular FFN with inter_size
            layer_weight_param_count = (
                layer_weight_param_count
                + self.num_layers * self.inter_size * hidden_size * ffn_w_count
            )

        # other small tensor
        layer_weight_param_count = (
            layer_weight_param_count + self.num_layers * hidden_size * 11
        )
        return layer_weight_param_count

    def apply_rope_scaling_override(self, model_override_args: Dict[str, Any]) -> None:
        """
        Apply rope_scaling configuration from model_override_args.

        Args:
            model_override_args: Dictionary containing model override arguments
        """
        if not model_override_args or "rope_scaling" not in model_override_args:
            return

        # be consistent with RopeStyle
        rope_type = {
            "no": 0,
            "base": 1,
            "glm2": 2,
            "dynamicntk": 3,
            "qwendynamicntk": 4,
            "yarn": 5,
            "llama3": 6,
            "mrope": 7,
        }
        rope_override_args = model_override_args["rope_scaling"]
        assert (
            "type" in rope_override_args and rope_override_args["type"] in rope_type
        ), f"Invalid rope_scaling type: {rope_override_args.get('type')}"

        self.attn_config.rope_config.style = rope_type[rope_override_args["type"]]

        if rope_override_args["type"] == "yarn":
            assert (
                "factor" in rope_override_args
                and "original_max_position_embeddings" in rope_override_args
            ), "yarn rope_scaling requires 'factor' and 'original_max_position_embeddings'"

            self.attn_config.rope_config.scale = rope_override_args["factor"]
            self.attn_config.rope_config.max_pos = rope_override_args[
                "original_max_position_embeddings"
            ]
            self.attn_config.rope_config.factor1 = rope_override_args.get(
                "beta_slow", 1.0
            )
            self.attn_config.rope_config.factor2 = rope_override_args.get(
                "beta_fast", 1.0
            )
            mscale = rope_override_args.get("mscale", 1.0)
            self.attn_config.rope_config.mscale = float(
                (
                    1.0
                    if self.attn_config.rope_config.scale <= 1
                    else 0.1 * math.log(self.attn_config.rope_config.scale) + 1.0
                )
                * mscale
            )
            self.attn_config.rope_config.extrapolation_factor = rope_override_args.get(
                "extrapolation_factor", 1.0
            )

            logging.info(
                f"Applied rope_scaling (yarn): "
                f"style: {self.attn_config.rope_config.style}, "
                f"scale: {self.attn_config.rope_config.scale}, "
                f"max_pos: {self.attn_config.rope_config.max_pos}, "
                f"factor1: {self.attn_config.rope_config.factor1}, "
                f"factor2: {self.attn_config.rope_config.factor2}, "
                f"mscale: {self.attn_config.rope_config.mscale}, "
                f"extrapolation_factor: {self.attn_config.rope_config.extrapolation_factor}"
            )
        else:
            logging.info(
                f"Applied rope_scaling: style: {self.attn_config.rope_config.style}"
            )

    def __init__(self, *args, **kwargs):
        """Initialize ModelConfig with quant_algo member and default values."""
        super().__init__(*args, **kwargs)
        # Additional Python-only fields
        self.is_mtp: bool = False
        self.normalize_lm_head_weight: bool = False
        self.has_lm_head_bias: bool = False
        self.tie_word_embeddings: bool = False
        # Model loading related fields
        # ptuning_path is now in C++ ModelConfig (as std::string, default "")
        self.quantization: str = (
            ""  # Quantization method string (e.g., "INT8", "FP8", etc.)
        )
        # mm_related_params will be set to VitParameters() if needed
        self.src_quantization_bit: int = 0
        self.config_dtype: Optional[str] = None

        # Model metadata fields (merged from function parameters)
        self.template_type: Optional[Any] = None  # TemplateType enum
        self.model_name: str = (
            ""  # Model name (also set to engine_config.runtime_config.model_name)
        )

        # Model architecture fields
        self.inter_size: int = 0  # FFN intermediate size (for regular FFN layers)
        self.moe_inter_size: int = (
            0  # MOE intermediate size (for MOE expert FFN layers)
        )

        # Renderer configuration fields
        self.generate_env_config: Optional[Any] = (
            None  # GenerateEnvConfig for renderer factory
        )
        self.render_config: Optional[Any] = None  # RenderConfig for renderer factory
        self.mm_related_params = VitParameters()
        self.quant_config = None

    def apply_override_args(self, json_model_override_args: str) -> None:
        """Apply model override arguments to ModelConfig.

        Args:
            json_model_override_args: JSON string with model override arguments
        """
        model_override_args = json.loads(json_model_override_args)
        if model_override_args:
            # Apply rope_scaling override via model_config
            self.apply_rope_scaling_override(model_override_args)

    def init_precision_config(
        self, kv_cache_config: Optional[Any], act_type: Optional[str]
    ):
        """Initialize precision configuration from checkpoint and quantization settings.

        This method:
        1. Loads quant_config from checkpoint or quantization string
        2. Sets quant_algo if quant_config exists
        3. Initializes data_type from act_type (or config_dtype if act_type is empty)
        4. Sets attn_config.kv_cache_dtype based on kv_cache_config (if provided)
        5. Applies quantization-specific overrides (e.g., fp8 quant_config sets kv_cache_dtype to FP8)
        6. Validates configuration with quant_config using kv_cache_dtype_to_torch_dtype
        7. Sets final data_type

        Args:
            kv_cache_config: Optional KVCacheConfig to set attn_config.kv_cache_dtype
        """
        # Load quant_config
        quant_config = QuantizationConfig.load_from_ckpt(self.ckpt_path)
        if not quant_config:
            if self.quantization:
                quant_config = init_quant_config(self.quantization)
                logging.info(f"need_load_quant by {quant_config.get_method()}")

        # Set quant_algo if quant_config exists
        if quant_config:
            self.quant_algo.setQuantAlgo(
                quant_config.get_algo().lower(),
                quant_config.bits,
                quant_config.group_size(),
            )

        # Initialize data_type: first try act_type, then config_dtype, finally default to FP16
        data_type: Optional[WEIGHT_TYPE] = None
        if act_type:
            data_type = WEIGHT_TYPE.from_str(act_type)
            logging.info(f"Initializing data_type from act_type: {data_type}")
        else:
            # Parse config_dtype if available
            config_dtype_parsed = None
            if self.config_dtype:
                config_dtype_parsed = WEIGHT_TYPE.from_str(self.config_dtype)
                logging.info(
                    f"act_type is empty, using config_dtype: {config_dtype_parsed}"
                )

            if config_dtype_parsed:
                data_type = config_dtype_parsed
                if data_type == WEIGHT_TYPE.FP32:
                    data_type = WEIGHT_TYPE.FP16
                    logging.info("auto convert embedding model to fp16")
            else:
                data_type = WEIGHT_TYPE.FP16
                logging.info(
                    f"act_type and config_dtype are both empty, using default: {data_type}"
                )

        # Apply quantization-specific overrides
        if quant_config and isinstance(quant_config, Fp8BlockWiseQuantConfig):
            original_data_type = data_type
            data_type = WEIGHT_TYPE.BF16
            logging.info(
                f"Overriding data_type from {original_data_type} to {data_type} "
                f"because fp8_block_wise quantization only supports BF16"
            )
        elif quant_config and quant_config.get_method().lower() in [
            "smooth_quant",
            "omni_quant",
        ]:
            original_data_type = data_type
            data_type = WEIGHT_TYPE.FP16
            logging.info(
                f"Overriding data_type from {original_data_type} to {data_type} "
                f"because {quant_config.get_method()} quantization requires FP16"
            )

        # Set attn_config.kv_cache_dtype based on kv_cache_config
        if kv_cache_config is not None:
            if kv_cache_config.int8_kv_cache:
                self.attn_config.kv_cache_dtype = KvCacheDataType.INT8
                logging.info(
                    "Setting attn_config.kv_cache_dtype to INT8 based on kv_cache_config.int8_kv_cache"
                )
            elif kv_cache_config.fp8_kv_cache:
                self.attn_config.kv_cache_dtype = KvCacheDataType.FP8
                logging.info(
                    "Setting attn_config.kv_cache_dtype to FP8 based on kv_cache_config.fp8_kv_cache"
                )
            else:
                self.attn_config.kv_cache_dtype = KvCacheDataType.BASE
                logging.info(
                    "Setting attn_config.kv_cache_dtype to BASE (default, no int8/fp8 kv_cache specified)"
                )

        if quant_config and quant_config.get_method().lower() == "fp8":
            self.attn_config.kv_cache_dtype = KvCacheDataType.FP8
            logging.info(
                "Setting attn_config.kv_cache_dtype to FP8 based on quant_config.get_method().lower() == 'fp8'"
            )

        # Validate configuration with quant_config
        if quant_config:
            kv_cache_torch_dtype = kv_cache_dtype_to_torch_dtype(
                self.attn_config.kv_cache_dtype, data_type
            )
            logging.info(
                f"Validating precision configuration with quant_config: "
                f"data_type={data_type}, kv_cache_dtype={self.attn_config.kv_cache_dtype}"
            )
            quant_config.verify_compute_dtype_and_kv_cache_dtype(
                data_type.to_torch_dtype(), kv_cache_torch_dtype
            )
            logging.info("Precision configuration validation passed")

        # Set final data_type
        # This uses ModelConfig's __setattr__ which handles string-to-enum conversion
        self.data_type = data_type.to_str()
        # Store quant_config as instance attribute for later use
        self.quant_config = quant_config

        # Print final type results
        logging.info(
            f"Final precision configuration - "
            f"quant_config: {quant_config}, "
            f"data_type: {self.data_type}, "
            f"attn_config.kv_cache_dtype: {self.attn_config.kv_cache_dtype}"
        )


def get_task_type_from_ckpt_path(
    task_type: Optional[TaskType],
    ckpt_path: str,
    embedding_config: Optional[Any] = None,
) -> TaskType:
    """
    Get task_type from checkpoint path or use provided task_type.

    Args:
        ckpt_path: Checkpoint path
        embedding_config: Optional EmbeddingConfig for embedding task detection

    Returns:
        TaskType enum value
    """
    if task_type is not None:
        logging.info(f"use {task_type} from args")
        return task_type

    def _is_dense_embedding_task(ckpt_path: str) -> bool:
        def _check_is_sentence_transformer_repo() -> bool:
            if os.path.exists(
                os.path.join(ckpt_path, "config_sentence_transformers.json")
            ):
                return True
            module_file_path = os.path.join(ckpt_path, "modules.json")
            if os.path.exists(module_file_path):
                with open(module_file_path, "r") as reader:
                    content = reader.read()
                if "sentence_transformers" in content:
                    return True
            return False

        return (
            embedding_config and embedding_config.embedding_model == 1
        ) or _check_is_sentence_transformer_repo()

    def _is_classifier_task(ckpt_path: str) -> bool:
        config_json = get_config_from_path(ckpt_path)
        if not config_json:
            return False
        if "architectures" in config_json and len(config_json["architectures"]) > 0:
            model_type = config_json["architectures"][0]
            if "SequenceClassification" in model_type:
                return True
        return False

    if _is_dense_embedding_task(ckpt_path):
        return TaskType.DENSE_EMBEDDING
    elif _is_classifier_task(ckpt_path):
        return TaskType.SEQ_CLASSIFICATION
    else:
        return TaskType.LANGUAGE_MODEL


def update_stop_words_from_env(special_tokens, generate_env_config) -> None:
    """
    Update stop_words_str and stop_words_id from environment variables.

    Args:
        special_tokens: SpecialTokens object to update
        generate_env_config: GenerateEnvConfig object containing stop_words configuration
    """
    env_stop_words_str = generate_env_config.stop_words_str
    env_stop_words_id = generate_env_config.stop_words_list
    env_stop_words_str_list = (
        json.loads(env_stop_words_str) if env_stop_words_str else []
    )
    env_stop_words_id_list = json.loads(env_stop_words_id) if env_stop_words_id else []
    env_force_stop = generate_env_config.force_stop_words
    if env_force_stop:
        special_tokens.stop_words_str_list = env_stop_words_str_list
        special_tokens.stop_words_id_list = env_stop_words_id_list
    else:
        special_tokens.stop_words_str_list = (
            special_tokens.stop_words_str_list + env_stop_words_str_list
        )
        special_tokens.stop_words_id_list = (
            special_tokens.stop_words_id_list + env_stop_words_id_list
        )

    logging.info(
        f"use stop_words_str_list [{special_tokens.stop_words_str_list }],"
        f" stop_words_id_list [{special_tokens.stop_words_id_list}]"
    )


def update_tokenizer_special_tokens(special_tokens, tokenizer: Any) -> None:
    """Update special tokens from tokenizer to ModelConfig.

    Args:
        special_tokens: SpecialTokens object to update
        tokenizer: Tokenizer instance with stop_words_id_list, stop_words_str_list, and eos_token_id
    """
    special_tokens.stop_words_id_list += tokenizer.stop_words_id_list
    special_tokens.stop_words_str_list += tokenizer.stop_words_str_list
    special_tokens.eos_token_id = tokenizer.eos_token_id


# ============================================================================
# ModelConfig setup and initialization functions
# ============================================================================


def build_model_config(
    model_config: ModelConfig,  # ModelConfig instance to build
    model_args: Any,  # ModelArgs from py_env_configs
    kv_cache_config,
    profiling_debug_logging_config: Any,  # ProfilingDebugLoggingConfig
    embedding_config: Optional[
        Any
    ] = None,  # EmbeddingConfig (optional, for check_task_type)
    quantization_config: Optional[
        Any
    ] = None,  # QuantizationConfig (optional, for quantization)
) -> None:
    """Build and initialize ModelConfig from model_args.

    This function initializes ModelConfig after EngineConfig is initialized.
    It copies values from model_args to model_config, then sets up model-specific fields.

        Args:
        model_config: ModelConfig instance to build
        model_args: ModelArgs instance from py_env_configs (contains user-provided arguments)
        kv_cache_config: KVCacheConfig for task_type and use_kvcache
        profiling_debug_logging_config: ProfilingDebugLoggingConfig for hack_layer_num
        embedding_config: Optional EmbeddingConfig (for check_task_type)
        quantization_config: Optional QuantizationConfig (for quantization settings)
    """
    model_config.ckpt_path = model_args.ckpt_path
    model_config.tokenizer_path = model_args.tokenizer_path
    model_config.extra_data_path = model_args.extra_data_path
    model_config.local_extra_data_path = model_args.local_extra_data_path
    model_config.model_type = model_args.model_type
    model_config.phy2log_path = model_args.phy2log_path

    if model_args.mla_ops_type:
        model_config.mla_ops_type = model_args.mla_ops_type

    if model_args.max_seq_len:
        model_config.max_seq_len = model_args.max_seq_len
    if not model_config.max_seq_len:
        model_config.max_seq_len = 8192
    logging.info(f"max_seq_len: {model_config.max_seq_len}")

    model_config.task_type = get_task_type_from_ckpt_path(
        model_args.task_type,
        model_config.ckpt_path,
        embedding_config,
    )

    # Set quantization from quantization_config
    if quantization_config is not None:
        model_config.quantization = quantization_config.get_quantization()

    # Initialize precision configuration (uses self.ckpt_path and self.quantization)
    # This will initialize data_type from act_type (or config_dtype), set attn_config.kv_cache_dtype
    # from kv_cache_config, and validate with quant_config
    model_config.init_precision_config(
        kv_cache_config=kv_cache_config, act_type=model_args.act_type
    )
    model_config.attn_config.tokens_per_block = kv_cache_config.seq_size_per_block

    model_config.use_kvcache = model_config.task_type == TaskType.LANGUAGE_MODEL
    logging.info(
        f"model task type: {model_config.task_type}, use_kvcache: {model_config.use_kvcache}"
    )

    if not model_config.hidden_size:
        model_config.hidden_size = (
            model_config.attn_config.size_per_head * model_config.attn_config.head_num
        )

    # Load cutlass gemm config
    load_cutlass_gemm_config(model_config.quant_algo)

    # Apply hack_layer_num if needed
    hack_layer_num = profiling_debug_logging_config.hack_layer_num
    if hack_layer_num:
        logging.info(f"hack layernum to {hack_layer_num}")
        model_config.num_layers = hack_layer_num

    # Apply model override args
    if model_args.json_model_override_args:
        model_config.apply_override_args(model_args.json_model_override_args)

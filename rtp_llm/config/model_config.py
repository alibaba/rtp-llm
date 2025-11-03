import json
import logging
import math
from typing import Any, Dict, Optional

from rtp_llm.config.quant_config import (
    Fp8BlockWiseQuantConfig,
    QuantizationConfig,
    init_quant_config,
)
from rtp_llm.config.task_type import TaskType, check_task_type
from rtp_llm.ops import ModelConfig as CppModelConfig, QuantAlgo
from rtp_llm.utils.weight_type import WEIGHT_TYPE

# Import for type hints
if False:
    from rtp_llm.config.py_config_modules import GenerateEnvConfig

def get_pad_size(size: int, align_size: int):
    """Calculate padding size to align to align_size."""
    return (align_size - (size % align_size)) % align_size


class ModelConfig(CppModelConfig):
    def eval_model_size(self, quant_algo, task_type: TaskType, vocab_size: int) -> float:
        """
        Evaluate total model size including weights, KV cache, and runtime buffers.
        
        Args:
            quant_algo: Quantization algorithm for determining weight bytes
            task_type: Task type (affects KV cache size)
            vocab_size: Vocabulary size
            
        Returns:
            Total model size in bytes
        """
        layer_param_bytes = 2
        if quant_algo.getWeightBits() == 8:
            layer_param_bytes = 1
        elif quant_algo.getWeightBits() == 4:
            layer_param_bytes = 0.54

        model_size = (
            self.word_emb_param_count(vocab_size) * 2
            + self.layer_weight_param_count() * layer_param_bytes
            + self.hidden_size * layer_param_bytes
            + self.word_emb_param_count(vocab_size) * 2
        )  # maybe some model donot have lm_head

        kv_cache_mem_size = self._eval_kv_cache_mem_size(task_type)
        runtime_buffer = self._eval_runtime_buffer_mem_size()
        total_size = model_size + kv_cache_mem_size + runtime_buffer
        logging.info(
            f"total_size(Bytes): {total_size}, model_size:{model_size}, kv_cache_mem_size:{kv_cache_mem_size}, runtime_buffer:{runtime_buffer}"
        )
        return total_size

    def _eval_kv_cache_mem_size(self, task_type: TaskType) -> float:
        """Evaluate KV cache memory size."""
        if task_type != TaskType.LANGUAGE_MODEL:
            return 0
        kv_cache_bytes = (
            1
            if self.kv_cache_data_type
            in [WEIGHT_TYPE.FP8.to_str(), WEIGHT_TYPE.INT8.to_str()]
            else 2
        )
        kv_cache_size = (
            2
            * self.num_layers
            * self.head_num_kv
            * self.size_per_head
            * kv_cache_bytes
            * self.max_seq_len
        )
        return kv_cache_size

    def _eval_runtime_buffer_mem_size(self) -> float:
        """Evaluate runtime buffer memory size."""
        input_buffer = self.max_seq_len * self.hidden_size
        qkv_gemm_buffer_size = (
            self.max_seq_len
            * (self.head_num_kv * 2 + self.head_num_kv)
            * self.size_per_head
        )
        attn_buffer_size = self.max_seq_len * self.hidden_size
        ffn_export_num = self.expert_num if self.moe_k else 1
        ffn_w_count = 1 if self.activation_type == "gelu" else 2
        ffn_buffer = (
            self.max_seq_len * self.hidden_size
            + ffn_w_count * self.max_seq_len * self.inter_size
        ) * ffn_export_num
        return input_buffer + qkv_gemm_buffer_size + attn_buffer_size + ffn_buffer

    def model_param_count(self, vocab_size: int) -> int:
        """
        Calculate total model parameter count.
        
        Args:
            vocab_size: Vocabulary size
            
        Returns:
            Total parameter count
        """
        return (
            self.word_emb_param_count(vocab_size) * 2
            + self.layer_weight_param_count()
            + self.hidden_size
        )

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
        if self.layer_head_num and isinstance(self.layer_head_num, list):
            for head_num in self.layer_head_num:
                layer_weight_param_count = (
                    layer_weight_param_count
                    + head_num * self.size_per_head * hidden_size * 3
                )
        elif self.head_num_kv != self.head_num:
            layer_weight_param_count = (
                layer_weight_param_count
                + self.num_layers * hidden_size * hidden_size
                + self.num_layers * (self.head_num_kv * self.size_per_head) * 2
            )
        else:
            layer_weight_param_count = (
                layer_weight_param_count
                + self.num_layers * hidden_size * hidden_size * 3
            )

        # attn_o_w
        if self.layer_head_num and isinstance(self.layer_head_num, list):
            for head_num in self.layer_head_num:
                layer_weight_param_count = (
                    layer_weight_param_count
                    + head_num * self.size_per_head * hidden_size
                )
        else:
            layer_weight_param_count = (
                layer_weight_param_count + self.num_layers * hidden_size * hidden_size
            )

        # ffn w1, w2, w3
        ffn_export_num = self.expert_num if self.expert_num > 0 else 1
        ffn_w_count = 2 if self.activation_type == "gelu" else 3
        if self.layer_inter_size and isinstance(self.layer_inter_size, list):
            for layer_inter_size in self.layer_inter_size:
                if self.moe_style == 1:
                    layer_weight_param_count = (
                        layer_weight_param_count
                        + layer_inter_size * hidden_size * ffn_w_count * ffn_export_num
                    )
                else:
                    layer_weight_param_count = (
                        layer_weight_param_count
                        + layer_inter_size * hidden_size * ffn_w_count
                    )
                    if self.moe_style == 2:
                        layer_weight_param_count = (
                            layer_weight_param_count
                            + self.moe_inter_padding_size
                            * hidden_size
                            * ffn_w_count
                            * ffn_export_num
                        )

        else:
            if self.moe_style == 1:
                layer_weight_param_count = (
                    layer_weight_param_count
                    + self.num_layers
                    * self.inter_size
                    * hidden_size
                    * ffn_w_count
                    * ffn_export_num
                )
            else:
                layer_weight_param_count = (
                    layer_weight_param_count + self.num_layers * self.inter_size * hidden_size * ffn_w_count
                )
                if self.moe_style == 2:
                    layer_weight_param_count = (
                        layer_weight_param_count
                        + len(self.moe_layer_index)
                        * self.moe_inter_padding_size
                        * hidden_size
                        * ffn_w_count
                        * ffn_export_num
                    )

        if ffn_export_num > 1:
            layer_weight_param_count = (
                layer_weight_param_count
                + len(self.moe_layer_index) * hidden_size * ffn_export_num
            )
        # other small tensor
        layer_weight_param_count = (
            layer_weight_param_count + self.num_layers * hidden_size * 11
        )
        return layer_weight_param_count

    def apply_rope_scaling_override(self, model_override_args: Dict[str, Any], target_model_config = None) -> None:
        """
        Apply rope_scaling configuration from model_override_args.
        
        Args:
            model_override_args: Dictionary containing model override arguments
            target_model_config: Optional ModelConfig instance to apply to. If None, applies to self.
        """
        if not model_override_args or "rope_scaling" not in model_override_args:
            return
        
        # Use target_model_config if provided, otherwise use self
        target = target_model_config if target_model_config is not None else self
        
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
            "type" in rope_override_args
            and rope_override_args["type"] in rope_type
        ), f"Invalid rope_scaling type: {rope_override_args.get('type')}"
        
        target.rope_config.style = rope_type[rope_override_args["type"]]
        
        if rope_override_args["type"] == "yarn":
            assert (
                "factor" in rope_override_args
                and "original_max_position_embeddings" in rope_override_args
            ), "yarn rope_scaling requires 'factor' and 'original_max_position_embeddings'"
            
            target.rope_config.scale = rope_override_args["factor"]
            target.rope_config.max_pos = rope_override_args["original_max_position_embeddings"]
            target.rope_config.factor1 = rope_override_args.get("beta_slow", 1.0)
            target.rope_config.factor2 = rope_override_args.get("beta_fast", 1.0)
            mscale = rope_override_args.get("mscale", 1.0)
            target.rope_config.mscale = float(
                (
                    1.0
                    if target.rope_config.scale <= 1
                    else 0.1 * math.log(target.rope_config.scale) + 1.0
                )
                * mscale
            )
            target.rope_config.extrapolation_factor = rope_override_args.get(
                "extrapolation_factor", 1.0
            )
            
            logging.info(
                f"Applied rope_scaling (yarn): "
                f"style: {target.rope_config.style}, "
                f"scale: {target.rope_config.scale}, "
                f"max_pos: {target.rope_config.max_pos}, "
                f"factor1: {target.rope_config.factor1}, "
                f"factor2: {target.rope_config.factor2}, "
                f"mscale: {target.rope_config.mscale}, "
                f"extrapolation_factor: {target.rope_config.extrapolation_factor}"
            )
        else:
            logging.info(
                f"Applied rope_scaling: style: {target.rope_config.style}"
            )

    def __init__(self, *args, **kwargs):
        """Initialize ModelConfig with quant_algo member and default values."""
        super().__init__(*args, **kwargs)
        self.quant_algo: QuantAlgo = QuantAlgo()
        
        # Set default model architecture values (can be overridden by subclasses)
        if not hasattr(self, 'activation_type') or self.activation_type is None:
            self.activation_type = "SiGLU"
        if not hasattr(self, 'norm_type') or self.norm_type is None:
            self.norm_type = "rmsnorm"
        if not hasattr(self, 'has_post_decoder_layernorm'):
            self.has_post_decoder_layernorm = True
        
        # Additional Python-only fields
        self.is_mtp: bool = False
        self.normalize_lm_head_weight: bool = False
        self.has_lm_head_bias: bool = False
        self.tie_word_embeddings: bool = False
        # Model loading related fields
        self.ckpt_path: str = ""
        self.ptuning_path: Optional[str] = None
        # mm_related_params will be set to VitParameters() by GptInitModelParameters
        self.mm_related_params: Any = None
        self.src_quantization_bit: int = 0
        self.config_dtype: Optional[str] = None

    def update_inter_padding_size(
        self, tp_size: int, ep_size: int, dp_size: int, hw_kernel_config: Any
    ) -> None:
        """
        Update inter_padding_size and moe_inter_padding_size based on quantization and hardware config.
        
        Args:
            tp_size: Tensor parallel size
            ep_size: Expert parallel size
            dp_size: Data parallel size
            hw_kernel_config: Hardware kernel config (used for use_swizzleA check)
        """
        if tp_size * dp_size != ep_size:
            raise ValueError(
                f"tp_size:{tp_size} * dp_size:{dp_size} != ep_size:{ep_size}"
            )
        # new tp_size just only for moe
        if self.quant_algo.isGroupwise():
            align_size = tp_size * self.quant_algo.getGroupSize()
            moe_align_size = self.quant_algo.getGroupSize()
        else:
            align_size = tp_size * 64
            moe_align_size = 64
            if self.quant_algo.isFp8PTPC():
                moe_align_size = 128
        
        self.inter_padding_size = self.inter_size + (
            get_pad_size(self.inter_size, align_size)
            if (self.quant_algo.isQuant() or hw_kernel_config.use_swizzleA)
            else 0
        )
        
        if self.head_num_kv <= 0:
            self.head_num_kv = self.head_num
        if self.inter_padding_size <= 0:
            self.inter_padding_size = self.inter_size

        if self.moe_inter_padding_size <= 0:
            self.moe_inter_padding_size = self.inter_size
        if self.moe_inter_padding_size > 0:
            moe_align_size = moe_align_size if self.quant_algo.isQuant() else 8
            self.moe_inter_padding_size = self.moe_inter_padding_size + (
                get_pad_size(self.moe_inter_padding_size, moe_align_size)
            )

        # Handle layer_inter_padding_size if layer_inter_size exists
        if hasattr(self, 'layer_inter_size') and self.layer_inter_size:
            layer_inter_padding_size = []
            for idx in range(len(self.layer_inter_size)):
                inter_size = self.layer_inter_size[idx]
                layer_inter_padding_size.append(
                    inter_size
                    + (
                        get_pad_size(inter_size, align_size)
                        if (self.quant_algo.isQuant() or hw_kernel_config.use_swizzleA)
                        else 0
                    )
                )
            self.layer_inter_padding_size = layer_inter_padding_size

        logging.info(
            f"update_inter_padding_size: {self.inter_padding_size}, moe_inter_padding_size: {self.moe_inter_padding_size}, layer_inter_size: {getattr(self, 'layer_inter_size', None)}"
        )

    def update_tokenizer_special_tokens(self, tokenizer:
        """Update special tokens from tokenizer to ModelConfig.
        
        Args:
            tokenizer: Tokenizer instance with stop_words_id_list, stop_words_str_list, and eos_token_idß
        """
        self.special_tokens.stop_words_id_list_ += tokenizer.stop_words_id_list
        self.special_tokens.stop_words_str_list_ += tokenizer.stop_words_str_list
        self.special_tokens.eos_token_id_ = tokenizer.eos_token_id

    def init_precision_config(
        self,
        ckpt_path: str,
        quantization: str,
        data_type_str: Optional[str],
        kv_cache_dtype_str: Optional[str],
        config_dtype: Optional[str],
    ) -> None:
        """
        Initialize precision configuration from checkpoint and quantization settings.
        This method sets quant_algo, data_type, and kv_cache_data_type directly on the model config.
        
        Args:
            ckpt_path: Path to checkpoint directory
            quantization: Quantization method string
            data_type_str: Data type string (optional)
            kv_cache_dtype_str: KV cache data type string (optional)
            config_dtype: Config data type string (optional)
        """
        quant_config = QuantizationConfig.load_from_ckpt(ckpt_path)
        if not quant_config:
            if quantization:
                quant_config = init_quant_config(quantization)
                logging.info(f"need_load_quant by {quant_config.get_method()}")
        
        if quant_config:
            self.quant_algo.setQuantAlgo(
                quant_config.get_algo().lower(),
                quant_config.bits,
                quant_config.group_size(),
            )

        # Verify the data_type
        data_type, kv_cache_data_type = self._get_and_verify_dtype(
            quant_config, data_type_str, kv_cache_dtype_str, config_dtype
        )

        # Set data_type and kv_cache_data_type directly
        # This uses ModelConfig's __setattr__ which handles string-to-enum conversion
        self.data_type = data_type.to_str()
        self.kv_cache_data_type = kv_cache_data_type.to_str()
        logging.info(
            f"quant_config: {quant_config}, data_type:{self.data_type}, kv_cache_data_type: {self.kv_cache_data_type}"
        )

    def _get_and_verify_dtype(
        self,
        quant_config: Optional[QuantizationConfig],
        data_type_str: Optional[str],
        kv_cache_dtype_str: Optional[str],
        config_dtype: Optional[str],
    ) -> tuple[WEIGHT_TYPE, WEIGHT_TYPE]:
        """
        Get and verify data types based on quantization config and input strings.
        
        Args:
            quant_config: Quantization configuration
            data_type_str: Data type string (optional)
            kv_cache_dtype_str: KV cache data type string (optional)
            config_dtype: Config data type string (optional)
            
        Returns:
            Tuple of (data_type, kv_cache_data_type)
        """
        data_type: Optional[WEIGHT_TYPE] = None
        config_dtype_parsed = (
            WEIGHT_TYPE.from_str(config_dtype) if config_dtype else None
        )
        if data_type_str:
            data_type = WEIGHT_TYPE.from_str(data_type_str)
            logging.info(f"set data_type by args: {data_type}")

        if not data_type or data_type == WEIGHT_TYPE.AUTO:
            data_type = config_dtype_parsed if config_dtype_parsed else WEIGHT_TYPE.FP16
            logging.info(
                f"data_type is not set or it's auto,we will use config_dtype:{config_dtype_parsed} or {WEIGHT_TYPE.FP16}"
            )
        if quant_config and isinstance(quant_config, Fp8BlockWiseQuantConfig):
            data_type = WEIGHT_TYPE.BF16  # now fp8_block_wise only support bf16
            logging.info(f"now fp8_block_wise only support bf16")
        elif quant_config and quant_config.get_method().lower() in [
            "smooth_quant",
            "omni_quant",
        ]:
            data_type = WEIGHT_TYPE.FP16

        if config_dtype_parsed and data_type != config_dtype_parsed:
            if data_type == WEIGHT_TYPE.FP32:
                # Upcasting to float32 is allowed.
                logging.info("Upcasting %s to %s.", config_dtype_parsed, data_type)
            elif config_dtype_parsed == WEIGHT_TYPE.FP32:
                # Downcasting from float32 to float16 or bfloat16 is allowed.
                logging.info("Downcasting %s to %s.", config_dtype_parsed, data_type)
            else:
                # Casting between float16 and bfloat16 is allowed with a warning.
                logging.warning("Casting %s to %s.", config_dtype_parsed, data_type)

        kv_cache_data_type: Optional[WEIGHT_TYPE] = (
            WEIGHT_TYPE.from_str(kv_cache_dtype_str)
            if kv_cache_dtype_str
            else data_type
        )
        if quant_config and quant_config.get_method().lower() == "fp8":
            kv_cache_data_type = WEIGHT_TYPE.FP8

        if kv_cache_data_type == WEIGHT_TYPE.AUTO:
            kv_cache_data_type: WEIGHT_TYPE = data_type

        if quant_config:
            quant_config.verify_compute_dtype_and_kv_cache_dtype(
                data_type.to_torch_dtype(), kv_cache_data_type.to_torch_dtype()
            )
        return (data_type, kv_cache_data_type)

    def update_task_type_use_kvcache(self) -> None:
        """
        Update task_type and use_kvcache based on checkpoint path.
        This method checks the task type from the checkpoint and updates the model configuration accordingly.
        """
        task_type = check_task_type(self.ckpt_path)
        # Set task_type using the setter method which handles string-to-enum conversion
        self.set_task_type(task_type.value)
        self.use_kvcache_ = task_type == TaskType.LANGUAGE_MODEL
        logging.info(
            f"model task type: {task_type}, use_kvcache: {self.use_kvcache_}"
        )

    def update_stop_words_from_env(self, generate_env_config) -> None:
        """
        Update stop_words_str and stop_words_id from environment variables.
        
        Args:
            generate_env_config: GenerateEnvConfig object containing stop_words configuration
        """
        env_stop_words_str = generate_env_config.stop_words_str
        env_stop_words_id = generate_env_config.stop_words_list
        env_stop_words_str_list = (
            json.loads(env_stop_words_str) if env_stop_words_str else []
        )
        env_stop_words_id_list = (
            json.loads(env_stop_words_id) if env_stop_words_id else []
        )
        env_force_stop = generate_env_config.force_stop_words
        if env_force_stop:
            self.special_tokens.stop_words_str_list_ = env_stop_words_str_list
            self.special_tokens.stop_words_id_list_ = env_stop_words_id_list
        else:
            self.special_tokens.stop_words_str_list_ = (
                self.special_tokens.stop_words_str_list_ + env_stop_words_str_list
            )
            self.special_tokens.stop_words_id_list_ = (
                self.special_tokens.stop_words_id_list_ + env_stop_words_id_list
            )

        logging.info(
            f"use stop_words_str_list [{self.special_tokens.stop_words_str_list_ }],"
            f" stop_words_id_list [{self.special_tokens.stop_words_id_list_}]"
        )


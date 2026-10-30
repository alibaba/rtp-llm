"""MiniMax-M3.1 checkpoint configuration on the current model contract."""

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.minimax_m31_kv_cache import build_m31_kv_cache_descs


def configure_m31_model(config: ModelConfig, checkpoint: dict) -> None:
    text = checkpoint.get("text_config", checkpoint)
    attn = config.attn_config
    config.hidden_size = int(text["hidden_size"])
    config.num_layers = int(text["num_hidden_layers"])
    attn.head_num = int(text["num_attention_heads"])
    attn.kv_head_num = int(text.get("num_key_value_heads", attn.head_num))
    attn.size_per_head = int(text.get("head_dim", config.hidden_size // attn.head_num))
    config.vocab_size = int(text["vocab_size"])
    config.layernorm_eps = float(text.get("rms_norm_eps", 1e-6))
    config.tie_word_embeddings = bool(text.get("tie_word_embeddings", False))
    config.max_seq_len = max(
        config.max_seq_len, int(text.get("max_position_embeddings", 8192))
    )
    attn.rope_config.style = 1
    attn.rope_config.base = int(text.get("rope_theta", attn.rope_config.base))
    attn.rope_config.dim = int(
        text.get(
            "rotary_dim",
            attn.size_per_head * float(text.get("partial_rotary_factor", 1.0)),
        )
    )
    config.qk_norm = bool(text.get("use_qk_norm", True))
    config.activation_type = "SiGLU"
    config.has_pre_decoder_layernorm = False
    config.has_post_decoder_layernorm = True
    config.norm_type = "rmsnorm"
    config.enable_fp32_lm_head = False
    freq = text.get("moe_layer_freq", [1] * config.num_layers)
    if len(freq) != config.num_layers:
        raise ValueError("MiniMax-M3.1 moe_layer_freq length does not match layers")
    config.moe_layer_index = [i for i, enabled in enumerate(freq) if enabled]
    config.expert_num = int(text["num_local_experts"])
    config.moe_k = int(text["num_experts_per_tok"])
    config.moe_inter_size = int(text["intermediate_size"])
    shared_count = int(text.get("n_shared_experts", 0))
    shared_size = int(text.get("shared_intermediate_size", text["intermediate_size"]))
    config.inter_size = max(
        shared_count * shared_size,
        int(text.get("dense_intermediate_size", shared_count * shared_size)),
    )
    config.n_shared_experts = shared_count
    config.moe_style = 2 if shared_count else 1
    config.scoring_func = 1 if text.get("scoring_func") == "sigmoid" else 0
    config.has_moe_norm = True
    config.routed_scaling_factor = float(text.get("routed_scaling_factor", 1.0))
    config.moe_n_group = 1
    config.moe_topk_group = 1
    config.swiglu_alpha = float(text.get("swiglu_alpha", 0.0))
    config.swiglu_limit = float(text.get("swiglu_limit", 0.0))

    sparse = text.get("sparse_attention_config", {})
    if not sparse.get("use_sparse_attention"):
        raise ValueError("MiniMax-M3.1 requires sparse attention")
    sparse_freq = sparse.get("sparse_attention_freq", [])
    if len(sparse_freq) != config.num_layers or not all(sparse_freq):
        raise ValueError("MiniMax-M3.1 requires sparse attention in every layer")
    block = int(sparse["sparse_block_size"])
    if block != 128:
        raise ValueError("MiniMax-M3.1 sparse block size must be 128")
    index_dim = int(sparse["sparse_index_dim"])
    attn.indexer_head_dim = index_dim
    init_blocks = int(
        sparse.get(
            "sparse_init_block",
            (int(sparse.get("sparse_init_tokens", 0)) + block - 1) // block,
        )
    )
    local_blocks = int(
        sparse.get(
            "sparse_local_block",
            (int(sparse.get("sparse_local_tokens", 0)) + block - 1) // block + 1,
        )
    )
    config.msa_sparse_config = {
        "sparse_layer_ids": list(range(config.num_layers)),
        "disable_value_layer_ids": [
            i
            for i, enabled in enumerate(sparse.get("sparse_disable_index_value", []))
            if enabled
        ],
        "idx_head_dim": index_dim,
        "num_idx_heads": int(sparse["sparse_num_index_heads"]),
        "topk_blocks": int(sparse["sparse_topk_blocks"]),
        "block_size": block,
        "init_blocks": init_blocks,
        "local_blocks": local_blocks,
        "score_type": str(sparse.get("sparse_score_type", "max")),
        "idx_k_fp8_mode": 3,
        "nvfp4_kv_cache": True,
    }
    quant = checkpoint.get("quantization_config", {})
    config.prepacked_nvfp4_moe = (
        str(quant.get("moe_quant_algo", "")).upper() == "NVFP4"
        and str(quant.get("moe_quant_format", "")).lower() == "nvfp4-pack-quantized"
    )


def finalize_m31_kv_cache(config: ModelConfig) -> None:
    """Declare native pages after runtime KV settings and layer overrides."""
    attn = config.attn_config
    physical = int(attn.tokens_per_block)
    kernel = int(attn.kernel_tokens_per_block)
    if physical == 64 and kernel == 64:
        attn.tokens_per_block = 128
        attn.kernel_tokens_per_block = 128
    elif physical != 128 or kernel != 128:
        raise ValueError(
            "MiniMax-M3.1 NVFP4 requires physical and kernel page size 128; "
            f"got {physical} and {kernel}"
        )
    config.kv_cache_spec_descs = build_m31_kv_cache_descs(
        config.num_layers,
        attn.kv_head_num,
        attn.size_per_head,
        int(attn.indexer_head_dim),
    )

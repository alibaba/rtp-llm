import torch

from rtp_llm.models_py.modules.base.common.zero_centered_rms_norm import (
    ZeroCenteredFusedQKRMSNorm,
)
from rtp_llm.ops import AttentionConfigs, RopeStyle
from rtp_llm.ops.fused_rope_kvcache_op import (
    FusedRopeAttnParams,
    FusedRopeKVCachePrefillOpQKVOut,
    _get_fused_rope_kvcache,
)


class MainQkNormRopeFixture:
    def setup_case(self, heads, rows, prefix=0):
        cfg = AttentionConfigs()
        cfg.head_num, cfg.kv_head_num, cfg.size_per_head = heads, 1, 256
        cfg.kernel_tokens_per_block, cfg.max_seq_len = 128, 32768
        cfg.use_logn_attn = False
        cfg.rope_config.style = RopeStyle.Mrope
        cfg.rope_config.index_factor, cfg.rope_config.dim = 3, 64
        cfg.rope_config.base, cfg.rope_config.scale = 10000000.0, 1.0
        (
            cfg.rope_config.mrope_dim1,
            cfg.rope_config.mrope_dim2,
            cfg.rope_config.mrope_dim3,
        ) = (11, 11, 10)
        plain = AttentionConfigs()
        for name in (
            "head_num",
            "kv_head_num",
            "size_per_head",
            "kernel_tokens_per_block",
            "max_seq_len",
            "use_logn_attn",
        ):
            setattr(plain, name, getattr(cfg, name))
        plain.rope_config.style = RopeStyle.No
        positions = torch.randint(0, 32768, (rows, 3), device="cuda", dtype=torch.int32)
        cu = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
        length = torch.tensor([rows], dtype=torch.int32).pin_memory()
        prefixes = torch.tensor([prefix], dtype=torch.int32).pin_memory()
        pages = (prefix + rows + 127) // 128
        table = torch.arange(1, pages + 1, device="cuda", dtype=torch.int32).reshape(
            1, pages
        )
        offset = _get_fused_rope_kvcache().convert_offset_to_block_array(table)
        params = FusedRopeAttnParams(
            offset,
            None,
            None,
            positions.flatten(),
            cu,
            cu,
            length,
            prefixes,
            length,
            rows,
            prefix,
            rows + prefix,
            False,
            torch.bfloat16,
        )
        norm = ZeroCenteredFusedQKRMSNorm(
            torch.randn(256, device="cuda") * 0.07,
            torch.randn(256, device="cuda") * 0.07,
            heads,
            1,
            256,
        )
        source = torch.randn(
            rows, (heads + 2) * 256, device="cuda", dtype=torch.bfloat16
        )
        return (
            cfg,
            norm,
            source,
            positions,
            params,
            FusedRopeKVCachePrefillOpQKVOut(cfg),
            FusedRopeKVCachePrefillOpQKVOut(plain),
            pages + 1,
        )

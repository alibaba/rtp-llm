import itertools
from dataclasses import dataclass
from typing import Optional, Tuple

import flashinfer
import torch
from flashinfer.cascade import merge_state
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from rtp_llm.models_py.modules.factory.attention.cuda_impl.headwise import (
    HeadWisePrefillAttnOp,
)
from rtp_llm.ops import AttentionConfigs, ParallelismConfig
from rtp_llm.ops.compute_ops import PyAttentionInputs

flex_attention = torch.compile(flex_attention, dynamic=True)


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class Case:
    batch_size: int
    kv_len: int
    qo_len: int
    window_left: int
    num_kv_heads: int
    num_qo_heads: int
    head_dim: int
    page_size: int


class KVCache:
    """rtp_llm 的 HeadWisePrefillAttnOp 期望的 cache wrapper（最小实现）"""

    def __init__(self, k_cache_base: torch.Tensor):
        self.k_cache_base = k_cache_base


# -----------------------------
# Runner
# -----------------------------
class HeadwisePrefillBenchmark:
    def __init__(
        self,
        device=torch.device("cuda:0"),
        dtype=torch.bfloat16,
        index_dtype=torch.int32,
        sink_token_num=4,
        swa_token_num=8192,
        seqlen_threshold=16384,
        workspace_bytes=256 * 1024 * 1024,
        flashinfer_backend="fa2",
    ):
        self.device = device
        self.dtype = dtype
        self.index_dtype = index_dtype

        self.sink_token_num = int(sink_token_num)
        self.swa_token_num = int(swa_token_num)
        self.seqlen_threshold = int(seqlen_threshold)
        self.flashinfer_backend = flashinfer_backend

        self.workspace_buffer = torch.empty(
            workspace_bytes, dtype=torch.int8, device=self.device
        )

        # cache op/config（当 case 的关键维度变化时重建）
        self._cached_op_key: Optional[Tuple[int, int, int, int]] = None
        self._op: Optional[HeadWisePrefillAttnOp] = None

    # -------------------------
    # Config / op factory
    # -------------------------
    def _build_attention_config(self, case: Case) -> AttentionConfigs:
        cfg = AttentionConfigs()
        cfg.head_num = case.num_qo_heads
        cfg.kv_head_num = case.num_kv_heads
        cfg.size_per_head = case.head_dim
        cfg.tokens_per_block = case.page_size
        cfg.headwise_config = {
            "sink_token_num": self.sink_token_num,
            "swa_token_num": self.swa_token_num,
            "seqlen_threshold": self.seqlen_threshold,
            "0": [0] * case.num_qo_heads,  # layer 0: all non-retrieval heads（示例）
        }
        return cfg

    @staticmethod
    def _build_parallelism_config() -> ParallelismConfig:
        pcfg = ParallelismConfig()
        pcfg.tp_rank = 0
        return pcfg

    def _get_or_create_op(self, case: Case) -> HeadWisePrefillAttnOp:
        # 这些变化会影响 op 的 plan/shape
        key = (case.num_qo_heads, case.num_kv_heads, case.head_dim, case.page_size)
        if self._op is None or self._cached_op_key != key:
            attn_cfg = self._build_attention_config(case)
            par_cfg = self._build_parallelism_config()
            self._op = HeadWisePrefillAttnOp(attn_cfg, par_cfg)
            self._cached_op_key = key
        return self._op

    # -------------------------
    # Tensor builders
    # -------------------------
    def _make_attn_inputs(self, case: Case) -> PyAttentionInputs:
        attn_inputs = PyAttentionInputs()
        attn_inputs.input_lengths = torch.tensor(
            [case.qo_len] * case.batch_size, device=self.device, dtype=torch.int32
        )
        attn_inputs.prefix_lengths = torch.tensor(
            [case.kv_len] * case.batch_size, device=self.device, dtype=torch.int32
        )

        num_pages_per_seq = (case.kv_len + case.page_size - 1) // case.page_size
        total_num_pages = num_pages_per_seq * case.batch_size

        # [B, num_pages]
        attn_inputs.kv_cache_block_id_device = torch.arange(
            0, total_num_pages, device=self.device, dtype=self.index_dtype
        ).view(case.batch_size, num_pages_per_seq)
        return attn_inputs

    def _make_paged_kv_cache(
        self, k: torch.Tensor, v: torch.Tensor, case: Case
    ) -> KVCache:
        num_pages_per_seq = (case.kv_len + case.page_size - 1) // case.page_size
        total_num_pages = num_pages_per_seq * case.batch_size

        # 期望：k/v 是 [B*kv_len, Hk, D]
        k_cache = k.view(
            total_num_pages, case.page_size, case.num_kv_heads, case.head_dim
        ).transpose(1, 2)
        v_cache = v.view(
            total_num_pages, case.page_size, case.num_kv_heads, case.head_dim
        ).transpose(1, 2)

        kv_cache_base = torch.stack(
            [k_cache, v_cache], dim=1
        )  # [pages, 2, Hk, page, D]
        return KVCache(kv_cache_base)

    def _make_qkv(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, case: Case
    ) -> torch.Tensor:
        """
        你的 HeadWisePrefillAttnOp.forward() 会把 fmha_input reshape 成 [q_len, Hq+2Hk, D]
        所以这里必须提供长度为 q_len 的“qkv token 序列”。

        对 qo_len != kv_len：
          - q 是当前 query tokens（长度 qo_len）
          - k/v 在这个 fmha_input 里通常只需要“占位/对齐”，真实 K/V 从 paged kv cache 里取
          - 你原代码用 k[:qo_len], v[:qo_len] 作为占位，这里继续沿用
        """
        return torch.cat(
            [q, k[: case.batch_size * case.qo_len], v[: case.batch_size * case.qo_len]],
            dim=1,
        )

    # -------------------------
    # References
    # -------------------------
    def _make_flex_mask(self, s_q: int, s_k: int, window_left: int) -> torch.Tensor:
        """
        q_idx ∈ [0, s_q)
        kv_idx ∈ [0, s_k)
        ret_len = s_k - s_q: 让 q 的第 0 个 token 对齐到 kv 的 ret_len 位置
        """
        ret_len = s_k - s_q

        def _sink_mask(b, h, q_idx, kv_idx):
            causal_window = q_idx + ret_len >= kv_idx
            sliding_window = q_idx + ret_len - kv_idx <= window_left
            sink_window = kv_idx < self.sink_token_num
            return (sliding_window | sink_window) & causal_window

        return create_block_mask(
            _sink_mask, None, None, s_q, s_k, device=self.device, _compile=False
        )

    def forward_flex(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, case: Case
    ) -> torch.Tensor:
        B, S_q, S_k = case.batch_size, case.qo_len, case.kv_len
        H_q, H_k, D = case.num_qo_heads, case.num_kv_heads, case.head_dim

        q_ = q.view(B, S_q, H_q, D).transpose(1, 2)  # [B, Hq, S_q, D]
        k_ = k.view(B, S_k, H_k, D).transpose(1, 2)  # [B, Hk, S_k, D]
        v_ = v.view(B, S_k, H_k, D).transpose(1, 2)  # [B, Hk, S_k, D]

        mask = self._make_flex_mask(S_q, S_k, case.window_left)
        out = flex_attention(
            q_, k_, v_, block_mask=mask, enable_gqa=True
        )  # [B, Hq, S_q, D]
        return out.transpose(1, 2).reshape(B * S_q, H_q, D)

    # -------------------------
    # rtp op path
    # -------------------------
    def forward_headwise_op(
        self,
        case: Case,
        qkv: torch.Tensor,
        cache: KVCache,
        attn_inputs: PyAttentionInputs,
    ) -> torch.Tensor:
        op = self._get_or_create_op(case)
        _ = op.support(attn_inputs)
        op.prepare(attn_inputs)
        op._get_headwise_config(0)
        return op.forward(qkv, cache, None)

    # -------------------------
    # Runner
    # -------------------------
    def run_one(self, case: Case, rtol=1e-2, atol=1e-2):
        torch.cuda.synchronize()

        # data
        q = torch.randn(
            case.batch_size * case.qo_len,
            case.num_qo_heads,
            case.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        k = torch.randn(
            case.batch_size * case.kv_len,
            case.num_kv_heads,
            case.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        v = torch.randn(
            case.batch_size * case.kv_len,
            case.num_kv_heads,
            case.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

        cache = self._make_paged_kv_cache(k, v, case)
        qkv = self._make_qkv(q, k, v, case)
        attn_inputs = self._make_attn_inputs(case)

        out_op = self.forward_headwise_op(
            case, qkv, cache, attn_inputs
        )  # [B*qo_len, Hq, D] flatten-ish
        out_ref = self.forward_flex(q, k, v, case)

        # 对齐 shape
        out_op2 = out_op.view(case.batch_size * case.qo_len, -1)
        out_ref2 = out_ref.view(case.batch_size * case.qo_len, -1)

        torch.testing.assert_close(out_ref2, out_op2, rtol=rtol, atol=atol)


def make_cases(
    batch_sizes,
    kv_lens,
    qo_lens,
    window_lefts,
    num_kv_heads,
    num_qo_heads,
    head_dims,
    page_sizes,
):
    return [
        Case(b, kv, qo, w, hk, hq, d, ps)
        for b, qo, w, hk, hq, d, ps, kv in itertools.product(
            batch_sizes,
            qo_lens,
            window_lefts,
            num_kv_heads,
            num_qo_heads,
            head_dims,
            page_sizes,
            kv_lens,
        )
    ]


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    BATCH_SIZES = [1]
    KV_LENS = [32768]
    QO_LENS = [27648, 22528, 17408, 12288]
    WINDOW_LEFTS = [8192]
    NUM_KV_HEADS = [1]
    NUM_QO_HEADS = [8]
    HEAD_DIMS = [128]
    PAGE_SIZES = [128]

    bench = HeadwisePrefillBenchmark()

    for c in make_cases(
        BATCH_SIZES,
        KV_LENS,
        QO_LENS,
        WINDOW_LEFTS,
        NUM_KV_HEADS,
        NUM_QO_HEADS,
        HEAD_DIMS,
        PAGE_SIZES,
    ):
        print(f"case = {c}")
        bench.run_one(c)

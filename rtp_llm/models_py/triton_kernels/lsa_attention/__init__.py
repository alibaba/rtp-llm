"""LongCat Sparse Attention (LSA) kernels and references.

Package layout:

- :mod:`.reference` — PyTorch semantic reference (indexer scores, SI budget
  selection, absorbed-MQA sparse MLA attention).
- :mod:`.sparse_mla_triton` — Triton sparse MLA forward kernel (first runnable
  baseline for the B300 attention operator).
- :mod:`.indexer_triton` — Triton Lightning-indexer scoring kernel (baseline).

See ``build_logs/longcat-lsa-b300-20261017/{attention-graph,plan}.md`` for the
computation graph, bottlenecks and the staged optimization plan.
"""

from rtp_llm.models_py.triton_kernels.lsa_attention.reference import (
    DEFAULT_INIT_TOKENS,
    DEFAULT_INDEX_TOPK,
    DEFAULT_KV_LORA_RANK,
    DEFAULT_LOCAL_TOKENS,
    absorb_queries,
    indexer_scores,
    lsa_forward,
    select_slots,
    sparse_mla_attention,
    sparse_mla_attention_naive,
)
from rtp_llm.models_py.triton_kernels.lsa_attention.lsa_forward import lsa_attention
from rtp_llm.models_py.triton_kernels.lsa_attention.merge_triton import merge_attention_triton
from rtp_llm.models_py.triton_kernels.lsa_attention.sparse_mla_triton import (
    sparse_mla_attention_auto,
    sparse_mla_fwd,
    sparse_mla_fwd_split,
)
from rtp_llm.models_py.triton_kernels.lsa_attention.swa_dense_triton import swa_dense_fwd

try:  # optional: B300 (sm_103) two-kernel prefill path, self-contained Triton
    from rtp_llm.models_py.triton_kernels.lsa_attention.sparse_mla_two_kernel import (
        pack_kv,
        pack_q,
        sparse_mla_two_kernel,
    )
except Exception:  # pragma: no cover - optional: needs triton + sm_103
    pack_kv = None  # type: ignore[assignment]
    pack_q = None  # type: ignore[assignment]
    sparse_mla_two_kernel = None  # type: ignore[assignment]

try:  # optional: requires the `flash-mla` package and SM90+
    from rtp_llm.models_py.triton_kernels.lsa_attention.lsa_attention_flashmla import (
        lsa_attention_flashmla,
    )
    from rtp_llm.models_py.triton_kernels.lsa_attention.sparse_mla_flashmla import (
        flashmla_available,
        sparse_flashmla_fwd,
    )
except Exception:  # pragma: no cover - optional dependency
    lsa_attention_flashmla = None  # type: ignore[assignment]
    flashmla_available = None  # type: ignore[assignment]
    sparse_flashmla_fwd = None  # type: ignore[assignment]

__all__ = [
    "DEFAULT_INIT_TOKENS",
    "DEFAULT_INDEX_TOPK",
    "DEFAULT_KV_LORA_RANK",
    "DEFAULT_LOCAL_TOKENS",
    "absorb_queries",
    "indexer_scores",
    "lsa_attention",
    "lsa_forward",
    "merge_attention_triton",
    "select_slots",
    "sparse_mla_attention",
    "sparse_mla_attention_auto",
    "sparse_mla_attention_naive",
    "sparse_mla_fwd",
    "sparse_mla_fwd_split",
    "swa_dense_fwd",
    "sparse_mla_two_kernel",
    "pack_q",
    "pack_kv",
    "flashmla_available",
    "lsa_attention_flashmla",
    "sparse_flashmla_fwd",
]
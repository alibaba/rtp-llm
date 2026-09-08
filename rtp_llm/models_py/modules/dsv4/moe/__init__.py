"""DeepSeek-V4 MoE: Gate + Expert + MoE.

Public API (preserved across the moe.py → moe/ refactor):
  - MoE: nn.Module instantiated by ``dsv4/block.py``
  - Gate: nn.Module used standalone in some unit tests
  - _has_fp8_fp4_grouped_kernel: probe used by ``test/grouped_moe_equivalence_test.py``

Internal layout (Phase 1 complete):
  - moe_layer.py: thin MoE orchestrator (gate + shared + strategy dispatch)
  - gate.py: Gate + _use_fused_gate
  - expert.py: Expert + mandatory fused SiLU path
  - quant_layouts.py: FP4_BLOCK / FP8_BLOCK / per-token-cast helper
  - mega_buf.py: symm-mem buffer cache + capability gates
  - strategies/{base,mega,grouped_fp4,local_loop,deepep}.py:
    routed-expert compute strategies + select_strategy

See ``.claude/plans/optimized-riding-mist.md`` for design + Phase 2 plan.
"""

from .gate import Gate
from .moe_layer import MoE
from .strategies import _has_fp8_fp4_grouped_kernel  # also populates registry

__all__ = ["MoE", "Gate", "_has_fp8_fp4_grouped_kernel"]

__all__ = ["MoE", "Gate", "_has_fp8_fp4_grouped_kernel"]

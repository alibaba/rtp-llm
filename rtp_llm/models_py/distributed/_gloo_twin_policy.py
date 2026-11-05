"""R20/C1+C2: pure routing policy for CPU-allgather gloo twins.

Deliberately free of torch / rtp_llm imports so bare-shell CPU suites can
load it through the hermetic namespace-stub pattern (see
test_v41_gloo_twin_policy_cpu.py). collective_torch owns the twin registry
and the wire calls; this module owns only the flag + predicate semantics.

Flags (value-based, "1" only; anything else — including presence with "0" —
keeps the qualified staged path):
  DSV41_CPU_ALLGATHER_GLOO  route pure-CPU cpp_allgather payloads (the V4.1
                            PP admission agreement packets) over the group's
                            gloo twin instead of pageable H2D + NCCL +
                            blocking D2H staging.
  DSV41_AGREE_DP_LEAN       exchange the _agree_dp signature over the gloo
                            twin on CPU (same staging removal for the
                            Python-side agreement).

Either flag also enables twin CREATION at group setup; with both off no
twins are created and every route is exactly the historical one.
"""

import os

CPU_ALLGATHER_GLOO_FLAG = "DSV41_CPU_ALLGATHER_GLOO"
AGREE_DP_LEAN_FLAG = "DSV41_AGREE_DP_LEAN"


def value_flag(name: str) -> bool:
    """Value-based gate: only an explicit "1" enables; "0"/absent/garbage do not."""
    return os.environ.get(name, "0") == "1"


def cpu_gloo_twins_enabled() -> bool:
    return value_flag(CPU_ALLGATHER_GLOO_FLAG) or value_flag(AGREE_DP_LEAN_FLAG)


def agree_dp_lean_enabled() -> bool:
    return value_flag(AGREE_DP_LEAN_FLAG)


def cpu_allgather_via_gloo(
    gloo_pg, recv_is_cuda: bool, send_is_cuda: bool, inplace: bool
) -> bool:
    """Routing predicate for cpp_allgather's gloo branch.

    Non-inplace only: the inplace path aliases its send slice inside the
    recv buffer, and gloo allgather-into-tensor with an aliased input is
    unverified — staged NCCL keeps that (unused by the V4.1 admission
    callers) case.
    """
    return gloo_pg is not None and not inplace and not recv_is_cuda and not send_is_cuda

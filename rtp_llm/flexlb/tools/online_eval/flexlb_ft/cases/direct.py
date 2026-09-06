"""Direct-path category: the client-direct gRPC contract.

Theme: requests that bypass the master entirely (the load-client direct
deployment shape) must surface engine-side fault injections on the
GenerateStreamCall entry and recover once the fault clears, with no
engine-side inflight residue.  direct_generate_error pins the
client-direct contract for the generate_error injection type — it runs
under every profile because the direct stub sequence never passes
through the master's dispatcher.
"""

from __future__ import annotations

from typing import Optional

from ..context import CaseContext, CaseDef, rid_base
from ..engine_ops import (
    StreamHandle,
    StreamSnapshot,
    clear_type_all,
    engine_inflight_clean,
    inject_type_all,
)

DIRECT_CASES: list[CaseDef] = []

STREAM_TIMEOUT_S = 15.0


def case(name: str, profiles=None, requires=None, source: str = ""):
    def deco(fn):
        DIRECT_CASES.append(
            CaseDef(
                name=name,
                category="direct",
                fn=fn,
                profiles=profiles,
                requires=requires,
                source=source,
            )
        )
        return fn

    return deco


def _prefill_names(ops) -> list[str]:
    snap = ops.snapshot()
    return [e["name"] for e in snap.get("engines", []) if e.get("role") == "prefill"]


# ===========================================================================
# Direct-path case (migrated from the legacy injection family, task #85
# category reorg — rid_base family "chaos" -> "direct")
# ===========================================================================


@case(
    "direct_generate_error",
    source="gap G6/G7: /inject type=generate_error (GenerateStreamCall entry, client-direct path)",
)
def inject_generate_error(ctx: CaseContext):
    """generate_error is checked ONLY at the engine's GenerateStreamCall
    entry (JavaMockEngineCluster.generateStreamCall: onError before any
    request state is registered).

    Profile semantics (v2, task #55): under the v1 mode axis ALL master
    modes delivered via enqueueBatch + FetchResponse (direct-run evidence:
    generate_stream_rpcs=0 while enqueue_rpcs=3 / fetch_response_rpcs=3),
    so the fault was structurally unreachable for master traffic and the
    case pinned the client-direct contract.  Under v2 the BATCH
    dispatcher is still unreachable, but the NON_BATCH dispatcher routes
    client-sent GenerateStreamCall traffic through this exact check — a
    master-routed variant of this case is dedicated-phase material.  The
    contract pinned here is the CLIENT-DIRECT path (the load-client
    direct deployment shape; same direct-stub sequence EngineOps already
    uses for worker_cancel), which does not pass through the master at
    all — so the case runs unconditionally under every profile:

    inject -> the direct stream fails immediately with the injected
    error and registers no engine-side inflight; clear -> a fresh direct
    request completes normally."""
    ops = ctx.ops()
    base = rid_base(ctx, "direct")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    snap = ops.snapshot_by_name()
    target = None
    for n in names:
        entry = snap.get(n) or {}
        addr = entry.get("grpc_addr")
        if addr:
            target = str(addr)
            break
    if not target:
        return False, "no prefill engine address in snapshot"

    def direct_request(rid: int) -> tuple[Optional[str], object]:
        input_pb = ops.build_generate_input(rid)
        stub = ops.pb2_grpc.RpcServiceStub(ops._channel(target))
        call = stub.GenerateStreamCall(input_pb, timeout=30.0)
        handle = StreamHandle(call, StreamSnapshot())
        handle.wait_end(STREAM_TIMEOUT_S)
        if handle.snap.error:
            return str(handle.snap.error), handle
        if not handle.snap.completed:
            return "stream did not complete", handle
        return None, handle

    try:
        rid0 = ops.next_request_id(base)
        err0, _ = direct_request(rid0)
        if err0:
            return False, f"baseline direct request failed: {err0}"

        inject_type_all(ops, names, "generate_error")
        try:
            rid1 = ops.next_request_id(base)
            err1, _ = direct_request(rid1)
            # Cross-process the engine's onError(RuntimeException("injected
            # generate_error")) reaches the client as grpc status 2
            # (UNKNOWN) with an EMPTY message — the text is not transmitted
            # (verified round 5: grpc_message:"", grpc_status:2) — so the
            # assertion is error-arrived, same contract as the fetch_error
            # case; causality comes from the inject/clear sandwich.
            error_ok = err1 is not None
        finally:
            clear_type_all(ops, names, "generate_error")

        rid2 = ops.next_request_id(base)
        err2, _ = direct_request(rid2)
        engine_clean, engine_detail = engine_inflight_clean(ops, names)

        passed = error_ok and err2 is None and engine_clean
        return passed, (
            f"direct_target={target}, "
            f"error_surfaced={error_ok} ({err1}), "
            f"recovered={err2 is None}"
            f"{'' if err2 is None else ' err=' + err2[:60]}, "
            f"engine_inflight_clean={engine_clean}({engine_detail})"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "generate_error")

"""Exact checkpoint FI adapter integration: original Torch vs fused glue.

One prepared adapter, checkpoint weights, workspace and FI callable serve both
arms. Only module-scope prepare/finish helper bindings are patched. The real FI
operator is never mocked. This equivalence gate cannot waive the separately
retained BF16 or intended-A8 stress numerical failures.
"""

from contextlib import contextmanager
from copy import copy
import json
import os
from pathlib import Path
import weakref

import pytest
import torch

from rtp_llm.models_py.modules.dsv41.test.test_sm90_flashinfer_checkpoint import (
    _inputs,
    checkpoint_case,
)
from rtp_llm.models_py.modules.dsv41.test.test_sm90_flashinfer_local import _bits, _sha

pytestmark = pytest.mark.skipif(
    not os.environ.get("DSV41_FLASHINFER_CHECKPOINT"),
    reason="explicit actual-checkpoint glue integration qualification",
)


def _torch_prepare(x, weights, indices, num_experts):
    # Original adapter operations, including legality before INT32 narrowing.
    valid = (indices >= 0) & (indices < num_experts)
    live_rows = valid.any(dim=1, keepdim=True)
    safe_indices = torch.where(valid, indices, num_experts).to(torch.int32)
    safe_weights = torch.where(valid, weights, 0.0)
    safe_x = torch.where(live_rows, x, 0.0)
    return safe_x, safe_weights, safe_indices, live_rows


def _torch_finish(out, live_rows, output_dtype=torch.float32):
    return torch.where(live_rows, out, 0.0).to(output_dtype)


@contextmanager
def _hooks(module, prepare, finish):
    calls = dict(prepare=0, finish=0)

    def tracked_prepare(*args, **kwargs):
        calls["prepare"] += 1
        return prepare(*args, **kwargs)

    def tracked_finish(*args, **kwargs):
        calls["finish"] += 1
        return finish(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "prepare_flashinfer_inputs", tracked_prepare)
        patch.setattr(module, "finish_flashinfer_output", tracked_finish)
        yield calls
    assert calls["prepare"] > 0 and calls["prepare"] == calls["finish"], calls


def _capture(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out


def _changed_inputs(amplitude, phase):
    if phase < 4:
        return _inputs(amplitude, phase)
    x, weights, ids = _inputs(amplitude, 2)
    # These INT64 invalid IDs would alias real experts if narrowed too early.
    ids[:, 0] = 2**32 + 47
    ids[:, 1] = -(2**63)
    ids[:, 3] = 2**63 - 1
    valid = (ids >= 0) & (ids < 48)
    weights.masked_fill_(~valid, float("nan"))
    x.masked_fill_(~valid.any(-1)[:, None], float("nan"))
    return x, weights, ids


@pytest.mark.parametrize("amplitude", [1.0, 16.0], ids=["normal", "amplified16"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.no_grad()
def test_checkpoint_adapter_glue_eager_and_graph_bitexact(
    checkpoint_case, amplitude, dtype
):
    from rtp_llm.models_py.modules.dsv41.moe import _sm90_flashinfer_moe as module
    from rtp_llm.models_py.modules.dsv41.moe._sm90_flashinfer_glue import (
        finish_flashinfer_output,
        prepare_flashinfer_inputs,
    )

    # Fail clearly if the intended real module-scope hook is absent or another
    # helper was selected. These assertions precede temporary baseline patches.
    assert module.prepare_flashinfer_inputs is prepare_flashinfer_inputs
    assert module.finish_flashinfer_output is finish_flashinfer_output
    model = checkpoint_case.model
    assert model.num_experts == 48 and model.hidden_size == 5120
    assert model.intermediate_size == 2304
    assert (8, 8) in model._workspaces
    workspace = model._workspaces[(8, 8)]
    operator = model._op
    live = _inputs(amplitude, 0)
    fn = lambda: model.forward_local(*live, output_dtype=dtype)
    with _hooks(module, _torch_prepare, _torch_finish) as baseline_capture_calls:
        baseline_graph, baseline_output = _capture(fn)
    with _hooks(
        module, prepare_flashinfer_inputs, finish_flashinfer_output
    ) as fused_capture_calls:
        fused_graph, fused_output = _capture(fn)
    records = []
    for phase in (0, 1, 2, 3, 4, 0):
        for destination, source in zip(live, _changed_inputs(amplitude, phase)):
            destination.copy_(source)
        saved = tuple(_sha(t) for t in live)
        with _hooks(module, _torch_prepare, _torch_finish):
            original_eager = fn()
        with _hooks(module, prepare_flashinfer_inputs, finish_flashinfer_output):
            fused_eager = fn()
        # Same workspace, serialized execution; never concurrently replay two
        # graphs that share FI scratch. No tactic retuning between either arm.
        baseline_graph.replay()
        fused_graph.replay()
        torch.cuda.synchronize()
        _bits(fused_eager, original_eager)
        _bits(baseline_output, original_eager)
        _bits(fused_output, original_eager)
        if phase == 3:
            _bits(fused_output, torch.zeros_like(fused_output))
        assert saved == tuple(_sha(t) for t in live)
        assert model._op is operator
        assert model._workspaces[(8, 8)] is workspace
        records.append(
            dict(
                phase=phase,
                eager_fused_vs_torch="BITEXACT",
                graph_fused_vs_torch="BITEXACT",
            )
        )
    assert module.prepare_flashinfer_inputs is prepare_flashinfer_inputs
    assert module.finish_flashinfer_output is finish_flashinfer_output
    report = dict(
        evidence="RTP_MEASURED",
        checkpoint=checkpoint_case.manifest["checkpoint"],
        index_sha256=checkpoint_case.manifest["index_sha256"],
        shape=dict(rows=8, topk=8, experts=48, dim=5120, intermediate=2304),
        amplitude=amplitude,
        output_dtype=str(dtype),
        same_prepared_adapter_operator_workspace=True,
        baseline_capture_helper_calls=baseline_capture_calls,
        fused_capture_helper_calls=fused_capture_calls,
        cases=records,
        scope="glue-only equivalence; no BF16/A8 numerical gate waiver; NOT full-model quality acceptance",
    )
    print("FLASHINFER_GLUE_INTEGRATION " + json.dumps(report), flush=True)
    root = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "."))
    root.mkdir(parents=True, exist_ok=True)
    (root / f"glue_integration_a{int(amplitude)}_{dtype}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


@torch.no_grad()
def test_checkpoint_eager_workspace_growth_preserves_graph(
    checkpoint_case, monkeypatch
):
    from rtp_llm.models_py.modules.dsv41.moe import _sm90_flashinfer_moe as module

    model = checkpoint_case.model
    peer = copy(model)
    peer._workspaces = {}
    pool = {}
    monkeypatch.setattr(module, "_EAGER_STREAM_WORKSPACES", pool)
    live = _inputs(1.0, 0)
    graph, graph_out = _capture(lambda: model.forward_local(*live))

    # EP1 decode graph warmup must prepare a smaller shape than setup's max,
    # even though ordinary eager prefill no longer populates graph workspaces.
    from rtp_llm.models_py.modules.dsv41.moe.strategies.sm90_flashinfer import (
        SM90FlashInferStrategy,
    )

    cfg = copy(model.cfg)
    cfg.ep_size = 1
    cfg.n_routed_experts = model.num_experts
    cfg.n_activated_experts = 8
    cfg.local_expert_start = 0
    cfg.local_expert_end = model.num_experts
    monkeypatch.setenv("DSV41_SM90_DECODE_GRAPH", "1")
    monkeypatch.setenv("DSV41_SM90_DECODE_GRAPH_MAX_ROWS", "8")
    strategy = SM90FlashInferStrategy(cfg)
    strategy._local = model
    small = tuple(t[:3].contiguous() for t in live)
    small_graph, small_out = _capture(lambda: strategy(*small))
    assert (3, 8) in model._workspaces
    assert not pool
    prepared = dict(model._workspaces)
    graph_pool = dict(module._STREAM_WORKSPACES)

    def unexpected_prepare(*args, **kwargs):
        pytest.fail("dynamic eager shapes must not prepare or warm up synthetic inputs")

    monkeypatch.setattr(model, "prepare", unexpected_prepare)
    monkeypatch.setattr(peer, "prepare", unexpected_prepare)
    operator = model._op
    calls = []

    def tracked_operator(**kwargs):
        calls.append(kwargs["input"].shape[0])
        return operator(**kwargs)

    monkeypatch.setattr(model, "_op", tracked_operator)
    monkeypatch.setattr(peer, "_op", tracked_operator)
    key = (live[0].device.index, torch.cuda.current_stream().cuda_stream)
    for iteration, rows in enumerate((3, 16, 5, 64, 8)):
        inputs = tuple(
            t.repeat((rows + 7) // 8, 1)[:rows].contiguous()
            for t in _changed_inputs(1.0, iteration)
        )
        saved = tuple(_sha(t) for t in inputs)
        required = model._workspace_nbytes(rows, 8)
        # Independent exact-sized storage supplies the same real FI operation.
        reference_scratch = torch.empty(
            required, device=inputs[0].device, dtype=torch.uint8
        )
        expected = model.forward_local(*inputs, workspace=reference_scratch)
        del reference_scratch
        previous = weakref.ref(pool[key]) if key in pool else None
        previous_size = pool[key].numel() if key in pool else 0
        before = len(calls)
        adapter = model if iteration % 2 == 0 else peer
        actual = adapter._forward_into_buf(*inputs, 0, 48)
        assert len(calls) == before + 1  # No synthetic warmup.
        _bits(actual, expected)
        assert saved == tuple(_sha(t) for t in inputs)
        assert len(pool) == 1 and pool[key].numel() >= required
        if previous is not None:
            if required > previous_size:
                assert previous() is None  # No per-layer stale strong reference.
            else:
                assert pool[key] is previous()
        assert model._workspaces.keys() == prepared.keys()
        assert all(model._workspaces[k] is v for k, v in prepared.items())
        assert not peer._workspaces
        assert module._STREAM_WORKSPACES.keys() == graph_pool.keys()
        assert all(module._STREAM_WORKSPACES[k] is v for k, v in graph_pool.items())

    # A second stream gets independent eager storage, with explicit ownership
    # dependencies for inputs and outputs. It never borrows the first slot.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        side_output = peer._forward_into_buf(*live, 0, 48)
    torch.cuda.current_stream().wait_stream(side)
    side_key = (live[0].device.index, side.cuda_stream)
    assert len(pool) == 2
    assert pool[side_key].data_ptr() != pool[key].data_ptr()
    _bits(side_output, model.forward_local(*live))

    for destination, source in zip(live, _changed_inputs(1.0, 4)):
        destination.copy_(source)
    graph.replay()
    _bits(graph_out, model.forward_local(*live))
    for destination, source in zip(small, live):
        destination.copy_(source[:3])
    small_graph.replay()
    _bits(small_out, model.forward_local(*small))
    assert all(model._workspaces[k] is v for k, v in prepared.items())

    def unexpected_size_query(*args, **kwargs):
        pytest.fail("zero-row eager input must not allocate workspace")

    monkeypatch.setattr(peer, "_workspace_nbytes", unexpected_size_query)
    empty = tuple(t[:0] for t in live)
    assert peer._forward_into_buf(*empty, 0, 48).shape == (0, 5120)
    assert len(pool) == 2

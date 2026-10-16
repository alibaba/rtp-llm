"""Exact checkpoint FI adapter integration: original Torch vs fused glue.

One prepared adapter, checkpoint weights, workspace and FI callable serve both
arms. Only module-scope prepare/finish helper bindings are patched. The real FI
operator is never mocked. This equivalence gate cannot waive the separately
retained BF16 or intended-A8 stress numerical failures.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path

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

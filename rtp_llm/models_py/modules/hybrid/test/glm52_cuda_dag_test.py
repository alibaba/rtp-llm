import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe import GLM5MegaMoE
from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_fp8 import GLM5MegaMoEFP8
from rtp_llm.models_py.modules.hybrid import glm52_cuda_dag as bridge


def test_switch_defaults_off(monkeypatch) -> None:
    monkeypatch.delenv("RTP_LLM_GLM52_CUDA_DAG", raising=False)
    assert bridge.resolve_glm52_cuda_dag_enabled() is False
    monkeypatch.setenv("RTP_LLM_GLM52_CUDA_DAG", "1")
    assert bridge.resolve_glm52_cuda_dag_enabled() is True


def test_page_table_keeps_64_token_contract() -> None:
    page_table = torch.empty((2, 16), dtype=torch.int32)
    expanded_table = torch.empty((2, 1024), dtype=torch.int32)
    inputs = SimpleNamespace(
        kv_cache_block_id_device=page_table,
        kv_cache_kernel_block_id_device=expanded_table,
    )
    assert bridge._page_block_table(inputs) is page_table


def _selection_adapter(*, has_indexer: bool, reuses_indexer: bool):
    adapter = object.__new__(bridge.Glm52CudaDagAdapter)
    adapter._disabled_reason = None
    adapter.self_attn = SimpleNamespace(
        has_indexer=has_indexer,
        reuse_topk_indices=reuses_indexer,
    )
    adapter._moe_bindings = lambda rows: object()
    return adapter


def test_model_execution_is_selected_once_for_all_layers() -> None:
    full = _selection_adapter(has_indexer=True, reuses_indexer=False)
    main = _selection_adapter(has_indexer=False, reuses_indexer=True)
    full._dispatch = lambda *args: bridge._Dispatch(
        implementation=object(),
        params=object(),
        rows=16,
        has_indexer=True,
        reuses_indexer=False,
        block_table=torch.empty((2, 1), dtype=torch.int32),
        context_lens=torch.empty((16,), dtype=torch.int32),
        reason=None,
    )
    layers = [
        SimpleNamespace(_glm52_cuda_dag_adapter=full),
        SimpleNamespace(_glm52_cuda_dag_adapter=main),
    ]
    kv_cache = SimpleNamespace(get_layer_cache=lambda _: object())
    assert bridge.should_enable_glm52_cudadag(
        layers, 2, torch.empty((16, 6144)), object(), kv_cache
    )

    full._disabled_reason = "unsupported"
    assert not bridge.should_enable_glm52_cudadag(
        layers, 2, torch.empty((16, 6144)), object(), kv_cache
    )


class _Plan:
    def __init__(self) -> None:
        self.calls = 0
        self.outputs = SimpleNamespace(
            attention_output=torch.empty((16, 6144)),
            residual=torch.empty((16, 6144)),
            sparse_indices=torch.empty((16, 2048), dtype=torch.int32),
            reused_indices=None,
            moe_hidden_states=torch.empty((16, 6144)),
            routed_indices=torch.empty((16, 8), dtype=torch.int64),
            routed_weights=torch.empty((16, 8)),
        )

    def __call__(self, hidden, residual):
        self.calls += 1
        return self.outputs


def test_forward_caches_full_indexer_plan(monkeypatch) -> None:
    adapter = object.__new__(bridge.Glm52CudaDagAdapter)
    adapter.layer_idx = 0
    adapter._plans = {}
    implementation = SimpleNamespace(_glm_cuda_dag_storage_plans={})
    dispatch = bridge._Dispatch(
        implementation=implementation,
        params=object(),
        rows=16,
        has_indexer=True,
        reuses_indexer=False,
        block_table=torch.empty((2, 1), dtype=torch.int32),
        context_lens=torch.empty((16,), dtype=torch.int32),
        reason=None,
    )
    moe = SimpleNamespace(pointer_key=(1, 2, 3, 4))
    plan = _Plan()
    prepared = 0

    adapter._dispatch = lambda *args: dispatch
    adapter._moe_bindings = lambda rows: moe

    def prepare(*args):
        nonlocal prepared
        prepared += 1
        return plan

    adapter._prepare_full_plan = prepare
    monkeypatch.setattr(bridge, "_is_capturing", lambda: False)

    hidden = torch.empty((16, 6144))
    residual = torch.empty_like(hidden)
    first = adapter.forward(hidden, residual, object(), object(), None)
    second = adapter.forward(hidden, residual, object(), object(), None)

    assert prepared == 1
    assert plan.calls == 2
    assert first.topk_indices is plan.outputs.sparse_indices
    assert second.moe_hidden_states is plan.outputs.moe_hidden_states


def test_main_only_requires_prior_topk_indices() -> None:
    adapter = object.__new__(bridge.Glm52CudaDagAdapter)
    adapter.layer_idx = 1
    adapter._plans = {}
    adapter._dispatch = lambda *args: bridge._Dispatch(
        implementation=object(),
        params=object(),
        rows=16,
        has_indexer=False,
        reuses_indexer=True,
        block_table=None,
        context_lens=None,
        reason=None,
    )
    adapter._moe_bindings = lambda rows: SimpleNamespace(pointer_key=())

    with pytest.raises(RuntimeError, match="prior TopK"):
        adapter.forward(
            torch.empty((16, 6144)),
            torch.empty((16, 6144)),
            object(),
            object(),
            None,
        )


@pytest.mark.parametrize(
    ("op_type", "expected_call", "sync_module"),
    [
        (GLM5MegaMoE, "fp8_fp4_mega_moe", "mega_moe"),
        (GLM5MegaMoEFP8, "fp8_fp8_mega_moe", "mega_moe_fp8"),
    ],
)
def test_prepacked_mega_moe_skips_input_packer(
    op_type, expected_call: str, sync_module: str
) -> None:
    op = object.__new__(op_type)
    torch.nn.Module.__init__(op)
    op.cfg = SimpleNamespace(layer_id=0)
    op._mega_buf = SimpleNamespace(num_max_tokens_per_rank=16)
    op._mega_y = torch.empty((16, 8), dtype=torch.bfloat16)
    op._mega_l1_w = op._mega_l1_sf = torch.empty(0)
    op._mega_l2_w = op._mega_l2_sf = torch.empty(0)
    op._input_packer = Mock()
    op._maybe_pre_kernel_barrier = Mock()
    deep_gemm = SimpleNamespace(
        fp8_fp4_mega_moe=Mock(),
        fp8_fp8_mega_moe=Mock(),
    )
    sync_target = (
        "rtp_llm.models_py.modules.glm5_mega_moe."
        f"{sync_module}._sync_cuda_graph_warmup_ranks"
    )

    with patch.dict(sys.modules, {"deep_gemm": deep_gemm}), patch(sync_target):
        result = op.forward_prepacked(torch.empty((6, 8), dtype=torch.bfloat16))

    assert result.shape == (6, 8)
    op._input_packer.pack.assert_not_called()
    getattr(deep_gemm, expected_call).assert_called_once()

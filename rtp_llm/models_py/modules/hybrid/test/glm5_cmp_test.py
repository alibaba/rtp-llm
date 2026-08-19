import inspect
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from rtp_llm.models_py.model_desc.generic_moe import (
    GenericMoeDecoderLayer,
    GenericMoeLayer,
)
from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe import GLM5MegaMoE
from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_fp8 import GLM5MegaMoEFP8
from rtp_llm.models_py.modules.hybrid import glm5_cmp as bridge


def test_switch_defaults_off(monkeypatch) -> None:
    monkeypatch.delenv("RTP_LLM_GLM5_CMP", raising=False)
    assert bridge.resolve_glm5_cmp_enabled() is False
    monkeypatch.setenv("RTP_LLM_GLM5_CMP", "1")
    assert bridge.resolve_glm5_cmp_enabled() is True


def test_page_table_keeps_64_token_contract() -> None:
    page_table = torch.empty((2, 16), dtype=torch.int32)
    expanded_table = torch.empty((2, 1024), dtype=torch.int32)
    inputs = SimpleNamespace(
        kv_cache_block_id_device=page_table,
        kv_cache_kernel_block_id_device=expanded_table,
    )
    assert bridge._page_block_table(inputs) is page_table


def test_router_weight_materializes_rtp_transpose_view() -> None:
    storage = torch.arange(6144 * 256, dtype=torch.bfloat16).reshape(6144, 256)
    weight = storage.T
    assert weight.shape == (256, 6144)
    assert not weight.is_contiguous()

    prepared = bridge.Glm5Cmp._prepare_router_weight(
        SimpleNamespace(gate=SimpleNamespace(weight=weight))
    )

    assert prepared is not None
    assert prepared.is_contiguous()
    torch.testing.assert_close(prepared, weight)


def test_dense_cmp_requires_ue8m0_fp8_input() -> None:
    cmp = object.__new__(bridge.Glm5Cmp)
    cmp.parallelism_config = SimpleNamespace(
        tp_size=1,
        get_attn_tp_size=lambda: 1,
        prefill_cp_config=None,
    )
    cmp.config = SimpleNamespace(
        model_type="glm_5",
        attn_config=SimpleNamespace(use_mla=True, kernel_tokens_per_block=64),
    )
    cmp.self_attn = SimpleNamespace(has_indexer=False)
    cmp._is_moe_layer = False
    cmp.mlp = SimpleNamespace(
        accepts_fp8_input=True,
        up_proj=SimpleNamespace(scale_ue8m0=True),
    )

    assert cmp._static_disabled_reason() is None
    cmp.mlp.up_proj.scale_ue8m0 = False
    assert cmp._static_disabled_reason() == (
        "Dense GLM5 CMP requires FP8 input with UE8M0 scales"
    )


def test_parallelism_ignores_prefill_cp_metadata_on_decode() -> None:
    prefill_cp_config = SimpleNamespace(
        is_enabled=lambda: True,
        is_prefill_enabled=lambda: True,
        kv_cache_sharded=True,
        prefill_cp_size=8,
    )
    decode_parallelism = SimpleNamespace(
        tp_size=1,
        get_attn_tp_size=lambda: 1,
        prefill_cp_config=prefill_cp_config,
    )
    prefill_parallelism = SimpleNamespace(
        tp_size=8,
        get_attn_tp_size=lambda: 1,
        prefill_cp_config=prefill_cp_config,
    )

    assert bridge._unsupported_parallelism_reason(decode_parallelism) is None
    assert (
        bridge._unsupported_parallelism_reason(prefill_parallelism)
        == "GLM5 CMP requires TP=1"
    )


def test_cmp_constructor_does_not_load_runtime_or_set_pdl() -> None:
    config = SimpleNamespace(
        model_type="glm_5",
        moe_layer_index=(0,),
        attn_config=SimpleNamespace(use_mla=True, kernel_tokens_per_block=64),
    )
    parallelism = SimpleNamespace(tp_size=1, get_attn_tp_size=lambda: 1)
    self_attn = SimpleNamespace(has_indexer=False)
    mlp = SimpleNamespace()

    with patch.object(bridge, "_load_ops") as load_ops, patch.object(
        bridge, "_configure_deep_gemm_pdl"
    ) as configure_pdl, patch.object(
        bridge.Glm5Cmp, "_prepare_router_weight"
    ) as prepare_router_weight:
        cmp = bridge.Glm5Cmp(
            layer_idx=0,
            config=config,
            parallelism_config=parallelism,
            self_attn=self_attn,
            input_layernorm=object(),
            mlp=mlp,
            post_attention_layernorm=object(),
        )

    assert cmp._disabled_reason is None
    assert cmp.ops is None
    load_ops.assert_not_called()
    configure_pdl.assert_not_called()
    prepare_router_weight.assert_not_called()


def _selection_cmp(*, has_indexer: bool, reuses_indexer: bool, is_moe_layer: bool):
    cmp = object.__new__(bridge.Glm5Cmp)
    cmp._disabled_reason = None
    cmp._is_moe_layer = is_moe_layer
    cmp.ops = object()
    cmp.self_attn = SimpleNamespace(
        has_indexer=has_indexer,
        reuse_topk_indices=reuses_indexer,
    )
    return cmp


def test_model_execution_is_selected_once_for_all_layers() -> None:
    full = _selection_cmp(has_indexer=True, reuses_indexer=False, is_moe_layer=False)
    main = _selection_cmp(has_indexer=False, reuses_indexer=True, is_moe_layer=True)
    full._unsupported_call_reason = lambda *args: None
    layers = [
        SimpleNamespace(cmp=full),
        SimpleNamespace(cmp=main),
    ]
    kv_cache = SimpleNamespace(get_layer_cache=lambda _: object())
    assert bridge.should_enable_glm5_cmp(
        layers, 2, torch.empty((16, 6144)), object(), kv_cache
    )

    main._moe_prepack_disabled_reason = "unsupported activation layout"
    assert bridge.should_enable_glm5_cmp(
        layers, 2, torch.empty((16, 6144)), object(), kv_cache
    )

    main._disabled_reason = "unsupported"
    assert not bridge.should_enable_glm5_cmp(
        layers, 2, torch.empty((16, 6144)), object(), kv_cache
    )


def test_dynamic_fallback_does_not_load_runtime_or_set_pdl() -> None:
    cmp = _selection_cmp(
        has_indexer=True,
        reuses_indexer=False,
        is_moe_layer=False,
    )
    cmp.ops = None
    cmp._unsupported_call_reason = Mock(return_value="ordinary prefill is unsupported")
    cmp._initialize_for_cmp = Mock()
    layers = [SimpleNamespace(cmp=cmp)]
    kv_cache = SimpleNamespace(get_layer_cache=lambda _: object())

    with patch.object(bridge, "_load_ops") as load_ops, patch.object(
        bridge, "_configure_deep_gemm_pdl"
    ) as configure_pdl:
        enabled = bridge.should_enable_glm5_cmp(
            layers, 1, torch.empty((16, 6144)), object(), kv_cache
        )

    assert not enabled
    load_ops.assert_not_called()
    configure_pdl.assert_not_called()
    cmp._initialize_for_cmp.assert_not_called()


def test_selected_call_initializes_all_layers_before_setting_pdl() -> None:
    full = _selection_cmp(
        has_indexer=True,
        reuses_indexer=False,
        is_moe_layer=False,
    )
    main = _selection_cmp(
        has_indexer=False,
        reuses_indexer=True,
        is_moe_layer=True,
    )
    full.ops = None
    main.ops = None
    full._unsupported_call_reason = Mock(return_value=None)
    ops = object()

    def initialize(cmp):
        def apply(loaded_ops):
            cmp.ops = loaded_ops

        cmp._initialize_for_cmp = Mock(side_effect=apply)

    initialize(full)
    initialize(main)
    layers = [SimpleNamespace(cmp=full), SimpleNamespace(cmp=main)]
    kv_cache = SimpleNamespace(get_layer_cache=lambda _: object())

    with patch.object(bridge, "_load_ops", return_value=ops) as load_ops, patch.object(
        bridge, "_configure_deep_gemm_pdl"
    ) as configure_pdl, patch.object(bridge.logger, "info") as log_info:
        assert bridge.should_enable_glm5_cmp(
            layers, 2, torch.empty((16, 6144)), object(), kv_cache
        )
        assert bridge.should_enable_glm5_cmp(
            layers, 2, torch.empty((16, 6144)), object(), kv_cache
        )

    load_ops.assert_called_once_with()
    full._initialize_for_cmp.assert_called_once_with(ops)
    main._initialize_for_cmp.assert_called_once_with(ops)
    configure_pdl.assert_called_once_with(ops)
    log_info.assert_called_once_with("GLM5 CMP activated for %d layers", 2)


def test_moe_prepacked_input_abi() -> None:
    valid = (
        torch.empty((16, 6144), dtype=torch.float8_e4m3fn),
        torch.empty((16, 48), dtype=torch.int32),
        torch.empty((16, 8), dtype=torch.int64),
        torch.empty((16, 8), dtype=torch.float32),
    )

    def make_cmp(views):
        cmp = object.__new__(bridge.Glm5Cmp)
        cmp._is_moe_layer = True
        cmp._moe_prepack_disabled_reason = None
        cmp._moe_prepack_abi_validated = False
        cmp.mlp = SimpleNamespace(
            fused_moe=SimpleNamespace(prepacked_input_views=Mock(return_value=views))
        )
        return cmp

    assert make_cmp(valid).moe_prepacked_input_views(16) == valid

    invalid = (valid[0], valid[1].float(), valid[2], valid[3])
    assert make_cmp(invalid).moe_prepacked_input_views(16) == (
        None,
        None,
        None,
        None,
    )


def test_fused_shared_expert_is_not_added_twice() -> None:
    expected = torch.empty((6, 8), dtype=torch.bfloat16)
    shared_expert = Mock(side_effect=AssertionError("shared expert ran twice"))
    layer = SimpleNamespace(
        fake_balance_expert=None,
        fused_moe=SimpleNamespace(forward_prepacked=Mock(return_value=expected)),
        _use_mega_moe_fused_shared=True,
        shared_expert=shared_expert,
    )

    result = GenericMoeLayer.forward_prepacked(
        layer,
        torch.empty((6, 8), dtype=torch.bfloat16),
        torch.empty((6, 8), dtype=torch.int64),
        torch.empty((6, 8), dtype=torch.float32),
    )

    assert result is expected
    shared_expert.assert_not_called()


def test_integration_has_no_whole_attention_resource_cache() -> None:
    source = inspect.getsource(bridge.Glm5Cmp)
    assert "_LayerResources" not in source
    assert "_get_resources" not in source
    assert "paged_indexer_score_qblock" not in source
    assert "exact_topk_and_globalize" not in source
    assert "ops.sparse_mla" not in source


def test_side_streams_are_shared_once_per_device(monkeypatch) -> None:
    created: list[tuple[torch.device, int]] = []

    def make_stream(*, device: torch.device, priority: int = 0) -> object:
        created.append((device, priority))
        return object()

    monkeypatch.setattr(bridge, "_is_capturing", lambda: False)
    monkeypatch.setattr(bridge.torch.cuda, "Stream", make_stream)
    monkeypatch.setattr(bridge.Glm5Cmp, "_streams_by_device", {})

    first = bridge.Glm5Cmp._side_streams(torch.device("cuda:2"))
    second = bridge.Glm5Cmp._side_streams(torch.device("cuda:2"))
    other = bridge.Glm5Cmp._side_streams(torch.device("cuda:3"))

    assert first is second
    assert other is not first
    assert created == [
        (torch.device("cuda:2"), -1),
        (torch.device("cuda:2"), 0),
        (torch.device("cuda:2"), 0),
        (torch.device("cuda:3"), -1),
        (torch.device("cuda:3"), 0),
        (torch.device("cuda:3"), 0),
    ]


def test_decoder_forward_consumes_cmp_topk_before_flashmla() -> None:
    calls: list[str] = []
    topk_indices = torch.empty((16, 2048), dtype=torch.int32)
    mla_output = torch.empty((16, 64, 512))
    residual = torch.empty((16, 6144))
    query = torch.empty((16, 64, 576))
    moe_hidden = torch.empty((16, 6144))
    routed_indices = torch.empty((16, 8), dtype=torch.int64)
    routed_weights = torch.empty((16, 8))
    cmp = SimpleNamespace(
        has_indexer=True,
        moe_prepacked_input_views=Mock(
            return_value=(
                torch.empty((16, 6144), dtype=torch.float8_e4m3fn),
                torch.empty((16, 48), dtype=torch.int32),
                routed_indices,
                routed_weights,
            )
        ),
        mla_prologue=Mock(
            side_effect=lambda *args, **_kwargs: calls.append("pre")
            or (residual, query, topk_indices)
        ),
        sparse_mla=Mock(
            side_effect=lambda *args: calls.append("flashmla") or mla_output
        ),
        mla_post_moe_pre=Mock(
            side_effect=lambda *args, **kwargs: calls.append("post")
            or (moe_hidden, residual, routed_indices, routed_weights)
        ),
    )
    mlp = SimpleNamespace(
        forward_prepacked=Mock(
            side_effect=lambda *args: calls.append("ffn") or torch.empty((16, 6144))
        )
    )
    layer = object.__new__(GenericMoeDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.cmp = cmp
    layer.mlp = mlp

    result = layer.forward(
        torch.empty((16, 6144)),
        torch.empty((16, 6144)),
        object(),
        object(),
        None,
        enable_cmp=True,
    )

    assert calls == ["pre", "flashmla", "post", "ffn"]
    assert cmp.sparse_mla.call_args.args[1] is topk_indices
    assert result.topk_indices is topk_indices


def test_dense_decoder_uses_existing_mlp_path() -> None:
    topk_indices = torch.empty((16, 2048), dtype=torch.int32)
    mla_output = torch.empty((16, 64, 512))
    residual = torch.empty((16, 6144))
    query = torch.empty((16, 64, 576))
    attention_output = torch.empty((16, 6144))
    dense_residual = torch.empty((16, 6144))
    normalized = torch.empty((16, 6144))
    cmp = SimpleNamespace(
        has_indexer=False,
        moe_prepacked_input_views=Mock(return_value=(None, None, None, None)),
        mla_prologue=Mock(return_value=(residual, query, topk_indices)),
        sparse_mla=Mock(return_value=mla_output),
        mla_post_moe_pre=Mock(return_value=(attention_output, residual)),
    )
    mlp = Mock(return_value=torch.empty((16, 6144)))
    mlp.up_proj = SimpleNamespace(scale_ue8m0=True)
    layer = object.__new__(GenericMoeDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.cmp = cmp
    layer.mlp = mlp
    layer.post_attention_layernorm = Mock(return_value=(normalized, dense_residual))
    layer._fuse_post_norm_quant = False
    layer._fuse_post_norm_quant_moe = False

    result = layer.forward(
        torch.empty((16, 6144)),
        torch.empty((16, 6144)),
        object(),
        object(),
        None,
        enable_cmp=True,
    )

    layer.post_attention_layernorm.assert_called_once_with(attention_output, residual)
    mlp.assert_called_once_with(normalized)
    assert result.residual is dense_residual


def test_sparse_mla_calls_rtp_flashmla_directly() -> None:
    query = torch.empty((16, 64, 576))
    cache = torch.empty((32, 64, 1, 656), dtype=torch.uint8)
    topk = torch.empty((16, 2048), dtype=torch.int32)
    output = torch.empty((16, 64, 512))
    flashmla = SimpleNamespace(expects_paged_kv=True, forward=Mock(return_value=output))
    implementation = SimpleNamespace(fmha_impl=flashmla)
    cmp = object.__new__(bridge.Glm5Cmp)
    cmp.layer_idx = 3
    cmp._attention_impl = Mock(return_value=implementation)

    result = cmp.sparse_mla(query, topk, object(), SimpleNamespace(kv_cache_base=cache))

    assert result is output
    flashmla.forward.assert_called_once_with(query, cache, topk, layer_id=3)


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

"""Focused tests for dynamic decode backend eligibility and fatal probes.

Pure-CPU: DECODE_MHA_IMPS is monkeypatched with fake impls; the *real*
_is_fmha_impl_disabled (class-name matching) is exercised. No GPU / torch compute.
The probe tests also replace torch allocation, benchmarking, fatal termination,
and TP broadcast so failure-boundary behavior is deterministic.
"""

import contextlib
import os
import sys
import types
import unittest
from unittest import mock

from rtp_llm.models_py.modules.factory.attention.cuda_impl.benchmark_workspace import (
    in_benchmark_workspace_scope,
)
from rtp_llm.models_py.modules.factory.attention.dispatch import backend_selector

_ATTN_FACTORY = "rtp_llm.models_py.modules.factory.attention.attn_factory"


def _make_fake_impl(
    name,
    *,
    supports=True,
    support_error=None,
    construction_error=None,
    support_observer=None,
    parallelism_supports=True,
    graph_supports=True,
):
    """A minimal decode impl stub that always 'supports' everything."""

    class _Fake:
        construction_count = 0
        graph_support_check_count = 0

        @classmethod
        def support(cls, attn_configs, attn_inputs):
            if support_observer is not None:
                support_observer()
            if support_error is not None:
                raise support_error
            return supports

        @classmethod
        def support_parallelism_config(cls, parallelism_config):
            return parallelism_supports

        def __init__(self, attn_configs, attn_inputs, parallelism_config=None):
            type(self).construction_count += 1
            if construction_error is not None:
                raise construction_error

        def support_cuda_graph(self):
            type(self).graph_support_check_count += 1
            return graph_supports

    _Fake.__name__ = name
    return _Fake


def _fmha_config(**overrides):
    """SimpleNamespace mirroring the fields FMHAConfig exposes."""
    cfg = dict(
        enable_fmha=True,
        enable_trt_fmha=True,
        enable_paged_trt_fmha=True,
        enable_open_source_fmha=True,
        enable_paged_open_source_fmha=True,
        enable_trtv1_fmha=True,
        disable_flashinfer_native=False,
        enable_xqa=True,
        use_aiter_pa=True,
        use_asm_pa=True,
        use_triton_pa=False,
    )
    cfg.update(overrides)
    return types.SimpleNamespace(**cfg)


def _patch_impls(names):
    fakes = [_make_fake_impl(n) for n in names]
    return mock.patch(f"{_ATTN_FACTORY}.DECODE_MHA_IMPS", fakes)


_NAMES = ["PyFlashinferDecodeImpl", "XQADecodeImpl"]


# _eligible respects fmha_config
def test_eligible_excludes_flashinfer_when_disabled():
    with _patch_impls(_NAMES):
        eligible = backend_selector._eligible(
            None, None, None, _fmha_config(disable_flashinfer_native=True)
        )
    assert "PyFlashinferDecodeImpl" not in eligible
    assert "XQADecodeImpl" in eligible


def test_eligible_excludes_xqa_when_disabled():
    with _patch_impls(_NAMES):
        eligible = backend_selector._eligible(
            None, None, None, _fmha_config(enable_xqa=False)
        )
    assert "XQADecodeImpl" not in eligible
    assert "PyFlashinferDecodeImpl" in eligible


def test_eligible_keeps_all_when_nothing_disabled():
    with _patch_impls(_NAMES):
        eligible = backend_selector._eligible(None, None, None, _fmha_config())
    assert set(eligible) == set(_NAMES)


def test_eligible_none_fmha_config_keeps_all():
    with _patch_impls(_NAMES):
        eligible = backend_selector._eligible(None, None, None, None)
    assert set(eligible) == set(_NAMES)


def test_eligible_support_false_is_normal_and_does_not_construct():
    support_scope = []
    unsupported = _make_fake_impl(
        "PyFlashinferDecodeImpl",
        supports=False,
        support_observer=lambda: support_scope.append(in_benchmark_workspace_scope()),
    )
    with mock.patch(f"{_ATTN_FACTORY}.DECODE_MHA_IMPS", [unsupported]):
        eligible = backend_selector._eligible(None, None, None, _fmha_config())

    assert eligible == []
    assert unsupported.construction_count == 0
    assert support_scope == [True]
    assert not in_benchmark_workspace_scope()


def test_eligible_does_not_construct_only_to_check_cuda_graph_support():
    candidate = _make_fake_impl("PyFlashinferDecodeImpl")
    with mock.patch(f"{_ATTN_FACTORY}.DECODE_MHA_IMPS", [candidate]):
        eligible = backend_selector._eligible(None, None, None, _fmha_config())

    assert eligible == ["PyFlashinferDecodeImpl"]
    assert candidate.construction_count == 0
    assert candidate.graph_support_check_count == 0


class _FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _FakeCode:
    def __init__(self):
        self.value = -1

    def __getitem__(self, _index):
        return _FakeScalar(self.value)

    def __setitem__(self, _index, value):
        self.value = value


class _FailingCodeWrite(_FakeCode):
    def __setitem__(self, _index, value):
        raise RuntimeError("winner code write failed")


class _FailingCodeRead(_FakeCode):
    def __getitem__(self, _index):
        raise RuntimeError("winner code read failed")


class _FakeInputLengths:
    def size(self, dim):
        assert dim == 0
        return 1


class _FakeSequenceLengths:
    def numel(self):
        return 1

    def flatten(self):
        return self

    def __getitem__(self, _index):
        return _FakeScalar(8)


def _selection_fixture(*, tp_size=1, tp_rank=0, dp_rank=0):
    attn_configs = types.SimpleNamespace(max_seq_len=8)
    parallelism_config = types.SimpleNamespace(
        tp_size=tp_size,
        tp_rank=tp_rank,
        dp_rank=dp_rank,
        get_attn_tp_size=lambda: tp_size,
    )
    model = types.SimpleNamespace(
        layer_num=32,
        parallelism_config=parallelism_config,
        fmha_config=_fmha_config(),
        config=types.SimpleNamespace(
            max_seq_len=8,
            hybrid_attention_config=None,
            headwise_config=None,
            getAttentionConfigs=lambda _tp: attn_configs,
        ),
        kv_cache=types.SimpleNamespace(get_layer_cache=lambda _idx: object()),
    )
    attention_inputs = types.SimpleNamespace(
        input_lengths=_FakeInputLengths(),
        sequence_lengths=_FakeSequenceLengths(),
    )
    return (
        model,
        types.SimpleNamespace(attention_inputs=attention_inputs),
        attn_configs,
        parallelism_config,
    )


def _assert_fatal_probe(
    impl_cls,
    *,
    bench_side_effect=None,
    bench_result=None,
    synchronize_side_effect=None,
    support_probe=False,
    impl_lookup_side_effect=None,
    code=None,
):
    model, inputs, _, _ = _selection_fixture(tp_size=2)

    collective_module = types.ModuleType("collective_torch")
    collective_module.Group = types.SimpleNamespace(TP=object())
    collective_module.broadcast = mock.Mock()
    fatal = mock.Mock()
    fixed_priority = mock.Mock()
    bench = mock.Mock(side_effect=bench_side_effect, return_value=bench_result)

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(
                backend_selector,
                "_decode_registry",
                return_value=[impl_cls.__name__],
            )
        )
        stack.enter_context(
            mock.patch.object(backend_selector, "kv_grid", return_value=[8])
        )
        stack.enter_context(
            mock.patch.object(backend_selector, "_terminate_probe_worker", fatal)
        )
        stack.enter_context(
            mock.patch(f"{_ATTN_FACTORY}.get_fmha_impl", fixed_priority)
        )
        stack.enter_context(
            mock.patch.object(
                backend_selector.torch,
                "full",
                return_value=code if code is not None else _FakeCode(),
            )
        )
        stack.enter_context(
            mock.patch.object(
                backend_selector.backend_bench,
                "bench_backend_grid",
                bench,
            )
        )
        stack.enter_context(
            mock.patch.object(
                backend_selector.torch.cuda,
                "synchronize",
                side_effect=synchronize_side_effect,
            )
        )
        stack.enter_context(
            mock.patch.dict(
                sys.modules,
                {
                    "rtp_llm.models_py.distributed.collective_torch": collective_module,
                },
            )
        )
        if support_probe:
            stack.enter_context(
                mock.patch(f"{_ATTN_FACTORY}.DECODE_MHA_IMPS", [impl_cls])
            )
        else:
            stack.enter_context(
                mock.patch.object(
                    backend_selector,
                    "_eligible",
                    return_value=[impl_cls.__name__],
                )
            )
            stack.enter_context(
                mock.patch.object(
                    backend_selector,
                    "_impl_by_name",
                    return_value=impl_cls,
                    side_effect=impl_lookup_side_effect,
                )
            )

        try:
            backend_selector.run_backend_selection(model, inputs, warmup=0, iters=1)
        except RuntimeError as error:
            assert str(error) == "fatal probe termination returned unexpectedly"
        else:
            raise AssertionError("fatal probe unexpectedly returned to selection")

    fatal.assert_called_once()
    assert fatal.call_args.args[0] == 1
    assert fatal.call_args.args[1] == impl_cls.__name__
    probe_error = fatal.call_args.args[2]
    assert isinstance(probe_error, backend_selector._FatalProbeError)
    assert isinstance(probe_error.__cause__, RuntimeError)
    collective_module.broadcast.assert_not_called()
    fixed_priority.assert_not_called()
    if support_probe:
        bench.assert_not_called()


def test_support_probe_exception_is_fatal():
    impl_cls = _make_fake_impl(
        "PyFlashinferDecodeImpl",
        support_error=RuntimeError("support probe failed"),
    )
    _assert_fatal_probe(impl_cls, support_probe=True)


def test_candidate_construction_exception_is_fatal():
    impl_cls = _make_fake_impl(
        "FailingConstructionImpl",
        construction_error=RuntimeError("construction failed"),
    )

    def construct_candidate(
        candidate_cls,
        attn_configs,
        attn_inputs,
        _layer_kv_cache,
        parallelism_config,
        _kv_list,
        **_kwargs,
    ):
        candidate_cls(attn_configs, attn_inputs, parallelism_config)

    _assert_fatal_probe(impl_cls, bench_side_effect=construct_candidate)


def test_post_construction_benchmark_exception_is_fatal():
    impl_cls = _make_fake_impl("FailingBenchmarkImpl")
    _assert_fatal_probe(
        impl_cls,
        bench_side_effect=RuntimeError("capture failed"),
    )


def test_post_benchmark_synchronize_exception_is_fatal():
    impl_cls = _make_fake_impl("FailingSynchronizationImpl")
    _assert_fatal_probe(
        impl_cls,
        bench_result=[1.0],
        synchronize_side_effect=RuntimeError("synchronize failed"),
    )


def test_post_support_impl_lookup_exception_is_fatal():
    impl_cls = _make_fake_impl("FailingLookupImpl")
    _assert_fatal_probe(
        impl_cls,
        impl_lookup_side_effect=RuntimeError("implementation lookup failed"),
    )


def test_post_probe_winner_code_write_exception_is_fatal():
    impl_cls = _make_fake_impl("FailingWinnerWriteImpl")
    _assert_fatal_probe(
        impl_cls,
        bench_result=[1.0],
        code=_FailingCodeWrite(),
    )


def _assert_selection_control_failure(
    *, tp_rank, full_side_effect=None, broadcast_side_effect=None, code=None
):
    model, inputs, _, _ = _selection_fixture(tp_size=2, tp_rank=tp_rank)
    collective_module = types.ModuleType("collective_torch")
    collective_module.Group = types.SimpleNamespace(TP=object())
    collective_module.broadcast = mock.Mock(side_effect=broadcast_side_effect)
    fatal = mock.Mock()

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(
                backend_selector, "_decode_registry", return_value=["XQADecodeImpl"]
            )
        )
        stack.enter_context(
            mock.patch.object(backend_selector, "kv_grid", return_value=[8])
        )
        stack.enter_context(
            mock.patch.object(
                backend_selector.torch,
                "full",
                side_effect=full_side_effect,
                return_value=code if code is not None else _FakeCode(),
            )
        )
        stack.enter_context(
            mock.patch.object(backend_selector, "_select_on_root", return_value=None)
        )
        stack.enter_context(
            mock.patch.object(backend_selector, "_terminate_probe_worker", fatal)
        )
        stack.enter_context(
            mock.patch.dict(
                sys.modules,
                {
                    "rtp_llm.models_py.distributed.collective_torch": collective_module,
                },
            )
        )

        try:
            backend_selector.run_backend_selection(model, inputs)
        except RuntimeError as error:
            assert str(error) == "fatal probe termination returned unexpectedly"
        else:
            raise AssertionError(
                "fatal selection-control failure unexpectedly returned"
            )

    fatal.assert_called_once()
    assert fatal.call_args.args[0:2] == (1, "selection-control")
    assert isinstance(fatal.call_args.args[2], RuntimeError)
    return collective_module.broadcast


def test_control_tensor_allocation_failure_is_fatal_on_every_tp_rank():
    for tp_rank in (0, 1):
        broadcast = _assert_selection_control_failure(
            tp_rank=tp_rank,
            full_side_effect=RuntimeError("control tensor allocation failed"),
        )
        broadcast.assert_not_called()


def test_tp_broadcast_failure_is_fatal():
    broadcast = _assert_selection_control_failure(
        tp_rank=1,
        broadcast_side_effect=RuntimeError("TP broadcast failed"),
    )
    broadcast.assert_called_once()


def test_control_tensor_readback_failure_is_fatal():
    broadcast = _assert_selection_control_failure(
        tp_rank=1,
        code=_FailingCodeRead(),
    )
    broadcast.assert_called_once()


def test_non_root_logs_received_plan_after_tp_broadcast():
    model, inputs, _, _ = _selection_fixture(tp_size=2, tp_rank=1)
    code = _FakeCode()
    collective_module = types.ModuleType("collective_torch")
    collective_module.Group = types.SimpleNamespace(TP=object())

    def broadcast(selected_code, _src, group):
        assert group is collective_module.Group.TP
        selected_code[0] = 0

    collective_module.broadcast = mock.Mock(side_effect=broadcast)

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(
                backend_selector, "_decode_registry", return_value=["XQADecodeImpl"]
            )
        )
        stack.enter_context(
            mock.patch.object(backend_selector, "kv_grid", return_value=[8])
        )
        stack.enter_context(
            mock.patch.object(backend_selector.torch, "full", return_value=code)
        )
        info = stack.enter_context(mock.patch.object(backend_selector.logger, "info"))
        stack.enter_context(
            mock.patch.dict(
                sys.modules,
                {
                    "rtp_llm.models_py.distributed.collective_torch": collective_module,
                },
            )
        )
        choice = backend_selector.run_backend_selection(model, inputs)

    assert choice == "XQADecodeImpl"
    collective_module.broadcast.assert_called_once()
    info.assert_called_once_with(
        "dynamic_decode_plan_received bs=%d registry_idx=%d backend=%s "
        "tp_rank=%d tp_size=%d dp_rank=%d",
        1,
        0,
        "XQADecodeImpl",
        1,
        2,
        0,
    )


def _root_selection_fixture():
    model, inputs, attn_configs, parallelism_config = _selection_fixture()
    return model, attn_configs, inputs.attention_inputs, parallelism_config


def _select_on_root_for_test(
    model, attn_configs, attn_inputs, parallelism_config, selector=None
):
    return backend_selector._select_on_root(
        model,
        attn_configs,
        attn_inputs,
        parallelism_config,
        [8],
        selector,
        1,
        8,
        0,
        1,
        "store",
    )


def test_no_latency_matrix_returns_no_plan_without_fixed_priority_construction():
    model, attn_configs, attn_inputs, parallelism_config = _root_selection_fixture()
    fixed_priority = mock.Mock()
    synchronize = mock.Mock()

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(backend_selector, "_eligible", return_value=[])
        )
        stack.enter_context(
            mock.patch(f"{_ATTN_FACTORY}.get_fmha_impl", fixed_priority)
        )
        stack.enter_context(
            mock.patch.object(backend_selector.torch.cuda, "synchronize", synchronize)
        )
        choice = _select_on_root_for_test(
            model, attn_configs, attn_inputs, parallelism_config
        )

    assert choice is None
    fixed_priority.assert_not_called()
    synchronize.assert_not_called()


def test_measured_baseline_uses_registry_priority_without_extra_construction():
    model, attn_configs, attn_inputs, parallelism_config = _root_selection_fixture()
    high_priority = _make_fake_impl("HighPriorityImpl")
    low_priority = _make_fake_impl("LowPriorityImpl")
    implementations = {
        high_priority.__name__: high_priority,
        low_priority.__name__: low_priority,
    }
    fixed_priority = mock.Mock()
    bench = mock.Mock(return_value=[10.0])

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(
                backend_selector,
                "_eligible",
                return_value=[low_priority.__name__, high_priority.__name__],
            )
        )
        stack.enter_context(
            mock.patch.object(
                backend_selector,
                "_decode_registry",
                return_value=[high_priority.__name__, low_priority.__name__],
            )
        )
        stack.enter_context(
            mock.patch.object(
                backend_selector,
                "_impl_by_name",
                side_effect=implementations.get,
            )
        )
        stack.enter_context(
            mock.patch.object(
                backend_selector.backend_bench,
                "bench_backend_grid",
                bench,
            )
        )
        stack.enter_context(
            mock.patch(f"{_ATTN_FACTORY}.get_fmha_impl", fixed_priority)
        )
        stack.enter_context(
            mock.patch.object(backend_selector.torch.cuda, "synchronize")
        )
        choice = _select_on_root_for_test(
            model, attn_configs, attn_inputs, parallelism_config
        )

    assert choice == high_priority.__name__
    assert bench.call_count == 2
    assert all(
        call.kwargs["attention_layer_count"] == 32 for call in bench.call_args_list
    )
    assert high_priority.construction_count == 0
    assert low_priority.construction_count == 0
    fixed_priority.assert_not_called()


def test_production_selector_receives_validated_values_without_clamping():
    model, attn_configs, attn_inputs, parallelism_config = _root_selection_fixture()
    candidate = _make_fake_impl("ConfiguredImpl")
    stable = mock.Mock(return_value=candidate.__name__)
    with (
        mock.patch.dict(
            os.environ,
            {
                "DYN_DECODE_THRESHOLD": "1.0",
                "DYN_DECODE_CLUSTER_MARGIN": "1.25",
            },
            clear=True,
        ),
        mock.patch.object(
            backend_selector, "_eligible", return_value=[candidate.__name__]
        ),
        mock.patch.object(
            backend_selector, "_decode_registry", return_value=[candidate.__name__]
        ),
        mock.patch.object(backend_selector, "_impl_by_name", return_value=candidate),
        mock.patch.object(
            backend_selector.backend_bench,
            "bench_backend_grid",
            return_value=[4.0],
        ),
        mock.patch.object(backend_selector, "select_stable", stable),
        mock.patch.object(backend_selector.torch.cuda, "synchronize"),
    ):
        choice = _select_on_root_for_test(
            model, attn_configs, attn_inputs, parallelism_config
        )

    assert choice == candidate.__name__
    stable.assert_called_once_with(
        {candidate.__name__: [4.0]},
        [candidate.__name__],
        candidate.__name__,
        1.0,
        1.25,
    )


def test_selector_config_defaults_and_legal_overrides():
    with mock.patch.dict(os.environ, {}, clear=True):
        assert backend_selector._read_selector_config() == (0.05, 0.05)

    legal = (
        ("DYN_DECODE_THRESHOLD", "0", (0.0, 0.05)),
        ("DYN_DECODE_THRESHOLD", "0.05", (0.05, 0.05)),
        ("DYN_DECODE_THRESHOLD", "1.0", (1.0, 0.05)),
        ("DYN_DECODE_CLUSTER_MARGIN", "0", (0.05, 0.0)),
        ("DYN_DECODE_CLUSTER_MARGIN", "0.05", (0.05, 0.05)),
        ("DYN_DECODE_CLUSTER_MARGIN", "1.25", (0.05, 1.25)),
    )
    for variable, value, expected in legal:
        with mock.patch.dict(os.environ, {variable: value}, clear=True):
            assert backend_selector._read_selector_config() == expected


def test_selector_config_rejects_each_invalid_explicit_value():
    for variable in ("DYN_DECODE_THRESHOLD", "DYN_DECODE_CLUSTER_MARGIN"):
        for value in ("not-a-number", "nan", "inf", "-inf", "-0.01"):
            with mock.patch.dict(os.environ, {variable: value}, clear=True):
                with unittest.TestCase().assertRaisesRegex(
                    backend_selector._SelectorConfigError,
                    variable,
                ) as raised:
                    backend_selector._read_selector_config()
            assert raised.exception.variable == variable
            assert raised.exception.raw_value == value


def test_invalid_production_selector_config_fails_before_probe():
    model, inputs, _, _ = _selection_fixture(dp_rank=3)
    fatal = mock.Mock()
    eligible = mock.Mock()
    bench = mock.Mock()
    synchronize = mock.Mock()
    fixed_priority = mock.Mock()

    with (
        mock.patch.dict(
            os.environ, {"DYN_DECODE_THRESHOLD": "not-a-number"}, clear=True
        ),
        mock.patch.object(backend_selector, "_decode_registry", return_value=[]),
        mock.patch.object(backend_selector, "kv_grid", return_value=[8]),
        mock.patch.object(backend_selector.torch, "full", return_value=_FakeCode()),
        mock.patch.object(backend_selector, "_eligible", eligible),
        mock.patch.object(backend_selector.backend_bench, "bench_backend_grid", bench),
        mock.patch.object(backend_selector.torch.cuda, "synchronize", synchronize),
        mock.patch.object(backend_selector, "_terminate_config_worker", fatal),
        mock.patch(f"{_ATTN_FACTORY}.get_fmha_impl", fixed_priority),
    ):
        with unittest.TestCase().assertRaisesRegex(
            backend_selector.DynamicDecodeFatalError,
            "fatal configuration termination returned unexpectedly",
        ):
            backend_selector.run_backend_selection(model, inputs)

    fatal.assert_called_once()
    assert fatal.call_args.args[0] == 1
    error = fatal.call_args.args[1]
    assert error.variable == "DYN_DECODE_THRESHOLD"
    eligible.assert_not_called()
    bench.assert_not_called()
    synchronize.assert_not_called()
    fixed_priority.assert_not_called()


def test_config_fatal_log_has_config_and_rank_context_once():
    error = backend_selector._SelectorConfigError(
        "DYN_DECODE_CLUSTER_MARGIN", "nan", "value is not finite"
    )
    parallelism_config = types.SimpleNamespace(tp_rank=0, tp_size=2, dp_rank=4)
    with (
        mock.patch.object(backend_selector.logger, "critical") as critical,
        mock.patch.object(
            backend_selector.os, "_exit", side_effect=SystemExit(70)
        ) as exit_process,
    ):
        with unittest.TestCase().assertRaises(SystemExit):
            backend_selector._terminate_config_worker(8, error, parallelism_config)

    critical.assert_called_once()
    message = critical.call_args.args[0]
    assert message.startswith("dynamic_decode_config_invalid")
    assert critical.call_args.args[1:8] == (
        "DYN_DECODE_CLUSTER_MARGIN",
        "nan",
        8,
        0,
        2,
        4,
        "value is not finite",
    )
    exit_process.assert_called_once_with(70)


def test_plan_application_fatal_log_has_stage_rank_and_traceback_once():
    error = RuntimeError("constructor failed")
    parallelism_config = types.SimpleNamespace(tp_rank=1, tp_size=2, dp_rank=3)
    with (
        mock.patch.object(backend_selector.logger, "critical") as critical,
        mock.patch.object(
            backend_selector.os, "_exit", side_effect=SystemExit(70)
        ) as exit_process,
    ):
        with unittest.TestCase().assertRaises(SystemExit):
            backend_selector._terminate_plan_application_worker(
                4,
                "WinnerImpl",
                7,
                parallelism_config,
                "constructor",
                error,
            )

    critical.assert_called_once()
    assert critical.call_args.args[0].startswith("dynamic_decode_plan_apply_failed")
    assert critical.call_args.args[1:9] == (
        4,
        "WinnerImpl",
        7,
        1,
        2,
        3,
        "constructor",
        error,
    )
    assert critical.call_args.kwargs["exc_info"] == (
        RuntimeError,
        error,
        error.__traceback__,
    )
    exit_process.assert_called_once_with(70)


def test_custom_selector_does_not_read_production_environment():
    model, attn_configs, attn_inputs, parallelism_config = _root_selection_fixture()
    candidate = _make_fake_impl("CustomSelectorImpl")
    custom_selector = mock.Mock(return_value=candidate.__name__)
    with (
        mock.patch.dict(os.environ, {"DYN_DECODE_THRESHOLD": "invalid"}, clear=True),
        mock.patch.object(
            backend_selector, "_eligible", return_value=[candidate.__name__]
        ),
        mock.patch.object(backend_selector, "_impl_by_name", return_value=candidate),
        mock.patch.object(
            backend_selector.backend_bench,
            "bench_backend_grid",
            return_value=[3.0],
        ),
        mock.patch.object(backend_selector.torch.cuda, "synchronize"),
    ):
        choice = _select_on_root_for_test(
            model,
            attn_configs,
            attn_inputs,
            parallelism_config,
            selector=custom_selector,
        )

    assert choice == candidate.__name__
    custom_selector.assert_called_once_with({candidate.__name__: [3.0]})


def _plan_application_fixture(tp_size=1, tp_rank=0):
    model, inputs, _, _ = _selection_fixture(
        tp_size=tp_size, tp_rank=tp_rank, dp_rank=2
    )
    return model, inputs.attention_inputs


def _assert_plan_application_failure(
    expected_stage, *, impl=None, disabled=False, tp_size=1, tp_rank=0
):
    name = "PlannedImpl"
    model, inputs = _plan_application_fixture(tp_size, tp_rank)
    fatal = mock.Mock()
    with (
        mock.patch.object(backend_selector, "_decode_registry", return_value=[name]),
        mock.patch.object(backend_selector, "_impl_by_name", return_value=impl),
        mock.patch(f"{_ATTN_FACTORY}._is_fmha_impl_disabled", return_value=disabled),
        mock.patch.object(
            backend_selector, "_terminate_plan_application_worker", fatal
        ),
    ):
        with unittest.TestCase().assertRaisesRegex(
            backend_selector.DynamicDecodeFatalError,
            "fatal plan application termination returned unexpectedly",
        ):
            backend_selector.instantiate_decode_impl(model, inputs, name, True)

    fatal.assert_called_once()
    args = fatal.call_args.args
    assert args[0:3] == (1, name, 0)
    assert args[3] is model.parallelism_config
    assert args[4] == expected_stage
    assert isinstance(args[5], BaseException)


def test_each_winner_application_failure_is_fatal():
    _assert_plan_application_failure("class-missing")
    _assert_plan_application_failure(
        "disabled", impl=_make_fake_impl("PlannedImpl"), disabled=True
    )
    _assert_plan_application_failure(
        "support", impl=_make_fake_impl("PlannedImpl", supports=False)
    )
    _assert_plan_application_failure(
        "parallelism",
        impl=_make_fake_impl("PlannedImpl", parallelism_supports=False),
    )
    _assert_plan_application_failure(
        "constructor",
        impl=_make_fake_impl(
            "PlannedImpl", construction_error=RuntimeError("constructor failed")
        ),
    )
    _assert_plan_application_failure(
        "cuda-graph-support",
        impl=_make_fake_impl("PlannedImpl", graph_supports=False),
    )


def test_winner_application_failure_semantics_cover_single_and_non_root_tp():
    _assert_plan_application_failure("class-missing", tp_size=2, tp_rank=1)


def test_invalid_nonnegative_broadcast_index_is_fatal_but_minus_one_is_miss():
    model, inputs, _, _ = _selection_fixture(tp_size=2, tp_rank=1)
    collective_module = types.ModuleType("collective_torch")
    collective_module.Group = types.SimpleNamespace(TP=object())

    for broadcast_index, should_fail in ((4, True), (-1, False)):
        code = _FakeCode()

        def broadcast(selected_code, _src, group, value=broadcast_index):
            assert group is collective_module.Group.TP
            selected_code[0] = value

        collective_module.broadcast = mock.Mock(side_effect=broadcast)
        fatal = mock.Mock()
        with (
            mock.patch.object(
                backend_selector, "_decode_registry", return_value=["OnlyImpl"]
            ),
            mock.patch.object(backend_selector, "kv_grid", return_value=[8]),
            mock.patch.object(backend_selector.torch, "full", return_value=code),
            mock.patch.object(
                backend_selector, "_terminate_plan_application_worker", fatal
            ),
            mock.patch.dict(
                sys.modules,
                {
                    "rtp_llm.models_py.distributed.collective_torch": collective_module,
                },
            ),
        ):
            if should_fail:
                with unittest.TestCase().assertRaises(
                    backend_selector.DynamicDecodeFatalError
                ):
                    backend_selector.run_backend_selection(model, inputs)
            else:
                assert backend_selector.run_backend_selection(model, inputs) is None

        if should_fail:
            fatal.assert_called_once()
            assert fatal.call_args.args[0:3] == (
                1,
                "<invalid-registry-index>",
                4,
            )
            assert fatal.call_args.args[4] == "registry-index"
        else:
            fatal.assert_not_called()


def test_instantiate_decode_impl_instantiates_when_enabled():
    model = types.SimpleNamespace(
        fmha_config=_fmha_config(),
        config=types.SimpleNamespace(
            getAttentionConfigs=lambda tp: None, headwise_config=None
        ),
        parallelism_config=types.SimpleNamespace(
            tp_size=1,
            tp_rank=0,
            dp_rank=0,
            get_attn_tp_size=lambda: 1,
        ),
    )
    with _patch_impls(_NAMES):
        inst = backend_selector.instantiate_decode_impl(
            model,
            types.SimpleNamespace(input_lengths=_FakeInputLengths()),
            "PyFlashinferDecodeImpl",
            True,
        )
    assert inst is not None


# Bind the module-level test_* functions onto a TestCase so bazel's unittest
# runner (no pytest available) discovers and runs them.
class BackendSelectorTest(unittest.TestCase):
    pass


for _name, _fn in list(globals().items()):
    if _name.startswith("test_") and callable(_fn):
        setattr(BackendSelectorTest, _name, staticmethod(_fn))
del _name, _fn  # don't leak a class/func ref that unittest would re-collect


if __name__ == "__main__":
    unittest.main()

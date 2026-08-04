"""backend_bench tests: clone isolation, timing boundaries, and workspace lifetime.

Some tests require CUDA; on GPU-less machines only the CPU logic runs.
"""

import gc
import statistics
import sys
import time
import unittest
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

from rtp_llm.models_py.modules.factory.attention.cuda_impl.benchmark_workspace import (
    benchmark_workspace_scope,
    in_benchmark_workspace_scope,
)
from rtp_llm.models_py.modules.factory.attention.dispatch.backend_bench import (
    _bench_graph,
    _clone_attn_inputs,
    _fill_synthetic_kv_block,
    _get_bench_pool,
    _model_step_times,
    _set_uniform_seq_len,
    bench_backend_grid,
)

requires_cuda = unittest.skipIf(not HAS_CUDA, "CUDA not available")


# _clone_attn_inputs tests


class _FakeAttnInputs:
    """Mimic the key field layout of PyAttentionInputs."""

    def __init__(self):
        if HAS_CUDA:
            self.input_lengths = torch.tensor([1, 2, 3], device="cuda")
            self.sequence_lengths = torch.tensor([10, 20, 30], device="cuda")
            self.kv_cache_kernel_block_id = torch.zeros(
                3, 4, dtype=torch.int32, device="cuda"
            )
            self.kv_cache_kernel_block_id_by_group = [
                torch.zeros(3, 4, dtype=torch.int32, device="cuda"),
                torch.ones(3, 4, dtype=torch.int32, device="cuda"),
            ]
        else:
            self.input_lengths = torch.tensor([1, 2, 3])
            self.sequence_lengths = torch.tensor([10, 20, 30])
            self.kv_cache_kernel_block_id = torch.zeros(3, 4, dtype=torch.int32)
            self.kv_cache_kernel_block_id_by_group = [
                torch.zeros(3, 4, dtype=torch.int32),
                torch.ones(3, 4, dtype=torch.int32),
            ]
        self.is_cuda_graph = True
        self.total_tokens = 6

    def __copy__(self):
        new = _FakeAttnInputs.__new__(_FakeAttnInputs)
        new.__dict__.update(self.__dict__)
        return new


def test_clone_attn_inputs_tensor_isolation():
    """After clone, modifying the clone's Tensor does not affect the original object."""
    orig = _FakeAttnInputs()
    cloned = _clone_attn_inputs(orig)

    # modify the clone
    cloned.sequence_lengths.fill_(999)
    cloned.kv_cache_kernel_block_id_by_group[0].fill_(777)

    # original unchanged
    assert orig.sequence_lengths.tolist() == [10, 20, 30]
    assert orig.kv_cache_kernel_block_id_by_group[0].sum().item() == 0


def test_clone_attn_inputs_tensors_are_different_objects():
    """The clone's Tensors are new objects (not the same storage)."""
    orig = _FakeAttnInputs()
    cloned = _clone_attn_inputs(orig)
    assert cloned.input_lengths.data_ptr() != orig.input_lengths.data_ptr()
    assert (
        cloned.kv_cache_kernel_block_id_by_group[1].data_ptr()
        != orig.kv_cache_kernel_block_id_by_group[1].data_ptr()
    )


# CUDA Graph timing-boundary tests


def _mocked_graph_actions(prepare_fn):
    import rtp_llm.models_py.modules.factory.attention.dispatch.backend_bench as bb

    actions = []

    class FakeStream:
        def wait_stream(self, other):
            actions.append("wait_stream")

    class FakeGraph:
        def replay(self):
            actions.append("replay")

    event_count = 0

    class FakeEvent:
        def __init__(self, enable_timing):
            nonlocal event_count
            self.name = "start" if event_count == 0 else "end"
            event_count += 1

        def record(self):
            actions.append(f"{self.name}.record")

        def synchronize(self):
            actions.append(f"{self.name}.synchronize")

        def elapsed_time(self, other):
            return 0.001

    with (
        patch.object(
            bb,
            "_make_l2_filler",
            return_value=(lambda: actions.append("fill"), ()),
        ),
        patch.object(bb, "_get_bench_pool", return_value=1),
        patch.object(bb.torch.cuda, "Event", FakeEvent),
        patch.object(bb.torch.cuda, "Stream", FakeStream),
        patch.object(bb.torch.cuda, "CUDAGraph", FakeGraph),
        patch.object(bb.torch.cuda, "current_stream", return_value=FakeStream()),
        patch.object(bb.torch.cuda, "stream", side_effect=lambda stream: nullcontext()),
        patch.object(
            bb.torch.cuda, "graph", side_effect=lambda graph, pool: nullcontext()
        ),
        patch.object(
            bb.torch.cuda,
            "synchronize",
            side_effect=lambda: actions.append("cuda.synchronize"),
        ),
        patch.object(bb.torch.cuda, "empty_cache"),
    ):
        _bench_graph(
            lambda: actions.append("fn"),
            warmup=0,
            iters=1,
            l2_fill_mode="none",
            prepare_fn=prepare_fn(actions) if prepare_fn else None,
        )
    return actions


def test_bench_graph_timing_order_includes_prepare():
    actions = _mocked_graph_actions(lambda log: lambda: log.append("prepare"))
    measured = actions.index("fill")
    assert actions[measured : measured + 6] == [
        "fill",
        "start.record",
        "start.synchronize",
        "prepare",
        "replay",
        "end.record",
    ]


def test_bench_graph_timing_order_without_prepare_uses_same_boundary():
    actions = _mocked_graph_actions(None)
    measured = actions.index("fill")
    assert actions[measured : measured + 5] == [
        "fill",
        "start.record",
        "start.synchronize",
        "replay",
        "end.record",
    ]


@requires_cuda
def test_bench_graph_measures_host_prepare_delay_with_and_without_l2_fill():
    buf = torch.zeros(1, device="cuda")

    def fn():
        buf.add_(1)

    prepare_delay_s = 0.02
    for fill_mode in ("none", "store"):
        times = _bench_graph(
            fn,
            warmup=1,
            iters=3,
            l2_fill_mode=fill_mode,
            prepare_fn=lambda: time.sleep(prepare_delay_s),
        )
        assert statistics.median(times) >= prepare_delay_s * 1_000_000 * 0.75


def test_model_step_times_amortize_prepare_once_across_attention_layers():
    fast_kernel = _model_step_times([10.0], [5.0], attention_layer_count=32)
    fast_prepare = _model_step_times([8.0], [7.0], attention_layer_count=32)

    # A one-layer prepare+replay measurement prefers fast_prepare (8 < 10), but
    # the 32-layer model-step score correctly prefers the faster replay kernel.
    assert fast_kernel == [165.0]
    assert fast_prepare == [225.0]
    assert fast_kernel[0] < fast_prepare[0]


# Benchmark workspace and failure-boundary tests


class _Workspace:
    pass


class _FakeBenchConfig:
    head_num = 8
    kv_head_num = 2
    size_per_head = 64
    dtype = torch.bfloat16


class _FakeBenchInputs:
    input_lengths = torch.ones(1)
    is_cuda_graph = False


class _FakeKVCache:
    kv_cache_base = None


def _assert_production_wrapper_workspace_ownership(
    module, pool_name, workspace_name, construct
):
    production_workspace = _Workspace()
    production_pool = [production_workspace]

    with (
        patch.object(module, pool_name, production_pool),
        patch.object(
            module.torch, "zeros", side_effect=lambda *args, **kwargs: _Workspace()
        ),
    ):
        production_op = construct()
        assert getattr(production_op, workspace_name) is production_workspace
        assert production_pool == []
        del production_op
        gc.collect()
        assert production_pool == [production_workspace]
        assert production_pool[0] is production_workspace

        with benchmark_workspace_scope():
            benchmark_op = construct()
        benchmark_workspace = getattr(benchmark_op, workspace_name)
        assert benchmark_workspace is not production_workspace
        assert production_pool == [production_workspace]
        benchmark_workspace_ref = weakref.ref(benchmark_workspace)
        del benchmark_workspace, benchmark_op
        gc.collect()

        assert benchmark_workspace_ref() is None
        assert production_pool == [production_workspace]
        assert production_pool[0] is production_workspace


def test_xqa_wrapper_preserves_production_workspace_pool_during_benchmark():
    from rtp_llm.models_py.modules.factory.attention.cuda_impl import xqa

    config = SimpleNamespace()
    inputs = SimpleNamespace(
        cu_seqlens_device=object(),
        is_prefill=False,
    )
    with (
        patch.object(xqa.torch.cuda, "get_device_capability", return_value=(9, 0)),
        patch.object(xqa, "get_num_device_sms", return_value=1),
        patch.object(xqa, "_load_xqa_fn", return_value=(object(), False)),
    ):
        _assert_production_wrapper_workspace_ownership(
            xqa,
            "_g_xqa_workspace_pool",
            "workspace_buffer",
            lambda: xqa.XQAWrapper(config, inputs),
        )


def test_py_flashinfer_wrapper_preserves_production_workspace_pool_during_benchmark():
    from rtp_llm.models_py.modules.factory.attention.cuda_impl import py_flashinfer_mha

    config = SimpleNamespace(
        head_num=8,
        kv_head_num=2,
        size_per_head=64,
        kernel_tokens_per_block=16,
        kv_cache_dtype=None,
    )
    inputs = SimpleNamespace(is_cuda_graph=True)
    fake_ops = SimpleNamespace(FlashInferMlaAttnParams=lambda: object())
    with (
        patch.object(
            py_flashinfer_mha,
            "BatchDecodeWithPagedKVCacheWrapper",
            new=lambda *args, **kwargs: object(),
        ),
        patch.object(py_flashinfer_mha, "rtp_llm_ops", fake_ops),
    ):
        _assert_production_wrapper_workspace_ownership(
            py_flashinfer_mha,
            "_g_py_flashinfer_workspace_pool",
            "g_workspace_buffer",
            lambda: py_flashinfer_mha.PyFlashinferDecodeAttnOp(config, inputs),
        )


def test_trt_wrapper_preserves_production_workspace_pool_during_benchmark():
    from rtp_llm.models_py.modules.factory.attention.cuda_impl import trtllm_gen

    config = SimpleNamespace(
        size_per_head=128,
        head_num=8,
        kernel_tokens_per_block=16,
        kv_head_num=2,
    )
    _assert_production_wrapper_workspace_ownership(
        trtllm_gen,
        "_g_trt_workspace_pool",
        "workspace_buffer",
        lambda: trtllm_gen.FlashInferTRTLLMDecodeOp(config),
    )


def test_bench_grid_uses_temporary_workspace_and_releases_before_empty_cache():
    import rtp_llm.models_py.modules.factory.attention.dispatch.backend_bench as bb

    class Workspace:
        pass

    production_workspace = Workspace()
    production_pool = [production_workspace]
    benchmark_workspace_refs = []

    class FakeImpl:
        def __init__(self, ac, ai, pc):
            assert in_benchmark_workspace_scope()
            if in_benchmark_workspace_scope():
                self.workspace = Workspace()
                self.workspace_from_pool = False
            else:
                self.workspace = production_pool.pop()
                self.workspace_from_pool = True
            benchmark_workspace_refs.append(weakref.ref(self.workspace))

        def forward(self, qkv, kv_cache, layer_idx):
            pass

        def support_cuda_graph(self):
            return True

        def __del__(self):
            if self.workspace_from_pool:
                production_pool.append(self.workspace)

    def assert_workspace_released():
        assert benchmark_workspace_refs
        assert all(ref() is None for ref in benchmark_workspace_refs)

    with (
        patch.object(bb.torch, "randn", return_value=object()),
        patch.object(bb, "_clone_attn_inputs", return_value=_FakeBenchInputs()),
        patch.object(bb, "_set_uniform_seq_len"),
        patch.object(bb, "_bench_graph", return_value=[3.0, 1.0, 2.0]),
        patch.object(bb.torch.cuda, "synchronize"),
        patch.object(
            bb.torch.cuda, "empty_cache", side_effect=assert_workspace_released
        ),
    ):
        result = bench_backend_grid(
            FakeImpl,
            _FakeBenchConfig(),
            _FakeBenchInputs(),
            _FakeKVCache(),
            None,
            [128],
        )

    assert result == [2.0]
    assert production_pool == [production_workspace]
    assert production_pool[0] is production_workspace
    assert all(ref() is None for ref in benchmark_workspace_refs)
    assert not in_benchmark_workspace_scope()


@requires_cuda
def test_synthetic_kv_fill_supports_fp8_cache_storage():
    kv_base = torch.zeros((2, 64), dtype=torch.float8_e4m3fn, device="cuda")
    _fill_synthetic_kv_block(kv_base)
    assert torch.isfinite(kv_base[0].float()).all().item()


def test_bench_grid_construction_exception_propagates_without_cuda_cleanup():
    import rtp_llm.models_py.modules.factory.attention.dispatch.backend_bench as bb

    class BrokenImpl:
        def __init__(self, *args):
            assert in_benchmark_workspace_scope()
            raise RuntimeError("cannot instantiate")

    with (
        patch.object(bb.torch, "randn", return_value=object()),
        patch.object(bb, "_clone_attn_inputs", return_value=_FakeBenchInputs()),
        patch.object(bb, "_set_uniform_seq_len"),
        patch.object(bb.torch.cuda, "synchronize") as synchronize,
        patch.object(bb.torch.cuda, "empty_cache") as empty_cache,
    ):
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "cannot instantiate"):
            bench_backend_grid(
                BrokenImpl,
                _FakeBenchConfig(),
                _FakeBenchInputs(),
                _FakeKVCache(),
                None,
                [128],
            )
        synchronize.assert_not_called()
        empty_cache.assert_not_called()
    assert not in_benchmark_workspace_scope()


def test_bench_grid_graph_exception_skips_healthy_cleanup():
    import rtp_llm.models_py.modules.factory.attention.dispatch.backend_bench as bb

    class FakeImpl:
        def __init__(self, *args):
            assert in_benchmark_workspace_scope()

        def forward(self, qkv, kv_cache, layer_idx):
            pass

        def support_cuda_graph(self):
            return True

    with (
        patch.object(bb.torch, "randn", return_value=object()),
        patch.object(bb, "_clone_attn_inputs", return_value=_FakeBenchInputs()),
        patch.object(bb, "_set_uniform_seq_len"),
        patch.object(bb, "_bench_graph", side_effect=RuntimeError("capture failed")),
        patch.object(bb.torch.cuda, "synchronize") as synchronize,
        patch.object(bb.torch.cuda, "empty_cache") as empty_cache,
    ):
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "capture failed"):
            bench_backend_grid(
                FakeImpl,
                _FakeBenchConfig(),
                _FakeBenchInputs(),
                _FakeKVCache(),
                None,
                [128],
            )
        assert synchronize.call_count == 2
        empty_cache.assert_not_called()


def test_bench_backend_grid_unsupported_graph_is_normal_unavailable():
    import rtp_llm.models_py.modules.factory.attention.dispatch.backend_bench as bb

    class UnsupportedImpl:
        def __init__(self, *args):
            assert in_benchmark_workspace_scope()

        def support_cuda_graph(self):
            return False

    with (
        patch.object(bb.torch, "randn", return_value=object()),
        patch.object(bb, "_clone_attn_inputs", return_value=_FakeBenchInputs()),
        patch.object(bb, "_set_uniform_seq_len"),
        patch.object(bb.torch.cuda, "synchronize") as synchronize,
        patch.object(bb.torch.cuda, "empty_cache") as empty_cache,
    ):
        result = bench_backend_grid(
            UnsupportedImpl,
            _FakeBenchConfig(),
            _FakeBenchInputs(),
            _FakeKVCache(),
            None,
            [128],
        )
        assert result is None
        synchronize.assert_called_once()
        empty_cache.assert_called_once()


@requires_cuda
def test_uniform_seq_len_updates_replay_mirror_in_place():
    inputs = SimpleNamespace(
        sequence_lengths=torch.zeros(3, dtype=torch.int32).pin_memory(),
        sequence_lengths_plus_1_device=torch.zeros(3, dtype=torch.int32, device="cuda"),
    )
    mirror_address = inputs.sequence_lengths_plus_1_device.data_ptr()

    for kv in (17, 257):
        _set_uniform_seq_len(inputs, bs=3, kv=kv)
        assert inputs.sequence_lengths.tolist() == [kv, kv, kv]
        assert inputs.sequence_lengths.is_pinned()
        assert inputs.sequence_lengths_plus_1_device.tolist() == [kv + 1] * 3
        assert inputs.sequence_lengths_plus_1_device.data_ptr() == mirror_address


@requires_cuda
def test_uniform_seq_len_validates_both_length_tensors():
    def valid_inputs():
        return SimpleNamespace(
            sequence_lengths=torch.zeros(2, dtype=torch.int32).pin_memory(),
            sequence_lengths_plus_1_device=torch.zeros(
                2, dtype=torch.int32, device="cuda"
            ),
        )

    invalid_values = {
        "sequence_lengths": (
            None,
            torch.zeros(2, dtype=torch.float32).pin_memory(),
            torch.zeros(2, dtype=torch.int32, device="cuda"),
            torch.zeros(1, dtype=torch.int32).pin_memory(),
        ),
        "sequence_lengths_plus_1_device": (
            None,
            torch.zeros(2, dtype=torch.float32, device="cuda"),
            torch.zeros(2, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32, device="cuda"),
        ),
    }
    for field, values in invalid_values.items():
        for value in values:
            inputs = valid_inputs()
            setattr(inputs, field, value)
            with unittest.TestCase().assertRaisesRegex(
                ValueError, rf"field={field} bs=2 kv=31"
            ):
                _set_uniform_seq_len(inputs, bs=2, kv=31)


@requires_cuda
def test_bench_grid_prepare_observes_each_replay_mirror_value():
    import rtp_llm.models_py.modules.factory.attention.dispatch.backend_bench as bb

    observed = []

    class GraphAwareImpl:
        def __init__(self, _config, inputs, _parallelism_config):
            self.inputs = inputs

        def support_cuda_graph(self):
            return True

        def forward(self, _qkv, _kv_cache, _layer_idx):
            pass

        def prepare_cuda_graph(self, inputs):
            assert inputs is self.inputs
            observed.append(inputs.sequence_lengths_plus_1_device.tolist())

    class FakeConfig:
        head_num = 8
        kv_head_num = 2
        size_per_head = 64
        dtype = torch.bfloat16

    class FakeInputs:
        def __init__(self):
            self.input_lengths = torch.ones(2, dtype=torch.int32, device="cuda")
            self.sequence_lengths = torch.zeros(2, dtype=torch.int32).pin_memory()
            self.sequence_lengths_plus_1_device = torch.zeros(
                2, dtype=torch.int32, device="cuda"
            )
            self.is_cuda_graph = False

        def __copy__(self):
            new = FakeInputs.__new__(FakeInputs)
            new.__dict__.update(self.__dict__)
            return new

    def run_graph(
        _fn,
        _warmup,
        _iters,
        _l2_fill_mode,
        prepare_fn=None,
        attention_layer_count=1,
    ):
        assert attention_layer_count == 1
        prepare_fn()
        return [1.0]

    with (
        patch.object(bb.torch, "randn", return_value=object()),
        patch.object(bb, "_bench_graph", side_effect=run_graph),
        patch.object(bb.torch.cuda, "synchronize"),
        patch.object(bb.torch.cuda, "empty_cache"),
    ):
        result = bench_backend_grid(
            GraphAwareImpl,
            FakeConfig(),
            FakeInputs(),
            SimpleNamespace(kv_cache_base=None),
            None,
            [16, 32],
            warmup=0,
            iters=1,
        )

    assert result == [1.0, 1.0]
    assert observed == [[17, 17], [33, 33]]


# _get_bench_pool tests


@requires_cuda
def test_get_bench_pool_is_stable():
    """Multiple calls within the same process return the same pool handle."""
    import rtp_llm.models_py.modules.factory.attention.dispatch.backend_bench as bb

    # reset global state
    bb._bench_graph_pool_id = None
    p1 = _get_bench_pool()
    p2 = _get_bench_pool()
    assert p1 == p2


# Bind the module-level test_* functions (incl. @requires_cuda skips) onto a
# TestCase so bazel's unittest runner (no pytest available) discovers them.
class BackendBenchTest(unittest.TestCase):
    pass


for _name, _fn in list(globals().items()):
    if _name.startswith("test_") and callable(_fn):
        setattr(BackendBenchTest, _name, staticmethod(_fn))
del _name, _fn  # don't leak a class/func ref that unittest would re-collect


if __name__ == "__main__":
    from rtp_llm.models_py.modules.factory.attention.dispatch.tests import (
        test_backend_selector,
        test_selector,
    )

    loader = unittest.TestLoader()
    suite = unittest.TestSuite(
        [
            loader.loadTestsFromModule(sys.modules[__name__]),
            loader.loadTestsFromModule(test_backend_selector),
            loader.loadTestsFromModule(test_selector),
        ]
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())

"""backend_bench unit tests: clone isolation / cleanup / graph pool safety.

Some tests require CUDA (marked with pytest.mark.skipif); on GPU-less machines
only the CPU logic runs.
"""

import copy
import unittest
from unittest.mock import MagicMock, patch

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

from rtp_llm.models_py.modules.factory.attention.dispatch.backend_bench import (
    _cleanup_flashinfer_impl,
    _clone_attn_inputs,
    _get_bench_pool,
)

requires_cuda = unittest.skipIf(not HAS_CUDA, "CUDA not available")


# ─── _clone_attn_inputs tests ────────────────────────────────────────────────


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


def test_clone_attn_inputs_scalar_copied():
    """Scalar fields are copied correctly."""
    orig = _FakeAttnInputs()
    cloned = _clone_attn_inputs(orig)
    assert cloned.is_cuda_graph == orig.is_cuda_graph
    assert cloned.total_tokens == orig.total_tokens


def test_clone_attn_inputs_tensors_are_different_objects():
    """The clone's Tensors are new objects (not the same storage)."""
    orig = _FakeAttnInputs()
    cloned = _clone_attn_inputs(orig)
    assert cloned.input_lengths.data_ptr() != orig.input_lengths.data_ptr()
    assert (
        cloned.kv_cache_kernel_block_id_by_group[1].data_ptr()
        != orig.kv_cache_kernel_block_id_by_group[1].data_ptr()
    )


# ─── _cleanup_flashinfer_impl tests ──────────────────────────────────────────


def test_cleanup_flashinfer_impl_resets_wrapper_flags():
    """For a mocked FlashInfer impl, the wrapper flags are reset after cleanup."""
    wrapper = MagicMock()
    wrapper._use_cuda_graph = True
    wrapper._fixed_batch_size = 32
    wrapper.end_forward = MagicMock()

    fmha_impl = MagicMock()
    fmha_impl.decode_wrapper = wrapper

    impl = MagicMock()
    impl.fmha_impl = fmha_impl

    _cleanup_flashinfer_impl(impl)

    assert wrapper._use_cuda_graph is False
    assert wrapper._fixed_batch_size == 0
    wrapper.end_forward.assert_called_once()


def test_cleanup_flashinfer_impl_noop_for_non_flashinfer():
    """A non-FlashInfer backend (no fmha_impl attribute) does not error."""
    impl = MagicMock(spec=[])  # empty spec -> no fmha_impl attribute
    # should not raise
    _cleanup_flashinfer_impl(impl)


def test_cleanup_flashinfer_impl_no_end_forward():
    """Does not error when the wrapper has no end_forward."""
    wrapper = MagicMock(spec=["_use_cuda_graph", "_fixed_batch_size"])
    wrapper._use_cuda_graph = True
    wrapper._fixed_batch_size = 16

    fmha_impl = MagicMock()
    fmha_impl.decode_wrapper = wrapper

    impl = MagicMock()
    impl.fmha_impl = fmha_impl

    _cleanup_flashinfer_impl(impl)
    assert wrapper._use_cuda_graph is False
    assert wrapper._fixed_batch_size == 0


# ─── _get_bench_pool tests ────────────────────────────────────────────────────


@requires_cuda
def test_get_bench_pool_is_stable():
    """Multiple calls within the same process return the same pool handle."""
    import rtp_llm.models_py.modules.factory.attention.dispatch.backend_bench as bb

    # reset global state
    bb._bench_graph_pool_id = None
    p1 = _get_bench_pool()
    p2 = _get_bench_pool()
    assert p1 == p2


# ─── bench_backend eager mode (requires CUDA) ────────────────────────────────────────


@requires_cuda
def test_bench_backend_eager_returns_median():
    """In eager mode, returns a reasonable latency for a trivial kernel."""
    from rtp_llm.models_py.modules.factory.attention.dispatch.backend_bench import (
        bench_backend,
    )

    class TrivialImpl:
        def __init__(self, ac, ai, *args):
            pass

        @classmethod
        def support(cls, ac, ai):
            return True

        @classmethod
        def support_parallelism_config(cls, pc):
            return True

        def support_cuda_graph(self):
            return True

        def forward(self, qkv, kv_cache, layer_idx):
            # minimal compute
            _ = qkv + 1

    class FakeAC:
        head_num = 8
        kv_head_num = 2
        size_per_head = 64
        dtype = torch.bfloat16

    class FakeAI:
        input_lengths = torch.ones(4, device="cuda")
        is_cuda_graph = True

        def __copy__(self):
            return copy.copy(self)

    result = bench_backend(
        TrivialImpl,
        FakeAC(),
        FakeAI(),
        None,
        None,
        warmup=2,
        iters=5,
        l2_fill_mode="none",
        use_graph=False,
    )
    assert result is not None
    assert result > 0  # positive latency


@requires_cuda
def test_bench_backend_graph_returns_median():
    """In graph mode, returns a reasonable latency for a trivial kernel."""
    from rtp_llm.models_py.modules.factory.attention.dispatch.backend_bench import (
        bench_backend,
    )

    class TrivialImpl:
        def __init__(self, ac, ai, *args):
            self.buf = torch.zeros(4, 64, device="cuda")

        @classmethod
        def support(cls, ac, ai):
            return True

        @classmethod
        def support_parallelism_config(cls, pc):
            return True

        def support_cuda_graph(self):
            return True

        def forward(self, qkv, kv_cache, layer_idx):
            self.buf.add_(1)

    class FakeAC:
        head_num = 8
        kv_head_num = 2
        size_per_head = 64
        dtype = torch.bfloat16

    class FakeAI:
        input_lengths = torch.ones(4, device="cuda")
        is_cuda_graph = True

        def __copy__(self):
            new = FakeAI.__new__(FakeAI)
            new.__dict__.update(self.__dict__)
            return new

    result = bench_backend(
        TrivialImpl,
        FakeAC(),
        FakeAI(),
        None,
        None,
        warmup=2,
        iters=5,
        l2_fill_mode="none",
        use_graph=True,
    )
    assert result is not None
    assert result > 0


@requires_cuda
def test_bench_backend_graph_does_not_pollute_shared_inputs():
    """In graph mode, the original attn_inputs tensors are not modified."""
    from rtp_llm.models_py.modules.factory.attention.dispatch.backend_bench import (
        bench_backend,
    )

    class MutatingImpl:
        """Deliberately modify attn_inputs tensors at init."""

        def __init__(self, ac, ai, *args):
            # attempt to contaminate ai's tensors
            ai.input_lengths.fill_(999)
            self.buf = torch.zeros(4, 64, device="cuda")

        def forward(self, qkv, kv_cache, layer_idx):
            self.buf.add_(1)

    class FakeAC:
        head_num = 8
        kv_head_num = 2
        size_per_head = 64
        dtype = torch.bfloat16

    orig_data = torch.ones(4, device="cuda")

    class FakeAI:
        input_lengths = orig_data.clone()
        is_cuda_graph = True

        def __copy__(self):
            new = FakeAI.__new__(FakeAI)
            new.__dict__.update(self.__dict__)
            return new

    ai = FakeAI()
    bench_backend(
        MutatingImpl,
        FakeAC(),
        ai,
        None,
        None,
        warmup=1,
        iters=3,
        l2_fill_mode="none",
        use_graph=True,
    )
    # graph mode: the original inputs are not contaminated
    assert ai.input_lengths.tolist() == [1.0, 1.0, 1.0, 1.0]


@requires_cuda
def test_bench_backend_exception_returns_none():
    """impl construction raises -> returns None (N/A)."""
    from rtp_llm.models_py.modules.factory.attention.dispatch.backend_bench import (
        bench_backend,
    )

    class BrokenImpl:
        def __init__(self, *args):
            raise RuntimeError("cannot instantiate")

    class FakeAC:
        head_num = 8
        kv_head_num = 2
        size_per_head = 64
        dtype = torch.bfloat16

    class FakeAI:
        input_lengths = torch.ones(4, device="cuda")
        is_cuda_graph = True

        def __copy__(self):
            new = FakeAI.__new__(FakeAI)
            new.__dict__.update(self.__dict__)
            return new

    result = bench_backend(
        BrokenImpl,
        FakeAC(),
        FakeAI(),
        None,
        None,
        warmup=1,
        iters=3,
        l2_fill_mode="none",
        use_graph=True,
    )
    assert result is None


# Bind the module-level test_* functions (incl. @requires_cuda skips) onto a
# TestCase so bazel's unittest runner (no pytest available) discovers them.
class BackendBenchTest(unittest.TestCase):
    pass


for _name, _fn in list(globals().items()):
    if _name.startswith("test_") and callable(_fn):
        setattr(BackendBenchTest, _name, staticmethod(_fn))
del _name, _fn  # don't leak a class/func ref that unittest would re-collect


if __name__ == "__main__":
    unittest.main()

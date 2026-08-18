import unittest
from unittest import mock

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import (
    CudaF16Linear,
)
from rtp_llm.models_py.modules.hybrid import indexer as indexer_module
from rtp_llm.models_py.modules.hybrid.indexer import Indexer

SOFTMAX_SCALE = 0.125
WEIGHTS_SCALE = 0.5


def check_cuda_version() -> bool:
    """Check if CUDA version is >= 12.9"""
    try:
        cuda_version = torch.version.cuda
        if cuda_version is None:
            return False
        major, minor = map(int, cuda_version.split(".")[:2])
        return (major, minor) >= (12, 9)
    except Exception:
        return False


def mm_call_succeeds_with_out_dtype(device: str) -> bool:
    """Whether a real ``torch.mm`` call with ``out_dtype`` succeeds on ``device``.

    Decided by calling rather than by reading the module's own probe, so a broken
    probe cannot hide behind the skip that is meant to guard it. Any failure
    counts as unusable, including a backend that accepts the keyword but has no
    kernel for it -- the fast-path cases must skip there, not error.
    """
    probe = torch.zeros((1, 1), dtype=torch.bfloat16, device=device)
    try:
        torch.mm(probe, probe, out_dtype=torch.float32)
    except Exception:
        return False
    return True


# The fast path needs both a device and a torch whose mm accepts out_dtype, so
# lanes without either skip instead of erroring.
FAST_PATH_OK = (
    torch.cuda.is_available()
    and check_cuda_version()
    and mm_call_succeeds_with_out_dtype("cuda")
)
FAST_PATH_SKIP_REASON = (
    "requires CUDA >= 12.9 and a torch whose mm accepts out_dtype"
)


class Fp32StagingRecorder(TorchDispatchMode):
    """Records the largest fp32 tensor any op produces inside the block.

    Bounds the fp32 staging copy by observing dispatched ops, so it is blind to
    which widening spelling the implementation happens to use.
    """

    def __init__(self) -> None:
        self.max_elements = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        candidates = result if isinstance(result, (tuple, list)) else (result,)
        for candidate in candidates:
            if (
                isinstance(candidate, torch.Tensor)
                and candidate.dtype is torch.float32
            ):
                self.max_elements = max(self.max_elements, candidate.numel())
        return result


def make_indexer(physical_weight: torch.Tensor) -> Indexer:
    indexer = Indexer.__new__(Indexer)
    torch.nn.Module.__init__(indexer)
    indexer.weights_proj = CudaF16Linear(physical_weight)
    indexer.softmax_scale = SOFTMAX_SCALE
    indexer.weights_scale = WEIGHTS_SCALE
    return indexer


def make_inputs(
    weight_dtype: torch.dtype,
    activation_dtype: torch.dtype,
    token_count: int = 3,
    hidden_size: int = 32,
    head_count: int = 4,
    device: str = "cpu",
):
    physical_weight = torch.randn(
        hidden_size, head_count, dtype=weight_dtype, device=device
    )
    hidden_states = torch.randn(
        token_count, hidden_size, dtype=activation_dtype, device=device
    )
    q_scale = torch.rand(
        token_count, head_count, 1, dtype=torch.float32, device=device
    )
    return physical_weight, hidden_states, q_scale


def fp32_reference_logits(
    hidden_states: torch.Tensor, physical_weight: torch.Tensor
) -> torch.Tensor:
    """Independent fp32 reference for the head-gate projection.

    Built from the raw checkpoint-layout weight with explicit per (token, head)
    fp32 dot products, which is the full-fp32 expansion the head gate used to
    do. It deliberately avoids the matmul entry points the implementation uses,
    so the expected value cannot be produced by the code under test.
    """
    x = hidden_states.float()
    weight = physical_weight.float()  # [hidden_size, head_count]
    return torch.stack(
        [
            torch.stack(
                [
                    torch.dot(row, weight[:, head])
                    for head in range(weight.shape[1])
                ]
            )
            for row in x
        ]
    )


def apply_gate_scale(logits: torch.Tensor, q_scale: torch.Tensor) -> torch.Tensor:
    return logits.unsqueeze(-1) * q_scale.float() * (SOFTMAX_SCALE * WEIGHTS_SCALE)


def fp32_reference_gate(
    hidden_states: torch.Tensor,
    physical_weight: torch.Tensor,
    q_scale: torch.Tensor,
) -> torch.Tensor:
    return apply_gate_scale(
        fp32_reference_logits(hidden_states, physical_weight), q_scale
    )


def max_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual - expected).abs().max().item()


class IndexerHeadGateFp32PathTest(unittest.TestCase):
    """The fp32 projection path, which is reachable without a device."""

    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_matches_independent_fp32_reference(self) -> None:
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                weight, hidden_states, q_scale = make_inputs(dtype, dtype)
                actual = make_indexer(weight)._get_logits_head_gate(
                    hidden_states, q_scale
                )
                expected = fp32_reference_gate(hidden_states, weight, q_scale)
                self.assertEqual(actual.dtype, torch.float32)
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_output_is_not_rounded_through_the_weight_dtype(self) -> None:
        weight, hidden_states, q_scale = make_inputs(
            torch.bfloat16, torch.bfloat16
        )
        actual = make_indexer(weight)._get_logits_head_gate(hidden_states, q_scale)
        logits = fp32_reference_logits(hidden_states, weight)
        expected = apply_gate_scale(logits, q_scale)
        # A gate whose logits are written out in 16 bit and only then widened,
        # i.e. what a bf16 GEMM output would produce.
        rounded = apply_gate_scale(logits.to(torch.bfloat16).float(), q_scale)
        self.assertLess(
            max_abs_error(actual, expected),
            max_abs_error(rounded, expected) / 100,
        )

    def test_mismatched_dtypes_keep_the_activation_precision(self) -> None:
        weight, hidden_states, q_scale = make_inputs(
            torch.bfloat16, torch.float16
        )
        actual = make_indexer(weight)._get_logits_head_gate(hidden_states, q_scale)
        expected = fp32_reference_gate(hidden_states, weight, q_scale)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        # The activation must not be cast down to the weight dtype.
        downcast = fp32_reference_gate(
            hidden_states.to(torch.bfloat16), weight, q_scale
        )
        self.assertLess(
            max_abs_error(actual, expected),
            max_abs_error(downcast, expected) / 100,
        )

    def test_streams_the_activation_in_row_blocks(self) -> None:
        # chunk > head_count keeps one row block larger than the fp32 weight, so
        # the bound below is about the staged activation rather than the weight.
        token_count, hidden_size, chunk = 20, 32, 8
        weight, hidden_states, q_scale = make_inputs(
            torch.bfloat16,
            torch.bfloat16,
            token_count=token_count,
            hidden_size=hidden_size,
        )
        indexer = make_indexer(weight)
        expected = fp32_reference_gate(hidden_states, weight, q_scale)
        recorder = Fp32StagingRecorder()

        with mock.patch.object(
            indexer_module, "_HEAD_GATE_FP32_ROW_CHUNK", chunk
        ):
            with recorder:
                actual = indexer._get_logits_head_gate(hidden_states, q_scale)

        self.assertEqual(actual.dtype, torch.float32)
        self.assertLessEqual(recorder.max_elements, chunk * hidden_size)
        self.assertLess(recorder.max_elements, token_count * hidden_size)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_shipped_row_chunk_bounds_the_fp32_staging(self) -> None:
        """No fp32 tensor may reach the size of the whole activation.

        Runs the constant that actually ships over a fixed row count larger
        than it, so a chunk inflated back to the whole activation fails here
        instead of making this test allocate the inflated size.
        """
        hidden_size = 32
        token_count = 8193
        weight, hidden_states, q_scale = make_inputs(
            torch.bfloat16,
            torch.bfloat16,
            token_count=token_count,
            hidden_size=hidden_size,
        )
        indexer = make_indexer(weight)
        recorder = Fp32StagingRecorder()

        with recorder:
            actual = indexer._get_logits_head_gate(hidden_states, q_scale)

        self.assertEqual(actual.dtype, torch.float32)
        self.assertLess(recorder.max_elements, token_count * hidden_size)

    def test_rank_three_activation_is_still_streamed(self) -> None:
        """The row loop strides the leading dim, which for rank 3 is not tokens.

        A [1, tokens, hidden] activation would take one iteration covering
        everything, staging the whole activation in fp32 -- the size this commit
        exists to avoid -- while the rank-2 bound above stays green.
        """
        hidden_size = 32
        token_count = 8193
        weight, hidden_states, q_scale = make_inputs(
            torch.bfloat16,
            torch.bfloat16,
            token_count=token_count,
            hidden_size=hidden_size,
        )
        indexer = make_indexer(weight)
        expected = indexer._get_logits_head_gate(hidden_states, q_scale)
        recorder = Fp32StagingRecorder()

        with recorder:
            actual = indexer._get_logits_head_gate(
                hidden_states.unsqueeze(0), q_scale.unsqueeze(0)
            )

        self.assertEqual(actual.dtype, torch.float32)
        self.assertLess(recorder.max_elements, token_count * hidden_size)
        self.assertEqual(actual.shape, (1, *expected.shape))
        torch.testing.assert_close(actual, expected.unsqueeze(0))

    def test_capability_probe_reports_absent_overload(self) -> None:
        """A torch without the overload must probe False, not be assumed capable.

        Driven by replacing the overload list rather than by comparing against a
        real call: on a torch that has the overload both sides are true, so such
        a comparison would pass even for a probe hardcoded to true.
        """
        with mock.patch.object(
            torch.ops.aten.mm, "overloads", lambda: ["default", "out"]
        ):
            self.assertFalse(indexer_module._mm_supports_out_dtype())

    def test_capability_probe_reports_present_overload(self) -> None:
        with mock.patch.object(
            torch.ops.aten.mm, "overloads", lambda: ["default", "out", "dtype"]
        ):
            self.assertTrue(indexer_module._mm_supports_out_dtype())

    def test_capability_probe_fails_closed_when_torch_ops_raise(self) -> None:
        with mock.patch.object(
            torch.ops.aten.mm, "overloads", side_effect=RuntimeError("no dispatcher")
        ):
            self.assertFalse(indexer_module._mm_supports_out_dtype())

    def test_fp32_weight_widened_once_and_reused(self) -> None:
        """The fallback caches the fp32-widened transposed weight against the source
        parameter, so long-context prefill does not rebuild it inside the CUDA-graph
        capture region on every layer. Removing the cache leaves this test red.
        """
        weight, hidden_states, q_scale = make_inputs(torch.bfloat16, torch.bfloat16)
        indexer = make_indexer(weight)
        expected = fp32_reference_gate(hidden_states, weight, q_scale)

        first = indexer._get_logits_head_gate(hidden_states, q_scale)
        first_cache = indexer._head_gate_fp32_weight
        second = indexer._get_logits_head_gate(hidden_states, q_scale)
        second_cache = indexer._head_gate_fp32_weight

        # Same source parameter -> cache hit -> same fp32 tensor across calls.
        self.assertIs(first_cache[1], second_cache[1])
        self.assertIs(first_cache[0], indexer.weights_proj.weight)
        torch.testing.assert_close(first, expected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(second, expected, rtol=1e-6, atol=1e-6)

        # Reassigning the source parameter must invalidate the cache.
        indexer.weights_proj = CudaF16Linear(
            torch.randn_like(weight).to(torch.bfloat16)
        )
        indexer._get_logits_head_gate(hidden_states, q_scale)
        third_cache = indexer._head_gate_fp32_weight
        self.assertIsNot(third_cache[1], second_cache[1])
        self.assertIs(third_cache[0], indexer.weights_proj.weight)

    def test_out_dtype_capability_gates_the_fast_path(self) -> None:
        """The gate must test the torch capability, not only device and dtype.

        A torch without ``out_dtype`` support is simulated on a tensor that
        reports itself as CUDA: with the capability flag off the head gate still
        returns an fp32-accurate result, while forcing the flag on makes the
        same call raise, which is what the flag exists to prevent.
        """
        weight, hidden_states, q_scale = make_inputs(
            torch.bfloat16, torch.bfloat16
        )
        indexer = make_indexer(weight)
        expected = fp32_reference_gate(hidden_states, weight, q_scale)
        original_mm = torch.mm

        def mm_without_out_dtype(*args, **kwargs):
            if "out_dtype" in kwargs:
                raise TypeError(
                    "mm() got an unexpected keyword argument 'out_dtype'"
                )
            return original_mm(*args, **kwargs)

        with mock.patch.object(
            torch.Tensor, "is_cuda", property(lambda _self: True)
        ), mock.patch.object(torch, "mm", mm_without_out_dtype):
            with mock.patch.object(
                indexer_module, "_MM_SUPPORTS_OUT_DTYPE", False
            ):
                actual = indexer._get_logits_head_gate(hidden_states, q_scale)
            with mock.patch.object(
                indexer_module, "_MM_SUPPORTS_OUT_DTYPE", True
            ):
                with self.assertRaises(TypeError):
                    indexer._get_logits_head_gate(hidden_states, q_scale)

        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_fast_path_accepts_rank_three_activation(self) -> None:
        """torch.mm takes matrices only, so rank 3 raised before the flattening.

        Driven on CPU by reporting is_cuda and answering the out_dtype overload
        locally, the same way the capability test above does, because the real
        overload is CUDA-only.
        """
        weight, hidden_states, q_scale = make_inputs(torch.bfloat16, torch.bfloat16)
        indexer = make_indexer(weight)
        original_mm = torch.mm

        def mm_with_cpu_out_dtype(*args, **kwargs):
            target = kwargs.pop("out_dtype", None)
            if target is None:
                return original_mm(*args, **kwargs)
            return original_mm(args[0].to(target), args[1].to(target), **kwargs)

        with mock.patch.object(
            torch.Tensor, "is_cuda", property(lambda _self: True)
        ), mock.patch.object(torch, "mm", mm_with_cpu_out_dtype), mock.patch.object(
            indexer_module, "_MM_SUPPORTS_OUT_DTYPE", True
        ):
            flat = indexer._get_logits_head_gate(hidden_states, q_scale)
            batched = indexer._get_logits_head_gate(
                hidden_states.unsqueeze(0), q_scale.unsqueeze(0)
            )

        self.assertEqual(batched.dtype, torch.float32)
        self.assertEqual(batched.shape, (1, *flat.shape))
        torch.testing.assert_close(batched, flat.unsqueeze(0))


@unittest.skipUnless(FAST_PATH_OK, FAST_PATH_SKIP_REASON)
class IndexerHeadGateFastPathTest(unittest.TestCase):
    """The out_dtype fast path, which needs a device and torch >= 2.8."""

    def test_bf16_path_matches_fp32_reference_without_expanding_input(
        self,
    ) -> None:
        torch.manual_seed(0)
        weight, hidden_states, q_scale = make_inputs(
            torch.bfloat16, torch.bfloat16, device="cuda"
        )
        indexer = make_indexer(weight)

        with mock.patch.object(torch, "mm", wraps=torch.mm) as mm:
            actual = indexer._get_logits_head_gate(hidden_states, q_scale)

        mm.assert_called_once()
        self.assertIs(mm.call_args.args[0], hidden_states)
        self.assertEqual(mm.call_args.args[0].dtype, torch.bfloat16)
        self.assertEqual(mm.call_args.args[1].dtype, torch.bfloat16)
        self.assertEqual(mm.call_args.kwargs["out_dtype"], torch.float32)
        self.assertEqual(actual.dtype, torch.float32)
        # Pins the claim that the 16-bit fast path reproduces the old full-fp32
        # expansion: the reference comes from the raw weight, so only the GEMM
        # accumulation order differs.
        expected = fp32_reference_gate(hidden_states, weight, q_scale)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

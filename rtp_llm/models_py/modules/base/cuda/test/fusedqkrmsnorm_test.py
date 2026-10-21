import hashlib
import itertools
from pathlib import Path
from unittest import SkipTest, TestCase, main, mock

import torch
from torch import dtype as _dtype

from rtp_llm.models_py.modules import FusedQKRMSNorm, QKRMSNorm


class FusedQKRMSNormTest(TestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 4096]
    HEAD_NUM = [40]
    KV_HEAD_NUM = [40, 8, 4]
    SIZE_PER_HEAD = [128]

    # DTYPES = [torch.bfloat16]
    # NUM_TOKENS = [4096]
    # HEAD_NUM = [40]
    # KV_HEAD_NUM =  [40]
    # SIZE_PER_HEAD = [128]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_fused_qk_rmsnorm_test(
        self,
        num_tokens: int,
        head_num: int,
        kv_head_num: int,
        size_per_head: int,
        dtype: _dtype,
    ):
        torch.manual_seed(0)

        hidden_size = head_num * size_per_head + 2 * kv_head_num * size_per_head

        q_weight = torch.randn(size_per_head, dtype=dtype)
        k_weight = torch.randn(size_per_head, dtype=dtype)

        qkrmsnorm = QKRMSNorm(q_weight, k_weight, head_num, kv_head_num, size_per_head)
        fused_qkrmsnorm = FusedQKRMSNorm(
            q_weight, k_weight, head_num, kv_head_num, size_per_head
        )

        x = torch.randn(num_tokens, hidden_size, dtype=dtype)

        # for _ in range(5):
        #     # out = qkrmsnorm(x)
        #     out = fused_qkrmsnorm(x)
        # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        #     for _ in range(10):
        #         # out = qkrmsnorm(x)
        #         out = fused_qkrmsnorm(x)
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))

        self.assertTrue(
            torch.allclose(qkrmsnorm(x), fused_qkrmsnorm(x), atol=1e-2, rtol=1e-2)
        )

    def test_fusedqkrmsnorm(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HEAD_NUM,
            self.KV_HEAD_NUM,
            self.SIZE_PER_HEAD,
            self.DTYPES,
        ):
            with self.subTest(
                num_tokens=params[0],
                head_num=params[1],
                kv_head_num=params[2],
                size_per_head=params[3],
                dtype=params[4],
            ):
                self._run_fused_qk_rmsnorm_test(*params)


# This suite deliberately uses the installed CUDA FlashInfer operator as the
# independent oracle. The legacy vec2 implementation and a PyTorch norm formula
# are not bitwise references for its vec8/FTZ/FMA arithmetic.
def _fi_qk_in_place(x, qw, kw, eps=1e-6, heads=16, kv_heads=8, enable_pdl=False):
    import flashinfer

    qkv = x.reshape(x.shape[0], heads + 2 * kv_heads, 128)
    q = qkv[:, :heads, :]
    k = qkv[:, heads : heads + kv_heads, :]
    flashinfer.norm.rmsnorm(q, qw, eps=eps, out=q, enable_pdl=enable_pdl)
    flashinfer.norm.rmsnorm(k, kw, eps=eps, out=k, enable_pdl=enable_pdl)
    return x


def _vec8_op(x, qw, kw, eps=1e-6):
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    rtp_llm_ops.fused_qk_rmsnorm_cuda_vec8(x, qw, kw, eps)
    return x


def _fixture(m, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, 4096, device="cuda", dtype=torch.bfloat16, generator=generator)
    qw = torch.randn(128, device="cuda", dtype=torch.bfloat16, generator=generator)
    kw = torch.randn(128, device="cuda", dtype=torch.bfloat16, generator=generator)
    return x, qw, kw


def _assert_bytes(case, actual, expected, label):
    case.assertEqual(actual.shape, expected.shape, label)
    case.assertEqual(actual.dtype, expected.dtype, label)
    # NaNs and signed zero are compared as bytes, not via allclose/equal on float.
    a = actual.contiguous().view(torch.uint8)
    b = expected.contiguous().view(torch.uint8)
    if not torch.equal(a, b):
        mismatches = int(torch.count_nonzero(a != b).item())
        case.fail(f"{label}: {mismatches} byte mismatches; no tolerance is allowed")


class FusedQKRMSNormCudaVec8ExactTest(TestCase):
    SHAPES = (
        1,
        7,
        127,
        128,
        129,
        256,
        257,
        384,
        500,
        512,
        768,
        1000,
        1024,
        1536,
        2000,
        2048,
    )

    @classmethod
    def setUpClass(cls):
        import flashinfer

        from rtp_llm.ops.compute_ops import rtp_llm_ops

        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        if torch.cuda.get_device_capability() != (12, 0):
            raise SkipTest("CUDA vec8 contract is SM120 only")
        if getattr(flashinfer.norm, "_USE_CUDA_NORM", False) is not True:
            raise SkipTest(
                "This exact oracle requires the current FlashInfer CUDA norm route"
            )
        if not hasattr(rtp_llm_ops, "fused_qk_rmsnorm_cuda_vec8"):
            raise AssertionError("The CUDA/BF16 vec8 binding was not built")
        norm_source = (
            Path(flashinfer.__file__).resolve().parent
            / "data/include/flashinfer/norm.cuh"
        )
        expected_source_sha = (
            "f32e86ca9e3a9caf0163ac619464f784f26cddaa8f19ef47a706ca7a602b0987"
        )
        if hashlib.sha256(norm_source.read_bytes()).hexdigest() != expected_source_sha:
            raise AssertionError(
                "FlashInfer CUDA oracle source changed; re-audit its numerical contract"
            )

    def compare(self, x, qw, kw, eps=1e-6, module=True):
        expected = x.clone()
        actual = x.clone()
        q_before, k_before = qw.clone(), kw.clone()
        _fi_qk_in_place(expected, qw, kw, eps)
        if module:
            norm = FusedQKRMSNorm(qw, kw, 16, 8, 128, eps, enable_pdl=False)
            returned = norm(actual)
            self.assertEqual(returned.data_ptr(), actual.data_ptr())
        else:
            returned = _vec8_op(actual, qw, kw, eps)
            self.assertEqual(returned.data_ptr(), actual.data_ptr())
        _assert_bytes(self, actual, expected, "Q/K and V")
        _assert_bytes(self, actual[:, 3072:], x[:, 3072:], "V remains unchanged")
        _assert_bytes(self, qw, q_before, "Q weight remains unchanged")
        _assert_bytes(self, kw, k_before, "K weight remains unchanged")

    def test_01_random_actual_and_odd_shapes_three_rounds(self):
        for m in self.SHAPES:
            for repeat in range(3):
                with self.subTest(m=m, repeat=repeat):
                    self.compare(*_fixture(m, 17 + repeat), module=repeat != 1)

    def test_02_ieee_edges_and_rounding_neighbors(self):
        # Full BF16 bit patterns, including both signs of zero/subnormals,
        # smallest normals, finite extrema, infinities and NaN payloads.
        patterns = [
            0x0000,
            0x8000,
            0x0001,
            0x8001,
            0x007F,
            0x807F,
            0x0080,
            0x8080,
            0x3F7F,
            0x3F80,
            0x3F81,
            0xBF7F,
            0xBF80,
            0xBF81,
            0x7F7F,
            0xFF7F,
            0x7F80,
            0xFF80,
            0x7FC1,
            0xFFC1,
            0x7F81,
            0xFF81,
        ]
        signed = [v if v < 32768 else v - 65536 for v in patterns]
        values = (
            torch.tensor(signed, dtype=torch.int16, device="cpu")
            .view(torch.bfloat16)
            .cuda()
        )
        x, qw, kw = _fixture(7, 391)
        for offset in range(len(patterns)):
            a = x.clone().view(7, 32, 128)
            # Distinct heads isolate infinity/NaN from finite boundary cases.
            for head, value in enumerate(values):
                a[:, head, :] = value
                a[:, head, (offset * 7) % 128] = values[(head + offset) % len(values)]
            q, k = qw.clone(), kw.clone()
            q[: len(values)] = values.roll(offset)
            k[: len(values)] = values.roll(-offset)
            with self.subTest(offset=offset):
                self.compare(a.view(7, 4096), q, k)
        # Rounding neighborhoods of BF16 powers, without infinities masking sums.
        for exponent in (1, 32, 63, 95, 126, 127, 128, 160, 190, 220, 253):
            bits = (
                torch.arange(
                    (exponent << 7) - 1,
                    (exponent << 7) + 2,
                    dtype=torch.int16,
                    device="cpu",
                )
                .view(torch.bfloat16)
                .cuda()
            )
            a, q, k = _fixture(1, exponent)
            a[:, :3072] = bits.repeat(1024)
            self.compare(a, q, k)

    def test_03_weight_alias_and_storage_canaries(self):
        x, qw, kw = _fixture(129, 92)
        self.compare(x, qw, qw)  # Both read-only weights may alias each other.
        slab = torch.full((x.numel() + 32,), 0.375, device="cuda", dtype=x.dtype)
        actual = slab[16:-16].view_as(x)
        actual.copy_(x)
        before = slab.clone()
        expected = x.clone()
        _fi_qk_in_place(expected, qw, kw)
        _vec8_op(actual, qw, kw)
        _assert_bytes(self, actual, expected, "aligned nonzero storage offset")
        _assert_bytes(self, slab[:16], before[:16], "prefix canary")
        _assert_bytes(self, slab[-16:], before[-16:], "suffix canary")

    def test_04_graph_changes_input_and_both_weights(self):
        for m in (1, 128, 256, 512, 1024, 1536, 2048):
            with self.subTest(m=m):
                sx, sq, sk = _fixture(m, 120)
                a, b = sx.clone(), sx.clone()
                aq, ak, bq, bk = sq.clone(), sk.clone(), sq.clone(), sk.clone()
                norm = FusedQKRMSNorm(aq, ak, 16, 8, 128, enable_pdl=False)
                for repeat in range(3):
                    fresh = _fixture(m, 121 + repeat)
                    self.compare(*fresh)
                # Compile and resolve all device metadata before capture.
                warm = torch.cuda.Stream()
                warm.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warm):
                    for _ in range(3):
                        a.copy_(sx)
                        b.copy_(sx)
                        norm(a)
                        _fi_qk_in_place(b, bq, bk)
                torch.cuda.current_stream().wait_stream(warm)
                torch.cuda.synchronize()
                ga, gb = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
                with torch.cuda.graph(ga):
                    norm(a)
                with torch.cuda.graph(gb):
                    _fi_qk_in_place(b, bq, bk)
                addresses = [t.data_ptr() for t in (a, aq, ak, b, bq, bk)]
                for repeat in range(3):
                    nx, nq, nk = _fixture(m, 400 + repeat)
                    eager = nx.clone()
                    _fi_qk_in_place(eager, nq, nk)
                    a.copy_(nx)
                    b.copy_(nx)
                    aq.copy_(nq)
                    bq.copy_(nq)
                    ak.copy_(nk)
                    bk.copy_(nk)
                    ga.replay()
                    gb.replay()
                    _assert_bytes(self, a, eager, "candidate Graph vs original eager")
                    _assert_bytes(self, b, eager, "original Graph vs original eager")
                    _assert_bytes(self, a, b, "candidate Graph vs original Graph")
                    _assert_bytes(self, a[:, 3072:], nx[:, 3072:], "Graph V")
                    self.assertEqual(
                        addresses, [t.data_ptr() for t in (a, aq, ak, b, bq, bk)]
                    )
                for repeat in range(3):
                    self.compare(*_fixture(m, 700 + repeat))

    def test_05_nondefault_stream_producer_and_consumer(self):
        x, q, k = _fixture(256, 625)
        expected = x.clone()
        _fi_qk_in_place(expected, q, k)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            actual = x.clone()
            _vec8_op(actual, q, k)
            consumed = actual.clone()
        torch.cuda.current_stream().wait_stream(stream)
        _assert_bytes(
            self, consumed, expected, "nondefault-stream producer/norm/consumer"
        )

    def test_06_direct_rejects_before_writing(self):
        x, q, k = _fixture(7, 12)
        unaligned_input = torch.empty(x.numel() + 1, device="cuda", dtype=x.dtype)[
            1:
        ].view_as(x)
        unaligned_input.copy_(x)
        unaligned_weight = torch.empty(129, device="cuda", dtype=x.dtype)[1:]
        unaligned_weight.copy_(q)
        noncontiguous = torch.empty(7, 8192, device="cuda", dtype=x.dtype)[:, ::2]
        noncontiguous.copy_(x)
        cases = [
            (x.half(), q.half(), k.half(), 1e-6),
            (x[:, :3968].contiguous(), q, k, 1e-6),
            (x.view(7, 32, 128), q, k, 1e-6),
            (noncontiguous, q, k, 1e-6),
            (unaligned_input, q, k, 1e-6),
            (x, unaligned_weight, k, 1e-6),
            (x, q, unaligned_weight, 1e-6),
            (x, q[:127], k, 1e-6),
            (x, q.view(1, 128), k, 1e-6),
            (x, q.repeat(2)[::2], k, 1e-6),
            (x, x[0, :128], k, 1e-6),
            (x, q, x[0, 2048:2176], 1e-6),
            (x, q.cpu(), k, 1e-6),
            (x.cpu(), q.cpu(), k.cpu(), 1e-6),
            (torch.zeros(0, 4096, device="cuda", dtype=x.dtype), q, k, 1e-6),
            (torch.zeros(2049, 4096, device="cuda", dtype=x.dtype), q, k, 1e-6),
        ] + [
            (x, q, k, eps)
            for eps in (0.0, -1.0, float("nan"), float("inf"), -float("inf"))
        ]
        for index, (a, wq, wk, eps) in enumerate(cases):
            snapshots = [t.clone() for t in (a, wq, wk)]
            with self.subTest(index=index):
                with self.assertRaises((RuntimeError, ValueError)):
                    _vec8_op(a, wq, wk, eps)
                for value, saved in zip((a, wq, wk), snapshots):
                    _assert_bytes(self, value, saved, "rejection must not write")

    def test_07_real_supported_fallbacks(self):
        # These are legal inputs to the original FI operator. Alignment-invalid
        # inputs are tested only for delegation/rejection, never sent to FI.
        for dtype, heads, kv_heads, pdl in (
            (torch.float16, 16, 8, False),
            (torch.bfloat16, 8, 8, False),
            (torch.bfloat16, 16, 8, True),
            (torch.bfloat16, 16, 8, None),
        ):
            with self.subTest(dtype=dtype, heads=heads, pdl=pdl):
                width = (heads + 2 * kv_heads) * 128
                x = torch.randn(7, width, device="cuda", dtype=dtype)
                q = torch.randn(128, device="cuda", dtype=dtype)
                k = torch.randn(128, device="cuda", dtype=dtype)
                expected, actual = x.clone(), x.clone()
                _fi_qk_in_place(
                    expected, q, k, heads=heads, kv_heads=kv_heads, enable_pdl=pdl
                )
                norm = FusedQKRMSNorm(q, k, heads, kv_heads, 128, enable_pdl=pdl)
                norm(actual)
                _assert_bytes(self, actual, expected, "real FI fallback")

    def test_08_delegation_guards_without_unsupported_kernel_launch(self):
        import flashinfer

        from rtp_llm.models_py.modules.base.cuda import norm as norm_module

        x, q, k = _fixture(7, 83)
        bad_x = torch.empty(x.numel() + 1, device="cuda", dtype=x.dtype)[1:].view_as(x)
        bad_x.copy_(x)
        bad_q = torch.empty(129, device="cuda", dtype=q.dtype)[1:]
        bad_q.copy_(q)
        strided = torch.empty(7, 8192, device="cuda", dtype=x.dtype)[:, ::2]
        strided.copy_(x)
        scenarios = [
            (bad_x, q, k, 1e-6),
            (x, bad_q, k, 1e-6),
            (strided, q, k, 1e-6),
            (x, x[0, :128], k, 1e-6),
            (x, q, k, 0.0),
            (x, q, k, float("nan")),
            (torch.zeros(2049, 4096, device="cuda", dtype=x.dtype), q, k, 1e-6),
        ]
        for index, (a, qw, kw, eps) in enumerate(scenarios):
            with self.subTest(index=index), mock.patch.object(
                flashinfer.norm, "rmsnorm"
            ) as fallback:
                saved = a.clone()
                FusedQKRMSNorm(qw, kw, 16, 8, 128, eps, enable_pdl=False)(a)
                self.assertEqual(fallback.call_count, 2)
                self.assertEqual(
                    fallback.call_args_list[0].kwargs["out"].data_ptr(), a.data_ptr()
                )
                self.assertEqual(
                    fallback.call_args_list[1].kwargs["out"].data_ptr(),
                    a.reshape(a.shape[0], 32, 128)[:, 16:24, :].data_ptr(),
                )
                _assert_bytes(self, a, saved, "mocked fallback control-flow only")
        # This is a branch contract test, not a claim that CuTe ran or matched.
        with mock.patch.object(
            flashinfer.norm, "_USE_CUDA_NORM", False
        ), mock.patch.object(flashinfer.norm, "rmsnorm") as fallback:
            FusedQKRMSNorm(q, k, 16, 8)(x)
            self.assertEqual(fallback.call_count, 2)
        with mock.patch.object(norm_module, "rtp_llm_ops", object()), mock.patch.object(
            flashinfer.norm, "rmsnorm"
        ) as fallback:
            FusedQKRMSNorm(q, k, 16, 8)(x)
            self.assertEqual(fallback.call_count, 2)

    def test_09_epsilon_double_to_float_contract(self):
        for eps in (1e-300, 1e-45, 1e-38, 1e-6, 0.5, 1e38, 1e100, 1e300):
            with self.subTest(eps=eps):
                self.compare(*_fixture(7, 299), eps=eps)
                self.compare(*_fixture(7, 300), eps=eps, module=False)

    def test_10_module_replaced_weight_storage(self):
        x, q, k = _fixture(128, 622)
        norm = FusedQKRMSNorm(q, k, 16, 8)
        for repeat in range(3):
            nx, nq, nk = _fixture(128, repeat + 621)
            norm.q_weight, norm.k_weight = nq, nk
            expected = nx.clone()
            _fi_qk_in_place(expected, nq, nk)
            norm(nx)
            _assert_bytes(self, nx, expected, "replacement weight addresses stay live")

    def test_11_actual_module_calls_new_binding_once(self):
        from rtp_llm.models_py.modules.base.cuda import norm as norm_module

        x, q, k = _fixture(128, 630)
        expected = x.clone()
        _fi_qk_in_place(expected, q, k)
        original_op = norm_module.rtp_llm_ops.fused_qk_rmsnorm_cuda_vec8
        with mock.patch.object(
            norm_module.rtp_llm_ops, "fused_qk_rmsnorm_cuda_vec8", wraps=original_op
        ) as invoked:
            returned = FusedQKRMSNorm(q, k, 16, 8, enable_pdl=False)(x)
            self.assertEqual(invoked.call_count, 1)
            self.assertEqual(returned.data_ptr(), x.data_ptr())
        _assert_bytes(self, x, expected, "real module -> CUDA binding")


if __name__ == "__main__":
    main()

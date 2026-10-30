"""Exercise newly reachable NVFP4 model/Linear branches on real SM103 kernels."""

import os
import socket
import sys
import unittest
from contextlib import contextmanager
from datetime import timedelta
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from flashinfer import fp4_quantize, mm_fp4

from rtp_llm.config.quant_config import ModelOptFp4Config
from rtp_llm.models.kimi_k3.nvfp4_tensor import (
    nvfp4_global_scale,
    unswizzle_nvfp4_scales,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_comm_fusion import (
    PackedNvfp4Activation,
    all_gather_nvfp4_gemm,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_linear import (
    CudaFp4GEMMLinear,
    Fp4Activation,
)
from rtp_llm.models_py.modules.hybrid.kda_attention import KdaAttention
from rtp_llm.models_py.modules.kimi_k3.dsa import KimiK31DSA
from rtp_llm.models_py.modules.kimi_k3.natural_dsa import (
    NaturalPairedProjection,
    split_natural_projection,
)
from rtp_llm.models_py.triton_kernels.common import nvfp4 as quant


@contextmanager
def no_flashinfer_quant():
    with patch(
        "flashinfer.fp4_quantize", side_effect=AssertionError("FI quant called")
    ):
        yield


def make_linear(n, k, *, dynamic=True):
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02
    scale = nvfp4_global_scale(weight)
    codes, scales = fp4_quantize(weight, scale.reciprocal())
    return CudaFp4GEMMLinear(
        codes,
        scales,
        weight_scale_2=scale,
        quant_config=ModelOptFp4Config(
            bits=4,
            group_size=16,
            is_quanted=False,
            activation_scheme="dynamic" if dynamic else "static",
        ),
        input_scales=None if dynamic else torch.ones(1, device="cuda"),
    )


def reference_quant(x):
    scale = nvfp4_global_scale(x)
    dense = x.contiguous()
    # A singleton row can already be contiguous while retaining an unaligned
    # storage offset. FlashInfer's reference packer needs an aligned buffer;
    # the native candidate still receives the original, unaligned view.
    if dense.data_ptr() % 32:
        dense = dense.clone(memory_format=torch.contiguous_format)
    values, scales = fp4_quantize(dense, scale.reciprocal())
    torch.cuda.synchronize()
    return Fp4Activation(values, scales, scale)


def check_quant(actual, expected):
    rows, packed = actual.values.shape
    assert torch.equal(actual.values, expected.values), "FP4 code mismatch"
    a = unswizzle_nvfp4_scales(actual.block_scales, rows, packed // 8)
    b = unswizzle_nvfp4_scales(expected.block_scales, rows, packed // 8)
    assert torch.equal(a, b), "Logical block-scale mismatch"
    torch.testing.assert_close(actual.scale, expected.scale, rtol=1e-6, atol=1e-12)


def reference_linear(linear, x):
    q = reference_quant(x)
    return mm_fp4(
        q.values,
        linear.weight.T,
        q.block_scales,
        linear.weight_scales.T,
        q.scale * linear.weight_scale_2,
        torch.bfloat16,
        backend="cutlass",
    )


def input_matrix(rows, width, layout):
    if layout == "dense":
        return torch.randn(rows, width, dtype=torch.bfloat16, device="cuda")
    if layout == "row_stride":
        return torch.randn(rows, width + 64, dtype=torch.bfloat16, device="cuda")[
            :, :width
        ]
    if layout == "inner_stride":
        return torch.randn(rows, width * 2, dtype=torch.bfloat16, device="cuda")[:, ::2]
    if layout == "misaligned":
        storage = torch.randn(
            rows * (width + 2) + 1, dtype=torch.bfloat16, device="cuda"
        )
        return storage[1:].as_strided((rows, width), (width + 2, 1))
    if layout == "expanded":
        return torch.randn(1, width, dtype=torch.bfloat16, device="cuda").expand(
            rows, -1
        )
    raise AssertionError(layout)


class KdaStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parallelism_config = SimpleNamespace(tp_size=8)
        self.local_num_k_heads = self.local_num_v_heads = 12
        self.head_k_dim = self.head_v_dim = 128


def make_dsa(q_rank=1536, *, ue8=True):
    # Invoke real production methods with real norm/quant/GEMM, replacing only
    # SFA/cache consumers outside the changed projection boundary.
    module = KimiK31DSA.__new__(KimiK31DSA)
    torch.nn.Module.__init__(module)
    module.q_rank, module.kv_rank, module.suffix_dim = q_rank, 512, 64
    module.q_norm = SimpleNamespace(
        weight=torch.ones(q_rank, dtype=torch.bfloat16, device="cuda"),
        variance_epsilon=1e-5,
    )
    module.kv_norm = SimpleNamespace(
        weight=torch.ones(512, dtype=torch.bfloat16, device="cuda"),
        variance_epsilon=1e-5,
    )
    module.q_b = make_linear(192, q_rank)
    module.indexer = SimpleNamespace(
        wq_b=SimpleNamespace(scale_ue8m0=ue8, FLASHINFER_M_THRESHOLD=32)
    )
    module.query_heads, module.q_dim, module.layer_idx = 1, 192, 0
    return module


class Nvfp4DispatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert torch.cuda.device_count() == 1
        assert torch.cuda.get_device_capability() == (10, 3)
        os.environ["RTP_LLM_FP4_GEMM_BACKEND"] = "cutlass"
        torch.set_num_threads(1)
        torch.manual_seed(41)

    def test_all_rows_and_layouts_use_native_dynamic(self):
        cases = list(
            product(
                (1, 31, 512, 513, 4096),
                (64, 128, 192, 256, 1536, 7168),
                ("dense", "row_stride"),
            )
        )
        cases += list(
            product((1, 513), (128, 1536), ("inner_stride", "misaligned", "expanded"))
        )
        cases += [(32768, k, "dense") for k in (128, 1536, 7168)]
        for rows, width, layout in cases:
            with self.subTest(rows=rows, width=width, layout=layout):
                x = input_matrix(rows, width, layout)
                expected = reference_quant(x)
                with no_flashinfer_quant(), patch.object(
                    quant, "quantize_strided_nvfp4", wraps=quant.quantize_strided_nvfp4
                ) as native, patch.object(
                    quant,
                    "quantize_nvfp4_from_partials",
                    wraps=quant.quantize_nvfp4_from_partials,
                ) as parallel:
                    actual = CudaFp4GEMMLinear.quantize_dynamic(x)
                if (rows, width) in ((32768, 1536), (32768, 7168)):
                    self.assertEqual(native.call_count, 0)
                    self.assertEqual(parallel.call_count, 1)
                    backend = "parallel"
                else:
                    self.assertEqual(native.call_count, 1)
                    self.assertEqual(parallel.call_count, 0)
                    staged = native.call_args.args[0]
                    if layout in ("dense", "row_stride", "misaligned") or (
                        layout == "expanded" and rows == 1
                    ):
                        self.assertEqual(staged.data_ptr(), x.data_ptr())
                    else:
                        self.assertTrue(staged.is_contiguous())
                        self.assertNotEqual(staged.data_ptr(), x.data_ptr())
                    backend = "cooperative"
                check_quant(actual, expected)

    def test_large_tensor_dispatch_boundary_and_graph_refresh(self):
        for rows, backend in (
            (4095, "cooperative"),
            (4096, "parallel"),
            (4097, "parallel"),
        ):
            for layout in ("dense", "row_stride", "inner_stride", "misaligned"):
                x = input_matrix(rows, 8192, layout)
                expected = reference_quant(x)
                with no_flashinfer_quant(), patch.object(
                    quant, "quantize_strided_nvfp4", wraps=quant.quantize_strided_nvfp4
                ) as cooperative, patch.object(
                    quant,
                    "quantize_nvfp4_from_partials",
                    wraps=quant.quantize_nvfp4_from_partials,
                ) as parallel:
                    actual = CudaFp4GEMMLinear.quantize_dynamic(x)
                self.assertEqual(cooperative.call_count, int(backend == "cooperative"))
                self.assertEqual(parallel.call_count, int(backend == "parallel"))
                check_quant(actual, expected)
        x = input_matrix(4096, 8192, "row_stride")
        for _ in range(3):
            quant.quantize_dynamic_nvfp4(x)
        graph = torch.cuda.CUDAGraph()
        with no_flashinfer_quant(), torch.cuda.graph(graph):
            actual = quant.quantize_dynamic_nvfp4(x)
        for magnitude in (0.125, 2.0, 0.0):
            x.copy_(torch.randn_like(x) * magnitude)
            expected = reference_quant(x)
            graph.replay()
            check_quant(actual, expected)

    def test_kda_forget_new_widths_and_linear_empty_contract(self):
        for width in (64, 128, 192, 256):
            linear = make_linear(1536, width)
            model = KdaAttention(
                input_projection=torch.nn.Identity(),
                forget_projection=linear,
                output_projection=torch.nn.Identity(),
                fa_width=width,
                norm=torch.nn.Identity(),
                prefill=KdaStub(),
                decode=KdaStub(),
            )
            for rows, layout in product(
                (1, 31, 512, 513), ("row_stride", "inner_stride")
            ):
                with self.subTest(width=width, rows=rows, layout=layout):
                    x = input_matrix(rows, width, layout)
                    expected = reference_linear(linear, x)
                    with no_flashinfer_quant(), patch.object(
                        quant,
                        "quantize_dynamic_nvfp4",
                        wraps=quant.quantize_dynamic_nvfp4,
                    ) as native:
                        actual = model._project_forget(x)
                    self.assertEqual(native.call_count, 1)
                    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.0625)
            with no_flashinfer_quant():
                empty = model._project_forget(
                    torch.empty(0, width, dtype=torch.bfloat16, device="cuda")
                )
            self.assertEqual(list(empty.shape), [0, 1536])

    def test_fp16_and_group16_compatibility_is_preserved(self):
        for dtype, width in product((torch.bfloat16, torch.float16), (16, 32, 96, 128)):
            x = torch.randn(513, width, dtype=dtype, device="cuda")
            expected = reference_quant(x)
            with no_flashinfer_quant():
                actual = CudaFp4GEMMLinear.quantize_dynamic(x)
            check_quant(actual, expected)

    def test_graph_replay_refreshes_scale_for_all_layouts(self):
        for rows, width, layout in (
            (32, 128, "row_stride"),
            (513, 128, "dense"),
            (513, 256, "inner_stride"),
            (513, 1536, "misaligned"),
        ):
            x = input_matrix(rows, width, layout)
            for _ in range(3):
                quant.quantize_dynamic_nvfp4(x)
            graph = torch.cuda.CUDAGraph()
            with no_flashinfer_quant(), torch.cuda.graph(graph):
                actual = quant.quantize_dynamic_nvfp4(x)
            for magnitude in (0.125, 2.0, 0.0):
                x.copy_(torch.randn_like(x) * magnitude)
                expected = reference_quant(x)
                graph.replay()
                check_quant(actual, expected)

    def test_prefill_query_normalizer_all_rows_and_indexer_flags(self):
        for ue8 in (False, True):
            module = make_dsa(ue8=ue8)
            for rows in (1, 31, 32, 512, 513, 4096):
                source = torch.randn(rows, 13824, dtype=torch.bfloat16, device="cuda")
                x = source[:, :1536]
                with no_flashinfer_quant():
                    normalized, fp8, activation = module._normalize_nvfp4_query(x)
                    actual = module.q_b.forward_quantized(activation)
                check_quant(activation, reference_quant(normalized))
                expected = reference_linear(module.q_b, normalized)
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.0625)
                self.assertEqual(list(fp8[0].shape), [rows, 1536])

    def test_decode_norm_and_query_ignore_packet_and_indexer_gating(self):
        for ue8, packet in product((False, True), (False, True)):
            module = make_dsa(ue8=ue8)
            sfa = SimpleNamespace(
                decode_query=lambda x, layer: x,
                write_decode_cache=lambda *args: None,
            )
            for rows in (1, 8, 32, 513):
                source = torch.randn(rows, 5184, dtype=torch.bfloat16, device="cuda")
                state = SimpleNamespace(
                    qkv=source[:, :2112],
                    fp4=object() if packet else None,
                    sfa=sfa,
                    sfa_cache=None,
                )
                with no_flashinfer_quant(), patch.object(
                    quant,
                    "quantize_nvfp4_from_partials",
                    wraps=quant.quantize_nvfp4_from_partials,
                ) as pack:
                    module._norm(state)
                    module._sfa_query(state)
                self.assertIsNotNone(state.q_amax)
                self.assertEqual(state.q_fp8 is not None, ue8)
                self.assertEqual(pack.call_count, 1)
                expected = reference_linear(module.q_b, state.q).view(rows, 1, 192)
                torch.testing.assert_close(
                    state.sfa_query, expected, rtol=0.02, atol=0.0625
                )
            # The real producer/quant/GEMM sequence must refresh on graph replay.
            for _ in range(3):
                module._norm(state)
                module._sfa_query(state)
            graph = torch.cuda.CUDAGraph()
            with no_flashinfer_quant(), torch.cuda.graph(graph):
                module._norm(state)
                module._sfa_query(state)
            for magnitude in (0.25, 2.0, 0.0):
                source.copy_(torch.randn_like(source) * magnitude)
                graph.replay()
                expected = reference_linear(module.q_b, state.q).view(rows, 1, 192)
                torch.testing.assert_close(
                    state.sfa_query, expected, rtol=0.02, atol=0.0625
                )

    def test_decode_unaligned_q_rank_uses_optimized_dynamic(self):
        module = make_dsa(1280, ue8=False)
        sfa = SimpleNamespace(
            decode_query=lambda x, layer: x, write_decode_cache=lambda *args: None
        )
        state = SimpleNamespace(
            qkv=torch.randn(513, 1856, dtype=torch.bfloat16, device="cuda"),
            fp4=None,
            sfa=sfa,
            sfa_cache=None,
        )
        module._norm(state)
        self.assertIsNone(state.q_amax)
        with no_flashinfer_quant(), patch.object(
            quant, "quantize_dynamic_nvfp4", wraps=quant.quantize_dynamic_nvfp4
        ) as native:
            module._sfa_query(state)
        self.assertEqual(native.call_count, 1)
        torch.testing.assert_close(
            state.sfa_query,
            reference_linear(module.q_b, state.q).view(513, 1, 192),
            rtol=0.02,
            atol=0.0625,
        )

    def test_real_prefill_query_branch_with_raw_or_packed_inputs(self):
        for ue8 in (False, True):
            module = make_dsa(ue8=ue8)
            module.heads, module.v_dim = 1, 128
            module._share_fp8_q_latent = False
            module._fp8_output_gate = False
            module.output = torch.nn.Identity()
            full = make_linear(2240, 7168)
            q, kv, gate = split_natural_projection(full, 1536, 576, 128, 0)
            module.natural_q_gate_input = NaturalPairedProjection(q, gate)
            module.natural_kv_input = kv
            module.natural_kv_indexer_input = None
            module.natural_gate_input = gate
            for rows, packet_input in product((1, 31, 32, 513), (False, True)):
                with self.subTest(rows=rows, ue8=ue8, packet=packet_input):
                    x = torch.randn(rows, 7168, dtype=torch.bfloat16, device="cuda")
                    captures = {}

                    class Exchange:
                        def exchange(
                            self, value, plan, direction, *, before_consume=None
                        ):
                            if before_consume is not None:
                                before_consume()
                            return value

                        def consume_nvfp4(
                            self,
                            packet,
                            plan,
                            project,
                            *,
                            before_consume,
                            project_heads,
                        ):
                            before_consume()
                            project(packet.unpack(packet.wire))
                            return None

                    owner = SimpleNamespace(
                        row_exchange=Exchange(), routing_plan=None, route_chunk_rows=32
                    )

                    def sfa_forward(query, latent, suffix, *args):
                        captures.update(
                            query=query.clone(),
                            latent=latent.clone(),
                            suffix=suffix.clone(),
                        )
                        return query[..., :128]

                    sfa = SimpleNamespace(workspace_owner=owner, forward=sfa_forward)
                    metadata = SimpleNamespace(
                        query_layout=SimpleNamespace(local_rows=rows, all_valid=True)
                    )
                    packet = None
                    if packet_input:
                        packet = PackedNvfp4Activation.empty(
                            rows, 7168, x.device, rowwise=False
                        )
                        quant.quantize_nvfp4_from_partials(
                            x, x.abs().amax().float().reshape(1), packet=packet
                        )
                    original = module._normalize_nvfp4_query

                    def normalize(value):
                        result = original(value)
                        captures["normalized"] = result[0].clone()
                        return result

                    with no_flashinfer_quant(), patch.object(
                        module, "_normalize_nvfp4_query", side_effect=normalize
                    ) as norm, patch(
                        "rtp_llm.models_py.modules.kimi_k3.dsa.resolve_mla_routes",
                        return_value=(sfa, None, None, None),
                    ), patch(
                        "rtp_llm.models_py.modules.kimi_k3.dsa.run_prepared_sparse_indexer",
                        return_value=None,
                    ):
                        actual = module._forward_natural(
                            x,
                            None,
                            None,
                            metadata,
                            natural_fp8=object() if packet_input else None,
                            natural_nvfp4=packet,
                        )
                    self.assertEqual(norm.call_count, 1)
                    expected_qgate = reference_linear(
                        module.natural_q_gate_input.nvfp4, x
                    )
                    raw_q, expected_gate = expected_qgate.split((1536, 128), dim=-1)
                    torch.testing.assert_close(
                        captures["normalized"],
                        torch.nn.functional.rms_norm(
                            raw_q.float(), (1536,), module.q_norm.weight.float(), 1e-5
                        ).bfloat16(),
                        rtol=0.02,
                        atol=0.0625,
                    )
                    expected_kv = reference_linear(kv, x)
                    expected_latent = torch.nn.functional.rms_norm(
                        expected_kv[:, :512].float(),
                        (512,),
                        module.kv_norm.weight.float(),
                        1e-5,
                    ).bfloat16()
                    torch.testing.assert_close(
                        captures["latent"], expected_latent, rtol=0.02, atol=0.0625
                    )
                    torch.testing.assert_close(
                        captures["suffix"], expected_kv[:, 512:], rtol=0.02, atol=0.0625
                    )
                    expected_query = reference_linear(
                        module.q_b, captures["normalized"]
                    ).view(rows, 1, 192)
                    torch.testing.assert_close(
                        captures["query"], expected_query, rtol=0.02, atol=0.0625
                    )
                    expected = (
                        expected_query[:, :, :128].reshape(rows, 128).float()
                        * expected_gate.float().sigmoid()
                    ).bfloat16()
                    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.0625)


def distributed_worker(rank, port):
    torch.cuda.set_device(rank)
    torch.manual_seed(91)
    os.environ["RTP_LLM_FP4_GEMM_BACKEND"] = "cutlass"
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(minutes=10),
        device_id=torch.device("cuda", rank),
    )
    try:
        linear = make_linear(192, 7168)
        torch.manual_seed(192 + rank)
        for rows in (31, 512, 513):
            x = torch.randn(rows, 7232, dtype=torch.bfloat16, device="cuda")[:, :7168]
            with no_flashinfer_quant(), patch.object(
                quant, "quantize_dynamic_nvfp4", wraps=quant.quantize_dynamic_nvfp4
            ) as native:
                actual = all_gather_nvfp4_gemm(
                    x, linear, dist.group.WORLD, overlap=False
                )
            assert native.call_count == 1
            pieces = [torch.empty_like(x) for _ in range(2)]
            dist.all_gather(pieces, x.contiguous(), group=dist.group.WORLD)
            # AG/GEMM calibrates each source shard separately, not the concat.
            expected = torch.cat([reference_linear(linear, piece) for piece in pieces])
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.0625)
    finally:
        dist.destroy_process_group()


def main():
    if "--distributed" not in sys.argv:
        unittest.main()
        return

    assert torch.cuda.device_count() == 2
    assert all(torch.cuda.get_device_capability(i) == (10, 3) for i in range(2))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(distributed_worker, args=(port,), nprocs=2, join=True)


if __name__ == "__main__":
    main()

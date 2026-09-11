"""Exact-byte oracle and optional microbenchmarks of GLM5 router/pack fusion.

This script does not patch production dispatch. GPU execution requires --gpu
and RTP_GLM5_MOE_ROUTER_PACK_GPU=1 on one explicitly reserved GPU. --benchmark
times GroupTopK+pack only; --abi --benchmark adds the real GLM5MegaMoE consumer
with identical weights and buffers at EP1. Neither benchmark includes norm,
gate, shared experts, initialization, or TP8 communication.
"""

import hashlib
import importlib.metadata
import json
import os
import statistics
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.models_py.triton_kernels.sparse_mla import (
    glm5_moe_router_pack as candidate,
)


class MetadataTest(unittest.TestCase):
    def test_bad_metadata_never_loads_extension(self):
        with patch.object(candidate, "_prepare_device") as prepare:
            with self.assertRaisesRegex(ValueError, "operands"):
                candidate.fused_router_pack(
                    torch.empty((1, 16)),
                    None,
                    None,
                    activation_out=None,
                    scales_out=None,
                    topk_ids_out=None,
                    topk_weights_out=None,
                )
            prepare.assert_not_called()

    def test_fixed_shape_contract_rejects_before_extension(self):
        for shape in ((0, 6144), (257, 6144), (1, 6143), (6144,)):
            with self.subTest(shape=shape):
                with patch.object(candidate, "_prepare_device") as prepare:
                    with self.assertRaisesRegex(ValueError, "hidden"):
                        candidate.fused_router_pack(
                            torch.empty(shape, dtype=torch.bfloat16),
                            torch.empty((1, 256)),
                            torch.empty(256),
                            **_kwargs(_outputs(1, "cpu")),
                        )
                    prepare.assert_not_called()

    def test_first_use_capture_rejects_before_extension(self):
        with (
            patch.object(torch.cuda, "device", side_effect=lambda _: nullcontext()),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=True),
            patch.object(candidate, "_load_extension") as load,
        ):
            with self.assertRaisesRegex(RuntimeError, "before capture"):
                candidate._prepare_device(-917)
            load.assert_not_called()

    def test_unsupported_architecture_rejects_before_extension(self):
        for capability in ((8, 0), (9, 0), (10, 1), (12, 0)):
            with self.subTest(capability=capability):
                with (
                    patch.object(
                        torch.cuda, "device", side_effect=lambda _: nullcontext()
                    ),
                    patch.object(
                        torch.cuda, "is_current_stream_capturing", return_value=False
                    ),
                    patch.object(
                        torch.cuda, "get_device_capability", return_value=capability
                    ),
                    patch.object(candidate, "_load_extension") as load,
                ):
                    with self.assertRaisesRegex(RuntimeError, "SM100 or SM103"):
                        candidate._prepare_device(-918)
                    load.assert_not_called()

    def test_function_initialization_cached_per_device(self):
        with (
            patch.object(
                torch.cuda, "device", side_effect=lambda _: nullcontext()
            ) as device,
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)),
            patch.object(candidate, "_load_extension") as load,
        ):
            candidate._prepare_device(-919)
            candidate._prepare_device(-919)
            candidate._prepare_device(-920)
            self.assertEqual(load.return_value.initialize.call_count, 2)
            self.assertEqual(
                [call.args[0] for call in device.call_args_list], [-919, -920]
            )

    def test_cpu_operands_rejected_without_cuda_init(self):
        tensors = _outputs(1, "cpu")
        with patch.object(
            torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA init")
        ):
            with self.assertRaisesRegex(ValueError, "CUDA"):
                candidate.fused_router_pack(
                    torch.zeros((1, 6144), dtype=torch.bfloat16),
                    torch.zeros((1, 256)),
                    torch.zeros(256),
                    **_kwargs(tensors),
                )

    def test_disjoint_symmetric_storage_slices_are_allowed(self):
        storage = torch.empty(70)
        tensors = [storage[index * 10 : index * 10 + 10] for index in range(7)]
        candidate._check_output_ranges(tensors)
        for output in range(3, 7):
            with self.subTest(output=output):
                aliased = list(tensors)
                aliased[output] = storage[9:19]
                with self.assertRaisesRegex(ValueError, "overlap"):
                    candidate._check_output_ranges(aliased)
        tensors[4] = tensors[3]
        with self.assertRaisesRegex(ValueError, "overlap"):
            candidate._check_output_ranges(tensors)


def _outputs(rows, device):
    return (
        torch.empty((rows, 6144), dtype=torch.float8_e4m3fn, device=device),
        torch.empty((rows, 48), dtype=torch.int32, device=device),
        torch.empty((rows, 8), dtype=torch.int64, device=device),
        torch.empty((rows, 8), dtype=torch.float32, device=device),
    )


def _kwargs(outputs):
    return dict(
        zip(
            ("activation_out", "scales_out", "topk_ids_out", "topk_weights_out"),
            outputs,
        )
    )


def _assert_bytes(actual, expected, label):
    for name, value, reference in zip(
        ("FP8", "scales", "IDs", "weights"), actual, expected
    ):
        try:
            torch.testing.assert_close(
                value.view(torch.uint8),
                reference.view(torch.uint8),
                rtol=0,
                atol=0,
                msg=f"{label}: {name} must match every byte",
            )
        except AssertionError:
            actual_bytes = value.view(torch.uint8).flatten()
            expected_bytes = reference.view(torch.uint8).flatten()
            offsets = (actual_bytes != expected_bytes).nonzero().flatten()
            first = offsets[:16]
            print(
                "MOE_ROUTER_PACK_MISMATCH "
                + json.dumps(
                    dict(
                        label=label,
                        field=name,
                        count=offsets.numel(),
                        offsets=first.tolist(),
                        actual=actual_bytes[first].tolist(),
                        expected=expected_bytes[first].tolist(),
                        actual_scales=actual[1][0, :4].tolist(),
                        expected_scales=expected[1][0, :4].tolist(),
                    )
                ),
                flush=True,
            )
            raise


class GpuFixture:
    def __init__(self, rows):
        from rtp_llm.models_py.modules import GroupTopK
        from rtp_llm.models_py.modules.glm5_mega_moe.input_packer_triton import (
            fused_pack_mega_moe_inputs,
        )

        self.rows, self.select, self.pack = (
            rows,
            GroupTopK(),
            fused_pack_mega_moe_inputs,
        )
        self.hidden = torch.empty((rows, 6144), device="cuda", dtype=torch.bfloat16)
        self.logits = torch.empty((rows, 256), device="cuda")
        self.bias = torch.empty(256, device="cuda")
        self.ordinary = _outputs(rows, "cuda")
        self.actual = _outputs(rows, "cuda")
        self.intermediate_ids = torch.empty((rows, 8), device="cuda", dtype=torch.int64)
        self.intermediate_weights = torch.empty((rows, 8), device="cuda")

    def reference(self):
        self.select(
            topk_weights=self.intermediate_weights,
            topk_ids=self.intermediate_ids,
            scores=self.logits,
            correction_bias=self.bias,
            n_group=1,
            topk_group=1,
            topk=8,
            renormalize=True,
            routed_scaling_factor=2.5,
        )
        self.pack(
            self.hidden,
            self.intermediate_weights,
            self.intermediate_ids,
            *self.ordinary,
        )
        return self.ordinary

    def fused(self):
        return candidate.fused_router_pack(
            self.hidden, self.logits, self.bias, **_kwargs(self.actual)
        )

    def stage(self, case, seed):
        torch.manual_seed(seed)
        self.hidden.copy_(torch.randn_like(self.hidden))
        self.logits.copy_(torch.randn_like(self.logits).bfloat16().float())
        self.bias.copy_(torch.randn_like(self.bias) * 0.1)
        if case == "random":
            return
        if case.startswith("norm_"):
            if case == "norm_zero":
                self.hidden.zero_()
            elif case == "norm_tiny":
                self.hidden.fill_(1e-20)
            elif case == "norm_boundaries":
                values = torch.tensor(
                    [0.0, -0.0, 448.0, -448.0, 0.875, -0.875, 1e-4, -1e-4],
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                self.hidden.copy_(values.repeat(768).expand(self.rows, -1))
            elif case == "norm_nan":
                self.hidden[:, 1] = float("nan")
            elif case == "norm_all_nan":
                self.hidden.fill_(float("nan"))
            elif case == "norm_nan_payloads":
                bits = torch.tensor(
                    [32704, -64, 32641, -127, 32767, -1, 0, -32768],
                    device="cuda",
                    dtype=torch.int16,
                )
                self.hidden.copy_(bits.view(torch.bfloat16).repeat(768))
            elif case == "norm_inf":
                self.hidden[:, 1:3] = torch.tensor(
                    [float("inf"), -float("inf")], device="cuda"
                )
            return
        self.bias.zero_()
        if case == "exact_ties":
            self.logits.zero_()
        elif case == "lane_ties":
            self.logits.copy_(
                (torch.arange(256, device="cuda") % 7).expand(self.rows, -1)
            )
        elif case == "near_ties":
            self.logits.zero_()
            self.logits[:, :7], self.logits[:, 7], self.logits[:, 8] = 2, 1, 1 + 1 / 512
        elif case == "ulp_ties":
            self.logits.fill_(1)
            self.logits[:, 8:16] = torch.nextafter(
                torch.tensor(1.0, device="cuda"), torch.tensor(2.0, device="cuda")
            )
        elif case == "low_scores":
            self.logits.copy_(-90 + torch.arange(256, device="cuda").float()[None] / 16)
        elif case == "all_zero_scores":
            self.logits.fill_(-float("inf"))
        elif case == "all_one_scores":
            self.logits.fill_(float("inf"))
        elif case == "nan_logits":
            self.logits[:, ::17] = float("nan")
        elif case == "all_nan_logits":
            self.logits.fill_(float("nan"))
        elif case == "few_valid_logits":
            self.logits.fill_(float("nan"))
            self.logits[:, :4] = 1
        elif case == "positive_infinite_logits":
            self.logits[:, ::17] = float("inf")
        elif case == "negative_infinite_logits":
            self.logits[:, ::17] = -float("inf")
        elif case == "nan_bias":
            self.bias[::17] = float("nan")
        elif case == "all_nan_bias":
            self.bias.fill_(float("nan"))
        elif case == "positive_infinite_bias":
            self.bias[0] = float("inf")
        elif case == "negative_infinite_bias":
            self.bias[0] = -float("inf")
        elif case == "group_score_overflow":
            self.bias.fill_(torch.finfo(torch.float32).max)
        else:
            raise ValueError(case)


def gpu_run(benchmark):
    if os.environ.get("RTP_GLM5_MOE_ROUTER_PACK_GPU") != "1":
        raise RuntimeError("Explicit GPU reservation opt-in required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one reserved GPU")
    cases = (
        "random",
        "exact_ties",
        "lane_ties",
        "near_ties",
        "ulp_ties",
        "low_scores",
        "all_zero_scores",
        "all_one_scores",
        "nan_logits",
        "all_nan_logits",
        "few_valid_logits",
        "positive_infinite_logits",
        "negative_infinite_logits",
        "nan_bias",
        "all_nan_bias",
        "positive_infinite_bias",
        "negative_infinite_bias",
        "group_score_overflow",
        "norm_zero",
        "norm_tiny",
        "norm_boundaries",
        "norm_nan",
        "norm_all_nan",
        "norm_nan_payloads",
        "norm_inf",
    )
    count = 0
    for rows in (1, 4, 6, 8, 16, 32, 64, 128, 256):
        fixture = GpuFixture(rows)
        for seed in (1, 817, 1933):
            for case in cases:
                fixture.stage(case, seed)
                original = tuple(
                    x.clone() for x in (fixture.hidden, fixture.logits, fixture.bias)
                )
                fixture.reference()
                fixture.fused()
                _assert_bytes(
                    fixture.actual,
                    fixture.ordinary,
                    f"rows={rows},case={case},seed={seed}",
                )
                for value, old in zip(
                    (fixture.hidden, fixture.logits, fixture.bias), original
                ):
                    torch.testing.assert_close(
                        value.view(torch.uint8), old.view(torch.uint8), rtol=0, atol=0
                    )
                count += 1
        print(f"MOE_ROUTER_PACK_EXACT rows={rows} cases={len(cases)*3}", flush=True)
        fixture.stage("random", 2)
        fixture.fused()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fixture.fused()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            fixture.fused()
        for step, case in enumerate(
            ("random", "exact_ties", "nan_bias", "near_ties", "low_scores", "random")
        ):
            fixture.stage(case, step + 27)
            fixture.reference()
            graph.replay()
            _assert_bytes(
                fixture.actual, fixture.ordinary, f"graph rows={rows},case={case}"
            )
        if benchmark:
            fixture.stage("random", 712)
            graphs = []
            for function in (fixture.reference, fixture.fused):
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        function()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    for _ in range(5):
                        function()
                graphs.append(graph)
            samples = [[], []]
            for index in range(31):
                for path in (0, 1) if index % 2 == 0 else (1, 0):
                    start, end = (
                        torch.cuda.Event(enable_timing=True) for _ in range(2)
                    )
                    start.record()
                    graphs[path].replay()
                    end.record()
                    end.synchronize()
                    samples[path].append(start.elapsed_time(end) * 1000 / 5)
            medians = [statistics.median(x) for x in samples]
            print(
                "MOE_ROUTER_PACK_BENCHMARK "
                + json.dumps(
                    dict(
                        rows=rows,
                        ordinary_us=medians[0],
                        fused_us=medians[1],
                        speedup=medians[0] / medians[1],
                        samples_us=samples,
                        boundary="ordinary GroupTopK+group32 pack only; not full MoE/CMP",
                    )
                ),
                flush=True,
            )
    print(
        f"MOE_ROUTER_PACK_ALL_EXACT eager_cases={count} graph_cases={9*6}", flush=True
    )


def _view_metadata(tensor, buffer):
    return dict(
        shape=list(tensor.shape),
        stride=list(tensor.stride()),
        dtype=str(tensor.dtype),
        contiguous=tensor.is_contiguous(),
        byte_offset=tensor.data_ptr() - buffer.data_ptr(),
        storage_matches_buffer=(
            tensor.untyped_storage().data_ptr() == buffer.untyped_storage().data_ptr()
        ),
        storage_bytes=tensor.untyped_storage().nbytes(),
    )


def _source_provenance():
    helper = Path(candidate.__file__)
    paths = (
        helper,
        helper.with_name("glm5_moe_router_pack_src") / "kernel.cu",
        Path(__file__),
    )
    print(
        "MOE_ROUTER_PACK_SOURCES "
        + json.dumps(
            {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        ),
        flush=True,
    )


def _benchmark_real_consumer(fixture, moe, stream):
    """Both graphs use the exact same weight pointers, input buffer and output."""

    def reference():
        fixture.reference()
        return moe.forward_prepacked(fixture.hidden)

    def fused():
        candidate.fused_router_pack(
            fixture.hidden, fixture.logits, fixture.bias, **_kwargs(fixture.ordinary)
        )
        return moe.forward_prepacked(fixture.hidden)

    fixture.stage("random", 771)
    y_reference = reference().clone()
    y_fused = fused().clone()
    torch.testing.assert_close(
        y_fused.view(torch.uint8), y_reference.view(torch.uint8), rtol=0, atol=0
    )
    graphs = []
    stream.wait_stream(torch.cuda.current_stream())
    for function in (reference, fused):
        with torch.cuda.stream(stream):
            for _ in range(3):
                function()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            function()
        graphs.append(graph)
    samples = [[], []]
    for index in range(31):
        for path in (0, 1) if index % 2 == 0 else (1, 0):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            graphs[path].replay()
            end.record()
            end.synchronize()
            samples[path].append(start.elapsed_time(end) * 1000)
    y_reference = reference().clone()
    y_fused = fused().clone()
    torch.testing.assert_close(
        y_fused.view(torch.uint8), y_reference.view(torch.uint8), rtol=0, atol=0
    )
    medians = [statistics.median(values) for values in samples]
    print(
        "MOE_ROUTER_PACK_REAL_CONSUMER_BENCHMARK "
        + json.dumps(
            dict(
                rows=fixture.rows,
                world_size=1,
                ordinary_us=medians[0],
                fused_us=medians[1],
                speedup=medians[0] / medians[1],
                samples_us=samples,
                same_physical_input_buffer=True,
                same_weight_pointers=True,
                output_byte_equal=True,
                boundary="EP1 GroupTopK+pack+real GLM5MegaMoE; excludes initialization/norm/gate/shared/TP8",
            )
        ),
        flush=True,
    )


def gpu_abi_run(benchmark=False):
    """Real SymmBuffer + GLM5MegaMoE EP1; explicitly NOT an eight-rank test."""
    if os.environ.get("RTP_GLM5_MOE_ROUTER_PACK_GPU") != "1":
        raise RuntimeError("Explicit GPU reservation opt-in required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one reserved GPU")
    import deep_gemm
    import rtp_kernel
    import torch.distributed as dist

    provenance = dict(
        deep_gemm=deep_gemm.__file__,
        deep_gemm_version=importlib.metadata.version("deep_gemm"),
        rtp_kernel=rtp_kernel.__file__,
        rtp_kernel_version=importlib.metadata.version("rtp_kernel"),
        torch=torch.__version__,
    )
    print("MOE_ROUTER_PACK_DEPENDENCIES " + json.dumps(provenance), flush=True)
    if provenance["deep_gemm_version"] != "2.6.1+7232a0c.cu132":
        raise RuntimeError(
            "Real ABI diagnostic requires the repository-locked DeepGEMM wheel"
        )

    from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe import GLM5MegaMoE

    if dist.is_initialized():
        raise RuntimeError("ABI test owns its isolated world-size=1 process group")
    torch.cuda.set_device(0)
    with tempfile.TemporaryDirectory(prefix="glm5-moe-router-pack-ep1-") as directory:
        dist.init_process_group(
            "nccl",
            rank=0,
            world_size=1,
            init_method="file://" + directory + "/rendezvous",
            device_id=torch.device("cuda:0"),
        )
        try:
            torch.manual_seed(9301)
            # Routed EP partitions experts, not their intermediate dimension:
            # use full GLM5 routed width 2048, not shared-expert TP width 256.
            # This diagnostic owns all 256 experts and has no TP/EP collectives.
            moe = GLM5MegaMoE.from_params(
                layer_id=917,
                dim=6144,
                moe_inter_dim=2048,
                n_routed_experts=256,
                n_activated_experts=8,
                ep_size=1,
                ep_rank=0,
                max_tokens_per_rank=256,
            )
            # Real nonzero FP4 weights through the production loading/layout
            # transforms, not a mocked consumer or a substituted matmul.
            w1 = torch.randint(
                -128, 128, (256, 4096, 3072), dtype=torch.int8, device="cuda"
            )
            w2 = torch.randint(
                -128, 128, (256, 6144, 1024), dtype=torch.int8, device="cuda"
            )
            s1 = torch.full((256, 4096, 192), 1 / 256, device="cuda")
            s2 = torch.full((256, 6144, 64), 1 / 256, device="cuda")
            with patch.dict(os.environ, {"GLM5_MEGA_MOE_JIT_WARMUP": "0"}):
                moe.setup_weights_from_fp4(w1, s1, w2, s2)
            del w1, w2, s1, s2
            clone = moe.clone_for_cuda_graph()
            assert clone._mega_buf is not moe._mega_buf
            assert clone._mega_y.data_ptr() != moe._mega_y.data_ptr()
            for name in ("_mega_l1_w", "_mega_l1_sf", "_mega_l2_w", "_mega_l2_sf"):
                assert getattr(clone, name).data_ptr() == getattr(moe, name).data_ptr()
            for label, module in (("original", moe), ("clone", clone)):
                buffer = module._mega_buf
                print(
                    "MOE_ROUTER_PACK_REAL_ABI "
                    + json.dumps(
                        dict(
                            label=label,
                            world_size=1,
                            ep_size=1,
                            routed_ffn_width=2048,
                            dependencies=provenance,
                            capacity=buffer.num_max_tokens_per_rank,
                            allocation_bytes=buffer.buffer.numel(),
                            views={
                                name: _view_metadata(
                                    getattr(buffer, name), buffer.buffer
                                )
                                for name in ("x", "x_sf", "topk_idx", "topk_weights")
                            },
                            caveat="Real DeepGEMM SymmBuffer factory uses torch.empty at world1; no NVLink rendezvous/TP8 proof",
                        )
                    ),
                    flush=True,
                )
            eager_count = graph_count = 0
            for rows in (1, 4, 6, 256):
                fixture = GpuFixture(rows)
                fixture.ordinary = moe.prepacked_input_views(rows)
                fixture.actual = clone.prepacked_input_views(rows)
                for module in (moe, clone):
                    assert module._mega_buf.num_max_tokens_per_rank >= rows
                    for view in module.prepacked_input_views(rows):
                        assert view.is_contiguous()

                def reference():
                    fixture.reference()
                    return moe.forward_prepacked(fixture.hidden)

                def fused():
                    fixture.fused()
                    return clone.forward_prepacked(fixture.hidden)

                for step, case in enumerate(
                    (
                        "random",
                        "exact_ties",
                        "near_ties",
                        "nan_bias",
                        "norm_nan",
                        "random",
                    )
                ):
                    fixture.stage(case, step + 510)
                    tails = []
                    for module in (moe, clone):
                        current = []
                        for name in ("x", "x_sf", "topk_idx", "topk_weights"):
                            tail = getattr(module._mega_buf, name)[rows:]
                            tail.view(torch.uint8).fill_(83)
                            current.append(tail.clone())
                        tails.append(current)
                    y_reference = reference().clone()
                    y_fused = fused().clone()
                    assert torch.isfinite(
                        y_reference
                    ).all(), "EP1 reference must be finite"
                    assert torch.count_nonzero(
                        y_reference
                    ), "EP1 reference must be nontrivial"
                    _assert_bytes(
                        fixture.actual,
                        fixture.ordinary,
                        f"real ABI rows={rows},case={case}",
                    )
                    torch.testing.assert_close(
                        y_fused.view(torch.uint8),
                        y_reference.view(torch.uint8),
                        rtol=0,
                        atol=0,
                        msg=f"real GLM5MegaMoE EP1 output rows={rows},case={case}",
                    )
                    for module, old_tails in zip((moe, clone), tails):
                        for name, old in zip(
                            ("x", "x_sf", "topk_idx", "topk_weights"), old_tails
                        ):
                            torch.testing.assert_close(
                                getattr(module._mega_buf, name)[rows:].view(
                                    torch.uint8
                                ),
                                old.view(torch.uint8),
                                rtol=0,
                                atol=0,
                            )
                    eager_count += 1
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    reference()
                    fused()
                stream.synchronize()
                graphs, outputs = [], []
                for function in (reference, fused):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        output = function()
                    graphs.append(graph)
                    outputs.append(output)
                for step, case in enumerate(
                    ("random", "exact_ties", "nan_bias", "near_ties", "random")
                ):
                    fixture.stage(case, step + 611)
                    # Sequential replay changes inputs at the same addresses;
                    # separate cloned mutable state must not leak across graphs.
                    for graph in graphs:
                        graph.replay()
                    _assert_bytes(
                        fixture.actual,
                        fixture.ordinary,
                        f"real ABI graph rows={rows},case={case}",
                    )
                    torch.testing.assert_close(
                        outputs[1].view(torch.uint8),
                        outputs[0].view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )
                    graph_count += 1
                print(
                    f"MOE_ROUTER_PACK_REAL_CONSUMER_EXACT rows={rows} eager=6 graph=5",
                    flush=True,
                )
                if benchmark and rows in (1, 4, 6):
                    _benchmark_real_consumer(fixture, moe, stream)
            print(
                f"MOE_ROUTER_PACK_REAL_ABI_ALL_EXACT eager_cases={eager_count} graph_cases={graph_count} world_size=1 NOT_TP8",
                flush=True,
            )
        finally:
            torch.cuda.synchronize()
            dist.destroy_process_group()


if __name__ == "__main__":
    if "--gpu" in sys.argv:
        with torch.inference_mode():
            _source_provenance()
            if "--abi" in sys.argv:
                gpu_abi_run("--benchmark" in sys.argv)
                if "--regression" in sys.argv:
                    gpu_run(False)
            else:
                gpu_run("--benchmark" in sys.argv)
    else:
        unittest.main()

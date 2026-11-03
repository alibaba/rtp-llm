"""CPU planning/dispatch checks; these do not claim GPU compilation coverage."""

import ast
import math
import os
import sys
import types
import unittest
from functools import lru_cache, partial
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Sequence
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "bounded_projection_jit_warmup.py").exists() else HERE.parent
PREFIX = "rtp_llm.models_py.modules."
BF16, FP8, SPLITS = "torch.bfloat16", "torch.float8_e4m3fn", (128, 64, 128)
DEPENDENCIES = dict(
    dense=PREFIX + "dsv41.dsv4_kernel_jit_warmup",
    switch="rtp_llm.utils.warmup",
    f16=PREFIX + "factory.linear.impl.cuda.f16_linear",
    factory=PREFIX + "factory.linear.factory",
    weights="rtp_llm.utils.model_weight",
    plan=PREFIX + "factory.attention.mla_prefill_plan",
    kda=PREFIX + "factory.attention.cuda_kda_impl",
    fla="fla.modules.l2norm",
    mla=PREFIX + "kimi_k3.mla_jit_warmup",
    directed=PREFIX + "kimi_k3.bounded_projection_jit_warmup",
)
WEIGHT_KEYS = NS(mla_kv_b_w="kv_up", mla_kv_b_s="kv_scale")
LAYOUT = dict(n_value=3072, k_value=512, kind="dg98_bf16_k_mn", num_sms=152)


def source_module(path, names=None, **bindings):
    tree = ast.parse(path.read_text())
    if names is not None:
        names = tuple(names)
        tree.body = [
            node
            for node in tree.body
            if getattr(node, "name", "").startswith(names)
            or isinstance(node, ast.Assign)
            and any(
                getattr(target, "id", "").startswith(names) for target in node.targets
            )
        ]
    result = types.ModuleType(path.stem)
    result.__file__ = str(path)
    if names is not None:
        result.__dict__.update(lru_cache=lru_cache, math=math, Sequence=Sequence)
    result.__dict__.update(bindings)
    exec(compile(tree, str(path), "exec"), result.__dict__)
    return result


def load_warmup():
    return source_module(ROOT / "bounded_projection_jit_warmup.py")


def load_sampling():
    names = """
    _DENSE_GEMM_FALLBACK_M_GRID _DENSE_GEMM_HEURISTIC_SCAN_LIMIT
    _SM100_DENSE_BLOCK_M_CANDIDATES _SM100_DENSE_BLOCK_N_CANDIDATES
    _sm100 _ceil_div _align _candidate_dense_gemm _generate_dense_gemm _dense_gemm_m_grid
    """.split()
    return source_module(ROOT.parent / "dsv41/dsv4_kernel_jit_warmup.py", names)


def dependencies(**items):
    return mock.patch.dict(
        sys.modules, {DEPENDENCIES.get(key, key): value for key, value in items.items()}
    )


class Tensor:
    def __init__(self, shape, strides=None, dtype=FP8, device="cuda:0"):
        self.shape, self._stride = shape, strides or (shape[1], 1)
        self.dtype = dtype
        self.device = NS(type=device.split(":")[0], name=device)

    def stride(self, dim=None):
        return self._stride if dim is None else self._stride[dim]


class CudaFp8GEMMLinear:
    def __init__(self, weight=None, scales=None, supported=True):
        self.weight = self.raw_weight = weight or Tensor((128, 512))
        self.weight_scales = scales or Tensor((128, 1), (1, 128), "torch.int32")
        self.N, self.K = self.weight.shape
        self.scale_ue8m0, self._flashinfer_linear = True, None
        self.supported, self.rows, self.packed, self.probes = supported, [], [], []

    def __call__(self, hidden):
        self.rows.append(hidden.shape[0])

    def supports_skip_head_mid(self, hidden, splits):
        self.probes.append((hidden.shape, splits))
        return self.supported

    def forward_skip_head_mid(self, hidden, splits):
        self.packed.append((hidden.shape[0], splits))


class CudaF16Linear(CudaFp8GEMMLinear):
    def __init__(self, weight, supported=True):
        view = Tensor(
            weight.shape, (1, weight.shape[0]), weight.dtype, weight.device.name
        )
        super().__init__(view, supported=supported)
        self.raw_weight = weight


class Model(NS):
    def __init__(self, page=128, cache_dtype=BF16, **config):
        cache = NS(seq_size_per_block=page, kv_cache_base=NS(dtype=cache_dtype))
        super().__init__(
            _max_moe_tokens=65536,
            parallelism_config=NS(tp_size=8),
            config=NS(
                max_seq_len=1048576,
                gen_num_per_cycle=4,
                attn_config=NS(nope_head_dim=128, rope_head_dim=64, v_head_dim=128),
            ),
            kv_cache=NS(get_layer_cache_groups=lambda _: [cache]),
            layers=[],
            weight=NS(weights=[]),
        )
        self.config.__dict__.update(config)


class KimiK3KDA:
    def __init__(self, heads):
        self.prefill = NS(local_num_k_heads=heads, head_k_dim=128)
        self.prefill.conv_weights = Tensor((128, 4), dtype=BF16)


def metadata_torch(free=1 << 50, capability=10):
    torch = mock.MagicMock(bfloat16=BF16, float32="torch.float32")
    torch.inference_mode.return_value = lambda function: function
    allocate = lambda shape, **_: NS(shape=shape)
    torch.zeros.side_effect = torch.empty.side_effect = allocate
    torch.empty_like.side_effect = lambda tensor: allocate(tensor.shape)
    torch.cuda.is_available.return_value = True
    torch.cuda.is_current_stream_capturing.return_value = False
    torch.cuda.mem_get_info.return_value = (free, 1 << 50)
    torch.cuda.get_device_capability.return_value = (capability, 3)
    return torch


class BoundedProjectionWarmupTest(unittest.TestCase):
    def setUp(self):
        self.warmup = load_warmup()
        self.run_warmup = self.warmup.warmup_kimi_k3_projection_jit

    def test_bf16_layout_coverage_sampling_and_version_layout_fallbacks(self):
        selector = load_sampling()
        grid = partial(selector._generate_dense_gemm_warmup_m_grid, **LAYOUT)
        signature = partial(selector._sm100_dense_layout_signature, **LAYOUT)
        layouts = {signature(m_value=m) for m in grid(max_m=838860)}
        self.assertTrue(
            {(1, 256, 128, 64, 1, 2, 6), (1, 160, 128, 64, 1, 2, 8)}
            <= {s[:6] + (s[13],) for s in layouts}
        )
        sample = partial(
            self.warmup._sample_m_values, n=3072, k=512, fp8=False, num_sms=152
        )
        with dependencies(dense=selector):
            with mock.patch(
                "importlib.metadata.version", return_value="2.8.2+98a672e.cu132"
            ):
                values = sample(838859, k_major=True)
            expected = set(grid(max_m=838859)) | set(
                selector._dense_gemm_m_grid(838859)
            )
            self.assertEqual(values, tuple(sorted(expected)))
            self.assertEqual(values[-1], 838859)
            self.assertNotEqual(grid(max_m=838859), grid(max_m=838859, num_sms=1))
            for version, k_major in (
                ("2.6.1+d029add", True),
                ("2.8.2+98a672e.cu132", False),
            ):
                with self.subTest(version=version, k_major=k_major), mock.patch(
                    "importlib.metadata.version",
                    return_value=version,
                ):
                    self.assertEqual(
                        sample(1003, k_major=k_major),
                        tuple(selector._dense_gemm_m_grid(1003)),
                    )

    def test_collection_preserves_outer_wrappers_lazy_weights_and_scale_layouts(self):
        first, other = CudaFp8GEMMLinear(), CudaFp8GEMMLinear()
        other.weight_scales._stride = (1, 256)
        cpu = CudaFp8GEMMLinear(Tensor((128, 512), device="cpu"))
        outer = [
            ("a", first),
            ("nested", NS(weight=first.weight)),
            ("b", CudaFp8GEMMLinear()),
            ("c", other),
            ("cpu", cpu),
        ]
        collected = self.warmup._fp8_projections(NS(named_modules=lambda: outer))
        self.assertEqual([name for name, _ in collected], ["a", "c"])
        raw = Tensor((512, 3072), dtype=BF16)
        quantized = Tensor((3072, 512))
        scales = [
            Tensor((3072, 1), (1, stride), "torch.int32") for stride in (3072, 6144)
        ]
        quant = object()

        def create(weights, key, scale, bias, config):
            self.assertIsNone(bias)
            self.assertIs(config, quant)
            return CudaFp8GEMMLinear(weights[key], weights[scale])

        model = Model(attention_projection_quant_config=quant)
        model.weight.weights = [{"kv_up": raw}] * 2 + [
            {"kv_up": quantized, "kv_scale": scale}
            for scale in (scales[0], scales[0], scales[1])
        ]
        attention = type("KimiK3MLA", (), {})()
        model.layers = [NS(index=i, attention=attention) for i in range(5)]
        with dependencies(
            f16=NS(CudaF16Linear=CudaF16Linear),
            factory=NS(LinearFactory=NS(create_linear_from_weights=create)),
            weights=NS(W=WEIGHT_KEYS),
        ):
            specs = list(self.warmup._kv_up_projections(model))
        self.assertEqual(len(specs), 3)
        self.assertIs(specs[0][1].raw_weight, raw)
        self.assertEqual(specs[0][2], SPLITS)
        self.assertIs(specs[1][1].weight, quantized)
        self.assertIs(specs[1][1].weight_scales, scales[0])
        self.assertIs(specs[2][1].weight_scales, scales[1])

    def _prepare(self, budget, dtype, seq, batch, fp8=False, supported=True, **opts):
        weight = Tensor((3072 if fp8 else 1024, 512), dtype=FP8 if fp8 else BF16)
        linear = CudaF16Linear if opts.get("native") else CudaFp8GEMMLinear
        projection = linear(weight, supported=supported)
        model = Model(cache_dtype=dtype, max_seq_len=seq)
        sample = mock.Mock(side_effect=lambda maximum, *_, **__: (1, maximum))

        bf16 = mock.Mock(wraps=self.warmup._warmup_bf16)
        plan = source_module(
            ROOT.parent / "factory/attention/mla_prefill_plan.py",
            ["plan_expanded_kv_capacity"],
        )
        dense = NS(
            _get_deep_gemm_num_sms=lambda *_: 152,
            _release_cuda_cache=lambda *_: None,
            _run_deepgemm_warmup_launches_serialized=lambda _, run: run(),
        )
        with dependencies(
            torch=metadata_torch(opts.get("free", 1 << 50), opts.get("capability", 10)),
            plan=plan,
            dense=dense,
            switch=NS(model_warm_up_enabled=lambda: True),
        ), mock.patch.dict(
            os.environ, {"RTP_MLA_PREFILL_EXPANDED_KV_BUDGET_GIB": budget}
        ), mock.patch.multiple(
            self.warmup,
            _fp8_projections=lambda _: (),
            _kv_up_projections=lambda _: [(0, projection, SPLITS)],
            _fla_l2norm_specs=lambda *_: (),
            _sample_m_values=sample,
            _warmup_bf16=bf16,
        ):
            self.run_warmup(
                model, NS(is_decode_role=False, max_context_batch_size=batch)
            )
        self.assertEqual(projection.probes[0], ((1, 512), SPLITS))
        self.assertEqual(bf16.called, not fp8 and supported)
        samples = [1, sample.call_args.args[0]] if sample.called else []
        packed = [(row, SPLITS) for row in samples] if supported else []
        outer = samples if fp8 and not supported else []
        self.assertEqual((projection.rows, projection.packed), (outer, packed))
        observed = sample.call_args.kwargs if sample.called else {}
        return (samples[-1] if samples else None), observed

    def test_scheduler_budget_precision_and_backend_bounds(self):
        model = Model()
        for decode, expected in ((False, 65536), (True, 80)):
            model._max_moe_tokens = expected
            self.assertEqual(
                self.warmup._projection_rows(model, NS(is_decode_role=decode)), expected
            )
        # Budget, cache dtype, max sequence, batch, FP8 projection, skip support, max M.
        cases = {
            "budget": ("1.0", BF16, 1000000, 2, False, True, 419430),
            "fp8_skip": ("6.0", FP8, 1048576, 1, True, True, 559240),
            "fp8_fallback": ("6.0", FP8, 1048576, 1, True, False, 559240),
            "cache_precision": ("1.0", FP8, 1000000, 2, False, True, 279620),
            "zero_budget": ("0", BF16, 500, 2, False, True, 1007),
            "request_clamp": ("1.0", BF16, 1000, 3, False, True, 3007),
        }
        for case, (*args, expected) in cases.items():
            with self.subTest(case=case):
                self.assertEqual(self._prepare(*args)[0], expected)
        self.assertIsNone(
            self._prepare("unparsed", BF16, 1000000, 2, supported=False, free=0)[0]
        )
        for native, capability, k_major, sms in (
            (True, 10, True, 152),
            (False, 10, False, 1),
            (True, 9, False, 1),
        ):
            with self.subTest(native=native, capability=capability):
                observed = self._prepare(
                    "6.0", BF16, 1048576, 1, native=native, capability=capability
                )[1]
                self.assertEqual(
                    (observed["k_major"], observed["num_sms"]), (k_major, sms)
                )
        with self.assertRaisesRegex(RuntimeError, "needs at least"):
            self._prepare("6.0", FP8, 1048576, 1, True, False, free=0)

    def test_launch_routes_and_scratch_memory_guards(self):
        with dependencies(torch=metadata_torch()):
            for fp8, n, expected in ((True, 128, 14464), (False, 320, 13312)):
                self.assertEqual(
                    self.warmup._projection_scratch_bytes(8, 512, n, fp8=fp8), expected
                )
            projection = CudaFp8GEMMLinear()
            self.warmup._warmup_fp8(projection, (1, 24, 65536))
            self.assertEqual(projection.rows, [1, 24, 65536])

    def test_fla_buckets_and_fixed_scratch_launch_metadata(self):
        for heads, page, rows, maximum in (
            ((12,), 4096, 65536, 12),
            ((96,), 128, 65536, 96),
            ((12, 16), 4096, 65536, 16),
            ((12,), None, 65536, 12),
            ((12,), 128, 65537, 13),
            ((12,), 128, 5461, 2),
        ):
            model = Model(page=page)
            if page is None:
                model.kv_cache = None
            model._max_moe_tokens = (rows + 7) // 8 * 8
            model.layers = [
                NS(index=i, attention=KimiK3KDA(count)) for i, count in enumerate(heads)
            ]
            # Planning must not import the old backend checkpoint constant.
            with self.subTest(heads=heads, page=page, rows=rows), dependencies(
                kda=None
            ):
                max_rows = self.warmup._projection_rows(model, NS(is_decode_role=False))
                specs = list(self.warmup._fla_l2norm_specs(model, max_rows))
                self.assertEqual(
                    [buckets for _, buckets in specs], [tuple(range(1, maximum + 1))]
                )
        kernel = mock.MagicMock()
        with dependencies(torch=metadata_torch(), fla=NS(l2norm_fwd_kernel=kernel)):
            self.warmup._warmup_fla("cuda:0", (1, 12, 64))
        self.assertEqual(kernel.__getitem__.call_args_list, [mock.call((1,))] * 3)
        calls = kernel.__getitem__.return_value.call_args_list
        self.assertEqual([call.kwargs["NB"] for call in calls], [1, 12, 64])
        keys = "T D BD BT num_warps num_stages num_ctas".split()
        for call in calls:
            args = call.kwargs
            self.assertEqual((args["x"].shape, args["rstd"].shape), ((8, 128), (8,)))
            self.assertEqual(
                tuple(args[key] for key in keys), (8, 128, 128, 8, 1, 3, 1)
            )

    def test_fla_device_bounds_and_ineligible_layers(self):
        model = Model()
        first, second, other = KimiK3KDA(12), KimiK3KDA(16), KimiK3KDA(8)
        other.prefill.conv_weights = Tensor((128, 4), device="cuda:1")
        wrong_dim, cpu = KimiK3KDA(96), KimiK3KDA(96)
        wrong_dim.prefill.head_k_dim = 64
        cpu.prefill.conv_weights = Tensor((128, 4), device="cpu")
        model.layers = [
            NS(index=i, attention=attention)
            for i, attention in enumerate(
                (first, second, other, wrong_dim, cpu, KimiK3KDA(0), NS())
            )
        ]
        with dependencies(kda=None):
            specs = list(self.warmup._fla_l2norm_specs(model, 65536))
            self.assertEqual(
                [(device.name, buckets) for device, buckets in specs],
                [
                    ("cuda:0", tuple(range(1, 17))),
                    ("cuda:1", tuple(range(1, 9))),
                ],
            )
            self.assertEqual(list(self.warmup._fla_l2norm_specs(model, 0)), [])

    def test_disabled_or_no_cuda_does_no_dependency_work(self):
        torch = NS(cuda=NS(is_available=lambda: False))
        for enabled in (False, True):
            with self.subTest(enabled=enabled), dependencies(
                switch=NS(model_warm_up_enabled=lambda: enabled),
                torch=torch if enabled else None,
                dense=None,
                fla=None,
            ):
                self.run_warmup(object(), object())

    def test_directed_failure_prevents_successful_startup(self):
        def fail(*_):
            raise RuntimeError("directed compile failed")

        noop = lambda *_: None
        with dependencies(
            mla=NS(warmup_kimi_k3_mla_jit=noop),
            directed=NS(warmup_kimi_k3_projection_jit=fail),
        ):
            startup = source_module(
                ROOT / "kernel_jit_warmup.py",
                ["warmup_kimi_k3_kernel_jit"],
                torch=metadata_torch(),
                model_warm_up_enabled=lambda: True,
                _warmup_replay=noop,
            )
            with self.assertRaisesRegex(RuntimeError, "directed compile failed"):
                startup.warmup_kimi_k3_kernel_jit(object(), object())


if __name__ == "__main__":
    unittest.main()

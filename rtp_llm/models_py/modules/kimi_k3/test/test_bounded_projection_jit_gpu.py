"""Real CUDA wrapper/key contracts with independent probes and isolated caches."""

import contextlib
import hashlib
import json
import os
import sys
import tempfile
import unittest
from importlib import import_module
from pathlib import Path


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def independent_probe_m_values(maximum, kind, extra=()):
    """Probe policy intentionally independent of the warmup representative grid."""
    require(maximum > 0, "positive probe upper bound required")
    if kind == "synthetic_unit_fixture":
        return tuple(range(1, maximum + 1))
    values = {1, maximum, max(1, maximum - 1)}
    power = 1
    while power <= maximum:
        values.update(v for v in (power - 1, power, power + 1) if 0 < v <= maximum)
        power *= 2
    for value in extra:
        require(type(value) is int and 0 < value <= maximum, "invalid explicit probe")
        values.add(value)
    return tuple(sorted(values))


@contextlib.contextmanager
def triton_keys():
    from triton import knobs

    previous, keys = knobs.runtime.jit_cache_hook, set()

    def listen(**event):
        function = event["fn"].jit_function
        keys.add((function.__module__, function.__name__, str(event["key"])))
        if previous is not None:
            return previous(**event)

    knobs.runtime.jit_cache_hook = listen
    try:
        yield keys
    finally:
        knobs.runtime.jit_cache_hook = previous


class BoundedProjectionGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(os.environ["RTP_K3_JIT_TEST_MANIFEST"])
        cls.manifest = json.loads(source.read_text())
        prov, geo = cls.manifest["geometry_provenance"], cls.manifest["geometry"]
        kind = cls.kind = prov["kind"]
        synthetic = kind == "synthetic_unit_fixture"
        require(kind in ("synthetic_unit_fixture", "model_derived_geometry"), "kind")
        require(prov.get("description") and prov.get("public_source_sha"), "source")
        require(bool(prov.get("model_config_sha256")) == (not synthetic), "config")
        families = ("fp8", "fp8_kv_up", "bf16_kv_up", "fla")
        na = cls.manifest.get("not_applicable_families", {})
        for family in families:
            require(family in geo, family)
            require(geo[family] or (not synthetic and na.get(family)), family)
        require(any(geo[f] for f in families), "one GPU family required")
        cls.temporary = tempfile.TemporaryDirectory(prefix="k3-directed-gpu-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.cache = Path(cls.temporary.name)
        imports = ("deep_gemm", "fla", "rtp_kernel", "rtp_ops")
        if any(name in sys.modules for name in imports):
            raise RuntimeError("dependency overlay must precede wrapper imports")
        caches = {"DG_JIT_CACHE_DIR": "dg", "TRITON_CACHE_DIR": "triton"}
        for name, directory in caches.items():
            os.environ[name] = str(cls.cache / directory)
        deps = cls.manifest["dependencies"]
        wheels = ("deep-gemm", "flash-linear-attention", "rtp-kernel")
        for name in (*wheels, "cutlass-site"):
            root = Path(deps[name]["root"]).resolve()
            require(root.is_dir(), f"dependency overlay: {name}")
            sys.path.insert(0, str(root))
        dsl = Path(deps["cutlass-site"]["root"]) / "nvidia_cutlass_dsl" / "dsl_packages"
        require(dsl.is_dir(), "Cutlass DSL overlay")
        sys.path.insert(0, str(dsl.resolve()))
        import torch as t

        require(
            t.cuda.is_available() and t.cuda.get_device_capability()[0] == 10,
            "real SM10 CUDA required",
        )
        cls.torch, cls.device = t, t.device("cuda")
        cls.args = dict(device=cls.device, dtype=t.bfloat16)
        import deep_gemm
        from fla.modules import l2norm
        from rtp_llm.models_py.modules.kimi_k3 import (
            bounded_projection_jit_warmup as warmup,
        )

        cls.warmup, cls.l2norm, cls.dg = warmup, l2norm, deep_gemm
        require(not list(cls.cache.rglob("kernel.cubin")), "DG process must start cold")
        cls.evidence = dict(
            checks=[],
            geometry_provenance=prov,
            cache_root=str(cls.cache),
            gpu=t.cuda.get_device_name(),
        )
        output = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", cls.temporary.name)
        cls.evidence_path = Path(output) / "k3_directed_gpu_evidence.json"
        cls.addClassCleanup(cls.write_evidence)

    @classmethod
    def dg_keys(cls):
        # DG98 exposes no listener: private source/cubin paths identify compile keys.
        return {
            (str(p.parent.relative_to(cls.cache)), sha256(p.parent / "kernel.cu"))
            for p in cls.cache.rglob("kernel.cubin")
        }

    @classmethod
    def write_evidence(cls):
        cls.evidence_path.write_text(json.dumps(cls.evidence, indent=2, sort_keys=True))

    def sampled_rows(self, rows):
        t = self.torch
        if rows <= 256:
            return t.arange(rows, device=self.device)
        return t.linspace(0, rows - 1, 17, device=self.device).long().unique()

    def family_items(self, family):
        items = self.manifest["geometry"][family]
        if not items:
            reason = self.manifest["not_applicable_families"][family]
            self.skipTest(f"{family} not applicable: {reason}")
        return items

    def bf16_values(self, *shape):
        return self.torch.empty(shape, **self.args).random_(-1, 2)

    def projection(self, n, k, fp8):
        t = self.torch
        name, wrapper = "fp8_gemm_linear", "CudaFp8GEMMLinear"
        if not fp8:
            name, wrapper = "f16_linear", "CudaF16Linear"
        namespace = "rtp_llm.models_py.modules.factory.linear.impl.cuda"
        factory = getattr(import_module(f"{namespace}.{name}"), wrapper)
        if not fp8:
            weight = self.bf16_values(k, n)
            return factory(weight), weight
        weight = self.bf16_values(n, k).to(t.float8_e4m3fn)
        stride = (n + 3) // 4 * 4
        scales = t.empty_strided(
            (n, (k + 511) // 512), (1, stride), device=self.device, dtype=t.int32
        ).fill_(0x7F7F7F7F)
        return factory(weight=weight, weight_scales=scales), weight

    def close(self, actual, expected, rtol=0.015625, atol=0.03125):
        self.torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    def call(self, projection, hidden, splits, output=None):
        if splits:
            if output is None:
                return projection.forward_skip_head_mid(hidden, splits)
            return projection.forward_skip_head_mid(hidden, splits, output=output)
        return projection(hidden)

    def cold_control(self, family, item, projection, splits):
        before = self.dg_keys()
        unprepared = self.torch.ones((1, item["k"]), **self.args)
        self.call(projection, unprepared, splits)
        self.torch.cuda.synchronize()
        observed = self.dg_keys() - before
        del unprepared
        if family == "fp8_kv_up":
            ordinary_done = any(
                c["kind"] == "fp8"
                and all(c["geometry"][key] == item[key] for key in ("n", "k"))
                for c in self.evidence["checks"]
            )
            self.assertTrue(observed or (not splits and ordinary_done))
        else:
            self.assertTrue(observed, "cold actual call must create observable DG keys")

    def check_projection_family(self, family):
        t, w, dg = self.torch, self.warmup, self.dg
        fp8, kv = family != "bf16_kv_up", family != "fp8"
        for index, item in enumerate(self.family_items(family)):
            n, k = item["n"], item["k"]
            splits = tuple(item["head_splits"]) if kv else ()
            heads = n // (splits[0] + splits[2]) if kv else 0
            projection, weight = self.projection(n, k, fp8)
            probe = t.zeros((1, k), **self.args) if kv else None
            supported = kv and projection.supports_skip_head_mid(probe, splits)
            del probe
            if kv:
                expected = item["expected_skip_head_mid"] if fp8 else True
                self.assertIs(type(expected), bool)
                self.assertEqual(supported, expected)
            active_splits = splits if supported else ()
            if fp8:
                k_major = not kv or (not supported and weight.stride(-1) == 1)
            else:
                k_major = projection.weight.T.is_contiguous()
            max_m = item["max_m"]
            mode = dict(fp8=fp8, k_major=k_major, num_sms=dg.get_num_sms())
            samples = w._sample_m_values(max_m, n, k, **mode)
            self.assertTrue(samples)
            extra = range(1, min(max_m, 32768) + 1, 127) if not fp8 else ()
            extra = (*item.get("probe_m", ()), *extra)
            probes = independent_probe_m_values(max_m, self.kind, extra)
            width = heads * sum(splits) if supported else n
            w._check_projection_memory(
                self.device,
                w._projection_scratch_bytes(max(samples), k, width, fp8=fp8),
            )
            if index == 0:
                self.cold_control(family, item, projection, active_splits)
            with triton_keys() if fp8 else contextlib.nullcontext(set()) as keys:
                if fp8:
                    options = dict(head_splits=splits) if kv else {}
                    w._warmup_fp8(projection, samples, **options)
                else:
                    self.assertTrue(w._warmup_bf16(projection, splits, samples))
                t.cuda.synchronize()
                prepared, prepared_triton = self.dg_keys(), set(keys)
                self.assertTrue(prepared)
                for rows in probes:
                    hidden = self.bf16_values(rows, k)
                    output = None
                    if supported:
                        output = t.full((rows, width), float("nan"), **self.args)
                    actual = self.call(projection, hidden, active_splits, output)
                    selected = self.sampled_rows(rows)
                    expected = hidden[selected].float() @ (
                        weight.float().T if fp8 else weight.float()
                    )
                    if supported:
                        view = output[selected].reshape(-1, heads, sum(splits))
                        ref = expected.reshape(-1, heads, splits[0] + splits[2])
                        ref = ref.to(t.bfloat16)
                        for region in (slice(None, splits[0]), slice(-splits[2], None)):
                            self.close(view[..., region], ref[..., region])
                        gap = view[..., splits[0] : splits[0] + splits[1]]
                        self.assertTrue(t.isnan(gap).all())
                        del view, ref, gap
                    else:
                        self.close(actual[selected], expected.to(t.bfloat16))
                    del hidden, actual, expected, output
                t.cuda.synchronize()
                observed = self.dg_keys()
                self.assertEqual(observed, prepared)
                if fp8:
                    self.assertEqual(keys, prepared_triton)
            self.evidence["checks"].append(
                dict(kind=family, geometry=item, probes=len(probes), numeric="passed")
            )
            del projection, weight
            t.cuda.empty_cache()

    def test_fp8_actual_outer_wrapper_preserves_sampled_keys_and_values(self):
        self.check_projection_family("fp8")

    def test_bf16_actual_skip_head_wrapper_preserves_sampled_keys_and_values(self):
        self.check_projection_family("bf16_kv_up")

    def test_fp8_kv_up_actual_backend_preserves_independent_probe_keys_and_values(self):
        self.check_projection_family("fp8_kv_up")

    def test_tiny_fla_preparation_covers_real_wrapper_sampled_nb_keys_and_values(self):
        t = self.torch
        for item in self.family_items("fla"):
            self.l2norm.l2norm_fwd_kernel.device_caches.clear()
            heads, maximum = item["heads"], item["max_call_rows"]
            buckets = tuple(range(1, (heads * maximum + 65535) // 65536 + 1))
            with triton_keys() as keys:
                self.warmup._warmup_fla(self.device, buckets)
                t.cuda.synchronize()
                prepared = set(keys)
                self.assertEqual(len(prepared), len(buckets), "one compile key per NB")
                for nb in buckets:
                    first = (nb - 1) * 65536 // heads + 1
                    last = min(nb * 65536 // heads, maximum)
                    for rows in sorted({first, last}):
                        self.assertGreaterEqual(rows, 1)
                        self.assertEqual((rows * heads + 65535) // 65536, nb)
                        self.warmup._check_projection_memory(
                            self.device, rows * heads * (128 * 4 + 4)
                        )
                        shape = (1, rows, heads, 128)
                        source = t.empty(shape, **self.args).uniform_(-1, 1)
                        actual, rstd = self.l2norm.l2norm_fwd(source)
                        selected = self.sampled_rows(rows * heads)
                        values = source.reshape(-1, 128)[selected].float()
                        inverse = (values.square().sum(-1) + 1e-6).rsqrt()
                        expected = (values * inverse[:, None]).to(t.bfloat16)
                        actual = actual.reshape(-1, 128)[selected]
                        self.close(actual, expected, atol=0.00390625)
                        self.close(rstd.flatten()[selected], inverse, 1e-5, 1e-6)
                        del source, actual, rstd, values, expected
                t.cuda.synchronize()
                self.assertEqual(keys, prepared)
            self.evidence["checks"].append(
                dict(kind="fla", geometry=item, numeric="passed")
            )


if __name__ == "__main__":
    unittest.main()

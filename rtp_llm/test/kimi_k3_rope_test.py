"""K3 fused RoPE correctness, JIT preparation, and compiled-cache reuse."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

RESULT_PREFIX = "K3_ROPE_JIT_RESULT="


def _reference_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rotated = pairs * freqs[:, None, :]
    return torch.view_as_real(rotated).flatten(-2).to(x.dtype)


class KimiK3RopeCpuTest(unittest.TestCase):
    def test_k3_vit_warmup_runs_after_weights_load(self) -> None:
        from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
            BaseMultiModalMixin,
        )
        from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_mixin import (
            KimiK3Mixin,
        )
        from rtp_llm.multimodal.multimodal_mixins.kimi_k3 import kimi_k3_rope_triton

        events = []
        mm_part = object()

        def load_weights(mixin):
            mixin.mm_part = mm_part
            events.append("weights_loaded")

        def warmup(part):
            self.assertIs(part, mm_part)
            events.append("warmup")

        with (
            mock.patch.object(BaseMultiModalMixin, "__init__", load_weights),
            mock.patch.object(
                kimi_k3_rope_triton, "warmup_kimi_k3_vit_rope", side_effect=warmup
            ),
        ):
            KimiK3Mixin()
        self.assertEqual(events, ["weights_loaded", "warmup"])

    def test_unsupported_cpu_input_falls_back(self) -> None:
        from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_rope_triton import (
            maybe_fused_apply_rope,
        )

        q = torch.zeros(2, 1, 8)
        freqs = torch.ones(2, 4, dtype=torch.complex64)
        self.assertIsNone(maybe_fused_apply_rope(q, q, freqs))


class KimiK3RopeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "kimi_k3_rope_test requires a CUDA worker; check GPU routing"
            )

    def test_fused_qk_matches_complex_rotation(self) -> None:
        from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_rope_triton import (
            maybe_fused_apply_rope,
        )

        torch.manual_seed(11)
        for length, heads, width in ((1, 1, 8), (64, 12, 128), (257, 5, 64)):
            with self.subTest(shape=(length, heads, width)):
                q = torch.randn(
                    length, heads, width, device="cuda", dtype=torch.bfloat16
                )
                k = torch.randn_like(q)
                angles = torch.randn(length, width // 2, device="cuda")
                freqs = torch.polar(torch.ones_like(angles), angles)
                fused = maybe_fused_apply_rope(q, k, freqs)
                self.assertIsNotNone(fused)
                q_out, k_out = fused
                torch.testing.assert_close(
                    q_out, _reference_rope(q, freqs), rtol=0.02, atol=0.02
                )
                torch.testing.assert_close(
                    k_out, _reference_rope(k, freqs), rtol=0.02, atol=0.02
                )


def run_rope(head_dim: int) -> None:
    import triton

    from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_rope_triton import (
        maybe_fused_apply_rope,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("K3 vision RoPE JIT probe requires CUDA")
    torch.manual_seed(29)
    qkv = torch.randn(7, 3, 5, head_dim, device="cuda", dtype=torch.bfloat16)
    q, k, _ = qkv.unbind(1)
    assert q.stride(0) != q.numel() // q.shape[0]
    angles = torch.randn(7, head_dim // 2, device="cuda")
    freqs = torch.polar(torch.ones_like(angles), angles)
    hits = []
    with triton.knobs.compilation.scope():
        triton.knobs.compilation.listener = lambda *, cache_hit, **_: hits.append(
            cache_hit
        )
        result = maybe_fused_apply_rope(q, k, freqs)
    if result is None or not hits:
        raise AssertionError("the fused K3 RoPE kernel did not launch")
    for actual, source in zip(result, (q, k)):
        torch.testing.assert_close(
            actual,
            _reference_rope(source, freqs).to(actual.dtype),
            rtol=0.02,
            atol=0.02,
        )
    torch.cuda.synchronize()
    print(RESULT_PREFIX + json.dumps({"compilations": hits.count(False)}))


def run_warmup_rope(dtype_name: str) -> None:
    import triton

    from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_rope_triton import (
        maybe_fused_apply_rope,
        warmup_kimi_k3_vit_rope,
    )
    from rtp_llm.utils.warmup import configure_warmup

    if not torch.cuda.is_available():
        raise RuntimeError("K3 vision RoPE JIT probe requires CUDA")
    configure_warmup(True, True)
    torch.manual_seed(31)
    heads, head_dim = 12, 128
    dtype = getattr(torch, dtype_name)
    # Match MoonViT's real projection and view/unbind layout without loading a
    # vision checkpoint or running attention over a large image.
    layer = torch.nn.Module()
    layer.num_heads, layer.head_dim = heads, head_dim
    layer.wqkv = torch.nn.Linear(
        heads * head_dim, 3 * heads * head_dim, device="cuda", dtype=dtype
    )
    hits = []
    with torch.inference_mode(), triton.knobs.compilation.scope():
        triton.knobs.compilation.listener = lambda *, cache_hit, **_: hits.append(
            cache_hit
        )
        warmup_kimi_k3_vit_rope(SimpleNamespace(vision_tower=layer))
        warmup_compilations = hits.count(False)
        requests = []
        for tokens in (7, 128, 257):
            hidden = torch.randn((tokens, heads * head_dim), device="cuda", dtype=dtype)
            qkv = layer.wqkv(hidden).view(tokens, 3, heads, head_dim)
            q, k, _ = qkv.unbind(1)
            angles = torch.randn((tokens, head_dim // 2), device="cuda")
            freqs = torch.polar(torch.ones_like(angles), angles)
            hits.clear()
            result = maybe_fused_apply_rope(q, k, freqs)
            if result is None:
                raise AssertionError("the warmed K3 RoPE layout fell back to eager")
            for actual, source in zip(result, (q, k)):
                torch.testing.assert_close(
                    actual,
                    _reference_rope(source, freqs).to(dtype),
                    rtol=0.02,
                    atol=0.02,
                )
            requests.append({"tokens": tokens, "compilations": hits.count(False)})
    torch.cuda.synchronize()
    print(
        RESULT_PREFIX
        + json.dumps({"warmup_compilations": warmup_compilations, "requests": requests})
    )


class KimiK3RopeJitCacheTest(unittest.TestCase):

    def test_warmup_reuses_production_layout_across_token_counts(self) -> None:
        for dtype in ("bfloat16", "float16"):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as temporary:
                with mock.patch.dict(os.environ, {"TRITON_CACHE_DIR": temporary}):
                    report = self._run_worker("--warmup-worker", dtype)
                print(f"K3 RoPE warmup {dtype}: {json.dumps(report, sort_keys=True)}")
                self.assertGreater(report["warmup_compilations"], 0)
                self.assertEqual(
                    report["requests"],
                    [
                        {"tokens": 7, "compilations": 0},
                        {"tokens": 128, "compilations": 0},
                        {"tokens": 257, "compilations": 0},
                    ],
                )

    def test_cold_compile_restore_and_new_layout(self) -> None:
        from rtp_llm.utils import jit_cache_manager as jit
        from rtp_llm.utils import jit_cache_store as store

        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ):
            root = Path(temporary).resolve()
            os.environ["MODEL_TYPE"] = "kimi_k3"
            os.environ["TEST_JIT_LOCAL_DIR"] = str(root / "local")
            for item in jit.COMPONENTS:
                os.environ[item.env_name] = str(root / "excluded" / item.name)
            os.environ.pop("TRITON_CACHE_DIR")
            jit.setup_jit_cache_env.cache_clear()
            self.addCleanup(jit.setup_jit_cache_env.cache_clear)
            (root / "remote").mkdir()
            config = SimpleNamespace(
                manage_jit_cache=True,
                remote_jit_dir=str(root / "remote"),
                jit_cache_setup_timeout_s=30,
            )
            producer = jit.start_from_config(config)
            self.assertIsNotNone(producer)
            try:
                self.assertEqual(
                    [item.name for item in producer.scope.components], ["triton"]
                )
                self.assertGreater(self._compile(64), 0)
                self.assertEqual(self._compile(64), 0)
            finally:
                producer.stop()
            snapshots = list(
                producer.store.remote_root.glob(f"*{store.SNAPSHOT_SUFFIX}")
            )
            self.assertTrue(snapshots)
            files = producer._snapshot_files()
            self.assertTrue(any(name.endswith(".cubin") for name in files))
            shutil.rmtree(producer.scope.root)
            consumer = jit.start_from_config(config)
            self.assertIsNotNone(consumer)
            try:
                self.assertIsNotNone(consumer._restored)
                self.assertEqual(self._compile(64), 0)
                self.assertGreater(self._compile(128), 0)
            finally:
                consumer.stop()

    def _compile(self, head_dim: int) -> int:
        return self._run_worker("--worker", str(head_dim))["compilations"]

    def _run_worker(self, *arguments: str) -> dict:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(sys.path)
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), *arguments],
            env=env,
            text=True,
            capture_output=True,
            timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        reports = [
            line[len(RESULT_PREFIX) :]
            for line in result.stdout.splitlines()
            if line.startswith(RESULT_PREFIX)
        ]
        self.assertEqual(len(reports), 1, result.stdout + result.stderr)
        return json.loads(reports[0])


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        run_rope(int(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == "--warmup-worker":
        run_warmup_rope(sys.argv[2])
    else:
        unittest.main()

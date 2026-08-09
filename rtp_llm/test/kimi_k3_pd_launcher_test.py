#!/usr/bin/env python3

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


def _launcher_path() -> Path:
    candidates = [Path(__file__).resolve().parents[2] / "example/start_kimi_k3_pd.sh"]
    test_srcdir = os.environ.get("TEST_SRCDIR")
    if test_srcdir:
        workspace = os.environ.get("TEST_WORKSPACE", "rtp_llm")
        candidates.extend(
            [
                Path(test_srcdir) / workspace / "example/start_kimi_k3_pd.sh",
                Path(test_srcdir) / "rtp_llm/example/start_kimi_k3_pd.sh",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"start_kimi_k3_pd.sh not found in {candidates}")


class KimiK3PdLauncherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        (self.root / "home").mkdir()
        (self.root / "operators").mkdir()
        (self.root / "model.safetensors.index.json").write_text("{}")
        (self.root / "config.json").write_text(json.dumps({"num_hidden_layers": 1}))

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _run(self, **overrides: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        for name in (
            "KIMI_K3_WORLD_SIZE",
            "KIMI_K3_LOCAL_WORLD_SIZE",
            "KIMI_K3_TP_SIZE",
            "KIMI_K3_DP_SIZE",
            "KIMI_K3_EP_SIZE",
            "KIMI_K3_DECODE_TOPOLOGY",
        ):
            env.pop(name, None)
        env.update(
            {
                "HOME": str(self.root / "home"),
                "CHECKPOINT_PATH": str(self.root),
                "TOKENIZER_PATH": str(self.root),
                "PREFILL_ENDPOINT": "127.0.0.1:12533",
                "DECODE_ENDPOINT": "127.0.0.1:12633",
                "RTP_LLM_PYTHON": "/usr/bin/true",
                "KIMI_K3_RUN_ROOT": str(self.root / "run"),
                "KIMI_K3_TMPDIR": str(self.root / "tmp"),
                "KIMI_K3_SKIP_BUILD": "1",
                "KIMI_K3_SERVER_BINARY": "/usr/bin/true",
                "KIMI_K3_DRY_RUN": "1",
                "KIMI_K3_OPERATOR_PYTHONPATH": str(self.root / "operators"),
                "KIMI_K3_DEEPGEMM_JIT_COMPILER": "nvrtc",
                "KIMI_K3_KDA_BACKEND": "kernel",
                "KIMI_K3_MLA_BACKEND": "kernel",
                "KIMI_K3_DECODE_CPU_OFFLOAD_START": "none",
            }
        )
        env.update(overrides)
        return subprocess.run(
            ["bash", str(_launcher_path()), "decode"],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def _assert_topology(
        self,
        result: subprocess.CompletedProcess[str],
        *,
        tp: int,
        dp: int,
        ep: int,
        world: int,
        local_world: int,
    ) -> None:
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(
            f"topology:        TP{tp}/DP{dp}/EP{ep}/world{world}/local{local_world}",
            result.stdout,
        )
        for flag, value in (
            ("--tp_size", tp),
            ("--dp_size", dp),
            ("--ep_size", ep),
            ("--world_size", world),
            ("--local_world_size", local_world),
        ):
            self.assertIn(f" {flag} {value}", result.stdout)

    def test_default_topology_remains_tp8(self) -> None:
        self._assert_topology(self._run(), tp=8, dp=1, ep=8, world=8, local_world=8)

    def test_tp_size_alone_derives_full_topology_generically(self) -> None:
        for size in (2, 4):
            with self.subTest(size=size):
                self._assert_topology(
                    self._run(KIMI_K3_TP_SIZE=str(size)),
                    tp=size,
                    dp=1,
                    ep=size,
                    world=size,
                    local_world=size,
                )

    def test_generic_legacy_alias_is_not_enumerated(self) -> None:
        self._assert_topology(
            self._run(KIMI_K3_DECODE_TOPOLOGY="tp4_ep4"),
            tp=4,
            dp=1,
            ep=4,
            world=4,
            local_world=4,
        )

    def test_world_and_dp_derive_tp(self) -> None:
        self._assert_topology(
            self._run(
                KIMI_K3_WORLD_SIZE="8",
                KIMI_K3_DP_SIZE="2",
                KIMI_K3_EP_SIZE="4",
                KIMI_K3_MOE_BACKEND="deepep",
                KIMI_K3_DEEP_EP_PYTHONPATH=str(self.root / "operators"),
                KIMI_K3_SP_MOE="0",
            ),
            tp=4,
            dp=2,
            ep=4,
            world=8,
            local_world=8,
        )

    def test_inconsistent_product_is_rejected(self) -> None:
        result = self._run(
            KIMI_K3_WORLD_SIZE="8",
            KIMI_K3_TP_SIZE="4",
            KIMI_K3_DP_SIZE="1",
        )
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("TP_SIZE * DP_SIZE must equal WORLD_SIZE", result.stdout)


if __name__ == "__main__":
    unittest.main()

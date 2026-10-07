"""Small real CUDA/Triton regressions; run under the repository GPU lock."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


log = load("rtp_llm.models_py.utils.prefill_input_log", "utils/prefill_input_log.py")
load(
    "rtp_llm.models_py.triton_kernels.common.offset", "triton_kernels/common/offset.py"
)
scatter = load("prefill_log_scatter", "triton_kernels/common/scatter_qkv.py")
norm = load("prefill_log_norm", "triton_kernels/common/gated_rmsnorm_prefill.py")
gating = load("prefill_log_gating", "triton_kernels/fla/gdn_gating_prefill.py")


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU")
class PrefillInputLogGpuTest(unittest.TestCase):
    def test_real_kernels_and_native_dispatch_match(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, MEGA_MOE_SNAPSHOT_DIR=directory
        ):
            torch.manual_seed(17)
            for tokens in (17, 4097):
                packed = torch.randn(tokens, 512, device="cuda", dtype=torch.bfloat16)
                x = torch.randn(tokens, 256, device="cuda", dtype=torch.bfloat16)
                gate = torch.randn_like(x)
                weight = torch.ones(128, device="cuda", dtype=torch.bfloat16)
                a = torch.randn(tokens, 8, device="cuda", dtype=torch.bfloat16)
                b = torch.randn_like(a)
                alog = torch.zeros(8, device="cuda", dtype=torch.float32)
                bias = torch.ones_like(alog)

                def run():
                    q, k, v = scatter.scatter_qkv(packed, 1, 2, 128, 128)
                    y = norm.gated_rmsnorm_prefill(x, gate, weight, group_size=128)
                    g, beta = gating.gdn_gating_prefill(alog, a, b, bias)
                    z = log.trace_call(
                        "native.mm", torch.mm, x.float(), x[:4].float().T
                    )
                    return q, k, v, y, g, beta, z

                with mock.patch.dict(os.environ, MEGA_MOE_LOG_INPUTS="0"):
                    reference = run()
                with mock.patch.dict(os.environ, MEGA_MOE_LOG_INPUTS="1"):
                    with log.prefill_input_snapshot(True):
                        with log.prefill_stage("decoder", 0):
                            # The recorder must never request an explicit CUDA sync.
                            with mock.patch.object(
                                torch.cuda,
                                "synchronize",
                                side_effect=AssertionError(
                                    "unexpected synchronization"
                                ),
                            ):
                                actual = run()
                for expected, value in zip(reference, actual):
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
                rows = [
                    json.loads(line)
                    for line in next(Path(directory).glob("*.jsonl"))
                    .read_text()
                    .splitlines()
                ]
                calls = [r for r in rows if r["event"] == "call"]
                self.assertGreaterEqual(
                    len([r for r in calls if r["backend"] == "triton"]), 3
                )
                self.assertTrue(any(r["backend"] == "torch_dispatch" for r in calls))
                self.assertTrue(any(r["op"] == "native.mm" for r in calls))
                self.assertEqual(rows[-1]["state"], "python_forward_returned")


if __name__ == "__main__":
    unittest.main()

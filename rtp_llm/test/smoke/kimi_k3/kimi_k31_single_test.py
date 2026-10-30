"""Start one full K3.1 PDFUSION service and run the PR smoke."""

import os
import tempfile
import time
import unittest
from pathlib import Path
from uuid import uuid4

from rtp_llm.test.smoke.kimi_k3.text_smoke import parse_args, run_smoke
from rtp_llm.test.utils.maga_server_manager import MagaServerManager

CHECKPOINT_ROOT = Path("/mnt/fuse/smoke-test/hf")


class KimiK31SingleTest(unittest.TestCase):
    def test_text_smoke(self):
        checkpoint = CHECKPOINT_ROOT / "Kimi-K3.1"
        draft = CHECKPOINT_ROOT / "Kimi-K3.1-MTP"
        output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]) / "kimi-k31"
        namespace = f"k3-single-{uuid4().hex[:24]}"
        with tempfile.TemporaryDirectory(prefix="k31-", dir="/tmp") as runtime:
            server = MagaServerManager(
                role_name="k31_single",
                smoke_args_str=os.environ["SMOKE_ARGS"],
                env_args={
                    "SP_CHECKPOINT_PATH": str(draft),
                    "TMPDIR": runtime,
                    "RTP_LLM_CPU_TP_BROADCASTER_DIR": runtime,
                    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                },
            )
            try:
                started = time.monotonic()
                self.assertTrue(
                    server.start_server(
                        model_path=str(checkpoint),
                        model_type="kimi_k31",
                        timeout=1800,
                    ),
                    f"K3.1 failed to start, process exit code={server.exit_code}",
                )
                print(
                    f"K3.1 startup elapsed_s={time.monotonic() - started:.3f}",
                    flush=True,
                )
                run_smoke(
                    parse_args(
                        [
                            "--base-url",
                            f"http://127.0.0.1:{server.port}",
                            "--output",
                            str(output / "accuracy.json"),
                            "--namespace",
                            namespace,
                            "--checkpoint",
                            str(checkpoint),
                            "--engine-log",
                            server.log_file_path,
                        ]
                    )
                )
            finally:
                server.stop_server()


if __name__ == "__main__":
    unittest.main()

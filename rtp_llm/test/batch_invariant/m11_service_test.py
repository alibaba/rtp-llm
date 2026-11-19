"""Launch the M11 smoke service and compare full-vocabulary BI RPC outputs."""

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

import requests

from rtp_llm.config.py_config_modules import ServerConfig
from rtp_llm.test.batch_invariant import service_test
from rtp_llm.test.smoke.qwen4_m11_smoke_test import Qwen4M11SmokeTest
from rtp_llm.test.utils.maga_server_manager import MagaServerManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, choices=(0, 256), default=0)
    args = parser.parse_args()
    if os.environ.get("RTP_BATCH_INVARIANT") != "1":
        parser.error("run this target with RTP_BATCH_INVARIANT=1")
    checkpoint = Path(os.environ["QWEN4_M11_CHECKPOINT"])
    output = Path(os.environ["RTP_M11_BI_OUTPUT"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference = output / "budget-0" / "solo.npz"
    run_dir = output / f"budget-{args.budget}"
    run_dir.mkdir()
    if args.budget and not reference.is_file():
        parser.error(
            "the budget-0 service must first save its single-request reference"
        )
    encode = service_test.load_encoder(checkpoint, m11=True)
    print(f"M11 fixed continuation IDs {encode(service_test.CONTINUATION)}", flush=True)

    with patch.dict(
        os.environ,
        {
            "TEST_UNDECLARED_OUTPUTS_DIR": str(run_dir),
            "MAGA_SERVER_WORK_DIR": str(run_dir),
            "RTP_LLM_RECORD_DECODE_SAMPLE_RATE": "1",
        },
    ):
        server = MagaServerManager(
            env_args={"FT_SERVER_TEST": "1"},
            role_name="qwen4_m11_bi",
            smoke_args_str=os.environ["SMOKE_ARGS"]
            + f" --prefill_chunk_size {args.budget}",
        )
        try:
            if not server.start_server(str(checkpoint), "qwen4_exp", timeout=1800):
                raise RuntimeError("M11 BI service did not become healthy")
            config = ServerConfig()
            config.start_port = int(server.port)
            addresses = []
            for rank in range(8):
                config.set_local_rank(rank)
                addresses.append(f"127.0.0.1:{config.rpc_server_port}")
            log_dir = run_dir / "logs"
            (run_dir / "service.json").write_text(
                json.dumps(
                    {
                        "port": int(server.port),
                        "rpc_addresses": addresses,
                        "log_dir": str(log_dir),
                        "process_log": server.log_file_path,
                    }
                )
            )
            service_test.main(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--log-dir",
                    str(log_dir),
                    "--save",
                    str(run_dir / "solo.npz"),
                    "--save-cases",
                    str(run_dir / "cases"),
                    "--m11",
                    "--target-length",
                    "257",
                    *(
                        ["--chunked", "--reference", str(reference)]
                        if args.budget
                        else []
                    ),
                    "--dp-addresses",
                    *addresses,
                ],
                encode=encode,
            )
            with requests.Session() as session:
                session.trust_env = False
                smoke = Qwen4M11SmokeTest()
                smoke.session = session
                smoke.server_url = f"http://127.0.0.1:{server.port}/"
                smoke.text_config = json.loads(
                    (checkpoint / "config.json").read_text()
                )["text_config"]
                smoke.test_chinese_answer()
                smoke.test_qsa_beyond_indexer_budget()
        finally:
            server.stop_server()


if __name__ == "__main__":
    main()

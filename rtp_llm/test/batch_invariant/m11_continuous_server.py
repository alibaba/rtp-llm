"""Run the real M11 PDFUSION ratio service until OUTPUT/stop is created."""

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.batch_invariant.service_test import rank_rpc_addresses
from rtp_llm.test.utils.maga_server_manager import MagaServerManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk", type=int, choices=(0, 256), default=0)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--kv-mb", type=int, default=10496)
    parser.add_argument("--linear-blocks", type=int, default=23)
    args = parser.parse_args()
    bi = int(os.environ["RTP_BATCH_INVARIANT"])
    if bi not in (0, 1):
        parser.error("RTP_BATCH_INVARIANT must be 0 or 1")
    checkpoint = Path(os.environ["QWEN4_M11_CHECKPOINT"]).resolve(strict=True)
    ple = Path(os.environ["QWEN4_PLE_SHARED_TABLE"]).resolve(strict=True)
    output = Path(os.environ["RTP_M11_OUTPUT"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ready = output / "service.json"
    stop = output / "stop"
    if ready.exists() or stop.exists():
        parser.error(f"use a fresh RTP_M11_OUTPUT directory: {output}")
    record = str(int(args.record))
    settings = {
        "role_type": "PDFUSION",
        "use_batch_decode_scheduler": "0",
        "pdfusion_scheduler_mode": "ratio",
        "decode_prefill_ratio": "1",
        "kv_cache_mem_mb": str(args.kv_mb),
        "linear_pool_blocks": str(args.linear_blocks),
        "reserve_block_ratio": "5",
        "seq_size_per_block": "128",
        "kernel_seq_size_per_block": "128",
        "max_seq_len": "8192",
        "max_batch_tokens_size": "8192",
        "concurrency_limit": "32",
        "prefill_chunk_size": str(args.chunk),
        "ssm_state_dtype": "fp32",
        "enable_cuda_graph": "0",
        "reuse_cache": "0",
        "sp_type": "",
        "gen_timeline_sync": "0",
        "enable_batch_record": record,
        "record_decode_sample_rate": record,
    }
    smoke_args = shlex.split(os.environ["SMOKE_ARGS"]) + [
        item for key, value in settings.items() for item in (f"--{key}", value)
    ]
    with patch.dict(
        os.environ,
        {
            "TEST_UNDECLARED_OUTPUTS_DIR": str(output),
            "MAGA_SERVER_WORK_DIR": str(output),
            "RTP_LLM_RECORD_DECODE_SAMPLE_RATE": record,
        },
    ):
        server = MagaServerManager(
            env_args={"FT_SERVER_TEST": "1"},
            role_name="m11_continuous",
            smoke_args_str=shlex.join(smoke_args),
        )
        try:
            if not server.start_server(str(checkpoint), "qwen4_exp", timeout=1800):
                raise RuntimeError("M11 ratio service did not become healthy")
            (output / "client.env").write_text(
                f"export PYTHONPATH={shlex.quote(os.pathsep.join(sys.path))}\n"
                f"export RTP_M11_CLIENT_PYTHON={shlex.quote(sys.executable)}\n"
            )
            manifest = {
                "bi": bi,
                "record": args.record,
                "prefill_chunk_size": args.chunk,
                "argv": smoke_args,
                "checkpoint": str(checkpoint),
                "ple_manifest": str(ple),
                "port": int(server.port),
                "rpc_addresses": rank_rpc_addresses(server.port, 8),
                "log_dir": str(output / "logs"),
                "process_log": server.log_file_path,
            }
            temporary = ready.with_suffix(".tmp")
            temporary.write_text(json.dumps(manifest, indent=2) + "\n")
            temporary.replace(ready)
            print(json.dumps(manifest), flush=True)
            while not stop.exists():
                pid = server.server_pid
                if pid is None or not Path(f"/proc/{pid}").is_dir():
                    raise RuntimeError(
                        f"M11 ratio service exited: {server.log_file_path}"
                    )
                time.sleep(1)
        finally:
            server.stop_server()


if __name__ == "__main__":
    main()

"""Real-weight standalone ViT service and multiprocess RPC acceptance test."""

import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest import TestCase, main

import grpc
from PIL import Image, ImageDraw

from rtp_llm.test.perf_test.vit_rpc_benchmark import benchmark
from rtp_llm.test.utils.port_util import PortsContext


def image_url(size, color):
    image = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 30, size[0] // 2, size[1] // 2), fill="white")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


class VitRpcStressTest(TestCase):
    def test_standalone_vit_batches_concurrent_rpc_requests(self):
        checkpoint = os.environ.get("CKPT_PATH")
        self.assertTrue(checkpoint, "CKPT_PATH must contain the real visual checkpoint")
        outputs = Path(
            os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", tempfile.mkdtemp())
        )
        outputs.mkdir(parents=True, exist_ok=True)
        log_path = outputs / "vit_server.log"
        with PortsContext(num_ports=12) as ports, log_path.open("w") as log:
            address = f"127.0.0.1:{ports[1]}"
            command = [
                sys.executable,
                "-m",
                "rtp_llm.start_server",
                "--model_type",
                "deepseek_v4",
                "--checkpoint_path",
                checkpoint,
                "--tokenizer_path",
                checkpoint,
                "--vit_separation",
                "1",
                "--tp_size",
                "1",
                "--dp_size",
                "1",
                "--world_size",
                "1",
                "--start_port",
                str(ports[0]),
                "--act_type",
                "BF16",
                "--enable_cuda_graph",
                "0",
                "--load_method",
                "scratch",
                "--mm_cache_item_num",
                "0",
                "--vit_batch_wait_ms",
                "10",
                "--vit_max_batch_images",
                "8",
                "--vit_max_concurrent_requests",
                "16",
            ]
            env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
            process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 600
                with grpc.insecure_channel(address) as channel:
                    ready = grpc.channel_ready_future(channel)
                    while True:
                        self.assertIsNone(process.poll(), log_path.read_text()[-12000:])
                        try:
                            ready.result(timeout=1)
                            break
                        except grpc.FutureTimeoutError:
                            self.assertLess(
                                time.monotonic(),
                                deadline,
                                log_path.read_text()[-12000:],
                            )
                # One image per request makes a multi-image GPU batch prove
                # aggregation across independent RPCs, not just within one RPC.
                report = benchmark(
                    address,
                    [image_url((384, 512), "blue")],
                    processes=8,
                    requests_per_process=128,
                )
                report["mixed_images"] = benchmark(
                    address,
                    [image_url((384, 512), "blue"), image_url((512, 384), "red")],
                    processes=4,
                    requests_per_process=32,
                )
                (outputs / "vit_rpc_benchmark.json").write_text(
                    json.dumps(report, indent=2)
                )
                print(json.dumps(report, indent=2), flush=True)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=15)


if __name__ == "__main__":
    main()

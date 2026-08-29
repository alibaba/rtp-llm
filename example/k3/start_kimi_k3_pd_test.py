import os
import pathlib
import socket
import subprocess
import tempfile
import unittest
from typing import Optional


@unittest.skipIf(
    os.uname().sysname == "Darwin",
    "the production launcher requires Bash 4+ and runs in lhc_GPU Linux",
)
class StartKimiK3PdDryRunTest(unittest.TestCase):
    def _run(
        self,
        role: str,
        topology: str,
        *,
        world_rank: str = "0",
        gang_config: Optional[str] = None,
        prefill_endpoint: str = "127.0.0.1:27188",
        decode_endpoint: str = "127.0.0.1:28188",
        dry_run: bool = True,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        script = pathlib.Path(__file__).with_name("start_kimi_k3_pd.sh")
        with tempfile.TemporaryDirectory() as checkpoint:
            root = pathlib.Path(checkpoint)
            (root / "config.json").write_text("{}\n", encoding="utf-8")
            (root / "model.safetensors.index.json").write_text(
                '{"weight_map": {}}\n', encoding="utf-8"
            )
            env = os.environ.copy()
            env.update(
                {
                    "CHECKPOINT_PATH": checkpoint,
                    "PREFILL_ENDPOINT": prefill_endpoint,
                    "DECODE_ENDPOINT": decode_endpoint,
                    "KIMI_K3_DECODE_TOPOLOGY": topology,
                    "WORLD_RANK": world_rank,
                    "RUN_ROOT": str(root / "run"),
                    "RTP_LLM_TMPDIR": str(root / "tmp"),
                    "HOME": checkpoint,
                }
            )
            if dry_run:
                env["RTP_LLM_DRY_RUN"] = "1"
            else:
                env.pop("RTP_LLM_DRY_RUN", None)
                env["RTP_LLM_SKIP_BUILD"] = "1"
                env["RTP_LLM_SERVER_BINARY"] = "/bin/true"
            if gang_config is None:
                env.pop("GANG_CONFIG_STRING", None)
            else:
                env["GANG_CONFIG_STRING"] = gang_config
            return subprocess.run(
                ["bash", str(script), role],
                check=check,
                capture_output=True,
                text=True,
                env=env,
            )

    def _dry_run(self, role: str, topology: str, **kwargs) -> str:
        return self._run(role, topology, **kwargs).stdout

    def test_prefill_is_fixed_tp8(self):
        output = self._dry_run("prefill", "dp16_ktp16_ep16")
        self.assertIn("--tp_size 8", output)
        self.assertIn("--dp_size 1", output)
        self.assertIn("--ktp_size 1", output)
        self.assertIn("--ep_size 8", output)
        self.assertIn("--world_size 8", output)

    def test_decode_projection_ktp8(self):
        output = self._dry_run("decode", "dp8_ktp8_ep8")
        self.assertIn("--tp_size 1", output)
        self.assertIn("--dp_size 8", output)
        self.assertIn("--ktp_size 8", output)
        self.assertIn("--ep_size 8", output)
        self.assertIn("--world_size 8", output)
        self.assertIn("--local_world_size 8", output)
        self.assertIn("worker_port_block: 28188-28259", output)

    def test_decode_projection_ktp16(self):
        gang = (
            "name:k3_part0,ip:10.0.0.1,port:28188;"
            "name:k3_part1,ip:10.0.0.2,port:28188"
        )
        output = self._dry_run(
            "decode", "dp16_ktp16_ep16", gang_config=gang
        )
        self.assertIn("--tp_size 1", output)
        self.assertIn("--dp_size 16", output)
        self.assertIn("--ktp_size 16", output)
        self.assertIn("--ep_size 16", output)
        self.assertIn("--world_size 16", output)
        self.assertIn("--world_rank 0", output)
        self.assertIn("--local_world_size 8", output)

        second_node = self._dry_run(
            "decode",
            "dp16_ktp16_ep16",
            world_rank="8",
            gang_config=gang,
        )
        self.assertIn("--world_rank 8", second_node)

    def test_decode_projection_ktp16_requires_gang_and_node_base_rank(self):
        missing_gang = self._run(
            "decode", "dp16_ktp16_ep16", check=False
        )
        self.assertNotEqual(missing_gang.returncode, 0)
        self.assertIn("GANG_CONFIG_STRING", missing_gang.stderr)

        bad_rank = self._run(
            "decode",
            "dp16_ktp16_ep16",
            world_rank="1",
            gang_config="name:k3_part0,ip:10.0.0.1,port:28188",
            check=False,
        )
        self.assertNotEqual(bad_rank.returncode, 0)
        self.assertIn("WORLD_RANK must be 0 or 8", bad_rank.stderr)

    def test_worker_port_block_rejects_occupied_derived_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            occupied_port = listener.getsockname()[1]
            base_port = occupied_port - 22  # local rank 2, RDMA offset 4
            if base_port < 1024 or base_port + 71 > 65535:
                self.skipTest("ephemeral port cannot form a valid worker block")
            result = self._run(
                "decode",
                "dp8_ktp8_ep8",
                decode_endpoint=f"127.0.0.1:{base_port}",
                dry_run=False,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"port {occupied_port} cannot bind", result.stderr)


if __name__ == "__main__":
    unittest.main()

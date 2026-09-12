import os
import pathlib
import socket
import subprocess
import tempfile
import unittest
from unittest import mock
from typing import Optional


@unittest.skipIf(
    os.uname().sysname == "Darwin",
    "the production launcher requires Bash 4+ and runs in lhc_GPU Linux",
)
class StartKimiK3PdDryRunTest(unittest.TestCase):
    @staticmethod
    def _find_free_port_block(span: int = 72) -> int:
        # Do not derive the block from an ephemeral port: subprocess startup
        # itself can consume another ephemeral port inside the released span.
        for base_port in range(20000, 32000, span + 1):
            held = []
            try:
                for port in range(base_port, base_port + span):
                    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    try:
                        listener.bind(("127.0.0.1", port))
                    except OSError:
                        listener.close()
                        raise
                    held.append(listener)
            except OSError:
                continue
            finally:
                for listener in held:
                    listener.close()
            return base_port
        raise RuntimeError(f"could not find a free {span}-port block")

    def _run(
        self,
        role: str,
        topology: Optional[str] = None,
        *,
        world_rank: str = "0",
        gang_config: Optional[str] = None,
        prefill_endpoint: str = "127.0.0.1:27188",
        decode_endpoint: str = "127.0.0.1:28188",
        server_binary: Optional[str] = None,
        dry_run: bool = True,
        check: bool = True,
        env_overrides: Optional[dict[str, str]] = None,
    ) -> subprocess.CompletedProcess[str]:
        script = pathlib.Path(__file__).with_name("start_kimi_k3_pd.sh")
        with tempfile.TemporaryDirectory() as checkpoint:
            root = pathlib.Path(checkpoint)
            (root / "config.json").write_text("{}\n", encoding="utf-8")
            (root / "model.safetensors.index.json").write_text(
                '{"weight_map": {}}\n', encoding="utf-8"
            )
            env = os.environ.copy()
            env.pop("THINK_START_TAG", None)
            env.pop("THINK_END_TAG", None)
            env.update(
                {
                    "CHECKPOINT_PATH": checkpoint,
                    "PREFILL_ENDPOINT": prefill_endpoint,
                    "DECODE_ENDPOINT": decode_endpoint,
                    "WORLD_RANK": world_rank,
                    "RUN_ROOT": str(root / "run"),
                    "RTP_LLM_TMPDIR": str(root / "tmp"),
                    "HOME": checkpoint,
                }
            )
            if topology is None:
                env.pop("KIMI_K3_DECODE_TOPOLOGY", None)
            else:
                env["KIMI_K3_DECODE_TOPOLOGY"] = topology
            if dry_run:
                env["RTP_LLM_DRY_RUN"] = "1"
            else:
                env.pop("RTP_LLM_DRY_RUN", None)
                env["RTP_LLM_SKIP_BUILD"] = "1"
                env["RTP_LLM_SERVER_BINARY"] = server_binary or "/bin/true"
            if gang_config is None:
                env.pop("GANG_CONFIG_STRING", None)
            else:
                env["GANG_CONFIG_STRING"] = gang_config
            if env_overrides:
                env.update(env_overrides)
            return subprocess.run(
                ["bash", str(script), role],
                check=check,
                capture_output=True,
                text=True,
                env=env,
            )

    def run_dry_run(self, role: str, **env_overrides: str) -> str:
        return self._run(role, env_overrides=env_overrides).stdout

    def _dry_run(self, role: str, topology: str, **kwargs) -> str:
        return self._run(role, topology, **kwargs).stdout

    def test_default_and_explicit_kv_budgets_reach_both_roles(self):
        for role, default in (("prefill", "43000"), ("decode", "46000")):
            with mock.patch.dict(os.environ):
                os.environ.pop("KV_CACHE_MEM_MB", None)
                output = self.run_dry_run(role)
                self.assertIn(f"--kv_cache_mem_mb {default}", output)
            for value in ("-1", "0", "24000"):
                with self.subTest(role=role, value=value):
                    output = self.run_dry_run(role, KV_CACHE_MEM_MB=value)
                    self.assertIn(f"--kv_cache_mem_mb {value}", output)

    def test_invalid_kv_budget_is_rejected_before_launch(self):
        for value in ("auto", "1.5"):
            result = self._run("decode", check=False, env_overrides={"KV_CACHE_MEM_MB": value})
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("command:", result.stdout)

    def test_configurable_parallelism_reaches_launcher(self):
        for tp in (1, 2, 4, 8, 16):
            for role in ("prefill", "decode"):
                with self.subTest(tp=tp, role=role):
                    output = self.run_dry_run(role, KIMI_K3_TP_SIZE=str(tp), KIMI_K3_EP_SIZE=str(tp))
                    self.assertIn(f"TP{tp}/DP1/KTP1/EP{tp}", output)
                    self.assertIn(f"--tp_size {tp}", output)
                    self.assertIn(f"--ep_size {tp}", output)
                    self.assertIn(f"--world_size {tp}", output)
                    self.assertIn(f"--local_world_size {tp}", output)
                    if role == "prefill":
                        self.assertIn(f"shard={int(tp % 2 == 0)}", output)

    def test_generic_topology_flags_override_k3_compatibility_aliases(self):
        for role in ("prefill", "decode"):
            with self.subTest(role=role):
                output = self.run_dry_run(
                    role, TP_SIZE="4", DP_SIZE="1", EP_SIZE="4",
                    WORLD_SIZE="4", LOCAL_WORLD_SIZE="4",
                    KIMI_K3_TP_SIZE="8", KIMI_K3_EP_SIZE="8",
                )
                self.assertIn("TP4/DP1/KTP1/EP4 world=4 local=4", output)
                self.assertIn("--tp_size 4", output)
                self.assertIn("--ep_size 4", output)
                self.assertIn("--local_world_size 4", output)
                if role == "prefill":
                    self.assertIn("MegaMoE tokens:  16384/rank", output)

    def test_defaults_to_complete_k3_think_boundary_for_both_roles(self) -> None:
        for role in ("prefill", "decode"):
            with self.subTest(role=role):
                output = self.run_dry_run(role)
                self.assertIn("think start:     <|open|>think<|sep|>", output)
                self.assertIn(
                    "think end:       <|close|>think<|sep|><|open|>response<|sep|>",
                    output,
                )
    def test_prefill_ignores_decode_topology(self):
        output = self._dry_run("prefill", "dp16_ktp16_ep16")
        self.assertIn("--tp_size 8", output)
        self.assertIn("--dp_size 1", output)
        self.assertIn("--ktp_size 1", output)
        self.assertIn("--ep_size 0", output)
        self.assertIn("--world_size 8", output)
        self.assertIn("think start:     <|open|>think<|sep|>", output)
        self.assertIn(
            "think end:       <|close|>think<|sep|><|open|>response<|sep|>",
            output,
        )

    def test_preserves_explicit_think_boundary_override(self):
        output = self._dry_run(
            "prefill",
            "tp8_ep8",
            env_overrides={
                "THINK_START_TAG": "custom-start",
                "THINK_END_TAG": "custom-end",
            },
        )
        self.assertIn("think start:     custom-start", output)
        self.assertIn("think end:       custom-end", output)

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

    def test_decode_projection_ktp16_exports_local_world_size(self):
        gang = (
            "name:k3_part0,ip:10.0.0.1,port:28188;"
            "name:k3_part1,ip:10.0.0.2,port:28188"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = pathlib.Path(temp_dir) / "print_local_world_size.sh"
            probe.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'LOCAL_WORLD_SIZE=%s\\n' \"${LOCAL_WORLD_SIZE:-}\"\n",
                encoding="utf-8",
            )
            probe.chmod(0o755)
            result = None
            for _ in range(8):
                decode_port = self._find_free_port_block()
                candidate = self._run(
                    "decode",
                    "dp16_ktp16_ep16",
                    gang_config=gang,
                    prefill_endpoint="127.0.0.1:27188",
                    decode_endpoint=f"127.0.0.1:{decode_port}",
                    server_binary=str(probe),
                    dry_run=False,
                    check=False,
                )
                if candidate.returncode == 0:
                    result = candidate
                    break
                if "worker port block" not in candidate.stderr:
                    self.fail(candidate.stderr)
            self.assertIsNotNone(result, "could not reserve a stable worker port block")
            assert result is not None
        self.assertIn("LOCAL_WORLD_SIZE=8", result.stdout)

    def test_worker_port_block_rejects_occupied_derived_port(self):
        last_stderr = ""
        for _ in range(16):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.bind(("127.0.0.1", 0))
                occupied_port = listener.getsockname()[1]
                base_port = occupied_port - 22  # local rank 2, RDMA offset 4
                if base_port < 1024 or base_port + 71 > 65535:
                    continue
                result = self._run(
                    "decode",
                    "dp8_ktp8_ep8",
                    decode_endpoint=f"127.0.0.1:{base_port}",
                    dry_run=False,
                    check=False,
                )
            self.assertNotEqual(result.returncode, 0)
            last_stderr = result.stderr
            if f"port {occupied_port} cannot bind" in result.stderr:
                return
        self.fail(
            "could not isolate the intended derived-port collision after "
            f"16 attempts; last stderr={last_stderr!r}"
        )


if __name__ == "__main__":
    unittest.main()

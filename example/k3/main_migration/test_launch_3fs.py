"""Checkpoint source checks for the K3 example launcher."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


LAUNCHER = Path(__file__).with_name("launch_bf16.py")
spec = importlib.util.spec_from_file_location("k3_launch_bf16", LAUNCHER)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class CheckpointSourceTest(unittest.TestCase):
    def test_asymmetric_pd_topologies_and_sixteen_card_planning(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            target = root / "target"
            draft = root / "draft"
            target.mkdir()
            draft.mkdir()
            (target / "config.json").write_text(json.dumps({"num_hidden_layers": 93}))
            for role, tp, dp, source_tp, capture in (
                ("PREFILL", 4, 1, 4, None),
                ("DECODE", 8, 1, 4, "1,2,4,8,16,32,64"),
                ("PREFILL", 8, 1, 8, None),
                ("DECODE", 4, 2, 8, "1,2,4,8,16,32"),
                ("DECODE", 8, 2, 8, "1,2,4,8,16,32"),
            ):
                args = SimpleNamespace(
                    checkpoint=str(target), draft_checkpoint=str(draft),
                    start_port=28000, peer_port=29000, peer_ip="127.0.0.1",
                    role=role, server="/bin/true", orthogonal_smoke=True,
                    tp_size=tp, dp_size=dp, ep_size=tp * dp,
                    prefill_source_tp_size=source_tp,
                )
                env, command = launch.launch_config(args)
                options = dict(zip(command[1::2], command[2::2]))
                self.assertEqual(options["--world_size"], str(tp * dp))
                self.assertEqual(options["--ep_size"], str(tp * dp))
                self.assertEqual(env["LOCAL_WORLD_SIZE"], str(tp * dp))
                self.assertEqual(len(env["CUDA_VISIBLE_DEVICES"].split(",")), tp * dp)
                if role == "DECODE":
                    self.assertEqual(options["--prefill_cp_size"], str(source_tp))
                    self.assertEqual(options["--decode_capture_config"], capture)

    def test_gpu_process_placeholder_is_not_an_occupier(self):
        gpu_rows = "".join(f"{index}, GPU-{index}, 274114\n" for index in range(8))
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(launch.subprocess, "check_output", side_effect=[
                gpu_rows, "GPU-0, [N/A], [N/A]\n",
            ]):
                launch.require_gpu_capacity(Path(tmp), min_free_gib=250)
            self.assertTrue((Path(tmp) / "gpu-preflight.json").is_file())

    def test_orthogonal_profile_enables_cache_graph_and_evidence_by_role(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            target = root / "target"
            draft = root / "draft"
            target.mkdir()
            draft.mkdir()
            (target / "config.json").write_text(json.dumps({"num_hidden_layers": 93}))
            for role in ("PREFILL", "DECODE"):
                args = SimpleNamespace(
                    checkpoint=str(target), draft_checkpoint=str(draft),
                    start_port=28000, peer_port=29000, peer_ip="127.0.0.1",
                    role=role, server="/bin/true", orthogonal_smoke=True,
                    memory_cache_size_mb=8192, kv_cache_mem_mb=4096,
                )
                env, command = launch.launch_config(args)
                options = dict(zip(command[1::2], command[2::2]))
                self.assertEqual(env["KIMI_K3_SMOKE_EVIDENCE"], "1")
                self.assertEqual(env["RTP_MLA_PREFILL_EXPANDED_KV_BUDGET_GIB"], "6.0")
                self.assertEqual(options["--max_seq_len"], "2097152")
                self.assertEqual(options["--max_batch_tokens_size"], "262144")
                self.assertEqual(options["--max_batch_tokens_without_cache"], "65536")
                self.assertEqual(options["--concurrency_limit"], "64")
                self.assertEqual(options["--kv_cache_mem_mb"], "4096")
                self.assertEqual(options["--enable_memory_cache"],
                                 "1" if role == "PREFILL" else "0")
                if role == "PREFILL":
                    self.assertEqual(options["--max_context_batch_size"], "64")
                    self.assertEqual(options["--memory_cache_size_mb"], "8192")
                    self.assertEqual(options["--prefill_cp_kv_cache_sharded"], "1")
                else:
                    self.assertEqual(env["NCCL_GRAPH_REGISTER"], "0")
                    self.assertEqual(env["NCCL_MAX_CTAS"], "8")
                    self.assertEqual(options["--prefill_cp_kv_cache_sharded"], "1")
                    self.assertEqual(options["--prefill_cp_size"], "8")
                    self.assertEqual(options["--decode_capture_config"],
                                     "1,2,4,8,16,32,64")

    def test_complete_prefill_smoke_bounds_device_cache_for_host_demotion(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            target, draft = root / "target", root / "draft"
            target.mkdir()
            draft.mkdir()
            (target / "config.json").write_text('{"num_hidden_layers":93}')
            for role in ("PREFILL", "DECODE"):
                args = SimpleNamespace(
                    checkpoint=str(target), draft_checkpoint=str(draft),
                    start_port=28000, peer_port=29000, peer_ip="127.0.0.1",
                    role=role, server="/bin/true", orthogonal_smoke=True,
                )
                _, command = launch.launch_config(args)
                options = dict(zip(command[1::2], command[2::2]))
                if role == "PREFILL":
                    self.assertEqual(options["--kv_cache_mem_mb"], "4096")
                else:
                    self.assertEqual(options["--kv_cache_mem_mb"], "34000")
            (target / "config.json").write_text(json.dumps({
                "num_hidden_layers": 4, "attn_res_block_size": 12,
                "linear_attn_config": {"kda_layers": [1, 2, 3],
                                       "full_attn_layers": [4]},
            }))
            args.role = "PREFILL"
            args.debug_four_layer = True
            _, command = launch.launch_config(args)
            self.assertEqual(dict(zip(command[1::2], command[2::2]))["--kv_cache_mem_mb"],
                             "256")

    def test_rpc_self_address_bypasses_proxy(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            target = root / "target"
            draft = root / "draft"
            target.mkdir()
            draft.mkdir()
            (target / "config.json").write_text(json.dumps({"num_hidden_layers": 93}))
            args = SimpleNamespace(
                checkpoint=str(target), draft_checkpoint=str(draft),
                start_port=28000, peer_port=29000, peer_ip="11.163.39.115",
                role="PREFILL", server="/bin/true", allow_hf3fs_root=str(root),
            )
            with patch.object(launch.socket, "socket") as route_socket:
                route_socket.return_value.getsockname.return_value = ("11.163.39.114", 54321)
                environment, _ = launch.launch_config(args)
            for key in ("NO_PROXY", "no_proxy"):
                self.assertIn("11.163.39.114", environment[key].split(","))
                self.assertIn("11.163.39.115", environment[key].split(","))

    def test_direct_3fs_launcher_selects_supported_fastsafetensors_copier(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            target = root / "target"
            draft = root / "draft"
            target.mkdir()
            draft.mkdir()
            (target / "config.json").write_text(json.dumps({"num_hidden_layers": 93}))
            args = SimpleNamespace(
                checkpoint=str(target), draft_checkpoint=str(draft),
                start_port=28000, peer_port=29000, peer_ip="127.0.0.1",
                role="PREFILL", server="/bin/true", allow_hf3fs_root=str(root),
            )
            environment, _ = launch.launch_config(args)
            self.assertEqual(environment["FASTSAFETENSORS_NOGDS"], "0")
            self.assertEqual(environment["MEGA_MOE_INPUT_PACKER_IMPL"], "fast_finite")

    def test_moe_input_packer_can_keep_safe_finite_check(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            target = root / "target"
            draft = root / "draft"
            target.mkdir()
            draft.mkdir()
            (target / "config.json").write_text(json.dumps({"num_hidden_layers": 93}))
            args = SimpleNamespace(
                checkpoint=str(target), draft_checkpoint=str(draft),
                start_port=28000, peer_port=29000, peer_ip="127.0.0.1",
                role="PREFILL", server="/bin/true", allow_hf3fs_root=str(root),
            )
            with patch.dict(launch.os.environ, {"MEGA_MOE_INPUT_PACKER_IMPL": "optimized"}):
                environment, _ = launch.launch_config(args)
            self.assertEqual(environment["MEGA_MOE_INPUT_PACKER_IMPL"], "optimized")

    def test_explicit_hf3fs_root_accepts_only_matching_fuse_mount(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp) / "weights"
            model = root / "target"
            other = Path(tmp) / "other"
            model.mkdir(parents=True)
            other.mkdir()
            with patch.object(launch.subprocess, "check_output", return_value="fuse.hf3fs\nfuse.hf3fs\n"):
                launch.require_checkpoint_source(model, root)
                with self.assertRaises(ValueError):
                    launch.require_checkpoint_source(other, root)
            with patch.object(launch.subprocess, "check_output", return_value="ext4\n"):
                with self.assertRaises(ValueError):
                    launch.require_checkpoint_source(model, root)

    def test_network_checkpoint_remains_rejected_without_explicit_root(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            with self.assertRaises(ValueError):
                launch.require_checkpoint_source(Path(tmp), None)


if __name__ == "__main__":
    unittest.main()

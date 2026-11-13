"""V4.1 full contract plus target/draft migration and a serving probe."""

import hashlib
import json
import logging
import os
import socket
import subprocess
import unittest
from pathlib import Path

import torch

from rtp_llm.test.smoke import deepseek_v41_full_contract_test as full_contract
from rtp_llm.test.smoke.eplb_smoke_checks import check_eplb_applied


class DeepSeekV41EplbContractTest(full_contract.DeepSeekV41FullContractTest):
    # Verified against target-only EPLB at 22655c65d0 with the same MegaSE config.
    golden_filename = "full_feature_cp2_dp2_mega_se.json"
    # Cover the native decode MegaMHC -> MegaGate -> MegaMoE-SE route.
    decode_env = {
        "DSV4_MOE_STRATEGY": "mega_se",
        "DSV4_USE_MEGA_MOE_SE": "1",
        "DSV41_MEGA_SHARED_EXPERT": "1",
        "DSV4_MEGA_GATE": "1",
    }

    def start_servers(self):
        # Server subprocesses change cwd to their output directory.
        if "LD_LIBRARY_PATH" in os.environ:
            os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
                os.path.abspath(path)
                for path in os.environ["LD_LIBRARY_PATH"].split(os.pathsep)
            )
        # Run inside the smoke GPU lock, on the actual execution worker.
        checkpoint = Path(self.model_path).resolve(strict=True)
        index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
        shards = sorted(set(index["weight_map"].values()))
        evidence = {
            "hostname": socket.gethostname(),
            "target_checkpoint": str(checkpoint),
            "draft_checkpoint": str(checkpoint),
            "metadata_sha256": {
                name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
                for name in (
                    "config.json",
                    "model.safetensors.index.json",
                    "tokenizer.json",
                )
            },
            "shard_sizes": {
                name: (checkpoint / name).stat().st_size for name in shards
            },
            "mount": subprocess.check_output(
                [
                    "findmnt",
                    "-T",
                    str(checkpoint),
                    "-o",
                    "TARGET,SOURCE,FSTYPE,OPTIONS",
                ],
                text=True,
                timeout=10,
            ),
            "gpus": [],
        }
        for device in range(torch.cuda.device_count()):
            torch.cuda.set_device(device)
            probe = torch.ones(1, device="cuda") + 1
            torch.cuda.synchronize()
            self.assertEqual(probe.item(), 2)
            properties = torch.cuda.get_device_properties(device)
            evidence["gpus"].append(
                {
                    "device": device,
                    "uuid": str(properties.uuid),
                    "capability": [properties.major, properties.minor],
                }
            )
            del probe
            torch.cuda.empty_cache()
        (self.output_dir / "eplb_execution_environment.json").write_text(
            json.dumps(evidence, indent=2) + "\n"
        )
        super().start_servers()

    def test_graph_engram_and_prefix_reuse(self):
        # Keep the original verdict while checking migration before unittest
        # runs the parent's registered server cleanups.
        with self.subTest(stage="original_contract"):
            super().test_graph_engram_and_prefix_reuse()

        if os.environ.get("SMOKE_ASSERT_EPLB_APPLIED", "1") != "1":
            return

        evidence = {"post_rebalance_request_passed": False}
        try:
            for role, manager in (
                ("prefill", self.server),
                ("decode", self.graph_server),
            ):
                evidence[role] = check_eplb_applied(
                    Path(manager.log_file_path).parent,
                    self.prefill_cp_size,
                    model_ids=(0, 1) if role == "decode" else (0,),
                )
            reference = json.loads(
                (self.output_dir / "short_uncached.json").read_text()
            )["actual"]
            actual = self.request("post_eplb_probe", self.short_prompt, False)
            self.assert_same(reference, actual)
            evidence["post_rebalance_request_passed"] = True
        finally:
            (self.output_dir / "eplb_migration_evidence.json").write_text(
                json.dumps(evidence, indent=2) + "\n"
            )


if __name__ == "__main__":
    os.environ.setdefault(
        "CHECKPOINT_PATH", "/mnt/hf3fs/3fs/models/DeepSeek-V4.1-Flash"
    )
    logging.basicConfig(level=logging.INFO)
    unittest.main()

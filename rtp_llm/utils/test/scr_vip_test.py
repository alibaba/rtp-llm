import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.utils import scr_vip


class ScrVipTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "ganginfo"
        self.rows = {
            "model-rank-0": {"ip": "22.0.1.2", "real_ip": "10.0.0.2"},
            "model-rank-1": {"ip": "22.0.1.3", "real_ip": "10.0.0.3"},
        }
        self.path.write_text(json.dumps(self.rows))
        env = patch.dict(
            os.environ,
            {
                "RTP_LLM_SCR_GANG_INFO": str(self.path),
                "RTP_LLM_SCR_VIP_INTERFACE": "scr_vxlan0",
                "RTPLLM_ENABLE_SCR": "1",
                "SCR_PHASE": "checkpoint",
                "RANK_SIZE": "2",
                "RANK_ID": "1",
            },
            clear=True,
        )
        env.start()
        self.addCleanup(env.stop)
        self.pc = SimpleNamespace(
            world_size=4, local_world_size=2, world_rank=2, local_rank=0
        )

    def test_control_plane_addresses_and_new_underlay(self):
        self.assertEqual(scr_vip.read_topology(4, 2), {0: "22.0.1.2", 1: "22.0.1.3"})
        self.rows["model-rank-1"]["real_ip"] = "10.1.2.3"
        self.path.write_text(json.dumps(self.rows))
        self.assertEqual(scr_vip.read_topology(4, 2, external=True)[1], "10.1.2.3")
        with patch.object(scr_vip, "validate_device") as check:
            self.assertEqual(scr_vip.internal_ip(self.pc, "10.0.0.3"), "22.0.1.3")
            check.assert_called_once_with("22.0.1.3")

    def test_worker_node_topology_must_match_gpu_topology(self):
        for name, wrong in (("RANK_SIZE", "1"), ("RANK_ID", "0"), ("RANK_SIZE", "")):
            with self.subTest(name=name, value=wrong), patch.dict(
                os.environ, {name: wrong}
            ), patch.object(scr_vip, "validate_device") as device:
                with self.assertRaisesRegex(ValueError, name):
                    scr_vip.topology(self.pc, wait=True)
                device.assert_not_called()

    def test_incomplete_duplicate_and_unmerged_maps_fail(self):
        bad_maps = [
            {},
            {"model-part0": self.rows["model-rank-0"]},
            {**self.rows, "other-rank-0": self.rows["model-rank-0"]},
            {
                "model-rank-0": self.rows["model-rank-0"],
                "model-rank-1": self.rows["model-rank-0"],
            },
            {"model-rank-0": {"ip": "10.0.0.2", "real_ip": "10.0.0.2"}},
        ]
        for rows in bad_maps:
            with self.subTest(rows=rows):
                self.path.write_text(json.dumps(rows))
                with self.assertRaises(ValueError):
                    scr_vip.read_topology(4, 2)

    def test_normal_and_single_node_do_not_read_vip(self):
        with patch.dict(os.environ, {"SCR_PHASE": "normal"}):
            self.assertEqual(scr_vip.internal_ip(self.pc, "10.0.0.3"), "10.0.0.3")
        self.pc.local_world_size = 4
        self.assertFalse(scr_vip.enabled(self.pc))

    def test_missing_interface_address_fails(self):
        with patch.object(scr_vip.subprocess, "check_output", return_value="[]"):
            with self.assertRaisesRegex(RuntimeError, "does not own"):
                scr_vip.validate_device("22.0.1.3")

    def test_interface_alone_does_not_prove_network_ready(self):
        addresses = json.dumps([{"addr_info": [{"local": "22.0.1.3"}]}])
        with patch.object(
            scr_vip.subprocess, "check_output", return_value=addresses
        ), patch.object(scr_vip.Path, "is_socket", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "readiness socket"):
                scr_vip.validate_device("22.0.1.3")

    def test_non_uniform_world_fails(self):
        with self.assertRaises(ValueError):
            scr_vip.read_topology(5, 2)


if __name__ == "__main__":
    unittest.main()

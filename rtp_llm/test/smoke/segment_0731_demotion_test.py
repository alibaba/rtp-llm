"""Verify DEVICE demotion with :segment_0731_demote_host and :segment_0731_demote_disk."""

from __future__ import annotations

import json
import logging
import os
import select
import subprocess
import sys
import time
import unittest
from pathlib import Path

from rtp_llm.test.smoke.deepseek_v4_tail_reuse_4layer_test import (
    DeepSeekV4TailReuseFourLayerTest,
)
from rtp_llm.test.smoke.segment_0731_test import Segment0731Mixin


class DemotionLaunch(DeepSeekV4TailReuseFourLayerTest):
    def _start_servers(self):
        self.extra_prefill_args += (
            " --enable_device_cache 1 --kv_cache_mem_mb 256"
            " --block_tree_device_evict_low_watermark_ratio 0.000000000001"
            " --block_tree_device_evict_high_watermark_ratio 0.000000000002"
        )
        path = self.output_dir / "segment_run_config.json"
        config = json.loads(path.read_text())
        config.update(
            device_pool_mb=256,
            route="DEVICE_DEMOTION",
            demotion_watermarks=[1e-12, 2e-12],
        )
        path.write_text(json.dumps(config, indent=2))
        super()._start_servers()


class Segment0731DemotionTest(Segment0731Mixin, DemotionLaunch):
    # The first-base sample does not depend on an ancestor under eager eviction.
    selected_g = 120

    def test_device_endpoint_demotes_before_lower_tier_reuse(self):
        self.assertIn(self.local_cache_tier, ("host", "disk"))
        self.assertEqual(self.profile, "full")
        branch = next(b for b in self.fixture["branches"] if b["G"] == self.selected_g)
        source = self.fixture["sources"][branch["source"]]
        g = branch["G"]
        self.assertEqual(branch["candidate_floor"][self.profile], g)
        cold, _, _ = self._segment_request("demotion_cold", branch["text"], False)
        self.assertEqual(cold["aux_info"]["prefill_total_reuse_len"], 0)
        prepared, _, _ = self._segment_request("demotion_source", source["text"], True)
        self.assertEqual(prepared["input_ids"][0], source["ids"])
        self.assertEqual(prepared["aux_info"]["prefill_total_reuse_len"], 0)
        events = []
        pending = b""
        deadline = time.monotonic() + 120
        marker = "event=prefix_cache_observation payload="
        # Follow the canonical prefill log and await the exact endpoint's settled release.
        # The timeout bounds observation; it is not a delay used to assume completion.
        with subprocess.Popen(
            ["tail", "-n", "+1", "-f", str(self.prefill.log_file_path)],
            stdout=subprocess.PIPE,
        ) as follower:
            try:
                self.assertIsNotNone(follower.stdout)
                while time.monotonic() < deadline:
                    ready, _, _ = select.select(
                        [follower.stdout], [], [], max(0, deadline - time.monotonic())
                    )
                    self.assertTrue(
                        ready, "no completed DEVICE release before deadline"
                    )
                    chunk = os.read(follower.stdout.fileno(), 65536)
                    self.assertTrue(
                        chunk, "prefill log follower closed before settlement"
                    )
                    pending += chunk
                    lines = pending.split(b"\n")
                    pending = lines.pop()
                    for line in lines:
                        text = line.decode(errors="replace")
                        if marker in text:
                            event, _ = json.JSONDecoder().raw_decode(
                                text.split(marker, 1)[1]
                            )
                            events.append(event)
                    points = {
                        (e["instance"], p["key"])
                        for e in events
                        if e.get("kind") == "plan" and e.get("target_tier") == "DEVICE"
                        for p in e.get("points", [])
                        if p["pos"] == g
                    }
                    released = [
                        e
                        for e in events
                        if e.get("kind") == "interval_end"
                        and (e.get("instance"), e.get("key")) in points
                        and e.get("interval") == "tier"
                        and e.get("medium") == "DEVICE"
                        and e.get("trigger") == "eviction"
                        and e.get("effect") == "tier_release"
                    ]
                    if released:
                        break
                else:
                    self.fail("selected endpoint never completed DEVICE eviction")
            finally:
                follower.terminate()
                follower.wait(timeout=5)
                (self.output_dir / "demotion_before_warm.json").write_text(
                    json.dumps(events, indent=2)
                )
        publications = [
            e
            for e in events
            if e.get("kind") == "publication" and e.get("insert_attempted")
        ]
        self.assertTrue(publications, "no successful source publication attempt")
        self.assertTrue(
            all(e["target_tier"] == "DEVICE" for e in publications),
            "direct lower-tier save detected",
        )
        selected = [e for e in publications if e.get("internal_fork_pos") == g]
        self.assertTrue(selected, "source did not publish selected segment point")
        self.assertTrue(
            any(
                e.get("internal_fork_ready") is True and e["accepted_resources"] > 0
                for e in selected
            )
        )
        warm, _, traces = self._segment_request("demotion_warm", branch["text"], True)
        self.assertEqual(warm["input_ids"], cold["input_ids"])
        self.assertEqual(warm["output_ids"], cold["output_ids"])
        self.assertEqual(warm["aux_info"]["prefill_total_reuse_len"], g)
        self.assertEqual(int(traces[0]["extra"]["prefix_length"]), g)
        # _request checks HOST/DISK attribution, while _segment_request checks matching-G direct tensors.
        (self.output_dir / "demotion_result.json").write_text(
            json.dumps(
                {
                    "G": g,
                    "source_case": source["case"],
                    "source_publication": "demotion_source",
                    "source_input_ids": prepared["input_ids"],
                    "cold_aux": cold["aux_info"],
                    "warm_aux": warm["aux_info"],
                    "device_release_events": released,
                    "source_publications": publications,
                    "numeric_oracle": "existing matching-G direct assertions",
                    "capacity_claim": "256 MiB requested; actual pool limits require startup/metric audit",
                    "pressure_claim": "completed selected-key eviction; aggressive watermark, not exhaustion",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    tier, cp, profile, capacity = sys.argv[1:5]
    if cp != "1":
        raise SystemExit("Demotion proof currently covers CP1 only")
    for arg in sys.argv[5:]:
        if arg.startswith("--profile="):
            profile = arg.split("=", 1)[1]
        elif arg.startswith("--capacity-mb="):
            capacity = arg.split("=", 1)[1]
        elif arg.startswith("--fork="):
            Segment0731DemotionTest.selected_g = int(arg.split("=", 1)[1])
        else:
            raise SystemExit("Unsupported argument: " + arg)
    Segment0731DemotionTest.local_cache_tier = tier
    Segment0731DemotionTest.profile = profile
    Segment0731DemotionTest.capacity_mb = int(capacity)
    logging.basicConfig(level=logging.INFO)
    unittest.main(
        argv=[
            sys.argv[0],
            "Segment0731DemotionTest.test_device_endpoint_demotes_before_lower_tier_reuse",
        ]
    )

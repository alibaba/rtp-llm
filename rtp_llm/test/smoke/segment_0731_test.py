"""Actual 0731 token forks, matching-G numerical checks, and local-tier attribution."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import unittest
from pathlib import Path
from shlex import quote

import torch

from rtp_llm.test.smoke.deepseek_v4_tail_reuse_4layer_cp4_test import (
    DeepSeekV4TailReuseFourLayerCP4Test,
)
from rtp_llm.test.smoke.deepseek_v4_tail_reuse_4layer_test import (
    DeepSeekV4TailReuseFourLayerTest,
)


class Segment0731Mixin:
    profile = "full"
    capacity_mb = 2048
    pressure = False
    cp_size = 1
    selected_g = None
    control_mode = None

    def _start_servers(self):
        fixture = json.loads(
            Path(__file__).with_name("segment_0731_tokens.json").read_text()
        )
        self.fixture = fixture
        for name, expected in fixture["provenance"]["sha256"].items():
            self.assertEqual(
                hashlib.sha256((self.model_path / name).read_bytes()).hexdigest(),
                expected,
            )
        rules = self.output_dir / "segment_rules.json"
        rules_payload = fixture["profiles"][self.profile]
        if self.control_mode == "matched_segment":
            source = next(
                source
                for source in fixture["sources"]
                if source["case"] == "tools_thinking"
            )
            rules_payload = {"before": [source["ids"][273:275]]}
        rules.write_text(json.dumps(rules_payload))
        self.extra_prefill_args = (
            " --enable_prefill_common_prefix_recovery 0"
            f" --memory_cache_size_mb {self.capacity_mb}"
            f" --disk_cache_size_mb {self.capacity_mb}"
        )
        if self.control_mode == "matched_common":
            self.extra_prefill_args += " --enable_prefill_common_prefix_recovery 1"
        if self.profile != "disabled" or self.control_mode == "matched_segment":
            self.extra_prefill_args += " --prefill_segment_rules_file " + quote(
                str(rules)
            )
        self.probe_options = {
            "segment_capture": True,
            "segment_source_ends": [len(source["ids"]) for source in fixture["sources"]]
            + ([self.selected_g] if self.selected_g is not None else []),
        }
        if self.control_mode in ("matched_common", "matched_segment"):
            self.probe_options["segment_source_ends"] = [278, 383]
            self.probe_options["segment_capture_resumed_source"] = 383
        (self.output_dir / "segment_run_config.json").write_text(
            json.dumps(
                {
                    "profile": self.profile,
                    "rules": rules_payload,
                    "control_mode": self.control_mode,
                    "capacity_mb_per_lower_tier": self.capacity_mb,
                    "device_pool_mb": 4096,
                    "cp_size": self.cp_size,
                    "tier": self.local_cache_tier,
                    "pressure_requested": self.pressure,
                    "tail_reuse": True,
                    "common_prefix_recovery": self.control_mode == "matched_common",
                },
                indent=2,
            )
        )
        super()._start_servers()

    def _segment_request(self, label, text, reuse):
        if self.cp_size == 4:
            actual, probes, traces, _ = self._request_cp4(label, text, reuse)
        else:
            actual, probe, trace, _ = self._request(label, text, reuse)
            probes, traces = {0: probe}, {0: trace}
        for rank, trace in traces.items():
            suffix = "tail128" if self.cp_size == 4 else "last"
            g = int(trace["extra"]["prefix_length"])
            n = len(actual["input_ids"][0])
            rank_part = f"rank{rank}_" if self.cp_size == 4 else ""
            direct = torch.load(
                self.probe_dir / f"direct_reference_{rank_part}{n}_g{g}.pt",
                weights_only=False,
            )
            for layer in range(4):
                torch.testing.assert_close(
                    trace["tensors"][f"layer{layer:02d}_{suffix}"],
                    direct["layers"][layer].float(),
                    rtol=0.002,
                    atol=0.002,
                )
            for key, reference in (
                ("lm_last_hidden", "final_last"),
                ("lm_logits_last", "logits_last"),
            ):
                torch.testing.assert_close(
                    trace["tensors"][key],
                    direct[reference].float(),
                    rtol=0.002,
                    atol=0.002,
                )
        return actual, probes, traces

    def test_matched_source_control(self):
        self.assertEqual(self.cp_size, 1)
        self.assertEqual(self.local_cache_tier, "device")
        index, source = next(
            (i, item)
            for i, item in enumerate(self.fixture["sources"])
            if item["case"] == "tools_thinking"
        )
        branch = next(
            item
            for item in self.fixture["branches"]
            if item["source"] == index and item["G"] == 273
        )
        query_ids = source["ids"][:273] + branch["suffix_ids"]
        prime_ids = query_ids[:-1] + self.tokenizer.encode("X")
        self.assertEqual(len(prime_ids), 278)
        prime_text = self.tokenizer.decode(prime_ids)
        self.assertEqual(self.tokenizer.encode(prime_text), prime_ids)
        self.assertNotEqual(prime_ids[-1], query_ids[-1])
        self.assertNotEqual(prime_ids[273], source["ids"][273])
        prime, _, _ = self._segment_request("matched_prime", prime_text, True)
        producer, snapshots, source_traces = self._segment_request(
            "matched_source", source["text"], True
        )
        self.assertEqual(producer["input_ids"][0], source["ids"])
        self.assertEqual(producer["aux_info"]["prefill_total_reuse_len"], 256)
        warm, _, traces = self._segment_request("matched_warm", branch["text"], True)
        self.assertEqual(warm["input_ids"][0], query_ids)
        self.assertEqual(warm["aux_info"]["prefill_total_reuse_len"], 273)
        for field in (
            "prefill_memory_reuse_len",
            "prefill_disk_reuse_len",
            "prefill_remote_reuse_len",
        ):
            self.assertEqual(warm["aux_info"][field], 0)
        (self.output_dir / "matched_control.json").write_text(
            json.dumps(
                {
                    "mode": self.control_mode,
                    "prime_ids": prime_ids,
                    "producer_ids": source["ids"],
                    "query_ids": query_ids,
                    "prime_aux": prime["aux_info"],
                    "producer_aux": producer["aux_info"],
                    "warm_aux": warm["aux_info"],
                    "matching_g_reference_passed": True,
                },
                indent=2,
            )
        )

    def test_old_tail_control(self):
        """Keep split-vs-unsplit failure visible after recording the paired control."""
        self.assertEqual(self.profile, "disabled")
        self.assertEqual(self.cp_size, 1)
        g = self.selected_g
        self.assertIsNotNone(g)
        source_index, source = next(
            (i, source)
            for i, source in enumerate(self.fixture["sources"])
            if source["case"] == ("hca_aligned" if g == 128 else "tools_thinking")
        )
        branch = next(
            branch
            for branch in self.fixture["branches"]
            if branch["source"] == source_index and branch["G"] == g
        )
        prefix_ids = source["ids"][:g]
        prefix_text = self.tokenizer.decode(prefix_ids)
        self.assertEqual(self.tokenizer.encode(prefix_text), prefix_ids)
        cold, _, cold_traces = self._segment_request(
            "control_cold", branch["text"], False
        )
        repeated, _, repeated_traces = self._segment_request(
            "control_cold_repeat", branch["text"], False
        )
        producer, _, _ = self._segment_request("control_tail_source", prefix_text, True)
        warm, _, warm_traces = self._segment_request(
            "control_tail_warm", branch["text"], True
        )
        self.assertEqual(producer["input_ids"][0], prefix_ids)
        self.assertEqual(warm["aux_info"]["prefill_total_reuse_len"], g)
        self.assertEqual(cold["input_ids"], warm["input_ids"])
        self.assertEqual(cold["output_ids"], warm["output_ids"])
        comparisons = {}
        for label, left in (
            ("repeat_cold", repeated_traces),
            ("tail_warm", warm_traces),
        ):
            comparisons[label] = {}
            for key, reference in cold_traces[0]["tensors"].items():
                value = left[0]["tensors"][key]
                delta = (value.double() - reference.double()).abs()
                comparisons[label][key] = {
                    "max_abs": float(delta.max()) if delta.numel() else 0.0,
                    "mean_abs": float(delta.mean()) if delta.numel() else 0.0,
                    "strict_failed": int(
                        (delta > 0.002 + 0.002 * reference.double().abs()).sum()
                    ),
                }
        (self.output_dir / "old_tail_control.json").write_text(
            json.dumps(
                {
                    "G": g,
                    "producer_ids": prefix_ids,
                    "cold_aux": cold["aux_info"],
                    "repeat_aux": repeated["aux_info"],
                    "warm_aux": warm["aux_info"],
                    "comparisons": comparisons,
                    "matching_g_direct_reference": "strictly checked for all four requests",
                },
                indent=2,
            )
        )
        for trace in (cold_traces[0], repeated_traces[0]):
            self.assertEqual(int(trace["extra"]["prefix_length"]), 0)
        for key, reference in cold_traces[0]["tensors"].items():
            torch.testing.assert_close(
                repeated_traces[0]["tensors"][key], reference, rtol=0.002, atol=0.002
            )
        for key in ("lm_last_hidden", "lm_logits_last"):
            torch.testing.assert_close(
                warm_traces[0]["tensors"][key],
                cold_traces[0]["tensors"][key],
                rtol=0.002,
                atol=0.002,
            )

    def test_exact_segment_forks(self):
        case = "tools_thinking_cross_cp4" if self.cp_size == 4 else "tools_thinking"
        chosen = {case} if self.cp_size == 4 else {case, "hca_aligned"}
        records = []
        for source_index, source in enumerate(self.fixture["sources"]):
            if source["case"] not in chosen:
                continue
            branches = [
                branch
                for branch in self.fixture["branches"]
                if branch["source"] == source_index
                and (self.selected_g is None or branch["G"] == self.selected_g)
            ]
            if not branches:
                continue
            cold_requests = {}
            self.assertEqual(self.tokenizer.encode(source["text"]), source["ids"])
            for branch in branches:
                g = branch["G"]
                label = f"{source['case']}_g{g}"
                ids = source["ids"][:g] + branch["suffix_ids"]
                self.assertEqual(self.tokenizer.encode(branch["text"]), ids)
                cold, _, cold_traces = self._segment_request(
                    label + "_cold", branch["text"], False
                )
                self.assertEqual(cold["input_ids"][0], ids)
                for trace in cold_traces.values():
                    self.assertEqual(int(trace["extra"]["prefix_length"]), 0)
                cold_requests[g] = (cold, cold_traces)
            prepared, snapshots, _ = self._segment_request(
                source["case"] + "_source", source["text"], True
            )
            self.assertEqual(prepared["input_ids"][0], source["ids"])
            for branch in branches:
                g = branch["G"]
                label = f"{source['case']}_g{g}"
                ids = source["ids"][:g] + branch["suffix_ids"]
                cold, cold_traces = cold_requests[g]
                actual, _, traces = self._segment_request(
                    label + "_warm", branch["text"], True
                )
                self.assertEqual(actual["input_ids"][0], ids)
                aux = actual["aux_info"]
                recovered = (
                    aux["local_reuse_len"]
                    if self.cp_size == 4
                    else aux["prefill_total_reuse_len"]
                )
                self.assertLessEqual(recovered, g)
                if self.local_cache_tier == "device":
                    for field in (
                        "prefill_memory_reuse_len",
                        "prefill_disk_reuse_len",
                        "prefill_remote_reuse_len",
                    ):
                        self.assertEqual(aux[field], 0, field)
                candidate = branch["candidate_floor"][self.profile]
                if self.profile == "full" and not self.pressure:
                    self.assertEqual(recovered, candidate, label)
                self.assertEqual(actual["output_ids"], cold["output_ids"])
                numeric = {}
                owner_count = 0
                for rank, trace in traces.items():
                    self.assertEqual(int(trace["extra"]["prefix_length"]), recovered)
                    for key in ("lm_last_hidden", "lm_logits_last"):
                        warm = trace["tensors"][key]
                        reference = cold_traces[rank]["tensors"][key]
                        self.assertTrue(torch.isfinite(warm).all())
                        delta = (warm.double() - reference.double()).abs()
                        numeric[f"rank{rank}_{key}"] = {
                            "max_abs": float(delta.max()) if delta.numel() else 0.0,
                            "mean_abs": float(delta.mean()) if delta.numel() else 0.0,
                            "strict_failed": int(
                                (delta > 0.002 + 0.002 * reference.double().abs()).sum()
                            ),
                            "reference": "whole-prefill G0 diagnostic; matching-G reference asserted separately",
                        }
                    owner_count += int(trace["tensors"]["lm_last_hidden"].numel() > 0)
                self.assertEqual(owner_count, 1)
                snapshot_bytes = 0
                saved = {}
                for rank, snapshot in snapshots.items():
                    for key, row in snapshot["rows"].items():
                        positions = row["current_save_positions"].reshape(-1).tolist()
                        saved[f"rank{rank}_{key}"] = positions
                        for field in ("current_save_values", "current_save_scales"):
                            tensor = row.get(field)
                            if tensor is not None:
                                snapshot_bytes += tensor.numel() * tensor.element_size()
                observations = []
                segment_candidates = []
                marker = "event=prefix_cache_observation payload="
                for log in self.output_dir.rglob("*.log"):
                    for line in log.read_text(errors="replace").splitlines():
                        if "event=prefix_segment_candidate " in line:
                            segment_candidates.append(
                                line.split("event=prefix_segment_candidate ", 1)[1]
                            )
                        if marker in line:
                            event, _ = json.JSONDecoder().raw_decode(
                                line.split(marker, 1)[1]
                            )
                            observations.append(event)
                (self.output_dir / (label + "_observations.json")).write_text(
                    json.dumps(observations, indent=2)
                )
                self.assertTrue(
                    observations, "prefix observation sink produced no events"
                )
                records.append(
                    {
                        "case": label,
                        "source_publication": source["case"] + "_source",
                        "segment_candidates_raw": sorted(set(segment_candidates)),
                        "cold_method": "reuse_cache=false; all cold requests precede source publication",
                        "G": g,
                        "candidate_floor": candidate,
                        "observed_G": recovered,
                        "aux_info": aux,
                        "numeric_max_abs": numeric,
                        "saved_positions_by_group": saved,
                        "captured_snapshot_bytes": snapshot_bytes,
                        "transfer_bytes": None,
                        "peak_live_pins": None,
                        "peak_store_pins": None,
                        "evicted_nodes": None,
                        "capacity_store_failures": None,
                        "measurement_gaps": [
                            "snapshot bytes are not transfer bytes",
                            "pin/eviction/store-failure counters require runtime metrics",
                        ],
                    }
                )
                (self.output_dir / "segment_results.json").write_text(
                    json.dumps(records, indent=2)
                )
        self.assertTrue(records, "no selected strict fork executed")


class Segment0731Test(Segment0731Mixin, DeepSeekV4TailReuseFourLayerTest):
    pass


class Segment0731CP4Test(Segment0731Mixin, DeepSeekV4TailReuseFourLayerCP4Test):
    cp_size = 4


if __name__ == "__main__":
    tier, cp, profile, capacity = sys.argv[1:5]
    del sys.argv[1:5]
    fixture_class = Segment0731CP4Test if cp == "4" else Segment0731Test
    method = "test_exact_segment_forks"
    remaining = []
    for argument in sys.argv[1:]:
        if argument in ("--control=matched_common", "--control=matched_segment"):
            fixture_class.control_mode = argument.split("=", 1)[1]
            method = "test_matched_source_control"
        elif argument == "--control=old_tail":
            method = "test_old_tail_control"
        elif argument.startswith("--profile="):
            profile = argument.split("=", 1)[1]
        elif argument.startswith("--capacity-mb="):
            capacity = argument.split("=", 1)[1]
        elif argument.startswith("--fork="):
            fixture_class.selected_g = int(argument.split("=", 1)[1])
        else:
            remaining.append(argument)
    sys.argv[1:] = remaining
    fixture_class.local_cache_tier = tier
    fixture_class.profile = profile
    fixture_class.capacity_mb = int(capacity)
    fixture_class.pressure = int(capacity) < 2048
    logging.basicConfig(level=logging.INFO)
    unittest.main(
        argv=[
            sys.argv[0],
            fixture_class.__name__ + "." + method,
            *sys.argv[1:],
        ]
    )

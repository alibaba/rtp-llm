"""Combined CI keeps both original request audits under one P/D lifecycle."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.test.smoke.kimi_k3 import k31_two_host_smoke as smoke
from rtp_llm.test.smoke.kimi_k3.k31_full_smoke import (
    cache_pressure_plan,
    request_budget,
)
from rtp_llm.test.smoke.kimi_k31_bazel import k31_ci


class SmokeFlowTest(unittest.TestCase):
    def test_readiness_checks_complete_capacity_instead_of_the_95_page_fixture(self):
        from rtp_llm.test.smoke.kimi_k3 import k31_smoke_capacity

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "logs").mkdir()
            (directory / "instance.json").write_text(
                json.dumps(dict(manager_pid=123, manager_start_ticks=456))
            )
            config = k31_ci.smoke_config(
                dict(
                    run_id="ready",
                    prefill=dict(repo_root=str(directory), ip="127.0.0.1", port=25000),
                    decode={},
                ),
            )

            def geometry(linear_blocks):
                return "".join(
                    f"CP sharded KV cache enabled: cp_rank={rank}, cp_size=8, "
                    "block_size=1024, virtual_block_size=8192\n"
                    f"[RANK {rank}] cache manager initialized with config CacheConfig{{\n"
                    "  layer_num=93\n  layer_all_num=94\n"
                    "  seq_size_per_block=1024\n  local_kv_cache_sharded=true\n"
                    "  reuse_cache_tail_blocks=5\n"
                    f"  group_block_nums=140,140,{linear_blocks}\n"
                    "    kernel_seq_size_per_block=128\n"
                    "  LinearKVCacheSpec{\n    cache_key_token_stride=1024\n  }\n"
                    "  group_tags=[full,indexer_kv,linear]\n"
                    "  layers_by_group=[[3,93],[3],[0]]\n"
                    for rank in range(8)
                )

            fake_worker = SimpleNamespace(
                pd_endpoints=lambda *_: [], listening_tcp_ports=lambda *_: set()
            )
            receipts = {str(rank): dict(pid=100 + rank, gib="8") for rank in range(8)}
            with (
                patch.object(sys, "path", list(sys.path)),
                patch.dict(sys.modules, {"worker": fake_worker}),
                patch.object(smoke.launcher, "run_dir", return_value=directory),
                patch.object(smoke.launcher, "start_ticks", return_value=456),
                patch.object(
                    smoke.launcher, "rank_workspace_receipts", return_value=receipts
                ),
                patch.object(
                    k31_smoke_capacity, "fastsafetensors_proof", return_value={}
                ),
            ):
                engine = directory / "logs/engine.log"
                engine.write_text(geometry(128))
                ready = smoke.launcher.status(config, "prefill")
                self.assertTrue(ready["healthy"], ready)
                self.assertEqual(
                    ready["runtime_geometry"]["usable_pages"]["0"]["full"], 139
                )
                engine.write_text(geometry(7))
                rejected = smoke.launcher.status(config, "prefill")
                self.assertFalse(rejected["healthy"])
                self.assertIn("capacity", rejected["error"])
                # Traditional regular keeps its own capacity contract even when
                # its configured maximum sequence length also admits long input.
                traditional = dict(config, k31_smoke_stage="regular")
                with patch.object(
                    k31_smoke_capacity, "cache_geometry", return_value={}
                ) as geometry_check:
                    self.assertTrue(smoke.launcher.status(traditional, "prefill")["healthy"])
                self.assertFalse(geometry_check.call_args.kwargs["long_case"])

    def test_default_config_keeps_regular_topology_and_query_coverage(self):
        base = dict(run_id="combined", prefill={}, decode={"test_block_num": 192})
        config = k31_ci.smoke_config(base)
        self.assertNotIn("k31_smoke_stage", config)
        self.assertEqual(config["decode_dp"], 2)
        self.assertEqual(config["prefill"]["linear_pool_blocks"], 128)
        self.assertEqual(config["prefill"]["reuse_cache_tail_blocks"], 5)
        self.assertNotIn("last_n_checkpoints", config)
        self.assertNotIn("reuse_cache_tail_blocks", config["decode"])
        self.assertEqual(config["decode"]["test_block_num"], 272)
        self.assertEqual(config["decode"]["linear_pool_blocks"], 192)
        self.assertEqual(config["prefill"]["prefill_token_capacity"], 65536)
        self.assertNotIn("allocator_large_segment_size_mb", config["prefill"])
        self.assertNotIn(
            "memory_cache_max_descriptors_per_transfer_batch", config["prefill"]
        )
        self.assertNotIn(
            "memory_cache_max_descriptors_per_transfer_batch",
            smoke.stage_config(base, "regular")["prefill"],
        )
        self.assertNotIn(
            "allocator_large_segment_size_mb",
            smoke.stage_config(base, "regular")["prefill"],
        )
        self.assertNotIn("request_capacity", config["decode"])
        self.assertNotIn("decode_capture_batches", config["decode"])
        self.assertEqual(config["prefill_kv_workspace_budget_gib"], 8)
        plan = cache_pressure_plan(139, 8192)
        self.assertGreater(plan["requests"] * plan["pages_per_probe"], 139)
        self.assertEqual(request_budget("smoke", plan["requests"])["formal"], 267)
        base["prefill"]["prefill_token_capacity"] = 8192
        with self.assertRaisesRegex(ValueError, "64K"):
            k31_ci.smoke_config(base)

    def test_prefill_retention_does_not_select_a_different_request_profile_or_decode_topology(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            worker = self.worker(Path(directory))
            config = k31_ci.smoke_config(
                dict(run_id="retention", prefill={}, decode={})
            )
            worker.host = config["prefill"]
            options = worker.launcher_options()
            self.assertEqual(
                options[options.index("--reuse-cache-tail-blocks") + 1], "5"
            )
            self.assertNotIn("--last-n-checkpoints", options)
            worker.host = config["decode"]
            self.assertNotIn("--reuse-cache-tail-blocks", worker.launcher_options())

    def worker(self, directory):
        worker = k31_ci.K31Worker(None)
        self.addCleanup(worker.locks.close)
        worker.task = directory
        worker.workspace = directory / "runfiles" / "rtp_llm"
        worker.config = {"external_revision": "a" * 40}
        return worker

    def test_retention_uses_tp0_allocation_and_requires_every_rank_alias_snapshot(self):
        span, prompt, chunk = 8192, 1_048_576, 65536
        selected = {(124 + index) * span: index + 1 for index in range(5)}
        events = {rank: [] for rank in range(8)}
        events[0].append(
            dict(
                event="linear_prefill_allocated",
                request_id=17,
                group_tag="linear",
                time_ns=1,
                prompt_tokens=prompt,
                initial_reuse_tokens=0,
                configured_tail_blocks=5,
                page_tokens=1024,
                checkpoint_tokens=span,
                planned_new_blocks=5,
                owned_state_blocks=5,
                has_separate_tail=False,
                borrowed_block=5,
            )
        )
        previous, tick = 0, 10
        for prefix in range(0, prompt, chunk):
            end = prefix + chunk
            writes = [
                dict(block=block, tokens=point, retained=True)
                for point, block in selected.items()
                if prefix < point <= end
            ]
            if end not in selected:
                writes.append(dict(block=5, tokens=end, retained=False))
            if previous and previous in [row["block"] for row in writes]:
                for rank in events:
                    events[rank].append(
                        dict(
                            event="linear_prefill_conv_snapshot_submitted",
                            world_rank=rank,
                            time_ns=tick,
                            rows=[
                                dict(
                                    prefix_tokens=prefix,
                                    query_tokens=chunk,
                                    input_block=previous,
                                )
                            ],
                        )
                    )
            events[0].append(
                dict(
                    event="linear_prefill_round_completed",
                    request_id=17,
                    group_tag="linear",
                    time_ns=tick + 1,
                    prefix_tokens=prefix,
                    end_tokens=end,
                    input_block=previous,
                    output_block=writes[-1]["block"],
                    writes=writes,
                )
            )
            previous, tick = writes[-1]["block"], tick + 10
        events[0].append(
            dict(
                event="linear_cache_publish_candidates",
                request_id=17,
                group_tag="linear",
                time_ns=tick,
                target_tier="HOST",
                prompt_tokens=prompt,
                checkpoints=[
                    dict(tokens=point, block=block) for point, block in selected.items()
                ],
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            worker = self.worker(Path(directory))
            worker.host = dict(reuse_cache_tail_blocks=5, prefill_token_capacity=chunk)
            (worker.task / "geometry.json").write_text(
                json.dumps(dict(page_size=1024, checkpoint_span=span))
            )
            proof = worker.audit_prefill_retention(events, require_one_million=True)
            self.assertEqual(set(proof), {"0"})
            self.assertTrue(proof["0"]["cold_one_million_verified"])
            events[7] = []
            with self.assertRaisesRegex(ValueError, "rank7.*conv input"):
                worker.audit_prefill_retention(events, require_one_million=True)

    def test_long_retention_matches_a_cold_seed_and_an_exact_one_million_continuation(
        self,
    ):
        lengths = {
            "seed": (17, 1_048_501, 0),
            "continuation": (18, 1_048_576, 1_040_384),
        }
        proof = dict(
            pd_native_route={
                name: dict(request_id=values[0]) for name, values in lengths.items()
            },
            cases={
                name: dict(input_len=values[1], prefill_total_reuse_len=values[2])
                for name, values in lengths.items()
            },
        )
        rows = [
            dict(
                request_id=identity,
                prompt_tokens=length,
                completed_tokens=length,
                initial_reuse_tokens=reuse,
            )
            for identity, length, reuse in lengths.values()
        ]
        retention = {"0": {"requests": rows}}
        result = k31_ci.audit_long_retention(proof, retention)
        self.assertEqual(result["seed"]["prompt_tokens"], 1_048_501)
        self.assertEqual(result["continuation"]["prompt_tokens"], 1_048_576)
        for field in (
            "request_id",
            "prompt_tokens",
            "completed_tokens",
            "initial_reuse_tokens",
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(retention)
                changed["0"]["requests"][1][field] += 1
                with self.assertRaisesRegex(ValueError, "continuation"):
                    k31_ci.audit_long_retention(proof, changed)

    def test_both_request_phases_run_without_starting_or_stopping_a_service(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = self.worker(Path(directory))
            requests = worker.task / "requests"
            requests.mkdir()
            calls = []

            def phase(_message):
                current = worker._request_phase
                calls.append(current)
                verdict = dict(
                    passed=True,
                    errors=[],
                    native_K31_MTP=True,
                    counts=(
                        dict(formal=267, preparation=42)
                        if current == "regular"
                        else dict(formal=1, preparation=1)
                    ),
                )
                name = "execution-audit.json"
                if current == "regular":
                    verdict["complete_smoke_passed"] = True
                    worker._regular_events = tuple(
                        {rank: [dict(time_ns=1)] for rank in range(8)} for _ in range(2)
                    )
                    (requests / "accuracy.json").write_text('{"passed": true}')
                else:
                    verdict["long_smoke_passed"] = True
                    name = "long-execution-audit.json"
                (requests / name).write_text(json.dumps(verdict))
                return {"requests_passed": True}

            with (
                patch.object(worker, "run_request_phase", side_effect=phase),
                patch.object(worker, "record_device_memory") as memory,
                patch.object(worker, "start") as start,
                patch.object(worker, "close") as close,
            ):
                self.assertTrue(worker.requests({"timeout_s": 60})["requests_passed"])
            self.assertEqual(calls, ["regular", "long"])
            self.assertEqual(
                [call.args[0] for call in memory.call_args_list],
                ["regular-complete", "long-complete"],
            )
            start.assert_not_called()
            close.assert_not_called()
            verdict = k31_ci.check_ci_result(requests / "accuracy.json")
            self.assertTrue(verdict["passed"])
            self.assertEqual(verdict["counts"], dict(formal=268, preparation=43))
            verdict["phases"].pop("long")
            (requests / "execution-audit.json").write_text(json.dumps(verdict))
            self.assertFalse(
                k31_ci.check_ci_result(requests / "accuracy.json")["passed"]
            )

    def test_memory_receipts_cover_only_the_eight_owned_gpus(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = self.worker(Path(directory))
            worker.role = "prefill"
            worker.gpu_uuids = ["GPU-" + str(rank) for rank in range(8)]
            raw = "\n".join(f"{gpu}, 280000, 260000, 20000" for gpu in worker.gpu_uuids)
            with patch.object(
                k31_ci.subprocess, "check_output", return_value=raw
            ) as query:
                worker.record_device_memory("ready")
            self.assertIn("--id=" + ",".join(worker.gpu_uuids), query.call_args.args[0])
            receipt = json.loads(
                (worker.task / "runtime-memory-ready.json").read_text()
            )
            self.assertEqual(set(receipt["gpus"]), set(worker.gpu_uuids))
            self.assertIn("excludes_reusable_torch", receipt["scope"])
            with (
                patch.object(
                    k31_ci.subprocess,
                    "check_output",
                    return_value=raw + "\nGPU-other, 1, 0, 1",
                ),
                self.assertRaisesRegex(ValueError, "owned GPUs"),
            ):
                worker.record_device_memory("ready")

    def test_long_events_exclude_regular_requests_and_reject_changed_history(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = self.worker(Path(directory))
            before = {rank: [dict(time_ns=1, request_id=10)] for rank in range(8)}
            after = {
                rank: rows + [dict(time_ns=2, request_id=20)]
                for rank, rows in before.items()
            }
            worker._regular_events = (before, before)
            for events in worker.long_phase_events(after, after):
                self.assertTrue(
                    all(
                        [row["request_id"] for row in rows] == [20]
                        for rows in events.values()
                    )
                )
            changed = copy.deepcopy(after)
            changed[4][0]["request_id"] = 99
            with self.assertRaisesRegex(ValueError, "changed across"):
                worker.long_phase_events(after, changed)

    def test_both_commands_keep_the_same_geometry_and_pin_the_long_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = self.worker(Path(directory))
            base = dict(
                run_id="complete",
                external_revision="a" * 40,
                prefill=dict(
                    ip="192.0.2.1",
                    port=27188,
                    checkpoint="/target",
                    draft_checkpoint="/draft",
                ),
                decode=dict(ip="192.0.2.2", port=28188),
            )
            worker.config = k31_ci.smoke_config(base)
            worker.host, worker.role = worker.config["prefill"], "prefill"
            with (
                patch.object(worker, "status"),
                patch.object(
                    k31_ci, "runtime_geometry", return_value={"runtime_verified": True}
                ) as geometry,
            ):
                worker._request_phase = "regular"
                regular = worker.request_command()
                worker._request_phase = "long"
                long = worker.request_command()
            geometry.assert_called_once()
            self.assertEqual(geometry.call_args.args[-1], "a" * 40)
            self.assertEqual(Path(regular[1]).name, "k31_full_smoke.py")
            self.assertEqual(Path(long[1]).name, "k31_long_prefix_smoke.py")
            self.assertEqual(regular.count("--decode-role-addr"), 2)
            self.assertEqual(long.count("--decode-role-addr"), 1)
            self.assertEqual(
                long[long.index("--decode-role-addr") + 1], "192.0.2.2:28188:28189"
            )

    def native_events(self):
        prefill, decode = ({rank: [] for rank in range(8)} for _ in range(2))
        proof = {
            "cases": {
                "seed": dict(input_len=10, prefill_total_reuse_len=0),
                "continuation": dict(input_len=12, prefill_total_reuse_len=8),
            }
        }
        for step, (name, query, prefix) in enumerate(
            (("seed", 10, 0), ("continuation", 4, 8)), 1
        ):
            for rank in range(8):
                prefill[rank].append(
                    dict(
                        event="mtp_target_prefill_forward",
                        execution_step=step,
                        world_rank=rank,
                        tp_rank=rank,
                        dp_rank=0,
                        fake=False,
                        logical_authoritative=rank == 0,
                        logical_batch=1,
                        request_ids=[step],
                        input_rows=query,
                        token_rows=query,
                        context_lengths=[query],
                        reuse_lengths=[prefix],
                        time_ns=step,
                    )
                )
            for rank in range(4):
                decode[rank].append(
                    dict(
                        event="pd_cache_loaded",
                        request_id=step,
                        dp_rank=0,
                        attn_tp_rank=rank,
                        loaded_blocks=1,
                    )
                )
                for source in range(8):
                    decode[rank].append(
                        dict(
                            event="pd_transfer_submitted",
                            request_id=step,
                            dp_rank=0,
                            attn_tp_rank=rank,
                            source_tp=8,
                            destination_tp=4,
                            source_tp_rank=source,
                            transfer_bytes=128,
                        )
                    )
        return proof, prefill, decode

    def test_long_route_requires_all_eight_sources_and_the_selected_tp4_owner(self):
        proof, prefill, decode = self.native_events()
        result = smoke.audit_long_events(proof, prefill, decode, decode_tp=4)
        self.assertEqual(set(result["seed"]["decode_ranks"]), {"0", "1", "2", "3"})
        missing = copy.deepcopy(decode)
        missing[3].pop()
        with self.assertRaisesRegex(ValueError, "incomplete source plan"):
            smoke.audit_long_events(proof, prefill, missing, decode_tp=4)
        wrong_owner = copy.deepcopy(decode)
        wrong_owner[4].append(dict(event="pd_cache_loaded", request_id=1))
        with self.assertRaisesRegex(ValueError, "unexpected cache load"):
            smoke.audit_long_events(proof, prefill, wrong_owner, decode_tp=4)


if __name__ == "__main__":
    unittest.main()

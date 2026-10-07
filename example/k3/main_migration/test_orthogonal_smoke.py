"""Offline failure gates for the complete K3 PD smoke."""

import pathlib
import json
import tempfile
import types
import unittest
from unittest.mock import patch
import sys
import contextlib
import io

from example.k3.main_migration.audit_orthogonal_smoke import (
    answer_from_question, audit, audit_raw_answers, cancel_host_load_evidence, frontend_ids,
    independent_answer_check, read_events,
)
from example.k3.main_migration.text_smoke import Runner, SmokeFailure, parse_args


def runner():
    args = types.SimpleNamespace(
        base_url="http://127.0.0.1:1", decode_health_url="http://127.0.0.1:2/health",
        decode_role_addrs=[], namespace="offline-orthogonal", block_size=4096,
        reuse_unit_tokens=32768, chunk_tokens=65536, max_tokens=32,
        mtp_chunk_max_tokens=32,
        suite="orthogonal-flow",
    )
    return Runner(args)


class OrthogonalSmokeOfflineTest(unittest.TestCase):
    def test_four_layer_flow_seeds_a_reusable_kda_stripe_with_short_output(self):
        smoke = runner()
        smoke.args.require_mtp = True
        smoke.decode_role_addrs = [{"role": "DECODE"}, {"role": "DECODE"}]
        smoke.fit_prompt = lambda head, tail, target: (head + tail, [1] * target)
        stages = []
        smoke.run_stage = lambda name, cases, concurrent=False: stages.append(
            (name, cases))
        smoke.run_flow()
        seed = stages[0][1][0]
        self.assertEqual(seed.expected_input_len, 32769)
        self.assertEqual(seed.max_tokens, 8)
        self.assertEqual([stages[owner + 1][1][0].decode_owner_rank
                          for owner in (0, 1)], [0, 1])
        self.assertTrue(all(case.max_tokens == 8
                            for _, cases in stages[3:5] for case in cases))

    def test_four_layer_cache_pressure_can_fill_its_device_pool(self):
        smoke = runner()
        smoke._orthogonal_device_blocks = lambda: 69
        smoke.fit_prompt = lambda head, tail, target: (head + tail, [1] * target)
        stages = []
        smoke._required_stage = lambda name, cases: stages.append((name, cases))
        smoke._orthogonal_bounded_cache_pressure()
        self.assertEqual(len(stages), 49)
        self.assertTrue(all(cases[0].preparation_only for _, cases in stages))
        smoke.args.suite = "main-text-64k-capped"
        stages.clear()
        smoke._orthogonal_bounded_cache_pressure()
        self.assertEqual(len(stages), 24)

    def test_checkpoint_selects_only_four_layer_or_full_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text('{"num_hidden_layers":93}')
            events = root / "events"
            events.mkdir()
            engine = root / "engine.log"
            engine.touch()
            proto = (root / "runfiles/rtp_llm/rtp_llm/cpp/model_rpc/proto/"
                     "model_rpc_service_pb2.py")
            proto.parent.mkdir(parents=True)
            proto.touch()
            argv = ["text_smoke.py", "--base-url", "http://127.0.0.1:1",
                    "--decode-health-url", "http://127.0.0.1:2/health",
                    "--decode-role-addr", "127.0.0.1:2:3", "--output",
                    str(root / "result.json"), "--namespace", "test",
                    "--block-size", "4096",
                    "--long-prefix-checkpoint", str(checkpoint),
                    "--prefill-event-dir", str(events), "--prefill-engine-log",
                    str(engine), "--prefill-rpc-runfiles", str(root / "runfiles"),
                    "--prefill-grpc-port", "4"]
            with patch.object(sys, "argv", argv):
                args = parse_args()
            self.assertEqual(args.suite, "main-text-64k-capped")
            self.assertEqual(args.run_kind, "full-93")
            self.assertEqual(args.decode_dp_size, 1)
            self.assertTrue(args.require_mtp)
            self.assertEqual(args.orthogonal_phases,
                             ("cache", "cancel", "page", "chunk", "decode"))
            for option in (["--orthogonal-phases", "cache"],
                           ["--suite", "flow"], ["--require-mtp"],
                           ["--decode-dp-size", "1"]):
                with self.subTest(option=option), patch.object(sys, "argv", argv + option):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit):
                            parse_args()
            (checkpoint / "config.json").write_text('{"num_hidden_layers":4}')
            with patch.object(sys, "argv", argv):
                args = parse_args()
            self.assertEqual(args.run_kind, "four-layer")
            self.assertEqual(args.suite, "orthogonal-flow")
            with patch.object(sys, "argv", argv + ["--decode-role-addr",
                                                  "127.0.0.1:5:6"]):
                args = parse_args()
            self.assertEqual(args.decode_dp_size, 2)

    def test_cancel_requires_server_ack_inside_host_load_window(self):
        started = {"event": "host_cache_load_started", "request_id": 7,
                   "time_ns": 100}
        accepted = {"event": "prefill_priority_cancel_accepted", "request_id": 7,
                    "time_ns": 105}
        item = {"request_id": 7, "load_started": started,
                "cancelled": accepted, "load_done": None,
                "cancel_status": 1,
                "fetch": {"code": "StatusCode.RESOURCE_EXHAUSTED",
                          "details": "preempted by a higher-priority request"}}
        events = {0: [started, accepted]}
        self.assertTrue(cancel_host_load_evidence(item, events))
        for change in ({"cancel_status": 2},
                       {"load_done": {"time_ns": 104}},
                       {"fetch": {"code": "StatusCode.OK"}}):
            self.assertFalse(cancel_host_load_evidence(dict(item, **change), events))
        self.assertFalse(cancel_host_load_evidence(item, {0: [started]}))

    def test_host_cache_trigger_reads_cpp_engine_events_incrementally(self):
        smoke = runner()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "main_0.log").write_text(
                '[K3_SMOKE_EVENT] {"event":"frontend_request","request_id":7}\n'
            )
            engine = root / "engine.log"
            engine.write_text(
                '[INFO] [RANK 0] [K3_SMOKE_EVENT] '
                '{"event":"host_cache_load_started","request_id":7}\n'
            )
            smoke.args.prefill_engine_log = engine
            offsets = {}
            first = smoke._new_log_events(root, offsets)
            self.assertEqual({item["event"] for item in first},
                             {"frontend_request", "host_cache_load_started"})
            self.assertEqual(smoke._new_log_events(root, offsets), [])
            with engine.open("a") as handle:
                handle.write('[INFO] [RANK 0] [K3_SMOKE_EVENT] '
                             '{"event":"host_cache_cancelled_during_load","request_id":7}\n')
            self.assertEqual(smoke._new_log_events(root, offsets)[0]["event"],
                             "host_cache_cancelled_during_load")

    def test_mixed_cache_requires_one_forward_on_every_rank(self):
        smoke = runner()
        names = {"device", "memory", "partial", "miss"}
        stage = {"start_time_ns": 1, "end_time_ns": 10}
        with tempfile.TemporaryDirectory() as tmp:
            smoke.args.prefill_event_dir = pathlib.Path(tmp)
            front = [{"time_ns": 2, "event": "frontend_request",
                      "case": name, "request_id": index}
                     for index, name in enumerate(sorted(names), 1)]
            shared = {"time_ns": 3, "kind": "target_prefill_forward",
                      "actual_batch": 4, "request_ids": [1, 2, 3, 4]}
            for rank in range(2):
                events = (front if rank == 0 else []) + [dict(shared, tp_rank=rank)]
                (smoke.args.prefill_event_dir / f"main_{rank}.log").write_text(
                    "".join("[K3_SMOKE_EVENT] " + json.dumps(item) + "\n"
                            for item in events))
            self.assertTrue(smoke._orthogonal_shared_prefill_forward(stage, names))
            (smoke.args.prefill_event_dir / "main_1.log").write_text("")
            self.assertFalse(smoke._orthogonal_shared_prefill_forward(stage, names))

    def test_pd_group_mapping_requires_all_four_exact_request_ids(self):
        smoke = runner()
        names = {"device", "memory", "partial", "miss"}
        mapping = {name: index for index, name in enumerate(sorted(names), 1)}
        stage = {"start_time_ns": 1, "end_time_ns": 10,
                 "transport": "pd_group_rpc", "request_ids": mapping}
        with tempfile.TemporaryDirectory() as tmp:
            smoke.args.prefill_event_dir = pathlib.Path(tmp)
            shared = {"time_ns": 3, "kind": "target_prefill_forward",
                      "actual_batch": 4, "request_ids": [1, 2, 3, 4]}
            for rank in range(2):
                (smoke.args.prefill_event_dir / f"main_{rank}.log").write_text(
                    "[K3_SMOKE_EVENT] " + json.dumps(dict(shared, tp_rank=rank)) + "\n")
            self.assertTrue(smoke._orthogonal_shared_prefill_forward(stage, names))
            self.assertEqual(frontend_ids({0: []}, [dict(stage, case_names=list(names))]), mapping)
            stage["request_ids"] = dict(mapping, miss=2)
            self.assertFalse(smoke._orthogonal_shared_prefill_forward(stage, names))
            with self.assertRaisesRegex(ValueError, "duplicate request IDs"):
                frontend_ids({0: []}, [dict(stage, case_names=list(names))])

    def test_cache_seed_reaches_kda_checkpoint_before_reuse_probe(self):
        smoke = runner()
        smoke.args.reuse_unit_tokens = 4096
        targets = []

        def fit_prompt(prefix, suffix, target):
            targets.append(target)
            return prefix + suffix, [0] * target

        smoke.fit_prompt = fit_prompt
        smoke._required_stage = lambda *args, **kwargs: (_ for _ in ()).throw(
            SmokeFailure("stop after seed"))
        with self.assertRaisesRegex(SmokeFailure, "stop after seed"):
            smoke._orthogonal_cache_seed()
        self.assertEqual(targets, [32769])

    def test_cache_seed_stays_within_64k_q(self):
        smoke = runner()
        targets = []

        def fit_prompt(head, tail, target):
            targets.append(target)
            return head + tail, [0] * target

        smoke.fit_prompt = fit_prompt
        smoke._required_stage = lambda *args, **kwargs: (_ for _ in ()).throw(
            SmokeFailure("stop after seed"))
        with self.assertRaisesRegex(SmokeFailure, "stop after seed"):
            smoke._orthogonal_cache_seed()
        self.assertEqual(targets, [32769])

    def test_cache_probe_rejects_missing_host_hit_without_extra_pressure(self):
        smoke = runner()
        smoke.fit_prompt = lambda head, tail, target: (head + tail, [0] * target)
        names = []

        def stage(name, cases, concurrent=False):
            names.append(name)
            if name.startswith("tier-device-probe-"):
                self.assertEqual(cases[0].max_tokens, 8)
                smoke.records.append({"prefill_cache_tiers": {
                    "device": 32768, "memory": 0}})
            if name == "tier-memory-probe":
                self.assertEqual(cases[0].max_tokens, 8)
                smoke.records.append({"prefill_cache_tiers": {
                    "device": 32768, "memory": 0}})

        smoke._required_stage = stage
        smoke._orthogonal_cache_seed()
        with self.assertRaisesRegex(SmokeFailure, "did not demote"):
            smoke._orthogonal_cache()
        self.assertEqual(names[:2], ["tier-memory-seed-0", "tier-device-probe-0"])
        self.assertFalse(any("evict" in name for name in names))
        self.assertEqual(names[-1], "tier-memory-probe")

    def test_long_history_seed_never_exceeds_uncached_64k_budget(self):
        smoke = runner()
        class Tokens:
            def __init__(self, count):
                self.count = count

            def __len__(self):
                return self.count

        smoke.tokenize = lambda prompt: Tokens(27 + prompt.count(" x") + prompt.count(" z"))
        calls = []

        def stage(name, cases, concurrent=False):
            case = cases[0]
            calls.append(case)
            length = len(smoke.tokenize(case.prompt))
            reuse = 0 if len(calls) == 1 else ((length - 32740) // 32768) * 32768
            if name == "orthogonal_kv_final":
                reuse = 589_824
            smoke.records.append({"input_len": length, "effective_reuse_len": reuse})
            smoke.stages.append({"name": name})

        smoke._required_stage = stage
        smoke._orthogonal_chunk_kv()
        seeds = [case for case in calls if case.preparation_only]
        self.assertLessEqual(len(seeds), 20)
        self.assertTrue(all(case.max_tokens == 4 for case in seeds))
        self.assertTrue(all(case.allow_long_history for case in calls))
        self.assertTrue(all(
            row["input_len"] - row["effective_reuse_len"] <= 65536
            for row in smoke.records[1:]
        ))
        self.assertEqual(calls[-1].name, "orthogonal_kv_final")

    def test_decode_global_64_routes_32_per_dp_owner(self):
        smoke = runner()
        smoke.decode_role_addrs = [{"role": "DECODE"}, {"role": "DECODE"}]
        groups = []
        smoke._required_stage = lambda name, cases, **kwargs: groups.append((name, cases, kwargs))
        smoke._orthogonal_decode_batches()
        self.assertEqual([len(cases) for _, cases, _ in groups],
                         [1, 4, 8, 63, 64])
        self.assertEqual([sum(case.decode_owner_rank == rank for case in groups[4][1])
                          for rank in (0, 1)], [32, 32])
        self.assertIsNone(groups[4][2].get("admission_wave_size"))
        self.assertEqual(groups[4][2].get("admission_gap_s", 0), 0)
        self.assertEqual(groups[4][1][0].max_tokens, 512)
        smoke.args.suite = "main-text-64k-capped"
        groups.clear()
        smoke._orthogonal_decode_batches()
        self.assertTrue(groups[4][2]["grouped_pd"])
        self.assertEqual(groups[4][1][0].max_tokens, 1280)
        self.assertFalse(groups[2][2]["grouped_pd"])

    def test_cached_64k_crosses_decode_dp_owner_without_extra_requests(self):
        smoke = runner()
        smoke.args.reuse_unit_tokens = 4096
        smoke.decode_role_addrs = [{"role": "DECODE"}, {"role": "DECODE"}]
        smoke.fit_prompt = lambda head, tail, target: (head + tail, [1] * target)
        stages = []
        smoke.run_stage = lambda name, cases, concurrent=False: stages.append(
            (name, cases))
        smoke._required_stage = smoke.run_stage
        smoke.run_single_prefill_64k()
        self.assertEqual(sum(len(cases) for _, cases in stages), 6)
        name, cases = stages[1]
        self.assertEqual(name, "decode_dp_cross_owner_cached_64k")
        self.assertEqual(cases[0].decode_owner_rank, 1)
        self.assertEqual(cases[0].expected_reuse_len, 61440)
        self.assertTrue(cases[0].require_mtp)

    def test_dp_cross_owner_audit_checks_isolation_and_kda_mla_mapping(self):
        cold = "prefill_64k_single_0_cold"
        switched = "decode_dp_cross_owner_cached_64k"
        stages = [{"name": cold, "start_time_ns": 0, "end_time_ns": 10},
                  {"name": switched, "start_time_ns": 20, "end_time_ns": 30}]
        cases = [{"name": cold, "phase": "preparation", "decode_owner_rank": 0},
                 {"name": switched, "phase": "preparation", "decode_owner_rank": 1,
                  "effective_reuse_len": 61440}]
        prefill = {rank: [] for rank in range(8)}
        prefill[0] = [
            {"event": "frontend_request", "case": cold, "request_id": 101,
             "time_ns": 1},
            {"event": "frontend_request", "case": switched, "request_id": 202,
             "time_ns": 21},
        ]
        decode = {rank: [] for rank in range(8)}
        for rank in range(8):
            owner, attention_rank = divmod(rank, 4)
            decode[rank].append({"event": "pd_cache_loaded", "dp_rank": owner,
                                 "request_id": 101 if owner == 0 else 202})
            if owner == 1:
                for source in range(8):
                    decode[rank].append({
                        "event": "pd_transfer_submitted", "request_id": 202,
                        "dp_rank": 1, "attn_tp_rank": attention_rank,
                        "source_tp_rank": source, "source_tp": 8,
                        "destination_tp": 4, "mla_pages": 1,
                        "first_mla_page": source, "last_mla_page": source,
                        "kda_blocks": int(source // 2 == attention_rank),
                        "kda_source_partitions": 1,
                        "kda_destination_partitions": 2,
                        "kda_destination_partition": source % 2,
                    })
        result = {"passed": True, "suite": "main-text-64k-capped",
                  "cases": cases, "stages": stages}
        with tempfile.TemporaryDirectory() as tmp:
            verdict = audit(result, prefill, decode, decode_dp=2,
                            request_dir=pathlib.Path(tmp))
            self.assertTrue(verdict["path_results"]["decode_dp_cross_owner_handoff"])
            decode[5] = [event for event in decode[5]
                         if not (event.get("event") == "pd_transfer_submitted"
                                 and event.get("source_tp_rank") == 2)]
            verdict = audit(result, prefill, decode, decode_dp=2,
                            request_dir=pathlib.Path(tmp))
            self.assertFalse(verdict["path_results"]["decode_dp_cross_owner_handoff"])

    def test_answer_audit_rejects_duplicate_json_and_truncation(self):
        row = {"name": "x", "phase": "formal", "pd_sep": True,
               "finish_reason": "length", "content": '{"value":"A","value":"A"}',
               "expected_json": {"value": "A"}, "output_len": 8}
        errors = independent_answer_check(row)
        self.assertTrue(any("duplicate" in error for error in errors))
        self.assertTrue(any("incomplete" in error for error in errors))

    def test_four_layer_diagnostic_garble_does_not_weaken_formal_audit(self):
        row = {"name": "filler", "phase": "formal", "pd_sep": True,
               "finish_reason": "length", "content": "x\ufffd", "expected_regex": ".",
               "output_len": 2}
        self.assertEqual(independent_answer_check(row, diagnostic=True), [])
        self.assertTrue(independent_answer_check(row, diagnostic=False))

    def test_raw_answer_is_derived_from_question(self):
        self.assertEqual(answer_from_question("只回答数字：81 的平方是多少？"), "6561")
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "0001.json"
            path.write_text(json.dumps({
                "name": "case", "phase": "formal", "status": 200,
                "request": {"messages": [{"content": '只输出 JSON {"value":"expected"}'}]},
                "response_body": json.dumps({"choices": [{"finish_reason": "stop",
                    "message": {"content": '{"value":"wrong"}'}}],
                    "aux_info": {"pd_sep": True}}),
            }))
            self.assertTrue(any("prompt-derived answer mismatch" in error
                                for error in audit_raw_answers(path.parent,
                                    {"case": {"phase": "formal"}})))

    def test_group_rpc_raw_answer_requires_rpc_ack_and_exact_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "0001.json"
            artifact = {
                "name": "case", "phase": "formal", "transport": "pd_group_rpc",
                "request_id": 17, "rpc_status": "OK",
                "request": {"messages": [{"content": '只输出 JSON {"value":"expected"}'}]},
                "response_body": json.dumps({"choices": [{"finish_reason": "stop",
                    "message": {"content": '{"value":"expected"}'}}],
                    "aux_info": {"pd_sep": True}}),
            }
            path.write_text(json.dumps(artifact))
            rows = {"case": {"phase": "formal"}}
            self.assertEqual(audit_raw_answers(path.parent, rows), [])
            artifact["rpc_status"] = "UNKNOWN"
            path.write_text(json.dumps(artifact))
            self.assertTrue(audit_raw_answers(path.parent, rows))

    def test_missing_rank_or_runtime_stage_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp)
            (path / "main_0.log").write_text("")
            with self.assertRaisesRegex(ValueError, "missing rank log"):
                read_events(path)
        empty = {rank: [] for rank in range(8)}
        verdict = audit({"passed": True, "suite": "orthogonal-flow", "cases": [], "stages": []},
                        empty, empty, decode_dp=1)
        self.assertFalse(verdict["passed"])
        self.assertIn("orthogonal smoke has no executed phase", verdict["errors"])

    def test_capped_suite_requires_orthogonal_paths_and_raw_answers(self):
        empty = {rank: [] for rank in range(8)}
        result = {"passed": True, "suite": "main-text-64k-capped",
                  "case_profile": "compact-64k-v1", "cases": [], "stages": []}
        with tempfile.TemporaryDirectory() as tmp:
            verdict = audit(result, empty, empty, decode_dp=1,
                            request_dir=pathlib.Path(tmp))
            self.assertFalse(verdict["passed"])
            self.assertIn("missing runtime stage: chunk_kv_executed", verdict["errors"])
            self.assertIn("missing runtime stage: orthogonal_decode_01_batch_4",
                          verdict["errors"])
            result["cases"] = [{"name": "case", "phase": "formal", "pd_sep": True,
                                "finish_reason": "stop", "content": "6561",
                                "expected_regex": "6561", "output_len": 2}]
            verdict = audit(result, empty, empty, decode_dp=1,
                            request_dir=pathlib.Path(tmp))
            self.assertFalse(verdict["answer_passed"])
            self.assertFalse(verdict["passed"])

    def test_dp2_decode_graph_accepts_exact_draft_and_requires_verify_padding(self):
        prefill = {rank: [] for rank in range(8)}
        decode = {rank: [] for rank in range(8)}
        cases = [{"name": f"single_{i}", "phase": "prepare", "decode_owner_rank": 0}
                 for i in range(1)]
        cases.extend({"name": f"uneven_{i}", "phase": "prepare",
                      "decode_owner_rank": i % 2} for i in range(63))
        stages = []
        for name, names, start in (
            ("orthogonal_decode_00_batch_1", ["single_0"], 100),
            ("orthogonal_decode_03_batch_63", [f"uneven_{i}" for i in range(63)], 300),
        ):
            stages.append({"name": name, "case_names": names,
                           "start_time_ns": start, "end_time_ns": start + 90})
            for rank in range(8):
                local = sum((case["decode_owner_rank"] == rank // 4)
                            for case in cases if case["name"] in names)
                if not local:
                    continue
                verify_bucket = 32 if local == 31 else local
                decode[rank].extend([
                    {"event": "mtp_target_verify_forward", "input_rows": local,
                     "stream_count": local, "time_ns": start + 10},
                    {"event": "cuda_graph_replay", "role": 2,
                     "real_batch": local, "bucket": verify_bucket,
                     "padding_rows": verify_bucket - local, "time_ns": start + 20},
                    {"event": "mtp_draft_decode_forward", "input_rows": local,
                     "stream_count": local, "time_ns": start + 30},
                    {"event": "cuda_graph_replay", "role": 4,
                     "real_batch": local, "real_tokens": local * 4,
                     "bucket": local * 4, "padding_rows": 0,
                     "time_ns": start + 40},
                ])
        result = {"passed": True, "suite": "orthogonal-flow", "cases": cases,
                  "stages": stages, "orthogonal_phases": ["decode"]}
        verdict = audit(result, prefill, decode, decode_dp=2)
        for stage in stages:
            self.assertTrue(verdict["path_results"][stage["name"]])
        for event in decode[4]:
            if event.get("role") == 2 and event["time_ns"] == 320:
                event["padding_rows"] = 0
        verdict = audit(result, prefill, decode, decode_dp=2)
        self.assertFalse(verdict["path_results"]["orthogonal_decode_03_batch_63"])

    def test_dp1_decode_graph_virtual_row_counts_as_padding(self):
        cases = [{"name": f"case_{i}", "phase": "prepare", "decode_owner_rank": 0}
                 for i in range(63)]
        stage = {"name": "orthogonal_decode_03_batch_63",
                 "case_names": [case["name"] for case in cases],
                 "start_time_ns": 0, "end_time_ns": 100}
        result = {"passed": True, "suite": "orthogonal-flow", "cases": cases,
                  "stages": [stage], "orthogonal_phases": ["decode"]}
        events = [
            {"event": "mtp_target_verify_forward", "input_rows": 63,
             "stream_count": 63, "time_ns": 10},
            {"event": "cuda_graph_replay", "role": 2, "real_batch": 64,
             "bucket": 64, "padding_rows": 0, "time_ns": 20},
            {"event": "mtp_draft_decode_forward", "input_rows": 63,
             "stream_count": 63, "time_ns": 30},
            {"event": "cuda_graph_replay", "role": 4, "real_batch": 64,
             "real_tokens": 256, "bucket": 256, "padding_rows": 0,
             "time_ns": 40},
        ]
        prefill = {rank: [] for rank in range(8)}
        decode = {rank: [dict(event) for event in events] for rank in range(8)}
        verdict = audit(result, prefill, decode, decode_dp=1)
        self.assertTrue(verdict["path_results"][stage["name"]])

    def test_frontend_ids_ignore_prior_run_with_same_case_name(self):
        events = {rank: [] for rank in range(8)}
        events[0] = [
            {"event": "frontend_request", "time_ns": 2, "case": "same", "request_id": 11},
            {"event": "frontend_request", "time_ns": 12, "case": "same", "request_id": 22},
        ]
        self.assertEqual(frontend_ids(events, [{"start_time_ns": 10,
                                                "end_time_ns": 20}]), {"same": 22})

    def test_engine_events_are_assigned_to_their_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            for rank in range(2):
                (root / f"main_{rank}.log").write_text("")
            cpp = root / "engine.log"
            cpp.write_text('[INFO] [RANK 1] [K3_SMOKE_EVENT] '
                           '{"event":"cuda_graph_replay","time_ns":12}\n')
            events = read_events(root, 2, cpp)
            self.assertEqual(events[0], [])
            self.assertEqual(events[1][0]["event"], "cuda_graph_replay")


if __name__ == "__main__":
    unittest.main()

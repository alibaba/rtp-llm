from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from openpyxl import load_workbook


TOOL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_DIR))
import build_workbook as workbook_module  # noqa: E402
import build_html as html_module  # noqa: E402


SHANGHAI = ZoneInfo("Asia/Shanghai")


def epoch_ms(hour: int, minute: int, second: int = 0) -> int:
    return int(
        datetime(2026, 8, 11, hour, minute, second, tzinfo=SHANGHAI).timestamp()
        * 1000
    )


def pv_line(log_time: str, record: dict) -> str:
    return (
        f"{log_time} INFO test pvLogger - "
        + json.dumps(record, separators=(",", ":"))
        + "\n"
    )


def route_record(
    request_id: str, request_time_ms: int, worker: str, role: str = "PREFILL"
) -> dict:
    return {
        "requestId": request_id,
        "requestTimeMs": request_time_ms,
        "totalUs": 1234,
        "success": True,
        "seqLen": 1000,
        "selectionReasons": {role: "SHORTEST_TTFT"},
        "response": {
            "code": 200,
            "server_status": [
                {"role": role, "server_ip": worker, "code": 200}
            ],
        },
        "shortestTtftDecisions": [
            {
                "role": role,
                "decisionTimeMs": request_time_ms,
                "routingAttempt": 1,
                "workers": [
                    {
                        "ip": worker,
                        "port": 8001,
                        "selected": True,
                        "estimatedTtftRank": 1,
                        "requestHitRatePct": 80,
                        "requestUncachedTokens": 200,
                    }
                ],
            }
        ],
    }


def cache_record(request_id: str, actual_hit: int) -> dict:
    return {
        "event": "cache_hit_comparison",
        "requestId": request_id,
        "inputTokens": 1000,
        "kvcm": {"hit": 800},
        "actual": {"hit": actual_hit},
        "state": "running",
    }


def status_record(request_id: str, worker: str, request_time_ms: int) -> dict:
    enqueue = request_time_ms + 100
    return {
        "event": "prefill_worker_status",
        "requestId": request_id,
        "workerIp": worker,
        "inputQueueEnqueueTimeMs": enqueue,
        "inputQueueDrainTimeMs": enqueue + 10,
        "firstTokenTimeMs": enqueue + 1010,
        "inputQueueWaitMs": 10,
        "schedulerWaitMs": 100,
        "remoteKvWaitMs": 200,
        "schedulerToRunningMs": 300,
        "runningToFirstTokenMs": 700,
        "hbmLocalMatchTokens": 300,
        "remoteKvAddedMatchTokens": 500,
        "prefillStepCount": 1,
        "firstPrefillStepId": 7,
        "lastPrefillStepId": 7,
    }


class CurrentRoutingDecisionTest(unittest.TestCase):
    def test_compact_pv_uses_top_level_outcome_and_selected_candidate(self):
        route = route_record("compact", epoch_ms(1, 45), "10.0.0.8")
        route["code"] = 200
        route["response"].pop("code", None)
        route.pop("selectionReasons", None)
        route.pop("cacheMatchSelections", None)
        route.pop("shortestTtftDecisions", None)
        route["routingDecisions"] = [{"role": "PREFILL", "selectionReason": "CACHE_LEADER",
            "candidates": [{"endpoint": "10.0.0.8:8001@1", "selected": True,
                            "routingMatchTokens": 512, "projectedTtftMs": 20}]}]
        self.assertTrue(workbook_module.route_success(route))
        self.assertEqual(workbook_module.get_route_cache_selection(route)["hitCacheTokens"], 512)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "pv.log"
            source.write_text(pv_line("2026-08-11 01:45:00.010", route))
            destination = Path(directory) / "analysis.xlsx"
            workbook_module.build_workbook([source], destination)
            workbook = load_workbook(destination)
            sheet = workbook["Requests"]
            headers = [cell.value for cell in sheet[1]]
            values = next(record for record in (dict(zip(headers, row))
                for row in sheet.iter_rows(min_row=2, values_only=True)) if record.get("request_id") == "compact")
            self.assertEqual(values["route_response_code"], 200)
            self.assertEqual(values["route_predicted_hit_tokens"], 512)
            self.assertIn("CACHE_LEADER", values.values())
            self.assertTrue(html_module._build_replay(destination)["candidates"])
            workbook.close()

    def test_current_candidates_keep_same_request_ids_separate_across_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            route = route_record("same-id", epoch_ms(1, 45), "10.0.0.8")
            del route["shortestTtftDecisions"]
            route["routingDecisions"] = [{"role": "PREFILL", "strategy": "CostBasedPrefill", "candidates": [
                {"endpoint": "10.0.0.8:8001@1", "selected": True, "projectedTtftMs": 20}]}]
            source = Path(directory) / "pv.log"
            source.write_text(pv_line("2026-08-11 01:45:00.010", route))
            destination = Path(directory) / "analysis.xlsx"
            workbook_module.build_workbook([workbook_module.PvSource(source, "instance-a"),
                                            workbook_module.PvSource(source, "instance-b")], destination)
            replay = html_module._build_replay(destination)
            self.assertEqual(set(replay["candidates"]), {"instance-a::same-id", "instance-b::same-id"})
            self.assertEqual([len(items) for items in replay["candidates"].values()], [1, 1])
            candidate = replay["candidates"]["instance-a::same-id"][0]
            self.assertEqual(candidate["endpoint"], "10.0.0.8:8001@1")
            self.assertEqual(candidate["host"], "10.0.0.8")
            self.assertEqual(candidate["port"], 8001)
            self.assertEqual(candidate["engineIndex"], 1)

    def test_current_and_historical_workbooks_keep_units_and_unknowns(self):
        for mixed in (False, True):
            with self.subTest(mixed=mixed), tempfile.TemporaryDirectory() as directory:
                current = route_record("current", epoch_ms(1, 45), "10.0.0.8")
                del current["shortestTtftDecisions"]
                current["selectionReasons"] = {"PREFILL": "BEST_ONLY"}
                current["decisionGroup"] = {"id": "group-1", "policy": "SINGLE", "committedSize": 1}
                current["routingDecisions"] = [
                    {"role": "PREFILL", "strategy": "CostBasedPrefill", "selectionReason": "BEST_ONLY",
                     "selectedEndpoint": "10.0.0.8:8001", "snapshotTruncated": True,
                     "candidateWorkerCount": 40, "prefillPolicy": {"minimumTtftMs": 12},
                     "rejections": {"OUTLIER": 2}, "candidates": [
                         {"endpoint": "10.0.0.8:8001", "selected": True, "projectedTtftMs": 12,
                          "projectedDrainMs": 3, "incomingPrefillMs": 9, "effectiveHitTokens": 100,
                          "pendingRequests": 0},
                         {"endpoint": "10.0.0.9:8001", "selected": False,
                          "predictionState": "UNMODELED_ENGINE_WORK"}]},
                    {"role": "DECODE", "strategy": "CostBasedDecode", "candidates": [
                        {"endpoint": "10.0.0.10:8001", "selected": True, "usedKvTokens": 55,
                         "availableKvTokens": 100, "logWeight": -0.5}]}]
                source = Path(directory) / "pv.log"
                content = pv_line("2026-08-11 01:45:00.010", current)
                if mixed:
                    content += pv_line("2026-08-11 01:45:00.010",
                                       route_record("legacy", epoch_ms(1, 45), "10.0.0.8"))
                source.write_text(content)
                destination = Path(directory) / "replay.xlsx"
                workbook_module.build_workbook(source, destination)
                workbook = load_workbook(destination, data_only=True)
                sheet = workbook["Routing Decisions"]
                headers = [cell.value for cell in sheet[1]]
                records = [dict(zip(headers, values)) for values in sheet.iter_rows(min_row=2, values_only=True)]
                self.assertEqual(len(records), 3)
                self.assertEqual(records[0]["projectedTtftMs"], 12)
                self.assertEqual(records[0]["pendingRequests"], 0)
                self.assertIsNone(records[1]["projectedTtftMs"])
                self.assertIsNone(records[1]["pendingRequests"])
                self.assertEqual(records[2]["role"], "DECODE")
                self.assertEqual(records[2]["usedKvTokens"], 55)
                self.assertEqual(records[0]["policy.minimumTtftMs"], 12)
                self.assertEqual(records[0]["decisionGroup.id"], "group-1")
                requests = workbook["Requests"]
                columns = [cell.value for cell in requests[1]]
                request_rows = [dict(zip(columns, values)) for values in requests.iter_rows(min_row=2, values_only=True)]
                row = next(item for item in request_rows if item.get("request_id") == "current")
                self.assertEqual(row["selected projected TTFT ms"], 12)
                self.assertIsNone(row.get("selected snapshot estimated TTFT"))
                self.assertEqual(row["decision_snapshot_status"], "COST_BASED")
                workbook.close()
                output_html = Path(directory) / "replay.html"
                summary = html_module.build_html(destination, TOOL_DIR / "replay_template.html", output_html)
                payload = html_module._build_replay(destination)
                current_request = next(request for request in payload["requests"] if request["requestId"] == "current")
                candidates = payload["candidates"][current_request["id"]]
                self.assertEqual(len(candidates), 3)
                self.assertEqual(candidates[0]["schema"], "routingDecisions")
                prefill = next(item for item in candidates if item["role"] == "PREFILL" and item["selected"])
                self.assertEqual(prefill["projectedTtftMs"], 12)
                self.assertNotIn("estimatedTtft", prefill)
                self.assertEqual(prefill["decisionGroup"]["id"], "group-1")
                self.assertEqual(summary["candidate_count"], 4 if mixed else 3)
                self.assertIn("预测首 Token 耗时", output_html.read_text())


class CacheComparisonReplayTest(unittest.TestCase):
    def test_raw_feedback_preserves_all_predictions_and_valid_zero_through_workbook_and_html(self):
        for source, actual, kvcm in (("KVCM", 500, True), ("LOCAL_STANDBY", 0, False)):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                route = route_record("feedback", epoch_ms(1, 45), "10.0.0.1")
                cache = {"event": "cache_hit_comparison", "requestId": "feedback",
                         "source": source, "inputTokens": 1000,
                         "routing": {"hit": 400 if kvcm else 300}, "actual": {"hit": actual},
                         "kvcm": {"hit": 400, "delta": 100,
                                  "local": {"hit": 200, "delta": 300},
                                  "p2pTotal": {"hit": 600, "delta": -100}} if kvcm else None,
                         "localStandby": {"hit": 300, "delta": actual - 300}}
                status = status_record("feedback", "10.0.0.1", epoch_ms(1, 45))
                log = Path(directory) / "pv.log"
                log.write_text("".join(pv_line("2026-08-11 01:45:02.000", item)
                                       for item in (route, status, cache)))
                xlsx = Path(directory) / "replay.xlsx"
                workbook_module.build_workbook(log, xlsx)
                workbook = load_workbook(xlsx, data_only=True)
                sheet = workbook["Requests"]
                headers = [cell.value for cell in sheet[1]]
                row = next(item for item in (dict(zip(headers, values)) for values in
                           sheet.iter_rows(min_row=2, values_only=True)) if item.get("request_id") == "feedback")
                self.assertEqual(row["actual_hit_tokens"], actual)
                self.assertEqual(row["local_standby_delta_tokens"], actual - 300)
                self.assertEqual(row["kvcm_p2p_total_delta_tokens"], -100 if kvcm else None)
                self.assertEqual(row["kvcm_minus_standby_tokens"], 100 if kvcm else None)
                workbook.close()
                output = Path(directory) / "replay.html"
                html_module.build_html(xlsx, TOOL_DIR / "replay_template.html", output)
                request = html_module._build_replay(xlsx)["requests"][0]
                self.assertEqual(request["actualHit"], actual)
                self.assertEqual(request["predictedHit"], 400 if kvcm else 300)
                comparison = request["cacheComparison"]
                self.assertEqual(comparison["source"], source)
                self.assertEqual(comparison["kvcm_local"]["hit"], 200 if kvcm else None)
                self.assertEqual(comparison["kvcm_p2p_total"]["hit"], 600 if kvcm else None)
                self.assertEqual(comparison["local_standby"], {"hit": 300, "delta": actual - 300})
                self.assertIn("renderCacheComparison", output.read_text())

    def test_worker_actual_hit_survives_missing_prediction_feedback(self):
        for valid, expected in ((True, 0), (False, None)):
            with self.subTest(valid=valid), tempfile.TemporaryDirectory() as directory:
                route = route_record("actual-only", epoch_ms(1, 45), "10.0.0.1")
                status = {"event": "prefill_worker_status", "requestId": "actual-only",
                          "prefixLengthValid": valid, "actualHitTokens": 0}
                log = Path(directory) / "pv.log"
                log.write_text("".join(pv_line("2026-08-11 01:45:02.000", item) for item in (route, status)))
                xlsx = Path(directory) / "replay.xlsx"
                workbook_module.build_workbook(log, xlsx)
                request = html_module._build_replay(xlsx)["requests"][0]
                self.assertEqual(request["actualHit"], expected)
                self.assertIsNone(request["cacheComparison"]["kvcm"]["hit"])

    def test_absent_feedback_keeps_predictions_and_actual_unknown(self):
        request = html_module.compact_request({"request_id": "unknown", "route_log_time (decision)": "2026-08-11 01:45:00.000"})
        self.assertIsNone(request["actualHit"])
        for key in ("kvcm", "kvcm_local", "kvcm_p2p_total", "local_standby"):
            self.assertEqual(request["cacheComparison"][key], {"hit": None, "delta": None})


class BuildWorkbookTest(unittest.TestCase):
    def test_p95_to_p99_band_uses_bright_yellow(self) -> None:
        self.assertEqual(workbook_module.PERCENTILE_COLORS["P95-P99"], "FFFF00")

    def test_joins_by_instance_and_filters_only_route_request_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_a = root / "raw" / "flexlb-a" / "pv.log.snapshot"
            source_b = root / "raw" / "flexlb-b" / "pv.log.snapshot"
            source_a.parent.mkdir(parents=True)
            source_b.parent.mkdir(parents=True)

            request_a_ms = epoch_ms(1, 45)
            request_b_ms = epoch_ms(1, 46)
            outside_ms = epoch_ms(2, 0)
            source_a.write_text(
                "".join(
                    [
                        pv_line(
                            "2026-08-11 01:45:00.010",
                            route_record("same-request", request_a_ms, "10.0.0.1"),
                        ),
                        # Terminal records deliberately arrive after the report
                        # window.  They must still join to the in-window route.
                        pv_line(
                            "2026-08-11 02:03:00.000",
                            cache_record("same-request", 750),
                        ),
                        pv_line(
                            "2026-08-11 02:03:01.000",
                            status_record("same-request", "10.0.0.1", request_a_ms),
                        ),
                        pv_line(
                            "2026-08-11 01:59:59.000",
                            route_record("end-exclusive", outside_ms, "10.0.0.9"),
                        ),
                        pv_line(
                            "2026-08-11 02:04:00.000",
                            status_record("end-exclusive", "10.0.0.9", outside_ms),
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            source_b.write_text(
                "".join(
                    [
                        pv_line(
                            "2026-08-11 01:46:00.010",
                            route_record("same-request", request_b_ms, "10.0.0.2"),
                        ),
                        pv_line(
                            "2026-08-11 02:02:00.000",
                            cache_record("same-request", 900),
                        ),
                        pv_line(
                            "2026-08-11 02:02:01.000",
                            status_record("same-request", "10.0.0.2", request_b_ms),
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            destination = root / "analysis.xlsx"
            summary = workbook_module.build_workbook(
                sources=[source_a, source_b],
                destination=destination,
                start=datetime(2026, 8, 11, 1, 40, tzinfo=SHANGHAI),
                end=datetime(2026, 8, 11, 2, 0, tzinfo=SHANGHAI),
            )

            self.assertEqual(summary["source_count"], 2)
            self.assertEqual(summary["instance_count"], 2)
            self.assertEqual(summary["request_count"], 2)
            self.assertEqual(summary["complete_request_count"], 2)
            self.assertEqual(summary["event_counts"]["route_outside_window"], 1)

            workbook = load_workbook(destination, data_only=True)
            self.assertEqual(
                workbook.sheetnames,
                [
                    "Requests",
                    "P99 Focus",
                    "Decision Snapshot Top5",
                    "Routing Decisions",
                    "Host Summary",
                    "Data Scope",
                ],
            )
            worksheet = workbook["Requests"]
            headers = [cell.value for cell in worksheet[1]]
            self.assertEqual(worksheet.freeze_panes, "A2")
            self.assertEqual(
                headers[:17],
                [
                    "request_id",
                    "flexlb_instance",
                    "prefill_host",
                    "host_sequence_no",
                    "route_log_time (decision)",
                    "selection_reason",
                    "prefill_engine_ttft_ms",
                    "input_tokens",
                    "actual_hit_rate_pct",
                    "uncache_tokens",
                    "selected outstanding uncache",
                    "selected outstanding after request",
                    "decision cache lead tokens",
                    "decision extra work tokens",
                    "actual_minus_predicted_pp",
                    "hbm_local_match_tokens",
                    "remote_kv_added_match_tokens",
                ],
            )
            request_id_col = headers.index("request_id")
            instance_col = headers.index("flexlb_instance")
            request_rows = [
                row
                for row in worksheet.iter_rows(min_row=2, values_only=True)
                if row[request_id_col] == "same-request"
            ]
            self.assertEqual(len(request_rows), 2)
            self.assertEqual(
                {row[instance_col] for row in request_rows}, {"flexlb-a", "flexlb-b"}
            )
            request_cell = next(
                row[request_id_col]
                for row in worksheet.iter_rows(min_row=2)
                if row[request_id_col].value == "same-request"
            )
            self.assertEqual(request_cell.alignment.horizontal, "center")

            snapshot_sheet = workbook["Decision Snapshot Top5"]
            snapshot_headers = [cell.value for cell in snapshot_sheet[4]]
            guard_col = snapshot_headers.index("outstanding_guard_eligible")
            guard_values = [
                row[guard_col]
                for row in snapshot_sheet.iter_rows(min_row=5, values_only=True)
            ]
            self.assertEqual(guard_values, [None, None])

    def test_prefill_extractors_do_not_fall_back_to_decode_role(self) -> None:
        decode_only = {
            "response": {"server_status": [{"role": "DECODE", "server_ip": "decode"}]},
            "cacheMatchSelections": [{"role": "DECODE", "selectedIp": "decode"}],
            "shortestTtftDecisions": [{"role": "DECODE", "workers": [{"ip": "decode"}]}],
        }
        self.assertEqual(workbook_module.get_prefill_server_status(decode_only), {})
        self.assertEqual(workbook_module.get_route_cache_selection(decode_only), {})
        self.assertEqual(workbook_module.get_prefill_decision(decode_only), {})

        legacy = {
            "response": {"server_status": [{"server_ip": "legacy"}]},
            "cacheMatchSelections": [{"selectedIp": "legacy"}],
            "shortestTtftDecisions": [{"workers": [{"ip": "legacy"}]}],
        }
        self.assertEqual(
            workbook_module.get_prefill_server_status(legacy)["server_ip"], "legacy"
        )
        self.assertEqual(
            workbook_module.get_route_cache_selection(legacy)["selectedIp"], "legacy"
        )
        self.assertEqual(
            workbook_module.get_prefill_decision(legacy)["workers"][0]["ip"],
            "legacy",
        )

        ambiguous_legacy = {
            "response": {"server_status": [{"server_ip": "one"}, {"server_ip": "two"}]},
            "cacheMatchSelections": [{"selectedIp": "one"}, {"selectedIp": "two"}],
            "shortestTtftDecisions": [{"workers": []}, {"workers": []}],
        }
        self.assertEqual(workbook_module.get_prefill_server_status(ambiguous_legacy), {})
        self.assertEqual(workbook_module.get_route_cache_selection(ambiguous_legacy), {})
        self.assertEqual(workbook_module.get_prefill_decision(ambiguous_legacy), {})

    def test_pdfusion_is_replayed_as_a_prefill_equivalent_role(self) -> None:
        request_time_ms = epoch_ms(1, 45)
        route = route_record("fusion-request", request_time_ms, "10.0.0.7", "PDFUSION")
        route["cacheMatchSelections"] = [
            {"role": "PDFUSION", "selectedIp": "10.0.0.7", "hitCacheTokens": 800}
        ]

        self.assertEqual(
            workbook_module.get_prefill_equivalent_role(route), "PDFUSION"
        )
        self.assertEqual(
            workbook_module.get_prefill_server_status(route)["server_ip"], "10.0.0.7"
        )
        self.assertEqual(
            workbook_module.get_route_cache_selection(route)["selectedIp"], "10.0.0.7"
        )
        self.assertEqual(
            workbook_module.get_prefill_decision(route)["role"], "PDFUSION"
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "pv.log"
            source.write_text(
                "".join(
                    [
                        pv_line("2026-08-11 01:45:00.010", route),
                        pv_line("2026-08-11 01:45:00.020", cache_record("fusion-request", 800)),
                    ]
                ),
                encoding="utf-8",
            )
            rows, _, selection_counts = workbook_module.build_rows(
                [workbook_module.PvSource(source, "fusion-flexlb")]
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["selection_reason"], "SHORTEST_TTFT")
        self.assertEqual(rows[0]["prefill_host"], "10.0.0.7")
        self.assertEqual(len(rows[0]["_decision_workers"]), 1)
        self.assertEqual(selection_counts["SHORTEST_TTFT"], 1)

    def test_prefill_keeps_priority_when_prefill_and_pdfusion_are_both_present(self) -> None:
        request_time_ms = epoch_ms(1, 45)
        route = route_record("mixed-request", request_time_ms, "10.0.0.8")
        fusion_worker = "10.0.0.9"
        route["response"]["server_status"].insert(
            0, {"role": "PDFUSION", "server_ip": fusion_worker, "code": 200}
        )
        route["cacheMatchSelections"] = [
            {"role": "PDFUSION", "selectedIp": fusion_worker, "hitCacheTokens": 100},
            {"role": "PREFILL", "selectedIp": "10.0.0.8", "hitCacheTokens": 800},
        ]
        route["shortestTtftDecisions"].insert(
            0,
            {
                "role": "PDFUSION",
                "workers": [{"ip": fusion_worker, "selected": True}],
            },
        )
        route["selectionReasons"]["PDFUSION"] = "FUSION_REASON"

        self.assertEqual(workbook_module.get_prefill_equivalent_role(route), "PREFILL")
        self.assertEqual(
            workbook_module.get_prefill_server_status(route)["server_ip"], "10.0.0.8"
        )
        self.assertEqual(
            workbook_module.get_route_cache_selection(route)["selectedIp"], "10.0.0.8"
        )
        self.assertEqual(
            workbook_module.get_prefill_decision(route)["role"], "PREFILL"
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "pv.log"
            source.write_text(
                pv_line("2026-08-11 01:45:00.010", route), encoding="utf-8"
            )
            rows, _, selection_counts = workbook_module.build_rows(
                [workbook_module.PvSource(source, "mixed-flexlb")]
            )

        self.assertEqual(rows[0]["selection_reason"], "SHORTEST_TTFT")
        self.assertEqual(selection_counts["SHORTEST_TTFT"], 1)

    def test_optional_boolean_keeps_unknown_distinct_from_false(self) -> None:
        self.assertEqual(workbook_module.yes_no_unknown(None), "")
        self.assertEqual(workbook_module.yes_no_unknown(False), "NO")
        self.assertEqual(workbook_module.yes_no_unknown(True), "YES")

    def test_discovers_collector_manifest_and_preserves_instance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            snapshot = root / "raw" / "instance-from-parent" / "pv.log.snapshot"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_text("", encoding="utf-8")
            manifest = root / "collect_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "snapshots": [
                            {"instance": "instance-from-manifest", "path": str(snapshot)}
                        ]
                    }
                ),
                encoding="utf-8",
            )

            direct = workbook_module.discover_sources(snapshot)
            manifested = workbook_module.discover_sources(root)
            self.assertEqual(direct[0].instance, "instance-from-parent")
            self.assertEqual(manifested[0].instance, "instance-from-manifest")

    def test_empty_route_selection_fails_before_creating_workbook(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "pv.log"
            destination = root / "analysis.xlsx"
            source.write_text("unrelated log line\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "No routing PV records matched"):
                workbook_module.build_workbook(source, destination)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()

import copy
import json
import tempfile
import unittest
from pathlib import Path

from summarize_modeling_graph_traces import (
    align_ranks,
    compare_runs,
    file_sha256,
    load_window,
)
from test_modeling_graph_trace import event, round_events


def round_record(number, rank=0):
    start = number * 1000
    spans = (10 + rank, 30 - rank, 100 + rank, 20 + rank)
    calls = []
    for index, span in enumerate(spans):
        begin = start + index * 200 + rank
        calls.append(
            dict(
                gpu_start_us=begin,
                gpu_end_us=begin + span,
                gpu_span_us=span,
                activities=100,
            )
        )
    return {
        "round": number,
        "proposal": calls[:2],
        "verify": [calls[2]],
        "update": [calls[3]],
    }


def run(version, tp, scale=1):
    metrics = {
        k: value * scale
        for k, value in {
            "proposal_1_gpu_us": 17,
            "proposal_2_gpu_us": 30,
            "update_gpu_us": 27,
            "target_verify_gpu_us": 107,
            "mtp_modeling_gpu_us": 74,
        }.items()
    }
    return {
        "version": version,
        "tp": tp,
        "commit": (
            "edaaf0d8aeea61593729dec43a811e29217c32dc"
            if version == "feat"
            else "test-only"
        ),
        "contract": {
            "history_tokens": 65536,
            "batch_per_owner": 32,
            "n_step": 3,
            "layers": 93,
            "world_size": 8,
            "tp": tp,
            "dp": 8 // tp,
            "ep": 8,
            "cuda_graph": True,
            "mtp_dtype": "bf16",
            "mla_backend": "tokenspeed_page_rr",
            "async": True,
            "device_states": True,
            "moe_strategy": "mega_moe",
            "input_ids_sha256": "1" * 64,
            "target_weight_index_sha256": "2" * 64,
            "mtp_weight_index_sha256": "3" * 64,
        },
        "windows": [{"window": w, "window_medians_us": metrics} for w in (1, 2, 3)],
    }


class EightRankSummaryTest(unittest.TestCase):
    def test_trace_start_offsets_do_not_pair_adjacent_rounds(self):
        ranks = {r: [round_record(i, r) for i in range(r % 2, 5)] for r in range(8)}
        result = align_ranks(ranks, sample_rounds=2)
        self.assertEqual(result["all_rank_matched_rounds"], 4)
        self.assertEqual(
            [r["source_round_by_rank"]["0"] for r in result["selected_rounds"]], [2, 3]
        )
        for row in result["selected_rounds"]:
            self.assertEqual(len(set(row["source_round_by_rank"].values())), 1)

    def test_mtp_sums_each_call_max_rank_even_when_slowest_rank_changes(self):
        ranks = {r: [round_record(1, r)] for r in range(8)}
        row = align_ranks(ranks, sample_rounds=1)["selected_rounds"][0]
        self.assertEqual(row["target_verify_gpu_us"], 107)
        self.assertEqual(row["mtp_modeling_gpu_us"], 17 + 30 + 27)
        self.assertNotEqual(
            row["mtp_modeling_gpu_us"], max(10 + r + 30 - r + 20 + r for r in range(8))
        )

    def test_missing_rank_ambiguous_matches_and_partial_gpu_graph_fail(self):
        ranks = {r: [round_record(1, r)] for r in range(8)}
        missing = copy.deepcopy(ranks)
        del missing[7]
        with self.assertRaisesRegex(ValueError, "all ranks"):
            align_ranks(missing, 1)
        ambiguous = copy.deepcopy(ranks)
        ambiguous[7].append(copy.deepcopy(ambiguous[7][0]))
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            align_ranks(ambiguous, 1)
        partial = {r: [round_record(i, r) for i in (1, 2)] for r in range(8)}
        partial[0][-1]["update"][0]["activities"] = 99
        with self.assertRaisesRegex(ValueError, "activity counts differ"):
            align_ranks(partial, 1)

    def test_verify_overlap_does_not_suffice_for_mtp_alignment(self):
        ranks = {r: [round_record(1, r)] for r in range(8)}
        ranks[7][0]["update"][0]["gpu_start_us"] += 1000
        ranks[7][0]["update"][0]["gpu_end_us"] += 1000
        with self.assertRaisesRegex(ValueError, "MTP forwards"):
            align_ranks(ranks, 1)

    def test_tp8_metrics_have_independent_five_percent_gates(self):
        original = [
            run(v, 8, 1.02 if v == "integration" else 1)
            for v in ("integration", "feat")
        ]
        self.assertTrue(compare_runs(original)["performance_pass"])
        for metric in ("mtp_modeling_gpu_us", "target_verify_gpu_us"):
            runs = copy.deepcopy(original)
            for window in runs[0]["windows"]:
                window["window_medians_us"][metric] *= 1.05
            self.assertFalse(compare_runs(runs)["performance_pass"])

    def test_dp2_tp4_cannot_replace_or_extend_tp8_performance_acceptance(self):
        original = [run(v, 8) for v in ("integration", "feat")]
        for runs in (
            [run(v, 4) for v in ("integration", "feat")],
            original + [run(v, 4) for v in ("integration", "feat")],
        ):
            with self.assertRaisesRegex(ValueError, "correctness-only"):
                compare_runs(runs)

    def test_wrong_baseline_configuration_or_window_count_cannot_pass(self):
        original = [run(v, 8) for v in ("integration", "feat")]
        for mutation in ("baseline", "contract", "windows"):
            runs = copy.deepcopy(original)
            if mutation == "baseline":
                runs[1]["commit"] = "newer-feat-is-not-the-reference"
            elif mutation == "contract":
                runs[0]["contract"]["batch_per_owner"] = 64
            else:
                runs[0]["windows"].pop()
            with self.assertRaises(ValueError):
                compare_runs(runs)

    def test_raw_trace_sha_and_completed_warmup_requests_are_required(self):
        # Synthetic events test evidence validation, never hardware performance.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for rank in range(8):
                events = round_events()
                for i, width in enumerate((1, 1, 4, 4)):
                    events.append(
                        event(
                            f"cuda_graph.modeling(role={i},logical_b=32,physical_b=32,q={width},"
                            f"bucket=32,capture_b=32,capture_t={32*width})",
                            100 + i * 100,
                            10,
                        )
                    )
                path = root / f"synthetic-rank{rank}.json"
                path.write_text(
                    json.dumps(
                        {
                            "baseTimeNanoseconds": 1000000000 + 100 * rank,
                            "traceEvents": events,
                        }
                    )
                )
                entries.append(
                    {"rank": rank, "path": path.name, "sha256": file_sha256(path)}
                )
            labels = [f"warmup-{i}" for i in range(10)]
            groups = [
                {
                    "label": label,
                    "requests": [
                        {
                            "owner": owner,
                            "request": request,
                            "http_status": 200,
                            "input_tokens": 65536,
                            "output_tokens": 512,
                        }
                        for owner in range(2)
                        for request in range(32)
                    ],
                    "queues_after": {
                        "decode": {
                            "0": {"running_task_info": []},
                            "1": {"running_task_info": []},
                        }
                    },
                }
                for label in labels
            ]
            client = {
                "input_tokens": 65536,
                "actual_batch_required_per_owner": 32,
                "dp": 2,
                "output_tokens": 512,
                "groups": groups,
                "windows": [
                    {"window": 1, "warmup_group_labels": labels, "traces": entries}
                ],
            }
            client_path = root / "synthetic-client.json"
            client_path.write_text(json.dumps(client))
            window = {
                "window": 1,
                "warmup_completed_batches": 10,
                "traces": entries,
                "client_summary_path": client_path.name,
                "client_summary_sha256": file_sha256(client_path),
            }
            self.assertEqual(load_window(window, root, 1)["all_rank_matched_rounds"], 1)
            client["groups"][0]["requests"].pop()
            client_path.write_text(json.dumps(client))
            window["client_summary_sha256"] = file_sha256(client_path)
            with self.assertRaisesRegex(ValueError, "omitted requests"):
                load_window(window, root, 1)
            window["client_summary_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                load_window(window, root, 1)


if __name__ == "__main__":
    unittest.main()

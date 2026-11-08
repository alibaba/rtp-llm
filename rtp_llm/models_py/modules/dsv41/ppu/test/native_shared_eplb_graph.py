"""Observe real expert migration and replay on the same captured PPU model."""

from collections import defaultdict
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from rtp_llm.models_py.modules.dsv41.kv_cache_utils import as_tagged_attention_inputs
from rtp_llm.models_py.modules.dsv41.ppu.shared_native import SharedNativeCache
from rtp_llm.models_py.modules.dsv41.ppu.test.native_ced_role import write_json
from rtp_llm.models_py.modules.dsv41.ppu.test.native_shared_ced_suite import validate_decode
from rtp_llm.models_py.modules.dsv41.ppu.tools.driver_shared_eplb import NativeEplbEvidence


class ReplayEvidence:
    def __init__(self, model, capture_count):
        self.model = model
        self.rank = dist.get_rank()
        self.runtime = model.shared_graph_runtime
        assert self.runtime.grouped is not None and capture_count > 0
        self.capture_count = capture_count
        self.weight_proof = NativeEplbEvidence(model)
        self.previous = self.weight_proof.initial_map
        self.sequence = 0
        self.initial_graph_replays = model.shared_runtime.graph_replays
        self.pending = None
        self.directory = Path(os.environ["RTP_V41_N8_EPLB_REPORT_DIR"])

    def before(self, inputs, capture_count):
        assert self.pending is None
        assert self.model.shared_graph_runtime is self.runtime and capture_count == self.capture_count
        assert self.weight_proof._bindings() == self.weight_proof.bindings
        self.runtime.validate()
        layer = self.weight_proof.layer
        current = layer.physical_to_logical.cpu().clone()
        moved = torch.where(current != self.previous)[0].tolist()
        self.previous = current
        proof = self.weight_proof._slot_proof(current, set(moved))
        tracked = torch.where(current != self.weight_proof.initial_map)[0].tolist()
        attn = as_tagged_attention_inputs(inputs.attention_inputs)
        prefixes = attn.sequence_lengths.reshape(-1).tolist()
        tokens = inputs.input_ids.reshape(-1).tolist()
        state = {}
        if not inputs.is_fake_stream:
            assert not attn.is_prefill and len(prefixes) == len(tokens)
            cache = SharedNativeCache(
                self.model.shared_weight_views.config, self.runtime.geometry, self.model.kv_cache,
                inputs.attention_inputs, tuple(prefixes), device=self.runtime.device,
            )
            state = cache.fingerprint(prefixes)
        self.pending = {
            "rank": self.rank, "call": self.sequence + 1, "idle": bool(inputs.is_fake_stream),
            "row_layout": {"prefixes": prefixes, "lengths": [1] * len(prefixes)},
            "input_token_ids": tokens, "prefill": False, "native_state_before": state,
            "capture_count": capture_count, "pid": os.getpid(), "runtime_identity": id(self.runtime),
            "layout_sha256": self.weight_proof._state_signature(), "moved_slots": moved,
            "local_moved_slot_bytes": proof, "tracked_slots": tracked,
            "layers_before": self.runtime.layer_calls.tolist(),
        }

    def after(self):
        row, self.pending = self.pending, None
        assert row is not None
        self.sequence += 1
        assert self.model.shared_runtime.graph_replays == self.initial_graph_replays + self.sequence
        layers = self.runtime.layer_calls.tolist()
        assert layers == [count + 1 for count in row.pop("layers_before")]
        counts = self.runtime.grouped.received[0].tolist()
        assert len(counts) == 49
        row.update({"actual_layer_calls": layers, "local_expert_counts": counts,
                    "local_assignments_to_migrated_slots": sum(
                        count for index, count in enumerate(counts)
                        if self.rank * 49 + index in row["tracked_slots"]
                    )})
        write_json(self.directory / f"graph-rank{self.rank}-replay{self.sequence}.json", row)
        if self.rank == 0 and self.sequence == 1:
            output = Path(os.environ["RTP_V41_SHARED_OUTPUT"])
            write_json(output / "replays-verified.json", {"native_graph_replays": self.sequence})


def validate(output, peer, native_text):
    from rtp_llm.models_py.modules.dsv41.ppu.test.test_dspark_evidence import (
        access_records, assert_cancellation,
    )

    grouped = defaultdict(dict)
    for path in (output / "eplb").glob("graph-rank*-replay*.json"):
        row = json.loads(path.read_text())
        assert row["rank"] not in grouped[row["call"]]
        grouped[row["call"]][row["rank"]] = row
    assert sorted(grouped) == list(range(1, len(grouped) + 1))
    migrations, consumed, fresh, reports = [], [], [], []
    bindings = {}
    multi_batch = 0
    for call, by_rank in sorted(grouped.items()):
        assert set(by_rank) == set(range(8))
        ranks = list(by_rank.values())
        assert len({row["layout_sha256"] for row in ranks}) == 1
        live = [row for row in ranks if not row["idle"]]
        assert live
        reports.extend(live)
        multi_batch += sum(len(row["row_layout"]["prefixes"]) > 1 for row in live)
        assert sum(sum(row["local_expert_counts"]) for row in ranks) == 6 * sum(
            len(row["input_token_ids"]) for row in live
        )
        for row in ranks:
            current = row["pid"], row["runtime_identity"], row["capture_count"]
            assert bindings.setdefault(row["rank"], current) == current
        moved = ranks[0]["moved_slots"]
        assert all(row["moved_slots"] == moved for row in ranks)
        if moved:
            assert call > 1
            actual = sorted(item["physical"] for row in ranks for item in row["local_moved_slot_bytes"])
            assert actual == moved
            migrations.append({"first_replay": call, "physical_slots": moved,
                               "layout_sha256": ranks[0]["layout_sha256"]})
        hits = sum(row["local_assignments_to_migrated_slots"] for row in ranks)
        if hits:
            assert migrations
            consumed.append({"replay": call, "received_assignments": hits})
            if any(257 in row["row_layout"]["prefixes"] for row in live):
                fresh.append(call)
    assert migrations and consumed and fresh and multi_batch
    result = validate_decode(output, peer, reports)
    cases = json.loads((peer / "http-cases.json").read_text())["cases"]
    cancel = next(case for case in cases if case["case_kind"] == "cancel")
    recovery = next(case for case in cases if case["case_kind"] == "recovery")
    result["cancellation"] = assert_cancellation(cancel, recovery, access_records(peer / "logs"), native_text)
    return {**result, "same_capture_runtime": True, "migrations": migrations,
            "post_migration_replays": consumed, "fresh_post_migration_replays": fresh,
            "multi_batch_rank_replays": multi_batch, "raw_processed_and_scale_bytes": "EXACT_SOURCE_MATCH"}

"""Read-only evidence on an actual native model before/after public EPLB.

The fixture never calls step/rearrange, edits a map/count, replaces a runtime,
or changes a model input. PyWrappedModel owns every balancing boundary.
"""

import hashlib
import inspect
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def _hash(value):
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _values(value):
    return [] if value is None else value.cpu().reshape(-1).tolist()


def _tensor(value):
    return {"pointer": value.data_ptr(), "storage_pointer": value.untyped_storage().data_ptr(),
            "storage_offset": value.storage_offset(), "shape": list(value.shape),
            "stride": list(value.stride()), "dtype": str(value.dtype),
            "storage_bytes": value.untyped_storage().nbytes()}


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _bindings(engine, layers):
    return [{"layer": layer.layer_id, "binding_identity": id(layer),
             "logical": layer.num_logical_experts, "physical": layer.num_physical_experts,
             "local": layer.num_local_experts, "ep_rank": layer.ep_rank,
             "log2phy": _tensor(layer.logical_to_physical), "replicas": _tensor(layer.logical_count),
             "phy2log": _tensor(layer.physical_to_logical),
             "weights": [_tensor(w) for w in layer.weights],
             "processed": [_tensor(w) for w in engine._processed[(layer.layer_id, layer.logical_count.device)][:4]],
             "inv13_inv2": engine._eplb_invariants[layer.layer_id]}
            for layer in layers]


def migration_window_basis(frames):
    """Classify actual samples of the fixture's public two-step window."""
    assert 1 <= len(frames) <= 2
    assert len(frames) == min(2, frames[-1]["round"])
    assert [f["round"] for f in frames] == list(range(frames[-1]["round"] - len(frames) + 1, frames[-1]["round"] + 1))
    for frame in frames:
        assert frame["kind"] in ("business", "all_empty"), frame
        count = frame["global_layer0_received"]
        assert (count > 0 if frame["kind"] == "business" else count == 0), frame
    return "actual_business_window" if any(f["kind"] == "business" for f in frames) else "zero_count_window"


def epoch_moved_slots(epoch):
    """Recompute movement from the delivered pre/post ownership, not a cache."""
    old, new = epoch["old_physical_to_logical"], epoch["physical_to_logical"]
    assert len(old) == len(new) == 392
    moved = [slot for slot, (a, b) in enumerate(zip(old, new)) if a != b]
    assert moved == epoch["global_moved_slots"] and len(moved) == epoch["global_moved_slot_count"]
    return set(moved)


def record_prefill_lineage(model, inputs, output):
    """Context IDs are public on P; normal D exposes only sequence prefixes."""
    path = Path(os.environ["RTP_V41_SHARED_EVIDENCE_DIR"]) / f"rank{dist.get_rank()}-call{model.shared_runtime.calls:05d}.json"
    native = json.loads(path.read_text())
    report = {"pid": os.getpid(), "rank": dist.get_rank(), "model_identity": id(model),
              "engine_identity": id(model._shared_ops.moe_engine), "request_ids": _values(inputs.request_id),
              "bindings": _bindings(model._shared_ops.moe_engine, model.get_eplb_layers()),
              "map_sha256": _hash(model.get_eplb_layers()[0].logical_to_physical),
              "yoco": native["yoco"], "native_state_after": native["native_state_after"],
              "output_finite": bool(torch.isfinite(output.hidden_states).all()), "native_forward_evidence": str(path)}
    _write(Path(os.environ["RTP_V41_N8_EPLB_REPORT_DIR"]) / f"rank{dist.get_rank()}-prefill{model.shared_runtime.calls}.json", report)


class NativeObservationLedger:
    """Pure validation of rank-synchronous observations, never a model clock."""

    def __init__(self, role):
        assert role in ("prefill", "decode")
        self.role, self.round, self.business_calls = role, 0, 0
        self.pending = None
        self.empty_total = 0

    def _rows(self, rows, phase, round_id):
        assert len(rows) == 8 and {r["rank"] for r in rows} == set(range(8)), rows
        assert all(type(r["rank"]) is int and type(r["round"]) is int
                   and r["round"] == round_id and r["phase"] == phase for r in rows), rows
        assert all(type(r["idle"]) is bool and not r["warmup"] and r["bindings_stable"] for r in rows), rows
        return sorted(rows, key=lambda r: r["rank"])

    def begin(self, rows):
        assert self.pending is None, "nested/out-of-order native observation"
        rows = self._rows(rows, "before", self.round + 1)
        assert all(r["business_call"] == self.business_calls and
                   (r["count_nonzero"] == 0 or self.role == "prefill" and r["count_nonzero"] is None)
                   for r in rows), rows
        assert len({r["map_signature"] for r in rows}) == 1, rows
        business = [r for r in rows if not r["idle"]]
        for row in rows:
            if row["idle"]:
                # NormalEngine.createMinFakeStream has one placeholder row;
                # Decode advances it once, giving seqLength-1 == 1. Context
                # IDs belong to fake P context rows, never to normal D rows.
                assert row["sequence_prefixes"] in ([], [0], [1]), row
                if self.role == "decode":
                    assert not row["request_ids"] and not row["prefill"] and len(row["input_token_ids"]) <= 1, row
                    assert row["input_lengths"] in ([], [1]) and row["context_prefixes"] in ([], [0]), row
                else:
                    assert len(row["request_ids"]) <= 1 and row["input_lengths"] in ([], [1]), row
                    assert row["context_prefixes"] in ([], [0]) and len(row["input_token_ids"]) <= 2, row
        stage = None
        if business:
            index = self.business_calls + 1
            if self.role == "decode":
                assert len(business) == 1 and 1 <= index <= 6, business
                expected = 257 + index - 1 if index <= 5 else 17
                assert business[0]["sequence_prefixes"] == [expected] and not business[0]["prefill"], business
                stage = "trigger" if index <= 5 else "fresh"
            else:
                assert 1 <= index <= 2, index
                expected = 257 if index == 1 else 17
                assert all(r["input_lengths"] == [expected] and r["context_prefixes"] == [0]
                           and r["prefill"] for r in business), business
                stage = "trigger" if index == 1 else "fresh"
        self.round += 1
        self.pending = {"round": self.round, "kind": "business" if business else "all_empty",
                        "before": rows, "business": business, "stage": stage}
        return self.pending

    def finish(self, rows):
        assert self.pending is not None, "after without matching before"
        pending = self.pending
        rows = self._rows(rows, "after", pending["round"])
        advance = int(pending["kind"] == "business")
        for before, after in zip(pending["before"], rows):
            assert after["business_call"] == self.business_calls + advance, after
            assert all(after[k] == before[k] for k in ("idle", "request_ids", "sequence_prefixes",
                       "input_token_ids", "input_lengths", "context_prefixes", "prefill")), after
            assert after["map_unchanged"] and after["output_finite"], after
            if not advance:
                assert (after["count_nonzero"] == 0 and after["received_assignments"] == after["layer0_received"] == 0
                        or self.role == "prefill" and after["count_nonzero"] is None
                        and after["received_assignments"] is after["layer0_received"] is None), after
                assert after["counters_unchanged"] and not after["raw_report"] and after["output_zero"], after
                assert not after["raw_writes"] and not after["raw_pools"], after
            else:
                assert after["raw_report"], after
                if after["idle"]:
                    assert not after["raw_writes"] and not after["raw_pools"] and after["idle_routes_zero"], after
                    assert after["output_zero"], after
                    # Idle source ranks may receive real EP assignments.
        received = sum(r["received_assignments"] or 0 for r in rows)
        layer0 = sum(r["layer0_received"] or 0 for r in rows)
        if advance and self.role == "decode":
            assert received > 0 and layer0 > 0, rows
        if not advance:
            self.empty_total += 1
        self.business_calls += advance
        frame = {"round": self.round, "kind": pending["kind"], "business_forward": self.business_calls,
                 "stage": pending["stage"], "lineage": pending["before"],
                 "global_received_assignments": received, "global_layer0_received": layer0}
        self.pending = None
        return frame


class NativeCallObserver:
    """Observe existing forward/report boundaries on both native roles."""

    def __init__(self, model):
        self.model, self.engine = model, model._shared_ops.moe_engine
        self.layers, self.rank = model.get_eplb_layers(), dist.get_rank()
        self.directory = Path(os.environ["RTP_V41_N8_EPLB_REPORT_DIR"])
        self.ledger = NativeObservationLedger("decode" if model._is_decode_role else "prefill")
        self.bindings = _bindings(self.engine, self.layers)
        self.calls = model.shared_runtime.calls
        self.frames = []
        self.count_pointer = None
        self.empty_phases = {"before_business": 0, "between_business": 0}
        self.empty_samples = {}
        self.done = False
        self._pending_observation = None

    def _counters(self):
        return (self.engine.grouped_forward_calls, self.engine.grouped_gemm_calls, self.engine.mega_moe_calls,
                tuple((layer.attn_norm.calls, layer.ffn_norm.calls, layer.ffn.calls) for layer in self.model.v4.layers))

    def _map_signature(self):
        return hashlib.sha256(json.dumps([[_hash(layer.logical_to_physical), _hash(layer.logical_count),
                                          _hash(layer.physical_to_logical)] for layer in self.layers]).encode()).hexdigest()

    def _raw_path(self, call):
        return Path(os.environ["RTP_V41_SHARED_EVIDENCE_DIR"]) / f"rank{self.rank}-call{call:05d}.json"

    def _row(self, inputs, phase):
        from rtp_llm.models_py.modules.dsv41.kv_cache_utils import as_tagged_attention_inputs

        attn = as_tagged_attention_inputs(inputs.attention_inputs)
        cp = getattr(attn, "context_parallel_info", None)
        counts = inputs.expert_token_counts
        if counts is None:
            assert self.ledger.role == "prefill", "Decode native stable counts required"
            pointer = None
        else:
            assert tuple(counts.shape) == (40, 49) and counts.dtype == torch.int32
            pointer = counts.data_ptr()
        if self.ledger.round == 0 and phase == "before":
            self.count_pointer = pointer
        return {"rank": self.rank, "round": self.ledger.round + int(phase == "before"), "phase": phase,
                "idle": bool(inputs.is_fake_stream), "warmup": bool(inputs.warmup),
                "business_call": self.model.shared_runtime.calls,
                "bindings_stable": pointer == self.count_pointer and _bindings(self.engine, self.layers) == self.bindings,
                "map_signature": self._map_signature(),
                "request_ids": _values(inputs.request_id), "sequence_prefixes": _values(attn.sequence_lengths),
                "input_token_ids": _values(inputs.input_ids), "prefill": bool(attn.is_prefill),
                "input_lengths": _values(cp.prefill_actual_input_lengths_cpu if cp is not None else attn.input_lengths),
                "context_prefixes": _values(cp.prefill_prefix_lengths_cpu if cp is not None else attn.prefix_lengths),
                "count_nonzero": int(torch.count_nonzero(counts)) if counts is not None else None,
                "received_assignments": int(counts.sum()) if counts is not None else None,
                "layer0_received": int(counts[0].sum()) if counts is not None else None}

    def before(self, inputs):
        assert self._pending_observation is None
        row = self._row(inputs, "before")
        rows = [None] * 8
        dist.all_gather_object(rows, row)
        pending = self.ledger.begin(rows)
        self._pending_observation = {"row": row, "counters": self._counters(), "classification": pending}
        return pending

    def after(self, inputs, output):
        pending = self._pending_observation
        assert pending is not None
        row = self._row(inputs, "after")
        path = self._raw_path(pending["row"]["business_call"] + 1)
        native = json.loads(path.read_text()) if path.is_file() else None
        if native is not None:
            assert native["rank"] == self.rank and native["call"] == row["business_call"]
            assert native["idle"] == row["idle"]
        row.update(map_unchanged=row["map_signature"] == pending["row"]["map_signature"],
                   counters_unchanged=self._counters() == pending["counters"], raw_report=native is not None,
                   raw_writes=bool(native and native["native_writes"]), raw_pools=bool(native and native["native_pool_bindings"]),
                   idle_routes_zero=bool(native and len(native["routed_active_rows"]) == 40 and not any(native["routed_active_rows"].values())),
                   output_finite=bool(torch.isfinite(output.hidden_states).all()),
                   output_zero=int(torch.count_nonzero(output.hidden_states)) == 0)
        rows = [None] * 8
        dist.all_gather_object(rows, row)
        frame = self.ledger.finish(rows)
        self.calls = self.model.shared_runtime.calls
        self.frames = (self.frames + [frame])[-2:]
        self._pending_observation = None
        if frame["kind"] == "all_empty":
            phase = "between_business" if self.calls else "before_business"
            self.empty_phases[phase] += 1
            self.empty_samples[phase] = {"native_round": frame["round"], "business_forward": self.calls,
                                         "ranks": [{key: item[key] for key in (
                                             "rank", "count_nonzero", "received_assignments", "layer0_received",
                                             "map_unchanged", "counters_unchanged", "raw_report", "raw_writes",
                                             "raw_pools", "output_zero")}
                                             for item in sorted(rows, key=lambda item: item["rank"])]}
            if self.empty_phases[phase] == 1:
                self._checkpoint()
        else:
            self._checkpoint()
        return frame, native

    def finalize(self):
        assert not self.done and self._pending_observation is None
        assert self.frames[-1]["stage"] == "fresh"
        self.done = True
        self._checkpoint()

    def _checkpoint(self):
        # One small overwritten receipt per rank, not one artifact per idle.
        _write(self.directory / f"rank{self.rank}-observations.json",
               {"rank": self.rank, "pid": os.getpid(), "model_identity": id(self.model),
                "native_rounds": self.ledger.round, "business_forwards": self.ledger.business_calls,
                "final": self.done,
                "all_empty_rounds": self.ledger.empty_total, "empty_phases": self.empty_phases,
                "last_empty_sample_by_phase": self.empty_samples,
                "observed_layout_changes": getattr(self, "epoch_serial", 0),
                "last_layout_change_basis": (getattr(self, "latest_epoch", None) or {}).get("stats_basis"),
                "last_round_kind": self.frames[-1]["kind"] if self.frames else None,
                "count_pointer": self.count_pointer,
                "all_empty_checks": "zero output and zero actual counts when present; no raw report/call increment/layer execution/map change inside forward",
                "public_step_boundary": "C++ step runs AFTER this observation and may legally change maps even for zero counts"})


class NativeEplbEvidence(NativeCallObserver):
    """One representative layer's complete expert bytes, all40 bindings."""

    def __init__(self, model):
        from rtp_llm.eplb.runtime import EplbLayer, RuntimeExpertBalancer
        from rtp_llm.utils.model_weight import W

        super().__init__(model)
        options = model.config.eplb_config
        if (options.eplb_update_time, options.eplb_stats_window_size) != (2, 2):
            raise ValueError("N8 Decode observer requires the explicit public interval2/window2 fixture")
        if self.layers is not model.get_eplb_layers() or len(self.layers) != 40:
            raise AssertionError("native target requires all40 cached EPLB bindings")
        if any(type(layer) is not EplbLayer or layer.layer_id != i for i, layer in enumerate(self.layers)):
            raise AssertionError("bindings must be exact public EplbLayer with target IDs0..39")
        if (model.shared_weight_views.geometry.n_routed_experts,
                model.shared_weight_views.geometry.n_activated_experts,
                self.engine.num_physical_experts, self.engine.num_local_experts) != (384, 6, 392, 49):
            raise AssertionError("smallest legal grouped EP8 fixture requires logical384/top6/physical392/local49")
        self.layer = self.layers[0]
        self.directory = Path(os.environ["RTP_V41_N8_EPLB_REPORT_DIR"])
        self.rank = dist.get_rank()
        self.calls = 0
        self.done = False
        self.continuation = None
        self.trigger_request_ids = None
        self.trigger_prefixes = None
        self.migration = None
        self.business_baseline = None
        self.epoch_serial = 0
        self.last_map = self.layer.physical_to_logical.cpu().clone()
        self.last_state = self._state_signature()
        self.latest_epoch = None
        self.latest_real_epoch = None
        self.bindings = self._bindings()
        self.initial_map = self.layer.physical_to_logical.cpu().clone()
        self.initial_inverse = self.layer.logical_to_physical.cpu().clone()
        self.inventory = self.engine._eplb_inventory[0]
        # Independent expected bytes are captured before any real model step.
        # Keep candidate source signatures for redundant replicas (layout
        # padding can legitimately differ), rather than deriving from new maps.
        local = []
        for slot in range(49):
            physical = self.rank * 49 + slot
            local.append({"physical": physical, "logical": int(self.initial_map[physical]),
                          "hashes": [_hash(view[slot]) for _, _, view in self.inventory]})
        gathered = [None] * 8
        dist.all_gather_object(gathered, local)
        self.expected = {}
        for rank_slots in gathered:
            for slot in rank_slots:
                self.expected.setdefault(slot["logical"], []).append(slot)
        if set(self.expected) != set(range(384)):
            raise AssertionError("loaded redundant experts do not cover all logical checkpoint experts")
        raw_indices = [i for i, (name, _, _) in enumerate(self.inventory) if not name.startswith("processed.")]
        for logical, slots in self.expected.items():
            if any(any(slot["hashes"][i] != slots[0]["hashes"][i] for i in raw_indices) for slot in slots[1:]):
                raise AssertionError(f"framework loader did not load identical packed/scale bytes for redundant logical{logical}")
        loaded_duplicates = sum(len(slots) - 1 for slots in self.expected.values())
        if loaded_duplicates != 8:
            raise AssertionError("framework loader did not populate all8 assigned redundant slots")
        self.count_pointer = None
        self.count_history = []
        self.seed = {"schema": "v41-ppu-native-eplb-v1", "rank": self.rank,
                     "pid": os.getpid(), "model_identity": id(model), "engine_identity": id(self.engine),
                     "backend": self.engine.backend, "precision": self.engine.precision,
                     "bindings": self.bindings, "representative_layer": 0,
                     "initial_physical_to_logical": self.initial_map.tolist(),
                     "initial_logical_to_physical": self.initial_inverse.tolist(),
                     "inventory": [{"name": name, "executor": _tensor(t), "migration_view": _tensor(v)}
                                   for name, t, v in self.inventory],
                     "inverse_scale_invariants": self.engine._eplb_invariants[0],
                     "public_runtime_source": {"path": inspect.getsourcefile(RuntimeExpertBalancer),
                         "sha256": hashlib.sha256(Path(inspect.getsourcefile(RuntimeExpertBalancer)).read_bytes()).hexdigest()},
                     "initial_slots": local, "independent_global_sources": gathered,
                     "loaded_redundant_slots": loaded_duplicates, "duplicate_raw_bytes_equal": True,
                     "vocab_aliases": {"embedding": model.v4.embed.weight is model.weight.global_weights[W.embedding],
                                       "lm_head": model.v4.head_weight is model.weight.global_weights[W.lm_head],
                                       "lm_head_pointer": model.v4.head_weight.data_ptr()},
                     "balancing_owner": "unchanged C++ PyWrappedModel/ExpertBalancerPythonWrapper"}
        _write(self.directory / f"rank{self.rank}-before.json", self.seed)

    def _bindings(self):
        return _bindings(self.engine, self.layers)

    def _state_signature(self):
        return hashlib.sha256("".join(_hash(t) for t in (self.layer.logical_to_physical,
                              self.layer.logical_count, self.layer.physical_to_logical)).encode()).hexdigest()

    def _slot_proof(self, current, slots):
        records = []
        for slot in range(49):
            physical = self.rank * 49 + slot
            if physical not in slots:
                continue
            logical = int(current[physical])
            hashes = [_hash(view[slot]) for _, _, view in self.inventory]
            sources = [source["physical"] for source in self.expected[logical] if source["hashes"] == hashes]
            if not sources:
                raise AssertionError(f"actual raw/processed/scale bytes do not follow slot{physical}/logical{logical}")
            records.append({"physical": physical, "local": slot,
                            "new_logical": logical, "hashes": hashes, "matching_pre_migration_sources": sources,
                            "bytes_by_inventory": [view.shape[1] for _, _, view in self.inventory]})
        return records

    def _observe_layout(self, current, signature):
        if signature == self.last_state:
            return
        basis = migration_window_basis(self.frames)
        assert self.frames[-1]["round"] + 1 == self.ledger.round
        moved = torch.where(current != self.last_map)[0].tolist()
        records = self._slot_proof(current, set(moved))
        for record in records:
            record["old_logical"] = int(self.last_map[record["physical"]])
        mapping = self.layer.logical_to_physical
        replicas = self.layer.logical_count
        valid = torch.arange(mapping.shape[1], device=mapping.device)[None] < replicas[:, None]
        if (bool(((replicas < 1) | (replicas > mapping.shape[1])).any())
                or not torch.equal(mapping[valid].sort().values.cpu(), torch.arange(392, dtype=mapping.dtype))
                or not torch.equal(current[mapping[valid].cpu().long()], torch.where(valid.cpu())[0].to(current.dtype))
                or bool((mapping[~valid] != -1).any())):
            raise AssertionError("post-migration maps do not match actual physical slot ownership")
        self.epoch_serial += 1
        self.latest_epoch = {"observed_layout_epoch": self.epoch_serial, "observed_before_native_round": self.ledger.round,
                             "business_forwards_already_completed": self.calls,
                             "stats_basis": basis, "source_window": list(self.frames),
                             "global_moved_slot_count": len(moved), "global_moved_slots": moved,
                             "local_moved_slots": records, "old_physical_to_logical": self.last_map.tolist(),
                             "physical_to_logical": current.tolist(), "logical_to_physical": mapping.tolist(),
                             "logical_count": replicas.tolist(), "mapping_sha256": _hash(mapping),
                             "layout_sha256": signature, "all_layer_pointers_and_inverse_scales_stable": True}
        if moved and basis == "actual_business_window":
            self.latest_real_epoch = self.latest_epoch
        self.last_map, self.last_state = current, signature

    def before(self, inputs):
        pending = super().before(inputs)
        current = self.layer.physical_to_logical.cpu().clone()
        signature = self._state_signature()
        self._observe_layout(current, signature)
        if pending["kind"] == "business" and self.calls == 0:
            # Startup empty updates may already have moved the loaded slots.
            # This second baseline is the actual layout entering prefix257.
            self.business_baseline = {"rank": self.rank, "native_round": self.ledger.round,
                                      "business_forward": 1, "prefix": 257, "layout_sha256": signature,
                                      "physical_to_logical": current.tolist(),
                                      "logical_to_physical": self.layer.logical_to_physical.tolist(),
                                      "logical_count": self.layer.logical_count.tolist(),
                                      "local_slots": self._slot_proof(current, set(range(392)))}
            _write(self.directory / f"rank{self.rank}-business-baseline.json", self.business_baseline)

    def after(self, inputs, output):
        frame, native_forward = super().after(inputs, output)
        if frame["kind"] == "all_empty":
            return
        counts = inputs.expert_token_counts
        lineage = frame["lineage"]
        local_lineage = lineage[self.rank]
        business = [item for item in lineage if not item["idle"]]
        request_ids = {rid for item in business for rid in item["request_ids"]}
        if self.trigger_request_ids is None:
            self.trigger_request_ids = request_ids
            self.trigger_prefixes = [prefix for item in business for prefix in item["sequence_prefixes"]]
        fresh = frame["stage"] == "fresh"
        self.count_history.append({"forward": self.calls, "native_round": frame["round"],
                                   "received_counts": counts.cpu().tolist(), "lineage": lineage})
        if self.continuation is not None and not fresh:
            return
        if fresh:
            if self.continuation is None:
                raise AssertionError("fresh prefix17 requires prior trigger consumption of a real-load migration")
            # A legal later public empty update may publish another layout.
            # Prove every current local slot independently in that layout.
            fresh_bytes = self._slot_proof(self.last_map, set(range(392)))
            # Both epochs are present in the delivered report. The receiver
            # can independently recompute the union from their ownership maps.
            moved_slots = epoch_moved_slots(self.migration) | epoch_moved_slots(self.latest_epoch)
        else:
            epoch = self.latest_real_epoch
            if epoch is None or epoch["layout_sha256"] != self.last_state:
                return
            moved_slots = set(epoch["global_moved_slots"])
            fresh_bytes = None
        routes = self.engine._eplb_routes.get(0)
        if routes is None:
            raise AssertionError("post-migration forward did not consume the bound logical map")
        logical, physical, weights = routes
        if not torch.equal(physical, self.layer.map_ids(logical)):
            raise AssertionError("subsequent actual dispatch IDs do not consume updated map")
        if bool((weights[physical < 0] != 0).any()) or not torch.equal(physical[logical < 0], logical[logical < 0]):
            raise AssertionError("logical-to-physical mapping damaged inactive/padded slots")
        slots = torch.tensor([slot in moved_slots for slot in range(392)], dtype=torch.bool, device=physical.device)
        consumed_moved = ((logical >= 0) & slots[physical.clamp_min(0).long()]).sum()
        dist.all_reduce(consumed_moved)
        if int(consumed_moved) <= 0:
            if fresh:
                raise AssertionError("fresh dispatch did not use any validated moved physical expert slot")
            return
        if not fresh:
            self.migration = epoch
        if self._bindings() != self.bindings:
            raise AssertionError("post-migration forward invalidated a live executor binding")
        forward_path = self._raw_path(self.calls)
        if not local_lineage["idle"] and (not native_forward["native_state_before"]
                or not any(item["nonzero_bytes"] > 0 for item in native_forward["native_state_before"].values())):
            raise AssertionError("post-migration real request lacks live native RDMA cache state")
        report = {**self.seed, "ok": True, "SMOKE": "PASS_NATIVE_EPLB_SAME_INSTANCE",
                  "migration": self.migration, "step_statistics": self.count_history,
                  "pre_first_business_baseline": self.business_baseline,
                  "post_migration": {"model_identity": id(self.model), "engine_identity": id(self.engine),
                      "forward": self.calls, "native_round": frame["round"],
                      "global_received_assignments": frame["global_received_assignments"],
                      "global_dispatch_assignments_to_moved_slots": int(consumed_moved),
                      "logical_ids": logical.cpu().tolist(), "physical_ids": physical.cpu().tolist(),
                      "route_weights": weights.cpu().tolist(), "output_finite": True,
                      "hidden_sha256": _hash(output.hidden_states.contiguous().view(torch.uint8)),
                      "count_pointer": self.count_pointer, "native_cache": type(self.model.kv_cache).__name__,
                      "purpose": "fresh_request_after_migration" if fresh else "trigger_request_continuation",
                      "request_lineage_all_ranks": lineage, "trigger_request_ids": sorted(self.trigger_request_ids),
                      "fresh_request_ids": sorted(request_ids) if fresh else [],
                      "request_id_scope": "normal inputs carry context IDs only; Decode IDs may be empty",
                      "lineage_basis": "serial HTTP traces/P context IDs plus actual native D prefix reset",
                      "trigger_initial_prefixes": self.trigger_prefixes,
                      "fresh_prefixes": [17] if fresh else [],
                      "native_forward_evidence": str(forward_path),
                      "native_state_before": native_forward["native_state_before"],
                      "native_writes": native_forward["native_writes"],
                      "yoco": native_forward["yoco"], "map_sha256": _hash(self.layer.logical_to_physical),
                      "layout_sha256": self.last_state, "observed_layout_epoch": self.epoch_serial,
                      "routing_movement_basis": "verified_trigger_and_current_epoch_slots" if fresh else "actual_business_window_migration",
                      "routed_movement_slots": sorted(moved_slots),
                      "fresh_current_epoch": self.latest_epoch if fresh else None,
                      "fresh_current_local_slot_bytes": fresh_bytes,
                      "moe": self.engine.execution_stats()},
                  "FULL_VALIDATION": "DEFERRED_TO_N10"}
        if fresh:
            report["trigger_continuation_forward"] = self.continuation["post_migration"]["forward"]
            _write(self.directory / f"rank{self.rank}-fresh-request.json", report)
            self.finalize()
        else:
            _write(self.directory / f"rank{self.rank}-after.json", report)
            self.continuation = report


def validate_eplb_forward_reports(role, reports):
    """Validate the two ordinary requests without changing their real reports.

    Rank-local call numbers identify the collective step. Classify that step
    using business rows, keeping its original idle peers in the same group.
    The caller applies the unchanged native validator to trigger_reports.
    """
    assert role in ("prefill", "decode") and reports, (role, reports)
    calls = {}
    seen = set()
    for item in reports:
        rank, call = item["rank"], item["call"]
        assert type(rank) is int and 0 <= rank < 8, item
        assert type(call) is int and call > 0, item
        assert (rank, call) not in seen, (rank, call)
        seen.add((rank, call))
        calls.setdefault(call, []).append(item)
        assert item["moe"]["backend"] == "grouped" and item["moe"]["precision"] == "w4a8", item
        assert item["model_layers"] == 40, item
        if role == "decode":
            assert item["prefill"] is False, item
        if item["idle"]:
            assert len(item["routed_active_rows"]) == 40 and not any(item["routed_active_rows"].values()), item
            assert not item["native_writes"] and not item["native_pool_bindings"], item

    trigger_calls, fresh_calls = [], []
    for call, items in sorted(calls.items()):
        assert {item["rank"] for item in items} == set(range(8)), (call, items)
        business = [item for item in items if not item["idle"]]
        assert business, (call, items)
        if role == "prefill":
            assert all(item["prefill"] is True for item in business), business
            lengths = {tuple(item["yoco"]["input_lengths"]) for item in business}
            assert lengths in ({(257,)}, {(17,)}), (call, lengths)
            fresh = lengths == {(17,)}
        else:
            assert len(business) == 1, business
            yoco = business[0]["yoco"]
            assert yoco["input_lengths"] == [1] and yoco["prefix_lengths"] == yoco["local_positions"], business
            positions = yoco["local_positions"]
            assert len(positions) == 1 and positions[0] in (17, 257, 258, 259, 260, 261), positions
            fresh = positions == [17]
        (fresh_calls if fresh else trigger_calls).append(call)

    assert trigger_calls and len(fresh_calls) == 1 and max(trigger_calls) < fresh_calls[0], (trigger_calls, fresh_calls)
    trigger = [item for item in reports if item["call"] in trigger_calls]
    fresh = [item for item in reports if item["call"] in fresh_calls]
    real = [item for item in fresh if not item["idle"]]
    assert real and any(item["moe"]["grouped_gemm_calls"] > 0 for item in real), fresh
    for item in real:
        assert item["yoco"]["executed_layers"] == list(range(40)), item
        assert item["profile"] == {"mode": "compact", "keep_tokens": 140, "window": 128,
                                   "max_rollback_tokens": 0, "encoder_replay": False}, item
        assert item["native_pool_bindings"] and all(pool["native_alias"] is True for pool in item["native_pool_bindings"]), item
        assert item["native_writes"] and item["output_finite"] is True, item
        normalization = item["normalization"]
        assert normalization["attention_norm_calls"] == 40 and normalization["ffn_calls"] == 40, item
        assert normalization["ffn_norm_calls"] + normalization["mega_transitions"] == 40, item
        state = item["native_state_after"] if role == "prefill" else item["native_state_before"]
        assert state and any(pool["nonzero_bytes"] > 0 for pool in state.values()), item

    if role == "prefill":
        assert len(trigger_calls) == 1 and {item["rank"] for item in real} == set(range(8)), real
        positions = set()
        for item in real:
            yoco = item["yoco"]
            assert yoco["prefix_lengths"] == [0] and yoco["logical_retained_rows"] == 17, item
            assert yoco["cp_size"] == 8 and yoco["compacted"] is False, item
            positions.update(yoco["local_positions"])
        assert positions == set(range(17)), positions
        fresh_evidence = {"CP_ranks": list(range(8)), "input_tokens": 17, "cold_prefix": 0,
                          "logical_rows": 17, "positions": [0, 16], "compacted": False}
    else:
        trigger_positions = sorted({p for item in trigger if not item["idle"] for p in item["yoco"]["local_positions"]})
        assert len(trigger_calls) == 5 and trigger_positions == list(range(257, 262)), trigger_positions
        assert len(real) == 1 and real[0]["yoco"]["local_positions"] == [17], real
        assert real[0]["mega_mhc"]["calls"].get("decode", 0) > 0, real
        fresh_evidence = {"destination_owners": [real[0]["rank"]], "prefill_executed": False,
                          "decode_positions": [17], "live_native_state": True}
    return {"trigger_reports": trigger, "fresh": fresh_evidence,
            "grouping": {"trigger_calls": trigger_calls, "fresh_calls": fresh_calls,
                         "trigger_report_count": len(trigger), "fresh_report_count": len(fresh),
                         "all_original_reports_partitioned": len(trigger) + len(fresh) == len(reports)}}


def _validate_slot_sources(report, slots):
    sources = {slot["physical"]: slot for rank in report["independent_global_sources"] for slot in rank}
    for slot in slots:
        matches = slot["matching_pre_migration_sources"]
        assert matches and all(sources[source]["logical"] == slot["new_logical"] and
                               sources[source]["hashes"] == slot["hashes"] for source in matches), slot


def validate_native_reports(directory):
    directory = Path(directory)
    reports = [json.loads((directory / f"rank{rank}-after.json").read_text()) for rank in range(8)]
    fresh_reports = [json.loads((directory / f"rank{rank}-fresh-request.json").read_text()) for rank in range(8)]
    observations = [json.loads((directory / f"rank{rank}-observations.json").read_text()) for rank in range(8)]
    reference = reports[0]["migration"]
    window = reference["source_window"]
    if migration_window_basis(window) != "actual_business_window" or not any(f["stage"] == "trigger" for f in window):
        raise AssertionError("zero-count-only public updates cannot establish real-load migration")
    for frame in window:
        if frame["kind"] == "business":
            measured = sum(sum(next(step["received_counts"][0] for step in report["step_statistics"]
                                    if step["native_round"] == frame["round"])) for report in reports)
            assert measured == frame["global_layer0_received"] > 0, frame
    for rank, report in enumerate(reports):
        migration, post = report["migration"], report["post_migration"]
        if (not report.get("ok") or report["SMOKE"] != "PASS_NATIVE_EPLB_SAME_INSTANCE"
                or report["rank"] != rank or migration["global_moved_slot_count"] <= 0
                or any(migration[k] != reference[k] for k in ("layout_sha256", "source_window", "global_moved_slots"))
                or migration["stats_basis"] != "actual_business_window"
                or post["model_identity"] != report["model_identity"] or post["engine_identity"] != report["engine_identity"]
                or not max(f["business_forward"] for f in window) < post["forward"] <= 5
                or post["native_round"] < migration["observed_before_native_round"]
                or migration["observed_before_native_round"] != window[-1]["round"] + 1
                or post["layout_sha256"] != migration["layout_sha256"]
                or post["global_dispatch_assignments_to_moved_slots"] <= 0
                or not all(report["vocab_aliases"][key] for key in ("embedding", "lm_head"))):
            raise AssertionError("N8 requires real migration followed by same-instance native execution")
        baseline = report["pre_first_business_baseline"]
        assert baseline["prefix"] == 257 and baseline["business_forward"] == 1 and len(baseline["local_slots"]) == 49
        assert baseline["native_round"] <= window[-1]["round"]
        assert post["routed_movement_slots"] == sorted(epoch_moved_slots(migration))
        _validate_slot_sources(report, baseline["local_slots"])
        _validate_slot_sources(report, migration["local_moved_slots"])
    for rank, (initial, report, observation) in enumerate(zip(reports, fresh_reports, observations)):
        fresh = report["post_migration"]
        current = fresh["fresh_current_epoch"]
        if (not report.get("ok") or fresh["purpose"] != "fresh_request_after_migration"
                or fresh["forward"] != 6 or fresh["native_round"] <= initial["post_migration"]["native_round"]
                or any(report[key] != initial[key] for key in ("pid", "model_identity", "engine_identity", "bindings"))
                or report["migration"] != initial["migration"]
                or fresh["count_pointer"] != initial["post_migration"]["count_pointer"]
                or fresh["fresh_prefixes"] != [17] or fresh["trigger_initial_prefixes"] != [257]
                or current["layout_sha256"] != fresh["layout_sha256"] or current["mapping_sha256"] != fresh["map_sha256"]
                or fresh["global_dispatch_assignments_to_moved_slots"] <= 0):
            raise AssertionError("fresh short request must use the same PID/model/engine/pointers and verified current layout")
        current_bytes = fresh["fresh_current_local_slot_bytes"]
        proven_slots = epoch_moved_slots(report["migration"]) | epoch_moved_slots(current)
        assert fresh["routed_movement_slots"] == sorted(proven_slots)
        assert [slot["physical"] for slot in current_bytes] == list(range(rank * 49, (rank + 1) * 49))
        assert all(slot["matching_pre_migration_sources"] and
                   slot["new_logical"] == current["physical_to_logical"][slot["physical"]] for slot in current_bytes)
        _validate_slot_sources(report, current_bytes)
        _validate_slot_sources(report, current["local_moved_slots"])
        assert observation["rank"] == rank and observation["pid"] == report["pid"] and observation["model_identity"] == report["model_identity"]
        assert observation["final"] is True and observation["native_rounds"] == fresh["native_round"]
        assert observation["business_forwards"] == 6 and observation["count_pointer"] == fresh["count_pointer"]
        assert observation["native_rounds"] == 6 + observation["all_empty_rounds"]
        assert sum(observation["empty_phases"].values()) == observation["all_empty_rounds"]
    actual_moved = sorted(slot["physical"] for report in reports for slot in report["migration"]["local_moved_slots"])
    if actual_moved != reference["global_moved_slots"] or len(actual_moved) != reference["global_moved_slot_count"]:
        raise AssertionError("moved expert byte evidence is incomplete across ranks")
    for group in (reports, fresh_reports):
        posts = [report["post_migration"] for report in group]
        slots = set(posts[0]["routed_movement_slots"])
        assert all(set(post["routed_movement_slots"]) == slots and post["layout_sha256"] == posts[0]["layout_sha256"]
                   for post in posts)
        actual_hits = sum(logical >= 0 and physical in slots for post in posts
                          for logical_row, physical_row in zip(post["logical_ids"], post["physical_ids"], strict=True)
                          for logical, physical in zip(logical_row, physical_row, strict=True))
        assert actual_hits > 0 and all(post["global_dispatch_assignments_to_moved_slots"] == actual_hits for post in posts)
    return {"ranks": list(range(8)), "moved_slots": len(actual_moved), "real_load_source_window": window,
            "same_instance_continuation": True, "raw_processed_aux_bytes": "EXACT_SOURCE_MATCH",
            "fresh_post_migration_request": True, "short_requests_in_one_load": 2,
            "fresh_layout_sha256": fresh_reports[0]["post_migration"]["layout_sha256"],
            "fresh_layout_matches_trigger_epoch": fresh_reports[0]["post_migration"]["layout_sha256"] == reference["layout_sha256"],
            "native_observations": observations,
            "native_exit_and_resource_reception": "L0_REQUIRED"}


def validate_prefill_lineage(directory):
    records = [json.loads(path.read_text()) for path in sorted(Path(directory).glob("rank*-prefill*.json"))]
    first = [r for r in records if r["yoco"]["input_lengths"] == [257]]
    fresh = [r for r in records if r["yoco"]["input_lengths"] == [17]]
    if {r["rank"] for r in first} != set(range(8)) or {r["rank"] for r in fresh} != set(range(8)):
        raise AssertionError("both ordinary HTTP requests must execute actual CP8 Prefill")
    first_ids = {rid for r in first for rid in r["request_ids"]}
    fresh_ids = {rid for r in fresh for rid in r["request_ids"]}
    if not first_ids or not fresh_ids or not first_ids.isdisjoint(fresh_ids):
        raise AssertionError("post-migration HTTP request must have a fresh native P context identity")
    for r in fresh:
        before = next(value for value in first if value["rank"] == r["rank"])
        if (any(r[key] != before[key] for key in ("pid", "model_identity", "engine_identity", "bindings", "map_sha256"))
                or r["yoco"]["prefix_lengths"] != [0] or not r["output_finite"]
                or not any(item["nonzero_bytes"] > 0 for item in r["native_state_after"].values())):
            raise AssertionError("fresh Prefill must populate native state in the same loaded P model")
    observations = [json.loads((Path(directory) / f"rank{rank}-observations.json").read_text()) for rank in range(8)]
    for item in observations:
        prefill = next(r for r in fresh if r["rank"] == item["rank"])
        assert item["final"] is True and item["business_forwards"] == 2
        assert item["native_rounds"] == 2 + item["all_empty_rounds"] == 2 + sum(item["empty_phases"].values())
        assert item["pid"] == prefill["pid"] and item["model_identity"] == prefill["model_identity"]
    return {"trigger_context_ids": sorted(first_ids), "fresh_context_ids": sorted(fresh_ids),
            "same_servers": True, "HTTP_traces": ["n8-trigger-0", "n8-postmigration-0"],
            "native_observations": observations}

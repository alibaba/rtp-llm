"""Receipts from the public Decode runner and a direct DSpark replay driver."""

import json
import math
import sys
from types import SimpleNamespace


def validate_native_receipts(paths):
    reports = {}
    for path in paths:
        record = json.loads(path.read_text())
        key = record["rank"], record["replay"]
        if key in reports or record["schema"] != "n9-native-grouped-decode-replay-v1":
            raise ValueError("invalid or duplicate native Graph receipt")
        reports[key] = record
    if set(reports) != {(rank, step) for rank in range(8) for step in (1, 2)}:
        raise ValueError("expected two actual runner replays per EP rank")
    owners = []
    for step in (1, 2):
        peers = [reports[rank, step] for rank in range(8)]
        live = [peer for peer in peers if peer["valid"] == [True]]
        if len(live) != 1 or live[0]["positions"] != [256 + step]:
            raise ValueError("runner did not consume the restored request at the current position")
        owner = live[0]
        owners.append(owner["rank"])
        if owner["swa_slots"][0][0] <= 0 or owner["engram_windows"][0][0] != owner["token_ids"][0]:
            raise ValueError("runner did not consume a real page and scheduler token window")
        if not any(domain["nonzero_bytes"] for domain in owner["domains"].values()):
            raise ValueError("runner did not write a live native cache row")
        for peer in peers:
            if len(peer["received"]) != 40 or len(peer["domains"]) != 51 or (
                peer["reserved_unchanged"] != [True] * 51
            ):
                raise ValueError("native full40 or reserved pool geometry differs")
            if peer is not owner and (
                peer["valid"] != [False] or peer["positions"] != [-1]
                or peer["swa_slots"] != [[-1]]
                or any(domain["nonzero_bytes"] for domain in peer["domains"].values())
            ):
                raise ValueError("idle owner acquired an active native slot")
        if [sum(sum(peer["received"][layer]) for peer in peers) for layer in range(40)] != [6] * 40:
            raise ValueError("physical EP ranks did not receive six actual routes per layer")
    if owners[0] != owners[1]:
        raise ValueError("the representative recovered request changed owners")
    first, second = (reports[owners[0], step] for step in (1, 2))
    if first["engram_windows"] == second["engram_windows"] or first["swa_slots"] == second["swa_slots"]:
        raise ValueError("the representative replay did not change Engram history and slot")
    for rank in range(8):
        a, b = reports[rank, 1], reports[rank, 2]
        if a["addresses"] != b["addresses"] or len(a["layer_calls"]) != 40 or (
            b["layer_calls"] != [count + 1 for count in a["layer_calls"]]
        ):
            raise ValueError("runner changed an address or did not execute all 40 layers")
    return {
        "ok": True, "bucket": 1, "ranks": list(range(8)), "replays_per_rank": 2,
        "owner": owners[0], "positions": [257, 258], "received_per_layer_per_step": 6,
        "fixed_addresses": True, "reserved_blocks_unchanged": True,
    }


def run_model_replays(
    model, mode, capture_inputs, steps, coordinates, *, permitted_writes=None, diagnostics=None,
    initial_states=None,
):
    """Use two scheduler inputs and their actual acceptance on a loaded model.

    This callable needs a real initialized target/draft model in the caller's
    process, isolated from serving by the fixture owner. It does not start,
    replace or instrument the public runner. The supplied coordinates are
    physical slots in all relevant live pages. A caller can additionally
    snapshot reserved block zero directly as a canary.
    """
    import hashlib

    import torch
    import torch.distributed as dist

    from rtp_llm.models_py.modules.dsv41.kv_cache_utils import primary_attention_inputs
    from rtp_llm.models_py.modules.dsv41.ppu.tools.driver_shared_graph_dp8 import (
        _hidden_comparison, _restore, _snapshot, _state_mismatches,
    )

    if len(steps) != 2 or type(model.kv_cache).__name__ != "KVCache" or dist.get_world_size() != 8:
        raise ValueError("require eight live native model ranks and two replay inputs")
    if mode in ("decode", "dense", "packed"):
        impl = model.prepare_fmha_impl(capture_inputs, is_cuda_graph=True)
        call = model.forward
    elif mode in ("proposal", "commit"):
        impl = model.prepare_dspark_graph(capture_inputs, mode)
        call = model.forward_propose if mode == "proposal" else model.forward_commit
    else:
        raise ValueError(f"unknown direct DSpark replay mode {mode}")
    if mode != "decode" and impl.mode != mode:
        raise ValueError("the supplied capture inputs selected another mode")
    cache = impl.cache if mode == "decode" else impl.native_cache
    if set(coordinates) != set(cache.geometry.pool_keys()) or any(
        not slots.numel() or bool((slots < cache.entries[cache._table_tag(*key)]).any())
        for key, slots in coordinates.items()
    ):
        raise ValueError("fixture must snapshot real slots in every touched native domain")

    def stage_accept(inputs, accepted, output, *, captured):
        if mode in ("dense", "packed"):
            if captured:
                model.shared_runtime.stage_graph_verify(
                    inputs, impl, output["mtp_target_hidden_states"]
                )
            attn = primary_attention_inputs(inputs.attention_inputs, model.kv_cache)
            model.finish_speculative_forward(inputs.request_id, attn.prefix_lengths, accepted)

    def update(inputs):
        count = primary_attention_inputs(inputs.attention_inputs, model.kv_cache).input_lengths.numel()
        impl.prepare_cuda_graph(inputs.attention_inputs, valid_request_count=0 if inputs.is_fake_stream else count)
        capture_inputs.input_ids.copy_(inputs.input_ids)
        if mode in ("decode", "dense", "packed"):
            capture_inputs.engram_token_windows.copy_(inputs.engram_token_windows)
        elif mode == "commit":
            capture_inputs.input_hiddens.copy_(inputs.input_hiddens)

    def outputs(result):
        fields = ("hidden_states", "mtp_target_hidden_states", "dspark_confidence_hidden_logits")
        return {
            key: value.clone()
            for key in fields
            if (value := getattr(result, key, None)) is not None and value.numel()
        }

    def pointers():
        return {
            "ids": capture_inputs.input_ids.data_ptr(),
            **{key: tensor.data_ptr() for key in ("engram_token_windows", "input_hiddens")
               if (tensor := getattr(capture_inputs, key, None)) is not None},
            **{f"table:{tag}": table.data_ptr() for tag, table in cache.tables.items()},
            **{f"pool:{owner}:{tag}": cache.pool(owner, tag).data_ptr()
               for owner, tag in coordinates},
        }

    def guarded_bytes():
        # Native row snapshots cover all entries in the reserved business
        # pages. Separately guard block zero and alignment outside entries.
        from rtp_llm.models_py.modules.dsv41.ppu.graph_cache import _WIDTH

        return {
            key: (
                cache.pool(*key)[0].view(torch.uint8).clone(),
                cache.pool(*key)[
                    torch.unique(slots // cache.entries[cache._table_tag(*key)]),
                    cache.entries[cache._table_tag(*key)] * _WIDTH[key[1]]:,
                ].view(torch.uint8).clone(),
            )
            for key, slots in coordinates.items()
        }

    def validate_step(step, initial, after, actual, independent, reference, mismatches,
                      before_accept, counters, addresses, reserved_unchanged):
        failures, comparisons = [], {}
        required = {"hidden_states"}
        if mode in ("dense", "packed"):
            required.add("mtp_target_hidden_states")
        if mode == "proposal" and model._dspark_confidence_enabled:
            required.add("dspark_confidence_hidden_logits")
        if not required <= actual.keys():
            failures.append("missing required hidden/aux fields")
        if actual.keys() != independent.keys() or actual.keys() != reference.keys():
            failures.append("output/aux fields changed")
        for key in actual.keys() & independent.keys() & reference.keys():
            tensor = actual[key]
            if any((other.shape, other.dtype) != (tensor.shape, tensor.dtype)
                   for other in (independent[key], reference[key])):
                failures.append(f"{key} shape/dtype changed")
                continue
            independent_check = _hidden_comparison(tensor, independent[key])
            graph_safe_check = _hidden_comparison(tensor, reference[key])
            comparisons[key] = {
                "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "sha256": hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
                "independent": independent_check, "graph_safe": graph_safe_check,
            }
            if (not independent_check["allclose"] or not graph_safe_check["exact"]
                or not independent_check["left_finite"] or not independent_check["right_finite"]):
                failures.append(f"{key} parity/finite")
        if any(count for arm in mismatches.values() for count in arm.values()):
            failures.append("native bytes differ from one of the references")
        if any(count for arm in before_accept.values() for count in arm.values()):
            failures.append("noncommitting arm modified native bytes before acceptance")
        if not reserved_unchanged or pointers() != addresses:
            failures.append("reserved bytes or captured addresses changed")
        changed_slots = {}
        for key, slots in coordinates.items():
            changed = (initial[key].view(torch.uint8) != after[key].view(torch.uint8)).any(-1)
            changed_ids = set(slots[changed].tolist())
            changed_slots[f"{key[0]}:{key[1]}"] = sorted(changed_ids)
            if permitted_writes is not None:
                allowed = permitted_writes[step][key]
                if not changed_ids <= allowed or (mode == "commit" and changed_ids != allowed):
                    failures.append(f"{key} changed rows differ from accepted physical footprint")
        if mode in ("dense", "packed") and not any(changed_slots.values()):
            failures.append("target acceptance did not change a native row")
        return {
            "outputs": comparisons, "native_byte_mismatches": mismatches,
            "native_changed_bytes": _state_mismatches(initial, after),
            "changed_slots": changed_slots, "native_before_accept_mismatches": before_accept,
            "fixed_addresses": pointers() == addresses, "reserved_unchanged": reserved_unchanged,
            "device_counters": counters, "failures": failures,
        }

    @torch.inference_mode()
    def execute():
        device = impl.device
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        graph = torch.cuda.CUDAGraph()
        reports = []
        failed_steps = []
        addresses = pointers()
        seed = guards = None
        try:
            with torch.cuda.stream(stream):
                seed = _snapshot(cache, coordinates)
                guards = guarded_bytes()
                update(capture_inputs)
                call(capture_inputs, impl)
                _restore(cache, coordinates, seed)
            stream.synchronize()
            dist.barrier()
            torch.cuda.synchronize(device)
            with torch.cuda.graph(graph, stream=stream):
                captured = call(capture_inputs, impl)
            with torch.cuda.stream(stream):
                for step, (inputs, accepted) in enumerate(steps):
                    initial = seed if initial_states is None else initial_states[step]
                    _restore(cache, coordinates, initial)
                    before_accept = {}
                    readonly_forward = mode in ("dense", "packed", "proposal")
                    independent = outputs(call(inputs))
                    if readonly_forward:
                        before_accept["eager"] = _state_mismatches(initial, _snapshot(cache, coordinates))
                    stage_accept(inputs, accepted, independent, captured=False)
                    eager_cache = _snapshot(cache, coordinates)
                    _restore(cache, coordinates, initial)
                    update(inputs)
                    if mode in ("dense", "packed"):
                        impl.arm_verify(inputs.request_id)
                    reference = outputs(call(capture_inputs, impl))
                    if readonly_forward:
                        before_accept["graph_safe"] = _state_mismatches(initial, _snapshot(cache, coordinates))
                    stage_accept(inputs, accepted, reference, captured=True)
                    reference_cache = _snapshot(cache, coordinates)
                    _restore(cache, coordinates, initial)
                    if mode in ("dense", "packed"):
                        impl.arm_verify(inputs.request_id)
                    if mode == "commit":
                        impl.writes.zero_()
                    else:
                        layers_before = impl.runtime.layer_calls.clone()
                    graph.replay()
                    actual = outputs(captured)
                    if readonly_forward:
                        before_accept["replay"] = _state_mismatches(initial, _snapshot(cache, coordinates))
                    stage_accept(inputs, accepted, actual, captured=True)
                    after = _snapshot(cache, coordinates)
                    mismatches = {
                        "graph_safe": _state_mismatches(reference_cache, after),
                        "eager": _state_mismatches(eager_cache, after),
                    }
                    current_guards = guarded_bytes()
                    unchanged = all(torch.equal(a, b) for key in guards
                                    for a, b in zip(guards[key], current_guards[key]))
                    if mode == "commit":
                        counters = {"writes": impl.writes.tolist()}
                        counter_ok = counters["writes"] == [impl.rows] * 3
                    else:
                        layer_calls = (impl.runtime.layer_calls - layers_before).tolist()
                        grouped = impl.runtime.grouped
                        counters = {"layer_calls": layer_calls}
                        counter_ok = layer_calls == [1] * len(model.v4.layers)
                        if grouped is not None:
                            received = grouped.received.clone()
                            dist.all_reduce(received)
                            counters["received_world_per_layer"] = received.sum(dim=1).tolist()
                            expected = impl.rows * model.shared_weight_views.geometry.n_activated_experts * dist.get_world_size()
                            counter_ok &= counters["received_world_per_layer"] == [expected] * len(model.v4.layers)
                    report = validate_step(
                        step, initial, after, actual, independent, reference, mismatches,
                        before_accept, counters, addresses, unchanged,
                    )
                    if not counter_ok:
                        report["failures"].append("device counters differ from replay geometry")
                    if diagnostics is not None:
                        path = diagnostics / f"rank{dist.get_rank()}-{mode}-step{step + 1}.json"
                        _write_direct_report(path, report)
                    reports.append(report)
                    failed = torch.tensor(int(bool(report["failures"])), device=device)
                    dist.all_reduce(failed)
                    failed_ranks = failed.item()
                    if failed_ranks:
                        failed_steps.append((step + 1, failed_ranks))
        finally:
            model.shared_runtime.abort_speculative_forward()
            torch.cuda.synchronize(device)
            graph.reset()
            if seed is not None:
                with torch.cuda.stream(stream):
                    _restore(cache, coordinates, seed)
                stream.synchronize()
        if failed_steps:
            raise DirectComparisonError(f"{mode} parity failed at (replay, ranks): {failed_steps}")
        return {
            "schema": "n9-direct-model-replay-v1", "mode": mode, "rank": dist.get_rank(),
            "captured": 1, "replayed": 2, "rows": impl.rows, "batch": impl.batch_size,
            "addresses": addresses,
            "steps": reports, "scope": "direct model capture using initialized native pools",
        }

    return execute()


DIRECT_MODES = ("proposal", "commit", "dense", "packed")
_PREFIXES = (129, 134)
_MAX_END = max(_PREFIXES) + 4


class DirectComparisonError(AssertionError):
    """A synchronized comparison failure after the completed replay."""


def _direct_case(mode):
    # Each geometry is (request order, query or commit rows). The first is
    # captured, the last two replace its live metadata and token contents.
    if mode == "proposal":
        return (((0,), (3,)), ((0,), (3,)), ((1,), (3,))), ((0,), (0,))
    if mode == "commit":
        return (((0, 1), (2, 2)), ((0, 1), (1, 3)), ((1, 0), (4, 0))), ((1, 3), (4, 0))
    if mode == "dense":
        return (((0,), (4,)), ((0,), (4,)), ((1,), (4,))), ((2,), (3,))
    if mode == "packed":
        return (((1, 0), (2, 2)), ((0, 1), (1, 3)), ((0, 1), (3, 1))), ((0, 2), (2, 1))
    raise ValueError(f"unsupported direct case {mode!r}")


def _write_direct_report(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("x") as writer:
        json.dump(value, writer, indent=2, allow_nan=False)
        writer.write("\n")
    temporary.replace(path)


class _FixturePages:
    """Two disjoint scratch requests in existing C++ native storage."""

    def __init__(self, model):
        import torch

        from rtp_llm.models_py.modules.dsv41.ppu.cache import CacheGeometry
        from rtp_llm.models_py.modules.dsv41.ppu.graph_cache import _FULL
        from rtp_llm.models_py.modules.dsv41.ppu.native_protocol import framework_pool_tag
        from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import PrefillPolicy

        native = model.kv_cache
        config = model.shared_weight_views.config
        self.device = model.shared_runtime.device
        self.geometry = CacheGeometry.from_config(config)
        self.tables = {}
        self.domains = {}
        policy = PrefillPolicy.from_config(config)
        for key in self.geometry.pool_keys():
            owner, tag = key
            name = framework_pool_tag(owner, tag, geometry=self.geometry, policy=policy)
            physical_span = native.get_seq_size_per_block(name)
            kernel_span = native.get_kernel_seq_size_per_block(name)
            if physical_span < 1 or kernel_span < 1 or physical_span % kernel_span:
                raise ValueError(f"invalid real physical/kernel page geometry for {name}")
            columns = math.ceil(_MAX_END / physical_span)
            factor = physical_span // kernel_span
            physical = torch.arange(1, 2 * columns + 1, dtype=torch.int32).reshape(2, columns)
            kernel = (physical[..., None] * factor + torch.arange(factor)).flatten(1).int()
            self.tables[name] = physical, kernel
            blocks, span = (kernel, kernel_span) if tag in _FULL else (physical, physical_span)
            pool = native.get_layer_cache(owner, name).kv_cache_base
            entries = native.get_entries_per_block(name)
            if entries <= 0 or int(blocks.max()) >= pool.shape[0] or pool.device != self.device:
                raise ValueError(f"real native pool {key} lacks reserved fixture page capacity")
            self.domains[key] = SimpleNamespace(
                pool=pool, blocks=blocks.tolist(), entries=entries, span=span,
                ratio=self.geometry.ratio(owner) if tag in _FULL else 1,
            )

    def inputs(self, requests, prefixes, token_rows, *, rank, prefill=False, verify=False,
               packed=False, windows=None, features=None):
        import torch

        from rtp_llm.models_py.modules.dsv41.ppu.tools.driver_shared_graph_dp8 import _inputs

        lengths = tuple(map(len, token_rows))
        host_lengths = torch.tensor(lengths, dtype=torch.int32)
        host_prefixes = torch.tensor(prefixes, dtype=torch.int32)
        cu = torch.cat((torch.zeros(1, dtype=torch.int32), host_lengths.cumsum(0, dtype=torch.int32)))
        lengths_device, prefixes_device = host_lengths.to(self.device), host_prefixes.to(self.device)
        cu_device = cu.to(self.device)
        ids = torch.tensor([token for row in token_rows for token in row],
                           dtype=torch.int32, device=self.device)
        positions = torch.tensor([
            pos for base, length in zip(prefixes, lengths) for pos in range(base, base + length)
        ], dtype=torch.int32, device=self.device)
        by_tag = {}
        for tag, (physical, kernel) in self.tables.items():
            physical, kernel = physical[list(requests)], kernel[list(requests)]
            by_tag[tag] = SimpleNamespace(
                is_prefill=prefill or verify, is_target_verify=verify,
                is_packed_target_verify=packed, is_cuda_graph=False, context_parallel_info=None,
                input_lengths=host_lengths, prefix_lengths=host_prefixes,
                input_lengths_device=lengths_device, prefix_lengths_device=prefixes_device,
                sequence_lengths=torch.empty(0, dtype=torch.int32),
                cu_seqlens=cu, cu_seqlens_device=cu_device, combo_position_ids=positions,
                kv_cache_block_id=physical, kv_cache_kernel_block_id=kernel,
                kv_cache_block_id_device=physical.to(self.device),
                kv_cache_kernel_block_id_device=kernel.to(self.device),
                cache_store_inputs=None, cache_store_writer=None,
            )
        inputs = _inputs(ids, by_tag, windows)
        inputs.request_id = torch.tensor([100 + rank * 2 + request for request in requests])
        inputs.input_hiddens = features
        inputs.combo_position_ids = positions
        return inputs

    def coordinates(self):
        import torch

        return {
            key: torch.tensor([
                block * domain.entries + entry
                for row in domain.blocks for block in row for entry in range(domain.entries)
            ], dtype=torch.int64, device=self.device)
            for key, domain in self.domains.items()
        }

    def writable(self, requests, accepted, *, prefixes=None):
        if prefixes is None:
            prefixes = tuple(_PREFIXES[request] for request in requests)
        allowed = {}
        for key, domain in self.domains.items():
            slots = set()
            for request, start, count in zip(requests, prefixes, accepted):
                end = start + count
                if key[1] in ("csa_state", "swa_kv"):
                    positions = range(max(start, end - domain.entries), end)
                else:
                    ratio = domain.ratio
                    positions = range(start // ratio * ratio, end // ratio * ratio, ratio)
                for pos in positions:
                    block = domain.blocks[request][pos // domain.span]
                    offset = (pos % domain.span) // domain.ratio if key[1] not in (
                        "swa_kv", "csa_state"
                    ) else pos % domain.entries
                    slots.add(block * domain.entries + offset)
            allowed[key] = slots
        return allowed

    def raw_snapshot(self):
        """Include reserved block zero and all physical padding."""
        import torch

        snapshot = []
        for domain in self.domains.values():
            blocks = torch.tensor([0, *domain.blocks[0], *domain.blocks[1]], device=self.device)
            snapshot.append((domain.pool, blocks, domain.pool.index_select(0, blocks)))
        return snapshot


def run_direct_device_fixture(target, draft, modes, output):
    """Run after both production model/KV bindings and before readiness."""
    import hashlib
    import os

    import torch
    import torch.distributed as dist

    from rtp_llm.model_loader.model_weight_info import ModelWeights
    from rtp_llm.models_py.modules.dsv41.ppu.engram import build_token_windows
    from rtp_llm.models_py.modules.dsv41.ppu.shared_model import DeepSeekV41SharedPPUModel
    from rtp_llm.models_py.modules.dsv41.transformer import V4Transformer
    from rtp_llm.utils.model_weight import W

    rank = dist.get_rank()
    models = target, draft
    if (not isinstance(target, DeepSeekV41SharedPPUModel) or dist.get_world_size() != 8
        or any(not isinstance(model.weight, ModelWeights) or type(model.kv_cache).__name__ != "KVCache"
               or not model._is_decode_role or not model._is_speculative
               or model._max_generate_batch_size < 2
               or (model._v4_args.dp_size, model._v4_args.ep_size, model._v4_args.tp_size) != (8, 8, 1)
               for model in models)):
        raise ValueError("fixture requires the two real cache-backed DP8/EP8 DSpark models")
    if draft.shared_mode.gamma != 3 or draft.shared_mode.query_width != 3 or not draft._dspark_confidence_enabled:
        raise ValueError("fixture requires gamma3 with sampled anchor and real confidence weights")
    if type(target.v4).forward_decode is not V4Transformer.forward_decode or any(
        (model._shared_ops.moe_engine.backend, model._shared_ops.moe_engine.precision) != ("grouped", "w4a8")
        for model in models
    ):
        raise ValueError("fixture requires inherited full target Decode and real grouped W4A8")
    aliases = {
        key: draft.weight.global_weights[key] is target.weight.global_weights[key]
        for key in (W.embedding, W.lm_head)
    }
    if not all(aliases.values()):
        raise ValueError("draft vocab is not aliased through the framework weight owner")

    @torch.inference_mode()
    def execute():
        target_pages, draft_pages = map(_FixturePages, models)
        if len(target_pages.domains) != 51 or len(draft_pages.domains) != 3:
            raise ValueError("unexpected native full40/draft3 domain counts")
        target_storage, draft_storage = [
            {domain.pool.untyped_storage().data_ptr() for domain in pages.domains.values()}
            for pages in (target_pages, draft_pages)
        ]
        # Target and draft may occupy distinct views in one allocator arena.
        for left in target_pages.domains.values():
            for right in draft_pages.domains.values():
                a, b = left.pool, right.pool
                if a.data_ptr() < b.data_ptr() + b.nbytes and b.data_ptr() < a.data_ptr() + a.nbytes:
                    raise ValueError("target and draft native domains overlap")
        g = target.shared_weight_views.geometry
        width = draft._dspark_aux_feature_dim
        depth = g.engram_max_ngram_size - 1
        if not target.v4.capture_aux_hidden_layer_ids or g.vocab_size <= 4096 or depth < 1:
            raise ValueError("fixture requires the checkpoint target aux and Engram geometry")
        histories = tuple(tuple(257 + rank * 17 + req * 257 + 5 * j for j in range(length))
                          for req, length in enumerate(_PREFIXES))

        def tokens(request, count, salt):
            return tuple(2048 + rank * 17 + 113 * request + salt * 43 + j for j in range(count))

        def target_inputs(requests, rows, *, seed=False, packed=False, previous_history=None):
            lengths = tuple(map(len, rows))
            prior = tuple(histories[request] for request in requests) if previous_history is None else previous_history
            prefixes = (0,) * len(rows) if seed else tuple(map(len, prior))
            previous = [[-1] * depth if seed else list(history[-depth:][::-1]) for history in prior]
            ids = torch.tensor([token for row in rows for token in row])
            cu = torch.cat((torch.zeros(1, dtype=torch.long), torch.tensor(lengths).cumsum(0)))
            windows = build_token_windows(ids, cu, torch.tensor(previous)).to(target_pages.device)
            return target_pages.inputs(
                requests, prefixes, rows, rank=rank, prefill=seed, verify=not seed,
                packed=packed, windows=windows,
            )

        def draft_inputs(requests, rows, *, features=None, seed=False):
            prefixes = (0,) * len(rows) if seed else tuple(_PREFIXES[request] for request in requests)
            return draft_pages.inputs(
                requests, prefixes, rows, rank=rank, prefill=seed,
                verify=not seed, packed=features is not None and not seed, features=features,
            )

        def actual_aux(result, rows):
            aux = result.mtp_target_hidden_states
            if aux is None or aux.shape != (rows, width) or not bool(torch.isfinite(aux).all()):
                raise ValueError("target did not produce its configured finite aux features")
            return aux.clone()

        def digest(tensor):
            return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

        def audit_commit_shape():
            from rtp_llm.models_py.modules.dsv41.ppu._quant import dequantize_swa_kv_rows
            from rtp_llm.models_py.modules.dsv41.ppu.shared_dspark_attention import project_commit

            prefix = _PREFIXES[0]
            all_ids = tokens(0, 4, 19)
            verify = target_inputs((0,), (all_ids,))
            try:
                features = actual_aux(target.forward(verify), len(all_ids))
            finally:
                target.abort_speculative_forward()
            saved = draft_pages.raw_snapshot()
            domains = list(draft_pages.domains.items())
            assert [key for key, _ in domains] == [(layer, "swa_kv") for layer in range(3)]

            def read_positions(domain, positions):
                blocks = torch.tensor(domain.blocks[0], device=positions.device)
                block_ids = blocks[positions // domain.span]
                offsets = (positions % domain.entries)[:, None] * 528
                columns = offsets + torch.arange(528, device=positions.device)
                return domain.pool.view(domain.pool.shape[0], -1)[block_ids[:, None], columns].clone()

            history = torch.arange(prefix - g.window, prefix, device=draft_pages.device)
            original = [read_positions(domain, history) for _, domain in domains]
            runs = {}
            try:
                for size in (4, 1, 2):
                    for pool, blocks, values in saved:
                        pool.index_copy_(0, blocks, values)
                    inputs = draft_inputs((0,), (all_ids[:size],), features=features[:size])
                    positions = inputs.combo_position_ids.long()
                    main_proj = draft.main_proj(inputs.input_hiddens)
                    main = draft.main_norm(main_proj)
                    wkv = [layer.attn.wkv(main) for layer in draft.v4.layers]
                    projections = [
                        project_commit(layer.attn, main, positions, draft.shared_runtime.freqs_for(index))
                        for index, layer in enumerate(draft.v4.layers)
                    ]
                    packed = [draft._shared_ops.encode_kv(value, pool_tag="swa_kv") for value in projections]
                    actual = draft.forward_commit(inputs).hidden_states
                    assert torch.equal(actual, main)
                    for i, (_, domain) in enumerate(domains):
                        assert torch.equal(read_positions(domain, positions), packed[i])
                        assert torch.equal(read_positions(domain, history), original[i])
                    runs[size] = (main_proj, main, wkv, projections, packed)
            finally:
                for pool, blocks, values in saved:
                    pool.index_copy_(0, blocks, values)

            def compare(full, compact):
                assert full.shape == compact.shape and torch.isfinite(full).all() and torch.isfinite(compact).all()
                return {
                    "different_elements": int(torch.count_nonzero(full != compact)),
                    "max_abs": float((full.float() - compact.float()).abs().max()),
                    "allclose_005": bool(torch.allclose(full, compact, atol=0.005, rtol=0.005)),
                }

            full_proj, full_main, full_wkv, full_kv, full_packed = runs[4]
            comparisons = []
            for size in (1, 2):
                compact_proj, compact_main, compact_wkv, compact_kv, compact_packed = runs[size]
                comparisons.append({
                    "compact_rows": size,
                    "shared_aux_sha256": digest(features[:size]),
                    "main_proj": compare(full_proj[:size], compact_proj),
                    "main_norm": compare(full_main[:size], compact_main),
                    "wkv_linear": [compare(a[:size], b) for a, b in zip(full_wkv, compact_wkv)],
                    "wkv_norm_rope": [compare(a[:size], b) for a, b in zip(full_kv, compact_kv)],
                    "packed_mismatch_bytes": [int(torch.count_nonzero(a[:size] != b))
                                              for a, b in zip(full_packed, compact_packed)],
                    "decoded_kv": [
                        compare(dequantize_swa_kv_rows(a[:size]), dequantize_swa_kv_rows(b))
                        for a, b in zip(full_packed, compact_packed)
                    ],
                })
            return {
                "full_rows": 4,
                "rank": rank,
                "prefix": prefix,
                "entries": [domain.entries for _, domain in domains],
                "spans": [domain.span for _, domain in domains],
                "native_write_matches_own_projection": True,
                "surviving_history_exact": True,
                "comparisons": comparisons,
            }

        def audit_rejected_tail():
            from rtp_llm.models_py.modules.dsv41.ppu.shared_dspark_attention import project_commit
            from rtp_llm.models_py.modules.dsv41.ppu.tools.driver_shared_graph_dp8 import (
                _restore, _snapshot, _state_mismatches,
            )

            request = (0,)
            first_ids, next_ids = tokens(0, 4, 20), tokens(0, 4, 21)
            prefix = len(histories[0])
            next_prefix = prefix + 1
            verify = target_inputs(request, (first_ids,))
            next_verify = target_inputs(
                request, (next_ids,), previous_history=(histories[0] + first_ids[:1],),
            )
            target_cache = target.prepare_fmha_impl(verify, is_cuda_graph=True).native_cache
            target_coords, draft_coords = target_pages.coordinates(), draft_pages.coordinates()
            saved = target_pages.raw_snapshot() + draft_pages.raw_snapshot()
            target_reports, failures = [], []

            def accept_target(inputs, start, count):
                before = _snapshot(target_cache, target_coords)
                features = actual_aux(target.forward(inputs), 4)
                staged = _state_mismatches(before, _snapshot(target_cache, target_coords))
                target.finish_speculative_forward(
                    inputs.request_id, torch.tensor([start], dtype=torch.int32),
                    torch.tensor([count], dtype=torch.int32),
                )
                after = _snapshot(target_cache, target_coords)
                allowed = target_pages.writable(request, (count,), prefixes=(start,))
                footprints = {}
                for key, slots in target_coords.items():
                    changed = (before[key].view(torch.uint8) != after[key].view(torch.uint8)).any(-1)
                    actual = set(slots[changed].tolist())
                    if actual - allowed[key] or (key[1] == "swa_kv" and actual != allowed[key]):
                        failures.append(f"target {start} {key} publication footprint")
                    footprints[f"{key[0]}:{key[1]}"] = sorted(actual)
                if any(staged.values()):
                    failures.append(f"target {start} verify wrote before acceptance")
                target_reports.append({
                    "input": describe(inputs, torch.tensor([count])), "staged_mismatch_bytes": staged,
                    "published_slots": footprints, "aux_sha256": digest(features),
                })
                return features

            def draft_commit(inputs, start, before, cache):
                main = draft.main_norm(draft.main_proj(inputs.input_hiddens))
                result = draft.forward_commit(inputs)
                if not torch.equal(result.hidden_states, main):
                    failures.append(f"draft {start} main projection")
                actual = _snapshot(cache, draft_coords)
                for layer, key in enumerate(draft_coords):
                    domain = draft_pages.domains[key]
                    positions = range(start, start + 4)
                    slots = [domain.blocks[0][p // domain.span] * domain.entries + p % domain.entries
                             for p in positions]
                    selected = torch.searchsorted(
                        draft_coords[key], torch.tensor(slots, device=draft_pages.device),
                    )
                    kv = project_commit(
                        draft.v4.layers[layer].attn, main, inputs.combo_position_ids.long(),
                        draft.shared_runtime.freqs_for(layer),
                    )
                    rows = draft._shared_ops.encode_kv(kv, pool_tag="swa_kv")
                    expected = before[key].clone()
                    expected[selected] = rows
                    if not torch.equal(actual[key], expected):
                        failures.append(f"draft {start} {key} full write/untouched footprint")
                return actual

            def commit_inputs(ids, start, features):
                return draft_pages.inputs(request, (start,), (ids,), rank=rank,
                                          verify=True, features=features)

            try:
                first = commit_inputs(first_ids, prefix, accept_target(verify, prefix, 1))
                cache = draft.prepare_dspark_graph(first, "commit").native_cache
                uncommitted = _snapshot(cache, draft_coords)
                baseline = draft_commit(first, prefix, uncommitted, cache)
                poisoned = {key: rows.clone() for key, rows in baseline.items()}
                slots_report = {}
                for key, slots in draft_coords.items():
                    domain = draft_pages.domains[key]
                    start = next_prefix - g.window
                    live = draft_pages.writable(request, (g.window,), prefixes=(start,))[key]
                    rejected = draft_pages.writable(request, (3,), prefixes=(next_prefix,))[key]
                    written = draft_pages.writable(request, (4,), prefixes=(prefix,))[key]
                    if live & rejected or len(live) != g.window or len(rejected) != 3:
                        failures.append(f"draft {key} overlapping control slots")
                    selected = torch.isin(slots, torch.tensor(sorted(rejected), device=slots.device))
                    visible = torch.isin(slots, torch.tensor(sorted(live), device=slots.device))
                    # Replace the rejected rows with valid encoded KV from the visible window.
                    poisoned[key][selected] = baseline[key][visible][:3]
                    changed = (poisoned[key] != baseline[key]).any(-1)
                    initially_written = (baseline[key] != uncommitted[key]).any(-1)
                    if set(slots[changed].tolist()) != rejected or set(slots[initially_written].tolist()) != written:
                        failures.append(f"draft {key} control or first commit did not change every expected slot")
                    slots_report[str(key[0])] = {
                        "ring_entries": domain.entries, "physical_span": domain.span,
                        "rejected_slots": sorted(rejected), "visible_count": int(visible.sum()),
                        "visible_sha256": digest(baseline[key][visible]),
                        "poison_visible_sha256": digest(poisoned[key][visible]),
                        "baseline_tail_sha256": digest(baseline[key][selected]),
                        "poison_tail_sha256": digest(poisoned[key][selected]),
                    }
                query_ids = (next_ids[0], draft._dspark_noise_token_id, draft._dspark_noise_token_id)
                query = draft_pages.inputs(request, (next_prefix,), (query_ids,), rank=rank, verify=True)
                capture = draft_pages.inputs(request, (next_prefix,), (query_ids,), rank=rank, verify=True)
                diagnostics = output / "tail-control"
                diagnostics.mkdir(exist_ok=True)
                replay = run_model_replays(
                    draft, "proposal", capture, ((query, None), (query, None)), draft_coords,
                    initial_states=(baseline, poisoned), diagnostics=diagnostics,
                )
                outputs = [{key: value["sha256"] for key, value in step["outputs"].items()}
                           for step in replay["steps"]]
                if outputs[0] != outputs[1]:
                    failures.append("proposal hidden or confidence output depends on rejected tail")

                later = commit_inputs(next_ids, next_prefix, accept_target(next_verify, next_prefix, 4))
                results = []
                for state in (baseline, poisoned):
                    _restore(cache, draft_coords, state)
                    results.append(draft_commit(later, next_prefix, state, cache))
                if any(_state_mismatches(*results).values()):
                    failures.append("later full commit did not converge after replacing rejected tail")
                for key, slots in draft_coords.items():
                    rejected = set(slots_report[str(key[0])]["rejected_slots"])
                    selected = torch.isin(slots, torch.tensor(sorted(rejected), device=slots.device))
                    for state, final in zip((baseline, poisoned), results):
                        if not bool((state[key][selected] != final[key][selected]).any(-1).all()):
                            failures.append(f"draft {key} later commit did not overwrite every rejected slot")
                    slots_report[str(key[0])]["overwritten_tail_sha256"] = digest(results[0][key][selected])
                return {
                    "physical_verify_width": 4, "accepted_rows": [1, 4], "proposal_width": 3,
                    "rejected_positions": list(range(next_prefix, prefix + 4)),
                    "overwritten_positions": list(range(next_prefix, prefix + 4)),
                    "proposal_input": describe(query, None), "proposal": replay, "target": target_reports,
                    "layers": slots_report, "failures": failures,
                }
            finally:
                target.abort_speculative_forward()
                for pool, blocks, values in saved:
                    pool.index_copy_(0, blocks, values)

        def describe(inputs, accepted):
            attn = next(iter(inputs.attention_inputs.values()))
            return {
                "request_ids": inputs.request_id.tolist(),
                "prefixes": attn.prefix_lengths.tolist(), "lengths": attn.input_lengths.tolist(),
                "tokens": inputs.input_ids.tolist(), "positions": inputs.combo_position_ids.tolist(),
                "physical_tables": {tag: meta.kv_cache_block_id.tolist()
                                    for tag, meta in inputs.attention_inputs.items()},
                "accepted_rows": accepted.tolist() if accepted is not None else None,
                "input_feature_sha256": digest(inputs.input_hiddens) if inputs.input_hiddens is not None else None,
            }

        original_pages = (target_pages.raw_snapshot(), draft_pages.raw_snapshot())
        report = {
            "schema": "n9-direct-dspark-rank-v1", "rank": rank, "ok": False, "SMOKE": "FAIL",
            "models": [type(model).__name__ for model in models],
            "model_weights": [type(model.weight).__name__ for model in models],
            "kv_cache": [type(model.kv_cache).__name__ for model in models],
            "vocab_aliases": aliases, "distinct_pool_views": True,
            "routed_precision": "grouped W4A8",
            "layer_counts": [len(model.v4.layers) for model in models],
            "native_domains": [len(target_pages.domains), len(draft_pages.domains)],
            "shares_backing_arena": bool(target_storage & draft_storage),
            "target_aux_layers": list(target.v4.capture_aux_hidden_layer_ids),
            "profile": draft.shared_mode.protocol(), "seeds": [], "modes": {},
        }
        try:
            report["phase"] = {"stage": "target_aux_capacity"}
            aux = target.get_mtp_target_hidden_states(sys.maxsize)
            shape = list(aux.shape) if aux is not None else []
            required_rows = max(_PREFIXES)
            report["target_aux_capacity"] = {
                "required_rows": required_rows, "required_width": width,
                "observed_shape": shape,
                "max_generate_batch_size": target._max_generate_batch_size,
            }
            if len(shape) != 2 or shape[0] < required_rows or shape[1] != width:
                raise ValueError(
                    f"target aux buffer shape {shape} must cover at least "
                    f"{required_rows} seed rows of width {width}"
                )
            for request, history in enumerate(histories):
                report["phase"] = {"stage": "target_prefill_seed", "request": request, "rows": len(history)}
                seed_input = target_inputs((request,), (history,), seed=True)
                # Same pre-admission seed as the N4 fixture, on the initialized
                # D target. Only this Python call selects the Prefill arm.
                role = target._is_decode_role
                try:
                    target._is_decode_role = False
                    result = target.forward(seed_input)
                finally:
                    target._is_decode_role = role
                if not bool(torch.isfinite(result.hidden_states).all()):
                    raise ValueError("real target Prefill returned non-finite hidden states")
                aux = actual_aux(result, len(history))
                report["phase"] = {"stage": "draft_commit_seed", "request": request, "rows": len(history)}
                seed_commit = draft_inputs((request,), (history,), features=aux, seed=True)
                committed = draft.forward_commit(seed_commit)
                if not bool(torch.isfinite(committed.hidden_states).all()):
                    raise ValueError("real draft feature commit returned non-finite hidden states")
                report["seeds"].append({
                    "request": request, "tokens": len(history), "target_aux_sha256": digest(aux),
                    "draft_commit_feature_sha256": digest(seed_commit.input_hiddens),
                })
            report["phase"] = {"stage": "check_seed_bytes"}
            for pages, original in zip((target_pages, draft_pages), original_pages):
                seed_pages = pages.raw_snapshot()
                if not all(not torch.equal(new[2][1:].view(torch.uint8), old[2][1:].view(torch.uint8))
                           for new, old in zip(seed_pages, original)):
                    raise ValueError("actual seed did not populate each native domain")
            if os.environ.get("RTP_V41_N9_COMMIT_SHAPE_AUDIT") == "1":
                report["phase"] = {"stage": "audit_commit_shape"}
                report["commit_shape_audit"] = audit_commit_shape()
            if os.environ.get("RTP_V41_N10_TAIL_CONTROL") == "1":
                report["phase"] = {"stage": "audit_rejected_tail"}
                control = report["rejected_tail_control"] = audit_rejected_tail()
                failed = torch.tensor(int(bool(control["failures"])), device=draft_pages.device)
                dist.all_reduce(failed)
                if failed.item():
                    raise DirectComparisonError(f"rejected tail control failed on {failed.item()} ranks")
            comparison_errors = {}
            for mode in modes:
                report["phase"] = {"stage": "prepare_mode_inputs", "mode": mode}
                shapes, accepts = _direct_case(mode)
                inputs = []
                is_draft = mode in ("proposal", "commit")
                pages = draft_pages if is_draft else target_pages
                for arm, (requests, lengths) in enumerate(shapes):
                    rows = tuple(tokens(request, count, arm + 4 * DIRECT_MODES.index(mode))
                                 for request, count in zip(requests, lengths))
                    if mode == "commit":
                        # Produce features by the real target verify path for
                        # exactly the tokens that will be committed. A zero
                        # draft row still has a one-row target anchor, which
                        # is discarded when gathering accepted features.
                        target_rows = tuple(row or tokens(req, 1, arm + 17) for req, row in zip(requests, rows))
                        verify = target_inputs(requests, target_rows, packed=True)
                        try:
                            features = actual_aux(target.forward(verify), sum(map(len, target_rows)))
                        finally:
                            target.abort_speculative_forward()
                        slices, offset = [], 0
                        for accepted, verified in zip(lengths, target_rows):
                            slices.append(features[offset:offset + accepted])
                            offset += len(verified)
                        inputs.append(draft_inputs(requests, rows, features=torch.cat(slices)))
                    elif mode == "proposal":
                        # Only the anchor is supplied as a real query token.
                        # Public proposal logic fills the remaining noise IDs.
                        rows = tuple((row[0], draft._dspark_noise_token_id, draft._dspark_noise_token_id)
                                     for row in rows)
                        inputs.append(draft_inputs(requests, rows))
                    else:
                        inputs.append(target_inputs(requests, rows, packed=mode == "packed"))
                acceptance = tuple(torch.tensor(counts, dtype=torch.int32) if not is_draft else None
                                   for counts in accepts)
                footprints = [pages.writable(shape[0], count) for shape, count in zip(shapes[1:], accepts)]
                # Save capture values before run_model_replays copies into
                # the capture tensors. Live observations come from replay.
                metadata = [describe(inputs[0], None),
                            *[describe(x, n) for x, n in zip(inputs[1:], acceptance)]]
                _write_direct_report(output / f"rank{rank}-{mode}-inputs.json", metadata)
                report["phase"] = {"stage": "capture_and_replay", "mode": mode}
                try:
                    report["modes"][mode] = run_model_replays(
                        draft if is_draft else target, mode, inputs[0],
                        tuple(zip(inputs[1:], acceptance)), pages.coordinates(),
                        permitted_writes=footprints, diagnostics=output,
                    )
                except DirectComparisonError as error:
                    comparison_errors[mode] = str(error)
                    continue
                report["modes"][mode]["inputs"] = metadata
            if comparison_errors:
                report["phase"] = {"stage": "check_mode_comparisons"}
                report["comparison_errors"] = comparison_errors
                raise DirectComparisonError(f"direct parity failed for {tuple(comparison_errors)}")
            report.update(ok=True, SMOKE="PASS_DIRECT_MODEL_GRAPH_FIDELITY", phase={"stage": "complete"})
        except Exception as error:
            report["error"] = repr(error)
            raise
        finally:
            raw_original = [item for original in original_pages for item in original]
            for pool, blocks, before in raw_original:
                pool.index_copy_(0, blocks, before)
            torch.cuda.synchronize(target_pages.device)
            report["fixture_pages_restored"] = all(
                torch.equal(pool.index_select(0, blocks).view(torch.uint8), before.view(torch.uint8))
                for pool, blocks, before in raw_original
            )
            if not report["fixture_pages_restored"]:
                report.update(ok=False, SMOKE="FAIL")
            _write_direct_report(output / f"rank{rank}.json", report)
        if not report["fixture_pages_restored"]:
            raise RuntimeError("failed to restore reserved engine pages before readiness")

    execute()

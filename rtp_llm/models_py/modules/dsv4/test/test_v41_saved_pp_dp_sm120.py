"""Sequential native CEP4PP2 export and cold eager DP8EP8 continuation."""

import copy
import gc
import os
import subprocess
import time
import traceback
import unittest
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from rtp_llm.models.dsv41_kv_cache import dsv41_cache_bindings, dsv41_page_layouts
from rtp_llm.models_py.model_desc.deepseek_v4_model import Dsv4SharedRuntimeBufferStore
from rtp_llm.models_py.model_desc.deepseek_v41_model import DeepSeekV41Model
from rtp_llm.models_py.model_desc.v41_errors import V41InputRejected
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_loader_worker_sm120 import (
    load_configuration,
    make_loader,
    save_checkpoint,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_worker_sm120 import (
    decode_inputs,
    resources,
    tagged_cache,
)
from rtp_llm.models_py.modules.dsv4.test import (
    test_v41_pp_checkpoint_engine_sm120 as producer,
)
from rtp_llm.models_py.modules.dsv4.test import test_v41_pp_loader_sm120 as pp_loader
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_preflight_sm120 import (
    clean_worker,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_stage_slices_sm120 import wide_fixture
from rtp_llm.models_py.modules.dsv4.test.v41_checkpoint_test_support import (
    byte_hash,
    digest,
    emit,
    loader_oracles,
    save_json,
)
from rtp_llm.models_py.modules.dsv4.v41_cache import V41PagedCache
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs
from rtp_llm.tools.dsv41_pp_smoke import checkpoint_identities
from rtp_llm.utils.model_weight import W

SENTINEL = 211
ROUNDS = 8
STATE_TAGS = {"swa_kv", "csa_state"}
REJECTIONS = (
    "generation",
    "policy",
    "missing_pool",
    "payload",
    "scale",
    "table",
    "replica",
    "checkpoint",
    "history",
    "sample",
    "shape",
    "dtype",
    "binding",
)


def pool_name(binding):
    return f"{binding.source_layer_id}:{binding.tag}"


def cpu_page_rows(raw, page, layout):
    raw = raw.reshape(-1, layout.stride_bytes)
    payload = raw[page, : layout.scale_plane_offset].reshape(
        layout.entries, layout.payload_bytes_per_entry
    )
    scales = raw[
        page,
        layout.scale_plane_offset : layout.scale_plane_offset
        + layout.entries * layout.scale_bytes_per_entry,
    ].reshape(layout.entries, layout.scale_bytes_per_entry)
    return torch.cat((payload, scales), 1).contiguous()


def merge_producer(root, choices, prompts, oracles):
    directory = root / "producer"
    checkpoint = root / "checkpoint"
    config, _ = load_configuration(checkpoint, max_seq_len=512)
    config.deepseek_v41_config.update(choices)
    text = config.deepseek_v41_config
    identities = checkpoint_identities(checkpoint)
    layouts = dsv41_page_layouts(0)
    snapshots = [
        torch.load(
            directory / f"rank{rank}-phase0.pt", map_location="cpu", weights_only=True
        )
        for rank in range(8)
    ]
    outputs = torch.load(
        directory / "outputs-phase0.pt", map_location="cpu", weights_only=True
    )
    assert len(outputs) == len(prompts) == 2
    canonical = [dict() for _ in prompts]
    replicas = [dict() for _ in prompts]
    source = []
    cp_checks = replica_checks = 0
    for rank, snapshot in enumerate(snapshots):
        owned = snapshot["ownership"]
        bindings = dsv41_cache_bindings(text, 2, rank // 4)
        assert owned["bindings"] == [asdict(binding) for binding in bindings]
        assert owned["layouts"] == {
            tag: asdict(layout) for tag, layout in layouts.items()
        }
        assert owned["identities"] == identities and owned["choices"] == choices
        assert (
            owned["config"] == text
            and owned["loader"] == oracles["source_loader"][str(rank)]
        )
        assert owned["live_native_leases"] and owned["worker_owner_exact"]
        assert set(snapshot["cache"]) == {
            (binding.storage_layer_id, binding.tag) for binding in bindings
        }
        assert len(owned["leases"]) == 2 and set(owned["tables"]) == {
            binding.tag for binding in bindings
        }
        assert snapshot["spans"] == snapshots[0]["spans"]
        assert [span[5] for span in snapshot["spans"]] == [
            len(prompt) for prompt in prompts
        ]
        assert all(span[1] > 0 and span[4] == 0 for span in snapshot["spans"])
        for tag, table in owned["tables"].items():
            assert (
                table.dtype == torch.int32
                and table.shape == (2, 1)
                and table.is_contiguous()
            )
            assert table.tolist() == [lease[tag] for lease in owned["leases"]]
            assert len(set(table.flatten().tolist())) == 2 and bool(
                ((table > 0) & (table < 8)).all()
            )
        base = snapshots[rank // 4 * 4]
        if rank % 4:
            producer.require_exact_tree(base["cache"], snapshot["cache"])
            producer.require_exact_tree(base["ownership"]["tables"], owned["tables"])
            assert base["ownership"]["leases"] == owned["leases"]
            cp_checks += len(bindings)
        else:
            for binding in bindings:
                raw = snapshot["cache"][binding.storage_layer_id, binding.tag]
                layout = layouts[binding.tag]
                assert (
                    raw.dtype == torch.uint8 and raw.numel() == 8 * layout.stride_bytes
                )
                assert bool((raw.reshape(8, -1)[0] == 37).all())
                for request in range(2):
                    page = owned["tables"][binding.tag][request, 0].item()
                    rows = cpu_page_rows(raw, page, layout)
                    name = pool_name(binding)
                    (replicas if binding.is_read_replica else canonical)[request][
                        name
                    ] = rows
        source.append(
            {
                "rank": rank,
                "stage": rank // 4,
                "snapshot_sha256": digest(directory / f"rank{rank}-phase0.pt"),
                "bindings": owned["bindings"],
                "tables": {
                    tag: table.tolist() for tag, table in owned["tables"].items()
                },
                "leases": owned["leases"],
                "live_native_leases": True,
                "worker_owner_exact": True,
            }
        )
    bindings = dsv41_cache_bindings(text)
    keys = {pool_name(binding) for binding in bindings}
    requests = []
    for request, prompt in enumerate(prompts):
        output = outputs[request]
        assert output["request_id"] == snapshots[0]["spans"][request][0]
        assert output["tokens"][:-1] == prompt and output["outstanding"] == 0
        assert output["tokens"][-1] == int(output["logits"].argmax().item()) < 127
        assert set(canonical[request]) == keys and len(keys) == 51
        assert set(replicas[request]) == {"20:hca_kv", "20:indexer_kv"}
        for name, rows in replicas[request].items():
            assert torch.equal(rows, canonical[request][name])
            replica_checks += 1
        requests.append(
            {
                "span": list(snapshots[0]["spans"][request]),
                "prompt": prompt,
                "start": len(prompt),
                "history": prompt[-3:][::-1],
                "next_token": output["tokens"][-1],
                "packed_sha256": {
                    name: byte_hash(value) for name, value in canonical[request].items()
                },
            }
        )
    admitted = {
        "format": "dsv41_native_pp_saved_dp_v1",
        "gen_num_per_cycle": 0,
        "label": root.name,
        "config": text,
        "choices": choices,
        "identities": identities,
        "checkpoint_sha256": {
            path.name: digest(path)
            for path in sorted(checkpoint.iterdir())
            if path.is_file()
        },
        "bindings": [asdict(binding) for binding in bindings],
        "layouts": {tag: asdict(layout) for tag, layout in layouts.items()},
        "sources": source,
        "requests": requests,
    }
    state = {"metadata": admitted, "packed": canonical, "replicas": replicas}
    producer.save_tensors(root / "state.pt", state)
    manifest = {"metadata": admitted, "state_sha256": digest(root / "state.pt")}
    save_json(root / "state.json", manifest)
    emit(
        "DSV41_HANDOFF_EXPORT",
        label=root.name,
        physical_pools=53,
        canonical_pools=51,
        requests=2,
        cp_pools_exact=cp_checks,
        replica_pages_exact=replica_checks,
        live_lease_tables=True,
        generation=0,
    )
    return manifest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_saved(state, admitted, model, root, checkpoint_hashes):
    model._check_execution_config()
    require(
        set(state) == {"metadata", "packed", "replicas"}
        and state["metadata"] == admitted,
        "saved metadata differs from the admitted producer",
    )
    require(
        admitted["format"] == "dsv41_native_pp_saved_dp_v1"
        and admitted["gen_num_per_cycle"] == model._gen_num_per_cycle == 0,
        "saved generation or format disagrees",
    )
    require(
        admitted["config"] == model._v4_args.v41_config,
        "saved model configuration disagrees",
    )
    require(model._sparse_attention_mode is None, "saved sparse policy disagrees")
    for component in producer.MHC_MODES:
        require(
            getattr(model, f"_mhc_{component}_mode")
            == admitted["choices"].get(f"mhc_{component}_mode"),
            "saved numerical policy disagrees",
        )
    require(
        admitted["bindings"]
        == [
            asdict(binding)
            for binding in dsv41_cache_bindings(model._v4_args.v41_config)
        ],
        "saved global source bindings disagree",
    )
    require(
        admitted["layouts"]
        == {tag: asdict(layout) for tag, layout in model.kv_cache.layouts.items()},
        "saved planar layout disagrees",
    )
    require(
        checkpoint_identities(root / "checkpoint") == admitted["identities"],
        "saved checkpoint/tokenizer identity disagrees",
    )
    require(
        checkpoint_hashes == admitted["checkpoint_sha256"],
        "checkpoint bytes changed after export",
    )
    require(
        len(state["packed"])
        == len(state["replicas"])
        == len(admitted["requests"])
        == 2,
        "saved request count disagrees",
    )
    bindings = dsv41_cache_bindings(model._v4_args.v41_config)
    keys = {pool_name(binding) for binding in bindings}
    for request, metadata in enumerate(admitted["requests"]):
        prompt = metadata["prompt"]
        require(
            metadata["start"] == len(prompt)
            and 1 <= len(prompt) <= 256
            and metadata["history"] == prompt[-3:][::-1]
            and all(type(token) is int and 0 <= token < 127 for token in prompt)
            and type(metadata["next_token"]) is int
            and 0 <= metadata["next_token"] < 127,
            "saved prompt/history/sample disagrees",
        )
        require(
            set(state["packed"][request]) == keys == set(metadata["packed_sha256"]),
            "saved continuation omits a canonical pool",
        )
        for binding in bindings:
            name = pool_name(binding)
            rows = state["packed"][request][name]
            layout = model.kv_cache.layouts[binding.tag]
            require(
                isinstance(rows, torch.Tensor)
                and rows.device.type == "cpu"
                and rows.dtype == torch.uint8
                and rows.is_contiguous()
                and rows.shape
                == (
                    layout.entries,
                    layout.payload_bytes_per_entry + layout.scale_bytes_per_entry,
                ),
                "saved packed shape/dtype disagrees",
            )
            require(
                byte_hash(rows) == metadata["packed_sha256"][name],
                "saved packed bytes disagree",
            )
        require(
            set(state["replicas"][request]) == {"20:hca_kv", "20:indexer_kv"},
            "saved L20 replica keys disagree",
        )
        for name, replica in state["replicas"][request].items():
            rows = state["packed"][request][name]
            require(
                isinstance(replica, torch.Tensor)
                and replica.dtype == torch.uint8
                and replica.device.type == "cpu"
                and replica.is_contiguous()
                and replica.shape == rows.shape
                and torch.equal(replica, rows),
                "saved L20 replica bytes disagree",
            )


def destination_tables(cache, first_page, second_page):
    return {
        tag: torch.tensor(
            [[first_page]] if tag in STATE_TAGS else [[first_page, second_page]],
            device="cuda",
            dtype=torch.int32,
        )
        for tag in cache.group_tags
    }


def import_saved(model, state, request, first_page):
    for binding in dsv41_cache_bindings(model._v4_args.v41_config):
        group = model.kv_cache.groups[binding.storage_layer_id, binding.tag]
        rows = state["packed"][request][pool_name(binding)].cuda()
        if binding.tag == "csa_state":
            group.kv_cache_base[first_page].copy_(rows.flatten())
        else:
            view = V41PagedCache(group.kv_cache_base, binding.tag, 0)
            slots = torch.arange(
                first_page * view.layout.entries,
                (first_page + 1) * view.layout.entries,
                device="cuda",
                dtype=torch.int32,
            )
            view.scatter(rows, slots)


def packed_destination(model, tables):
    packed = {}
    for binding in dsv41_cache_bindings(model._v4_args.v41_config):
        group = model.kv_cache.groups[binding.storage_layer_id, binding.tag]
        layout = model.kv_cache.layouts[binding.tag]
        pages = tables[binding.tag].flatten().tolist()
        if binding.tag == "csa_state":
            rows = (
                group.kv_cache_base[pages]
                .reshape(-1, layout.payload_bytes_per_entry)
                .cpu()
                .clone()
            )
        else:
            view = V41PagedCache(group.kv_cache_base, binding.tag, 0)
            slots = torch.cat(
                [
                    torch.arange(
                        page * layout.entries,
                        (page + 1) * layout.entries,
                        device="cuda",
                        dtype=torch.int32,
                    )
                    for page in pages
                ]
            )
            rows = view.gather(slots).cpu()
        packed[pool_name(binding)] = rows
    return packed


def guard_pools(cache, tables):
    checks = 0
    for key, raw in cache.storage.items():
        layout = cache.layouts[key[1]]
        allowed = set(tables[key[1]].flatten().tolist())
        assert 0 not in allowed
        for page in range(raw.size(0)):
            if page not in allowed:
                assert bool((raw[page] == SENTINEL).all())
                checks += 1
        end = layout.scale_plane_offset + layout.entries * layout.scale_bytes_per_entry
        assert bool((raw[:, end:] == SENTINEL).all())
        checks += 1
    return checks


def idle_decode(cache):
    tagged = {}
    for tag in cache.group_tags:
        attn = PyAttentionInputs()
        attn.is_prefill = False
        attn.input_lengths = torch.empty(0, dtype=torch.int32)
        attn.prefix_lengths = torch.empty(0, dtype=torch.int32)
        attn.sequence_lengths = torch.empty(0, dtype=torch.int32)
        attn.cu_seqlens = torch.zeros(1, dtype=torch.int32)
        attn.kv_cache_kernel_block_id_device = torch.empty(
            (0, 1), device="cuda", dtype=torch.int32
        )
        tagged[tag] = attn
    return PyModelInputs(
        input_ids=torch.empty(0, device="cuda", dtype=torch.int32),
        attention_inputs=tagged,
        engram_token_windows=torch.empty((0, 4), dtype=torch.int32),
    )


def mutate_saved(state, variant):
    if variant == "generation":
        state["metadata"]["gen_num_per_cycle"] = 1
    elif variant == "policy":
        state["metadata"]["choices"]["sparse_attention_mode"] = "short_preserved"
    elif variant == "missing_pool":
        state["packed"][0].pop("0:swa_kv")
    elif variant in ("payload", "scale"):
        state["packed"][0]["2:csa_kv"][0, 0 if variant == "payload" else -1] ^= 1
    elif variant == "table":
        state["metadata"]["sources"][0]["tables"]["swa_kv"][0][0] = 0
    elif variant == "replica":
        state["replicas"][0]["20:hca_kv"][0, 0] ^= 1
    elif variant == "checkpoint":
        state["metadata"]["identities"]["checkpoint_sha256"] = "0" * 64
    elif variant == "history":
        state["metadata"]["requests"][0]["history"][0] ^= 1
    elif variant == "sample":
        state["metadata"]["requests"][0]["next_token"] = 127
    elif variant == "shape":
        state["packed"][0]["2:csa_kv"] = state["packed"][0]["2:csa_kv"].flatten()
    elif variant == "dtype":
        state["packed"][0]["2:csa_kv"] = state["packed"][0]["2:csa_kv"].to(torch.int16)
    elif variant == "binding":
        state["metadata"]["bindings"][0]["source_layer_id"] = 40
    else:
        assert variant == "precision"
        torch.set_float32_matmul_precision("high")


def active_rank(rank, step):
    return (
        step in (4, 6, 7)
        or (step == 1 and rank == 0)
        or (step == 2 and rank % 2 == 0)
        or (step == 3 and rank % 2 == 1)
        or (step == 5 and rank == 7)
    )


@torch.inference_mode()
def receiver_rank(
    rank, rendezvous, root, manifest, oracles, phase, first_page, second_page
):
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability() == (12, 0)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    root = Path(root)
    directory = root / phase
    admitted = manifest["metadata"]
    os.environ.update(
        {
            "NCCL_P2P_DISABLE": "1",
            "NCCL_DEBUG": "WARN",
            "DSV41_ENGRAM_HOST_BYTES": str(2**20),
            "DSV41_ENGRAM_CACHE_DIR": str(directory / "tables"),
            "DSV4_MOE_STRATEGY": "auto",
            "DSV4_MOE_CHUNK_PREFILL": "1",
            "DSV4_MOE_CHUNK_TOKENS": "4",
            "DSV4_MOE_EXTENT_COUNT_FUSE": "1",
            "MOEDBG": "0",
        }
    )
    Dsv4SharedRuntimeBufferStore._reset_for_test()
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=180),
    )
    model = None
    try:
        with producer.pp_mhc_equations(admitted["choices"], False):
            config, kv_config = load_configuration(root / "checkpoint", max_seq_len=512)
            config.deepseek_v41_config.update(admitted["choices"])
            parallelism, loader = make_loader(config, kv_config, 8, rank)
            loaded = loader.load_weights(f"cuda:{rank}")
            signature = producer.loader_signature(loaded)
            assert signature == oracles["destination_loader"][str(rank)]
            source_weights = {
                f"{layer_id}:{name}": value
                for layer_id, mapping in enumerate(loaded.weights)
                for name, value in mapping.items()
                if name
                in (W.v4_compressor_wkv, W.v4_compressor_wgate, W.v41_indexer_wk)
            }
            source_weight_hashes = {
                name: byte_hash(value) for name, value in source_weights.items()
            }
            checkpoint_hashes = {
                name: digest(root / "checkpoint" / name)
                for name in admitted["checkpoint_sha256"]
            }
            head = loaded.global_weights[W.lm_head].clone()
            model = DeepSeekV41Model(config, parallelism, loaded, None, 2)
            del loaded, loader
            assert model.initialize(resources())
            assert model.initialize(resources(tagged_cache(model, pages=8)))
            assert len(model.v4.layers) == 40 and len(model.kv_cache.groups) == 51
            for layer in model.v4.layers:
                for unit in (layer.attn_hc, layer.ffn_hc):
                    for component in producer.MHC_MODES:
                        assert getattr(unit, f"{component}_mode") == admitted[
                            "choices"
                        ].get(f"mhc_{component}_mode")
            emit(
                "DSV41_HANDOFF_LOADER",
                label=root.name,
                rank=rank,
                role=phase,
                ep_size=8,
                expert_begin=rank * 2,
                expert_count=2,
                entries_exact=len(signature),
            )
            table_paths = [table._path for table in model._engram_tables.values()]
            state = torch.load(root / "state.pt", map_location="cpu", weights_only=True)
            assert digest(root / "state.pt") == manifest["state_sha256"]
            for raw in model.kv_cache.storage.values():
                raw.fill_(SENTINEL)
            before = {
                key: value.clone() for key, value in model.kv_cache.storage.items()
            }
            before_hash = {
                f"{key[0]}:{key[1]}": byte_hash(value) for key, value in before.items()
            }
            rejected = []
            variants = (*REJECTIONS, "precision") if admitted["choices"] else REJECTIONS
            for index, variant in enumerate(variants):
                actor = index % 8
                mutated = copy.deepcopy(state)
                if rank == actor:
                    mutate_saved(mutated, variant)
                error = None
                try:
                    validate_saved(mutated, admitted, model, root, checkpoint_hashes)
                except ValueError as exc:
                    error = exc
                finally:
                    if variant == "precision" and rank == actor:
                        assert torch.get_float32_matmul_precision() == "high"
                        torch.set_float32_matmul_precision("highest")
                try:
                    model._agree_dp(
                        error,
                        ("saved-pp-admission", variant),
                        head.device,
                        native_dispatch=True,
                    )
                except V41InputRejected:
                    assert (error is not None) == (rank == actor)
                else:
                    raise AssertionError("one-rank saved-state failure was ignored")
                for key, value in before.items():
                    assert torch.equal(value, model.kv_cache.storage[key])
                assert {
                    name: byte_hash(value) for name, value in source_weights.items()
                } == source_weight_hashes
                assert model._worker_state == "ready"
                clean_worker(model)
                validate_saved(state, admitted, model, root, checkpoint_hashes)
                model._agree_dp(
                    None,
                    ("saved-pp-recovery", variant),
                    head.device,
                    native_dispatch=True,
                )
                rejected.append(
                    {
                        "variant": variant,
                        "actor": actor,
                        "local_error": error is not None,
                        "cache_unchanged": True,
                        "source_weights_unchanged": True,
                        "ready": True,
                        "recovered": True,
                    }
                )
                emit(
                    "DSV41_HANDOFF_REJECT",
                    label=root.name,
                    phase=phase,
                    rank=rank,
                    **rejected[-1],
                )
            validate_saved(state, admitted, model, root, checkpoint_hashes)
            model._agree_dp(None, "saved-pp-import", head.device, native_dispatch=True)
            request = rank % 2
            metadata = admitted["requests"][request]
            tables = destination_tables(model.kv_cache, first_page, second_page)
            import_saved(model, state, request, first_page)
            packed = packed_destination(model, tables)
            for binding in dsv41_cache_bindings(model._v4_args.v41_config):
                name = pool_name(binding)
                entries = model.kv_cache.layouts[binding.tag].entries
                assert torch.equal(
                    packed[name][:entries], state["packed"][request][name]
                )
                if binding.tag not in STATE_TAGS:
                    assert bool((packed[name][entries:] == SENTINEL).all())
            checks = guard_pools(model.kv_cache, tables)
            imported = {
                key: raw.cpu().clone() for key, raw in model.kv_cache.storage.items()
            }
            start, history, token = (
                metadata["start"],
                metadata["history"].copy(),
                metadata["next_token"],
            )
            steps = []
            for step in range(ROUNDS):
                active = active_rank(rank, step)
                previous = {
                    key: raw.clone() for key, raw in model.kv_cache.storage.items()
                }
                inputs = (
                    decode_inputs(
                        model.kv_cache,
                        [token],
                        [start],
                        [history],
                        prompt_lengths=[metadata["start"]],
                        tables=tables,
                    )
                    if active
                    else idle_decode(model.kv_cache)
                )
                windows = inputs.engram_token_windows.cpu().clone()
                hidden = model.forward(inputs).hidden_states
                clean_worker(model)
                assert (
                    hidden.shape == (1 if active else 0, 5120)
                    and torch.isfinite(hidden).all()
                )
                logits = F.linear(hidden, head).float()
                output_token = int(logits.argmax(-1).item()) if active else None
                record = {
                    "step": step,
                    "active": active,
                    "start": start,
                    "history": history.copy(),
                    "input_token": token if active else None,
                    "output_token": output_token,
                    "windows": windows,
                    "hidden": hidden.cpu().clone(),
                    "logits": logits.cpu().clone(),
                    "packed": packed_destination(model, tables),
                }
                if active:
                    assert 0 <= output_token < 127
                    assert windows.tolist() == [[token, *history]]
                    history, token, start = (
                        [token, *history[:2]],
                        output_token,
                        start + 1,
                    )
                else:
                    assert all(
                        torch.equal(raw, model.kv_cache.storage[key])
                        for key, raw in previous.items()
                    )
                checks += guard_pools(model.kv_cache, tables)
                steps.append(record)
                emit(
                    "DSV41_HANDOFF_ROUND",
                    label=root.name,
                    phase=phase,
                    rank=rank,
                    step=step,
                    active=active,
                    start=record["start"],
                    hidden_sha256=byte_hash(hidden),
                    output_token=output_token,
                    idle_cache_exact=not active,
                )
            final = {
                key: raw.cpu().clone() for key, raw in model.kv_cache.storage.items()
            }
            result = {
                "rank": rank,
                "phase": phase,
                "request": request,
                "tables": {tag: table.cpu().clone() for tag, table in tables.items()},
                "loader": signature,
                "rejections": rejected,
                "initial_sha256": before_hash,
                "source_weight_sha256": source_weight_hashes,
                "imported": imported,
                "steps": steps,
                "final": final,
                "start": start,
                "history": history,
                "next_token": token,
                "guard_checks": checks,
            }
            producer.save_tensors(directory / f"rank{rank}.pt", result)
            dist.barrier()
            model.close()
            dist.barrier()
            assert all(not path.exists() for path in table_paths)
            assert not list((directory / "tables").glob("*.lease.*"))
            emit(
                "DSV41_HANDOFF_LIFETIME",
                label=root.name,
                phase=phase,
                rank=rank,
                worker_closed=model._worker_state == "closed",
                engram_released=True,
                cache_pools=51,
                guard_checks=checks,
                standalone_decode_role=False,
            )
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if model is not None:
            model.close()
        Dsv4SharedRuntimeBufferStore._reset_for_test()
        dist.destroy_process_group()


def gpu_observation():
    memory = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=5,
    )
    clients = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
        timeout=5,
    )
    return {
        "memory_mib": [int(line.split(",")[1]) for line in memory.splitlines()],
        "compute_pids": sorted({int(pid) for pid in clients.splitlines()}),
    }


def drain_to_baseline(baseline):
    deadline = time.monotonic() + 30
    while True:
        observed = gpu_observation()
        if observed == baseline:
            return observed
        if time.monotonic() >= deadline:
            raise AssertionError(
                ("GPU teardown did not restore baseline", baseline, observed)
            )
        time.sleep(0.25)


def compare_receivers(root):
    tensors = 0
    for rank in range(8):
        left = torch.load(
            root / "receiver-a" / f"rank{rank}.pt",
            map_location="cpu",
            weights_only=True,
        )
        right = torch.load(
            root / "receiver-b" / f"rank{rank}.pt",
            map_location="cpu",
            weights_only=True,
        )
        for name in ("steps", "loader", "rejections", "start", "history", "next_token"):
            tensors += producer.require_exact_tree(left[name], right[name])
        assert all(
            not torch.equal(left["tables"][tag], right["tables"][tag])
            for tag in left["tables"]
        )
    emit(
        "DSV41_HANDOFF_COMPARISON",
        label=root.name,
        ranks_exact=8,
        rounds=ROUNDS,
        logical_tensors_exact=tensors,
        relocated=True,
        bounded_weights=True,
        live_pd=False,
        standalone_decode_role=False,
    )
    return tensors


class V41SavedPPDPSM120Test(unittest.TestCase):
    def test_bounded_native_control(self):
        producer.V41PPCheckpointEngineSM120Test().test_bounded_named_checkpoint()

    def test_named_loader_control(self):
        pp_loader.V41PPLoaderSM120Test().test_named_stage_loader_and_cold_background_engine()

    def handoff_case(self, boundary, selected):
        self.assertEqual(torch.cuda.device_count(), 8)
        lengths = [127, 129] if boundary == 128 else [255, 256]
        choices = (
            {
                f"mhc_{component}_mode": mode
                for component, mode in producer.MHC_MODES.items()
            }
            if selected
            else {}
        )
        label = f"{'composed' if selected else 'default'}-boundary{boundary}"
        root = Path(os.environ["DSV41_TEST_PP_DP_ROOT"]).resolve() / label
        checkpoint = root / "checkpoint"
        checkpoint.mkdir(parents=True)
        torch.manual_seed(41701)
        args, weights = wide_fixture(experts=16)
        weights.global_weights[W.lm_head][127].copy_(
            weights.global_weights[W.lm_head][126]
        )
        save_checkpoint(checkpoint, args, weights)
        oracles = loader_oracles(args, weights)
        oracles["label"] = label
        save_json(root / "loader-oracles.json", oracles)
        del weights
        gc.collect()
        torch.cuda.empty_cache()
        baseline = gpu_observation()
        prompts = [
            [(3 + request * 17 + token) % 127 for token in range(length)]
            for request, length in enumerate(lengths)
        ]
        directory = root / "producer"
        (directory / "tables").mkdir(parents=True)
        producer.V41PPCheckpointEngineSM120Test().run_checkpoint(
            checkpoint,
            directory / "tables",
            prompts,
            full=False,
            mhc_choices=choices,
            artifact_directory=directory,
            max_seq_len=512,
            saved_handoff=oracles,
        )
        drain = [{"phase": "producer", "observation": drain_to_baseline(baseline)}]
        manifest = merge_producer(root, choices, prompts, oracles)
        for phase, first_page, second_page in (
            ("receiver-a", 2, 4),
            ("receiver-b", 3, 1),
        ):
            directory = root / phase
            (directory / "tables").mkdir(parents=True)
            rendezvous = directory / "world.rendezvous"
            assert not rendezvous.exists()
            mp.spawn(
                receiver_rank,
                args=(
                    rendezvous.as_uri(),
                    str(root),
                    manifest,
                    oracles,
                    phase,
                    first_page,
                    second_page,
                ),
                nprocs=8,
                join=True,
            )
            rendezvous.unlink(missing_ok=True)
            assert not rendezvous.exists()
            drain.append({"phase": phase, "observation": drain_to_baseline(baseline)})
            self.assertTrue(
                all(
                    path.name.endswith(".lock")
                    for path in (directory / "tables").iterdir()
                )
            )
        tensors = compare_receivers(root)
        save_json(
            root / "completion.json",
            {
                "label": label,
                "choices": choices,
                "prompts": prompts,
                "baseline": baseline,
                "drain": drain,
                "logical_tensors_exact": tensors,
                "file_rendezvous_removed": True,
                "producer_released_before_receivers": True,
                "standalone_decode_role": False,
                "live_pd": False,
            },
        )

    def test_default_window_handoff(self):
        self.handoff_case(128, False)

    def test_composed_window_handoff(self):
        self.handoff_case(128, True)

    def test_default_page_handoff(self):
        self.handoff_case(256, False)

    def test_composed_page_handoff(self):
        self.handoff_case(256, True)


if __name__ == "__main__":
    unittest.main()

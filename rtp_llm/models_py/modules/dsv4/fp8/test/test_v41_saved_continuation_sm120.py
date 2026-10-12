"""Cold DP8 continuation from saved packed pages; no DECODE RPC admission."""

import copy
import hashlib
import json
import os
import socket
import subprocess
import time
import traceback
import unittest
from contextlib import ExitStack
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from rtp_llm.models.dsv41_kv_cache import dsv41_cache_bindings
from rtp_llm.models_py.model_desc.deepseek_v4_model import Dsv4SharedRuntimeBufferStore
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_dp_decode_sm120 import idle_decode
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_dp_worker_sm120 import (
    assert_released,
    make_workers,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_worker_sm120 import (
    checkpoint,
    decode_inputs,
    inputs,
    resources,
    tagged_cache,
)
from rtp_llm.models_py.modules.dsv4.hc import delayed as delayed_hc
from rtp_llm.models_py.modules.dsv4.v41_cache import V41PagedCache
from rtp_llm.utils.model_weight import W

CHOICES = {
    "rotary_compute_device": "cuda",
    "sparse_attention_mode": "short_preserved",
}
SOURCE_PAGES = {
    "swa_kv": 2,
    "csa_kv": 2,
    "hca_kv": 1,
    "indexer_kv": 2,
    "csa_state": 1,
}
DESTINATION_PAGE = 3
SENTINEL = 211
ROUNDS = 12


def worker_choices(model):
    choices = dict(CHOICES)
    for component in ("collapse", "post"):
        key = f"mhc_{component}_mode"
        mode = model._v4_args.v41_config.get(key)
        if mode is not None:
            choices[key] = mode
    return choices


def emit(marker, **values):
    print(marker + " " + json.dumps(values), flush=True)


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


def drain_to_control_baseline(baseline):
    # Process exit can precede CUDA/NVML teardown. Keep the GPU lease and poll.
    deadline = time.monotonic() + 30
    while True:
        observed = gpu_observation()
        if observed == baseline:
            return observed
        if time.monotonic() >= deadline:
            raise AssertionError(
                ("GPU drain did not restore control baseline", baseline, observed)
            )
        time.sleep(0.25)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def byte_hash(value):
    raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def save_tensor(path, value):
    with path.open("xb") as stream:
        torch.save(value, stream)


def save_json(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def pool_name(key):
    return f"{key[0]}:{key[1]}"


def weight_hash(model):
    hasher = hashlib.sha256()
    for layer, mapping in enumerate(
        (model.weight.global_weights, *model.weight.weights)
    ):
        for name, value in sorted(mapping.items()):
            hasher.update(
                json.dumps((layer, name, list(value.shape), str(value.dtype))).encode()
            )
            hasher.update(bytes.fromhex(byte_hash(value)))
    return hasher.hexdigest()


def tables(cache, pages):
    return {
        tag: torch.tensor([[pages[tag]]], device="cuda", dtype=torch.int32)
        for tag in cache.group_tags
    }


def packed_pages(cache, generation, pages):
    result = {}
    for key, group in sorted(cache.groups.items()):
        if key[1] == "csa_state":
            # Compressor state is raw FP32 storage, without quantized scales.
            layout = cache.layouts[key[1]]
            result[pool_name(key)] = (
                group.kv_cache_base[pages[key[1]]]
                .reshape(layout.entries, layout.payload_bytes_per_entry)
                .cpu()
                .clone()
            )
            continue
        view = V41PagedCache(group.kv_cache_base, key[1], generation)
        begin = pages[key[1]] * view.layout.entries
        slots = torch.arange(
            begin, begin + view.layout.entries, device="cuda", dtype=torch.int32
        )
        result[pool_name(key)] = view.gather(slots).cpu()
    return result


def metadata(model, generation, rank, identity, prompt, hidden, head, packed):
    binding = dsv41_cache_bindings(model._v4_args.v41_config)
    assert len(binding) == len(model.kv_cache.groups) == 51
    assert {pool_name((b.storage_layer_id, b.tag)) for b in binding} == set(packed)
    return {
        "format": "dsv41_saved_worker_continuation_v1",
        "generation": generation,
        "rank": rank,
        "weight_sha256": identity,
        "config": model._v4_args.v41_config,
        "choices": worker_choices(model),
        "bindings": [asdict(b) for b in binding],
        "layouts": {
            tag: asdict(layout) for tag, layout in model.kv_cache.layouts.items()
        },
        "source_pages": SOURCE_PAGES,
        "checkpoint_sha256": {
            path.name: file_hash(path)
            for path in sorted(Path(model._ckpt_path).iterdir())
            if path.is_file()
        },
        "prompt": prompt,
        "start": len(prompt),
        "history": prompt[-3:][::-1],
        "next_token": sample(hidden, head),
        "packed_sha256": {key: byte_hash(value) for key, value in packed.items()},
    }


def sample(hidden, head):
    return int(F.linear(hidden[-1:], head).float().argmax().item())


def validate_saved(state, admitted, model, generation, rank, identity):
    # Bind externally admitted metadata before scatter.
    model._check_execution_config()
    if set(state) != {"metadata", "packed"} or state["metadata"] != admitted:
        raise ValueError("saved metadata differs from the admitted producer")
    if (
        admitted["format"] != "dsv41_saved_worker_continuation_v1"
        or admitted["generation"] != generation
        or admitted["rank"] != rank
        or admitted["weight_sha256"] != identity
        or admitted["config"] != model._v4_args.v41_config
        or admitted["choices"] != worker_choices(model)
        or model._mhc_collapse_mode != admitted["choices"].get("mhc_collapse_mode")
        or model._mhc_post_mode != admitted["choices"].get("mhc_post_mode")
        or model._rotary_compute_device != "cuda"
        or model._sparse_attention_mode != "short_preserved"
    ):
        raise ValueError("saved producer and cold worker identities disagree")
    expected_bindings = [
        asdict(b) for b in dsv41_cache_bindings(model._v4_args.v41_config)
    ]
    if admitted["bindings"] != expected_bindings:
        raise ValueError("saved physical source bindings disagree")
    if admitted["layouts"] != {
        tag: asdict(layout) for tag, layout in model.kv_cache.layouts.items()
    }:
        raise ValueError("saved page generation/layout disagrees")
    for name, expected in admitted["checkpoint_sha256"].items():
        if file_hash(Path(model._ckpt_path) / name) != expected:
            raise ValueError("saved checkpoint identity changed")
    prompt = admitted["prompt"]
    if (
        not 1 <= len(prompt) <= 128
        or admitted["start"] != len(prompt)
        or admitted["history"] != prompt[-3:][::-1]
        or any(type(token) is not int or not 0 <= token < 127 for token in prompt)
        or type(admitted["next_token"]) is not int
        or not 0 <= admitted["next_token"] < 127
    ):
        raise ValueError("saved request/history is outside the admitted fixture")
    expected_keys = {pool_name(key) for key in model.kv_cache.groups}
    if (
        set(state["packed"]) != expected_keys
        or set(admitted["packed_sha256"]) != expected_keys
    ):
        raise ValueError("saved continuation omits or adds a physical pool")
    for key, group in model.kv_cache.groups.items():
        name = pool_name(key)
        rows = state["packed"][name]
        layout = model.kv_cache.layouts[key[1]]
        if (
            rows.device.type != "cpu"
            or rows.dtype != torch.uint8
            or not rows.is_contiguous()
            or rows.shape
            != (
                layout.entries,
                layout.payload_bytes_per_entry + layout.scale_bytes_per_entry,
            )
            or byte_hash(rows) != admitted["packed_sha256"][name]
        ):
            raise ValueError("saved packed row shape or hash disagrees")


def import_pages(model, generation, state):
    for key, group in sorted(model.kv_cache.groups.items()):
        if key[1] == "csa_state":
            group.kv_cache_base[DESTINATION_PAGE].copy_(
                state["packed"][pool_name(key)].cuda().reshape(-1)
            )
            continue
        view = V41PagedCache(group.kv_cache_base, key[1], generation)
        begin = DESTINATION_PAGE * view.layout.entries
        slots = torch.arange(
            begin, begin + view.layout.entries, device="cuda", dtype=torch.int32
        )
        view.scatter(state["packed"][pool_name(key)].cuda(), slots)
    # Every byte outside the payload/scale planes must retain the sentinel.
    for key, raw in model.kv_cache.storage.items():
        layout = model.kv_cache.layouts[key[1]]
        assert (raw[:DESTINATION_PAGE] == SENTINEL).all()
        assert (raw[DESTINATION_PAGE, layout.used_bytes :] == SENTINEL).all()


def active_round(rank, step):
    return (
        step in (0, 3, 5, 7, 8, 9, 10, 11)
        or (step == 1 and rank == 0)
        or (step == 2 and rank % 2 == 0)
        or (step == 4 and rank % 2 == 1)
    )


def continue_worker(model, head, generation, rank, admitted, pages, phase):
    start, history, token = (
        admitted["start"],
        list(admitted["history"]),
        admitted["next_token"],
    )
    result = []
    for step in range(ROUNDS):
        active = active_round(rank, step)
        before = packed_pages(model.kv_cache, generation, pages)
        if active:
            request = decode_inputs(
                model.kv_cache,
                [token],
                [start],
                [history],
                prompt_lengths=[admitted["start"]],
                tables=tables(model.kv_cache, pages),
            )
        else:
            request = idle_decode(model.kv_cache)
        hidden = model.forward(request).hidden_states.cpu()
        assert_released(model)
        after = packed_pages(model.kv_cache, generation, pages)
        record = {
            "step": step,
            "active": active,
            "start": start,
            "input_token": token if active else None,
            "history": list(history),
            "hidden": hidden,
            "cache_sha256": {key: byte_hash(value) for key, value in after.items()},
            "output_token": sample(hidden.cuda(), head) if active else None,
        }
        if active:
            assert hidden.shape == (1, 512) and torch.isfinite(hidden).all()
            assert record["output_token"] < 127
            history, token = [token, *history[:2]], record["output_token"]
            start += 1
        else:
            assert hidden.shape == (0, 512)
            for key in before:
                assert torch.equal(before[key], after[key])
        result.append(record)
        emit(
            "DSV41_SAVED_ROUND",
            generation=generation,
            rank=rank,
            phase=phase,
            step=step,
            active=active,
            start=record["start"],
            output_token=record["output_token"],
            hidden_sha256=byte_hash(hidden),
            idle_cache_exact=not active,
        )
    return {
        "steps": result,
        "final": packed_pages(model.kv_cache, generation, pages),
        "start": start,
        "history": history,
        "next_token": token,
    }


def run_rank(rank, port, config, generation, phase, directory):
    with ExitStack() as stack:
        for component in ("collapse", "post"):
            if config.get(f"mhc_{component}_mode") == "ordered_fp32":
                stack.enter_context(
                    patch.object(
                        delayed_hc,
                        f"{component}_delayed",
                        side_effect=AssertionError(
                            f"Selected continuation reached default {component}"
                        ),
                    )
                )
        run_rank_impl(rank, port, config, generation, phase, directory)


@torch.inference_mode()
def run_rank_impl(rank, port, config, generation, phase, directory):
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability() == (12, 0)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    directory = Path(directory)
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
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=180),
    )
    model = peer = None
    try:
        model, peer = make_workers(rank, config, generation, initialize_model=False)
        peer.close()
        peer = None
        head = model.weight.global_weights[W.lm_head]
        # Match the earlier bounded smoke's deterministic image-token tie.
        head[127].copy_(head[126])
        identity = weight_hash(model)
        head = head.clone()
        assert model.initialize(resources())
        assert model.initialize(resources(tagged_cache(model)))
        assert len(model.v4.layers) == 40 and len(model._engram_tables) == 2
        for layer in model.v4.layers:
            layer.attn._check_execution_config()
            assert layer.attn.freqs_cis.device == torch.device("cuda", rank)
            mode = config.get("mhc_collapse_mode")
            assert layer.attn_hc.collapse_mode == mode
            assert layer.ffn_hc.collapse_mode == mode
            assert layer.attn_hc.post_mode == config.get("mhc_post_mode")
            assert layer.ffn_hc.post_mode == config.get("mhc_post_mode")
        if config.get("mhc_collapse_mode") is not None:
            emit(
                "DSV41_SAVED_COLLAPSE_OWNER",
                generation=generation,
                rank=rank,
                phase=phase,
                layers=len(model.v4.layers),
                mhc_collapse_mode="ordered_fp32",
                diagnostic_collapse_hook=False,
            )
        if config.get("mhc_post_mode") is not None:
            emit(
                "DSV41_SAVED_POST_OWNER",
                generation=generation,
                rank=rank,
                phase=phase,
                layers=len(model.v4.layers),
                mhc_post_mode="ordered_fp32",
                mhc_collapse_mode=config.get("mhc_collapse_mode"),
                diagnostic_post_hook=False,
            )
        state_path = directory / f"rank{rank}.state.pt"
        manifest_path = directory / f"rank{rank}.state.json"
        if phase == "source":
            prompt = [(3 + rank * 7 + offset) % 127 for offset in range(3 + rank % 2)]
            hidden = model.forward(
                inputs(
                    model.kv_cache,
                    prompt,
                    [len(prompt)],
                    tables=tables(model.kv_cache, SOURCE_PAGES),
                )
            ).hidden_states
            assert_released(model)
            packed = packed_pages(model.kv_cache, generation, SOURCE_PAGES)
            admitted = metadata(
                model, generation, rank, identity, prompt, hidden, head, packed
            )
            state = {"metadata": admitted, "packed": packed}
            save_tensor(state_path, state)
            save_json(
                manifest_path,
                {"state_sha256": file_hash(state_path), "metadata": admitted},
            )
            expected = continue_worker(
                model, head, generation, rank, admitted, SOURCE_PAGES, phase
            )
            save_tensor(directory / f"rank{rank}.expected.pt", expected)
            emit(
                "DSV41_SAVED_EXPORT",
                generation=generation,
                rank=rank,
                pools=len(packed),
                prompt_length=len(prompt),
                weight_sha256=identity,
                state_sha256=file_hash(state_path),
                pages=SOURCE_PAGES,
            )
        else:
            assert phase == "resume"
            manifest = json.loads(manifest_path.read_text())
            assert file_hash(state_path) == manifest["state_sha256"]
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            admitted = manifest["metadata"]
            for raw in model.kv_cache.storage.values():
                raw.fill_(SENTINEL)
            before = {key: raw.clone() for key, raw in model.kv_cache.storage.items()}
            variants = [
                "generation",
                "policy",
                "missing_pool",
                "corrupt_row",
                "history",
            ]
            if (
                config.get("mhc_post_mode") is not None
                and config.get("mhc_collapse_mode") is not None
            ):
                variants.append("collapse_policy")
            for variant in variants:
                mutated = copy.deepcopy(state)
                if rank == 0:
                    if variant == "generation":
                        mutated["metadata"]["generation"] = 1 - generation
                    elif variant in ("policy", "collapse_policy"):
                        policy = (
                            "mhc_collapse_mode"
                            if variant == "collapse_policy"
                            else (
                                "mhc_post_mode"
                                if config.get("mhc_post_mode") is not None
                                else (
                                    "mhc_collapse_mode"
                                    if config.get("mhc_collapse_mode") is not None
                                    else "sparse_attention_mode"
                                )
                            )
                        )
                        mutated["metadata"]["choices"][policy] = None
                    elif variant == "history":
                        mutated["metadata"]["history"][0] += 1
                    elif variant == "missing_pool":
                        mutated["packed"].pop(next(iter(mutated["packed"])))
                    else:
                        mutated["packed"][next(iter(mutated["packed"]))][0, 0] ^= 1
                error = None
                try:
                    validate_saved(mutated, admitted, model, generation, rank, identity)
                except ValueError as exc:
                    error = exc
                try:
                    model._agree_dp(error, ("saved-preflight", variant), head.device)
                except (ValueError, RuntimeError):
                    assert (error is not None) == (rank == 0)
                else:
                    raise AssertionError("one-rank saved-state failure was ignored")
                for key in before:
                    assert torch.equal(before[key], model.kv_cache.storage[key])
                assert model._worker_state == "ready"
                emit(
                    "DSV41_SAVED_REJECT",
                    generation=generation,
                    rank=rank,
                    actor=0,
                    variant=variant,
                    cache_exact=True,
                    ready=True,
                )
            validate_saved(state, admitted, model, generation, rank, identity)
            model._agree_dp(None, "saved-preflight-ready", head.device)
            import_pages(model, generation, state)
            pages = {tag: DESTINATION_PAGE for tag in model.kv_cache.group_tags}
            imported = packed_pages(model.kv_cache, generation, pages)
            for key in state["packed"]:
                assert torch.equal(state["packed"][key], imported[key])
            imported_storage = {
                pool_name(key): raw.cpu().clone()
                for key, raw in model.kv_cache.storage.items()
            }
            actual = continue_worker(
                model, head, generation, rank, admitted, pages, phase
            )
            expected = torch.load(
                directory / f"rank{rank}.expected.pt",
                map_location="cpu",
                weights_only=True,
            )
            for left, right in zip(expected["steps"], actual["steps"]):
                assert left.keys() == right.keys()
                for key in left:
                    if key == "hidden":
                        assert torch.equal(
                            left[key].view(torch.uint8), right[key].view(torch.uint8)
                        )
                    else:
                        assert left[key] == right[key], (
                            generation,
                            rank,
                            key,
                            left,
                            right,
                        )
            for key in expected["final"]:
                assert torch.equal(expected["final"][key], actual["final"][key])
            for key in ("start", "history", "next_token"):
                assert expected[key] == actual[key]
            actual.update(imported=imported, imported_storage=imported_storage)
            save_tensor(directory / f"rank{rank}.actual.pt", actual)
            emit(
                "DSV41_SAVED_RESUME",
                generation=generation,
                rank=rank,
                pools=len(imported),
                weight_sha256=identity,
                destination_page=DESTINATION_PAGE,
                hidden_token_cache_exact=True,
                padding_reserved_pages_exact=True,
                ready=True,
            )
        model.close()
        assert not model._engram_tables
        dist.barrier()
        emit(
            "DSV41_SAVED_RELEASE",
            generation=generation,
            rank=rank,
            phase=phase,
            tables_empty=True,
        )
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if peer is not None:
            peer.close()
        if model is not None:
            model.close()
        Dsv4SharedRuntimeBufferStore._reset_for_test()
        dist.destroy_process_group()


class SavedContinuationSM120Test(unittest.TestCase):
    def run_case(self, generation, *, collapse_mode=None, post_mode=None):
        self.assertEqual(torch.cuda.device_count(), 8)
        suffix = ("-collapse" if collapse_mode is not None else "") + (
            "-post" if post_mode is not None else ""
        )
        root = Path(os.environ["DSV41_TEST_HANDOFF_ROOT"]) / f"gen{generation}{suffix}"
        root.mkdir(parents=True)
        checkpoint_root = root / "checkpoint"
        checkpoint_root.mkdir()
        (root / "tables").mkdir()
        torch.manual_seed(41993)
        config = checkpoint(checkpoint_root, full_schedule=True)
        config.update(CHOICES)
        if collapse_mode is not None:
            config["mhc_collapse_mode"] = collapse_mode
        if post_mode is not None:
            config["mhc_post_mode"] = post_mode
        save_json(root / "fixture.json", config)
        for phase in ("source", "resume"):
            baseline = gpu_observation()
            emit(
                "DSV41_SAVED_CONTROL_BASELINE",
                generation=generation,
                phase=phase,
                observation=baseline,
            )
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            mp.spawn(
                run_rank,
                args=(port, config, generation, phase, str(root)),
                nprocs=8,
                join=True,
            )
            observed = drain_to_control_baseline(baseline)
            self.assertTrue(
                all(path.name.endswith(".lock") for path in (root / "tables").iterdir())
            )
            emit(
                "DSV41_SAVED_PHASE_RELEASE",
                generation=generation,
                phase=phase,
                observation=observed,
                control_baseline=baseline,
                table_locks_only=True,
            )
        emit(
            "DSV41_SAVED_CONTINUATION",
            generation=generation,
            layers=40,
            pools=51,
            rounds=ROUNDS,
            cold_receivers=8,
            completed=True,
            standalone_decode_role=False,
            **({} if collapse_mode is None else {"mhc_collapse_mode": collapse_mode}),
            **({} if post_mode is None else {"mhc_post_mode": post_mode}),
        )

    def test_generation_zero(self):
        self.run_case(0)

    def test_generation_one(self):
        self.run_case(1)

    def test_ordered_collapse_generation_zero(self):
        self.run_case(0, collapse_mode="ordered_fp32")

    def test_ordered_collapse_generation_one(self):
        self.run_case(1, collapse_mode="ordered_fp32")

    def test_ordered_post_generation_zero(self):
        self.run_case(0, post_mode="ordered_fp32")

    def test_ordered_post_generation_one(self):
        self.run_case(1, post_mode="ordered_fp32")

    def test_ordered_collapse_post_generation_zero(self):
        self.run_case(0, collapse_mode="ordered_fp32", post_mode="ordered_fp32")

    def test_ordered_collapse_post_generation_one(self):
        self.run_case(1, collapse_mode="ordered_fp32", post_mode="ordered_fp32")


if __name__ == "__main__":
    unittest.main()

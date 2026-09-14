"""Actual CP8 collectives versus complete-page attention on the same CUDA inputs.

Run on one authorized CUDA13 Blackwell eight-GPU host. Direct invocation starts
eight workers; torchrun may also launch this file with eight ranks explicitly.
This is a component integration test, not model or deployment acceptance.
"""

import json
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.modules.dsv4.cp import build_cp_context_for_forward
from rtp_llm.models_py.modules.dsv41.attention import V41Attention, V41AttentionCache
from rtp_llm.models_py.modules.dsv41.cache_layout import (
    PAIR_OWNERS,
    CacheLayout,
    CacheRegion,
    RegionSlot,
    layer_sources,
)
from rtp_llm.models_py.modules.dsv41.ced import ReplayConfig, ReplayMode
from rtp_llm.models_py.modules.dsv41.compressor import OwnerCompressor, PairCarry
from rtp_llm.models_py.modules.dsv41.cp import begin_cp_request

_LAYERS = (0, 2, 3, 8, 9, 14, 15, 20, 21, 24, 25, 28, 29, 32, 33, 36, 39)
_FLAGS = (
    "DSV41_ATTENTION",
    "DSV41_OWNER_COMPRESSOR",
    "DSV41_SPARSE_INDEXER",
    "DSV41_NATIVE_COMPACT_READER",
    "DSV41_NATIVE_COMPACT_WRITER",
)


def _metadata(lengths, starts, rank, device):
    chunks = [2 * ((length + 15) // 16) for length in lengths]
    padded = [8 * length for length in chunks]
    mask = torch.zeros(sum(padded), dtype=torch.int32)
    restore = torch.empty(sum(padded), dtype=torch.int32)
    local_first = global_first = 0
    for length, chunk, size in zip(lengths, chunks, padded):
        mask[global_first : global_first + length] = 1
        half = chunk // 2
        for peer in range(8):
            positions = list(range(peer * half, (peer + 1) * half))
            positions += list(range(size - (peer + 1) * half, size - peer * half))
            for local, position in enumerate(positions):
                restore[global_first + position] = (
                    peer * sum(chunks) + local_first + local
                )
        global_first += size
        local_first += chunk
    info = SimpleNamespace(
        prefill_actual_input_lengths_cpu=torch.tensor(lengths, dtype=torch.int32),
        prefill_cp_chunk_lengths=torch.tensor(chunks, dtype=torch.int32),
        prefill_qkv_padding_mask=mask.to(device),
        prefill_qkv_restore_indice=restore.to(device),
    )
    prefixes = torch.tensor(starts, dtype=torch.int32)
    return build_cp_context_for_forward(
        info,
        8,
        rank,
        sum(chunks),
        device,
        prefix_lengths=prefixes.to(device),
        prefix_lengths_host=prefixes,
        chunk_lengths_device=torch.tensor(chunks, dtype=torch.int32, device=device),
        kv_cache_sharded=True,
    )


def _framework_pages(layout, rank, device):
    pools, tables, pair_pools, pair_tables = {}, {}, {}, {}
    for page in layout.pages:
        pools[page.slot] = torch.zeros(
            (40, page.prefill_shard_bytes), dtype=torch.uint8, device=device
        )
        tables[page.slot] = torch.tensor(
            [[rank + 2, rank + 19]], dtype=torch.int32, device=device
        )
    for owner in PAIR_OWNERS:
        snapshots = next(
            state.snapshots
            for state in layout.pair_states
            if state.owner_layer == owner
        )
        shard_bytes = ((snapshots * 4112 + 511) // 512) * 512 // layout.cp_size
        pair_pools[owner] = torch.zeros(
            (40, shard_bytes), dtype=torch.uint8, device=device
        )
        pair_tables[owner] = torch.tensor(
            [[rank + 3, rank + 20]], dtype=torch.int32, device=device
        )
    return dict(
        pools=pools, tables=tables, pair_pools=pair_pools, pair_tables=pair_tables
    )


def _models(layout, device):
    def linear(inputs, outputs):
        module = nn.Linear(
            inputs, outputs, bias=False, dtype=torch.bfloat16, device=device
        )
        module.weight.data.zero_()
        return module

    wq_a, wq_b = linear(5120, 1280), linear(1280, 64 * 512)
    wkv, wo_b = linear(5120, 512), linear(8192, 5120)
    index_wq_b = linear(1280, 32 * 128)
    channels = torch.arange(512, device=device)
    wkv.weight.data[channels, channels] = 1
    wq_a.weight.data[
        torch.arange(1280, device=device), torch.arange(1280, device=device)
    ] = 1
    wo_b.weight.data[
        torch.arange(5120, device=device),
        (torch.arange(5120, device=device) % 8) * 1024,
    ] = 1
    wo_a = torch.zeros((8, 1024, 4096), dtype=torch.bfloat16, device=device)
    wo_a[:, 0, 0] = 1
    index_wk = torch.zeros((128, 512), dtype=torch.bfloat16, device=device)
    index_wk[torch.arange(128, device=device), torch.arange(128, device=device)] = 1
    models = {}
    for layer in _LAYERS:
        source = layer_sources(layer)
        compressor = None
        if source.writes_global:
            compressor = OwnerCompressor(
                layer,
                wkv.weight.detach().to(
                    torch.float32 if source.ratio == 2 else torch.bfloat16
                ),
                torch.ones(512, dtype=torch.bfloat16, device=device),
                (
                    torch.zeros((512, 5120), dtype=torch.float32, device=device)
                    if source.ratio == 2
                    else None
                ),
                layout=layout,
            )
        models[layer] = V41Attention(
            layer,
            wq_a=wq_a,
            wq_b=wq_b,
            wkv=wkv,
            wo_b=wo_b,
            wo_a=wo_a,
            q_norm=torch.ones(1280, dtype=torch.bfloat16, device=device),
            kv_norm=torch.ones(512, dtype=torch.bfloat16, device=device),
            sinks=torch.zeros(64, dtype=torch.float32, device=device),
            compressor=compressor,
            index_wq_b=index_wq_b if source.scores_queries else None,
            index_weights=(
                torch.zeros((32, 5120), dtype=torch.bfloat16, device=device)
                if source.scores_queries
                else None
            ),
            index_wk=index_wk if source.writes_index_k else None,
            index_norm=(
                torch.ones(128, dtype=torch.bfloat16, device=device)
                if source.writes_index_k
                else None
            ),
        )
    return models


def _hidden(first, last, request, device):
    token = torch.arange(first, last, device=device, dtype=torch.float32)[:, None]
    channel = torch.arange(5120, device=device, dtype=torch.float32)[None, :]
    return (0.75 + torch.sin(token * 0.17 + channel * 0.07 + request)).bfloat16()


def _equal(actual, expected, description):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=description)


def _compare_pages(layout, context, reference):
    for layer in _LAYERS:
        slot = RegionSlot(CacheRegion.SWA, layer)
        spec = context._page_specs[slot]
        page = int(context.tables[slot][0, context.current].item())
        source = reference.swa[layer]
        begin, end = spec.swa_byte_slice(context.cp.cp_rank)
        _equal(
            context.pools[slot][page],
            source.pages.data[source.page_ids[0], begin:end],
            f"SWA layer {layer}",
        )
        _equal(
            context.cache.swa[layer].valid_starts,
            source.valid_starts,
            f"SWA start {layer}",
        )
        _equal(
            context.cache.swa[layer].valid_ends, source.valid_ends, f"SWA end {layer}"
        )
    for owner in (2, 8, 14, 20):
        for region in (CacheRegion.GLOBAL, CacheRegion.INDEX_K):
            slot = RegionSlot(region, owner)
            table, pool = context.tables[slot], context.pools[slot]
            ref = reference.owners[owner]
            pages, ref_table = (
                (ref.global_kv.pages, ref.global_kv.page_table)
                if region == CacheRegion.GLOBAL
                else (ref.index_pages, ref.index_table)
            )
            blocks = (
                context.end + layout.token_block_size - 1
            ) // layout.token_block_size
            for logical in range(context.cp.cp_rank, blocks, 8):
                actual_id = table[0, logical // 8]
                expected_id = ref_table[0, logical]
                _equal(
                    pool[actual_id],
                    pages.data[expected_id],
                    f"{region} owner {owner} block {logical}",
                )
        if owner in PAIR_OWNERS:
            actual, expected = (
                context.cache.owners[owner].pair,
                reference.owners[owner].pair,
            )
            assert actual.next_position == expected.next_position == context.end
            if context.end % 2:
                _equal(actual.partial_kv, expected.partial_kv, f"pair KV {owner}")
                _equal(
                    actual.partial_score, expected.partial_score, f"pair score {owner}"
                )


@torch.inference_mode()
def _pair_checkpoint_restore(rank, device):
    observations = []
    for speculative in (0, 5):
        layout = CacheLayout(
            cp_size=8, speculative_tokens=speculative, draft_enabled=bool(speculative)
        )
        identity = ReplayConfig(ReplayMode.FULL).cache_identity("pair-memory", layout)
        framework = _framework_pages(layout, rank, device)
        for name in ("tables", "pair_tables"):
            for slot in framework[name]:
                framework[name][slot] = torch.arange(
                    1, 18, dtype=torch.int32, device=device
                )[None, :].contiguous()

        def begin(start, end, ready=False):
            cp = _metadata((end - start,), (start,), rank, device)
            return begin_cp_request(
                cp, 0, request_id="pair-memory", identity=identity, layout=layout,
                max_tokens=16384, restored_state_ready=ready, **framework,
            )

        published = begin(0, 15360)
        for owner in PAIR_OWNERS:
            published.publish_pair(
                owner, PairCarry.empty(owner, "pair-memory", identity, 15360)
            )
        # Native aligned D2H canonicalizes the entire pair region to zero.
        # Restore into a different physical page on each rank, as H2D does.
        for owner, pool in framework["pair_pools"].items():
            host = pool[14].cpu().clone().zero_()
            pool[30 - rank].copy_(host)
            framework["pair_tables"][owner][0, 14] = 30 - rank
        restored = begin(15360, 15361, True)
        for owner in PAIR_OWNERS:
            pair = restored.cache.owners[owner].pair
            assert pair.next_position == 15360
            assert pair.partial_kv is pair.partial_score is None
            page = int(framework["pair_tables"][owner][0, 14])
            assert not bool(framework["pair_pools"][owner][page].any())
            values = torch.arange(512, dtype=torch.float32, device=device) + owner
            restored.publish_pair(
                owner, PairCarry(owner, "pair-memory", identity, 15361, values, -values)
            )
        odd = begin(15361, 15362, True)
        for owner in PAIR_OWNERS:
            pair = odd.cache.owners[owner].pair
            expected = torch.arange(512, dtype=torch.float32, device=device) + owner
            assert pair.next_position == 15361
            _equal(pair.partial_kv, expected, "restored odd KV")
            _equal(pair.partial_score, -expected, "restored odd scores")
        rejected = []
        for label, start, ready, dirty_byte in (
            ("not_ready", 15360, False, None),
            ("unaligned_even", 15362, True, None),
            ("odd_missing_payload", 15361, True, None),
            ("other_snapshot_nonzero", 15360, True, 0),
            ("trailing_padding_nonzero", 15360, True, -1),
            ("stale_position", 15360, True, (speculative + 1) * 4112 + 4096),
            ("invalid_flag", 15360, True, (speculative + 1) * 4112 + 4104),
        ):
            for pool in framework["pair_pools"].values():
                pool.zero_()
            if dirty_byte is not None:
                pool = framework["pair_pools"][2]
                full_bytes = pool.shape[1] * 8
                byte = dirty_byte % full_bytes
                peer, offset = divmod(byte, pool.shape[1])
                if peer == rank:
                    page = int(framework["pair_tables"][2][0, (start - 1) // 1024])
                    pool[page, offset] = 1
            try:
                begin(start, start + 1, ready)
            except ValueError as error:
                assert "restored execution boundary" in str(error)
                rejected.append(label)
            else:
                raise AssertionError("malformed pair state accepted: " + label)
            torch.distributed.barrier()
        observations.append({"speculative_tokens": speculative, "restored_start":15360,
                             "odd_continuation":15361,"rejected":rejected})
    return observations


@torch.inference_mode()
def _run_rank():
    rank, device = int(os.environ["RANK"]), torch.device(
        "cuda", int(os.environ["LOCAL_RANK"])
    )
    if os.getuid() == 0 or not str(torch.version.cuda).startswith("13."):
        raise RuntimeError("CP integration must run as a non-root user with CUDA13")
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device)[0] != 10:
        raise RuntimeError("CP integration requires Blackwell")
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=15))
    if torch.distributed.get_world_size() != 8:
        raise RuntimeError("CP integration requires eight distinct CUDA ranks")
    # Use the real NCCL world as the framework's CP/TP group in this component fixture.
    collective_torch._group_map[Group.TP] = torch.distributed.group.WORLD
    collective_torch._group_map[Group.DP_AND_TP] = torch.distributed.group.WORLD
    collective_torch._parallelism_config = SimpleNamespace(
        tp_size=8, dp_size=1, world_size=8
    )
    collective_torch._initialized = True
    torch.backends.cuda.matmul.allow_tf32 = False
    pair_checkpoints = _pair_checkpoint_restore(rank, device)
    layout = CacheLayout(cp_size=8, speculative_tokens=0, draft_enabled=False)
    identity = ReplayConfig(ReplayMode.FULL).cache_identity(
        "cp8-attention-integration", layout
    )
    models = _models(layout, device)
    frameworks = [_framework_pages(layout, rank, device) for _ in range(2)]
    references = [
        V41AttentionCache.allocate_local(
            str(index), identity, layout, 2048, device=device
        )
        for index in range(2)
    ]
    starts, records = [0, 0], []
    for epoch, lengths in enumerate(((3, 1), (130, 18), (1000, 1028))):
        cp = _metadata(lengths, starts, rank, device)
        contexts = [
            begin_cp_request(
                cp,
                index,
                request_id=str(index),
                identity=identity,
                layout=layout,
                max_tokens=2048,
                epoch=epoch,
                **frameworks[index],
            )
            for index in range(2)
        ]
        local_contexts = [
            references[index].begin_forward(
                epoch=epoch, start=starts[index], end=starts[index] + lengths[index]
            )
            for index in range(2)
        ]
        for layer in _LAYERS:
            for index, context in enumerate(contexts):
                canonical = _hidden(context.start, context.end, index, device)
                local = canonical.index_select(
                    0, (context.positions - context.start).long()
                ).contiguous()
                local.masked_fill_(~context.valid[:, None], 0)
                expected = models[layer](canonical, local_contexts[index])
                actual = models[layer](local, context)
                selected = expected.index_select(
                    0, (context.positions - context.start).long()
                )
                selected.masked_fill_(~context.valid[:, None], 0)
                _equal(
                    actual,
                    selected,
                    f"rank {rank}, request {index}, epoch {epoch}, layer {layer}",
                )
                if layer_sources(layer).scores_queries:
                    ref_selection = local_contexts[index].selections[layer]
                    selected_ids = ref_selection.topk.index_select(
                        0, (context.positions - context.start).long()
                    )
                    selected_ids.masked_fill_(~context.valid[:, None], -1)
                    _equal(
                        context.selections[layer].topk, selected_ids, f"top-k {layer}"
                    )
        for index, context in enumerate(contexts):
            _compare_pages(layout, context, references[index])
            assert context.completed_layers == set(_LAYERS)
            assert context.max_gather_live_bytes <= 64 * 1024 * 1024
            assert context.gather_count > 0
            if epoch == 0 and rank >= 3:
                assert not bool(contexts[0].valid.any().item())
            records.append(
                dict(
                    epoch=epoch,
                    request=index,
                    start=context.start,
                    end=context.end,
                    local_rows=context.query_rows,
                    valid_rows=int(context.valid.sum().item()),
                    gathers=context.gather_count,
                    max_receive_bytes=context.max_receive_bytes,
                    max_gather_live_bytes=context.max_gather_live_bytes,
                )
            )
            starts[index] = context.end
        torch.distributed.barrier()
    torch.cuda.synchronize()
    result = dict(
        scope="real CP8 attention component comparison; no model acceptance",
        rank=rank,
        gpu=str(torch.cuda.get_device_properties(device).uuid),
        cases=records,
        pair_checkpoints=pair_checkpoints,
    )
    destination = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if destination:
        Path(destination, f"cp_attention_rank{rank}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    print(json.dumps(result), flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def main():
    for flag in _FLAGS:
        os.environ[flag] = "1"
    os.environ["DSV41_ATTENTION_BACKEND"] = "native"
    if "LOCAL_RANK" not in os.environ:
        if torch.cuda.device_count() != 8:
            raise RuntimeError("launch CP comparison with exactly eight visible GPUs")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=8",
                str(Path(__file__).resolve()),
            ],
            check=True,
        )
    else:
        _run_rank()


if __name__ == "__main__":
    main()

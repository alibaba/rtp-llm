"""CP8 CED scheduling, retained metadata, and real draft/shard publication.

Lightweight target blocks make row ownership and gate counts independently
observable. CP collectives, compact draft writers, and stage orchestration are
the product implementations; this does not replace full-model numerical tests.
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

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.modules.dsv41.cache_layout import (
    PAIR_OWNERS,
    CacheLayout,
    CacheRegion,
    RegionSlot,
    layer_sources,
)
from rtp_llm.models_py.modules.dsv41.ced import ReplayConfig, ReplayMode
from rtp_llm.models_py.modules.dsv41.compressor import PairCarry
from rtp_llm.models_py.modules.dsv41.cp import begin_cp_request
from rtp_llm.models_py.modules.dsv41.draft import V41PrefillDraftCommit
from rtp_llm.models_py.modules.dsv41.indexer import IndexSelection
from rtp_llm.models_py.modules.dsv41.inputs import V41ModelRows
from rtp_llm.models_py.modules.dsv41.prefill import V41CPPrefillExecutor
from rtp_llm.models_py.modules.dsv41.test.fixture import flash_config
from rtp_llm.models_py.modules.dsv41.test.test_cp_attention import (
    _FLAGS,
    _framework_pages,
    _metadata,
)
from rtp_llm.models_py.modules.dsv41.transformer import (
    V41ImageFeatures,
    V41TargetModel,
    V41TargetOutput,
)


_IMAGE_STARTS = (581, 588, 894, 1090)


def _tokens(positions, image_starts=_IMAGE_STARTS):
    return torch.where(
        _types(positions, image_starts) >= 0, 129264, positions % 127 + 1
    ).to(torch.int32)


def _types(positions, image_starts=_IMAGE_STARTS):
    kinds = torch.full_like(positions, -1, dtype=torch.int32)
    for first in image_starts:
        selected = (positions >= first) & (positions < first + 7)
        kinds[selected] = 1
        kinds[positions == first] = 0
        kinds[positions == first + 5] = 2
        kinds[positions == first + 6] = 3
    return kinds


def _rows(context, image_starts=_IMAGE_STARTS):
    positions = context.positions
    predecessors = positions[:, None] + torch.arange(-3, 0, device=positions.device)
    return V41ModelRows(
        _tokens(positions, image_starts).masked_fill(~context.valid, 0),
        _types(positions, image_starts).masked_fill(~context.valid, -1),
        context.valid.clone(),
        _tokens(predecessors, image_starts).masked_fill(predecessors < 0, 0),
        (
            (predecessors >= 0)
            & (_types(predecessors, image_starts) == -1)
            & context.valid[:, None]
        ).contiguous(),
    )


def _execute(executor, context, *, image_starts=_IMAGE_STARTS, **kwargs):
    rows = _rows(context, image_starts)
    selected = rows.image_mask.nonzero().flatten()
    features = None
    if selected.numel():
        values = torch.ones(
            (selected.numel(), 5120), dtype=torch.bfloat16, device=context.query_device
        )
        values[:, 0] = context.positions[selected].bfloat16()
        features = V41ImageFeatures(selected, rows.token_types[selected], values)
    return executor.run_extend(rows, context, image_features=features, **kwargs)


class _Projection(nn.Module):
    def __init__(self, inputs, outputs, device):
        super().__init__()
        self.in_features, self.out_features = inputs, outputs
        self.register_buffer(
            "weight", torch.ones(1, dtype=torch.bfloat16, device=device)
        )
        self.rows = []

    def forward(self, values):
        self.rows.append(values.shape[0])
        return values[:, : self.out_features].contiguous()


class _Engram(nn.Module):
    def forward(self, hidden, hashes, text_mask, *, lookup_output=None):
        return hidden


class _Block(nn.Module):
    def __init__(self, layer, records):
        super().__init__()
        self.layer, self.records = layer, records

    def forward(self, hidden, pre_mix, context, image_mask):
        context.validate()
        layer, source = self.layer, layer_sources(self.layer)
        self.records.append(
            (layer, context.start, context.end, int(context.valid.sum()))
        )
        if source.writes_global:
            owner = context.cache.owners[layer]
            assert owner.materialized_end == context.start
            owner.materialized_end = context.end
            if layer in PAIR_OWNERS:
                partial = torch.full(
                    (512,), context.end, dtype=torch.float32, device=hidden.device
                )
                owner.pair = PairCarry(
                    layer,
                    context.cache.request_id,
                    context.cache.identity,
                    context.end,
                    partial if context.end % 2 else None,
                    -partial if context.end % 2 else None,
                )
                context.publish_pair(layer, owner.pair)
            context.published_sources.add(layer)
        if source.scores_queries:
            topk = torch.full(
                (context.query_rows, 512), -1, dtype=torch.int32, device=hidden.device
            )
            topk[:, 0] = torch.where(
                context.valid, context.positions // source.ratio, -1
            )
            candidates = None
            if layer == 20:
                candidates = torch.full(
                    (context.query_rows, 2048),
                    -1,
                    dtype=torch.int32,
                    device=hidden.device,
                )
                candidates[:, 0] = torch.where(
                    context.valid, context.positions // 8, -1
                )
            context.publish_selection(
                IndexSelection(
                    topk,
                    candidates,
                    torch.zeros(
                        context.query_rows, dtype=torch.int32, device=hidden.device
                    ),
                    layer,
                    source.index_k_owner,
                    0,
                    0,
                    0,
                )
            )
        if layer == 21:
            selected = context.selection_for(21)
            assert torch.equal(
                selected.topk[:, 0], torch.where(context.valid, context.positions, -1)
            )
            assert torch.equal(
                selected.candidate_blocks[:, 0],
                torch.where(context.valid, context.positions // 8, -1),
            )
        initial = context.restore_swa(layer)
        encoded = (
            (context.positions % 251)
            .to(torch.uint8)[:, None]
            .expand(-1, 528)
            .contiguous()
        )
        encoded[:, 0] = layer
        context.publish_swa(layer, initial, encoded)
        context.completed_layers.add(layer)
        return hidden, pre_mix


class _Target(V41TargetModel):
    def __init__(self, device):
        nn.Module.__init__(self)
        self.hidden_size, self.hc_mult = 5120, 4
        self.aux_layer_ids = (37, 38, 39)
        self.register_buffer(
            "embedding", torch.ones((1, 5120), dtype=torch.bfloat16, device=device)
        )
        self.records = []
        self.blocks = nn.ModuleList(
            [_Block(layer, self.records) for layer in range(40)]
        )
        self.engrams = nn.ModuleDict({"1": _Engram(), "14": _Engram()})
        self.token_hasher = lambda ids, history, valid, text: torch.zeros(
            (ids.shape[0], 1, 2, 24), dtype=torch.int64, device=ids.device
        )

    def _embed_rows(self, rows, image_features):
        rows.validate()
        if image_features is None:
            assert not bool(rows.image_mask.any())
        else:
            image_features.validate(rows, self.hidden_size)
        hidden = self.embedding.new_ones((rows.token_ids.numel(), 4, 5120))
        hidden[:, :, 0] = rows.token_ids[:, None].to(torch.bfloat16)
        hidden.masked_fill_(~rows.valid[:, None, None], 0)
        return hidden, torch.ones(
            (hidden.shape[0], 4), dtype=torch.float32, device=hidden.device
        )

    def prefill_encoder(
        self, rows, context, *, image_features=None, lookup_outputs=None
    ):
        if image_features is not None:
            assert torch.equal(
                image_features.values[:, 0],
                context.positions[image_features.row_indices].bfloat16(),
            )
        return super().prefill_encoder(
            rows, context, image_features=image_features, lookup_outputs=lookup_outputs
        )

    def prefill_decoder(self, l20, context):
        expected = _rows(context)
        for name in ("token_ids", "token_types", "history_ids", "history_valid"):
            assert torch.equal(
                getattr(l20.rows, name)[context.valid],
                getattr(expected, name)[context.valid],
            )
        return super().prefill_decoder(l20, context)

    def _finish(self, hidden, pre_mix, aux, indices):
        return V41TargetOutput(
            hidden[:, 0].contiguous(),
            pre_mix,
            torch.cat(aux, dim=-1),
            indices,
            self.aux_layer_ids,
        )


def _executor(layout, identity, device, request, **kwargs):
    target = _Target(device)
    projections = [_Projection(15360, 5120, device)] + [
        _Projection(5120, 512, device) for _ in range(3)
    ]
    draft = V41PrefillDraftCommit(
        V41Config.from_dict(flash_config()),
        projections[0],
        torch.ones(5120, dtype=torch.bfloat16, device=device),
        projections[1:],
        [torch.ones(512, dtype=torch.bfloat16, device=device) for _ in range(3)],
    )
    return V41CPPrefillExecutor(
        target,
        request_id=request,
        identity=identity,
        layout=layout,
        draft_commit=draft,
        **kwargs,
    )


def _context(executor, pages, length, rank, device, epoch):
    start = executor.progress.encoder_materialized_end
    return begin_cp_request(
        _metadata((length,), (start,), rank, device),
        0,
        request_id=executor.request_id,
        identity=executor.identity,
        layout=executor.layout,
        max_tokens=2048,
        epoch=epoch,
        decoder_ready_end=executor.progress.decoder_checkpoint_end,
        **pages,
    )


@torch.inference_mode()
def _run_rank():
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    if os.getuid() == 0 or not str(torch.version.cuda).startswith("13."):
        raise RuntimeError("CP CED integration requires non-root CUDA13 execution")
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device)[0] != 10:
        raise RuntimeError("CP CED integration requires Blackwell")
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=15))
    if torch.distributed.get_world_size() != 8:
        raise RuntimeError("CP CED integration requires eight actual ranks")
    collective_torch._group_map[Group.TP] = torch.distributed.group.WORLD
    collective_torch._group_map[Group.DP_AND_TP] = torch.distributed.group.WORLD
    collective_torch._parallelism_config = SimpleNamespace(
        tp_size=8, dp_size=1, world_size=8
    )
    collective_torch._initialized = True
    layout = CacheLayout(cp_size=8, speculative_tokens=5, draft_enabled=True)
    identity = ReplayConfig(ReplayMode.BOUNDED).cache_identity(
        "cp-ced-scheduling-test", layout
    )
    pages = _framework_pages(layout, rank, device)
    executor = _executor(layout, identity, device, "1")
    protected = {}

    def protect(context, history):
        assert context.end == 1024
        assert all(context.cache.swa_ends[layer] == 1024 for layer in range(43))
        assert not any(start >= 1024 for _, start, _, _ in executor.target.records)
        protected["history"] = history
        for key in ("pools", "pair_pools"):
            tables = pages["tables" if key == "pools" else "pair_tables"]
            protected[key] = {
                slot: pool[tables[slot][0, 0]].clone()
                for slot, pool in pages[key].items()
            }
        return True

    for epoch, length in enumerate((1000, 165)):
        context = _context(executor, pages, length, rank, device, epoch)
        result = _execute(
            executor,
            context,
            protected_checkpoint_end=1024,
            final_handoff_end=1165,
            protect_checkpoint=protect,
        )
        if epoch == 0:
            assert result.output is None and result.progress.decoder_checkpoint_end == 0
            assert all(layer <= 20 for layer, *_ in executor.target.records)
        else:
            assert result.progress.decoder_checkpoint_end == 1165
            assert result.progress.protected_checkpoint_end == 1024
            assert (result.decoder_context.start, result.decoder_context.end) == (
                1037,
                1165,
            )
            wanted = context.valid & (context.positions >= 1037)
            assert torch.equal(
                result.hidden_states[wanted, 0],
                _tokens(context.positions[wanted]).bfloat16(),
            )
            assert result.aux_rows.positions == tuple(
                result.decoder_context.positions[result.decoder_context.valid].tolist()
            )
    assert [row["decoder_range"] for row in executor.observations] == [
        None,
        (896, 1024),
        (1037, 1165),
    ]
    total_late = torch.tensor(
        sum(count for layer, _, _, count in executor.target.records if layer == 39),
        device=device,
    )
    torch.distributed.all_reduce(total_late)
    assert int(total_late) == 256
    projected = torch.tensor(
        sum(executor.draft_commit.main_projection.rows), device=device
    )
    torch.distributed.all_reduce(projected)
    assert int(projected) == 256

    restored_pages = _framework_pages(layout, rank, device)
    for key in ("pools", "pair_pools"):
        tables = restored_pages["tables" if key == "pools" else "pair_tables"]
        for slot, pool in restored_pages[key].items():
            tables[slot][0, 0] += 9
            pool[tables[slot][0, 0]].copy_(protected[key][slot])
    resumed = _executor(
        layout,
        identity,
        device,
        "1",
        initial_encoder_end=1024,
        initial_decoder_end=1024,
        initial_protected_end=1024,
        history=protected["history"],
    )
    context = _context(resumed, restored_pages, 3, rank, device, 0)
    result = _execute(
        resumed, context, protected_checkpoint_end=1024, final_handoff_end=1027
    )
    assert (
        result.decoder_context.start == 1024
        and result.decoder_context.replay_floor == 896
    )
    assert all(
        int(result.decoder_context.cache.swa[layer].valid_starts[0]) == 896
        for layer in range(21, 43)
    )

    for request, lengths, final in (("2", (126, 3), 129), ("3", (3,), 3)):
        runner = _executor(layout, identity, device, request)
        backing = _framework_pages(layout, rank, device)
        for epoch, length in enumerate(lengths):
            context = _context(runner, backing, length, rank, device, epoch)
            result = _execute(
                runner,
                context,
                protected_checkpoint_end=0,
                final_handoff_end=final,
            )
        assert result.decoder_context.start == max(0, final - 128)
        assert result.progress.decoder_checkpoint_end == final
        if request == "3" and rank >= 3:
            assert result.aux_rows.positions == ()
            assert sum(runner.draft_commit.main_projection.rows) == 0
            assert {layer for layer, *_ in runner.target.records} == set(range(40))

    rejected = _executor(layout, identity, device, "4")
    context = _context(
        rejected, _framework_pages(layout, rank, device), 1027, rank, device, 0
    )
    try:
        _execute(
            rejected,
            context,
            protected_checkpoint_end=1024,
            final_handoff_end=1027,
            protect_checkpoint=lambda *_: False,
        )
        raise AssertionError(
            "failed checkpoint protection unexpectedly consumed the suffix"
        )
    except RuntimeError as error:
        assert "checkpoint copy failed" in str(error)
    assert rejected.poisoned and not any(
        start >= 1024 for _, start, _, _ in rejected.target.records
    )

    chunked = _executor(layout, identity, device, "5", max_tokens_per_rank=74)
    context = _context(
        chunked, _framework_pages(layout, rank, device), 1165, rank, device, 0
    )
    result = _execute(
        chunked,
        context,
        protected_checkpoint_end=1024,
        final_handoff_end=1165,
        protect_checkpoint=lambda *_: True,
    )
    assert result.progress.decoder_checkpoint_end == 1165
    assert all(
        record["encoder_local_rows"] <= 74 and record["decoder_local_rows"] <= 74
        for record in chunked.observations
    )
    assert (
        sum(record["decoder_range"] is not None for record in chunked.observations) == 2
    )
    assert len(chunked.observations) > 2
    assert chunked.observations[0]["encoder_range"][1] == 581
    assert all(
        not any(first < boundary < first + 7 for first in _IMAGE_STARTS)
        for record in chunked.observations
        for boundary in record["encoder_range"]
    )

    small = _executor(layout, identity, device, "6", max_tokens_per_rank=32)
    context = _context(
        small, _framework_pages(layout, rank, device), 257, rank, device, 0
    )
    result = _execute(small, context, protected_checkpoint_end=0, final_handoff_end=257)
    wanted = context.valid & (context.positions >= 129)
    assert torch.equal(
        result.hidden_states[wanted, 0], _tokens(context.positions[wanted]).bfloat16()
    )
    assert len(small.observations) == 2
    assert all(
        record["encoder_local_rows"] <= 32 and record["decoder_local_rows"] <= 32
        for record in small.observations
    )
    assert result.decoder_context.start < result.encoder_context.start

    checkpoint_tail = _executor(layout, identity, device, "14", max_tokens_per_rank=128)
    context = _context(
        checkpoint_tail, _framework_pages(layout, rank, device), 2000, rank, device, 0
    )
    result = _execute(
        checkpoint_tail,
        context,
        protected_checkpoint_end=1024,
        final_handoff_end=2000,
        protect_checkpoint=lambda *_: True,
    )
    assert [record["encoder_range"] for record in checkpoint_tail.observations] == [
        (0, 1003),
        (1003, 1024),
        (1024, 2000),
    ]
    wanted = context.valid & (
        ((context.positions >= 896) & (context.positions < 1024))
        | (context.positions >= 1872)
    )
    assert torch.equal(
        result.hidden_states[wanted, 0], _tokens(context.positions[wanted]).bfloat16()
    )
    assert not bool(result.hidden_states[~wanted].any())
    assert all(
        record["encoder_local_rows"] <= 128 and record["decoder_local_rows"] <= 128
        for record in checkpoint_tail.observations
    )

    full_identity = ReplayConfig(ReplayMode.FULL).cache_identity(
        "cp-ced-scheduling-test", layout
    )
    full = _executor(layout, full_identity, device, "7", max_tokens_per_rank=4)
    context = _context(
        full, _framework_pages(layout, rank, device), 65, rank, device, 0
    )
    result = _execute(full, context, protected_checkpoint_end=0, final_handoff_end=65)
    assert full.tail is None and len(full.observations) > 1
    assert all(
        record["encoder_range"] == record["decoder_range"]
        and record["encoder_local_rows"] <= 4
        and record["decoder_local_rows"] <= 4
        for record in full.observations
    )
    counts = torch.tensor(
        [
            sum(count for actual, _, _, count in full.target.records if actual == layer)
            for layer in range(40)
        ],
        device=device,
    )
    torch.distributed.all_reduce(counts)
    assert counts.tolist() == [65] * 40
    assert torch.equal(
        result.hidden_states[context.valid, 0],
        _tokens(context.positions[context.valid]).bfloat16(),
    )

    for request, length, budget, checkpoint, images, message in (
        ("8", 1165, 32, 1024, _IMAGE_STARTS, "bounded decoder requires"),
        ("9", 898, 128, 0, _IMAGE_STARTS, "ends inside a canonical image span"),
        ("10", 1040, 128, 1024, (1022,), "checkpoint cannot split"),
        ("11", 64, 2, 0, (1,), "budget cannot fit a complete image span"),
        ("12", 16, 128, 0, (-2,), "incomplete canonical image span"),
    ):
        invalid = _executor(
            layout, identity, device, request, max_tokens_per_rank=budget
        )
        context = _context(
            invalid, _framework_pages(layout, rank, device), length, rank, device, 0
        )
        progress = invalid.progress
        try:
            _execute(
                invalid,
                context,
                image_starts=images,
                protected_checkpoint_end=checkpoint,
                final_handoff_end=length,
                protect_checkpoint=lambda *_: True,
            )
            raise AssertionError(
                "invalid CP admission unexpectedly executed target rows"
            )
        except ValueError as error:
            assert message in str(error)
        assert not invalid.target.records and invalid.progress == progress
        assert invalid._input_epoch == invalid._execution_epoch == -1
        assert invalid._boundaries is None and not invalid.poisoned
        assert not context.cache.poisoned

    adjacent = _executor(layout, identity, device, "13", max_tokens_per_rank=7)
    context = _context(
        adjacent, _framework_pages(layout, rank, device), 256, rank, device, 0
    )
    assert adjacent._segments(context, _rows(context, (0, 7)), 0)[:3] == [0, 7, 14]
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "rank": rank,
                "status": "passed",
                "cases": [
                    "N split",
                    "bounded gate",
                    "restored short suffix",
                    "cross-chunk tail",
                    "empty valid ranks",
                    "failed copy blocks suffix",
                    "bounded per-rank chunks",
                    "cross-segment tail scatter",
                    "cross-segment checkpoint scatter",
                    "chunked full diagnostic",
                    "complete adjacent images",
                    "preflight rejects image/capacity conflicts",
                ],
            }
        ),
        flush=True,
    )
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def main():
    for flag in _FLAGS:
        os.environ[flag] = "1"
    if "LOCAL_RANK" not in os.environ:
        if torch.cuda.device_count() != 8:
            raise RuntimeError("CP CED test requires exactly eight visible GPUs")
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

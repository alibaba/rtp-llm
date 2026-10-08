"""Bounded CP4 stage qualification; run with torchrun --nproc_per_node=8."""

import argparse
import copy
import dataclasses
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--without-engine-bindings", action="store_true")
    parser.add_argument("--flashinfer-source-jit", action="store_true")
    parser.add_argument("--extra-package-path", action="append", default=[])
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    sys.path.extend(args.extra_package_path)
    if args.flashinfer_source_jit:
        sys.modules["flashinfer_jit_cache"] = None
    if args.without_engine_bindings:
        package = types.ModuleType("rtp_llm.models_py.modules")
        package.__path__ = [str(root / "rtp_llm/models_py/modules")]
        sys.modules[package.__name__] = package

    import torch
    import torch.distributed as dist
    from rtp_llm.config.dsv41_config import V41Config
    from rtp_llm.models.dsv41_kv_cache import dsv41_page_layouts
    from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_block_hardware import fixture
    from rtp_llm.models_py.modules.dsv4.v41_block import (
        V41Block,
        V41ResidualState,
        readout_v41_residual,
        start_v41_residual,
    )
    from rtp_llm.models_py.modules.dsv4.v41_cache import V41PagedCache
    from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import (
        V41CacheIdentity,
        V41L20Contract,
        V41L20ReplicaReceiver,
        V41L20Span,
    )
    from rtp_llm.models_py.modules.dsv4.v41_compressor import (
        V41PairMetadata,
        V41PairState,
    )
    from rtp_llm.models_py.modules.dsv4.v41_cp import V41CPExchange, V41CPPlan
    from rtp_llm.models_py.modules.dsv4.v41_engram_stage import V41EngramStage
    from rtp_llm.models_py.modules.dsv4.v41_projection import V41KVProjection, _rms_norm
    from rtp_llm.models_py.modules.dsv4.v41_stage import V41Stage, V41StageResult
    from rtp_llm.models_py.modules.dsv4.v41_stage_attention import (
        V41AttentionPlan,
        V41ForwardRows,
        V41StageAttention,
    )
    from rtp_llm.models_py.modules.dsv4.v41_stage_pp import (
        V41PPDispatch,
        export_v41_stage_result,
        run_v41_pp_stage,
    )

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", rank)
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("CP qualification requires SM120")
    dist.init_process_group(
        "nccl", timeout=datetime.timedelta(seconds=120), device_id=device
    )
    if dist.get_world_size() != 8:
        raise RuntimeError("CP qualification requires eight workers")
    groups = [dist.new_group(list(range(start, start + 4))) for start in (0, 4)]
    pp_rank, peer = rank // 4, rank % 4
    group = groups[pp_rank]
    engram = None
    try:
        plan = V41CPPlan(
            tuple(dist.get_process_group_ranks(group)), pp_rank, "cp-hardware-1", 2, 256
        )
        wrong = dataclasses.replace(
            plan, generation="mismatch" if peer == 1 else plan.generation
        )
        try:
            V41CPExchange(wrong, group, device).admit()
        except RuntimeError as error:
            assert "mismatch" in str(error)
        else:
            raise AssertionError("CP peer generation mismatch was admitted")
        peers = tuple(
            V41ForwardRows(
                "cp-hardware",
                torch.full((len(positions),), 19, dtype=torch.int64),
                torch.tensor(positions, dtype=torch.int64),
                torch.ones(len(positions), dtype=torch.bool),
            )
            for positions in ([1, 4], [3], [0, 2], [])
        )

        def begin():
            session = V41CPExchange(plan, group, device)
            session.admit()
            return session.begin(peers)

        forward = begin()
        full, local, part = forward.rows, forward.local_rows, forward.local_slice
        torch.manual_seed(419)
        prototypes = {ratio: fixture(ratio) for ratio in (1, 2)}
        data = json.loads(
            (root / "rtp_llm/config/test/testdata/deepseek_v41_flash.json").read_text()
        )
        data["text_config"].update(
            n_routed_experts=8, engram_layer_ids=[], engram_num_embeddings=[]
        )
        config = V41Config.from_dict(data)
        blocks = []
        begin_layer, end_layer = (0, 21) if pp_rank == 0 else (21, 40)
        for sources in config.layer_sources()[begin_layer:end_layer]:
            prototype = prototypes[2 if sources.compress_ratio == 2 else 1]
            attention = copy.copy(prototype.attention)
            attention.sources = sources
            if not sources.owns_index:
                attention.index_wq_b = attention.index_weights = None
            if sources.owns_kv:
                kv = copy.copy(prototype.kv)
                kv.sources = sources
            else:
                kv = V41KVProjection(
                    sources,
                    prototype.kv.wkv,
                    prototype.kv.kv_norm,
                    prototype.kv.frequencies,
                )
            ffn = copy.copy(prototype.feed_forward)
            ffn.layer_id = sources.layer_id
            blocks.append(
                V41Block(
                    attention,
                    kv,
                    prototype.attn_hc,
                    prototype.ffn_hc,
                    prototype.attn_norm,
                    prototype.ffn_norm,
                    ffn,
                )
            )
        engram = V41EngramStage(
            {}, None, device=device, max_host_bytes=0, max_weight_bytes=0
        ).initialize()
        driver = V41Stage(
            blocks,
            config,
            role="prefill",
            pp_rank=pp_rank,
            engram=engram,
            final_norm=(
                torch.ones(5120, dtype=torch.bfloat16, device=device)
                if pp_rank
                else None
            ),
            max_positions=256,
        )

        def cache(tag):
            return V41PagedCache(
                torch.zeros(
                    3,
                    dsv41_page_layouts(1)[tag].stride_bytes,
                    dtype=torch.uint8,
                    device=device,
                ),
                tag,
            )

        def plans_for(operators, *, memory=None, page=2):
            memory = {} if memory is None else memory
            plans = {}
            n = full.positions.numel()
            for op in operators:
                sources = op.attention.sources
                reads = torch.full((n, 8), -1, dtype=torch.int64, device=device)
                for row in range(n):
                    if full.valid_rows[row]:
                        length = int(full.positions[row]) + 1
                        reads[row, :length] = torch.arange(length, device=device)
                kwargs = dict(
                    swa=cache("swa_kv"),
                    swa_writes=full.positions.clone(),
                    swa_reads=reads,
                )
                if sources.compress_ratio:
                    source = sources.kv_source_layer_id
                    if source not in memory:
                        memory[source] = (
                            cache(
                                "csa_kv" if sources.compress_ratio == 2 else "hca_kv"
                            ),
                            cache("indexer_kv"),
                        )
                    global_cache, index_cache = memory[source]
                    kwargs.update(
                        kv_source_layer_id=source,
                        global_cache=global_cache,
                        index_cache=index_cache,
                        global_pages=torch.full(
                            (n, 1), page, dtype=torch.int32, device=device
                        ),
                        index_pages=torch.full(
                            (n, 1), (page + 1) % 3, dtype=torch.int32, device=device
                        ),
                    )
                    if sources.owns_kv:
                        ratio = sources.compress_ratio
                        kwargs.update(
                            global_writes=full.positions // ratio
                            + page * global_cache.layout.entries,
                            index_writes=full.positions // ratio
                            + ((page + 1) % 3) * index_cache.layout.entries,
                        )
                        if ratio == 2:
                            previous = torch.full((n,), -1, device=device)
                            for row in range(n):
                                if full.valid_rows[row] and full.positions[row] % 2:
                                    previous[row] = (
                                        (full.positions == full.positions[row] - 1)
                                        & full.valid_rows
                                    ).nonzero()[0, 0]
                            kwargs.update(
                                pair_state=V41PairState(
                                    torch.zeros(2, 4096, device=device)
                                ),
                                pair_metadata=V41PairMetadata(
                                    full.request_ids,
                                    full.positions,
                                    full.valid_rows,
                                    previous,
                                    torch.full((n,), -1, device=device),
                                    torch.where(
                                        full.valid_rows & (full.positions > 0),
                                        full.positions % 4,
                                        -1,
                                    ),
                                ),
                            )
                plans[op.layer_id] = V41AttentionPlan(**kwargs)
            return plans, memory

        torch.manual_seed(4191)
        initial = start_v41_residual(
            torch.randn(full.positions.numel(), 5120, device=device).bfloat16(),
            full.valid_rows,
        )
        seed, source_plan, wire, receiver, dispatch = None, None, None, None, None
        reference_memory = None
        if pp_rank:
            source = prototypes[1]
            source_plan = plans_for([source])[0][20]
            frame = source.prepare(
                dataclasses.replace(initial, next_layer_id=20),
                full.positions,
                full.valid_rows,
            )
            attended, seed = V41StageAttention(
                source.attention.sources, max_positions=256
            )(frame, source.attention.sinks, source_plan, full, None)
            initial = source.finish(frame, attended)
            reference_memory = {20: (source_plan.global_cache, source_plan.index_cache)}
        reference_plans, _ = plans_for(blocks, memory=reference_memory)
        unsplit = driver.run(initial, full, reference_plans, selections=seed)
        reference_plans, _ = plans_for(blocks, memory=reference_memory)
        slices = [
            slice(rank * 2, rank * 2 + rows.positions.numel())
            for rank, rows in enumerate(peers)
        ]
        local_maps = [
            V41ForwardRows(
                full.invocation_id,
                full.request_ids[part].contiguous(),
                full.positions[part].contiguous(),
                full.valid_rows[part].contiguous(),
            )
            for part in slices
        ]
        states = [
            V41ResidualState(
                initial.residual[part].contiguous(),
                initial.previous_pre_mix[part].contiguous(),
                begin_layer,
            )
            for part in slices
        ]
        selected = [
            (
                dataclasses.replace(
                    seed,
                    rows=rows,
                    result=dataclasses.replace(
                        seed.result,
                        **{
                            name: value[part].contiguous()
                            for name, value in vars(seed.result).items()
                        }
                    ),
                )
                if seed is not None
                else None
            )
            for rows, part in zip(local_maps, slices)
        ]
        # Independent reference: explicitly execute each rank's batch on this GPU
        # and concatenate its projected rows. No CP session or collective is used.
        for op, step in zip(blocks, driver.attentions):
            current = reference_plans[op.layer_id]
            projected = []
            for state, rows in zip(states, local_maps):
                collapsed, _ = op.attn_hc.pre(state.residual, state.previous_pre_mix)
                projected.append(
                    op.kv.project(
                        _rms_norm(collapsed, op.attn_norm),
                        rows.positions,
                        rows.valid_rows,
                    )
                )
            gathered = {}
            for name in ("swa", "values", "scores"):
                if getattr(projected[0], name) is None:
                    continue
                padded = []
                for item in projected:
                    value = getattr(item, name)
                    storage = torch.zeros(2, 512, dtype=value.dtype, device=device)
                    storage[: len(value)] = value
                    padded.append(storage)
                gathered[name] = torch.cat(padded)
            packet = dataclasses.replace(
                projected[0],
                **gathered,
                positions=full.positions,
                valid_rows=full.valid_rows
            )
            frames = [
                op.prepare(
                    state,
                    rows.positions,
                    rows.valid_rows,
                    pair_state=current.pair_state,
                    pair_metadata=current.pair_metadata,
                    kv_exchange=lambda _: packet,
                )
                for state, rows in zip(states, local_maps)
            ]
            for index, (frame, rows, part_for_rank) in enumerate(
                zip(frames, local_maps, slices)
            ):
                attended, selected[index] = step(
                    frame,
                    op.attention.sinks,
                    current,
                    rows,
                    selected[index],
                    read_rows=part_for_rank,
                )
                states[index] = op.finish(frame, attended)
        expected = V41StageResult(
            states[peer],
            local_maps[peer],
            selected[peer],
            (
                readout_v41_residual(states[peer], driver.final_norm, local.valid_rows)
                if pp_rank
                else None
            ),
        )
        local_state = V41ResidualState(
            initial.residual[part].contiguous(),
            initial.previous_pre_mix[part].contiguous(),
            begin_layer,
        )
        if not pp_rank:
            plans, _ = plans_for(blocks, page=peer % 3)
            actual = forward.run(driver, local_state, local, plans)
        else:
            identity = V41CacheIdentity(19, 1, 7)
            span = V41L20Span(identity, 0, 0, 5)
            contract = V41L20Contract("a" * 64, "b" * 64, "c" * 64, 256)
            dispatch = V41PPDispatch(
                local.invocation_id,
                peer,
                0,
                (span,),
                torch.tensor(
                    [[0, p, 1] for p in local.positions.cpu().tolist()],
                    dtype=torch.int64,
                ).reshape(-1, 3),
            )
            local_seed = dataclasses.replace(
                seed,
                rows=local,
                result=dataclasses.replace(
                    seed.result,
                    **{
                        name: value[part].contiguous()
                        for name, value in vars(seed.result).items()
                    }
                ),
            )
            result = V41StageResult(
                local_state, local, local_seed, None, forward.local_plan(source_plan)
            )
            wire = export_v41_stage_result(
                result,
                dispatch,
                contract,
                {identity: {"hca_kv": [2], "indexer_kv": [0]}},
            )
            target = {"hca_kv": cache("hca_kv"), "indexer_kv": cache("indexer_kv")}
            receiver = V41L20ReplicaReceiver(contract, peer, target)
            receiver.begin_request(
                identity, {"hca_kv": [peer % 3], "indexer_kv": [(peer + 1) % 3]}
            )
            plans, _ = plans_for(
                blocks,
                memory={20: (target["hca_kv"], target["indexer_kv"])},
                page=peer % 3,
            )
            bad = dict(wire)
            if peer == 1:
                bad["v41_delta_header"] = wire["v41_delta_header"].clone()
                bad["v41_delta_header"][3] += 1
            try:
                run_v41_pp_stage(
                    driver, bad, receiver, dispatch, plans, cp_forward=forward
                )
            except RuntimeError as error:
                assert "mismatch" in str(error)
            else:
                raise AssertionError("Bad peer PP delta was imported")
            assert all(bool((c.raw == 0).all()) for c in target.values())
            forward = begin()
            actual = run_v41_pp_stage(
                driver, wire, receiver, dispatch, plans, cp_forward=forward
            )
            receiver.require_ready(identity, 5, 0)
        torch.testing.assert_close(
            actual.state.residual, expected.state.residual, rtol=0, atol=0
        )
        torch.testing.assert_close(
            actual.state.previous_pre_mix,
            expected.state.previous_pre_mix,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            actual.selections.result.topk,
            expected.selections.result.topk,
            rtol=0,
            atol=0,
        )
        if pp_rank:
            torch.testing.assert_close(actual.hidden, expected.hidden, rtol=0, atol=0)
        torch.cuda.synchronize()
        reports = [None] * 8
        dist.all_gather_object(
            reports,
            {
                "rank": rank,
                "pp_rank": pp_rank,
                "local_rows": local.positions.numel(),
                "layers": len(blocks),
                "exact_stage_parity": True,
                "reference": "explicit per-rank batches with local concatenation",
                "unsplit_residual_max_abs": (
                    (
                        actual.state.residual.float()
                        - unsplit.state.residual[part].float()
                    )
                    .abs()
                    .max()
                    .item()
                    if actual.state.residual.numel()
                    else 0
                ),
                "pp_import": bool(pp_rank),
                "peer_rejection": True,
            },
        )
        if rank == 0:
            print(json.dumps(reports), flush=True)
            if args.manifest:
                args.manifest.parent.mkdir(parents=True, exist_ok=True)
                args.manifest.write_text(
                    json.dumps(
                        {
                            "torch": torch.__version__,
                            "nccl": torch.cuda.nccl.version(),
                            "engine_bindings_bypassed": args.without_engine_bindings,
                            "fixture": "D5120/I2304/E8; synthetic shared weights; Engram disabled",
                            "reports": reports,
                            "source_sha256": {
                                str(p.relative_to(root)): hashlib.sha256(
                                    p.read_bytes()
                                ).hexdigest()
                                for p in (root / "rtp_llm/models_py/modules/dsv4").glob(
                                    "v41_*.py"
                                )
                            },
                        },
                        indent=2,
                    )
                    + "\n"
                )
    finally:
        if engram is not None:
            engram.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

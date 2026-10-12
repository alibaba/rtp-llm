"""Actual PPExecutor messaging with live stage-local native allocation leases.

Frames and bytes are independent fixtures; maintained owner/export execution is
qualified separately. This does not open worker PP forwards or run a scheduler.
"""

import gc
import json
import os
import socket
import sys
import traceback
import unittest
import weakref
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.cpp.normal_engine.test import (
    libth_v41_native_pp_transport_test as native_ops,
)
from rtp_llm.models.dsv41_kv_cache import configure_dsv41_kv_cache, dsv41_page_layouts
from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.distributed.rank_layout import Group
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_cp_owner_sm120 import zigzag
from rtp_llm.models_py.modules.dsv4.native_l20_transfer import (
    NativeL20ReplicaReceiver,
    export_native_l20_delta,
    native_l20_cache_views,
)
from rtp_llm.models_py.modules.dsv4.test.test_native_l20_transfer import (
    CONTRACT,
    TAGS,
    TEXT,
    WIRE_ROWS,
    frame_for,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_cp_stage_worker_sm120 import (
    parallelism_for,
)
from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import (
    V41CacheIdentity,
    V41L20Span,
)
from rtp_llm.models_py.modules.dsv4.v41_pp import (
    V41PPFrame,
    import_v41_pp_frame,
    pack_v41_pp_frame,
)
from rtp_llm.ops import NcclCommConfig

REQUESTS = (1301, 1302)
ROUNDS = ([1, 1], [6, 4], [122, 124], [126, 126], [1, 1], [1, 1])
FRAME_KEYS = (
    "hidden_states",
    "v41_prev_pre_mix",
    "v41_row_map",
    "v41_l20_topk",
    "v41_candidate_blocks",
    "v41_candidate_counts",
)


def cache_config():
    config = ModelConfig()
    config.model_type = "deepseek_v41"
    config.num_layers = 40
    config.hidden_size = 5120
    config.vocab_size = config.input_vocab_size = 128
    config.max_seq_len = 512
    config.hc_mult = 4
    config.gen_num_per_cycle = 0
    config.deepseek_v41_config = TEXT
    config.attn_config.tokens_per_block = 256
    config.attn_config.kernel_tokens_per_block = 256
    configure_dsv41_kv_cache(config)
    return config


def snapshot(cache):
    return {
        (layer, group.tag): group.kv_cache_base.cpu().clone()
        for layer in range(cache.layer_count)
        for group in cache.get_layer_cache_groups(layer)
    }


def require_snapshot(cache, expected):
    for key, value in snapshot(cache).items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def pattern(page, offsets, tag, lane):
    return (
        (page * 17 + offsets * 7 + TAGS.index(tag) * 31 + lane * 13)
        .remainder(256)
        .to(torch.uint8)
    )


def expected_delta(spans, tables, tag, lane):
    layout = dsv41_page_layouts(0)[tag]
    rows = []
    for span in spans:
        for position in range(span.start, span.start + span.count):
            page = tables[span.identity][tag][position // 256]
            payload, scale = layout.entry_offsets(position % 256)
            offsets = torch.cat(
                (
                    torch.arange(payload, payload + layout.payload_bytes_per_entry),
                    torch.arange(scale, scale + layout.scale_bytes_per_entry),
                )
            )
            rows.append(pattern(page, offsets, tag, lane))
    return torch.stack(rows)


def update_expected(expected, spans, tables, deltas):
    for tag, delta in zip(TAGS, deltas):
        layout = dsv41_page_layouts(0)[tag]
        raw = expected[0, tag]
        row = 0
        for span in spans:
            for position in range(span.start, span.start + span.count):
                page = tables[span.identity][tag][position // 256]
                payload, scale = layout.entry_offsets(position % 256)
                width = layout.payload_bytes_per_entry
                raw[page, payload : payload + width] = delta[row, :width]
                raw[page, scale : scale + layout.scale_bytes_per_entry] = delta[
                    row, width:
                ]
                row += 1


def frame_for_round(rows, step, lane, device):
    torch.manual_seed(41850 + step * 10 + lane)
    frame = frame_for(rows)
    return V41PPFrame(
        *(
            value if name == "row_map" else value.to(device)
            for name, value in frame.__dict__.items()
        )
    )


def run_rank(rank, port):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    os.environ["NCCL_P2P_DISABLE"] = "1"
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=180),
    )
    counts = {"cpu": 0, "cuda": 0, "snapshots": 0}
    registered = native_ops.register_pp_ops

    def register_checked(send, receive, exchange):
        def checked(fn):
            def call(tensor, peer, backend):
                assert peer == (rank + 4 if rank < 4 else rank - 4)
                assert backend == (
                    collective._CPP_P2P_BACKEND_NCCL
                    if tensor.is_cuda
                    else collective._CPP_P2P_BACKEND_GLOO
                )
                counts["cuda" if tensor.is_cuda else "cpu"] += 1
                return fn(tensor, peer, backend)

            return call

        def snapshots(value):
            counts["snapshots"] += 1
            result = exchange(value)
            assert len(result) == 2
            return result

        registered(checked(send), checked(receive), snapshots)

    native = None
    new_group = dist.new_group

    def bounded_group(*args, **kwargs):
        # Production groups intentionally have an effectively infinite timeout.
        # Test failures must surface instead of hiding in collective teardown.
        kwargs["timeout"] = timedelta(seconds=180)
        return new_group(*args, **kwargs)

    with patch.dict(sys.modules, {"librtp_compute_ops": native_ops}), patch.object(
        native_ops, "register_pp_ops", side_effect=register_checked
    ), patch.object(dist, "new_group", side_effect=bounded_group):
        try:
            pcfg = parallelism_for(rank)
            collective.init_distributed_environment(
                pcfg, NcclCommConfig(nccl_ip="127.0.0.1"), port
            )
            assert dist.get_process_group_ranks(collective._get_group(Group.PP)) == [
                rank % 4,
                rank % 4 + 4,
            ]
            with torch.inference_mode():
                native = native_ops.NativePPTransport(cache_config(), pcfg)
                assert native.capacity == 12 and counts["snapshots"] == 1
                free_before = native.free_blocks
                cache = native.cache
                assert cache.layer_count == (21 if rank < 4 else 19)
                allocations = {
                    request: native.allocate(request, 257)
                    for request in (REQUESTS if rank < 4 else REQUESTS[::-1])
                }
                local = {
                    request: {
                        "identity": allocation.identity,
                        "tables": {
                            tag: tuple(allocation.block_tables[tag]) for tag in TAGS
                        },
                    }
                    for request, allocation in allocations.items()
                }
                peers = [None] * 8
                dist.all_gather_object(peers, local)
                source = peers[rank % 4]
                target = peers[rank % 4 + 4]
                identities = [
                    V41CacheIdentity(*source[request]["identity"])
                    for request in REQUESTS
                ]
                source_tables = {
                    identity: source[request]["tables"]
                    for identity, request in zip(identities, REQUESTS)
                }
                target_tables = {
                    identity: target[request]["tables"]
                    for identity, request in zip(identities, REQUESTS)
                }
                for request, identity in zip(REQUESTS, identities):
                    assert target[request]["identity"] != list(
                        identity.__dict__.values()
                    )
                    assert source[request]["tables"] != target[request]["tables"]
                views = native_l20_cache_views(TEXT, cache, rank // 4)
                for layer in range(cache.layer_count):
                    for group in cache.get_layer_cache_groups(layer):
                        group.kv_cache_base.zero_()
                if rank < 4:
                    for tag, view in views.items():
                        offsets = torch.arange(view.raw.shape[1], device="cuda")
                        for pages in source_tables.values():
                            for page in pages[tag]:
                                view.raw[page].copy_(
                                    pattern(page, offsets, tag, rank % 4)
                                )
                expected_cache = snapshot(cache)
                receiver = None
                if rank >= 4:
                    receiver = NativeL20ReplicaReceiver(TEXT, cache, CONTRACT, rank % 4)
                    for identity in identities:
                        receiver.begin_request(identity, target_tables[identity])
                prefixes = [0, 0]
                for step, lengths in enumerate(ROUNDS):
                    cp, _, _ = zigzag(
                        lengths, prefixes, rank % 4, torch.device("cuda", rank)
                    )
                    rows = torch.stack(
                        (
                            cp.req_id_per_token.long(),
                            cp.global_positions,
                            cp.local_is_real.long(),
                        ),
                        -1,
                    ).cpu()
                    rows[~rows[:, 2].bool()] = torch.tensor([-1, -1, 0])
                    spans = [
                        V41L20Span(identity, step, start, count)
                        for identity, start, count in zip(identities, prefixes, lengths)
                    ]
                    expected_frame = frame_for_round(
                        rows, step, rank % 4, torch.device("cuda", rank)
                    )
                    deltas = [
                        expected_delta(spans, source_tables, tag, rank % 4)
                        for tag in TAGS
                    ]
                    owners = [allocations[request] for request in REQUESTS]
                    wrapper_retained = None
                    if rank < 4:
                        outgoing = pack_v41_pp_frame(
                            export_native_l20_delta(
                                TEXT,
                                cache,
                                CONTRACT,
                                rank % 4,
                                step,
                                spans,
                                source_tables,
                            ),
                            frame_for_round(
                                rows, step, rank % 4, torch.device("cuda", rank)
                            ),
                            spans,
                            rows,
                        )
                        weak = weakref.ref(outgoing["hidden_states"])
                        native.send(outgoing, owners)
                        del outgoing
                        gc.collect()
                        # TensorImpl can retain its Python wrapper while the
                        # native ticket owns it. Inspect that behavior as evidence;
                        # exact receive bytes establish the lifetime contract.
                        wrapper_retained = weak() is not None
                        churn = torch.empty_like(expected_frame.hidden_states)
                        churn.fill_(-13)
                    else:
                        incoming = native.receive(owners)
                    assert native.pending_tickets > 0 and native.retained_leases == 2
                    try:
                        native.finish()
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("round released with undrained PP tickets")
                    final = step == len(ROUNDS) - 1
                    if final:
                        in_use = native.free_blocks
                        for allocation in owners:
                            allocation.release_owner()
                            allocation.release_lease()
                        assert native.free_blocks == in_use
                        probe = native.allocate(9999, 257)
                        for tag in TAGS:
                            occupied = {
                                page
                                for request in local.values()
                                for page in request["tables"][tag]
                            }
                            assert not occupied.intersection(probe.block_tables[tag])
                        probe.release_owner()
                        probe.release_lease()
                        assert native.free_blocks == in_use
                    native.wait()
                    assert native.pending_tickets == 0 and native.retained_leases == 2
                    if rank >= 4:
                        for key, value in zip(
                            FRAME_KEYS, expected_frame.__dict__.values()
                        ):
                            torch.testing.assert_close(
                                incoming[key], value, rtol=0, atol=0
                            )
                        for key, value in zip(WIRE_ROWS, deltas):
                            torch.testing.assert_close(
                                incoming[key].cpu(), value, rtol=0, atol=0
                            )
                        assert incoming["v41_pp_header"].tolist() == [
                            1,
                            len(rows),
                            4,
                            5120,
                        ]
                        assert incoming["v41_delta_header"].tolist() == [
                            1,
                            20,
                            rank % 4,
                            step,
                            2,
                            sum(lengths),
                        ]
                        assert incoming["v41_delta_spans"].tolist() == [
                            list(span.wire_values()) for span in spans
                        ]
                        assert (
                            bytes(incoming["v41_cache_fingerprint"].tolist())
                            == CONTRACT.fingerprint
                        )
                        before = snapshot(cache)
                        bad = dict(incoming)
                        del bad["v41_prev_pre_mix"]
                        try:
                            import_v41_pp_frame(bad, receiver, spans, rows)
                        except ValueError:
                            pass
                        else:
                            raise AssertionError("partial transmitted frame admitted")
                        require_snapshot(cache, before)
                        try:
                            import_v41_pp_frame(incoming, receiver, spans[::-1], rows)
                        except ValueError:
                            pass
                        else:
                            raise AssertionError("reordered dispatch admitted")
                        require_snapshot(cache, before)
                        import_v41_pp_frame(incoming, receiver, spans, rows)
                        update_expected(expected_cache, spans, target_tables, deltas)
                        for span in spans:
                            receiver.require_ready(
                                span.identity, span.start + span.count, step
                            )
                    require_snapshot(cache, expected_cache)
                    native.finish()
                    assert native.retained_leases == 0
                    if final:
                        assert native.free_blocks == free_before
                    prefixes = [p + n for p, n in zip(prefixes, lengths)]
                    print(
                        "DSV41_NATIVE_PP_TRANSPORT "
                        + json.dumps(
                            {
                                "rank": rank,
                                "step": step,
                                "ends": prefixes,
                                "real_cp_rows": int(rows[:, 2].sum()),
                                "transport_exact": True,
                                "native_bytes_exact": True,
                                "relocated_local_allocations": True,
                                "leases_through_import": True,
                                "release_while_pending": final,
                                "wrapper_retained_while_pending": wrapper_retained,
                            }
                        ),
                        flush=True,
                    )
                for allocation in owners:
                    try:
                        if rank < 4:
                            native.send({"control": torch.ones(1)}, [allocation])
                        else:
                            native.receive([allocation])
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("released allocation admitted")
                    assert native.pending_tickets == native.retained_leases == 0
                if receiver is not None:
                    for identity in identities:
                        receiver.cancel(identity)
                        receiver.release(identity)
                fresh = native.allocate(REQUESTS[0], 1)
                assert fresh.identity[1] > local[REQUESTS[0]]["identity"][1]
                if receiver is not None:
                    new = V41CacheIdentity(
                        REQUESTS[0],
                        identities[-1].generation + 1,
                        identities[0].cache_incarnation + 1,
                    )
                    receiver.begin_request(
                        new, {tag: fresh.block_tables[tag] for tag in TAGS}
                    )
                    fresh_snapshot = snapshot(cache)
                    try:
                        receiver.validate(incoming, spans)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError(
                            "retired wire identity admitted after native reuse"
                        )
                    require_snapshot(cache, fresh_snapshot)
                    receiver.cancel(new)
                    receiver.release(new)
                fresh.release_owner()
                fresh.release_lease()
                assert native.free_blocks == free_before
                assert counts["cpu"] > 0 and counts["cuda"] > 0
                print(
                    "DSV41_NATIVE_PP_LIFETIME "
                    + json.dumps(
                        {
                            "rank": rank,
                            "counts": counts,
                            "capacity": native.capacity,
                            "free_before": free_before,
                            "free_after": native.free_blocks,
                            "pending_release_fenced": True,
                            "released_lease_rejected": True,
                            "fresh_generation": True,
                        }
                    ),
                    flush=True,
                )
                native.close()
                native = None
            dist.barrier()
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            if native is not None:
                native.wait()
                native.close()
            collective.destroy_distributed_environment()


class V41NativePPTransportSM120Test(unittest.TestCase):
    def test_native_pp_lanes_and_live_allocations(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 8)
        for rank in range(8):
            self.assertEqual(torch.cuda.get_device_capability(rank), (12, 0))
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        mp.spawn(run_rank, args=(port,), nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main()

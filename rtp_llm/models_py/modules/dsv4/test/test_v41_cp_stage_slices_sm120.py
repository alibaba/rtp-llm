"""Padded CP frames through the maintained 40-layer and 21/19 controls."""

import json
import os
import socket
import unittest
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    _get_group,
    destroy_distributed_environment,
    init_distributed_environment,
)
from rtp_llm.models_py.modules.dsv4.fp8.attention import bind_attn_cache
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_attention_owner_sm120 import (
    NativeTaggedCache,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_cp_owner_sm120 import (
    restore,
    zigzag,
)
from rtp_llm.models_py.modules.dsv4.hc.pp_state import imported_delayed_pre_mix
from rtp_llm.models_py.modules.dsv4.native_l20_prefill import (
    imported_native_l20_prefill,
    prepare_native_l20_prefill_source,
)
from rtp_llm.models_py.modules.dsv4.native_l20_transfer import (
    NativeL20ReplicaReceiver,
    export_native_l20_delta,
    native_l20_cache_views,
)
from rtp_llm.models_py.modules.dsv4.prefill.forward import forward_layers
from rtp_llm.models_py.modules.dsv4.test.test_native_l20_transfer import (
    CONTRACT,
    TEXT,
    native_fixture,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_stage_slices_sm120 import (
    assert_cleared,
    capture_frame,
    model_for,
    project_control,
    tables,
    wide_fixture,
)
from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import (
    V41CacheIdentity,
    V41L20Span,
)
from rtp_llm.models_py.modules.dsv4.v41_cache import logical_to_physical_slots
from rtp_llm.models_py.modules.dsv4.v41_cp_router import canonical_v41_cp_routes
from rtp_llm.models_py.modules.dsv4.v41_pp import import_v41_pp_frame, pack_v41_pp_frame
from rtp_llm.ops import NcclCommConfig, ParallelismConfig


@contextmanager
def trace_boundaries(model, snapshots, rank, step, cp=None, group=None):
    """Opt-in first-divergence evidence; never replace an acceptance check."""
    if os.environ.get("DSV41_TEST_CP_STAGE_TRACE") != "1":
        yield
        return
    reported = False

    def record(key, value):
        nonlocal reported
        if cp is None:
            snapshots[key] = value.clone()
        elif not reported:
            actual = restore(value, cp, group)
            expected = snapshots[key]
            if not torch.equal(actual, expected):
                difference = (actual.float() - expected.float()).abs()
                print(
                    "DSV41_CP_STAGE_FIRST_DIFFERENCE "
                    + json.dumps(
                        {
                            "rank": rank,
                            "step": step,
                            "boundary": key,
                            "shape": list(actual.shape),
                            "mismatches": int((actual != expected).sum()),
                            "max_abs": float(difference.max()),
                        }
                    ),
                    flush=True,
                )
                reported = True

    with ExitStack() as stack:
        for block in model.layers:
            for label, module in (
                ("gate", block.ffn.gate),
                ("shared", block.ffn.shared_experts),
                ("routed", block.ffn._strategy),
            ):
                original = module.forward
                name = f"L{block.layer_id}.ffn.{label}"

                def forward(x, *args, _fn=original, _name=name, **kwargs):
                    record(_name + ".input", x)
                    result = _fn(x, *args, **kwargs)
                    values = result if isinstance(result, tuple) else (result,)
                    for index, value in enumerate(values):
                        record(_name + f".output{index}", value)
                    return result

                stack.enter_context(
                    patch.object(module, "forward", side_effect=forward)
                )
            for part in ("attn", "ffn"):
                unit = getattr(block, part + "_hc")
                name = f"L{block.layer_id}.{part}"
                original_pre, original_post = unit._pre_impl, unit._post_impl

                def pre(x, *args, _fn=original_pre, _unit=unit, _name=name, **kwargs):
                    record(_name + ".residual_in", x)
                    result = _fn(x, *args, **kwargs)
                    for label, value in zip(("readout", "post", "comb"), result):
                        record(_name + "." + label, value)
                    record(_name + ".pre_mix", _unit.pre_mix_out)
                    return result

                def post(x, residual, mix, comb, _fn=original_post, _name=name):
                    record(_name + ".sublayer_out", x)
                    result = _fn(x, residual, mix, comb)
                    record(_name + ".residual_out", result)
                    return result

                stack.enter_context(patch.object(unit, "_pre_impl", side_effect=pre))
                stack.enter_context(patch.object(unit, "_post_impl", side_effect=post))
        yield


def call(model, cache, ids, positions, lengths, prefixes, pages, cp=None, hidden=None):
    model.set_cp_info(
        None if cp is None else cp.cp_info,
        1 if cp is None else 4,
        0 if cp is None else cp.cp_rank,
    )
    local_lengths = lengths if cp is None else cp.chunk_lengths_per_req
    cu = torch.tensor(
        [0, local_lengths[0], sum(local_lengths)], device="cuda", dtype=torch.int32
    )
    inputs = SimpleNamespace(
        is_prefill=True,
        cache_store_inputs=None,
        input_lengths=torch.tensor(local_lengths, device="cuda", dtype=torch.int32),
        prefix_lengths=torch.tensor(prefixes, device="cuda", dtype=torch.int32),
    )
    try:
        return forward_layers(
            model,
            cache,
            ids,
            positions,
            cu,
            pages,
            inputs,
            prepare_hidden_fn=None if hidden is None else lambda **kwargs: hidden,
        )
    finally:
        model._propagate_cp_ctx(None)
        model.set_cp_info(None, 1, 0)


def run_controls(rank, group):
    torch.manual_seed(41600 + (rank // 4) * 100)
    args, weights = wide_fixture()
    cp1, whole = model_for(args, weights), model_for(args, weights)
    first, _ = project_control(model_for(args, weights), 0)
    second, shared = project_control(model_for(args, weights), 1)
    oracle_cache = NativeTaggedCache([l.attn for l in cp1.layers], gen=0, pages=6)
    whole_cache = NativeTaggedCache([l.attn for l in whole.layers], gen=0, pages=6)
    first_cache = NativeTaggedCache([l.attn for l in first.layers], gen=0, pages=6)
    first_cache.layer_count = 21
    replica, backing = native_fixture(1, "cuda")
    second_cache = NativeTaggedCache([l.attn for l in second.layers], gen=0, pages=6)
    replica.layers.update(second_cache.groups)
    replica.layouts = second_cache.layouts
    pages = tables(whole_cache)
    destination = tables(replica, (3, 4), (4, 3))
    identities = (V41CacheIdentity(1301, 1, 0), V41CacheIdentity(1302, 2, 1))
    source_pages = {
        identities[0]: {"hca_kv": (1,), "indexer_kv": (2,)},
        identities[1]: {"hca_kv": (2,), "indexer_kv": (1,)},
    }
    target_pages = {
        identities[0]: {"hca_kv": (3,), "indexer_kv": (4,)},
        identities[1]: {"hca_kv": (4,), "indexer_kv": (3,)},
    }
    receiver = NativeL20ReplicaReceiver(TEXT, replica, CONTRACT, rank % 4)
    for identity in identities:
        receiver.begin_request(identity, target_pages[identity])
    prefixes = [0, 0]
    for step, lengths in enumerate(([1, 1], [6, 4], [122, 124])):
        cp, indices, _ = zigzag(lengths, prefixes, rank % 4, torch.device("cuda", rank))
        ids = (
            torch.arange(sum(lengths), device="cuda") + step * 17 + (rank // 4) * 23
        ).remainder(args.vocab_size)
        positions = torch.cat(
            [torch.arange(p, p + n, device="cuda") for p, n in zip(prefixes, lengths)]
        )
        local_ids = ids.index_select(0, indices)
        local_ids[~cp.local_is_real] = args.vocab_size - 1
        if step == 0:
            x = torch.zeros(
                (cp.chunk_length, args.dim), device="cuda", dtype=torch.bfloat16
            )
            for bad in (
                replace(cp, cp_size=2),
                replace(cp, kv_cache_sharded=True),
                replace(cp, cp_rank=(cp.cp_rank + 1) % 4),
                replace(cp, input_lengths_global_host=(513, 1)),
                replace(cp, unpad_restore=None),
            ):
                try:
                    canonical_v41_cp_routes(whole.layers[0].ffn.gate, x, bad)
                except (ValueError, NotImplementedError):
                    pass
                else:
                    raise AssertionError("invalid CP router scope was admitted")
        rows = torch.stack(
            (cp.req_id_per_token.long(), cp.global_positions, cp.local_is_real.long()),
            -1,
        ).cpu()
        rows[~rows[:, 2].bool()] = torch.tensor([-1, -1, 0])
        snapshots = {}
        with trace_boundaries(cp1, snapshots, rank, step):
            expected_cp1 = call(
                cp1, oracle_cache, ids, positions, lengths, prefixes, pages
            )
        captures = []
        expected_selections = {}

        def snapshot(block, inputs, hidden):
            frame = capture_frame(block, rows, hidden)
            frame.topk[~cp.local_is_real] = -1
            captures.append(frame)

        def select_expected(block, inputs, hidden):
            expected_selections[block.layer_id] = block.attn._shared_attention["topk"][
                block.layer_id
            ].clone()

        hook = whole.layers[20].register_forward_hook(snapshot)
        selection_hooks = [
            block.register_forward_hook(select_expected)
            for block in whole.layers[24:]
            if block.attn.is_index_source
        ]
        try:
            with trace_boundaries(whole, snapshots, rank, step, cp, group):
                expected = call(
                    whole,
                    whole_cache,
                    local_ids,
                    cp.global_positions,
                    lengths,
                    prefixes,
                    pages,
                    cp,
                )
        finally:
            hook.remove()
            for handle in selection_hooks:
                handle.remove()
        torch.testing.assert_close(
            restore(expected, cp, group), expected_cp1, rtol=0, atol=0
        )
        whole_frame = captures.pop()
        hook = first.layers[-1].register_forward_hook(snapshot)
        try:
            hidden = call(
                first,
                first_cache,
                local_ids,
                cp.global_positions,
                lengths,
                prefixes,
                pages,
                cp,
            )
        finally:
            hook.remove()
        frame = captures.pop()
        torch.testing.assert_close(hidden, frame.hidden_states, rtol=0, atol=0)
        for key, value in whole_frame.__dict__.items():
            torch.testing.assert_close(frame.__dict__[key], value, rtol=0, atol=0)
        spans = [
            V41L20Span(identity, step, start, count)
            for identity, start, count in zip(identities, prefixes, lengths)
        ]
        delta = export_native_l20_delta(
            TEXT, first_cache, CONTRACT, rank % 4, step, spans, source_pages
        )
        frame = import_v41_pp_frame(
            pack_v41_pp_frame(delta, frame, spans, rows), receiver, spans, rows
        )
        source = prepare_native_l20_prefill_source(
            frame, receiver, spans, rows, cp_ctx=cp
        )
        source_views = native_l20_cache_views(TEXT, first_cache, 0)
        replica_views = native_l20_cache_views(TEXT, replica, 1)
        for span in spans:
            logical = torch.arange(span.start + span.count, device="cuda")
            for tag in ("hca_kv", "indexer_kv"):
                values = []
                for views, mapping in (
                    (source_views, source_pages),
                    (replica_views, target_pages),
                ):
                    slots = logical_to_physical_slots(
                        torch.tensor(mapping[span.identity][tag], device="cuda"),
                        logical,
                        tag,
                        1,
                    )
                    values.append(views[tag].gather(slots))
                torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
        frame.row_map.fill_(99)
        frame.topk.zero_()
        before = backing.clone()
        if step == 0:
            with imported_native_l20_prefill(second.layers[0].attn, source):
                entry = second.layers[0].attn
                for bad in (
                    None,
                    replace(cp, cp_rank=(cp.cp_rank + 1) % 4),
                    replace(cp, local_is_real=~cp.local_is_real),
                ):
                    with bind_attn_cache(entry, replica, destination, cp_ctx=bad):
                        try:
                            source.require_owner(entry)
                        except (ValueError, NotImplementedError):
                            pass
                        else:
                            raise AssertionError(
                                "stale CP source dispatch was admitted"
                            )
                with patch.object(
                    receiver, "cp_rank", (receiver.cp_rank + 1) % 4
                ), bind_attn_cache(entry, replica, destination, cp_ctx=cp):
                    try:
                        source.require_owner(entry)
                    except NotImplementedError:
                        pass
                    else:
                        raise AssertionError("wrong receiver lane was admitted")
            try:
                with imported_native_l20_prefill(
                    second.layers[0].attn, source
                ), imported_delayed_pre_mix(
                    second.layers[0].attn_hc, frame.hidden_states, frame.prev_pre_mix
                ), patch.object(
                    second.layers[3].attn,
                    "_prefill_compute_qkv",
                    side_effect=RuntimeError("CP stage selection failure"),
                ):
                    call(
                        second,
                        replica,
                        local_ids,
                        cp.global_positions,
                        lengths,
                        prefixes,
                        destination,
                        cp,
                        frame.hidden_states.clone(),
                    )
            except RuntimeError as error:
                assert str(error) == "CP stage selection failure"
            else:
                raise AssertionError("injected CP failure was ignored")
            assert_cleared(unittest.TestCase(), second, shared)
            torch.testing.assert_close(backing, before, rtol=0, atol=0)
        selected = {}

        def select_actual(block, inputs, hidden):
            selected[block.layer_id] = block.attn._shared_attention["topk"][
                block.layer_id
            ].clone()

        selection_hooks = [
            block.register_forward_hook(select_actual)
            for block in second.layers
            if block.attn.is_index_source
        ]
        try:
            with imported_native_l20_prefill(
                second.layers[0].attn, source
            ), imported_delayed_pre_mix(
                second.layers[0].attn_hc, frame.hidden_states, frame.prev_pre_mix
            ):
                frame.prev_pre_mix.zero_()
                actual = call(
                    second,
                    replica,
                    local_ids,
                    cp.global_positions,
                    lengths,
                    prefixes,
                    destination,
                    cp,
                    frame.hidden_states,
                )
        finally:
            for handle in selection_hooks:
                handle.remove()
        real = cp.local_is_real
        assert set(selected) == set(expected_selections) == {24, 28, 32, 36}
        for layer_id, value in selected.items():
            torch.testing.assert_close(
                value[real], expected_selections[layer_id][real], rtol=0, atol=0
            )
        torch.testing.assert_close(actual[real], expected[real], rtol=0, atol=0)
        restored = restore(actual, cp, group)
        torch.testing.assert_close(restored, expected_cp1, rtol=0, atol=0)
        torch.testing.assert_close(
            F.linear(restored, second.head_weight),
            F.linear(expected_cp1, cp1.head_weight),
            rtol=0,
            atol=0,
        )
        assert bool(torch.isfinite(restored).all())
        torch.testing.assert_close(backing, before, rtol=0, atol=0)
        assert_cleared(unittest.TestCase(), second, shared)
        for key, value in whole_cache.storage.items():
            torch.testing.assert_close(value, oracle_cache.storage[key], rtol=0, atol=0)
        for (local, tag), value in first_cache.storage.items():
            torch.testing.assert_close(
                value, whole_cache.storage[local, tag], rtol=0, atol=0
            )
        for (local, tag), value in second_cache.storage.items():
            torch.testing.assert_close(
                value, whole_cache.storage[local + 21, tag], rtol=0, atol=0
            )
        prefixes = [p + n for p, n in zip(prefixes, lengths)]
        print(
            "DSV41_CP_STAGE "
            + json.dumps(
                {
                    "rank": rank,
                    "step": step,
                    "ends": prefixes,
                    "frames_exact": True,
                    "hidden_logits_exact": True,
                    "cache_bytes_exact": True,
                    "transport_bytes_exact": True,
                    "selections_exact": True,
                }
            ),
            flush=True,
        )


def run_rank(rank, port):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ.update(
        {
            "NCCL_P2P_DISABLE": "1",
            "DSV4_MOE_STRATEGY": "auto",
            "MOEDBG": "0",
            "DSV4_PREFILL_FAST_PATH": "0",
            "RTP_V41_BATCHED_PRODUCER": "0",
        }
    )
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=120),
    )
    config = ParallelismConfig()
    config.world_size = config.local_world_size = 8
    config.world_rank = config.local_rank = rank
    config.tp_size, config.dp_size, config.pp_size = 4, 1, 2
    try:
        init_distributed_environment(config, NcclCommConfig(nccl_ip="127.0.0.1"), port)
        group = _get_group(Group.TP)
        assert tuple(dist.get_process_group_ranks(group)) == tuple(
            range(rank // 4 * 4, rank // 4 * 4 + 4)
        )
        with torch.inference_mode():
            run_controls(rank, group)
        dist.barrier()
    finally:
        destroy_distributed_environment()


class V41CPStageSlicesSM120Test(unittest.TestCase):
    def test_imported_cp_frames_match_full_maintained_owner(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 8)
        for rank in range(8):
            self.assertEqual(torch.cuda.get_device_capability(rank), (12, 0))
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        mp.spawn(run_rank, args=(port,), nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main()

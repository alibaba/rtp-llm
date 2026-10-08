"""SM120 planner/codec qualification with synthetic KV and native-shaped pools.

This does not instantiate the C++ allocator or qualify full-model execution.
"""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.models.dsv41_kv_cache import dsv41_cache_bindings, dsv41_page_layouts
from rtp_llm.models_py.modules.dsv4.v41_cache import quantize_rows_eager
from rtp_llm.models_py.modules.dsv4.v41_cache_plan import V41StageCache
from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import (
    V41CacheIdentity,
    V41L20Span,
)
from rtp_llm.models_py.modules.dsv4.v41_compressor import prepare_ratio2
from rtp_llm.models_py.modules.dsv4.v41_projection import V41KVFrame
from rtp_llm.models_py.modules.dsv4.v41_stage_attention import V41ForwardRows


def owner_fixture():
    path = (
        Path(__file__).resolve().parents[5]
        / "config/test/testdata/deepseek_v41_flash.json"
    )
    config = V41Config.from_dict(json.loads(path.read_text()))
    bindings, layouts = dsv41_cache_bindings(config.text, 2, 0), dsv41_page_layouts(1)
    raw = torch.full(
        (sum(4 * layouts[b.tag].stride_bytes for b in bindings),),
        165,
        dtype=torch.uint8,
        device="cuda",
    )
    groups, offset = [[] for _ in range(21)], 0
    for binding in bindings:
        width = layouts[binding.tag].stride_bytes
        view = raw[offset : offset + 4 * width].view(4, width)
        offset += view.numel()
        if binding.tag == "csa_state":
            view = view.view(torch.float32)
        groups[binding.storage_layer_id].append(
            SimpleNamespace(
                tag=binding.tag,
                layer_id=binding.storage_layer_id,
                seq_size_per_block=256,
                kv_scale_base=None,
                kv_cache_base=view,
                read_replica_source_layer_id=None,
            )
        )
    native = SimpleNamespace(layer_count=21, get_layer_cache_groups=lambda i: groups[i])
    owner = V41StageCache.from_native(
        config,
        native,
        role="prefill",
        pp_rank=0,
        max_positions=512,
        max_scratch_bytes=64 * 1024**2,
    )
    blocks = {
        "swa_kv": (1,),
        "csa_state": (2,),
        "csa_kv": (1, 3),
        "hca_kv": (2, 1),
        "indexer_kv": (3, 2),
    }
    identity = V41CacheIdentity(41, 1, 0)
    owner.begin_request(identity, blocks)
    return owner, identity


def values_at(positions, layer):
    return (
        (positions % 17).float()[:, None] / 4
        + layer / 8
        + torch.arange(512, device=positions.device)[None, :] % 31 / 16
    ).bfloat16()


def forward_rows(span):
    positions = torch.arange(span.start, span.start + span.count, device="cuda").flip(0)
    if span.count < 128:
        positions = torch.cat((positions, positions.new_tensor([-1])))
    return V41ForwardRows(
        "cache-plan-hardware", torch.full_like(positions, 41), positions, positions >= 0
    )


class V41CachePlanHardwareTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))

    def publish(self, batch):
        rows = batch.rows
        for layer, plan in batch.plans.items():
            values = values_at(rows.positions, layer)
            kwargs, pair, complete = {}, None, rows.valid_rows
            if plan.global_writes is not None:
                ratio = 2 if plan.global_cache.tag == "csa_kv" else 1
                complete = rows.valid_rows & ((rows.positions + 1) % ratio == 0)
                kwargs = dict(
                    global_cache=plan.global_cache,
                    global_slots=plan.global_writes,
                    index_cache=plan.index_cache,
                    index_slots=plan.index_writes,
                )
                if ratio == 2:
                    old = plan.pair_metadata.previous_slots >= 0
                    previous = plan.pair_state.read(plan.pair_metadata.previous_slots)
                    torch.testing.assert_close(
                        previous[old, :512].cpu(),
                        values_at((rows.positions[old] - 1).cpu(), layer).float(),
                        rtol=0,
                        atol=0,
                    )
                    pair = prepare_ratio2(
                        values.float(),
                        torch.zeros_like(values, dtype=torch.float32),
                        torch.ones(512, dtype=torch.bfloat16, device="cuda"),
                        plan.pair_state,
                        plan.pair_metadata,
                    )
            frame = V41KVFrame(
                values,
                values if kwargs else None,
                values[:, :128].contiguous() if kwargs else None,
                rows.valid_rows,
                complete,
                plan.global_cache.tag if kwargs else None,
                pair,
                "triton",
                torch.cuda.current_stream().cuda_stream,
            )
            frame.publish(plan.swa, plan.swa_writes, **kwargs)
            if layer in (0, 2, 20):
                mask = plan.swa_reads >= 0
                positions = (rows.positions - 127).clamp_min(0)[:, None] + torch.arange(
                    128, device="cuda"
                )
                torch.testing.assert_close(
                    plan.swa.gather(plan.swa_reads[mask]).cpu(),
                    quantize_rows_eager(
                        values_at(positions[mask].cpu(), layer), "swa_kv"
                    ),
                    rtol=0,
                    atol=0,
                )
            if kwargs:
                for cache, slots, width in (
                    (plan.global_cache, plan.global_writes, 512),
                    (plan.index_cache, plan.index_writes, 128),
                ):
                    torch.testing.assert_close(
                        cache.gather(slots[complete]).cpu(),
                        quantize_rows_eager(values[complete, :width].cpu(), cache.tag),
                        rtol=0,
                        atol=0,
                    )

    def test_chunks_preserve_ring_bytes_pair_carry_and_relocated_full_pages(self):
        owner, identity = owner_fixture()
        self.assertEqual(
            len({v.raw.untyped_storage().data_ptr() for v in owner.caches.values()}), 1
        )
        for chunk, start, count in (
            (0, 0, 128),
            (1, 128, 127),
            (2, 255, 2),
            (3, 257, 1),
        ):
            span = V41L20Span(identity, chunk, start, count)
            batch = owner.prepare(forward_rows(span), (span,))
            self.assertEqual(bool(batch.scratch_bytes), chunk == 1)
            for sources in owner.sources:
                batch.plans[sources.layer_id].validate(sources, batch.rows, 512)
            if chunk == 3:
                self.assertEqual(int(batch.plans[2].global_writes[0]), 3 * 128)
                self.assertEqual(int(batch.plans[2].index_writes[0]), 2 * 256)
            batch._execute(lambda _: self.publish(batch))
            positions = torch.arange(max(0, start + count - 128), start + count)
            for sources in owner.sources:
                cache = owner.caches[sources.layer_id, "swa_kv"]
                slots = cache.layout.entries + positions % cache.layout.entries
                torch.testing.assert_close(
                    cache.gather(slots.cuda()).cpu(),
                    quantize_rows_eager(
                        values_at(positions, sources.layer_id), "swa_kv"
                    ),
                    rtol=0,
                    atol=0,
                )
            self.assertEqual(owner._state(identity).visible, start + count)
        for cache in owner.caches.values():
            self.assertTrue(bool((cache.raw[0].view(torch.uint8) == 165).all()))
        owner.release(identity)

    def test_stream_and_budget_rejections_preserve_native_pools(self):
        owner, identity = owner_fixture()
        span = V41L20Span(identity, 0, 0, 128)
        batch = owner.prepare(forward_rows(span), (span,))
        with torch.cuda.stream(torch.cuda.Stream()):
            with self.assertRaisesRegex(RuntimeError, "preparation stream"):
                batch.discard()
        batch._execute(lambda _: self.publish(batch))
        before = {key: cache.raw.clone() for key, cache in owner.caches.items()}
        owner.max_scratch_bytes = 0
        span = V41L20Span(identity, 1, 128, 4)
        with self.assertRaisesRegex(ValueError, "budget"):
            owner.prepare(forward_rows(span), (span,))
        for key, cache in owner.caches.items():
            torch.testing.assert_close(
                cache.raw.view(torch.uint8),
                before[key].view(torch.uint8),
                rtol=0,
                atol=0,
            )
        self.assertTrue(owner._drained)
        self.assertIsNone(owner._active)
        self.assertEqual(owner._state(identity).visible, 128)


if __name__ == "__main__":
    unittest.main()

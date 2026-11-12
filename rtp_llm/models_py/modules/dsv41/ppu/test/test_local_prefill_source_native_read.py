"""FULL source reads through fresh pages and the unchanged native recovery.

Register this module with the existing shared sources and the source-cache,
SWA-segments/native-write/fused/operator test fixtures imported below. CPU
substitutes only physical PPU admission, pinned upload and the existing kernel
launcher. Explicit device cases require a real PPU89 and use native kernels.
"""

import gc
import json
import math
import os
import unittest
import weakref
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41 import cp
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_swa_metadata as metadata
from rtp_llm.models_py.modules.dsv41.ppu import _quant, graph_cache_kernels, prefill, prefill_readback
from rtp_llm.models_py.modules.dsv41.ppu.kv_codec import decode_kv
from rtp_llm.models_py.modules.dsv41.ppu.native_cache import _WIDTH
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import (
    LOCAL_PREFILL_SOURCE_CACHE_ENV, LOCAL_PREFILL_SOURCE_NATIVE_READ_ENV,
    SharedNativeCache,
)
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import PrefillPolicy
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_fused import assert_bytes
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_operator_flags import _trace_events
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_source_cache import _SourceFixture
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_swa_native_write import fixture as swa_fixture
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_swa_segments import arguments, compact_plan, stage


ENV, SCC = LOCAL_PREFILL_SOURCE_NATIVE_READ_ENV, LOCAL_PREFILL_SOURCE_CACHE_ENV
_DEVICE_GUARD = SharedNativeCache._local_source_native_read_device


class _FullNative:
    """True E/S and planar native pages, with optional real 2880-block extent."""

    def __init__(self, device, totals, ratios, *, padding=16, spans=None,
                 blocks=None, configure=None, alias=False):
        self.inputs, self.handles, self.raw = {}, {}, {}
        self.spans = dict(hca_kv=256, csa_kv=256, indexer_kv=256, swa_kv=512, csa_state=512)
        self.spans.update(spans or {})
        self.entries = dict(hca_kv=self.spans['hca_kv'], csa_kv=self.spans['csa_kv'] // 2,
                            indexer_kv=self.spans['indexer_kv'], swa_kv=128, csa_state=2)
        keys = [(0, 'swa_kv')]
        for source, ratio in ratios.items():
            keys.extend(((source, 'csa_kv' if ratio == 2 else 'hca_kv'), (source, 'indexer_kv')))
            if ratio == 2:
                keys.append((source, 'csa_state'))
        self.group_tags = sorted({tag for _, tag in keys})
        for owner, tag in keys:
            columns = max(1, math.ceil(max(totals, default=0) / self.spans[tag]))
            count = blocks or max(3, len(totals) * columns + 2)
            if tag not in self.inputs:
                host = (torch.arange(len(totals) * columns, dtype=torch.int32).reshape(len(totals), columns) + 1) % count
                if configure is not None:
                    configure(tag, host, count)
                table = host.to(device=device, copy=True)
                self.inputs[tag] = NS(kv_cache_kernel_block_id=host,
                                      kv_cache_kernel_block_id_device=table,
                                      kv_cache_block_id=host, kv_cache_block_id_device=table)
            width = dict(hca_kv=288, csa_kv=288, indexer_kv=68, swa_kv=528, csa_state=1024)[tag]
            stride = self.entries[tag] * width + padding
            dtype = torch.float32 if tag == 'csa_state' else torch.uint8
            # Explicit outer canaries and a nonzero native storage offset.
            raw = torch.full((count * stride + 32,), 173, dtype=dtype, device=device)
            tensor = raw[16:-16].view(count, stride)
            if dtype == torch.uint8:
                base = (torch.arange(stride, device=device) % 251).to(torch.uint8)
                tensor.copy_(base[None].expand_as(tensor))
                tensor.add_((torch.arange(count, device=device) % 251).to(torch.uint8)[:, None])
                tensor.add_(owner * 7)
                if padding:
                    tensor[:, -padding:].fill_(173)
            self.raw[owner, tag] = raw
            self.handles[owner, tag] = NS(kv_cache_base=tensor)
        if alias and len(ratios) > 1:
            first, second = tuple(ratios)[:2]
            for tag in ('csa_kv' if ratios[first] == 2 else 'hca_kv', 'indexer_kv'):
                self.handles[second, tag].kv_cache_base = self.handles[first, tag].kv_cache_base
                self.raw[second, tag] = self.raw[first, tag]

    def get_layer_cache(self, owner, tag):
        return self.handles[owner, tag]

    def get_entries_per_block(self, tag):
        return self.entries[tag]

    def get_kernel_seq_size_per_block(self, tag):
        return self.spans[tag]

    get_seq_size_per_block = get_kernel_seq_size_per_block


def fixture(device, totals=(1031,), *, ratios=None, **kwargs):
    ratios = ratios or {2: 2, 20: 1}
    geometry = NS(num_layers=1, kv_source_layer_ids=tuple(ratios), head_dim=512, window=128,
                  ratio=lambda source: ratios.get(source, 0), kv_source_for=lambda layer: next(iter(ratios)))
    native = _FullNative(device, totals, ratios, **kwargs)
    cache = SharedNativeCache(PrefillPolicy().freeze({}), geometry, native, native.inputs, totals,
                              device=device, prefill=True, defer_producer_slots=True)
    return cache


def cpu_address_oracle(cache, key, rid, count):
    """Independent raw-position / two-plane formula; not a restore double."""
    pool, ratio = cache.pools[key], cache.geometry.ratio(key[0])
    table = pool.table.detach().cpu()
    native = pool.tensor.detach().cpu()
    width = _WIDTH[key[1]]
    output = torch.empty((count, width), dtype=pool.tensor.dtype)
    column = torch.arange(width)[None]
    for first in range(0, count, 8192):
        last = min(first + 8192, count)
        positions = torch.arange(first, last, dtype=torch.int64) * ratio
        pages, offset = positions // pool.span, (positions % pool.span) // ratio
        blocks = table[rid, pages].long()
        physical = (torch.where(column < 64, offset[:, None] * 64 + column,
                                pool.entries * 64 + offset[:, None] * 4 + column - 64)
                    if key[1] == 'indexer_kv' else offset[:, None] * 288 + column)
        output[first:last] = native[blocks[:, None], physical]
    return output.to(pool.tensor.device)


def original(cache, source, rid=0, count=None):
    ratio = cache.geometry.ratio(source)
    count = cache.lengths[rid] // ratio if count is None else count
    positions = torch.arange(count, device=cache.device) * ratio
    tag = 'global_kv_2' if ratio == 2 else 'global_kv_1'
    return tuple(cache._copy_rows((source, kind), rid, positions) for kind in (tag, 'indexer_kv'))


def read(cache, source, rid=0, count=None):
    count = cache.lengths[rid] // cache.geometry.ratio(source) if count is None else count
    return cache.read_source(source, rid, global_entries=count)


class _RestoreCPU:
    """Existing kernel ABI double; only device tests prove its native launch."""

    def __getitem__(self, grid):
        def launch(packet, plan, out, scale, stripe_count, owner_rows, page_rows,
                   cp_size, block_bytes, page_bytes, entries, width, planar, split, tile):
            assert cp_size == 1 and stripe_count == 1 and not split and scale is None
            assert page_bytes == packet.stride(0) and owner_rows == page_rows
            count, start = int(plan[0, 2]), int(plan[0, 3])
            position = torch.arange(count)
            entry = position % page_rows
            column = torch.arange(width)[None]
            address = (torch.where(column < 64, entry[:, None] * 64 + column,
                                   entries * 64 + entry[:, None] * 4 + column - 64)
                       if planar else entry[:, None] * width + column)
            out[start:start + count].copy_(packet[position[:, None] // page_rows, address])
        return launch


class _ReadCases:
    def setUp(self):
        environment = patch.dict(os.environ, {ENV: '1', SCC: '0', 'TP_SIZE': '1', 'PREFILL_CP_SIZE': '1',
                                              'RTP_V41_PPU_LOCAL_PREFILL_SWA_NATIVE_READ': '0'})
        environment.start()
        self.addCleanup(environment.stop)

    @torch.no_grad()
    def test_true_tag_geometry_partial_pages_order_zero_duplicates_and_bytes(self):
        def configure(tag, table, blocks):
            if table.shape[1] > 2:
                table[0, 0], table[0, 1], table[0, 2] = blocks - 1, 0, blocks - 1
        for ratio in (1, 2):
            for count in (0, 1, 127, 128, 129, 255, 256, 257, 4095, 4096, 4097):
                with self.subTest(ratio=ratio, count=count):
                    cache = fixture(self.device, (max(1, count * ratio),), ratios={2: ratio}, configure=configure)
                    snapshots = {k: v.clone() for k, v in cache.native.raw.items()}
                    expected = original(cache, 2, count=count)
                    actual = read(cache, 2, count=count)
                    for kind, new, old in zip(('global_kv_2' if ratio == 2 else 'global_kv_1', 'indexer_kv'), actual, expected):
                        assert_bytes(self, new, old)
                        assert_bytes(self, new, cpu_address_oracle(cache, (2, kind), 0, count))
                        self.assertFalse(any(torch._C._is_alias_of(new, p.tensor) for p in cache.pools.values()))
                    self.assertEqual(cache._local_source_native_read_stats['reads'], 2 * int(count > 0))
                    for key, before in snapshots.items():
                        assert_bytes(self, cache.native.raw[key], before)
        # Per-tag S is independent; indexer has P256/E512, main P128/E128.
        cache = fixture(self.device, (1031,), ratios={2: 2}, spans={'indexer_kv': 512})
        for actual, expected in zip(read(cache, 2), original(cache, 2)):
            assert_bytes(self, actual, expected)
        self.assertEqual({g['span'] for g in cache._local_source_native_read_stats['geometries']}, {256, 512})
        cache = fixture(self.device, (1031,), ratios={2: 1},
                        spans={'hca_kv': 257, 'indexer_kv': 259})
        for actual, expected, kind in zip(read(cache, 2), original(cache, 2), ('global_kv_1', 'indexer_kv')):
            assert_bytes(self, actual, expected)
            assert_bytes(self, actual, cpu_address_oracle(cache, (2, kind), 0, 1031))

    @torch.no_grad()
    def test_ragged_inactive_holes_errors_and_main_before_indexer_priority(self):
        def unused(tag, table, blocks):
            table[1].fill_(-1)
        cache = fixture(self.device, (1031, 1024, 511), ratios={2: 2}, configure=unused)
        cache.prepare_source(2)
        self.assertEqual(cache._local_source_native_read_stats['reads'], 0)
        for rid in (0, 2):
            for actual, expected in zip(read(cache, 2, rid), original(cache, 2, rid)):
                assert_bytes(self, actual, expected)
        self.assertEqual(cache._local_source_native_read_stats['reads'], 4)
        for kind in ('main', 'indexer', 'both', 'short', 'outside'):
            def bad(tag, table, blocks):
                if kind == 'short':
                    table.resize_(1, 1).fill_(1)
                elif kind == 'both' or (kind == 'main' and tag == 'csa_kv') or (kind == 'indexer' and tag == 'indexer_kv'):
                    table[0, 0] = -1
                elif kind == 'outside':
                    table[0, 0] = blocks
            cache = fixture(self.device, (1031,), ratios={2: 2}, configure=bad)
            with self.subTest(kind=kind), self.assertRaises((ValueError, IndexError)) as old:
                original(cache, 2)
            with self.assertRaises(type(old.exception)) as new:
                read(cache, 2)
            self.assertEqual(str(new.exception), str(old.exception))
            self.assertEqual(cache._local_source_native_read_stats['reads'], int(kind == 'indexer'))
        cache = fixture(self.device)
        for rid, count in ((-4, 1), (len(cache.lengths), 1), (0, -1)):
            with self.subTest(rid=rid, count=count):
                with self.assertRaises((ValueError, RuntimeError, IndexError)) as old:
                    original(cache, 2, rid, count)
                with self.assertRaises(type(old.exception)) as new:
                    read(cache, 2, rid, count)
                self.assertEqual(str(new.exception), str(old.exception))

    @torch.no_grad()
    def test_metadata_reuse_fresh_ordinary_native_and_cross_alias_full_writes(self):
        for payload_cache in ('0', '1'):
            def alias(tag, table, blocks):
                table[1].copy_(table[0])
            with self.subTest(source_cache=payload_cache), patch.dict(os.environ, {SCC: payload_cache}):
                cache = fixture(self.device, (1031, 1031), ratios={2: 2, 8: 2}, configure=alias, alias=True)
                read(cache, 2)
                read(cache, 8, 1)
                stats = cache._local_source_native_read_stats
                before_uploads = stats['ids_uploads'], stats['plan_uploads']
                for tag in ('global_kv_2', 'indexer_kv'):
                    key = 2, tag
                    rows = torch.full((1, _WIDTH[tag]), 57, device=self.device, dtype=torch.uint8)
                    cache._copy_rows(key, 0, torch.tensor([0], device=self.device), rows)
                    self.assertFalse(cache._source_reads, 'every FULL write clears all aliased source banks')
                    for new, old in zip(read(cache, 8, 1), original(cache, 8, 1)):
                        assert_bytes(self, new, old)
                    # Real existing direct native writer, entered only after
                    # the unchanged adapter's conservative FULL invalidation.
                    cache._try_store_prefill_range(key, 0, 0, rows)
                    pool = cache.pools[key]
                    physical = int(pool.table[0, 0])
                    self.assertGreater(physical, 0, 'this separate provider reserves physical0')
                    slots = torch.tensor([physical * pool.entries], device=self.device)
                    graph_cache_kernels.write_packed_rows(pool.tensor, slots, rows + 1,
                                                         width=_WIDTH[tag], entries=pool.entries,
                                                         planar=tag == 'indexer_kv')
                    self.assertFalse(cache._source_reads)
                    for new, old in zip(read(cache, 8, 1), original(cache, 8, 1)):
                        assert_bytes(self, new, old)
                self.assertEqual((stats['ids_uploads'], stats['plan_uploads']), before_uploads)
                self.assertGreater(stats['ids_hits'], 0)
                if payload_cache == '1':
                    before = stats['reads']
                    read(cache, 8, 1)
                    self.assertEqual(stats['reads'], before, 'only original source cache may save final banks')

    @torch.no_grad()
    def test_birth_normal_mutations_rebind_mirror_plans_nonce_are_sticky(self):
        cases = ('host', 'device', 'meta', 'pool_table', 'pool', 'base', 'geometry',
                 'mirror', 'ids', 'plan', 'nonce')
        for kind in cases:
            with self.subTest(kind=kind):
                cache = fixture(self.device, ratios={2: 2})
                if kind in ('mirror', 'ids', 'plan'):
                    read(cache, 2)
                meta = cache.inputs['csa_kv']
                pool = cache.pools[2, 'global_kv_2']
                if kind in ('host', 'device'):
                    tensor = getattr(meta, 'kv_cache_kernel_block_id' + ('_device' if kind == 'device' else ''))
                    tensor.add_(1).sub_(1)
                elif kind == 'meta':
                    cache.inputs['csa_kv'] = NS(**vars(meta))
                elif kind == 'pool_table':
                    pool.table = pool.table.clone()
                elif kind == 'pool':
                    cache.pools[2, 'global_kv_2'] = replace(pool, tensor=pool.tensor.clone())
                elif kind == 'base':
                    pool.handle.kv_cache_base = pool.handle.kv_cache_base.clone()
                elif kind == 'geometry':
                    pool.span += 2
                    pool.span -= 2
                    pool.entries -= 1
                elif kind == 'nonce':
                    cache._local_source_native_read_nonce = object()
                elif kind == 'mirror':
                    cache._local_source_native_read_mirrors['csa_kv'][0].add_(1).sub_(1)
                elif kind == 'ids':
                    next(iter(cache._local_source_native_read_ids.values()))[0].add_(1).sub_(1)
                else:
                    next(iter(cache._local_source_native_read_plans.values()))[0].add_(1).sub_(1)
                before = cache._local_source_native_read_stats['reads']
                count = cache.lengths[0] // cache.geometry.ratio(2)
                self.assertIsNone(cache._try_read_local_source_pool((2, 'global_kv_2'), 0, count))
                self.assertIsNone(cache._try_read_local_source_pool((2, 'global_kv_2'), 0, count))
                self.assertEqual(cache._local_source_native_read_stats['reads'], before)
        for kind in ('meta', 'pool', 'geometry'):
            cache = fixture(self.device, ratios={2: 2})
            pool, meta = cache.pools[2, 'global_kv_2'], cache.inputs['csa_kv']
            if kind == 'meta':
                cache.inputs['csa_kv'] = NS(**vars(meta))
            elif kind == 'pool':
                cache.pools[2, 'global_kv_2'] = replace(pool, tensor=pool.tensor.clone())
            else:
                pool.entries -= 1
            self.assertIsNone(cache._try_read_local_source_pool((2, 'global_kv_2'), 0, 10))
            cache.inputs['csa_kv'], cache.pools[2, 'global_kv_2'] = meta, pool
            if kind == 'geometry':
                pool.entries += 1
            self.assertIsNone(cache._try_read_local_source_pool((2, 'global_kv_2'), 0, 10),
                              'observed rebind cannot remint the old birth after restoration')

    @torch.no_grad()
    def test_constructor_latch_scope_new_forward_inference_and_host_only_close(self):
        with patch.dict(os.environ, {ENV: '0'}):
            cache = fixture(self.device)
        for new, old in zip(read(cache, 2), original(cache, 2)):
            assert_bytes(self, new, old)
        self.assertEqual(cache._local_source_native_read_stats['reads'], 0)
        for field in ('_is_prefill', 'sharded'):
            cache = fixture(self.device)
            setattr(cache, field, field == 'sharded')
            self.assertIsNone(cache._try_read_local_source_pool((2, 'global_kv_2'), 0, 10))
        for env in ({'TP_SIZE': '8'}, {'PREFILL_CP_SIZE': '8'}):
            cache = fixture(self.device)
            with patch.dict(os.environ, env):
                self.assertIsNone(cache._try_read_local_source_pool((2, 'global_kv_2'), 0, 10))
        with torch.inference_mode():
            first = fixture(self.device, ratios={2: 2})
            old = read(first, 2)
            newer = SharedNativeCache(first.config, first.geometry, first.native, first.inputs,
                                      first.lengths, device=self.device, prefill=True, defer_producer_slots=True)
            self.assertIsNot(first._local_source_native_read_nonce, newer._local_source_native_read_nonce)
            first.close_local_source_native_reads()
            pool = newer.pools[2, 'global_kv_2']
            pool.tensor.fill_(49)  # Legal new-forward payload, inference table readonly.
            new = read(newer, 2)
            self.assertFalse(torch.equal(old[0], new[0]))
            self.assertEqual(newer._local_source_native_read_stats['mirrors'], 2)
        cache = fixture(self.device)
        read(cache, 2)
        stats = cache._local_source_native_read_stats
        with patch.object(torch.cuda, 'current_stream', side_effect=AssertionError('host-only close')):
            cache.close_local_source_native_reads()
            cache.close_local_source_native_reads()
            SharedNativeCache.__new__(SharedNativeCache).close_local_source_native_reads()
        self.assertIs(stats, cache._local_source_native_read_stats)
        self.assertTrue(stats['closed'])
        self.assertFalse(cache._local_source_native_read_bindings)
        self.assertFalse(cache._local_source_native_read_sources)
        self.assertFalse(cache._local_source_native_read_plans)
        self.assertFalse(cache._local_source_native_read_ids)
        self.assertIsNone(cache._try_read_local_source_pool((2, 'global_kv_2'), 0, 10))

    @torch.no_grad()
    def test_actual_query_compaction_before_source20_read_keeps_full_domain_alive(self):
        state = swa_fixture(self.device, (128,), (4096,), layers=22, decoder=True, seed=False)
        cache = state.cache
        stage(state, arguments(state, layer=20), layer=20)
        expected = original(cache, 20)
        compact_plan(state)
        self.assertTrue(cache._local_swa_native_read_stats['closed'])
        self.assertFalse(cache._local_source_native_read_stats['closed'])
        self.assertEqual(state.context.chunk_length, 140)
        for actual, old in zip(read(cache, 20), expected):
            assert_bytes(self, actual, old)
        self.assertEqual([g['count'] for g in cache._local_source_native_read_stats['geometries']], [4224, 4224])

    @torch.no_grad()
    def test_existing_workspace_indexer_decode_and_mutable_swa_match_old(self):
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                f = _SourceFixture(self.device, ratio)
                cache = f.cache
                positions = torch.cat([torch.arange(p, p + n, device=self.device) for p, n in zip(f.prefixes, f.fresh)])
                requests = torch.tensor([rid for rid, n in enumerate(f.fresh) for _ in range(n)], device=self.device)
                cu = torch.tensor([0, f.fresh[0], f.fresh[0] + f.fresh[1], sum(f.fresh)], device=self.device)
                context = prefill.build_step_context(positions, requests, cu)
                args = dict(ops=NS(decode_kv=decode_kv), cache=cache, layer_id=0, geometry=f.geometry,
                            context=context, selected_blocks=torch.zeros(positions.numel(), 5, device=self.device, dtype=torch.int32),
                            window=128, swa_starts=None)
                with patch.dict(os.environ, {ENV: '0', SCC: '0'}):
                    expected = prefill._build_reference_workspace(**args)
                actual = prefill._build_reference_workspace(**args)
                for new, old in zip(actual, expected):
                    assert_bytes(self, new, old)
                packed = read(cache, 0, 0)
                for new, old, tag in zip(packed, original(cache, 0), (f.tag, 'indexer_kv')):
                    dimension = _quant.INDEXER_HEAD_DIM if tag == 'indexer_kv' else _quant.KV_DIM
                    new_values = torch.empty((new.shape[0], dimension), device=new.device, dtype=torch.bfloat16)
                    old_values = torch.empty_like(new_values)
                    self.assertIs(decode_kv(new, pool_tag=tag, out=new_values), new_values)
                    self.assertIs(decode_kv(old, pool_tag=tag, out=old_values), old_values)
                    assert_bytes(self, new_values, old_values)
                f.stage_swa(shift=3)
                changed = prefill._build_reference_workspace(**args)
                self.assertFalse(torch.equal(changed[0], actual[0]))
                with patch.dict(os.environ, {ENV: '0', SCC: '0'}):
                    for new, old in zip(changed, prefill._build_reference_workspace(**args)):
                        assert_bytes(self, new, old)


class LocalSourceNativeReadCPU(_ReadCases, unittest.TestCase):
    device = 'cpu'

    def setUp(self):
        super().setUp()
        for mocked in (
            # A Mock would retain original _Pool arguments in call history,
            # defeating the owner-release weakref assertion after close.
            patch.object(SharedNativeCache, '_local_source_native_read_device', new=lambda cache, pool: True),
            patch.object(prefill_readback, '_restore_rows', _RestoreCPU()),
            patch.object(metadata, '_upload', new=lambda values, device, dtype: torch.tensor(values, device=device, dtype=dtype)),
        ):
            mocked.start()
            self.addCleanup(mocked.stop)

    @torch.no_grad()
    def test_replaced_constructor_owners_live_until_close_without_payload_cache(self):
        cache = fixture('cpu', ratios={2: 2})
        read(cache, 2)
        def rebind():
            refs = []
            for key, pool in list(cache.pools.items()):
                if key[1] not in ('global_kv_2', 'indexer_kv'):
                    continue
                meta = cache.inputs[pool.framework_tag]
                refs.extend(weakref.ref(value) for value in (
                    pool.tensor, pool.table, pool.handle.kv_cache_base,
                    meta.kv_cache_kernel_block_id, meta.kv_cache_kernel_block_id_device,
                ))
                new = pool.handle.kv_cache_base.clone()
                pool.handle.kv_cache_base = new
                cache.pools[key] = replace(pool, tensor=new.view(pool.tensor.shape), table=pool.table.clone())
                for field in ('kv_cache_kernel_block_id', 'kv_cache_kernel_block_id_device',
                              'kv_cache_block_id', 'kv_cache_block_id_device'):
                    setattr(meta, field, getattr(meta, field).clone())
            return refs
        refs = rebind()
        gc.collect()
        self.assertTrue(all(reference() is not None for reference in refs))
        self.assertFalse(cache._source_reads, 'SOURCE_CACHE=0 must not add a payload cache')
        cache.close_local_source_native_reads()
        gc.collect()
        self.assertTrue(all(reference() is None for reference in refs))

    @torch.no_grad()
    def test_raw_unsupported_tables_pool_layout_and_postlaunch_error_no_retry(self):
        with patch.object(SharedNativeCache, '_local_source_native_read_device', return_value=False):
            cache = fixture('cpu', ratios={2: 2})
        self.assertIsNone(cache._try_read_local_source_pool((2, 'global_kv_2'), 0, 10),
                          'unsupported construction cannot mint a later birth')
        for kind in ('host-mismatch', 'negative-view', 'pool-view', 'graph', 'device'):
            cache = fixture('cpu', ratios={2: 2})
            if kind == 'host-mismatch':
                cache.inputs['csa_kv'].kv_cache_kernel_block_id[0, 0] += 1
            elif kind == 'negative-view':
                cache.pools[2, 'global_kv_2'].table = torch._neg_view(cache.pools[2, 'global_kv_2'].table)
            elif kind == 'pool-view':
                pool = cache.pools[2, 'global_kv_2']
                pool.tensor = pool.tensor[:, ::2]
            elif kind == 'graph':
                cache.inputs['csa_kv'].is_cuda_graph = True
            with self.subTest(kind=kind), patch.object(metadata, '_upload', side_effect=AssertionError('unsupported upload')):
                if kind == 'device':
                    with patch.object(cache, '_local_source_native_read_device', return_value=False):
                        self.assertIsNone(cache._try_read_local_source_pool((2, 'global_kv_2'), 0, 10))
                else:
                    self.assertIsNone(cache._try_read_local_source_pool((2, 'global_kv_2'), 0, 10))
        cache = fixture('cpu', ratios={2: 2})
        with patch.object(prefill_readback, '_restore_rows') as kernel, patch.object(cache, '_copy_rows', side_effect=AssertionError('serial retry')):
            kernel.__getitem__.return_value.side_effect = RuntimeError('native restore failure')
            with self.assertRaisesRegex(RuntimeError, 'native restore failure'):
                read(cache, 2)

    @torch.no_grad()
    def test_real_device_metadata_guard_and_constructor_mirror_value_mismatch(self):
        tensor = NS(is_cuda=True, device=torch.device('cuda:0'))
        pool, cache, stream = NS(tensor=tensor), SharedNativeCache.__new__(SharedNativeCache), object()
        cache.device = 'cuda:0'
        patches = (patch.object(_quant, 'is_ppu', return_value=True),
                   patch.object(torch.cuda, 'get_device_capability', return_value=(8, 9)),
                   patch.object(torch.cuda, 'current_device', return_value=0),
                   patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False),
                   patch.object(torch.cuda, 'current_stream', return_value=stream),
                   patch.object(torch.cuda, 'default_stream', return_value=stream))
        from contextlib import ExitStack
        with ExitStack() as stack:
            for mocked in patches:
                stack.enter_context(mocked)
            self.assertTrue(_DEVICE_GUARD(cache, pool))
            for owner, name, value in ((_quant, 'is_ppu', False),
                                       (torch.cuda, 'get_device_capability', (8, 0)),
                                       (torch.cuda, 'current_device', 1),
                                       (torch.cuda, 'is_current_stream_capturing', True),
                                       (torch.cuda, 'current_stream', object())):
                with patch.object(owner, name, return_value=value):
                    self.assertFalse(_DEVICE_GUARD(cache, pool), name)
            with torch.enable_grad():
                self.assertFalse(_DEVICE_GUARD(cache, pool))
            cache.device = 'cuda:1'
            self.assertFalse(_DEVICE_GUARD(cache, pool))
        first = fixture('cpu', ratios={2: 2})
        first.inputs['csa_kv'].kv_cache_kernel_block_id[0, 0] += 1
        # Mismatch exists before this new constructor captures both versions.
        newer = SharedNativeCache(first.config, first.geometry, first.native, first.inputs,
                                  first.lengths, device='cpu', prefill=True, defer_producer_slots=True)
        self.assertIsNone(newer._try_read_local_source_pool((2, 'global_kv_2'), 0, 10))
        self.assertEqual(newer._local_source_native_read_stats['mirrors'], 1)
        self.assertIn('csa_kv', newer._local_source_native_read_invalid)


@unittest.skipUnless(os.environ.get('RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE') == '1', 'explicit PPU device test')
class LocalSourceNativeReadDevice(_ReadCases, unittest.TestCase):
    device = 'cuda'

    def setUp(self):
        super().setUp()
        self.assertTrue(_quant.is_ppu(), 'requires an actual ZW PPU')
        self.assertEqual(torch.cuda.get_device_capability(), (8, 9))

    @torch.no_grad()
    def test_actual_2880_blocks_512_pages_profile_and_exact_eight_bank_receipt(self):
        def arbitrary(tag, table, blocks):
            table.copy_(table.flip(1))
            table[0, 0], table[0, 1], table[0, 2] = 0, blocks - 1, 0
        cache = fixture('cuda', (131072,), ratios={2: 2, 8: 2, 14: 2, 20: 1}, blocks=2880, padding=0, configure=arbitrary)
        snapshots = {key: value.clone() for key, value in cache.native.raw.items()}
        # Compile all four unchanged address specializations outside profiling.
        warm_plan = torch.tensor([(0, 0, 1, 0)], device='cuda', dtype=torch.int64)
        for source in (2, 20):
            for kind in ('global_kv_2' if source == 2 else 'global_kv_1', 'indexer_kv'):
                pool = cache.pools[source, kind]
                rows = pool.tensor.new_empty((1, _WIDTH[kind]))
                page = pool.span // cache.geometry.ratio(source)
                prefill_readback._restore_rows[(1, 1)](
                    pool.tensor, warm_plan, rows, None, 1, page, page, 1,
                    pool.tensor.shape[1], pool.tensor.shape[1], pool.entries, _WIDTH[kind],
                    kind == 'indexer_kv', False, 1024,
                )
        arms = []
        for enabled in (False, True):
            with patch.dict(os.environ, {ENV: str(int(enabled))}), patch.object(
                prefill_readback, 'gather_pool', side_effect=AssertionError('no whole-pool wrapper'),
            ), patch.object(prefill_readback, 'cp_gather_rank_major', side_effect=AssertionError('no collective')), patch.object(
                cp, 'cp_gather_rank_major', side_effect=AssertionError('no collective'),
            ):
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
                    arms.append([read(cache, source) for source in (2, 8, 14, 20)])
                    torch.cuda.synchronize()
            events = _trace_events(profile, f'source-native-eight-banks-flag{int(enabled)}')
            kernels = [e['name'] for e in events if e.get('cat') == 'kernel']
            cpu = [e for e in events if e.get('cat') == 'cpu_op']
            self.assertEqual(sum('_restore_rows' in name for name in kernels), 8 * int(enabled))
            byte_gathers = [e for e in cpu if e['name'] == 'aten::index'
                            and e.get('args', {}).get('Input type') == ['unsigned char', '']]
            self.assertEqual(len(byte_gathers), 0 if enabled else 160)
            if enabled:
                self.assertEqual(len(kernels), 16, kernels)
                self.assertEqual(sum(e['name'] == 'aten::index_select' for e in cpu), 8)
                self.assertFalse(any(e['name'] in ('aten::min', 'aten::max', 'aten::_local_scalar_dense') for e in cpu))
        for source, new_source, old_source in zip((2, 8, 14, 20), arms[1], arms[0]):
            for kind, new, old in zip(('global_kv_1' if source == 20 else 'global_kv_2', 'indexer_kv'), new_source, old_source):
                assert_bytes(self, new, old)
                assert_bytes(self, new, cpu_address_oracle(cache, (source, kind), 0, new.shape[0]))
        for key, before in snapshots.items():
            assert_bytes(self, cache.native.raw[key], before)
        stats = cache._local_source_native_read_stats
        self.assertEqual({name: stats[name] for name in ('reads', 'packet_calls', 'restore_calls', 'mirrors', 'ids_uploads', 'plan_uploads', 'packet_bytes', 'output_bytes', 'ids_bytes', 'plan_bytes', 'fallbacks')},
                         dict(reads=8, packet_calls=8, restore_calls=8, mirrors=3, ids_uploads=3, plan_uploads=2,
                              packet_bytes=130023424, output_bytes=116654080, ids_bytes=12288, plan_bytes=64, fallbacks=0))
        self.assertEqual(len(stats['geometries']), 8)
        for geometry in stats['geometries']:
            self.assertEqual(geometry['pages'], 512)
            self.assertEqual(geometry['blocks'], 2880)
            self.assertEqual(geometry['span'], 256)
        cache.close_local_source_native_reads()
        self.assertTrue(stats['closed'])

    @torch.no_grad()
    def test_actual_pinned_upload_gc_before_observation_and_stream_rollback(self):
        cache = fixture('cuda', (1031,), ratios={2: 2})
        expected = original(cache, 2)
        actual = read(cache, 2)
        # No CPU result observation/synchronize before forcing Python collection
        # and pinned allocator pressure. _upload's allocator tracks queued copy.
        gc.collect()
        junk = [torch.empty(size, dtype=torch.int64, pin_memory=True).fill_(-7)
                for size in (4, 5, 512) for _ in range(32)]
        del junk
        gc.collect()
        for new, old in zip(actual, expected):
            assert_bytes(self, new, old)
        cache = fixture('cuda', ratios={2: 2})
        before = cache._local_source_native_read_stats['ids_uploads']
        with torch.cuda.stream(torch.cuda.Stream()):
            self.assertIsNone(cache._try_read_local_source_pool((2, 'global_kv_2'), 0, 10))
        self.assertEqual(cache._local_source_native_read_stats['ids_uploads'], before)


if __name__ == '__main__':
    unittest.main()

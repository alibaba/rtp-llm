"""Exact fresh BF16 scatter and production continuation routing contracts."""

import ast
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_swa_triton as codec


def load_attention():
    """Compile unchanged SWA methods without unrelated native model imports."""
    path = Path(codec.__file__).with_name("attention_v41.py")
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "AttentionV41FP8"
    )
    names = {
        "_can_fuse_swa_fresh",
        "_swa_prefill_workspace",
        "_swa_prefill_concat",
        "_prefill_write_swa_fp8_paged",
        "_prefill_produce",
        "_host_prefill_lengths",
        "_host_prefill_prefixes",
    }
    methods = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    test_cls = ast.ClassDef(
        name="Attention", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    env = dict(
        torch=torch,
        swa_codec=codec,
        SWA_KV="swa",
        record_function_range=lambda *a: nullcontext(),
        _use_small_cp_x_gather=lambda cp: False,
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[test_cls], type_ignores=[])),
            str(path),
            "exec",
        ),
        env,
    )
    return env["Attention"]


def fixture(
    lengths=(139, 37, 1),
    prefixes=(127, 0, 5),
    rank=0,
    dtype=torch.int64,
    all_negative=False,
):
    n, width = sum(lengths), max(p + s for p, s in zip(prefixes, lengths))
    # Include every BF16 bit pattern for longer inputs, with padded row stride.
    bits = (
        (torch.arange(n * 528, device="cuda", dtype=torch.int32) % 65536)
        .to(torch.int16)
        .reshape(n, 528)
    )
    keys = bits.view(torch.bfloat16)[:, :512]
    raw = torch.full((4, 18080), 0x5A, device="cuda", dtype=torch.uint8)[:, :18048]
    slots = torch.full((n,), -1, device="cuda", dtype=dtype)
    if not all_negative:
        count = min(n, 136)
        slots[-count:] = torch.arange(count, device="cuda", dtype=dtype)
    unique = torch.tensor([] if all_negative else [1], device="cuda", dtype=dtype)
    compaction = SimpleNamespace(compact_slots=slots, unique_blocks=unique)
    dest = torch.tensor(
        [
            b * width + p + i
            for b, (p, s) in enumerate(zip(prefixes, lengths))
            for i in range(s)
        ],
        device="cuda",
        dtype=dtype,
    )
    # Offset-one contiguous metadata must remain supported.
    dest = torch.cat((torch.zeros(1, device="cuda", dtype=dtype), dest))[1:]
    out = torch.full(
        (len(lengths), width, 512), -3, device="cuda", dtype=torch.bfloat16
    )
    return keys, raw, slots, compaction, dest, out


def write(keys, raw, slots, compaction, dest, out, rank=0):
    codec.quantize_and_insert_k_cache_cp_byte_sliced(
        keys,
        raw,
        slots,
        136,
        rank,
        4,
        compaction,
        fresh_out=out,
        fresh_slots=dest if out is not None else None,
    )


class SwaWorkspaceViewTest(unittest.TestCase):
    def test_ragged_views_preserve_offsets_strides_and_aliasing(self):
        owner = load_attention()()
        owner.swa_bounded_replay = False
        owner.window_size = 128
        lengths, prefixes = [139, 37, 0, 1], [127, 0, 5, 64]
        owner._host_prefill_lengths = lambda _: lengths
        owner._host_prefill_prefixes = lambda _: prefixes
        common = SimpleNamespace(batch_size=4, any_cont=True)
        for padded in (False, True):
            with self.subTest(padded=padded):
                storage = torch.arange(5 * 270 * 16).view(5, 270, 16)
                buf = storage[1:, 2:, ::2] if padded else storage[:4]
                owner._swa_prefill_concat = lambda *args, **kw: buf
                expected = [
                    buf[b, : min(p, 127) + n]
                    for b, (p, n) in enumerate(zip(prefixes, lengths))
                ]

                class RecordViews(TorchDispatchMode):
                    def __init__(self):
                        self.ops = []

                    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                        self.ops.append(func)
                        return func(*args, **(kwargs or {}))

                with RecordViews() as views:
                    actual, starts = owner._swa_prefill_workspace(None, common)
                self.assertEqual(starts, [0, 0, 0, 0])
                self.assertNotIn(torch.ops.aten.select.int, views.ops)
                self.assertNotIn(torch.ops.aten.slice.Tensor, views.ops)
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(got, want)
                    self.assertEqual(got.stride(), want.stride())
                    self.assertEqual(got.storage_offset(), want.storage_offset())
                actual[1].fill_(-1)
                torch.testing.assert_close(expected[1], actual[1])


class SwaFreshGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

    def test_exact_bits_cache_and_padding_all_ranks(self):
        for rank in range(4):
            for dtype in (torch.int32, torch.int64):
                for empty in (False, True):
                    with self.subTest(rank=rank, dtype=dtype, empty=empty):
                        k, raw, slots, comp, dst, out = fixture(
                            rank=rank, dtype=dtype, all_negative=empty
                        )
                        old_raw, expected = raw.clone(), out.clone()
                        write(k, old_raw, slots, comp, dst, None, rank)
                        expected.view(-1, 512).index_copy_(0, dst.long(), k)
                        write(k, raw, slots, comp, dst, out, rank)
                        self.assertTrue(torch.equal(raw, old_raw))
                        self.assertTrue(
                            torch.equal(
                                out.view(torch.int16), expected.view(torch.int16)
                            )
                        )

    def test_unsupported_dtype_and_slot_stride_fall_back(self):
        k, raw, slots, comp, dst, _ = fixture()
        self.assertFalse(
            codec.is_supported_fresh_store(k.float(), raw, slots, 136, 0, 4, comp, dst)
        )
        self.assertFalse(
            codec.is_supported_fresh_store(k, raw, slots, 136, 0, 4, comp, dst[::2])
        )

    def test_graph_changed_bits_slots_and_side_stream(self):
        k, raw, slots, comp, dst, out = fixture()
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        producer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(producer):
            write(k, raw, slots, comp, dst, out)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                write(k, raw, slots, comp, dst, out)
        torch.cuda.current_stream().wait_stream(producer)
        for i in range(3):
            k.view(torch.int16).fill_([0x7F81, -1, -32768][i])
            slots.fill_(-1)
            dst.copy_(dst.flip(0))
            old_raw, expected = raw.clone(), out.clone()
            expected.view(-1, 512).index_copy_(0, dst, k)
            producer.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(producer):
                graph.replay()
            consumer.wait_stream(producer)
            with torch.cuda.stream(consumer):
                seen = out.clone()
            torch.cuda.current_stream().wait_stream(consumer)
            self.assertTrue(torch.equal(raw, old_raw))
            self.assertTrue(
                torch.equal(seen.view(torch.int16), expected.view(torch.int16))
            )

    def test_product_produce_exact_and_single_write(self):
        Attention = load_attention()
        for ratio in (1, 2):
            for empty in (False, True):
                k, raw, slots, comp, dst, out = fixture(all_negative=empty)
                meta = SimpleNamespace(
                    M=out.shape[1],
                    slot_mapping=slots,
                    slot_compaction=comp,
                    slot_in_flat=dst,
                    cache_slot_mapping=torch.full(
                        (3, 127), -1, device="cuda", dtype=torch.int64
                    ),
                    cache_gather_lens=torch.tensor(
                        [127, 0, 5], device="cuda", dtype=torch.int32
                    ),
                    prefix_len_max=1,
                    cache_compaction=None,
                )
                cp = SimpleNamespace(
                    cp_rank=0,
                    cp_size=4,
                    swa_replay_start=None,
                    input_lengths_global_host=[139, 37, 1],
                    prefix_lengths_host=[127, 0, 5],
                )
                common = SimpleNamespace(
                    swa_meta=meta, cp_on=True, cp_ctx=cp, any_cont=True, batch_size=3
                )
                qkv = SimpleNamespace(kv_full=k)
                a = Attention()
                (
                    a.compress_ratio,
                    a.swa_bounded_replay,
                    a.head_dim,
                    a.window_size,
                    a.is_kv_source,
                ) = (ratio, False, 512, 128, False)
                a._swa_cp_byte_sliced = lambda: True
                a._pool_raw_u8 = lambda region: raw
                a._swa_entries_per_block = lambda: 136
                a._begin_forward = lambda: None
                a._prefill_common_setup = lambda *args: common
                a._prefill_compute_qkv = lambda *args, **kw: qkv
                events = []
                before = raw.clone()
                original = codec.quantize_and_insert_k_cache_cp_byte_sliced
                full = torch.zeros((1, 72192), device="cuda", dtype=torch.uint8)
                verify_prefix = True

                def gather(**kw):
                    if verify_prefix:
                        self.assertTrue(torch.equal(raw, before))
                    events.append("prefix")
                    # Real prefix decoder; collective transport is outside this single-GPU proof.
                    codec.dequantize_and_gather_k_cache_slots(
                        kw["out"],
                        full.as_strided((1, 136, 528), (72192, 528, 1)),
                        meta.cache_slot_mapping,
                        meta.cache_gather_lens,
                        0,
                    )

                def writer(*args, **kw):
                    events.append("write")
                    return original(*args, **kw)

                with patch.object(
                    codec, "dequantize_and_gather_k_cache_slots_cp_byte_sliced", gather
                ), patch.object(
                    codec, "quantize_and_insert_k_cache_cp_byte_sliced", writer
                ):
                    result = a._prefill_produce(k, None)
                self.assertEqual(events, ["prefix", "write"])
                offset = 0
                for sw, p, n in zip(result[2], [127, 0, 5], [139, 37, 1]):
                    self.assertTrue(torch.equal(sw[:p], torch.zeros_like(sw[:p])))
                    self.assertTrue(
                        torch.equal(
                            sw[p:].view(torch.int16),
                            k[offset : offset + n].view(torch.int16),
                        )
                    )
                    offset += n
                verify_prefix = False
                with patch.object(
                    codec, "dequantize_and_gather_k_cache_slots_cp_byte_sliced", gather
                ):
                    # The unchanged allocation/metadata/caller chain must capture too.
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        a._prefill_produce(k, None)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            captured = a._prefill_produce(k, None)
                    torch.cuda.current_stream().wait_stream(stream)
                    k.view(torch.int16).fill_(0x7F81)
                    slots.fill_(-1)
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        graph.replay()
                    torch.cuda.current_stream().wait_stream(stream)
                    for sw, p in zip(captured[2], [127, 0, 5]):
                        self.assertTrue(torch.all(sw[p:].view(torch.int16) == 0x7F81))
                    # Force a rejected capability gate and exercise the original
                    # zero/index_copy/write route with identical valid outputs.
                    with patch.object(a, "_can_fuse_swa_fresh", return_value=False):
                        legacy = a._prefill_produce(k, None)
                    for got, want in zip(captured[2], legacy[2]):
                        self.assertTrue(
                            torch.equal(got.view(torch.int16), want.view(torch.int16))
                        )
                for name, value in (
                    ("compress_ratio", 0),
                    ("swa_bounded_replay", True),
                ):
                    old = getattr(a, name)
                    setattr(a, name, value)
                    self.assertFalse(a._can_fuse_swa_fresh(qkv, common))
                    setattr(a, name, old)
                cp.swa_replay_start = 0
                self.assertFalse(a._can_fuse_swa_fresh(qkv, common))


if __name__ == "__main__":
    unittest.main()

"""GPU regression tests for the production 656 -> TRT sparse decode adapter.

Run on an explicitly reserved GPU, for example CUDA_VISIBLE_DEVICES=5.
The reference below models the adapter's additional FP8 conversion; it is not
permission to waive the separate strict FlashMLA replacement precision gate.
"""

import importlib
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


def load_backend_class():
    for root in Path(__file__).resolve().parents:
        if (root / "rtp_llm" / "models_py").is_dir():
            sys.path.insert(0, str(root))
            break
    module = importlib.import_module(
        "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.trtllm_sparse_impl"
    )
    return module.TrtllmSparseMlaFp8Op


PAGE, TOPK, LATENT, ROPE = 64, 2048, 512, 64


def fp8_rne_sat(value):
    return value.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


class PackedFixture:
    """Independent packed-cache fixture: no conversion kernel under test reused."""

    def __init__(self, batch=2, queries=1, heads=8, length=2112, seed=914):
        self.batch, self.queries, self.heads, self.length = (
            batch,
            queries,
            heads,
            length,
        )
        self.rows = batch * queries
        self.device = torch.device("cuda:0")
        pages_per_request = math.ceil(length / PAGE)
        self.pages = batch * pages_per_request + 1
        rng = torch.Generator(device=self.device).manual_seed(seed)
        self.table = (
            (torch.randperm(self.pages - 1, device=self.device, generator=rng) + 1)
            .reshape(batch, pages_per_request)
            .int()
        )
        self.request_ids = torch.arange(
            batch, device=self.device, dtype=torch.int32
        ).repeat_interleave(queries)
        self.seq_lens = torch.arange(
            length - queries + 1, length + 1, device=self.device, dtype=torch.int32
        ).repeat(batch)
        self.params = SimpleNamespace(
            batch_indice_d=self.request_ids,
            kvlen_d=self.seq_lens,
            expanded_seq_lens=self.seq_lens,
        )
        self.attn_inputs = SimpleNamespace(
            is_prefill=False, is_target_verify=queries > 1, is_draft_extend=False
        )
        self.q = (
            torch.randn(
                (self.rows, heads, LATENT + ROPE), device=self.device, generator=rng
            )
            * 0.25
        ).bfloat16()
        self.cache = torch.zeros(
            (self.pages, PAGE, 656), device=self.device, dtype=torch.uint8
        )
        flat = self.cache.view(-1, 656)
        payload = fp8_rne_sat(
            torch.randn((flat.shape[0], LATENT), device=self.device, generator=rng)
            * 2.0
        )
        scales = (
            torch.rand((flat.shape[0], 4), device=self.device, generator=rng) * 0.03
            + 0.005
        )
        rope = (
            torch.randn((flat.shape[0], ROPE), device=self.device, generator=rng) * 0.1
        ).bfloat16()
        flat[:, :512].copy_(payload.view(torch.uint8))
        flat[:, 512:528].copy_(scales.contiguous().view(torch.uint8))
        flat[:, 528:].copy_(rope.contiguous().view(torch.uint8))
        self.cache[0].zero_()
        self.topk = torch.full(
            (self.rows, TOPK), -1, device=self.device, dtype=torch.int32
        )
        self.set_counts([65] * self.rows)

    def set_counts(self, counts):
        if len(counts) != self.rows:
            raise ValueError("one count per query is required")
        self.topk.fill_(-1)
        for row, count in enumerate(counts):
            # Distribute the valid prefix's entries throughout the physical
            # index-array columns; the implementation must really compact.
            if count:
                columns = (torch.arange(count, device=self.device) * 997) % TOPK
                logical = torch.arange(count, device=self.device).remainder(self.length)
                self.topk[row, columns] = logical.int()

    def new_op(self, cuda_graph=False):
        op = load_backend_class()(
            num_heads=self.heads,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            qk_nope_head_dim=192,
            page_size=PAGE,
            softmax_extra_scale=1.0,
            top_k=TOPK,
            use_cuda_graph=cuda_graph,
        )
        op.plan(self.params, self.table, self.attn_inputs)
        return op

    def forward(self, op, four_dim=False, fp8_storage=False):
        cache = self.cache.view(torch.float8_e4m3fn) if fp8_storage else self.cache
        cache = cache.unsqueeze(2) if four_dim else cache
        indices = self.topk.unsqueeze(1) if four_dim else self.topk
        return op.forward(self.q, cache, indices, layer_id=0)

    def selected(self, row):
        logical = self.topk[row].long()
        req = self.request_ids[row].long()
        valid = (logical >= 0) & (logical < self.seq_lens[row])
        block = logical.clamp_min(0) // PAGE
        valid &= block < self.table.shape[1]
        page = self.table[req, block.clamp_max(self.table.shape[1] - 1)].long()
        valid &= (page > 0) & (page < self.pages)
        physical = page * PAGE + logical.remainder(PAGE)
        return self.cache.view(-1, 656)[physical[valid]]

    @staticmethod
    def flash_reader_values(packed):
        latent = (
            packed[:, :512]
            .contiguous()
            .view(torch.float8_e4m3fn)
            .bfloat16()
            .reshape(-1, 4, 128)
        )
        # The installed cb10b79 reader rounds the stored FP32 scales to BF16
        # before multiplying, then rounds the product to BF16.
        scales = packed[:, 512:528].contiguous().view(torch.float32).bfloat16()
        latent = (latent * scales[..., None]).bfloat16().flatten(1)
        rope = packed[:, 528:].contiguous().view(torch.bfloat16)
        return torch.cat((latent, rope), dim=-1)

    def reference(self):
        outputs = []
        for row in range(self.rows):
            packed = self.selected(row)
            if packed.shape[0] == 0:
                outputs.append(torch.zeros((self.heads, LATENT), device=self.device))
                continue
            key = fp8_rne_sat(self.flash_reader_values(packed)).float()
            query = fp8_rne_sat(self.q[row]).float()
            outputs.append(((query @ key.T) / 16.0).softmax(-1) @ key[:, :LATENT])
        return torch.stack(outputs).bfloat16()


class TrtllmSparseDecodeGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if (
            not torch.cuda.is_available()
            or torch.cuda.get_device_capability(0)[0] != 10
        ):
            raise unittest.SkipTest("TRT sparse decode requires an available SM10x GPU")
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        load_backend_class()

    def assert_reference(self, fixture, actual):
        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertEqual(tuple(actual.shape), (fixture.rows, fixture.heads, LATENT))
        torch.testing.assert_close(actual, fixture.reference(), atol=0.0008, rtol=0.02)

    @staticmethod
    def capture(fixture, op):
        for _ in range(3):
            fixture.forward(op)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = fixture.forward(op)
        return graph, output

    def test_compact_count_boundaries_and_invalid_rows(self):
        counts = [0, 1, 63, 64, 65, 2047, 2048]
        fixture = PackedFixture(batch=len(counts))
        fixture.set_counts(counts)
        # Empty fake rows may contain uninitialized/nonfinite Q. The adapter
        # must not feed these values into its dummy attention calculation.
        fixture.q[0].fill_(float("nan"))
        op = fixture.new_op()
        actual = fixture.forward(op)
        self.assert_reference(fixture, actual)
        self.assertEqual(op.valid_counts.cpu().tolist(), counts)
        for row, count in enumerate(counts):
            if count:
                expected = fp8_rne_sat(
                    fixture.flash_reader_values(fixture.selected(row))
                )
                self.assertTrue(
                    torch.equal(
                        op.selected_kv[row, :count].view(torch.uint8),
                        expected.view(torch.uint8),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        op.q_fp8[row].view(torch.uint8),
                        fp8_rne_sat(fixture.q[row]).view(torch.uint8),
                    )
                )
            else:
                self.assertEqual(
                    int(op.q_fp8[row].view(torch.uint8).count_nonzero()), 0
                )
                self.assertEqual(
                    int(op.selected_kv[row, 0].view(torch.uint8).count_nonzero()), 0
                )
        torch.testing.assert_close(
            actual[0], torch.zeros_like(actual[0]), atol=0, rtol=0
        )

    def test_supported_heads_and_multi_query_rows(self):
        for heads in (8, 16, 32, 64):
            for queries in (1, 4, 6):
                with self.subTest(heads=heads, queries=queries):
                    fixture = PackedFixture(heads=heads, queries=queries)
                    fixture.set_counts([63 + row % 3 for row in range(fixture.rows)])
                    self.assert_reference(
                        fixture,
                        fixture.forward(
                            fixture.new_op(),
                            four_dim=True,
                            fp8_storage=queries == 4,
                        ),
                    )

    def test_duplicates_invalid_pages_and_causal_filter(self):
        fixture = PackedFixture(batch=3)
        fixture.set_counts([2048, 2048, 2048])
        fixture.topk[0, 0:512] = 17  # Must preserve repeated-token weighting.
        fixture.table[1, 0] = 0
        fixture.table[1, 1] = -1
        fixture.table[2, 0] = fixture.pages + 3
        fixture.seq_lens[2] = 65
        op = fixture.new_op()
        original = fixture.cache.clone()
        self.assert_reference(fixture, fixture.forward(op))
        self.assertTrue(torch.equal(original, fixture.cache))

    def test_forward_does_not_read_gpu_values_on_host(self):
        fixture = PackedFixture()
        op = fixture.new_op()
        fixture.forward(op)  # Exclude one-time compiler/runtime initialization.
        with (
            patch.object(torch.Tensor, "item", side_effect=AssertionError("item sync")),
            patch.object(
                torch.Tensor, "tolist", side_effect=AssertionError("tolist sync")
            ),
            patch.object(torch.Tensor, "cpu", side_effect=AssertionError("cpu sync")),
            patch.object(
                torch.Tensor, "__bool__", side_effect=AssertionError("bool sync")
            ),
            patch.object(
                torch.cuda, "synchronize", side_effect=AssertionError("cuda sync")
            ),
        ):
            actual = fixture.forward(op)
        self.assert_reference(fixture, actual)

    def test_graph_reads_dynamic_metadata(self):
        changes = {
            "seq_lens": lambda f: f.seq_lens.fill_(64),
            "page_table": lambda f: f.table.copy_(f.table.roll(1, dims=1)),
            "request_ids": lambda f: f.request_ids.copy_(f.request_ids.flip(0)),
            "topk_empty_to_nonempty": lambda f: f.set_counts([2048, 0]),
            "q": lambda f: f.q.mul_(3.0),
            "cache": lambda f: f.cache[:, :, 528:].zero_(),
        }
        for name, change in changes.items():
            with self.subTest(changed=name):
                fixture = PackedFixture()
                fixture.set_counts(
                    [0, 2048] if name == "topk_empty_to_nonempty" else [2048, 2048]
                )
                op = fixture.new_op(cuda_graph=True)
                graph, output = self.capture(fixture, op)
                graph.replay()
                before = output.clone()
                change(fixture)
                graph.replay()
                replay = output.clone()
                self.assert_reference(fixture, replay)
                torch.testing.assert_close(
                    replay, fixture.forward(op), atol=0.0008, rtol=0.02
                )
                self.assertFalse(
                    torch.equal(before, replay), "input change was not observed"
                )

    def test_instances_do_not_share_mutable_scratch(self):
        a, b = PackedFixture(seed=117), PackedFixture(seed=119)
        a.set_counts([2048, 0])
        b.set_counts([65, 2048])
        oa, ob = a.new_op(cuda_graph=True), b.new_op(cuda_graph=True)
        for name in (
            "selected_kv",
            "q_fp8",
            "source_indices",
            "physical_indices",
            "valid_counts",
            "trt_seq_lens",
            "output",
            "workspace_buffer",
        ):
            self.assertNotEqual(
                getattr(oa, name).data_ptr(), getattr(ob, name).data_ptr(), name
            )
        ga, ya = self.capture(a, oa)
        gb, yb = self.capture(b, ob)
        # Distinct instances may be captured/executed by distinct runners.
        sa, sb = torch.cuda.Stream(), torch.cuda.Stream()
        ready = torch.cuda.Event()
        ready.record()
        sa.wait_event(ready)
        sb.wait_event(ready)
        for _ in range(10):
            with torch.cuda.stream(sa):
                ga.replay()
                actual_a = ya.clone()
            with torch.cuda.stream(sb):
                gb.replay()
                actual_b = yb.clone()
        sa.synchronize()
        sb.synchronize()
        self.assert_reference(a, actual_a)
        self.assert_reference(b, actual_b)

    def test_single_selected_token_matches_actual_flash_reader_rounding(self):
        from flash_mla import flash_mla_with_kvcache, get_mla_metadata

        fixture = PackedFixture(batch=1, heads=64)
        fixture.set_counts([1])
        physical = int(fixture.table[0, 0]) * PAGE
        indices = torch.full((1, 1, TOPK), -1, dtype=torch.int32, device=fixture.device)
        indices[0, 0, 0] = physical
        metadata, _ = get_mla_metadata()
        flash, _ = flash_mla_with_kvcache(
            q=fixture.q.view(1, 1, 64, 576),
            k_cache=fixture.cache.unsqueeze(2),
            block_table=fixture.table,
            cache_seqlens=None,
            head_dim_v=LATENT,
            tile_scheduler_metadata=metadata,
            num_splits=None,
            softmax_scale=1 / 16,
            is_fp8_kvcache=True,
            indices=indices,
        )
        actual_reader = flash.view(1, 64, LATENT)
        independent_reader = fixture.flash_reader_values(fixture.selected(0))[
            :, :LATENT
        ]
        torch.testing.assert_close(
            actual_reader,
            independent_reader[:, None].expand_as(actual_reader),
            atol=0,
            rtol=0,
        )
        actual_trt = fixture.forward(fixture.new_op())
        torch.testing.assert_close(
            actual_trt, fp8_rne_sat(actual_reader).bfloat16(), atol=0, rtol=0
        )


if __name__ == "__main__":
    unittest.main()

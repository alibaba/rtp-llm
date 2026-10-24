"""Compatibility edges for batch preparation, independent of model weights."""

import copy
import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8.decode._fused_prepare_meta_triton import (
    fill_state_pool_slots,
    fused_update_decode_meta_pure,
    try_snapshot_block_tables,
)
from rtp_llm.models_py.speculative._dspark_prepare_triton import (
    try_map_context_rows,
    try_prepare_proposal,
)
from rtp_llm.models_py.speculative.dspark_proposer_mixin import map_context_rows


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class BoundaryCompatibilityTest(unittest.TestCase):
    def test_variable_batch_and_proposal_widths(self):
        batches = (0, 1, 2, 3, 7, 8, 16, 31, 32, 63, 64, 65, 127, 128, 129, 257)
        for batch in batches:
            for width in (1, 2, 3, 5, 6, 8):
                ids = torch.arange(batch * width, device="cuda", dtype=torch.int32)
                prefix = torch.arange(batch, device="cuda", dtype=torch.int64) * 127
                actual = try_prepare_proposal(
                    ids, prefix, width=width, noise_token_id=9
                )
                expected = torch.full(
                    (batch, width), 9, device="cuda", dtype=torch.int32
                )
                expected[:, 0] = ids.reshape(batch, width)[:, 0]
                self.assertTrue(torch.equal(actual[0], expected))
                self.assertTrue(
                    torch.equal(
                        actual[1], prefix[:, None] + torch.arange(width, device="cuda")
                    )
                )
                self.assertTrue(torch.equal(actual[3], prefix > 0))
                if not batch:
                    continue
                # Include full/partial acceptance and graph-padding requests.
                lengths = torch.arange(batch, device="cuda", dtype=torch.int64) % (
                    width + 1
                )
                starts = lengths.cumsum(0) - lengths
                ends = prefix + lengths
                rows = batch * width + 2
                with patch.dict(os.environ, {"DSV4_FUSED_DSPARK_COMMIT_ROWS": "0"}):
                    reference = map_context_rows(starts, lengths, ends, rows)
                fused = try_map_context_rows(starts, lengths, ends, rows)
                for a, b in zip(fused, reference):
                    self.assertTrue(torch.equal(a, b), (batch, width))

    def test_context_limits_and_host_device_metadata(self):
        from rtp_llm.models_py.modules.dsv41.fp8.decode.decode_attn_metadata import (
            allocate_decode_metadata_fp8,
            update_decode_metadata_in_place_fp8,
        )

        for limit in (128, 4096, 32768, 65536, 131072, 1048576):
            for width in (1, 3, 8):
                for flags in ({}, {"is_prefill": True}, {"is_target_verify": True}):
                    meta = allocate_decode_metadata_fp8(
                        4, width, 128, 128, limit, [], 0, torch.device("cuda")
                    )
                    starts = torch.tensor(
                        [-1, 0, limit - 1, limit + 99], device="cuda", dtype=torch.int64
                    )
                    inputs = SimpleNamespace(
                        prefix_lengths_device=starts,
                        sequence_lengths_plus_1_device=starts + 1,
                        prefix_lengths=torch.full((4,), -999),
                        sequence_lengths=torch.full((4,), -999),
                        **flags,
                    )
                    update_decode_metadata_in_place_fp8(
                        meta, inputs, require_device_metadata=True
                    )
                    expected = starts.int().clamp(0, limit - width)
                    expected = expected[:, None] + torch.arange(
                        width, device="cuda", dtype=torch.int32
                    )
                    self.assertTrue(
                        torch.equal(meta.position_ids, expected.flatten()),
                        (limit, width, flags),
                    )

    def test_snapshot_pool_subsets_and_ragged_widths(self):
        for pool_count in (1, 2, 9):
            srcs, dsts, refs = {}, {}, {}
            for key in range(pool_count):
                srcs[key] = torch.arange(
                    5 * (key + 2), device="cuda", dtype=torch.int32
                ).reshape(5, key + 2)
                dsts[key] = torch.full((7, 7), -99, device="cuda", dtype=torch.int32)
                refs[key] = dsts[key].clone()
                rows, cols = 5, min(key + 2, 7)
                refs[key][3:].zero_()
                refs[key][:rows, :cols].copy_(srcs[key][:rows, :cols])
                refs[key][:rows, cols:].zero_()
            self.assertTrue(try_snapshot_block_tables(dsts, srcs, 3))
            for key in srcs:
                self.assertTrue(torch.equal(dsts[key], refs[key]), (pool_count, key))

    def test_unsupported_inputs_preserve_reference_route(self):
        ids = torch.arange(30, device="cuda", dtype=torch.int64)[::2]
        prefix = torch.arange(3, device="cuda", dtype=torch.int32)
        self.assertIsNone(try_prepare_proposal(ids, prefix, width=5, noise_token_id=1))
        self.assertIsNone(
            try_prepare_proposal(ids.float(), prefix, width=5, noise_token_id=1)
        )
        self.assertIsNone(
            try_prepare_proposal(ids.cpu(), prefix.cpu(), width=5, noise_token_id=1)
        )
        with patch.dict(os.environ, {"DSV4_FUSED_DSPARK_QUERY_INPUTS": "0"}):
            self.assertIsNone(
                try_prepare_proposal(
                    ids.contiguous(), prefix, width=5, noise_token_id=1
                )
            )
        src = torch.ones((2, 3), device="cuda", dtype=torch.int64)
        dst = torch.zeros_like(src)
        self.assertFalse(try_snapshot_block_tables({0: dst}, {0: src}, 2))
        self.assertTrue((dst == 0).all().item())
        self.assertFalse(try_snapshot_block_tables({0: dst.cpu()}, {0: src.cpu()}, 2))
        self.assertTrue(try_snapshot_block_tables({}, {}, 0))

    def test_aux_outputs_match_original_with_strides_and_ratio_subsets(self):
        from rtp_llm.models_py.modules.dsv41.fp8.decode.decode_attn_metadata import (
            allocate_decode_metadata_fp8,
        )

        original = None
        reference = os.environ.get("BOUNDARY_REFERENCE")
        if reference:
            spec = importlib.util.spec_from_file_location(
                "original_prepare_meta", Path(reference) / "prepare_meta.py"
            )
            original = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(original)
        cases = [
            (b, (1, 2, 3, 5, 6, 8)[i % 6], 128, 0, [])
            for i, b in enumerate((0, 1, 3, 7, 32, 63, 64, 65, 128, 129, 257))
        ]
        cases += [
            (3, q, w, k, ratios)
            for q in (1, 6)
            for w, k, ratios in (
                (3, 33, []),
                (16, 1, []),
                (128, 64, [4]),
                (128, 64, [128]),
                (128, 64, [4, 128]),
            )
        ]
        for batch, width, window, topk, ratios in cases:
            ref = allocate_decode_metadata_fp8(
                batch + 2, width, window, 128, 65536, ratios, topk, torch.device("cuda")
            )
            ref.topk_buffer_compressed.fill_(123)
            new = copy.deepcopy(ref)
            # .copy_ originally accepted strided destination views as well.
            for key in (
                "position_ids",
                "position_ids_long",
                "decode_seq_start_per_req",
            ):
                value = getattr(new, key)
                if value is not None:
                    setattr(new, key, value.repeat_interleave(2)[::2])
            new.topk_buffer_compressed = new.topk_buffer_compressed.repeat_interleave(
                2, -1
            )[..., ::2]
            if topk == 1:
                ref.position_ids_long = new.position_ids_long = None
                ref.decode_seq_start_per_req = new.decode_seq_start_per_req = None
            starts = (
                torch.arange(batch, device="cuda", dtype=torch.int32) * 127
            ).clamp(max=65536 - width)
            if batch > 1:
                starts[-1] = 65536 - width
            expected_pos = (
                starts[:, None] + torch.arange(width, device="cuda", dtype=torch.int32)
            ).flatten()
            ref.position_ids[: batch * width].copy_(expected_pos)
            if ref.position_ids_long is not None:
                ref.position_ids_long[: batch * width].copy_(expected_pos.long())
            if ref.decode_seq_start_per_req is not None:
                ref.decode_seq_start_per_req[:batch].copy_(starts)
            ref.topk_buffer_compressed[:batch].fill_(-1)
            if original is not None:
                original.fused_update_decode_meta_pure(ref, starts, 65536)
            fused_update_decode_meta_pure(new, starts, 65536)

            def check(a, b):
                if isinstance(a, torch.Tensor):
                    self.assertTrue(torch.equal(a, b), (batch, width, ratios))
                elif isinstance(a, dict):
                    for key in a:
                        check(a[key], b[key])

            keys = (
                vars(ref)
                if original is not None
                else (
                    "position_ids",
                    "position_ids_long",
                    "decode_seq_start_per_req",
                    "topk_buffer_compressed",
                )
            )
            for key in keys:
                check(getattr(ref, key), getattr(new, key))

    def test_state_fallback_for_host_indices_and_float_tables(self):
        from rtp_llm.models_py.modules.dsv41.fp8.decode.decode_attn_metadata import (
            _update_compressor_state_slot_mappings,
        )

        for host_indices, float_table, host_table in (
            (True, False, False),
            (False, True, False),
            (True, False, True),
        ):
            device = "cpu" if host_indices else "cuda"
            positions = torch.arange(6, device=device, dtype=torch.int64)
            requests = torch.arange(2, device=device).repeat_interleave(3)
            table = torch.arange(1, 9, device="cpu" if host_table else "cuda").reshape(
                2, 4
            )
            if float_table:
                table = table.float()
            out = torch.full((6,), -99, device=table.device, dtype=torch.int64)
            self.assertFalse(
                fill_state_pool_slots(table, positions, requests, out, 2, 4)
            )
            meta = SimpleNamespace(
                q_len_per_req=3,
                position_ids_long=positions,
                req_id_per_token_long=requests,
                pool_block_tables={"state": table},
                compressor_state_slot_mappings={"state": out},
                paged_pool_tokens_per_block={"state": 4},
            )
            if host_indices and not host_table:
                # This mixture also fails in the original Torch expression;
                # keep its error instead of launching with inaccessible pointers.
                with self.assertRaisesRegex(RuntimeError, "same device"):
                    _update_compressor_state_slot_mappings(meta, 2, {"state": 2})
                continue
            _update_compressor_state_slot_mappings(meta, 2, {"state": 2})
            expected = (
                table.long()[requests, (positions // 4) % 4] * 2
                + positions.to(table.device) % 2
            )
            self.assertTrue(torch.equal(out, expected))

    def test_state_slots_preserve_strided_vectors(self):
        table = torch.arange(1, 22, device="cuda", dtype=torch.int32).view(3, 7)
        positions = torch.arange(36, device="cuda", dtype=torch.int64)[::2]
        request_storage = torch.zeros(36, device="cuda", dtype=torch.int64)
        requests = request_storage[::2]
        requests.copy_(torch.arange(18, device="cuda") % 3)
        requests[::3] = -1
        storage = torch.full((40,), -99, device="cuda", dtype=torch.int64)
        out = storage[::2]
        fill_state_pool_slots(table, positions, requests, out, 4, 8)
        block = table[requests, (positions // 8) % 7].long()
        self.assertTrue(torch.equal(out[:18], block * 4 + positions % 4))
        self.assertTrue((out[18:] == -99).all().item())
        self.assertTrue((storage[1::2] == -99).all().item())

    def test_cross_table_alias_uses_ordered_fallback(self):
        x = torch.arange(12, device="cuda", dtype=torch.int32).view(3, 4)
        y = x.clone() + 100
        self.assertFalse(
            try_snapshot_block_tables({"a": x, "b": y}, {"a": y, "b": x}, 3)
        )
        from rtp_llm.models_py.modules.dsv41.fp8.decode.decode_attn_metadata import (
            allocate_decode_metadata_fp8,
            update_decode_metadata_in_place_fp8,
        )

        expected = y.clone()
        meta = allocate_decode_metadata_fp8(3, 1, 3, 128, 64, [], 0, x.device)
        meta.pool_block_tables = {"a": x, "b": y}
        update_decode_metadata_in_place_fp8(
            meta,
            torch.zeros(3, device=x.device, dtype=torch.int32),
            paged_block_tables={"a": y, "b": x},
            paged_pool_entries_per_block={"a": 2, "b": 2},
            paged_pool_tokens_per_block={"a": 2, "b": 2},
        )
        self.assertTrue(torch.equal(x, expected))
        self.assertTrue(torch.equal(y, expected))

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA devices required")
    def test_helpers_use_tensor_device(self):
        with torch.cuda.device(1):
            ids = torch.arange(15, device="cuda", dtype=torch.int32)
            prefix = torch.tensor([0, 7, 19], device="cuda", dtype=torch.int32)
        with torch.cuda.device(0):
            actual = try_prepare_proposal(ids, prefix, width=5, noise_token_id=1)
            self.assertEqual(torch.cuda.current_device(), 0)
        expected = prefix.long()[:, None] + torch.arange(5, device=prefix.device)
        self.assertTrue(torch.equal(actual[1], expected))
        with torch.cuda.device(0):
            starts = torch.tensor([0, 5, 10], device=prefix.device, dtype=torch.int64)
            lengths = torch.full_like(starts, 5)
            req, pos = try_map_context_rows(starts, lengths, prefix.long() + 5, 15)
            self.assertTrue(
                torch.equal(
                    req,
                    torch.arange(3, device=prefix.device).repeat_interleave(5).int(),
                )
            )
            self.assertTrue(torch.equal(pos, expected.flatten().int()))
            table = torch.ones((3, 4), device=prefix.device, dtype=torch.int32)
            output = torch.empty(15, device=prefix.device, dtype=torch.int64)
            self.assertTrue(
                fill_state_pool_slots(table, pos.long(), req.long(), output, 4, 8)
            )
            dst = torch.zeros_like(table)
            self.assertTrue(try_snapshot_block_tables({0: dst}, {0: table}, 3))
            self.assertTrue(torch.equal(dst, table))
            self.assertEqual(torch.cuda.current_device(), 0)


if __name__ == "__main__":
    unittest.main()

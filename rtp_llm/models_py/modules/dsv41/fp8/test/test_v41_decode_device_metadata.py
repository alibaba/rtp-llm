"""Device length mirrors stay authoritative across decode graph replays."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8.decode.decode_attn_metadata import (
    _build_position_ids_2d,
    _build_start_pos_from_attention_inputs,
)


class DecodeDeviceMetadataTest(unittest.TestCase):

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_global_metadata_reuse_preserves_architecture_and_compatibility(self):
        from dataclasses import replace

        from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import (
            _decode_global_metadata_views,
        )
        from rtp_llm.models_py.modules.dsv41.fp8.decode.decode_attn_metadata import (
            allocate_decode_metadata_fp8,
            update_decode_metadata_in_place_fp8,
        )

        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("requires an SM90 GPU")
        meta = allocate_decode_metadata_fp8(
            max_batch_size=2,
            q_len=6,
            window_size=128,
            head_dim=512,
            max_seq_len=512,
            compress_ratios=[0],
            index_topk=0,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        update_decode_metadata_in_place_fp8(
            meta, torch.tensor([7, 255], device="cuda", dtype=torch.int32)
        )
        expected = (meta.req_id_per_token.long(), meta.start_pos.long())
        with patch.object(
            torch.cuda,
            "get_device_capability",
            side_effect=AssertionError("no per-call architecture probe"),
        ):
            ordinary = _decode_global_metadata_views(meta, 12, 2, sm90=False)
            reused = _decode_global_metadata_views(meta, 12, 2, sm90=True)
        for actual, reference in zip(ordinary, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        self.assertNotEqual(
            ordinary[0].data_ptr(), meta.req_id_per_token_long.data_ptr()
        )
        self.assertNotEqual(
            ordinary[1].data_ptr(), meta.decode_seq_start_per_req.data_ptr()
        )
        self.assertEqual(reused[0].data_ptr(), meta.req_id_per_token_long.data_ptr())
        self.assertEqual(reused[1].data_ptr(), meta.decode_seq_start_per_req.data_ptr())

        # Existing int64/custom metadata and absent companions retain the
        # old conversion contract instead of requiring new owner fields.
        alternatives = (
            replace(meta, req_id_per_token=expected[0], start_pos=expected[1]),
            replace(meta, req_id_per_token_long=None, decode_seq_start_per_req=None),
            SimpleNamespace(
                req_id_per_token=meta.req_id_per_token, start_pos=meta.start_pos
            ),
        )
        for alternate in alternatives:
            actual = _decode_global_metadata_views(alternate, 12, 2, sm90=True)
            original = (
                alternate.req_id_per_token[:12].long(),
                alternate.start_pos[:2].long(),
            )
            for value, reference in zip(actual, original):
                torch.testing.assert_close(value, reference, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_global_metadata_dense_views_follow_changed_graph_inputs(self):
        from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import (
            _decode_global_metadata_views,
        )
        from rtp_llm.models_py.modules.dsv41.fp8.decode.decode_attn_metadata import (
            allocate_decode_metadata_fp8,
            update_decode_metadata_in_place_fp8,
        )

        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("requires an SM90 GPU")
        for q_len in (6, 8):
            with self.subTest(q_len=q_len):
                meta = allocate_decode_metadata_fp8(
                    max_batch_size=4,
                    q_len=q_len,
                    window_size=128,
                    head_dim=512,
                    max_seq_len=512,
                    compress_ratios=[0],
                    index_topk=0,
                    device=torch.device("cuda"),
                )
                starts = torch.zeros(4, device="cuda", dtype=torch.int32)
                for _ in range(3):
                    update_decode_metadata_in_place_fp8(meta, starts)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    update_decode_metadata_in_place_fp8(meta, starts)
                    request_view, start_view = _decode_global_metadata_views(
                        meta, 4 * q_len, 4, sm90=True
                    )
                    observed_requests = request_view.clone()
                    observed_starts = start_view.clone()
                self.assertEqual(
                    request_view.data_ptr(), meta.req_id_per_token_long.data_ptr()
                )
                self.assertEqual(
                    start_view.data_ptr(), meta.decode_seq_start_per_req.data_ptr()
                )
                for values in (
                    [0, 127, 255, 399],
                    [1, 128, 256, 511],
                    [7, 0, 257, 400],
                ):
                    starts.copy_(torch.tensor(values, device="cuda", dtype=torch.int32))
                    graph.replay()
                    torch.testing.assert_close(
                        observed_requests,
                        torch.arange(4, device="cuda").repeat_interleave(q_len),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        observed_starts,
                        starts.clamp(0, 512 - q_len).long(),
                        rtol=0,
                        atol=0,
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_global_metadata_packed_requests_update_and_starts_keep_old_cast(self):
        from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import (
            _decode_global_metadata_views,
        )
        from rtp_llm.models_py.modules.dsv41.fp8.decode.packed_metadata import (
            allocate_packed_decode_metadata,
            update_packed_decode_metadata,
        )

        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("requires an SM90 GPU")
        meta = allocate_packed_decode_metadata(
            max_batch_size=4,
            q_len=6,
            window_size=128,
            head_dim=512,
            max_seq_len=512,
            compress_ratios=[0],
            index_topk=0,
            device=torch.device("cuda"),
        )
        lengths = torch.tensor([4, 8, 5, 7], device="cuda", dtype=torch.int32)
        starts = torch.tensor([7, 127, 255, 399], device="cuda", dtype=torch.int32)
        inputs = SimpleNamespace(
            input_lengths_device=lengths, prefix_lengths_device=starts
        )
        for _ in range(3):
            update_packed_decode_metadata(meta, inputs, {}, {}, {})
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            update_packed_decode_metadata(meta, inputs, {}, {}, {})
            requests, prefixes = _decode_global_metadata_views(meta, 24, 4, sm90=True)
            observed_requests = requests.clone()
            observed_prefixes = prefixes.clone()
        self.assertEqual(requests.data_ptr(), meta.req_id_per_token_long.data_ptr())
        self.assertNotEqual(
            prefixes.data_ptr(), meta.decode_seq_start_per_req.data_ptr()
        )
        for sizes in ([4, 8, 5, 7], [8, 4, 7, 5], [6, 6, 6, 6]):
            lengths.copy_(torch.tensor(sizes, device="cuda", dtype=torch.int32))
            starts.add_(1)
            graph.replay()
            expected_requests = [r for r, size in enumerate(sizes) for _ in range(size)]
            torch.testing.assert_close(
                observed_requests,
                torch.tensor(expected_requests, device="cuda", dtype=torch.int64),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(observed_prefixes, starts.long(), rtol=0, atol=0)

        # Selector-only boundary: never launch a cache read at this illegal
        # prefix. Preserve the old int32 truncation even with an int64 mirror.
        overflow = torch.full((4,), 2**31 + 9, device="cuda", dtype=torch.int64)
        meta.start_pos.copy_(overflow.int())
        meta.decode_seq_start_per_req.copy_(overflow)
        _, actual = _decode_global_metadata_views(meta, 24, 4, sm90=True)
        torch.testing.assert_close(actual, meta.start_pos.long(), rtol=0, atol=0)

    def test_graph_replay_requires_explicit_live_request_count(self):
        from rtp_llm.models_py.modules.dsv41.fp8.decode import decode_fmha_impl as fmha

        impl = fmha.DSv4DecodeFmhaImplFP8.__new__(fmha.DSv4DecodeFmhaImplFP8)
        impl.metadata = SimpleNamespace()
        with patch.object(fmha, "update_decode_metadata_in_place_fp8") as update:
            with self.assertRaisesRegex(ValueError, "live request count"):
                impl.prepare_cuda_graph(SimpleNamespace())
            update.assert_not_called()

    def test_host_only_eager_compatibility(self):
        for q_len, flags in (
            (1, {}),
            (3, {"is_prefill": True}),
            (6, {"is_target_verify": True}),
        ):
            inputs = SimpleNamespace(
                sequence_lengths=torch.tensor([-1, 5, 260]),
                prefix_lengths=torch.tensor([-1, 5, 260]),
                **flags,
            )
            actual = _build_start_pos_from_attention_inputs(
                inputs, torch.device("cpu"), 256, q_len
            )
            torch.testing.assert_close(
                actual, torch.tensor([0, 5, 256 - q_len], dtype=torch.int32)
            )
            with self.assertRaisesRegex(ValueError, "requires device"):
                _build_start_pos_from_attention_inputs(
                    inputs,
                    torch.device("cpu"),
                    256,
                    q_len,
                    require_device_metadata=True,
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_validity_clears_capacity_tail_and_regrows_at_fixed_address(self):
        from rtp_llm.models_py.modules.dsv41.fp8.decode.decode_attn_metadata import (
            allocate_decode_metadata_fp8,
            update_decode_metadata_in_place_fp8,
        )

        device = torch.device("cuda", torch.cuda.current_device())
        for q_len in (1, 3, 6):
            meta = allocate_decode_metadata_fp8(
                max_batch_size=4, q_len=q_len, window_size=128,
                head_dim=512, max_seq_len=256, compress_ratios=[4, 128],
                index_topk=512, device=device,
            )
            validity_ptr = meta.valid_requests.data_ptr()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                observed_valid = meta.valid_requests.clone()
                observed_positions = meta.position_ids.clone()
            for prepared, live in ((4, 4), (4, 1), (2, 2), (4, 0), (4, 3)):
                starts = torch.arange(prepared, device=device, dtype=torch.int32) + 128
                update_decode_metadata_in_place_fp8(meta, starts, valid_request_count=live)
                graph.replay()
                self.assertEqual(meta.valid_requests.data_ptr(), validity_ptr)
                torch.testing.assert_close(observed_valid, torch.arange(4, device=device) < live)
                torch.testing.assert_close(
                    observed_positions[:prepared * q_len].view(prepared, q_len),
                    starts[:, None] + torch.arange(q_len, device=device),
                    check_dtype=False,
                )
                torch.testing.assert_close(meta.start_pos[:prepared], starts)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_device_mirrors_update_without_host_values_or_source_reuse(self):
        device = torch.device("cuda", torch.cuda.current_device())
        for batch in (2, 4):
            for q_len, flags in (
                (1, {}),
                (3, {"is_prefill": True}),
                (6, {"is_target_verify": True}),
            ):
                with self.subTest(batch=batch, q_len=q_len):
                    mirror = torch.zeros(batch, device=device, dtype=torch.int32)
                    offset = int(q_len == 1)
                    name = (
                        "sequence_lengths_plus_1_device"
                        if offset
                        else "prefix_lengths_device"
                    )
                    inputs = SimpleNamespace(**flags, **{name: mirror})
                    # No host length fields exist. Mirror updates and reads
                    # remain ordered on this stream without a host wait.
                    positions = torch.empty_like(mirror)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = _build_position_ids_2d(positions, q_len, device)
                    observed = []
                    expected = []
                    for starts in (
                        list(range(batch)),
                        [129] * (batch - 1) + [0],
                        [255] * batch,
                    ):
                        values = torch.tensor(starts, device=device, dtype=torch.int32)
                        mirror.copy_(values + offset)
                        positions.copy_(
                            _build_start_pos_from_attention_inputs(
                                inputs,
                                device,
                                256,
                                q_len,
                                require_device_metadata=True,
                            )
                        )
                        graph.replay()
                        observed.append(output.clone())
                        expected.append(
                            values.clamp(0, 256 - q_len)[:, None]
                            + torch.arange(q_len, device=device, dtype=torch.int32)
                        )
                    # Compare only after multiple queued source overwrites and
                    # replays. A copy from stale host state fails this check.
                    for actual, reference in zip(observed, expected):
                        torch.testing.assert_close(actual, reference)


if __name__ == "__main__":
    unittest.main()

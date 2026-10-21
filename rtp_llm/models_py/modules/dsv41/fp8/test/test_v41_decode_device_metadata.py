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
    def test_graph_replay_requires_explicit_live_request_count(self):
        from rtp_llm.models_py.modules.dsv41.fp8.decode import decode_fmha_impl as fmha

        impl = fmha.DSv4DecodeFmhaImplFP8.__new__(fmha.DSv4DecodeFmhaImplFP8)
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

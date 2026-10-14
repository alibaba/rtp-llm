"""Device length mirrors stay authoritative across decode graph replays."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv41.fp8.decode.decode_attn_metadata import (
    _build_position_ids_2d,
    _build_start_pos_from_attention_inputs,
)


class DecodeDeviceMetadataTest(unittest.TestCase):
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

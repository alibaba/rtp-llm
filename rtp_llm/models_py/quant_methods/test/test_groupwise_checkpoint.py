import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rtp_llm.models_py.quant_methods.groupwise_checkpoint import (
    GroupwiseCheckpointLayout,
    PackedGroupwiseWeights,
)


def checkpoint(layout):
    k, n, bits = layout.input_size, layout.output_size, layout.bits
    values = torch.arange(k * n).reshape(k, n) % (1 << bits)
    zeros = torch.arange(k // layout.group_size * n).reshape(-1, n) % ((1 << bits) - 1)
    scales = (torch.arange(zeros.numel()).reshape(zeros.shape) % 7 + 1).half() / 16

    def pack(tensor):
        groups = tensor.long().reshape(tensor.shape[0], -1, layout.pack_factor)
        result = torch.zeros(groups.shape[:2], dtype=torch.int64)
        for index in range(layout.pack_factor):
            result |= groups[:, :, index] << (bits * index)
        return result.to(torch.int32)

    if layout.method == "awq":
        # Inverse of the exporter interleave, expressed as explicit indices.
        order = [0, 2, 4, 6, 1, 3, 5, 7]
        packed_weight = pack(values.reshape(-1, 8)[:, order].reshape(k, n))
        packed_zero = pack(zeros.reshape(-1, 8)[:, order].reshape(zeros.shape))
        actual_zeros = zeros
    else:
        packed_weight = pack(values.T).T.contiguous()
        packed_zero = pack(zeros)
        actual_zeros = zeros + 1
    dense = (
        values - actual_zeros.repeat_interleave(layout.group_size, 0)
    ) * scales.repeat_interleave(layout.group_size, 0)
    return {"qweight": packed_weight, "qzeros": packed_zero, "scales": scales}, dense


def reconstruct(module):
    packed, offsets, scales = module.convert(lambda weight, bits: weight)
    if module.layout.bits == 4:
        unsigned = packed.to(torch.uint8)
        values = (
            torch.stack([unsigned & 15, unsigned >> 4], -1)
            .reshape(packed.shape[0], -1)
            .short()
        )
        values = torch.where(values >= 8, values - 16, values)
    else:
        values = packed.short()
    return values * scales.repeat_interleave(
        module.layout.group_size, 0
    ) + offsets.repeat_interleave(module.layout.group_size, 0)


class GroupwiseCheckpointTest(unittest.TestCase):
    def test_real_safetensors_all_formats_and_tp_axes(self):
        for method in ("awq", "gptq"):
            for bits in (4, 8):
                layout = GroupwiseCheckpointLayout(method, bits, 16, 64, 32)
                tensors, dense = checkpoint(layout)
                with tempfile.TemporaryDirectory() as directory:
                    path = str(Path(directory) / "model.safetensors")
                    save_file(tensors, path)
                    for dim in (0, 1):
                        for rank in range(2):
                            with self.subTest(
                                method=method, bits=bits, dim=dim, rank=rank
                            ):
                                module = PackedGroupwiseWeights(
                                    layout.local_layout(dim, 2, rank)
                                )
                                with safe_open(
                                    path, framework="pt", device="cpu"
                                ) as handle:
                                    for name in handle.keys():
                                        source = handle.get_slice(name)
                                        spec = layout.read_spec(
                                            name,
                                            tuple(source.get_shape()),
                                            dim,
                                            2,
                                            rank,
                                        )
                                        module.load_checkpoint(
                                            {name: spec.materialize(source)},
                                            layout,
                                            partition_dim=dim,
                                            tp_size=2,
                                            tp_rank=rank,
                                        )
                                torch.testing.assert_close(
                                    reconstruct(module),
                                    dense.chunk(2, dim)[rank],
                                    rtol=0,
                                    atol=0,
                                )

    def test_raw_and_presliced_are_equal(self):
        layout = GroupwiseCheckpointLayout("gptq", 4, 16, 64, 32)
        tensors, _ = checkpoint(layout)
        raw = PackedGroupwiseWeights(layout.local_layout(0, 2, 1))
        raw.load_checkpoint(tensors, layout, partition_dim=0, tp_size=2, tp_rank=1)
        sliced = PackedGroupwiseWeights(raw.layout)
        sliced.load_checkpoint(
            {
                name: layout.read_spec(name, tuple(value.shape), 0, 2, 1).materialize(
                    value
                )
                for name, value in tensors.items()
            },
            layout,
            partition_dim=0,
            tp_size=2,
            tp_rank=1,
        )
        for name in tensors:
            torch.testing.assert_close(getattr(raw, name), getattr(sliced, name))

    def test_gqa_local_fusion_preserves_projection_order(self):
        for method in ("awq", "gptq"):
            for bits in (4, 8):
                projections, expected = [], []
                for n, shards, rank in ((64, 4, 3), (32, 2, 1), (32, 2, 1)):
                    layout = GroupwiseCheckpointLayout(method, bits, 16, 64, n)
                    tensors, dense = checkpoint(layout)
                    local = PackedGroupwiseWeights(layout.local_layout(1, shards, rank))
                    local.load_checkpoint(tensors, layout, tp_size=shards, tp_rank=rank)
                    projections.append(local)
                    expected.append(dense.chunk(shards, 1)[rank])
                fused = PackedGroupwiseWeights.fuse(projections)
                torch.testing.assert_close(
                    reconstruct(fused), torch.cat(expected, 1), rtol=0, atol=0
                )

    def test_rejects_incomplete_duplicate_wrong_rank_and_dtype(self):
        layout = GroupwiseCheckpointLayout("awq", 4, 16, 64, 32)
        tensors, _ = checkpoint(layout)
        module = PackedGroupwiseWeights(layout.local_layout(1, 2, 0))
        with self.assertRaisesRegex(RuntimeError, "Missing"):
            module.convert(lambda weight, bits: weight)
        spec = layout.read_spec("qweight", tuple(tensors["qweight"].shape), 1, 2, 1)
        with self.assertRaisesRegex(ValueError, "ReadSpec"):
            module.load_checkpoint(
                {"qweight": spec.materialize(tensors["qweight"])}, layout, tp_size=2
            )
        with self.assertRaisesRegex(ValueError, "dtype"):
            module.load_checkpoint(
                {"scales": tensors["scales"].float()}, layout, tp_size=2
            )
        module.load_checkpoint(tensors, layout, tp_size=2)
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            module.load_checkpoint(tensors, layout, tp_size=2)

    def test_rejects_group_or_word_split(self):
        layout = GroupwiseCheckpointLayout("gptq", 4, 64, 64, 32)
        with self.assertRaisesRegex(ValueError, "complete quantization groups"):
            layout.local_layout(0, 2, 0)
        with self.assertRaisesRegex(ValueError, "complete zero-point words"):
            layout.local_layout(1, 8, 0)


if __name__ == "__main__":
    unittest.main()

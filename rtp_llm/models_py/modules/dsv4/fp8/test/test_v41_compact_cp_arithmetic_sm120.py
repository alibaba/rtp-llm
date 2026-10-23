"""Check owner GEMM shapes against captured raw-X producer arithmetic."""

import json
import os
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4.fp8 import _v41_fp4_triton as codec
from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import (
    compress_pairs,
    rms_norm,
    rope_only,
)
from rtp_llm.models_py.modules.dsv4.fp8.compressor import _linear_bf16_bf16_fp32


def produce(values, scores, weights, fixture):
    ratio = fixture["ratio"]
    eps = fixture["eps"]
    if ratio == 2:
        latent = compress_pairs(
            values.reshape(-1, 2, 512),
            scores.reshape(-1, 2, 512),
            weights["global_norm"],
            eps,
        )
    else:
        latent = rms_norm(values, weights["global_norm"], eps).bfloat16()
    projected_index = F.linear(latent, weights["index_wk"])
    main = rope_only(latent.clone(), fixture["freqs"], fixture["rope_dim"])
    index = rope_only(
        rms_norm(projected_index, weights["index_k_norm"], eps),
        fixture["freqs"],
        fixture["rope_dim"],
    )
    rows = len(latent)
    main_entries = 128 if ratio == 2 else 256
    main_pool = torch.zeros(
        ((rows + main_entries - 1) // main_entries, main_entries, 288),
        dtype=torch.uint8,
        device=values.device,
    )
    index_pool = torch.zeros(
        ((rows + 255) // 256, 256, 68),
        dtype=torch.uint8,
        device=values.device,
    )
    slots = torch.arange(rows, device=values.device)
    codec.quantize_and_insert_k_cache_fp4(main.contiguous(), main_pool, slots)
    codec.quantize_indexer_k_fp4(index.contiguous(), slots, index_pool)
    payload, scales = codec.gather_indexer_k_fp4(index_pool, slots)
    return {
        "latent": latent,
        "projected_index": projected_index,
        "main_bytes": codec.gather_k_cache_bytes_fp4(main_pool, slots),
        "index_payload": payload,
        "index_scales": scales,
    }


def difference(expected, actual):
    a = expected.contiguous().view(torch.uint8)
    b = actual.contiguous().view(torch.uint8)
    bad = a != b
    result = {"exact": not bool(bad.any()), "mismatched_bytes": int(bad.sum())}
    if not result["exact"]:
        result["first_byte_coordinate"] = torch.nonzero(bad)[0].tolist()
        if expected.is_floating_point():
            result["max_abs_difference"] = float(
                (expected.float() - actual.float()).abs().max()
            )
    return result


class CompactCPArithmetic(unittest.TestCase):
    def test_real_producer_partitioning(self):
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        torch.backends.cuda.matmul.allow_tf32 = False
        root = Path(os.environ["DSV41_CP_FIXTURES"])
        paths = sorted(root.glob("layer*-chunk*.pt"))
        self.assertEqual(len(paths), 8)
        report = {"fixtures": [], "passed": True}
        for path in paths:
            fixture = torch.load(path, map_location="cuda", weights_only=True)
            weights = torch.load(
                root / f'weights-layer{fixture["layer"]}.pt',
                map_location="cuda",
                weights_only=True,
            )
            fixture.update(fixture.pop("tensors"))
            x = fixture["x"]
            values = _linear_bf16_bf16_fp32(x, weights["global_wkv"])
            scores = (
                _linear_bf16_bf16_fp32(x, weights["global_wgate"])
                if fixture["ratio"] == 2
                else None
            )
            baseline = produce(values, scores, weights, fixture)
            entry = {"file": path.name, "checks": {}}
            for key in ("main_bytes", "index_payload", "index_scales"):
                entry["checks"]["captured_" + key] = difference(
                    fixture[key], baseline[key]
                )
            if scores is not None:
                entry["checks"]["captured_state_tail"] = difference(
                    fixture["state_tail"], torch.cat((values[-2:], scores[-2:]), dim=1)
                )
            projected = torch.cat(
                [
                    _linear_bf16_bf16_fp32(part, weights["global_wkv"])
                    for part in x.split(512)
                ]
            )
            projected_scores = (
                torch.cat(
                    [
                        _linear_bf16_bf16_fp32(part, weights["global_wgate"])
                        for part in x.split(512)
                    ]
                )
                if scores is not None
                else None
            )
            entry["checks"]["projected_values"] = difference(values, projected)
            if scores is not None:
                entry["checks"]["projected_scores"] = difference(
                    scores, projected_scores
                )
            projected_result = produce(projected, projected_scores, weights, fixture)
            for key, expected in baseline.items():
                entry["checks"]["projected_" + key] = difference(
                    expected, projected_result[key]
                )
            pieces = []
            for start in range(0, 4096, 512):
                local_fixture = dict(fixture)
                local_fixture["freqs"] = fixture["freqs"][
                    start // fixture["ratio"] : (start + 512) // fixture["ratio"]
                ]
                pieces.append(
                    produce(
                        projected[start : start + 512],
                        (
                            projected_scores[start : start + 512]
                            if scores is not None
                            else None
                        ),
                        weights,
                        local_fixture,
                    )
                )
            for key, expected in baseline.items():
                entry["checks"]["packed_" + key] = difference(
                    expected, torch.cat([part[key] for part in pieces])
                )
            if not all(check["exact"] for check in entry["checks"].values()):
                report["passed"] = False
            report["fixtures"].append(entry)
        Path(os.environ["DSV41_CP_PROBE_OUTPUT"]).write_text(
            json.dumps(report, indent=2) + "\n"
        )
        self.assertTrue(
            report["passed"],
            "CP owner arithmetic diverged; inspect the boundary report",
        )


if __name__ == "__main__":
    unittest.main()

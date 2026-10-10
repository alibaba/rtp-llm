"""Hold qualified arithmetic L1 inputs across WKV, norm/RoPE and SWA encoding.

This probe does not execute experts, substitute a full-model result or change
production dispatch. Each stream must reproduce its saved current SWA rows.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch

from rtp_llm.tools.dsv41_attention_replay import (
    NASReader,
    difference,
    native_attention,
    official_attention,
)
from rtp_llm.tools.dsv41_reference_probe import model_args

LAYER = 1
WIDTH = 512
ROPE_WIDTH = 64
ENTRIES = 128
POISON = 173


def rotate_released(module, value, freqs):
    result = value.unsqueeze(0).clone()
    module.apply_rotary_emb(result[..., -ROPE_WIDTH:], freqs)
    return result[0]


def encode_pair(value, start, module, kernel):
    """Compare both codecs on one held BF16 value and a padded planar page."""
    from rtp_llm.models_py.modules.dsv4.fp8 import _v41_swa_triton as swa
    from rtp_llm.tools.dsv41_query_boundary_replay import (
        quantized_difference,
        require_bytes,
    )

    value = value.contiguous()
    rows = len(value)
    stride = ENTRIES * swa.ENTRY_BYTES + 512
    backing = torch.full((2 * stride,), POISON, device="cuda", dtype=torch.uint8)
    before = backing.clone()
    pool = backing.as_strided(
        (2, ENTRIES, swa.ENTRY_BYTES), (stride, swa.ENTRY_BYTES, 1)
    )
    slots = torch.arange(ENTRIES + start, ENTRIES + start + rows, device="cuda")
    operand = value.clone()
    swa.quantize_and_insert_swa_k_cache(value, pool, slots)
    decoded = swa.dequantize_swa_k_cache(pool, slots)
    reference_payload, reference_scales = kernel.act_quant(
        value, module.fp8_block_size, module.scale_fmt, module.scale_dtype
    )
    reference_decoded = value.clone()
    kernel.act_quant(
        reference_decoded,
        module.fp8_block_size,
        module.scale_fmt,
        module.scale_dtype,
        True,
    )
    payload = backing[stride + start * WIDTH : stride + (start + rows) * WIDTH]
    payload = payload.reshape(rows, WIDTH).clone()
    scales_start = stride + ENTRIES * WIDTH + start * 16
    scales = backing[scales_start : scales_start + rows * 16].reshape(rows, 16).clone()
    require_bytes(operand, value, "held SWA codec operand")
    data = {
        "input": operand,
        "input_after": value.clone(),
        "slots": slots,
        "stride": stride,
        "before": before,
        "after": backing.clone(),
        "payload": payload,
        "scales": scales,
        "reference_payload": reference_payload.view(torch.uint8).clone(),
        "reference_scales": reference_scales.view(torch.uint8).clone(),
        "decoded": decoded,
        "reference_decoded": reference_decoded,
    }
    metrics = {
        "payload": quantized_difference(
            reference_payload, payload.view(torch.float8_e4m3fn)
        ),
        "scales": quantized_difference(
            reference_scales, scales.view(torch.float8_e8m0fnu)
        ),
        "decoded": difference(reference_decoded, decoded),
    }
    return data, metrics


def norm_pair(projected, native, released, native_freqs, released_freqs, module):
    from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import (
        fused_rmsnorm_rope,
    )
    from rtp_llm.models_py.modules.dsv4._rope_only_triton import rope_only_inplace

    identity_freqs = torch.ones_like(native_freqs)
    # Identity rotation exposes the fused kernel's own BF16 normalization
    # boundary, including its explicit ROUND_NORM store on the rotary tail.
    normalized = fused_rmsnorm_rope(
        projected.unsqueeze(0),
        native.kv_norm,
        identity_freqs,
        ROPE_WIDTH,
        eps=native.eps,
        round_norm_to_input_dtype=True,
    )[0]
    reference_norm = released.kv_norm(projected)
    fused = fused_rmsnorm_rope(
        projected.unsqueeze(0),
        native.kv_norm,
        native_freqs,
        ROPE_WIDTH,
        eps=native.eps,
        round_norm_to_input_dtype=True,
    )[0]
    split = normalized.unsqueeze(0).clone()
    rope_only_inplace(split[..., -ROPE_WIDTH:], native_freqs)
    common_freqs = rotate_released(module, reference_norm, native_freqs)
    reference = rotate_released(module, reference_norm, released_freqs)
    data = {
        "projection": projected.clone(),
        "identity_freqs": identity_freqs,
        "native_norm": normalized,
        "reference_norm": reference_norm,
        "native_fused": fused,
        "native_split": split[0],
        "reference_common_freqs": common_freqs,
        "reference_rotated": reference,
    }
    metrics = {
        "normalization": difference(reference_norm, normalized),
        "fused_vs_split": difference(split[0], fused),
        "common_freqs": difference(common_freqs, fused),
        "rotated": difference(reference, fused),
        "nope": difference(reference[:, :448], fused[:, :448]),
        "rope": difference(reference[:, 448:], fused[:, 448:]),
    }
    return data, metrics


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "checkpoint",
        "history_qualification",
        "history_report",
        "history_artifact",
        "admission_manifest",
        "report",
    ):
        parser.add_argument(name, type=Path)
    parser.add_argument("--tensor-artifact", type=Path, required=True)
    cli = parser.parse_args()
    if cli.report.exists() or cli.tensor_artifact.exists():
        raise ValueError("KV boundary evidence must be fresh")
    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("KV boundary probe requires leased SM120")
    from rtp_llm.tools.dsv41_block_norm_replay import admit
    from rtp_llm.tools.dsv41_corrected_history_replay import cpu_tree
    from rtp_llm.tools.dsv41_query_boundary_replay import (
        linear_pair,
        require_bytes,
        sha256,
    )

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision("highest")
    manifest, history, raw = admit(cli)
    if manifest["prompt_index"] != 1 or history["prompt_index"] != 1:
        raise ValueError("KV boundary probe requires the pinned arithmetic prompt")
    if raw["identities"] != [
        {"input_ids": [20, 940, 223, 20, 438], "start_pos": 0},
        {"input_ids": [223], "start_pos": 5},
        {"input_ids": [22], "start_pos": 6},
        {"input_ids": [201], "start_pos": 7},
    ]:
        raise ValueError("KV boundary consumed prefixes differ")
    root = cli.checkpoint.resolve()
    sys.path.insert(0, str(root / "inference"))
    module, converter, kernel = [
        importlib.import_module(name) for name in ("model", "convert", "kernel")
    ]
    for loaded in (module, converter, kernel):
        if Path(loaded.__file__).resolve() != root / "inference" / (
            loaded.__name__ + ".py"
        ):
            raise ValueError("Unexpected released KV module")
    args = model_args(module, root, 256)
    config = json.loads((root / "config.json").read_text())["text_config"]
    reader = NASReader(root)
    native = native_attention(args, config, reader, LAYER)
    shards = official_attention(module, converter, args, reader, LAYER)
    released = shards[0]
    if (native.head_dim, native.rope_head_dim, native.compress_ratio) != (512, 64, 0):
        raise ValueError("KV boundary geometry differs")
    weights = {
        "native_wkv": native.wkv.weight,
        "native_scales": native.wkv._sm120_linear.scales,
        "native_packed_scales": native.wkv.weight_scales,
        "native_norm": native.kv_norm,
        "released_wkv": released.wkv.weight,
        "released_scales": released.wkv.scale,
        "released_norm": released.kv_norm.weight,
        "native_freqs": native.freqs_cis,
        "released_freqs": released.freqs_cis,
    }
    before = {name: value.clone() for name, value in weights.items()}
    require_bytes(native.wkv.weight, released.wkv.weight, "WKV weights")
    require_bytes(native.kv_norm, released.kv_norm.weight, "WKV norm weights")
    records, cases = [], []
    cumulative = {"native": [], "official": []}
    own_controls = 0
    for step, identity in enumerate(raw["identities"]):
        start, rows = identity["start_pos"], len(identity["input_ids"])
        nf = native.freqs_cis[start : start + rows].contiguous()
        rf = released.freqs_cis[start : start + rows].contiguous()
        require_bytes(
            raw["native_inputs"][step][LAYER],
            raw["official_inputs"][step][LAYER],
            "L1 inputs",
        )
        for stream in ("native", "official"):
            value = raw[stream + "_inputs"][step][LAYER].cuda().contiguous()
            saved_input = value.clone()
            quant, packed = native.wkv._quantize_input(value)
            linear_scales = (
                packed.clone(memory_format=torch.contiguous_format)
                .view(torch.uint8)[:, :160]
                .contiguous()
            )
            rq, rs = kernel.act_quant(
                value, module.fp8_block_size, module.scale_fmt, module.scale_dtype
            )
            a, reference_a, substituted, restored, projection_metrics = linear_pair(
                native.wkv, released.wkv, value, module, kernel
            )
            require_bytes(a, restored, "WKV restoration")
            norms, norm_metrics, codecs, codec_metrics = {}, {}, {}, {}
            for kind, projected in (("native", a), ("released", reference_a)):
                norms[kind], norm_metrics[kind] = norm_pair(
                    projected, native, released, nf, rf, module
                )
                for operator in ("native_fused", "reference_rotated"):
                    key = kind + "." + operator
                    codecs[key], codec_metrics[key] = encode_pair(
                        norms[kind][operator], start, module, kernel
                    )
            selected = (
                "native.native_fused"
                if stream == "native"
                else "released.reference_rotated"
            )
            current = (
                codecs[selected]["decoded"]
                if stream == "native"
                else codecs[selected]["reference_decoded"]
            )
            saved = raw[stream + "_attention"][step][LAYER]["kv"].reshape(-1, WIDTH)
            require_bytes(
                saved[start : start + rows], current.cpu(), "saved current SWA"
            )
            cumulative[stream].append(current.clone())
            require_bytes(
                saved[: start + rows],
                torch.cat(cumulative[stream]).cpu(),
                "saved retained SWA",
            )
            require_bytes(saved_input, value, "WKV input operand")
            own_controls += 1
            identity_fields = {
                "step": step,
                "stream": stream,
                "layer": LAYER,
                "identity": identity,
            }
            records.append(
                dict(
                    identity_fields,
                    projection=projection_metrics,
                    normalization=norm_metrics,
                    encoding=codec_metrics,
                )
            )
            cases.append(
                cpu_tree(
                    dict(
                        identity_fields,
                        input=saved_input,
                        input_after=value.clone(),
                        native_payload=quant,
                        native_scales=linear_scales,
                        released_payload=rq,
                        released_scales=rs.view(torch.uint8),
                        native_projection=a,
                        released_projection=reference_a,
                        substituted_projection=substituted,
                        restored_projection=restored,
                        norms=norms,
                        codecs=codecs,
                        saved_current=saved[start : start + rows].clone(),
                        saved_retained=saved[: start + rows].clone(),
                        current=current.clone(),
                        retained=torch.cat(cumulative[stream]).clone(),
                    )
                )
            )
    for name, expected in before.items():
        require_bytes(expected, weights[name], "KV immutable " + name)
    for name, expected in reader.hashes.items():
        if (
            expected != history["weight_sha256"][name]
            or expected != history["cached_weight_sha256"][name]
        ):
            raise ValueError("KV NAS weight differs: " + name)
    if own_controls != 8:
        raise AssertionError("KV boundary own-stream controls incomplete")
    with cli.tensor_artifact.open("xb") as handle:
        torch.save(
            {
                "cases": cases,
                "weights_before": cpu_tree(before),
                "weights_after": cpu_tree(weights),
            },
            handle,
        )
    report = {
        "prompt_index": 1,
        "layer": LAYER,
        "eps": native.eps,
        "identities": raw["identities"],
        "records": records,
        "own_current_controls": own_controls,
        "own_retained_controls": own_controls,
        "restoration_controls": own_controls,
        "weight_sha256": reader.hashes,
        "tensor_artifact_sha256": sha256(cli.tensor_artifact),
        "history_qualification_sha256": manifest["history_qualification_sha256"],
        "replay_source_sha256": sha256(Path(__file__)),
        "numerical_promotion": False,
    }
    with cli.report.open("x") as handle:
        handle.write(
            json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
    print(
        "DSV41_KV_BOUNDARY_CONTROLS "
        + json.dumps(
            {
                key: report[key]
                for key in (
                    "own_current_controls",
                    "own_retained_controls",
                    "restoration_controls",
                    "numerical_promotion",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

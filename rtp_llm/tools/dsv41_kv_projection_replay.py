"""Replay explicit WKV selection on qualified operands and changing graphs."""

import argparse
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.tools.dsv41_attention_replay import (
    NASReader,
    difference,
    native_attention,
    official_attention,
)
from rtp_llm.tools.dsv41_reference_probe import model_args


def graph_cases(native, released, module, kernel, seeds):
    from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import (
        fused_rmsnorm_rope,
    )
    from rtp_llm.models_py.modules.dsv4.fp8 import _v41_swa_triton as swa
    from rtp_llm.models_py.modules.dsv4.fp8.test.v41_kv_projection_reference import (
        kv_projection_reference,
    )
    from rtp_llm.tools.dsv41_corrected_history_replay import cpu_tree
    from rtp_llm.tools.dsv41_kv_boundary_replay import rotate_released
    from rtp_llm.tools.dsv41_query_boundary_replay import (
        quantized_difference,
        require_bytes,
    )

    raw, records = [], []
    for name, seed, start in seeds:
        static = seed.cuda().contiguous().clone()
        rows = len(static)
        stride = 128 * 528 + 512
        backing = torch.full((3 * stride,), 173, device="cuda", dtype=torch.uint8)
        pool = backing.as_strided((3, 128, 528), (stride, 528, 1))
        slots = torch.arange(128 + start, 128 + start + rows, device="cuda")
        nf = native.freqs_cis[start : start + rows].contiguous()
        rf = released.freqs_cis[start : start + rows].contiguous()

        def body():
            projected = native._lin(native.wkv, static)
            rotated = fused_rmsnorm_rope(
                projected.unsqueeze(0),
                native.kv_norm,
                nf,
                64,
                eps=native.eps,
                round_norm_to_input_dtype=True,
            )[0]
            swa.quantize_and_insert_swa_k_cache(rotated, pool, slots)
            decoded = swa.dequantize_swa_k_cache(pool, slots)
            return projected, rotated, decoded

        replays, metrics = [], []
        with kv_projection_reference(True, attentions=[native], kernel=kernel):
            body()
            body()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                projected, rotated, decoded = body()
            for repeat, page in enumerate((1, 2, 1)):
                changed = (
                    torch.roll(seed.cuda(), repeat + 1, -1).float()
                    * (1 + (repeat + 1) / 16)
                ).bfloat16()
                static.copy_(changed)
                slots.copy_(
                    torch.arange(
                        page * 128 + start, page * 128 + start + rows, device="cuda"
                    )
                )
                backing.fill_(173)
                before = backing.clone()
                reference_projection = released.wkv(changed).clone()
                reference_rotated = rotate_released(
                    module, released.kv_norm(reference_projection), rf
                )
                reference_decoded = reference_rotated.clone()
                reference_payload, reference_scales = kernel.act_quant(
                    reference_rotated,
                    module.fp8_block_size,
                    module.scale_fmt,
                    module.scale_dtype,
                )
                kernel.act_quant(
                    reference_decoded,
                    module.fp8_block_size,
                    module.scale_fmt,
                    module.scale_dtype,
                    True,
                )
                graph.replay()
                require_bytes(changed, static, "graph WKV input")
                payload = (
                    backing[
                        page * stride
                        + start * 512 : page * stride
                        + (start + rows) * 512
                    ]
                    .reshape(rows, 512)
                    .clone()
                )
                offset = page * stride + 128 * 512 + start * 16
                scales = backing[offset : offset + rows * 16].reshape(rows, 16).clone()
                metric = {
                    "projection": difference(reference_projection, projected),
                    "rotation": difference(reference_rotated, rotated),
                    "decoded": difference(reference_decoded, decoded),
                    "payload": quantized_difference(
                        reference_payload, payload.view(torch.float8_e4m3fn)
                    ),
                    "scales": quantized_difference(
                        reference_scales, scales.view(torch.float8_e8m0fnu)
                    ),
                }
                metrics.append(metric)
                replays.append(
                    cpu_tree(
                        {
                            "repeat": repeat,
                            "page": page,
                            "input": changed,
                            "input_after": static.clone(),
                            "slots": slots.clone(),
                            "before": before,
                            "after": backing.clone(),
                            "projection": projected.clone(),
                            "reference_projection": reference_projection,
                            "rotated": rotated.clone(),
                            "reference_rotated": reference_rotated,
                            "decoded": decoded.clone(),
                            "reference_decoded": reference_decoded,
                            "payload": payload,
                            "scales": scales,
                            "reference_payload": reference_payload.view(
                                torch.uint8
                            ).clone(),
                            "reference_scales": reference_scales.view(
                                torch.uint8
                            ).clone(),
                        }
                    )
                )
        raw.append(
            cpu_tree(
                {
                    "name": name,
                    "start": start,
                    "stride": stride,
                    "seed": seed,
                    "native_freqs": nf,
                    "released_freqs": rf,
                    "replays": replays,
                }
            )
        )
        records.append({"name": name, "start": start, "metrics": metrics})
    return raw, records


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
    parser.add_argument("--boundary-report", type=Path, required=True)
    parser.add_argument("--boundary-artifact", type=Path, required=True)
    cli = parser.parse_args()
    if cli.report.exists() or cli.tensor_artifact.exists():
        raise ValueError("Require fresh WKV projection evidence")
    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("WKV replay requires leased SM120 hardware")
    from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import (
        fused_rmsnorm_rope,
    )
    from rtp_llm.models_py.modules.dsv4.fp8.test.v41_kv_projection_reference import (
        kv_projection_reference,
    )
    from rtp_llm.tools.dsv41_block_norm_replay import admit
    from rtp_llm.tools.dsv41_corrected_history_replay import cpu_tree
    from rtp_llm.tools.dsv41_kv_boundary_replay import encode_pair
    from rtp_llm.tools.dsv41_query_boundary_replay import (
        quantized_difference,
        require_bytes,
        sha256,
    )

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision("highest")
    manifest, history, _ = admit(cli)
    report = json.loads(cli.boundary_report.read_text())
    assert sha256(cli.boundary_artifact) == report["tensor_artifact_sha256"]
    boundary = torch.load(cli.boundary_artifact, map_location="cpu", weights_only=True)
    source_cases = {(c["stream"], c["step"]): c for c in boundary["cases"]}
    assert (
        len(source_cases) == 8
        and manifest["prompt_index"] == report["prompt_index"] == 1
    )
    root = cli.checkpoint.resolve()
    sys.path.insert(0, str(root / "inference"))
    module, converter, kernel = [
        importlib.import_module(name) for name in ("model", "convert", "kernel")
    ]
    for loaded in (module, converter, kernel):
        assert Path(loaded.__file__).resolve() == root / "inference" / (
            loaded.__name__ + ".py"
        )
    args = model_args(module, root, 256)
    config = json.loads((root / "config.json").read_text())["text_config"]
    reader = NASReader(root)
    native = native_attention(args, config, reader, 1)
    released = official_attention(module, converter, args, reader, 1)[0]
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
    for name, value in before.items():
        require_bytes(
            boundary["weights_before"][name],
            value.cpu(),
            "qualified WKV weight/frequency " + name,
        )
    cases, records = [], []
    cumulative = {"native": [], "official": []}
    for step in range(4):
        for stream in ("native", "official"):
            source = source_cases[stream, step]
            target = source_cases["official", step]
            identity = source["identity"]
            start, rows = identity["start_pos"], len(identity["input_ids"])
            value = source["input"].cuda().contiguous().clone()
            input_before = value.clone()
            nf = native.freqs_cis[start : start + rows].contiguous()
            native_projection = native._lin(native.wkv, value).clone()
            released_projection = released.wkv(value).clone()
            query_before = native._lin(native.wq_a, value).clone()
            native_rotated = fused_rmsnorm_rope(
                native_projection.unsqueeze(0),
                native.kv_norm,
                nf,
                64,
                eps=native.eps,
                round_norm_to_input_dtype=True,
            )[0]
            native_codec, native_encoding = encode_pair(
                native_rotated, start, module, kernel
            )
            quantized = []
            original_quant = native.wkv._quantize_input

            def capture_quant(values):
                payload, packed = original_quant(values)
                linear_scales = (
                    packed.clone(memory_format=torch.contiguous_format)
                    .view(torch.uint8)[:, : native.dim // 32]
                    .contiguous()
                )
                quantized.append((payload.clone(), linear_scales.clone()))
                return payload, packed

            with patch.object(
                native.wkv, "_quantize_input", side_effect=capture_quant
            ), kv_projection_reference(True, attentions=[native], kernel=kernel):
                candidate_projection = native._lin(native.wkv, value).clone()
                query_selected = native._lin(native.wq_a, value).clone()
            assert len(quantized) == 1
            restored_projection = native._lin(native.wkv, value).clone()
            query_restored = native._lin(native.wq_a, value).clone()
            candidate_rotated = fused_rmsnorm_rope(
                candidate_projection.unsqueeze(0),
                native.kv_norm,
                nf,
                64,
                eps=native.eps,
                round_norm_to_input_dtype=True,
            )[0]
            candidate_codec, candidate_encoding = encode_pair(
                candidate_rotated, start, module, kernel
            )
            cumulative[stream].append(candidate_codec["decoded"].clone())
            retained = torch.cat(cumulative[stream])
            metrics = {
                "native_projection": difference(
                    source["native_projection"], native_projection
                ),
                "released_projection": difference(
                    source["released_projection"], released_projection
                ),
                "candidate_projection": difference(
                    released_projection, candidate_projection
                ),
                "restored_projection": difference(
                    native_projection, restored_projection
                ),
                "native_current": difference(
                    source["codecs"]["native.native_fused"]["decoded"],
                    native_codec["decoded"],
                ),
                "candidate_current": difference(
                    target["saved_current"], candidate_codec["decoded"]
                ),
                "candidate_retained": difference(target["saved_retained"], retained),
                "activation_payload": quantized_difference(
                    source["native_payload"], quantized[0][0].cpu()
                ),
                "activation_scales": quantized_difference(
                    source["native_scales"].view(torch.float8_e8m0fnu),
                    quantized[0][1].cpu().view(torch.float8_e8m0fnu),
                ),
                "unselected_query": difference(query_before, query_selected),
                "restored_query": difference(query_before, query_restored),
                "candidate_rotation": difference(
                    source["norms"]["released"]["native_fused"], candidate_rotated
                ),
            }
            for name, metric in metrics.items():
                assert metric["bitwise_equal"], (stream, step, name, metric)
            require_bytes(input_before, value, "held WKV input")
            fields = {"step": step, "stream": stream, "identity": identity}
            records.append(
                dict(
                    fields,
                    metrics=metrics,
                    native_encoding=native_encoding,
                    candidate_encoding=candidate_encoding,
                )
            )
            cases.append(
                cpu_tree(
                    dict(
                        fields,
                        input=input_before,
                        input_after=value.clone(),
                        native_projection=native_projection,
                        released_projection=released_projection,
                        candidate_projection=candidate_projection,
                        restored_projection=restored_projection,
                        quantized_payload=quantized[0][0],
                        quantized_scales=quantized[0][1],
                        native_rotated=native_rotated,
                        candidate_rotated=candidate_rotated,
                        native_codec=native_codec,
                        candidate_codec=candidate_codec,
                        candidate_retained=retained,
                        query_before=query_before,
                        query_selected=query_selected,
                        query_restored=query_restored,
                    )
                )
            )
    graphs, graph_records = graph_cases(
        native,
        released,
        module,
        kernel,
        [
            ("prefill", source_cases["native", 0]["input"], 0),
            ("decode", source_cases["native", 1]["input"], 5),
        ],
    )
    for name, value in before.items():
        require_bytes(value, weights[name], "unchanged WKV weight/frequency " + name)
    assert reader.hashes == report["weight_sha256"]
    for name, value in reader.hashes.items():
        assert (
            value
            == history["weight_sha256"][name]
            == history["cached_weight_sha256"][name]
        )
    projection_graphs = sum(
        m["projection"]["bitwise_equal"] for g in graph_records for m in g["metrics"]
    )
    cache_graphs = sum(
        m["decoded"]["bitwise_equal"] for g in graph_records for m in g["metrics"]
    )
    controls = {
        "native_projection_controls": 8,
        "released_projection_controls": 8,
        "candidate_projection_controls": 8,
        "native_current_controls": 8,
        "candidate_current_controls": 8,
        "candidate_retained_controls": 8,
        "restoration_controls": 8,
        "unselected_query_controls": 16,
        "graphs": 2,
        "projection_graph_replays_exact": projection_graphs,
        "cache_graph_replays_exact": cache_graphs,
        "numerical_promotion": False,
    }
    with cli.tensor_artifact.open("xb") as handle:
        torch.save(
            {
                "cases": cases,
                "graphs": graphs,
                "weights_before": cpu_tree(before),
                "weights_after": cpu_tree(weights),
            },
            handle,
        )
    result = {
        "records": records,
        "graphs": graph_records,
        "controls": controls,
        "weight_sha256": reader.hashes,
        "tensor_artifact_sha256": sha256(cli.tensor_artifact),
        "boundary_artifact_sha256": sha256(cli.boundary_artifact),
        "boundary_qualification_sha256": manifest["boundary_qualification_sha256"],
        "replay_source_sha256": sha256(Path(__file__)),
        "numerical_promotion": False,
    }
    with cli.report.open("x") as handle:
        handle.write(
            json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
    print("DSV41_KV_PROJECTION_CONTROLS " + json.dumps(controls), flush=True)
    if projection_graphs != 6 or cache_graphs != 6:
        raise AssertionError("WKV graph mismatch preserved in raw evidence")


if __name__ == "__main__":
    main()

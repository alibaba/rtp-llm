"""Reconstruct official blocks, then isolate attention on their real inputs.

Only selected NAS Engram rows are read. Saved official FFN outputs replace the
expensive experts, but each reconstructed FFN input must match the official
trace bitwise before any result is accepted. No latency claim is made.
"""

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from rtp_llm.tools.dsv41_reference_probe import model_args, shard_spec
from rtp_llm.tools.dsv41_trace_compare import tensor_difference, trace_identity


def difference(left, right):
    return tensor_difference(
        left.reshape(-1, left.shape[-1]).cpu(),
        right.reshape(-1, right.shape[-1]).cpu(),
    )


def aligned_sparse_inputs(reference, captured, window_rows):
    """Compare logical positions across official/native KV storage orders."""
    if type(window_rows) is not int or window_rows < 1:
        raise ValueError("Sparse SWA segment must contain at least one row")
    reference_kv = reference["kv"].reshape(-1, 512)
    native_kv = captured["kv"].reshape(-1, 512)
    if reference_kv.shape != native_kv.shape:
        raise ValueError("Reference/native sparse KV row counts differ")
    global_rows = len(reference_kv) - window_rows
    if global_rows < 0:
        raise ValueError("Sparse KV is missing the initial SWA rows")
    # Official stores [SWA, global]; native stores [global, SWA].
    aligned_kv = torch.cat((native_kv[global_rows:], native_kv[:global_rows]))
    lengths = captured["lengths"]
    if lengths.ndim != 1 or not lengths.numel():
        raise ValueError("Sparse lengths must describe nonempty query rows")
    query_rows = lengths.numel()
    if (
        captured["indices"].numel() % query_rows
        or reference["indices"].numel() % query_rows
        or reference["indices"].shape[-2] != query_rows
        or captured["indices"].shape[0] != query_rows
    ):
        raise ValueError("Sparse index plans disagree with query row count")
    native_indices = captured["indices"].reshape(query_rows, -1).cpu()
    native_lengths = lengths.cpu().tolist()
    reference_indices = reference["indices"].reshape(query_rows, -1).cpu()
    selections = []
    for row, limit in enumerate(native_lengths):
        if not 0 <= limit <= native_indices.shape[1]:
            raise ValueError("Native sparse row length is outside its index plan")
        raw = native_indices[row, :limit]
        aligned = torch.where(raw < global_rows, raw + window_rows, raw - global_rows)
        aligned = torch.where(raw >= 0, aligned, -1)
        expected = reference_indices[row]
        actual_ids, expected_ids = (
            aligned[aligned >= 0].tolist(),
            expected[expected >= 0].tolist(),
        )
        native_invalid = int(((raw < -1) | (raw >= len(native_kv))).sum())
        reference_invalid = int(
            ((expected < -1) | (expected >= len(reference_kv))).sum()
        )
        selection = {
            "set_equal": sorted(actual_ids) == sorted(expected_ids),
            "order_equal": actual_ids == expected_ids,
            "reference": expected_ids,
            "native_aligned": actual_ids,
            "native_raw": raw[raw >= 0].tolist(),
            "native_invalid": native_invalid,
            "reference_invalid": reference_invalid,
            "native_duplicates": len(actual_ids) - len(set(actual_ids)),
            "reference_duplicates": len(expected_ids) - len(set(expected_ids)),
        }
        if not selection["set_equal"] or any(
            selection[name]
            for name in (
                "native_invalid",
                "reference_invalid",
                "native_duplicates",
                "reference_duplicates",
            )
        ):
            raise AssertionError(
                f"Short-replay sparse position set differs: {selection}"
            )
        selections.append(selection)
    return difference(reference_kv, aligned_kv), selections


def history_identities(traces, native_traces=None):
    """Admit one contiguous short prefill/decode history, before cache writes."""
    if not traces:
        raise ValueError("Replay history requires an initial prefill trace")
    identities = [trace_identity(trace["metadata"]) for trace in traces]
    end = 0
    for step, identity in enumerate(identities):
        span = len(identity["input_ids"])
        if identity["start_pos"] != end:
            raise ValueError("Replay trace history has a gap, overlap or reset")
        if (step == 0 and span < 2) or (step > 0 and span != 1):
            raise ValueError("Replay requires prefill then single-token decode")
        end += span
        if end > 128:
            raise ValueError("Replay history must fit within the first SWA ring")
    if native_traces is not None:
        if len(native_traces) != len(traces):
            raise ValueError("Native/reference histories have different step counts")
        native = [trace_identity(trace["metadata"]) for trace in native_traces]
        if native != identities:
            raise ValueError(
                "Native/reference histories have divergent consumed prefixes"
            )
    return identities


class SelectedEmbedding(torch.nn.Module):
    """Prepared official dequantized rows; no host reads inside forward."""

    def __init__(self, prepared):
        super().__init__()
        self.prepared = prepared

    def forward(self, indices):
        if indices.shape != self.prepared.shape[:-1]:
            raise ValueError("Prepared Engram row shape differs from hash shape")
        return self.prepared


class NASReader:
    def __init__(self, root):
        self.root = root
        self.index = json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        self.values = {}
        self.hashes = {}
        self.selected_rows = {}
        self.selected_row_reads = []

    def load(self, name):
        if name not in self.values:
            with safe_open(
                self.root / self.index[name], framework="pt", device="cpu"
            ) as handle:
                value = handle.get_tensor(name)
                self.hashes[name] = hashlib.sha256(
                    value.contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest()
                self.values[name] = value.cuda()
        return self.values[name]

    def rows(self, name, ids):
        unique, inverse = torch.unique(ids.flatten(), sorted=True, return_inverse=True)
        with safe_open(
            self.root / self.index[name], framework="pt", device="cpu"
        ) as handle:
            source = handle.get_slice(name)
            shape = source.get_shape()
            if ids.min() < 0 or ids.max() >= shape[0]:
                raise ValueError("Engram hash row is outside the checkpoint table")
            rows = torch.cat([source[row : row + 1] for row in unique.tolist()])
        self.selected_rows[name] = {
            "row_ids": unique.tolist(),
            "shape": list(rows.shape),
            "dtype": str(rows.dtype),
            "sha256": hashlib.sha256(
                rows.contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        }
        self.selected_row_reads.append({"name": name, **self.selected_rows[name]})
        # CPU float8 index_select is unavailable in some Torch versions.
        rows = rows.view(torch.uint8)[inverse].view(rows.dtype)
        return rows.reshape(*ids.shape, shape[1])

    def engram_rows(self, layer, ids, block_size):
        prefix = f"layers.{layer}.engram.embed."
        values = self.rows(prefix + "weight", ids).float()
        scales = self.rows(prefix + "scale", ids).float()
        return (
            (values.unflatten(-1, (-1, block_size)) * scales.unsqueeze(-1))
            .flatten(-2)
            .bfloat16()
            .cuda()
        )


def official_attention(module, converter, args, reader, layer):
    shards = []
    with torch.device("cuda"), patch.object(module, "world_size", 8):
        for rank in range(8):
            shard = module.Attention(layer, args)
            for name, target in shard.named_parameters():
                full_name = f"layers.{layer}.attn.{name}"
                value = reader.load(full_name)
                if name == "wo_a.weight":
                    scale = reader.load(full_name.removesuffix("weight") + "scale")
                    value = (
                        value.float()
                        * scale.float()
                        .repeat_interleave(32, 0)
                        .repeat_interleave(32, 1)
                    ).bfloat16()
                spec = shard_spec(
                    full_name,
                    value.shape,
                    rank,
                    8,
                    args.n_routed_experts,
                    converter.mapping,
                )
                shape, axis, start, stop = spec
                if axis is not None:
                    value = value.narrow(axis, start, stop - start)
                if tuple(target.shape) != shape or target.shape != value.shape:
                    raise ValueError(f"Reference shard shape mismatch: {full_name}")
                target.copy_(value)
            shards.append(shard)
    return shards


def run_official_attention(module, shards, x, start_pos=0, *, history=False):
    partials, captures, latent_captures = [], [], []
    sparse = module.sparse_attn

    def reduce(value, *args, **kwargs):
        # Short replay selects every reachable compressed position. Indexer
        # reductions cannot change that set; output partials are reduced below.
        if value.shape[-1] == x.shape[-1]:
            partials.append(value.clone())

    def record(q, kv, sinks, indices, scale):
        result = sparse(q, kv, sinks, indices, scale)
        captures.append(
            {
                "q": q.clone(),
                "kv": kv.clone(),
                "attention": result.clone(),
                "indices": indices.clone(),
            }
        )
        return result

    def record_latent(owner, inputs, output):
        if output is not None:
            latent_captures.append(output.clone())

    with (
        ExitStack() as stack,
        torch.device("cuda"),
        patch.object(module, "world_size", 8),
        patch.object(module.dist, "all_reduce", reduce),
        patch.object(module, "sparse_attn", record),
    ):
        for rank, shard in enumerate(shards):
            stack.enter_context(patch.object(module, "rank", rank))
            if shard.compressor is not None:
                stack.callback(
                    shard.compressor.register_forward_hook(record_latent).remove
                )
            shard(x, start_pos)
    if len(partials) != 8 or len(captures) != 8:
        raise ValueError("Replay did not capture all eight TP output partials")
    for capture in captures[1:]:
        for name in ("kv", "indices"):
            torch.testing.assert_close(capture[name], captures[0][name], rtol=0, atol=0)
    for value in latent_captures[1:]:
        torch.testing.assert_close(value, latent_captures[0], rtol=0, atol=0)
    result = {
        "q": torch.cat([c["q"] for c in captures], dim=2),
        "kv": captures[0]["kv"],
        "indices": captures[0]["indices"],
        "attention": torch.cat([c["attention"] for c in captures], dim=2),
        "output": torch.stack(partials).sum(0).bfloat16(),
    }
    if latent_captures:
        result["latent"] = latent_captures[0]
    if history:
        result["window_cache"] = shards[0].window_kv_cache.clone()
        if shards[0].compressor is not None:
            result["global_cache"] = shards[0].compress_kv_cache.clone()
            result["index_cache"] = shards[0].indexer.k_cache.clone()
            if shards[0].compress_ratio == 2:
                for name in ("kv_state", "score_state"):
                    result[name] = getattr(shards[0].compressor, name).clone()
        for shard in shards[1:]:
            torch.testing.assert_close(
                shard.window_kv_cache, result["window_cache"], rtol=0, atol=0
            )
            if shard.compressor is not None:
                torch.testing.assert_close(
                    shard.compress_kv_cache, result["global_cache"], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    shard.indexer.k_cache, result["index_cache"], rtol=0, atol=0
                )
                if shard.compress_ratio == 2:
                    for name in ("kv_state", "score_state"):
                        torch.testing.assert_close(
                            getattr(shard.compressor, name),
                            result[name],
                            rtol=0,
                            atol=0,
                        )
    return result


def native_attention(args, config, reader, layer):
    from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import AttentionV41FP8
    from rtp_llm.utils.model_weight import W

    suffixes = {
        W.v4_attn_q_norm: "q_norm.weight",
        W.v4_attn_kv_norm: "kv_norm.weight",
        W.v4_attn_sink: "attn_sink",
    }
    for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
        for suffix, field in (("w", "weight"), ("s", "scale")):
            suffixes[getattr(W, f"v4_attn_{name}_{suffix}")] = f"{name}.{field}"
    if layer in args.kv_source_layers:
        suffixes.update(
            {
                W.v4_compressor_wkv: "compressor.wkv.weight",
                W.v4_compressor_norm: "compressor.norm.weight",
                W.v41_indexer_wk: "indexer.wk.weight",
                W.v41_indexer_k_norm: "indexer.k_norm.weight",
            }
        )
        if args.compress_ratios[layer] == 2:
            suffixes[W.v4_compressor_wgate] = "compressor.wgate.weight"
    if layer in args.index_source_layers:
        suffixes.update(
            {
                W.v4_indexer_wq_b_w: "indexer.wq_b.weight",
                W.v4_indexer_wq_b_s: "indexer.wq_b.scale",
                W.v4_indexer_weights_proj_w: "indexer.weights_proj.weight",
            }
        )
    weights = {
        key: reader.load(f"layers.{layer}.attn.{suffix}")
        for key, suffix in suffixes.items()
    }
    native_config = dict(config)
    native_config.update(
        kv_source_layer_ids=list(args.kv_source_layers),
        index_source_layer_ids=list(args.index_source_layers),
    )
    owner = AttentionV41FP8(
        layer_id=layer,
        cache_layer_id=layer,
        dim=args.dim,
        n_heads=args.n_heads,
        q_lora_rank=args.q_lora_rank,
        head_dim=args.head_dim,
        rope_head_dim=args.rope_head_dim,
        o_lora_rank=args.o_lora_rank,
        o_groups=args.o_groups,
        window_size=args.window_size,
        compress_ratio=args.compress_ratios[layer],
        compress_rope_theta=args.compress_rope_theta,
        rope_theta=args.rope_theta,
        rope_factor=args.rope_factor,
        beta_fast=args.beta_fast,
        beta_slow=args.beta_slow,
        original_seq_len=args.original_seq_len,
        max_batch_size=1,
        max_seq_len=args.max_seq_len,
        index_n_heads=args.index_n_heads,
        index_head_dim=args.index_head_dim,
        index_topk=args.index_topk,
        norm_eps=args.norm_eps,
        layer_weights=weights,
        v41_config=native_config,
        shared_attention={},
        gen_num_per_cycle=0,
    )
    owner.reset_rope_cache(torch.device("cuda"))
    return owner


def compare_native(owner, x, reference):
    from rtp_llm.models_py.modules.dsv4.fp8 import (
        _v41_prefill_attention_sm120 as provider,
        attention_v41,
        compressor,
    )
    from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_attention_owner_sm120 import (
        NativeTaggedCache,
        prepare,
    )
    from rtp_llm.models_py.modules.dsv4.fp8.test.v41_compressor_precision import (
        compressor_bf16,
    )
    from rtp_llm.models_py.modules.dsv4.fp8.test.v41_tp8_projection import (
        tp8_projection,
    )

    results = {}
    sparse = provider.sparse_prefill_attention
    profiles = [
        ("baseline", False, False, False, False, False, False),
        ("compressor_bf16", True, False, False, False, False, False),
        ("tp8_projection", False, True, False, False, False, False),
        ("tp8_projection_compressor_bf16", True, True, False, False, False, False),
        ("tp8_projection_official_latent", False, True, True, False, False, False),
        (
            "tp8_projection_compressor_bf16_official_q",
            True,
            True,
            False,
            True,
            False,
            False,
        ),
        ("tp8_projection_official_attention", False, True, False, False, True, False),
    ]
    if owner.compress_ratio == 1:
        profiles.append(
            (
                "tp8_projection_hca_bf16_projection",
                True,
                True,
                False,
                False,
                False,
                True,
            )
        )
    for (
        name,
        rounded,
        tp8,
        fixed_latent,
        fixed_q,
        fixed_attention,
        hca_bf16,
    ) in profiles:
        captured = {}

        def record(q, kv, indices, lengths, sinks):
            if fixed_q:
                q = reference["q"].reshape_as(q).clone()
            out = sparse(q, kv, indices, lengths, sinks)
            if fixed_attention:
                out = reference["attention"].reshape_as(out).clone()
            captured.update(
                q=q.clone(),
                kv=kv.clone(),
                indices=indices.clone(),
                lengths=lengths.clone(),
                attention=out.clone(),
            )
            return out

        cache = NativeTaggedCache([owner], gen=0)
        positions, tables, _ = prepare(owner, x, cache, [len(x)])
        with ExitStack() as stack:
            stack.enter_context(compressor_bf16(rounded))
            stack.enter_context(tp8_projection(tp8))
            if hca_bf16:
                projection = compressor._linear_bf16_bf16_fp32

                def bf16_projection(value, weight):
                    if weight is owner.global_wkv:
                        # Ratio one uses an actual BF16-output GEMM, which can
                        # differ from casting an FP32-output GEMM to BF16.
                        return torch.nn.functional.linear(value, weight).float()
                    return projection(value, weight)

                stack.enter_context(
                    patch.object(compressor, "_linear_bf16_bf16_fp32", bf16_projection)
                )
            norm = attention_v41.rms_norm

            def record_norm(value, weight, eps):
                if value.dtype == torch.float32 and value.shape[-1] == 512:
                    captured["pooled_latent"] = value.clone()
                    if fixed_latent:
                        out = reference["latent"].reshape(-1, 512).clone()
                        if out.shape != value.shape:
                            raise ValueError(
                                "Reference latent shape differs from producer"
                            )
                    else:
                        out = norm(value, weight, eps)
                    captured["latent"] = out.bfloat16().clone()
                    return out
                return norm(value, weight, eps)

            stack.enter_context(patch.object(attention_v41, "rms_norm", record_norm))
            stack.enter_context(
                patch.object(provider, "sparse_prefill_attention", record)
            )
            actual = owner(x.clone(), positions, cache, tables).clone()
        result = {"output": difference(reference["output"], actual)}
        for key in ("q", "attention", "latent"):
            if key in captured and key in reference:
                result[key] = difference(reference[key], captured[key])
        result["kv"], result["selected_positions"] = aligned_sparse_inputs(
            reference, captured, len(x)
        )
        results[name] = result
        print(json.dumps({"profile": name, "output": result["output"]}), flush=True)
    return results


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--input-artifact", type=Path, required=True)
    parser.add_argument("--layer", type=int, choices=(2, 8, 14, 20), default=2)
    cli = parser.parse_args()
    if cli.report.exists() or cli.input_artifact.exists():
        raise ValueError("Replay artifacts must be fresh")
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("Early attention replay requires SM120 under GPU lock")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision("highest")
    trace = torch.load(cli.trace, map_location="cpu", weights_only=True)
    identity = trace_identity(trace["metadata"])
    if identity["start_pos"] != 0 or not 2 <= len(identity["input_ids"]) <= 128:
        raise ValueError("This replay requires a short initial prefill trace")
    root = cli.checkpoint.resolve()
    sys.path.insert(0, str(root / "inference"))
    official, converter = [
        importlib.import_module(name) for name in ("model", "convert")
    ]
    if Path(official.__file__).resolve() != root / "inference/model.py":
        raise ValueError("Unexpected official reference module")
    args = model_args(official, root, 256)
    if tuple(args.compress_ratios[:3]) != (0, 0, 2):
        raise ValueError("Replay requires the released L0-L2 source schedule")
    reader = NASReader(root)
    tokenizer = AutoTokenizer.from_pretrained(
        root, trust_remote_code=False, local_files_only=True
    )
    layout = official.EngramLayout.from_args(args)
    tokens = torch.tensor([identity["input_ids"]], device="cpu", dtype=torch.long)
    with torch.device("cpu"):
        hashes = official.NgramHashState(args, layout, tokenizer)(tokens, 0)
    embedded = reader.rows("embed.weight", tokens).cuda()
    hidden = embedded.unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
    pre_mix = official.make_identity_pre_mix(hidden, args.hc_mult)
    block = SimpleNamespace(
        norm_eps=args.norm_eps,
        hc_mult=args.hc_mult,
        hc_sinkhorn_iters=args.hc_sinkhorn_iters,
        hc_eps=args.hc_eps,
    )
    report = {
        "identity": identity,
        "layer": cli.layer,
        "reconstruction": {},
        "variants": {},
        "sparse_layout": {
            "reference": "SWA then global",
            "native": "global then SWA",
            "comparison": "KV and indices aligned to reference storage",
        },
    }
    target_x, target_reference = None, None
    for layer in range(cli.layer + 1):
        if layer in layout.layer_ids:
            ids = hashes[:, :, layout.layer_ids.index(layer), :]
            prepared = reader.engram_rows(layer, ids, official.fp8_block_size)
            with (
                torch.device("cuda"),
                patch.object(
                    official,
                    "ParallelEngramEmbedding",
                    lambda rows, dim: SelectedEmbedding(prepared),
                ),
            ):
                engram = official.Engram(args, layer, layout)
            for name, parameter in engram.named_parameters():
                parameter.copy_(reader.load(f"layers.{layer}.engram.{name}"))
            hidden = engram(hidden, ids.cuda())
            del engram, prepared
        with torch.device("cuda"):
            attn_pre, attn_post, attn_comb = official.Block.hc_mixes(
                block,
                hidden,
                reader.load(f"layers.{layer}.hc_attn_fn").float(),
                reader.load(f"layers.{layer}.hc_attn_scale").float(),
                reader.load(f"layers.{layer}.hc_attn_base").float(),
            )
            norm = official.RMSNorm(args.dim, args.norm_eps)
            norm.weight.copy_(reader.load(f"layers.{layer}.attn_norm.weight"))
            x = norm(official.Block.hc_pre(block, hidden, pre_mix))
            shards = official_attention(official, converter, args, reader, layer)
            attention = run_official_attention(official, shards, x)
            mixed = official.Block.hc_post(
                block, attention["output"], hidden, attn_post, attn_comb
            )
            ffn_pre, ffn_post, ffn_comb = official.Block.hc_mixes(
                block,
                mixed,
                reader.load(f"layers.{layer}.hc_ffn_fn").float(),
                reader.load(f"layers.{layer}.hc_ffn_scale").float(),
                reader.load(f"layers.{layer}.hc_ffn_base").float(),
            )
            norm.weight.copy_(reader.load(f"layers.{layer}.ffn_norm.weight"))
            ffn_input = norm(official.Block.hc_pre(block, mixed, attn_pre))
        label = f"layer{layer:02d}.ffn.input"
        result = difference(trace["tensors"][label], ffn_input)
        report["reconstruction"][label] = result
        print(json.dumps({"boundary": label, **result}), flush=True)
        if not result["bitwise_equal"]:
            raise AssertionError(
                f"Official reconstruction is not exact at {label}: {result}"
            )
        if layer == cli.layer:
            target_x, target_reference = x[0].clone(), attention
        else:
            ffn_output = trace["tensors"][f"layer{layer:02d}.ffn.output"].cuda()
            if ffn_output.ndim == 2:
                ffn_output = ffn_output[None]
            hidden = official.Block.hc_post(
                block, ffn_output, mixed, ffn_post, ffn_comb
            )
            pre_mix = ffn_pre
        del shards, norm
    with cli.input_artifact.open("xb") as handle:
        torch.save(
            {"identity": identity, "layer": cli.layer, "input": target_x.cpu()}, handle
        )
    config = json.loads((root / "config.json").read_text())["text_config"]
    owner = native_attention(args, config, reader, cli.layer)
    report["variants"] = compare_native(owner, target_x, target_reference)
    report["weight_sha256"] = reader.hashes
    report["selected_rows"] = reader.selected_rows
    report["official_source_sha256"] = {
        name: hashlib.sha256((root / "inference" / name).read_bytes()).hexdigest()
        for name in ("model.py", "kernel.py", "engram.py", "convert.py", "config.json")
    }
    report["reference_trace_sha256"] = hashlib.sha256(
        cli.trace.read_bytes()
    ).hexdigest()
    repo = Path(__file__).resolve().parents[2]
    report["native_revision"] = subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report["native_source_sha256"] = {
        name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
        for name in (
            "rtp_llm/tools/dsv41_attention_replay.py",
            "rtp_llm/models_py/modules/dsv4/fp8/attention_v41.py",
            "rtp_llm/models_py/modules/dsv4/fp8/compressor.py",
            "rtp_llm/models_py/modules/dsv4/fp8/_v41_prefill_attention_sm120.py",
            "rtp_llm/models_py/modules/dsv4/fp8/test/v41_compressor_precision.py",
            "rtp_llm/models_py/modules/dsv4/fp8/test/v41_tp8_projection.py",
        )
    }
    report["limitations"] = [
        f"Short L{cli.layer} prefill only; source inputs and initial cache history are fixed.",
        "Saved official FFN outputs are replayed; experts are not re-executed.",
        "TP8 output partials are simulated serially; no live collective qualification.",
        "Full-model, continuation, serving and latency acceptance remain separate.",
    ]
    with cli.report.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()

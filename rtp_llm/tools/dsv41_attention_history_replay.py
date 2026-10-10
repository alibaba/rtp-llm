"""Replay actual L0-L2 attention with retained prefill/decode cache history.

Saved FFN outputs replace expert execution. Every reconstructed FFN input must
match its full-run trace bitwise. Official-input and native-input experiments
have separate caches; this diagnostic makes no serving or performance claim.
"""

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import AutoTokenizer

from rtp_llm.tools.dsv41_attention_replay import (
    NASReader,
    SelectedEmbedding,
    aligned_sparse_inputs,
    difference,
    history_identities,
    native_attention,
    official_attention,
    run_official_attention,
)
from rtp_llm.tools.dsv41_reference_probe import model_args


def require_exact(expected, actual, label):
    result = difference(expected, actual)
    print(json.dumps({"boundary": label, **result}), flush=True)
    if not result["bitwise_equal"]:
        raise AssertionError(
            f"History reconstruction is not exact at {label}: {result}"
        )
    return result


def cpu_snapshot(values):
    return {key: value.cpu() for key, value in values.items() if torch.is_tensor(value)}


def validate_native_recipe(traces, *, reference_kernel_sha256=None):
    base = "tp8_projection_mhc_fp32_compressor_recipe"
    recipes = {
        base: (False, False, False, False),
        f"{base}_shared_down_reference": (False, True, False, False),
        f"{base}_expert_down_reference": (False, True, True, False),
        f"{base}_sparse_sink_div_rn_prefill_shared_down_reference": (
            True,
            True,
            False,
            False,
        ),
        f"{base}_sparse_sink_div_rn_prefill_expert_down_reference": (
            True,
            True,
            True,
            False,
        ),
        f"{base}_expert_down_reference_mhc_post_reference": (
            False,
            True,
            True,
            True,
        ),
        f"{base}_sparse_sink_div_rn_prefill_expert_down_reference_mhc_post_reference": (
            True,
            True,
            True,
            True,
        ),
    }
    if not traces:
        raise ValueError("Native history requires nonempty recipe traces")
    selected = None
    for trace in traces:
        metadata = trace["metadata"]
        profile = metadata.get("precision_profile")
        if (
            profile not in recipes
            or metadata.get("compressor_fused") is not True
            or metadata.get("hca_bf16_projection") is not True
        ):
            raise ValueError(
                "Native history requires the composed TP8/mHC/compressor recipe"
            )
        sparse, shared, routed, post = recipes[profile]
        expected = {
            "precision_profile": profile,
            "compressor_fused": True,
            "hca_bf16_projection": True,
            "sparse_sink_div_rn_prefill": sparse,
            "sparse_sink_div_rn_decode": False,
            "shared_down_reference": shared,
            "routed_down_reference": routed,
        }
        for field in (
            "sparse_sink_div_rn_prefill",
            "sparse_sink_div_rn_decode",
            "shared_down_reference",
            "routed_down_reference",
        ):
            # The original frozen recipe predates the optional diagnostic flags.
            actual = (
                metadata.get(field, False) if profile == base else metadata.get(field)
            )
            if actual is not expected[field]:
                raise ValueError(f"Native history recipe flag differs: {field}")
        actual_post = (
            metadata.get("mhc_post_reference", False)
            if not post
            else metadata.get("mhc_post_reference")
        )
        if actual_post is not post:
            raise ValueError("Native history recipe flag differs: mhc_post_reference")
        # Preserve the canonical recipe recorded by the older history reports.
        if post:
            expected["mhc_post_reference"] = True
        digest = metadata.get("down_reference_kernel_sha256")
        if shared:
            if (
                not isinstance(reference_kernel_sha256, str)
                or len(reference_kernel_sha256) != 64
                or any(c not in "0123456789abcdef" for c in reference_kernel_sha256)
                or digest != reference_kernel_sha256
            ):
                raise ValueError("Native history down-reference kernel hash differs")
            expected["down_reference_kernel_sha256"] = digest
        elif digest is not None:
            raise ValueError(
                "Native history recipe has an unexpected down-reference hash"
            )
        if selected is not None and expected != selected:
            raise ValueError("Native history cannot mix recipes across steps")
        selected = expected
    return selected


def official_history_control(recipe):
    name = "tp8_projection_compressor_recipe"
    if recipe and recipe["sparse_sink_div_rn_prefill"]:
        name += "_sparse_sink_div_rn_prefill"
    return name


def prepared_engram(module, args, layout, reader, layer, ids):
    rows = reader.engram_rows(layer, ids, module.fp8_block_size)
    with (
        torch.device("cuda"),
        patch.object(
            module, "ParallelEngramEmbedding", lambda n, d: SelectedEmbedding(rows)
        ),
    ):
        owner = module.Engram(args, layer, layout)
    for name, parameter in owner.named_parameters():
        parameter.copy_(reader.load(f"layers.{layer}.engram.{name}"))
    return owner, rows


def reconstruct_official(
    module, converter, args, layout, reader, traces, identities, hashes
):
    shards = [
        official_attention(module, converter, args, reader, layer) for layer in range(3)
    ]
    block = SimpleNamespace(
        norm_eps=args.norm_eps,
        hc_mult=args.hc_mult,
        hc_sinkhorn_iters=args.hc_sinkhorn_iters,
        hc_eps=args.hc_eps,
    )
    steps, boundaries, rows_by_step = [], {}, []
    split, raw_mixes = module.hc_split_sinkhorn, []

    def record_split(mixes, *args, **kwargs):
        raw_mixes.append(mixes.clone())
        return split(mixes, *args, **kwargs)

    for step, (trace, identity, step_hashes) in enumerate(
        zip(traces, identities, hashes)
    ):
        tokens = torch.tensor([identity["input_ids"]], dtype=torch.long, device="cpu")
        hidden = (
            reader.rows("embed.weight", tokens)
            .cuda()
            .unsqueeze(2)
            .repeat(1, 1, args.hc_mult, 1)
        )
        pre_mix = module.make_identity_pre_mix(hidden, args.hc_mult)
        inputs, references, selected, mhc = [], [], {}, []
        for layer in range(3):
            captured = {"layer_input": hidden.clone()}
            if layer in layout.layer_ids:
                ids = step_hashes[:, :, layout.layer_ids.index(layer), :]
                engram, rows = prepared_engram(module, args, layout, reader, layer, ids)
                hidden = engram(hidden, ids.cuda())
                selected[layer] = rows
            captured["attention_residual"] = hidden.clone()
            captured["attn_read_mix"] = pre_mix.clone()
            with torch.device("cuda"), patch.object(
                module, "hc_split_sinkhorn", record_split
            ):
                attn_pre, attn_post, attn_comb = module.Block.hc_mixes(
                    block,
                    hidden,
                    reader.load(f"layers.{layer}.hc_attn_fn").float(),
                    reader.load(f"layers.{layer}.hc_attn_scale").float(),
                    reader.load(f"layers.{layer}.hc_attn_base").float(),
                )
                captured.update(
                    attn_raw_mixes=raw_mixes.pop(),
                    attn_pre_mix=attn_pre.clone(),
                    attn_post_mix=attn_post.clone(),
                    attn_comb_mix=attn_comb.clone(),
                )
                norm = module.RMSNorm(args.dim, args.norm_eps)
                norm.weight.copy_(reader.load(f"layers.{layer}.attn_norm.weight"))
                collapsed = module.Block.hc_pre(block, hidden, pre_mix)
                captured["attn_collapsed"] = collapsed.clone()
                x = norm(collapsed)
                captured["attn_norm"] = x.clone()
                reference = run_official_attention(
                    module, shards[layer], x, identity["start_pos"], history=True
                )
                mixed = module.Block.hc_post(
                    block, reference["output"], hidden, attn_post, attn_comb
                )
                captured.update(
                    attention_output=reference["output"].clone(),
                    attn_post=mixed.clone(),
                    ffn_read_mix=attn_pre.clone(),
                )
                ffn_pre, ffn_post, ffn_comb = module.Block.hc_mixes(
                    block,
                    mixed,
                    reader.load(f"layers.{layer}.hc_ffn_fn").float(),
                    reader.load(f"layers.{layer}.hc_ffn_scale").float(),
                    reader.load(f"layers.{layer}.hc_ffn_base").float(),
                )
                captured.update(
                    ffn_raw_mixes=raw_mixes.pop(),
                    ffn_pre_mix=ffn_pre.clone(),
                    ffn_post_mix=ffn_post.clone(),
                    ffn_comb_mix=ffn_comb.clone(),
                )
                norm.weight.copy_(reader.load(f"layers.{layer}.ffn_norm.weight"))
                collapsed = module.Block.hc_pre(block, mixed, attn_pre)
                captured["ffn_collapsed"] = collapsed.clone()
                ffn_input = norm(collapsed)
                captured["ffn_norm"] = ffn_input.clone()
            label = f"step{step}.layer{layer:02d}.ffn.input"
            boundaries[label] = require_exact(
                trace["tensors"][f"layer{layer:02d}.ffn.input"], ffn_input, label
            )
            inputs.append(x[0].clone())
            references.append(reference)
            if layer < 2:
                ffn_output = (
                    trace["tensors"][f"layer{layer:02d}.ffn.output"]
                    .cuda()
                    .reshape_as(x)
                )
                hidden = module.Block.hc_post(
                    block, ffn_output, mixed, ffn_post, ffn_comb
                )
                captured.update(ffn_output=ffn_output.clone(), ffn_post=hidden.clone())
                pre_mix = ffn_pre
            mhc.append(captured)
        steps.append({"inputs": inputs, "reference": references, "mhc": mhc})
        rows_by_step.append(selected)
    return steps, boundaries, rows_by_step


def cache_values(owner, cache, tables, tag, count):
    from rtp_llm.models_py.modules.dsv4.v41_cache import V41PagedCache

    view = V41PagedCache(
        cache.get_layer_cache(owner.cache_layer_id, tag).kv_cache_base, tag, 0
    )
    slots = tables[tag][0, 0].long() * view.layout.entries + torch.arange(
        count, device="cuda"
    )
    return view.read(slots.contiguous())


def decode_capture(owner, cache, tables, captured, end):
    """Gather after forward; physical indices become logical ring/global rows."""
    window = cache_values(owner, cache, tables, "swa_kv", 128)
    count = end // owner.compress_ratio if owner.compress_ratio else 0
    global_values = (
        cache_values(owner, cache, tables, owner._global_region(), count)
        if count
        else window[:0]
    )
    swa_base = tables["swa_kv"][0, 0].long() * cache.layouts["swa_kv"].entries
    swa = captured["swa_slots"].reshape(1, -1)
    indices = torch.where(swa >= 0, swa - swa_base + count, -1)
    if count:
        tag = owner._global_region()
        global_base = tables[tag][0, 0].long() * cache.layouts[tag].entries
        selected = captured["global_slots"].reshape(1, -1)
        indices = torch.cat(
            (indices, torch.where(selected >= 0, selected - global_base, -1)), -1
        )
    captured.update(
        kv=torch.cat((global_values, window)).unsqueeze(1),
        indices=indices.unsqueeze(1),
        lengths=torch.tensor([indices.shape[-1]], device="cuda", dtype=torch.int32),
    )


def execute_native(
    owner, cache, tables, x, start, reference, *, fixed_q=False, fixed_attention=False
):
    from rtp_llm.models_py.modules.dsv4.fp8 import _v41_decode_attention_sm120 as decode
    from rtp_llm.models_py.modules.dsv4.fp8 import (
        _v41_prefill_attention_sm120 as prefill,
    )
    from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_attention_owner_sm120 import (
        prepare,
    )
    from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_decode_owner_sm120 import (
        decode_metadata,
    )

    captured = {}
    prefill_sparse, decode_sparse = (
        prefill.sparse_prefill_attention,
        decode.sparse_decode_attention,
    )
    linear = torch.nn.functional.linear

    def record_linear(value, weight, *args, **kwargs):
        if owner.compress_ratio and weight is owner.index_wk:
            captured["latent"] = value.clone()
        return linear(value, weight, *args, **kwargs)

    def record_prefill(q, kv, indices, lengths, sinks):
        captured["native_q"] = q.clone()
        if fixed_q:
            q = reference["q"].reshape_as(q).clone()
        out = prefill_sparse(q, kv, indices, lengths, sinks)
        captured["native_attention"] = out.clone()
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

    def record_decode(
        q, swa_pool, swa_slots, sinks, global_pool=None, global_slots=None
    ):
        captured["native_q"] = q.clone()
        if fixed_q:
            q = reference["q"].reshape_as(q).clone()
        out = decode_sparse(q, swa_pool, swa_slots, sinks, global_pool, global_slots)
        captured["native_attention"] = out.clone()
        if fixed_attention:
            out = reference["attention"].reshape_as(out).clone()
        captured.update(q=q.clone(), swa_slots=swa_slots.clone(), attention=out.clone())
        if global_slots is not None:
            captured["global_slots"] = global_slots.clone()
        return out

    with (
        patch.object(torch.nn.functional, "linear", record_linear),
        patch.object(prefill, "sparse_prefill_attention", record_prefill),
        patch.object(decode, "sparse_decode_attention", record_decode),
    ):
        if start == 0:
            positions, _, _ = prepare(owner, x, cache, [len(x)], tables=tables)
            output = owner(x.clone(), positions, cache, tables).clone()
        else:
            metadata = decode_metadata(owner, cache, [start], tables)
            output = owner.forward_decode(x[None].clone(), metadata, cache).clone()
    end = start + len(x)
    if start:
        decode_capture(owner, cache, tables, captured, end)
    if owner.compress_ratio:
        captured["index_cache"] = cache_values(
            owner, cache, tables, "indexer_kv", end // owner.compress_ratio
        )
        if owner.compress_ratio == 2 and end % 2:
            raw = cache.get_layer_cache(owner.cache_layer_id, "csa_state").kv_cache_base
            state = raw.view(torch.float32).reshape(
                raw.shape[0], cache.layouts["csa_state"].entries, 1024
            )
            page = int(tables["csa_state"][0, 0])
            captured["pending_state"] = state[page, (end - 1) % state.shape[1]].clone()
    captured["output"] = output
    return output, captured


def compare_capture(owner, reference, captured, identity):
    start, span = identity["start_pos"], len(identity["input_ids"])
    end, window_rows = start + span, 128 if start else span
    result = {
        key: difference(reference[key], captured[key])
        for key in ("q", "attention", "output")
    }
    result["native_q"] = difference(reference["q"], captured["native_q"])
    result["native_attention"] = difference(
        reference["attention"], captured["native_attention"]
    )
    result["kv"], result["selected_positions"] = aligned_sparse_inputs(
        reference, captured, window_rows
    )
    count = end // owner.compress_ratio if owner.compress_ratio else 0
    native = captured["kv"].reshape(-1, 512)
    expected = reference["kv"].reshape(-1, 512)

    def segment(label, left, right):
        result[label] = difference(left, right) if len(left) else {"rows": 0}

    segment("current_swa", expected[start:end], native[count + start : count + end])
    segment("retained_swa", expected[:start], native[count : count + start])
    if owner.compress_ratio:
        previous = start // owner.compress_ratio
        segment(
            "new_global", expected[window_rows + previous :], native[previous:count]
        )
        segment(
            "retained_global",
            expected[window_rows : window_rows + previous],
            native[:previous],
        )
        result["index_cache"] = difference(
            reference["index_cache"][:, :count], captured["index_cache"]
        )
        if "latent" in reference:
            result["latent"] = difference(reference["latent"], captured["latent"])
        if "pending_state" in captured:
            pending = torch.cat(
                (reference["kv_state"][0, 0], reference["score_state"][0, 0])
            ).reshape(1, -1)
            result["pending_state"] = difference(
                pending, captured["pending_state"].reshape(1, -1)
            )
            result["pending_position"] = end - 1
    return result


def fresh_cache(owner):
    from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_attention_owner_sm120 import (
        NativeTaggedCache,
    )

    cache = NativeTaggedCache([owner], gen=0)
    tables = {
        tag: torch.tensor([[2]], device="cuda", dtype=torch.int32)
        for tag in cache.layouts
    }
    return cache, tables


def compare_official_history(args, config, reader, steps, identities, recipe=None):
    from rtp_llm.models_py.modules.dsv4.fp8.test.v41_compressor_precision import (
        compressor_bf16,
    )
    from rtp_llm.models_py.modules.dsv4.fp8.test.v41_sparse_precision import (
        sparse_sink_div_rn,
    )
    from rtp_llm.models_py.modules.dsv4.fp8.test.v41_tp8_projection import (
        tp8_projection,
    )

    profiles = [
        ("baseline", False, False, False, False, False),
        ("tp8_projection", True, False, False, False, False),
        ("tp8_projection_compressor_bf16", True, True, False, False, False),
        ("tp8_projection_compressor_recipe", True, True, True, False, False),
        ("tp8_recipe_official_q", True, True, True, True, False),
        ("tp8_recipe_official_attention", True, True, True, False, True),
    ]
    profiles = [(*profile, False) for profile in profiles]
    if recipe and recipe["sparse_sink_div_rn_prefill"]:
        profiles.append(
            (official_history_control(recipe), True, True, True, False, False, True)
        )
    results, snapshots = {}, {}
    for name, tp8, rounded, fused, fixed_q, fixed_attention, sparse in profiles:
        results[name], snapshots[name] = {}, {}
        for layer in range(3):
            owner = native_attention(args, config, reader, layer)
            cache, tables = fresh_cache(owner)
            records, saved = [], []
            with tp8_projection(tp8), compressor_bf16(
                rounded, fused=fused, hca_projection=fused
            ), sparse_sink_div_rn(sparse, decode=False):
                for step, (data, identity) in enumerate(zip(steps, identities)):
                    reference = data["reference"][layer]
                    _, captured = execute_native(
                        owner,
                        cache,
                        tables,
                        data["inputs"][layer],
                        identity["start_pos"],
                        reference,
                        fixed_q=fixed_q,
                        fixed_attention=fixed_attention,
                    )
                    result = compare_capture(owner, reference, captured, identity)
                    records.append(result)
                    saved.append(cpu_snapshot(captured))
                    print(
                        json.dumps(
                            {
                                "history": "official_inputs",
                                "profile": name,
                                "layer": layer,
                                "step": step,
                                "output": result["output"],
                            }
                        ),
                        flush=True,
                    )
            results[name][str(layer)], snapshots[name][str(layer)] = records, saved
    return results, snapshots


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("reference_traces", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--input-artifact", type=Path, required=True)
    parser.add_argument("--native-traces", type=Path)
    parser.add_argument("--prompt-index", type=int, default=0, choices=(0, 1))
    parser.add_argument("--steps", type=int, default=3, choices=(2, 3))
    cli = parser.parse_args()
    if cli.report.exists() or cli.input_artifact.exists():
        raise ValueError("Replay artifacts must be fresh")
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("Attention history replay requires SM120 under GPU lock")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision("highest")
    paths = [
        cli.reference_traces / f"reference-prompt{cli.prompt_index}-step{step}.pt"
        for step in range(cli.steps)
    ]
    traces = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    native_paths = (
        [
            cli.native_traces / f"native-prompt{cli.prompt_index}-step{step}.pt"
            for step in range(cli.steps)
        ]
        if cli.native_traces is not None
        else []
    )
    native_traces = [
        torch.load(path, map_location="cpu", weights_only=True) for path in native_paths
    ]
    identities = history_identities(traces, native_traces if native_paths else None)
    root = cli.checkpoint.resolve()
    reference_kernel_sha256 = hashlib.sha256(
        (root / "inference/kernel.py").read_bytes()
    ).hexdigest()
    native_recipe = (
        validate_native_recipe(
            native_traces, reference_kernel_sha256=reference_kernel_sha256
        )
        if native_paths
        else None
    )
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
    with torch.device("cpu"):
        hash_state = official.NgramHashState(args, layout, tokenizer)
        hashes = [
            hash_state(
                torch.tensor([identity["input_ids"]], device="cpu", dtype=torch.long),
                identity["start_pos"],
            )
            for identity in identities
        ]
    steps, reconstruction, prepared_rows = reconstruct_official(
        official, converter, args, layout, reader, traces, identities, hashes
    )
    config = json.loads((root / "config.json").read_text())["text_config"]
    results, snapshots = compare_official_history(
        args, config, reader, steps, identities, native_recipe
    )
    (
        native_reconstruction,
        native_results,
        native_snapshots,
        native_inputs,
        native_mhc,
    ) = (
        {},
        [],
        [],
        [],
        [],
    )
    input_causality, causality_snapshots = {}, {}
    if native_paths:
        from rtp_llm.tools.dsv41_native_history_replay import (
            compare_prefix_current_histories,
            reconstruct_native,
        )

        (
            native_reconstruction,
            native_results,
            native_snapshots,
            native_inputs,
            native_mhc,
        ) = reconstruct_native(
            args,
            config,
            reader,
            steps,
            native_traces,
            identities,
            hashes,
            prepared_rows,
            hash_state.token_map.tolist(),
            native_recipe,
        )
        input_causality, causality_snapshots = compare_prefix_current_histories(
            args,
            config,
            reader,
            steps,
            identities,
            native_inputs,
            snapshots,
            native_snapshots,
            native_recipe,
        )
    with cli.input_artifact.open("xb") as handle:
        torch.save(
            {
                "identities": identities,
                "official_inputs": [
                    [value.cpu() for value in step["inputs"]] for step in steps
                ],
                "official_attention": [
                    [cpu_snapshot(value) for value in step["reference"]]
                    for step in steps
                ],
                "official_mhc": [
                    [cpu_snapshot(value) for value in step["mhc"]] for step in steps
                ],
                "official_input_history": snapshots,
                "native_inputs": [
                    [value.cpu() for value in step] for step in native_inputs
                ],
                "native_input_history": native_snapshots,
                "native_mhc": [
                    [cpu_snapshot(value) for value in step] for step in native_mhc
                ],
                "native_recipe": native_recipe,
                "prefix_current_substitutions": causality_snapshots,
            },
            handle,
        )
    repo = Path(__file__).resolve().parents[2]
    source_names = (
        "rtp_llm/tools/dsv41_attention_history_replay.py",
        "rtp_llm/tools/dsv41_attention_replay.py",
        "rtp_llm/tools/dsv41_native_history_replay.py",
        "rtp_llm/models_py/modules/dsv4/hc/delayed.py",
        "rtp_llm/models_py/modules/dsv4/hc/v41_prenorm.py",
        "rtp_llm/models_py/modules/dsv4/engram.py",
        "rtp_llm/models_py/modules/dsv4/_engram_triton.py",
        "rtp_llm/models_py/modules/base/cuda/norm.py",
        "rtp_llm/models_py/modules/dsv4/fp8/attention_v41.py",
        "rtp_llm/models_py/modules/dsv4/fp8/compressor.py",
        "rtp_llm/models_py/modules/dsv4/fp8/_v41_prefill_attention_sm120.py",
        "rtp_llm/models_py/modules/dsv4/fp8/_v41_decode_attention_sm120.py",
        "rtp_llm/models_py/modules/dsv4/fp8/_v41_decode_global.py",
        "rtp_llm/models_py/modules/dsv4/fp8/test/v41_compressor_precision.py",
        "rtp_llm/models_py/modules/dsv4/fp8/test/v41_tp8_projection.py",
        "rtp_llm/models_py/modules/dsv4/fp8/test/v41_mhc_precision.py",
        "rtp_llm/models_py/modules/dsv4/fp8/test/v41_sparse_precision.py",
        "rtp_llm/models_py/modules/dsv4/fp8/test/test_v41_attention_owner_sm120.py",
        "rtp_llm/models_py/modules/dsv4/fp8/test/test_v41_decode_owner_sm120.py",
        "rtp_llm/models_py/modules/dsv4/v41_attention.py",
        "rtp_llm/models_py/modules/dsv4/v41_cache.py",
        "rtp_llm/models_py/modules/dsv4/fp8/_v41_attention_triton.py",
    )
    report = {
        "prompt_index": cli.prompt_index,
        "identities": identities,
        "native_recipe": native_recipe,
        "official_reconstruction": reconstruction,
        "official_input_history": results,
        "native_reconstruction": native_reconstruction,
        "native_input_history": native_results,
        "prefix_current_substitutions": input_causality,
        "weight_sha256": reader.hashes,
        "selected_row_reads": reader.selected_row_reads,
        "official_source_sha256": {
            name: hashlib.sha256((root / "inference" / name).read_bytes()).hexdigest()
            for name in (
                "model.py",
                "kernel.py",
                "engram.py",
                "convert.py",
                "config.json",
            )
        },
        "reference_trace_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
        "native_trace_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in native_paths
        },
        "native_revision": subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repo}",
                "-C",
                str(repo),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "native_source_sha256": {
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
            for name in source_names
        },
        "input_artifact_sha256": hashlib.sha256(
            cli.input_artifact.read_bytes()
        ).hexdigest(),
        "limitations": [
            "One short L0-L2 request: retained caches use exact official attention inputs.",
            "Saved FFN outputs are substituted from each experiment's own trace; experts are not re-executed.",
            "Native-input history uses the validated composed recipe and its optional sparse-prefill/output-GEMM/mHC-post diagnostics.",
            "TP8 output partials are simulated serially, without live collectives.",
            "Serving, distributed topology, graphs, full-model accuracy and latency remain separate.",
        ],
    }
    with cli.report.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()

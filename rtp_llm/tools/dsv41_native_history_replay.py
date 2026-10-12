"""Reconstruct maintained native L0-L2 boundaries using saved FFN outputs.

This companion to the attention history diagnostic uses native Engram, delayed
mHC, RMSNorm and attention modules. Its caches and residuals are independent
of the experiment that fixes all attention inputs to the official history.
"""

import json
from unittest.mock import patch

import torch

from rtp_llm.tools.dsv41_attention_history_replay import (
    compare_capture,
    cpu_snapshot,
    execute_native,
    fresh_cache,
    official_history_control,
    require_exact,
)
from rtp_llm.tools.dsv41_attention_replay import difference, native_attention


class PreparedNativeEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rows = None

    def forward(self, indices, device):
        if self.rows is None or indices.shape != self.rows.shape[:-1]:
            raise ValueError("Prepared native Engram rows do not match hash rows")
        if self.rows.device != device:
            raise ValueError("Prepared native Engram rows are on the wrong device")
        return self.rows


def reconstruct_native(
    args,
    config,
    reader,
    steps,
    traces,
    identities,
    hashes,
    prepared_rows,
    token_map,
    recipe,
):
    from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm
    from rtp_llm.models_py.modules.dsv4.engram import (
        Engram,
        EngramLayout,
        NgramHashState,
        make_token_windows,
    )
    from rtp_llm.models_py.modules.dsv4.fp8.test.v41_compressor_precision import (
        compressor_bf16,
    )
    from rtp_llm.models_py.modules.dsv4.fp8.test.v41_mhc_precision import (
        fp32_mhc_prenorm,
        mhc_fp32,
    )
    from rtp_llm.models_py.modules.dsv4.fp8.test.v41_sparse_precision import (
        sparse_sink_div_rn,
    )
    from rtp_llm.models_py.modules.dsv4.fp8.test.v41_tp8_projection import (
        tp8_projection,
    )
    from rtp_llm.models_py.modules.dsv4.hc import v41_prenorm
    from rtp_llm.models_py.modules.dsv4.hc.delayed import DelayedHCUnit
    from rtp_llm.models_py.modules.dsv4.utils import V41MXFP8Linear

    owners = [native_attention(args, config, reader, layer) for layer in range(3)]
    caches = [fresh_cache(owner) for owner in owners]
    hcs, norms, engrams = [], [], {}
    previous = None
    layout = EngramLayout(config)
    hasher = NgramHashState(layout, token_map=token_map, device="cuda")
    for layer in range(3):
        layer_hcs, layer_norms = [], []
        for kind in ("attn", "ffn"):
            unit = DelayedHCUnit(
                reader.load(f"layers.{layer}.hc_{kind}_fn").float(),
                reader.load(f"layers.{layer}.hc_{kind}_base").float(),
                reader.load(f"layers.{layer}.hc_{kind}_scale").float(),
                dim=args.dim,
                hc_mult=args.hc_mult,
                hc_sinkhorn_iters=args.hc_sinkhorn_iters,
                hc_eps=args.hc_eps,
                norm_eps=args.norm_eps,
                layer_id=layer,
                name=kind,
            )
            unit.set_previous(previous)
            previous = unit
            layer_hcs.append(unit)
            layer_norms.append(
                RMSNorm(
                    reader.load(f"layers.{layer}.{kind}_norm.weight"), args.norm_eps
                )
            )
        hcs.append(layer_hcs)
        norms.append(layer_norms)
        if layer in layout.layer_ids:
            prefix = f"layers.{layer}.engram."
            embedding = PreparedNativeEmbedding()
            engrams[layer] = Engram(
                layout,
                layout.layer_ids.index(layer),
                embedding,
                V41MXFP8Linear(
                    reader.load(prefix + "wkv.weight"),
                    reader.load(prefix + "wkv.scale"),
                ),
                reader.load(prefix + "q_weight"),
                reader.load(prefix + "k_weight"),
                args.norm_eps,
            )
    inputs, captures, mhc, history, raw_mixes = [], [], [], [], []

    def record_prenorm(*args, **kwargs):
        result = fp32_mhc_prenorm(*args, **kwargs)
        raw_mixes.append(result.clone())
        return result

    with tp8_projection(True), mhc_fp32(True), compressor_bf16(
        True, fused=True, hca_projection=True
    ), sparse_sink_div_rn(
        recipe["sparse_sink_div_rn_prefill"],
        decode=recipe["sparse_sink_div_rn_decode"],
    ), patch.object(
        v41_prenorm, "prenorm", record_prenorm
    ):
        for step, (trace, identity) in enumerate(zip(traces, identities)):
            tokens = torch.tensor(identity["input_ids"], device="cpu", dtype=torch.long)
            depth = layout.max_ngram_size - 1
            lookback = list(reversed(history[-depth:]))
            lookback += [-1] * (depth - len(lookback))
            windows = make_token_windows(
                tokens, torch.tensor([0, len(tokens)]), torch.tensor([lookback])
            )
            native_hashes = hasher.hash_token_windows(windows.cuda())
            if not torch.equal(native_hashes.cpu(), hashes[step][0]):
                raise AssertionError(
                    f"Native/official Engram history hashes differ at step {step}"
                )
            hidden = (
                reader.rows("embed.weight", tokens[None])
                .cuda()[0]
                .unsqueeze(1)
                .repeat(1, args.hc_mult, 1)
            )
            if step:
                hidden = hidden[None]
            pre_mix = torch.zeros(hidden.shape[:-1], device="cuda", dtype=torch.float32)
            pre_mix[..., 0] = 1
            step_captures, step_mhc, step_inputs = [], [], []
            for layer in range(3):
                internal = {"layer_input": hidden.clone()}
                owner = owners[layer]
                attn_hc, ffn_hc = hcs[layer]
                attn_norm, ffn_norm = norms[layer]
                if layer in engrams:
                    engram = engrams[layer]
                    engram.embed_tokens.rows = prepared_rows[step][layer][0]
                    shape = hidden.shape
                    hidden = engram(
                        hidden.reshape(-1, args.hc_mult, args.dim),
                        native_hashes[:, layout.layer_ids.index(layer)],
                    ).reshape(shape)
                residual = hidden
                internal.update(
                    attention_residual=hidden.clone(), attn_read_mix=pre_mix.clone()
                )
                collapsed, post, comb = attn_hc.pre(hidden)
                internal.update(
                    attn_raw_mixes=raw_mixes.pop(),
                    attn_pre_mix=attn_hc.pre_mix_out.clone(),
                    attn_post_mix=post.squeeze(-1).clone(),
                    attn_comb_mix=comb.clone(),
                    attn_collapsed=collapsed.clone(),
                )
                x = attn_norm(collapsed.reshape(-1, args.dim)).reshape(-1, args.dim)
                internal["attn_norm"] = x.clone()
                step_inputs.append(x.clone())
                cache, tables = caches[layer]
                output, captured = execute_native(
                    owner,
                    cache,
                    tables,
                    x,
                    identity["start_pos"],
                    steps[step]["reference"][layer],
                )
                hidden = attn_hc.post(
                    output.reshape_as(collapsed), residual, post, comb
                )
                internal.update(
                    attention_output=output.clone(),
                    attn_post=hidden.clone(),
                    ffn_read_mix=attn_hc.pre_mix_out.clone(),
                )
                residual = hidden
                collapsed, post, comb = ffn_hc.pre(hidden)
                internal.update(
                    ffn_raw_mixes=raw_mixes.pop(),
                    ffn_pre_mix=ffn_hc.pre_mix_out.clone(),
                    ffn_post_mix=post.squeeze(-1).clone(),
                    ffn_comb_mix=comb.clone(),
                    ffn_collapsed=collapsed.clone(),
                )
                ffn_input = ffn_norm(collapsed.reshape(-1, args.dim)).reshape_as(
                    collapsed
                )
                internal["ffn_norm"] = ffn_input.clone()
                if layer < 2:
                    ffn_output = (
                        trace["tensors"][f"layer{layer:02d}.ffn.output"]
                        .cuda()
                        .reshape_as(collapsed)
                    )
                    internal["ffn_output"] = ffn_output.clone()
                    hidden = ffn_hc.post(ffn_output, residual, post, comb)
                    internal["ffn_post"] = hidden.clone()
                    pre_mix = ffn_hc.pre_mix_out
                step_mhc.append(internal)
                step_captures.append(captured)
            mhc.append(step_mhc)
            captures.append(step_captures)
            inputs.append(step_inputs)
            history.extend(identity["input_ids"])
    # Transfers and comparisons occur after all maintained forwards. Reject
    # reconstruction drift before interpreting the new internal boundaries.
    boundaries = {}
    for step, trace in enumerate(traces):
        for layer in range(3):
            label = f"native.step{step}.layer{layer:02d}.ffn.input"
            boundaries[label] = require_exact(
                trace["tensors"][f"layer{layer:02d}.ffn.input"],
                mhc[step][layer]["ffn_norm"],
                label,
            )
    records, snapshots = [], []
    for step, identity in enumerate(identities):
        step_records, step_snapshots = [], []
        for layer, owner in enumerate(owners):
            captured = captures[step][layer]
            result = compare_capture(
                owner, steps[step]["reference"][layer], captured, identity
            )
            result["attention_input"] = difference(
                steps[step]["inputs"][layer], inputs[step][layer]
            )
            result["mhc"] = {
                key: difference(steps[step]["mhc"][layer][key], value)
                for key, value in mhc[step][layer].items()
            }
            step_records.append(result)
            step_snapshots.append(cpu_snapshot(captured))
            print(
                json.dumps(
                    {
                        "history": "native_inputs",
                        "layer": layer,
                        "step": step,
                        "ffn_input": result["mhc"]["ffn_norm"],
                    }
                ),
                flush=True,
            )
        records.append(step_records)
        snapshots.append(step_snapshots)
    return boundaries, records, snapshots, inputs, mhc


def compare_prefix_current_histories(
    args,
    config,
    reader,
    steps,
    identities,
    native_inputs,
    official_snapshots,
    native_snapshots,
    recipe,
):
    """Hold prior attention inputs and the current input independently fixed."""
    from rtp_llm.models_py.modules.dsv4.fp8.test.v41_compressor_precision import (
        compressor_bf16,
    )
    from rtp_llm.models_py.modules.dsv4.fp8.test.v41_tp8_projection import (
        tp8_projection,
    )
    from rtp_llm.models_py.modules.dsv4.fp8.test.v41_sparse_precision import (
        sparse_sink_div_rn,
    )

    streams = {
        "official": [step["inputs"][2] for step in steps],
        "native": [step[2] for step in native_inputs],
    }
    controls = {
        "official": official_snapshots[official_history_control(recipe)]["2"],
        "native": [step[2] for step in native_snapshots],
    }
    results, snapshots = {}, {}
    with tp8_projection(True), compressor_bf16(
        True, fused=True, hca_projection=True
    ), sparse_sink_div_rn(
        recipe["sparse_sink_div_rn_prefill"],
        decode=recipe["sparse_sink_div_rn_decode"],
    ):
        for prefix in streams:
            for current in streams:
                name = f"{prefix}_prefix_{current}_current"
                records, saved = [], []
                for target in range(1, len(steps)):
                    owner = native_attention(args, config, reader, 2)
                    cache, tables = fresh_cache(owner)
                    for step in range(target):
                        execute_native(
                            owner,
                            cache,
                            tables,
                            streams[prefix][step],
                            identities[step]["start_pos"],
                            steps[step]["reference"][2],
                        )
                    _, captured = execute_native(
                        owner,
                        cache,
                        tables,
                        streams[current][target],
                        identities[target]["start_pos"],
                        steps[target]["reference"][2],
                    )
                    result = compare_capture(
                        owner,
                        steps[target]["reference"][2],
                        captured,
                        identities[target],
                    )
                    result["attention_input"] = difference(
                        steps[target]["inputs"][2], streams[current][target]
                    )
                    result["step"] = target
                    if prefix == current:
                        result["control_reconstruction"] = {
                            key: require_exact(
                                controls[prefix][target][key],
                                captured[key],
                                f"{name}.step{target}.{key}",
                            )
                            for key in ("q", "kv", "attention", "output")
                        }
                        for key in ("indices", "lengths"):
                            if not torch.equal(
                                controls[prefix][target][key], captured[key].cpu()
                            ):
                                raise AssertionError(
                                    f"History control index plan differs: {name} step {target} {key}"
                                )
                    records.append(result)
                    saved.append(cpu_snapshot(captured))
                    print(
                        json.dumps(
                            {
                                "history": name,
                                "step": target,
                                "output": result["output"],
                            }
                        ),
                        flush=True,
                    )
                results[name], snapshots[name] = records, saved
    return results, snapshots

"""Reconstruct maintained native L0-L2 boundaries using saved FFN outputs.

This companion to the attention history diagnostic uses native Engram, delayed
mHC, RMSNorm and attention modules. Its caches and residuals are independent
of the experiment that fixes all attention inputs to the official history.
"""

import torch

from rtp_llm.tools.dsv41_attention_history_replay import (
    compare_capture,
    cpu_snapshot,
    execute_native,
    fresh_cache,
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
    args, config, reader, steps, traces, identities, hashes, prepared_rows, token_map
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
    from rtp_llm.models_py.modules.dsv4.fp8.test.v41_mhc_precision import mhc_fp32
    from rtp_llm.models_py.modules.dsv4.fp8.test.v41_tp8_projection import (
        tp8_projection,
    )
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
    boundaries, records, snapshots, inputs, history = {}, [], [], [], []
    with tp8_projection(True), mhc_fp32(True), compressor_bf16(
        True, fused=True, hca_projection=True
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
            step_records, step_snapshots, step_inputs = [], [], []
            for layer in range(3):
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
                collapsed, post, comb = attn_hc.pre(hidden)
                x = attn_norm(collapsed.reshape(-1, args.dim)).reshape(-1, args.dim)
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
                residual = hidden
                collapsed, post, comb = ffn_hc.pre(hidden)
                ffn_input = ffn_norm(collapsed.reshape(-1, args.dim)).reshape_as(
                    collapsed
                )
                label = f"native.step{step}.layer{layer:02d}.ffn.input"
                boundaries[label] = require_exact(
                    trace["tensors"][f"layer{layer:02d}.ffn.input"], ffn_input, label
                )
                result = compare_capture(
                    owner, steps[step]["reference"][layer], captured, identity
                )
                result["attention_input"] = difference(steps[step]["inputs"][layer], x)
                step_records.append(result)
                step_snapshots.append(cpu_snapshot(captured))
                if layer < 2:
                    ffn_output = (
                        trace["tensors"][f"layer{layer:02d}.ffn.output"]
                        .cuda()
                        .reshape_as(collapsed)
                    )
                    hidden = ffn_hc.post(ffn_output, residual, post, comb)
            records.append(step_records)
            snapshots.append(step_snapshots)
            inputs.append(step_inputs)
            history.extend(identity["input_ids"])
    return boundaries, records, snapshots, inputs

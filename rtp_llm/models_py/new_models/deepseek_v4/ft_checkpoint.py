"""Read the V4 family's weights_convert format into its checkpoint owners.

The V4 writer preserves checkpoint-native projection bits (sp_id), stacks
rank-local routed experts and concatenates shared W1/W3. Only the vocabulary
uses the legacy hidden-column / padded-row TP layout. No legacy loader or
runtime ModelWeights object is involved.
"""

import re

import torch

from rtp_llm.models_py.expert_placement import ExpertPlacement
from rtp_llm.models_py.ft_checkpoint import ft_lm_head_source


def _vocab(checkpoint, model, name):
    tp = checkpoint.topology.tp_size
    vocab, dim = model._v4_args.vocab_size, model._v4_args.dim
    shapes = [checkpoint.read_rank_slice(name, rank) for rank in range(tp)]
    if name == "global.embedding":
        if shapes == [(vocab, dim)]:
            return checkpoint.read(name)
        if dim % tp or any(shape != (vocab, dim // tp) for shape in shapes):
            raise ValueError("Invalid V4 FT hidden-column embedding partition")
        # A V4.1/PPU embedding can consume its own hidden-column shard.
        leaf = model.v4.embed
        spec = leaf.checkpoint_read_spec("weight", (vocab, dim))
        needs_tied_head = (
            getattr(model, "tied_target_vocabulary", False)
            and "global.lm_head" not in checkpoint.entries
        )
        if (
            spec
            and spec.shard_dim == 1
            and spec.shard_count == tp
            and not needs_tied_head
        ):
            part = checkpoint.read_rank_slice(
                name, spec.shard_rank, (slice(None), slice(None))
            )
            return spec.wrap_sharded_tensor(part)
        result = None
        for rank in range(tp):
            part = checkpoint.read_rank_slice(name, rank, (slice(None), slice(None)))
            if result is None:
                result = part.new_empty((vocab, dim))
            if part.dtype != result.dtype:
                raise ValueError("V4 FT embedding peer dtype mismatch")
            result.narrow(1, rank * part.shape[1], part.shape[1]).copy_(part)
        return result

    head = model.v4.head
    if head.output_vocab_ids:
        raise ValueError("V4 FT does not encode the pruned head's source token IDs")
    block = shapes[0][0]
    if block * tp < vocab or any(shape != (block, dim) for shape in shapes):
        raise ValueError("Invalid V4 FT padded output-head partition")
    spec = head.tied_checkpoint_read_spec("weight", (vocab, dim))
    start, end, _ = spec.slices[0].indices(vocab)
    parts = []
    for rank in range(tp):
        left, right = max(start, rank * block), min(end, (rank + 1) * block)
        if left < right:
            parts.append(
                checkpoint.read_rank_slice(
                    name,
                    rank,
                    (slice(left - rank * block, right - rank * block), slice(None)),
                )
            )
    local = torch.cat(parts) if parts else head.weight.new_empty((0, dim))
    return spec.wrap_sharded_tensor(local)


def _hc_value(tensor, field):
    # AtomicWeight adds a trailing singleton to 1-D checkpoint scale tensors.
    if (
        field.startswith("v4.hc.")
        and field.endswith("_scale")
        and tensor.ndim == 2
        and tensor.shape[1] == 1
    ):
        return tensor[:, 0]
    return tensor


def _source_values(model, checkpoint):
    cfg = model.config
    role = cfg.model_type
    draft = role.endswith(("_mtp", "_dspark"))
    dspark = role.endswith("_dspark")
    v41 = role.startswith("deepseek_v41")
    last = len(model.v4.layers) - 1
    base = "mtp" if draft else "layers"
    final = f"mtp.{last}." if draft else ""
    include = model.checkpoint_weight_name_filter()
    loans = getattr(model, "_rtp_borrowed_parameter_bindings", {})
    globals_ = {
        "embedding": "embed.weight",
        "lm_head": "head.weight",
        "final_layernorm.gamma": final + "norm.weight",
    }
    if not v41:
        globals_.update(
            {"v4.hc.head_" + k: final + "hc_head_" + k for k in ("fn", "base", "scale")}
        )
    if role.endswith("_mtp"):
        for field in (
            "enorm.weight",
            "hnorm.weight",
            "e_proj.weight",
            "e_proj.scale",
            "h_proj.weight",
            "h_proj.scale",
        ):
            globals_["v4.mtp." + field] = "mtp.0." + field
    if dspark:
        for field in ("main_proj.weight", "main_proj.scale", "main_norm.weight"):
            globals_["v4.dspark." + field] = "mtp.0." + field
        for key, field in (
            ("w1", "embed" if v41 else "markov_w1"),
            ("w2", "head" if v41 else "markov_w2"),
        ):
            globals_[f"dspark_markov_{key}.weight"] = (
                final + f"markov_head.{field}.weight"
            )
    for source in checkpoint.entries:
        if source.startswith("global."):
            field = source[7:]
            if field == "final_layernorm.beta":
                beta = checkpoint.read(source)
                if tuple(beta.shape) != (model._v4_args.dim,) or torch.count_nonzero(
                    beta
                ):
                    raise ValueError(
                        "V4 RMSNorm cannot import a nonzero or malformed FT beta"
                    )
                continue
            if field not in globals_:
                raise ValueError(f"Unexpected V4 FT global: {source}")
            key = globals_[field]
            if (
                (key == "embed.weight" and "v4.embed.weight" in loans)
                or (key == "head.weight" and "v4.head.weight" in loans)
                or not include(key)
            ):
                continue
            if field in ("embedding", "lm_head"):
                yield key, _vocab(checkpoint, model, source)
            else:
                value = _hc_value(checkpoint.read(source), field)
                # Historical MTP exports also declared a standalone scale
                # after the FP8 projection. It stored native E8M0 as BF16.
                if (
                    field in ("v4.mtp.e_proj.scale", "v4.mtp.h_proj.scale")
                    and value.dtype == torch.bfloat16
                ):
                    recovered = value.to(torch.float8_e8m0fnu)
                    if not torch.equal(recovered.to(value.dtype), value):
                        raise ValueError(f"Non-E8M0 values in {source}")
                    value = recovered
                yield key, value
            continue

        _, raw_id, field = source.split(".", 2)
        index = int(raw_id)
        if not 0 <= index <= last:
            raise ValueError(f"V4 FT layer outside this model: {source}")
        prefix = f"{base}.{index}."
        expert = re.fullmatch(
            r"v4\.(shared|routed)\.(w[123]|w13)\.(weight|scale)", field
        )
        if expert:
            if model.v4.commit_only:
                continue
            kind, proj, attr = expert.groups()
            moe = model.v4.layers[index].ffn
            prepare = getattr(moe, "prepare_ft_projection", None)

            def expert_item(local_name, value):
                if prepare is not None:
                    value = prepare(local_name, value)
                return prefix + "ffn." + local_name, value

            if kind == "shared":
                value = checkpoint.read(source)
                if proj == "w13":
                    if not value.ndim or value.shape[0] % 2:
                        raise ValueError(f"Invalid fused V4 shared W13: {source}")
                    for p, part in zip(("w1", "w3"), value.chunk(2)):
                        yield expert_item(f"shared_experts.{p}.{attr}", part)
                else:
                    yield expert_item(f"shared_experts.{proj}.{attr}", value)
                continue
            placement = ExpertPlacement(
                moe.num_experts, moe.ep_size, moe.ep_rank, tuple(range(moe.num_experts))
            )
            values = checkpoint.read_expert_rows(source, index, placement)
            for logical, row in zip(placement.local_physical_to_logical, values):
                yield expert_item(f"experts.{logical}.{proj}.{attr}", row)
            continue
        hc = re.fullmatch(r"v4\.hc\.(attn|ffn)_(base|fn|scale)", field)
        if hc:
            suffix = f"hc_{hc[1]}_{hc[2]}"
        elif field.startswith("v4.router."):
            suffix = "ffn.gate." + field[len("v4.router.") :]
        elif field.startswith(("v4.compressor.", "v4.indexer.", "v41.indexer.")):
            suffix = "attn." + field.split(".", 1)[1]
        elif field.startswith(("v4.attn.", "v4.attn_norm.", "v4.ffn_norm.")):
            suffix = field[3:]
        else:
            raise ValueError(f"Unexpected V4 FT layer tensor: {source}")
        key = prefix + suffix
        if include(key):
            yield key, _hc_value(checkpoint.read(source), field)


def load_v4_ft_checkpoint(model, checkpoint):
    # FT usually serializes the head even when it was derived from embedding.
    # Keep the serialized bytes; recover who may write them from export metadata.
    serialized_tied_head = (
        getattr(model, "tied_target_vocabulary", False)
        and "global.lm_head" in checkpoint.entries
    )
    source = ft_lm_head_source(checkpoint) if serialized_tied_head else None
    model.load_weights(_source_values(model, checkpoint))
    if serialized_tied_head:
        model._ft_unknown_tied_vocab = source is None
        if source == "embedding":
            head = model.v4.head
            head._explicit_checkpoint_loaded = False
            head._tied_checkpoint_loaded = True
            head._tied_checkpoint_source_rows = model._v4_args.vocab_size

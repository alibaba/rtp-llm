"""Read-only rank-local physical LINEAR cache snapshots for the CP smoke."""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed.collective_torch import Group, _get_group, all_gather
from rtp_llm.models_py.model_desc.block_map import select_attention_inputs_for_layer
from rtp_llm.ops.attention_input_utils import select_prefill_position_ids
from rtp_llm.utils.model_weight import W, sp_0_pad8_size

_source_ids: dict[tuple[int, str], torch.Tensor] = {}
_sequence = 0


def _rank() -> int:
    return (
        dist.get_rank()
        if dist.is_available() and dist.is_initialized()
        else int(os.environ.get("RANK", "0"))
    )


def _capture(ctx, phase: str) -> None:
    global _sequence
    if os.environ.get("T09_CACHE_PROBE_ROLE") != "prefill":
        return
    model, inputs = ctx.args[:2]
    cache = model.kv_cache
    if cache is None:
        return
    linear_inputs = select_attention_inputs_for_layer(inputs, cache, 0)
    if isinstance(linear_inputs, list) or not linear_inputs.is_prefill:
        return
    cp_info = linear_inputs.context_parallel_info
    prefixes = linear_inputs.prefix_lengths.reshape(-1)
    lengths = (
        cp_info.prefill_actual_input_lengths_cpu
        if cp_info is not None
        else linear_inputs.input_lengths
    ).reshape(-1)
    if prefixes.numel() != 1 or lengths.numel() != 1:
        return
    prefix = int(prefixes[0])
    endpoint = prefix + int(lengths[0])
    observed_endpoints = (
        (800, 1300, 2000)
        if os.environ.get("T10_INTERNAL_FORK_STATE_REFERENCE") == "1"
        else (1300, 2000)
    )
    if endpoint not in observed_endpoints or (phase == "before" and prefix == 0):
        return

    rows = {}
    for layer_id in range(
        min(int(os.environ.get("T09_CACHE_PROBE_LAYERS", "3")), int(cache.layer_count))
    ):
        group_inputs = select_attention_inputs_for_layer(inputs, cache, layer_id)
        if isinstance(group_inputs, list):
            raise AssertionError(f"layer {layer_id}: expected one cache group")
        table = group_inputs.kv_cache_kernel_block_id_device
        if table is None or table.numel() == 0:
            continue
        for layer_cache in cache.get_layer_cache_groups(layer_id):
            tag = str(layer_cache.tag)
            key = (layer_id, tag)
            pages = cache.get_seq_size_per_block(
                tag
            ) // cache.get_kernel_seq_size_per_block(tag)
            ids = table.reshape(-1).to(torch.long)
            ids = ids[ids >= 0]
            save_positions = group_inputs.cache_save_positions
            save_blocks = group_inputs.cache_save_block_ids
            if save_blocks is not None and save_blocks.numel():
                physical = save_blocks.reshape(-1).to(
                    device=ids.device, dtype=torch.long
                )
                physical = physical[physical > 0]
                if physical.numel():
                    offsets = torch.arange(pages, device=ids.device)
                    ids = torch.cat(
                        (ids, (physical[:, None] * pages + offsets).reshape(-1))
                    )
            if phase == "after" and prefix == 0:
                _source_ids[key] = torch.unique(ids, sorted=True)
            if key in _source_ids:
                ids = torch.cat((ids, _source_ids[key]))
            ids = torch.unique(ids, sorted=True)
            ids = ids[ids < layer_cache.kv_cache_base.shape[0]]
            rows[key] = {
                "ids": ids.detach().cpu(),
                "values": layer_cache.kv_cache_base.index_select(0, ids)
                .detach()
                .cpu()
                .clone(),
                "runtime_table": table.detach().cpu().clone(),
                "save_positions": (
                    save_positions.detach().cpu().clone()
                    if save_positions is not None
                    else None
                ),
                "save_block_ids": (
                    save_blocks.detach().cpu().clone()
                    if save_blocks is not None
                    else None
                ),
                "seq_size_per_block": cache.get_seq_size_per_block(tag),
                "kernel_seq_size_per_block": cache.get_kernel_seq_size_per_block(tag),
            }
    if not rows:
        return
    directory = Path(os.environ["T09_CACHE_PROBE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    file = directory / f"prefill_rank{_rank()}_{phase}_{_sequence:03d}_end{endpoint}.pt"
    hidden = None
    positions = None
    if phase == "after" and os.environ.get("T09_CACHE_PROBE_LAYERS") == "4":
        hidden = ctx.result.hidden_states.detach().cpu().clone()
        positions = select_prefill_position_ids(linear_inputs).detach().cpu().clone()
    torch.save(
        {
            "rank": _rank(),
            "phase": phase,
            "prefix": prefix,
            "end_position": endpoint,
            "captured_unix_ns": time.time_ns(),
            "rows": rows,
            "hidden_states": hidden,
            "position_ids": positions,
        },
        file,
    )
    _sequence += 1


def before_model_forward(ctx) -> None:
    _begin_numeric(ctx)
    _capture(ctx, "before")


def after_model_forward(ctx) -> None:
    _capture(ctx, "after")
    _finish_numeric(ctx)
    if os.environ.get("T10_COMMON_PREFIX_BOTH_OFF") == "1":
        return
    if os.environ.get("T10_INTERNAL_FORK_STATE_REFERENCE") == "1":
        _run_private_fork_reference(ctx)
    else:
        _run_private_reference(ctx)


_numeric = None
_numeric_sequence = 0


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, (tuple, list)):
        return [_snapshot(item) for item in value]
    return value


def _begin_numeric(ctx):
    global _numeric
    _numeric = None
    if (
        os.environ.get("T10_GDN_NUMERIC_PROBE") != "1"
        or os.environ.get("T09_CACHE_PROBE_ROLE") != "prefill"
    ):
        return
    model, inputs = ctx.args[:2]
    if model.kv_cache is None:
        return
    attention = select_attention_inputs_for_layer(inputs, model.kv_cache, 0)
    if not attention.is_prefill:
        return
    prefix = attention.prefix_lengths.reshape(-1)
    cp = attention.context_parallel_info
    lengths = (
        cp.prefill_actual_input_lengths_cpu
        if cp is not None
        else attention.input_lengths
    ).reshape(-1)
    if prefix.numel() != 1 or lengths.numel() != 1:
        return
    end = int(prefix[0]) + int(lengths[0])
    if end not in (1300, 2000):
        return
    _numeric = {
        "prefix": int(prefix[0]),
        "end_position": end,
        "rank": _rank(),
        "position_ids": _snapshot(select_prefill_position_ids(attention)),
        "layers": {},
        "current_layer": None,
    }


def _finish_numeric(ctx):
    global _numeric, _numeric_sequence
    if _numeric is None:
        return
    _numeric["final_hidden"] = _snapshot(ctx.result.hidden_states)
    directory = Path(os.environ["T09_CACHE_PROBE_DIR"]).parent / "gdn_numeric_probe"
    directory.mkdir(exist_ok=True)
    torch.save(
        _to_cpu(_numeric),
        directory
        / f"rank{_rank()}_req{_numeric_sequence}_G{_numeric['prefix']}_N{_numeric['end_position']}.pt",
    )
    _numeric_sequence += 1
    _numeric = None


def before_decoder(ctx):
    if _numeric is None:
        return
    layer = int(ctx.args[0].layer_idx)
    _numeric["current_layer"] = layer
    _numeric["layers"][layer] = {
        "input_hidden": _snapshot(ctx.args[1]),
        "input_residual": _snapshot(ctx.args[2]),
    }


def after_decoder(ctx):
    if _numeric is None:
        return
    _numeric["layers"][int(ctx.args[0].layer_idx)]["output"] = _snapshot(ctx.result)


def after_input_project(ctx):
    if _numeric is not None:
        _numeric["layers"][_numeric["current_layer"]]["projection"] = _snapshot(
            ctx.result
        )


def before_conv(ctx):
    if _numeric is not None and _rank() == 0:
        _numeric["layers"][_numeric["current_layer"]]["conv_before"] = {
            key: _snapshot(ctx.kwargs.get(key))
            for key in (
                "x",
                "initial_states",
                "prefix_lengths",
                "query_start_loc",
                "block_map",
            )
        }


def after_conv(ctx):
    if _numeric is not None and _rank() == 0:
        _numeric["layers"][_numeric["current_layer"]]["conv_output"] = _snapshot(
            ctx.result
        )


def before_gdn_chunk(ctx):
    if _numeric is not None and _rank() == 0:
        _numeric["layers"][_numeric["current_layer"]]["fla_before"] = {
            "qkvgb": _snapshot(ctx.args[:5]),
            **{
                key: _snapshot(ctx.kwargs.get(key))
                for key in ("initial_state", "cu_seqlens", "state_positions")
            },
        }


def after_gdn_chunk(ctx):
    if _numeric is not None and _rank() == 0:
        _numeric["layers"][_numeric["current_layer"]]["fla_after"] = {
            "output": _snapshot(ctx.result[0]),
            "final_state": _snapshot(ctx.result[2]),
            "selected_states": _snapshot(ctx.kwargs.get("selected_states")),
        }


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_cpu(item) for item in value]
    return value


class _PrivateReferenceCache:
    """Small, independent real KV pools; no production cache bytes are copied."""

    def __init__(self, production, fork_reference=False, fork_cp_size=1):
        from rtp_llm.ops.compute_ops import LayerKVCache

        self._production = production
        self.group_tags = list(production.group_tags)
        self.layer_count = int(production.layer_count)
        self._layers = {}
        self.store_isolation = []
        self.fork_reference = fork_reference
        self.fork_cp_size = fork_cp_size
        self.fork_tables = None
        self.fork_full_source_table = None
        self.fork_selected = {}
        self.fork_stage = 0
        self.fork_seed_copies = []
        for layer_id in range(self.layer_count):
            for layer in production.get_layer_cache_groups(layer_id):
                tag = str(layer.tag)
                base = production.get_seq_size_per_block(tag)
                pages = base // production.get_kernel_seq_size_per_block(tag)
                # Reserve the actual 2000-token logical sequence, plus block 0.
                # FULL CP ownership uses fewer slots; keeping the same maximum
                # here avoids any assumption about LINEAR vs FULL tag names.
                runtime_blocks = (2000 + base - 1) // base
                count = (
                    (4 * (runtime_blocks + 1) + 3) * pages
                    if fork_reference
                    else (runtime_blocks + 1) * pages
                )
                values = layer.kv_cache_base
                private = torch.zeros(
                    (count, *values.shape[1:]),
                    dtype=values.dtype,
                    device=values.device,
                )
                scale = layer.kv_scale_base
                private_scale = None
                if scale is not None and scale.numel():
                    private_scale = torch.zeros(
                        (count, *scale.shape[1:]),
                        dtype=scale.dtype,
                        device=scale.device,
                    )
                self._layers[(layer_id, tag)] = LayerKVCache(
                    private,
                    layer.seq_size_per_block,
                    layer_id=layer_id,
                    tag=tag,
                    kv_scale_base=private_scale,
                )

    def get_layer_cache(self, layer_id):
        groups = self.get_layer_cache_groups(layer_id)
        if len(groups) != 1:
            raise AssertionError("Qwen reference expects one cache group per layer")
        return groups[0]

    def get_layer_cache_groups(self, layer_id):
        return [
            self._layers[(layer_id, tag)]
            for tag in self.group_tags
            if (layer_id, tag) in self._layers
        ]

    def get_seq_size_per_block(self, tag):
        return self._production.get_seq_size_per_block(tag)

    def get_kernel_seq_size_per_block(self, tag):
        return self._production.get_kernel_seq_size_per_block(tag)

    def get_entries_per_block(self, tag):
        return self._production.get_entries_per_block(tag)

    def attention_inputs(self, original, tag, *, stage=0, save_endpoint=None):
        original_writer = original.cache_store_writer
        had_store_inputs = original.cache_store_inputs is not None
        result = original.clone_without_cache_store()
        if (
            result.cache_store_inputs is not None
            or result.cache_store_writer is not None
        ):
            raise AssertionError(
                "private reference retained external publication handles"
            )
        if (
            original.cache_store_writer is not original_writer
            or (original.cache_store_inputs is not None) != had_store_inputs
        ):
            raise AssertionError(
                "reference cloning modified production publication handles"
            )
        for name in ("prefix_lengths_device", "input_lengths_device"):
            actual, cloned = getattr(original, name), getattr(result, name)
            if actual is None:
                if cloned is not None:
                    raise AssertionError(f"reference changed undefined mirror: {name}")
            elif cloned is None or actual.data_ptr() != cloned.data_ptr():
                raise AssertionError(f"reference lost the real device mirror: {name}")
        self.store_isolation.append(
            {
                "tag": tag,
                "production_had_store_inputs": had_store_inputs,
                "production_had_store_writer": original_writer is not None,
                "reference_store_inputs": False,
                "reference_store_writer": False,
            }
        )
        pages = self.get_seq_size_per_block(tag) // self.get_kernel_seq_size_per_block(
            tag
        )
        for field, subdivision in (
            ("kv_cache_block_id", 1),
            ("kv_cache_block_id_device", 1),
            ("kv_cache_kernel_block_id", pages),
            ("kv_cache_kernel_block_id_device", pages),
        ):
            table = getattr(original, field)
            if table is None or not table.numel():
                continue
            if table.shape[0] != 1:
                raise AssertionError("reference requires one real request")
            # Logical column j always maps to private physical block j+1;
            # kernel columns retain contiguous subpages within that block.
            columns = torch.arange(
                table.shape[1], device=table.device, dtype=table.dtype
            )
            base = self.get_seq_size_per_block(tag)
            runtime_blocks = (2000 + base - 1) // base
            stage_offset = stage * (runtime_blocks + 1) * subdivision
            remapped = columns + subdivision + stage_offset
            limit = runtime_blocks * subdivision
            remapped = torch.where(
                columns < limit, remapped, torch.zeros_like(remapped)
            )
            setattr(result, field, remapped.reshape(1, -1).contiguous())
        device = original.kv_cache_kernel_block_id_device.device
        positions = original.cache_save_positions
        blocks = original.cache_save_block_ids
        selected = (
            positions is not None
            and blocks is not None
            and save_endpoint is not None
            and positions.numel() > 0
        )
        if selected:
            matches = (positions == save_endpoint) & (blocks > 0)
            selected = bool(matches.any())
        if selected:
            selected_id = 3 * (runtime_blocks + 1) + (1 if stage == 0 else 2)
            result.cache_save_positions = torch.tensor(
                [[save_endpoint]], dtype=positions.dtype, device=positions.device
            )
            result.cache_save_block_ids = torch.tensor(
                [[selected_id]], dtype=blocks.dtype, device=blocks.device
            )
        else:
            result.cache_save_positions = torch.empty(
                (1, 0), dtype=torch.int32, device=device
            )
            result.cache_save_block_ids = torch.empty(
                (1, 0), dtype=torch.int32, device=device
            )
        return result

    def remember_fork_inputs(self, groups, *, stage, save_endpoint):
        self.fork_tables = {
            tag: group.kv_cache_kernel_block_id_device.detach().clone()
            for tag, group in groups.items()
        }
        if self.fork_cp_size > 1 and stage == 0:
            self.fork_full_source_table = self.fork_tables.get("full")
            if self.fork_full_source_table is None:
                raise AssertionError("private CP fork lacks A FULL source table")
        self.fork_selected = {}
        for tag, group in groups.items():
            positions = group.cache_save_positions
            blocks = group.cache_save_block_ids
            if positions is None or blocks is None or not positions.numel():
                continue
            matched = blocks[positions == save_endpoint]
            if matched.numel() != 1 or int(matched[0]) <= 0:
                raise AssertionError(f"private {tag} selected {save_endpoint} invalid")
            self.fork_selected[tag] = int(matched[0])
        self.fork_stage = stage

    def seed_fork_continuation(self, groups, *, stage, resume_endpoint):
        if self.fork_tables is None or self.fork_stage != stage - 1:
            raise AssertionError("private fork source stage missing")
        self.fork_seed_copies = []
        for tag, group in groups.items():
            source = self.fork_tables[tag]
            if self.fork_cp_size > 1 and tag == "full" and stage == 2:
                if self.fork_full_source_table is None:
                    raise AssertionError("private CP fork lost A FULL source table")
                source = self.fork_full_source_table
            destination = group.kv_cache_kernel_block_id_device
            pages = self.get_seq_size_per_block(
                tag
            ) // self.get_kernel_seq_size_per_block(tag)
            base = self.get_seq_size_per_block(tag)
            kernel = self.get_kernel_seq_size_per_block(tag)
            cp_full = self.fork_cp_size > 1 and tag == "full"
            last_slot = (resume_endpoint - 1) // (
                base * self.fork_cp_size if cp_full else base
            )
            for column in range((last_slot + 1) * pages):
                valid_tokens = kernel
                if cp_full:
                    global_block = (column // pages) * self.fork_cp_size + _rank()
                    remaining = (
                        resume_endpoint
                        - global_block * base
                        - (column % pages) * kernel
                    )
                    valid_tokens = min(kernel, max(0, remaining))
                    if valid_tokens == 0:
                        continue
                source_id = int(source[0, column])
                if tag in self.fork_selected and column // pages == last_slot:
                    source_id = self.fork_selected[tag] * pages + column % pages
                destination_id = int(destination[0, column])
                if source_id <= 0 or destination_id <= 0:
                    raise AssertionError(
                        f"private {tag} continuation lacks G{resume_endpoint} column {column}"
                    )
                self.fork_seed_copies.append(
                    (tag, column, source_id, destination_id, valid_tokens)
                )
                for layer_id in range(self.layer_count):
                    layer = self._layers.get((layer_id, tag))
                    if layer is None:
                        continue
                    base_values = layer.kv_cache_base
                    if valid_tokens == kernel or not cp_full:
                        base_values[destination_id].copy_(base_values[source_id])
                    else:
                        base_values[destination_id, ..., :valid_tokens, :].copy_(
                            base_values[source_id, ..., :valid_tokens, :]
                        )
                    scale = layer.kv_scale_base
                    if scale is not None and scale.numel():
                        if valid_tokens != kernel:
                            raise AssertionError(
                                "CP FULL partial scale layout unsupported"
                            )
                        scale[destination_id].copy_(scale[source_id])


_private_reference = None
_private_reference_endpoint = 0


def _private_model_forward(ctx, private_cache, private_groups):
    from rtp_llm.ops.compute_ops import PyModelInputs

    model, inputs = ctx.args[:2]
    private_inputs = PyModelInputs(
        input_ids=inputs.input_ids,
        input_hiddens=(
            inputs.input_hiddens if inputs.input_hiddens is not None else torch.empty(0)
        ),
        combo_position_ids=inputs.combo_position_ids,
        embedding_inputs=inputs.embedding_inputs,
        multimodal_inputs=inputs.multimodal_inputs,
        attention_inputs=private_groups,
        bert_embedding_inputs=inputs.bert_embedding_inputs,
    )
    private_inputs.capture_hidden_states = inputs.capture_hidden_states
    private_inputs.need_all_hidden_states = inputs.need_all_hidden_states
    private_inputs.need_all_logits = inputs.need_all_logits
    original = ctx.runtime._patched[ctx.target].original_callable
    production_cache = model.kv_cache
    production_hidden = ctx.result.hidden_states.detach().clone()
    try:
        model.kv_cache = private_cache
        with torch.inference_mode():
            result = original(model, private_inputs, fmha_impl=None)
            return result.hidden_states.detach().cpu().clone()
    finally:
        model.kv_cache = production_cache
        ctx.result.hidden_states = production_hidden


def _resolved_head_projection_comparison(
    model,
    produced,
    private,
    position_ids,
    *,
    valid_start: int,
    valid_end: int,
    atol: float,
    rtol: float,
):
    """Compare every real token against this rank's resolved vocabulary shard."""
    head = model.weight.get_global_weight(W.lm_head)
    if not isinstance(head, torch.Tensor):
        raise AssertionError("resolved W.lm_head is not a tensor")
    tp = int(model.parallelism_config.tp_size)
    rank = int(model.parallelism_config.tp_rank)
    vocab = int(model.config.vocab_size)
    hidden_width = int(model.config.hidden_size)
    if tp < 1 or not 0 <= rank < tp or vocab < 1:
        raise AssertionError(
            f"invalid resolved-head TP layout: tp={tp}, rank={rank}, vocab={vocab}"
        )
    if int(model.config.output_vocab_padded_size) != 0:
        raise AssertionError(
            "resolved-head oracle does not support output-vocabulary pruning"
        )
    local_rows = vocab if tp == 1 else sp_0_pad8_size(vocab, tp) // tp
    if (
        head.ndim != 2
        or head.shape != (local_rows, hidden_width)
        or head.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or produced.ndim != 2
        or private.ndim != 2
        or produced.shape != private.shape
        or produced.dtype != private.dtype
        or produced.shape[1] != hidden_width
        or produced.device != head.device
    ):
        raise AssertionError(
            f"unsupported resolved-head projection: head={tuple(head.shape)}/{head.dtype}/{head.device}, "
            f"produced={tuple(produced.shape)}/{produced.device}, private={tuple(private.shape)}, "
            f"expected local head=({local_rows},{hidden_width})"
        )
    if (
        produced.shape[0] <= 0
        or position_ids is None
        or position_ids.numel() % produced.shape[0]
    ):
        raise AssertionError("resolved-head position IDs do not match hidden rows")
    positions = position_ids.detach().cpu().reshape(produced.shape[0], -1)
    if positions.shape[1] not in (1, 3) or not torch.equal(
        positions, positions[:, :1].expand_as(positions)
    ):
        raise AssertionError(
            "resolved-head rows lack one canonical absolute token position"
        )
    positions = positions[:, 0]
    gathered_rank_rows = [produced.shape[0]]
    if tp > 1:
        if not dist.is_initialized():
            raise AssertionError(
                "resolved-head TP gather requires initialized distributed group"
            )
        process_group = _get_group(Group.TP)
        if (
            dist.get_world_size(process_group) != tp
            or dist.get_rank(process_group) != rank
        ):
            raise AssertionError(
                "resolved-head TP group differs from loaded LM-head ownership"
            )
        gathered_lengths = all_gather(
            torch.tensor([produced.shape[0]], device=head.device, dtype=torch.int64),
            group=Group.TP,
        )
        gathered_rank_rows = [int(value) for value in gathered_lengths.cpu().tolist()]
        max_rows = max(gathered_rank_rows)
        if min(gathered_rank_rows) <= 0:
            raise AssertionError("resolved-head TP rank has no hidden rows")

        def gather_padded(tensor):
            padded = torch.zeros(
                (max_rows, *tensor.shape[1:]), device=head.device, dtype=tensor.dtype
            )
            padded[: tensor.shape[0]].copy_(tensor.to(head.device))
            gathered = all_gather(padded.contiguous(), group=Group.TP)
            return torch.cat(
                [
                    gathered[i * max_rows : i * max_rows + rows]
                    for i, rows in enumerate(gathered_rank_rows)
                ]
            )

        produced = gather_padded(produced)
        private = gather_padded(private)
        positions = gather_padded(positions.to(torch.int64)).cpu()
    valid_indices = torch.nonzero(
        (positions >= valid_start) & (positions < valid_end)
    ).reshape(-1)
    if valid_indices.numel() == 0:
        raise AssertionError(
            f"resolved-head has no real token rows in [{valid_start},{valid_end})"
        )
    real_positions = positions.index_select(0, valid_indices)
    ordered_positions, order = torch.sort(real_positions)
    if not torch.equal(ordered_positions, torch.arange(valid_start, valid_end)):
        raise AssertionError(
            f"resolved-head TP rows do not cover every absolute position [{valid_start},{valid_end}) exactly once"
        )
    valid_indices = valid_indices.index_select(0, order)
    vocab_start = rank * local_rows
    valid_rows = min(local_rows, vocab - vocab_start)
    if valid_rows <= 0:
        raise AssertionError("resolved-head TP rank owns no real vocabulary rows")

    token_chunk, vocab_chunk = 64, 8192
    count = outside_count = 0
    maxabs = 0.0
    with torch.inference_mode():
        for token_start in range(0, valid_indices.numel(), token_chunk):
            token_end = min(token_start + token_chunk, valid_indices.numel())
            indices = valid_indices[token_start:token_end]
            actual_hidden = produced.index_select(0, indices.to(produced.device)).to(
                head.dtype
            )
            private_hidden = private.index_select(0, indices.to(private.device)).to(
                device=head.device, dtype=head.dtype
            )
            for local_start in range(0, valid_rows, vocab_chunk):
                local_end = min(local_start + vocab_chunk, valid_rows)
                weight = head[local_start:local_end]
                actual = torch.mm(actual_hidden, weight.T).float()
                expected = torch.mm(private_hidden, weight.T).float()
                delta = (actual - expected).abs()
                outside = ~torch.isclose(
                    actual, expected, atol=atol, rtol=rtol, equal_nan=False
                )
                outside_count += int(outside.sum().item())
                maxabs = max(
                    maxabs, float(delta.nan_to_num(nan=float("inf")).max().item())
                )
                count += actual.numel()
    return {
        "reference": "independent same-boundary private no-store whole-model hidden",
        "method": "resolved W.lm_head local-vocab torch.mm; cast hidden to head dtype; compare fp32 outputs",
        "native_cpp_logits_capture": False,
        "head_shape": list(head.shape),
        "head_dtype": str(head.dtype),
        "full_vocab_size": vocab,
        "model_layers": int(model.layer_num),
        "tp_size": tp,
        "tp_rank": rank,
        "local_vocab_range": [vocab_start, vocab_start + valid_rows],
        "token_chunk": token_chunk,
        "vocab_chunk": vocab_chunk,
        "absolute_positions": ordered_positions.tolist(),
        "gathered_tp_hidden": tp > 1,
        "gathered_rank_rows": gathered_rank_rows,
        "excluded_padding_rows": produced.shape[0] - valid_indices.numel(),
        "valid_position_range": [valid_start, valid_end],
        "compared_element_count": count,
        "atol": atol,
        "rtol": rtol,
        "maxabs": maxabs,
        "outside_tolerance_count": outside_count,
        "passes_original_tolerance": outside_count == 0,
    }


def _run_private_reference(ctx):
    """Two complete model forwards at 1300 and 700, with no selected writer."""
    global _private_reference, _private_reference_endpoint
    if (
        os.environ.get("T09_CACHE_PROBE_ROLE") != "prefill"
        or os.environ.get("T09_CACHE_PROBE_LAYERS") != "4"
    ):
        return
    model, inputs = ctx.args[:2]
    production = model.kv_cache
    if production is None:
        return
    attention = select_attention_inputs_for_layer(inputs, production, 0)
    if not attention.is_prefill or attention.prefix_lengths.numel() != 1:
        return
    cp = attention.context_parallel_info
    prefix = int(attention.prefix_lengths.reshape(-1)[0])
    lengths = (
        cp.prefill_actual_input_lengths_cpu
        if cp is not None
        else attention.input_lengths
    )
    endpoint = prefix + int(lengths.reshape(-1)[0])
    if (prefix, endpoint) not in ((0, 1300), (1300, 2000)):
        return
    if prefix == 0:
        if _private_reference is not None:
            raise AssertionError("duplicate private reference source")
        _private_reference = _PrivateReferenceCache(production)
    elif _private_reference is None or _private_reference_endpoint != prefix:
        raise AssertionError(
            "private reference must retain its own A1300 runtime state"
        )
    groups = inputs.attention_inputs
    if not isinstance(groups, dict):
        raise AssertionError("FULL+LINEAR reference requires real tagged inputs")
    hidden = _private_model_forward(
        ctx,
        _private_reference,
        {
            tag: _private_reference.attention_inputs(group, tag)
            for tag, group in groups.items()
        },
    )
    _private_reference_endpoint = endpoint
    directory = Path(os.environ["T09_CACHE_PROBE_DIR"]).parent / "private_reference"
    directory.mkdir(exist_ok=True)
    torch.save(
        {
            "rank": _rank(),
            "prefix": prefix,
            "end_position": endpoint,
            "hidden_states": hidden,
            "position_ids": select_prefill_position_ids(attention)
            .detach()
            .cpu()
            .clone(),
            "provenance": "whole_model_no_save_private_runtime_1300_then_700",
            "production_cache_bytes_copied": 0,
            "selected_save_slots": 0,
            "cache_store_isolation": _private_reference.store_isolation,
        },
        directory / f"rank{_rank()}_G{prefix}_N{endpoint}.pt",
    )
    if endpoint == 2000:
        _private_reference = None


_private_fork_reference = None


def _run_private_fork_reference(ctx):
    """Independent A, B and C forwards at the producer's real CP boundary."""
    global _private_fork_reference
    if (
        os.environ.get("T10_INTERNAL_FORK_STATE_REFERENCE") != "1"
        or os.environ.get("T09_CACHE_PROBE_ROLE") != "prefill"
        or os.environ.get("T09_CACHE_PROBE_LAYERS") != "4"
    ):
        return
    model, inputs = ctx.args[:2]
    production = model.kv_cache
    if production is None:
        return
    attention = select_attention_inputs_for_layer(inputs, production, 0)
    if not attention.is_prefill or attention.prefix_lengths.numel() != 1:
        return
    cp = attention.context_parallel_info
    cp_size = int(os.environ["T10_FORK_CP_SIZE"])
    if cp_size not in (1, 2) or (cp is not None) != (cp_size == 2):
        raise AssertionError("private fork CP topology differs from the real request")
    b_resume = 512 if cp_size == 2 else 768
    prefix = int(attention.prefix_lengths.reshape(-1)[0])
    lengths = (
        cp.prefill_actual_input_lengths_cpu
        if cp is not None
        else attention.input_lengths
    )
    endpoint = prefix + int(lengths.reshape(-1)[0])
    stages = {
        (0, 1300): (0, b_resume),
        (b_resume, 2000): (1, 800),
        (800, 2000): (2, None),
    }
    if (prefix, endpoint) not in stages:
        return
    stage, save_endpoint = stages[(prefix, endpoint)]
    if stage == 0:
        if _private_fork_reference is not None:
            raise AssertionError("duplicate private fork A1300")
        _private_fork_reference = _PrivateReferenceCache(
            production, fork_reference=True, fork_cp_size=cp_size
        )
    elif (
        _private_fork_reference is None
        or _private_fork_reference.fork_stage != stage - 1
    ):
        raise AssertionError(f"missing private fork source for stage {stage}")
    cache = _private_fork_reference
    groups = inputs.attention_inputs
    if not isinstance(groups, dict):
        raise AssertionError("FULL+LINEAR fork reference requires tagged inputs")
    private_groups = {
        tag: cache.attention_inputs(
            group, tag, stage=stage, save_endpoint=save_endpoint
        )
        for tag, group in groups.items()
    }
    if stage:
        cache.seed_fork_continuation(
            private_groups, stage=stage, resume_endpoint=prefix
        )
    hidden = _private_model_forward(ctx, cache, private_groups)
    projection = None
    if os.environ.get("T10_QWEN35_HEAD_PROJECTION") == "1":
        projection = _resolved_head_projection_comparison(
            model,
            ctx.result.hidden_states,
            hidden,
            select_prefill_position_ids(attention),
            valid_start=prefix,
            valid_end=endpoint,
            atol=1e-2,
            rtol=1e-2,
        )
    if save_endpoint is not None:
        cache.remember_fork_inputs(
            private_groups, stage=stage, save_endpoint=save_endpoint
        )
    directory = (
        Path(os.environ["T09_CACHE_PROBE_DIR"]).parent / "private_fork_reference"
    )
    directory.mkdir(exist_ok=True)
    state_rows = {}
    if stage == 1:
        for (layer_id, tag), layer in cache._layers.items():
            selected = cache.fork_selected.get(tag)
            if selected is None:
                if layer_id != 3:
                    raise AssertionError(
                        f"private F800 LINEAR state absent: {layer_id}/{tag}"
                    )
                continue
            pages = cache.get_seq_size_per_block(
                tag
            ) // cache.get_kernel_seq_size_per_block(tag)
            ids = selected * pages + torch.arange(
                pages, dtype=torch.long, device=layer.kv_cache_base.device
            )
            state_rows[(layer_id, tag)] = (
                layer.kv_cache_base.index_select(0, ids).detach().cpu().clone()
            )
    torch.save(
        {
            "rank": _rank(),
            "prefix": prefix,
            "end_position": endpoint,
            "hidden_states": hidden,
            "resolved_head_projection": projection,
            "position_ids": select_prefill_position_ids(attention)
            .detach()
            .cpu()
            .clone(),
            "state_rows": state_rows,
            "private_stage": stage,
            "private_selected_block_ids": dict(cache.fork_selected),
            "private_seed_copies": list(cache.fork_seed_copies),
            "cp_size": cp_size,
            "production_cache_bytes_copied": 0,
            "cache_store_isolation": list(cache.store_isolation),
            "provenance": f"whole_model_private_no_store_A0_1300_G{b_resume}_B2000_F800_C2000_cp{cp_size}",
        },
        directory / f"rank{_rank()}_G{prefix}_N{endpoint}.pt",
    )
    if stage == 2:
        _private_fork_reference = None

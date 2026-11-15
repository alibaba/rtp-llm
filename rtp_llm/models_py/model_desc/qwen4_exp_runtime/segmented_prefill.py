"""Qwen's opt-in prefill plan: captured GDN and bounded QSA score regions.

FP8 scale producers and their projections always see the original live rows.
Only prepared GDN receives capacity padding. PLE/HC/MoE and QSA state writers
remain live in the ordinary forward loop and its invocation finally.
"""

import bisect
from collections.abc import Mapping

import torch
from torch import nn

from rtp_llm.models_py.model_desc.block_map import get_attention_inputs_value
from rtp_llm.models_py.model_desc.cuda_graph.segmented_prefill import GraphSegment
from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextMetadata
from rtp_llm.models_py.triton_kernels.fla.prepared import GdnPrefillPrepared
from rtp_llm.ops import HybridAttentionType
from rtp_llm.ops.compute_ops import PyAttentionInputs


def _signature(tensor):
    return (
        tensor.data_ptr(),
        tensor.shape,
        tensor.stride(),
        tensor.dtype,
        tensor.device,
    )


def _has_tensor(tensor):
    return isinstance(tensor, torch.Tensor) and tensor.numel() > 0


def _qsa_prefill_model_supported(model):
    pc = model.parallelism_config
    return (
        model.__dict__.get("_ordinary_decode_graph_plan", False)
        and not bool(getattr(model.config, "is_mtp", False))
        and not bool(getattr(model, "_capture_mtp_target_hidden", False))
        and bool(getattr(model.config, "_qwen4_full_fp8_nvfp4", False))
        and str(pc.role_type).rsplit(".", 1)[-1] == "PDFUSION"
        and pc.tp_size in (2, 4, 8)
        and pc.tp_size == pc.get_attn_tp_size() == pc.get_ffn_tp_size()
        and pc.ep_size == pc.dp_size == pc.pp_size == 1
    )


def _qsa_prefill_request_supported(inputs, capacity):
    """Read this invocation's native host metadata and actual text payload."""
    if capacity not in (2048, 8192) or inputs.input_ids.numel() != capacity:
        return False
    mm, embedding = inputs.multimodal_inputs, inputs.embedding_inputs
    if (
        _has_tensor(inputs.input_hiddens)
        or mm.multimodal_features
        or mm.mm_extra_input
        or _has_tensor(mm.mm_features_spans)
        or _has_tensor(mm.mm_features_locs)
        or _has_tensor(embedding.text_tokens_mask)
        or _has_tensor(embedding.combo_tokens_type_ids)
    ):
        return False
    groups = get_attention_inputs_value(inputs)
    if not isinstance(groups, Mapping) or not groups:
        return False
    for item in groups.values():
        if (
            not item.is_prefill
            or item.is_target_verify
            or item.is_cuda_graph
            or item.is_s_padded
            or item.context_parallel_info is not None
            or item.cache_store_inputs is not None
        ):
            return False
        for tensor, expected in (
            (item.input_lengths, capacity),
            (item.prefix_lengths, 0),
        ):
            if not (
                tensor.device.type == "cpu"
                and tensor.dtype == torch.int32
                and tensor.shape == (1,)
                and tensor.tolist() == [expected]
            ):
                return False
    return True


class _QsaPrefillScoreCore:
    """One weight-free score/selection graph per complete geometry and bucket."""

    def __init__(self, owner, indexers):
        from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import Qwen4ExpQSAIndexer

        self.owner = owner
        self.indexers = tuple(indexers)
        self.indexer_type = Qwen4ExpQSAIndexer
        self.select = Qwen4ExpQSAIndexer._select_causal_scores
        self.forward = Qwen4ExpQSAIndexer.forward_causal_from_projected
        self._borrowed = None

    def geometry(self, indexer):
        if (
            type(indexer) is not self.indexer_type
            or type(indexer)._select_causal_scores is not self.select
            or type(indexer).forward_causal_from_projected is not self.forward
            or any(
                name in indexer.__dict__
                for name in ("_select_causal_scores", "forward_causal_from_projected")
            )
        ):
            return None
        geometry = (
            indexer.n_heads,
            indexer.kv_heads,
            indexer.head_dim,
            indexer.compress_ratio,
            indexer.token_budget,
            indexer.block_topk,
            indexer.max_selected,
        )
        return geometry if geometry == (4, 1, 128, 4, 2048, 512, 2051) else None

    def check_bindings(self):
        if any(
            indexer._prefill_score_core is not self or self.geometry(indexer) is None
            for indexer in self.indexers
        ):
            raise RuntimeError(
                "segmented QSA score binding changed; recapture required"
            )

    def __call__(self, indexer, q, keys, positions):
        self._borrowed = None
        owner, context = self.owner, self.owner.current
        geometry = self.geometry(indexer)
        if not (
            owner.executing
            and owner.qsa_prefill_active
            and context is not None
            and any(indexer is item for item in self.indexers)
            and geometry is not None
            and context.capacity in (2048, 8192)
            and context.live_tokens == context.capacity
            and q.shape == (1, context.capacity, 4, 128)
            and keys.shape == (1, context.capacity // 4, 128)
            and positions.shape == (context.capacity,)
            and q.dtype == keys.dtype == torch.bfloat16
            and positions.dtype == torch.int64
            and q.device == keys.device == positions.device == owner.device
            and q.stride(-1) == keys.stride(-1) == 1
            and positions.is_contiguous()
            and not any(t.requires_grad for t in (q, keys, positions))
        ):
            return None
        arguments = (q, keys, positions)
        key = geometry, tuple(
            (tuple(t.shape), t.stride(), t.dtype, t.device) for t in arguments
        )
        region = context.qsa_regions.get(key)
        if region is None:
            if not owner.recording:
                return None
            from rtp_llm.models_py.kernels.cuda.qsa.indexer_prefill_score import (
                try_prefill_score,
            )

            bridges = tuple(
                torch.empty(t.shape, dtype=t.dtype, device=t.device) for t in arguments
            )
            output = torch.empty(
                (1, context.capacity, 2051), dtype=torch.int32, device=owner.device
            )
            for bridge, value in zip(bridges, arguments):
                bridge.copy_(value)

            def run(q, keys, positions):
                scores = try_prefill_score(q, keys)
                if scores is None:
                    raise RuntimeError("segmented QSA score backend is unavailable")
                return self.select(indexer, scores, positions)

            # Keep even a partially captured graph alive in the outer quarantine.
            region = GraphSegment.__new__(GraphSegment)
            region._qsa_function = run
            context.qsa_regions[key] = region
            GraphSegment.__init__(
                region, run, bridges, output, owner.pool, owner.capture_stream
            )
        for bridge, value in zip(region.arguments, arguments):
            bridge.copy_(value)
        result = self._execute(region)
        if owner.recording:
            context.qsa_captured.add(id(indexer))
        self._borrowed = context, owner.qsa_epoch, indexer, region.output
        return result

    def _execute(self, region):
        return region.replay()

    def borrowed_selection(self, indexer, selected):
        borrowed, self._borrowed = self._borrowed, None
        owner = self.owner
        if not (
            borrowed is not None
            and owner.executing
            and owner.qsa_prefill_active
            and borrowed[0] is owner.current
            and borrowed[1] == owner.qsa_epoch
            and borrowed[2] is indexer
        ):
            return False
        bank = borrowed[3].squeeze(0)
        return _signature(selected) == _signature(bank) and (
            selected.storage_offset() == bank.storage_offset()
        )


def warmup_shapes(longest, live_limit, rows, page):
    """Cover max-Q/max-KV, ragged grids and the SM120 paged-TMA boundary."""
    if min(longest, live_limit, rows, page) <= 0:
        raise ValueError("segmented prefill requires positive budget dimensions")
    ragged = [(live_limit - rows + 1, 0)] + [(1, 0)] * (rows - 1)
    prefix = (longest - ragged[0][0]) // page * page
    candidates = [
        [(live_limit, 0)],
        [(live_limit // rows, 0)] * rows,
        ragged,
        [(ragged[0][0], prefix)] + ragged[1:],
    ]
    # One legal prefix request warms the paged backend without materializing
    # Bcap full histories. The remaining batched scratch is bounded below.
    if live_limit >= 128:
        bound = min(4096, longest)
        limit = min(bound - page, live_limit)
        # Prefixes are page aligned, including the last partial context page.
        query = (limit - bound % page) // page * page + bound % page
        if query >= 128:
            candidates.append([(query, bound - query)])
    return list(dict.fromkeys(tuple(shape) for shape in candidates))


def extra_runtime_bytes(model, live_limit, rows, longest, page):
    extra = sum(
        ple.ple_embedding.remaining_prefetch_budget_bytes()
        for ple in model.ple_layers.values()
    )
    # Eager SM120 CP has three process-wide caches, separate from prepared
    # graph workspaces. Smaller intermediate calls can choose a smaller CP
    # chunk than either extreme probe. CP>=64 gives a finite growth bound;
    # adding its full capacity covers old/new cache storage during growth too.
    linear = [
        layer.self_attn.prefill_gdn
        for layer in model.layers
        if layer.layer_type == HybridAttentionType.LINEAR
    ]
    heads = max(getattr(gdn, "original", gdn).local_num_v_heads for gdn in linear)
    shapes = warmup_shapes(longest, live_limit, rows, page)
    if live_limit >= 2048:
        state_bytes = heads * 128 * 128 * 4
        extra += 3 * (rows + (live_limit - rows) // 64) * state_bytes
        measured_pages = max(
            (
                sum(n // 128 for n, _ in shape)
                for shape in shapes
                if any(p for _, p in shape)
            ),
            default=0,
        )
        extra += max(0, live_limit // 128 - measured_pages) * state_bytes
    tma_rows = min(32, rows, live_limit // 128)
    bound = min(4096, longest)
    if tma_rows and bound >= page + 128:
        columns = (bound + page - 1) // page
        # Membership also grows with live tokens, independently of page-table
        # width. A same-width prefix probe need not cover that high-water mark.
        extra += 4 * live_limit * ((bound + 31) // 32)
        probed = any(
            len(shape) == tma_rows
            and min(n for n, _ in shape) >= 128
            and max(p for _, p in shape) > 0
            and max(n for n, _ in shape) + max(p for _, p in shape) <= 4096
            and (max(n for n, _ in shape) + max(p for _, p in shape) + page - 1) // page
            >= columns
            for shape in shapes
        )
        if not probed:
            # FP8 paged staging, bounded by the TMA gate.
            blocks = tma_rows * columns
            extra += (blocks + 1) * (2 * page * 256 * 2 + 2 * page * 4)
            extra += 4 * blocks
    return extra


class _QkvProjectionRegion(nn.Module):
    def __init__(self, original, owner):
        super().__init__()
        self.original = original
        self.owner = owner

    def forward(self, hidden):
        # Keep the original small-M backend and all non-segmented calls.
        if (
            self.owner.executing
            and hidden.ndim == 2
            and hidden.dtype == torch.bfloat16
            and not self.original._should_use_flashinfer(hidden)
        ):
            return self.owner.current.project(self.original, hidden)
        return self.original(hidden)


class _GdnRegion(nn.Module):
    def __init__(self, original, index):
        super().__init__()
        self.original = original
        self.index = index

    def forward(self, qkv, b, a, attention_inputs, cache, metadata):
        execution = getattr(metadata, "segmented_prefill", None)
        if execution is None:
            return self.original(qkv, b, a, attention_inputs, cache, metadata)
        output = execution.gdn(self.index, self.original, (qkv, b, a), cache)
        # Keep the native GDN publication at its eager layer boundary.
        if cache is not None and attention_inputs.cache_store_inputs is not None:
            attention_inputs.cache_store_writer.write(
                attention_inputs.cache_store_inputs, cache
            )
        return output


class QwenPrefillContext:
    def __init__(self, owner, capacity, dummy):
        self.owner = owner
        self.capacity = capacity
        self.prepared = GdnPrefillPrepared(
            capacity, owner.max_requests, owner.device, output_live_only=True
        )
        self.prepared.refresh([capacity], [0])
        self.tables = {}
        groups = get_attention_inputs_value(dummy)
        for index, layer in owner.linear_layers:
            tag = owner.model._attention_tag(index)
            if tag not in self.tables:
                table = groups[tag].kv_cache_kernel_block_id_device
                page = owner.model.kv_cache.get_kernel_seq_size_per_block(tag)
                # Prefix history consumes table columns, not captured token
                # capacity. Keep all deployment-valid checkpoint addresses.
                width = max(
                    table.shape[1], (owner.model.config.max_seq_len + page - 1) // page
                )
                self.tables[tag] = torch.full(
                    (owner.max_requests, width),
                    -1,
                    device=owner.device,
                    dtype=torch.int32,
                )
                self.tables[tag][:1, : table.shape[1]].copy_(table)
        self.metadata = Qwen3NextMetadata(
            prefill_conv1d_meta=self.prepared.conv_metadata,
            gdn_prefill_prepared=self.prepared,
        )
        self.regions = {}
        self.bridges = {}
        self.projections = {}
        self.live_tokens = capacity
        self.qsa_regions = {}
        self.qsa_captured = set()

    def project(self, original, hidden):
        key = (original.N, hidden.dtype, hidden.device)
        if key not in self.projections:
            self.projections[key] = hidden.new_empty((self.capacity, original.N))
        # Quantization still sees only the live input. The original DeepGEMM
        # route writes the packed QKV/Z into the plan-owned output bank.
        return original(hidden, out=self.projections[key][: hidden.shape[0]])

    def gdn(self, index, original, arguments, cache):
        if index not in self.regions:
            if not self.owner.recording:
                raise RuntimeError("segmented GDN region was not captured")
            tag = self.owner.model._attention_tag(index)
            # Identical GDN geometries share caller-owned bridges. Graph-pool
            # temporaries die inside capture and can be reused in model order.
            geometry = tuple((tuple(x.shape[1:]), x.dtype) for x in arguments)
            bridges = self.bridges.get(geometry)
            if bridges is None:

                def bridge(value):
                    # Borrow only the view actually written by the producer.
                    for packed in self.projections.values():
                        live = packed[: value.shape[0], : value.shape[1]]
                        if _signature(live) == _signature(value):
                            return packed[:, : value.shape[1]]
                    return value.new_zeros((self.capacity, *value.shape[1:]))

                bridges = tuple(bridge(value) for value in arguments)
                self.bridges[geometry] = bridges
            output_key = ("output", geometry)
            output = self.bridges.get(output_key)
            if output is None:
                # The SM120 producer already writes this plan-owned buffer.
                # Other backends still copy their result into it in GraphSegment.
                output = self.prepared.get_workspace(
                    "gdn_sm120_output",
                    (self.capacity, original.local_num_v_heads, original.head_v_dim),
                    arguments[0].dtype,
                    arguments[0].device,
                )
                self.bridges[output_key] = output
            static_inputs = PyAttentionInputs()
            static_inputs.is_prefill = True
            static_inputs.input_lengths = torch.zeros(
                self.owner.max_requests, dtype=torch.int32
            )
            static_inputs.kv_cache_kernel_block_id_device = self.tables[tag]
            for bridge, value in zip(bridges, arguments):
                bridge[: value.shape[0]].copy_(value)

            def run(qkv, b, a):
                return original(qkv, b, a, static_inputs, cache, self.metadata)

            self.regions[index] = GraphSegment(
                run, bridges, output, self.owner.pool, self.owner.capture_stream
            )
        region = self.regions[index]
        for bridge, value in zip(region.arguments, arguments):
            if value.dtype != bridge.dtype or value.shape[1:] != bridge.shape[1:]:
                raise RuntimeError("segmented GDN projection geometry changed")
            live = bridge[: self.live_tokens]
            if _signature(live) != _signature(value):
                live.copy_(value)
            bridge[self.live_tokens :].zero_()
        return region.replay()[: self.live_tokens]

    def refresh(self, inputs, lengths):
        groups = get_attention_inputs_value(inputs)
        prefixes = next(iter(groups.values())).prefix_lengths
        self.prepared.refresh(lengths, prefixes)
        self.live_tokens = sum(lengths)
        for tag, destination in self.tables.items():
            source = groups[tag].kv_cache_kernel_block_id_device
            destination.fill_(-1)
            # Admission checks the required columns. Native tables may contain
            # extra unused capacity, which is safe to omit from this fixed bank.
            width = min(source.shape[1], destination.shape[1])
            destination[: len(lengths), :width].copy_(source[:, :width])


class QwenSegmentedPrefillAdapter:
    def __init__(self, model, buckets, max_requests, dummy_factory):
        from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpModel

        if type(model) is not Qwen4ExpModel or model.kv_cache is None:
            raise ValueError(
                "segmented prefill requires the Qwen4 target and bound caches"
            )
        if bool(getattr(model.config, "is_mtp", False)):
            raise ValueError("segmented prefill does not yet support MTP")
        self.model = model
        self.buckets = sorted(set(buckets))
        if not self.buckets or any(n < 128 for n in self.buckets):
            raise ValueError("Qwen segmented prefill requires buckets >= 128")
        self.max_requests = max_requests
        self.dummy_factory = dummy_factory
        self.linear_layers = [
            (i, layer)
            for i, layer in enumerate(model.layers)
            if layer.layer_type == HybridAttentionType.LINEAR
        ]
        if not self.linear_layers:
            raise ValueError("Qwen segmented prefill requires GDN layers")
        self.linear_tags = {model._attention_tag(i) for i, _ in self.linear_layers}
        self.pools = []
        self.page_geometry = {}
        for i in range(len(model.layers)):
            for cache in model.kv_cache.get_layer_cache_groups(i):
                span, pages = self.page_geometry.get(
                    cache.tag,
                    (cache.seq_size_per_block, cache.kv_cache_base.shape[0]),
                )
                if span != cache.seq_size_per_block:
                    raise ValueError(
                        "segmented prefill requires consistent tag geometry"
                    )
                self.page_geometry[cache.tag] = (
                    span,
                    min(pages, cache.kv_cache_base.shape[0]),
                )
                for field in ("kv_cache_base", "kv_scale_base"):
                    tensor = getattr(cache, field)
                    if isinstance(tensor, torch.Tensor) and tensor.numel():
                        self.pools.append(
                            (i, cache.tag, field, tensor, _signature(tensor))
                        )
        self.device = self.pools[0][3].device
        if self.device.type != "cuda" or torch.version.hip is not None:
            raise ValueError("segmented prefill requires NVIDIA CUDA")
        self.pool = torch.cuda.graph_pool_handle()
        self.capture_stream = torch.cuda.Stream(device=self.device)
        self.contexts = {}
        self.current = None
        self.recording = False
        self.executing = False
        self.installed = []
        self.qsa_core = None
        self.qsa_prefill_active = False
        self.qsa_epoch = 0

    def capture(self):
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
            CudaFp8GEMMLinear,
        )

        self.recording = True
        if _qsa_prefill_model_supported(self.model) and (
            torch.cuda.get_device_capability(self.device) == (12, 0)
        ):
            indexers = tuple(
                layer.self_attn.qsa_indexer
                for layer in self.model.layers
                if getattr(layer.self_attn, "qsa_indexer", None) is not None
            )
            core = _QsaPrefillScoreCore(self, indexers)
            if indexers and all(
                core.geometry(indexer) is not None for indexer in indexers
            ):
                self.qsa_core = core
                for indexer in indexers:
                    self.installed.append(
                        (indexer, "_prefill_score_core", indexer._prefill_score_core)
                    )
                    indexer._prefill_score_core = core
        for index, layer in self.linear_layers:
            original = layer.self_attn.prefill_gdn
            layer.self_attn.prefill_gdn = _GdnRegion(original, index)
            self.installed.append((layer.self_attn, "prefill_gdn", original))
            projection = layer.self_attn.in_proj_qkvz
            if type(projection) is CudaFp8GEMMLinear and projection.bias is None:
                layer.self_attn.in_proj_qkvz = _QkvProjectionRegion(projection, self)
                self.installed.append((layer.self_attn, "in_proj_qkvz", projection))
        for capacity in reversed(self.buckets):
            dummy = self.dummy_factory(capacity)
            context = QwenPrefillContext(self, capacity, dummy)
            self.contexts[capacity] = context
            self.current = context
            # Real cache-backed execution measures every eager unit too; it
            # cannot take the model's cacheless embedding warmup shortcut.
            expect_qsa = self.qsa_core is not None and _qsa_prefill_request_supported(
                dummy, capacity
            )
            self.forward(dummy)
            if len(context.regions) != len(self.linear_layers):
                raise RuntimeError("segmented prefill did not capture every GDN layer")
            if expect_qsa and context.qsa_captured != {
                id(indexer) for indexer in self.qsa_core.indexers
            }:
                raise RuntimeError(
                    "segmented prefill did not capture every QSA score producer"
                )
        self.recording = False
        self.current = None
        self.dummy_factory = None

    def select(self, inputs, preparing=False):
        if (
            inputs.multimodal_inputs.multimodal_features
            and str(self.model.parallelism_config.role_type).rsplit(".", 1)[-1]
            != "PDFUSION"
        ):
            return 0, "request_not_supported"
        groups = get_attention_inputs_value(inputs)
        if not isinstance(groups, Mapping) or not groups:
            return 0, "input_metadata_invalid"
        lengths = prefixes = None
        for item in groups.values():
            if not item.is_prefill or item.is_target_verify or item.is_cuda_graph:
                return 0, "request_not_supported"
            if any(
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or tensor.dtype != torch.int32
                or tensor.ndim != 1
                for tensor in (item.input_lengths, item.prefix_lengths)
            ):
                return 0, "input_metadata_invalid"
            values = tuple(item.input_lengths.tolist())
            prefix_values = tuple(item.prefix_lengths.tolist())
            if lengths is None:
                lengths, prefixes = values, prefix_values
            if (
                values != lengths
                or prefix_values != prefixes
                or len(prefix_values) != len(values)
                or not values
                or any(n <= 0 for n in values)
                or any(p < 0 or p + n >= 2**31 for p, n in zip(prefixes, lengths))
            ):
                return 0, "input_metadata_invalid"
            if any(p % 128 for p in prefixes):
                return 0, "prefix_cache_not_supported"
        if any(
            p + n > self.model.config.max_seq_len for p, n in zip(prefixes, lengths)
        ):
            return 0, "request_not_supported"
        if len(lengths) > self.max_requests:
            return 0, "request_count_exceed_capture_limit"
        total = sum(lengths)
        position = bisect.bisect_left(self.buckets, total)
        if total < 128 or position == len(self.buckets):
            return 0, "input_tokens_exceed_capture_limit"
        bucket = self.buckets[position]
        # Retain the existing SM120 eager/DSL boundary at 2048 live tokens.
        if total < 2048 <= bucket and torch.cuda.get_device_capability(self.device) == (
            12,
            0,
        ):
            return 0, "request_not_supported"
        if not preparing:
            if inputs.input_ids.numel() != total or _has_tensor(inputs.input_hiddens):
                return 0, "input_metadata_invalid"
            # The ordinary forward owns live multimodal embedding/positions.
            # Only its projected GDN inputs enter these captured regions.
            if _has_tensor(inputs.embedding_inputs.combo_tokens_type_ids):
                return 0, "token_type_input_not_supported"
            if not self._pages_supported(groups):
                return 0, "request_not_supported"
        return bucket, "replayed"

    def _pages_supported(self, groups):
        supported = True
        for tag, item in groups.items():
            span, pages = self.page_geometry[tag]
            # Opaque FULL state views keep physical pages, unlike MHA's
            # kernel-page view. Their native host table carries that geometry.
            host = (
                item.kv_cache_kernel_block_id
                if span == self.model.kv_cache.get_kernel_seq_size_per_block(tag)
                else item.kv_cache_block_id
            )
            if (
                not isinstance(host, torch.Tensor)
                or host.device.type != "cpu"
                or host.dtype != torch.int32
                or host.ndim != 2
            ):
                raise ValueError("segmented prefill requires native host page tables")
            if host.shape[0] != item.input_lengths.numel():
                return False
            reads, writes = set(), set()
            if tag in self.linear_tags:
                source = item.kv_cache_kernel_block_id_device
                if (
                    not isinstance(source, torch.Tensor)
                    or source.device != self.device
                    or source.dtype != torch.int32
                    or source.ndim != 2
                    or source.shape[0] != host.shape[0]
                ):
                    return False
                capacity = (self.model.config.max_seq_len + span - 1) // span
            for row, (length, prefix) in enumerate(
                zip(item.input_lengths.tolist(), item.prefix_lengths.tolist())
            ):
                count = (prefix + length + span - 1) // span
                first = prefix // span if tag in self.linear_tags else 0
                selected = host[row, first:count]
                if bool(torch.any(selected >= pages)):
                    raise ValueError(
                        f"segmented prefill out-of-range live pages for {tag}"
                    )
                if selected.numel() != count - first:
                    supported = False
                elif tag in self.linear_tags:
                    if prefix % span or count > min(capacity, source.shape[1]):
                        return False
                    if bool(torch.any((selected == 0) | (selected < -1))):
                        raise ValueError(
                            "segmented prefill has invalid LINEAR checkpoint pages"
                        )
                    # The allocator omits unused intermediate checkpoints when
                    # reuse is off. GDN/conv scatter masks those -1 pages; the
                    # live final state must still have its own positive page.
                    supported = supported and bool(selected[-1] > 0)
                    for page in selected.tolist():
                        if page > 0:
                            if page in writes:
                                return False
                            writes.add(page)
                    if prefix:
                        read_page = (prefix - 1) // span
                        initial = host[row, read_page : read_page + 1]
                        if initial.numel() != 1 or not 0 < int(initial[0]) < pages:
                            raise ValueError(
                                "segmented prefill has invalid LINEAR prefix page"
                            )
                        reads.add(int(initial[0]))
                elif bool(torch.any(selected <= 0)):
                    # Sparse/cyclic layouts remain on their existing eager
                    # route until this plan validates their writer positions.
                    supported = False
            if reads & writes:
                return False
        return supported

    def prepare(self, inputs):
        bucket, reason = self.select(inputs)
        if not bucket:
            raise ValueError(
                f"segmented prefill admission changed after prepare: {reason}"
            )
        for i, tag, field, owner, signature in self.pools:
            current = getattr(self.model.kv_cache.get_layer_cache(i, tag), field)
            if _signature(current) != signature:
                raise RuntimeError(
                    "segmented prefill cache binding changed; recapture required"
                )
        if self.qsa_core is not None:
            self.qsa_core.check_bindings()
        groups = get_attention_inputs_value(inputs)
        self.current = self.contexts[bucket]
        lengths = next(iter(groups.values())).input_lengths.tolist()
        self.current.refresh(inputs, lengths)

    def forward(self, inputs):
        fmha = self.model.prepare_fmha_impl(inputs, False)
        self.qsa_epoch += 1
        self.qsa_prefill_active = (
            self.qsa_core is not None
            and _qsa_prefill_model_supported(self.model)
            and _qsa_prefill_request_supported(inputs, self.current.capacity)
        )
        self.executing = True
        try:
            return self.model.forward(inputs, fmha, segmented_prefill=self.current)
        finally:
            self.executing = False
            self.qsa_prefill_active = False
            if self.qsa_core is not None:
                self.qsa_core._borrowed = None

    def close(self):
        # The outer owner has drained the last invocation and its auxiliary work.
        self.contexts.clear()
        self.current = None
        for module, field, original in self.installed:
            setattr(module, field, original)
        self.installed.clear()
        self.qsa_core = None
        self.pools.clear()

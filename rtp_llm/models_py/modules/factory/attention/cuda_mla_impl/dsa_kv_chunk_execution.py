"""Prepare KV-only DSA plans and reusable phase buffers before execution."""

from dataclasses import dataclass

import torch

from rtp_llm.models_py.modules.dsv4.indexer_topk import (
    native_candidate_selection_supported,
    native_page_rr_selection_supported,
)

from ..mla_prefill_plan import prefill_kv_workspace_budget_bytes
from .dsa_kv_chunk_plan import (
    indexer_selection_scratch_bytes,
    plan_dsa_kv_chunks,
    prefill_indexer_score_bytes,
)
from .dsa_kv_workspace import _bytes, indexer_candidate_workspace_shapes, staged_workspace_bytes


@dataclass
class DsaPreparedKVBlock:
    descriptor: object
    start: int
    width: int
    key_positions: torch.Tensor
    key_requests: object = None
    slices: object = None
    dcp_positions: object = None
    host_slices: object = None
    indexer_starts: object = None
    indexer_ends: object = None
    native_context: object = None
    native_table: object = None
    native_schedule: object = None
    indexer_decode_positions: object = None
    fp8_dequant_metadata: object = None


def _prefill_block_layout(block, request_count, device):
    # Generate token-sized metadata on the device. CPU staging is bounded
    # by request count and kept alive for its nonblocking H2D copy.
    positions, requests = [], []
    slices = [[0, 0, 0] for _ in range(request_count)]
    offset = 0
    for request, start, length in zip(block.requests, block.starts, block.lengths):
        positions.append(
            torch.arange(start, start + length, dtype=torch.int32, device=device)
        )
        requests.append(
            torch.full((length,), request, dtype=torch.int32, device=device)
        )
        slices[request] = [start, start + length, offset]
        offset += length
    host = torch.tensor(slices, dtype=torch.int32, device="cpu", pin_memory=True)
    return (
        positions[0] if len(positions) == 1 else torch.cat(positions),
        requests[0] if len(requests) == 1 else torch.cat(requests),
        host.to(device=device, non_blocking=True),
        host,
    )


class DsaKVExecution:
    @staticmethod
    def planning_key(params, name):
        owner = params.workspace_owner
        prefill = params.is_prefill
        rows = params.query_positions.numel() if prefill else params.positions_d.numel()
        batch = 1 if prefill else params.decode.local_causal_lens.shape[0]
        graph_key = (batch, rows // batch) if params.is_cuda_graph else None
        phase = owner.phase(name, graph_key=graph_key)
        lengths = (
            tuple(params.kv_lens_host)
            if prefill
            else (params.block_table.shape[1] * params.kernel_page_size,)
        )
        # Descriptors depend on cache geometry, not physical block IDs or
        # query contents. Refresh block IDs separately on every preparation.
        # Captured/resident buffers affect the budget and invalidate this key.
        return (
            prefill,
            params.is_cuda_graph,
            rows,
            batch,
            params.positions_d.numel(),
            params.shard_size,
            params.shard_rank,
            params.page_size,
            params.kernel_page_size,
            tuple(params.block_table.shape),
            lengths,
            prefill_kv_workspace_budget_bytes() if prefill else owner.budgets[name],
            owner.other_resident_bytes(phase),
            owner.native_exact_topk,
            owner.indexer_use_fp8,
            owner.sfa_use_fp8,
            owner.layout_bytes,
            owner.routing_plan is not None,
            owner.route_bytes,
            owner.route_capacity,
            owner.route_chunk_rows,
            owner.route_allocation_granularity,
        )

    def __init__(self, params, name):
        owner = params.workspace_owner
        self.owner, self.name = owner, name
        self.fp8_indexer = name == "indexer" and owner.indexer_use_fp8
        self.direct_native_page_size = None
        self.fp8_sfa = name == "sfa" and owner.sfa_use_fp8
        self.prefill = params.is_prefill
        kv_budget = prefill_kv_workspace_budget_bytes() if self.prefill else None
        self.rows = (
            params.query_positions.numel()
            if self.prefill
            else params.positions_d.numel()
        )
        self.graph = params.is_cuda_graph
        self.batch = 1 if self.prefill else params.decode.local_causal_lens.shape[0]
        self.queries = self.rows // self.batch
        self.shards = params.shard_size
        self.selection_rows = (
            self.rows if self.prefill else (self.rows + self.shards - 1) // self.shards
        )
        graph_key = (self.batch, self.queries) if self.graph else None
        self.workspace = owner.phase(name, graph_key=graph_key)
        graph_decode = self.graph and not self.prefill
        capacity_batch = max(owner.graph_max_requests, self.batch) if graph_decode else self.batch
        capacity_queries = owner.graph_max_queries if graph_decode else self.queries
        if self.queries > capacity_queries:
            raise ValueError("DSA Graph query width exceeds its initialized speculative capacity")
        capacity_rows = capacity_batch * capacity_queries if graph_decode else self.rows
        capacity_selection_rows = (
            (capacity_rows + self.shards - 1) // self.shards
            if graph_decode else self.selection_rows
        )
        frozen = self.workspace.graph_capacity if self.graph else None
        geometry = (tuple(params.block_table.shape[1:]), params.page_size,
                    params.kernel_page_size, self.shards, params.shard_rank)
        if frozen is not None and (
            capacity_batch > frozen["batch"] or geometry != frozen["geometry"]
        ):
            raise ValueError("DSA Graph scratch must capture its largest bucket first with fixed KV geometry")
        self.blocks = []
        lengths = (
            params.kv_lens_host
            if self.prefill
            else [params.block_table.shape[1] * params.kernel_page_size]
        )
        features = (
            (132 if self.fp8_indexer else owner.index_dim)
            if name == "indexer"
            else (656 if self.fp8_sfa else owner.latent_dim + owner.suffix_dim)
        )
        staged_sfa = self.prefill and name == "sfa"
        mega_topk = name == "indexer" and not self.prefill and owner.mega_topk
        total_history = sum(lengths)
        tokens = capacity_rows if graph_decode else params.positions_d.numel()
        external = owner.external_projection_bytes(tokens, indexer=name == "indexer")
        # Source Q, CP/DCP layouts, FlashMLA's kernel-owned padded partial
        # latent output/max/LSE and collective restore buffers are live too.
        # These buffers are allocated by existing kernels/collectives, rather
        # than incorrectly counted as caller-owned workspace.
        if name == "indexer":
            if owner.routing_plan is None:
                external += capacity_rows * owner.index_heads * (owner.index_dim * 2 + 4)
            if self.fp8_indexer:
                # BF16 projection source, new E4M3 Q, packed/unpacked scales
                # and folded FP32 head weights coexist until QK completion.
                if owner.routing_plan is None:
                    external += capacity_rows * owner.index_heads * (owner.index_dim + 16)
                if not self.prefill:
                    if params.kernel_page_size % 64 or params.page_size % 64:
                        raise ValueError("FP8 paged Indexer needs whole native64 subpages")
                    from rtp_llm.models_py.modules.base.cuda.fp8_indexer_score import NativeFp8IndexerScore

                    if owner.indexer_score is None:
                        owner.indexer_score = NativeFp8IndexerScore()
                    self.num_sms = torch.cuda.get_device_properties(owner.device).multi_processor_count
                    if owner.indexer_score.supports_direct_pages(params.kernel_page_size, owner.device):
                        self.direct_native_page_size = params.kernel_page_size
                    probe = owner.indexer_score.prepare_paged_metadata(
                        torch.zeros(capacity_batch, capacity_queries, device=owner.device, dtype=torch.int32),
                        num_sms=self.num_sms, page_size=self.direct_native_page_size or 64,
                    )
                    self.native_schedule_bytes = probe.numel() * probe.element_size()
        else:
            kernel_heads = 64 if owner.heads <= 64 else 128
            external += (
                capacity_rows * (3 * owner.heads + kernel_heads)
                * (owner.latent_dim + owner.suffix_dim) * 2
                + capacity_rows * kernel_heads * (owner.latent_dim * 2 + 8)
                + capacity_rows * owner.heads * owner.value_dim * 4
                + capacity_rows * owner.topk * 4
            )
        # Page descriptors, global positions/request IDs, staged block tables,
        # per-block mapping metadata and small kernel/BLAS alignment scratch.
        # This reserve is checked against observed allocator peaks in tests.
        external += (
            64 * 1024 * 1024 + sum(lengths) * 16 + params.block_table.numel() * 16
        )
        external += owner.other_resident_bytes(self.workspace)

        def specifications(block):
            width = block.tokens
            q, k = capacity_selection_rows, owner.topk
            s = {}

            def add(n, shape, dtype):
                s[n] = (shape, dtype)

            if self.prefill:
                pages = block.owner_pages(params.page_size, self.shards)
                dtype = torch.uint8 if self.fp8_indexer or self.fp8_sfa else torch.bfloat16
                add("cache.pack", (pages, params.page_size, features), dtype)
                add(
                    "cache.gather",
                    (self.shards, pages, params.page_size, features),
                    dtype,
                )
                add("cache.restore", (width, features), dtype)
                if self.fp8_sfa:
                    add(
                        "cache.dequant",
                        (width, owner.latent_dim + owner.suffix_dim),
                        torch.bfloat16,
                    )
            elif name == "indexer":
                if self.fp8_indexer:
                    add("cache.native64", (1 + capacity_batch * (width // 64), 64, 1, 132), torch.uint8)
                else:
                    add("cache.restore", (capacity_batch, width, features), torch.bfloat16)
            if self.graph and not self.prefill:
                # Immutable owner/global positions are shared by all batch
                # buckets. Mutable block IDs and native schedules are not.
                key = str(block.starts[0])
                add("positions." + key, (width,), torch.int32)
                add("dcp_positions." + key, (self.shards * width,), torch.int32)
            if name == "indexer" and mega_topk:
                # The MQA output (charged in scratch) feeds MegaTopk directly; several KV blocks
                # are first placed side by side.
                if block.tokens < total_history:
                    add("mega_topk.logits", (capacity_rows, total_history), torch.float32)
                add("mega_topk.positions", (capacity_rows, owner.topk), torch.int32)
                add("mega_topk.count", (capacity_rows,), torch.int32)
            elif name == "indexer":
                if not self.fp8_indexer or not self.prefill:
                    add("score", (capacity_rows, width), torch.float32)
                select_width = width if self.prefill else self.shards * width
                if not self.prefill:
                    padded = q * self.shards
                    for n in ("wire", "received"):
                        add(n, (padded, width), torch.float32)
                    page_rr = self.fp8_indexer and owner.native_exact_topk and native_page_rr_selection_supported(
                        width, k, shards=self.shards, page=params.page_size, start=block.starts[0],
                    )
                    if not page_rr:
                        add("complete", (q, select_width), torch.float32)
                        if self.fp8_indexer and owner.native_exact_topk:
                            add("decode.ends", (q,), torch.int32)
                            if self.graph:
                                add("decode.positions." + str(block.starts[0]), (select_width,), torch.int32)
                    add("gathered_indices", (padded, k), torch.int32)
                for prefix, columns in (("block", select_width), ("merge", 2 * k)):
                    s.update(indexer_candidate_workspace_shapes(
                        q, columns, k, prefix=prefix, native=owner.native_exact_topk,
                        # The first block writes directly to running. A second
                        # candidate pair is needed only by later KV blocks.
                        needs_block_output=block.tokens < total_history,
                    ))
                add("running.scores", (q, k), torch.float32)
                add("running.positions", (q, k), torch.int32)
                add("merge.input_scores", (q, 2 * k), torch.float32)
                add("merge.input_positions", (q, 2 * k), torch.int32)
            else:
                add("slots", (capacity_rows, owner.topk), torch.int32)
                if not self.prefill:
                    add("lengths", (capacity_rows,), torch.int32)
                    if owner.sfa_metadata_free:
                        # topk_length of the metadata-free FP8 decode kernel: how
                        # many of the packed slots this owner holds. The operator
                        # reads the same owner flag, so the two cannot disagree.
                        add("owned_slots", (capacity_rows,), torch.int32)
                    add("partial_lse", (capacity_rows, owner.heads), torch.float32)
                if not staged_sfa:
                    if block.tokens < total_history:
                        add(
                            "projected",
                            (capacity_rows, owner.heads, owner.value_dim),
                            torch.bfloat16,
                        )
                        add(
                            "running",
                            (capacity_rows, owner.heads, owner.value_dim),
                            torch.float32,
                        )
                        add("running_lse", (capacity_rows, owner.heads), torch.float32)
                    add(
                        "output",
                        (capacity_rows, owner.heads, owner.value_dim),
                        torch.bfloat16,
                    )
                else:
                    final = {
                        "output": (
                            (self.rows, owner.heads, owner.value_dim),
                            torch.bfloat16,
                        )
                    }
                    if block.tokens < total_history:
                        add(
                            "running",
                            (self.rows, owner.heads, owner.latent_dim),
                            torch.float32,
                        )
                        add("running_lse", (self.rows, owner.heads), torch.float32)
                        final.update({n: s[n] for n in ("running", "running_lse")})
                        final["latent"] = (
                            (self.rows, owner.heads, owner.latent_dim),
                            torch.bfloat16,
                        )
                    # Cache pack/gather/restore and final BF16 latent/output
                    # never coexist. Reuse their arena after the last KV launch.
                    return (s, final)
            return s

        def scratch(block):
            if name != "indexer":
                return 0
            native = (((capacity_rows + 63) // 64) * 64
                      * ((block.tokens + 63) // 64) * 64 * 4) if self.fp8_indexer else 0
            if mega_topk:
                return native
            columns = block.tokens * (1 if self.prefill else self.shards)
            # Native selection and tie repair use only the declared buffers
            # plus CTA shared memory; no hidden Torch select/sort allocation.
            selection = max(
                0 if owner.native_exact_topk and native_candidate_selection_supported(
                    width, owner.topk, prefix=prefix
                ) else indexer_selection_scratch_bytes(
                    capacity_selection_rows, width, owner.topk
                )
                for prefix, width in (("block", columns), ("merge", 2 * owner.topk))
            )
            # Charge MQA's padded output allocation in both phases.
            native = 0
            if self.fp8_indexer:
                if kv_budget is None:
                    native = ((capacity_rows + 63) // 64 * 64
                              * ((block.tokens + 63) // 64 * 64) * 4)
                else:
                    native = prefill_indexer_score_bytes(capacity_rows, block.tokens)
            return selection + native

        def metadata_bytes(plan):
            if self.fp8_sfa and self.prefill:
                return len(plan.blocks) * 3 * 4
            if self.graph and not self.prefill:
                # All capture buckets/cache groups remain live. Reserve their
                # mutable metadata up front, not just the current bucket.
                # Capturing every B in [1,R] is a conservative superset of
                # the engine's alignment-deduplicated capture grid.
                requests = capacity_batch
                groups = owner.graph_cache_groups
                summed_batches = requests * (requests + 1) // 2
                query_widths = capacity_queries
                total = groups * query_widths * summed_batches * params.block_table.shape[1] * 8
                if self.fp8_indexer:
                    total += groups * sum(
                        query_widths * summed_batches * (block.tokens // 64) * 4
                        + summed_batches * query_widths * (query_widths + 1) // 2 * 4
                        + query_widths * requests * self.native_schedule_bytes
                        for block in plan.blocks
                    )
                return total
            if not self.fp8_indexer:
                return 0
            if self.prefill:
                return len(plan.blocks) * self.rows * 8
            return sum(self.batch * (block.tokens // 64) * 4
                       + self.rows * 4 + self.native_schedule_bytes
                       for block in plan.blocks)

        def temporary(block):
            specs = specifications(block)
            return (
                staged_workspace_bytes(specs)
                if staged_sfa
                else sum(_bytes(*v) for v in specs.values())
            ) + scratch(block)

        def kv_workspace_bytes(shapes):
            # BF16 scores and fallback selector keys live in the workspace.
            # Native FP8 scores and selector scratch are charged by scratch().
            return sum(
                _bytes(*value) for key, value in shapes.items()
                if key.startswith("cache.") or key in ("score", "block.keys")
            )

        def temporary_kv(block):
            specs = specifications(block)
            return kv_workspace_bytes(specs[0] if staged_sfa else specs) + scratch(block)

        budget = owner.budgets[name] if kv_budget is None else kv_budget

        def plan_within_budget():
            planning_budget = budget
            page_limit = frozen["pages"] if frozen is not None else None
            while True:
                plan = plan_dsa_kv_chunks(
                    lengths,
                    page_tokens=params.page_size,
                    budget_bytes=planning_budget,
                    fixed_bytes=external if kv_budget is None else 0,
                    temporary_bytes=temporary if kv_budget is None else temporary_kv,
                    max_page_capacity=page_limit,
                )
                stages = [{}, {}] if staged_sfa else [{}]
                for block in plan.blocks:
                    specs = specifications(block)
                    for shapes, spec in zip(stages, specs if staged_sfa else (specs,)):
                        for n, (shape, dtype) in spec.items():
                            if n not in shapes or _bytes(shape, dtype) > _bytes(*shapes[n]):
                                shapes[n] = (shape, dtype)
                shapes = {n: value for stage in stages for n, value in stage.items()}
                required = external + (
                    self.workspace.required_staged_bytes(stages)
                    if staged_sfa
                    else self.workspace.required_bytes(shapes)
                )
                scratch_bytes = max((scratch(b) for b in plan.blocks), default=0)
                required += scratch_bytes + metadata_bytes(plan)
                # Different ragged blocks can maximize different named buffers.
                # Account for the union of capacities, not only the largest block.
                if kv_budget is not None:
                    if kv_workspace_bytes(shapes) + scratch_bytes <= kv_budget:
                        return plan, stages, shapes, required
                    page_limit = plan.page_capacity - 1
                    if page_limit <= 0:
                        raise ValueError("Prefill KV workspace budget cannot fit the ragged arena")
                else:
                    if required <= budget:
                        return plan, stages, shapes, required
                    planning_budget -= required - budget

        # MegaTopk needs the whole local history's logits at once. When the budget left beside the
        # resident Graph arenas cannot hold them (eager steps), select with the chunked DCP path.
        for mega_topk in (True, False) if mega_topk else (False,):
            try:
                plan, stages, shapes, required = plan_within_budget()
                break
            except ValueError:
                if not mega_topk:
                    raise
        self.mega_topk = mega_topk
        # Keep the original chunk geometry and graph phase budget.
        planning_reserve = 0
        if self.fp8_indexer and not self.prefill and "score" in shapes:
            planning_reserve += _bytes(*shapes.pop("score"))
        if self.direct_native_page_size is not None and "cache.native64" in shapes:
            planning_reserve += _bytes(*shapes.pop("cache.native64"))
        self.plan, self.shapes, self.required_bytes = plan, shapes, required
        self.stages = stages if staged_sfa else None
        if frozen is not None and plan.page_capacity != frozen["pages"]:
            raise ValueError("DSA Graph KV chunk capacity cannot change after capture")
        if self.graph:
            forbid = params.forbid_realloc or frozen is not None
            if self.stages is None:
                self.workspace.reserve(shapes, forbid_realloc=forbid)
            else:
                self.workspace.reserve_stages(self.stages, forbid_realloc=forbid)
            if frozen is None:
                self.workspace.graph_capacity = {
                    "batch": capacity_batch, "geometry": geometry,
                    "pages": plan.page_capacity,
                }
            self.workspace.metadata_budget_bytes = max(
                self.workspace.metadata_budget_bytes, metadata_bytes(plan) + planning_reserve
            )
        self.block_ids = None
        if not self.prefill:
            self.block_ids = params.block_table.to(torch.int64).contiguous()
        device = owner.device
        for block in plan.blocks:
            if self.prefill:
                descriptor = params.cache_adapter.build_prefix_chunk_descriptor(
                    block.requests, block.starts, block.lengths, feature_width=features
                )
                from rtp_llm.models_py.triton_kernels.mla.page_rr import _metadata

                _metadata(descriptor, device)
                positions, requests, slices, host_slices = _prefill_block_layout(
                    block, len(lengths), device
                )
                self.blocks.append(
                    DsaPreparedKVBlock(
                        descriptor,
                        0,
                        block.tokens,
                        positions,
                        requests,
                        slices,
                        host_slices=host_slices,
                        fp8_dequant_metadata=(
                            (
                                torch.zeros((1, 1), device=device, dtype=torch.int32),
                                torch.full((1,), block.tokens, device=device, dtype=torch.int32),
                                torch.zeros((1,), device=device, dtype=torch.int32),
                            ) if self.fp8_sfa else None
                        ),
                    )
                )
            else:
                start, width = block.starts[0], block.lengths[0]
                geometry_key = (start, width, geometry)
                cached = self.workspace.graph_geometry.get(geometry_key) if self.graph else None
                if cached is None:
                    local = torch.arange(
                        start, start + width, dtype=torch.int32, device=device
                    )
                    page = params.page_size
                    positions = (
                        (local // page) * self.shards + params.shard_rank
                    ) * page + local % page
                    owners = torch.arange(self.shards, dtype=torch.int32, device=device)[:, None]
                    gathered_positions = (
                        ((local // page)[None, :] * self.shards + owners) * page
                        + local[None, :] % page
                    ).reshape(-1)
                    if self.graph:
                        destination = self.workspace("positions." + str(start), (width,), torch.int32)
                        destination.copy_(positions)
                        positions = destination
                        destination = self.workspace("dcp_positions." + str(start),
                                                     (self.shards * width,), torch.int32)
                        destination.copy_(gathered_positions)
                        gathered_positions = destination
                        self.workspace.graph_geometry[geometry_key] = (positions, gathered_positions)
                else:
                    positions, gathered_positions = cached
                prepared = DsaPreparedKVBlock(
                    None, start, width, positions, dcp_positions=gathered_positions
                )
                if (self.fp8_indexer and owner.native_exact_topk and not self.mega_topk
                        and not native_page_rr_selection_supported(
                            width, owner.topk, shards=self.shards, page=params.page_size, start=start,
                        )):
                    if start % params.page_size or width % params.page_size:
                        raise ValueError("Bounded Indexer Decode chunks must contain whole owner pages")
                    key = ("decode.positions", start, width, geometry)
                    compact_positions = self.workspace.graph_geometry.get(key) if self.graph else None
                    if compact_positions is None:
                        compact_positions = torch.arange(
                            start * self.shards, (start + width) * self.shards,
                            dtype=torch.int32, device=device,
                        )
                        if self.graph:
                            destination = self.workspace("decode.positions." + str(start),
                                                         (self.shards * width,), torch.int32)
                            destination.copy_(compact_positions)
                            compact_positions = destination
                            self.workspace.graph_geometry[key] = compact_positions
                    prepared.indexer_decode_positions = compact_positions
                self.blocks.append(prepared)
        self._planning_key = self.planning_key(params, name)
        self.metadata_ready = None
        if self.prefill or self.fp8_indexer:
            stream = torch.cuda.current_stream(owner.device)
            self.metadata_stream = stream.cuda_stream
            # Decode metadata is produced outside capture and consumed by a
            # captured stream: represent this as an external Graph wait node.
            self.metadata_ready = torch.cuda.Event(external=self.graph)
            self.metadata_ready.record(stream)
        self.metadata_waited = False

    def update_indexer_metadata(self, params):
        """Refresh bounded native metadata during prepare, before Graph replay."""
        if not self.fp8_indexer:
            return
        for block in self.blocks:
            if self.prefill:
                info = block.slices.index_select(0, params.query_requests.long())
                start, end, offset = info.unbind(1)
                stop = offset + (torch.minimum(params.query_positions + 1, end) - start).clamp_min(0)
                valid = params.query_positions >= 0
                block.indexer_starts = torch.where(valid, offset, 0).int().contiguous()
                block.indexer_ends = torch.where(valid, stop, 0).int().contiguous()
            else:
                context = (params.decode.local_causal_lens - block.start).clamp(0, block.width)
                context = torch.where(params.positions_d.view(self.batch, self.queries) >= 0,
                                      context, 0).int().contiguous()
                if block.native_context is None:
                    if params.forbid_realloc:
                        raise ValueError("FP8 Indexer native metadata must exist before Graph replay")
                    block.native_context = context
                    if self.direct_native_page_size is None:
                        block.native_table = torch.arange(
                            1, 1 + self.batch * (block.width // 64), device=self.owner.device,
                            dtype=torch.int32,
                        ).view(self.batch, -1)
                    else:
                        block.native_table = torch.empty(
                            (self.batch, block.width // self.direct_native_page_size),
                            device=self.owner.device, dtype=torch.int32,
                        )
                else:
                    block.native_context.copy_(context)
                if self.direct_native_page_size is not None:
                    page = self.direct_native_page_size
                    if block.start % page or block.width % page:
                        raise ValueError("Direct Indexer chunks require whole physical pages")
                    first = block.start // page
                    block.native_table.copy_(params.block_table[:, first:first + block.width // page])
                schedule = self.owner.indexer_score.prepare_paged_metadata(
                    block.native_context, num_sms=self.num_sms,
                    page_size=self.direct_native_page_size or 64,
                )
                if block.native_schedule is None:
                    block.native_schedule = schedule
                else:
                    if schedule.shape != block.native_schedule.shape or schedule.dtype != block.native_schedule.dtype:
                        raise ValueError("FP8 MQA scheduling capacity changed after prepare")
                    block.native_schedule.copy_(schedule)
        if self.metadata_ready is not None:
            stream = torch.cuda.current_stream(self.owner.device)
            self.metadata_stream = stream.cuda_stream
            self.metadata_ready.record(stream)
        self.metadata_waited = False

    def begin(self):
        if self.metadata_ready is not None and not self.metadata_waited:
            stream = torch.cuda.current_stream(self.owner.device)
            if stream.cuda_stream != self.metadata_stream and not (
                self.graph and torch.cuda.is_current_stream_capturing()
            ):
                stream.wait_event(self.metadata_ready)
                for block in self.blocks:
                    for tensor in (
                        block.key_positions,
                        block.key_requests,
                        block.slices,
                        block.indexer_starts,
                        block.indexer_ends,
                        block.indexer_decode_positions,
                        block.native_context,
                        block.native_table,
                        block.native_schedule,
                    ):
                        if isinstance(tensor, torch.Tensor):
                            tensor.record_stream(stream)
            self.metadata_waited = True
        if not self.graph:
            if self.stages is None:
                self.workspace.reserve(self.shapes)
            else:
                self.workspace.reserve_stages(self.stages)
        return self.workspace

    def finish(self):
        if not self.graph:
            # Returned TopK positions/output retain their own tensor lease.
            # All other phase buffers can be reused by SFA or following MoE/KDA.
            self.workspace.release()

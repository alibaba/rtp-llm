"""CED stage execution over local pages or framework-owned CP8 cache shards."""

from dataclasses import dataclass, fields, replace

import torch
from rtp_llm.models_py.modules.dsv41.attention import (
    V41AttentionCache,
    V41AttentionContext,
)
from rtp_llm.models_py.modules.dsv41.cache_layout import (
    SWA_WINDOW,
    DRAFT_LAYERS,
    CacheRegion,
    MemoryCheckpoint,
    RegionSlot,
)
from rtp_llm.models_py.modules.dsv41.ced import (
    AuxRowMap,
    LateCompletion,
    PrefillPlan,
    PrefillExtend,
    PrefillProgress,
    ReplayConfig,
    ReplayMode,
    RowRange,
    build_prefill_plan,
)
from rtp_llm.models_py.modules.dsv41.compressor import PairCarry
from rtp_llm.models_py.modules.dsv41.indexer import IndexSelection
from rtp_llm.models_py.modules.dsv41.inputs import V41CanonicalInputs, V41ModelRows
from rtp_llm.models_py.modules.dsv41.transformer import V41L20Output, V41TargetOutput


def _copy_rows(rows, selection):
    return V41ModelRows(
        *(getattr(rows, f.name)[selection].clone() for f in fields(V41ModelRows))
    )


def _tensor_range(tensor):
    first = tensor.data_ptr()
    return first, first + tensor.numel() * tensor.element_size()


def _snapshot_page_map(pages, ids, spec, count, device):
    pages.validate(device)
    if (
        pages.region != spec.slot.region
        or pages.entries_per_page != spec.entries
        or pages.data.shape[1] != spec.page_stride_bytes
    ):
        raise ValueError("snapshot page pool differs from the declared layout")
    if (
        ids.ndim != 1
        or ids.numel() != count
        or ids.dtype not in (torch.int32, torch.int64)
        or ids.device != device
        or not ids.is_contiguous()
    ):
        raise ValueError("snapshot is missing its complete page map")
    physical = ids.tolist()
    if len(set(physical)) != count or any(
        page <= 0 or page >= pages.data.shape[0] for page in physical
    ):
        raise ValueError("snapshot page map contains missing or aliased pages")
    base, stride = pages.data.data_ptr(), pages.data.stride(0)
    ranges = [
        (base + page * stride, base + page * stride + spec.page_stride_bytes)
        for page in physical
    ]
    return ids.long(), ranges


def _snapshot_disjoint(writable, retained=()):
    # Compare actual page intervals, so disjoint views of one allocation remain
    # valid while shifted views cannot hide an overlapping physical page.
    latest_write, latest_read = 0, 0
    ranges = [(first, last, True) for first, last in writable if first < last]
    ranges += [(first, last, False) for first, last in retained if first < last]
    for first, last, write in sorted(ranges):
        if first < latest_write or (write and first < latest_read):
            raise ValueError("snapshot copy aliases a retained payload or cache region")
        if write:
            latest_write = max(latest_write, last)
        else:
            latest_read = max(latest_read, last)


@dataclass(frozen=True)
class V41L20Tail:
    request_id: str
    cache_fingerprint: str
    epoch: int
    positions: RowRange
    l20: V41L20Output
    selection: IndexSelection

    @classmethod
    @torch.inference_mode()
    def append(cls, previous, l20, context: V41AttentionContext):
        context.validate()
        if (
            context.cache.identity.replay_fingerprint
            != ReplayConfig(ReplayMode.BOUNDED).fingerprint
        ):
            raise ValueError("retained L20 tails require bounded policy identity")
        count = context.end - context.start
        if count <= 0 or l20.rows.token_ids.numel() != count:
            raise ValueError("L20 rows must match the complete nonempty encoder range")
        if not set(range(21)).issubset(
            context.completed_layers
        ) or context.published_sources != {2, 8, 14, 20}:
            raise ValueError(
                "L20 tail requires every encoder layer and all four sources"
            )
        selected = context.selection_for(20)
        if (
            selected is None
            or selected.topk.shape != (count, 512)
            or selected.candidate_blocks is None
            or selected.candidate_blocks.shape != (count, 2048)
            or selected.status.shape != (count,)
        ):
            raise ValueError(
                "L20 tail requires its query top-k, candidates and valid status"
            )
        selected.check()
        torch._assert_async(
            l20.rows.valid.all(), "local L20 tail cannot retain padding"
        )
        start = max(0, context.end - SWA_WINDOW)
        if previous is not None:
            if (
                previous.request_id != context.cache.request_id
                or previous.cache_fingerprint != context.cache.identity.fingerprint
                or previous.epoch >= context.epoch
                or previous.positions.end != context.start
            ):
                raise ValueError(
                    "retained L20 tail has stale request, epoch or position identity"
                )
            start = max(start, previous.positions.start)
        else:
            start = max(start, context.start)

        keep = context.end - start
        new_keep = min(count, keep)
        old_keep = keep - new_keep

        def join(old, new):
            # Slice before allocating: the encoder chunk can be much larger
            # than the tail. A slice alone would retain its entire backing.
            new = new[-new_keep:]
            if old_keep == 0:
                return new.clone()
            return torch.cat((old[-old_keep:], new), dim=0)

        rows = V41ModelRows(
            *(
                join(
                    None if previous is None else getattr(previous.l20.rows, f.name),
                    getattr(l20.rows, f.name),
                )
                for f in fields(V41ModelRows)
            )
        )
        return cls(
            context.cache.request_id,
            context.cache.identity.fingerprint,
            context.epoch,
            RowRange(start, context.end),
            V41L20Output(
                rows,
                join(
                    None if previous is None else previous.l20.hidden_states,
                    l20.hidden_states,
                ),
                join(None if previous is None else previous.l20.pre_mix, l20.pre_mix),
            ),
            IndexSelection(
                join(
                    None if previous is None else previous.selection.topk, selected.topk
                ),
                join(
                    None if previous is None else previous.selection.candidate_blocks,
                    selected.candidate_blocks,
                ),
                join(
                    None if previous is None else previous.selection.status,
                    selected.status,
                ),
                20,
                20,
                0,
                0,
                0,
            ),
        )

    @property
    def storage_bytes(self):
        tensors = [getattr(self.l20.rows, f.name) for f in fields(V41ModelRows)]
        tensors += [
            self.l20.hidden_states,
            self.l20.pre_mix,
            self.selection.topk,
            self.selection.candidate_blocks,
            self.selection.status,
        ]
        return sum(t.untyped_storage().nbytes() for t in tensors)

    @torch.inference_mode()
    def select(self, cache, rows: RowRange, replay_floor):
        if (
            self.request_id != cache.request_id
            or self.cache_fingerprint != cache.identity.fingerprint
            or self.epoch != cache.active_epoch
            or cache.poisoned
        ):
            raise ValueError("cannot consume a stale retained L20 tail")
        if not self.positions.start <= rows.start < rows.end == self.positions.end:
            raise ValueError("required decoder rows are no longer retained at L20")
        if not 0 <= replay_floor <= rows.start:
            raise ValueError("late SWA floor must precede its first query")
        if any(owner.materialized_end < rows.end for owner in cache.owners.values()):
            raise ValueError("retained tail has incomplete global/index sources")
        offset = rows.start - self.positions.start
        context = V41AttentionContext(
            cache, self.epoch, rows.start, rows.end, replay_floor
        )
        context.published_sources = {2, 8, 14, 20}
        context.publish_selection(
            IndexSelection(
                self.selection.topk[offset:].clone(),
                self.selection.candidate_blocks[offset:].clone(),
                self.selection.status[offset:].clone(),
                20,
                20,
                0,
                0,
                0,
            )
        )
        return (
            V41L20Output(
                _copy_rows(self.l20.rows, slice(offset, None)),
                self.l20.hidden_states[offset:].clone(),
                self.l20.pre_mix[offset:].clone(),
            ),
            context,
        )


@dataclass(frozen=True)
class V41LocalSnapshot:
    checkpoint: MemoryCheckpoint
    swa: dict
    owners: dict
    history_rows: V41ModelRows
    canonical_prefix_sha256: str | None = None

    @classmethod
    @torch.inference_mode()
    def protect(cls, cache, *, end, replay_floor, history_rows):
        """Finish a real joint in-memory copy before any suffix can wrap SWA."""
        if (
            cache.poisoned
            or type(end) is not int
            or not 0 < end <= min(cache.max_tokens, 1048576)
            or end % cache.layout.reuse_unit
            or cache.identity.layout_fingerprint != cache.layout.fingerprint
        ):
            raise ValueError("local snapshot requires a healthy aligned checkpoint")
        expected_layers = set(range(43 if cache.layout.draft_enabled else 40))
        if set(cache.swa) != expected_layers or set(cache.owners) != {2, 8, 14, 20}:
            raise ValueError(
                "local snapshot is missing a required physical cache region"
            )
        if set(cache.swa_ends) != set(cache.swa) or any(
            value != end for value in cache.swa_ends.values()
        ):
            raise ValueError("local checkpoint is missing complete target/draft SWA")
        if any(owner.materialized_end != end for owner in cache.owners.values()):
            raise ValueError("local checkpoint is missing complete global/index KV")
        history_rows.validate()
        device = next(iter(cache.swa.values())).pages.data.device
        if (
            history_rows.token_ids.device != device
            or not min(3, end) <= history_rows.token_ids.numel() <= SWA_WINDOW
            or not bool(history_rows.valid.all())
        ):
            raise ValueError("checkpoint must retain actual canonical tail history")
        specs = {page.slot: page for page in cache.layout.pages}
        source_swa, source_owners, starts = {}, {}, {}
        payload_ranges, metadata_ranges = [], []
        for layer, binding in cache.swa.items():
            if binding.validate(device) != 1:
                raise ValueError("local snapshot requires one complete request ring")
            ids, ranges = _snapshot_page_map(
                binding.pages,
                binding.page_ids,
                specs[RegionSlot(CacheRegion.SWA, layer)],
                1,
                device,
            )
            payload_ranges.extend(ranges)
            metadata_ranges.extend(
                _tensor_range(tensor)
                for tensor in (
                    binding.page_ids,
                    binding.valid_starts,
                    binding.valid_ends,
                )
            )
            first, last = int(binding.valid_starts[0]), int(binding.valid_ends[0])
            if (
                not max(0, end - binding.pages.entries_per_page)
                <= first
                <= max(0, end - SWA_WINDOW)
                or last != end
            ):
                raise ValueError(
                    "checkpoint SWA valid range does not cover its required window"
                )
            source_swa[layer] = (binding.pages.data, ids, first, last)
            starts[layer] = first
        for layer, owner in cache.owners.items():
            if owner.pair is not None:
                owner.pair.validate(
                    layer,
                    cache.request_id,
                    cache.identity,
                    end,
                    device,
                )
            page_count = end // cache.layout.token_block_size
            owner.global_kv.validate(1, device)
            sources = []
            for pages, table, region in (
                (owner.global_kv.pages, owner.global_kv.page_table, CacheRegion.GLOBAL),
                (owner.index_pages, owner.index_table, CacheRegion.INDEX_K),
            ):
                spec = specs[RegionSlot(region, layer)]
                if table.ndim != 2 or table.shape[0] != 1:
                    raise ValueError("local snapshot requires one request page table")
                if owner.global_kv.compress_ratio != spec.ratio:
                    raise ValueError(
                        "snapshot owner compression ratio differs from layout"
                    )
                ids = table[0, :page_count]
                selected, ranges = _snapshot_page_map(
                    pages, ids, spec, page_count, device
                )
                payload_ranges.extend(ranges)
                metadata_ranges.append(_tensor_range(table))
                sources.append((pages.data, selected))
            source_owners[layer] = sources
        _snapshot_disjoint(payload_ranges, metadata_ranges)
        saved_swa = {
            layer: (data.index_select(0, ids).clone(), first, last)
            for layer, (data, ids, first, last) in source_swa.items()
        }
        saved_owners = {
            layer: tuple(data.index_select(0, ids).clone() for data, ids in sources)
            for layer, sources in source_owners.items()
        }
        history = _copy_rows(history_rows, slice(-3, None))
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(history.token_ids.device))
        event.synchronize()
        checkpoint = MemoryCheckpoint(
            end,
            cache.identity,
            cache.layout.required_slots,
            max(starts[layer] for layer in range(40)),
            (
                max(starts[layer] for layer in (40, 41, 42))
                if cache.layout.draft_enabled
                else None
            ),
            replay_floor,
            history_ready=True,
            copy_complete=True,
            backing_protected=True,
        )
        if not checkpoint.is_complete(cache.layout, cache.identity):
            raise ValueError(
                "copied local checkpoint does not satisfy the complete state contract"
            )
        return cls(checkpoint, saved_swa, saved_owners, history)

    @torch.inference_mode()
    def restore(self, cache):
        """Restore into fresh page IDs without retaining old physical mappings."""
        c = self.checkpoint
        expected_layers = {
            slot.owner_layer
            for slot in cache.layout.required_slots
            if slot.region == CacheRegion.SWA
        }
        if (
            set(self.swa) != expected_layers
            or set(cache.swa) != expected_layers
            or set(self.owners) != {2, 8, 14, 20}
            or set(cache.owners) != set(self.owners)
        ):
            raise ValueError(
                "joint snapshot regions do not match the destination cache"
            )
        if (
            not c.is_complete(cache.layout, cache.identity)
            or c.materialized_end > cache.max_tokens
        ):
            raise ValueError(
                "snapshot cannot restore into a different policy or capacity"
            )
        if (
            cache.poisoned
            or cache.swa_ends
            or any(owner.materialized_end for owner in cache.owners.values())
        ):
            raise ValueError("local restore requires a fresh destination cache")
        if (
            self.history_rows.token_ids.device
            != next(iter(cache.swa.values())).pages.data.device
        ):
            raise ValueError("local snapshot restore cannot cross devices")
        self.history_rows.validate()
        if self.history_rows.token_ids.numel() != min(
            3, c.materialized_end
        ) or not bool(self.history_rows.valid.all()):
            raise ValueError("snapshot is missing its canonical tail history")
        device = self.history_rows.token_ids.device
        specs = {page.slot: page for page in cache.layout.pages}
        copies, starts = [], {}
        writable_ranges, retained_ranges = [], [
            _tensor_range(getattr(self.history_rows, field.name))
            for field in fields(V41ModelRows)
        ]

        def prepare_copy(data, pages, ids, slot, count):
            spec = specs[slot]
            if (
                tuple(data.shape) != (count, spec.page_stride_bytes)
                or data.dtype != torch.uint8
                or data.device != device
                or not data.is_contiguous()
            ):
                raise ValueError(
                    "snapshot payload does not contain every declared page"
                )
            selected, ranges = _snapshot_page_map(pages, ids, spec, count, device)
            writable_ranges.extend(ranges)
            retained_ranges.extend((_tensor_range(ids), _tensor_range(data)))
            copies.append((pages.data, selected, data))

        # Validate every payload before writing: a complete descriptor alone cannot
        # prove that all global/index pages and target/draft rings were received.
        for layer, (data, first, end) in self.swa.items():
            binding = cache.swa[layer]
            if binding.validate(device) != 1:
                raise ValueError("local snapshot requires one complete request ring")
            if end != c.materialized_end or not max(
                0, end - cache.layout.swa_entries
            ) <= first <= max(0, end - SWA_WINDOW):
                raise ValueError("snapshot SWA payload does not cover its checkpoint")
            starts[layer] = first
            writable_ranges.extend(
                (_tensor_range(binding.valid_starts), _tensor_range(binding.valid_ends))
            )
            prepare_copy(
                data,
                binding.pages,
                binding.page_ids,
                RegionSlot(CacheRegion.SWA, layer),
                1,
            )
        # Encoder rings can retain rows preceding the decoder replay floor.
        # protect() records the common valid start across each complete layer set.
        if max(starts[layer] for layer in range(40)) != c.target_swa_start or (
            cache.layout.draft_enabled
            and max(starts[layer] for layer in (40, 41, 42)) != c.draft_swa_start
        ):
            raise ValueError("snapshot SWA ranges disagree with checkpoint metadata")
        page_count = c.materialized_end // cache.layout.token_block_size
        for layer, (global_data, index_data) in self.owners.items():
            owner = cache.owners[layer]
            owner.global_kv.validate(1, device)
            for data, pages, table, region in (
                (
                    global_data,
                    owner.global_kv.pages,
                    owner.global_kv.page_table,
                    CacheRegion.GLOBAL,
                ),
                (index_data, owner.index_pages, owner.index_table, CacheRegion.INDEX_K),
            ):
                slot = RegionSlot(region, layer)
                if table.ndim != 2 or table.shape[0] != 1:
                    raise ValueError("local snapshot requires one request page table")
                if owner.global_kv.compress_ratio != specs[slot].ratio:
                    raise ValueError(
                        "snapshot owner compression ratio differs from layout"
                    )
                retained_ranges.append(_tensor_range(table))
                prepare_copy(data, pages, table[0, :page_count], slot, page_count)
        _snapshot_disjoint(writable_ranges, retained_ranges)
        try:
            for destination, ids, data in copies:
                destination.index_copy_(0, ids, data)
            for layer, (_, first, end) in self.swa.items():
                cache.swa[layer].valid_starts.fill_(first)
                cache.swa[layer].valid_ends.fill_(end)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device))
            event.synchronize()
            for layer, owner in cache.owners.items():
                if owner.global_kv.compress_ratio == 2:
                    owner.pair = PairCarry.empty(
                        layer, cache.request_id, cache.identity, c.materialized_end
                    )
                owner.materialized_end = c.materialized_end
            cache.swa_ends.update({layer: c.materialized_end for layer in cache.swa})
        except Exception:
            cache.poisoned = True
            raise


@dataclass(frozen=True)
class V41PrefillResult:
    output: V41TargetOutput | None
    aux_rows: AuxRowMap | None
    context: V41AttentionContext
    checkpoint: V41LocalSnapshot | None


class V41PrefillExecutor:
    @classmethod
    def from_prepared(
        cls,
        target,
        cache,
        prepared,
        *,
        config,
        chunk_tokens,
        draft_commit,
        restored=None,
        **plan_options,
    ):
        """Connect validated request IDs/images to local CED execution.

        Image encoding runs once per image. Global LM boundaries still come
        from the image-aware plan; selecting a retained L20 tail never reruns ViT.
        CP assembly and engine checkpoint transport remain separate integrations.
        """
        canonical = V41CanonicalInputs(prepared)
        if restored is not None and restored.canonical_prefix_sha256 != (
            canonical.prefix_fingerprint(restored.checkpoint.materialized_end)
        ):
            raise ValueError("restored checkpoint has a different canonical prefix")
        plan = build_prefill_plan(
            total_tokens=len(prepared.token_ids),
            chunk_tokens=chunk_tokens,
            layout=cache.layout,
            model_revision=cache.identity.model_revision,
            config=config,
            restored_checkpoint=None if restored is None else restored.checkpoint,
            images=tuple(
                RowRange(image.start, image.start + image.length)
                for image in prepared.images
            ),
            **plan_options,
        )
        features = target.prepare_images(prepared) if prepared.images else None
        result = cls(target, cache, plan, draft_commit=draft_commit, restored=restored)
        result.canonical = canonical
        result.image_features = features
        return result

    def __init__(
        self,
        target,
        cache: V41AttentionCache,
        plan: PrefillPlan,
        *,
        draft_commit,
        restored=None,
    ):
        if plan.identity != cache.identity or not plan.extends:
            raise ValueError("prefill plan and cache require the same actual identity")
        if plan.extends[-1].encoder_rows.end > cache.max_tokens:
            raise ValueError("prefill plan exceeds the local cache capacity")
        start = plan.extends[0].encoder_rows.start
        if start:
            if restored is None or restored.checkpoint.materialized_end != start:
                raise ValueError(
                    "prefix continuation requires its actual joint snapshot"
                )
            restored.restore(cache)
        elif (
            restored is not None
            or cache.swa_ends
            or any(owner.materialized_end for owner in cache.owners.values())
        ):
            raise ValueError("cold prefill requires fresh local state")
        if cache.layout.draft_enabled != (draft_commit is not None):
            raise ValueError("joint prefill requires the three draft commit writers")
        self.target, self.cache, self.plan, self.draft_commit = (
            target,
            cache,
            plan,
            draft_commit,
        )
        self.progress = PrefillProgress(start, start, start)
        self.next_extend = 0
        self.tail = None
        self.history_rows = (
            None
            if restored is None
            else _copy_rows(restored.history_rows, slice(-3, None))
        )
        self.protected = restored
        self.observations = []
        self.canonical = None
        self.image_features = None

    def run_next(self, *, epoch, lookup_outputs=None):
        if self.canonical is None:
            raise ValueError("automatic extends require a validated canonical request")
        if self.next_extend >= len(self.plan.extends):
            raise ValueError("all planned prefill extends already completed")
        selected = self.plan.extends[self.next_extend].encoder_rows
        rows = self.canonical.rows(
            selected.start, selected.end, device=self.target.embedding.device
        )
        features = (
            None
            if self.image_features is None
            else self.image_features.for_extend(selected.start, selected.end)
        )
        return self.run_extend(
            rows, epoch=epoch, image_features=features, lookup_outputs=lookup_outputs
        )

    def _commit_draft(self, output, aux_map, context):
        first = max(context.start, context.end - SWA_WINDOW)
        positions = tuple(range(first, context.end))
        bindings = {layer: self.cache.swa[layer] for layer in (40, 41, 42)}
        starts = {}
        for layer, binding in bindings.items():
            previous = self.cache.swa_ends.get(layer, 0)
            if previous == first:
                starts[layer] = int(binding.valid_starts[0])
            elif len(positions) == SWA_WINDOW or first == 0:
                starts[layer] = first
            else:
                raise ValueError(
                    "short draft continuation requires its complete SWA history"
                )
        result = self.draft_commit.commit(
            output.aux_hidden_states,
            aux_map,
            required_positions=positions,
            swa_bindings=bindings,
            request_id=aux_map.request_id,
            forward_epoch=aux_map.forward_epoch,
            replay_fingerprint=aux_map.replay_fingerprint,
            replay_floor=context.replay_floor,
        )
        if not result.write_completed or result.positions != positions:
            raise ValueError(
                "draft writer did not complete every required valid aux row"
            )
        for layer, binding in bindings.items():
            binding.valid_starts.fill_(
                max(
                    starts[layer],
                    context.replay_floor,
                    context.end - binding.pages.entries_per_page,
                )
            )
            binding.valid_ends.fill_(context.end)
            self.cache.swa_ends[layer] = context.end
        return {
            "main_projection_rows": result.main_projection_rows,
            "stage_projection_rows": result.stage_projection_rows,
        }

    @torch.inference_mode()
    def run_extend(self, rows, *, epoch, image_features=None, lookup_outputs=None):
        if self.next_extend >= len(self.plan.extends):
            raise ValueError("all planned prefill extends already completed")
        extend = self.plan.extends[self.next_extend]
        progress = self.progress.encoder_completed(extend)
        if rows.token_ids.numel() != len(extend.encoder_rows):
            raise ValueError(
                "local prefill input must contain every planned encoder row"
            )
        rows.validate()
        if self.canonical is not None:
            expected_rows = self.canonical.rows(
                extend.encoder_rows.start,
                extend.encoder_rows.end,
                device=rows.token_ids.device,
            )
            if any(
                not torch.equal(getattr(rows, f.name), getattr(expected_rows, f.name))
                for f in fields(V41ModelRows)
            ):
                raise ValueError("prefill rows disagree with the canonical request")
            expected_features = (
                None
                if self.image_features is None
                else self.image_features.for_extend(
                    extend.encoder_rows.start, extend.encoder_rows.end
                )
            )
            if (image_features is None) != (expected_features is None) or (
                expected_features is not None
                and any(
                    not torch.equal(
                        getattr(image_features, name), getattr(expected_features, name)
                    )
                    for name in ("row_indices", "token_types", "values")
                )
            ):
                raise ValueError("prefill features disagree with the canonical images")
        torch._assert_async(
            rows.valid.all(), "local prefill cannot substitute padded rows"
        )
        context = self.cache.begin_forward(
            epoch=epoch, start=extend.encoder_rows.start, end=extend.encoder_rows.end
        )
        output = aux_map = checkpoint = None
        draft_rows = None
        try:
            history = _copy_rows(rows, slice(-3, None))
            if self.history_rows is not None and history.token_ids.numel() < 3:
                history = V41ModelRows(
                    *(
                        torch.cat(
                            (
                                getattr(self.history_rows, f.name),
                                getattr(history, f.name),
                            ),
                            dim=0,
                        )[-3:].clone()
                        for f in fields(V41ModelRows)
                    )
                )
            l20 = self.target.prefill_encoder(
                rows,
                context,
                image_features=image_features,
                lookup_outputs=lookup_outputs,
            )
            if self.plan.config.mode == ReplayMode.BOUNDED:
                self.tail = V41L20Tail.append(self.tail, l20, context)
            encoder_observations = list(context.observations)
            if extend.decoder_rows is not None:
                if self.plan.config.mode == ReplayMode.BOUNDED:
                    l20, context = self.tail.select(
                        self.cache, extend.decoder_rows, extend.replay_floor
                    )
                output = self.target.prefill_decoder(l20, context)
                positions = tuple(range(context.start, context.end))
                aux_map = AuxRowMap(
                    self.cache.request_id,
                    epoch,
                    self.plan.identity.replay_fingerprint,
                    positions,
                    positions,
                    tuple(l20.rows.image_mask.cpu().tolist()),
                )
                if self.draft_commit is not None:
                    draft_rows = self._commit_draft(output, aux_map, context)
                complete = all(
                    self.cache.swa_ends.get(layer) == context.end
                    for layer in self.cache.swa
                )
                if (
                    not set(range(21, 40)).issubset(context.completed_layers)
                    or not complete
                ):
                    raise ValueError(
                        "late prefill did not materialize all target/draft SWA"
                    )
                progress = progress.decoder_completed(
                    extend, LateCompletion(extend.decoder_rows, True, True)
                )
                if extend.checkpoint_end is not None:
                    checkpoint = V41LocalSnapshot.protect(
                        self.cache,
                        end=extend.checkpoint_end,
                        replay_floor=extend.replay_floor,
                        history_rows=history,
                    )
                    if self.canonical is not None:
                        checkpoint = replace(
                            checkpoint,
                            canonical_prefix_sha256=self.canonical.prefix_fingerprint(
                                extend.checkpoint_end
                            ),
                        )
                    progress = progress.checkpoint_protected(
                        checkpoint.checkpoint, self.cache.layout, self.cache.identity
                    )
                    self.protected = checkpoint
            if extend.final_handoff:
                progress.require_handoff(
                    extend.encoder_rows.end, self.plan.protected_checkpoint_end
                )
            self.progress = progress
            self.history_rows = history
            self.next_extend += 1
            self.observations.append(
                {
                    "epoch": epoch,
                    "mode": self.plan.config.mode.value,
                    "fallback_reason": self.plan.fallback_reason,
                    "encoder_range": (
                        extend.encoder_rows.start,
                        extend.encoder_rows.end,
                    ),
                    "decoder_range": (
                        None
                        if extend.decoder_rows is None
                        else (context.start, context.end)
                    ),
                    "encoder_layers": encoder_observations,
                    "decoder_layers": [
                        o for o in context.observations if o["layer"] > 20
                    ],
                    "retained_bytes": (
                        0 if self.tail is None else self.tail.storage_bytes
                    ),
                    "protected_end": progress.protected_checkpoint_end,
                    "draft_rows": draft_rows,
                }
            )
            return V41PrefillResult(output, aux_map, context, checkpoint)
        except Exception:
            self.cache.poisoned = True
            raise


@dataclass(frozen=True)
class V41CPHistory:
    token_ids: tuple[int, int, int]
    image_mask: tuple[bool, bool, bool]

    @classmethod
    def at_boundary(cls, context, rows):
        packed = torch.cat(
            (
                rows.history_ids[:, -2:],
                rows.token_ids[:, None],
                (~rows.history_valid[:, -2:]).to(torch.int32),
                rows.image_mask[:, None].to(torch.int32),
            ),
            dim=1,
        )
        values = (
            context.gather_rows(packed, context.end - 1, context.end)[0].cpu().tolist()
        )
        for index, position in enumerate(range(context.end - 3, context.end)):
            if position < 0:
                values[index], values[index + 3] = -1, 0
        return cls(tuple(values[:3]), tuple(bool(value) for value in values[3:]))


@dataclass(frozen=True)
class V41CPPrefillResult:
    hidden_states: torch.Tensor
    output: V41TargetOutput | None
    aux_rows: AuxRowMap | None
    encoder_context: object
    decoder_context: object | None
    history: V41CPHistory
    progress: PrefillProgress
    checkpoint_protected: bool


class V41CPPrefillExecutor:
    """Request-owned late-stage scheduling using the existing all-worker copy.

    Every CP rank runs the same request/segment order, including empty-valid
    ranks. The adapter binds fresh framework tables for each engine forward and
    supplies native protection/progress callbacks; no KV payload is owned here.
    """

    def __init__(
        self,
        target,
        *,
        request_id,
        identity,
        layout,
        initial_encoder_end=0,
        initial_decoder_end=0,
        initial_protected_end=0,
        history=None,
        draft_commit=None,
        max_tokens_per_rank=None,
    ):
        if (
            not request_id
            or layout.cp_size not in (4, 8)
            or identity.layout_fingerprint != layout.fingerprint
            or identity.replay_fingerprint
            not in (
                ReplayConfig(ReplayMode.FULL).fingerprint,
                ReplayConfig(ReplayMode.BOUNDED).fingerprint,
            )
            or layout.draft_enabled != (draft_commit is not None)
            or (
                max_tokens_per_rank is not None
                and (type(max_tokens_per_rank) is not int or max_tokens_per_rank <= 0)
            )
        ):
            raise ValueError(
                "CP prefill requires actual request/replay identity and draft writers"
            )
        self.target, self.request_id, self.identity, self.layout = (
            target,
            request_id,
            identity,
            layout,
        )
        self.draft_commit = draft_commit
        self.max_tokens_per_rank = max_tokens_per_rank
        self.progress = PrefillProgress(
            initial_encoder_end, initial_decoder_end, initial_protected_end
        )
        self.history = history
        self.tail = None
        self.observations = []
        self.poisoned = False
        self._input_epoch = -1
        self._execution_epoch = -1
        self._boundaries = None

    def _extend(self, start, end, checkpoint, final):
        full = (
            self.identity.replay_fingerprint
            == ReplayConfig(ReplayMode.FULL).fingerprint
        )
        publish, handoff = end == checkpoint and checkpoint > 0, end == final
        rows = floor = None
        history = False
        if full:
            rows, floor, history = RowRange(start, end), 0, start > 0
        elif publish or handoff:
            previous = self.progress.decoder_checkpoint_end
            history = previous > 0 and end - previous <= SWA_WINDOW
            first = previous if history else max(0, end - SWA_WINDOW)
            rows = RowRange(first, end)
            floor = max(0, previous - SWA_WINDOW) if history else first
        return PrefillExtend(
            RowRange(start, end),
            rows,
            floor,
            history,
            end if publish else None,
            handoff,
            RowRange(max(0, end - SWA_WINDOW), end),
            checkpoint if checkpoint and start >= checkpoint else 0,
        )

    def _validate_continuation(self, context, rows):
        if self.history is None or not context.start:
            return
        packed = torch.cat(
            (rows.history_ids, rows.history_valid.to(torch.int32)), dim=1
        )
        values = (
            context.gather_rows(packed, context.start, context.start + 1)[0]
            .cpu()
            .tolist()
        )
        for index, position in enumerate(range(context.start - 3, context.start)):
            if position >= 0 and (
                values[index] != self.history.token_ids[index]
                or bool(values[index + 3]) == self.history.image_mask[index]
            ):
                raise ValueError(
                    "CP continuation history differs from its committed token/image tail"
                )

    def _segments(self, context, rows, checkpoint):
        from rtp_llm.models.multimodal.deepseek_v41_processor import (
            IMAGE_END,
            IMAGE_START,
            TEXT,
        )

        # Only token-type metadata is global; all hidden/image rows retain their
        # original CP owners throughout the internal encoder chunks.
        types = (
            context.gather_rows(rows.token_types[:, None], context.start, context.end)
            .flatten()
            .cpu()
            .tolist()
        )
        images, image_start = [], None
        for offset, token_type in enumerate(types):
            if token_type == IMAGE_START:
                if image_start is not None:
                    raise ValueError(
                        "CP canonical image starts before the previous image ends"
                    )
                image_start = context.start + offset
            elif token_type == IMAGE_END:
                if image_start is None:
                    raise ValueError("CP input starts inside a canonical image span")
                images.append((image_start, context.start + offset + 1))
                image_start = None
            elif (token_type == TEXT) != (image_start is None):
                raise ValueError("CP input contains an incomplete canonical image span")
        if image_start is not None:
            raise ValueError("CP input ends inside a canonical image span")
        if any(first < checkpoint < last for first, last in images):
            raise ValueError(
                "CP protected checkpoint cannot split a canonical image span"
            )
        cuts, start = [context.start], context.start
        while start < context.end:
            end = checkpoint if start < checkpoint < context.end else context.end
            if self.max_tokens_per_rank is not None:
                counts = [0] * context.cp.cp_size
                candidate = start
                while candidate < end:
                    rank = (
                        context._restore_host[candidate - context.start]
                        // context.local_count
                    )
                    if counts[rank] == self.max_tokens_per_rank:
                        break
                    counts[rank] += 1
                    candidate += 1
                end = candidate
            for first, last in images:
                if first < end < last:
                    end = first
                    break
            if end <= start:
                raise ValueError(
                    "CP per-rank token budget cannot fit a complete image span"
                )
            cuts.append(end)
            start = end
        return cuts

    def _validate_late_capacity(self, context, checkpoint, final):
        if (
            self.max_tokens_per_rank is None
            or self.identity.replay_fingerprint
            == ReplayConfig(ReplayMode.FULL).fingerprint
        ):
            return
        previous = self.progress.decoder_checkpoint_end
        for end in sorted({checkpoint, final}):
            if not context.start < end <= context.end:
                continue
            first = (
                previous
                if previous > 0 and end - previous <= SWA_WINDOW
                else max(0, end - SWA_WINDOW)
            )
            counts = [0] * context.cp.cp_size
            if first < context.start:
                if self.tail is None or self.tail.positions.start > first:
                    raise ValueError("CP bounded decoder is missing retained L20 rows")
                for rank, positions in enumerate(self.tail.rank_positions):
                    counts[rank] = sum(
                        first <= pos < context.start for pos in positions
                    )
            for position in range(max(first, context.start), end):
                rank = (
                    context._restore_host[position - context.start]
                    // context.local_count
                )
                counts[rank] += 1
            required = max(counts)
            if required > self.max_tokens_per_rank:
                raise ValueError(
                    f"CP bounded decoder requires {required} rows per rank; "
                    f"admitted token budget is {self.max_tokens_per_rank}"
                )
            previous = end

    @torch.inference_mode()
    def run_extend(
        self,
        rows,
        context,
        *,
        protected_checkpoint_end,
        final_handoff_end,
        protect_checkpoint=None,
        report_progress=None,
        image_features=None,
        lookup_outputs=None,
    ):
        from rtp_llm.models_py.modules.dsv41.cp import V41CPL20Tail

        context.validate()
        rows.validate()
        cache = context.cache
        boundaries = (protected_checkpoint_end, final_handoff_end)
        if (
            self.poisoned
            or cache.request_id != self.request_id
            or cache.identity != self.identity
            or cache.layout != self.layout
            or context.decoder_only
            or context.epoch <= self._input_epoch
            or context.start != self.progress.encoder_materialized_end
            or rows.token_ids.numel() != context.query_rows
            or not torch.equal(rows.valid, context.valid)
            or not 0
            <= protected_checkpoint_end
            <= final_handoff_end
            <= cache.max_tokens
            or protected_checkpoint_end % self.layout.reuse_unit
            or context.end > final_handoff_end
            or (self._boundaries is not None and self._boundaries != boundaries)
            or any(
                cache.swa_ends.get(layer) != self.progress.decoder_checkpoint_end
                for layer in range(21, 43 if self.layout.draft_enabled else 40)
            )
        ):
            raise ValueError(
                "CP extend has stale request, epoch, progress or checkpoint boundaries"
            )
        if (
            protected_checkpoint_end > self.progress.protected_checkpoint_end
            and protect_checkpoint is None
        ):
            raise ValueError(
                "CP checkpoint N requires the engine's completed all-worker copy callback"
            )
        self._validate_continuation(context, rows)
        split = self._segments(context, rows, protected_checkpoint_end)
        self._validate_late_capacity(context, *boundaries)
        self._input_epoch, self._boundaries = context.epoch, boundaries
        hidden = self.target.embedding.new_zeros(
            (context.query_rows, self.target.hidden_size)
        )
        output = aux_map = decoder = None
        protected = False
        bounded = (
            self.identity.replay_fingerprint
            == ReplayConfig(ReplayMode.BOUNDED).fingerprint
        )
        try:
            for start, end in zip(split, split[1:]):
                extend = self._extend(start, end, *boundaries)
                progress = self.progress.encoder_completed(extend)
                self._execution_epoch = (
                    max(self._execution_epoch, cache.active_epoch) + 1
                )
                encoder = context.encoder_slice(start, end, epoch=self._execution_epoch)
                current_rows = encoder.select_model_rows(rows)
                l20 = self.target.prefill_encoder(
                    current_rows,
                    encoder,
                    image_features=encoder.select_image_features(image_features),
                    lookup_outputs=(
                        None
                        if lookup_outputs is None
                        else {
                            layer: encoder.select_rows(values)
                            for layer, values in lookup_outputs.items()
                        }
                    ),
                )
                if not set(range(21)).issubset(
                    encoder.completed_layers
                ) or encoder.published_sources != {2, 8, 14, 20}:
                    raise ValueError(
                        "CP encoder did not finish every L0-L20 layer and source"
                    )
                history = V41CPHistory.at_boundary(encoder, current_rows)
                if bounded:
                    self.tail = V41CPL20Tail.append(self.tail, l20, encoder)
                if report_progress is not None:
                    report_progress(progress)
                decoder, draft_rows = None, None
                if extend.decoder_rows is not None:
                    if bounded:
                        l20, decoder = self.tail.select(
                            encoder, extend.decoder_rows, extend.replay_floor
                        )
                    else:
                        decoder = encoder.for_decoder(extend)
                    output = self.target.prefill_decoder(l20, decoder)
                    indices = output.aux_row_indices
                    if indices is None or not torch.equal(
                        indices, l20.rows.valid.nonzero().flatten()
                    ):
                        raise ValueError(
                            "CP late execution returned a different valid aux row set"
                        )
                    positions = tuple(decoder.positions[indices].cpu().tolist())
                    aux_map = AuxRowMap(
                        self.request_id,
                        decoder.epoch,
                        self.identity.replay_fingerprint,
                        positions,
                        positions,
                        tuple(l20.rows.image_mask[indices].cpu().tolist()),
                    )
                    if self.draft_commit is not None:
                        aux_map, committed = decoder.commit_draft(
                            self.draft_commit, output, l20.rows
                        )
                        draft_rows = {
                            "main_projection_rows": committed.main_projection_rows,
                            "stage_projection_rows": committed.stage_projection_rows,
                        }
                    if not set(range(21, 40)).issubset(decoder.completed_layers) or any(
                        cache.swa_ends.get(layer) != end
                        for layer in range(43 if self.layout.draft_enabled else 40)
                    ):
                        raise ValueError(
                            "CP late execution did not materialize all target/draft SWA"
                        )
                    progress = progress.decoder_completed(
                        extend, LateCompletion(extend.decoder_rows, True, True)
                    )
                    decoder.scatter_rows(output.hidden_states, output=hidden)
                    if report_progress is not None:
                        report_progress(progress)
                    if extend.checkpoint_end is not None:
                        if protect_checkpoint(decoder, history) is not True:
                            raise RuntimeError(
                                "CP checkpoint copy failed; source suffix cannot overwrite N"
                            )
                        checkpoint = MemoryCheckpoint(
                            end,
                            self.identity,
                            self.layout.required_slots,
                            max(
                                int(cache.swa[layer].valid_starts[0])
                                for layer in range(40)
                            ),
                            (
                                max(
                                    int(cache.swa[layer].valid_starts[0])
                                    for layer in DRAFT_LAYERS
                                )
                                if self.layout.draft_enabled
                                else None
                            ),
                            decoder.replay_floor,
                            True,
                            True,
                            True,
                        )
                        progress = progress.checkpoint_protected(
                            checkpoint, self.layout, self.identity
                        )
                        protected = True
                if extend.final_handoff:
                    progress.require_handoff(
                        final_handoff_end, protected_checkpoint_end
                    )
                self.progress, self.history = progress, history
                self.observations.append(
                    {
                        "epoch": encoder.epoch,
                        "encoder_range": (start, end),
                        "encoder_local_rows": encoder.query_rows,
                        "decoder_range": (
                            None if decoder is None else (decoder.start, decoder.end)
                        ),
                        "decoder_local_rows": (
                            0 if decoder is None else decoder.query_rows
                        ),
                        "encoder_layers": tuple(encoder.observations),
                        "decoder_layers": (
                            ()
                            if decoder is None
                            else tuple(
                                row
                                for row in decoder.observations
                                if row.get("layer", 0) > 20
                            )
                        ),
                        "retained_bytes": (
                            0 if self.tail is None else self.tail.storage_bytes
                        ),
                        "draft_rows": draft_rows,
                        "protected_end": progress.protected_checkpoint_end,
                    }
                )
            return V41CPPrefillResult(
                hidden,
                output,
                aux_map,
                encoder,
                decoder,
                self.history,
                self.progress,
                protected,
            )
        except Exception:
            self.poisoned = cache.poisoned = True
            raise

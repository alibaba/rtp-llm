"""Standard fixed PROPOSE5/TAIL6 entrypoints for the V4.1 DSpark engine."""

import torch

from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.dsv41.cache_layout import CacheLayout, CacheRegion
from rtp_llm.models_py.modules.dsv41.compact_reader import CompactPages, SwaBinding
from rtp_llm.models_py.modules.dsv41.decode_attention import _same_pages
from rtp_llm.models_py.modules.dsv41.decode_compressor import _tensor
from rtp_llm.models_py.modules.dsv41.decode_draft import (
    V41DraftAttentionBuffers,
    V41DraftModel,
)
from rtp_llm.models_py.modules.dsv41.decode_fmha_impl import _copy_bytes, _host
from rtp_llm.models_py.modules.dsv41.linear import warmup_block32_linears
from rtp_llm.ops.compute_ops import KVCacheRegionName, PyModelOutputs
from rtp_llm.utils.model_weight import W


def _prefill_role(parallelism):
    return str(parallelism.role_type).upper().rsplit(".", 1)[-1] == "PREFILL"


def _draft_query_width(inputs):
    lengths = inputs.attention_inputs.input_lengths
    if (
        not isinstance(lengths, torch.Tensor)
        or lengths.ndim != 1
        or lengths.numel() == 0
    ):
        raise ValueError("draft inputs require one query length per request")
    width, remainder = divmod(inputs.input_ids.numel(), lengths.numel())
    if remainder or width not in (5, 6):
        raise ValueError("draft inputs must preserve fixed PROPOSE5 or TAIL6 rows")
    return width


class V41DraftFmhaImpl:
    def __init__(self, model, inputs, *, query_width):
        if query_width not in (5, 6) or model._prefill_only or model.kv_cache is None:
            raise ValueError("draft graph needs native decode pages and width5/6")
        self.model = model
        self.query_width = query_width
        self.batch_size, remainder = divmod(inputs.input_ids.numel(), query_width)
        if not self.batch_size or remainder:
            raise ValueError("draft graph needs a positive fixed BxQ bucket")
        self.device = model.device
        self.starts = torch.zeros(
            self.batch_size, dtype=torch.int64, device=self.device
        )
        self.counts = torch.zeros(
            self.batch_size, dtype=torch.int32, device=self.device
        )
        self.active = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self.positions = torch.zeros(
            self.batch_size * query_width, dtype=torch.int64, device=self.device
        )
        self.row_active = torch.zeros(
            self.batch_size * query_width, dtype=torch.bool, device=self.device
        )
        self.offsets = torch.arange(query_width, dtype=torch.int64, device=self.device)
        self.token_ids = torch.zeros(
            self.batch_size * query_width, dtype=torch.int32, device=self.device
        )
        self.aux = (
            torch.zeros(
                (self.batch_size * query_width, 15360),
                dtype=torch.bfloat16,
                device=self.device,
            )
            if query_width == 6
            else None
        )
        self.proposal_buffers = (
            V41DraftAttentionBuffers(self.batch_size, self.device)
            if query_width == 5
            else None
        )
        self.tables, self.swa, self.floors = {}, {}, {}
        self.copy_status, self.writer_status, self.reader_status = {}, {}, {}
        supplied = inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group
        for group in set(model._groups.values()):
            if not 0 <= group < len(supplied):
                raise ValueError("capture is missing a native draft cache group")
            table = supplied[group]
            _tensor(table, table.shape, torch.int32, self.device, "draft graph table")
            if (
                table.ndim != 2
                or table.shape[0] != self.batch_size
                or table.shape[1] < 1
            ):
                raise ValueError("draft table differs from its graph bucket")
            self.tables[group] = torch.zeros_like(table)
        for stage, pages in model._pages.items():
            pages.validate(self.device)
            if pages.entries_per_page != model.layout.swa_entries:
                raise ValueError("draft cache must retain the gamma5 SWA layout")
            self.swa[stage] = SwaBinding(
                pages,
                torch.zeros_like(self.counts),
                torch.zeros_like(self.counts),
                torch.zeros_like(self.counts),
            )
            self.floors[stage] = torch.zeros_like(self.counts)
            self.copy_status[stage] = torch.zeros_like(self.counts)
            self.reader_status[stage] = torch.zeros_like(self.counts)
            self.writer_status[stage] = torch.zeros_like(self.token_ids)
        self.prepared_epoch = torch.zeros((), dtype=torch.int64, device=self.device)
        self.executed_epoch = torch.full_like(self.prepared_epoch, -1)
        self.generation = 0
        self.signature = None

    def support_cuda_graph(self):
        return True

    def prepare_cuda_graph(self, attention_inputs):
        if attention_inputs.context_parallel_info is not None:
            raise ValueError("decode draft graph cannot execute CP prefill")

    def _signature(self, inputs):
        batch = inputs.request_id.numel()
        ids = _host(inputs.request_id, (batch,), torch.int64, "draft request IDs")
        starts = _host(
            inputs.attention_inputs.prefix_lengths,
            (batch,),
            torch.int32,
            "draft prefix lengths",
        )
        fake = _host(inputs.v41_is_fake, (batch,), torch.bool, "draft fake flags")
        return tuple(zip(ids, starts, fake))

    @torch.inference_mode()
    def prepare_model_inputs(self, inputs):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("refresh original draft inputs before replay")
        requests = self._signature(inputs)
        batch = len(requests)
        if (
            batch > self.batch_size
            or _draft_query_width(inputs) != self.query_width
            or inputs.input_ids.numel() != batch * self.query_width
        ):
            raise ValueError("draft request-major rows exceed the graph bucket")
        _tensor(
            inputs.input_ids,
            (batch * self.query_width,),
            torch.int32,
            self.device,
            "draft token IDs",
        )
        _tensor(
            inputs.v41_token_valid,
            inputs.input_ids.shape,
            torch.bool,
            self.device,
            "draft row validity",
        )
        row_valid = inputs.v41_token_valid.view(batch, self.query_width)
        counts = _host(
            row_valid.sum(1, dtype=torch.int32),
            (batch,),
            torch.int32,
            "draft valid rows",
        )
        expected = (
            torch.arange(self.query_width, device=self.device)[None, :]
            < row_valid.sum(1)[:, None]
        )
        torch._assert_async(
            (row_valid == expected).all(),
            "draft valid rows must form a request-local prefix",
        )
        ready = _host(
            inputs.v41_state_ready, (batch,), torch.bool, "draft state-ready flags"
        )
        execution = _host(
            inputs.v41_execution_context,
            (batch, 4),
            torch.int64,
            "draft execution bounds",
        )
        ranges = _host(
            inputs.v41_swa_ranges, (batch, 43, 3), torch.int64, "draft SWA ranges"
        )
        live_ids = [rid for rid, _, fake in requests if not fake]
        if len(set(live_ids)) != len(live_ids):
            raise ValueError("live draft request IDs must be unique")
        for index, (rid, start, fake) in enumerate(requests):
            if (counts[index] == 0) != fake:
                raise ValueError(
                    "draft row validity disagrees with fake request metadata"
                )
            if fake:
                continue
            if (
                rid < 0
                or start <= 0
                or not ready[index]
                or not 0 < counts[index] <= self.query_width
                or start + counts[index] > self.model.config.max_seq_len
                or execution[index][1] != start
                or execution[index][2] != start
            ):
                raise ValueError("draft requires complete canonical target/draft state")
            for begin, end, floor in ranges[index][40:43]:
                if (
                    begin < 0
                    or begin > max(start - 128, 0)
                    or end != start
                    or end - begin > self.model.layout.swa_entries
                    or floor < 0
                    or floor > begin
                ):
                    raise ValueError(
                        "draft continuation has an incomplete SWA interval"
                    )
        self.starts.zero_()
        self.counts.zero_()
        self.starts[:batch].copy_(
            torch.tensor(
                [value[1] for value in requests], dtype=torch.int64, device=self.device
            )
        )
        self.counts[:batch].copy_(
            torch.tensor(counts, dtype=torch.int32, device=self.device)
        )
        self.active.copy_(self.counts > 0)
        self.token_ids.zero_()
        self.token_ids[: batch * self.query_width].copy_(inputs.input_ids)
        if self.query_width == 5:
            self.token_ids.view(self.batch_size, 5)[:, 1:].fill_(
                self.model.config.dspark_noise_token_id
            )
        else:
            aux = inputs.input_hiddens
            if (
                aux.dtype != torch.bfloat16
                or aux.device != self.device
                or aux.numel() != batch * 6 * 15360
            ):
                raise ValueError(
                    "draft TAIL needs matching target L37/L38/L39 feature rows"
                )
            self.aux.zero_()
            self.aux[: batch * 6].copy_(aux.reshape(batch * 6, 15360))
        supplied = inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group
        for group, table in self.tables.items():
            if group >= len(supplied):
                raise ValueError("draft inputs are missing a native cache group")
            source = supplied[group]
            _tensor(
                source, source.shape, torch.int32, self.device, "updated draft table"
            )
            if (
                source.ndim != 2
                or source.shape[0] < batch
                or source.shape[1] > table.shape[1]
            ):
                raise ValueError("updated draft table exceeds captured capacity")
            table.zero_()
            table[:batch, : source.shape[1]].copy_(source[:batch])
        for stage, binding in self.swa.items():
            _same_pages(binding.pages, self.model._pages[stage])
            group = self.model._groups[stage]
            table = self.tables[group]
            previous = (self.starts - 1).clamp_min(0) // self.model.layout.reuse_unit
            logical = (
                previous
                if self.query_width == 5
                else (self.starts + self.counts - 1).clamp_min(0)
                // self.model.layout.reuse_unit
            )
            torch._assert_async(
                (~self.active | (logical < table.shape[1])).all(),
                "draft table is too short",
            )
            source = table.gather(
                1, previous.clamp_max(table.shape[1] - 1)[:, None]
            ).squeeze(1)
            destination = table.gather(
                1, logical.clamp_max(table.shape[1] - 1)[:, None]
            ).squeeze(1)
            torch._assert_async(
                (
                    ~self.active
                    | (
                        (source > 0)
                        & (source < binding.pages.data.shape[0])
                        & (destination > 0)
                        & (destination < binding.pages.data.shape[0])
                    )
                ).all(),
                "draft SWA page is unmapped",
            )
            other = ~torch.eye(self.batch_size, dtype=torch.bool, device=self.device)
            conflict = (destination[:, None] == destination[None, :]) | (
                destination[:, None] == source[None, :]
            )
            torch._assert_async(
                ~(conflict & self.active[:, None] & self.active[None, :] & other).any(),
                "live draft requests share writable SWA",
            )
            binding.page_ids.copy_(destination)
            metadata = [
                ranges[index][40 + stage] if not requests[index][2] else [0, 0, 0]
                for index in range(batch)
            ]
            metadata += [[0, 0, 0]] * (self.batch_size - batch)
            metadata = torch.tensor(metadata, dtype=torch.int32, device=self.device)
            binding.valid_starts.copy_(metadata[:, 0])
            binding.valid_ends.copy_(metadata[:, 1])
            self.floors[stage].copy_(metadata[:, 2])
            _copy_bytes(
                binding.pages.data,
                binding.pages.data,
                source,
                destination,
                self.active & (source != destination),
                self.copy_status[stage],
                copy_bytes=binding.pages.data.shape[1],
            )
        self.signature = requests
        self.generation += 1
        self.prepared_epoch.fill_(self.generation)
        self.model._active_v41_draft_impl = self

    def begin_forward(self):
        active = self.offsets[None, :] < self.counts[:, None]
        self.row_active.copy_(active.flatten())
        positions = self.starts[:, None] + self.offsets[None, :]
        self.positions.copy_(torch.where(active, positions, 0).flatten())

    def finish_forward(self):
        self.executed_epoch.copy_(self.prepared_epoch)

    def check(self, original_inputs):
        if (
            self.signature != self._signature(original_inputs)
            or int(self.executed_epoch) != self.generation
        ):
            raise RuntimeError(
                "draft state publication does not belong to the executed round"
            )
        statuses = [
            *self.copy_status.values(),
            *self.writer_status.values(),
            *self.reader_status.values(),
        ]
        if bool(torch.cat(statuses).ne(0).any()):
            raise RuntimeError("draft compact state transfer or attention failed")


class DeepSeekV41DSparkModel(GptModelBase):
    def __init__(
        self,
        config,
        parallelism_config,
        weights,
        *,
        kv_cache_config,
        max_tokens_per_rank,
        max_generate_batch_size,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None
    ):
        super().__init__(
            config,
            parallelism_config,
            weights,
            max_generate_batch_size,
            fmha_config,
            py_hw_kernel_config,
            device_resource_config,
        )
        if config.num_layers != 3 or config.gen_num_per_cycle != 5:
            raise ValueError("V4.1 DSpark requires three stages and gamma5")
        self._prefill_only = _prefill_role(parallelism_config)
        self.layout = CacheLayout(
            token_block_size=kv_cache_config.seq_size_per_block,
            cp_size=parallelism_config.prefill_cp_config.prefill_cp_size,
            speculative_tokens=5,
            draft_enabled=True,
        )
        self.device = weights.global_weights[W.embedding].device
        self._pages, self._groups = {}, {}
        self._active_v41_draft_impl = None
        self._max_graph_rows = max_generate_batch_size * 6
        self.draft = (
            None
            if self._prefill_only
            else V41DraftModel(
                config.dsv41_config,
                weights,
                ep_size=parallelism_config.ep_size,
                ep_rank=parallelism_config.ep_rank,
                max_tokens_per_rank=max_tokens_per_rank,
            )
        )

    def initialize(self, init_resource):
        super().initialize(init_resource)
        if self.kv_cache is None:
            return True
        cache = self.kv_cache
        raw = cache.kv_cache_base_by_layer_region
        offset = 40 if len(raw) == 43 else 0
        if len(raw) not in (3, 43):
            raise ValueError(
                "native draft cache must expose three local or 43 joint layers"
            )
        region = int(KVCacheRegionName.SWA_KV)
        groups = []
        for stage in range(3):
            group = cache.layer_region_to_group_id[offset + stage][region]
            if (
                not 0 <= group < len(cache.group_seq_size_per_block)
                or cache.group_seq_size_per_block[group] != self.layout.reuse_unit
            ):
                raise ValueError("draft SWA group has the wrong native reuse unit")
            groups.append(group)
        if self._prefill_only:
            # P owns byte-sharded CP pools. Its target committer writes them;
            # this adapter only verifies that handoff and never binds D pages.
            return True
        for stage, group in enumerate(groups):
            pages = CompactPages(
                cache.get_raw_pool_tensor(offset + stage, KVCacheRegionName.SWA_KV),
                CacheRegion.SWA,
                self.layout.swa_entries,
            )
            pages.validate(self.device)
            self._pages[stage], self._groups[stage] = pages, group
        if self.draft is not None:
            warmup_block32_linears(self.draft, max_rows=self._max_graph_rows)
        return True

    def cuda_graph_input_hidden_size(self):
        return 15360

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
        if self.kv_cache is None:
            if is_cuda_graph:
                raise RuntimeError(
                    "capture requires fully initialized native draft pages"
                )
            return None
        if self._prefill_only:
            if is_cuda_graph:
                raise ValueError("V4.1 CP prefill commit does not use decode graphs")
            return None
        # GraphRunner allocates input_hiddens for both entrypoints. Only the
        # fixed request-major geometry distinguishes PROPOSE5 from TAIL6.
        width = _draft_query_width(inputs)
        context = V41DraftFmhaImpl(self, inputs, query_width=width)
        if not is_cuda_graph:
            context.prepare_model_inputs(inputs)
        return context

    def _context(self, inputs, fmha_impl, width):
        context = fmha_impl
        if context is None:
            context = V41DraftFmhaImpl(self, inputs, query_width=width)
            context.prepare_model_inputs(inputs)
        if (
            not isinstance(context, V41DraftFmhaImpl)
            or context.model is not self
            or context.query_width != width
        ):
            raise ValueError("draft entrypoint and graph geometry disagree")
        return context

    @torch.inference_mode()
    def forward_propose(self, inputs, fmha_impl=None):
        if self._prefill_only:
            raise RuntimeError("dedicated P role cannot execute draft proposals")
        if self.kv_cache is None:
            return PyModelOutputs(
                torch.zeros(
                    (inputs.input_ids.numel(), 5120),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
            )
        context = self._context(inputs, fmha_impl, 5)
        hidden = self.draft.propose(context)
        context.finish_forward()
        if (
            not torch.cuda.is_current_stream_capturing()
            and self._active_v41_draft_impl is context
        ):
            context.check(inputs)
        return PyModelOutputs(hidden)

    @torch.inference_mode()
    def forward_commit(self, inputs, fmha_impl=None):
        if self.kv_cache is None:
            return PyModelOutputs(
                torch.zeros((0, 5120), dtype=torch.bfloat16, device=self.device)
            )
        if self._prefill_only:
            return self._already_committed_prefill(inputs)
        context = self._context(inputs, fmha_impl, 6)
        hidden = self.draft.commit(context)
        context.finish_forward()
        if (
            not torch.cuda.is_current_stream_capturing()
            and self._active_v41_draft_impl is context
        ):
            context.check(inputs)
        return PyModelOutputs(hidden)

    def _already_committed_prefill(self, inputs):
        batch = inputs.request_id.numel()
        state = _host(
            inputs.v41_execution_context,
            (batch, 4),
            torch.int64,
            "P draft execution certificate",
        )
        ranges = _host(
            inputs.v41_swa_ranges,
            (batch, 43, 3),
            torch.int64,
            "P draft SWA certificate",
        )
        fake = _host(inputs.v41_is_fake, (batch,), torch.bool, "P fake flags")
        for index in range(batch):
            if fake[index]:
                continue
            end = state[index][2]
            if (
                end <= 0
                or state[index][1] != end
                or any(value[1] != end for value in ranges[index][40:43])
            ):
                raise RuntimeError(
                    "P target must complete all draft projections before its DSpark handoff"
                )
        return PyModelOutputs(
            torch.zeros((0, 5120), dtype=torch.bfloat16, device=self.device)
        )

    def get_execution_states(self, original_inputs):
        if self._active_v41_draft_impl is not None:
            self._active_v41_draft_impl.check(original_inputs)
        return []

    def forward(self, inputs, fmha_impl=None):
        raise RuntimeError(
            "V4.1 DSpark requires fixed forward_propose or forward_commit entrypoints"
        )

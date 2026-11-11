"""PPU resources for the real public V4.1 DSpark/proposer execution loops."""

import json
import hashlib
import os
import weakref
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.deepseek_v41_dspark_model import DeepSeekV41DSparkModel
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.dsv41.kv_cache_utils import as_tagged_attention_inputs
from rtp_llm.models_py.modules.dsv41.ppu.engram import DistributedEnpmGroup
from rtp_llm.models_py.modules.dsv41.ppu.native_protocol import native_cache_key_hash_seed, native_cache_protocol
from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import PrefillPolicy, SharedPrefillPolicy
from rtp_llm.models_py.modules.dsv41.ppu.dspark_attention import SharedPPUDSparkBlock, forward_query, project_commit, unsampled_anchor_metadata
from rtp_llm.models_py.modules.dsv41.ppu.dspark_modes import SharedDSparkMode
from rtp_llm.models_py.modules.dsv41.ppu.layers import PPUSharedRMSNorm
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache
from rtp_llm.models_py.modules.dsv41.ppu.prefill_runtime import SharedForwardState
from rtp_llm.models_py.modules.dsv41.ppu.prefill_runtime import _Overlay
from rtp_llm.models_py.modules.dsv41.ppu.model_runtime import SharedNativeRuntime
from rtp_llm.models_py.modules.dsv41.ppu.transformer import SharedPPUTransformer
from rtp_llm.models_py.modules.dsv41.ppu.model_weights import SharedWeightViews
from rtp_llm.utils.model_weight import W


def _draft_config(config):
    """Use the draft's layer and expert geometry."""
    result = dict(config)
    layers = int(config["num_nextn_predict_layers"])
    if layers != 3:
        raise ValueError("V4.1 DSpark requires exactly three real draft layers")
    result.update(
        num_hidden_layers=layers,
        num_layers=layers,
        compress_ratios=[0] * layers,
        kv_source_layer_ids=[],
        index_source_layer_ids=[],
        engram_layer_ids=[],
        engram_num_embeddings=[],
        n_routed_experts=int(config["dspark_n_routed_experts"]),
        num_experts_per_tok=int(config["dspark_num_experts_per_tok"]),
        original_seq_len=0,
        ppu_model_identity="dspark-draft",
    )
    return result


def _private_draft_request_ids(request_ids):
    if (
        not isinstance(request_ids, torch.Tensor)
        or request_ids.device.type != "cpu"
        or request_ids.dtype not in (torch.int32, torch.int64)
        or request_ids.shape != (8,)
    ):
        return None
    ids = tuple(request_ids.tolist())
    return ids if len(set(ids)) == len(ids) else None


class DeepSeekV41PPUDSparkModel(DeepSeekV41DSparkModel):

    def __init__(self, model_config, parallelism_config, weights, moe_config,
                 max_generate_batch_size, fmha_config=None,
                 py_hw_kernel_config=None, device_resource_config=None):
        if not isinstance(weights, ModelWeights):
            raise TypeError("shared PPU DSpark requires actual framework ModelWeights")
        super().__init__(model_config, parallelism_config, weights, moe_config,
                         max_generate_batch_size, fmha_config=fmha_config,
                         py_hw_kernel_config=py_hw_kernel_config,
                         device_resource_config=device_resource_config)
        config = _draft_config(self._v4_args.v41_config)
        config["ppu_target_capture_layer_ids"] = list(self._capture_aux_hidden_layer_ids)
        config = PrefillPolicy.from_config(config).freeze(config)
        policy = SharedPrefillPolicy.from_config(config)
        mode = SharedDSparkMode.from_config(config)
        if mode is None:
            mode = SharedDSparkMode(
                gamma=int(self._gen_num_per_cycle),
                confidence="observe" if self._dspark_confidence_enabled else "off",
                sample_from_anchor=bool(getattr(model_config, "dspark_sample_from_anchor", True)),
            ).validate()
        if (mode.gamma != self._gen_num_per_cycle
                or mode.max_verified_rows != policy.max_rollback_tokens
                or mode.sample_from_anchor != bool(getattr(model_config, "dspark_sample_from_anchor", True))):
            raise ValueError("shared DSpark model geometry disagrees with frozen startup profile")
        self.shared_mode = mode
        # Existing public proposer API also supports an unsampled anchor row.
        self.init_dspark_proposer(
            width=mode.gamma, query_width=mode.query_width,
            noise_token_id=self._dspark_noise_token_id,
            aux_feature_dim=self._dspark_aux_feature_dim, hidden_dim=self._dspark_hidden_dim,
        )
        config = mode.freeze(config)
        self.shared_weight_views = SharedWeightViews(
            weights, policy.freeze(config), ep_rank=self._v4_args.ep_rank,
            ep_size=self._v4_args.ep_size,
            num_physical_experts=self._v4_args.n_physical_experts,
        )
        self.shared_weight_views.commit_only = self._commit_only_prefill
        self.config.cache_key_hash_seed = native_cache_key_hash_seed(self.shared_weight_views.config)
        self._v4_args.v41_config = self.shared_weight_views.config
        self._shared_ops = None
        self.shared_runtime = None
        self.shared_graph_runtime = None
        self._private_draft_capture = None
        self._private_draft_armed_ids = None
        self._private_draft_pending = None

    def _load_extra_weights(self, weights):
        views, g = self.shared_weight_views, self.shared_weight_views.geometry
        self.main_proj = views.linear(None, W.v4_dspark_main_proj_w)
        self.main_norm = PPUSharedRMSNorm(views.tensor(None, W.v4_dspark_main_norm), g.norm_eps)
        if tuple(self.main_proj.weight.shape) != (g.hidden_size, self._dspark_aux_feature_dim):
            raise ValueError("shared main projection disagrees with public target-feature width")
        if self._dspark_confidence_enabled:
            confidence = weights.global_weights[W.dspark_confidence_head]
            if confidence.dtype != torch.float32 or tuple(confidence.shape) != (1, g.hidden_size + self._dspark_markov_rank):
                raise ValueError("shared confidence head requires real FP32 [1,H+Markov] weights")
            self._dspark_confidence_hidden_weight = confidence[:, :g.hidden_size]
        # No CUDA stacked-WKV packer. The public commit loop calls the PPU
        # per-layer hook with each resident, independently owned WKV module.

    def _initialize_impl(self, resources):
        GptModelBase.initialize(self, resources)
        self._is_decode_role = bool(resources.is_decode_role)
        if not self._commit_only_prefill and self._dspark_confidence_enabled != (self.shared_mode.confidence != "off"):
            raise ValueError("shared confidence weights disagree with selected public mode")
        self._is_speculative = True
        self._max_context_batch_size = int(resources.max_context_batch_size)
        self._max_prefill_batch_tokens = int(getattr(resources, "max_batch_tokens_size", 0))
        self._v4_args.is_decode_role = self._is_decode_role
        if self._materialized:
            return True
        from rtp_llm.models_py.modules.dsv41.ppu.cp_sharding import bind_cache_placement

        self.shared_weight_views.config = bind_cache_placement(
            self.shared_weight_views.config, self.parallelism_config, self._is_decode_role,
        )
        self._v4_args.v41_config = self.shared_weight_views.config
        self.config.cache_key_hash_seed = native_cache_key_hash_seed(self.shared_weight_views.config)
        g, views = self.shared_weight_views.geometry, self.shared_weight_views
        owner = getattr(self, "shared_vocab_owner", None)
        if owner is None or any(self.weight.global_weights[key] is not owner.global_weights[key] for key in (W.embedding, W.lm_head)):
            raise ValueError("shared draft must preserve the actual target embedding/LM-head aliases")
        if (g.hidden_size, g.hc_mult, g.num_layers, g.n_routed_experts, g.n_activated_experts) != (5120, 4, 3, 128, 3):
            raise ValueError("shared draft requires actual H5120/HC4/L3/E128/top3")
        group = dist.group.WORLD if dist.is_initialized() else None
        if self._v4_args.ep_size > 1 and (group is None or dist.get_world_size(group) != self._v4_args.ep_size):
            raise ValueError("draft EP must match the initialized native role world")
        enpm = DistributedEnpmGroup(group) if group is not None else None
        ops = PPUOps(
            ep_rank=self._v4_args.ep_rank, ep_size=self._v4_args.ep_size,
            num_routed_experts=g.n_routed_experts, num_local_experts=views.local_experts,
            num_physical_experts=views.physical_experts,
            hidden_size=g.hidden_size, inter_dim=g.moe_inter_dim,
            swiglu_limit=g.swiglu_limit, mhc_config=g.mhc_config(), enpm_group=enpm,
            deepep_group=group, sm_scale=g.sm_scale(), window=g.window,
        )
        self._shared_ops = ops
        self._v4_args.fp8_kv_cache = self.fp8_kv_cache = True
        self.v4 = SharedPPUTransformer(self._v4_args, views, ops, enpm_group=enpm, block_factory=SharedPPUDSparkBlock)
        self._shared_eplb_layers = tuple(
            layer.ffn.eplb_layer for layer in self.v4.layers
            if getattr(layer, "ffn", None) is not None and layer.ffn.eplb_layer is not None
        )
        self._load_extra_weights(self.weight)
        self._materialized = True
        self.shared_runtime = SharedNativeRuntime(self)
        if self._is_decode_role:
            # Commit has its own Graph state and can capture before proposal.
            self.main_norm._full_graph_runtime = self.shared_runtime
            for layer in self.v4.layers:
                layer.attn.kv_norm._full_graph_runtime = self.shared_runtime
        if hasattr(self, "_shared_stop_callbacks"):
            self.shared_runtime.bind_native_stop(*self._shared_stop_callbacks)
        return True

    def bind_native_stop(self, requested, observed):
        self._shared_stop_callbacks = requested, observed
        if self.shared_runtime is not None:
            self.shared_runtime.bind_native_stop(requested, observed)

    def get_eplb_layers(self):
        if self.v4 is None:
            raise RuntimeError("EPLB bindings require initialized shared draft modules")
        # The public ModelFactory and native draft-init policy keep draft
        # EPLB NONE/R0. Expose only maps actually loaded for this model.
        return self._shared_eplb_layers

    cuda_graph_uses_physical_block_tables = True

    def prepare_fmha_impl_for_graph(self, inputs, forward_method):
        mode = {"forward_propose": "proposal", "forward_commit": "commit"}[forward_method]
        return self.prepare_dspark_graph(inputs, mode)

    def prepare_dspark_graph(self, inputs, mode):
        """Prepare the native draft pools for the selected public forward."""
        from rtp_llm.models_py.modules.dsv41.ppu.dspark_graph import SharedDSparkGraphImpl
        from rtp_llm.models_py.modules.dsv41.ppu.graph_runtime import SharedGraphRuntime

        if mode not in ("proposal", "commit") or self._commit_only_prefill:
            raise ValueError("select a decode-side draft Graph mode")
        for layer in range(self.shared_runtime.geometry.num_layers):
            self.shared_runtime.freqs_for(layer)
        if mode == "proposal" and self.shared_graph_runtime is None:
            self.shared_graph_runtime = SharedGraphRuntime(
                self, row_width=self.shared_mode.query_width, draft=True
            )
        impl = SharedDSparkGraphImpl(self, inputs, mode, runtime=self.shared_graph_runtime)
        if mode == "commit":
            impl.writes = torch.zeros(
                self.shared_runtime.geometry.num_layers, dtype=torch.int64, device=impl.device
            )
            if (
                getattr(impl, "private_draft_journal", None) is not None
                and (impl.batch_size, impl.width) == (8, 4)
            ):
                self._private_draft_capture = weakref.ref(impl)
        return impl

    def arm_private_draft_commit(self, request_ids):
        if self._private_draft_armed_ids is not None or self._private_draft_pending is not None:
            raise RuntimeError("another private Draft KV commit is active")
        if os.environ.get("RTP_V41_PPU_DSPARK_PRIVATE_DRAFT_KV") != "1":
            return False
        impl = self._private_draft_capture() if self._private_draft_capture is not None else None
        if impl is None or not impl.cuda_graph_preparation_in_graph:
            return False
        ids = _private_draft_request_ids(request_ids)
        if ids is None:
            return False
        self._private_draft_armed_ids = ids
        return True

    def note_private_draft_replay(self, impl, inputs):
        """Bind a native Commit replay to the active private-KV lease."""
        if inputs.warmup or inputs.is_fake_stream:
            return
        if self._private_draft_pending is not None:
            raise RuntimeError("private Draft KV replay is already staged")
        ids = self._private_draft_armed_ids
        if ids is None:
            impl.publish_dense_private_draft_commit(inputs)
            return
        if (
            self._private_draft_capture() is not impl
            or _private_draft_request_ids(inputs.request_id) != ids
        ):
            raise ValueError("private Draft KV replay does not match the armed requests")
        receipt = impl.stage_private_draft_commit(inputs)
        if receipt is not None:
            self._private_draft_pending = (ids, receipt)

    def finish_private_draft_stage(self):
        self._private_draft_armed_ids = None
        return self._private_draft_pending is not None

    def private_draft_commit_receipt(self, request_ids):
        if self._private_draft_armed_ids is not None or self._private_draft_pending is None:
            raise RuntimeError("private Draft KV commit requires a finished stage")
        ids, receipt = self._private_draft_pending
        if request_ids != ids:
            raise ValueError("private Draft KV acceptance request order changed")
        return receipt

    def publish_private_draft_commit(self, request_ids, prefixes, accepted_rows):
        try:
            receipt = self.private_draft_commit_receipt(_private_draft_request_ids(request_ids))
            receipt.publish(prefixes, accepted_rows)
        finally:
            self.abort_private_draft_commit()

    def abort_private_draft_commit(self):
        """Discard the lease after the caller joins the Draft stage stream."""
        self._private_draft_armed_ids = None
        self._private_draft_pending = None

    def _commit_layer_features(self, layer_idx, main_x, context_req_ids, context_positions,
                               committed_ends, block_table, tokens_per_block, batch_size,
                               gathered_req_ids=None, gathered_positions=None, cp_ctx=None,
                               metadata_by_geometry=None, host_block_table=None, projected_kv=None):
        runtime, g = self.shared_runtime, self.shared_weight_views.geometry
        state = runtime.current
        if getattr(state, "is_graph_commit", False):
            if cp_ctx is not None or gathered_positions is not None:
                raise ValueError("Prefill commit stays eager")
            from rtp_llm.models_py.modules.dsv41.ppu.dspark_attention import graph_commit_layer

            graph_commit_layer(self, state, layer_idx, main_x, context_req_ids,
                               context_positions, committed_ends)
            return
        if state.idle:
            return
        attn = self.v4.layers[layer_idx].attn
        kv = project_commit(attn, main_x, context_positions, runtime.freqs_for(layer_idx))
        req, pos = context_req_ids, context_positions
        if gathered_positions is not None:
            from rtp_llm.models_py.distributed.collective_torch import Group, all_gather

            kv = all_gather(kv.contiguous(), group=Group.TP)
            req, pos = gathered_req_ids, gathered_positions
        packed = self._shared_ops.encode_kv(kv, pool_tag="swa_kv")
        for rid, end in enumerate(committed_ends.tolist()):
            valid = (req == rid) & (pos >= max(0, int(end) - g.window)) & (pos < end)
            chosen = torch.where(valid)[0]
            chosen = chosen[torch.argsort(pos[chosen])]
            state.cache._copy_rows((layer_idx, "swa_kv"), rid, pos[chosen], packed[chosen])
            state.cache.writes.append({"layer": layer_idx, "request": rid, "positions": pos[chosen].tolist(), "rows": chosen.numel()})
        state.executed_layers.append(layer_idx)

    def _forward_dspark_attention(self, layer_idx, x, query_positions, prefix_lengths,
                                  active_requests, block_table, tokens_per_block,
                                  graph_metadata, metadata_by_geometry):
        if getattr(self.shared_runtime.current, "is_graph", False):
            from rtp_llm.models_py.modules.dsv41.ppu.dspark_attention import forward_graph_query

            return forward_graph_query(
                self, layer_idx, x, query_positions, prefix_lengths, active_requests
            )
        g = self.shared_weight_views.geometry
        # Virtual slots identify ephemeral BF16 workspace rows; the native
        # feature ring has a separate real table and is never proposal scratch.
        geometry = (g.max_seq_len, g.max_seq_len)
        metadata = metadata_by_geometry.get(geometry)
        if metadata is None:
            if self.shared_mode.sample_from_anchor:
                table = torch.arange(1, prefix_lengths.numel() + 1, dtype=torch.int32, device=x.device)[:, None]
                metadata = self._build_noncausal_indices(prefix_lengths, active_requests, table, *geometry)
            else:
                metadata = unsampled_anchor_metadata(
                    prefix_lengths, active_requests, width=self._dspark_query_width,
                    window=g.window, virtual_span=g.max_seq_len,
                )
            metadata_by_geometry[geometry] = metadata
        return forward_query(self, layer_idx, x, query_positions, prefix_lengths, active_requests, metadata)

    def forward_query_block(self, query_ids, query_positions, prefix_lengths,
                            active_requests, inputs, fmha_impl):
        if fmha_impl is not None:
            active = self.shared_runtime.current.metadata.active_requests
            return super().forward_query_block(
                query_ids, query_positions, prefix_lengths, active, inputs, fmha_impl
            )
        if not self.shared_runtime.current.idle:
            return super().forward_query_block(query_ids, query_positions, prefix_lengths, active_requests, inputs, fmha_impl)
        # Idle ranks own no business page table. Keep the real public layer
        # loop and every EP collective, supplying only invalid placeholder rows.
        return self._forward_layers(query_ids, query_positions, prefix_lengths, active_requests,
                                    None, 1, SimpleNamespace(sched_meta_cache={}), inputs.expert_token_counts)

    def _call(self, kind, inputs, fmha_impl):
        runtime = self.shared_runtime
        if runtime is None or runtime.current is not None:
            raise RuntimeError("shared DSpark resources are uninitialized or reentrant")
        if kind == "commit":
            if self._private_draft_armed_ids is not None:
                from rtp_llm.ops import PyModelOutputs

                return PyModelOutputs(
                    torch.empty((0, self._dspark_hidden_dim), dtype=torch.bfloat16, device=runtime.device)
                )
            if self._private_draft_pending is not None:
                raise RuntimeError("private Draft KV commit is awaiting publication")
        if fmha_impl is not None:
            return self._call_graph(kind, inputs, fmha_impl)
        idle = bool(inputs.is_fake_stream)
        if kind == "proposal" and not runtime.vote(0 if idle else inputs.input_ids.numel()):
            return self.dspark_empty_outputs(max(inputs.input_ids.numel() // self._dspark_query_width, 1), runtime.device)
        attn = as_tagged_attention_inputs(inputs.attention_inputs)
        if idle:
            width = self._dspark_query_width if kind == "proposal" else 0
            placeholder = SimpleNamespace(
                input_lengths=torch.tensor([width], dtype=torch.int32),
                input_lengths_device=torch.tensor([width], dtype=torch.int32, device=runtime.device),
                prefix_lengths=torch.zeros(1, dtype=torch.int32),
                prefix_lengths_device=torch.zeros(1, dtype=torch.int32, device=runtime.device),
                context_parallel_info=None,
            )
            inputs = _Overlay(inputs, attention_inputs=placeholder, input_hiddens=None,
                              input_ids=torch.zeros(width, dtype=torch.int32, device=runtime.device))
            attn = placeholder
        cp = getattr(attn, "context_parallel_info", None)
        lengths = tuple(map(int, attn.input_lengths.tolist()))
        prefixes = tuple(map(int, attn.prefix_lengths.tolist()))
        if kind == "commit" and cp is not None and self._dspark_commit_cp_enabled:
            lengths = tuple(map(int, cp.prefill_actual_input_lengths_cpu.tolist()))
            prefixes = tuple(map(int, cp.prefill_prefix_lengths_cpu.tolist()))
        if kind == "proposal":
            lengths = (self._dspark_query_width,) * len(prefixes)
            if any(p < 0 or p + n > runtime.geometry.max_seq_len for p, n in zip(prefixes, lengths)):
                raise ValueError("draft query rows exceed actual absolute position capacity")
        totals = tuple(p + n for p, n in zip(prefixes, lengths))
        cache = None if idle or self.kv_cache is None else SharedNativeCache(
            self.shared_weight_views.config, runtime.geometry, self.kv_cache,
            inputs.attention_inputs, totals, device=runtime.device,
        )
        features = getattr(inputs, "input_hiddens", None)
        if kind == "commit" and features is not None and not bool(torch.isfinite(features).all()):
            raise RuntimeError("shared draft commit received non-finite target features")
        state = SharedForwardState(runtime, inputs, attn, cache, prefill=False, idle=idle, prefixes=prefixes, lengths=lengths)
        if kind == "proposal":
            state.active_token_mask = (torch.tensor(prefixes, device=runtime.device) > 0).repeat_interleave(self._dspark_query_width)
        state.proposal_native_before = cache.fingerprint(prefixes) if cache is not None and kind == "proposal" and os.environ.get("RTP_V41_SHARED_VALIDATE_STATE") == "1" else None
        runtime.current = state
        counts_before = [(layer.attn_norm.calls, layer.ffn_norm.calls, layer.ffn.calls) for layer in self.v4.layers] if not self._commit_only_prefill else []
        try:
            for layer in self.v4.layers:
                if not self._commit_only_prefill:
                    layer.attn_hc.state = layer.ffn_hc.state = None
                    layer.bind_execution_context("decode", speculative=True)
            if kind == "proposal":
                result = super().forward_propose(inputs, fmha_impl)
            else:
                result = super().forward_commit(inputs, fmha_impl)
            confidence = getattr(result, "dspark_confidence_hidden_logits", None)
            if not bool(torch.isfinite(result.hidden_states).all()) or (confidence is not None and not bool(torch.isfinite(confidence).all())):
                raise RuntimeError("shared DSpark returned non-finite hidden/confidence output")
            runtime.calls += 1
            self._report(kind, state, result, counts_before)
            return result
        finally:
            for layer in self.v4.layers:
                if not self._commit_only_prefill:
                    layer.attn_hc.state = layer.ffn_hc.state = None
                    layer.attn_hc.pre_mix_out = layer.ffn_hc.pre_mix_out = None
            runtime.current = None

    def _call_graph(self, kind, inputs, impl):
        from rtp_llm.models_py.modules.dsv41.ppu.dspark_graph import SharedDSparkGraphImpl
        from rtp_llm.ops import PyModelOutputs

        if not isinstance(impl, SharedDSparkGraphImpl) or impl.mode != kind or impl.cache.native is not self.kv_cache:
            raise ValueError("draft Graph metadata must belong to this model and forward method")
        impl.assert_inputs_bound(inputs)
        live = impl.live_requests()
        if kind == "proposal":
            active = live & (impl.prefixes > 0) & (
                impl.prefixes + self._dspark_query_width <= impl.cache.max_seq_len
            )
            valid = active.repeat_interleave(self._dspark_query_width)
            metadata = SimpleNamespace(valid_mask=valid, active_requests=active)
            # The inherited proposal head consumes the final FFN mHC state.
            with impl.runtime.bind(impl, metadata):
                result = super().forward_propose(inputs, impl)
            valid = valid[:, None]
            output = PyModelOutputs(torch.where(valid, result.hidden_states, 0))
            confidence = getattr(result, "dspark_confidence_hidden_logits", None)
            if confidence is not None and confidence.numel():
                output.dspark_confidence_hidden_logits = torch.where(valid, confidence, 0)
            return output
        self.shared_runtime.current = SimpleNamespace(
            is_graph_commit=True, live=live, cache=impl.cache, writes=impl.writes,
        )
        try:
            return super().forward_commit(inputs, impl)
        finally:
            self.shared_runtime.current = None

    @torch.inference_mode()
    def forward_propose(self, inputs, fmha_impl=None):
        if self._commit_only_prefill:
            raise ValueError("dedicated P role is public commit-only")
        if self.kv_cache is None and self.shared_runtime is not None:
            return super().forward_propose(inputs, fmha_impl)
        return self._call("proposal", inputs, fmha_impl)

    @torch.inference_mode()
    def forward_commit(self, inputs, fmha_impl=None):
        if self.kv_cache is None and self.shared_runtime is not None:
            return super().forward_commit(inputs, fmha_impl)
        return self._call("commit", inputs, fmha_impl)

    def _report(self, kind, state, result, counts_before):
        directory = os.environ.get("RTP_V41_SHARED_EVIDENCE_DIR")
        if not directory:
            return
        if state.idle and os.environ.get("RTP_V41_SHARED_COMPACT_IDLE_EVIDENCE") == "1":
            if state.cache is not None or any(state.routed_rows.values()):
                raise AssertionError("idle draft forward owns cache or routed rows")
            return
        rank = dist.get_rank() if dist.is_initialized() else 0
        cache = state.cache
        features = getattr(state.inputs, "input_hiddens", None)
        report = {
            "schema": "v41-ppu-shared-dspark-v1", "rank": rank,
            "quant_mode": self.shared_weight_views.quant_mode,
            "call": self.shared_runtime.calls, "event": kind, "idle": state.idle,
            "commit_only": self._commit_only_prefill, "model_layers": 3,
            "geometry": {"hidden_size": self.shared_weight_views.geometry.hidden_size, "hc_mult": self.shared_weight_views.geometry.hc_mult, "routed_experts": self.shared_weight_views.geometry.n_routed_experts,
                         "top_k": self.shared_weight_views.geometry.n_activated_experts},
            "executed_layers": state.executed_layers,
            "prefixes": state.query_prefixes, "lengths": state.lengths,
            "profile": self.shared_runtime.policy.__dict__,
            "dspark_mode": self.shared_mode.protocol(),
            "cache_protocol": native_cache_protocol(self.shared_weight_views.config),
            "native_config_cache_key_hash_seed": self.config.cache_key_hash_seed,
            "native_writes": cache.writes if cache is not None else [],
            "native_aliases": all(pool.tensor.data_ptr() == pool.handle.kv_cache_base.data_ptr() for pool in cache.pools.values()) if cache is not None else True,
            "native_state": cache.fingerprint(state.query_prefixes if kind == "proposal" else cache.lengths) if cache is not None else {},
            "feature_rows": features.shape[0] if features is not None and features.numel() else 0,
            "feature_finite": bool(torch.isfinite(features).all()) if features is not None else None,
            "output_rows": result.hidden_states.shape[0],
            "output_finite": bool(torch.isfinite(result.hidden_states).all()),
            "confidence_hidden": getattr(result, "dspark_confidence_hidden_logits", None) is not None,
            "input_feature_sha256": hashlib.sha256(features.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest() if features is not None else None,
            "proposal_native_unchanged": cache.fingerprint(state.query_prefixes) == state.proposal_native_before if cache is not None and state.proposal_native_before is not None else None,
            "routed_active_rows": state.routed_rows,
            "normalization": [{"layer": layer.layer_id,
                "attention_norm_calls": layer.attn_norm.calls - before[0],
                "ffn_norm_calls": layer.ffn_norm.calls - before[1],
                "ffn_calls": layer.ffn.calls - before[2]}
                for layer, before in zip(self.v4.layers, counts_before)],
            "resident_conversions": dict(self.shared_weight_views.conversions),
            "vocab_framework_alias": self._commit_only_prefill or self.v4.embed.weight.data_ptr() == self.weight.global_weights[W.embedding].data_ptr(),
            "target_vocab_aliases": {key: self.weight.global_weights[key] is self.shared_vocab_owner.global_weights[key] for key in (W.embedding, W.lm_head)},
            "model_class": type(self).__module__ + "." + type(self).__name__,
        }
        Path(directory).mkdir(parents=True, exist_ok=True)
        with (Path(directory) / f"draft-rank{rank}-call{self.shared_runtime.calls:05d}-{kind}.json").open("x") as writer:
            json.dump(report, writer, indent=2)

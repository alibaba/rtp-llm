"""Explicit shared PPU target, inheriting the actual public V4.1 model.

Never import the dynamic deepseek_v41_model entry as a parent: it selects
this class on PPU and would recurse. Native resources and per-forward state
are PPU-local; complete execution loops remain inherited or delegated.
"""

import os

import torch
import torch.distributed as dist

from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.deepseek_v41_cuda_model import DeepSeekV41Model
from rtp_llm.models_py.modules.dsv41.ppu.engram import DistributedEnpmGroup
from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
from rtp_llm.models_py.modules.dsv41.ppu.block import SharedPPUBlock
from rtp_llm.models_py.modules.dsv41.ppu.transformer import SharedPPUTransformer
from rtp_llm.models_py.modules.dsv41.ppu.model_weights import SharedWeightViews


class SharedPPUTargetBlock(SharedPPUBlock):
    """Keep the public auxiliary-feature producer at the layer entry."""

    def forward_decode(self, x, attn_metadata, input_ids, expert_load=None, **kwargs):
        runtime = self.attn.runtime
        if getattr(runtime.current, "verify", False) and getattr(runtime.current, "is_graph", False):
            runtime.model.v4.capture_aux_hidden(self.layer_id, x)
        return super().forward_decode(x, attn_metadata, input_ids, expert_load=expert_load, **kwargs)


class DeepSeekV41PPUModel(DeepSeekV41Model):
    """Inherited public dispatch/control; PPU-local construction and init only."""

    # Native capture allocates the per-tag tables before creating the adapter.
    cuda_graph_uses_physical_block_tables = True

    def __init__(
        self, model_config, parallelism_config, weight, moe_config,
        max_generate_batch_size, fmha_config=None, py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        if not isinstance(weight, ModelWeights):
            raise TypeError("shared PPU target requires real framework ModelWeights")
        super().__init__(
            model_config, parallelism_config, weight, moe_config,
            max_generate_batch_size, fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        if self._v4_args.tp_size != 1:
            raise ValueError("shared PPU requires Attention TP1 (physical CP may be 8)")
        self.shared_weight_views = SharedWeightViews(
            weight, self._v4_args.v41_config,
            ep_rank=self._v4_args.ep_rank, ep_size=self._v4_args.ep_size,
            num_physical_experts=self._v4_args.n_physical_experts,
        )
        self.shared_weight_views.config["ppu_target_capture_layer_ids"] = list(self._capture_aux_hidden_layer_ids)
        from rtp_llm.models_py.modules.dsv41.ppu.native_protocol import native_cache_key_hash_seed

        # ModelFactory attaches DSpark capture IDs after initial cache-spec
        # parsing. Bind the actual PPU cache identity before native engine init.
        self.config.cache_key_hash_seed = native_cache_key_hash_seed(self.shared_weight_views.config)
        self._v4_args.v41_config = self.shared_weight_views.config
        self._shared_ops = None
        self.shared_runtime = None
        self.shared_graph_runtime = None
        self._graph_receipt = None

    @staticmethod
    def build_weight_consumer(views, layer_id, ops, *, query_only=False, enpm_group=None):
        """Production Block factory, also used by the bounded real-weight smoke."""
        return SharedPPUBlock(
            views, layer_id, ops, query_only=query_only, enpm_group=enpm_group
        )

    def materialize_modules(self):
        """N2 consumer: bind once from loaded ModelWeights, without reloading."""
        if self.v4 is not None:
            return self.v4
        g = self.shared_weight_views.geometry
        group = dist.group.WORLD if dist.is_initialized() else None
        if self._v4_args.ep_size > 1 and group is None:
            raise RuntimeError("shared PPU EP/ENP requires the initialized role world")
        if group is not None and dist.get_world_size(group) != self._v4_args.ep_size:
            raise ValueError("shared PPU EP/ENP group differs from the role world")
        enpm = DistributedEnpmGroup(group) if group is not None else None
        ops = PPUOps(
            ep_rank=self._v4_args.ep_rank, ep_size=self._v4_args.ep_size,
            num_routed_experts=g.n_routed_experts,
            num_physical_experts=self.shared_weight_views.physical_experts,
            num_local_experts=self.shared_weight_views.local_experts,
            hidden_size=g.hidden_size, inter_dim=g.moe_inter_dim,
            swiglu_limit=g.swiglu_limit, mhc_config=g.mhc_config(),
            enpm_group=enpm, deepep_group=group,
            sm_scale=g.sm_scale(), window=g.window,
        )
        ops.moe_engine.configure_fake_balance(
            self._v4_args.fake_balance_expert,
            top_k=g.n_activated_experts,
            dp_rank=self._v4_args.dp_rank, dp_size=self._v4_args.dp_size,
        )
        self._shared_ops = ops
        block_factory = SharedPPUTargetBlock if getattr(self, "_is_speculative", False) else self.build_weight_consumer
        self.v4 = SharedPPUTransformer(
            self._v4_args, self.shared_weight_views, ops, enpm_group=enpm,
            block_factory=block_factory,
        )
        self.v4.set_aux_hidden_capture_layer_ids(self._capture_aux_hidden_layer_ids)
        self.v4.embed.bind_owner(self)
        self._shared_eplb_layers = tuple(
            layer.ffn.eplb_layer for layer in self.v4.layers
            if getattr(layer, "ffn", None) is not None and layer.ffn.eplb_layer is not None
        )
        self._materialized = True
        return self.v4

    def _initialize_impl(self, init_resource):
        from rtp_llm.models_py.model_desc.module_base import GptModelBase
        from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import PrefillPolicy, SharedPrefillPolicy
        from rtp_llm.models_py.modules.dsv41.ppu.model_runtime import SharedNativeRuntime

        GptModelBase.initialize(self, init_resource)
        self._is_decode_role = bool(init_resource.is_decode_role)
        self._is_speculative = bool(init_resource.is_speculative)
        self._max_context_batch_size = int(init_resource.max_context_batch_size)
        self._max_prefill_batch_tokens = int(getattr(init_resource, "max_batch_tokens_size", 0))
        self._v4_args.is_decode_role = self._is_decode_role
        from rtp_llm.models_py.modules.dsv41.ppu.cp_sharding import bind_cache_placement

        config = PrefillPolicy.from_config(self.shared_weight_views.config).freeze(self.shared_weight_views.config)
        config = bind_cache_placement(config, self.parallelism_config, self._is_decode_role)
        policy = SharedPrefillPolicy.from_config(config)
        g = self.shared_weight_views.geometry
        if (g.hidden_size, g.hc_mult, g.num_layers, g.n_routed_experts, g.n_activated_experts, g.window) != (5120, 4, 40, 384, 6, 128):
            raise ValueError("shared native target requires H5120/HC4/L40/E384/top6")
        if self._v4_args.n_layers != g.num_layers:
            raise ValueError("native layer count differs from loaded target geometry")
        if self._is_speculative:
            from rtp_llm.models_py.modules.dsv41.ppu.dspark_modes import SharedDSparkMode

            mode = SharedDSparkMode.from_config(config)
            if mode is None or policy.max_rollback_tokens != mode.max_verified_rows:
                raise ValueError("shared speculative target requires its actual maximum verified row capacity")
        self.shared_weight_views.config = policy.freeze(config)
        self._v4_args.v41_config = self.shared_weight_views.config
        from rtp_llm.models_py.modules.dsv41.ppu.native_protocol import native_cache_key_hash_seed

        self.config.cache_key_hash_seed = native_cache_key_hash_seed(self.shared_weight_views.config)
        # PPU per-tag descriptors use FP8 SWA and packed Global/Indexer bytes.
        # Validate those real pools in SharedNativeCache, independent of the
        # legacy generic KV dtype hint used by the common constructor.
        self._v4_args.fp8_kv_cache = self.fp8_kv_cache = True
        self.materialize_modules()
        self._bind_runtime_buffers(self.v4.embed.weight.device)
        self.shared_runtime = SharedNativeRuntime(self)
        if hasattr(self, "_shared_stop_callbacks"):
            self.shared_runtime.bind_native_stop(*self._shared_stop_callbacks)
        return True

    def bind_native_stop(self, requested, observed):
        self._shared_stop_callbacks = requested, observed
        if self.shared_runtime is not None:
            self.shared_runtime.bind_native_stop(requested, observed)

    def prepare_speculative_forward_completion(self):
        self.shared_runtime.prepare_speculative_forward_completion()

    def prepare_speculative_forward_completion_with_private_draft(self, draft, prefixes):
        self.shared_runtime.prepare_speculative_forward_completion_with_private_draft(draft, prefixes)

    def finish_speculative_forward(self, request_ids, prefixes, accepted_rows):
        return self.shared_runtime.finish_speculative_forward(request_ids, prefixes, accepted_rows)

    def try_finish_speculative_forward_with_private_draft(self, draft, request_ids, prefixes, accepted_rows):
        return self.shared_runtime.try_finish_speculative_forward_with_private_draft(
            draft, request_ids, prefixes, accepted_rows,
        )

    def abort_speculative_forward(self):
        if self.shared_runtime is not None:
            self.shared_runtime.abort_speculative_forward()

    def get_eplb_layers(self):
        if self.v4 is None:
            raise RuntimeError("EPLB bindings require initialized shared PPU modules")
        return self._shared_eplb_layers

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False, **kwargs):
        if is_cuda_graph:
            from rtp_llm.config.v41_runtime import shared_grouped_graph_enabled
            from rtp_llm.models_py.modules.dsv41.kv_cache_utils import as_tagged_attention_inputs
            from rtp_llm.models_py.modules.dsv41.ppu.decode_graph import SharedPPUDecodeGraphImpl
            from rtp_llm.models_py.modules.dsv41.ppu.graph_runtime import SharedGraphRuntime

            attn = as_tagged_attention_inputs(inputs.attention_inputs)
            if bool(attn.is_prefill) and not bool(getattr(attn, "is_target_verify", False)):
                return None
            if self.kv_cache is None:
                return None
            if self._is_speculative and bool(getattr(attn, "is_target_verify", False)):
                from rtp_llm.models_py.modules.dsv41.ppu.dspark_graph import SharedDSparkGraphImpl

                if self.shared_graph_runtime is None:
                    self.shared_graph_runtime = SharedGraphRuntime(
                        self, row_width=self.shared_runtime.policy.max_rollback_tokens
                    )
                mode = "packed" if attn.is_packed_target_verify else "dense"
                return SharedDSparkGraphImpl(self, inputs, mode, runtime=self.shared_graph_runtime)
            if (self._shared_ops.moe_engine.backend == "grouped"
                and not shared_grouped_graph_enabled()):
                raise ValueError("grouped Decode Graph requires the explicit shared PPU opt-in")
            if self.shared_graph_runtime is None:
                self.shared_graph_runtime = SharedGraphRuntime(self)
            impl = SharedPPUDecodeGraphImpl(self, inputs)
            directory = os.environ.get("RTP_V41_N9_GRAPH_REPORT_DIR")
            if directory and self.shared_graph_runtime.grouped is not None:
                from rtp_llm.models_py.modules.dsv41.ppu.graph_metadata import DecodeGraphReceipt

                if impl.batch_size != 1:
                    raise ValueError("N9 Graph replay receipt requires the single bucket1 smoke")
                if self._graph_receipt is None:
                    self._graph_receipt = DecodeGraphReceipt(directory)
                self._graph_receipt.bind(impl, inputs.input_ids, inputs.engram_token_windows)
            return impl
        return None

    def cuda_graph_engram_window_size(self):
        return self.shared_weight_views.geometry.engram_max_ngram_size

    def _uses_host_graph_vote(self):
        return (
            self._is_speculative
            and self._shared_ops.moe_engine.backend == "megamoe"
            and os.environ.get("RTP_V41_PPU_DSPARK_HOST_GRAPH_VOTE", "0") == "1"
        )

    def start_dspark_verify_host_vote(self, local_proposal_rows):
        if self._uses_host_graph_vote():
            self.shared_runtime.start_verify_host_vote(local_proposal_rows)

    def prepare_native_graph_step(self, inputs):
        if self._graph_receipt is not None:
            self._graph_receipt.flush()
            self._graph_receipt.armed = not bool(inputs.warmup)
            if os.environ.get("RTP_V41_N9_GRAPH_CHECK_RESTORE") == "1":
                self._graph_receipt.observe_restored_prefix(self, inputs)
        if bool(inputs.warmup):
            return True
        if self.shared_graph_runtime is not None:
            self.shared_graph_runtime.validate()
        count = 0 if inputs.is_fake_stream else inputs.input_ids.numel()
        # Use a model-wide choice so fake and live ranks vote on the same group.
        return self.shared_runtime.vote(count, host=self._uses_host_graph_vote())

    def native_graph_idle_output(self, inputs):
        from rtp_llm.models_py.modules.dsv41.kv_cache_utils import as_tagged_attention_inputs
        from rtp_llm.ops import PyModelOutputs

        rows = inputs.input_ids.numel()
        hidden = self.v4.embed.weight.new_zeros((rows, self.shared_weight_views.geometry.hidden_size))
        if self._is_speculative and as_tagged_attention_inputs(inputs.attention_inputs).is_target_verify:
            self.shared_runtime.stage_idle_graph_verify(inputs, "all_empty_vote")
            aux = self.get_mtp_target_hidden_states(rows)
            return PyModelOutputs(hidden, torch.zeros_like(aux))
        return PyModelOutputs(hidden)

    def note_native_graph_replay(self):
        self.shared_runtime.graph_replays += 1
        if self._graph_receipt is not None:
            self._graph_receipt.observe()

    @torch.inference_mode()
    def forward(self, inputs, fmha_impl=None):
        from rtp_llm.models_py.model_desc.deepseek_v41_base_model import DeepSeekV4Model

        if self.shared_runtime is None:
            raise RuntimeError("shared PPU native resources are not initialized")
        self._prepare_image_features(inputs)
        try:
            return DeepSeekV4Model.forward(self, inputs, fmha_impl)
        finally:
            self._image_plan = None

    def _forward_impl(self, inputs, fmha_impl=None):
        if fmha_impl is not None:
            from rtp_llm.models_py.modules.dsv41.ppu.decode_graph import SharedPPUDecodeGraphImpl
            from rtp_llm.models_py.modules.dsv41.ppu.dspark_graph import SharedDSparkGraphImpl

            if not isinstance(fmha_impl, (SharedPPUDecodeGraphImpl, SharedDSparkGraphImpl)):
                raise TypeError("shared PPU requires its own Graph metadata adapter")
            return self.shared_graph_runtime.forward(inputs, fmha_impl)
        return self.shared_runtime.dispatch(inputs, fmha_impl)

    def _load_extra_weights(self, weights):
        # Engram's rank-owned assets are bound by SharedPPUBlock, not the
        # CUDA Engram/ViT loader. No second table/projection load here.
        return None

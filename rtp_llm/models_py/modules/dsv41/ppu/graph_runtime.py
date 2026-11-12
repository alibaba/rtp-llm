"""Fixed PPU resources around the inherited V4Transformer Decode loop.

There is no model computation loop here. The only layer iteration binds and
clears PPU device resources. Work/stop votes and asset preparation stay outside
capture; the same forward is used for capture and graph-safe eager.
"""

import os
import weakref
from collections import Counter
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from rtp_llm.config.v41_runtime import env_bool
from rtp_llm.models_py.modules.dsv41.ppu._quant import prepare_e2m1_codebooks
from rtp_llm.models_py.modules.dsv41.ppu.graph_engram import GraphEngram
from rtp_llm.models_py.modules.dsv41.ppu.graph_metadata import increment_decode_counters
from rtp_llm.ops import PyModelOutputs


_GRAPH_WEIGHT_NAMES = ("weight", "bias", "fn", "base", "scale", "q_weight", "k_weight")
_GRAPH_WEIGHT_NAME_SET = frozenset(_GRAPH_WEIGHT_NAMES)
_MODULE_LOOKUP_HOOKS = frozenset(("__dict__", "__getattribute__", "__getattr__"))
_STANDARD_MODULE_GETATTR = torch.nn.Module.__getattr__


def _plain_module_weight_lookup(cls):
    if type(cls) is not type:
        return False
    for base in cls.__mro__:
        members = vars(base)
        if not _GRAPH_WEIGHT_NAME_SET.isdisjoint(members):
            return False
        if base not in (torch.nn.Module, object) and not _MODULE_LOOKUP_HOOKS.isdisjoint(members):
            return False
    return True


def _graph_module_weights(root, *, direct):
    tensor_type = torch.Tensor
    # Cache the lookup rules only for this traversal. Modules and their
    # attributes are still read live for every validation.
    plain_types = {}
    direct = (
        direct and torch.nn.Module.__getattr__ is _STANDARD_MODULE_GETATTR
        and torch.nn.Module.__getattribute__ is object.__getattribute__
    )
    for module in root.modules():
        if direct:
            cls = type(module)
            plain = plain_types.get(cls)
            if plain is None:
                plain = plain_types[cls] = _plain_module_weight_lookup(cls)
            if plain:
                attrs = vars(module)
                params = attrs.get("_parameters")
                buffers = attrs.get("_buffers")
                children = attrs.get("_modules")
                if type(params) is type(buffers) is type(children) is dict:
                    for name in _GRAPH_WEIGHT_NAMES:
                        if name in attrs:
                            value = attrs[name]
                        elif name in params:
                            value = params[name]
                        elif name in buffers:
                            value = buffers[name]
                        else:
                            value = children.get(name)
                        if isinstance(value, tensor_type):
                            yield value
                    continue
        for name in _GRAPH_WEIGHT_NAMES:
            value = getattr(module, name, None)
            if isinstance(value, tensor_type):
                yield value


class SharedGraphRuntime:
    def __init__(self, model, *, row_width=1, draft=False):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("create shared Graph resources before capture")
        self._model = weakref.ref(model)
        self.device = model.v4.embed.weight.device
        self.ops = model._shared_ops
        self.geometry = model.shared_weight_views.geometry
        self.max_batch_size = int(model._max_generate_batch_size)
        self.max_rows = self.max_batch_size * row_width
        g = self.geometry
        expected = (3, 128, 3) if draft else (40, 384, 6)
        if (g.hidden_size, g.hc_mult) != (5120, 4) or (
            g.num_layers, g.n_routed_experts, g.n_activated_experts
        ) != expected:
            raise ValueError("shared Graph model geometry disagrees with its target/draft role")
        prepare_e2m1_codebooks(self.device)
        engine = self.ops.moe_engine
        example = torch.empty((self.max_rows, g.hidden_size), dtype=torch.bfloat16, device=self.device)
        self.grouped = None
        if (engine.backend, engine.precision) == ("grouped", "w4a8"):
            from rtp_llm.models_py.modules.dsv41.ppu.graph_moe import GroupedW4A8Graph

            if model._v4_args.fake_balance_expert:
                raise ValueError("grouped Graph does not support fake expert balance")
            self.grouped = GroupedW4A8Graph(engine, example, range(g.num_layers))
            self.workspace = self.grouped
        elif (engine.backend, engine.precision) == ("megamoe", "w4a4"):
            engine.prepare_shared_graph(example, range(g.num_layers))
            self.workspace = engine._mega_buffer
        else:
            raise ValueError("PPU Graph requires grouped W4A8 or MegaMoE W4A4")
        self.verify_moe_overlap = None
        if (
            not draft and row_width == 4 and self.max_rows == 32 and self.grouped is None
            and env_bool(os.environ, "RTP_V41_PPU_VERIFY_MOE_OVERLAP", default=True)
        ):
            from rtp_llm.models_py.modules.dsv41.ppu.verify_moe_overlap import VerifyMoEOverlapPlans

            self.verify_moe_overlap = VerifyMoEOverlapPlans(
                self.device, engine, target_layers=g.num_layers, max_rows=self.max_rows,
            )
        self.freqs = tuple(model.shared_runtime.freqs_for(i) for i in range(g.num_layers))
        self.layer_calls = torch.zeros(g.num_layers, dtype=torch.int64, device=self.device)
        self.moe_calls = torch.zeros_like(self.layer_calls)
        self.engram_calls = torch.zeros_like(self.layer_calls)
        layers = model.v4.layers
        self._bulk_counters = (
            self.device.type == "cuda" and self.grouped is None and not draft and row_width == 1
            and not model._is_speculative and len(layers) == g.num_layers
            and all(
                layer.layer_id == index and layer.ffn.routed_enabled
                and (layer.engram is not None) == (index in g.engram_layer_ids)
                for index, layer in enumerate(layers)
            )
        )
        if self._bulk_counters:
            self._engram_layer_mask = torch.tensor(
                [index in g.engram_layer_ids for index in range(g.num_layers)],
                dtype=torch.bool, device=self.device,
            )
            model.v4.norm.runtime = model.shared_runtime
        elif (draft or row_width > 1) and self.grouped is None and model._is_speculative:
            # Verify and proposal RMSNorm read the current Graph state.
            model.v4.norm.runtime = model.shared_runtime
            for layer in layers:
                layer.ffn_norm.runtime = model.shared_runtime
        # Bind the optional full-fusion path independently of the existing
        # Verify epilogue, which stays available for unsupported inputs.
        model.v4.norm._full_graph_runtime = model.shared_runtime
        for layer in layers:
            layer.ffn_norm._full_graph_runtime = model.shared_runtime
        self.engram = GraphEngram(
            self.ops, g, device=self.device, max_batch_size=self.max_rows,
            fuse_metadata=self._bulk_counters,
        )
        self.routed_rows_device = torch.zeros_like(self.layer_calls)
        self.attention_capture_calls = Counter()
        self._fast_storage_signature = env_bool(
            os.environ, "RTP_V41_PPU_GRAPH_FAST_STORAGE_SIGNATURE", default=model._is_speculative,
        )
        self._pointers = self._storage_signature()

    def _storage_signature(self):
        model = self._model()
        weights = [model.v4.embed.weight, model.v4.norm.weight]
        weights.extend(model.shared_weight_views._resident.values())
        weights.extend(_graph_module_weights(model.v4, direct=self._fast_storage_signature))
        for values in self.ops.moe_engine._weights.values():
            weights.extend(values)
        for values in self.ops.moe_engine._processed.values():
            weights.extend(v for v in values if isinstance(v, torch.Tensor))
        signature = tuple((t.data_ptr(), tuple(t.shape), t.dtype) for t in weights)
        return signature

    def validate(self):
        self.engram.validate()
        if self.grouped is None and self.workspace is not self.ops.moe_engine._mega_buffer:
            raise RuntimeError("shared Graph MegaMoE workspace changed; recapture required")
        if self._pointers != self._storage_signature():
            raise RuntimeError("shared Graph weight storage changed; recapture required")
        if self.verify_moe_overlap is not None:
            self.verify_moe_overlap.validate()

    @contextmanager
    def bind(self, impl, metadata, *, windows=None):
        model = self._model()
        native = model.shared_runtime
        if native.current is not None or native.pending_verify is not None or native.pending_idle_verify is not None:
            raise RuntimeError("shared Graph cannot overlap an eager/verify transaction")
        if impl.runtime is not self or impl.cache.native is not model.kv_cache:
            raise ValueError("shared Graph metadata belongs to another model/native pool")
        mode = getattr(impl, "mode", None)
        verify = mode in ("dense", "packed")
        proposal_q3 = (
            mode == "proposal"
            and self.grouped is None
            and impl.width == 3
            and metadata.valid_mask.numel() == 24
        )
        mhc_post_mode = None
        mhc_reduce_block_m = None
        if (
            mode == "dense"
            and self.grouped is None
            and getattr(impl, "packed_attention", None) is not None
            and metadata.q_len_per_req == 4
        ):
            selected = os.environ.get("RTP_V41_PPU_VERIFY_MHC_POST", "dot")
            if selected != "off":
                if selected not in ("dot", "epilogue"):
                    raise ValueError("RTP_V41_PPU_VERIFY_MHC_POST must be off, dot or epilogue")
                mhc_post_mode = selected
            if metadata.valid_mask.numel() == 32:
                rows = os.environ.get("RTP_V41_PPU_VERIFY_MHC_REDUCE_BLOCK_M", "1")
                if rows not in ("0", "1", "2", "4", "8"):
                    raise ValueError("RTP_V41_PPU_VERIFY_MHC_REDUCE_BLOCK_M must be 0, 1, 2, 4 or 8")
                if rows != "0":
                    mhc_reduce_block_m = int(rows)
        elif proposal_q3:
            selected = os.environ.get("RTP_V41_PPU_PROPOSAL_MHC_POST", "dot")
            if selected not in ("off", "dot", "epilogue"):
                raise ValueError("RTP_V41_PPU_PROPOSAL_MHC_POST must be off, dot or epilogue")
            if selected != "off":
                mhc_post_mode = selected
            rows = os.environ.get("RTP_V41_PPU_PROPOSAL_MHC_REDUCE_BLOCK_M", "1")
            if rows not in ("0", "1", "2", "4", "8"):
                raise ValueError("RTP_V41_PPU_PROPOSAL_MHC_REDUCE_BLOCK_M must be 0, 1, 2, 4 or 8")
            if rows != "0":
                mhc_reduce_block_m = int(rows)
        mhc_post = (mhc_post_mode, proposal_q3) if mhc_post_mode is not None else None
        proposal_batched_wo_a = proposal_q3 and env_bool(
            os.environ, "RTP_V41_PPU_PROPOSAL_BATCHED_WO_A", default=True,
        )
        bulk_counters = self._bulk_counters and not verify
        hashes = None
        if self.geometry.engram_layer_ids:
            if windows is None or tuple(windows.shape) != (
                metadata.valid_mask.numel(), self.geometry.engram_max_ngram_size,
            ):
                raise ValueError("shared Graph requires scheduler token windows")
            hashes = self.engram.hash(windows)
        state = SimpleNamespace(
            is_graph=True, prefill=False, metadata=metadata, cache=impl.cache,
            shared={}, engram=self.engram, freqs=self.freqs,
            layer_calls=self.layer_calls, moe_calls=self.moe_calls,
            engram_calls=self.engram_calls, routed_rows_device=self.routed_rows_device,
            grouped_moe=self.grouped,
            verify=verify, proposal=mode == "proposal", bulk_counters=bulk_counters,
            verify_moe_overlap=(
                self.verify_moe_overlap.lookup_current_stream()
                if self.verify_moe_overlap is not None and mode == "dense"
                and metadata.valid_mask.numel() == 32 else None
            ),
            verify_moe_overlap_allowed=mode == "dense",
            proposal_batched_wo_a=proposal_batched_wo_a,
            packed_attention=getattr(impl, "packed_attention", None),
            visible_indexer=getattr(impl, "visible_indexer", False),
            attention_calls=Counter(),
        )
        native.current = state
        try:
            for index, layer in enumerate(model.v4.layers):
                layer.attn_hc.state = layer.ffn_hc.state = None
                layer.attn_hc.pre_mix_out = layer.ffn_hc.pre_mix_out = None
                if mhc_post is not None:
                    layer.attn_hc._graph_mhc_post = layer.ffn_hc._graph_mhc_post = mhc_post
                if mhc_reduce_block_m is not None:
                    layer.attn_hc._graph_mhc_reduce_block_m = mhc_reduce_block_m
                    layer.ffn_hc._graph_mhc_reduce_block_m = mhc_reduce_block_m
                if layer.engram is not None:
                    column = self.geometry.engram_layer_ids.index(index)
                    layer.engram_hashes = hashes[:, column]
                    layer.engram_token_mask = metadata.valid_mask & (layer.engram_hashes[:, 0] >= 0)
                    layer.engram._graph_state = state
            yield state
        finally:
            for layer in model.v4.layers:
                layer.attn_hc.state = layer.ffn_hc.state = None
                layer.attn_hc.pre_mix_out = layer.ffn_hc.pre_mix_out = None
                if mhc_post is not None:
                    layer.attn_hc._graph_mhc_post = layer.ffn_hc._graph_mhc_post = None
                if mhc_reduce_block_m is not None:
                    layer.attn_hc._graph_mhc_reduce_block_m = None
                    layer.ffn_hc._graph_mhc_reduce_block_m = None
                if layer.engram is not None:
                    layer.engram._graph_state = None
                    layer.engram_hashes = layer.engram_token_mask = None
            native.current = None

    @torch.inference_mode()
    def forward(self, inputs, impl):
        model = self._model()
        rows = inputs.input_ids.numel()
        if rows != impl.rows or rows > self.max_rows:
            raise ValueError("shared Graph input does not fit captured row geometry")
        verify = getattr(impl, "mode", None) in ("dense", "packed")
        if self.verify_moe_overlap is not None and getattr(impl, "mode", None) == "dense" and rows == 32:
            if torch.cuda.is_current_stream_capturing():
                self.verify_moe_overlap.lookup_current_stream(required=True)
            elif os.environ.get("RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD") == "1":
                # initCapture's dtype probe and the actual capture warmup use
                # different callers. Each gets its own strict stream plan.
                self.verify_moe_overlap.prepare_current_stream()
        if verify:
            impl.assert_inputs_bound(inputs)
        elif self._bulk_counters:
            impl._fuse_decode_metadata = True
        impl.prepare_in_graph(inputs.attention_inputs)
        metadata = impl.metadata
        with self.bind(impl, metadata, windows=inputs.engram_token_windows) as state:
            ids = torch.where(metadata.valid_mask, inputs.input_ids.reshape(rows), 0)
            hidden = model.v4.forward_decode(
                ids, metadata, expert_token_counts=inputs.expert_token_counts,
                kv_cache=model.kv_cache,
            )
            if state.bulk_counters:
                increment_decode_counters(
                    self.layer_calls, self.moe_calls, self.engram_calls, self._engram_layer_mask,
                )
            if state.attention_calls and torch.cuda.is_current_stream_capturing():
                self.attention_capture_calls.update(state.attention_calls)
            if state.verify:
                aux = model.v4._mtp_hidden_buffer
                if aux is not None:
                    aux[:rows].masked_fill_(~metadata.valid_mask[:, None], 0)
                impl.consumed_ids.copy_(inputs.input_ids)
                impl.consumed_windows.copy_(inputs.engram_token_windows)
                impl.cache.completions.add_(1)
        return PyModelOutputs(torch.where(metadata.valid_mask[:, None], hidden, 0))

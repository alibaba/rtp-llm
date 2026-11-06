"""PPU model classes for DeepSeek V4.1 Flash.

* :class:`PPUModelCore` — the STANDALONE model: a V4.1 config dict + a bound
  :class:`PPUWeightBundle` + a bound ``PPUOps`` (+ optional device). It owns
  the rank's :class:`PPUKVCache` and exposes ``prefill`` (this wave) and a
  ``decode`` seam for L3 (``ppu/decode.py`` plugs itself in via
  ``core.decode_impl = ...`` or subclasses override :meth:`PPUModelCore.decode`;
  until then decode fails loudly).

* :class:`DeepSeekV41PPUModel` — the framework wrapper satisfying the A0
  instantiation contract
  ``(model_config, parallelism_config, weight, moe_config,
  max_generate_batch_size, fmha_config, py_hw_kernel_config,
  device_resource_config)`` with ``forward(PyModelInputs) -> PyModelOutputs``
  returning the pre-lm-head hidden. The framework loads the global weights
  used by native sampling; the core loads Attention TP1 and EP/ENP shards
  directly from the checkpoint and uses the framework's tagged cache pools.

All compute goes through the ``PPUOps`` contract ops (see ``prefill.py``);
this module only wires weights, cache and IO.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch
from rtp_llm.models_py.modules.dsv41.ppu.cache import PPUKVCache
from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
from rtp_llm.models_py.modules.dsv41.ppu.prefill import PrefillResult, run_prefill
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import PrefillPolicy
from rtp_llm.models_py.modules.dsv41.ppu.weight import PPUWeightBundle, V41Geometry

logger = logging.getLogger(__name__)

__all__ = ["PPUModelCore", "DeepSeekV41PPUModel"]


class PPUModelCore:
    """Standalone single-rank V4.1 model (config + weights + ops + cache)."""

    def __init__(
        self,
        config: dict,
        weights: PPUWeightBundle,
        ops: PPUOps,
        *,
        cache: Optional[PPUKVCache] = None,
        device=None,
    ):
        self.prefill_policy = PrefillPolicy.from_config(config)
        self.config = self.prefill_policy.freeze(config)
        self.weights = weights
        self.ops = ops
        self.geometry: V41Geometry = weights.geometry
        if cache is None:
            cache = PPUKVCache(self.config, device=device)
        self.cache = cache
        self.device = (
            torch.device(device) if device is not None else weights.embedding.device
        )
        # L3 decode seam: assigned by the decode orchestrator (ppu/decode.py).
        self.decode_impl: Optional[Callable[..., object]] = None
        self.capture_layer_ids = ()

    # ---------------------------------------------------------------- prefill

    def prefill(
        self,
        input_ids: torch.Tensor,
        *,
        positions=None,
        cu_seqlens=None,
        request_ids=None,
        lookback_token_ids=None,
        cache: Optional[PPUKVCache] = None,
        apply_lm_head: bool = False,
        parallel_context: Optional[object] = None,
        allow_ced: bool = True,
        replay_prefix_lengths=None,
    ) -> PrefillResult:
        """Single-rank prefill (see ``prefill.run_prefill``)."""
        return run_prefill(
            self.weights,
            self.ops,
            self.cache if cache is None else cache,
            input_ids,
            positions=positions,
            cu_seqlens=cu_seqlens,
            request_ids=request_ids,
            lookback_token_ids=lookback_token_ids,
            apply_lm_head=apply_lm_head,
            parallel_context=parallel_context,
            policy=self.prefill_policy,
            allow_ced=allow_ced,
            replay_prefix_lengths=replay_prefix_lengths,
            capture_layer_ids=self.capture_layer_ids,
        )

    # ---------------------------------------------------------------- decode

    def decode(self, *args, **kwargs):
        """Decode seam: L3's ``ppu/decode.py`` plugs the step orchestration in.

        Until the decode batch lands, calling this fails loudly instead of
        silently running a prefill-shaped forward.
        """
        if self.decode_impl is not None:
            return self.decode_impl(self, *args, **kwargs)
        raise NotImplementedError(
            "V4.1 PPU decode orchestration is the L3 deliverable "
            "(rtp_llm/models_py/modules/dsv41/ppu/decode.py); wire it via "
            "PPUModelCore.decode_impl or a subclass override. No silent "
            "prefill fallback."
        )


def _v41_config_from_model_config(model_config) -> dict:
    config = getattr(model_config, "deepseek_v41_config", None)
    if not config:
        raise ValueError(
            "DeepSeek V4.1 PPU requires model_config.deepseek_v41_config (the "
            "checkpoint text config parsed by the entry model)"
        )
    return dict(config)


class DeepSeekV41PPUModel(torch.nn.Module):
    """Framework-facing V4.1 PPU model (A0 instantiation contract)."""

    cuda_graph_uses_physical_block_tables = True

    def __init__(
        self,
        model_config,
        parallelism_config,
        weight,
        moe_config,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__()
        config = _v41_config_from_model_config(model_config)
        config["ppu_target_capture_layer_ids"] = list(
            getattr(model_config, "capture_aux_hidden_layer_ids", None) or ()
        )
        ep_size = int(getattr(parallelism_config, "ep_size", 1) or 1)
        ep_rank = int(getattr(parallelism_config, "ep_rank", 0) or 0)
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        checkpoint_path = config.get("checkpoint_path") or getattr(
            model_config, "ckpt_path", None
        )
        if checkpoint_path is None:
            raise ValueError("native PPU model requires its checkpoint path")
        if "engram_token_map" not in config:
            import os
            from pathlib import Path

            from rtp_llm.models_py.modules.dsv41.ppu.tools.driver_prefill import (
                _load_token_map,
            )

            config["engram_token_map"] = _load_token_map(
                str(Path(checkpoint_path) / "tokenizer.json"),
                os.environ.get("RTP_V41_TOKEN_MAP_CACHE"),
            )
        geometry = V41Geometry.from_config(config)
        import torch.distributed as dist
        from rtp_llm.models_py.modules.dsv41.ppu.engram import DistributedEnpmGroup
        from rtp_llm.models_py.modules.dsv41.ppu.weight import load_ppu_weights

        group = dist.group.WORLD if dist.is_initialized() else None
        if ep_size > 1 and group is None:
            raise RuntimeError(
                "native PPU EP8/ENP8 requires the initialized role world"
            )
        enpm_group = DistributedEnpmGroup(group) if group is not None else None
        ops = PPUOps(
            ep_rank=ep_rank,
            ep_size=ep_size,
            num_routed_experts=geometry.n_routed_experts,
            num_local_experts=geometry.num_local_experts(ep_size),
            hidden_size=geometry.hidden_size,
            inter_dim=geometry.moe_inter_dim,
            swiglu_limit=geometry.swiglu_limit,
            mhc_config=geometry.mhc_config(),
            sm_scale=geometry.sm_scale(),
            window=geometry.window,
            enpm_group=enpm_group,
            deepep_group=group,
        )
        bundle = load_ppu_weights(
            config,
            checkpoint_path,
            ep_rank=ep_rank,
            ep_size=ep_size,
            enp_rank=ep_rank,
            enp_size=ep_size,
            device=device,
        )
        bundle.bind(ops, enpm_group=enpm_group)
        # Sampling and vocabulary sharding remain in native forwardPostLayers.
        bundle.lm_head = None
        self.core = PPUModelCore(config, bundle, ops, device=device)
        from rtp_llm.models_py.modules.dsv41.ppu.native_model import NativeModelRunner

        dp_size = int(getattr(parallelism_config, "dp_size", 1))
        tp_size = int(getattr(parallelism_config, "tp_size", 1))
        self.runner = NativeModelRunner(
            self.core,
            dp_rank=int(getattr(parallelism_config, "dp_rank", 0)),
            dp_size=dp_size,
            cp_rank=int(getattr(parallelism_config, "tp_rank", 0)),
            cp_size=tp_size if dp_size == 1 else 1,
            group=group,
        )
        self._max_generate_batch_size = int(max_generate_batch_size)
        self._decode_graph_enabled = bool(
            getattr(py_hw_kernel_config, "enable_cuda_graph", False)
        )
        self._decode_graph_config = dict(self.core.config)
        runtime_max_seq_len = int(getattr(model_config, "max_seq_len", 0) or 0)
        if runtime_max_seq_len > 0:
            self._decode_graph_config["max_position_embeddings"] = min(
                int(config["max_position_embeddings"]), runtime_max_seq_len
            )
        self._decode_graph_runner = None
        self._target_capture_layer_ids = tuple(
            map(int, getattr(model_config, "capture_aux_hidden_layer_ids", None) or ())
        )
        logger.info(
            "[DeepSeekV41PPUModel] layers=%d hidden=%d heads=%d experts=%d/%d "
            "topk=%d kv_sources=%s engram=%s ep=%d/%d",
            geometry.num_layers,
            geometry.hidden_size,
            geometry.n_heads,
            geometry.n_routed_experts,
            geometry.num_local_experts(ep_size),
            geometry.n_activated_experts,
            list(geometry.kv_source_layer_ids),
            list(geometry.engram_layer_ids),
            ep_rank,
            ep_size,
        )

    # ---------------------------------------------------------------- forward

    def bind_native_stop(self, requested, observed):
        self.runner.bind_native_stop(requested, observed)

    def initialize(self, resources):
        from rtp_llm.ops import PyModelInputs

        if not hasattr(PyModelInputs(), "is_fake_stream"):
            raise RuntimeError(
                "V4.1 native PPU requires rebuilt bindings with is_fake_stream/warmup"
            )
        if bool(getattr(resources, "is_speculative", False)):
            self.core.capture_layer_ids = self._target_capture_layer_ids
        return self.runner.initialize(resources)

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False, **kwargs):
        if not is_cuda_graph:
            return None
        if self.runner.kv_cache is None:
            return None
        if not self.runner.decode_role:
            raise ValueError("PPU Graph supports the DP8 decode role only")
        if self.core.ops.moe_engine.backend != "megamoe":
            raise ValueError("PPU decode Graph requires MegaMoE/W4A4")
        from rtp_llm.models_py.modules.dsv41.ppu.graph_cache import PPUDecodeGraphImpl

        return PPUDecodeGraphImpl(
            self._decode_graph_config,
            self.runner.kv_cache,
            inputs.attention_inputs,
            device=self.core.device,
            batch_size=int(inputs.input_ids.numel()),
        )

    def cuda_graph_engram_window_size(self):
        return (
            self.core.geometry.engram_max_ngram_size
            if self.core.geometry.engram_enabled()
            else 0
        )

    def _graph_runner(self):
        if self._decode_graph_runner is None:
            from rtp_llm.models_py.modules.dsv41.ppu.graph_decode import (
                GraphDecodeRunner,
            )

            self._decode_graph_runner = GraphDecodeRunner(
                self.core,
                max_batch_size=self._max_generate_batch_size,
                dp_rank=self.runner.dp_rank,
                dp_size=self.runner.dp_size,
            )
        return self._decode_graph_runner

    def prepare_native_graph_step(self, inputs):
        """Run the role-wide work/stop vote outside capture and every replay."""
        if not self.runner.decode_role or bool(inputs.warmup):
            return True
        if self._decode_graph_runner is not None:
            self._decode_graph_runner.validate()
        return self.runner.prepare_graph_step(inputs)

    def native_graph_idle_output(self, inputs):
        from rtp_llm.ops import PyModelOutputs

        return PyModelOutputs(
            self.core.weights.embedding.new_zeros(
                (inputs.input_ids.numel(), self.core.geometry.hidden_size)
            )
        )

    def note_native_graph_replay(self):
        self.runner.note_graph_replay()

    def finish_speculative_forward(self, request_ids, prefixes, accepted_rows):
        self.runner.finish_speculative_forward(request_ids, prefixes, accepted_rows)

    def abort_speculative_forward(self):
        self.runner.abort_speculative_forward()

    def has_mtp_hidden_buffer(self):
        # CP admission asks for a rank-local feature producer. Values travel
        # through each forward's PyModelOutputs, never a mutable accessor.
        return bool(self.core.capture_layer_ids)

    @torch.inference_mode()
    def forward(self, inputs, fmha_impl=None):
        """``forward(PyModelInputs) -> PyModelOutputs`` (pre-lm-head hidden)."""
        from rtp_llm.ops import PyModelOutputs

        if fmha_impl is not None:
            return PyModelOutputs(self._graph_runner().forward(inputs, fmha_impl))
        if (
            self._decode_graph_enabled
            and self.runner.decode_role
            and not bool(inputs.warmup)
            and self.runner.kv_cache is not None
        ):
            raise RuntimeError("DP8 PPU decode Graph input missed its captured bucket")
        hidden, aux = self.runner.forward(inputs, return_aux=True)
        return (
            PyModelOutputs(hidden, aux) if aux is not None else PyModelOutputs(hidden)
        )

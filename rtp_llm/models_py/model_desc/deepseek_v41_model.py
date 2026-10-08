"""Upstream V4.1 text worker on the base paged-cache runtime.

Adapted from e4a9162f6f. Serving registration remains gated while this worker
is qualified incrementally; only eager short PP1/CP1/TP1/EP1 prefill is admitted.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from rtp_llm.models_py.model_desc.deepseek_v4_model import DeepSeekV4Model
from rtp_llm.models_py.modules.dsv4.engram import Engram, EngramLayout, NgramHashState
from rtp_llm.models_py.modules.dsv4.fp8.prefill_meta import (
    clear_prefill_meta_shared_fp8,
)
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import primary_attention_inputs
from rtp_llm.models_py.modules.dsv4.utils import V41MXFP8Linear
from rtp_llm.models_py.modules.dsv4.v41_engram_host import _locked, _PinnedSharedTable
from rtp_llm.models_py.modules.dsv4.v41_engram_weights import (
    load_engram_weights,
    plan_stage_engram_sources,
)


class _SharedTableEmbedding:
    """Use upstream lookup arithmetic with the qualified explicit table lease."""

    def __init__(self, table):
        self._table = table
        self._num_sms = torch.cuda.get_device_properties(
            table._device
        ).multi_processor_count

    @property
    def weight(self):
        self._table._check_open()
        return self._table._weight.view(torch.float8_e4m3fn)

    def __call__(self, indices, device):
        from rtp_llm.models_py.modules.dsv4._engram_triton import lookup_host_rows

        table = self._table
        with table._mutex:
            table._check_open()
            if indices.device != table._device or torch.device(device) != table._device:
                raise ValueError("Engram lookup must use its registered worker device")
            return lookup_host_rows(
                table._cuda_weight, table._cuda_scales, indices, self._num_sms
            )


class DeepSeekV41Model(DeepSeekV4Model):
    def __init__(self, model_config, *args, **kwargs):
        super().__init__(model_config, *args, **kwargs)
        config = dict(getattr(model_config, "deepseek_v41_config", None) or {})
        if not config:
            raise ValueError("DeepSeek V4.1 requires deepseek_v41_config")
        self._v4_args.v41_config = config
        self._v4_args.n_hash_layers = 0
        self._engram_hash_state = None
        self._engram_layers = ()
        self._engram_tables = {}
        self._engram_sources = {}
        self._worker_lock = threading.RLock()
        self._worker_pid = os.getpid()
        self._worker_state = "new"
        self._worker_stream = None

    def _check_worker(self):
        if os.getpid() != self._worker_pid:
            raise RuntimeError("V4.1 workers must be spawned, not inherited after fork")
        if self._worker_state in ("failed", "closing", "closed"):
            raise RuntimeError(f"V4.1 worker is {self._worker_state}")

    def _plan_engram(self):
        config = self._v4_args.v41_config
        owned = set(self.pp_layer_ids())
        if not owned.intersection(config.get("engram_layer_ids", ())):
            return {}
        # This is a metadata identity, not a hash of the checkpoint payload.
        # Every admitted span additionally binds file identity and header hash.
        identity = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        sources = plan_stage_engram_sources(
            SimpleNamespace(text=config), config["checkpoint_path"], owned, identity
        )
        total = sum(source.table.nbytes for source in sources.values())
        budget = int(os.environ.get("DSV41_ENGRAM_HOST_BYTES", "0"))
        if total > budget:
            raise ValueError(
                f"Engram tables need {total} host bytes; DSV41_ENGRAM_HOST_BYTES={budget}"
            )
        return sources

    def _check_engram_capacity(self):
        needed = 0
        for source in self._engram_sources.values():
            source.check_sources()
            table = source.table
            path = Path("/dev/shm") / f"rtp_llm_engram_{os.getuid()}_{table.key}"
            ready = Path(str(path) + ".json")
            if not (
                path.exists()
                and path.stat().st_size == table.nbytes
                and ready.exists()
                and ready.read_bytes() == table.manifest
            ):
                needed += table.nbytes
        if needed > shutil.disk_usage("/dev/shm").free:
            raise RuntimeError(
                f"Engram combined tables need {needed} additional /dev/shm bytes"
            )

    def _initialize_impl(self, init_resource):
        with self._worker_lock:
            self._check_worker()
            args = self._v4_args
            if (
                args.pp_size != 1
                or args.tp_size != 1
                or args.ep_size != 1
                or args.dp_size != 1
                or args.world_size != 1
                or self.parallelism_config.local_cp_enabled()
                or not args.fp8_kv_cache
                or args.commit_only
                or init_resource.is_speculative
                or init_resource.is_decode_role
                or self._gen_num_per_cycle != 1
            ):
                raise NotImplementedError(
                    "V4.1 worker currently requires eager PP1/CP1/TP1/DP1/EP1 prefill"
                )
            if self._materialized:
                return super()._initialize_impl(init_resource)
            device = next(iter(self.weight.global_weights.values())).device
            if device.type != "cuda" or torch.cuda.get_device_capability(device) != (
                12,
                0,
            ):
                raise NotImplementedError(
                    "V4.1 worker currently requires SM120 weights"
                )
            self._engram_sources = self._plan_engram()
            self._check_engram_capacity()
            try:
                result = super()._initialize_impl(init_resource)
            except BaseException:
                self._worker_state = "failed"
                self.close()
                raise
            self._worker_state = "ready"
            self._worker_stream = torch.cuda.current_stream(self._weight_device())
            return result

    def _load_extra_weights(self, weights):
        if not self._engram_sources:
            return
        config = self._v4_args.v41_config
        layout = EngramLayout(config)
        device = self._weight_device()
        hash_state = NgramHashState(layout, config["checkpoint_path"], device=device)
        if hash_state.token_map.numel() != self._v4_args.vocab_size:
            raise ValueError("Engram tokenizer size disagrees with model vocabulary")
        by_global_id = {layer.layer_id: layer for layer in self.v4.layers}
        modules = []
        # Serialize aggregate admission and staging across these workers. The
        # table provider also reserves bytes before copying, preventing SIGBUS.
        lock = Path("/tmp") / f"rtp_llm_v41_engram_admission_{os.getuid()}.lock"
        with _locked(lock):
            self._check_engram_capacity()
            for layer_id, source in self._engram_sources.items():
                table = _PinnedSharedTable(source.table, max_bytes=source.table.nbytes)
                self._engram_tables[layer_id] = table
                table.register_cuda(device)
                tensors = load_engram_weights(
                    source, max_bytes=source.weight_bytes, device=device
                )
                projection = V41MXFP8Linear(
                    tensors["weight"], tensors["scales"].view(torch.float8_e8m0fnu)
                )
                index = layout.layer_ids.index(layer_id)
                module = Engram(
                    layout,
                    index,
                    _SharedTableEmbedding(table),
                    projection,
                    tensors["q_weight"],
                    tensors["k_weight"],
                    source.norm_eps,
                )
                modules.append((by_global_id[layer_id], index, module))
        # Publish only after every owned table and projection is ready.
        for layer, _, module in modules:
            layer.engram = module
        self._engram_layers = tuple((layer, index) for layer, index, _ in modules)
        self._engram_hash_state = hash_state

    def _prepare_engram(self, inputs):
        if not self._engram_layers:
            return
        windows = inputs.engram_token_windows
        if (
            windows is None
            or windows.ndim != 2
            or windows.shape != (inputs.input_ids.numel(), 4)
            or windows.dtype != torch.int32
            or not windows.is_contiguous()
        ):
            raise ValueError(
                "Engram requires contiguous int32 [tokens, 4] token windows"
            )
        windows = windows.to(device=self._weight_device(), non_blocking=True)
        ids = inputs.input_ids.reshape(-1).to(windows.device)
        if bool(((windows < -1) | (windows >= self._v4_args.vocab_size)).any()):
            raise ValueError("Engram token history contains an invalid token ID")
        if bool(((windows[:, 0] >= 0) & (windows[:, 0] != ids)).any()):
            raise ValueError("Engram current tokens disagree with input_ids")
        hashes = self._engram_hash_state.hash_token_windows(windows)
        token_mask = windows[:, 0] >= 0
        for layer, index in self._engram_layers:
            layer.engram_hashes = hashes[:, index].contiguous()
            layer.engram_token_mask = token_mask

    def _clear_forward_state(self):
        if self.v4 is None:
            return
        for layer in self.v4.layers:
            layer.engram_hashes = None
            layer.engram_token_mask = None
            layer.attn_hc.pre_mix_out = None
            layer.ffn_hc.pre_mix_out = None
        clear_prefill_meta_shared_fp8(self.v4)

    def _validate_prefill(self, inputs, attn):
        lengths, prefixes = attn.input_lengths, attn.prefix_lengths
        if (
            lengths is None
            or prefixes is None
            or lengths.ndim != 1
            or prefixes.shape != lengths.shape
            or not 0 < lengths.numel() <= self._max_generate_batch_size
            or lengths.dtype not in (torch.int32, torch.int64)
            or prefixes.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("V4.1 prefill requires matching integer request lengths")
        lengths, prefixes = lengths.cpu(), prefixes.cpu()
        if (
            bool((lengths <= 0).any())
            or bool((prefixes < 0).any())
            or int(lengths.sum()) != inputs.input_ids.numel()
        ):
            raise ValueError("V4.1 prefill lengths must cover nonempty requests")
        if bool(((prefixes + lengths) > min(256, self._v4_args.max_seq_len)).any()):
            raise NotImplementedError(
                "V4.1 prefill currently requires the first 256-token page"
            )
        cu = attn.cu_seqlens
        expected_cu = torch.cat((torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)))
        if (
            cu is not None
            and cu.numel()
            and (
                cu.dtype not in (torch.int32, torch.int64)
                or not torch.equal(cu.cpu(), expected_cu)
            )
        ):
            raise ValueError("V4.1 prefill cumulative lengths disagree with requests")
        positions = attn.combo_position_ids
        if positions is not None and positions.numel():
            expected = torch.cat(
                [
                    torch.arange(p, p + n)
                    for p, n in zip(prefixes.tolist(), lengths.tolist())
                ]
            )
            if positions.dtype not in (torch.int32, torch.int64) or not torch.equal(
                positions.cpu(), expected
            ):
                raise ValueError("V4.1 prefill positions disagree with requests")

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
        if is_cuda_graph:
            raise NotImplementedError("V4.1 worker CUDA graphs are not integrated")
        return None

    @torch.inference_mode()
    def forward(self, inputs, fmha_impl: Any = None):
        with self._worker_lock:
            self._check_worker()
            if self._worker_state != "ready":
                raise RuntimeError("V4.1 worker is not initialized")
            try:
                if fmha_impl is not None or torch.cuda.is_current_stream_capturing():
                    raise NotImplementedError("V4.1 worker requires eager prefill")
                if (
                    torch.cuda.current_stream(self._weight_device())
                    != self._worker_stream
                ):
                    raise NotImplementedError(
                        "V4.1 worker requires its initialization CUDA stream"
                    )
                mm = inputs.multimodal_inputs
                if (
                    mm.multimodal_features
                    or mm.mm_extra_input
                    or (mm.mm_features_locs is not None and mm.mm_features_locs.numel())
                ):
                    raise NotImplementedError(
                        "V4.1 worker image inputs are not integrated"
                    )
                if bool(
                    (
                        inputs.input_ids
                        == self._v4_args.v41_config.get("image_token_id", 129264)
                    ).any()
                ):
                    raise NotImplementedError(
                        "V4.1 worker image tokens are not integrated"
                    )
                if self.kv_cache is not None:
                    ids = inputs.input_ids
                    if (
                        ids.ndim != 1
                        or ids.device != self._weight_device()
                        or not ids.numel()
                        or ids.dtype not in (torch.int32, torch.int64)
                        or bool(((ids < 0) | (ids >= self._v4_args.vocab_size)).any())
                    ):
                        raise ValueError(
                            "V4.1 prefill requires nonempty valid token IDs"
                        )
                    attn = primary_attention_inputs(
                        inputs.attention_inputs, self.kv_cache
                    )
                    if (
                        attn is None
                        or not attn.is_prefill
                        or attn.is_cuda_graph
                        or getattr(attn, "is_target_verify", False)
                    ):
                        raise NotImplementedError(
                            "V4.1 worker decode/verify is not integrated"
                        )
                    self._validate_prefill(inputs, attn)
                    self._prepare_engram(inputs)
                return super().forward(inputs, fmha_impl)
            finally:
                self._clear_forward_state()

    def close(self):
        with self._worker_lock:
            if os.getpid() != self._worker_pid:
                raise RuntimeError("V4.1 worker cannot close inherited CUDA resources")
            if self._worker_state == "closed":
                return
            self._worker_state = "closing"
            self._clear_forward_state()
            for layer_id, table in tuple(self._engram_tables.items()):
                # close drains CUDA and retains mappings/leases on failure.
                table.close()
                del self._engram_tables[layer_id]
            for layer, _ in self._engram_layers:
                layer.engram = None
            self._engram_layers = ()
            self._engram_hash_state = None
            self._worker_state = "closed"

    def __del__(self):
        if hasattr(self, "_worker_lock") and os.getpid() == self._worker_pid:
            try:
                self.close()
            except BaseException:
                # Individual table owners retain failed CUDA teardown safely.
                pass


__all__ = ["DeepSeekV41Model"]

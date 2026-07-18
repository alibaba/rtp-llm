from __future__ import annotations

import logging
import re
import threading
from typing import Mapping

import torch

from rtp_llm.config.sleep_mode_compatibility import reject_dynamic_weight_update
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.model_loader.weight_memory_saver import is_enabled as sleep_mode_enabled
from rtp_llm.model_loader.weight_memory_saver import (
    sleep_mode_level,
    suppress_weights_region,
)

# Assuming these imports are from your project and accessible
from rtp_llm.model_loader.weight_module import WeightModule

from .tipc import CudaIpcHelper, SharedMemIpcMeta, SharedMemoryIPCHelper

# Dictionary for renaming specific layer weight names from an external format
# (e.g., 'verl') to the internal 'rtp-llm' format.
RENAME_DICTIONARY = {
    # verl
    "embed_tokens.weight": "embedding",
    "norm.weight": "final_layernorm.gamma",
    "norm.bias": "final_layernorm.beta",
    "lm_head.weight": "lm_head",
    "input_layernorm.weight": "pre_layernorm_weights.gamma",
    "post_attention_layernorm.weight": "post_layernorm_weights.gamma",
    "self_attn.qkv_proj.weight": "self_attention_weights.query_weight.kernel",
    "self_attn.qkv_proj.bias": "self_attention_weights.query_weight.bias",
    "self_attn.o_proj.weight": "self_attention_weights.attention_output_weight.kernel",
    "mlp.gate_proj.weight": "ffn_weights.intermediate_weight.kernel",
    "mlp.up_proj.weight": "ffn_weights.intermediate_weight3.kernel",
    "mlp.down_proj.weight": "ffn_weights.intermediate_weight2.kernel",
    # roll - megatron
    "mbedding.word_embeddings.weight": "embedding",
    "self_attention.linear_proj.weight": "self_attention_weights.attention_output_weight.kernel",
    "self_attention.linear_proj.bias": "self_attention_weights.attention_output_weight.bias",
    "self_attention.linear_qkv.weight": "self_attention_weights.query_weight.kernel",
    "self_attention.linear_qkv.bias": "self_attention_weights.query_weight.bias",
    "mlp.linear_fc1.layer_norm_weight": "post_layernorm_weights.gamma",
    # ???
    "mlp.linear_fc1.weight": "",
}


def rename_function(layer_name: str) -> str:
    """
    Transforms a layer weight name from an external format (e.g., 'verl')
    into the format required by 'rtp-llm'.
    The input format is expected to be like 'model.layers.1.self_attn_qkv_proj.bias'.
    Args:
        layer_name: The layer weight name string from an external source.
    Returns:
        The transformed layer weight name in 'rtp-llm's internal format.
        For example, 'model.layers.1.self_attn_qkv_proj.bias' might become
        'self_attention_weights.query_weight.bias' if it matches a pattern
        and is in the RENAME_DICTIONARY.
    Error Handling:
        This function does not explicitly raise errors but performs string manipulations
        and dictionary lookups. If an unexpected `layer_name` format is provided,
        it might return a string that is not correctly transformed or recognized
        by downstream components.
    """
    # Remove the "model." prefix
    if layer_name.startswith("model."):
        name: str = layer_name[len("model.") :]
    elif layer_name.startswith("decoder."):
        name: str = layer_name[len("decoder.") :]
    else:
        name: str = layer_name
    if "layers" in layer_name:
        # Remove "layers." prefix
        name = name[len("layers.") :]
        # Remove the layer number and the dot following it (e.g., "1." from "1.self_attn...")
        # This assumes the format "layers.<number>.<rest_of_name>"
        first_dot_after_layers = name.find(".")
        if first_dot_after_layers != -1:
            name = name[first_dot_after_layers + 1 :]
        if name in RENAME_DICTIONARY:
            return RENAME_DICTIONARY[name]
        return name
    else:
        if name in RENAME_DICTIONARY:
            return RENAME_DICTIONARY[name]
        return name


class WeightManager:
    """
    Manages model weight updates, including renaming weights from an external
    source and handling inter-process communication (IPC) for tensor transfer.
    It ensures that incoming tensors are correctly processed and sharded/replicated
    as per the rtp-llm model's internal structure (e.g., for Tensor Parallelism (TP)
    or Pipeline Parallelism (PP)).
    """

    def __init__(
        self, device, weight: ModelWeights, model_weights_loader: ModelLoader
    ) -> None:
        """
        Initializes the WeightManager with a model's weights, device information, and weight loader.
        """
        self._s_helper = SharedMemoryIPCHelper()

        # Use the explicit device/weights/loader passed in by the caller (e.g. BaseModel),
        # instead of relying on any global "engine" object.
        if isinstance(device, torch.device):
            self._device = device
        else:
            self._device = torch.device(device)

        self._weights: ModelWeights = weight
        self._weights_loader: ModelLoader = model_weights_loader
        self._weight_module = self._weights_loader._model_weights_info
        self._working_stream: torch.cuda.Stream = torch.cuda.Stream(
            device=self._device,
        )
        # TODO: Consider the actual need for this lock. If updates are always
        # serialized via the server's request handling, a per-update lock might
        # be redundant or require finer-grained locking within _weights.update_...
        self._lock = threading.Lock()

    def extract_layer_number(self, s: str) -> int | None:
        """
        Extracts the layer number (an integer) from a string that follows
        the pattern 'layers.<number>'.
        Args:
            s: The input string, e.g., 'model.layers.2.mlp.gate_proj.weight'.
        Returns:
            The extracted layer number as an integer if found; otherwise, returns `None`.
        Error Handling:
            Returns `None` if the pattern 'layers.<number>' is not found,
            or if the captured group cannot be converted to an integer.
        """
        match = re.search(r"layers\.(\d+)", s)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        else:
            return None

    def update(self, req: dict[str, str]) -> None:
        """
        Receives an Inter-Process Communication (IPC) tensor description and
        updates the corresponding model weights.
        For models with Tensor Parallelism (TP) or Pipeline Parallelism (PP),
        this function expects the transmitted tensor to be a complete, unsharded tensor.
        It then handles the internal sharding or replication according to the
        rtp-llm's specific model parallelism configuration.
        Args:
            req: A dictionary containing the IPC request details. Expected keys are:
                 - "desc": A string describing the tensor's IPC metadata
                           (e.g., `CuIpcTensorMeta` or `SharedMemIpcMeta` encoded string).
                 - "name": A string representing the original name of the weight
                           (e.g., 'model.layers.1.self_attn_qkv_proj.bias').
                 - "method": A string indicating the IPC method used ("cuda_ipc" or "shm").
        Returns:
            None. The method updates internal model weights directly.
        Error Handling:
            - `KeyError`: If "desc", "name", or "method" fields are missing from `req`.
            - `ValueError`: If the "method" is invalid (not "cuda_ipc" or "shm"),
                            or if a layer weight name is invalid and its ID cannot be extracted.
            - `NotImplementedError`: If "cuda_ipc" method is attempted (currently disallowed).
            - `Exception`: If the tensor cannot be built from the IPC metadata (e.g., invalid descriptor).
                          This is a general catch-all for unexpected failures in `_t_helper.build_from_meta`.
        """
        # Level-2 wake reloads weights from the on-disk checkpoint, which would silently
        # revert any runtime weight update pushed here. Reject rather than lose the update.
        reject_dynamic_weight_update(
            enable_sleep_mode=sleep_mode_enabled(),
            sleep_mode_level=sleep_mode_level(),
        )
        if "desc" not in req:
            raise KeyError(
                "Update request is missing the 'desc' field. "
                "It must contain IPC tensor metadata."
            )
        if "name" not in req:
            raise KeyError(
                "Update request is missing the 'name' field. "
                "It must specify the weight name to update."
            )
        if "method" not in req:
            raise KeyError(
                "Update request is missing the 'method' field. "
                "It must specify the IPC method (e.g., 'cuda_ipc' or 'shm')."
            )
        method: str = req["method"]
        desc: str = req["desc"]
        name: str = req["name"]
        stored_name: str = name

        if method not in {"cuda_ipc", "shm"}:
            raise ValueError(
                f"Invalid IPC method '{method}' provided. Only 'cuda_ipc' and 'shm' are allowed."
            )
        tensor: torch.Tensor | None = None

        if method == "cuda_ipc":
            helper = CudaIpcHelper()
            tensor = helper.build_from_meta(bytes.fromhex(desc))
        else:  # method == "shm"
            sm_meta: SharedMemIpcMeta = SharedMemIpcMeta.decode(desc)
            tensor = self._s_helper.build_from_meta(sm_meta)

        if tensor is None:
            logging.error(
                f"Fail to build tensor from ipc description {desc}, method: {method}"
            )
            # This should ideally not be reached if build_from_meta consistently returns a tensor or raises an error.
            raise Exception(
                f"Failed to build tensor from IPC description '{desc}' using method '{method}'. Tensor is None."
            )

        logging.info(
            f"update weight request: {name}, shape: {tensor.shape}, device: {tensor.device}, dtype: {tensor.dtype}"
        )
        with torch.cuda.stream(self._working_stream):
            config = self._weights_loader.get_load_config()
            if "layers" in name:
                # This is a layer-specific weight
                layer_id: int | None = self.extract_layer_number(name)
                if layer_id is None:
                    raise ValueError(
                        f"Invalid layer weight name format: '{name}'. "
                        "Could not extract layer number. Expected format like 'model.layers.<id>...'"
                    )
                name: str = rename_function(name)
                fail: bool = True

                for receptor in self._weight_module.layer_weights[layer_id]:
                    if receptor.name == name or (
                        "ffn_weights" in name and receptor.name == "__ffn_weights__"
                    ):
                        assert isinstance(receptor, WeightModule)

                        # split tensor into shards
                        shard = receptor.update(
                            tensor=tensor,
                            device=self._device,
                            load_config=config,
                            module_name=name,
                        )
                        if isinstance(shard, dict):
                            shard = next(iter(shard.values()))

                        # update tensor weight
                        self._weights.update_layer_weight(
                            layer_id=layer_id, name=name, data=shard
                        )
                        fail = False

                if fail:
                    raise KeyError(
                        f"{stored_name} not found. wanted name list is {[w.name for w in self._weight_module.layer_weights[layer_id]]}"
                    )

            else:
                # weight is global weight

                name: str = rename_function(name)
                fail: bool = True
                for weight in self._weight_module.weights:
                    if weight.name == name:
                        shard: dict = weight.update(
                            tensor,
                            self._device,
                            load_config=self._weights_loader.get_load_config(),
                        )
                        if isinstance(shard, dict):
                            shard = next(iter(shard.values()))
                        self._weights.update_global_weight(name=name, data=shard)
                        fail = False

                if fail:
                    raise KeyError(
                        f"{stored_name} not found. wanted name list is {[w.name for w in self._weight_module.weights]}"
                    )

            self._working_stream.synchronize()

    # ------------------------------------------------------------------
    # Sleep level 2 (discard weights): in-place reload from the model loader.
    #
    # In level-2 sleep the weights region is opened without torch_memory_saver
    # host cpu_backup, so ``pause("weights")`` frees GPU *and* host memory and
    # ``resume("weights")`` remaps blank pages at the same virtual address. Sleep
    # itself writes nothing — there is no on-disk backup. To bring the *same*
    # weights back on wake, the C++ wake hook calls
    # :meth:`reload_weights_from_loader` after resume.
    #
    # The reload re-runs the loader's per-tensor pipeline
    # (:meth:`ModelLoader.prepare_weights`) from the original checkpoint. It
    # yields already-processed tensors (post dequant / MoE fusion / TP split),
    # matching the live layout exactly, and each is ``copy_``-ed in place into
    # the existing GPU storage — preserving every tensor's ``data_ptr`` (aliased
    # by the C++ engine and baked into captured CUDA graphs). ``prepare_weights``
    # is a generator, so only a bounded amount of processed weight is
    # materialized at a time (no 2x-weights GPU peak) and nothing hits disk.
    # Scope: the base ``ModelWeights`` only; LoRA adapters, multimodal ViT, and
    # C++-side dynamic EPLB expert buffers are out of scope for v1 (see
    # weight_memory_saver.py coverage checklist).
    # ------------------------------------------------------------------

    def _live_weight_keys(self) -> set[tuple[int | None, str]]:
        """Every (layer_id, name) tracked in the live ModelWeights.

        layer_id is None for global weights. Used by
        :meth:`reload_weights_from_loader` to assert full coverage.
        """
        keys: set[tuple[int | None, str]] = set()
        for layer_id, layer_dict in enumerate(self._weights.weights):
            for name in layer_dict:
                keys.add((layer_id, name))
        for name in self._weights.global_weights:
            keys.add((None, name))
        return keys

    def reload_weights_from_loader(self) -> None:
        """Reload weights in place from the model loader (level-2 wake).

        Called after ``resume("weights")`` has remapped blank pages at the
        original VA. Streams the loader's processed tensors from the original
        checkpoint and ``copy_`` s each into the matching live GPU tensor,
        preserving ``data_ptr`` so C++ aliases and captured CUDA graphs stay
        valid. Raises on any shape/dtype mismatch or if some live weight is
        never covered, so a failed reload propagates to the caller (C++ hook ->
        ERROR) instead of leaving blank-page garbage behind.

        Source selection: prefer the fast bulk fastsafetensors path
        (:meth:`ModelLoader.prepare_weights_fastsafetensor`) — for a large MoE
        the per-tensor scratch path is too slow (30B took >10min and exceeded
        the wake timeout). Both generators yield the identical
        ``(layer_id, name, tensor)`` set (same weight modules), so switching the
        source changes only load speed, not coverage/correctness. Fall back to
        the load-from-scratch per-tensor path when fastsafetensors is
        unavailable or the checkpoint is not fast-loadable.
        """
        device = str(self._device)
        pending = self._live_weight_keys()
        expected = len(pending)
        restored = 0
        seen = 0
        # Baseline: driver-free right now = after weights region is restored
        # (resident weights + base/context) but BEFORE any reload transient is
        # allocated. Diff vs the post-teardown free tells us the true stuck
        # transient and how close base+weights already is to total capacity.
        with torch.cuda.device(self._device):
            _free_baseline, _total_baseline = torch.cuda.mem_get_info(self._device)
        logging.info(
            "reload_weights_from_loader: baseline driver-free %.0f MiB "
            "(total %.0f MiB) = base+weights, pre-reload-transient",
            _free_baseline / (1024.0**2),
            _total_baseline / (1024.0**2),
        )
        if self._weights_loader.can_reload_from_fastsafetensor():
            # in_weights_region=False: the reloaded tensors are transient copy_
            # sources, not the resident weights (those already occupy their fixed
            # VA). Keeping them OUT of the torch_memory_saver weights region lets
            # the end-of-reload empty_cache return every transient shard/dequant
            # buffer to the driver, so the following KV-cache resume has full
            # headroom (region-scoped transients were stuck and OOM'd cu_mem_create).
            source = self._weights_loader.prepare_weights_fastsafetensor(
                device, in_weights_region=False
            )
            method = "fastsafetensors"
        else:
            source = self._weights_loader.prepare_weights(device)
            method = "scratch"
        logging.info("reload_weights_from_loader: reloading via %s path", method)

        # suppress_weights_region: the resident weights already occupy their VA;
        # keep every reload transient (scratch WeightModule.load intermediates and
        # the fastsafetensors shard/split buffers) OUT of the torch_memory_saver
        # weights region so empty_cache can return them to the driver. Without this
        # the scratch path commits per-tensor intermediates as region-backed pages
        # that stick (growing with weight count) and OOM the KV-cache resume -- the
        # same failure prepare_weights_fastsafetensor(in_weights_region=False) fixes
        # for the fast path.
        with self._lock, suppress_weights_region():
            with torch.cuda.stream(self._working_stream), torch.inference_mode():
                for layer_id, name, tensor in source:
                    seen += 1
                    if layer_id is not None:
                        ori = (
                            self._weights.weights[layer_id].get(name)
                            if 0 <= layer_id < len(self._weights.weights)
                            else None
                        )
                    else:
                        ori = self._weights.global_weights.get(name)
                    if ori is None:
                        # The loader can yield tensors not tracked in the live
                        # ModelWeights (e.g. misc weights); nothing to reload in
                        # place for those. Live-weight coverage is asserted below.
                        logging.debug(
                            "reload_weights_from_loader: loader tensor "
                            "(layer=%s, name=%s) has no live tensor, skip",
                            layer_id,
                            name,
                        )
                        del tensor
                        continue
                    if ori.shape != tensor.shape or ori.dtype != tensor.dtype:
                        raise ValueError(
                            f"reload_weights_from_loader: mismatch for {name}: live "
                            f"{tuple(ori.shape)}/{ori.dtype} vs loader "
                            f"{tuple(tensor.shape)}/{tensor.dtype}"
                        )
                    # In-place copy into the existing storage: data_ptr is
                    # preserved so C++ aliases and captured graphs stay valid.
                    ori.copy_(tensor)
                    pending.discard((layer_id, name))
                    del tensor
                    restored += 1
                self._working_stream.synchronize()
            # Reclaim the reload transients. Every yielded copy_ source is dropped
            # inside the loop and the loader frees its shard / dequant / split
            # intermediates as it goes, so torch's *allocated* bytes are already
            # ~0 here; what remains is caching-allocator segments. Drop the
            # generator (so its frame + the loader objects are finalized), force a
            # GC pass, sync, then return every 100%-free segment to the driver in a
            # single empty_cache so the following KV-cache resume has headroom.
            # The residual-vs-baseline number below measures whether any transient
            # is still stuck (co-tenanted with a resident engine segment); if it is
            # large on a big FP8 MoE we revisit isolating the reload allocations.
            import gc

            mib = 1024.0**2
            with torch.cuda.device(self._device):
                free_before = torch.cuda.mem_get_info(self._device)[0]
                resv_before = torch.cuda.memory_reserved(self._device)
                alloc_before = torch.cuda.memory_allocated(self._device)
                del source
                gc.collect()
                torch.cuda.synchronize()
                # Best-effort transient reclaim. On a decode role with captured CUDA
                # graphs, empty_cache() walks the graph-private MemPool whose blocks are
                # torch_memory_saver VMM-backed; its release path issues a
                # cuMemUnmap/cudaFree that returns "CUDA error: invalid argument" (same
                # MemPool-under-TMS failure the sleep-side release swallows -- reproduced
                # even with the cuda_graph region resumed, so it is the pool walk itself).
                # The weight copy_ above is already done, so this reclaim is optional: the
                # transient stays in torch's caching pool but is bounded (in_weights_region
                # =False keeps peak per-tensor, not per-model), and the KV cache is sized
                # with runtime-reserve headroom that absorbs it. Swallow the failure (the
                # error is a non-sticky runtime-API return, drained by the c10 check that
                # raised it) so it cannot abort an otherwise-successful wake.
                try:
                    torch.cuda.empty_cache()
                except Exception as e:  # noqa: BLE001 - best-effort teardown
                    logging.warning(
                        "reload_weights_from_loader: best-effort empty_cache() failed "
                        "(%s); continuing, weights already reloaded in place",
                        e,
                    )
                free_after, total = torch.cuda.mem_get_info(self._device)
                resv_after = torch.cuda.memory_reserved(self._device)
                alloc_after = torch.cuda.memory_allocated(self._device)
            logging.info(
                "reload_weights_from_loader: teardown driver-free %.0f -> %.0f MiB "
                "(empty_cache reclaimed %.0f); residual vs baseline %.0f MiB "
                "(total %.0f)",
                free_before / mib,
                free_after / mib,
                (free_after - free_before) / mib,
                (_free_baseline - free_after) / mib,
                total / mib,
            )
            # Attribute the residual. NOTE: under torch_memory_saver the caching
            # allocator's reserved/allocated bookkeeping is decoupled from the
            # physical pages (TMS unmaps/remaps below torch's view), so only the
            # driver-free number above is physically truthful and reserved is not
            # comparable to it. What IS interpretable is the *allocated* delta:
            # it drops by ~the reload transient, confirming del/gc freed every
            # copy_ source back into torch's pool. The driver-free staying flat
            # then means those freed blocks sit in segments co-tenanted with
            # resident engine allocations, so no segment is 100% free and
            # empty_cache cannot hand them back. That residual is bounded (each
            # tensor is del'd before the next, so peak simultaneous transient is
            # tiny) and independent of model size — unlike the old in-region path
            # whose intermediates accumulated multi-GB and OOM'd.
            logging.info(
                "reload_weights_from_loader: torch allocated %.0f -> %.0f MiB "
                "(freed %.0f transient); reserved %.0f MiB (TMS-decoupled, not "
                "vs driver-free)",
                alloc_before / mib,
                alloc_after / mib,
                (alloc_before - alloc_after) / mib,
                resv_after / mib,
            )
            # Force-flush: the KV-cache resume that follows can OOM and abort the
            # process before buffered logs reach disk; keep the teardown numbers.
            try:
                import sys as _sys

                for _h in logging.getLogger().handlers:
                    _h.flush()
                _sys.stdout.flush()
                _sys.stderr.flush()
            except Exception:
                pass
        if pending:
            sample = sorted(str(k) for k in pending)[:10]
            raise RuntimeError(
                f"reload_weights_from_loader: {len(pending)} of {expected} live "
                f"weights were not reloaded (would remain blank pages), e.g. {sample}"
            )
        logging.info(
            "reload_weights_from_loader: reloaded %d/%d live tensors in place "
            "from checkpoint via %s path (%d loader tensors seen)",
            restored,
            expected,
            method,
            seen,
        )

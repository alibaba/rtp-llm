from __future__ import annotations

import logging
import re
import threading
from typing import Any, Mapping, Sequence

import torch

from rtp_llm.config.sleep_mode_compatibility import reject_dynamic_weight_update
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.model_loader.weight_memory_saver import expandable_segments_disabled
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
        self,
        device,
        weight: ModelWeights,
        model_weights_loader: ModelLoader,
        non_owned_global_weights: Sequence[str] = (),
        model_scope=None,
    ) -> None:
        """
        Initializes the WeightManager with a model's weights, device information, and weight loader.

        model_scope: token identifying the owning model (``id(base_model)``). Used
        by :meth:`reload_weights_from_loader` to filter the process-global DSV4
        Mega-MoE / compressor registries down to this model's own instances, so a
        coexisting checkpoint-backed propose/draft model (which registers into the
        same registries and collides on layer ids) does not get re-derived here.
        """
        self._s_helper = SharedMemoryIPCHelper()
        self._model_scope = model_scope
        # Additional WeightManagers whose weights must also be reloaded during this
        # manager's level-2 wake reload (e.g. a checkpoint-backed MTP draft model).
        # The C++ wake hook only calls reload on the main model's manager, so the
        # main manager fans out to these after its own reload. See
        # :meth:`register_chained_reload`.
        self._chained_reload: list["WeightManager"] = []

        # Use the explicit device/weights/loader passed in by the caller (e.g. BaseModel),
        # instead of relying on any global "engine" object.
        if isinstance(device, torch.device):
            self._device = device
        else:
            self._device = torch.device(device)

        self._weights: ModelWeights = weight
        self._weights_loader: ModelLoader = model_weights_loader
        self._weight_module = self._weights_loader._model_weights_info
        self._non_owned_global_weights = frozenset(non_owned_global_weights)
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
                if name in self._non_owned_global_weights:
                    raise PermissionError(
                        f"global weight {name!r} is a non-owning alias; update its owner instead"
                    )
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

    def release_runtime_gpu_caches(self, reason: str = "sleep") -> None:
        """Sleep-hook entry: hand freeable GPU memory back to the driver.

        Called from the C++ sleep release hook (after the weights/cuda_graph VMM
        pauses and KV release) via the ``weight_manager`` py::object seam. Drops
        long-lived Python-held device caches, then empties the torch caching
        allocator so segments they co-tenanted are returned to the driver, and logs
        a ``[SleepReclaim]`` segment breakdown attributing any physical residual.
        Best-effort; never raises into the hook.
        """
        from rtp_llm.utils.sleep_gpu_reclaim import release_and_trim

        release_and_trim(self._device, reason=reason)

    def nccl_memory_status(self) -> str:
        """Why NCCL failed the current sleep/wake transition, or ``""``.

        Read-only and lock-free; touches neither CUDA nor NCCL. Its only caller is
        a C++ hook that is already reporting a failure, so it must not become the
        second failure: :func:`~rtp_llm.utils.nccl_memory.status_text` reads one
        module global and formats it, and the hook catches a throw regardless.

        The empty string is the normal answer and is load-bearing. The hook this
        feeds appends the result to ``SleepStatus.last_error`` whenever it fails,
        and it fails for several reasons that have nothing to do with NCCL (the
        cuda_graph/weights VMM pause, the level-2 weight reload). Returning
        anything non-empty on those paths would name a healthy subsystem in the
        operator-visible error, so the C++ side drops an empty detail and keeps its
        own message.
        """
        from rtp_llm.utils.nccl_memory import status_text

        return status_text()

    # This method and :meth:`resume_collectives_for_wake` are hosted here only
    # because ``LocalRpcServer`` already holds a ``py::object weight_manager_`` and
    # there is no other Python handle on the C++ sleep-hook seam. Neither touches
    # weight state, so do not infer that collectives are a weight-manager concern.
    def suspend_collectives_for_sleep(self, reason: str = "sleep") -> None:
        """Sleep-hook entry: release NCCL communicator GPU memory.

        A separate seam from :meth:`release_runtime_gpu_caches` rather than a step
        inside it, because the two have opposite failure semantics: that one is
        best-effort, whereas a failure here leaves memory the wake path will
        dereference, so it must propagate and put the instance in ERROR.

        No-op unless ``--sleep_release_collective_memory`` is on, and no-op on a
        runtime NCCL without the suspend API.
        """
        from rtp_llm.model_loader.weight_memory_saver import release_collective_memory

        if not release_collective_memory():
            return
        from rtp_llm.utils.nccl_memory import suspend_for_sleep

        suspend_for_sleep(self._device, reason=reason)

    def resume_collectives_for_wake(self, reason: str = "wake") -> None:
        """Wake-hook entry: remap NCCL communicator memory. Must run FIRST.

        Deliberately not gated on the config switch: what has to be undone is
        whatever was actually suspended, and :func:`resume_after_wake` no-ops when
        nothing was. Reading the switch here would turn a mid-sleep config change
        into a communicator left unmapped.

        Ordering is load-bearing, which is why this is not a step inside
        :meth:`restore_runtime_gpu_caches` -- see
        :mod:`rtp_llm.utils.nccl_memory` rule (7).
        """
        from rtp_llm.utils.nccl_memory import resume_after_wake

        resume_after_wake(self._device, reason=reason)

    def restore_runtime_gpu_caches(self, reason: str = "wake") -> None:
        """Rebuild inexpensive Python-owned caches explicitly dropped at sleep."""
        # The TP symmetric-memory staging buffer is deliberately dropped while
        # sleeping (the process group remains valid). Recreate it before the
        # engine warmup so the normal TP fast path is restored on wake.
        try:
            from rtp_llm.models_py.distributed import collective_torch
            from rtp_llm.models_py.distributed.collective_torch import Group
            from rtp_llm.models_py.distributed.symm_mem import (
                restore_symm_mem_communicator_after_wake,
            )

            parallelism_config = getattr(collective_torch, "_parallelism_config", None)
            if parallelism_config is None or parallelism_config.tp_size <= 1:
                raise RuntimeError(
                    "TP symmetric-memory communicator is disabled for TP<=1"
                )
            tp_group = collective_torch._get_group(Group.TP)
            restored_symm = restore_symm_mem_communicator_after_wake(tp_group)
            logging.info(
                "restore_runtime_gpu_caches[%s]: TP symmetric-memory communicator restored=%s",
                reason,
                restored_symm,
            )
        except Exception as e:
            # Some deployments have TP=1 or no torch.distributed process group;
            # those paths never had a symmetric-memory communicator to restore.
            logging.info(
                "restore_runtime_gpu_caches[%s]: TP symmetric-memory restore skipped: %s",
                reason,
                e,
            )
        from rtp_llm.models_py.modules.dsv4.fp8.attention import (
            restore_rope_caches_after_wake,
        )

        restored = restore_rope_caches_after_wake()
        logging.info(
            "restore_runtime_gpu_caches[%s]: restored DSV4 RoPE caches for %d owner(s)",
            reason,
            restored,
        )

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
            #
            # Keep the default SHM copier on wake. Earlier force_nogds handling
            # targeted a suspected copier failure, but repeated-cycle profiling
            # localized the slowdown to the downstream Mega pageable-D2H stash;
            # both copier paths feed the same shuffle/transform pipeline.
            source = self._weights_loader.prepare_weights_fastsafetensor(
                device, in_weights_region=False
            )
            method = "fastsafetensors"
        else:
            source = self._weights_loader.prepare_weights(device)
            method = "scratch"
        # Both checkpoint-replay paths omit computed dynamic weights (e.g.
        # rotary_embedding.cos_sin_cache), which are generated at cold load by
        # ModelLoader._load_dynamic_weights, not read from the checkpoint. Chain
        # the dynamic-weight regenerator so the wake reload covers them too;
        # otherwise the coverage assertion below fires on those blank pages.
        import itertools

        source = itertools.chain(
            source, self._weights_loader.prepare_dynamic_weights(device)
        )
        logging.info("reload_weights_from_loader: reloading via %s path", method)

        # Level-2 wake also blanked py-model *computed* weights that are neither
        # streamed from the checkpoint nor reloadable in place: the DSV4 Mega-MoE
        # kernel weights (derived from raw routed weights that are POPPED at cold
        # load -> not in ModelWeights) and each attention compressor's fused
        # wkv/wgate. Reach the live modules via their registries and re-derive
        # them after the stream loop (see _rederive_dsv4_computed_weights). The
        # raw routed weights are re-streamed but hit the "no live tensor" branch
        # below, so capture them per MoE layer here. Guarded: non-DSV4 models have
        # empty/absent registries and are unaffected.
        mega_by_layer: dict = {}
        mega_routed_keys: set = set()
        mega_acc: dict = {}
        mega_done: set = set()
        mega_active_gpu_layer = None
        mega_peak_layers = 0
        mega_peak_gpu_bytes = 0
        mega_peak_host_bytes = 0
        compressors: list = []
        try:
            from rtp_llm.models_py.modules.dsv4.fp8.compressor import iter_compressors
            from rtp_llm.models_py.modules.dsv4.moe.mega_buf import iter_mega_strategies
            from rtp_llm.utils.model_weight import W

            mega_routed_keys = {
                W.v4_routed_w1_w,
                W.v4_routed_w1_s,
                W.v4_routed_w2_w,
                W.v4_routed_w2_s,
                W.v4_routed_w3_w,
                W.v4_routed_w3_s,
            }

            # Filter the process-global registries to THIS model's own instances.
            # A coexisting checkpoint-backed propose/draft model (e.g. DSV4 MTP)
            # registers its strategies/compressors into the same registries and its
            # lone draft layer collides on layer_id=0 with the main model's layer 0.
            # Each instance is stamped with its owning model's build scope at
            # registration; match it against this manager's scope so the main reload
            # never grabs the draft's layer-0 strategy (and vice versa). When scope
            # is None on both sides (non-sleep / untagged builds) the match is a
            # no-op that preserves the original all-instances behavior.
            def _owned(obj) -> bool:
                return getattr(obj, "_sleep_model_scope", None) == self._model_scope

            for strat in iter_mega_strategies():
                if _owned(strat):
                    mega_by_layer[strat.cfg.layer_id] = strat
            compressors = [c for c in iter_compressors() if _owned(c)]
        except Exception:
            logging.info(
                "reload_weights_from_loader: DSV4 computed-weight registries "
                "unavailable; skipping mega/compressor re-derivation",
                exc_info=True,
            )

        # suppress_weights_region: the resident weights already occupy their VA;
        # keep every reload transient (scratch WeightModule.load intermediates and
        # the fastsafetensors shard/split buffers) OUT of the torch_memory_saver
        # weights region so empty_cache can return them to the driver. Without this
        # the scratch path commits per-tensor intermediates as region-backed pages
        # that stick (growing with weight count) and OOM the KV-cache resume -- the
        # same failure prepare_weights_fastsafetensor(in_weights_region=False) fixes
        # for the fast path.
        # expandable_segments_disabled: the reload streams checkpoint tensors as
        # copy_ SOURCES (fastsafetensors shards / dequant / TP-split intermediates)
        # and copies them into the resident weights. suppress_weights_region keeps
        # those transients out of the torch_memory_saver weights region, but with
        # runtime expandable_segments enabled they would otherwise land in torch
        # expandable segments -- and a tensor read/written across the tms
        # pause/resume boundary comes back CORRUPTED (silent: coverage stays
        # correct, values are wrong -> garbage output after wake). Force the whole
        # reload non-expandable so the copy_ sources are plain, driver-backed
        # allocations, matching the verified-correct expandable-off wake. No-op
        # unless expandable coexistence is active.
        with self._lock, suppress_weights_region(), expandable_segments_disabled():
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
                        # DSV4 Mega-MoE raw routed weights are popped at cold load
                        # (their transform outputs are the resident kernel weights),
                        # so they have no live ModelWeights tensor to copy into.
                        # Keep each raw ON GPU and re-derive the layer's Mega
                        # kernel weights as soon as all six routed keys have
                        # arrived, mirroring the cold-load inline transform. This
                        # removes the pageable D2H stash that degrades on fresh
                        # default-pool allocations on coherent arm64 platforms.
                        # The current DSV4 checkpoint stream completes one layer
                        # at a time, so the normal path needs no host staging. The
                        # iterator does not, however, promise layer ordering: if a
                        # different layer arrives before the active one completes,
                        # spill the incomplete layer to temporary pinned host
                        # buffers. This keeps GPU residency bounded without paying
                        # the old pageable-D2H penalty or retaining the pinned
                        # buffers across reloads.
                        #
                        # reload_routed_weights runs the wake transform in the
                        # default pool and copies the results IN PLACE into the
                        # existing tagged buffers, so calling it mid-loop does
                        # not replace resident weight allocations.
                        if layer_id in mega_by_layer and name in mega_routed_keys:
                            if (
                                mega_active_gpu_layer is not None
                                and mega_active_gpu_layer != layer_id
                            ):
                                previous = mega_acc.get(mega_active_gpu_layer, {})
                                for previous_name, previous_tensor in list(
                                    previous.items()
                                ):
                                    if previous_tensor.is_cuda:
                                        host_tensor = torch.empty(
                                            previous_tensor.shape,
                                            dtype=previous_tensor.dtype,
                                            device="cpu",
                                            pin_memory=True,
                                        )
                                        host_tensor.copy_(previous_tensor)
                                        previous[previous_name] = host_tensor
                                self._working_stream.synchronize()
                            mega_active_gpu_layer = layer_id
                            acc = mega_acc.setdefault(layer_id, {})
                            if name in acc:
                                raise RuntimeError(
                                    "reload_weights_from_loader: duplicate Mega raw "
                                    f"weight for layer {layer_id}: {name}"
                                )
                            acc[name] = tensor
                            del tensor
                            if set(acc) == mega_routed_keys:
                                mega_acc.pop(layer_id, None)
                                gpu_acc = {
                                    key: (
                                        value
                                        if value.is_cuda
                                        else value.to(device, non_blocking=False)
                                    )
                                    for key, value in acc.items()
                                }
                                del acc
                                mega_by_layer[layer_id].reload_routed_weights(gpu_acc)
                                del gpu_acc
                                self._working_stream.synchronize()
                                mega_done.add(layer_id)
                                mega_active_gpu_layer = None
                            pending_gpu_bytes = sum(
                                t.nelement() * t.element_size()
                                for values in mega_acc.values()
                                for t in values.values()
                                if t.is_cuda
                            )
                            pending_host_bytes = sum(
                                t.nelement() * t.element_size()
                                for values in mega_acc.values()
                                for t in values.values()
                                if not t.is_cuda
                            )
                            mega_peak_layers = max(mega_peak_layers, len(mega_acc))
                            mega_peak_gpu_bytes = max(
                                mega_peak_gpu_bytes, pending_gpu_bytes
                            )
                            mega_peak_host_bytes = max(
                                mega_peak_host_bytes, pending_host_bytes
                            )
                            continue
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
        # Re-derive DSV4 computed weights blanked by the level-2 resume. Most Mega
        # layers were already re-derived in-loop when their six raws completed;
        # only incomplete layers plus the compressors remain. This runs outside
        # suppress_weights_region so fresh buffers are tagged as weights for the
        # next sleep, and before KV-cache resume while full headroom is available.
        if mega_by_layer:
            logging.info(
                "reload_weights_from_loader: in-loop mega rederive %d/%d layers, "
                "peak pending %d layer(s) / %.0f MiB GPU raws / "
                "%.0f MiB pinned fallback",
                len(mega_done),
                len(mega_by_layer),
                mega_peak_layers,
                mega_peak_gpu_bytes / (1024.0**2),
                mega_peak_host_bytes / (1024.0**2),
            )
        mega_remaining = {
            layer_id: strategy
            for layer_id, strategy in mega_by_layer.items()
            if layer_id not in mega_done
        }
        # Keep the fallback transform non-expandable for the same TMS corruption
        # reason as the reload loop above.
        with expandable_segments_disabled():
            self._rederive_dsv4_computed_weights(
                mega_remaining, mega_acc, mega_routed_keys, compressors
            )
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
        # Fan out to chained managers (e.g. a checkpoint-backed MTP draft model).
        # The C++ level-2 wake hook only calls reload on the main model's manager;
        # the draft model's GPU weights are also blank-remapped by resume("weights")
        # and must be reloaded before the engine loop restarts. Runs after the main
        # reload (transients already reclaimed) so each draft reload streams into a
        # fresh headroom window. A chained failure propagates (draft weights left
        # blank would silently corrupt speculative decoding).
        for mgr in self._chained_reload:
            logging.info(
                "reload_weights_from_loader: reloading chained model weights "
                "(scope=%s)",
                getattr(mgr, "_model_scope", None),
            )
            mgr.reload_weights_from_loader()

    def register_chained_reload(self, other: "WeightManager") -> None:
        """Register another WeightManager to reload during this manager's level-2 wake.

        Used to attach a checkpoint-backed propose/draft model's manager to the
        main model's manager: the C++ wake hook only invokes
        :meth:`reload_weights_from_loader` on the main manager, so the draft
        model's blank-remapped GPU weights would otherwise never be restored.
        Idempotent; ignores self / duplicate registrations."""
        if other is None or other is self or other in self._chained_reload:
            return
        self._chained_reload.append(other)

    def _rederive_dsv4_computed_weights(
        self, mega_by_layer, mega_acc, mega_routed_keys, compressors
    ) -> None:
        """Rebuild DSV4 py-model computed weights blanked by the level-2 resume.

        Two classes of resident weight are ``weights``-tagged (blank-remapped by
        ``resume("weights")``) yet not reloadable in place from the checkpoint:

        * Mega-MoE kernel weights (``_mega_l1_w`` / ``_l1_sf`` / ``_l2_w`` /
          ``_l2_sf``) — derived by ``transform_weights_for_mega_moe`` from the raw
          routed stacks, which are popped at cold load (so absent from
          ModelWeights). Re-derived per MoE layer from the raws captured during the
          reload stream (``mega_acc``).
        * Each attention compressor's fused wkv/wgate — a ``cat`` of the raw
          wkv/wgate, which stay in ModelWeights and were reloaded in place, so the
          compressor rebuilds ``_wkv_wgate_fused`` from its retained raw refs.

        Runs with the weights_region NOT suppressed so the fresh buffers are
        re-tagged for the next sleep, and before the KV-cache resume so headroom is
        ample for the held raws + transform transients. Raises if any registered
        consumer was not fully covered (surfaces to the C++ wake hook as an ERROR
        rather than leaving blank-page garbage)."""
        if not mega_by_layer and not compressors:
            return
        import gc

        device = self._device
        mib = 1024.0**2
        # Per-phase memory probe is diagnostic only (used to root-cause the
        # private-weights-MemPool fragmentation OOM); off by default to keep the
        # wake path quiet. Set RTP_LLM_SLEEP_MEM_DEBUG=1 to re-enable.
        import os as _os

        _probe_enabled = _os.environ.get("RTP_LLM_SLEEP_MEM_DEBUG", "0") == "1"

        def _rd_probe(tag: str) -> None:
            # High-signal per-phase memory probe for the L2 wake re-derive. Logs
            # physical driver-free (truthful) alongside torch's caching-allocator
            # reserved/allocated so we can attribute any residual to the private
            # weights MemPool (reserved>>allocated, not returnable) vs the default
            # pool (returnable). Best-effort; never aborts the wake.
            if not _probe_enabled:
                return
            try:
                free_b, total_b = torch.cuda.mem_get_info(device)
                reserved = torch.cuda.memory_reserved(device)
                allocated = torch.cuda.memory_allocated(device)
                logging.info(
                    "[L2Rederive][%s] driver phys used %.0f MiB (free %.0f) | "
                    "torch reserved %.0f MiB allocated %.0f MiB (reserved-alloc "
                    "gap %.0f MiB)",
                    tag,
                    (total_b - free_b) / mib,
                    free_b / mib,
                    reserved / mib,
                    allocated / mib,
                    (reserved - allocated) / mib,
                )
            except Exception:  # noqa: BLE001 - probe must never break wake
                pass

        _rd_probe("start")
        with self._lock:
            with torch.cuda.stream(self._working_stream), torch.inference_mode():
                for layer_id, strat in mega_by_layer.items():
                    acc = mega_acc.get(layer_id)
                    have = set(acc) if acc else set()
                    if have != mega_routed_keys:
                        missing = sorted(str(k) for k in (mega_routed_keys - have))
                        raise RuntimeError(
                            f"reload_weights_from_loader: mega layer {layer_id} "
                            f"missing routed weights for re-transform: {missing}"
                        )
                    # Bring THIS layer's host-staged raws back to GPU just before
                    # transforming, so at most one layer of raws is GPU-resident.
                    # reload_routed_weights recomputes the kernel weights in the
                    # DEFAULT pool and copies them IN PLACE into the existing
                    # blank-remapped tagged buffers (no free + realloc of the
                    # private weights MemPool, which would fragment and leak ~35 GiB
                    # of unreturnable reserved segments), so the only fresh GPU
                    # demand is this one layer's raws + transform scratch in the
                    # returnable default pool -- well within headroom. Drop the host
                    # + GPU refs immediately so neither accumulates across layers.
                    gpu_acc = {
                        k: v.to(device, non_blocking=False) for k, v in acc.items()
                    }
                    mega_acc.pop(layer_id, None)
                    del acc
                    strat.reload_routed_weights(gpu_acc)
                    del gpu_acc
                    self._working_stream.synchronize()
                    # Return this layer's default-pool transients (raws + transform
                    # scratch) to the driver before the next layer. Guarded: under
                    # torch_memory_saver the private weights MemPool walk can raise
                    # (invalid-argument), and default-pool blocks are reused across
                    # layers anyway, so a failure must not abort the wake.
                    try:
                        torch.cuda.empty_cache()
                    except Exception:  # noqa: BLE001 - best-effort per-layer reclaim
                        pass
                self._working_stream.synchronize()
                _rd_probe("after_mega")
                for cmp in compressors:
                    cmp.reload_fused_weights()
                self._working_stream.synchronize()
                _rd_probe("after_compressors")
            mega_acc.clear()
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001 - best-effort transient reclaim
                logging.warning(
                    "reload_weights_from_loader: re-derive empty_cache() failed "
                    "(%s); continuing, computed weights already rebuilt",
                    e,
                )
            _rd_probe("after_reclaim")
        logging.info(
            "reload_weights_from_loader: re-derived level-2 computed weights "
            "(%d Mega-MoE layer(s) + %d compressor(s))",
            len(mega_by_layer),
            len(compressors),
        )

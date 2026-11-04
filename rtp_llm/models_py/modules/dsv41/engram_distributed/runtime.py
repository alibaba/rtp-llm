# SPDX-License-Identifier: Apache-2.0
"""Model attachment, table sources and process-lifetime storage ownership."""
from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .checkpoint import hash_state_digest
from .config import (
    EXTERNAL_GRAPH_MODE_ENV,
    HARD_MAX_ALLOCATION_BYTES,
    HARD_MAX_TOTAL_ALLOCATION_BYTES,
    MAX_SHARDS,
    EngramOptions,
    StoragePolicy,
)
from .manifest import (
    ExpectedLayout,
    Limits,
    ManifestError,
    ParsedManifest,
    bind_manifest,
)
from .storage import (
    BootstrapError,
    CleanupPendingError,
    FabricEngramStorage,
    _load_native,
)

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


# A consumer imports the entire cohort. Publisher budgets constrain physical
# allocations on each host and are checked before any allocation at startup.
_MAPPING_LIMITS = Limits(
    max_tensors=1024,
    max_allocation_bytes=HARD_MAX_ALLOCATION_BYTES,
    max_total_allocation_bytes=HARD_MAX_TOTAL_ALLOCATION_BYTES * MAX_SHARDS,
)


def _validate_mapping_capacity(validated, shard_count):
    if validated.physical_bytes > HARD_MAX_TOTAL_ALLOCATION_BYTES * shard_count:
        raise ManifestError("TOTAL_ALLOCATION_LIMIT", "tensors")
    allocations = {
        part.allocation_id: part
        for tensor in validated.manifest.tensors
        for part in tensor.shards
    }
    hosts = {}
    for part in allocations.values():
        hosts[part.host_id] = hosts.get(part.host_id, 0) + part.allocation_bytes
    if any(total > HARD_MAX_TOTAL_ALLOCATION_BYTES for total in hosts.values()):
        raise ManifestError("OWNER_ALLOCATION_LIMIT", "tensors")


def _digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _verify_samples(views, validated, samples, device):
    import torch

    for tensor in validated.manifest.tensors:
        sample = samples[tensor.name]
        indices = torch.tensor(sample["row_ids"], dtype=torch.int64, device=device)
        selected = views[tensor.name].index_select(0, indices).cpu()
        if _digest(selected.numpy().tobytes()) != sample["sha256"]:
            raise BootstrapError("GPU_READ_SAMPLE_MISMATCH")
    torch.cuda.synchronize(device)
    return True


_PACKAGE_FIELDS = frozenset(
    (
        "manifest",
        "generation",
        "model_revision",
        "capabilities",
        "samples",
    )
)


PROCESS_LIFETIME_GRAPH = "process_lifetime"
_LOGGER = logging.getLogger(__name__)


class EngramModelInitializationError(RuntimeError):
    def __init__(self, code: str, *, resource=None):
        self.code = code
        self.resource = resource
        super().__init__(code)


@dataclass
class EngramModelAttachment:
    phase: str = "INITIALIZING"
    external: bool = False
    graph_mode: str = "eager"
    graph_pin_held: bool = False
    process_retained: bool = False
    storage: Any = field(default=None, repr=False)
    hash_state: Any = field(default=None, repr=False)
    layers: tuple = field(default=(), repr=False)
    failed_resources: list = field(default_factory=list, repr=False)

    layer_indices: tuple[tuple[int, int], ...] = ()

    def require_ready(
        self,
        model,
        *,
        is_cuda_graph=False,
        capture_check=None,
        is_prefill=False,
        is_target_verify=False,
        check_binding=False,
    ):
        """Keep eager default; admit opted-in decode graphs with a permanent pin.

        Replay bypasses Python. The pin, not this per-forward check, prevents close
        from invalidating captured pointers. No successful binding can be detached
        or rebound in-process; even model/graph destruction retains its mapping.
        """
        if not self.external:
            return
        if self.phase != "READY":
            raise EngramModelInitializationError("EXTERNAL_ENGRAM_NOT_READY")
        capturing = is_cuda_graph or (
            (
                self.graph_mode != PROCESS_LIFETIME_GRAPH
                or is_prefill
                or is_target_verify
            )
            and capture_check is not None
            and capture_check()
        )
        if (check_binding or capturing) and any(
            model.v4.layers[lid].engram is not module for lid, _, module in self.layers
        ):
            raise EngramModelInitializationError("EXTERNAL_ENGRAM_BINDING_CHANGED")
        if self.graph_mode == PROCESS_LIFETIME_GRAPH:
            if not self.graph_pin_held or not self.process_retained:
                raise EngramModelInitializationError(
                    "EXTERNAL_ENGRAM_GRAPH_LIFETIME_UNPINNED"
                )
            # Ordinary prefill remains eager. Decode replay never re-enters this
            # Python guard; decode graph capture is checked below.
            if is_prefill or is_target_verify:
                if capturing:
                    # A decode graph warms up on a prefill-shaped batch and on the
                    # speculative target-verify batch, so the input tag alone does
                    # not prove a prefill capture. A decode-role model owns both
                    # graphs; only a prefill-role capture stays unsupported.
                    if getattr(model, "_is_decode_role", False):
                        return
                    raise EngramModelInitializationError(
                        "EXTERNAL_ENGRAM_DECODE_GRAPH_ONLY"
                    )
            return
        if capturing:
            raise EngramModelInitializationError("EXTERNAL_ENGRAM_GRAPH_UNSUPPORTED")

    @classmethod
    def bind(
        cls,
        model,
        runtime: EngramRuntime,
        *,
        layout_factory=None,
        hash_factory=None,
        layer_factory=None,
        attach_factory=None,
        retain=None,
        environ=None,
        scale_group=None,
    ):
        """Build every model-local Engram layer, then publish the complete binding.

        ``model`` supplies ``_v4_args.v41_config``, ``v4.layers``, and
        ``v4.embed.weight.device``. A successful repeated call returns the same
        attachment. A failed call cannot be retried in-process.

        External identity and GPU content verification belong to
        ``attach_from_package``. This function never reads capabilities or infers
        expected identity from an untrusted manifest.
        """
        previous = getattr(model, "_engram_attachment", None)
        if previous is not None:
            if previous.phase in ("READY", "EMPTY"):
                previous.require_ready(model, check_binding=True)
                return previous
            raise EngramModelInitializationError("ENGRAM_INITIALIZATION_NOT_RETRYABLE")

        attachment = cls()
        model._engram_attachment = attachment
        retain = retain or runtime.retain
        constructed = []
        published = []
        storage = None
        device = None
        phase = "layout"
        started = time.monotonic()
        try:
            if layout_factory is None:
                from ..engram import EngramLayout

                layout_factory = EngramLayout
            config = model._v4_args.v41_config
            layout = layout_factory(config)
            if not layout.layer_ids:
                # Draft/no-Engram models do not inspect the external launch bundle.
                attachment.phase = "EMPTY"
                return attachment

            env = os.environ if environ is None else environ
            prepared = runtime.prepared()
            if prepared is None and StoragePolicy.from_environ(env).shared:
                raise EngramModelInitializationError(
                    "DISTRIBUTED_ENGRAM_REQUIRES_EXCHANGE"
                )
            attachment.external = prepared is not None
            phase = "graph_config"
            if attachment.external:
                package, options = prepared
                attachment.graph_mode = options.graph_mode
                if attachment.graph_mode not in ("eager", PROCESS_LIFETIME_GRAPH):
                    raise EngramModelInitializationError(
                        "INVALID_EXTERNAL_ENGRAM_GRAPH_MODE"
                    )
                hw = getattr(model, "py_hw_kernel_config", None)
                if (
                    hw is not None
                    and bool(getattr(hw, "enable_cuda_graph", False))
                    and attachment.graph_mode != PROCESS_LIFETIME_GRAPH
                ):
                    raise EngramModelInitializationError(
                        "EXTERNAL_ENGRAM_GRAPH_UNSUPPORTED"
                    )

            if hash_factory is None:
                from ..engram import NgramHashState

                hash_factory = NgramHashState
            if layer_factory is None:
                from ..engram import Engram

                layer_factory = Engram.from_checkpoint
            device = model.v4.embed.weight.device
            checkpoint_path = config["checkpoint_path"]
            phase = "hash_init"
            hash_state = hash_factory(layout, checkpoint_path, device=device)
            if attachment.external:
                if attach_factory is None:
                    attach_factory = runtime.attach_from_package
                phase = "storage_attach"
                storage = attach_factory(
                    package,
                    layout=layout,
                    hash_state=hash_state,
                    device=device,
                    options=options,
                )
                attachment.storage = storage

            phase = "layer_construction"
            for hash_index, layer_id in enumerate(layout.layer_ids):
                if layer_id >= len(model.v4.layers):
                    continue  # Preserve main's model-local layer selection.
                if layer_id < 0:
                    raise EngramModelInitializationError("INVALID_ENGRAM_LAYER_ID")
                kwargs = {"external_storage": storage} if attachment.external else {}
                if scale_group is not None:
                    kwargs["scale_group"] = scale_group
                module = layer_factory(
                    config, layer_id, checkpoint_path, device, **kwargs
                )
                constructed.append((layer_id, hash_index, module))

            if attachment.external and attachment.graph_mode == PROCESS_LIFETIME_GRAPH:
                # Pin before any layer is published or the engine can capture. This
                # pin is deliberately never removed after READY: worker exit owns
                # cleanup, including capture/drain failure and model destruction.
                phase = "graph_pin"
                storage._pin_graph()
                attachment.graph_pin_held = True

            # No layer can become visible while another layer is still being loaded.
            phase = "model_binding"
            for layer_id, _, module in constructed:
                layer = model.v4.layers[layer_id]
                old = getattr(layer, "engram", None)
                published.append((layer, old))
                layer.engram = module
            if attachment.external:
                retain(storage)
                attachment.process_retained = True
            attachment.hash_state = hash_state
            attachment.layers = tuple(constructed)
            attachment.layer_indices = tuple(
                (lid, index) for lid, index, _ in constructed
            )
            attachment.phase = "READY"
            if attachment.external:
                _log_model_state(attachment, phase, started, device)
            return attachment
        except BaseException as original:
            attachment.phase = "FAILED"
            cleanup_errors = []
            for layer, old in reversed(published):
                try:
                    layer.engram = old
                except BaseException as error:
                    cleanup_errors.append(error)
            if attachment.external:
                # READY was never published, so no forward/capture was admitted.
                # Only this pre-READY rollback may remove the lifetime pin.
                if attachment.graph_pin_held:
                    try:
                        storage._unpin_graph()
                        attachment.graph_pin_held = False
                    except BaseException as error:
                        cleanup_errors.append(error)
                for _, _, module in reversed(constructed):
                    try:
                        module.release_external_views()
                    except BaseException as error:
                        # Keep the module as well as storage: it may still own views.
                        attachment.failed_resources.append(module)
                        cleanup_errors.append(error)
                resource = (
                    storage
                    if storage is not None
                    else getattr(original, "resource", None)
                )
                if resource is not None:
                    try:
                        resource.close()
                    except BaseException as error:
                        cleanup_errors.append(error)
                    if cleanup_errors:
                        attachment.failed_resources.append(resource)
                        retain(resource)
                _log_model_state(attachment, phase, started, device, original)
            if cleanup_errors:
                raise EngramModelInitializationError(
                    "ENGRAM_CLEANUP_INCOMPLETE", resource=attachment
                ) from original
            raise


def _log_model_state(attachment, phase, started, device, error=None):
    """Report actual model publication, separately from control-plane readiness."""
    info = {}
    try:
        if attachment.storage is not None:
            info = attachment.storage.info()
    except Exception:
        # A missing diagnostic snapshot must not interfere with publication or
        # rollback. Capacity is also recorded at the storage attach boundary.
        pass
    try:
        _LOGGER.log(
            logging.INFO if error is None else logging.ERROR,
            "engram event=%s phase=%s state=%s generation=%s device=%s shard_count=%s "
            "physical_bytes=%s allocation_count=%s mapping_count=%s graph_pins=%s "
            "graph_mode=%s process_retained=%s layer_count=%s duration_ms=%.3f "
            "error_class=%s cleanup_pending=%s",
            "model_ready" if error is None else "model_failed",
            phase,
            attachment.phase,
            info.get("generation", "none"),
            info.get("device", device),
            info.get("shard_count", "none"),
            info.get("physical_bytes", "none"),
            info.get("allocation_count", "none"),
            info.get("mapping_count", "none"),
            info.get("graph_pins", int(attachment.graph_pin_held)),
            (
                attachment.graph_mode
                if attachment.graph_mode in ("eager", PROCESS_LIFETIME_GRAPH)
                else "invalid"
            ),
            attachment.process_retained,
            len(attachment.layers),
            (time.monotonic() - started) * 1000,
            type(error).__name__[:64] if error is not None else "none",
            bool(attachment.failed_resources),
        )
    except Exception:
        pass


class EngramRuntime:
    """Per-worker package handoff, model binding and process-lifetime ownership.

    No GC/atexit close: the CUDA/OS process teardown owns final cleanup until
    the engine supplies an explicit drain contract, including captured graphs.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._prepared: tuple[dict, EngramOptions] | None = None
        self._storages = []

    def install(self, package: dict, options: EngramOptions) -> None:
        """Commit the verified startup package before this rank acknowledges it."""
        if not options.policy.shared:
            raise BootstrapError("LOCAL_ENGRAM_HAS_SHARED_PACKAGE")
        prepared = (package, options)
        with self._lock:
            if self._prepared is not None and self._prepared != prepared:
                raise BootstrapError("PACKAGE_ALREADY_STAGED")
            self._prepared = prepared

    def prepared(self):
        with self._lock:
            return self._prepared

    def retain(self, storage):
        with self._lock:
            if not any(item is storage for item in self._storages):
                self._storages.append(storage)

    def _attach_validated(self, validated, handles, samples, target):
        """Map the validated contract once; GPU sample verification gates READY."""
        import torch

        ordinal = (
            target.index if target.index is not None else torch.cuda.current_device()
        )
        manifest = validated.manifest
        native = _load_native()
        native_limit = getattr(native, "max_allocation_bytes", None)
        if (
            type(native_limit) is not int
            or native_limit <= 0
            or any(
                shard.allocation_bytes > native_limit
                for t in manifest.tensors
                for shard in t.shards
            )
        ):
            raise BootstrapError("NATIVE_ALLOCATION_LIMIT")
        try:
            storage = FabricEngramStorage._attach_validated(
                validated, handles=handles, device=ordinal, native=native
            )
        except CleanupPendingError as error:
            self.retain(error.resource)
            raise
        del handles
        try:
            storage.verify(
                lambda views, verified: _verify_samples(
                    views, verified, samples, target
                )
            )
            return storage
        except BaseException as original:
            try:
                storage.close()
            except BaseException:
                self.retain(storage)
                raise CleanupPendingError(
                    "external Engram verification cleanup incomplete", storage
                ) from original
            raise

    def attach_from_package(
        self, package, *, layout, hash_state, device, options: EngramOptions
    ):
        """Attach from an in-memory startup package. Serving never reads bundle files."""
        if type(package) is not dict or set(package) != _PACKAGE_FIELDS:
            raise BootstrapError("INVALID_PACKAGE")
        parsed = package["manifest"]
        if not isinstance(parsed, ParsedManifest):
            raise BootstrapError("INVALID_PACKAGE")
        for key in ("generation", "model_revision"):
            if type(package[key]) is not str or not package[key]:
                raise BootstrapError("INVALID_PACKAGE")
        import torch

        target = torch.device(device)
        if target.type != "cuda":
            raise BootstrapError("CUDA_DEVICE_REQUIRED")
        expected = ExpectedLayout(
            layers=tuple(zip(layout.layer_ids, layout.num_embeddings)),
            head_dim=layout.head_dim,
            model_revision=package["model_revision"],
            hash_abi=hash_state_digest(hash_state),
            backend=options.policy.backend,
        )
        validated = bind_manifest(
            parsed,
            expected,
            package["generation"],
            _MAPPING_LIMITS,
        )
        manifest = validated.manifest
        options.shards.validate_manifest(manifest)
        _validate_mapping_capacity(validated, options.shards.shard_count)
        names = {tensor.name for tensor in manifest.tensors}
        samples = package["samples"]
        if type(samples) is not dict or samples.keys() != names:
            raise BootstrapError("INCOMPLETE_READ_SAMPLES")
        for tensor in manifest.tensors:
            sample = samples[tensor.name]
            required = sorted(
                {0, tensor.shape[0] // 2, tensor.shape[0] - 1}
                | {
                    row
                    for part in tensor.shards
                    for row in (part.row_begin, part.row_end - 1)
                }
            )
            if (
                type(sample) is not dict
                or sample.keys() != {"row_ids", "sha256"}
                or type(sample["row_ids"]) is not list
                or any(type(row) is not int for row in sample["row_ids"])
                or sample["row_ids"] != required
                or type(sample["sha256"]) is not str
                or not _DIGEST.fullmatch(sample["sha256"])
            ):
                raise BootstrapError("INVALID_READ_SAMPLE")
        return self._attach_validated(
            validated, package["capabilities"], samples, target
        )


process_runtime = EngramRuntime()

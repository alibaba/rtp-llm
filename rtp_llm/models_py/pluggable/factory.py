"""Prebind implementations before allocation, then consume bindings explicitly."""

import hashlib
import inspect
import json
import logging
import uuid
import weakref
from dataclasses import dataclass
from pathlib import Path

from rtp_llm.config.module_dispatch_config import ModuleDispatchConfig
from rtp_llm.device.runtime import DeviceRuntimeContext
from rtp_llm.models_py.pluggable.spec import (
    BuildRequest,
    ModuleBinding,
    canonical_json,
    load_entrypoint,
)


@dataclass(frozen=True)
class ModuleSelectionContext:
    platform: DeviceRuntimeContext
    # Only rank-invariant selection/communication facts belong in this object.
    # Device addresses, local rank and resource handles live in DeviceRuntimeContext
    # or builder arguments, not in the protocol digest.
    model_metadata_json: str

    def __post_init__(self):
        metadata = json.loads(self.model_metadata_json)
        if not isinstance(metadata, dict):
            raise TypeError("Model metadata must be a JSON object")
        object.__setattr__(self, "model_metadata_json", canonical_json(metadata))

    @property
    def model_metadata(self):
        return json.loads(self.model_metadata_json)


class ModuleBuildContext:
    """Owned by one model, including its deferred initialize() construction.

    A separate context is required for an MTP/draft model. Only descriptors are
    shared across contexts. This object is confined to the model startup thread.
    """

    def __init__(
        self, registry, selection, config, *, world_size=1, model_adapter=None
    ):
        if not registry.frozen:
            raise RuntimeError("Build context requires a frozen registry")
        if config.mode != "auto":
            raise ValueError("Legacy construction must bypass ModuleFactory")
        if (
            config.platform != "auto"
            and config.platform != selection.platform.device_type.name.lower()
        ):
            raise ValueError(
                "Module config platform conflicts with DeviceRuntimeContext"
            )
        if type(world_size) is not int or world_size < 1:
            raise ValueError("world_size must be a positive integer")
        self.registry = registry
        self.selection = selection
        self.config: ModuleDispatchConfig = config
        self.world_size = world_size
        self.factory = ModuleFactory(self)
        self._bindings = {}
        self._built = set()
        self._building = set()
        self._state = "new"
        self._digest = None
        self.model_instance_id = uuid.uuid4().hex
        self._instance_refs = {}
        self._class_sources = {}
        self.model_adapter = model_adapter
        self.resource_plan = None
        self.weights_loaded = False
        self._weight_loader = None
        self._weight_preparation = None

    def fail(self):
        """Poison this context; resource owners retain teardown responsibility."""
        self._state = "failed"

    def configure_weight_loader(self, loader):
        if (
            self.state != "verified"
            or self.resource_plan is None
            or self._weight_loader is not None
        ):
            raise RuntimeError(
                "Weight preparation requires an unused verified resource plan"
            )
        try:
            entry = self.resource_plan.weight_preparation
            preparation = load_entrypoint(entry)() if entry else None
            loader.get_load_config().weight_preparation = preparation
            self._weight_preparation = preparation
            self._weight_loader = weakref.ref(loader)
        except BaseException:
            self.fail()
            raise

    def finish_weight_loading(self, loader):
        if (
            self.state != "verified"
            or self.weights_loaded
            or self._weight_loader is None
            or self._weight_loader() is not loader
            or loader.get_load_config().weight_preparation
            is not self._weight_preparation
        ):
            self.fail()
            raise RuntimeError(
                "Weight loader did not consume the frozen preparation plan"
            )
        self.weights_loaded = True

    @property
    def bindings(self):
        return tuple(self._bindings[path] for path in sorted(self._bindings))

    @property
    def state(self):
        return self._state

    def _record_instance(self, binding, module):
        cls = type(module)
        class_name = cls.__module__ + "." + cls.__qualname__
        if class_name not in self._class_sources:
            source = inspect.getsourcefile(cls)
            self._class_sources[class_name] = {
                "path": source,
                "sha256": (
                    hashlib.sha256(Path(source).read_bytes()).hexdigest()
                    if source is not None
                    else None
                ),
            }
        record = {
            **binding.protocol_record(),
            "model_instance_id": self.model_instance_id,
            "actual_class": class_name,
            "source": self._class_sources[class_name],
            "selection_source": binding.selection_source,
            "support_check": "accepted",
            "platform": self.selection.platform.device_type.name,
            "local_rank": self.selection.platform.local_rank,
            "role": self.selection.model_metadata.get("role"),
            "protocol_digest": self.protocol_digest,
        }
        self._instance_refs[binding.request.path] = weakref.ref(module)
        if (
            self.model_adapter is not None
            and binding.request == self.model_adapter.root_request(self.selection)
        ):
            from .lifecycle import bind_model_context

            bind_model_context(module, self)
        logging.info("module_dispatch bound: %s", canonical_json(record))

    def validate_built_tree(self, root, *, root_path):
        """Confirm the initialized module tree still contains the objects built."""
        if self._state not in ("verified", "closed"):
            raise RuntimeError(f"Cannot validate module tree in state {self._state}")
        try:
            if set(self._instance_refs) != set(self._bindings):
                raise RuntimeError(
                    "Module tree validation requires all planned objects"
                )
            if root_path not in self._instance_refs:
                raise ValueError(f"Unbound model root {root_path}")
            for path, reference in self._instance_refs.items():
                actual = root if path == root_path else root.get_submodule(path)
                if reference() is not actual:
                    raise RuntimeError(
                        f"Initialized module differs from binding: {path}"
                    )
        except BaseException:
            self._state = "failed"
            raise

    @property
    def protocol_digest(self):
        if self._digest is None:
            raise RuntimeError("Module plan has not been prepared")
        return self._digest

    def prepare(self, root_requests):
        """Expand light descriptors without importing any selected builder."""
        if self._state != "new":
            raise RuntimeError(f"Cannot prepare module context in state {self._state}")
        self._state = "planning"
        module_overrides = dict(self.config.impl_overrides)
        path_overrides = dict(self.config.path_overrides)
        used_modules, used_paths = set(), set()
        try:
            pending = list(root_requests)
            cursor = 0
            while cursor < len(pending):
                request = pending[cursor]
                cursor += 1
                if not isinstance(request, BuildRequest):
                    raise TypeError("Descriptors must return BuildRequest objects")
                if request.path in self._bindings:
                    raise ValueError(f"Duplicate planned module path {request.path}")
                # Bound expansion so a recursive descriptor fails before allocation.
                if cursor > 100000:
                    raise ValueError("Module description exceeds 100000 requests")
                source = "auto"
                explicit = None
                if request.path in path_overrides:
                    source, explicit = "path_override", path_overrides[request.path]
                    used_paths.add(request.path)
                elif request.module_id in module_overrides:
                    source, explicit = (
                        "module_override",
                        module_overrides[request.module_id],
                    )
                    used_modules.add(request.module_id)
                impl = self.registry.resolve(request, self.selection, explicit)
                self._bindings[request.path] = ModuleBinding(request, impl, source)
                if impl.describe_build_requests:
                    children = load_entrypoint(impl.describe_build_requests)(
                        self.selection, request
                    )
                    for child in children:
                        if not isinstance(
                            child, BuildRequest
                        ) or not child.path.startswith(request.path + "."):
                            raise ValueError(
                                f"Descriptor children must be beneath {request.path}"
                            )
                        pending.append(child)
            unused = (set(module_overrides) - used_modules) | (
                set(path_overrides) - used_paths
            )
            if unused:
                raise ValueError(
                    f"Unused or shadowed module overrides: {sorted(unused)}"
                )
            if not self._bindings:
                raise ValueError("Module plan must contain at least one request")
            if self.model_adapter is not None:
                from .resources import ResourcePlan

                self.resource_plan = self.model_adapter.plan_resources(
                    self.selection, self.bindings
                )
                if not isinstance(self.resource_plan, ResourcePlan):
                    raise TypeError("Model adapter must return a ResourcePlan")
            protocol = {
                "schema": 1,
                "world_size": self.world_size,
                "device_type": self.selection.platform.device_type.name,
                "model_metadata": self.selection.model_metadata,
                "bindings": [binding.protocol_record() for binding in self.bindings],
                "resources": (
                    self.resource_plan.record() if self.resource_plan else None
                ),
            }
            self._digest = hashlib.sha256(canonical_json(protocol).encode()).hexdigest()
            self._state = "planned"
            return self._digest
        except BaseException:
            self._state = "failed"
            raise

    def verify_protocol(self, verifier=None):
        """Call an existing CPU startup channel before any selected builder.

        The verifier must compare every rank in the applicable execution group,
        return True on agreement and raise on timeout/mismatch. It must not use
        a collective supplied by one of the implementations being checked.
        """
        if self._state != "planned":
            raise RuntimeError(f"Cannot verify module context in state {self._state}")
        try:
            if verifier is None:
                if self.world_size > 1:
                    raise RuntimeError(
                        "Multi-rank module construction requires protocol verification"
                    )
            elif verifier(self.protocol_digest) is not True:
                raise RuntimeError("Protocol verifier did not confirm agreement")
            self._state = "verified"
            logging.info(
                "module_dispatch preflight: digest=%s local_rank=%s bindings=%s",
                self.protocol_digest,
                self.selection.platform.local_rank,
                canonical_json([b.protocol_record() for b in self.bindings]),
            )
        except BaseException:
            self._state = "failed"
            raise

    def close(self):
        if self._state == "closed":
            return
        if self._state != "verified" or self._building:
            raise RuntimeError(f"Cannot close module context in state {self._state}")
        missing = set(self._bindings) - self._built
        if missing:
            self._state = "failed"
            raise RuntimeError(
                f"Planned modules were not constructed: {sorted(missing)}"
            )
        self._state = "closed"


class ModuleFactory:
    def __init__(self, build_ctx):
        self._ctx = build_ctx

    def build(self, request: BuildRequest, **kwargs):
        ctx = self._ctx
        if ctx.state != "verified":
            raise RuntimeError(f"Module construction is forbidden in state {ctx.state}")
        try:
            binding = ctx._bindings.get(request.path)
            if binding is None or binding.request != request:
                raise ValueError(
                    f"Build metadata differs from preflight: {request.path}"
                )
            if request.path in ctx._built or request.path in ctx._building:
                raise RuntimeError(
                    f"Module already constructed or constructing: {request.path}"
                )
            ctx._building.add(request.path)
            impl = binding.implementation
            builder = load_entrypoint(impl.builder)
            inspect.signature(builder).bind(build_ctx=ctx, request=request, **kwargs)
            module = builder(build_ctx=ctx, request=request, **kwargs)
            spec = ctx.registry.module(request.module_id)
            for method in spec.required_methods:
                if not callable(getattr(module, method, None)):
                    raise TypeError(f"{impl.impl_id} lacks required method {method}")
            if spec.validate_instance:
                load_entrypoint(spec.validate_instance)(module, ctx, request)
            # A nested builder failure poisons the context even if its caller
            # catches the exception; no fallback after partial state allocation.
            if ctx.state != "verified":
                raise RuntimeError(
                    "Module build context failed during nested construction"
                )
            ctx._record_instance(binding, module)
            ctx._building.remove(request.path)
            ctx._built.add(request.path)
            return module
        except BaseException:
            ctx._state = "failed"
            raise

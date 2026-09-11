"""A frozen descriptor registry shared by independent model build contexts."""

from threading import RLock

from rtp_llm.models_py.pluggable.spec import (
    ModuleImplSpec,
    ModuleSpec,
    SupportResult,
    load_entrypoint,
)


class ModuleRegistry:
    def __init__(self):
        self._modules = {}
        self._implementations = {}
        self._frozen = False
        self._lock = RLock()

    @property
    def frozen(self):
        return self._frozen

    def _register(self, table, key, spec):
        with self._lock:
            if self._frozen:
                raise RuntimeError("Module registry is frozen")
            previous = table.get(key)
            if previous is not None and previous != spec:
                raise ValueError(f"Conflicting module registration: {key!r}")
            table[key] = spec

    def register_module(self, spec: ModuleSpec):
        self._register(self._modules, spec.module_id, spec)

    def register_implementation(self, spec: ModuleImplSpec):
        self._register(self._implementations, (spec.module_id, spec.impl_id), spec)

    def freeze(self):
        with self._lock:
            for impl in self._implementations.values():
                spec = self.module(impl.module_id)
                if (impl.api_version, impl.contract_id) != (
                    spec.api_version,
                    spec.contract_id,
                ):
                    raise ValueError(f"API/contract mismatch for {impl.impl_id}")
            self._frozen = True
        return self

    def module(self, module_id):
        try:
            return self._modules[module_id]
        except KeyError:
            raise ValueError(f"Unknown module {module_id!r}") from None

    def has_module(self, module_id):
        """Let platform discovery attach only to registered model contracts."""
        return module_id in self._modules

    def implementation(self, module_id, impl_id):
        """Read a frozen descriptor without importing its builder or probing a device."""
        if not self._frozen:
            raise RuntimeError("Freeze the module registry before descriptor lookup")
        try:
            return self._implementations[(module_id, impl_id)]
        except KeyError:
            raise ValueError(
                f"Unknown implementation {impl_id!r} for {module_id}"
            ) from None

    def resolve(self, request, selection, explicit_impl=None):
        if not self._frozen:
            raise RuntimeError("Freeze the module registry before selection")
        self.module(request.module_id)
        candidates = sorted(
            (
                i
                for i in self._implementations.values()
                if i.module_id == request.module_id
            ),
            key=lambda i: i.impl_id,
        )
        if explicit_impl is not None:
            candidates = [i for i in candidates if i.impl_id == explicit_impl]
            if not candidates:
                raise ValueError(
                    f"Unknown implementation {explicit_impl!r} for {request.path}"
                )
        accepted, rejected = [], []
        for impl in candidates:
            reason = ""
            if selection.platform.device_type not in impl.supported_devices:
                reason = "device type is unsupported"
            elif request.weight_format_id != impl.weight_format_id:
                reason = "weight format mismatch"
            elif request.state_format_id != impl.state_format_id:
                reason = "state format mismatch"
            elif not request.required_capabilities <= impl.capabilities:
                reason = f"missing capabilities {sorted(request.required_capabilities - impl.capabilities)}"
            elif explicit_impl is None and not impl.auto_selectable:
                reason = "explicit selection required"
            else:
                # Only lightweight predicates may run here; a broken dependency
                # propagates, rather than silently becoming an unsupported result.
                support = load_entrypoint(impl.predicate)(selection, request)
                if not isinstance(support, SupportResult):
                    raise TypeError(
                        f"Predicate for {impl.impl_id} must return SupportResult"
                    )
                if not support.supported:
                    reason = support.reason or "predicate rejected configuration"
            if reason:
                rejected.append(f"{impl.impl_id}: {reason}")
            else:
                accepted.append(impl)
        if not accepted:
            raise ValueError(
                f"No compatible implementation for {request.path}: {'; '.join(rejected)}"
            )
        priority = max(impl.priority for impl in accepted)
        winners = [impl for impl in accepted if impl.priority == priority]
        if len(winners) != 1:
            raise ValueError(
                f"Ambiguous implementation for {request.path}: {[i.impl_id for i in winners]}"
            )
        return winners[0]

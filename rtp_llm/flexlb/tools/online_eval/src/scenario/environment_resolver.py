"""Preset loading, profile rendering, deviation checks and frozen effective environments."""
from __future__ import annotations
import copy
import json
from dataclasses import dataclass
from flexlb_cfg import PROFILE_CAPS, PROFILES, VICTIM_STAGES, ProfileIdentityError, render_env, validate_profile_identity
from adapters.perf_presets import load_preset_bundle
from document_io import DocumentError as ScenarioError
from scenario.validation import fail, mapping
from scenario.master_patch import MasterPatch
from scenario.environment_declaration import EnvironmentDeclaration, captured_fields
from scenario.environment_snapshot import EnvironmentSnapshot

CAPABILITIES = set().union(*PROFILE_CAPS.values()) | {
    "priority",
    "preemption",
    "engine_cancellation",
}


def effective_capabilities(config):
    scheduler, dispatcher = config["scheduler"], config["dispatcher"]
    axes = dict(
        scheduler=scheduler["type"],
        ordering=scheduler["ordering"]["type"],
        decision=scheduler["decision"]["type"],
        dispatcher=dispatcher["type"],
    )
    caps = {"queue", axes["ordering"].lower(), axes["decision"].lower()}
    if dispatcher["type"] == "BATCH":
        caps.update({"batch_dispatch", "enqueue_batch", "fetch_response"})
    elif dispatcher["type"] == "NON_BATCH":
        caps.update({"non_batch_dispatch", "frontend_send", "generate_stream"})
    else:
        raise ValueError("unrecognized effective dispatcher")
    preemption = scheduler["ordering"].get("preemption")
    if preemption is None and scheduler["ordering"]["type"] == "PRIORITY":
        preemption = {"allowedVictimStages": list(VICTIM_STAGES)}
    if preemption:
        caps.add("preemption")
        if "DECODE_ENGINE_OWNED" in preemption.get("allowedVictimStages", []):
            caps.add("engine_cancellation")
    return axes, sorted(caps)


@dataclass(frozen=True)
class ResolvedEnvironment:
    declaration: EnvironmentDeclaration
    master_patch: MasterPatch
    snapshot: "EnvironmentSnapshot"
    effective_axes: dict[str, str]
    effective_capabilities: tuple[str, ...]

    def to_dict(self):
        return dict(self.declaration.to_dict(), config_overrides=copy.deepcopy(self.master_patch.declared),
                    resolved_config=json.loads(self.snapshot.master_json), rendered=self.snapshot.to_dict(),
                    effective_axes=dict(self.effective_axes), effective_capabilities=list(self.effective_capabilities))


@dataclass(frozen=True)
class EnvironmentResolver:
    """One preset load and one patch parse per environment declaration."""

    performance: dict
    runtime: dict
    common: MasterPatch
    profiles: dict[str, MasterPatch]
    captured: dict[str, int]

    @classmethod
    def read(cls, value, path, preset):
        bundle = load_preset_bundle(preset)
        performance, runtime = bundle.performance, bundle.runtime
        common = MasterPatch.read(value.get("config_overrides", {}), path + ".config_overrides")
        patches = mapping(value.get("profile_overrides", {}), path + ".profile_overrides", PROFILES)
        profiles = {profile: MasterPatch.read(patch, path + ".profile_overrides." + profile)
                    for profile, patch in patches.items()}
        paired = runtime.get("paired_master")
        # Check excluded profiles too; this never recursively renders an environment.
        for profile, patch in profiles.items():
            cls._identity(profile, patch.declared, path)
            cls._identity(profile, common.declared, path)
            baseline = MasterPatch.read(paired.for_profile(profile) if paired else {}, path + ".preset")
            baseline.overlay(common).overlay(patch)
        return cls(performance, runtime, common, profiles, bundle.captured)

    @staticmethod
    def _identity(profile, patch, path):
        try:
            validate_profile_identity(profile, patch)
        except ProfileIdentityError as exc:
            raise ScenarioError(f"{path}: {exc}") from exc

    def master(self, profile, path):
        paired = self.runtime.get("paired_master")
        baseline = MasterPatch.read(paired.for_profile(profile) if paired else {}, path + ".preset")
        patch = self.common.overlay(self.profiles.get(profile, MasterPatch.read({}, path)))
        for candidate in (baseline, patch):
            self._identity(profile, candidate.declared, path)
        resolved = baseline.overlay(patch)
        try:
            config = json.loads(render_env(profile, resolved.configuration))
        except (TypeError, ValueError) as exc:
            fail(path + ".config_overrides", str(exc))
        return baseline, patch, resolved, config

    def resolve(self, declaration, path, profile):
        baseline, declared, resolved, config = self.master(profile, path)
        paired_master, declared_overrides = baseline.declared, declared.declared
        captured = self.captured
        if captured:
            changed = [field for field, source_field in captured_fields().items() if getattr(declaration, field) is not None and getattr(declaration, field) != captured[source_field]]
            if (declared_overrides.get("prefill_expression") is not None and declared_overrides["prefill_expression"] != paired_master.get("prefill_expression")) or declaration.prefill_perf is not None:
                changed.append("prefill model")
            changed.extend("master." + key for key, val in declared_overrides.items()
                           if val is not None and key in paired_master and val != paired_master[key])
            if declaration.prefill_cache_policy is not None:
                performance = self.performance
                policy = declaration.prefill_cache_policy
                original = performance.get("prefill", {})
                memory = original.get("memory_cache", {})
                if (policy.device_tree != original.get("enable_gpu_prefix_tree")
                    or policy.memory_tree != memory.get("enable_prefix_tree")
                    or (policy.memory_blocks is not None and policy.memory_blocks != memory.get("capacity_blocks"))):
                    changed.append("prefill_cache_policy")
            if changed and declaration.model_override is None:
                fail(path + ".model_override", "test deviation requires baseline and reason: " + ", ".join(changed))
        axes, capabilities = effective_capabilities(config)
        from scenario.environment_snapshot import EnvironmentSnapshot
        snapshot = EnvironmentSnapshot.freeze(
            config, self.effective_performance(declaration),
            {key: value for key, value in self.runtime.items() if key != "paired_master"},
        )
        return ResolvedEnvironment(declaration, resolved, snapshot, axes, tuple(capabilities))

    def effective_performance(self, declaration):
        return declaration.apply_performance(self.performance)

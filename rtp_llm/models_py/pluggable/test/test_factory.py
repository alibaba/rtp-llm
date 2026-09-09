import dataclasses
import gc
import json
import os
import pickle
import subprocess
import sys
import unittest
import weakref

from rtp_llm.config.module_dispatch_config import ModuleDispatchConfig
from rtp_llm.device.device_type import DeviceType
from rtp_llm.models_py.pluggable.factory import (
    ModuleBuildContext,
    ModuleSelectionContext,
)
from rtp_llm.models_py.pluggable.platform import PlatformContext
from rtp_llm.models_py.pluggable.registry import ModuleRegistry
from rtp_llm.models_py.pluggable.spec import (
    BuildRequest,
    ModuleImplSpec,
    ModuleSpec,
    SupportResult,
)

ENTRY = __name__
BUILD_CALLS = []


def supports(selection, request):
    if selection.model_metadata.get("tp_size", 4) != 4:
        return SupportResult(False, "requires TP4")
    return SupportResult(True)


def broken_predicate(selection, request):
    raise ImportError("broken installed backend dependency")


class ActualModule:
    def __init__(self, weight, build_ctx):
        self.weight = weight
        self.build_ctx = build_ctx

    def forward(self, value):
        return value

    def get_submodule(self, path):
        module = self
        for component in path.split("."):
            module = getattr(module, component)
        return module


class RegistryDiscoveryTest(unittest.TestCase):
    def test_other_model_registry_does_not_acquire_v4_implementations(self):
        from unittest.mock import patch

        from rtp_llm.models_py.pluggable.bootstrap import get_module_registry

        with patch(
            "rtp_llm.models_py.pluggable.bootstrap.import_optional_internal_source_entrypoint",
            return_value=False,
        ):
            registry = get_module_registry(lambda registry: None)
        self.assertTrue(registry.frozen)
        self.assertFalse(registry.has_module("rtp.dsv4.model"))
        with self.assertRaisesRegex(ValueError, "Unknown implementation"):
            registry.implementation("rtp.dsv4.model", "ppu.dsv4.model.v1")

    def test_unknown_implementation_module_still_rejected(self):
        registry = ModuleRegistry()
        registry.register_implementation(impl_for())
        with self.assertRaisesRegex(ValueError, "Unknown module"):
            registry.freeze()


def build(*, build_ctx, request, weight):
    BUILD_CALLS.append(request.path)
    return ActualModule(weight, build_ctx)


def broken_builder(*, build_ctx, request, weight):
    BUILD_CALLS.append(request.path)
    raise RuntimeError("builder allocation failed")


def bad_interface(*, build_ctx, request, weight):
    return object()


def validate_instance(module, build_ctx, request):
    if not isinstance(module, ActualModule):
        raise TypeError("requires ActualModule base")


def describe_children(selection, request):
    return [request_for(request.path + ".attn", module_id="attention")]


def invalid_children(selection, request):
    return [request_for("outside.attn", module_id="attention")]


def request_for(path="model", module_id="model", **kwargs):
    return BuildRequest.create(
        module_id=module_id,
        path=path,
        weight_format_id="loader.v1",
        state_format_id="fp8-cache.v1",
        required_capabilities={"prefill"},
        **kwargs,
    )


def impl_for(impl_id="ppu.base", module_id="model", **kwargs):
    values = dict(
        module_id=module_id,
        impl_id=impl_id,
        api_version=1,
        builder=ENTRY + ":build",
        supported_devices={DeviceType.Ppu},
        predicate=ENTRY + ":supports",
        priority=0,
        contract_id="hidden.v1",
        weight_format_id="loader.v1",
        state_format_id="fp8-cache.v1",
        collective_protocol_id="tp4-fp32.v1",
        capabilities={"prefill"},
        auto_selectable=True,
    )
    values.update(kwargs)
    return ModuleImplSpec(**values)


def registry_for(*impls):
    registry = ModuleRegistry()
    for module in {impl.module_id for impl in impls}:
        registry.register_module(
            ModuleSpec(
                module, 1, "hidden.v1", validate_instance=ENTRY + ":validate_instance"
            )
        )
    for impl in impls:
        registry.register_implementation(impl)
    return registry.freeze()


def context_for(registry=None, config=None, *, rank=0, world_size=1, metadata=None):
    return ModuleBuildContext(
        registry or registry_for(impl_for()),
        ModuleSelectionContext(
            PlatformContext(DeviceType.Ppu, "ZW-M890P", rank),
            json.dumps(metadata or {"tp_size": 4}),
        ),
        config or ModuleDispatchConfig(mode="auto"),
        world_size=world_size,
    )


class ModuleFactoryTest(unittest.TestCase):
    def test_bound_records_identify_actual_object_without_retaining_it(self):
        ctx = context_for()
        ctx.prepare([request_for()])
        ctx.verify_protocol()
        module = ctx.factory.build(request_for(), weight=object())
        ctx.validate_built_tree(module, root_path="model")
        ctx.close()
        ctx.validate_built_tree(module, root_path="model")
        record = ctx.bound_instances[0]
        self.assertEqual(record["actual_class"], ENTRY + ".ActualModule")
        self.assertEqual(record["model_instance_id"], ctx.model_instance_id)
        self.assertEqual(record["protocol_digest"], ctx.protocol_digest)
        self.assertEqual(len(record["source"]["sha256"]), 64)
        record["source"]["sha256"] = "mutated snapshot"
        self.assertNotEqual(
            ctx.bound_instances[0]["source"]["sha256"], "mutated snapshot"
        )
        reference = weakref.ref(module)
        del module
        gc.collect()
        self.assertIsNone(reference())

    def test_initialized_tree_rejects_replaced_child(self):
        ctx = context_for(
            registry_for(
                impl_for(describe_build_requests=ENTRY + ":describe_children"),
                impl_for("ppu.attn", module_id="attention"),
            )
        )
        ctx.prepare([request_for()])
        ctx.verify_protocol()
        root = ctx.factory.build(request_for(), weight=object())
        child = ctx.factory.build(
            request_for("model.attn", module_id="attention"), weight=object()
        )
        root.model = type("Children", (), {})()
        root.model.attn = child
        ctx.validate_built_tree(root, root_path="model")
        root.model.attn = ActualModule(object(), ctx)
        with self.assertRaisesRegex(RuntimeError, "differs from binding: model.attn"):
            ctx.validate_built_tree(root, root_path="model")
        self.assertEqual(ctx.state, "failed")

    def test_instance_identity_is_distinct_and_excluded_from_protocol(self):
        first, second = context_for(), context_for()
        first.prepare([request_for()])
        second.prepare([request_for()])
        self.assertNotEqual(first.model_instance_id, second.model_instance_id)
        self.assertEqual(first.protocol_digest, second.protocol_digest)

    def setUp(self):
        BUILD_CALLS.clear()

    def test_registration_idempotence_conflicts_and_freeze(self):
        registry = ModuleRegistry()
        spec = ModuleSpec("model", 1, "hidden.v1")
        registry.register_module(spec)
        registry.register_module(spec)
        impl = impl_for()
        registry.register_implementation(impl)
        registry.register_implementation(impl)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            registry.register_implementation(dataclasses.replace(impl, priority=10))
        registry.freeze()
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            registry.register_module(spec)

    def test_bad_contract_rejected_at_freeze(self):
        registry = ModuleRegistry()
        registry.register_module(ModuleSpec("model", 1, "other-layout"))
        registry.register_implementation(impl_for())
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            registry.freeze()

    def test_filter_does_not_import_unselected_builder_or_predicate(self):
        impl = impl_for(
            "cuda.fast",
            supported_devices={DeviceType.Cuda},
            builder="missing_cuda_dependency:build",
            predicate="missing_cuda_dependency:predicate",
            priority=100,
        )
        ctx = context_for(registry_for(impl, impl_for()))
        ctx.prepare([request_for()])
        self.assertEqual(ctx.bindings[0].implementation.impl_id, "ppu.base")
        self.assertEqual(BUILD_CALLS, [])

    def test_unsupported_override_fails_without_fallback(self):
        config = ModuleDispatchConfig.from_dict(
            {
                "mode": "auto",
                "impl_overrides": {"model": "cuda.fast"},
            }
        )
        ctx = context_for(
            registry_for(
                impl_for("cuda.fast", supported_devices={DeviceType.Cuda}),
                impl_for(),
            ),
            config,
        )
        with self.assertRaisesRegex(ValueError, "device type is unsupported"):
            ctx.prepare([request_for()])
        self.assertEqual(ctx.state, "failed")
        self.assertEqual(BUILD_CALLS, [])

    def test_no_implicit_cuda_fallback(self):
        ctx = context_for(registry_for(impl_for(supported_devices={DeviceType.Cuda})))
        with self.assertRaisesRegex(ValueError, "No compatible"):
            ctx.prepare([request_for()])

    def test_predicate_rejection_and_broken_dependency_are_distinct(self):
        with self.assertRaisesRegex(ValueError, "requires TP4"):
            context_for(metadata={"tp_size": 8}).prepare([request_for()])
        ctx = context_for(registry_for(impl_for(predicate=ENTRY + ":broken_predicate")))
        with self.assertRaisesRegex(ImportError, "broken installed"):
            ctx.prepare([request_for()])

    def test_tie_fails_in_both_registration_orders(self):
        for impls in [(impl_for("a"), impl_for("b")), (impl_for("b"), impl_for("a"))]:
            with self.subTest(impls=impls):
                with self.assertRaisesRegex(ValueError, "Ambiguous"):
                    context_for(registry_for(*impls)).prepare([request_for()])

    def test_explicit_only_candidate(self):
        registry = registry_for(impl_for(auto_selectable=False))
        with self.assertRaisesRegex(ValueError, "explicit selection required"):
            context_for(registry).prepare([request_for()])
        ctx = context_for(
            registry,
            ModuleDispatchConfig.from_dict(
                {
                    "mode": "auto",
                    "impl_overrides": {"model": "ppu.base"},
                }
            ),
        )
        ctx.prepare([request_for()])

    def test_capability_and_formats_are_required(self):
        request = request_for()
        for changed, expected in [
            (
                dataclasses.replace(request, state_format_id="fp4-cache.v1"),
                "state format",
            ),
            (
                dataclasses.replace(request, weight_format_id="private.v2"),
                "weight format",
            ),
            (
                dataclasses.replace(request, required_capabilities={"decode"}),
                "missing capabilities",
            ),
        ]:
            with self.subTest(request=changed):
                with self.assertRaisesRegex(ValueError, expected):
                    context_for().prepare([changed])

    def test_path_override_priority_and_unused_override(self):
        registry = registry_for(impl_for("a"), impl_for("b", priority=1))
        config = ModuleDispatchConfig.from_dict(
            {
                "mode": "auto",
                "impl_overrides": {"model": "a"},
                "path_overrides": {"model.layers.1": "b"},
            }
        )
        ctx = context_for(registry, config)
        ctx.prepare([request_for("model.layers.0"), request_for("model.layers.1")])
        self.assertEqual([b.implementation.impl_id for b in ctx.bindings], ["a", "b"])
        config = dataclasses.replace(config, path_overrides=(("missing.path", "b"),))
        with self.assertRaisesRegex(ValueError, "Unused or shadowed"):
            context_for(registry, config).prepare([request_for()])

    def test_shadowed_module_override_is_not_silently_accepted(self):
        config = ModuleDispatchConfig.from_dict(
            {
                "mode": "auto",
                "impl_overrides": {"model": "ppu.base"},
                "path_overrides": {"model": "ppu.base"},
            }
        )
        with self.assertRaisesRegex(ValueError, "Unused or shadowed"):
            context_for(config=config).prepare([request_for()])

    def test_delayed_construction_keeps_context_and_weight_identity(self):
        registry = registry_for(
            impl_for(describe_build_requests=ENTRY + ":describe_children"),
            impl_for(module_id="attention"),
        )
        ctx = context_for(registry)
        ctx.prepare([request_for()])
        ctx.verify_protocol()
        weight = object()
        module = ctx.factory.build(request_for(), weight=weight)
        self.assertIs(type(module), ActualModule)
        self.assertIs(module.weight, weight)
        self.assertIs(module.build_ctx, ctx)
        attn = module.build_ctx.factory.build(
            request_for("model.attn", "attention"),
            weight=weight,
        )
        self.assertIs(attn.weight, weight)
        ctx.close()
        ctx.close()
        with self.assertRaisesRegex(RuntimeError, "forbidden"):
            ctx.factory.build(request_for(), weight=weight)

    def test_metadata_and_capabilities_are_immutable_snapshots(self):
        metadata, capabilities = {"shape": [8192, 4096]}, {"prefill"}
        request = BuildRequest.create(
            module_id="model",
            path="model",
            weight_format_id="loader.v1",
            state_format_id="fp8-cache.v1",
            metadata=metadata,
            required_capabilities=capabilities,
        )
        metadata["shape"][0] = 1
        capabilities.add("decode")
        self.assertEqual(request.metadata["shape"][0], 8192)
        self.assertEqual(request.required_capabilities, {"prefill"})
        request.metadata["shape"][0] = 2
        self.assertEqual(request.metadata["shape"][0], 8192)

    def test_changed_build_metadata_poison_context(self):
        ctx = context_for()
        ctx.prepare([request_for(metadata={"layer": 0})])
        ctx.verify_protocol()
        with self.assertRaisesRegex(ValueError, "differs from preflight"):
            ctx.factory.build(request_for(metadata={"layer": 1}), weight=object())
        self.assertEqual(BUILD_CALLS, [])
        self.assertEqual(ctx.state, "failed")

    def test_protocol_verification_precedes_builder(self):
        ctx = context_for(world_size=4)
        ctx.prepare([request_for()])
        with self.assertRaisesRegex(RuntimeError, "forbidden"):
            ctx.factory.build(request_for(), weight=object())
        self.assertEqual(BUILD_CALLS, [])
        with self.assertRaisesRegex(RuntimeError, "requires protocol verification"):
            ctx.verify_protocol()

    def test_protocol_disagreement_is_terminal(self):
        ctx = context_for(world_size=4)
        ctx.prepare([request_for()])
        with self.assertRaisesRegex(RuntimeError, "did not confirm"):
            ctx.verify_protocol(lambda digest: False)
        with self.assertRaisesRegex(RuntimeError, "forbidden"):
            ctx.factory.build(request_for(), weight=object())
        self.assertEqual(BUILD_CALLS, [])

    def test_digest_excludes_rank_but_includes_protocol_and_metadata(self):
        first = context_for(rank=0, world_size=4).prepare([request_for()])
        peer = context_for(rank=3, world_size=4).prepare([request_for()])
        self.assertEqual(first, peer)
        changed = context_for(
            registry_for(impl_for(collective_protocol_id="tp4-bf16.v2")), world_size=4
        ).prepare([request_for()])
        self.assertNotEqual(first, changed)
        changed = context_for(
            world_size=4, metadata={"tp_size": 4, "reduce_dtype": "bf16"}
        )
        self.assertNotEqual(first, changed.prepare([request_for()]))

    def test_two_models_share_only_descriptors(self):
        registry = registry_for(impl_for("a"), impl_for("b", priority=1))
        first = context_for(
            registry,
            ModuleDispatchConfig.from_dict(
                {
                    "mode": "auto",
                    "impl_overrides": {"model": "a"},
                }
            ),
        )
        second = context_for(registry)
        for ctx in (first, second):
            ctx.prepare([request_for()])
            ctx.verify_protocol()
        first.factory.build(request_for(), weight=object())
        first.close()
        self.assertEqual(second.state, "verified")
        self.assertEqual(first.bindings[0].implementation.impl_id, "a")
        self.assertEqual(second.bindings[0].implementation.impl_id, "b")
        second.factory.build(request_for(), weight=object())
        second.close()

    def test_builder_failure_does_not_try_another_candidate(self):
        ctx = context_for(
            registry_for(
                impl_for("fallback"),
                impl_for(
                    "fast",
                    builder=ENTRY + ":broken_builder",
                    priority=10,
                ),
            )
        )
        ctx.prepare([request_for()])
        ctx.verify_protocol()
        with self.assertRaisesRegex(RuntimeError, "allocation failed"):
            ctx.factory.build(request_for(), weight=object())
        self.assertEqual(BUILD_CALLS, ["model"])
        with self.assertRaisesRegex(RuntimeError, "forbidden"):
            ctx.factory.build(request_for(), weight=object())

    def test_interface_mismatch_fails(self):
        ctx = context_for(registry_for(impl_for(builder=ENTRY + ":bad_interface")))
        ctx.prepare([request_for()])
        ctx.verify_protocol()
        with self.assertRaisesRegex(TypeError, "required method forward"):
            ctx.factory.build(request_for(), weight=object())

    def test_duplicate_paths_and_invalid_child_boundaries_fail(self):
        with self.assertRaisesRegex(ValueError, "Duplicate planned"):
            context_for().prepare([request_for(), request_for()])
        ctx = context_for(
            registry_for(impl_for(describe_build_requests=ENTRY + ":invalid_children"))
        )
        with self.assertRaisesRegex(ValueError, "beneath model"):
            ctx.prepare([request_for()])

    def test_close_checks_that_whole_plan_was_consumed(self):
        ctx = context_for()
        ctx.prepare([request_for()])
        ctx.verify_protocol()
        with self.assertRaisesRegex(RuntimeError, "not constructed"):
            ctx.close()

    def test_config_round_trip_and_strict_validation(self):
        config = ModuleDispatchConfig.from_dict(
            {
                "mode": "auto",
                "platform": "ppu",
                "impl_overrides": {"model": "ppu.base"},
            }
        )
        self.assertEqual(config, pickle.loads(pickle.dumps(config)))
        self.assertEqual(
            config, ModuleDispatchConfig.from_dict(json.loads(config.to_string()))
        )
        for data in [
            {"mode": "typo"},
            {"platform": "unknown"},
            {"unknown": 1},
            {"impl_overrides": {"model": "ppu.base"}},
            {"mode": "auto", "path_overrides": []},
        ]:
            with self.subTest(data=data), self.assertRaises((ValueError, TypeError)):
                ModuleDispatchConfig.from_dict(data)

    def test_descriptors_import_without_device_or_kernel_packages(self):
        code = """
import importlib.abc, sys
class RejectRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'triton', 'deep_gemm', 'flashinfer') or fullname.startswith('rtp_llm.ops'):
            raise AssertionError('eager runtime import: ' + fullname)
sys.meta_path.insert(0, RejectRuntime())
import rtp_llm.models_py.pluggable.factory
import rtp_llm.models_py.pluggable.registry
from rtp_llm.device.device_type import DeviceType
assert DeviceType.Ppu.value == 5
from rtp_llm.utils.import_util import import_optional_internal_source_entrypoint
import_optional_internal_source_entrypoint('models_py')
"""
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

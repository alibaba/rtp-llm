"""Factory-to-initialize lifecycle contracts; no hardware qualification claim."""

import unittest
from types import SimpleNamespace

from rtp_llm.models_py.pluggable.lifecycle import initialize_model
from rtp_llm.models_py.pluggable.resources import ResourcePlan
from rtp_llm.models_py.pluggable.test.test_factory import (
    ActualModule,
    context_for,
    impl_for,
    registry_for,
    request_for,
)


class InitModule(ActualModule):
    def __init__(self, weight, build_ctx):
        super().__init__(weight, build_ctx)
        self.module_build_context = build_ctx
        self.initializations = 0
        self.action = lambda: True

    def initialize(self, resources):
        self.initializations += 1
        return self.action()


def build_model(*, build_ctx, request, weight):
    return InitModule(weight, build_ctx)


class InitAdapter:
    def root_request(self, selection):
        return request_for()

    def plan_resources(self, selection, bindings):
        return ResourcePlan('{"cache":"engine-owned"}')

    def validate_resources(self, model, resources, context):
        if resources.cache is not model.weight:
            raise ValueError("Resource binding differs from the frozen plan")

    def validate_initialized_model(self, model, resources, context):
        self.validate_resources(model, resources, context)


class Loader:
    def __init__(self):
        self.config = SimpleNamespace(weight_preparation=None)

    def get_load_config(self):
        return self.config


class InitializationTest(unittest.TestCase):
    def make_model(self, *, loaded=True):
        ctx = context_for(registry_for(impl_for(builder=__name__ + ":build_model")))
        ctx.model_adapter = InitAdapter()
        ctx.prepare([request_for()])
        ctx.verify_protocol()
        loader = Loader()
        ctx.configure_weight_loader(loader)
        if loaded:
            ctx.finish_weight_loading(loader)
        root = ctx.factory.build(request_for(), weight=object())
        return root, ctx, SimpleNamespace(cache=root.weight)

    def test_capability_omission_does_not_bypass_real_finalization(self):
        model, ctx, resources = self.make_model()
        self.assertFalse(hasattr(model, "get_execution_capabilities"))
        self.assertTrue(initialize_model(model, resources))
        self.assertEqual(ctx.state, "closed")
        self.assertEqual(model.initializations, 1)

    def test_missing_weight_consumption_rejects_before_initialize(self):
        model, ctx, resources = self.make_model(loaded=False)
        with self.assertRaisesRegex(RuntimeError, "consumed weight"):
            initialize_model(model, resources)
        self.assertEqual(model.initializations, 0)
        self.assertEqual(ctx.state, "failed")

    def test_warmup_then_executor_rebind_keeps_construction_closed(self):
        model, ctx, resources = self.make_model()
        initialize_model(model, resources)
        initial_bindings = ctx.bindings
        self.assertTrue(initialize_model(model, resources))
        self.assertEqual(model.initializations, 2)
        self.assertEqual(ctx.state, "closed")
        self.assertEqual(ctx.bindings, initial_bindings)
        with self.assertRaisesRegex(RuntimeError, "forbidden in state closed"):
            ctx.factory.build(request_for(), weight=model.weight)

    def test_rebinding_checks_resources_and_actual_tree_again(self):
        for corrupt in ("resources", "tree"):
            model, ctx, resources = self.make_model()
            initialize_model(model, resources)
            if corrupt == "resources":
                resources.cache = object()
            else:
                ctx._instance_refs.clear()
            with self.assertRaises((ValueError, RuntimeError)):
                initialize_model(model, resources)
            self.assertEqual(model.initializations, 1)
            self.assertEqual(ctx.state, "failed")

    def test_failed_second_initialize_poisons_context(self):
        model, ctx, resources = self.make_model()
        initialize_model(model, resources)
        model.action = lambda: False
        with self.assertRaisesRegex(RuntimeError, "did not succeed"):
            initialize_model(model, resources)
        self.assertEqual(ctx.state, "failed")
        with self.assertRaisesRegex(RuntimeError, "verified factory context"):
            initialize_model(model, resources)

    def test_invalid_resources_reject_before_initialize(self):
        model, ctx, _ = self.make_model()
        with self.assertRaisesRegex(ValueError, "Resource binding"):
            initialize_model(model, SimpleNamespace(cache=object()))
        self.assertEqual(model.initializations, 0)
        self.assertEqual(ctx.state, "failed")

    def test_removed_context_cannot_turn_bound_root_into_legacy_model(self):
        model, ctx, resources = self.make_model()
        del model.module_build_context
        with self.assertRaisesRegex(RuntimeError, "verified factory context"):
            initialize_model(model, resources)
        self.assertEqual(ctx.state, "failed")

    def test_initialize_failure_and_exception_poison_context(self):
        for action in (
            lambda: False,
            lambda: (_ for _ in ()).throw(ValueError("init failed")),
        ):
            model, ctx, resources = self.make_model()
            model.action = action
            with self.assertRaises((RuntimeError, ValueError)):
                initialize_model(model, resources)
            self.assertEqual(ctx.state, "failed")

    def test_initialize_cannot_replace_context_or_factory_binding(self):
        for attribute, value in (("module_build_context", None), ("weight", object())):
            model, ctx, resources = self.make_model()
            model.action = lambda: (setattr(model, attribute, value) or True)
            with self.assertRaises((RuntimeError, ValueError)):
                initialize_model(model, resources)
            self.assertEqual(ctx.state, "failed")

    def test_missing_plan_consumption_and_early_close_are_rejected(self):
        for action in ("missing", "early_close"):
            model, ctx, resources = self.make_model()
            if action == "missing":
                ctx._built.clear()
            else:
                model.action = lambda: (ctx.close() or True)
            with self.assertRaises(RuntimeError):
                initialize_model(model, resources)
            self.assertEqual(ctx.state, "failed")

    def test_unbound_declared_context_and_legacy_entry(self):
        self.assertTrue(initialize_model(LegacyModel(), object()))
        model, ctx, resources = self.make_model()
        unbound = InitModule(model.weight, ctx)
        with self.assertRaisesRegex(RuntimeError, "no factory root binding"):
            initialize_model(unbound, resources)


class LegacyModel:
    def initialize(self, resources):
        return True


if __name__ == "__main__":
    unittest.main()

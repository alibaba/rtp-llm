import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_DSV4_DIR = Path(__file__).resolve().parents[1]
_PROVIDER_SPEC = importlib.util.spec_from_file_location(
    "dsv4_platform_provider_under_test", _DSV4_DIR / "platform_provider.py"
)
assert _PROVIDER_SPEC is not None and _PROVIDER_SPEC.loader is not None
_PROVIDER_MODULE = importlib.util.module_from_spec(_PROVIDER_SPEC)
sys.modules[_PROVIDER_SPEC.name] = _PROVIDER_MODULE
_PROVIDER_SPEC.loader.exec_module(_PROVIDER_MODULE)

DefaultDsv4PlatformProvider = _PROVIDER_MODULE.DefaultDsv4PlatformProvider
Dsv4PlatformProviderRegistry = _PROVIDER_MODULE.Dsv4PlatformProviderRegistry
Dsv4ProviderCapability = _PROVIDER_MODULE.Dsv4ProviderCapability
Dsv4AttentionLayout = _PROVIDER_MODULE.Dsv4AttentionLayout
resolve_dsv4_attention_layout = _PROVIDER_MODULE.resolve_dsv4_attention_layout
build_dsv4_fp8_linear = _PROVIDER_MODULE.build_dsv4_fp8_linear
build_dsv4_wo_a_fp8_linear = _PROVIDER_MODULE.build_dsv4_wo_a_fp8_linear
build_dsv4_bf16_fp32_linear = _PROVIDER_MODULE.build_dsv4_bf16_fp32_linear
run_dsv4_fp8_mqa_logits = _PROVIDER_MODULE.run_dsv4_fp8_mqa_logits
build_dsv4_fp4_linear = _PROVIDER_MODULE.build_dsv4_fp4_linear
prepare_dsv4_fp4_weight_scale = _PROVIDER_MODULE.prepare_dsv4_fp4_weight_scale


class _Provider:
    name = "test-provider"
    capabilities = frozenset(
        {
            Dsv4ProviderCapability.BLOCK,
            Dsv4ProviderCapability.TRANSFORMER,
        }
    )

    def build_block(self, default_factory, *args, **kwargs):
        return ("block", default_factory, args, kwargs)

    def build_transformer(self, default_factory, *args, **kwargs):
        return ("transformer", default_factory, args, kwargs)


class _FalseyProvider(_Provider):
    name = "falsey-provider"

    def __bool__(self):
        return False


class _BlockOnlyProvider:
    name = "block-only"
    capabilities = frozenset({Dsv4ProviderCapability.BLOCK})

    def build_block(self, default_factory, *args, **kwargs):
        return default_factory(*args, **kwargs)


class _InvalidProvider:
    name = "invalid"
    capabilities = frozenset({Dsv4ProviderCapability.TRANSFORMER})


class _AttentionProvider(_Provider):
    capabilities = frozenset(
        {
            Dsv4ProviderCapability.BLOCK,
            Dsv4ProviderCapability.TRANSFORMER,
            Dsv4ProviderCapability.ATTENTION,
            Dsv4ProviderCapability.MOE,
        }
    )
    attention_layout = Dsv4AttentionLayout.FLAT

    def build_attention(self, default_factory, *args, **kwargs):
        return ("attention", default_factory, args, kwargs)

    def build_moe(self, default_factory, *args, **kwargs):
        return ("moe", default_factory, args, kwargs)


class _Fp8Provider(_Provider):
    capabilities = _Provider.capabilities | {
        Dsv4ProviderCapability.FP8_LINEAR,
    }

    def build_fp8_linear(self, default_factory, *args, **kwargs):
        return ("provider-fp8", default_factory, args, kwargs)


class _WoAFp8Provider(_Provider):
    capabilities = _Provider.capabilities | {
        Dsv4ProviderCapability.WO_A_FP8_LINEAR,
    }

    def build_wo_a_fp8_linear(self, default_factory, *args, **kwargs):
        return ("provider-wo-a-fp8", default_factory, args, kwargs)


class _Bf16Fp32Provider(_Provider):
    capabilities = _Provider.capabilities | {
        Dsv4ProviderCapability.BF16_FP32_LINEAR,
    }

    def run_bf16_fp32_linear(self, default_factory, *args, **kwargs):
        return ("provider-bf16-fp32", default_factory, args, kwargs)


class _Fp8MqaLogitsProvider(_Provider):
    capabilities = _Provider.capabilities | {
        Dsv4ProviderCapability.FP8_MQA_LOGITS,
    }

    def run_fp8_mqa_logits(self, default_factory, *args, **kwargs):
        return ("provider-fp8-mqa-logits", default_factory, args, kwargs)


class _Fp4Provider(_Provider):
    capabilities = _Provider.capabilities | {
        Dsv4ProviderCapability.FP4_LINEAR,
    }

    def prepare_fp4_weight_scale(self, default_factory, *args, **kwargs):
        return ("provider-fp4-scale", default_factory, args, kwargs)

    def build_fp4_linear(self, default_factory, *args, **kwargs):
        return ("provider-fp4-linear", default_factory, args, kwargs)


class Dsv4PlatformProviderRegistryTest(unittest.TestCase):
    def test_deterministic_hc_requires_provider_and_preserves_outputs(self):
        class HCProvider(_Provider):
            capabilities = _Provider.capabilities | {Dsv4ProviderCapability.HC_PRENORM}

            def run_hc_prenorm(self, *args, **kwargs):
                return args, kwargs

        original_registry = _PROVIDER_MODULE._PROVIDER_REGISTRY
        try:
            registry = Dsv4PlatformProviderRegistry()
            _PROVIDER_MODULE._PROVIDER_REGISTRY = registry
            tensors = tuple(object() for _ in range(4))
            with self.assertRaisesRegex(RuntimeError, "hc_prenorm"):
                _PROVIDER_MODULE.run_dsv4_hc_prenorm(*tensors, requested_splits=1)
            registry.register(HCProvider())
            args, kwargs = _PROVIDER_MODULE.run_dsv4_hc_prenorm(
                *tensors, requested_splits=1
            )
            for actual, expected in zip(args, tensors):
                self.assertIs(actual, expected)
            self.assertEqual(kwargs, {"requested_splits": 1})
            with self.assertRaisesRegex(RuntimeError, "construction started"):
                registry.register(HCProvider())
        finally:
            _PROVIDER_MODULE._PROVIDER_REGISTRY = original_registry

    def test_hc_capability_requires_callable(self):
        class InvalidHCProvider(_Provider):
            capabilities = _Provider.capabilities | {Dsv4ProviderCapability.HC_PRENORM}

        with self.assertRaisesRegex(TypeError, "run_hc_prenorm"):
            Dsv4PlatformProviderRegistry().register(InvalidHCProvider())

    def test_fp4_scale_and_linear_dispatch_are_owned_together(self):
        original_registry = _PROVIDER_MODULE._PROVIDER_REGISTRY
        try:
            _PROVIDER_MODULE._PROVIDER_REGISTRY = Dsv4PlatformProviderRegistry()
            sentinel = object()
            self.assertIs(
                prepare_dsv4_fp4_weight_scale(lambda value: value, sentinel),
                sentinel,
            )
            self.assertIs(
                build_dsv4_fp4_linear(lambda value: value, sentinel), sentinel
            )

            registry = Dsv4PlatformProviderRegistry()
            registry.register(_Fp4Provider())
            _PROVIDER_MODULE._PROVIDER_REGISTRY = registry
            prepared = prepare_dsv4_fp4_weight_scale(
                lambda value: value, sentinel, groups=2
            )
            linear = build_dsv4_fp4_linear(
                lambda value: value, sentinel, out_features=128
            )
            self.assertEqual(prepared[0], "provider-fp4-scale")
            self.assertEqual(linear[0], "provider-fp4-linear")
            self.assertIs(prepared[2][0], sentinel)
            self.assertEqual(prepared[3]["groups"], 2)
            self.assertEqual(linear[3]["out_features"], 128)
        finally:
            _PROVIDER_MODULE._PROVIDER_REGISTRY = original_registry

    def test_fp8_linear_dispatch_is_explicit_and_preserves_default(self):
        original_registry = _PROVIDER_MODULE._PROVIDER_REGISTRY
        try:
            _PROVIDER_MODULE._PROVIDER_REGISTRY = Dsv4PlatformProviderRegistry()
            sentinel = object()
            self.assertIs(
                build_dsv4_fp8_linear(lambda value: value, sentinel), sentinel
            )

            registry = Dsv4PlatformProviderRegistry()
            provider = _Fp8Provider()
            registry.register(provider)
            _PROVIDER_MODULE._PROVIDER_REGISTRY = registry
            result = build_dsv4_fp8_linear(lambda value: value, sentinel, flag=True)
            self.assertEqual(result[0], "provider-fp8")
            self.assertIs(result[2][0], sentinel)
            self.assertTrue(result[3]["flag"])
        finally:
            _PROVIDER_MODULE._PROVIDER_REGISTRY = original_registry

    def test_wo_a_fp8_dispatch_is_separate_and_explicit(self):
        original_registry = _PROVIDER_MODULE._PROVIDER_REGISTRY
        try:
            _PROVIDER_MODULE._PROVIDER_REGISTRY = Dsv4PlatformProviderRegistry()
            sentinel = object()
            self.assertIs(
                build_dsv4_wo_a_fp8_linear(lambda value, **_: value, sentinel),
                sentinel,
            )

            registry = Dsv4PlatformProviderRegistry()
            registry.register(_WoAFp8Provider())
            _PROVIDER_MODULE._PROVIDER_REGISTRY = registry
            result = build_dsv4_wo_a_fp8_linear(
                lambda *_args, **_kwargs: self.fail("default factory used"),
                sentinel,
                groups=1,
            )
            self.assertEqual(result[0], "provider-wo-a-fp8")
            self.assertIs(result[2][0], sentinel)
            self.assertEqual(result[3]["groups"], 1)
        finally:
            _PROVIDER_MODULE._PROVIDER_REGISTRY = original_registry

    def test_bf16_fp32_dispatch_is_separate_and_explicit(self):
        original_registry = _PROVIDER_MODULE._PROVIDER_REGISTRY
        try:
            _PROVIDER_MODULE._PROVIDER_REGISTRY = Dsv4PlatformProviderRegistry()
            sentinel = object()
            self.assertIs(
                build_dsv4_bf16_fp32_linear(lambda value, **_: value)(sentinel),
                sentinel,
            )

            registry = Dsv4PlatformProviderRegistry()
            registry.register(_Bf16Fp32Provider())
            _PROVIDER_MODULE._PROVIDER_REGISTRY = registry
            result = build_dsv4_bf16_fp32_linear(
                lambda *_args, **_kwargs: self.fail("default factory used"),
            )(sentinel, weight="weight")
            self.assertEqual(result[0], "provider-bf16-fp32")
            self.assertIs(result[2][0], sentinel)
            self.assertEqual(result[3]["weight"], "weight")
        finally:
            _PROVIDER_MODULE._PROVIDER_REGISTRY = original_registry

    def test_fp8_mqa_logits_dispatch_is_separate_and_explicit(self):
        original_registry = _PROVIDER_MODULE._PROVIDER_REGISTRY
        try:
            _PROVIDER_MODULE._PROVIDER_REGISTRY = Dsv4PlatformProviderRegistry()
            sentinel = object()
            self.assertIs(
                run_dsv4_fp8_mqa_logits(lambda value, **_: value, sentinel),
                sentinel,
            )

            registry = Dsv4PlatformProviderRegistry()
            registry.register(_Fp8MqaLogitsProvider())
            _PROVIDER_MODULE._PROVIDER_REGISTRY = registry
            result = run_dsv4_fp8_mqa_logits(
                lambda *_args, **_kwargs: self.fail("default factory used"),
                sentinel,
                clean_logits=False,
                max_seqlen_k=1024,
            )
            self.assertEqual(result[0], "provider-fp8-mqa-logits")
            self.assertIs(result[2][0], sentinel)
            self.assertFalse(result[3]["clean_logits"])
            self.assertEqual(result[3]["max_seqlen_k"], 1024)
        finally:
            _PROVIDER_MODULE._PROVIDER_REGISTRY = original_registry

    def test_default_provider_delegates_arguments_and_exceptions_exactly(self):
        provider = DefaultDsv4PlatformProvider()
        sentinel = object()

        def factory(*args, **kwargs):
            self.assertEqual(args, ("positional",))
            self.assertEqual(kwargs, {"sentinel": sentinel, "count": 3})
            return sentinel

        self.assertIs(
            provider.build_block(factory, "positional", sentinel=sentinel, count=3),
            sentinel,
        )
        self.assertIs(
            provider.build_transformer(
                factory, "positional", sentinel=sentinel, count=3
            ),
            sentinel,
        )

        failure = ValueError("factory failure")

        def failing_factory(*args, **kwargs):
            raise failure

        with self.assertRaises(ValueError) as caught:
            provider.build_block(failing_factory)
        self.assertIs(caught.exception, failure)

    def test_capability_query_does_not_freeze_registration(self):
        registry = Dsv4PlatformProviderRegistry()
        self.assertEqual(
            registry.capabilities(),
            frozenset(
                {
                    Dsv4ProviderCapability.BLOCK,
                    Dsv4ProviderCapability.TRANSFORMER,
                }
            ),
        )
        provider = _Provider()
        registry.register(provider)
        self.assertIs(
            registry.resolve(
                {
                    Dsv4ProviderCapability.BLOCK,
                    Dsv4ProviderCapability.TRANSFORMER,
                }
            ),
            provider,
        )

    def test_falsey_provider_identity_is_preserved(self):
        default_provider = _FalseyProvider()
        default_registry = Dsv4PlatformProviderRegistry(default_provider)
        self.assertIs(
            default_registry.resolve({Dsv4ProviderCapability.BLOCK}),
            default_provider,
        )

        registered_provider = _FalseyProvider()
        registered_registry = Dsv4PlatformProviderRegistry()
        registered_registry.register(registered_provider)
        self.assertIs(
            registered_registry.resolve({Dsv4ProviderCapability.TRANSFORMER}),
            registered_provider,
        )

    def test_second_registration_is_rejected(self):
        registry = Dsv4PlatformProviderRegistry()
        registry.register(_Provider())
        with self.assertRaisesRegex(RuntimeError, "already registered"):
            registry.register(_Provider())

    def test_registration_after_construction_resolution_is_rejected(self):
        registry = Dsv4PlatformProviderRegistry()
        registry.resolve({Dsv4ProviderCapability.BLOCK})
        with self.assertRaisesRegex(RuntimeError, "construction started"):
            registry.register(_Provider())

    def test_missing_capability_fails_before_construction(self):
        registry = Dsv4PlatformProviderRegistry()
        registry.register(_BlockOnlyProvider())
        with self.assertRaisesRegex(RuntimeError, "transformer"):
            registry.resolve({Dsv4ProviderCapability.TRANSFORMER})

    def test_declared_capability_requires_a_factory(self):
        registry = Dsv4PlatformProviderRegistry()
        with self.assertRaisesRegex(TypeError, "build_transformer"):
            registry.register(_InvalidProvider())

    def test_attention_capability_requires_explicit_layout(self):
        provider = _AttentionProvider()
        registry = Dsv4PlatformProviderRegistry()
        registry.register(provider)
        self.assertEqual(provider.attention_layout, Dsv4AttentionLayout.FLAT)

        class MissingLayout(_AttentionProvider):
            attention_layout = None

        with self.assertRaisesRegex(ValueError, "attention_layout"):
            Dsv4PlatformProviderRegistry().register(MissingLayout())

    def test_non_attention_provider_cannot_override_default_layout(self):
        class BlockTransformerOnly:
            capabilities = frozenset(
                {
                    Dsv4ProviderCapability.BLOCK,
                    Dsv4ProviderCapability.TRANSFORMER,
                }
            )
            attention_layout = Dsv4AttentionLayout.PADDED

        self.assertEqual(
            resolve_dsv4_attention_layout(BlockTransformerOnly()),
            Dsv4AttentionLayout.FLAT,
        )


class Dsv4InstanceAdapterTest(unittest.TestCase):
    def test_explicit_operator_adapters_bypass_global_registry(self):
        def adapter(label):
            methods = {
                method: (lambda *args, _label=label, **kwargs: (_label, args, kwargs))
                for method in (
                    "build_fp8_linear",
                    "build_wo_a_fp8_linear",
                    "run_bf16_fp32_linear",
                    "run_fp8_mqa_logits",
                    "prepare_fp4_weight_scale",
                    "build_fp4_linear",
                    "run_hc_prenorm",
                )
            }
            return SimpleNamespace(
                name=label, capabilities=frozenset(Dsv4ProviderCapability), **methods
            )

        first, second = adapter("first"), adapter("second")
        with patch.object(
            _PROVIDER_MODULE._PROVIDER_REGISTRY,
            "resolve",
            side_effect=AssertionError("global selection used"),
        ):
            for name in (
                "build_dsv4_fp8_linear",
                "build_dsv4_wo_a_fp8_linear",
                "run_dsv4_fp8_mqa_logits",
                "prepare_dsv4_fp4_weight_scale",
                "build_dsv4_fp4_linear",
                "run_dsv4_hc_prenorm",
            ):
                fn = getattr(_PROVIDER_MODULE, name)
                for provider in (first, second, first):
                    value = fn("sentinel", platform_provider=provider, option=7)
                    self.assertEqual(
                        value, (provider.name, ("sentinel",), {"option": 7})
                    )


class Dsv4PlatformProviderWiringTest(unittest.TestCase):
    def test_model_and_transformer_use_the_provider_seam(self):
        dsv4_dir = _DSV4_DIR
        transformer_source = (dsv4_dir / "transformer.py").read_text()
        model_source = (
            dsv4_dir.parent.parent / "model_desc" / "deepseek_v4_model.py"
        ).read_text()

        transformer_tree = ast.parse(transformer_source)
        model_tree = ast.parse(model_source)

        model_class = next(
            node
            for node in model_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "DeepSeekV4Model"
        )
        model_init = next(
            node
            for node in model_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        model_resolves = [
            node
            for node in ast.walk(model_init)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "resolve_dsv4_platform_provider"
        ]
        self.assertEqual(len(model_resolves), 1)
        provider_guard = next(
            node
            for node in model_init.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "platform_provider"
        )
        # Explicit providers, including falsey objects, bypass legacy lookup.
        self.assertIsInstance(provider_guard.test.ops[0], ast.Is)
        self.assertIsNone(provider_guard.test.comparators[0].value)
        self.assertIn(model_resolves[0], list(ast.walk(provider_guard)))
        self.assertFalse(provider_guard.orelse)
        self.assertTrue(
            any(
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr == "_platform_provider"
                    for target in node.targets
                )
                and isinstance(node.value, ast.Name)
                and node.value.id == "platform_provider"
                for node in model_init.body
            )
        )

        initialize_impl = next(
            node
            for node in model_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "_initialize_impl"
        )
        transformer_builds = [
            node
            for node in ast.walk(initialize_impl)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "build_transformer"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "_platform_provider"
        ]
        self.assertEqual(len(transformer_builds), 1)
        injected_keyword = next(
            keyword
            for keyword in transformer_builds[0].keywords
            if keyword.arg == "platform_provider"
        )
        self.assertIsInstance(injected_keyword.value, ast.Attribute)
        self.assertEqual(injected_keyword.value.attr, "_platform_provider")

        transformer_class = next(
            node
            for node in transformer_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "V4Transformer"
        )
        transformer_init = next(
            node
            for node in transformer_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        provider_guard = next(
            node
            for node in transformer_init.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "platform_provider"
        )
        self.assertIsInstance(provider_guard.test.ops[0], ast.Is)
        self.assertIsInstance(provider_guard.test.comparators[0], ast.Constant)
        self.assertIsNone(provider_guard.test.comparators[0].value)
        self.assertEqual(
            sum(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "resolve_dsv4_platform_provider"
                for node in ast.walk(provider_guard)
            ),
            1,
        )
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "resolve_dsv4_platform_provider"
                for statement in provider_guard.orelse
                for node in ast.walk(statement)
            )
        )


class BoundLinearTest(unittest.TestCase):
    def test_bound_operation_retains_its_provider_without_reselection(self):
        first, second = _Bf16Fp32Provider(), _Bf16Fp32Provider()
        a = build_dsv4_bf16_fp32_linear(object(), platform_provider=first)
        b = build_dsv4_bf16_fp32_linear(object(), platform_provider=second)
        with patch.object(
            _PROVIDER_MODULE,
            "_normalize_capabilities",
            side_effect=AssertionError("reselection"),
        ):
            self.assertIs(a.func.__self__, first)
            self.assertIs(b.func.__self__, second)
            self.assertEqual(a("x")[0], "provider-bf16-fp32")
            self.assertEqual(b("y")[0], "provider-bf16-fp32")


if __name__ == "__main__":
    unittest.main()

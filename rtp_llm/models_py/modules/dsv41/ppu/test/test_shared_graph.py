"""N4 stdlib boundary checks; these do not qualify device computation."""

import ast
import runpy
import types
import unittest
from pathlib import Path

PPU = Path(__file__).resolve().parents[1]


def _class(path, name, namespace):
    tree = ast.parse(path.read_text(), feature_version=(3, 10))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class MetadataProtocol:
    def __init__(self, config, native, tagged, **kwargs):
        self.bound = config, native, tagged, kwargs


class SharedGraphBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.impl_class = _class(PPU / "shared_graph.py", "SharedPPUDecodeGraphImpl", {
            "PPUDecodeGraphImpl": MetadataProtocol,
            "as_tagged_attention_inputs": lambda value: value,
        })
        ns = types.SimpleNamespace
        self.attn = ns(is_prefill=False, input_lengths=ns(numel=lambda: 1))
        self.inputs = ns(attention_inputs=self.attn, input_ids=ns(numel=lambda: 1),
                         multimodal_inputs=ns(multimodal_features=[]))
        self.validated = []
        self.model = ns(
            _is_decode_role=True, _is_speculative=False, kv_cache=object(),
            _v4_args=ns(dp_size=8, ep_size=8, tp_size=1, max_seq_len=512),
            shared_weight_views=ns(config={"origin": "ModelWeights"}),
            shared_graph_runtime=ns(
                device="ppu", max_batch_size=1, grouped=None,
                validate=lambda: self.validated.append(True),
            ),
        )

    def test_existing_metadata_protocol_receives_native_resources(self):
        impl = self.impl_class(self.model, self.inputs)
        config, native, tagged, kwargs = impl.bound
        self.assertIs(native, self.model.kv_cache)
        self.assertIs(tagged, self.attn)
        self.assertEqual(kwargs, {"device": "ppu", "batch_size": 1})
        self.assertEqual(config["max_position_embeddings"], 512)
        self.assertEqual(self.validated, [True])

    def test_n4_excludes_prefill_verify_other_topology_and_bucket(self):
        for obj, name, bad in (
            (self.model, "_is_decode_role", False),
            (self.model, "_is_speculative", True),
            (self.attn, "is_prefill", True),
            (self.model._v4_args, "dp_size", 1),
            (self.model._v4_args, "ep_size", 1),
            (self.model._v4_args, "tp_size", 8),
            (self.inputs, "input_ids", types.SimpleNamespace(numel=lambda: 2)),
            (self.inputs.multimodal_inputs, "multimodal_features", [object()]),
        ):
            original = getattr(obj, name)
            try:
                setattr(obj, name, bad)
                with self.subTest(name=name), self.assertRaises(ValueError):
                    self.impl_class(self.model, self.inputs)
            finally:
                setattr(obj, name, original)
        self.attn.is_target_verify = True
        with self.assertRaises(ValueError):
            self.impl_class(self.model, self.inputs)

    def test_existing_bucket_capacity_and_single_token_shape_are_distinct(self):
        self.model.shared_graph_runtime.max_batch_size = 2
        self.attn.input_lengths.numel = lambda: 2
        self.inputs.input_ids.numel = lambda: 2
        self.assertEqual(self.impl_class(self.model, self.inputs).bound[3]["batch_size"], 2)
        self.inputs.input_ids.numel = lambda: 4
        with self.assertRaises(ValueError):
            self.impl_class(self.model, self.inputs)

    def test_source_parent_bootstrap_defers_native_bindings(self):
        namespace = runpy.run_path(str(PPU / "test/native_shared_graph_role.py"))
        self.assertTrue(callable(namespace["native_smoke_model_class"]))
        self.assertNotIn("DeepSeekV41SharedPPUModel", namespace)

    def test_capture_body_has_no_host_value_reads_or_old_executor(self):
        banned = {"item", "tolist", "cpu", "numpy", "prefetch", "dispatch", "load_ppu_weights"}
        for name in ("shared_graph_runtime.py", "shared_graph_attention.py"):
            tree = ast.parse((PPU / name).read_text(), feature_version=(3, 10))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    symbol = getattr(node.func, "attr", getattr(node.func, "id", ""))
                    self.assertNotIn(symbol, banned, (name, node.lineno, symbol))
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn("graph_decode", node.module or "")
        transformer = ast.parse((PPU / "shared_transformer.py").read_text())
        cls = next(n for n in transformer.body if isinstance(n, ast.ClassDef))
        self.assertNotIn("forward_decode", {getattr(n, "name", "") for n in cls.body})

    def test_replay_count_is_fixed_and_reference_uses_eager_dispatch(self):
        tree = ast.parse((PPU / "tools/driver_shared_graph_dp8.py").read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        replays = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "replay"]
        self.assertEqual(len(replays), 1)
        # The replay statement sits in the loop over the fixed two tokens;
        # there is no argv-driven replay grid and no old full-model runner.
        self.assertNotIn("argparse", (PPU / "tools/driver_shared_graph_dp8.py").read_text())
        forwards = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "forward"]
        self.assertTrue(any(len(n.args) == 1 for n in forwards), "independent eager arm missing")
        self.assertTrue(any(len(n.args) == 2 for n in forwards), "graph-safe arm missing")

    def test_server_entry_consumes_private_sentinel_before_public_parser(self):
        import os
        import sys
        from unittest.mock import patch

        namespace = runpy.run_path(str(PPU / "test/native_shared_graph_role.py"))
        calls = []
        registry = types.ModuleType("rtp_llm.model_factory_register")
        registry.register_lazy_model = lambda *args: calls.append(("register", args))
        provider = types.ModuleType("rtp_llm.models_py.modules.dsv41.ppu.backend")
        selector_globals = {"os": os, "MODEL_PATH_ENV": "RTP_V41_PPU_MODEL_PATH"}
        selector = next(n for n in ast.parse((PPU / "backend.py").read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == "_native_graph_smoke_selected")
        exec(compile(ast.Module(body=[selector], type_ignores=[]), "N4 provider selector", "exec"), selector_globals)
        provider._native_graph_smoke_selected = selector_globals[selector.name]
        server = types.ModuleType("rtp_llm.start_server")
        server.main = lambda: calls.append(("start", list(sys.argv), os.environ["MODEL_TYPE"]))
        entry = "native_shared_graph_role.py"
        with patch.dict(sys.modules, {
            registry.__name__: registry, server.__name__: server, provider.__name__: provider,
        }), patch.dict(os.environ, {
            "MODEL_TYPE": "deepseek_v41", "RTP_V41_N4_NATIVE_GRAPH_SMOKE": "1",
            "RTP_V41_BACKEND": "ppu", "RTP_V41_PPU_MODEL_PATH": "shared", "ROLE_TYPE": "DECODE",
            "RTP_V41_PPU_MOE_BACKEND": "megamoe", "RTP_V41_PPU_MOE_PRECISION": "w4a4",
        }), patch.object(sys, "argv", [entry, "--server"]):
            namespace["main"]()
        self.assertEqual(calls, [("start", [entry], "deepseek_v41")])

    def test_fixture_uses_public_disabled_cp_default_without_string_override(self):
        import argparse
        import logging
        import os
        import sys
        import typing
        from enum import IntEnum
        from unittest.mock import patch

        # Only the native enum is stubbed; parser, converter, argument
        # declaration, production preset and fixture statements are real source.
        class CPRotateMethod(IntEnum):
            DISABLED = 0
            ALL_GATHER = 1
            ALL_GATHER_WITH_OVERLAP = 2
            ALLTOALL = 3
            PREFILL_CP = 4

        root = Path(os.environ.get("RTP_V41_TEST_SOURCE_ROOT", str(PPU.parents[4])))
        args_root = root / "rtp_llm/server/server_args"
        namespace = {
            **vars(typing), "argparse": argparse, "logging": logging,
            "os": os, "sys": sys, "CPRotateMethod": CPRotateMethod,
        }
        for name in ("ConfigBinding", "EnvArgumentGroup", "EnvArgumentParser"):
            _class(args_root / "server_args.py", name, namespace)
        for filename in ("util.py", "parallel_group_args.py"):
            nodes = [n for n in ast.parse((args_root / filename).read_text()).body
                     if isinstance(n, ast.FunctionDef)]
            exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)
        convert = namespace["str2_cp_rotate_method"]
        self.assertIs(convert(CPRotateMethod.DISABLED), CPRotateMethod.DISABLED)
        with self.assertRaises(ValueError):
            convert("DISABLED")
        for name in ("ALL_GATHER", "ALL_GATHER_WITH_OVERLAP", "ALLTOALL", "PREFILL_CP"):
            self.assertIs(convert(name), getattr(CPRotateMethod, name))

        preset_globals = runpy.run_path(str(root / "rtp_llm/config/v41_runtime.py"))
        preset_globals.update(os=os, json=__import__("json"),
                              DEFAULT_PREFILL_WAIT_TIMEOUT_MS=1800000,
                              MAX_PREFILL_WAIT_TIMEOUT_MS=86400000)
        launch = root / "rtp_llm/tools/v41/launch.py"
        nodes = [n for n in ast.parse(launch.read_text()).body if isinstance(n, ast.FunctionDef)
                 and n.name in ("preset_env", "bounded_prefill_wait_timeout_ms")]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(launch), "exec"), preset_globals)
        precision_name = "rtp_llm.models_py.modules.dsv41.ppu.moe_precision"
        precision = types.ModuleType(precision_name)
        precision.__dict__.update(runpy.run_path(str(PPU / "moe_precision.py")))
        records = {}
        fixture = PPU / "test/native_shared_graph_role.py"
        main = next(n for n in ast.parse(fixture.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
        first = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "env" for t in n.targets))
        last = next(i for i, n in enumerate(main.body) if i > first and isinstance(n, ast.With))
        fixture_globals = {
            "os": os, "sys": sys, "preset_env": preset_globals["preset_env"],
            "reports_dir": Path("/fixture/ranks"), "output": Path("/fixture/output"),
            "runtime": Path("/fixture/runtime"), "port": 19090,
            "snapshot": {}, "checkpoint": {},
            "write_json": lambda path, value: records.update({path.name: value}),
        }
        with patch.dict(sys.modules, {precision_name: precision}), patch.dict(os.environ, {
            "RTP_V41_TEST_CHECKPOINT": "/fixture/checkpoint",
            "RTP_V41_PPU_MOE_BACKEND": "megamoe", "RTP_V41_PPU_MOE_PRECISION": "w4a4",
        }, clear=True):
            before = preset_globals["preset_env"](
                "decode", "/fixture/checkpoint", rdma=True, prefill_parallel="dp",
                max_seq_len=512, start_port=19090, ppu_model_path="shared", fake_balance_expert=False,
            )
            exec(compile(ast.Module(body=main.body[first:last], type_ignores=[]),
                         str(fixture), "exec"), fixture_globals)
        child = fixture_globals["env"]
        self.assertEqual(child["MODEL_TYPE"], "deepseek_v41")
        self.assertEqual(child["RTP_V41_N4_NATIVE_GRAPH_SMOKE"], "1")
        self.assertNotIn("CP_ROTATE_METHOD", before)
        self.assertNotIn("CP_ROTATE_METHOD", child)
        for key, value in {"TP_SIZE": "1", "DP_SIZE": "8", "EP_SIZE": "8",
                           "WORLD_SIZE": "8", "RTP_V41_ENP_SIZE": "8", "PREFILL_CP_SIZE": "1",
                           "PREFILL_CP_KV_CACHE_SHARDED": "0"}.items():
            self.assertEqual(child[key], value)
            self.assertEqual(child[key], before[key])
        parser = namespace["EnvArgumentParser"](add_help=False)
        parallel, ffn, cp = types.SimpleNamespace(), types.SimpleNamespace(), types.SimpleNamespace(prefill_cp_size=0)
        namespace["init_parallel_group_args"](parser, parallel, ffn, cp)
        parser.set_root_config(types.SimpleNamespace())
        with patch.dict(os.environ, child, clear=True), patch.object(sys, "argv", ["server"]):
            parsed = parser.parse_args()
        self.assertIs(parsed.cp_rotate_method, CPRotateMethod.DISABLED)
        self.assertIs(cp.method, CPRotateMethod.DISABLED)
        self.assertEqual((parallel.tp_size, parallel.dp_size, parallel.ep_size, cp.prefill_cp_size), (1, 8, 8, 1))
        self.assertEqual(records["launch.json"]["context_parallel"], {
            "method_environment": None, "method_default": "CPRotateMethod.DISABLED", "prefill_cp_size": 1,
        })
        self.assertEqual(records["launch.json"]["environment"]["PREFILL_CP_SIZE"], "1")
        self.assertEqual(records["launch.json"]["environment"]["MODEL_TYPE"], "deepseek_v41")
        self.assertEqual(records["launch.json"]["environment"]["RTP_V41_N4_NATIVE_GRAPH_SMOKE"], "1")

    def test_canonical_guard_and_actual_ppu_provider_selection(self):
        import os
        import sys
        from unittest.mock import patch

        root = Path(os.environ.get("RTP_V41_TEST_SOURCE_ROOT", str(PPU.parents[4])))
        runtime_name = "rtp_llm.config.v41_runtime"
        runtime = types.ModuleType(runtime_name)
        runtime.__dict__.update(runpy.run_path(str(root / "rtp_llm/config/v41_runtime.py")))
        public_name = "rtp_llm.models_py.modules.dsv41.backend"
        public = types.ModuleType(public_name)
        public_source = runpy.run_path(str(root / "rtp_llm/models_py/modules/dsv41/backend.py"))
        public.__dict__.update(public_source)
        precision_name = "rtp_llm.models_py.modules.dsv41.ppu.moe_precision"
        precision = types.ModuleType(precision_name)
        precision.__dict__.update(runpy.run_path(str(PPU / "moe_precision.py")))
        actual_device = {"ppu": True}
        device_name = "rtp_llm.device.device_type"
        device = types.ModuleType(device_name)
        device.get_device_type = lambda: "PPU" if actual_device["ppu"] else "CPU"
        device.is_ppu = lambda: actual_device["ppu"]
        shared_cls = type("NormalSharedModel", (), {})
        shared_weight = type("SharedWeight", (), {"framework_base_available": classmethod(lambda cls: True)})
        modules = {runtime_name: runtime, public_name: public, precision_name: precision, device_name: device}
        for suffix, symbol, value in (
            ("shared_model", "DeepSeekV41SharedPPUModel", shared_cls),
            ("shared_weight", "DeepSeekV41SharedPPUWeight", shared_weight),
        ):
            name = "rtp_llm.models_py.modules.dsv41.ppu." + suffix
            module = types.ModuleType(name)
            setattr(module, symbol, value)
            modules[name] = module
        role_name = "rtp_llm.models_py.modules.dsv41.ppu.test.native_shared_graph_role"
        role = types.ModuleType(role_name)
        role.__dict__.update(runpy.run_path(str(PPU / "test/native_shared_graph_role.py")))
        modules[role_name] = role
        ops = types.SimpleNamespace(**{name: (lambda *args, **kwargs: None) for name in public.OP_NAMES})
        ops_name = "rtp_llm.models_py.modules.dsv41.ppu.ops"
        ops_module = types.ModuleType(ops_name)
        ops_module.PPUOps = lambda **kwargs: ops
        modules[ops_name] = ops_module
        # Only imports/native handles are stubbed. Public runtime/guard,
        # V41Backend.validate/check_runtime and PPU get_backend are actual code.
        for name, symbols in {
            "deep_gemm": ("fp4_fp4_mega_moe", "get_symm_buffer_for_mega_moe",
                          "transform_weights_for_mega_moe", "mega_moe_pre_dispatch"),
            "deep_ep": ("Buffer",), "flash_mla": ("flash_mla_sparse_fwd",),
            "fast_hadamard_transform": ("hadamard_transform",),
        }.items():
            module = types.ModuleType(name)
            for symbol in symbols:
                setattr(module, symbol, lambda *args, **kwargs: None)
            modules[name] = module
        qualified = {
            "RTP_V41_BACKEND": "ppu", "RTP_V41_PPU_MODEL_PATH": "shared",
            "MODEL_TYPE": "deepseek_v41", "ROLE_TYPE": "DECODE", "RTP_V41_ENP_SIZE": "8",
            "RTP_V41_PPU_MOE_BACKEND": "megamoe", "RTP_V41_PPU_MOE_PRECISION": "w4a4",
            "RTP_V41_N4_NATIVE_GRAPH_SMOKE": "1",
        }
        ns = types.SimpleNamespace
        config = ns(
            model_args=ns(model_type="deepseek_v41", max_seq_len=512),
            role_config=ns(role_type="DECODE"),
            parallelism_config=ns(world_size=8, local_world_size=8, tp_size=1, dp_size=8, ep_size=8,
                                  prefill_cp_config=ns(method="DISABLED", prefill_cp_size=1, kv_cache_sharded=False)),
            py_hw_kernel_config=ns(enable_cuda_graph=False), sp_config=ns(type="NONE"),
            eplb_config=ns(eplb_mode="NONE", redundant_expert=0),
            moe_config=ns(use_deepep_low_latency=False, use_deepep_moe=True, use_all_gather=False),
        )
        lookups = []
        with patch.dict(sys.modules, modules), patch.dict(os.environ, qualified, clear=True):
            provider = runpy.run_path(str(PPU / "backend.py"))
            public.get_v41_backend = lambda name: (lookups.append(name), provider["get_backend"]())[1]
            chosen = provider["get_backend"]()
            self.assertEqual(chosen.name, "ppu")
            self.assertIs(chosen.weight_cls, shared_weight)
            self.assertTrue(issubclass(chosen.model_cls, shared_cls))
            self.assertEqual(chosen.model_cls.__name__, "NativeSharedGraphSmokeModel")
            config.model_args.model_type = "deepseek_v41_ppu_shared_graph_smoke"
            with self.assertRaisesRegex(ValueError, "text target"):
                runtime.validate_v41_server_config(config)
            self.assertEqual(lookups, [])
            config.model_args.model_type = "deepseek_v41"
            config.parallelism_config.local_world_size = 7
            with self.assertRaisesRegex(ValueError, "local_world_size=8"):
                runtime.validate_v41_server_config(config)
            self.assertEqual(lookups, [])
            config.parallelism_config.local_world_size = 8
            runtime.validate_v41_server_config(config)
            self.assertEqual(lookups, ["ppu"])
            mla = modules["flash_mla"].flash_mla_sparse_fwd
            del modules["flash_mla"].flash_mla_sparse_fwd
            with self.assertRaisesRegex(RuntimeError, "kernels missing"):
                runtime.validate_v41_server_config(config)
            modules["flash_mla"].flash_mla_sparse_fwd = mla
            normal = {k: v for k, v in qualified.items()
                      if k not in ("RTP_V41_N4_NATIVE_GRAPH_SMOKE", "RTP_V41_PPU_MODEL_PATH")}
            with patch.dict(os.environ, normal, clear=True):
                backend = provider["get_backend"]()
                self.assertIs(backend.model_cls, shared_cls)
                self.assertIs(backend.weight_cls, shared_weight)
            for key, value in (
                ("RTP_V41_N4_NATIVE_GRAPH_SMOKE", "0"), ("RTP_V41_N4_NATIVE_GRAPH_SMOKE", "true"),
                ("RTP_V41_N4_NATIVE_GRAPH_SMOKE", ""), ("RTP_V41_BACKEND", "cuda"),
                ("RTP_V41_PPU_MODEL_PATH", "legacy"), ("MODEL_TYPE", "deepseek_v41_ppu_shared_graph_smoke"),
                ("MODEL_TYPE", "deepseek_v41_dspark"), ("ROLE_TYPE", "PREFILL"),
                ("RTP_V41_PPU_MOE_BACKEND", "grouped"), ("RTP_V41_PPU_MOE_PRECISION", "w4a8"),
            ):
                invalid = {**qualified, key: value}
                with self.subTest(key=key, value=value), patch.dict(os.environ, invalid, clear=True):
                    with self.assertRaises(ValueError):
                        provider["get_backend"]()
            for key in ("RTP_V41_BACKEND", "RTP_V41_PPU_MOE_PRECISION"):
                missing = {name: value for name, value in qualified.items() if name != key}
                with self.subTest(missing=key), patch.dict(os.environ, missing, clear=True):
                    with self.assertRaises(ValueError):
                        provider["get_backend"]()
            actual_device["ppu"] = False
            with self.assertRaisesRegex(RuntimeError, "actual PPU device"):
                provider["get_backend"]()
            self.assertEqual(public_source["_backends"], {})


if __name__ == "__main__":
    unittest.main()

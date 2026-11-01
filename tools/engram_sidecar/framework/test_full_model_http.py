"""Preflight negative cases only; these fake files never run a model."""
import json
import io
import struct
import subprocess
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.engram_sidecar.framework.full_model_http import (
    EXTERNAL_ENV, NATIVE_LIBS, TEXT_FIELDS, chat_payload, checkpoint_preflight, launch_environment, runtime_preflight, main,
)


class FullModelPreflightTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        text = dict.fromkeys(TEXT_FIELDS, 1)
        text.update(num_hidden_layers=2, engram_layer_ids=[0, 1], engram_num_embeddings=[100, 101], engram_head_dim=64)
        self.config = {"text_config": text, "vision_config": {}, "quantization_config": {}, "dtype": "bfloat16",
                       "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2, "image_token_id": 3,
                       "architectures": ["DeepseekV41ForCausalLM"]}
        self.weights = {key: "weights.safetensors" for key in (
            "embed.weight", "head.weight", "norm.weight", "image_start", "image_newline", "image_end",
            "vision.fake", "aligner.fake")}
        for layer in range(2):
            for key in ("attn.fake", "ffn.fake", "engram.wkv.weight", "engram.wkv.scale", "engram.q_weight", "engram.k_weight"):
                self.weights[f"layers.{layer}.{key}"] = "weights.safetensors"
        (self.root / "encoding").mkdir()
        for name in ("tokenizer.json", "tokenizer_config.json", "encoding/encoding.py"):
            (self.root / name).write_text("{}")
        self.save()

    def save(self):
        (self.root / "config.json").write_text(json.dumps(self.config))
        (self.root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": self.weights}))
        self.write_header()

    def write_header(self, overrides=None, extra=None):
        header, offset = {}, 0
        for name in sorted(set(self.weights) | set(extra or [])):
            dtype, shape, count = "BF16", [1], 2
            if ".engram.embed." in name:
                layer = int(name.split(".")[1])
                width = 64 if name.endswith("weight") else 2
                dtype, shape = ("F8_E4M3" if width == 64 else "U8"), [100 + layer, width]
                count = shape[0] * shape[1]
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + count]}
            offset += count
        for name, fields in (overrides or {}).items():
            header[name].update(fields)
        data = json.dumps(header).encode()
        (self.root / "weights.safetensors").write_bytes(struct.pack("<Q", len(data)) + data + bytes(offset))

    def add_tables(self):
        for layer in (0, 1):
            for role in ("weight", "scale"):
                self.weights[f"layers.{layer}.engram.embed.{role}"] = "weights.safetensors"
        self.save()

    def test_local_requires_complete_exact_tables_and_checks_headers(self):
        with self.assertRaisesRegex(ValueError, "exactly every"):
            checkpoint_preflight(self.root, "local-pinned")
        self.add_tables()
        report = checkpoint_preflight(self.root, "local-pinned")
        self.assertTrue(report["consumer_has_table_entries"])
        self.assertEqual(len(report["shard_header_sha256"]), 1)
        self.write_header({"layers.0.engram.embed.weight": {"dtype": "U8"}})
        with self.assertRaisesRegex(ValueError, "dtype/shape"):
            checkpoint_preflight(self.root, "local-pinned")

    def test_external_refuses_hidden_header_table_and_unindexed_files(self):
        self.write_header(extra=["layers.0.engram.embed.weight"])
        with self.assertRaisesRegex(ValueError, "header/index"):
            checkpoint_preflight(self.root)
        self.save()
        (self.root / "extra.safetensors").touch()
        with self.assertRaisesRegex(ValueError, "unindexed"):
            checkpoint_preflight(self.root)

    def test_local_preflight_does_not_read_external_bundle(self):
        self.add_tables()
        with patch("tools.engram_sidecar.framework.full_model_http.runtime_preflight", return_value={}), patch(
            "tools.engram_sidecar.framework.full_model_http.cohort_preflight", side_effect=AssertionError("must not attach")), patch(
            "sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(main(["preflight", "--checkpoint", str(self.root), "--storage-mode", "local-pinned",
                                   "--model-revision", "test", "--cuda-graph"]), 0)
        report = json.loads(out.getvalue())
        self.assertNotIn("cohort", report)
        self.assertEqual(report["experiment_environment"]["DECODE_CAPTURE_CONFIG"], "1")

    def test_local_clears_external_env_with_same_normal_graph_configuration(self):
        args = SimpleNamespace(checkpoint=self.root, bundle=self.root / "bundle.json", port=18630,
                               max_allocation_bytes=128 << 30, budget_bytes=256 << 30, cuda_graph=True)
        external = launch_environment(args, {})
        args.storage_mode = "local-pinned"
        local = launch_environment(args, dict.fromkeys(EXTERNAL_ENV, "stale-sensitive-value"))
        for name in EXTERNAL_ENV:
            self.assertNotIn(name, local)
        self.assertEqual({k: v for k, v in external.items() if k not in EXTERNAL_ENV}, local)

    def test_metadata_screen_counts_geometry_without_weight_loading(self):
        report = checkpoint_preflight(self.root)
        self.assertEqual(report["engram_logical_bytes"], 201 * 66)
        self.assertEqual(report["full_layer_count"], 2)
        self.assertFalse(report["consumer_has_table_entries"])

    def test_storage_comparison_cannot_inherit_asymmetric_release_prefetch(self):
        for graph in (False, True):
            for inherited in ({}, {"DSV41_CPU_ENGRAM_LOOKUP": "1", "DSV41_ASYNC_ENGRAM_LOOKUP": "1"}):
                for storage in ("local-pinned", "external-egm"):
                    with self.subTest(graph=graph, inherited=inherited, storage=storage):
                        args = SimpleNamespace(checkpoint=self.root, bundle=self.root / "bundle.json", port=18630,
                                               max_allocation_bytes=128 << 30, budget_bytes=256 << 30,
                                               cuda_graph=graph, storage_mode=storage)
                        env = launch_environment(args, inherited)
                        self.assertEqual(env["DSV41_CPU_ENGRAM_LOOKUP"], "0")
                        self.assertEqual(env["DSV41_ASYNC_ENGRAM_LOOKUP"], "0")

    def test_component_checkpoint_rejected(self):
        self.config["fixture_component_only"] = True
        self.save()
        with self.assertRaisesRegex(ValueError, "not a full LLM"):
            checkpoint_preflight(self.root)

    def test_missing_chat_encoder_rejected(self):
        (self.root / "encoding/encoding.py").unlink()
        with self.assertRaisesRegex(ValueError, "encoding asset"):
            checkpoint_preflight(self.root)

    def test_wrong_hf_architecture_rejected(self):
        self.config["architectures"] = ["Qwen3ForCausalLM"]
        self.save()
        with self.assertRaisesRegex(ValueError, "architecture"):
            checkpoint_preflight(self.root)

    def test_consumer_full_tables_rejected(self):
        self.weights["layers.0.engram.embed.weight"] = "weights.safetensors"
        self.save()
        with self.assertRaisesRegex(ValueError, "must omit"):
            checkpoint_preflight(self.root)

    def test_incomplete_backbone_rejected(self):
        del self.weights["layers.1.ffn.fake"]
        self.save()
        with self.assertRaisesRegex(ValueError, "missing MoE"):
            checkpoint_preflight(self.root)

    def test_missing_local_engram_projection_rejected(self):
        del self.weights["layers.1.engram.k_weight"]
        self.save()
        with self.assertRaisesRegex(ValueError, "local Engram projection"):
            checkpoint_preflight(self.root)

    def test_missing_vision_weights_rejected(self):
        del self.weights["vision.fake"]
        self.save()
        with self.assertRaisesRegex(ValueError, "local ViT"):
            checkpoint_preflight(self.root)

    def test_unsafe_shard_rejected(self):
        self.weights["head.weight"] = "../outside.safetensors"
        self.save()
        with self.assertRaisesRegex(ValueError, "unsafe"):
            checkpoint_preflight(self.root)

    def test_runtime_rejects_only_config_and_compute_before_imports(self):
        libs = self.root / "rtp_llm/libs"
        libs.mkdir(parents=True)
        for name in NATIVE_LIBS[:2]:
            (libs / name).touch()
        with self.assertRaisesRegex(RuntimeError, "libth_transformer.so,libth_grammar_tokenizer_info.so"):
            runtime_preflight(self.root)

    def test_runtime_child_imports_real_service_entry_classes_with_gpu_hidden(self):
        libs = self.root / "rtp_llm/libs"
        libs.mkdir(parents=True)
        for name in NATIVE_LIBS:
            (libs / name).write_bytes(b"metadata-only-unit-test")
        with patch("tools.engram_sidecar.framework.full_model_http.importlib.util.find_spec", return_value=object()), patch(
            "tools.engram_sidecar.framework.full_model_http.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
        ) as run:
            report = runtime_preflight(self.root, load_native=True)
        children = [call for call in run.call_args_list if "-c" in call.args[0]]
        self.assertEqual(len(children), 1)
        child = children[0]
        self.assertEqual(child.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertTrue(child.kwargs["check"])
        for statement in (
            "from rtp_llm.dash_sc.app import DashScApp",
            "from rtp_llm.frontend.frontend_app import FrontendApp",
            "from rtp_llm.server.backend_manager import BackendManager",
            "import rtp_llm.start_backend_server",
            "from flash_mla import flash_mla_sparse_fwd, flash_mla_with_kvcache, get_mla_metadata",
            "from flash_mla import cuda as flash_mla_cuda",
            "flash_mla_cuda.sparse_prefill_fwd, flash_mla_cuda.sparse_decode_fwd",
            "assert all(callable(f)",
            "assert not torch.cuda.is_initialized()",
        ):
            self.assertIn(statement, child.args[0][2])
        self.assertNotIn("from tools.engram_sidecar", child.args[0][2])
        self.assertTrue(report["native_import_checked"])

    def test_runtime_rejects_missing_flash_mla_before_child(self):
        libs = self.root / "rtp_llm/libs"
        libs.mkdir(parents=True)
        for name in NATIVE_LIBS:
            (libs / name).touch()
        with patch("tools.engram_sidecar.framework.full_model_http.importlib.util.find_spec",
                   side_effect=lambda name: None if name == "flash_mla" else object()), patch(
            "tools.engram_sidecar.framework.full_model_http.subprocess.run",
        ) as run:
            with self.assertRaisesRegex(RuntimeError, "missing runtime Python packages: flash_mla"):
                runtime_preflight(self.root, load_native=True)
        run.assert_not_called()

    def test_runtime_child_missing_dependency_is_fatal_before_launch(self):
        libs = self.root / "rtp_llm/libs"
        libs.mkdir(parents=True)
        for name in NATIVE_LIBS:
            (libs / name).touch()
        def run(command, **kwargs):
            if "-c" in command:
                raise subprocess.CalledProcessError(1, command)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        with patch("tools.engram_sidecar.framework.full_model_http.importlib.util.find_spec", return_value=object()), patch(
            "tools.engram_sidecar.framework.full_model_http.subprocess.run", side_effect=run,
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                runtime_preflight(self.root, load_native=True)

    def test_launch_enforces_full_eager_no_draft_and_trusted_capacity(self):
        args = SimpleNamespace(checkpoint=self.root, bundle=self.root / "bundle.json", port=18630,
                               max_allocation_bytes=128 << 30, budget_bytes=256 << 30)
        env = launch_environment(args, {"ENABLE_CUDA_GRAPH": "1", "HACK_LAYER_NUM": "1", "SP_TYPE": "dspark",
                                        "MODEL_SERVICE_CONFIG": "old", "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                                        "FASTSAFETENSORS_NOGDS": "0"})
        self.assertEqual(env["ENABLE_CUDA_GRAPH"], "0")
        self.assertEqual(env["HACK_LAYER_NUM"], "0")
        self.assertEqual(env["SP_TYPE"], "")
        self.assertEqual(env["THINK_MODE"], "0")
        self.assertEqual(env["TP_SIZE"], "4")
        self.assertEqual(env["EP_SIZE"], "4")
        self.assertEqual(env["LOAD_METHOD"], "fastsafetensors")
        self.assertEqual(env["FASTSAFETENSORS_NOGDS"], "1")
        self.assertEqual(env["RTP_ENGRAM_MAX_TOTAL_ALLOCATION_BYTES"], str(256 << 30))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1,2,3")
        self.assertNotIn("MODEL_SERVICE_CONFIG", env)

    def test_decode_graph_opt_in_clears_prefill_and_bounds_capture_to_batch_one(self):
        args = SimpleNamespace(checkpoint=self.root, bundle=self.root / "bundle.json", port=18630,
                               max_allocation_bytes=128 << 30, budget_bytes=256 << 30, cuda_graph=True)
        env = launch_environment(args, {"GENERATION_PREFILL_CAPTURE_CONFIG": "64,128",
                                       "GENERATION_PREFILL_CUDA_GRAPH_MAX_REQUESTS": "8",
                                       "DECODE_CAPTURE_CONFIG": "1,16,1024", "PREFILL_CAPTURE_CONFIG": "1024:1",
                                       "ENABLE_NATIVE_CUDA_GRAPH": "1", "SP_TYPE": "eagle"})
        self.assertEqual(env["ENABLE_CUDA_GRAPH"], "1")
        self.assertEqual(env["RTP_ENGRAM_GRAPH_MODE"], "process_lifetime")
        self.assertEqual(env["DECODE_CAPTURE_CONFIG"], "1")
        self.assertEqual(env["ENABLE_NATIVE_CUDA_GRAPH"], "0")
        self.assertEqual(env["SP_TYPE"], "")
        self.assertEqual(env["CONCURRENCY_LIMIT"], "1")
        for key in ("GENERATION_PREFILL_CAPTURE_CONFIG", "GENERATION_PREFILL_CUDA_GRAPH_MAX_REQUESTS", "PREFILL_CAPTURE_CONFIG"):
            self.assertNotIn(key, env)

    def test_graph_mode_defaults_to_eager_despite_inherited_opt_in(self):
        args = SimpleNamespace(checkpoint=self.root, bundle=self.root / "bundle.json", port=18630,
                               max_allocation_bytes=128 << 30, budget_bytes=256 << 30)
        env = launch_environment(args, {"RTP_ENGRAM_GRAPH_MODE": "process_lifetime", "ENABLE_CUDA_GRAPH": "1"})
        self.assertEqual(env["ENABLE_CUDA_GRAPH"], "0")
        self.assertEqual(env["RTP_ENGRAM_GRAPH_MODE"], "eager")
        args.cuda_graph = "1"
        with self.assertRaisesRegex(ValueError, "explicit boolean"):
            launch_environment(args, {})

    def test_local_pdfusion_clears_external_routing_residue(self):
        args = SimpleNamespace(checkpoint=self.root, bundle=self.root / "bundle.json", port=18630,
                               max_allocation_bytes=128 << 30, budget_bytes=256 << 30)
        stale_keys = ("MODEL_SERVICE_CONFIG", "REMOTE_RPC_SERVER_IP", "REMOTE_VIT_SERVER_IP",
                      "RTP_LLM_DECODE_CM2_CONFIG", "RTP_LLM_MULTIMODAL_PART_CM2_CONFIG")
        env = launch_environment(args, {**dict.fromkeys(stale_keys, "stale-endpoint"), "USE_LOCAL": "0"})
        self.assertEqual(env["USE_LOCAL"], "1")
        self.assertEqual(env["ROLE_TYPE"], "PDFUSION")
        for key in stale_keys:
            self.assertNotIn(key, env)

    def test_short_chat_uses_real_v41_thinking_disabled_schema(self):
        from rtp_llm.openai.renderers.v41_recipe import resolve_thinking

        payload = chat_payload(max_tokens=32)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["max_tokens"], 32)
        for server_default in (True, False):
            self.assertIs(resolve_thinking(payload, server_default)[0], False)


if __name__ == "__main__":
    unittest.main()

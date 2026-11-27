"""Main ABI + direct-core checkpoint regression; no GPU inference claim.

Run after the NewLoader core dependency is integrated. No native import stubs,
legacy weight mapping or fallback loader is allowed in this test.
"""

import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.checkpoint_reader_test_support import checkpoint_readers
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.qwen3.language import Qwen3ForCausalLM
from rtp_llm.models_py.new_weight_manager import NewLoaderWeightManager
from rtp_llm.models_py.quant_methods import QuantizationConfig
from rtp_llm.models_py.registry import register_model

register_model("_test_main_pp_qwen3")(Qwen3ForCausalLM)


def model_config():
    return SimpleNamespace(
        model_type="_test_main_pp_qwen3",
        num_layers=3,
        is_mtp=False,
        vocab_size=8,
        hidden_size=4,
        inter_size=4,
        attn_config=SimpleNamespace(head_num=2, kv_head_num=1, size_per_head=2),
        layernorm_eps=1e-6,
        enable_fp32_lm_head=False,
        tie_word_embeddings=True,
        capture_aux_hidden_layer_ids=[],
        hidden_state_capture_layer_ids=[],
    )


def checkpoint():
    values = {
        "model.embed_tokens.weight": torch.arange(32, dtype=torch.float32).reshape(
            8, 4
        ),
        "model.norm.weight": torch.ones(4),
    }
    for layer in range(3):
        prefix = f"model.layers.{layer}."
        for name, shape in (
            ("input_layernorm.weight", (4,)),
            ("post_attention_layernorm.weight", (4,)),
            ("self_attn.q_norm.weight", (2,)),
            ("self_attn.k_norm.weight", (2,)),
            ("self_attn.q_proj.weight", (4, 4)),
            ("self_attn.k_proj.weight", (2, 4)),
            ("self_attn.v_proj.weight", (2, 4)),
            ("self_attn.o_proj.weight", (4, 4)),
            ("mlp.gate_proj.weight", (4, 4)),
            ("mlp.up_proj.weight", (4, 4)),
            ("mlp.down_proj.weight", (4, 4)),
        ):
            values[prefix + name] = torch.full(shape, layer + 1.0)
    return values


class MainPPCheckpointTest(unittest.TestCase):
    @staticmethod
    def stage_parallel(stage, tp=1, rank=0):
        return SimpleNamespace(
            tp_size=tp,
            tp_rank=rank,
            ep_size=1,
            ep_rank=0,
            pp_size=2,
            pp_rank=stage,
            pp_stage_layer_counts=[1, 2],
            dp_size=1,
            dp_rank=0,
            world_size=2 * tp,
            world_rank=stage * tp + rank,
            get_attn_tp_size=lambda: tp,
            get_attn_tp_rank=lambda: rank,
            get_ffn_tp_size=lambda: tp,
            get_ffn_tp_rank=lambda: rank,
        )

    def test_tail_warmup_preserves_all_upstream_capture_slots(self):
        config = model_config()
        config.capture_aux_hidden_layer_ids = [0, 1]
        pc = self.stage_parallel(1)
        pc.pp_stage_layer_counts = [2, 1]
        load = NewLoaderConfig(
            device="cpu", load_method="scratch", compute_dtype=torch.float32,
            parallelism_config=pc,
        )
        with tempfile.TemporaryDirectory() as folder:
            save_file(checkpoint(), os.path.join(folder, "model.safetensors"))
            model = NewModelLoader(config, load, model_path=folder).load()
            template = torch.ones(3, 4)
            warmup = model.make_empty_intermediate_tensors(template)
            self.assertIs(warmup["hidden_states"], template)
            prefix = warmup["mtp_aux_hidden_states"]
            torch.testing.assert_close(prefix, template.new_zeros((3, 8)))
            model.begin_aux_hidden_capture(template, False, prefix)
            model.finish_aux_hidden_capture()
            torch.testing.assert_close(model.get_mtp_target_hidden_states(3), prefix)

    def test_tied_pp_update_validates_embedding_source_on_both_stages(self):
        config = model_config()
        config.enable_fp32_lm_head = True
        data = checkpoint()
        name = "model.embed_tokens.weight"
        replacement = data[name].bfloat16() + 16
        for explicit in (False, True):
            if explicit:
                data["lm_head.weight"] = data[name] + 100
            with self.subTest(explicit=explicit), tempfile.TemporaryDirectory() as root:
                save_file(data, os.path.join(root, "model.safetensors"))
                for stage in (0, 1):
                    load = NewLoaderConfig(
                        device="cpu",
                        load_method="scratch",
                        compute_dtype=torch.bfloat16,
                        quant_config=QuantizationConfig("none"),
                        parallelism_config=self.stage_parallel(stage),
                    )
                    loader = NewModelLoader(config, load, model_path=root)
                    model = loader.load()
                    manager = NewLoaderWeightManager(model, loader)
                    owner = model.embed_tokens if stage == 0 else model.lm_head
                    target = owner.weight
                    old = target.detach().clone()
                    ptr = target.data_ptr()
                    if not explicit:
                        for invalid, error in (
                            (replacement.float(), TypeError),
                            (torch.cat([replacement, replacement[:1]]), ValueError),
                        ):
                            with self.assertRaises(error):
                                manager.update_tensors({name: invalid})
                            torch.testing.assert_close(target, old)
                            self.assertEqual(manager.version, 0)
                        if stage == 1:
                            with self.assertRaisesRegex(ValueError, "non-owning"):
                                manager.update_tensors({"lm_head.weight": old})
                    manager.update_tensors({name: replacement})
                    mirrored = not explicit or stage == 0
                    expected = replacement.to(target.dtype) if mirrored else old
                    torch.testing.assert_close(target, expected)
                    self.assertEqual(manager.version, int(mirrored))
                    self.assertEqual(target.data_ptr(), ptr)
                    self.assertEqual(
                        target.dtype, torch.float32 if stage else torch.bfloat16
                    )
                    # Published names are broadcast too, including on a stage
                    # with no local view to reverse-map to a Parameter.
                    manager.update_tensors({"embedding": replacement + 1})
                    expected = (replacement + 1).to(target.dtype) if mirrored else old
                    torch.testing.assert_close(target, expected)
                    norm = torch.full((4,), 3, dtype=torch.bfloat16)
                    manager.update_tensors({"final_layernorm.gamma": norm})
                    if stage:
                        torch.testing.assert_close(model.norm.weight, norm)
                    if explicit or stage == 0:
                        manager.update_tensors({"lm_head": replacement.float() + 7})
                        if stage:
                            torch.testing.assert_close(target, replacement.float() + 7)
                    else:
                        with self.assertRaisesRegex(ValueError, "non-owning"):
                            manager.update_tensors({"lm_head": replacement.float()})

    def test_pp_head_updates_only_reach_the_last_stage(self):
        values = checkpoint()
        values["lm_head.weight"] = values["model.embed_tokens.weight"] + 100
        with tempfile.TemporaryDirectory() as root:
            save_file(values, os.path.join(root, "model.safetensors"))
            for stage in range(3):
                with self.subTest(stage=stage):
                    parallel = self.stage_parallel(stage)
                    parallel.pp_size = parallel.world_size = 3
                    parallel.pp_stage_layer_counts = [1, 1, 1]
                    load = NewLoaderConfig(
                        device="cpu",
                        load_method="scratch",
                        compute_dtype=torch.float32,
                        parallelism_config=parallel,
                    )
                    loader = NewModelLoader(model_config(), load, model_path=root)
                    model = loader.load()
                    manager = NewLoaderWeightManager(model, loader)
                    self.assertEqual(model.lm_head is not None, stage == 2)
                    before = {
                        name: param.detach().clone()
                        for name, param in model.named_parameters()
                    }
                    for epoch, alias in enumerate(
                        ("lm_head.weight", "model.lm_head.weight", "lm_head"), 1
                    ):
                        updated = values["lm_head.weight"] + epoch
                        manager.update_tensors({alias: updated})
                        self.assertEqual(manager.version, epoch if stage == 2 else 0)
                        for name, param in model.named_parameters():
                            expected = (
                                updated if name == "lm_head.weight" else before[name]
                            )
                            torch.testing.assert_close(param, expected)
                    with self.assertRaisesRegex(RuntimeError, "no_such_weight"):
                        manager.update_tensors({"model.lm_head.no_such_weight": updated})

    def test_non_generation_heads_are_handler_owned_and_norm_is_preserved(self):
        from rtp_llm.ops import TaskType

        pc = SimpleNamespace(
            tp_size=1,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            pp_size=1,
            pp_rank=0,
            dp_size=1,
            dp_rank=0,
        )
        for method in checkpoint_readers("scratch", "fastsafetensors"):
            for explicit_head, needs_head in (
                (False, False),
                (True, False),
                (True, True),
                (False, True),
            ):
                with self.subTest(
                    method=method, explicit_head=explicit_head, needs_head=needs_head
                ), tempfile.TemporaryDirectory() as root:
                    cfg = model_config()
                    cfg.task_type = (
                        TaskType.RERANKER if needs_head else TaskType.DENSE_EMBEDDING
                    )
                    cfg.tie_word_embeddings = False
                    data = checkpoint()
                    if explicit_head:
                        data["lm_head.weight"] = data["model.embed_tokens.weight"] + 2
                    save_file(data, os.path.join(root, "model.safetensors"))
                    load = NewLoaderConfig(
                        device="cpu",
                        compute_dtype=torch.float32,
                        load_method=method,
                        parallelism_config=pc,
                        quant_config=QuantizationConfig("none"),
                        custom_weight_mappings=(
                            (("__custom__.lm_head.weight", "lm_head.weight"),)
                            if needs_head
                            else ()
                        ),
                    )
                    loader = NewModelLoader(cfg, load, model_path=root)
                    if needs_head and not explicit_head:
                        with self.assertRaisesRegex(RuntimeError, "Missing downstream"):
                            loader.load()
                        continue
                    model = loader.load()
                    self.assertIsNone(model.lm_head)
                    exports = model.runtime_weight_view()
                    self.assertIs(exports["final_layernorm.gamma"], model.norm.weight)
                    torch.testing.assert_close(
                        model.norm.weight, data["model.norm.weight"]
                    )
                    self.assertNotIn("lm_head", exports)
                    if needs_head:
                        torch.testing.assert_close(
                            exports["__custom__.lm_head.weight"], data["lm_head.weight"]
                        )

    def test_resolved_native_prefill_cp_stage_loads_replicas_and_sharded_head(self):
        from rtp_llm.config.server_config_setup import set_parallelism_config
        from rtp_llm.ops import CPRotateMethod, ParallelismConfig

        values = checkpoint()
        with tempfile.TemporaryDirectory() as folder:
            save_file(values, os.path.join(folder, "model.safetensors"))
            for physical_rank in range(4):
                pc = ParallelismConfig()
                pc.tp_size, pc.pp_size, pc.dp_size, pc.ep_size = 2, 2, 1, 0
                pc.world_size, pc.local_world_size = 4, 4
                pc.pp_stage_layer_counts = [1, 2]
                pc.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
                set_parallelism_config(pc, world_rank=physical_rank)
                load = NewLoaderConfig.from_runtime(
                    pc,
                    device="cpu",
                    load_method="scratch",
                    compute_dtype=torch.float32,
                    quant_config=QuantizationConfig("none"),
                )
                model = NewModelLoader(model_config(), load, model_path=folder).load()
                ids = ["0"] if pc.pp_rank == 0 else ["1", "2"]
                self.assertEqual(list(model.layers), ids)
                for layer_id in ids:
                    layer = model.layers[layer_id]
                    self.assertEqual(layer.self_attn.tp_size, 1)
                    self.assertEqual(layer.mlp.tp_size, 1)
                    torch.testing.assert_close(
                        layer.mlp.down_proj.weight,
                        values[f"model.layers.{layer_id}.mlp.down_proj.weight"],
                    )
                if pc.pp_rank == 1:
                    self.assertIsNone(model.embed_tokens)
                    torch.testing.assert_close(
                        model.lm_head.weight,
                        values["model.embed_tokens.weight"].chunk(2)[pc.tp_rank],
                    )

    def test_pp1_keeps_full_owner_and_main_capture_capability(self):
        parallel = SimpleNamespace(
            tp_size=1,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            pp_size=1,
            pp_rank=0,
            pp_stage_layer_counts=[],
            dp_size=1,
            dp_rank=0,
            world_size=1,
            world_rank=0,
            get_attn_tp_size=lambda: 1,
            get_attn_tp_rank=lambda: 0,
            get_ffn_tp_size=lambda: 1,
            get_ffn_tp_rank=lambda: 0,
        )
        load = NewLoaderConfig(
            device="cpu",
            load_method="scratch",
            compute_dtype=torch.float32,
            quant_config=QuantizationConfig("none"),
            parallelism_config=parallel,
        )
        with tempfile.TemporaryDirectory() as folder:
            save_file(checkpoint(), os.path.join(folder, "model.safetensors"))
            model = NewModelLoader(model_config(), load, model_path=folder).load()
        self.assertIsInstance(model.layers, torch.nn.ModuleList)
        self.assertEqual(len(model.layers), 3)
        self.assertIsNotNone(model.embed_tokens)
        self.assertIsNotNone(model.lm_head)
        self.assertTrue(model.supports_hidden_state_capture)
        torch.testing.assert_close(model.embed_tokens.weight, model.lm_head.weight)

    def test_stage_ownership_and_global_tied_head_rows(self):
        for stage in (0, 1):
            for tp, rank in ((1, 0), (2, 0), (2, 1)):
                for pruned in (False, True):
                    for explicit in (False, True):
                        with self.subTest(
                            stage=stage,
                            tp=tp,
                            rank=rank,
                            pruned=pruned,
                            explicit=explicit,
                        ):
                            config = model_config()
                            if pruned:
                                config.enable_output_vocab_pruning = True
                                config.output_vocab_ids = [1, 3, 7]
                                config.output_vocab_padded_size = 4
                            parallel = self.stage_parallel(stage, tp, rank)
                            load = NewLoaderConfig(
                                tp_size=tp,
                                tp_rank=rank,
                                compute_dtype=torch.float32,
                                device="cpu",
                                load_method="scratch",
                                quant_config=QuantizationConfig("none"),
                                parallelism_config=parallel,
                            )
                            values = checkpoint()
                            if explicit:
                                values["lm_head.weight"] = (
                                    values["model.embed_tokens.weight"] + 17
                                )
                            with tempfile.TemporaryDirectory() as folder:
                                save_file(
                                    values, os.path.join(folder, "model.safetensors")
                                )
                                model = NewModelLoader(
                                    config, load, model_path=folder
                                ).load()
                            self.assertEqual(
                                list(model.layers), ["0"] if stage == 0 else ["1", "2"]
                            )
                            self.assertEqual(model.embed_tokens is not None, stage == 0)
                            self.assertEqual(model.lm_head is not None, stage == 1)
                            for layer_id, layer in model.layers.items():
                                torch.testing.assert_close(
                                    layer.input_layernorm.weight,
                                    torch.full((4,), int(layer_id) + 1.0),
                                )
                            template = torch.zeros(2, 4)
                            self.assertIs(
                                model.make_empty_intermediate_tensors(template)[
                                    "hidden_states"
                                ],
                                template,
                            )
                            if stage == 1:
                                expected = values[
                                    (
                                        "lm_head.weight"
                                        if explicit
                                        else "model.embed_tokens.weight"
                                    )
                                ]
                                if pruned:
                                    selected = torch.zeros(4, 4)
                                    selected[:3] = expected[[1, 3, 7]]
                                    expected = selected
                                torch.testing.assert_close(
                                    model.lm_head.weight, expected.chunk(tp)[rank]
                                )
                                self.assertEqual(
                                    model.lm_head._explicit_checkpoint_loaded, explicit
                                )


if __name__ == "__main__":
    unittest.main()

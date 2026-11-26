"""Exercise the real service loader and output contracts, without a GPU kernel."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.py_config_modules import GenerateEnvConfig, RenderConfig, ResidentWeightsConfig
from rtp_llm.cpp.model_rpc.model_rpc_client import StreamState, trans_output
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    GenerateOutputsPB,
    TensorPB,
)
from rtp_llm.model_factory import ModelFactory
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models_py.checkpoint_reader_test_support import checkpoint_readers, requires_native_fst
from rtp_llm.ops import ParallelismConfig, TaskType
from rtp_llm.utils.base_model_datatypes import GenerateInput


def fixture(root, *, head, extra_weights=None, tied=False):
    raw = dict(
        architectures=["Qwen2ForCausalLM"],
        model_type="qwen2",
        hidden_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=8,
        vocab_size=8,
        max_position_embeddings=32,
        tie_word_embeddings=tied,
        rms_norm_eps=1e-6,
    )
    (root / "config.json").write_text(json.dumps(raw))
    tokenizer = Tokenizer(WordLevel({str(i): i for i in range(8)}, unk_token="0"))
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="0", eos_token="7"
    ).save_pretrained(root)
    values = {
        "model.embed_tokens.weight": torch.eye(8, 4),
        "model.norm.weight": torch.ones(4),
    }
    if head:
        values["lm_head.weight"] = torch.arange(32).reshape(8, 4).float()
    prefix = "model.layers.0."
    for role in ("input_layernorm", "post_attention_layernorm"):
        values[prefix + role + ".weight"] = torch.ones(4)
    for role, width in (("q_proj", 4), ("k_proj", 2), ("v_proj", 2)):
        values[prefix + "self_attn." + role + ".weight"] = torch.eye(width, 4)
        values[prefix + "self_attn." + role + ".bias"] = torch.zeros(width)
    values[prefix + "self_attn.o_proj.weight"] = torch.eye(4)
    for role, rows, cols in (
        ("gate_proj", 8, 4),
        ("up_proj", 8, 4),
        ("down_proj", 4, 8),
    ):
        values[prefix + "mlp." + role + ".weight"] = torch.eye(rows, cols)
    values.update(extra_weights or {})
    shards = {}
    for index, (key, tensor) in enumerate(values.items()):
        shard_name = f"model-{index % 2}.safetensors"
        shards.setdefault(shard_name, {})[key] = tensor
    weight_map = {}
    for shard_name, tensors in shards.items():
        save_file(tensors, str(root / shard_name))
        weight_map.update(dict.fromkeys(tensors, shard_name))
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return values


def load(root, name, method, *, task=None, data_type="fp32"):
    cfg = ModelFactory.get_model_cls(name).create_config(str(root))
    cfg.model_type, cfg.tokenizer_path, cfg.data_type = name, str(root), data_type
    cfg.task_type = task if task is not None else (
        TaskType.DENSE_EMBEDDING
        if name == "qwen_2_embedding"
        else TaskType.LANGUAGE_MODEL
    )
    cfg.use_kvcache = cfg.task_type == TaskType.LANGUAGE_MODEL
    cfg.render_config, cfg.generate_env_config = RenderConfig(), GenerateEnvConfig()
    ns = SimpleNamespace
    engine = ns(
        parallelism_config=ParallelismConfig(),
        runtime_config=ns(
            warm_up=False, model_warm_up=False, max_generate_batch_size=4
        ),
        hw_kernel_config=ns(enable_cuda_graph=False, use_swizzleA=False),
        kv_cache_config=ns(multi_task_prompt="", multi_task_prompt_str=""),
        fmha_config=None,
        moe_config=None,
        load_config=ns(
            resident_weights_config=ResidentWeightsConfig(),
            load_method=method,
            force_cpu_load_weights=False,
            loader_recycle_handles=False,
            moe_pure_tp_preshard=False,
        ),
    )
    with patch.object(BaseModel, "_get_device_str", return_value="cpu"):
        result = ModelFactory._create_model(cfg, engine)
    assert result.model_weights_loader.loaded_method.value == method
    return result


class Qwen2ServiceVariantsTest(unittest.TestCase):
    def test_fp32_lm_head_preserves_checkpoint_precision(self):
        # Neither FP16 nor BF16 can keep the low bits of these values.
        table = 1 + (torch.arange(32).reshape(8, 4).float() + 1) / 131072
        for explicit_head in (False, True):
            weights = {"model.embed_tokens.weight": table}
            if explicit_head:
                weights["lm_head.weight"] = -table
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture(root, head=explicit_head, extra_weights=weights, tied=True)
                for method in checkpoint_readers("scratch", "fastsafetensors"):
                    for name, dtype in (
                        ("fp16", torch.float16),
                        ("bf16", torch.bfloat16),
                    ):
                        with self.subTest(
                            head=explicit_head, method=method, dtype=name
                        ):
                            svc = load(root, "qwen_2", method, data_type=name)
                            exports = svc.py_model.runtime_weight_view()
                            torch.testing.assert_close(
                                exports["embedding"], table.to(dtype), rtol=0, atol=0
                            )
                            torch.testing.assert_close(
                                exports["lm_head"],
                                weights.get("lm_head.weight", table),
                                rtol=0,
                                atol=0,
                            )

    def test_score_handler_role_is_independent_of_checkpoint_path(self):
        from rtp_llm.model_loader.weight_module import CustomAtomicWeight
        from rtp_llm.models.downstream_modules.custom_module import (
            CustomHandler,
            CustomModule,
        )
        from rtp_llm.utils.model_weight import CkptWeightInfo

        matrix, offset = "document_head.matrix_values", "metadata.label_offsets"
        wt = torch.arange(12).reshape(3, 4).float() + 0.013
        bias = torch.tensor([1.001, 2.003, 3.005], dtype=torch.float64)
        mappings = (
            ("__custom__.score.weight", matrix, torch.bfloat16),
            ("__custom__.score.bias", offset, torch.float32),
        )

        class Head(CustomHandler):
            def init(self, params):
                self.params = params

            def score(self, hidden):
                return torch.nn.functional.linear(
                    hidden.float(),
                    self.params["score.weight"].float(),
                    self.params.get("score.bias"),
                )

        class DeclaredHead(Head):
            def checkpoint_weight_mappings(self):
                return mappings

        class ExtensionHead(Head):
            def custom_weight_info(self):
                return [
                    CustomAtomicWeight(role, [CkptWeightInfo(source)], data_type=dtype)
                    for role, source, dtype in mappings
                ]

        class WeightOnly(Head):
            def checkpoint_weight_mappings(self):
                return mappings[:1]

        def start(root, method, head_type):
            def custom_module(model):
                module = CustomModule(model.model_config, model.tokenizer)
                module.handler = head_type(model.model_config)
                return module

            with patch.object(BaseModel, "_init_custom_module", custom_module):
                return load(root, "qwen_2", method, task=TaskType.SEQ_CLASSIFICATION)

        for method in checkpoint_readers("scratch", "fastsafetensors"):
            for head in (DeclaredHead, ExtensionHead, WeightOnly):
                with self.subTest(
                    method=method, head=head
                ), tempfile.TemporaryDirectory() as path:
                    root = Path(path)
                    tensors = {matrix: wt}
                    expected = {"score.weight": wt.bfloat16()}
                    if head is not WeightOnly:
                        tensors[offset] = bias
                        expected["score.bias"] = bias.float()
                    fixture(root, head=False, extra_weights=tensors)
                    svc = start(root, method, head)
                    handler = svc.custom_module.handler
                    params = handler.params
                    self.assertEqual(set(params), set(expected))
                    self.assertIsNone(svc.py_model.lm_head)
                    for role, value in expected.items():
                        self.assertIs(
                            params[role],
                            svc.weight.get_global_weight("__custom__." + role),
                        )
                        torch.testing.assert_close(params[role], value, rtol=0, atol=0)
                    hidden = torch.tensor([[0.0, 1.0, 0.0, -1.0]])
                    projected = hidden @ expected["score.weight"].float().T
                    torch.testing.assert_close(
                        handler.score(hidden),
                        projected + expected.get("score.bias", 0),
                    )
            for source, tensors, error in (
                ("model.layers.0.input_layernorm.weight", {}, "model-owned"),
                ("model.norm.weight", {}, "model-owned"),
                ("model.embed_tokens.weight", {}, "model-owned"),
                (matrix, {matrix: wt, offset: torch.ones(2)}, "label counts"),
                (matrix, {matrix: torch.ones(3, 5), offset: bias}, "Invalid downstream"),
                (matrix, {matrix: wt.long(), offset: bias}, "Invalid downstream"),
                (matrix, {matrix: wt}, "Missing downstream"),
            ):
                class InvalidHead(Head):
                    def checkpoint_weight_mappings(self):
                        return ((mappings[0][0], source), mappings[1])

                with self.subTest(
                    method=method, source=source, error=error
                ), tempfile.TemporaryDirectory() as path:
                    root = Path(path)
                    fixture(root, head=False, extra_weights=tensors)
                    with self.assertRaisesRegex((ValueError, RuntimeError), error):
                        start(root, method, InvalidHead)

    @requires_native_fst
    def test_smoke_model_owners_accept_parallel_fst_worker_batches(self):
        # A normal CPU FST load uses the range reader, not the installed
        # ParallelLoader worker used by accelerator smoke. Run that actual
        # worker on host tensors so its batch protocol is covered on CPU CI.
        for variant in ("qwen_2", "qwen_2_embedding"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture(root, head=variant == "qwen_2")
                baseline = load(root, variant, "scratch")
                loader = baseline.model_weights_loader
                owner = loader._create_model()
                keep = owner.checkpoint_weight_name_filter()
                paths = sorted(str(path) for path in root.glob("*.safetensors"))
                self.assertEqual(len(paths), 2)
                resolver = loader._checkpoint_read_spec_resolver(owner, keep, None)
                weights = loader._fastsafetensors_weights_iterator(
                    paths, "cpu", keep, None, resolver
                )
                try:
                    owner.load_weights(weights)
                finally:
                    weights.close()
                loader._validate_loaded_weights(owner)
                loader._run_post_load_hooks(owner)
                torch.testing.assert_close(
                    owner.state_dict(), baseline.py_model.state_dict(), rtol=0, atol=0
                )

    def test_qwen2_vl_awq_packed_checkpoint(self):
        from rtp_llm.config.quant_config import AWQConfig
        from rtp_llm.models.qwen2_vl import QWen2_VL
        from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
        from rtp_llm.models_py.quant_methods.base import QuantizationConfig

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = dict(
                vocab_size=16,
                num_attention_heads=2,
                num_key_value_heads=1,
                hidden_size=32,
                num_hidden_layers=1,
                intermediate_size=64,
                rms_norm_eps=1e-6,
                rope_theta=10000,
                vision_config={},
                vision_start_token_id=10,
                vision_end_token_id=11,
            )
            (root / "config.json").write_text(json.dumps(raw))
            cfg = QWen2_VL.create_config(directory)
            cfg.model_type, cfg.data_type = "qwen2_vl", "fp16"
            source = AWQConfig(bits=4, group_size=16, is_quanted=True)
            tensors = {
                "model.embed_tokens.weight": torch.ones(16, 32).half(),
                "model.norm.weight": torch.ones(32).half(),
                "lm_head.weight": torch.ones(16, 32).half(),
                "visual.probe": torch.ones(1),
            }
            prefix = "model.layers.0."
            for role in ("input_layernorm", "post_attention_layernorm"):
                tensors[prefix + role + ".weight"] = torch.ones(32).half()
            base = []
            for index, (role, k, n) in enumerate(
                (
                    ("self_attn.q_proj", 32, 32),
                    ("self_attn.k_proj", 32, 16),
                    ("self_attn.v_proj", 32, 16),
                    ("self_attn.o_proj", 32, 32),
                    ("mlp.gate_proj", 32, 64),
                    ("mlp.up_proj", 32, 64),
                    ("mlp.down_proj", 64, 32),
                )
            ):
                fields = {
                    "qweight": torch.full(
                        (k, n // 8), 0x11111111 * (index + 1), dtype=torch.int32
                    ),
                    "qzeros": torch.full(
                        (k // 16, n // 8), 0x11111111, dtype=torch.int32
                    ),
                    "scales": torch.full(
                        (k // 16, n), (index + 1) / 8, dtype=torch.float16
                    ),
                }
                if index < 3:
                    tensors[prefix + role + ".bias"] = torch.full(
                        (n,), float(index + 1)
                    ).half()
                tensors.update(
                    {prefix + role + "." + key: value for key, value in fields.items()}
                )
                base.append(fields)
            save_file(tensors, str(root / "model.safetensors"))
            for method in checkpoint_readers("scratch", "fastsafetensors"):
                with self.subTest(method=method):
                    load = NewLoaderConfig.from_runtime(
                        ParallelismConfig(),
                        device="cpu",
                        load_method=method,
                        compute_dtype=torch.float16,
                        quant_config=QuantizationConfig("awq", source),
                    )
                    loader = NewModelLoader(cfg, load, model_path=directory)
                    model = loader.load()
                    self.assertEqual(loader.loaded_method.value, method)
                    layer = model.layers[0]
                    for target, parts in (
                        (layer.self_attn.qkv_proj, base[:3]),
                        (layer.self_attn.o_proj, base[3:4]),
                        (layer.mlp.gate_up_proj, base[4:6]),
                        (layer.mlp.down_proj, base[6:]),
                    ):
                        self.assertEqual(target.quant_method.method, "awq")
                        for key in ("qweight", "qzeros", "scales"):
                            torch.testing.assert_close(
                                getattr(target, key),
                                torch.cat([p[key] for p in parts], dim=1),
                            )
                    torch.testing.assert_close(
                        layer.self_attn.qkv_proj.bias.flatten(),
                        torch.cat(
                            [
                                tensors[prefix + "self_attn." + s + "_proj.bias"]
                                for s in ("q", "k", "v")
                            ]
                        ),
                    )

    def test_qwen2_vl_real_vision_and_biased_language_checkpoint(self):
        self._check_qwen2_vl_checkpoint("qwen2_vl")

    def test_qwen25_vl_real_vision_and_biased_language_checkpoint(self):
        self._check_qwen2_vl_checkpoint("qwen2_5_vl")

    def _check_qwen2_vl_checkpoint(self, variant):
        from rtp_llm.multimodal.multimodal_mixin_factory import MultimodalMixinFactory
        from rtp_llm.multimodal.multimodal_mixins.qwen2_vl.image_processing_qwen2_vl import (
            Qwen2VLImageProcessor,
        )
        from rtp_llm.multimodal.multimodal_mixins.qwen2_vl.modeling_qwen2_vl import (
            Qwen2VisionTransformerPretrainedModel,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensors = fixture(root, head=True)
            config_path = root / "config.json"
            config = json.loads(config_path.read_text())
            vision = dict(
                depth=1,
                embed_dim=8,
                hidden_size=4,
                num_heads=2,
                patch_size=2,
                spatial_merge_size=1,
                temporal_patch_size=1,
            )
            tower = Qwen2VisionTransformerPretrainedModel
            architecture = "Qwen2VLForConditionalGeneration"
            if variant == "qwen2_5_vl":
                from rtp_llm.multimodal.multimodal_mixins.qwen2_5_vl.modeling_qwen2_5_vl import (
                    Qwen2_5_VisionTransformerPretrainedModel,
                )

                tower = Qwen2_5_VisionTransformerPretrainedModel
                architecture = "Qwen2_5_VLForConditionalGeneration"
                vision.update(
                    hidden_size=8,
                    intermediate_size=16,
                    out_hidden_size=4,
                    fullatt_block_indexes=[0],
                )
            config.update(
                architectures=[architecture],
                model_type=variant,
                vision_config=vision,
                rope_theta=10000,
                vision_start_token_id=5,
                vision_end_token_id=6,
            )
            config_path.write_text(json.dumps(config))
            Qwen2VLImageProcessor().save_pretrained(root)
            reference = tower(vision)
            visual = {
                "visual." + n: p.detach().half()
                for n, p in reference.named_parameters()
            }
            save_file(visual, str(root / "vision.safetensors"))
            index_path = root / "model.safetensors.index.json"
            index = json.loads(index_path.read_text())
            index["weight_map"].update(dict.fromkeys(visual, "vision.safetensors"))
            index_path.write_text(json.dumps(index))
            for method in checkpoint_readers("scratch", "fastsafetensors"):
                with self.subTest(method=method):
                    model = load(root, variant, method)
                    owner = model.py_model
                    torch.testing.assert_close(
                        owner.lm_head.weight, tensors["lm_head.weight"]
                    )
                    self.assertIsNotNone(owner.layers[0].self_attn.qkv_proj.bias)
                    mm = MultimodalMixinFactory._create_multimodal_mixin(
                        model.model_config,
                        SimpleNamespace(
                            load_config=SimpleNamespace(load_method=method)
                        ),
                        None,
                        "cpu",
                    )
                    for name, param in mm.mm_part.visual.named_parameters():
                        torch.testing.assert_close(
                            param.float(), visual["visual." + name].float()
                        )
                    features = torch.tensor([[2.0, 4.0, 6.0, 8.0]])
                    inputs = SimpleNamespace(
                        input_ids=torch.tensor([1, 80]),
                        embedding_inputs=SimpleNamespace(
                            text_tokens_mask=torch.tensor([1, 0])
                        ),
                        multimodal_inputs=SimpleNamespace(
                            multimodal_features=[features],
                            mm_features_locs=torch.tensor([1]),
                            mm_extra_input=[],
                        ),
                    )
                    hidden, _, _ = owner._embed_multimodal_inputs(inputs)
                    torch.testing.assert_close(
                        hidden[0], tensors["model.embed_tokens.weight"][1]
                    )
                    torch.testing.assert_close(hidden[1], features[0])

    def test_tool_selected_logits_survive_terminal_frame(self):
        for method in checkpoint_readers("scratch", "fastsafetensors"):
            with self.subTest(
                method=method
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                tensors = fixture(root, head=True)
                svc = load(root, "qwen_tool", method)
                self.assertTrue(svc.py_model.config.attn_config.is_causal)
                self.assertIsNone(svc.custom_module)
                weight = svc.py_model.runtime_weight_view()["lm_head"]
                torch.testing.assert_close(weight, tensors["lm_head.weight"])
                request = GenerateInput(
                    request_id=1,
                    token_ids=torch.tensor([0]),
                    mm_inputs=[],
                    generate_config=GenerateConfig(
                        return_logits=True,
                        logits_index=2,
                        select_tokens_id=[1, 2],
                        aux_info=True,
                        is_streaming=True,
                    ),
                )
                stream = StreamState()
                for step in (1, 2, 3):
                    logits = weight[[1, 2]] @ torch.tensor([float(step), 0, 0, 0])
                    response = GenerateOutputsPB()
                    flat = response.flatten_output
                    flat.finished.append(step == 3)
                    flat.aux_info.add().output_len = step
                    flat.logits.data_type = TensorPB.FP32
                    flat.logits.shape.extend([1, 1, 2])
                    flat.logits.fp32_data = logits.numpy().tobytes()
                    output = trans_output(request, response, stream).generate_outputs[0]
                self.assertTrue(output.finished)
                torch.testing.assert_close(output.logits, torch.tensor([[8.0, 16.0]]))

    def test_embedding_publishes_noncausal_backbone_and_pooling_not_a_head(self):
        for method in checkpoint_readers("scratch", "fastsafetensors"):
            with self.subTest(
                method=method
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture(root, head=False)
                svc = load(root, "qwen_2_embedding", method)
                owner = svc.py_model
                self.assertFalse(owner.config.attn_config.is_causal)
                self.assertIsNone(owner.lm_head)
                self.assertNotIn("lm_head", owner.runtime_weight_view())
                torch.testing.assert_close(owner.norm.weight, torch.ones(4))
                # Two packed sequences; bidirectional NormalHandler pools the
                # first token, not the last. Use the weights actually loaded.
                ids = torch.tensor([1, 3, 2, 0, 1], dtype=torch.int32)
                hidden = owner.embed_tokens.weight[ids] * 3
                result = svc.custom_module.get_handler().forward(
                    ids, hidden, torch.tensor([2, 3])
                )
                torch.testing.assert_close(result, torch.eye(4)[[1, 2]])

    def test_embedding_respects_checkpoint_pooling_program(self):
        # A sentence-transformers release may explicitly select the last
        # token even with fully bidirectional attention in the transformer.
        from sentence_transformers.models import Pooling

        for method in checkpoint_readers("scratch", "fastsafetensors"):
            with self.subTest(
                method=method
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture(root, head=False)
                pool_path = root / "1_Pooling"
                pool_path.mkdir()
                Pooling(4, pooling_mode="lasttoken").save(str(pool_path))
                (root / "modules.json").write_text(
                    json.dumps(
                        [
                            {
                                "idx": 0,
                                "name": "0",
                                "path": "",
                                "type": "sentence_transformers.models.Transformer",
                            },
                            {
                                "idx": 1,
                                "name": "1",
                                "path": "1_Pooling",
                                "type": "sentence_transformers.models.Pooling",
                            },
                        ]
                    )
                )
                svc = load(root, "qwen_2_embedding", method)
                self.assertFalse(svc.py_model.config.attn_config.is_causal)
                ids = torch.tensor([1, 3, 2, 0, 1], dtype=torch.int32)
                hidden = svc.py_model.embed_tokens.weight[ids]
                handler = svc.custom_module.get_handler()
                result = handler.extend_forward(
                    input_ids=ids,
                    hidden_states=hidden,
                    input_lengths=torch.tensor([2, 3]),
                    attention_mask=torch.ones_like(ids),
                )
                torch.testing.assert_close(result, torch.eye(4)[[3, 1]])


if __name__ == "__main__":
    unittest.main()

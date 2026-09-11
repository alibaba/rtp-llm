import functools
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import List

import torch

from rtp_llm.config.output_vocab_config import OUTPUT_TOKENS_FILENAME
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
    select_output_vocab_rows,
)
from rtp_llm.model_loader.tensor_source import TensorCollector
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.base_model import BaseModel
from rtp_llm.ops import RoleType
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    WeightStyle,
    sp_0_pad8,
    sp_0_pad8_size,
    transpose,
    transpose_pad,
)


class FakeCkptFileInfo:
    def __init__(self, file_name: str, tensor_names: List[str], file_size: int = 1):
        self.file_name = file_name
        self._tensor_names = tensor_names
        self._file_size = file_size

    @property
    def file_size(self) -> int:
        return self._file_size

    def get_tensor_names(self) -> List[str]:
        return self._tensor_names


class FakeWeight:
    def __init__(self, ckpt_names: List[str]):
        self.weights = [CkptWeightInfo(name) for name in ckpt_names]

    def get_components(self):
        return [self]


class FakeCompositeWeight:
    def __init__(self, weights: List[FakeWeight]):
        self._weights = weights

    def get_components(self):
        return self._weights


def make_database(files: List[FakeCkptFileInfo]) -> CkptDatabase:
    database = CkptDatabase(None)
    database.pretrain_file_list = files
    database.finetune_file_list = []
    database._is_ft_style = False
    return database


class RecordingDeployWeightInfo(ModelDeployWeightInfo):
    def __init__(self, database: CkptDatabase, returned_weight_info: ModelWeightInfo):
        self.database = database
        self.returned_weight_info = returned_weight_info
        self.output_vocab_ids = ()
        self.events = []

    def process_meta_from_ckpt(self, ckpt_metas):
        self.events.append(
            (
                "process_meta_from_ckpt",
                len(ckpt_metas),
                len(self.database.pretrain_file_list),
            )
        )

    def get_weight_info(self) -> ModelWeightInfo:
        self.events.append(("get_weight_info", len(self.database.pretrain_file_list)))
        return self.returned_weight_info


class ModelDeployWeightInfoCkptRegexTest(unittest.TestCase):
    def test_atomic_weight_transpose_layout_uses_callable_identity(self):
        source = [CkptWeightInfo("weight")]
        self.assertTrue(AtomicWeight("real", source, transpose).need_transpose)
        self.assertTrue(
            AtomicWeight(
                "partial",
                source,
                functools.partial(transpose_pad, align_size=8, dim=0),
            ).need_transpose
        )

        @functools.wraps(transpose)
        def wrapped(tensors):
            return transpose(tensors)

        self.assertTrue(AtomicWeight("wrapped", source, wrapped).need_transpose)

        def same_name_but_different_layout(tensors):
            return tensors[0]

        same_name_but_different_layout.__name__ = "transpose"
        self.assertFalse(
            AtomicWeight(
                "lookalike", source, same_name_but_different_layout
            ).need_transpose
        )

    def test_ckpt_tensor_name_regex_matches_layer_and_expert_placeholders(self):
        pattern = ModelDeployWeightInfo._ckpt_tensor_name_to_regex(
            "model.layers.{i_1}.mlp.experts.{expert_id}.down_proj.weight"
        )

        self.assertIsNotNone(
            pattern.fullmatch("model.layers.12.mlp.experts.3.down_proj.weight")
        )
        self.assertIsNone(
            pattern.fullmatch("model.layers.x.mlp.experts.3.down_proj.weight")
        )
        self.assertIsNone(
            pattern.fullmatch("model.layers.12.mlp.experts.3.down_proj.weight.extra")
        )

    def test_ckpt_tensor_name_regex_escapes_literal_dots(self):
        pattern = ModelDeployWeightInfo._ckpt_tensor_name_to_regex("lm_head.weight")

        self.assertIsNotNone(pattern.fullmatch("lm_head.weight"))
        self.assertIsNone(pattern.fullmatch("lm_headXweight"))

    def test_collect_ckpt_tensor_regexes_from_global_layer_and_composite_weights(self):
        weight_info = ModelWeightInfo(
            weights=[
                FakeWeight(["model.embed_tokens.weight"]),
                FakeCompositeWeight([FakeWeight(["lm_head.weight"])]),
            ],
            layer_weights=[
                [
                    FakeWeight(["model.layers.{i}.self_attn.q_proj.weight"]),
                    FakeCompositeWeight(
                        [FakeWeight(["model.layers.{i}.mlp.experts.{expert_id}.w1"])]
                    ),
                ]
            ],
        )

        patterns = ModelDeployWeightInfo._collect_ckpt_tensor_name_regexes(weight_info)

        self.assertEqual(len(patterns), 4)
        self.assertTrue(
            any(pattern.fullmatch("model.embed_tokens.weight") for pattern in patterns)
        )
        self.assertTrue(
            any(pattern.fullmatch("lm_head.weight") for pattern in patterns)
        )
        self.assertTrue(
            any(
                pattern.fullmatch("model.layers.0.self_attn.q_proj.weight")
                for pattern in patterns
            )
        )
        self.assertTrue(
            any(
                pattern.fullmatch("model.layers.1.mlp.experts.7.w1")
                for pattern in patterns
            )
        )

    def test_collect_ckpt_tensor_regexes_ignores_empty_weight_info(self):
        weight_info = ModelWeightInfo(weights=[], layer_weights=[])

        patterns = ModelDeployWeightInfo._collect_ckpt_tensor_name_regexes(weight_info)

        self.assertEqual(patterns, [])


class AttentionOutputStaticQuantReciprocalTest(unittest.TestCase):
    def test_reciprocal_added_only_to_layers_with_attention_output(self):
        # layer 0 mimics a hybrid linear-attention layer: no attention output
        # projection at all. layer 1 mimics a normal MHA layer.
        weight_info = ModelWeightInfo(
            weights=[],
            layer_weights=[
                [
                    AtomicWeight(
                        W.linear_attn_out_w,
                        [CkptWeightInfo("model.layers.{i}.linear_attn.out_proj.w")],
                    )
                ],
                [
                    AtomicWeight(
                        W.attn_o_w,
                        [CkptWeightInfo("model.layers.{i}.self_attn.o_proj.weight")],
                    )
                ],
            ],
        )
        deploy_info = RecordingDeployWeightInfo(make_database([]), weight_info)

        result = deploy_info._add_attention_output_static_quant_reciprocal(weight_info)

        linear_layer, mha_layer = result.layer_weights
        self.assertEqual([w.name for w in linear_layer], [W.linear_attn_out_w])
        self.assertEqual(
            [w.name for w in mha_layer],
            [W.attn_o_w, W.attention_output_static_quant_reciprocal],
        )

        # The scale has no ckpt dependency, so it is loaded from an empty
        # collector and must still yield a float32 one on the target device.
        reciprocal = mha_layer[-1]
        self.assertEqual(reciprocal.get_tensor_names(1, None), set())
        loaded = reciprocal.load(
            TensorCollector(set(), make_database([])),
            1,
            "cpu",
            LoadConfig.model_construct(
                tp_size=1,
                dp_size=1,
                ep_size=1,
                merge_lora=False,
                exported_device=SimpleNamespace(
                    maybe_rewrite_weight_by_key=lambda _, tensor: tensor
                ),
            ),
        )
        torch.testing.assert_close(
            loaded[W.attention_output_static_quant_reciprocal],
            torch.ones(1, dtype=torch.float32),
        )


class CkptDatabaseFilterTest(unittest.TestCase):
    def test_get_max_file_size_returns_zero_for_empty_pretrain_files(self):
        database = make_database([])

        self.assertEqual(database.get_max_file_size(), 0)

    def test_filter_by_tensor_name_regexes_keeps_only_matching_files(self):
        database = make_database(
            [
                FakeCkptFileInfo(
                    "base.safetensors",
                    ["model.layers.0.self_attn.q_proj.weight"],
                ),
                FakeCkptFileInfo(
                    "mtp.safetensors",
                    ["mtp.layers.12.self_attn.q_proj.weight"],
                ),
                FakeCkptFileInfo(
                    "prefix_only.safetensors",
                    ["mtp.layers.12.self_attn.q_proj.weight.extra"],
                ),
            ]
        )
        patterns = [
            ModelDeployWeightInfo._ckpt_tensor_name_to_regex(
                "mtp.layers.{i}.self_attn.q_proj.weight"
            )
        ]

        database.filter_by_tensor_name_regexes(patterns)

        self.assertEqual(
            [ckpt.file_name for ckpt in database.pretrain_file_list],
            ["mtp.safetensors"],
        )

    def test_filter_by_tensor_name_regexes_is_noop_for_single_file(self):
        original_file = FakeCkptFileInfo(
            "single.safetensors",
            ["irrelevant.weight"],
        )
        database = make_database([original_file])
        patterns = [ModelDeployWeightInfo._ckpt_tensor_name_to_regex("required.weight")]

        database.filter_by_tensor_name_regexes(patterns)

        self.assertEqual(database.pretrain_file_list, [original_file])

    def test_filter_by_tensor_name_regexes_is_noop_for_empty_patterns(self):
        files = [
            FakeCkptFileInfo("a.safetensors", ["a.weight"]),
            FakeCkptFileInfo("b.safetensors", ["b.weight"]),
        ]
        database = make_database(files)

        database.filter_by_tensor_name_regexes([])

        self.assertEqual(database.pretrain_file_list, files)

    def test_filter_by_tensor_name_regexes_keeps_original_when_no_file_matches(self):
        files = [
            FakeCkptFileInfo("a.safetensors", ["a.weight"]),
            FakeCkptFileInfo("b.safetensors", ["b.weight"]),
        ]
        database = make_database(files)
        patterns = [ModelDeployWeightInfo._ckpt_tensor_name_to_regex("missing.weight")]

        database.filter_by_tensor_name_regexes(patterns)

        self.assertEqual(database.pretrain_file_list, files)


class CreateModelWeightInfoFilterOrderTest(unittest.TestCase):
    def test_create_model_weight_info_filters_after_meta_and_final_weight_info(self):
        database = make_database(
            [
                FakeCkptFileInfo(
                    "base.safetensors",
                    ["model.layers.0.self_attn.q_proj.weight"],
                ),
                FakeCkptFileInfo(
                    "mtp.safetensors",
                    ["mtp.layers.0.self_attn.q_proj.weight"],
                ),
            ]
        )
        returned_weight_info = ModelWeightInfo(
            weights=[],
            layer_weights=[[FakeWeight(["mtp.layers.{i}.self_attn.q_proj.weight"])]],
        )
        weight_info = RecordingDeployWeightInfo(database, returned_weight_info)

        result = weight_info.create_model_weight_info(database)

        self.assertIs(result, returned_weight_info)
        self.assertEqual(
            weight_info.events,
            [
                ("process_meta_from_ckpt", 2, 2),
                ("process_meta_from_ckpt", 0, 2),
                ("get_weight_info", 2),
            ],
        )
        self.assertEqual(
            [ckpt.file_name for ckpt in database.pretrain_file_list],
            ["mtp.safetensors"],
        )

    def test_create_model_weight_info_returns_none_for_ft_style_database(self):
        database = make_database([])
        database._is_ft_style = True
        weight_info = RecordingDeployWeightInfo(
            database,
            ModelWeightInfo(weights=[], layer_weights=[]),
        )

        self.assertIsNone(weight_info.create_model_weight_info(database))
        self.assertEqual(weight_info.events, [])

    def test_create_model_weight_info_rejects_pruning_for_ft_style_database(self):
        database = make_database([])
        database._is_ft_style = True
        weight_info = RecordingDeployWeightInfo(
            database,
            ModelWeightInfo(weights=[], layer_weights=[]),
        )
        weight_info.output_vocab_ids = (1, 3)

        with self.assertRaisesRegex(ValueError, "pre-sharded FT checkpoints"):
            weight_info.create_model_weight_info(database)

    def test_create_model_weight_info_raises_for_unknown_database_type(self):
        class UnknownDatabase:
            is_ft_style = False

        weight_info = RecordingDeployWeightInfo(
            make_database([]),
            ModelWeightInfo(weights=[], layer_weights=[]),
        )

        with self.assertRaisesRegex(Exception, "Unknown database class"):
            weight_info.create_model_weight_info(UnknownDatabase())


class OutputVocabWeightTest(unittest.TestCase):
    def test_select_output_vocab_rows_composes_after_original_process(self):
        calls = []

        def original_process(tensors):
            calls.append(True)
            return tensors[0] + 1

        # The checkpoint may contain tail padding beyond the declared model vocabulary.
        source = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        result = select_output_vocab_rows(
            [source],
            original_process,
            (1, 4),
            expected_vocab_size=5,
            expected_hidden_size=4,
        )

        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(
            result, (source + 1).index_select(0, torch.tensor([1, 4]))
        )

    def test_select_output_vocab_rows_rejects_unsupported_shapes_and_dtype(self):
        invalid_cases = [
            (torch.ones(4), (1, 4), 5, 4, "2-D"),
            (torch.ones((5, 3)), (1, 4), 5, 4, "hidden size mismatch"),
            (
                torch.ones((5, 4), dtype=torch.int32),
                (1, 4),
                5,
                4,
                "FP16, BF16, or FP32",
            ),
            (
                torch.ones((5, 4), dtype=torch.float64),
                (1, 4),
                5,
                4,
                "FP16, BF16, or FP32",
            ),
            (
                torch.ones((4, 4)),
                (1, 3),
                5,
                4,
                "does not cover the model vocabulary",
            ),
            (torch.ones((3, 4)), (1, 4), 3, 4, "exceeds LM head rows"),
        ]
        for tensor, output_vocab_ids, vocab_size, hidden_size, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    select_output_vocab_rows(
                        [tensor],
                        lambda tensors: tensors[0],
                        output_vocab_ids,
                        vocab_size,
                        hidden_size,
                    )

    def test_replacement_is_plain_and_does_not_mutate_original_descriptor(self):
        original_process = lambda tensors: tensors[0]
        original = AtomicWeight(
            W.lm_head,
            [CkptWeightInfo("lm_head.weight")],
            original_process,
            data_type=torch.float16,
        )
        weight_info = ModelWeightInfo([original], [])
        deploy_info = object.__new__(ModelDeployWeightInfo)
        deploy_info.output_vocab_ids = (1, 3)
        deploy_info.model_config = SimpleNamespace(vocab_size=5)
        deploy_info._hidden_size = 2
        deploy_info.enable_fp32_lm_head = True

        result = deploy_info._replace_output_vocab_lm_head(weight_info)
        replacement = result.weights[0]

        self.assertIs(result, weight_info)
        self.assertIsNot(replacement, original)
        self.assertIs(original.process_fun, original_process)
        self.assertIs(type(replacement), AtomicWeight)
        self.assertEqual(replacement.data_type, torch.float32)
        source = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        torch.testing.assert_close(
            replacement.process_fun([source]),
            source.index_select(0, torch.tensor([1, 3])),
        )

    def test_plain_descriptor_gate_rejects_subclasses_and_lora(self):
        class DerivedAtomicWeight(AtomicWeight):
            pass

        deploy_info = object.__new__(ModelDeployWeightInfo)
        derived = DerivedAtomicWeight(W.lm_head, [CkptWeightInfo("lm_head.weight")])
        with self.assertRaisesRegex(ValueError, "exact plain AtomicWeight"):
            deploy_info._validate_plain_output_weight(derived, W.lm_head)

        lora = AtomicWeight(
            W.lm_head,
            [CkptWeightInfo("lm_head.weight")],
            lora_a_process_func=lambda tensors: tensors[0],
        )
        with self.assertRaisesRegex(ValueError, "does not support LoRA"):
            deploy_info._validate_plain_output_weight(lora, W.lm_head)

    def test_sp_0_pad8_uses_one_layout_for_every_rank(self):
        self.assertEqual(sp_0_pad8_size(3, 2), 16)
        source = torch.arange(6, dtype=torch.float32).reshape(3, 2)

        rank_zero = sp_0_pad8(source, tp=2, tp_rank=0)
        rank_one = sp_0_pad8(source, tp=2, tp_rank=1)

        self.assertEqual(rank_zero.shape, (8, 2))
        self.assertEqual(rank_one.shape, (8, 2))
        torch.testing.assert_close(rank_zero[:3], source)
        self.assertEqual(torch.count_nonzero(rank_zero[3:]).item(), 0)
        self.assertEqual(torch.count_nonzero(rank_one).item(), 0)

    def test_sp_0_pad8_supports_one_dimensional_tensors(self):
        source = torch.tensor([1.0, 2.0, 3.0])

        self.assertEqual(sp_0_pad8(source, tp=2, tp_rank=0).shape, (8,))
        self.assertEqual(sp_0_pad8(source, tp=2, tp_rank=1).shape, (8,))

    def test_sp_0_pad8_divisible_rows_split_evenly(self):
        source = torch.arange(32, dtype=torch.float32).reshape(16, 2)

        rank_zero = sp_0_pad8(source, tp=2, tp_rank=0)
        rank_one = sp_0_pad8(source, tp=2, tp_rank=1)

        torch.testing.assert_close(rank_zero, source[:8])
        torch.testing.assert_close(rank_one, source[8:])
        torch.testing.assert_close(torch.cat([rank_zero, rank_one], 0), source)

    def test_sp_0_pad8_indivisible_rows_keep_real_rows_and_zero_tail(self):
        source = torch.arange(40, dtype=torch.float32).reshape(20, 2)

        rank_zero = sp_0_pad8(source, tp=2, tp_rank=0)
        rank_one = sp_0_pad8(source, tp=2, tp_rank=1)

        self.assertEqual(rank_zero.shape, (16, 2))
        self.assertEqual(rank_one.shape, (16, 2))
        torch.testing.assert_close(rank_zero, source[:16])
        torch.testing.assert_close(rank_one[:4], source[16:])
        self.assertEqual(torch.count_nonzero(rank_one[4:]).item(), 0)
        torch.testing.assert_close(torch.cat([rank_zero, rank_one], 0)[:20], source)

    def test_sp_0_pad8_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "row_count must be non-negative"):
            sp_0_pad8_size(-1, 2)
        with self.assertRaisesRegex(ValueError, "tp must be positive"):
            sp_0_pad8_size(8, 0)
        with self.assertRaisesRegex(ValueError, "at least one dimension"):
            sp_0_pad8(torch.tensor(1.0), tp=2, tp_rank=0)

        source = torch.ones(3, 2)
        with self.assertRaisesRegex(ValueError, "tp_rank must be in"):
            sp_0_pad8(source, tp=2, tp_rank=2)
        with self.assertRaisesRegex(ValueError, "tp_rank must be in"):
            sp_0_pad8(source, tp=2, tp_rank=-1)


class FakeQuantAlgo:
    def isQuant(self):
        return False

    def isGroupwise(self):
        return False

    def isFp8PTPC(self):
        return False


def make_minimal_configs(
    output_vocab_ids=(),
    tie_word_embeddings=False,
    enable_fp32_lm_head=False,
    has_lm_head_bias=False,
):
    attn_config = SimpleNamespace(
        head_num=2,
        kv_head_num=2,
        size_per_head=4,
        is_sparse=False,
        kv_lora_rank=0,
        nope_head_dim=0,
        rope_head_dim=0,
        v_head_dim=0,
        kv_cache_dtype="AUTO",
        use_mla=False,
    )
    eplb_config = SimpleNamespace(
        enable_eplb=lambda: False, phy_exp_num=lambda expert_num: expert_num
    )
    model_config = SimpleNamespace(
        qk_norm=False,
        hidden_size=4,
        quant_algo=FakeQuantAlgo(),
        attn_config=attn_config,
        quant_config=None,
        num_layers=1,
        src_quantization_bit=0,
        isGatedActivation=lambda: False,
        expert_num=0,
        moe_n_group=0,
        eplb_config=eplb_config,
        moe_k=0,
        moe_layer_index=[],
        moe_style=None,
        tie_word_embeddings=tie_word_embeddings,
        enable_fp32_lm_head=enable_fp32_lm_head,
        output_vocab_ids=list(output_vocab_ids),
        has_lm_head_bias=has_lm_head_bias,
        vocab_size=5,
    )
    parallelism_config = SimpleNamespace(
        get_attn_tp_size=lambda: 1,
        get_attn_tp_rank=lambda: 0,
        ep_size=1,
        ep_rank=0,
        dp_size=1,
        dp_rank=0,
        world_size=1,
        local_world_size=1,
        get_ffn_tp_rank=lambda: 0,
        get_ffn_tp_size=lambda: 1,
        ffn_disaggregate_config=SimpleNamespace(
            is_ffn_service=lambda: False, enable_ffn_disaggregate=False
        ),
        tp_size=1,
        tp_rank=0,
        role_type=RoleType.PDFUSION,
    )
    hw_kernel_config = SimpleNamespace(use_swizzleA=False)
    return model_config, parallelism_config, hw_kernel_config


class MinimalLmHeadDeployWeightInfo(ModelDeployWeightInfo):
    """Runs the production get_weight_info pipeline with a minimal descriptor set."""

    def __init__(self, configs, weights):
        model_config, parallelism_config, hw_kernel_config = configs
        super().__init__(model_config, parallelism_config, hw_kernel_config, object())
        self._weights = weights

    def _get_weight_info(self):
        return ModelWeightInfo(list(self._weights), [])

    def process_meta_from_ckpt(self, ckpt_metas):
        pass


def make_lm_head(ckpt_names=("lm_head.weight",)):
    return AtomicWeight(
        W.lm_head,
        [CkptWeightInfo(name) for name in ckpt_names],
        lambda tensors: tensors[0],
        data_type=torch.float16,
    )


class GetWeightInfoOutputVocabPipelineTest(unittest.TestCase):
    def build(self, weights, **config_kwargs):
        configs = make_minimal_configs(**config_kwargs)
        return MinimalLmHeadDeployWeightInfo(configs, weights)

    def test_pipeline_rejects_special_weight_style(self):
        deploy = self.build([make_lm_head()], output_vocab_ids=(1, 3))
        deploy.weight_style = WeightStyle.TRT_ENGINE
        with self.assertRaisesRegex(ValueError, "special checkpoint weight styles"):
            deploy.get_weight_info()

    def test_pipeline_rejects_lm_head_bias(self):
        deploy = self.build(
            [make_lm_head()], output_vocab_ids=(1, 3), has_lm_head_bias=True
        )
        with self.assertRaisesRegex(ValueError, "does not support LM head bias"):
            deploy.get_weight_info()

        bias_weight = AtomicWeight(W.lm_head_b, [CkptWeightInfo("lm_head.bias")])
        deploy = self.build([make_lm_head(), bias_weight], output_vocab_ids=(1, 3))
        with self.assertRaisesRegex(ValueError, "does not support LM head bias"):
            deploy.get_weight_info()

    def test_pipeline_requires_exactly_one_lm_head_descriptor(self):
        deploy = self.build([], output_vocab_ids=(1, 3))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            deploy.get_weight_info()

        deploy = self.build([make_lm_head(), make_lm_head()], output_vocab_ids=(1, 3))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            deploy.get_weight_info()

    def test_pipeline_rejects_tied_multi_checkpoint_source(self):
        tied_lm_head = make_lm_head(("lm_head.weight", "extra.weight"))
        embedding = AtomicWeight(
            W.embedding, [CkptWeightInfo("embed.weight")], data_type=torch.float16
        )
        deploy = self.build(
            [tied_lm_head, embedding],
            output_vocab_ids=(1, 3),
            tie_word_embeddings=True,
        )
        with self.assertRaisesRegex(ValueError, "exactly one checkpoint source"):
            deploy.get_weight_info()

    def test_pipeline_replaces_after_tie_fix_with_fp32(self):
        embedding = AtomicWeight(
            W.embedding,
            [CkptWeightInfo("embed.weight")],
            lambda tensors: tensors[0],
            data_type=torch.float16,
        )
        deploy = self.build(
            [make_lm_head(), embedding],
            output_vocab_ids=(1, 3),
            tie_word_embeddings=True,
            enable_fp32_lm_head=True,
        )
        result = deploy.get_weight_info()
        lm_heads = [w for w in result.weights if w.name == W.lm_head]
        self.assertEqual(len(lm_heads), 1)
        replacement = lm_heads[0]
        self.assertIs(type(replacement), AtomicWeight)
        self.assertEqual(replacement.data_type, torch.float32)
        # Replacement must be built from the tie-fixed descriptor, which carries
        # both checkpoint sources (lm_head first, embedding as fallback).
        self.assertEqual(
            [w.name for w in replacement.weights], ["lm_head.weight", "embed.weight"]
        )

    def test_pipeline_keeps_original_dtype_without_fp32(self):
        deploy = self.build(
            [make_lm_head()], output_vocab_ids=(1, 3), enable_fp32_lm_head=False
        )
        result = deploy.get_weight_info()
        replacement = [w for w in result.weights if w.name == W.lm_head][0]
        self.assertEqual(replacement.data_type, torch.float16)
        source = torch.arange(20, dtype=torch.float16).reshape(5, 4)
        torch.testing.assert_close(
            replacement.process_fun([source]),
            source.index_select(0, torch.tensor([1, 3])),
        )


class FinalizeOutputVocabConfigTest(unittest.TestCase):
    """Coverage for BaseModel._finalize_output_vocab_config P derivation."""

    class _FakeRealTokenizer:
        def get_vocab(self):
            return {}

    class _FakeTokenizer:
        def get_real_tokenizer(self):
            return FinalizeOutputVocabConfigTest._FakeRealTokenizer()

    def _make_fake(self, tp, dp, ep, pruning=True, has_lm_head=True):
        model_config = SimpleNamespace(
            enable_output_vocab_pruning=pruning,
            has_lm_head=has_lm_head,
            vocab_size=100,
            input_vocab_size=0,
            ckpt_path="",
            output_vocab_ids=[99],  # dirty value that must be overwritten or reset
            output_vocab_padded_size=999,
            special_tokens=SimpleNamespace(eos_token_id=0),
        )
        return SimpleNamespace(
            model_config=model_config,
            parallelism_config=SimpleNamespace(tp_size=tp, dp_size=dp, ep_size=ep),
            tokenizer=self._FakeTokenizer(),
        )

    def _finalize_with_manifest(self, fake, ids):
        with tempfile.TemporaryDirectory() as ckpt_path:
            manifest_path = os.path.join(ckpt_path, OUTPUT_TOKENS_FILENAME)
            with open(manifest_path, "w", encoding="utf-8") as writer:
                json.dump(list(ids), writer)
            fake.model_config.ckpt_path = ckpt_path
            fake.model_config.vocab_size = max(ids) + 50  # manifest is a proper subset
            BaseModel._finalize_output_vocab_config(fake)

    def test_single_device_padded_size_equals_k(self):
        fake = self._make_fake(tp=1, dp=1, ep=1)
        self._finalize_with_manifest(fake, [1, 3, 5])
        # eos 0 is merged automatically: K = {0, 1, 3, 5}; single device -> P == K.
        self.assertEqual(fake.model_config.output_vocab_ids, [0, 1, 3, 5])
        self.assertEqual(fake.model_config.output_vocab_padded_size, 4)

    def test_tp_gt1_pads_to_multiple_of_tp_times_8(self):
        fake = self._make_fake(tp=2, dp=1, ep=1)
        self._finalize_with_manifest(fake, [1, 3, 5])
        padded = fake.model_config.output_vocab_padded_size
        self.assertEqual(padded, 16)  # sp_0_pad8_size(K=4, tp=2)
        self.assertEqual(padded % (2 * 8), 0)
        self.assertGreaterEqual(padded, 4)

    def test_dp_only_distributed_pads_to_multiple_of_8(self):
        fake = self._make_fake(tp=1, dp=2, ep=1)
        self._finalize_with_manifest(fake, [1, 3, 5])
        self.assertEqual(fake.model_config.output_vocab_padded_size, 8)

    def test_ep_only_distributed_pads_to_multiple_of_8(self):
        fake = self._make_fake(tp=1, dp=1, ep=2)
        self._finalize_with_manifest(fake, [1, 3, 5])
        self.assertEqual(fake.model_config.output_vocab_padded_size, 8)

    def test_disabled_pruning_resets_both_fields(self):
        fake = self._make_fake(tp=1, dp=1, ep=1, pruning=False)
        BaseModel._finalize_output_vocab_config(fake)
        self.assertEqual(fake.model_config.output_vocab_ids, [])
        self.assertEqual(fake.model_config.output_vocab_padded_size, 0)

    def test_missing_lm_head_raises(self):
        fake = self._make_fake(tp=1, dp=1, ep=1, has_lm_head=False)
        with self.assertRaisesRegex(ValueError, "requires a model LM head"):
            BaseModel._finalize_output_vocab_config(fake)


if __name__ == "__main__":
    unittest.main()

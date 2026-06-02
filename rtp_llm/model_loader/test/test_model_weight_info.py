import unittest
from typing import List

from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.model_weight import CkptWeightInfo


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
        pattern = ModelDeployWeightInfo._ckpt_tensor_name_to_regex(
            "lm_head.weight"
        )

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
        self.assertTrue(any(pattern.fullmatch("lm_head.weight") for pattern in patterns))
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
        patterns = [
            ModelDeployWeightInfo._ckpt_tensor_name_to_regex("required.weight")
        ]

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
        patterns = [
            ModelDeployWeightInfo._ckpt_tensor_name_to_regex("missing.weight")
        ]

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
            layer_weights=[
                [FakeWeight(["mtp.layers.{i}.self_attn.q_proj.weight"])]
            ],
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

    def test_create_model_weight_info_raises_for_unknown_database_type(self):
        class UnknownDatabase:
            is_ft_style = False

        weight_info = RecordingDeployWeightInfo(
            make_database([]),
            ModelWeightInfo(weights=[], layer_weights=[]),
        )

        with self.assertRaisesRegex(Exception, "Unknown database class"):
            weight_info.create_model_weight_info(UnknownDatabase())


if __name__ == "__main__":
    unittest.main()

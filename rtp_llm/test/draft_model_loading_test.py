import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rtp_llm.config.layer_partition import local_layer_range
from rtp_llm.config.model_args import ModelArgs
from rtp_llm.model_factory import ModelFactory
from rtp_llm.models.base_model import BaseModel
from rtp_llm.ops import SpeculativeType, TaskType


def make_engine_config(sp_type, pp_size, pp_rank):
    return SimpleNamespace(
        parallelism_config=SimpleNamespace(
            pp_size=pp_size,
            pp_rank=pp_rank,
            tp_size=2,
            tp_rank=1,
            dp_size=1,
            dp_rank=0,
            pp_stage_layer_counts=[2, 3, 2] if pp_size > 1 else [],
        ),
        sp_config=SimpleNamespace(type=sp_type, gen_num_per_cycle=2),
        runtime_config=SimpleNamespace(
            warm_up=False, model_warm_up=False, max_generate_batch_size=8
        ),
        load_config=SimpleNamespace(
            load_method="auto",
            loader_recycle_handles=False,
            moe_pure_tp_preshard=False,
            force_cpu_load_weights=False,
        ),
        hw_kernel_config=MagicMock(),
        kv_cache_config=MagicMock(),
        fmha_config=MagicMock(),
        moe_config=MagicMock(),
        device_resource_config=MagicMock(),
    )


class DraftModelLoadingTest(unittest.TestCase):
    def test_engine_config_finalizes_partition_from_completed_model_config(self):
        for pp_size, counts, expected in (
            (3, [], [3, 2, 2]),
            (3, [2, 3, 2], [2, 3, 2]),
            (1, [], []),
            (1, [2, 5], [2, 5]),
        ):
            with self.subTest(pp_size=pp_size, counts=counts):
                engine = make_engine_config(SpeculativeType.MTP, pp_size, 0)
                engine.parallelism_config.pp_stage_layer_counts = counts
                engine.runtime_config.fifo_scheduler_config = SimpleNamespace(
                    max_context_batch_size=1, max_batch_tokens_size=0
                )
                model = SimpleNamespace(
                    num_layers=12, max_seq_len=128, to_string=lambda: "target"
                )
                model_cls = MagicMock(__name__="TestTarget")
                model_cls._create_config.return_value = model

                def apply_overrides(**kwargs):
                    kwargs["model_config"].num_layers = 9

                def post_build_model_config(config):
                    config.num_layers = 7

                model_cls._post_build_model_config.side_effect = post_build_model_config
                with patch.object(
                    ModelFactory, "get_model_cls", return_value=model_cls
                ), patch(
                    "rtp_llm.model_factory.build_model_config",
                    side_effect=apply_overrides,
                ):
                    result = ModelFactory.create_model_config(
                        model_args=SimpleNamespace(
                            model_type="test_target", ckpt_path="target"
                        ),
                        lora_config=SimpleNamespace(lora_info=""),
                        kv_cache_config=engine.kv_cache_config,
                        profiling_debug_logging_config=MagicMock(),
                    )
                self.assertIs(result, model)
                self.assertEqual(model.num_layers, 7)
                self.assertEqual(
                    engine.parallelism_config.pp_stage_layer_counts, counts
                )
                ModelFactory.update_engine_config_from_model_config(engine, model)
                self.assertEqual(
                    engine.parallelism_config.pp_stage_layer_counts, expected
                )

    def test_factory_sets_draft_role_before_building_config(self):
        for sp_type in (
            SpeculativeType.MTP,
            SpeculativeType.EAGLE,
            SpeculativeType.DSPARK,
        ):
            with self.subTest(sp_type=sp_type):
                engine = make_engine_config(sp_type, 3, 2)
                engine.sp_config.checkpoint_path = "draft"
                engine.sp_config.model_type = "test_draft"
                engine.sp_config.quantization = None
                engine.profiling_debug_logging_config = MagicMock()
                draft = SimpleNamespace(
                    num_layers=2,
                    is_mtp=False,
                    ckpt_path="draft",
                    tokenizer_path="",
                )
                model_cls = MagicMock()
                model_cls._create_config.return_value = draft

                def check_draft_role(model_config, *_args, **_kwargs):
                    self.assertTrue(model_config.is_mtp)
                    self.assertEqual(
                        local_layer_range(model_config, engine.parallelism_config),
                        range(2),
                    )

                model_cls._apply_kv_cache_config.side_effect = check_draft_role
                model_cls._post_build_model_config.side_effect = check_draft_role
                with patch.object(
                    ModelFactory, "get_model_cls", return_value=model_cls
                ), patch(
                    "rtp_llm.model_factory.build_model_config",
                    side_effect=check_draft_role,
                ) as build_config, patch.object(
                    ModelFactory, "_setup_dspark_configs"
                ):
                    result = ModelFactory.create_propose_model_config(
                        engine,
                        SimpleNamespace(num_layers=7, max_seq_len=128),
                        ModelArgs(),
                    )

                self.assertIs(result, draft)
                check_draft_role(result)
                build_config.assert_called_once()
                model_cls._apply_kv_cache_config.assert_called_once()
                model_cls._post_build_model_config.assert_called_once()
                self.assertEqual(
                    engine.parallelism_config.pp_stage_layer_counts, [2, 3, 2]
                )
                self.assertEqual(
                    (
                        engine.parallelism_config.pp_size,
                        engine.parallelism_config.pp_rank,
                    ),
                    (3, 2),
                )

    def test_draft_is_created_only_on_last_stage(self):
        for sp_type in (
            SpeculativeType.MTP,
            SpeculativeType.EAGLE,
            SpeculativeType.DSPARK,
        ):
            for pp_size, pp_rank in ((1, 0), (3, 0), (3, 1), (3, 2)):
                with self.subTest(sp_type=sp_type, pp_size=pp_size, pp_rank=pp_rank):
                    engine_config = make_engine_config(sp_type, pp_size, pp_rank)
                    target_config = SimpleNamespace(
                        max_seq_len=128, gen_num_per_cycle=2
                    )
                    draft_config = SimpleNamespace(
                        model_type="test_draft", num_layers=2, is_mtp=True
                    )
                    model_cls = MagicMock()
                    with patch.object(
                        ModelFactory, "get_model_cls", return_value=model_cls
                    ), patch("rtp_llm.model_factory.configure_warmup"):
                        propose = ModelFactory.get_sp_model(
                            target_config, draft_config, engine_config
                        )
                    self.assertEqual(engine_config.sp_config.type, sp_type)
                    self.assertEqual(engine_config.sp_config.gen_num_per_cycle, 2)
                    if pp_rank != pp_size - 1:
                        self.assertIsNone(propose)
                        model_cls.from_config.assert_not_called()
                        continue
                    self.assertIs(propose.model, model_cls.from_config.return_value)
                    self.assertEqual(propose.sp_type, sp_type)
                    self.assertEqual(propose.gen_num_per_circle, 2)
                    args = model_cls.from_config.call_args.kwargs
                    self.assertIs(args["model_config"], draft_config)
                    self.assertEqual(
                        local_layer_range(draft_config, args["parallelism_config"]),
                        range(2),
                    )
                    self.assertEqual(
                        engine_config.parallelism_config.pp_stage_layer_counts,
                        [2, 3, 2] if pp_size > 1 else [],
                    )
                    self.assertIs(
                        args["parallelism_config"], engine_config.parallelism_config
                    )
                    self.assertEqual(
                        (
                            args["parallelism_config"].pp_size,
                            args["parallelism_config"].pp_rank,
                        ),
                        (pp_size, pp_rank),
                    )

    def test_non_last_stage_normalizes_speculative_type(self):
        for model_type, expected in (
            ("deepseek-v3-mtp", SpeculativeType.MTP),
            ("qwen_3_moe-mtp", SpeculativeType.EAGLE3),
        ):
            with self.subTest(model_type=model_type):
                engine_config = make_engine_config(SpeculativeType.EAGLE, 3, 0)
                with patch.object(ModelFactory, "get_model_cls") as factory, patch(
                    "rtp_llm.model_factory.configure_warmup"
                ):
                    propose = ModelFactory.get_sp_model(
                        SimpleNamespace(),
                        SimpleNamespace(model_type=model_type),
                        engine_config,
                    )
                self.assertIsNone(propose)
                factory.assert_not_called()
                self.assertEqual(engine_config.sp_config.type, expected)

    def test_pp_load_requires_explicit_model_support_including_mtp(self):
        from rtp_llm.models.qwen3_next.qwen3_next import (
            Qwen3Next,
            Qwen35Dense,
            Qwen35Moe,
        )
        from rtp_llm.models.qwen3_next.qwen3_next_mtp import (
            Qwen3NextMTP,
            Qwen35DenseMTP,
            Qwen35MoeMTP,
        )
        from rtp_llm.models.qwen3_vl import QWen3_VL
        from rtp_llm.models.qwen_3_dspark import Qwen3DSpark
        from rtp_llm.models.qwen_v3 import QwenV3

        for model_cls, supports_pp in (
            (QWen3_VL, False),
            (Qwen3DSpark, False),
            (Qwen3NextMTP, False),
            (Qwen35MoeMTP, False),
            (Qwen35DenseMTP, False),
            (QwenV3, True),
            (Qwen3Next, True),
            (Qwen35Moe, True),
            (Qwen35Dense, True),
        ):
            for pp_size, is_mtp in ((2, False), (2, True), (1, False), (1, True)):
                with self.subTest(
                    model=model_cls.__name__,
                    pp_size=pp_size,
                    is_mtp=is_mtp,
                ):
                    model = object.__new__(model_cls)
                    model.model_config = SimpleNamespace(
                        task_type=TaskType.LANGUAGE_MODEL,
                        num_layers=2,
                        is_mtp=is_mtp,
                    )
                    model.parallelism_config = SimpleNamespace(pp_size=pp_size)
                    model.hw_kernel_config = SimpleNamespace(enable_cuda_graph=False)
                    self.assertEqual(model.support_pp(), supports_pp)
                    reject = pp_size > 1 and not supports_pp
                    with patch.object(
                        model, "_init_custom_module", return_value=None
                    ), patch.object(
                        model,
                        "create_model_loader",
                        side_effect=RuntimeError("reached loader"),
                    ) as loader:
                        expected = (
                            "can't support pipeline parallelism"
                            if reject
                            else "reached loader"
                        )
                        with self.assertRaisesRegex(Exception, expected):
                            model.load()
                        if reject:
                            loader.assert_not_called()
                        else:
                            loader.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

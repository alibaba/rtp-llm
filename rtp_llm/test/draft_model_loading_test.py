import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rtp_llm.model_factory import ModelFactory
from rtp_llm.models.base_model import BaseModel
from rtp_llm.ops import SpeculativeType


def make_engine_config(sp_type, pp_size, pp_rank):
    return SimpleNamespace(
        parallelism_config=SimpleNamespace(
            pp_size=pp_size, pp_rank=pp_rank, tp_size=2, tp_rank=1,
            dp_size=1, dp_rank=0, pp_stage_layer_counts=[2, 3, 2] if pp_size > 1 else [],
        ),
        sp_config=SimpleNamespace(type=sp_type, gen_num_per_cycle=2),
        runtime_config=SimpleNamespace(warm_up=False, model_warm_up=False, max_generate_batch_size=8),
        load_config=SimpleNamespace(load_method="auto", loader_recycle_handles=False, moe_pure_tp_preshard=False),
        hw_kernel_config=MagicMock(), kv_cache_config=MagicMock(),
        fmha_config=MagicMock(), moe_config=MagicMock(), device_resource_config=MagicMock(),
    )


class DraftModelLoadingTest(unittest.TestCase):
    def test_all_draft_types_load_only_on_last_stage_including_pp1(self):
        for sp_type in (
            SpeculativeType.VANILLA, SpeculativeType.MTP, SpeculativeType.EAGLE,
            SpeculativeType.EAGLE3, SpeculativeType.DSPARK,
        ):
            for pp_size, pp_rank in ((1, 0), (3, 0), (3, 1), (3, 2)):
                with self.subTest(sp_type=sp_type, pp_size=pp_size, pp_rank=pp_rank):
                    engine_config = make_engine_config(sp_type, pp_size, pp_rank)
                    target_config = SimpleNamespace(max_seq_len=128, gen_num_per_cycle=2)
                    draft_config = SimpleNamespace(model_type="test_draft")
                    model_cls = MagicMock()
                    with patch.object(ModelFactory, "get_model_cls", return_value=model_cls), patch(
                        "rtp_llm.model_factory.configure_warmup"
                    ):
                        propose = ModelFactory.get_sp_model(target_config, draft_config, engine_config)
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
                    self.assertFalse(args["apply_pp_partition"])
                    self.assertIs(args["parallelism_config"], engine_config.parallelism_config)
                    self.assertEqual((args["parallelism_config"].pp_size, args["parallelism_config"].pp_rank), (pp_size, pp_rank))

    def test_non_last_stage_still_normalizes_speculative_type(self):
        for model_type, expected in (
            ("deepseek-v3-mtp", SpeculativeType.MTP),
            ("qwen_3_moe-mtp", SpeculativeType.EAGLE3),
        ):
            with self.subTest(model_type=model_type):
                engine_config = make_engine_config(SpeculativeType.VANILLA, 3, 0)
                with patch.object(ModelFactory, "get_model_cls") as factory, patch(
                    "rtp_llm.model_factory.configure_warmup"
                ):
                    propose = ModelFactory.get_sp_model(
                        SimpleNamespace(), SimpleNamespace(model_type=model_type), engine_config
                    )
                self.assertIsNone(propose)
                factory.assert_not_called()
                self.assertEqual(engine_config.sp_config.type, expected)

    def test_pp_capability_check_applies_to_target_only(self):
        for apply_pp in (True, False):
            with self.subTest(apply_pp=apply_pp):
                model = object.__new__(BaseModel)
                model.parallelism_config = SimpleNamespace(pp_size=3)
                model.hw_kernel_config = SimpleNamespace(enable_cuda_graph=False)
                with patch.object(model, "_init_custom_module", return_value=None), patch.object(
                    model, "create_model_loader", side_effect=RuntimeError("reached loader")
                ) as loader:
                    expected = "can't support pipeline parallelism" if apply_pp else "reached loader"
                    with self.assertRaisesRegex(Exception, expected):
                        model.load(apply_pp_partition=apply_pp)
                    if apply_pp:
                        loader.assert_not_called()
                    else:
                        loader.assert_called_once_with(apply_pp_partition=False)

    def test_model_loader_receives_load_scope_without_changing_parallelism(self):
        model = object.__new__(BaseModel)
        model.model_config = SimpleNamespace(ckpt_path="", ptuning_path="", lora_infos={})
        model.parallelism_config = make_engine_config(SpeculativeType.MTP, 3, 2).parallelism_config
        model.hw_kernel_config = MagicMock()
        model.kv_cache_config = MagicMock()
        model.merge_lora = False
        model.load_method = "auto"
        model.loader_recycle_handles = False
        model.force_cpu_load_weights = False
        model.moe_pure_tp_preshard = False
        model.custom_module = None
        loader = MagicMock()
        weight_cls = MagicMock()
        with patch.object(model, "get_weight_cls", return_value=weight_cls), patch(
            "rtp_llm.models.base_model.CkptDatabase"
        ), patch("rtp_llm.models.base_model.get_model_loader", return_value=loader) as factory:
            self.assertIs(model.create_model_loader(apply_pp_partition=False), loader)
        self.assertFalse(factory.call_args.kwargs["apply_pp_partition"])
        self.assertIs(weight_cls.call_args.kwargs["parallelism_config"], model.parallelism_config)
        self.assertEqual((model.parallelism_config.pp_size, model.parallelism_config.pp_rank), (3, 2))


if __name__ == "__main__":
    unittest.main()

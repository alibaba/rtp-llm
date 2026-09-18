import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.utils.database import CkptDatabase


class FastsafetensorsCopyoutTest(unittest.TestCase):
    def database(self):
        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [
            types.SimpleNamespace(file_name="weights.safetensors")
        ]
        return database

    def test_model_loader_filters_rank_local_and_alias_weights_before_copyout(self):
        collector = MagicMock()
        collector.store_tensor.return_value = True
        collector.is_collection_complete.return_value = True
        weight = MagicMock()
        weight.load.return_value = {"resident": torch.ones(1)}
        info = ModelLoader.WeightInfo(weight, 7, collector)
        database = MagicMock()

        def iterate(*args, **kwargs):
            predicate = kwargs["local_copyout_filter"]
            self.assertTrue(predicate("experts.1.weight"))
            for excluded in ("experts.0.weight", "stacked.raw", "aliased.lm_head"):
                self.assertFalse(predicate(excluded))
            self.assertEqual(
                kwargs["stacked_key_config"],
                {"stacked.raw": "experts.{expert_id}.weight"},
            )
            yield "experts.1.weight", torch.ones(1)

        database.fastsafetensors_weights_iterator.side_effect = iterate
        loader = object.__new__(ModelLoader)
        loader._load_config = types.SimpleNamespace(database=database)
        loader._is_online_ptpc = lambda: False
        loader._create_model_weights = MagicMock()
        loader._generate_weight_info = lambda: ({"experts.1.weight": info}, [info])
        loader._build_stacked_key_config = lambda _: {
            "stacked.raw": "experts.{expert_id}.weight"
        }
        result = loader._load_from_fastsafetensor("cuda:0")
        result.set_layer_weight.assert_called_once()
        weight.load.assert_called_once()

    def test_database_forwards_filter_and_closes_once(self):
        for outcome in (
            "success",
            "iteration_error",
            "early_close",
            "close_error",
            "both_errors",
        ):
            with self.subTest(outcome=outcome):
                loader = MagicMock()
                active_error = RuntimeError("read failed")

                def items():
                    yield "local", torch.ones(1)
                    if outcome in ("iteration_error", "both_errors"):
                        raise active_error

                loader.iterate_weights.side_effect = items
                if outcome in ("close_error", "both_errors"):
                    loader.close.side_effect = RuntimeError("close failed")
                module = types.ModuleType("fastsafetensors")
                module.AutoLoader = MagicMock(return_value=loader)
                module.SingleGroup = MagicMock()
                predicate = lambda key: key == "local"
                with patch.dict(sys.modules, {"fastsafetensors": module}), patch(
                    "torch.distributed.is_initialized", return_value=False
                ):
                    iterator = self.database().fastsafetensors_weights_iterator(
                        "cuda:0", False, {"stacked": "experts.{expert_id}"}, predicate
                    )
                    if outcome == "early_close":
                        next(iterator)
                        iterator.close()
                    elif outcome in ("iteration_error", "both_errors"):
                        with self.assertRaises(RuntimeError) as raised:
                            list(iterator)
                        self.assertIs(raised.exception, active_error)
                    elif outcome == "close_error":
                        with self.assertRaisesRegex(RuntimeError, "close failed"):
                            list(iterator)
                    else:
                        self.assertEqual([key for key, _ in iterator], ["local"])
                self.assertIs(
                    module.AutoLoader.call_args.kwargs["local_copyout_filter"],
                    predicate,
                )
                loader.close.assert_called_once()

    def test_transient_estimate_uses_bounded_config_or_legacy_wheel(self):
        for estimate, expected in ((1024, 1024), (None, 3 * 4096)):
            module = types.ModuleType("fastsafetensors")
            module.load_config = lambda: types.SimpleNamespace(
                estimated_peak_device_bytes=estimate
            )
            with patch.dict(sys.modules, {"fastsafetensors": module}):
                self.assertEqual(
                    ModelLoader._fastsafetensors_transient_budget_bytes(4096), expected
                )
        with patch.dict(
            sys.modules, {"fastsafetensors": types.ModuleType("fastsafetensors")}
        ):
            self.assertEqual(
                ModelLoader._fastsafetensors_transient_budget_bytes(4096), 3 * 4096
            )


if __name__ == "__main__":
    unittest.main()

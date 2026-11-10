import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.model_args import ModelArgs
from rtp_llm.config.model_config import ModelConfig, build_model_config
from rtp_llm.model_factory import ModelFactory
from rtp_llm.ops import SpeculativeType


class ModelFactoryLayerOverrideTest(unittest.TestCase):
    def test_target_retains_requested_layer_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps({}))
            args = ModelArgs()
            args.ckpt_path = directory
            args.model_type = "hy_v4_mtp"
            args.act_type = "BF16"
            config = ModelConfig()
            config.num_layers = 78
            profile = SimpleNamespace(hack_layer_num=4)

            build_model_config(config, args, KVCacheConfig(), profile)

            self.assertEqual(config.num_layers, 4)

    def test_proposal_keeps_architecture_layers_when_target_is_reduced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps({}))
            args = ModelArgs()
            args.ckpt_path = directory
            args.model_type = "hy_v4"
            args.act_type = "BF16"
            target = ModelConfig()
            target.max_seq_len = 8192
            proposal = ModelConfig()
            proposal.num_layers = 1
            architecture = SimpleNamespace(_create_config=lambda _: proposal)
            profile = SimpleNamespace(hack_layer_num=4)
            engine = SimpleNamespace(
                sp_config=SimpleNamespace(
                    type=SpeculativeType.EAGLE,
                    checkpoint_path=directory,
                    model_type="hy_v4_mtp",
                    quantization=None,
                ),
                kv_cache_config=KVCacheConfig(),
                profiling_debug_logging_config=profile,
            )

            with patch.object(ModelFactory, "get_model_cls", return_value=architecture):
                actual = ModelFactory.create_propose_model_config(engine, target, args)

            self.assertEqual(actual.num_layers, 1)
            self.assertEqual(profile.hack_layer_num, 4)


if __name__ == "__main__":
    unittest.main()

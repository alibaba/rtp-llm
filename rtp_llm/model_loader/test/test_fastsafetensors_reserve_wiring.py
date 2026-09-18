import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.model_factory import ModelFactory
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.models.base_model import BaseModel
from rtp_llm.ops import SpeculativeType, TaskType, VitSeparation
from rtp_llm.utils.database import BaseDatabase


class _EmptyDatabase(BaseDatabase):
    def has_lora(self):
        return False


class _EmptyWeightsInfo(ModelDeployWeightInfo):
    def __init__(self, **kwargs):
        self._num_layers = 1
        self._hidden_size = 1
        self._head_num = 1
        self._head_num_kv = 1
        self._size_per_head = 1
        self.tp_size = 1
        self.ep_size = 1
        self.dp_size = 1
        self.num_nodes = 1
        self.lm_head_tp_size = 1
        self.ffn_tp_size = 1
        self._align_size = 0
        self._moe_align_size_for_padding = 0
        self.moe_n_group_ = 0
        self.expert_num_ = 0
        self.phy_exp_num_ = 0
        self.tp_rank = 0
        self.ep_rank = 0
        self.dp_rank = 0
        self.lm_head_tp_rank = 0
        self.ffn_tp_rank = 0
        self._moe_pure_tp_mode = False
        self.enable_eplb_ = False
        self._use_swizzleA = False
        self.is_attn_model = False
        self.merge_lora = False
        self.vit_separation = VitSeparation.VIT_SEPARATION_LOCAL
        self.moe_layer_index_ = []
        self._quant_algo = SimpleNamespace(getWeightBits=lambda: 16)

    def create_model_weight_info(self, database):
        return ModelWeightInfo([], [])


class _LoaderOnlyModel(BaseModel):
    @staticmethod
    def get_weight_cls():
        return _EmptyWeightsInfo

    def load_tokenizer(self):
        pass

    def _finalize_output_vocab_config(self):
        pass

    def load(self, skip_python_model=False):
        self.runtime_loader = self.create_model_loader()


class FastsafetensorsReserveWiringTest(unittest.TestCase):
    def test_main_and_propose_model_reach_runtime_load_config(self):
        for reserve in (0, 137):
            with self.subTest(reserve=reserve):
                service_config = PyEnvConfigs()
                service_config.load_config.fastsafetensors_reserve_mb = reserve
                service_config.sp_config.type = SpeculativeType.VANILLA
                engine_config = EngineConfig.create(service_config)

                def model_config():
                    return SimpleNamespace(
                        model_type="reserve-wiring-test",
                        model_name="",
                        task_type=TaskType.LANGUAGE_MODEL,
                        compute_dtype=torch.float16,
                        ckpt_path="unused",
                        ptuning_path=None,
                        lora_infos={},
                        mm_related_params=None,
                        generate_env_config=None,
                        max_seq_len=128,
                        gen_num_per_cycle=1,
                    )

                target_config, draft_config = model_config(), model_config()
                with (
                    patch.object(
                        ModelFactory, "get_model_cls", return_value=_LoaderOnlyModel
                    ),
                    patch(
                        "rtp_llm.models.base_model.CkptDatabase",
                        return_value=_EmptyDatabase(),
                    ),
                    patch.dict(
                        sys.modules,
                        {
                            "rtp_llm.device": SimpleNamespace(
                                get_current_device=lambda: None
                            )
                        },
                    ),
                    patch.object(ModelLoader, "create_eplb", return_value=(None, None)),
                ):
                    target = ModelFactory._create_model(target_config, engine_config)
                    propose = ModelFactory.get_sp_model(
                        target_config, draft_config, engine_config, target_model=target
                    )
                for model in (target, propose.model):
                    runtime = model.runtime_loader.get_load_config()
                    self.assertIsInstance(runtime, LoadConfig)
                    self.assertEqual(runtime.fastsafetensors_reserve_mb, reserve)


if __name__ == "__main__":
    unittest.main()

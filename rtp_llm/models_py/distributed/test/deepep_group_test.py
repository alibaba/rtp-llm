"""DeepEP initialization contracts; buffer allocation is mocked."""

from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import torch.distributed as dist

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.distributed import collective_torch as ct
from rtp_llm.models_py.distributed.deepep_wrapper import (
    DeepEPMode,
    DeepEPWrapper,
    DeepepWrapperConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig


class DeepEPGroupTest(TestCase):
    def setUp(self):
        DeepEPWrapper.reset()
        self.addCleanup(DeepEPWrapper.reset)
        self.model = ModelConfig()
        self.model.hidden_size = 1024
        self.model.expert_num = 16
        self.model.moe_k = 2
        self.parallel = ParallelismConfig()
        self.parallel.tp_size = 1
        self.parallel.dp_size = 2
        self.parallel.ep_size = 2
        self.parallel.world_size = 2
        self.parallel.local_world_size = 2
        self.moe = MoeConfig()
        self.moe.use_deepep_low_latency = False
        self.config = DeepepWrapperConfig.from_config_adapter(
            MoEConfigAdapter(self.model, self.parallel, self.moe)
        )

    def test_world_default_does_not_require_rtp_registry(self):
        world = object()
        with (
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "group", SimpleNamespace(WORLD=world)),
            patch.object(
                ct, "_get_group", side_effect=RuntimeError("no registry")
            ) as lookup,
            patch.object(
                DeepEPWrapper,
                "_init_deepep_buffer",
                return_value=(DeepEPMode.NORMAL, object()),
            ) as init_buffer,
        ):
            DeepEPWrapper.create(self.config)
            init_buffer.assert_called_once_with(world)
            lookup.assert_not_called()

    def test_create_passes_explicit_group_to_buffer(self):
        stage = object()
        with (
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(
                DeepEPWrapper,
                "_init_deepep_buffer",
                return_value=(DeepEPMode.NORMAL, object()),
            ) as init_buffer,
        ):
            DeepEPWrapper.create(self.config, group=stage)
            init_buffer.assert_called_once_with(stage)


if __name__ == "__main__":
    main()

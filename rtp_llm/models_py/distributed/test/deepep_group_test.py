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
    init_deepep_wrapper,
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
        self.parallel.pp_size = 1
        self.parallel.tp_size = 1
        self.parallel.dp_size = 2
        self.parallel.ep_size = 2
        self.parallel.world_size = 2
        self.parallel.local_world_size = 2
        self.moe = MoeConfig()
        self.moe.use_deepep_low_latency = False
        self.engine = SimpleNamespace(
            hw_kernel_config=None,
            parallelism_config=self.parallel,
            moe_config=self.moe,
        )

    def test_non_pp_init_uses_world_without_rtp_registry(self):
        world = object()
        with (
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "group", SimpleNamespace(WORLD=world)),
            patch.object(DeepEPWrapper, "supported", return_value=True),
            patch.object(
                ct, "_get_group", side_effect=RuntimeError("no registry")
            ) as lookup,
            patch.object(
                DeepEPWrapper,
                "_init_deepep_buffer",
                return_value=(DeepEPMode.NORMAL, object()),
            ) as init_buffer,
        ):
            init_deepep_wrapper(self.engine, self.model)
            init_buffer.assert_called_once_with(world)
            lookup.assert_not_called()

    def test_pp_init_uses_stage_group(self):
        self.parallel.pp_size = 2
        self.parallel.world_size = 4
        stage = object()
        with (
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(DeepEPWrapper, "supported", return_value=True),
            patch.object(ct, "_get_group", return_value=stage) as lookup,
            patch.object(
                DeepEPWrapper,
                "_init_deepep_buffer",
                return_value=(DeepEPMode.NORMAL, object()),
            ) as init_buffer,
        ):
            init_deepep_wrapper(self.engine, self.model)
            lookup.assert_called_once_with(ct.Group.STAGE)
            init_buffer.assert_called_once_with(stage)


if __name__ == "__main__":
    main()

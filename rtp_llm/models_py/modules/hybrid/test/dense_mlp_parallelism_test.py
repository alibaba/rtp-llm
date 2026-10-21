import unittest
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP
from rtp_llm.ops import ActivationType, ParallelismConfig


class GeluProjection(nn.Identity):
    def forward_with_bias_gelu(self, x):
        return torch.nn.functional.gelu(x)


class DenseMLPParallelismTest(unittest.TestCase):
    def make_mlp(self, activation, tp_size):
        # Exercise the actual forward with CPU projections: only the collective
        # is mocked, so no process group or GPU is needed for this contract.
        mlp = DenseMLP.__new__(DenseMLP)
        nn.Module.__init__(mlp)
        mlp.parallelism_config = ParallelismConfig()
        mlp.parallelism_config.tp_size = tp_size
        mlp.parallelism_config.ffn_tp_size = tp_size
        mlp.parallelism_config.ep_size = 1
        mlp.activation_type = activation
        mlp.is_gated = activation == ActivationType.Swiglu
        mlp.up_proj = GeluProjection()
        mlp.down_proj = nn.Identity()
        mlp.act_fn = nn.SiLU()
        mlp._fuse_silu_quant = False
        return mlp

    def test_sharded_outputs_use_tp_collective(self):
        x = torch.tensor([[-1.0, 0.0, 1.0, 2.0]])
        for activation in (ActivationType.Gelu, ActivationType.Swiglu):
            with self.subTest(activation=activation):
                mlp = self.make_mlp(activation, 2)
                local = mlp(x, skip_allreduce=True)
                with patch(
                    "rtp_llm.models_py.modules.hybrid.dense_mlp.all_reduce",
                    side_effect=lambda output, group: output * 2,
                ) as reduce:
                    torch.testing.assert_close(mlp(x), local * 2)
                reduce.assert_called_once()
                self.assertIs(reduce.call_args.kwargs["group"], Group.TP)

    def test_single_rank_and_deferred_reduction_do_not_communicate(self):
        x = torch.ones(1, 4)
        for activation in (ActivationType.Gelu, ActivationType.Swiglu):
            for tp_size, skip in ((1, False), (2, True)):
                with self.subTest(activation=activation, tp_size=tp_size, skip=skip):
                    mlp = self.make_mlp(activation, tp_size)
                    with patch(
                        "rtp_llm.models_py.modules.hybrid.dense_mlp.all_reduce"
                    ) as reduce:
                        output = mlp(x, skip_allreduce=skip)
                    reduce.assert_not_called()
                    self.assertEqual(output.shape, x.shape)


if __name__ == "__main__":
    unittest.main()

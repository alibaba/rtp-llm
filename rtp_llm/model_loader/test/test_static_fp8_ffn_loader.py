import unittest
from types import SimpleNamespace

import torch

from rtp_llm.config.quant_config import Fp8PerTensorCompressedQuantConfig
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, FfnConfig
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.static_fp8_quant_weight import Fp8PerTensorCompressedWeight
from rtp_llm.model_loader.tensor_source import TensorCollector
from rtp_llm.model_loader.weight_module import WeightModule
from rtp_llm.utils.database import BaseDatabase
from rtp_llm.utils.model_weight import CkptWeightInfo, W

_DEVICE = SimpleNamespace(maybe_rewrite_weight_by_key=lambda _, tensor: tensor)


def _load_config(database, tp_size, tp_rank):
    return LoadConfig(
        database=database,
        num_layers=1,
        hidden_size=2,
        head_num=1,
        head_num_kv=1,
        size_per_head=2,
        moe_pure_tp_mode=False,
        align_size=4,
        moe_align_size=1,
        moe_layer_index=[],
        moe_n_group=1,
        expert_num=0,
        enable_eplb=False,
        phy_exp_num=0,
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=1,
        ep_rank=0,
        dp_size=1,
        dp_rank=0,
        lm_head_tp_size=tp_size,
        lm_head_tp_rank=tp_rank,
        ffn_tp_size=tp_size,
        ffn_tp_rank=tp_rank,
        num_nodes=1,
        compute_dtype=torch.float16,
        exported_device=_DEVICE,
    )


class StaticFp8FfnLoaderTest(unittest.TestCase):
    def test_fused_w13_padding_tp_shards_and_scale_requantization(self):
        gate_name = "model.layers.{i}.mlp.gate_proj.weight"
        up_name = "model.layers.{i}.mlp.up_proj.weight"
        gate_q = torch.tensor(
            [[4.0, 8.0], [12.0, 16.0], [20.0, 24.0]],
            dtype=torch.float8_e4m3fn,
        )
        up_q = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            dtype=torch.float8_e4m3fn,
        )
        gate_scale = 0.5
        up_scale = 2.0
        max_scale = up_scale

        for scale_shape in ((), (1,)):
            for tp_size, ranks in ((1, (0,)), (2, (0, 1))):
                for tp_rank in ranks:
                    with self.subTest(
                        scale_shape=scale_shape, tp_size=tp_size, tp_rank=tp_rank
                    ):
                        source = FfnAtomicWeight(
                            W.ffn_w13,
                            [CkptWeightInfo(gate_name), CkptWeightInfo(up_name)],
                            config=FfnConfig(is_gated_activation=True, align_size=4),
                        )
                        quant_config = Fp8PerTensorCompressedQuantConfig(
                            is_quanted=True
                        )
                        weight = WeightModule.create(source, quant_config)
                        self.assertIsInstance(weight, Fp8PerTensorCompressedWeight)

                        database = BaseDatabase()
                        load_config = _load_config(database, tp_size, tp_rank)
                        tensor_names = weight.get_tensor_names(0, load_config)
                        collector = TensorCollector(tensor_names, database)
                        collector.store_tensor(gate_name.format(i=0), gate_q)
                        collector.store_tensor(up_name.format(i=0), up_q)
                        gate_scale_name = gate_name.replace(
                            ".weight", ".weights_scaling_factor"
                        ).format(i=0)
                        collector.store_tensor(
                            gate_scale_name,
                            torch.full(scale_shape, gate_scale, dtype=torch.float32),
                        )
                        collector.store_tensor(
                            up_name.replace(
                                ".weight", ".weights_scaling_factor"
                            ).format(i=0),
                            torch.full(scale_shape, up_scale, dtype=torch.float32),
                        )
                        activation_scale_name = gate_name.replace(
                            ".weight", ".activation_scaling_factor"
                        ).format(i=0)
                        self.assertTrue(
                            collector.store_tensor(
                                activation_scale_name,
                                torch.tensor(1.0, dtype=torch.float32),
                            )
                        )

                        loaded = weight.load(collector, 0, "cpu", load_config)
                        kernel = loaded[W.ffn_w13]
                        final_scale = loaded[W.ffn_s13]

                        padded_gate = torch.cat(
                            [
                                gate_q.float() * gate_scale,
                                torch.zeros(1, gate_q.shape[1]),
                            ]
                        )
                        padded_up = torch.cat(
                            [
                                up_q.float() * up_scale,
                                torch.zeros(1, up_q.shape[1]),
                            ]
                        )
                        rows_per_rank = 4 // tp_size
                        row_start = tp_rank * rows_per_rank
                        row_stop = row_start + rows_per_rank
                        local_gate = padded_gate[row_start:row_stop]
                        local_up = padded_up[row_start:row_stop]
                        expected_dequant = torch.cat([local_gate, local_up])
                        expected_kernel = (expected_dequant / max_scale).to(
                            torch.float8_e4m3fn
                        )

                        self.assertEqual(
                            tuple(kernel.shape), (rows_per_rank * 2, gate_q.shape[1])
                        )
                        self.assertEqual(kernel.dtype, torch.float8_e4m3fn)
                        torch.testing.assert_close(
                            kernel[:rows_per_rank].float(),
                            expected_kernel[:rows_per_rank].float(),
                            rtol=0,
                            atol=0,
                        )
                        torch.testing.assert_close(
                            kernel[rows_per_rank:].float(),
                            expected_kernel[rows_per_rank:].float(),
                            rtol=0,
                            atol=0,
                        )
                        if tp_rank == tp_size - 1:
                            self.assertEqual(
                                torch.count_nonzero(
                                    kernel[rows_per_rank - 1].float()
                                ).item(),
                                0,
                            )
                            self.assertEqual(
                                torch.count_nonzero(kernel[-1].float()).item(), 0
                            )
                        torch.testing.assert_close(
                            kernel.float() * final_scale,
                            expected_dequant,
                            rtol=0,
                            atol=0,
                        )
                        self.assertEqual(final_scale.shape, torch.Size([1]))
                        self.assertEqual(final_scale.item(), max_scale)


if __name__ == "__main__":
    unittest.main()

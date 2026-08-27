import unittest
from types import SimpleNamespace

import torch

from rtp_llm.model_loader.linear_attn_weight import (
    split_kda_qkv_fa_beta,
    split_kda_qkvg_fa_beta,
    split_kda_qkvg_fa_beta_sections,
)
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3Weight
from rtp_llm.utils.model_weight import W


class _DictTensorSource(TensorSource):
    def __init__(self, tensors):
        self.tensors = tensors

    def load_tensor(self, name, data_type=torch.float16):
        return [self.tensors[name].to(data_type)]

    def has_tensor(self, name):
        return name in self.tensors

    def get_database(self):
        return None


class _IdentityExportedDevice:
    @staticmethod
    def maybe_rewrite_weight_by_key(name, tensor):
        return tensor


class KdaFusedProjectionSplitTest(unittest.TestCase):
    def test_glm_low_rank_gate_qkv_are_sharded_and_fa_beta_replicated(self):
        hidden = 2
        heads = 8
        head_dim = 2
        projection_width = heads * head_dim
        f_a_width = 3
        sections = [
            torch.full((hidden, width), float(section_id))
            for section_id, width in enumerate(
                (
                    projection_width,
                    projection_width,
                    projection_width,
                    f_a_width,
                    heads,
                ),
                start=1,
            )
        ]
        fused = torch.cat(sections, dim=1)
        linear_config = SimpleNamespace(
            linear_num_key_heads=heads,
            linear_num_value_heads=heads,
            linear_key_head_dim=head_dim,
            linear_value_head_dim=head_dim,
        )

        load_config = SimpleNamespace(tp_size=4, tp_rank=2)
        actual = split_kda_qkv_fa_beta(fused, load_config, linear_config)
        local_width = projection_width // load_config.tp_size
        begin = load_config.tp_rank * local_width
        expected = torch.cat(
            tuple(section[:, begin : begin + local_width] for section in sections[:3])
            + tuple(sections[3:]),
            dim=1,
        )
        torch.testing.assert_close(actual, expected)

    def test_glm_low_rank_gate_split_rejects_nondivisible_tp_width(self):
        linear_config = SimpleNamespace(
            linear_num_key_heads=3,
            linear_num_value_heads=3,
            linear_key_head_dim=2,
            linear_value_head_dim=2,
        )
        fused = torch.zeros(2, 3 * 6 + 2 + 3)
        load_config = SimpleNamespace(tp_size=4, tp_rank=0)
        with self.assertRaisesRegex(ValueError, "not divisible by TP"):
            split_kda_qkv_fa_beta(fused, load_config, linear_config)

    def test_qkvg_are_sharded_and_fa_beta_are_replicated(self):
        hidden = 3
        heads = 8
        head_dim = 2
        f_a_width = 4
        qkv_width = heads * head_dim
        beta_width = heads
        sections = []
        column = 0
        for width in (
            qkv_width,
            qkv_width,
            qkv_width,
            qkv_width,
            f_a_width,
            beta_width,
        ):
            section = torch.arange(
                hidden * width,
                dtype=torch.float32,
            ).reshape(hidden, width)
            sections.append(section + column * 1000)
            column += 1
        global_fused = torch.cat(sections, dim=1)
        linear_config = SimpleNamespace(
            linear_num_key_heads=heads,
            linear_num_value_heads=heads,
            linear_key_head_dim=head_dim,
            linear_value_head_dim=head_dim,
        )

        for tp_size in (1, 2, 4, 8):
            local_width = qkv_width // tp_size
            for tp_rank in range(tp_size):
                load_config = SimpleNamespace(tp_size=tp_size, tp_rank=tp_rank)
                actual = split_kda_qkvg_fa_beta(
                    global_fused,
                    load_config,
                    linear_config,
                )
                begin = tp_rank * local_width
                expected = torch.cat(
                    (
                        sections[0][:, begin : begin + local_width],
                        sections[1][:, begin : begin + local_width],
                        sections[2][:, begin : begin + local_width],
                        sections[3][:, begin : begin + local_width],
                        sections[4],
                        sections[5],
                    ),
                    dim=1,
                )
                self.assertTrue(actual.is_contiguous())
                self.assertTrue(torch.equal(actual, expected))

    def test_checkpoint_manifest_loader_and_forward_layout_contract(self):
        hidden = 3
        heads = 96
        head_dim = 128
        f_a_width = 128
        projection_width = heads * head_dim
        layer_id = 0
        prefix = f"language_model.model.layers.{layer_id}.self_attn."

        def checkpoint_tensor(width, section_id):
            return (
                torch.arange(width * hidden, dtype=torch.float32).reshape(width, hidden)
                + section_id * 1000
            )

        checkpoint_sections = {
            "q_proj.weight": checkpoint_tensor(projection_width, 1),
            "k_proj.weight": checkpoint_tensor(projection_width, 2),
            "v_proj.weight": checkpoint_tensor(projection_width, 3),
            "g_proj.weight": checkpoint_tensor(projection_width, 4),
            "f_a_proj.weight": checkpoint_tensor(f_a_width, 5),
            "b_proj.weight": checkpoint_tensor(heads, 6),
        }
        source = _DictTensorSource(
            {prefix + suffix: tensor for suffix, tensor in checkpoint_sections.items()}
        )
        linear_attention_config = SimpleNamespace(
            linear_num_key_heads=heads,
            linear_num_value_heads=heads,
            linear_key_head_dim=head_dim,
            linear_value_head_dim=head_dim,
        )
        manifest = object.__new__(KimiK3Weight)
        manifest.model_config = SimpleNamespace(
            linear_attention_config=linear_attention_config
        )
        fused_weight_info = next(
            weight
            for weight in manifest._kda_weights()
            if weight.name == W.linear_attn_qkvg_fa_beta_w
        )

        for tp_size in (1, 2, 4, 8):
            local_projection_width = projection_width // tp_size
            consumer_widths = (
                local_projection_width,
                local_projection_width,
                local_projection_width,
                local_projection_width,
                f_a_width,
                heads,
            )
            for tp_rank in range(tp_size):
                load_config = SimpleNamespace(
                    compute_dtype=torch.float32,
                    exported_device=_IdentityExportedDevice(),
                    merge_lora=False,
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                )
                loaded = fused_weight_info.load(
                    source,
                    layer_id=layer_id,
                    device="cpu",
                    load_config=load_config,
                )[W.linear_attn_qkvg_fa_beta_w]
                loaded_sections = split_kda_qkvg_fa_beta_sections(
                    loaded,
                    *consumer_widths,
                    dim=1,
                )

                begin = tp_rank * local_projection_width
                expected_weights = tuple(
                    checkpoint_sections[suffix][
                        begin : begin + local_projection_width
                    ].T.contiguous()
                    for suffix in (
                        "q_proj.weight",
                        "k_proj.weight",
                        "v_proj.weight",
                        "g_proj.weight",
                    )
                ) + (
                    checkpoint_sections["f_a_proj.weight"].T.contiguous(),
                    checkpoint_sections["b_proj.weight"].T.contiguous(),
                )
                self.assertTrue(loaded.is_contiguous())
                for actual, expected in zip(loaded_sections, expected_weights):
                    self.assertTrue(torch.equal(actual, expected))

                hidden_states = torch.arange(2 * hidden, dtype=torch.float32).reshape(
                    2, hidden
                )
                projected_sections = split_kda_qkvg_fa_beta_sections(
                    hidden_states @ loaded,
                    *consumer_widths,
                    dim=1,
                )
                for actual, weight in zip(projected_sections, expected_weights):
                    self.assertTrue(torch.equal(actual, hidden_states @ weight))


if __name__ == "__main__":
    unittest.main()

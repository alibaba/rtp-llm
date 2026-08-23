import unittest
from types import SimpleNamespace

import torch

from rtp_llm.model_loader.linear_attn_weight import (
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


class _CopyableLoadConfig(SimpleNamespace):
    """Minimal LoadConfig double that preserves per-weight KTP overrides."""

    def model_copy(self, *, update):
        values = vars(self).copy()
        values.update(update)
        return _CopyableLoadConfig(**values)


class KdaFusedProjectionSplitTest(unittest.TestCase):
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
            loaded_by_rank = []
            for tp_rank in range(tp_size):
                # Decode's model-wide loader intentionally remains TP1/DP8.
                # The KDA atomic weight must override only its split view with
                # the independent KTP rank carried by the manifest.
                manifest.kda_parallel_context = SimpleNamespace(
                    size=tp_size,
                    rank=tp_rank,
                )
                fused_weight_info = next(
                    weight
                    for weight in manifest._kda_weights()
                    if weight.name == W.linear_attn_qkvg_fa_beta_w
                )
                load_config = _CopyableLoadConfig(
                    compute_dtype=torch.float32,
                    exported_device=_IdentityExportedDevice(),
                    merge_lora=False,
                    tp_size=1,
                    tp_rank=0,
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
                loaded_by_rank.append(loaded_sections)

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

            for section_idx, suffix in enumerate(
                (
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "g_proj.weight",
                )
            ):
                reconstructed = torch.cat(
                    [sections[section_idx] for sections in loaded_by_rank], dim=1
                )
                self.assertTrue(
                    torch.equal(
                        reconstructed,
                        checkpoint_sections[suffix].T.contiguous(),
                    )
                )
            for replicated_idx, suffix in (
                (4, "f_a_proj.weight"),
                (5, "b_proj.weight"),
            ):
                expected = checkpoint_sections[suffix].T.contiguous()
                for sections in loaded_by_rank:
                    self.assertTrue(torch.equal(sections[replicated_idx], expected))


if __name__ == "__main__":
    unittest.main()

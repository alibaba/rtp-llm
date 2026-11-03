import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.factory.attention.attn_factory import AttnImplFactory
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.glm53_prefill_workspace import (
    Glm53PrefillWorkspace,
)


class Hy4PrefillWorkspaceRoutingTest(unittest.TestCase):
    def select_impl(self, model_type: str, *, is_cuda_graph: bool = False):
        implementation = Mock()
        model = SimpleNamespace(
            model_type=model_type,
            getAttentionConfigs=lambda _: SimpleNamespace(use_mla=True),
            quant_config=None,
            max_seq_len=8192,
            force_sparse_mla=False,
        )
        parallelism = SimpleNamespace(get_attn_tp_size=lambda: 8)
        with patch.dict(
            AttnImplFactory.FMHA_IMPL_REGISTRY,
            {"mla": Mock(return_value=implementation)},
        ):
            result = AttnImplFactory.get_fmha_impl(
                model,
                parallelism,
                None,
                SimpleNamespace(),
                is_cuda_graph=is_cuda_graph,
            )
        self.assertIs(result, implementation)
        return implementation

    def test_hy4_prefill_enables_workspace(self):
        self.select_impl(
            "hy_v4"
        ).enable_glm53_prefill_workspace.assert_called_once_with()

    def test_hy4_mtp_prefill_enables_workspace(self):
        self.select_impl(
            "hy_v4_mtp"
        ).enable_glm53_prefill_workspace.assert_called_once_with()

    def test_graph_capture_does_not_enable_reallocating_workspace(self):
        implementation = self.select_impl("hy_v4", is_cuda_graph=True)
        implementation.enable_glm53_prefill_workspace.assert_not_called()

    def test_unrelated_model_does_not_enable_workspace(self):
        self.select_impl("qwen_3").enable_glm53_prefill_workspace.assert_not_called()


class Glm53PrefillWorkspaceTest(unittest.TestCase):
    def test_scatter_aliases_compact_prefix_of_q_transform(self) -> None:
        workspace = Glm53PrefillWorkspace()
        q_transform = workspace.q_transformed(
            7, 4, 9, dtype=torch.bfloat16, device=torch.device("cpu")
        )
        scatter = workspace.scatter_output(7, 4, 6)

        self.assertEqual(q_transform.data_ptr(), scatter.data_ptr())
        self.assertEqual(scatter.shape, (7, 4, 6))
        self.assertTrue(scatter.is_contiguous())
        scatter.fill_(3)
        self.assertTrue(
            torch.equal(q_transform.view(-1)[: scatter.numel()], scatter.view(-1))
        )

    def test_reuses_same_owner_for_all_layers(self) -> None:
        workspace = Glm53PrefillWorkspace()
        first = workspace.q_transformed(
            3, 2, 5, dtype=torch.float32, device=torch.device("cpu")
        )
        second = workspace.q_transformed(
            3, 2, 5, dtype=torch.float32, device=torch.device("cpu")
        )
        self.assertEqual(first.data_ptr(), second.data_ptr())

        workspace.release()
        third = workspace.q_transformed(
            4, 2, 5, dtype=torch.float32, device=torch.device("cpu")
        )
        self.assertEqual(third.shape, (4, 2, 5))

    def test_rejects_shape_change_and_oversized_scatter(self) -> None:
        workspace = Glm53PrefillWorkspace()
        workspace.q_transformed(
            3, 2, 5, dtype=torch.float32, device=torch.device("cpu")
        )
        with self.assertRaisesRegex(RuntimeError, "changed within one forward"):
            workspace.q_transformed(
                4, 2, 5, dtype=torch.float32, device=torch.device("cpu")
            )
        with self.assertRaisesRegex(RuntimeError, "does not fit"):
            workspace.scatter_output(3, 2, 6)


if __name__ == "__main__":
    unittest.main()

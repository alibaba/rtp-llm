import unittest

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.glm53_prefill_workspace import (
    Glm53PrefillWorkspace,
)


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

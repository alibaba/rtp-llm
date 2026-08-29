import unittest
from types import SimpleNamespace

from rtp_llm.multimodal.mm_ingress import should_create_local_mm_process_engine
from rtp_llm.ops import RoleType, VitSeparation


class MultimodalIngressOwnershipTest(unittest.TestCase):
    """Truth table of the Python-side ownership rule.

    LanguageCppEngine calls this predicate to decide whether to build a local ViT engine,
    and the C++ table (resolveMMProcessorKind, covered by mm_processor_config_test) must
    agree: if Python declines while C++ still expects a LOCAL processor the startup
    decision is INVALID and the process refuses to start. EmbeddingCppEngine intentionally
    retains its existing placement behavior because its TP input synchronization differs
    from NormalEngine. Keep the Python and C++ decision-table tests in step.
    """

    @staticmethod
    def _model(is_multimodal=True, vit_separation=VitSeparation.VIT_SEPARATION_LOCAL):
        return SimpleNamespace(
            is_multimodal=lambda: is_multimodal,
            vit_config=SimpleNamespace(vit_separation=vit_separation),
        )

    @staticmethod
    def _engine_config(tp_rank=0, role_type=RoleType.PDFUSION):
        return SimpleNamespace(
            parallelism_config=SimpleNamespace(tp_rank=tp_rank),
            pd_sep_config=SimpleNamespace(role_type=role_type),
        )

    def test_local_process_engine_ownership_truth_table(self):
        cases = (
            (True, VitSeparation.VIT_SEPARATION_LOCAL, 0, RoleType.PDFUSION, True),
            (True, VitSeparation.VIT_SEPARATION_LOCAL, 0, RoleType.PREFILL, True),
            (True, VitSeparation.VIT_SEPARATION_LOCAL, 1, RoleType.PDFUSION, False),
            (True, VitSeparation.VIT_SEPARATION_LOCAL, 0, RoleType.DECODE, False),
            (True, VitSeparation.VIT_SEPARATION_REMOTE, 0, RoleType.PDFUSION, False),
            (True, VitSeparation.VIT_SEPARATION_ROLE, 0, RoleType.PDFUSION, False),
            (False, VitSeparation.VIT_SEPARATION_LOCAL, 0, RoleType.PDFUSION, False),
        )
        for is_multimodal, vit_separation, tp_rank, role_type, expected in cases:
            with self.subTest(
                is_multimodal=is_multimodal,
                vit_separation=vit_separation,
                tp_rank=tp_rank,
                role_type=role_type,
            ):
                self.assertEqual(
                    should_create_local_mm_process_engine(
                        self._model(is_multimodal, vit_separation),
                        self._engine_config(tp_rank, role_type),
                    ),
                    expected,
                )

if __name__ == "__main__":
    unittest.main()

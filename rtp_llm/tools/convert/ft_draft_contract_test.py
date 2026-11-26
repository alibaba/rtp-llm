import unittest
from unittest.mock import Mock, patch

from rtp_llm.tools.convert.ft_export_job import create_ft_export_job


class DraftExportContractTest(unittest.TestCase):
    def test_contract_checked_before_database_open(self):
        class Dense:
            def load_ft_checkpoint(self, checkpoint):
                raise AssertionError("not a draft adapter")

        class MissingPrivateAdapter(Dense):
            pass

        class MissingVocabularyLoan:
            def load_ft_checkpoint(self, checkpoint):
                pass

        model = Mock()
        model.model_config.attn_config.head_num = 8
        model.parallelism_config.get_attn_tp_size.return_value = 1
        for kind, owner, reason in (
            ("qwen_2-mtp", MissingPrivateAdapter, "draft FT format"),
            ("qwen_3_dflash", MissingVocabularyLoan, "vocabulary FT loan contract"),
            ("qwen_3_moe_eagle3", MissingVocabularyLoan, "vocabulary FT loan contract"),
        ):
            model.model_config.model_type = kind
            with self.subTest(kind=kind), patch(
                "rtp_llm.models_py.registry.get_model_class", return_value=owner
            ), patch("rtp_llm.utils.database.CkptDatabase") as database:
                with self.assertRaisesRegex(ValueError, reason):
                    create_ft_export_job(model)
                database.assert_not_called()


if __name__ == "__main__":
    unittest.main()

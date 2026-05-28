import unittest

from pydantic import BaseModel
from smoke.aux_info_assertions import assert_aux_info_assertions
from smoke.common_def import SmokeException


class AuxInfo(BaseModel):
    pd_sep: bool = False
    prefill_local_reuse_len: int = 0


class SmokeResult(BaseModel):
    aux_info: AuxInfo


class AuxInfoAssertionsTest(unittest.TestCase):

    def test_accepts_equal_and_lower_bound_assertions(self):
        result = SmokeResult(aux_info=AuxInfo(pd_sep=True, prefill_local_reuse_len=512))

        assert_aux_info_assertions(
            result,
            {
                "mode": "aux_info_only",
                "fields": {
                    "aux_info.pd_sep": {"eq": True},
                    "aux_info.prefill_local_reuse_len": {"ge": 512},
                },
            },
        )

    def test_failure_reports_field_path(self):
        result = SmokeResult(aux_info=AuxInfo(pd_sep=True, prefill_local_reuse_len=448))

        with self.assertRaises(SmokeException) as ctx:
            assert_aux_info_assertions(
                result,
                {
                    "fields": {
                        "aux_info.prefill_local_reuse_len": {"ge": 512},
                    },
                },
            )

        self.assertIn("aux_info.prefill_local_reuse_len", str(ctx.exception))
        self.assertIn("ge", str(ctx.exception))

    def test_type_mismatch_is_reported_as_assertion_failure(self):
        result = SmokeResult(aux_info=AuxInfo(pd_sep=True, prefill_local_reuse_len=448))

        with self.assertRaises(SmokeException) as ctx:
            assert_aux_info_assertions(
                result,
                {
                    "fields": {
                        "aux_info.pd_sep": {"ge": "yes"},
                    },
                },
            )

        self.assertIn("aux_info.pd_sep", str(ctx.exception))
        self.assertIn("can not be compared", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

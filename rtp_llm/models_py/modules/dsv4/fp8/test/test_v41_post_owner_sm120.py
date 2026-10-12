"""Worker post selection alone and composed with ordered stream collapse."""

import unittest

from rtp_llm.models_py.modules.dsv4.fp8.test import (
    test_v41_collapse_owner_sm120 as owner_tests,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_mhc_precision import mhc_post_reference


class PostOwnerSM120Test(owner_tests.CollapseOwnerSM120Test):
    component = "post"
    reference_context = staticmethod(mhc_post_reference)


class PostCollapseOwnerSM120Test(PostOwnerSM120Test):
    extra_choices = {"mhc_collapse_mode": "ordered_fp32"}


if __name__ == "__main__":
    unittest.main()

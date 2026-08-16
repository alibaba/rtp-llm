"""Regression tests for the ``expert_dtype`` checkpoint key.

The key travels: a V4-Flash ``config.json`` -> ``DeepSeekV4._from_hf`` ->
``ModelConfig.expert_dtype`` -> ``DeepSeekV4Weight._process_meta`` -> the dtype the
loader declares for routed-expert tensors (``float8_e4m3fn`` vs ``int8``-packed FP4
nibble pairs). Getting it wrong does not fail loudly: the wrong declared dtype
surfaces later as a shape or dtype error deep inside MoE weight assembly, after
minutes of weight loading.

Two layers are covered here:

* the allow-list -- ``ModelConfig`` gates attribute writes on a hand-maintained set
  of names, so a name that is not in it silently is not settable;
* ``parse_expert_dtype`` -- the default when the key is absent (FP4, which keeps
  every released checkpoint on its existing path), case/whitespace normalisation,
  and rejection of anything else.

Host-only.
"""

import os
import sys
import unittest

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.deepseek_v4 import (
    EXPERT_DTYPE_FP4,
    EXPERT_DTYPE_FP8,
    EXPERT_DTYPES,
    parse_expert_dtype,
)


class ExpertDtypeConfigFieldTest(unittest.TestCase):
    def test_defaults_to_none_and_is_settable(self):
        config = ModelConfig()
        # None rather than "fp4": the None -> fp4 mapping lives in
        # parse_expert_dtype, so that the config layer needs no import from models.
        self.assertIsNone(config.expert_dtype)
        config.expert_dtype = EXPERT_DTYPE_FP8
        self.assertEqual(config.expert_dtype, EXPERT_DTYPE_FP8)

    def test_a_misspelled_name_is_rejected(self):
        """Guards the allow-list entry itself, which is maintained by hand."""
        config = ModelConfig()
        with self.assertRaises(AttributeError):
            config.expert_dtypes = "fp8"  # note the plural


class ParseExpertDtypeTest(unittest.TestCase):
    def test_absent_key_falls_back_to_fp4(self):
        """Every released FP4 checkpoint has no such key and must be unaffected."""
        self.assertEqual(parse_expert_dtype(None), EXPERT_DTYPE_FP4)

    def test_normalises_case_and_whitespace(self):
        for raw in ("fp8", "FP8", "Fp8", " fp8", "fp8 ", "  FP8  "):
            with self.subTest(raw=raw):
                self.assertEqual(parse_expert_dtype(raw), EXPERT_DTYPE_FP8)
        for raw in ("fp4", "FP4", " fp4 "):
            with self.subTest(raw=raw):
                self.assertEqual(parse_expert_dtype(raw), EXPERT_DTYPE_FP4)

    def test_rejects_anything_else(self):
        for raw in ("fp16", "int8", "fp", "", "  ", "fp8x", "bf16"):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, "unsupported expert_dtype"):
                    parse_expert_dtype(raw)

    def test_error_names_the_value_and_the_legal_set(self):
        with self.assertRaises(ValueError) as caught:
            parse_expert_dtype("FP16")
        message = str(caught.exception)
        self.assertIn("FP16", message)
        for legal in EXPERT_DTYPES:
            self.assertIn(legal, message)

    def test_the_only_two_values_are_the_ones_the_consumer_branches_on(self):
        """A third value would silently take the int8 branch at the consumer."""
        self.assertEqual(set(EXPERT_DTYPES), {EXPERT_DTYPE_FP4, EXPERT_DTYPE_FP8})


if __name__ == "__main__":
    unittest.main()

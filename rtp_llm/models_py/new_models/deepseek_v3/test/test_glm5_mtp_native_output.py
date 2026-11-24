"""The real serving output class is supplied by the driver-linked compute DSO."""

import unittest

from rtp_llm.models_py.new_models.deepseek_v3.test.test_glm5_mtp_route import (
    _DraftServiceFixture,
)


class Glm5DraftNativeOutputTest(_DraftServiceFixture, unittest.TestCase):
    def test_service_owner_and_both_vocab_layouts(self):
        self.check_service_owner_and_both_vocab_layouts(native_output=True)


if __name__ == "__main__":
    unittest.main()

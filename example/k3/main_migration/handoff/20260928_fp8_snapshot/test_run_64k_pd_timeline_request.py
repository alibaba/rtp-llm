"""Check that vLLM's NIXL holdback keeps the measured 64K prefix intact."""

import gzip
import hashlib
import json
import unittest

import run_64k_pd_timeline_request as runner


class VllmHoldbackInputTest(unittest.TestCase):
    def test_extra_token_preserves_the_fixed_prefill_prefix(self):
        fixture = json.load(gzip.open(runner.FIXTURE, "rt"))
        original = fixture["input_ids"]
        submitted = runner.vllm_prompt_ids(original, append_holdback_token=True)

        self.assertEqual(len(original), 65536)
        self.assertEqual(len(submitted), 65537)
        self.assertEqual(submitted[:65536], original)
        self.assertEqual(submitted[-1], original[-1])
        self.assertEqual(
            hashlib.sha256(json.dumps(submitted[:65536], separators=(",", ":")).encode()).hexdigest(),
            runner.TOKEN_SHA,
        )

    def test_default_prompt_does_not_change_the_input(self):
        original = [101, 102, 103]
        submitted = runner.vllm_prompt_ids(original, append_holdback_token=False)
        self.assertEqual(submitted, original)
        self.assertIsNot(submitted, original)


if __name__ == "__main__":
    unittest.main()

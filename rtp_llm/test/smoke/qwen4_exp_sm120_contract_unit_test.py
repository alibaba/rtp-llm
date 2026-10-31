"""Check that the smoke harness rejects missing coverage and wrong tokens."""

import copy
import unittest

import qwen4_exp_sm120_contract_test as contract


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class Qwen4SM120ContractUnitTest(unittest.TestCase):
    def result(self):
        return {
            "finished": True,
            "input_ids": [[1, 2, 3]],
            "output_ids": [[4, 5]],
            "aux_info": {
                "input_len": 3,
                "output_len": 2,
                "reuse_len": 0,
                "speculative_draft_rounds": 2,
                "speculative_accepted_tokens_per_pos": [2, 1, 0],
            },
        }

    def test_prompts_exceed_budget_share_pages_and_end_at_page_tail(self):
        tokenizer = WordTokenizer()
        first, second = contract.build_prompt_pair(tokenizer, 8192, 4)
        first_ids, second_ids = tokenizer.encode(first), tokenizer.encode(second)
        self.assertGreater(len(first_ids), 8192)
        self.assertEqual(len(first_ids) % 128, 127)
        self.assertNotEqual(len(first_ids) % 4, 0)
        self.assertEqual(first_ids[:8192], second_ids[:8192])
        self.assertNotEqual(first, second)

    def test_page_tail_counts_the_rendered_chat_template(self):
        tokenizer = WordTokenizer()

        def render(text):
            return "user message " + text + " assistant response"

        first, second = contract.build_prompt_pair(
            tokenizer, 2048, 4, render_prompt=render
        )
        self.assertTrue(first.startswith("user message "))
        self.assertTrue(first.endswith(" assistant response"))
        self.assertEqual(len(tokenizer.encode(first)) % 128, 127)
        self.assertIn("The archive keyword is MAP.", second)

    def test_invalid_qsa_geometry_fails(self):
        for budget, ratio in ((0, 4), (2048, 1), (2048, 3)):
            with self.subTest(budget=budget, ratio=ratio):
                with self.assertRaisesRegex(ValueError, "QSA geometry"):
                    contract.build_prompt_pair(WordTokenizer(), budget, ratio)

    def test_missing_mtp_rounds_acceptance_and_wrong_tokens_fail(self):
        result = self.result()
        contract.validate_generation(result, [1, 2, 3], 2, 10, True)
        for field, value in (
            ("speculative_draft_rounds", 0),
            ("speculative_accepted_tokens_per_pos", [0, 0, 0]),
            ("speculative_accepted_tokens_per_pos", [3, 0, 0]),
            ("reuse_len", 3),
            ("input_len", 2),
            ("output_len", 1),
        ):
            with self.subTest(field=field, value=value):
                broken = copy.deepcopy(result)
                broken["aux_info"][field] = value
                with self.assertRaises(AssertionError):
                    contract.validate_generation(broken, [1, 2, 3], 2, 10, True)
        for key, value in (
            ("input_ids", [[1, 2, 9]]),
            ("output_ids", [[4, 10]]),
            ("finished", False),
        ):
            with self.subTest(field=key):
                broken = copy.deepcopy(result)
                broken[key] = value
                with self.assertRaises(AssertionError):
                    contract.validate_generation(broken, [1, 2, 3], 2, 10, True)

    def test_one_token_reference_cannot_silently_run_speculation(self):
        result = self.result()
        result["output_ids"] = [[4]]
        result["aux_info"]["output_len"] = 1
        with self.assertRaisesRegex(AssertionError, "One-token prefill"):
            contract.validate_generation(result, [1, 2, 3], 1, 10, False)
        result["aux_info"]["speculative_draft_rounds"] = 0
        contract.validate_generation(result, [1, 2, 3], 1, 10, False)


if __name__ == "__main__":
    unittest.main()

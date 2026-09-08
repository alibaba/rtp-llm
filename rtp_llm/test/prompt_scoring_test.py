import logging
import os
from unittest import TestCase, main

import requests

from rtp_llm.test.utils.maga_server_manager import MagaServerManager

logging.basicConfig(level=logging.INFO)


class PromptScoringTest(TestCase):
    server_manager = None

    @classmethod
    def setUpClass(cls):
        import unittest

        checkpoint_path = os.environ.get("CHECKPOINT_PATH")
        if not checkpoint_path:
            raise unittest.SkipTest("CHECKPOINT_PATH not set")
        model_type = os.environ.get("MODEL_TYPE", "qwen_3")

        env_args = {
            "TP_SIZE": "1",
            "DP_SIZE": "1",
            "EP_SIZE": "1",
            "WORLD_SIZE": "1",
            "MAX_SEQ_LEN": "5120",
            "WARM_UP": "0",
            "DEVICE_RESERVE_MEMORY_BYTES": "-2048000000",
            "RESERVER_RUNTIME_MEM_MB": "2048",
            "NCCL_DISABLE_ABORT": "1",
            "FT_DISABLE_CUSTOM_AR": "1",
        }

        cls.server_manager = MagaServerManager(env_args=env_args)
        ret = cls.server_manager.start_server(
            model_path=checkpoint_path,
            model_type=model_type,
            tokenizer_path=checkpoint_path,
        )
        if not ret:
            raise RuntimeError("Server failed to start")
        logging.info("Server is ready")

    @classmethod
    def tearDownClass(cls):
        if cls.server_manager:
            cls.server_manager.stop_server()

    def _post(self, endpoint, payload):
        import json

        ret, response = self.server_manager.visit(
            payload, retry_times=3, endpoint=endpoint
        )
        self.assertTrue(ret, f"Request to {endpoint} failed: {response}")
        if isinstance(response, str):
            response = json.loads(response)
        return response

    def test_01_normal_generation(self):
        """Regression: normal generation still works."""
        data = self._post(
            "/",
            {
                "prompt": "Hello, world!",
                "generate_config": {"max_new_tokens": 5},
            },
        )
        self.assertIn("response", data)
        self.assertTrue(data.get("finished"))
        self.assertTrue(len(data["response"]) > 0)

    def test_02_general_protocol_prompt_logits(self):
        """General protocol returns prompt_logits with correct shape."""
        prompt = "The quick brown fox jumps over the lazy dog"
        data = self._post(
            "/",
            {
                "prompt": prompt,
                "generate_config": {
                    "return_prompt_logits": True,
                    "prompt_logits_top_k": 16,
                    "max_new_tokens": 1,
                    "return_target_logprob": True,
                },
            },
        )
        self.assertIn("prompt_logprobs", data)
        pl = data["prompt_logprobs"]
        self.assertIn("topk_logprobs", pl)
        self.assertIn("topk_token_ids", pl)
        self.assertIn("target_logprobs", pl)
        self.assertIn("start_pos", pl)
        self.assertIn("end_pos", pl)

        num_positions = len(pl["topk_logprobs"])
        expected_positions = pl["end_pos"] - pl["start_pos"]
        self.assertEqual(num_positions, expected_positions)
        self.assertGreater(num_positions, 1)
        self.assertEqual(len(pl["topk_logprobs"][0]), 16)
        self.assertEqual(len(pl["topk_token_ids"][0]), 16)
        self.assertEqual(len(pl["topk_token_ids"]), num_positions)
        self.assertEqual(len(pl["target_logprobs"]), num_positions - 1)

        for row in pl["topk_logprobs"]:
            for v in row:
                self.assertLessEqual(v, 0.0)

        for row in pl["topk_logprobs"]:
            for i in range(len(row) - 1):
                self.assertGreaterEqual(row[i], row[i + 1])

        for row in pl["topk_token_ids"]:
            for t in row:
                self.assertIsInstance(t, int)
                self.assertGreaterEqual(t, 0)

    def test_03_general_protocol_range_clipping(self):
        """Range clipping (start/end) works correctly."""
        data = self._post(
            "/",
            {
                "prompt": "The quick brown fox jumps over the lazy dog and runs away",
                "generate_config": {
                    "return_prompt_logits": True,
                    "prompt_logits_top_k": 8,
                    "prompt_logits_start": 2,
                    "prompt_logits_end": 6,
                    "max_new_tokens": 1,
                    "return_target_logprob": True,
                },
            },
        )
        pl = data["prompt_logprobs"]
        self.assertEqual(pl["start_pos"], 2)
        self.assertEqual(pl["end_pos"], 6)
        self.assertEqual(len(pl["topk_logprobs"]), 4)
        self.assertEqual(len(pl["topk_logprobs"][0]), 8)
        # end=6 < prompt_len, so every position has a next token as label
        self.assertEqual(len(pl["target_logprobs"]), 4)
        for v in pl["target_logprobs"]:
            self.assertLessEqual(v, 0.0)

    def test_03b_single_token_prompt(self):
        """Single token prompt returns 1 row of top-k and empty target_logprobs."""
        data = self._post(
            "/",
            {
                "prompt": "Hi",
                "generate_config": {
                    "return_prompt_logits": True,
                    "prompt_logits_top_k": 8,
                    "max_new_tokens": 1,
                    "return_target_logprob": True,
                },
            },
        )
        self.assertIn("prompt_logprobs", data)
        pl = data["prompt_logprobs"]
        self.assertIsNotNone(pl)
        num_positions = len(pl["topk_logprobs"])
        self.assertGreaterEqual(num_positions, 1)
        self.assertEqual(len(pl["topk_logprobs"][0]), 8)
        # For very short prompts, target_logprobs may be shorter than topk positions
        # (last position has no next token as label)
        if num_positions == 1:
            # Single position: no next token, target_logprobs should be empty or absent
            target = pl.get("target_logprobs")
            self.assertTrue(target is None or len(target) == 0)

    def test_03c_no_target_logprobs_when_disabled(self):
        """return_target_logprob=false omits target_logprobs from response."""
        data = self._post(
            "/",
            {
                "prompt": "The quick brown fox jumps over the lazy dog",
                "generate_config": {
                    "return_prompt_logits": True,
                    "prompt_logits_top_k": 8,
                    "max_new_tokens": 1,
                    "return_target_logprob": False,
                },
            },
        )
        self.assertIn("prompt_logprobs", data)
        pl = data["prompt_logprobs"]
        self.assertIn("topk_logprobs", pl)
        self.assertIn("topk_token_ids", pl)
        self.assertGreater(len(pl["topk_logprobs"]), 0)
        # target_logprobs should be absent when explicitly disabled
        self.assertNotIn("target_logprobs", pl)

    def test_04_openai_protocol_prompt_logprobs(self):
        """OpenAI protocol returns prompt_logprobs."""
        data = self._post(
            "/v1/chat/completions",
            {
                "model": "qwen",
                "messages": [
                    {
                        "role": "user",
                        "content": "What is 2+2? Answer with just the number.",
                    }
                ],
                "prompt_logprobs": 10,
                "max_tokens": 1,
            },
        )
        self.assertIn("prompt_logprobs", data)
        pl = data["prompt_logprobs"]
        self.assertIsNotNone(pl)
        self.assertIn("topk_logprobs", pl)
        self.assertIn("topk_token_ids", pl)
        self.assertIn("target_logprobs", pl)
        self.assertIn("start_pos", pl)
        self.assertIn("end_pos", pl)

        num_positions = len(pl["topk_logprobs"])
        self.assertGreater(num_positions, 0)
        self.assertEqual(len(pl["topk_logprobs"][0]), 10)

    def test_05_no_prompt_logits_when_disabled(self):
        """Normal request does not include prompt_logits."""
        data = self._post(
            "/",
            {
                "prompt": "Hello",
                "generate_config": {"max_new_tokens": 3},
            },
        )
        pl = data.get("prompt_logprobs")
        self.assertTrue(pl is None or pl == {})

    def test_05c_general_batch_prompt_logits(self):
        """General protocol batch (prompt_batch) returns prompt_logits for each item."""
        prompts = [
            "The quick brown fox jumps over the lazy dog",
            "Hello world, this is a test of prompt scoring",
        ]
        data = self._post(
            "/batch_infer",
            {
                "prompt_batch": prompts,
                "generate_config": {
                    "return_prompt_logits": True,
                    "prompt_logits_top_k": 8,
                    "max_new_tokens": 1,
                    "return_target_logprob": True,
                },
            },
        )
        self.assertIn("response_batch", data)
        self.assertEqual(len(data["response_batch"]), 2)
        for idx, item in enumerate(data["response_batch"]):
            pl = item.get("prompt_logprobs")
            self.assertIsNotNone(pl, f"prompt_logits missing for batch item {idx}")
            self.assertIn("topk_logprobs", pl)
            self.assertIn("topk_token_ids", pl)
            self.assertIn("target_logprobs", pl)
            self.assertGreater(len(pl["topk_logprobs"]), 0)
            self.assertEqual(len(pl["topk_logprobs"][0]), 8)

    def test_05d_openai_batch_prompt_logprobs(self):
        """OpenAI batch protocol returns prompt_logprobs for each item."""
        data = self._post(
            "/v1/batch/chat/completions",
            {
                "requests": [
                    {
                        "model": "qwen",
                        "messages": [{"role": "user", "content": "What is 2+2?"}],
                        "prompt_logprobs": 8,
                        "max_tokens": 1,
                    },
                    {
                        "model": "qwen",
                        "messages": [{"role": "user", "content": "Hello world"}],
                        "prompt_logprobs": 8,
                        "max_tokens": 1,
                    },
                ]
            },
        )
        self.assertIn("responses", data)
        self.assertEqual(len(data["responses"]), 2)
        for idx, resp in enumerate(data["responses"]):
            pl = resp.get("prompt_logprobs")
            self.assertIsNotNone(pl, f"prompt_logprobs missing for batch item {idx}")
            self.assertIn("topk_logprobs", pl)
            self.assertIn("topk_token_ids", pl)
            self.assertGreater(len(pl["topk_logprobs"]), 0)
            self.assertEqual(len(pl["topk_logprobs"][0]), 8)

    def test_05b_dump_smoke_golden(self):
        """Dump golden data for smoke test (only when SAVE_GOLDEN is set)."""
        import json as json_mod

        if not os.environ.get("SAVE_GOLDEN"):
            self.skipTest("SAVE_GOLDEN not set")
        data = self._post(
            "/",
            {
                "prompt": "Which recommended Flask alternative is most similar to Django North in syntax?",
                "generate_config": {
                    "return_prompt_logits": True,
                    "prompt_logits_top_k": 8,
                    "max_new_tokens": 1,
                    "return_target_logprob": True,
                    "top_k": 1,
                    "top_p": 0,
                },
            },
        )
        pl = data["prompt_logprobs"]
        golden = {
            "response": data.get("response", ""),
            "finished": True,
            "prompt_logprobs": {
                "start_pos": pl["start_pos"],
                "end_pos": pl["end_pos"],
                "topk_logprobs_head": pl["topk_logprobs"][:3],
                "topk_token_ids_head": pl["topk_token_ids"][:3],
                "target_logprobs": pl["target_logprobs"],
            },
        }
        print("=== SMOKE GOLDEN BEGIN ===")
        print(json_mod.dumps(golden, indent=2))
        print("=== SMOKE GOLDEN END ===")

    def test_06_precision_vs_hf_ground_truth(self):
        """Compare prompt scoring logprobs against HF FP32 ground truth.

        RTP-LLM uses BF16 for attention computation while HF ground truth is FP32.
        BF16 precision loss accumulates through layers, causing logits differences
        that can reorder tokens with similar probabilities. We verify:
        1. Top-1 token matches at every position
        2. Top-1 logprob value is close (atol=0.5 for BF16 tolerance)
        3. Target logprobs are close (atol=0.5)
        """
        import torch

        gt_path = os.environ.get("GROUND_TRUTH_PATH")
        if not gt_path:
            gt_path = os.path.join(
                os.getcwd(), "rtp_llm/test/prompt_scoring_data", "hf_ground_truth.pt"
            )
        if not os.path.exists(gt_path):
            self.skipTest(f"Ground truth file not found: {gt_path}")

        gt_results = torch.load(gt_path, weights_only=True)
        top_k = 16

        for gt in gt_results:
            prompt = gt["prompt"]
            hf_topk_logprobs = gt["topk_logprobs"]
            hf_topk_ids = gt["topk_indices"]
            hf_target_logprobs = gt["target_logprobs"]

            data = self._post(
                "/",
                {
                    "prompt": prompt,
                    "generate_config": {
                        "return_prompt_logits": True,
                        "prompt_logits_top_k": top_k,
                        "max_new_tokens": 1,
                        "return_target_logprob": True,
                    },
                },
            )
            self.assertIn("prompt_logprobs", data)
            pl = data["prompt_logprobs"]

            rtp_topk_logprobs = torch.tensor(pl["topk_logprobs"], dtype=torch.float32)
            rtp_topk_ids = torch.tensor(pl["topk_token_ids"], dtype=torch.int32)
            rtp_start = pl["start_pos"]
            rtp_end = pl["end_pos"]

            hf_slice_logprobs = hf_topk_logprobs[rtp_start:rtp_end, :top_k]
            hf_slice_ids = hf_topk_ids[rtp_start:rtp_end, :top_k]

            # 1. Top-1 token ID must match at every position
            rtp_top1 = rtp_topk_ids[:, 0]
            hf_top1 = hf_slice_ids[:, 0]
            top1_match = torch.equal(rtp_top1.int(), hf_top1.int())
            if not top1_match:
                mismatch_pos = (
                    (rtp_top1.int() != hf_top1.int()).nonzero().flatten().tolist()
                )
                logging.warning(
                    f"[{prompt[:30]}] top-1 ID mismatch at positions: {mismatch_pos}"
                )
            self.assertTrue(
                top1_match,
                f"Top-1 token ID mismatch for '{prompt[:30]}'",
            )

            # 2. Top-1 logprob value close (BF16 tolerance)
            rtp_top1_val = rtp_topk_logprobs[:, 0]
            hf_top1_val = hf_slice_logprobs[:, 0]
            top1_max_diff = (rtp_top1_val - hf_top1_val).abs().max().item()
            top1_mean_diff = (rtp_top1_val - hf_top1_val).abs().mean().item()
            all_max_diff = (rtp_topk_logprobs - hf_slice_logprobs).abs().max().item()
            all_mean_diff = (rtp_topk_logprobs - hf_slice_logprobs).abs().mean().item()
            # Cosine similarity (flatten tensors as vectors)
            topk_cos = torch.nn.functional.cosine_similarity(
                rtp_topk_logprobs.flatten().unsqueeze(0),
                hf_slice_logprobs.flatten().unsqueeze(0),
            ).item()
            # Per-row cosine similarity (each position as a vector)
            row_cos = torch.nn.functional.cosine_similarity(
                rtp_topk_logprobs, hf_slice_logprobs, dim=1
            )
            row_cos_min = row_cos.min().item()
            row_cos_mean = row_cos.mean().item()
            print(
                f"  [{prompt[:40]}] top1_max_diff={top1_max_diff:.6f}, "
                f"top1_mean_diff={top1_mean_diff:.6f}, "
                f"all_topk_max_diff={all_max_diff:.6f}, "
                f"all_topk_mean_diff={all_mean_diff:.6f}"
            )
            print(
                f"  [{prompt[:40]}] cosine_sim(flatten)={topk_cos:.6f}, "
                f"cosine_sim(per_row): min={row_cos_min:.6f}, mean={row_cos_mean:.6f}"
            )
            self.assertLess(
                top1_max_diff,
                0.2,
                f"Top-1 logprob diff too large for '{prompt[:30]}': {top1_max_diff:.6f}",
            )

            # 3. Target logprobs close
            if pl.get("target_logprobs"):
                rtp_target = torch.tensor(pl["target_logprobs"], dtype=torch.float32)
                hf_target_slice = hf_target_logprobs[
                    rtp_start : rtp_start + len(rtp_target)
                ]
                target_max_diff = (rtp_target - hf_target_slice).abs().max().item()
                target_mean_diff = (rtp_target - hf_target_slice).abs().mean().item()
                target_cos = torch.nn.functional.cosine_similarity(
                    rtp_target.unsqueeze(0), hf_target_slice.unsqueeze(0)
                ).item()
                print(
                    f"  [{prompt[:40]}] target_max_diff={target_max_diff:.6f}, "
                    f"target_mean_diff={target_mean_diff:.6f}, "
                    f"target_cosine_sim={target_cos:.6f}"
                )
                self.assertLess(
                    target_max_diff,
                    0.2,
                    f"Target logprob diff too large for '{prompt[:30]}': {target_max_diff:.6f}",
                )


if __name__ == "__main__":
    main()

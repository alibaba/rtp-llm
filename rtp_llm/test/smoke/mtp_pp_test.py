import json
import logging
import os
import shlex
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from rtp_llm.test.utils.maga_server_manager import MagaServerManager


class MtpPPTest(unittest.TestCase):
    def generate(self, server, prompt, max_new_tokens, sampling_options=None):
        generate_config = {
            "is_streaming": False,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": max_new_tokens,
            "top_k": 1,
            "top_p": 1.0,
            "random_seed": 1234,
            "return_output_ids": True,
            "aux_info": True,
        }
        generate_config.update(sampling_options or {})
        response = requests.post(
            f"http://127.0.0.1:{server.port}/",
            json={
                "prompt": prompt,
                "generate_config": generate_config,
            },
            timeout=180,
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result["finished"], result)
        self.assertEqual(len(result["output_ids"]), 1, result)
        self.assertEqual(len(result["output_ids"][0]), max_new_tokens, result)
        self.assertGreater(result["aux_info"]["iter_count"], 0, result)
        return result

    def run_variant(self, checkpoint, propose_step, reuse_cache=False):
        variant = f"pp2_mtp{propose_step}_reuse{int(reuse_cache)}"
        args = shlex.split(os.environ["SMOKE_ARGS"])
        args += [
            "--role_type",
            "PDFUSION",
            "--reuse_cache",
            str(int(reuse_cache)),
            "--sp_type",
            "mtp" if propose_step else "none",
        ]
        if propose_step:
            args += [
                "--sp_model_type",
                "qwen35_dense_mtp",
                "--sp_checkpoint_path",
                checkpoint,
                "--sp_act_type",
                "BF16",
                "--gen_num_per_cycle",
                str(propose_step),
            ]
        server = MagaServerManager(
            env_args={"RTP_LLM_STREAM_ASYNC": "0"},
            role_name=variant,
            smoke_args_str=shlex.join(args),
        )
        cases = [
            ("The capital of France is", 1),
            ("Count the positive integers in order: 1, 2, 3,", 64),
            ("The quick brown fox jumps over the lazy dog. " * 220 + "Continue:", 64),
        ]
        prefix = cases[-1][0]
        # Repeat the full prompt, then change only its continuation. Reused MTP
        # KV must not retain the successor token from the previous request.
        cases += [
            (prefix, 64),
            (prefix + " Write a poem:", 32),
            (prefix + " Write a recipe:", 32),
            (
                "Continue this pattern: red blue red blue red blue",
                32,
                {
                    "repetition_penalty": 1.2,
                    "presence_penalty": 0.3,
                    "frequency_penalty": 0.2,
                    "no_repeat_ngram_size": 3,
                },
            ),
        ]
        outputs = {}
        output_dir = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "."))
        try:
            self.assertTrue(
                server.start_server(
                    model_path=checkpoint,
                    model_type="qwen35_dense",
                    tokenizer_path=checkpoint,
                ),
                f"{variant} failed to start: {server.log_file_path}",
            )
            outputs["serial"] = []
            for case in cases:
                outputs["serial"].append(self.generate(server, *case))
            if reuse_cache:
                for result in outputs["serial"][3:6]:
                    self.assertGreater(result["aux_info"]["reuse_len"], 0, result)
            else:
                for result in outputs["serial"]:
                    self.assertEqual(result["aux_info"]["reuse_len"], 0, result)
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(self.generate, server, *case) for case in cases[1:]
                ]
                outputs["concurrent"] = [future.result() for future in futures]
        finally:
            server.stop_server()
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{variant}.json").write_text(
                json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return outputs

    def test_pdfusion_mtp_matches_target_generation(self):
        checkpoint = os.environ.get("CHECKPOINT_PATH")
        self.assertTrue(
            checkpoint, "Pass --test_env=CHECKPOINT_PATH=<Qwen3.5-27B checkpoint>"
        )
        baseline = self.run_variant(checkpoint, 0)
        variants = ((step, reuse) for step in (1, 3, 4) for reuse in (False, True))
        for propose_step, reuse_cache in variants:
            with self.subTest(propose_step=propose_step, reuse_cache=reuse_cache):
                actual = self.run_variant(
                    checkpoint, propose_step, reuse_cache=reuse_cache
                )
                for mode in ("serial", "concurrent"):
                    self.assertEqual(
                        [result["output_ids"] for result in actual[mode]],
                        [result["output_ids"] for result in baseline[mode]],
                        f"{mode}: MTP {propose_step} differs from target-only PP",
                    )
                self.assertTrue(
                    any(
                        result["aux_info"]["iter_count"]
                        < len(result["output_ids"][0])
                        for result in actual["serial"][1:]
                    ),
                    f"MTP {propose_step} did not accept any draft tokens",
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()

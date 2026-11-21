"""Compare sequential target-only and real target+NVIDIA draft greedy serving."""

import os
import shlex
import unittest
from pathlib import Path

import requests

from rtp_llm.test.utils.maga_server_manager import MagaServerManager


class Qwen3MoeEagle3Smoke(unittest.TestCase):
    def test_greedy_speculative_tokens(self):
        # CI supplies release snapshots with --test_env; never download model
        # weights or quietly skip this correctness check when an asset is absent.
        def checkpoint(variable):
            path = os.environ.get(variable)
            if not path or not (Path(path) / "config.json").is_file():
                self.fail(f"Set --test_env={variable} to the release checkpoint directory")
            return path

        target = checkpoint("QWEN3_235B_TARGET_CHECKPOINT")
        draft = checkpoint("QWEN3_235B_EAGLE3_CHECKPOINT")
        prompts = (
            "Count from one to twenty: one, two, three, four,",
            "Repeat the word map, separated by spaces: map map map map",
        )
        session = requests.Session()
        self.addCleanup(session.close)
        session.trust_env = False

        def run(spec):
            args = os.environ["SMOKE_ARGS"]
            if spec:
                args += (
                    " --sp_type eagle3 --sp_model_type qwen_3_moe_eagle3 --sp_act_type BF16 --gen_num_per_cycle 3 --sp_checkpoint_path "
                    + shlex.quote(draft)
                )
            server = MagaServerManager(
                env_args={"LOG_LEVEL": "INFO", "FT_SERVER_TEST": "1"},
                role_name="qwen3_235b_eagle3_" + ("draft" if spec else "baseline"),
                smoke_args_str=args,
            )
            tokens, accepted = [], 0
            try:
                self.assertTrue(
                    server.start_server(
                        model_path=target, model_type="qwen_3_moe", timeout=1800
                    )
                )
                for prompt in prompts:
                    response = session.post(
                        f"http://127.0.0.1:{server.port}/",
                        json={
                            "prompt": prompt,
                            "generate_config": {
                                "top_k": 1,
                                "random_seed": 42,
                                "max_new_tokens": 32,
                                "ignore_eos": True,
                                "return_output_ids": True,
                                "aux_info": True,
                            },
                        },
                        timeout=240,
                    )
                    response.raise_for_status()
                    actual = response.json()
                    self.assertIs(actual["finished"], True)
                    ids = actual["output_ids"][0]
                    self.assertEqual(len(ids), 32)
                    tokens.append(ids)
                    aux = actual["aux_info"]
                    if spec:
                        rounds = aux["speculative_draft_rounds"]
                        self.assertGreater(rounds, 1)
                        counts = aux["speculative_accepted_tokens_per_pos"]
                        self.assertTrue(
                            counts and all(0 <= n <= rounds for n in counts)
                        )
                        accepted += sum(counts)
                    else:
                        self.assertEqual(aux["speculative_draft_rounds"], 0)
                log = Path(server.log_file_path).read_text(errors="replace")
                for model in (
                    ("qwen_3_moe", "qwen_3_moe_eagle3") if spec else ("qwen_3_moe",)
                ):
                    self.assertIn(
                        f"model loaded successfully: model_type={model} selected=fastsafetensors",
                        log,
                    )
                if spec:
                    self.assertGreater(
                        accepted, 0, "Target never accepted a draft proposal"
                    )
                return tokens
            finally:
                server.stop_server()

        self.assertEqual(run(False), run(True))


if __name__ == "__main__":
    unittest.main()

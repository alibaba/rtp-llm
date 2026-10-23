"""Five-GPU contract: native FP8 TP4, weighted 64-token cache, and remote ViT.

QWEN4_EXP_CHECKPOINT must point to a complete original FP8 checkpoint. The
fifth GPU hosts two independent ViT workers. QWEN4_SERVING_GRAPH selects
eager (0) or CUDA graph (1); both runs use the same assertions. The animal
variant checks functional correctness; description also diagnoses exact reuse parity.
"""

import argparse
import base64
import json
import logging
import os
import re
import shlex
import sys
import unicodedata
import unittest
from pathlib import Path

import requests
from transformers import AutoTokenizer

from rtp_llm.test.utils.device_resource import get_gpu_ids
from rtp_llm.test.utils.maga_server_manager import MagaServerManager


class Qwen4ServingContractTest(unittest.TestCase):
    image_question = "description"

    def setUp(self):
        self.model_path = os.environ["QWEN4_EXP_CHECKPOINT"]
        self.assertTrue(
            (Path(self.model_path) / "model.safetensors.index.json").is_file()
        )
        self.output_dir = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, local_files_only=True
        )
        generation = json.loads(
            (Path(self.model_path) / "generation_config.json").read_text()
        )
        eos = generation.get("eos_token_id", [])
        self.eos_token_ids = set(eos if isinstance(eos, list) else [eos])
        self.eos_token_ids.add(self.tokenizer.eos_token_id)
        self.eos_token_ids.discard(None)
        devices = list(get_gpu_ids())
        self.assertGreaterEqual(len(devices), 5, "TP4 plus a separate ViT GPU required")
        graph = os.environ.get("QWEN4_SERVING_GRAPH", "0")
        self.assertIn(graph, ("0", "1"))
        self.graph_enabled = graph == "1"
        llm_port, vit_port = (MagaServerManager.get_free_port() for _ in range(2))

        def endpoint(port):
            return dict(
                type="Vipserver", address=f"127.0.0.1:{port}", protocol="http", path="/"
            )

        route = json.dumps(
            dict(
                service_id="qwen4_serving_contract",
                use_local=True,
                role_endpoints=[
                    dict(
                        group="default",
                        pd_fusion_endpoint=endpoint(llm_port),
                        vit_endpoint=endpoint(vit_port),
                    )
                ],
            )
        )
        common = {
            "LOAD_PYTHON_MODEL": "1",
            "FT_SERVER_TEST": "1",
            "FRONTEND_SERVER_COUNT": "1",
            "MODEL_SERVICE_CONFIG": route,
            "TP_SIZE": "1",
            "EP_SIZE": "1",
            "WORLD_SIZE": "1",
        }
        self.vit = MagaServerManager(
            env_args={**common, "VIT_SEPARATION": "1", "ROLE_TYPE": "VIT"},
            port=vit_port,
            device_ids=devices[4:5],
            role_name="qwen4_vit",
            smoke_args_str="--tp_size 1 --world_size 1 --act_type BF16 --use_local 1 --warm_up 0 --vit_server_count 2 --vit_proxy_min_healthy_workers 2 --mm_preprocess_max_workers 1 --use_local_preprocess 0",
        )
        self.addCleanup(self.vit.stop_server)
        self.assertTrue(
            self.vit.start_server(
                model_path=self.model_path, model_type="qwen4_exp", timeout=1800
            )
        )
        weights = json.dumps(
            dict(
                full=4,
                linear=1,
                ple_conv_state=4,
                ple_ngram_ctx=4,
                indexer_kv=4,
                indexer_state=4,
            )
        )
        llm_args = (
            "--tp_size 4 --ep_size 1 --world_size 4 --act_type BF16 "
            "--quantization FP8_PER_BLOCK --use_local 1 --warm_up 0 "
            "--load_method scratch --use_deepep_moe 0 --enable_fp32_lm_head 0 "
            "--max_seq_len 8192 --max_context_batch_size 1 --concurrency_limit 4 "
            "--seq_size_per_block 64 --kernel_seq_size_per_block 64 "
            "--reuse_cache 1 --enable_device_cache 1 --enable_linear_cache_reclaim 1 "
            "--kv_cache_mem_mb 4096 --reserver_runtime_mem_mb 4096 "
            f"--enable_cuda_graph {graph} --decode_capture_config 1,2,4 "
            f"--kv_cache_pool_block_weights {shlex.quote(weights)}"
        )
        self.llm = MagaServerManager(
            env_args={
                **common,
                "WORLD_SIZE": "4",
                "TP_SIZE": "4",
                "ROLE_TYPE": "PDFUSION",
                "VIT_SEPARATION": "2",
                "REMOTE_VIT_SERVER_IP": "localhost",
                "REMOTE_SERVER_PORT": str(vit_port),
            },
            port=llm_port,
            device_ids=devices[:4],
            role_name="qwen4_native_fp8",
            smoke_args_str=llm_args,
        )
        self.addCleanup(self.llm.stop_server)
        (self.output_dir / "configuration.json").write_text(
            json.dumps(
                dict(
                    checkpoint=self.model_path,
                    devices=devices[:5],
                    llm_args=llm_args,
                    vit_workers=2,
                    graph=int(graph),
                    image_question=self.image_question,
                ),
                indent=2,
            )
        )
        self.assertTrue(
            self.llm.start_server(
                model_path=self.model_path, model_type="qwen4_exp", timeout=3600
            )
        )
        self.session = requests.Session()
        self.session.trust_env = False
        self.addCleanup(self.session.close)
        if self.graph_enabled:
            self.assertIn(
                "capture success for batch size",
                Path(self.llm.log_file_path).read_text(errors="replace"),
            )
            level = self.session.post(
                f"http://127.0.0.1:{self.llm.port}/set_log_level",
                json={"log_level": "DEBUG"},
                timeout=10,
            )
            level.raise_for_status()
            self.assertEqual(level.json(), {"status": "ok"})

    def post(self, label, path, query):
        log_path = Path(self.llm.log_file_path)
        offset = log_path.stat().st_size if self.graph_enabled else 0
        response = self.session.post(
            f"http://127.0.0.1:{self.llm.port}{path}", json=query, timeout=600
        )
        (self.output_dir / f"{label}.http.txt").write_text(response.text)
        response.raise_for_status()
        actual = response.json()
        (self.output_dir / f"{label}.json").write_text(
            json.dumps(dict(query=query, actual=actual), indent=2, ensure_ascii=False)
        )
        self.assertNotIn("error_code", actual, actual)
        self.assertNotIn("error", actual, actual)
        if self.graph_enabled and label != "math":
            # Only serving traffic after this request began counts. Startup
            # capture alone cannot distinguish replay from an eager fallback.
            with log_path.open("rb") as log:
                log.seek(offset)
                runtime = log.read().decode(errors="replace")
            batches = re.findall(
                r"\[PyWrappedModel\] using CUDA graph forward, is_target_verify=0, is_prefill=0, graph_bs=(\d+)",
                runtime,
            )
            normal = runtime.count(
                "[PyWrappedModel] using normal forward, is_target_verify=0, is_prefill=0"
            )
            completed = runtime.count("[PyWrappedModel] CUDA graph forward completed")
            (self.output_dir / f"{label}.graph.json").write_text(
                json.dumps(
                    dict(
                        runtime_decode_graph_batches=sorted(set(batches)),
                        graph_forwards=len(batches),
                        completed_forwards=completed,
                        normal_decode_forwards=normal,
                    ),
                    indent=2,
                )
            )
            self.assertIn("1", batches, f"{label}: no runtime decode graph replay")
            self.assertGreater(completed, 0, label)
            self.assertEqual(normal, 0, f"{label}: ordinary decode fell back to eager")
        return actual

    def text_request(self, label, prompt, reuse, new_tokens=32, start=1):
        actual = self.post(
            label,
            "/",
            dict(
                prompt=prompt,
                generate_config=dict(
                    max_new_tokens=new_tokens,
                    min_new_tokens=new_tokens,
                    ignore_eos=True,
                    top_k=1,
                    random_seed=42,
                    reuse_cache=reuse,
                    return_input_ids=True,
                    return_output_ids=True,
                    aux_info=True,
                ),
            ),
        )
        self.assertIs(actual.get("finished"), True, actual)
        self.assertEqual(len(actual["output_ids"]), 1)
        self.assertEqual(len(actual["output_ids"][0]), new_tokens)
        self.assertTrue(
            all(
                type(token) is int and 0 <= token < len(self.tokenizer)
                for token in actual["output_ids"][0]
            )
        )
        self.assertGreater(actual["aux_info"]["input_len"], 3 * 64)
        if self.image_question == "animal":
            answer = self.tokenizer.decode(
                actual["output_ids"][0], skip_special_tokens=True
            )
            self.assertTrue(
                all(
                    char in "0123456789"
                    or char.isspace()
                    or unicodedata.category(char).startswith("P")
                    for char in answer
                ),
                f"{label}: unexpected counting response: {answer!r}",
            )
            matches = list(re.finditer(r"[0-9]+", answer))
            # A fixed token budget may stop inside the next integer. Only an
            # undelimited final strict prefix is incomplete; all earlier
            # integers, and any final integer followed by a separator, count.
            if matches and matches[-1].end() == len(answer):
                tail = matches[-1].group()
                expected = str(start + len(matches) - 1)
                if len(tail) < len(expected) and expected.startswith(tail):
                    matches.pop()
            numbers = [int(match.group()) for match in matches]
            self.assertGreaterEqual(len(numbers), 2, label)
            self.assertEqual(numbers, list(range(start, start + len(numbers))), label)
        return actual

    def test_combined_generation_and_prefix_reuse(self):
        math_result = self.post(
            "math",
            "/v1/chat/completions",
            dict(
                messages=[
                    dict(
                        role="user",
                        content="What is 4 multiplied by 5? Return only the number.",
                    )
                ],
                max_tokens=16,
                temperature=0,
                seed=42,
                chat_template_kwargs=dict(enable_thinking=False),
            ),
        )
        self.assertEqual(math_result["choices"][0]["message"]["content"].strip(), "20")
        prefix = "Here is an archive of independent numbered records.\n" + "".join(
            f"Record {i:04d}: books, maps, and letters are in the library.\n"
            for i in range(40)
        )
        prompts = [
            self.tokenizer.apply_chat_template(
                [
                    dict(
                        role="user",
                        content=prefix
                        + f"\nWrite the integers from {start} to 1000 in order, separated by spaces.",
                    )
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for start in (1, 2)
        ]
        ids = [self.tokenizer.encode(prompt) for prompt in prompts]
        common_len = next(
            (i for i, (a, b) in enumerate(zip(*ids)) if a != b), min(map(len, ids))
        )
        self.assertGreaterEqual(common_len, 3 * 64)
        references = [
            self.text_request(f"uncached_{i}", prompt, False, start=i + 1)
            for i, prompt in enumerate(prompts)
        ]
        for reference in references:
            self.assertEqual(reference["aux_info"]["reuse_len"], 0)
        for label, index, reuse, expect_hit in (
            ("fill", 0, True, False),
            ("repeat", 0, True, True),
            ("fork", 1, True, True),
            ("fork_repeat", 1, True, True),
            ("bypass", 0, False, False),
        ):
            actual = self.text_request(label, prompts[index], reuse, start=index + 1)
            if self.image_question == "description":
                for key in ("input_ids", "output_ids", "response"):
                    self.assertEqual(
                        actual[key], references[index][key], f"{label}: {key}"
                    )
            reused = actual["aux_info"]["reuse_len"]
            if expect_hit:
                self.assertGreaterEqual(reused, 2 * 64, label)
                self.assertEqual(reused % 64, 0)
                self.assertLess(reused, actual["aux_info"]["input_len"])
                if label == "fork":
                    self.assertLessEqual(reused, common_len)
            else:
                self.assertEqual(reused, 0, label)

        # The last sampled token is not forwarded, and checkpoint reclamation
        # drains on the following allocation. Leave enough decode steps for
        # three completed pages and their subsequent allocation, even when the
        # input ends on a page boundary. Keep this inside the intended answer:
        # forced continuation after EOS is not a useful deterministic fixture.
        long_reference = self.text_request("long_uncached", prompts[0], False, 256)
        long_cached = self.text_request("long_cached", prompts[0], True, 256)
        for actual in (long_reference, long_cached):
            self.assertTrue(self.eos_token_ids.isdisjoint(actual["output_ids"][0]))
        self.assertEqual(long_reference["aux_info"]["reuse_len"], 0)
        self.assertGreaterEqual(long_cached["aux_info"]["reuse_len"], 2 * 64)
        if self.image_question == "description":
            for key in ("input_ids", "output_ids", "response"):
                self.assertEqual(long_reference[key], long_cached[key], key)

        # The image is transferred over the actual LLM -> proxy -> GPU worker path.
        image_path = Path(__file__).parent / "data/model/qwen_vl/1.jpeg"
        image_url = (
            "data:image/jpeg;base64,"
            + base64.b64encode(image_path.read_bytes()).decode()
        )
        image_prompt = (
            "What animal is next to the woman? Answer with exactly one lowercase word."
            if self.image_question == "animal"
            else "Describe the main visible objects in this picture."
        )
        vision_query = dict(
            messages=[
                dict(
                    role="user",
                    content=[
                        dict(type="image_url", image_url=dict(url=image_url)),
                        dict(
                            type="text",
                            text=image_prompt,
                        ),
                    ],
                )
            ],
            max_tokens=32,
            temperature=0,
            seed=42,
            chat_template_kwargs=dict(enable_thinking=False),
            extra_configs=dict(
                reuse_cache=False, return_input_ids=True, return_output_ids=True
            ),
            aux_info=True,
        )
        if self.image_question == "description":
            vision_query.update(logprobs=True, top_logprobs=5)
        image_reference = self.post(
            "vision_uncached", "/v1/chat/completions", vision_query
        )
        self.assertEqual(image_reference["aux_info"]["reuse_len"], 0)
        vision_query["extra_configs"]["reuse_cache"] = True
        image_fill = self.post("vision_fill", "/v1/chat/completions", vision_query)
        image_hit = self.post("vision_hit", "/v1/chat/completions", vision_query)
        self.assertEqual(image_fill["aux_info"]["reuse_len"], 0)
        reused = image_hit["aux_info"]["reuse_len"]
        self.assertGreaterEqual(reused, 64)
        self.assertEqual(reused % 64, 0)
        self.assertLess(reused, image_hit["aux_info"]["input_len"])
        for actual in (image_reference, image_fill, image_hit):
            self.assertGreater(actual["usage"]["prompt_tokens"], 100)
            self.assertGreaterEqual(actual["aux_info"]["input_len"], 128)
            self.assertGreater(actual["usage"]["completion_tokens"], 0)
            self.assertTrue(actual["choices"][0]["message"]["content"].strip())
            if self.image_question == "animal":
                self.assertEqual(
                    re.findall(
                        r"[a-z]+",
                        actual["choices"][0]["message"]["content"].casefold(),
                    ),
                    ["dog"],
                )
            self.assertTrue(actual["extra_outputs"]["input_ids"])
            self.assertTrue(actual["extra_outputs"]["output_ids"])
            if self.image_question == "description":
                self.assertEqual(
                    image_reference["choices"][0]["message"],
                    actual["choices"][0]["message"],
                )
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    self.assertEqual(
                        image_reference["usage"][key], actual["usage"][key]
                    )
                for key in ("input_ids", "output_ids"):
                    self.assertEqual(
                        image_reference["extra_outputs"][key],
                        actual["extra_outputs"][key],
                    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--image-question", choices=("description", "animal"), default="description"
    )
    options, test_args = parser.parse_known_args()
    Qwen4ServingContractTest.image_question = options.image_question
    unittest.main(argv=[sys.argv[0], *test_args])

"""Real K3 target/MTP inference with a visual feature crossing a 32K checkpoint."""

import base64
import copy
import io
import json
import math
import os
import re
import unittest
from contextlib import ExitStack
from pathlib import Path

import requests
from PIL import Image

from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_image_processor import (
    KimiK3VisionProcessor,
    load_kimi_k3_media_config,
)
from rtp_llm.test.utils.device_resource import get_gpu_ids
from rtp_llm.test.utils.maga_server_manager import MagaServerManager
from rtp_llm.test.utils.port_util import PortManager

CHUNK_SIZE = 32768
OUTPUT_TOKENS = 32
PAD_UNIT = "A quiet river flows beside the village. "
DESCRIBE = "Write at least 100 words describing the image colors, shapes, and their arrangement in detail."
SERVER_ARGS = (
    "--role_type PDFUSION --tp_size {tp_size} --dp_size 1 --ep_size {tp_size} "
    "--ffn_sp_size {tp_size} --world_size {tp_size} --local_world_size {tp_size} "
    "--act_type BF16 --quantization FP8_PER_BLOCK "
    "--prefill_cp_size 1 --prefill_cp_kv_cache_sharded 1 "
    "--seq_size_per_block 4096 --kernel_seq_size_per_block 128 --linear_step 1 "
    "--ssm_state_dtype fp32 --max_seq_len 65536 --max_batch_tokens_size 65536 "
    "--max_context_batch_size 1 --concurrency_limit 1 --kv_cache_mem_mb 8192 "
    "--reuse_cache 1 --enable_device_cache 1 --enable_memory_cache 0 "
    "--enable_cuda_graph 0 --warm_up 0 --use_local 1 --use_local_preprocess 1 "
    "--moe_strategy mega_moe --load_method fastsafetensors --cache_store_rdma_mode 0 "
    "--reserver_runtime_mem_mb 14336 --enable_detail_log 1"
)


class ChunkedKimiK3Test(unittest.TestCase):
    def setUp(self):
        self.checkpoint = Path(os.environ["K3_CKPT_PATH"])
        self.draft = Path(os.environ["K3_MTP_CKPT_PATH"])
        config, weights = self.check_checkpoint(self.checkpoint)
        draft_config, _ = self.check_checkpoint(self.draft)
        self.assertEqual(config["model_type"], "kimi_k3")
        self.assertEqual(draft_config["model_type"], "kimi_k3")
        self.assertTrue(config["vision_config"])
        text = config.get("text_config", config)
        self.layers = text["num_hidden_layers"]
        self.assertIn(self.layers, (4, 93))
        self.tp_size = int(os.environ.get("WORLD_SIZE", "8"))
        self.assertIn(self.tp_size, (4, 8))
        if self.tp_size < 8:
            self.assertEqual(self.layers, 4, "93-layer K3 requires eight GPUs")
        self.server_args = SERVER_ARGS.format(tp_size=self.tp_size)
        self.assertTrue(text["linear_attn_config"]["kda_layers"])
        self.assertTrue(text["linear_attn_config"]["full_attn_layers"])
        self.assertTrue(any(name.startswith("vision_tower.") for name in weights))
        self.assertTrue(any(name.startswith("mm_projector.") for name in weights))
        self.placeholder = config["media_placeholder_token_id"]
        self.assertEqual(draft_config["media_placeholder_token_id"], self.placeholder)
        self.processor = KimiK3VisionProcessor(
            load_kimi_k3_media_config(str(self.checkpoint))
        )
        self.output_dir = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_root = self.output_dir
        self.locks = ExitStack()
        self.addCleanup(self.locks.close)
        self.session = requests.Session()
        self.session.trust_env = False
        self.addCleanup(self.session.close)
        self.images = []
        for color in ((192, 32, 32), (32, 32, 192)):
            image = Image.new("RGB", (896, 896), color)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            self.images.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,"
                        + base64.b64encode(buffer.getvalue()).decode()
                    },
                }
            )
        self.native_stats = []
        self.save(
            "model_profile.json",
            {
                "target": str(self.checkpoint),
                "draft": str(self.draft),
                "target_layers": self.layers,
                "full_model": self.layers == 93,
                "tp_size": self.tp_size,
                "act_type": "BF16",
                "target_quantization": "FP8_PER_BLOCK",
                "draft_quantization": "",
                "chunk_size": CHUNK_SIZE,
                "server_args": self.server_args,
            },
        )

    def check_checkpoint(self, path):
        config = json.loads((path / "config.json").read_text())
        weights = json.loads((path / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        self.assertTrue(weights, f"empty checkpoint index: {path}")
        for name in set(weights.values()):
            self.assertGreater((path / name).stat().st_size, 8, str(path / name))
        return config, weights

    def save(self, name, value):
        (self.output_dir / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2)
        )

    def start_server(self, chunk_size, native_mtp, stack):
        devices = get_gpu_ids()
        self.assertGreaterEqual(
            len(devices), self.tp_size, "reserve the configured GPUs with gpu_lock"
        )
        ports, locks = PortManager(
            lock_dir=self.output_dir / "port_locks"
        ).get_consecutive_ports(200)
        for lock in locks:
            self.locks.callback(lock.__exit__, None, None, None)
        mode = "native_mtp" if native_mtp else "ordinary"
        env = {
            "LOAD_PYTHON_MODEL": "1",
            "WORLD_SIZE": str(self.tp_size),
            "FRONTEND_SERVER_COUNT": "1",
            "LOG_LEVEL": "DEBUG",
            "FT_SERVER_TEST": "1",
            "SP_TYPE": "",
            "SP_MODEL_TYPE": "",
            "SP_CHECKPOINT_PATH": "",
            "SP_QUANTIZATION": "",
            "FT_DISABLE_CUSTOM_AR": "1",
            "ENABLE_LINEAR_REPLAY": "1",
            "FASTSAFETENSORS_NOGDS": "0",
            "RTP_LLM_SMOKE_EVIDENCE": "1",
            "MEGA_MOE_INPUT_PACKER_IMPL": "fast_finite",
        }
        if native_mtp:
            env.update(
                SP_TYPE="mtp",
                SP_MODEL_TYPE="kimi_k3_mtp",
                SP_CHECKPOINT_PATH=str(self.draft),
                SP_ACT_TYPE="BF16",
                GEN_NUM_PER_CIRCLE="3",
            )
        manager = MagaServerManager(
            device_ids=devices[: self.tp_size],
            port=str(ports[0] + 100),
            role_name=f"kimi_k3_{mode}_chunk{chunk_size}",
            env_args=env,
            smoke_args_str=self.server_args + f" --prefill_chunk_size {chunk_size}",
        )
        stack.callback(manager.stop_server)
        startup_offsets = self.log_offsets()
        self.assertTrue(
            manager.start_server(
                model_type="kimi_k3", model_path=str(self.checkpoint), timeout=1800
            ),
            "real K3 target/MTP server failed to start",
        )
        url = f"http://127.0.0.1:{manager.port}"
        # Engine initialization reloads alog.conf after LOG_LEVEL is read.
        self.assertEqual(
            self.post_json(url, "/set_log_level", {"log_level": "DEBUG"}),
            {"status": "ok"},
        )
        startup = self.collect_evidence(startup_offsets)
        self.save(f"{mode}_chunk{chunk_size}.startup_evidence.json", startup)
        if chunk_size:
            self.assertTrue(
                any(
                    re.search(
                        rf"token_budget={CHUNK_SIZE}, intermediate_chunk_alignment={4096 * self.tp_size}\b",
                        row,
                    )
                    for row in startup["alignment_rows"]
                ),
                "scheduler did not confirm the configured chunk budget/alignment",
            )
        return url

    def post_json(self, url, endpoint, query):
        response = self.session.post(url + endpoint, json=query, timeout=900)
        response.raise_for_status()
        return response.json()

    def query(self, content, streaming):
        return {
            "model": "kimi_k3",
            "messages": [{"role": "user", "content": content}],
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": OUTPUT_TOKENS,
            "temperature": 1.0,
            "top_p": 1.0,
            "stream": streaming,
            **({"stream_options": {"include_usage": True}} if streaming else {}),
            "logprobs": True,
            "top_logprobs": 5,
            "logprobs_mode": "original",
            "extra_configs": {
                "top_k": 1,
                "do_sample": True,
                "min_new_tokens": OUTPUT_TOKENS,
                "return_all_probs": 2,
                "aux_info": True,
                "reuse_cache": True,
            },
        }

    def render_layout(self, url, query):
        rendered = self.post_json(url, "/v1/chat/render", query)
        ids = rendered["input_ids"]
        locations = [i for i, token in enumerate(ids) if token == self.placeholder]
        image_count = sum(
            item["type"] == "image_url" for item in query["messages"][0]["content"]
        )
        self.assertEqual(len(locations), image_count)
        spans, added = [], 0
        for location in locations:
            start = location + added
            spans.append(
                {
                    "start": start,
                    "end": start + self.feature_length,
                    "vision_start": start + self.vision_offset,
                    "vision_end": start + self.vision_offset + self.vision_tokens,
                }
            )
            added += self.feature_length - 1
        self.assertLess(spans[0]["vision_start"], CHUNK_SIZE)
        self.assertGreater(spans[0]["vision_end"], CHUNK_SIZE)
        return {
            "input_ids": ids,
            "spans": spans,
            "physical_prompt_tokens": len(ids) + added,
            "usage_prompt_tokens": len(ids) + added - self.pending_response_count,
        }

    def make_plan(self, url, label, images, native_mtp=False, streaming=False):
        # Fit using the service's actual XTML tokenizer, not an estimated token ratio.
        target = CHUNK_SIZE - self.vision_tokens // 2 - self.vision_offset
        unit_ids = self.post_json(url, "/tokenizer/encode", {"prompt": PAD_UNIT})[
            "token_ids"
        ]
        lower, upper = 0, CHUNK_SIZE // len(unit_ids) + 64

        def build(count):
            content = [
                {"type": "text", "text": f"Fixture {label}. " + PAD_UNIT * count},
                *images,
                {"type": "text", "text": DESCRIBE},
            ]
            return self.query(content, streaming)

        def placeholder_position(count):
            rendered = self.post_json(url, "/v1/chat/render", build(count))
            return rendered["input_ids"].index(self.placeholder)

        # Repeating a sentence can merge tokens across its boundaries. Grow a
        # rendered bracket before searching the standalone token estimate.
        while placeholder_position(upper) < target:
            lower = upper + 1
            upper *= 2
            self.assertLessEqual(upper, CHUNK_SIZE)
        while lower < upper:
            middle = (lower + upper) // 2
            if placeholder_position(middle) < target:
                lower = middle + 1
            else:
                upper = middle
        query = build(max(0, lower - 1))
        layout = self.render_layout(url, query)
        self.assertLess(layout["physical_prompt_tokens"] + OUTPUT_TOKENS, 65536)
        return {"query": query, "layout": layout, "native_mtp": native_mtp}

    def log_offsets(self):
        return {path: path.stat().st_size for path in self.log_root.rglob("*.log")}

    def collect_evidence(self, offsets):
        context, events, alignment = [], [], []
        for path in self.log_root.rglob("*.log"):
            with path.open("rb") as reader:
                reader.seek(offsets.get(path, 0))
                for raw_line in reader:
                    line = raw_line.decode(errors="replace")
                    if "context stream:" in line:
                        context.append(line.rstrip())
                    if "intermediate_chunk_alignment=" in line:
                        alignment.append(line.rstrip())
                    if "[RTP_SMOKE_EVENT] " in line:
                        payload = line.split("[RTP_SMOKE_EVENT] ", 1)[1]
                        event, _ = json.JSONDecoder().raw_decode(payload)
                        events.append(event)
        prefixes = sorted(
            {
                int(match.group(1))
                for line in context
                if (match := re.search(r"reuse_length:\s*(\d+)", line))
            }
        )
        return {
            "prefixes": prefixes,
            "context_rows": context,
            "events": events,
            "alignment_rows": alignment,
        }

    def infer(self, url, label, plan, reuse=0):
        prefix = f"chunk{self.chunk_size}_{label}"
        query = plan["query"]
        self.save(prefix + ".query.json", query)
        self.save(prefix + ".layout.json", plan["layout"])
        offsets = self.log_offsets()
        with self.session.post(
            url + "/v1/chat/completions",
            json=query,
            stream=query["stream"],
            timeout=900,
        ) as response:
            if not response.ok:
                self.save(
                    prefix + ".error.json",
                    {"status": response.status_code, "body": response.text},
                )
            response.raise_for_status()
            events, raw, done = [], [], False
            if query["stream"]:
                for line in response.iter_lines(decode_unicode=True):
                    raw.append(line)
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload.lower() == "[done]":
                        done = True
                    else:
                        events.append(json.loads(payload))
                self.save(prefix + ".sse.json", raw)
            else:
                events = [response.json()]
        self.save(prefix + ".http.json", events)
        rows, text, usages, auxiliaries, finished = [], [], [], [], False
        for event in events:
            self.assertNotIn("error", event, event)
            if event.get("usage"):
                usages.append(event["usage"])
            if event.get("aux_info"):
                auxiliaries.append(event["aux_info"])
            for choice in event.get("choices", []):
                self.assertEqual(choice["index"], 0)
                rows.extend((choice.get("logprobs") or {}).get("content") or [])
                text.append(
                    (choice.get("delta") or choice.get("message") or {}).get("content")
                    or ""
                )
                finished |= bool(choice.get("finish_reason"))
        self.assertTrue(finished or done, "missing terminal completion")
        self.assertGreaterEqual(len(rows), 1)
        self.assertLessEqual(len(rows), OUTPUT_TOKENS)
        self.assertEqual(
            b"".join(bytes(row["bytes"]) for row in rows).decode("utf-8"),
            "".join(text),
        )
        self.assertTrue(usages, "missing usage")
        self.assertTrue(auxiliaries, "missing execution metadata")
        usage, aux = usages[-1], auxiliaries[-1]
        self.assertEqual(usage["completion_tokens"], OUTPUT_TOKENS)
        self.assertEqual(usage["prompt_tokens"], plan["layout"]["usage_prompt_tokens"])
        self.assertEqual(aux["reuse_len"], reuse)
        expected_features = self.feature_length * len(plan["layout"]["spans"])
        self.assertEqual(sum(aux["multimodal_lengths"].values()), expected_features)
        for row in rows:
            self.assertEqual(len(row["top_logprobs"]), 5)
            for score in [
                row["logprob"],
                *(item["logprob"] for item in row["top_logprobs"]),
            ]:
                self.assertTrue(math.isfinite(score))
                self.assertLessEqual(score, 1e-6)
        evidence = self.collect_evidence(offsets)
        self.save(prefix + ".execution_evidence.json", evidence)
        self.assertIn(reuse, evidence["prefixes"])
        prefill_events = [
            event
            for event in evidence["events"]
            if event.get("event") == "mla_page_rr_prefill"
        ]
        prefill_lengths = {event["logical_tokens"] for event in prefill_events}
        remaining = plan["layout"]["physical_prompt_tokens"] - reuse
        tail = remaining % self.chunk_size if self.chunk_size else remaining
        self.assertIn(tail or self.chunk_size, prefill_lengths)
        for event in prefill_events:
            self.assertEqual(event["page_tokens"], 4096)
            self.assertEqual(event["shards"], self.tp_size)
        if self.chunk_size and not reuse:
            self.assertIn(CHUNK_SIZE, evidence["prefixes"], "no intermediate chunk")
            self.assertIn(CHUNK_SIZE, prefill_lengths, "no real full chunk forward")
        if plan["native_mtp"]:
            rounds = aux["speculative_draft_rounds"]
            accepted = aux["speculative_accepted_tokens_per_pos"]
            self.assertGreater(rounds, 0, "native MTP did not draft")
            self.assertTrue(
                any(
                    e.get("event") == "mtp_target_verify_forward"
                    for e in evidence["events"]
                )
            )
            self.assertTrue(all(0 <= count <= rounds for count in accepted))
            self.native_stats.append(
                {"case": prefix, "rounds": rounds, "accepted": sum(accepted)}
            )
        else:
            self.assertEqual(aux["speculative_draft_rounds"], 0)
        actual = {"text": "".join(text), "rows": rows, "usage": usage, "aux": aux}
        self.save(prefix + ".json", actual)
        return actual

    def compare(self, label, reference, actual, reused=False):
        sample_deltas, top_deltas = [], []
        divergence = None
        compared_rows = 0
        for index, (expected, observed) in enumerate(
            zip(reference["rows"], actual["rows"])
        ):
            # Scores after a different generated token have different input
            # prefixes and cannot measure numerical agreement.
            compared_rows += 1
            expected_top = {
                (tuple(c["bytes"]), c["token"]): c["logprob"]
                for c in expected["top_logprobs"]
            }
            observed_top = {
                (tuple(c["bytes"]), c["token"]): c["logprob"]
                for c in observed["top_logprobs"]
            }
            top_deltas.extend(
                abs(observed_top[k] - expected_top[k])
                for k in expected_top.keys() & observed_top.keys()
            )
            if observed["bytes"] != expected["bytes"]:
                divergence = {
                    "visible_token_index": index,
                    "reference": expected,
                    "actual": observed,
                }
                break
            sample_deltas.append(abs(observed["logprob"] - expected["logprob"]))
        # BF16/FP8 prefill shapes can change rounded scores; report the drift
        # while enforcing exact greedy outputs and the probability-row contract.
        self.save(
            label + ".precision.json",
            {
                "compared_rows_with_identical_input_prefix": compared_rows,
                "first_divergence": divergence,
                "max_sample_logprob_delta": max(sample_deltas, default=None),
                "max_common_top_logprob_delta": max(top_deltas, default=None),
                "strict_score_parity": False,
            },
        )
        self.assertEqual(actual["text"], reference["text"])
        self.assertEqual(
            [r["bytes"] for r in actual["rows"]],
            [r["bytes"] for r in reference["rows"]],
        )
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self.assertEqual(actual["usage"][field], reference["usage"][field])
        if not reused:
            self.assertEqual(actual["usage"], reference["usage"])

    def run_mode(self, native_mtp):
        plans, references = {}, {}
        mode = "native_mtp" if native_mtp else "ordinary"
        for chunk_size in (0, CHUNK_SIZE):
            self.chunk_size = chunk_size
            with ExitStack() as stack:
                url = self.start_server(chunk_size, native_mtp, stack)
                if not chunk_size:
                    self.pending_response_count = len(
                        self.post_json(
                            url,
                            "/tokenizer/encode",
                            {"prompt": "<|open|>response<|sep|>"},
                        )["token_ids"]
                    )
                    image_prompt = self.processor.make_image_prompt(896, 896)
                    image_ids = self.post_json(
                        url, "/tokenizer/encode", {"prompt": image_prompt}
                    )["token_ids"]
                    self.assertEqual(image_ids.count(self.placeholder), 1)
                    self.vision_offset = image_ids.index(self.placeholder)
                    self.vision_tokens = self.processor.resize_config_for_size(
                        896, 896
                    )["num_tokens"]
                    self.feature_length = len(image_ids) - 1 + self.vision_tokens
                for case, images, streaming in (
                    ("one_image", self.images[:1], False),
                    ("two_different_images", self.images, False),
                    ("streaming_image", self.images[1:], True),
                ):
                    label = f"{case}_{mode}"
                    if not chunk_size:
                        plans[label] = self.make_plan(
                            url, label, images, native_mtp, streaming
                        )
                    actual = self.infer(url, label, plans[label])
                    if not chunk_size:
                        references[label] = actual
                    else:
                        with self.subTest(case=label):
                            self.compare(label, references[label], actual)
                if not chunk_size:
                    reuse_plan = self.make_plan(
                        url, f"image_prefix_reuse_{mode}", self.images[:1], native_mtp
                    )
                    reuse_plan["query"]["messages"][0]["content"][-1]["text"] = (
                        DESCRIBE + " Compare lighter and darker areas."
                    )
                    reuse_plan["layout"] = self.render_layout(url, reuse_plan["query"])
                    reuse_reference = self.infer(
                        url, f"reuse_fresh_reference_{mode}", reuse_plan
                    )
                else:
                    seed = copy.deepcopy(reuse_plan)
                    seed["query"]["messages"][0]["content"][-1]["text"] = (
                        DESCRIBE + " Explain which color is dominant."
                    )
                    seed["layout"] = self.render_layout(url, seed["query"])
                    self.assertEqual(
                        seed["layout"]["spans"], reuse_plan["layout"]["spans"]
                    )
                    self.infer(url, f"reuse_seed_{mode}", seed)
                    actual = self.infer(
                        url,
                        f"reuse_inside_image_{mode}",
                        reuse_plan,
                        reuse=CHUNK_SIZE,
                    )
                    with self.subTest(case=f"reuse_inside_image_{mode}"):
                        self.compare(
                            f"reuse_inside_image_{mode}",
                            reuse_reference,
                            actual,
                            reused=True,
                        )

    def test_real_k3_images_chunks_mtp_and_image_prefix_reuse(self):
        for native_mtp in (False, True):
            self.run_mode(native_mtp)
        self.save("native_mtp_summary.json", self.native_stats)
        # The truncated target need not agree with the full-model draft.
        # Both modes still require real draft rounds and target verification.
        if self.layers == 93:
            for chunk_size in (0, CHUNK_SIZE):
                self.assertTrue(
                    any(
                        row["case"].startswith(f"chunk{chunk_size}_")
                        and row["accepted"] > 0
                        for row in self.native_stats
                    ),
                    f"no real MTP tokens accepted with chunk_size={chunk_size}",
                )


if __name__ == "__main__":
    unittest.main()

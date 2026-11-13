"""Real-model scenarios for single-service and P/D-separated Qwen4 profiles.

Every cache restore scenario establishes its own fill/hit sequence. Uncached
requests provide references without relying on unittest method ordering.
"""

import copy
import json
import logging
import os
import re
import shlex
import sys
import unittest
from pathlib import Path

# The shared comparers import their peers through the smoke package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PAGE_SIZE = 128
OUTPUT_TOKENS = 160


def build_prompt_pair(tokenizer, token_budget, compress_ratio, render_prompt=None):
    if token_budget <= 0 or compress_ratio <= 1 or PAGE_SIZE % compress_ratio:
        raise ValueError("Expected page-aligned QSA geometry with a compressed tail")
    minimum = max(4096, token_budget + 2 * PAGE_SIZE)
    line = "The archive records books, maps, letters, and journeys.\n"
    prefix = "The archive keyword is MAP.\n" + line
    while len(tokenizer.encode(prefix, add_special_tokens=False)) < minimum:
        prefix += line
    for padding in range(2 * PAGE_SIZE):
        shared = prefix + " x" * padding
        first = shared + (
            "\nRepeat the archive keyword 200 times, separated by spaces. "
            "Output no explanation or other text."
        )
        if render_prompt is not None:
            first = render_prompt(first)
        tokens = tokenizer.encode(first, add_special_tokens=False)
        # Begin at the final token of a page. Decode crosses two page boundaries
        # and the incomplete compression block before the next complete block.
        if len(tokens) % PAGE_SIZE == PAGE_SIZE - 1:
            second = shared + (
                "\nRepeat the archive keyword 200 times, one per line. "
                "Output no explanation or other text."
            )
            if render_prompt is not None:
                second = render_prompt(second)
            return first, second
    raise AssertionError("Tokenizer could not construct the requested QSA page tail")


def validate_generation(actual, expected_ids, output_tokens, vocab_size, speculative):
    assert "error_code" not in actual, actual
    assert actual.get("finished") is True, actual
    assert actual["input_ids"] == [
        expected_ids
    ], "Server consumed different prompt tokens"
    tokens = actual["output_ids"]
    assert len(tokens) == 1 and len(tokens[0]) == output_tokens, tokens
    assert all(type(t) is int and 0 <= t < vocab_size for t in tokens[0]), tokens
    aux = actual["aux_info"]
    assert aux["input_len"] == len(expected_ids), aux
    assert aux["output_len"] == output_tokens, aux
    assert 0 <= aux["reuse_len"] < len(expected_ids), aux
    if speculative:
        rounds = aux["speculative_draft_rounds"]
        accepted = aux["speculative_accepted_tokens_per_pos"]
        assert rounds >= 2, "No repeated MTP draft/verify rounds"
        assert accepted and all(
            type(n) is int and 0 <= n <= rounds for n in accepted
        ), aux
        assert sum(accepted) > 0, "No speculative token was accepted"
    else:
        assert (
            aux["speculative_draft_rounds"] == 0
        ), "Non-speculative generation ran speculation"


def server_option(smoke_args, name, default=None):
    tokens = shlex.split(smoke_args)
    value = default
    for index, token in enumerate(tokens):
        if token == name:
            value = tokens[index + 1]
        elif token.startswith(name + "="):
            value = token.split("=", 1)[1]
    return value


def execution_modes(smoke_args):
    return (
        server_option(smoke_args, "--enable_cuda_graph", "0") == "1",
        server_option(smoke_args, "--sp_type", "none") == "mtp",
    )


def role_specs(smoke_args, gpu_ids, pd_separation):
    if not pd_separation:
        return {"main": {"args": smoke_args, "devices": []}}
    if any(
        int(server_option(smoke_args, flag, default)) != expected
        for flag, default, expected in (
            ("--tp_size", "1", 2),
            ("--dp_size", "1", 1),
            ("--pp_size", "1", 1),
            ("--world_size", "2", 2),
        )
    ):
        raise ValueError("Qwen4 P/D smoke requires TP2, DP1, PP1 per role")
    if len(gpu_ids) < 4 or len(set(gpu_ids[:4])) != 4:
        raise ValueError("Qwen4 P/D smoke requires four distinct reserved GPUs")
    specs = {}
    for role, devices in (("prefill", gpu_ids[:2]), ("decode", gpu_ids[2:4])):
        # The server parser uses the last occurrence of an option.
        args = (
            smoke_args
            + f" --role_type {role.upper()} --world_size 2 --local_world_size 2"
        )
        graph = (
            "0"
            if role == "prefill"
            else server_option(smoke_args, "--enable_cuda_graph", "0")
        )
        args += " --enable_cuda_graph " + graph
        specs[role] = {
            "devices": list(devices),
            "args": args,
        }
    return specs


def routed_query(query, endpoint, pd_separation):
    query = copy.deepcopy(query)
    field = "generate_config" if endpoint == "/" else "extra_configs"
    query.setdefault(field, {})["can_use_pd_separation"] = pd_separation
    return query


def historical_reuse(aux, pd_separation):
    return aux["prefill_total_reuse_len" if pd_separation else "reuse_len"]


def validate_gr_coverage(log, enabled, mtp_enabled):
    observed = set(
        re.findall(
            r"Qwen4 fused GR enabled: rank=([01]) role=DECODE is_mtp=(True|False)", log
        )
    )
    expected = (
        {
            (str(rank), mtp)
            for rank in (0, 1)
            for mtp in (("False", "True") if mtp_enabled else ("False",))
        }
        if enabled
        else set()
    )
    assert (
        observed == expected
    ), f"GR admission coverage: expected {expected}, observed {observed}"


class Qwen4SM120ScenariosTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import requests
        from transformers import AutoTokenizer

        fixture = (
            Path(__file__).parent / "data/model/qwen4_exp/q_r_qwen4_exp_fp8_full.json"
        )
        cls.task_info = json.loads(fixture.read_text())
        cls.model_path = os.environ.get(
            "QWEN4_EXP_CHECKPOINT", cls.task_info["model_path"]
        )
        raw = json.loads((Path(cls.model_path) / "config.json").read_text())
        if os.environ.get("QWEN4_REQUIRE_NVFP4") == "1":
            quant = raw.get("quantization_config") or {}
            if (
                quant.get("quant_method") != "modelopt"
                or str(quant.get("quant_algo", "")).upper() != "NVFP4"
            ):
                raise ValueError("This smoke profile requires an NVFP4 checkpoint")
        cls.text_config = raw.get("text_config", raw)
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_path, trust_remote_code=True, local_files_only=True
        )
        cls.output_dir = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "."))
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.observed = []
        cls.graph_batches = []
        cls.baseline = None
        cls.session = requests.Session()
        cls.session.trust_env = False
        cls.addClassCleanup(cls.session.close)
        smoke_args = os.environ["SMOKE_ARGS"]
        cls.graph_enabled, cls.mtp_enabled = execution_modes(smoke_args)
        if cls.mtp_enabled:
            smoke_args += " --sp_checkpoint_path " + shlex.quote(cls.model_path)
        cls.pd_separation = os.environ.get("QWEN4_SMOKE_PD", "0") == "1"
        cls.gr_enabled = os.environ.get("RTP_QWEN4_FUSED_GR", "0") == "1"
        cls.gr_mtp_enabled = (
            cls.mtp_enabled and os.environ.get("RTP_QWEN4_FUSED_GR_MTP", "1") == "1"
        )
        if cls.gr_enabled and not cls.pd_separation:
            raise ValueError("GR smoke coverage requires P/D separation")
        cls.start_servers(smoke_args)
        cls.addClassCleanup(cls.write_coverage)

    @classmethod
    def start_servers(cls, smoke_args):
        from concurrent.futures import ThreadPoolExecutor

        from rtp_llm.server.host_service import EndPoint, GroupEndPoint, ServiceRoute
        from rtp_llm.test.utils.device_resource import get_gpu_ids
        from rtp_llm.test.utils.maga_server_manager import MagaServerManager

        specs = role_specs(
            smoke_args,
            list(get_gpu_ids()) if cls.pd_separation else [],
            cls.pd_separation,
        )
        ports = {role: MagaServerManager.get_free_port() for role in specs}
        common_env = {
            "RTP_LLM_STREAM_ASYNC": "1",
            "LOG_LEVEL": "INFO",
            "FT_SERVER_TEST": "1",
        }
        if cls.pd_separation:

            def endpoint(role):
                return EndPoint(
                    type="Vipserver",
                    address=f"127.0.0.1:{ports[role]}",
                    protocol="http",
                    path="/",
                )

            route = ServiceRoute(
                service_id="qwen4_smoke",
                use_local=True,
                role_endpoints=[
                    GroupEndPoint(
                        group="default",
                        prefill_endpoint=endpoint("prefill"),
                        decode_endpoint=endpoint("decode"),
                    )
                ],
            )
            common_env.update(
                WORLD_SIZE="2",
                LOCAL_WORLD_SIZE="2",
                WORLD_RANK="0",
                REMOTE_RPC_SERVER_IP="localhost",
                MODEL_SERVICE_CONFIG=route.model_dump_json(),
            )
        cls.servers = {}
        for role, spec in specs.items():
            env = dict(common_env)
            if cls.pd_separation:
                env["REMOTE_SERVER_PORT"] = ports[
                    "decode" if role == "prefill" else "prefill"
                ]
            manager = MagaServerManager(
                env_args=env,
                port=ports[role],
                device_ids=spec["devices"],
                role_name=f"qwen4_sm120_{role}",
                smoke_args_str=spec["args"],
            )
            cls.servers[role] = manager
            cls.addClassCleanup(manager.stop_server)
        cls.server = cls.servers["prefill" if cls.pd_separation else "main"]
        cls.graph_server = cls.servers["decode"] if cls.pd_separation else cls.server
        with ThreadPoolExecutor(max_workers=len(specs)) as pool:
            futures = {
                role: pool.submit(
                    manager.start_server,
                    model_path=cls.model_path,
                    model_type="qwen4_exp",
                    timeout=1200,
                )
                for role, manager in cls.servers.items()
            }
            for role, future in futures.items():
                if not future.result():
                    cls.dump_startup_failure(cls.servers[role])
                    raise RuntimeError(f"Qwen4 {role} server did not become healthy")
        if cls.pd_separation:
            from smoke.multi_inst_case_runner import _wait_tcp_port_ready

            from rtp_llm.config.py_config_modules import MIN_WORKER_INFO_PORT_NUM

            for role, port in ports.items():
                stride = int(
                    server_option(
                        specs[role]["args"],
                        "--worker_info_port_num",
                        str(MIN_WORKER_INFO_PORT_NUM),
                    )
                )
                for rank in range(2):
                    _wait_tcp_port_ready(
                        "127.0.0.1",
                        int(port) + rank * stride + 1,
                        120,
                        f"{role} rank {rank} model RPC",
                    )

    @classmethod
    def dump_startup_failure(cls, server):
        """Keep the live server ranks' stacks before the outer watchdog kills them."""
        import shutil
        import subprocess

        import psutil
        from smoke.gpu_diagnostics import dump_gpu_state

        dump_gpu_state(
            failure_context="Qwen4 server did not become healthy",
            log_path=str(cls.output_dir / "gpu_state_server_failed.log"),
            server_pid=server.server_pid,
        )
        debugger = shutil.which("gdb")
        pid = server.server_pid
        if debugger is None or pid is None:
            return
        try:
            children = psutil.Process(pid).children(recursive=True)
            log = Path(server.log_file_path).read_text(errors="replace")
            ranks = set(re.findall(r"\[(\d+):\d+\] \[RANK [01]\]", log))
            for process in children:
                if str(process.pid) not in ranks or not process.is_running():
                    continue
                with (cls.output_dir / f"startup_rank_{process.pid}.stack").open(
                    "w"
                ) as output:
                    subprocess.run(
                        [
                            debugger,
                            "-batch",
                            "-nx",
                            "-ex",
                            "set pagination off",
                            "-ex",
                            f"attach {process.pid}",
                            "-ex",
                            "thread apply all bt 20",
                            "-ex",
                            "py-bt",
                            "-ex",
                            "detach",
                        ],
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        timeout=45,
                        check=False,
                    )
        except (OSError, psutil.Error, subprocess.TimeoutExpired):
            logging.exception("Could not collect live Qwen4 startup stacks")

    @classmethod
    def write_coverage(cls):
        (cls.output_dir / "qwen4_sm120_coverage.json").write_text(
            json.dumps(
                {
                    "profile": {
                        "mtp": cls.mtp_enabled,
                        "graph": cls.graph_enabled,
                        "pd_separation": cls.pd_separation,
                        "fused_gr": cls.gr_enabled,
                    },
                    "requests": cls.observed,
                    "target_graph_batches": cls.graph_batches,
                },
                indent=2,
            )
        )

    def reuse_len(self, result):
        return historical_reuse(result["aux_info"], self.pd_separation)

    def url(self, endpoint):
        return f"http://127.0.0.1:{self.server.port}{endpoint}"

    def post(self, label, endpoint, query):
        label = self._testMethodName + "_" + label
        query = routed_query(query, endpoint, self.pd_separation)
        response = self.session.post(self.url(endpoint), json=query, timeout=600)
        (self.output_dir / f"{label}.http.txt").write_text(response.text)
        response.raise_for_status()
        actual = response.json()
        (self.output_dir / f"{label}.json").write_text(
            json.dumps({"query": query, "actual": actual}, ensure_ascii=False, indent=2)
        )
        return actual

    def check_chat_goldens(self, phase):
        from smoke.openai_comparer import OpenaiComparer

        for index, item in enumerate(self.task_info["query_result"]):
            with self.subTest(phase=phase, golden=index):
                comparer = OpenaiComparer(None, item["endpoint"], item, None, False)
                actual = comparer.format_result(
                    self.post(
                        f"{phase}_golden_{index}", item["endpoint"], item["query"]
                    )
                )
                comparer.compare_result(comparer.format_result(item["result"]), actual)

    def request(self, label, prompt, reuse, output_tokens=OUTPUT_TOKENS):
        expected_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        query = {
            "prompt": prompt,
            "generate_config": {
                "max_new_tokens": output_tokens,
                "ignore_eos": True,
                "top_k": 1,
                "top_p": 1.0,
                "random_seed": 42,
                "reuse_cache": reuse,
                "return_input_ids": True,
                "return_output_ids": True,
                "aux_info": True,
            },
        }
        actual = self.post(label, "/", query)
        validate_generation(
            actual,
            expected_ids,
            output_tokens,
            self.text_config["vocab_size"],
            self.mtp_enabled and output_tokens > 1,
        )
        if output_tokens > 1:
            # A high-confidence task with an answer stored before the long
            # prefix keeps FP8 prefill/decode rounding from turning this cache
            # check into a comparison of unconstrained creative completions.
            words = actual["response"].split()
            self.assertGreaterEqual(len(words), output_tokens // 4)
            self.assertEqual(set(words), {"MAP"})
        if self.pd_separation and output_tokens > 1:
            self.assertTrue(
                actual["aux_info"]["pd_sep"], "Request bypassed P/D routing"
            )
        if not reuse:
            self.assertEqual(self.reuse_len(actual), 0)
        self.observed.append(
            {
                "label": label,
                "input_len": len(expected_ids),
                "aux_info": actual["aux_info"],
            }
        )
        return actual

    def assert_same_tokens(self, reference, actual):
        self.assertEqual(reference["input_ids"], actual["input_ids"])
        self.assertEqual(reference["output_ids"], actual["output_ids"])
        self.assertEqual(reference["response"], actual["response"])

    def check_fresh_prefill_reference(self, generated):
        input_ids = generated["input_ids"][0]
        output_ids = generated["output_ids"][0]
        # Check initial MTP decisions and both sides of the next page boundary.
        for step in (0, 1, 2, PAGE_SIZE, PAGE_SIZE + 1):
            prefix = input_ids + output_ids[:step]
            prompt = self.tokenizer.decode(prefix, skip_special_tokens=False)
            self.assertEqual(
                self.tokenizer.encode(prompt, add_special_tokens=False), prefix
            )
            fresh = self.request(
                f"fresh_prefill_{step}", prompt, False, output_tokens=1
            )
            self.assertEqual(fresh["output_ids"][0], [output_ids[step]])

    def long_prompts(self):
        budget = int(self.text_config["indexer_budget"])
        ratio = int(self.text_config["indexer_compress_ratio"])

        def render_prompt(text):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

        first, second = build_prompt_pair(
            self.tokenizer, budget, ratio, render_prompt=render_prompt
        )
        first_ids = self.tokenizer.encode(first, add_special_tokens=False)
        second_ids = self.tokenizer.encode(second, add_special_tokens=False)
        common = next(
            (
                i
                for i, (left, right) in enumerate(zip(first_ids, second_ids))
                if left != right
            ),
            min(len(first_ids), len(second_ids)),
        )
        self.assertGreaterEqual(common, 2 * PAGE_SIZE)
        self.assertGreater(len(first_ids), budget)
        self.assertNotEqual(len(first_ids) % ratio, 0)
        self.assertLess(max(len(first_ids), len(second_ids)) + OUTPUT_TOKENS, 32768)

        return first, second

    def uncached_baseline(self):
        if type(self).baseline is None:
            first, _ = self.long_prompts()
            type(self).baseline = self.request("long_uncached", first, False)
        return type(self).baseline

    def test_01_basic_generation(self):
        self.check_chat_goldens("before_state_checks")

    def test_02_long_context_qsa_tail(self):
        self.check_fresh_prefill_reference(self.uncached_baseline())

    def test_03_prefix_cache_restore(self):
        first, second = self.long_prompts()
        baseline = self.uncached_baseline()
        # GDN decode retains only its active tail. Publish prompt checkpoints
        # before a long generation advances beyond those state pages.
        fill = self.request("long_cache_fill", first, True, output_tokens=1)
        self.assertEqual(fill["output_ids"][0], baseline["output_ids"][0][:1])
        warm = self.request("long_cache_hit", first, True)
        self.assertGreaterEqual(self.reuse_len(warm), PAGE_SIZE)
        self.assertEqual(self.reuse_len(warm) % PAGE_SIZE, 0)
        self.assert_same_tokens(baseline, warm)

        other = self.request("changed_suffix_uncached", second, False)
        fill = self.request("changed_suffix_cache_fill", second, True, output_tokens=1)
        self.assertEqual(fill["output_ids"][0], other["output_ids"][0][:1])
        reused = self.request("changed_suffix_cache_hit", second, True)
        self.assertGreaterEqual(self.reuse_len(reused), PAGE_SIZE)
        self.assert_same_tokens(other, reused)
        self.assert_same_tokens(
            baseline, self.request("long_after_changed_suffix", first, True)
        )
        self.assert_same_tokens(
            baseline, self.request("populated_cache_bypass", first, False)
        )

    def test_04_speculative_execution_mode(self):
        actual = self.uncached_baseline()
        validate_generation(
            actual,
            actual["input_ids"][0],
            OUTPUT_TOKENS,
            self.text_config["vocab_size"],
            self.mtp_enabled,
        )

    def set_log_level(self, level):
        response = self.session.post(
            f"http://127.0.0.1:{self.graph_server.port}/set_log_level",
            json={"log_level": level},
            timeout=10,
        )
        response.raise_for_status()

    def test_05_graph_execution_mode(self):
        # Observe this request only: warmup and earlier scenarios cannot satisfy it.
        offset = Path(self.graph_server.log_file_path).stat().st_size
        first, _ = self.long_prompts()
        # Only the replay observation needs DEBUG; all correctness and timing
        # scenarios run at INFO without the debug kernel synchronization.
        self.set_log_level("DEBUG")
        try:
            self.request("graph_probe", first, False)
        finally:
            self.set_log_level("INFO")
        with Path(self.graph_server.log_file_path).open("rb") as reader:
            reader.seek(offset)
            runtime = reader.read().decode(errors="replace")
        replay = re.findall(
            r"\[PyWrappedModel\] using CUDA graph forward, is_target_verify=1, is_prefill=[01], graph_bs=(\d+)",
            runtime,
        )
        type(self).graph_batches.extend(replay)
        if self.graph_enabled and self.mtp_enabled:
            self.assertIn(
                "1", replay, "No real MTP target CUDA graph replay after startup"
            )
        else:
            self.assertFalse(replay, "Eager target replayed an MTP CUDA graph")

        if self.pd_separation:
            decode_log = Path(self.graph_server.log_file_path).read_text(
                errors="replace"
            )
            validate_gr_coverage(decode_log, self.gr_enabled, self.gr_mtp_enabled)
            if self.gr_enabled:
                rank_pids = dict(
                    re.findall(
                        r"\[process-(\d+)\].*Qwen4 fused GR enabled: rank=([01])",
                        decode_log,
                    )
                )
                self.assertEqual(set(rank_pids.values()), {"0", "1"})
                for pid in rank_pids:
                    rows = {
                        int(n)
                        for n in re.findall(
                            rf"\[process-{pid}\].*SM120 GR workspace prepared: rows=(\d+)",
                            decode_log,
                        )
                    }
                    self.assertIn(
                        4 if self.mtp_enabled else 1, rows, "No target GR workspace"
                    )
                    if self.gr_mtp_enabled:
                        self.assertIn(1, rows, "No draft GR workspace")
            prefill_log = Path(self.server.log_file_path).read_text(errors="replace")
            self.assertNotIn("Qwen4 fused GR enabled:", prefill_log)

    @staticmethod
    def image_part(color):
        import base64
        from io import BytesIO

        from PIL import Image

        buffer = BytesIO()
        Image.new("RGB", (384, 256), color).save(buffer, format="PNG")
        return {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64,"
                + base64.b64encode(buffer.getvalue()).decode()
            },
        }

    def media_request(
        self, label, parts, question, *, reuse=False, max_tokens=64, ignore_eos=False
    ):
        return self.post(
            label,
            "/v1/chat/completions",
            {
                "model": "qwen4_exp",
                "messages": [
                    {
                        "role": "user",
                        "content": [*parts, {"type": "text", "text": question}],
                    }
                ],
                "enable_thinking": False,
                "temperature": 0,
                "max_tokens": max_tokens,
                "extra_configs": {"reuse_cache": reuse, "ignore_eos": ignore_eos},
            },
        )

    @staticmethod
    def answer(actual):
        return actual["choices"][0]["message"]["content"].strip().upper()

    def test_06_single_image_and_text_regression(self):
        for color in ("red", "blue"):
            actual = self.media_request(
                color,
                [self.image_part(color)],
                "What is the color of this image? Answer with one English color word only.",
            )
            self.assertEqual(self.answer(actual).strip(". "), color.upper())
        self.check_chat_goldens("after_images")

    def test_07_multi_image_order_and_prefix_restore(self):
        question = "Give the colors of the first and second images, in that order. Answer only: COLOR, COLOR."
        parts = [self.image_part("red"), self.image_part("blue")]
        reference = self.media_request("two_images", parts, question)
        self.assertEqual(
            re.findall(r"RED|BLUE", self.answer(reference)), ["RED", "BLUE"]
        )
        self.media_request("cache_fill", parts, question, reuse=True, max_tokens=1)
        reused = self.media_request("cache_hit", parts, question, reuse=True)
        self.assertEqual(self.answer(reference), self.answer(reused))
        self.assertGreaterEqual(self.reuse_len(reused), PAGE_SIZE)
        reversed_result = self.media_request(
            "reverse_images", parts[::-1], question, reuse=True
        )
        self.assertEqual(
            re.findall(r"RED|BLUE", self.answer(reversed_result)), ["BLUE", "RED"]
        )

    def test_08_video_temporal_order(self):
        import av
        import numpy as np

        path = self.output_dir / "red_then_blue.mp4"
        with av.open(str(path), "w") as container:
            stream = container.add_stream("libx264", rate=4)
            stream.width = stream.height = 256
            stream.pix_fmt = "yuv420p"
            for color in ([255, 0, 0], [0, 0, 255]):
                pixels = np.full((256, 256, 3), color, dtype=np.uint8)
                for _ in range(8):
                    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        actual = self.media_request(
            "video",
            [{"type": "video_url", "video_url": {"url": str(path)}}],
            "Which color appears first and which appears last? Answer only: COLOR, COLOR.",
        )
        self.assertEqual(re.findall(r"RED|BLUE", self.answer(actual)), ["RED", "BLUE"])

    def test_09_visual_context_exceeds_qsa_budget(self):
        line = "The archive records books, maps, letters, and journeys.\n"
        text = line
        budget = int(self.text_config["indexer_budget"])
        while (
            len(self.tokenizer.encode(text, add_special_tokens=False))
            <= budget + PAGE_SIZE
        ):
            text += line
        text += (
            "\nRepeat the single English word for the image background color 200 times, "
            "separated by spaces. Output no explanation or other text."
        )
        parts = [self.image_part("red")]
        actual = self.media_request(
            "visual_long", parts, text, max_tokens=OUTPUT_TOKENS, ignore_eos=True
        )
        self.assertEqual(set(self.answer(actual).split()), {"RED"})
        self.assertGreater(actual["aux_info"]["input_len"], budget)
        self.assertEqual(actual["aux_info"]["output_len"], OUTPUT_TOKENS)
        if self.mtp_enabled:
            self.assertGreater(
                sum(actual["aux_info"]["speculative_accepted_tokens_per_pos"]), 0
            )
        self.media_request("visual_long_fill", parts, text, reuse=True, max_tokens=1)
        reused = self.media_request(
            "visual_long_hit",
            parts,
            text,
            reuse=True,
            max_tokens=OUTPUT_TOKENS,
            ignore_eos=True,
        )
        self.assertGreaterEqual(self.reuse_len(reused), PAGE_SIZE)
        self.assertEqual(self.answer(reused), self.answer(actual))

    def test_10_concurrent_text_image_and_repeated_new_media(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier

        red = self.image_part("red")
        question = (
            "What is the color of this image? Answer with one English color word only."
        )
        # Make the image ready before overlapping its request with long text.
        self.media_request("warm_red", [red], question)
        reference = self.uncached_baseline()
        prompt, _ = self.long_prompts()
        barrier = Barrier(2)

        def text_request():
            barrier.wait(timeout=30)
            return self.request("concurrent_text", prompt, False)

        def image_request():
            barrier.wait(timeout=30)
            return self.media_request("concurrent_image", [red], question)

        with ThreadPoolExecutor(max_workers=2) as pool:
            text = pool.submit(text_request)
            image = pool.submit(image_request)
            self.assert_same_tokens(reference, text.result(timeout=600))
            self.assertEqual(self.answer(image.result(timeout=600)).strip(". "), "RED")
        actual = self.media_request(
            "repeated_red_new_green",
            [red, self.image_part("green")],
            "Give the colors of the first and second images, in that order. Answer only: COLOR, COLOR.",
        )
        self.assertEqual(
            re.findall(r"RED|GREEN", self.answer(actual)), ["RED", "GREEN"]
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()

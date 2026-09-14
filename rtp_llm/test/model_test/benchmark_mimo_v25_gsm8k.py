"""MiMo V2.5 GSM8K benchmark matching the official SGLang cookbook recipe.

Protocol: OpenAI GSM8K test split, fixed 5-shot prompt, 200 evaluated examples,
temperature 0, max_tokens 4096, and last-number exact matching.
"""

import ast
import concurrent.futures
import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import requests

from rtp_llm.test.utils.maga_server_manager import MagaServerManager

GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)
GSM8K_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
REQUIRED_SHOTS = 5
INVALID_ANSWER = -9999999

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,
)


def _checkpoint_path() -> str:
    path = os.environ.get("CHECKPOINT_PATH", "")
    if not os.path.isfile(os.path.join(path, "config.json")):
        raise FileNotFoundError(
            "Set CHECKPOINT_PATH to a MiMo V2.5 checkpoint containing config.json "
            f"(got {path!r})"
        )
    return path


def _download_dataset(path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    logging.info("Downloading GSM8K test split from %s", GSM8K_URL)
    urllib.request.urlretrieve(GSM8K_URL, tmp_path)
    tmp_path.replace(path)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _validate_dataset(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != GSM8K_SHA256:
        raise ValueError(
            f"Unexpected GSM8K test split checksum for {path}: {digest}; "
            f"expected {GSM8K_SHA256}"
        )
    return digest


def _one_example(example: dict, include_answer: bool) -> str:
    text = f"Question: {example['question']}\nAnswer:"
    if include_answer:
        text += f" {example['answer']}"
    return text


def _few_shot_prompt(examples: list[dict]) -> str:
    return "".join(
        _one_example(example, include_answer=True) + "\n\n"
        for example in examples[:REQUIRED_SHOTS]
    )


def _answer_value(answer: str):
    numbers = re.findall(r"-?\d+\.?\d*", answer.replace(",", ""))
    if not numbers:
        return INVALID_ANSWER
    try:
        return ast.literal_eval(numbers[-1])
    except (SyntaxError, ValueError):
        return INVALID_ANSWER


class MiMoV25GSM8KBenchmark:
    def __init__(self):
        self.checkpoint = _checkpoint_path()
        self.num_examples = int(os.environ.get("GSM8K_NUM_EXAMPLES", "200"))
        self.num_threads = int(os.environ.get("GSM8K_NUM_THREADS", "8"))
        self.max_tokens = int(os.environ.get("GSM8K_MAX_TOKENS", "4096"))
        if self.num_examples <= 0 or self.num_threads <= 0 or self.max_tokens <= 0:
            raise ValueError(
                "GSM8K_NUM_EXAMPLES, GSM8K_NUM_THREADS, and GSM8K_MAX_TOKENS "
                "must all be positive"
            )
        self.dataset_path = Path(
            os.environ.get(
                "GSM8K_DATA_PATH", "/home/renkun.ren/dataset/gsm8k_test.jsonl"
            )
        )
        self.log_dir = Path(
            os.environ.get("MIMO_BENCHMARK_LOG_DIR", "/home/renkun.ren/log/mimov25")
        )
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_path = self.log_dir / f"gsm8k_results_{self.timestamp}.jsonl"
        self.summary_path = self.log_dir / f"gsm8k_summary_{self.timestamp}.json"
        self.server = None
        self._progress_lock = threading.Lock()
        self._completed = 0

    def _start_server(self) -> None:
        tp_size = int(os.environ.get("TP_SIZE", "4"))
        reuse_cache = os.environ.get("REUSE_CACHE", "1")
        max_seq_len = int(os.environ.get("GSM8K_MAX_SEQ_LEN", "8192"))
        smoke_args = (
            f"--model_type mimo_v25 "
            f"--checkpoint_path {self.checkpoint} "
            f"--tokenizer_path {self.checkpoint} "
            f"--tp_size {tp_size} "
            f"--world_size {tp_size} "
            f"--max_seq_len {max_seq_len} "
            f"--concurrency_limit {self.num_threads} "
            f"--reuse_cache {reuse_cache}"
        )
        self.server = MagaServerManager(
            process_file_name=f"mimo_v25_gsm8k_server_{self.timestamp}.log",
            smoke_args_str=smoke_args,
        )
        logging.info(
            "Starting MiMo V2.5 server on GPUs %s",
            os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        if not self.server.start_server(timeout=1600):
            raise RuntimeError("MiMo V2.5 server failed to start")

    def _evaluate_one(self, index: int, example: dict, prefix: str) -> dict:
        prompt = prefix + _one_example(example, include_answer=False)
        query = {
            "model": "mimo_v25",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
        }
        started = time.perf_counter()
        error = None
        response = {}
        for attempt in range(1, 4):
            try:
                raw = requests.post(
                    f"http://127.0.0.1:{self.server.port}/v1/chat/completions",
                    json=query,
                    timeout=900,
                )
                raw.raise_for_status()
                response = raw.json()
                break
            except Exception as exc:
                error = f"attempt {attempt}: {type(exc).__name__}: {exc}"
                if attempt < 3:
                    time.sleep(1)

        choices = response.get("choices", [])
        if not choices and error is None:
            error = f"response does not contain choices: {response}"
        message = choices[0].get("message", {}) if choices else {}
        answer_text = message.get("content") or ""
        reasoning_text = message.get("reasoning_content") or ""
        scored_text = answer_text or reasoning_text
        expected = _answer_value(example["answer"])
        extracted = _answer_value(scored_text)
        usage = response.get("usage", {})
        result = {
            "index": index,
            "question": example["question"],
            "expected": expected,
            "extracted": extracted,
            "correct": extracted == expected,
            "latency_seconds": time.perf_counter() - started,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "answer": answer_text,
            "reasoning": reasoning_text,
            "finish_reason": choices[0].get("finish_reason") if choices else None,
            "error": error if not choices else None,
        }
        with self._progress_lock:
            self._completed += 1
            logging.info(
                "GSM8K progress %d/%d, item=%d, correct=%s",
                self._completed,
                self.num_examples,
                index,
                result["correct"],
            )
        return result

    def run(self) -> dict:
        _download_dataset(self.dataset_path)
        dataset_sha256 = _validate_dataset(self.dataset_path)
        all_examples = _read_jsonl(self.dataset_path)
        if len(all_examples) < REQUIRED_SHOTS + self.num_examples:
            raise ValueError(
                f"GSM8K has {len(all_examples)} examples; need at least "
                f"{REQUIRED_SHOTS + self.num_examples}"
            )
        prefix = _few_shot_prompt(all_examples)
        examples = all_examples[REQUIRED_SHOTS : REQUIRED_SHOTS + self.num_examples]
        self.log_dir.mkdir(parents=True, exist_ok=True)

        results = []
        server_started = time.perf_counter()
        try:
            self._start_server()
            server_start_seconds = time.perf_counter() - server_started
            evaluation_started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_threads
            ) as executor:
                futures = {
                    executor.submit(self._evaluate_one, i, example, prefix): i
                    for i, example in enumerate(examples)
                }
                with self.results_path.open("w", encoding="utf-8") as stream:
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        results.append(result)
                        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                        stream.flush()
        finally:
            server_log = None
            if self.server is not None:
                if self.server.log_file_path:
                    server_log = Path(self.server.log_file_path)
                self.server.stop_server()
            if server_log and server_log.is_file():
                shutil.copy2(
                    server_log,
                    self.log_dir / f"gsm8k_server_{self.timestamp}.log",
                )
        correct = sum(result["correct"] for result in results)
        failed_requests = sum(result["error"] is not None for result in results)
        completion_tokens = sum(result["completion_tokens"] or 0 for result in results)
        duration = time.perf_counter() - evaluation_started
        summary = {
            "dataset": "openai/grade-school-math test split",
            "dataset_path": str(self.dataset_path),
            "dataset_sha256": dataset_sha256,
            "protocol": "5-shot, last-number exact match",
            "num_examples": len(results),
            "correct": correct,
            "score": correct / len(results),
            "failed_requests": failed_requests,
            "server_start_seconds": server_start_seconds,
            "duration_seconds": duration,
            "output_throughput_tokens_per_second": completion_tokens / duration,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "num_threads": self.num_threads,
            "results_path": str(self.results_path),
        }
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logging.info("GSM8K summary: %s", json.dumps(summary, ensure_ascii=False))
        return summary


def main() -> None:
    summary = MiMoV25GSM8KBenchmark().run()
    if summary["failed_requests"]:
        raise RuntimeError(f"{summary['failed_requests']} GSM8K requests failed")


if __name__ == "__main__":
    main()

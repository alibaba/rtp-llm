"""H20 single-replica A/B acceptance test for single-request GPU OOM isolation.

Proves on a physical H20 GPU (TP=1) with Qwen3-VL that:
1. Baseline (RTP_LLM_DISABLE_STEP_OOM_ISOLATION=1): a single oversized
   long-prompt logits request (prompt_logprobs / calculate_loss) kills the
   live Qwen3-VL backend replica.
2. With fix enabled: after the identical single request triggers a real GPU OOM
   in NormalExecutor::process and is isolated cleanly, the exact same backend
   worker PID stays alive and serves subsequent multimodal and text requests
   with HTTP 200.
"""

import base64
import io
import json
import logging
import os
import subprocess
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
import requests
from PIL import Image
from transformers import AutoTokenizer

from rtp_llm.test.utils.maga_server_manager import MagaServerManager

CANDIDATE_MODELS: List[Tuple[str, str]] = [
    ("/mnt/nas1/hf/Qwen3-VL-32B-Instruct", "qwen3_vl"),
    ("/mnt/nas1/hf/Qwen3-VL-8B-Instruct", "qwen3_vl"),
    ("/mnt/nas1/hf/Qwen3-VL-4B-Instruct", "qwen3_vl"),
]

COMMON_SMOKE_ARGS = (
    "--act_type BF16 "
    "--use_local 1 "
    "--use_local_preprocess 1 "
    "--tp_size 1 "
    "--dp_size 1 "
    "--pp_size 1 "
    "--world_size 1 "
    "--max_seq_len 32768 "
    "--prefill_chunk_size 0 "
    "--reserver_runtime_mem_mb 3072 "
    "--enable_cuda_graph 0"
)


def _discover_qwen3_vl_model() -> Tuple[str, str, List[str]]:
    nas_root = Path("/mnt/nas1/hf")
    available_entries: List[str] = []
    if nas_root.is_dir():
        available_entries = sorted(
            entry.name
            for entry in nas_root.iterdir()
            if "qwen" in entry.name.lower() or "vl" in entry.name.lower()
        )
    for model_path, model_type in CANDIDATE_MODELS:
        cfg = Path(model_path) / "config.json"
        if cfg.is_file():
            return model_path, model_type, available_entries
    raise FileNotFoundError(
        f"No Qwen3-VL checkpoint found under /mnt/nas1/hf; saw entries: {available_entries}"
    )


def _read_vocab_size(model_path: str) -> int:
    cfg_path = Path(model_path) / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if isinstance(cfg.get("text_config"), dict) and "vocab_size" in cfg["text_config"]:
        return int(cfg["text_config"]["vocab_size"])
    return int(cfg.get("vocab_size", 151936))


def _query_gpu_memory_mib() -> Dict[str, Any]:
    """Query GPU name/total/free memory for the visible H20 device without opening CUDA."""
    visible = [
        item.strip()
        for item in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
        if item.strip()
    ]
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=10)
    except Exception as exc:
        logging.warning("nvidia-smi memory query failed: %s", exc)
        return {"gpu_name": "unknown", "total_mib": 96000, "free_mib": 5000}

    rows: List[Tuple[str, str, int, int]] = []
    for raw_line in out.splitlines():
        parts = [p.strip() for p in raw_line.split(",")]
        if len(parts) == 4:
            rows.append((parts[0], parts[1], int(parts[2]), int(parts[3])))
    for row in rows:
        if visible and row[0] == visible[0]:
            return {"gpu_name": row[1], "total_mib": row[2], "free_mib": row[3]}
    if rows:
        return {"gpu_name": rows[0][1], "total_mib": rows[0][2], "free_mib": rows[0][3]}
    return {"gpu_name": "unknown", "total_mib": 96000, "free_mib": 5000}


def _query_compute_app_pids() -> Dict[int, int]:
    """Return {pid: used_gpu_mib} from nvidia-smi compute apps."""
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=10)
    except Exception:
        return {}
    usage: Dict[int, int] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
            mem = int(parts[1]) if parts[1].isdigit() else 0
            usage[pid] = mem
        except ValueError:
            continue
    return usage


def _get_backend_worker_pid(server: MagaServerManager) -> Optional[int]:
    """Resolve the live backend worker child PID for this server instance."""
    parent_pid = server.server_pid
    if parent_pid is None or not psutil.pid_exists(parent_pid):
        return None
    try:
        parent = psutil.Process(parent_pid)
        children = [
            proc
            for proc in parent.children(recursive=True)
            if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        ]
    except psutil.Error:
        return None

    if not children:
        return None

    gpu_usage = _query_compute_app_pids()
    gpu_children = [(gpu_usage[c.pid], c.pid) for c in children if c.pid in gpu_usage]
    if gpu_children:
        gpu_children.sort(reverse=True)
        return gpu_children[0][1]

    named_backend = [
        c
        for c in children
        if "backend" in (c.name() or "").lower()
        or any("backend" in arg.lower() for arg in _safe_cmdline(c))
    ]
    if named_backend:
        named_backend.sort(key=lambda proc: _safe_rss(proc), reverse=True)
        return named_backend[0].pid

    children.sort(key=lambda proc: _safe_rss(proc), reverse=True)
    return children[0].pid


def _safe_cmdline(proc: psutil.Process) -> List[str]:
    try:
        return proc.cmdline()
    except psutil.Error:
        return []


def _safe_rss(proc: psutil.Process) -> int:
    try:
        return proc.memory_info().rss
    except psutil.Error:
        return 0


def _make_jpeg_data_url(width: int, height: int, seed_offset: int = 0) -> str:
    """Create a deterministic patterned RGB image encoded as JPEG base64 data URI."""
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (
                (x * 3 + seed_offset * 37) % 256,
                (y * 5 + seed_offset * 73) % 256,
                ((x ^ y) + seed_offset * 19) % 256,
            )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _build_token_sized_prompt(
    tokenizer: Any, target_tokens: int, seed_label: str
) -> Tuple[str, int]:
    """Construct an offline text prompt with ~target_tokens BPE tokens."""
    unit = " a"
    unit_ids = tokenizer.encode(unit, add_special_tokens=False)
    repeat = max(1, target_tokens // max(1, len(unit_ids)))
    text = f"[{seed_label}] Summarize and validate:" + (unit * repeat)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) > 28000:
        token_ids = token_ids[:28000]
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    return text, len(token_ids)


class SingleRequestGpuOomH20Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_path, cls.model_type, cls.nas_entries = _discover_qwen3_vl_model()
        cls.vocab_size = _read_vocab_size(cls.model_path)
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_path, local_files_only=True
        )
        cls.output_dir = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp"))
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(
            "Selected Qwen3-VL checkpoint=%s vocab_size=%d nas_entries=%s",
            cls.model_path,
            cls.vocab_size,
            cls.nas_entries,
        )

    def _start_qwen3_vl_server(
        self, role_name: str, extra_env: Dict[str, str]
    ) -> MagaServerManager:
        env_args = {
            "FT_SERVER_TEST": "1",
            "LOG_LEVEL": "INFO",
            **extra_env,
        }
        server = MagaServerManager(
            env_args=env_args,
            role_name=role_name,
            smoke_args_str=COMMON_SMOKE_ARGS,
        )
        self.addCleanup(server.stop_server)
        started = server.start_server(
            model_type=self.model_type,
            model_path=self.model_path,
            tokenizer_path=self.model_path,
            timeout=1200,
        )
        self.assertTrue(started, f"Failed to start {role_name} for {self.model_path}")
        return server

    def _post_chat_completion(
        self,
        port: int,
        image_url: Optional[str],
        text: str,
        max_tokens: int = 16,
        prompt_logprobs: Optional[int] = None,
        calculate_loss: Optional[int] = None,
        timeout: int = 180,
    ) -> Tuple[Optional[int], str]:
        content: Any
        if image_url is not None:
            content = [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": text},
            ]
        else:
            content = text

        extra_configs: Dict[str, Any] = {"add_vision_id": False}
        if calculate_loss is not None:
            extra_configs["calculate_loss"] = calculate_loss

        payload: Dict[str, Any] = {
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 0.01,
            "stream": False,
            "extra_configs": extra_configs,
        }
        if prompt_logprobs is not None:
            payload["prompt_logprobs"] = prompt_logprobs

        session = requests.Session()
        session.trust_env = False
        try:
            resp = session.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json=payload,
                timeout=timeout,
            )
            return resp.status_code, resp.text
        except requests.RequestException as exc:
            return None, repr(exc)
        finally:
            session.close()

    def _compute_oom_token_targets(self, free_mib: int) -> List[int]:
        fp32_logits_mib_per_tok = (self.vocab_size * 4) / (1024.0 * 1024.0)
        estimated = int((max(free_mib, 2048) * 2.4) / fp32_logits_mib_per_tok)
        first_target = min(24000, max(12000, estimated))
        targets = [first_target]
        if first_target < 28000:
            targets.append(28000)
        return targets

    def _wait_health(self, port: int, expect_healthy: bool, timeout_s: int = 20) -> bool:
        deadline = time.monotonic() + timeout_s
        session = requests.Session()
        session.trust_env = False
        try:
            while time.monotonic() < deadline:
                try:
                    resp = session.get(f"http://127.0.0.1:{port}/health", timeout=3)
                    healthy = resp.status_code == 200
                except requests.RequestException:
                    healthy = False
                if healthy == expect_healthy:
                    return True
                time.sleep(0.5)
            return False
        finally:
            session.close()

    def _read_log_delta(self, log_path: str, offset: int) -> str:
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            fh.seek(offset)
            return fh.read()

    def _verify_baseline_crashes_on_single_oom(
        self,
        oom_specs: List[Tuple[int, Optional[int], Optional[int]]],
    ) -> Dict[str, Any]:
        server = self._start_qwen3_vl_server(
            role_name="qwen3_vl_baseline_no_isolation",
            extra_env={"RTP_LLM_DISABLE_STEP_OOM_ISOLATION": "1"},
        )
        backend_pid_before = _get_backend_worker_pid(server)
        self.assertIsNotNone(backend_pid_before, "Baseline backend worker PID missing")
        log_offset = Path(server.log_file_path).stat().st_size

        normal_img = _make_jpeg_data_url(160, 160, seed_offset=1)
        status_pre, body_pre = self._post_chat_completion(
            server.port,
            normal_img,
            "Describe the colors in this image concisely.",
            max_tokens=16,
        )
        self.assertEqual(status_pre, 200, f"Baseline healthy req failed: {body_pre}")

        oom_attempts: List[Dict[str, Any]] = []
        crashed = False
        for idx, (tok_target, prompt_lp, calc_loss) in enumerate(oom_specs):
            prompt_text, actual_tokens = _build_token_sized_prompt(
                self.tokenizer, tok_target, f"baseline_oom_{idx}"
            )
            status, body = self._post_chat_completion(
                server.port,
                None,
                prompt_text,
                max_tokens=1,
                prompt_logprobs=prompt_lp,
                calculate_loss=calc_loss,
                timeout=90,
            )
            crashed = self._wait_backend_exit(server, backend_pid_before, timeout_s=20)
            if not crashed and status != 200:
                crashed = self._wait_health(server.port, expect_healthy=False, timeout_s=10)
            oom_attempts.append(
                {
                    "target_tokens": tok_target,
                    "prompt_logprobs": prompt_lp,
                    "calculate_loss": calc_loss,
                    "text_tokens": actual_tokens,
                    "http_status": status,
                    "body_preview": body[:300],
                    "backend_exited_or_unhealthy": crashed,
                }
            )
            if crashed or status != 200:
                break

        log_delta = self._read_log_delta(server.log_file_path, log_offset)
        self.assertNotIn(
            "[normal-executor-oom] isolated GPU OOM",
            log_delta,
            "Isolation should be disabled in Scenario A",
        )
        self.assertTrue(
            crashed,
            f"Baseline Scenario A expected backend crash/unhealthy after single-req OOM: {oom_attempts}",
        )
        server.stop_server()
        return {
            "backend_pid_before": backend_pid_before,
            "pre_oom_status": status_pre,
            "oom_attempts": oom_attempts,
            "crashed_or_unhealthy": crashed,
        }

    def _wait_backend_exit(
        self, server: MagaServerManager, backend_pid: int, timeout_s: int = 20
    ) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not psutil.pid_exists(backend_pid):
                return True
            try:
                proc = psutil.Process(backend_pid)
                if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
                    return True
            except psutil.Error:
                return True
            parent_pid = server.server_pid
            if parent_pid is None or not psutil.pid_exists(parent_pid):
                return True
            time.sleep(0.5)
        return False

    def _verify_fixed_isolates_oom_and_keeps_serving(self) -> Dict[str, Any]:
        server = self._start_qwen3_vl_server(
            role_name="qwen3_vl_oom_isolated",
            extra_env={},
        )
        backend_pid_before = _get_backend_worker_pid(server)
        self.assertIsNotNone(backend_pid_before, "Fixed server backend PID missing")
        gpu_mem = _query_gpu_memory_mib()
        oom_token_targets = self._compute_oom_token_targets(gpu_mem["free_mib"])
        oom_specs: List[Tuple[int, Optional[int], Optional[int]]] = [
            (t, 1, None) for t in oom_token_targets
        ] + [(28000, None, 1)]
        log_offset = Path(server.log_file_path).stat().st_size

        # 1) Pre-OOM normal multimodal request
        normal_before_img = _make_jpeg_data_url(160, 160, seed_offset=2)
        status_pre, body_pre = self._post_chat_completion(
            server.port,
            normal_before_img,
            "Summarize this pattern in one short sentence.",
            max_tokens=16,
        )
        self.assertEqual(status_pre, 200, f"Pre-OOM multimodal failed: {body_pre}")
        pre_json = json.loads(body_pre)
        self.assertTrue(pre_json["choices"][0]["message"]["content"])

        # 2) Worst-case single request triggering real GPU OOM in NormalExecutor::process
        oom_attempts: List[Dict[str, Any]] = []
        oom_isolated = False
        winning_spec: Optional[Tuple[int, Optional[int], Optional[int]]] = None
        for idx, (tok_target, prompt_lp, calc_loss) in enumerate(oom_specs):
            prompt_text, actual_tokens = _build_token_sized_prompt(
                self.tokenizer, tok_target, f"fixed_oom_{idx}"
            )
            status, body = self._post_chat_completion(
                server.port,
                None,
                prompt_text,
                max_tokens=1,
                prompt_logprobs=prompt_lp,
                calculate_loss=calc_loss,
                timeout=120,
            )
            log_delta = self._read_log_delta(server.log_file_path, log_offset)
            isolated_hit = "[normal-executor-oom] isolated GPU OOM" in log_delta
            oom_attempts.append(
                {
                    "target_tokens": tok_target,
                    "prompt_logprobs": prompt_lp,
                    "calculate_loss": calc_loss,
                    "text_tokens": actual_tokens,
                    "http_status": status,
                    "isolated_log_seen": isolated_hit,
                    "body_preview": body[:300],
                }
            )
            if status != 200 and isolated_hit:
                oom_isolated = True
                winning_spec = (tok_target, prompt_lp, calc_loss)
                break

        self.assertTrue(
            oom_isolated and winning_spec is not None,
            f"Expected single request to trigger isolated OOM in NormalExecutor: {oom_attempts}",
        )

        # 3) Same replica must stay alive with identical backend PID & healthy endpoint
        self.assertTrue(
            self._wait_health(server.port, expect_healthy=True, timeout_s=20),
            "Fixed server /health did not remain HTTP 200 after isolated OOM",
        )
        backend_pid_after_oom = _get_backend_worker_pid(server)
        self.assertEqual(
            backend_pid_after_oom,
            backend_pid_before,
            "Backend worker PID changed after single-request GPU OOM!",
        )

        # 4) Next normal multimodal and text requests must pass HTTP 200 on the same replica
        normal_after_img = _make_jpeg_data_url(192, 192, seed_offset=3)
        status_post_mm, body_post_mm = self._post_chat_completion(
            server.port,
            normal_after_img,
            "Describe the dominant pattern in this image concisely.",
            max_tokens=24,
        )
        self.assertEqual(
            status_post_mm, 200, f"Post-OOM multimodal req failed: {body_post_mm}"
        )
        post_mm_json = json.loads(body_post_mm)
        self.assertTrue(post_mm_json["choices"][0]["message"]["content"])

        status_post_text, body_post_text = self._post_chat_completion(
            server.port,
            None,
            "Reply with the single word OK.",
            max_tokens=16,
        )
        self.assertEqual(
            status_post_text, 200, f"Post-OOM text req failed: {body_post_text}"
        )
        post_text_json = json.loads(body_post_text)
        self.assertTrue(post_text_json["choices"][0]["message"]["content"])

        backend_pid_after_recovery = _get_backend_worker_pid(server)
        self.assertEqual(
            backend_pid_after_recovery,
            backend_pid_before,
            "Backend worker PID changed while serving post-OOM recovery requests!",
        )
        server.stop_server()
        return {
            "gpu_mem_after_start": gpu_mem,
            "backend_pid_before": backend_pid_before,
            "backend_pid_after_oom": backend_pid_after_oom,
            "backend_pid_after_recovery": backend_pid_after_recovery,
            "same_backend_pid": backend_pid_after_recovery == backend_pid_before,
            "pre_oom_status": status_pre,
            "oom_attempts": oom_attempts,
            "winning_spec": list(winning_spec),
            "post_oom_multimodal_status": status_post_mm,
            "post_oom_text_status": status_post_text,
        }

    def test_single_request_gpu_oom_ab_acceptance_on_h20(self) -> None:
        gpu_mem_init = _query_gpu_memory_mib()

        scenario_b = self._verify_fixed_isolates_oom_and_keeps_serving()
        winning_spec = tuple(scenario_b["winning_spec"])
        scenario_a = self._verify_baseline_crashes_on_single_oom([winning_spec])

        report = {
            "model_path": self.model_path,
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
            "gpu_mem_init": gpu_mem_init,
            "smoke_args": COMMON_SMOKE_ARGS,
            "nas_qwen_vl_entries": self.nas_entries,
            "scenario_b_fixed": scenario_b,
            "scenario_a_baseline": scenario_a,
        }
        report_path = self.output_dir / "acceptance_single_req_oom_h20.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"ACCEPTANCE_SINGLE_REQ_OOM_H20_JSON:{json.dumps(report)}", flush=True)


if __name__ == "__main__":
    unittest.main()

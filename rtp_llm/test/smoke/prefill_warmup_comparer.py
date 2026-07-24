import glob
import logging
import math
import os
import re
import shlex
from typing import Any, Dict, List

from pydantic import BaseModel
from smoke.base_comparer import BaseComparer


class WarmupComparerMixin:
    def _server_env(self, name: str, default: str) -> str:
        if not hasattr(self.server_manager, "env_args"):
            raise AssertionError(
                "server manager must expose env_args for warmup validation"
            )
        server_env = self.server_manager.env_args
        return str(server_env.get(name, os.environ.get(name, default)))

    def _configured_ep_size(self) -> int:
        value = self._server_env("EP_SIZE", "1")
        if not hasattr(self.server_manager, "smoke_args_str"):
            raise AssertionError(
                "server manager must expose smoke_args_str for warmup validation"
            )
        tokens = shlex.split(self.server_manager.smoke_args_str)
        for index, token in enumerate(tokens):
            if token == "--ep_size" and index + 1 < len(tokens):
                value = tokens[index + 1]
            elif token.startswith("--ep_size="):
                value = token.split("=", 1)[1]
        return int(value)

    @staticmethod
    def _evidence_lines(logs: str) -> str:
        markers = (
            "[PREFILL_WARMUP]",
            "[DECODE_WARMUP]",
            "[MOE_WARMUP]",
            "warm up done",
            "[KV_ALLOC]",
        )
        lines = [
            line
            for line in logs.splitlines()
            if any(marker in line for marker in markers)
        ]
        return "\n".join(lines[-20:]) or "<no warmup evidence lines>"

    def _assert_moe_warmup_skew(self, logs: str, expected_ep_size: int) -> None:
        if expected_ep_size <= 1:
            return

        moe_match = re.search(
            r"\[MOE_WARMUP\][^\n]*mode=slot[^\n]*ep_size=(\d+)"
            r"[^\n]*experts=(\d+)[^\n]*top_k=(\d+)"
            r"[^\n]*skew_fraction=([0-9]+(?:\.[0-9]+)?)",
            logs,
        )
        if moe_match is None:
            raise AssertionError(
                "warmup did not emit a structured MoE skew line\n"
                f"actual evidence:\n{self._evidence_lines(logs)}"
            )

        actual_ep_size = int(moe_match.group(1))
        expert_num = int(moe_match.group(2))
        top_k = int(moe_match.group(3))
        actual_fraction = float(moe_match.group(4))
        try:
            skew_mult = float(self._server_env("MOE_SKEW_MULT", "1.5"))
            skew_add = float(self._server_env("MOE_SKEW_ADD", "0.1"))
        except ValueError:
            skew_mult, skew_add = 1.5, 0.1
        expected_fraction = (
            1.0
            if expert_num <= top_k
            else max(
                1.0 / expected_ep_size,
                min(1.0, (1.0 / expected_ep_size) * skew_mult + skew_add),
            )
        )
        if actual_ep_size != expected_ep_size or not math.isclose(
            actual_fraction, expected_fraction, rel_tol=0.0, abs_tol=1e-6
        ):
            raise AssertionError(
                "MoE warmup skew mismatch: "
                f"actual ep_size={actual_ep_size} skew_fraction={actual_fraction:.6f}; "
                f"expected ep_size={expected_ep_size} skew_fraction={expected_fraction:.6f} "
                f"from experts={expert_num} top_k={top_k} "
                f"MOE_SKEW_MULT={skew_mult} MOE_SKEW_ADD={skew_add}"
            )


class _WarmupSmokeQuery(BaseModel):
    max_seq_len: int
    max_batch_tokens: int
    num_seqs: int


class PrefillWarmupComparer(WarmupComparerMixin, BaseComparer):
    """Validate that a standalone PD-prefill server completes warmup and KV sizing."""

    def format_query(self, query_json: Dict[str, Any]) -> BaseModel:
        return _WarmupSmokeQuery(**self.qr_info["prefill_warmup_smoke"])

    def format_result(self, result_json: Dict[str, Any]) -> BaseModel:
        raise NotImplementedError

    def curl_response_to_json(
        self, query_info: BaseModel, curl_response: Any
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def compare_result(self, expect_result: Any, actual_result: Any) -> None:
        raise NotImplementedError

    @staticmethod
    def _read_logs(root: str) -> str:
        chunks: List[str] = []
        for path in sorted(glob.glob(os.path.join(root, "*_logs", "*.log"))):
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    chunks.append(f.read())
            except OSError:
                continue
        return "\n".join(chunks)

    def _assert_prefill_warmup_logs(self) -> None:
        process_log = self.server_manager.log_file_path
        if not process_log:
            raise AssertionError("prefill process log path is unavailable")
        output_root = os.path.dirname(os.path.dirname(process_log))
        logs = self._read_logs(output_root)

        query = self.format_query({})
        required = {
            "prefill shape": rf"\[PREFILL_WARMUP\].*max_seq_len={query.max_seq_len}"
            rf".*max_batch_tokens={query.max_batch_tokens}.*num_seqs={query.num_seqs}",
            "KV sizing": r"\[KV_ALLOC\] warm_up=1.*torch_peak=\d+ MiB",
            "final blocks": r"\[KV_ALLOC\] final block_num=[1-9][0-9]*",
        }
        missing = [
            name
            for name, pattern in required.items()
            if re.search(pattern, logs) is None
        ]
        if missing:
            raise AssertionError(
                "prefill warmup did not emit: "
                + ", ".join(missing)
                + "\nactual evidence:\n"
                + self._evidence_lines(logs)
            )

        memory_match = re.search(
            r"warm up done, max runtime used memory: (\d+) bytes", logs
        )
        if memory_match is None or int(memory_match.group(1)) <= 0:
            actual = memory_match.group(1) if memory_match else "missing"
            raise AssertionError(
                f"prefill warmup max runtime memory must be positive, got {actual} bytes\n"
                f"actual evidence:\n{self._evidence_lines(logs)}"
            )

        expected_ep_size = self._configured_ep_size()
        self._assert_moe_warmup_skew(logs, expected_ep_size)

    def run(self) -> None:
        query = self.format_query({})
        self.tracer.query = query
        self._assert_prefill_warmup_logs()

        output_root = os.path.dirname(
            os.path.dirname(self.server_manager.log_file_path)
        )
        logs = self._read_logs(output_root)
        fatal = re.search(
            r"out.?of.?memory|CUDA out of|illegal memory access|"
            r"kv.?cache[^\n]*\bfull\b|no free block",
            logs,
            flags=re.IGNORECASE,
        )
        if fatal:
            raise AssertionError(
                f"GPU/KV failure found in smoke logs: {fatal.group(0)}"
            )
        logging.info(
            "prefill-only warmup smoke passed: num_seqs=%d tokens_per_seq=%d",
            query.num_seqs,
            query.max_seq_len - 1,
        )

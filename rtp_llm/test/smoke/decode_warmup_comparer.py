import glob
import logging
import os
import re
from typing import Any, Dict, List

from pydantic import BaseModel
from smoke.base_comparer import BaseComparer
from smoke.prefill_warmup_comparer import WarmupComparerMixin


class _DecodeWarmupSmokeQuery(BaseModel):
    max_seq_len: int
    max_generate_batch_size: int
    expected_ep_size: int


class DecodeWarmupComparer(WarmupComparerMixin, BaseComparer):
    """Validate that a standalone PD-decode server completes decode warmup and KV sizing."""

    def format_query(self, query_json: Dict[str, Any]) -> BaseModel:
        return _DecodeWarmupSmokeQuery(**self.qr_info["decode_warmup_smoke"])

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

    def _assert_decode_warmup_logs(self) -> None:
        process_log = self.server_manager.log_file_path
        if not process_log:
            raise AssertionError("decode process log path is unavailable")
        output_root = os.path.dirname(os.path.dirname(process_log))
        logs = self._read_logs(output_root)

        query = self.format_query({})
        required = [
            # Engine constructor log: "warm up ... query begin"
            r"warm up.*max_seq_len.*query begin",
            # Engine constructor log: "warm up done" with memory info
            r"warm up done.*max runtime used memory.*device reserved memory",
            # torch_peak may legitimately be zero when the measured growth is non-torch.
            r"\[KV_ALLOC\] warm_up=1.*torch_peak=\d+ MiB",
            # Final block allocation succeeded
            r"\[KV_ALLOC\] final block_num=[1-9][0-9]*",
            # CUDA graph capture completed rather than silently falling back.
            r"capture success for batch size: [1-9][0-9]*",
        ]
        missing = [pattern for pattern in required if re.search(pattern, logs) is None]
        if missing:
            raise AssertionError(
                "decode warmup did not emit the expected evidence: "
                + ", ".join(missing)
            )

        memory_match = re.search(
            r"warm up done, max runtime used memory: (\d+) bytes", logs
        )
        if memory_match is None or int(memory_match.group(1)) <= 0:
            actual = memory_match.group(1) if memory_match else "missing"
            raise AssertionError(
                f"decode warmup max runtime memory must be positive, got {actual} bytes"
            )

        configured_ep_size = self._configured_ep_size()
        if query.expected_ep_size != configured_ep_size:
            raise AssertionError(
                "decode warmup EP configuration mismatch: "
                f"JSON expected_ep_size={query.expected_ep_size}, "
                f"server launched with ep_size={configured_ep_size}"
            )
        self._assert_moe_warmup_skew(logs, configured_ep_size)

        # Exact-value contract: the warmup must actually run at the declared max_seq_len and
        # concurrency (num_return_sequences == max_generate_batch_size == concurrency_limit).
        # A regex-presence check alone would silently pass at the default concurrency (32) while
        # the JSON declares 256.
        match = re.search(
            r"\[DECODE_WARMUP\] max_seq_len=(\d+) num_return_sequences=(\d+)", logs
        )
        if match is None:
            raise AssertionError(
                "decode warmup did not emit the [DECODE_WARMUP] structured line"
            )
        actual_max_seq_len = int(match.group(1))
        actual_num_return_sequences = int(match.group(2))
        if actual_max_seq_len != query.max_seq_len:
            raise AssertionError(
                f"decode warmup max_seq_len mismatch: expected {query.max_seq_len}, "
                f"got {actual_max_seq_len}"
            )
        if actual_num_return_sequences != query.max_generate_batch_size:
            raise AssertionError(
                "decode warmup ran at the wrong concurrency: expected "
                f"num_return_sequences={query.max_generate_batch_size}, "
                f"got {actual_num_return_sequences} (pass --concurrency_limit "
                f"{query.max_generate_batch_size})"
            )

    def run(self) -> None:
        query = self.format_query({})
        self.tracer.query = query
        self._assert_decode_warmup_logs()

        output_root = os.path.dirname(
            os.path.dirname(self.server_manager.log_file_path)
        )
        logs = self._read_logs(output_root)
        fatal = re.search(
            r"out.?of.?memory|CUDA out of|illegal memory access|"
            r"cudaErrorStreamCapture|capture[^\n]*(?:abort|failed)|"
            r"kv.?cache[^\n]*\bfull\b|no free block",
            logs,
            flags=re.IGNORECASE,
        )
        if fatal:
            raise AssertionError(
                f"GPU/KV failure found in smoke logs: {fatal.group(0)}"
            )
        logging.info(
            "decode warmup smoke passed: max_seq_len=%d max_generate_batch_size=%d",
            query.max_seq_len,
            query.max_generate_batch_size,
        )

import logging
import os
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from rtp_llm.test.perf_test.batch_perf_impl import BatchPerfImpl
from rtp_llm.test.perf_test.dataclass import (
    MetricState,
    TableType,
    TestResultMetrics,
    create_metrics_table,
)


def _kv_blocks_per_seq(input_len: int) -> int:
    block = max(int(os.environ.get("PERF_KV_SEQ_SIZE_PER_BLOCK", "256")), 1)
    return (int(input_len) + block - 1) // block


def _kv_skip_reason(batch_size: int, input_len: int) -> Optional[str]:
    """Check per-rank FULL KV capacity; a zero total disables the check."""
    total = int(os.environ.get("PERF_KV_TOTAL_BLOCKS", "0"))
    if total <= 0:
        return None
    reserve = int(os.environ.get("PERF_KV_GROUP_RESERVE", "1490"))
    per_seq = _kv_blocks_per_seq(input_len)
    need = int(batch_size) * per_seq + reserve
    if need > total:
        return (
            f"need={need} (bs={batch_size}*{per_seq}+reserve={reserve}) "
            f"> PERF_KV_TOTAL_BLOCKS={total}"
        )
    return None


def _empty_metrics() -> TestResultMetrics:
    return TestResultMetrics(total_requests=0, success_requests=0, fail_requests=0)


class GridRunner:
    """Grid-mode performance test (batch_size x input_len cartesian product)."""

    def __init__(
        self,
        port: int,
        dp_size: int,
        batch_size_list: List[int],
        input_len_list: List[int],
        input_query_dict: Dict[int, str],
        *,
        is_decode: bool = True,
        dump_json_path: str = ".",
        decode_test_length: int = 10,
        tp_size: int = 1,
        generate_config: Optional[Dict[str, Any]] = None,
    ):
        self._port = port
        self._dp_size = dp_size
        self._batch_size_list = batch_size_list
        self._input_len_list = input_len_list
        self._input_query_dict = input_query_dict
        self._is_decode = is_decode
        self._dump_json_path = dump_json_path
        self._decode_test_length = decode_test_length
        self._tp_size = tp_size
        self._generate_config = generate_config or {}
        self._title = "Decode Result" if is_decode else "Prefill Result"

    def warmup(self) -> None:
        warmup_runs = int(os.environ.get("PERF_GRID_WARMUP_RUNS", "1"))
        input_len = next(
            (seq for seq in self._input_len_list if _kv_skip_reason(1, seq) is None),
            None,
        )
        if warmup_runs <= 0 or input_len is None:
            return
        logging.info(
            f"in warmup, base_port: {self._port}, dp_size: {self._dp_size}, "
            f"batch_size: {1 * self._dp_size}, "
            f"input_len: {input_len}, runs: {warmup_runs}"
        )
        BatchPerfImpl(
            self._port,
            self._dp_size,
            1 * self._dp_size,
            self._input_query_dict[input_len],
            self._is_decode,
            1000,
            self._decode_test_length,
            False,
            self._generate_config,
            warmup_runs=0,
            measure_runs=warmup_runs,
            profile_runs=0,
        ).run()

    def run(self) -> List[MetricState]:
        """Run sequence-first so failed cells only suppress larger batches there."""
        self.warmup()
        logging.info("start to run perf test")
        metrics_list: List[MetricState] = []
        skip_bs_ge: Dict[int, int] = {}
        skip_on_fail = os.environ.get("PERF_SKIP_ON_FAIL", "0") == "1"

        total_tests = len(self._batch_size_list) * len(self._input_len_list)

        with tqdm(
            total=total_tests, desc=f"Running {self._title}", unit="test"
        ) as pbar:
            for input_len in self._input_len_list:
                for batch_size in self._batch_size_list:
                    pbar.set_description(
                        f"Running {self._title} - "
                        f"batch_size: {batch_size}, input_len: {input_len}"
                    )
                    limit = skip_bs_ge.get(input_len)
                    reason = _kv_skip_reason(batch_size, input_len)
                    if limit is not None and batch_size >= limit:
                        reason = f"limit bs>={limit} after KV/fail"
                    if reason:
                        logging.warning(
                            "[PERF_SKIP] bs=%d seq=%d %s",
                            batch_size,
                            input_len,
                            reason,
                        )
                        skip_bs_ge[input_len] = min(
                            skip_bs_ge.get(input_len, batch_size), batch_size
                        )
                        metrics_list.append(
                            MetricState(input_len, batch_size, _empty_metrics())
                        )
                        pbar.update(1)
                        continue

                    phase = "decode" if self._is_decode else "prefill"
                    trace_name = f"bs{batch_size}_seq{input_len}_{phase}"
                    metric = BatchPerfImpl(
                        self._port,
                        self._dp_size,
                        batch_size * self._dp_size,
                        self._input_query_dict[input_len],
                        self._is_decode,
                        500,
                        self._decode_test_length,
                        True,
                        self._generate_config,
                        trace_name,
                    ).run()
                    metrics_list.append(MetricState(input_len, batch_size, metric))
                    if skip_on_fail and metric.success_requests == 0:
                        skip_bs_ge[input_len] = batch_size
                        logging.warning(
                            "[PERF_SKIP_ON_FAIL] bs=%d seq=%d success=0, "
                            "skip larger bs for this seq",
                            batch_size,
                            input_len,
                        )
                    pbar.update(1)

        metrics_table = create_metrics_table(
            TableType.Decode if self._is_decode else TableType.Prefill,
            metrics_list,
            self._dump_json_path,
            {"dp_size": self._dp_size, "tp_size": self._tp_size},
            self._title,
            self._generate_config,
        )
        logging.info("metrics_table: \n" + str(metrics_table))
        return metrics_list

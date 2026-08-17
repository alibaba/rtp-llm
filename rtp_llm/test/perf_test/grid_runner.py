import logging
import os
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from rtp_llm.test.perf_test.batch_perf_impl import BatchPerfImpl
from rtp_llm.test.perf_test.dataclass import (
    MetricState,
    TableType,
    create_metrics_table,
)


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
        logging.info(
            f"in warmup, base_port: {self._port}, dp_size: {self._dp_size}, "
            f"batch_size: {1 * self._dp_size}, "
            f"input_len: {self._input_len_list[0]}, runs: {warmup_runs}"
        )
        BatchPerfImpl(
            self._port,
            self._dp_size,
            1 * self._dp_size,
            self._input_query_dict[self._input_len_list[0]],
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
        """Warmup then iterate batch_size x input_len, return metrics."""
        self.warmup()
        logging.info("start to run perf test")
        metrics_list: List[MetricState] = []
        profile_cases = []

        total_tests = len(self._batch_size_list) * len(self._input_len_list)

        with tqdm(
            total=total_tests, desc=f"Running {self._title}", unit="test"
        ) as pbar:
            for batch_size in self._batch_size_list:
                for input_len in self._input_len_list:
                    pbar.set_description(
                        f"Running {self._title} - "
                        f"batch_size: {batch_size}, input_len: {input_len}"
                    )

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
                        False,
                        self._generate_config,
                        trace_name,
                        profile_runs=0,
                    ).run()
                    metrics_list.append(MetricState(input_len, batch_size, metric))
                    profile_cases.append((batch_size, input_len, trace_name))
                    if (
                        os.environ.get("PERF_REQUIRE_ALL_SUCCESS", "1") == "1"
                        and metric.success_requests != metric.total_requests
                    ):
                        raise RuntimeError(
                            f"perf case {trace_name} succeeded for "
                            f"{metric.success_requests}/{metric.total_requests} requests"
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

        # Profile only after every timed case has completed. Timeline export is
        # asynchronous and DSpARK all-rank traces can be hundreds of MB per rank;
        # profiling an earlier grid point inline can therefore contaminate the
        # latency of the next point with trace serialization/flush time.
        profile_runs = int(os.environ.get("PERF_PROFILE_RUNS", "1"))
        if profile_runs > 0:
            logging.info("start profile-only pass after all timed measurements")
            for batch_size, input_len, trace_name in profile_cases:
                BatchPerfImpl(
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
                    warmup_runs=0,
                    measure_runs=0,
                    profile_runs=profile_runs,
                ).run()
        return metrics_list

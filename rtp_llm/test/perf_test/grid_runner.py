import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from rtp_llm.test.perf_test.batch_perf_impl import BatchPerfImpl
from rtp_llm.test.perf_test.dataclass import (
    MetricState,
    TableType,
    create_metrics_table,
)
from rtp_llm.test.perf_test.test_util import ReuseCacheQuery


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
        grid_cases: Optional[List[Tuple[int, int]]] = None,
        reuse_cache_query_dict: Optional[Dict[int, ReuseCacheQuery]] = None,
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
        self._grid_cases = grid_cases
        self._reuse_cache_query_dict = reuse_cache_query_dict or {}
        self._title = "Decode Result" if is_decode else "Prefill Result"

    def warmup(self) -> None:
        if self._reuse_cache_query_dict:
            logging.info("skip global grid warmup for reuse-cache hit-rate mode")
            return
        warmup_runs = int(os.environ.get("PERF_GRID_WARMUP_RUNS", "1"))
        logging.info(
            f"in warmup, base_port: {self._port}, dp_size: {self._dp_size}, "
            f"batch_size: {1 * self._dp_size}, "
            f"input_len: {self._warmup_input_len()}, runs: {warmup_runs}"
        )
        BatchPerfImpl(
            self._port,
            self._dp_size,
            1 * self._dp_size,
            self._input_query_dict[self._warmup_input_len()],
            self._is_decode,
            1000,
            self._decode_test_length,
            False,
            self._generate_config,
            warmup_runs=0,
            measure_runs=warmup_runs,
            profile_runs=0,
        ).run()

    def _warmup_input_len(self) -> int:
        return self._grid_cases[0][1] if self._grid_cases else self._input_len_list[0]

    def _iter_cases(self) -> List[Tuple[int, int]]:
        if self._grid_cases:
            return self._grid_cases
        return [
            (batch_size, input_len)
            for batch_size in self._batch_size_list
            for input_len in self._input_len_list
        ]

    def run(self) -> List[MetricState]:
        """Warmup then iterate batch_size x input_len, return metrics."""
        self.warmup()
        logging.info("start to run perf test")
        metrics_list: List[MetricState] = []

        grid_cases = self._iter_cases()

        with tqdm(
            total=len(grid_cases), desc=f"Running {self._title}", unit="test"
        ) as pbar:
            for batch_size, input_len in grid_cases:
                pbar.set_description(
                    f"Running {self._title} - "
                    f"batch_size: {batch_size}, input_len: {input_len}"
                )

                phase = "decode" if self._is_decode else "prefill"
                trace_name = f"bs{batch_size}_seq{input_len}_{phase}"
                reuse_cache_query = self._reuse_cache_query_dict.get(input_len)
                query = (
                    reuse_cache_query.hit_queries[0]
                    if reuse_cache_query
                    else self._input_query_dict[input_len]
                )
                query_variants = (
                    reuse_cache_query.hit_queries if reuse_cache_query else None
                )
                seed_query = reuse_cache_query.seed_query if reuse_cache_query else None
                target_reuse_len = (
                    reuse_cache_query.target_reuse_len if reuse_cache_query else 0
                )
                metric = BatchPerfImpl(
                    self._port,
                    self._dp_size,
                    batch_size * self._dp_size,
                    query,
                    self._is_decode,
                    500,
                    self._decode_test_length,
                    True,
                    self._generate_config,
                    trace_name,
                    reuse_cache_seed_query=seed_query,
                    query_variants=query_variants,
                    target_reuse_len=target_reuse_len,
                ).run()
                metrics_list.append(MetricState(input_len, batch_size, metric))

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

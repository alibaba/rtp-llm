import logging
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from rtp_llm.test.perf_test.batch_perf_impl import BatchPerfImpl
from rtp_llm.test.perf_test.dataclass import (
    DistributionMetricState,
    create_distribution_metrics_table,
)


class DistributionRunner:
    """Distribution-mode performance test (real-world seq_len combinations)."""

    def __init__(
        self,
        port: int,
        dp_size: int,
        test_config: Dict[str, Any],
        input_query_dict: Dict[int, str],
        *,
        dump_json_path: str = ".",
        decode_test_length: int = 10,
        generate_config: Optional[Dict[str, Any]] = None,
    ):
        self._port = port
        self._dp_size = dp_size
        self._test_config = test_config
        self._input_query_dict = input_query_dict
        self._dump_json_path = dump_json_path
        self._decode_test_length = decode_test_length
        self._generate_config = generate_config or {}
        self._title = "Distribution Decode Result"

    def warmup(self) -> None:
        batch_seq_len_map = self._test_config["batch_seq_len_map"]
        sorted_bs_keys = sorted(batch_seq_len_map.keys(), key=lambda x: int(x))
        first_seq_lens = batch_seq_len_map[sorted_bs_keys[0]]
        logging.info(f"in warmup, port: {self._port}, dp_size: {self._dp_size}")
        BatchPerfImpl(
            self._port,
            self._dp_size,
            1 * self._dp_size,
            self._input_query_dict[first_seq_lens[0]],
            True,
            1000,
            self._decode_test_length,
            False,
            self._generate_config,
        ).run()

    def run(self) -> List[DistributionMetricState]:
        """Warmup then iterate batch_seq_len_map, return metrics."""
        self.warmup()
        logging.info("start to run distribution perf test")

        batch_seq_len_map = self._test_config["batch_seq_len_map"]
        sorted_bs_keys = sorted(batch_seq_len_map.keys(), key=lambda x: int(x))
        metrics_list: List[DistributionMetricState] = []

        with tqdm(
            total=len(sorted_bs_keys),
            desc=f"Running {self._title}",
            unit="test",
        ) as pbar:
            for bs_str in sorted_bs_keys:
                seq_len_list = batch_seq_len_map[bs_str]
                actual_bs = len(seq_len_list)
                pbar.set_description(
                    f"Running {self._title} - bs: {actual_bs}, "
                    f"seq_lens: {min(seq_len_list)}~{max(seq_len_list)}"
                )

                queries = [self._input_query_dict[sl] for sl in seq_len_list]
                queries_dp = queries * self._dp_size

                trace_name = (
                    f"bs{actual_bs}_seq{min(seq_len_list)}-{max(seq_len_list)}_decode"
                )
                metric = BatchPerfImpl(
                    self._port,
                    self._dp_size,
                    actual_bs * self._dp_size,
                    queries_dp,
                    True,
                    500,
                    self._decode_test_length,
                    True,
                    self._generate_config,
                    trace_name,
                ).run()

                metrics_list.append(
                    DistributionMetricState(actual_bs, seq_len_list, metric)
                )
                pbar.update(1)

        distribution_source = self._test_config.get("source_table", "unknown")
        metrics_table = create_distribution_metrics_table(
            metrics_list,
            self._dump_json_path,
            distribution_source,
            self._title,
            self._generate_config,
        )
        logging.info("metrics_table: \n" + str(metrics_table))
        return metrics_list

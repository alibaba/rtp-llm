import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from rtp_llm.test.perf_test.batch_perf_impl import BatchPerfImpl
from rtp_llm.test.perf_test.dataclass import (
    TpsResult,
    TpsSearchStep,
    create_tps_result_table,
)


class TpsBinarySearchRunner:
    """Binary-search for the maximum batch size that satisfies a target TPOT."""

    def __init__(
        self,
        port: int,
        dp_size: int,
        target_tpot: float,
        max_bs: int,
        decode_test_length: int = 10,
        generate_config: Optional[Dict[str, Any]] = None,
        dump_json_path: str = ".",
        num_measures: int = 5,
    ):
        self._port = port
        self._dp_size = dp_size
        self._target_tpot = target_tpot
        self._max_bs = max_bs
        self._decode_test_length = decode_test_length
        self._generate_config = generate_config or {}
        self._dump_json_path = dump_json_path
        self._num_measures = num_measures

    def warmup(self, query: str) -> None:
        logging.info(f"TPS warmup: port={self._port}, dp_size={self._dp_size}")
        BatchPerfImpl(
            self._port,
            self._dp_size,
            1 * self._dp_size,
            query,
            True,
            1000,
            self._decode_test_length,
            False,
            self._generate_config,
        ).run()

    def _test_bs(
        self, bs: int, queries: Any, trace_label: str = ""
    ) -> Tuple[bool, float, float]:
        trace_name = f"bs{bs}_{trace_label}_decode" if trace_label else f"bs{bs}_decode"
        metric = BatchPerfImpl(
            self._port,
            self._dp_size,
            bs * self._dp_size,
            queries,
            True,
            500,
            self._decode_test_length,
            True,
            self._generate_config,
            trace_name,
        ).run(num_measures=self._num_measures)
        sr = (
            metric.success_requests / metric.total_requests
            if metric.total_requests > 0
            else 0
        )
        ok = sr == 1.0 and metric.avg_decode_time <= self._target_tpot
        return ok, metric.avg_decode_time, sr

    @staticmethod
    def _make_bs_candidates(max_bs: int, step: int = 4) -> List[int]:
        """Generate candidate BS list: [1, step, 2*step, ..., max_bs]."""
        candidates = [1]
        bs = step
        while bs <= max_bs:
            candidates.append(bs)
            bs += step
        if candidates[-1] != max_bs:
            candidates.append(max_bs)
        return candidates

    def _binary_search(
        self,
        queries_fn: Callable[[int], Any],
        label: str,
        max_bs: Optional[int] = None,
    ) -> TpsResult:
        candidates = self._make_bs_candidates(max_bs or self._max_bs)
        lo, hi = 0, len(candidates) - 1
        best_bs, best_tpot = 0, 0.0
        steps: List[TpsSearchStep] = []

        while lo <= hi:
            mid_idx = (lo + hi) // 2
            bs = candidates[mid_idx]
            queries = queries_fn(bs)
            ok, tpot, sr = self._test_bs(bs, queries, label)
            steps.append(TpsSearchStep(bs, tpot, sr, ok))
            status = "PASS" if ok else "FAIL"
            logging.debug(
                f"  [{label}] BS={bs}: TPOT={tpot:.2f}ms, "
                f"success_rate={sr:.2f}, target={self._target_tpot}ms -> {status}"
            )
            if ok:
                best_bs, best_tpot = bs, tpot
                lo = mid_idx + 1
            else:
                hi = mid_idx - 1

        if best_bs == 0:
            logging.warning(f"  [{label}] No BS satisfies target TPOT!")

        tps = best_bs / (best_tpot / 1000) if best_tpot > 0 else 0.0
        return TpsResult(
            target_tpot=self._target_tpot,
            best_bs=best_bs,
            actual_tpot=best_tpot,
            tps=tps,
            search_steps=steps,
        )

    def run_grid(
        self,
        input_len_list: List[int],
        input_query_dict: Dict[int, str],
        max_bs_per_len: Optional[Dict[int, int]] = None,
    ) -> List[TpsResult]:
        """Grid mode: binary search for each input_len."""
        self.warmup(input_query_dict[input_len_list[0]])
        results: List[TpsResult] = []

        for input_len in input_len_list:
            query = input_query_dict[input_len]
            queries_fn = lambda bs, q=query: q  # type: ignore[arg-type]
            effective_max = (
                max_bs_per_len.get(input_len, self._max_bs)
                if max_bs_per_len
                else self._max_bs
            )
            result = self._binary_search(
                queries_fn, f"seq{input_len}", max_bs=effective_max
            )
            result.input_len = input_len
            results.append(result)

        create_tps_result_table(
            results,
            self._dump_json_path,
            "TPS Grid Result",
            self._generate_config,
        )
        return results

    def run_distribution(
        self,
        test_config: Dict[str, Any],
        input_query_dict: Dict[int, str],
    ) -> List[TpsResult]:
        """Distribution mode: binary search BS using pre-sampled seq_lens."""
        batch_seq_len_map = test_config["batch_seq_len_map"]
        sorted_keys = sorted(batch_seq_len_map.keys(), key=int)
        # Clamp to self._max_bs (concurrency_limit) to avoid exceeding configured limit
        max_sampled_bs = min(int(sorted_keys[-1]), self._max_bs)

        def queries_fn(bs: int) -> List[str]:
            # Find the smallest pre-sampled key >= bs
            for k in sorted_keys:
                if int(k) >= bs:
                    seq_lens = batch_seq_len_map[k][:bs]
                    queries = [input_query_dict[sl] for sl in seq_lens]
                    return queries * self._dp_size
            # Fallback: use largest pre-sampled seq_lens
            seq_lens = batch_seq_len_map[sorted_keys[-1]][:bs]
            queries = [input_query_dict[sl] for sl in seq_lens]
            return queries * self._dp_size

        first_seq_lens = batch_seq_len_map[sorted_keys[0]]
        self.warmup(input_query_dict[first_seq_lens[0]])

        result = self._binary_search(queries_fn, "distribution", max_bs=max_sampled_bs)
        create_tps_result_table(
            [result],
            self._dump_json_path,
            "TPS Distribution Result",
            self._generate_config,
        )
        return [result]

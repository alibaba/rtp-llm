"""Perf test configuration: argument parsing, path resolution, PerfTestConfig generation."""

import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple

from rtp_llm.test.perf_test.dataclass import PerfTestConfig
from rtp_llm.test.perf_test.dataset import KNOWN_DATASETS, extract_arg
from rtp_llm.test.perf_test.hub_download import (
    needs_perf_hub_resolve,
    resolve_checkpoint_or_tokenizer_for_perf,
)
from rtp_llm.test.perf_test.perf_utils import auto_generate_bs_list
from rtp_llm.test.perf_test.sampling import prepare_distribution_config


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="RTP-LLM batch decode performance test runner. "
        "Unrecognized arguments are forwarded to the engine server.",
    )

    perf = parser.add_argument_group("perf test configuration")
    perf.add_argument(
        "--batch_size",
        type=str,
        default="1,8,16",
        help="Comma-separated batch sizes for grid mode",
    )
    perf.add_argument(
        "--input_len",
        type=str,
        default="1024,4096",
        help="Comma-separated input lengths for grid mode",
    )
    dataset_group = perf.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--dataset_name",
        type=str,
        default="",
        help=f"Known dataset name. Choices: {list(KNOWN_DATASETS.keys())}",
    )
    dataset_group.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Local path to dataset JSON.",
    )
    perf.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Path to distribution.csv for runtime stratified sampling.",
    )
    perf.add_argument(
        "--test_json",
        type=str,
        default="",
        help="Path to previously saved test config JSON for replay",
    )
    perf.add_argument(
        "--partial",
        type=int,
        default=1,
        choices=[1, 2],
        help="1: decode only (default), 2: prefill only (grid mode only, not supported in distribution mode)",
    )
    perf.add_argument("--generate_config", type=str, default="{}")
    perf.add_argument(
        "--result_dir",
        type=str,
        default=os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "./perf_results"),
    )
    perf.add_argument("--decode_test_length", type=int, default=10)
    perf.add_argument(
        "--num_measures",
        type=int,
        default=5,
        help="Number of measurements per BS. Trim min/max and average the rest.",
    )
    perf.add_argument(
        "--target_tpot",
        type=float,
        default=0,
        help="Target TPOT (ms). When set, binary search for max BS satisfying TPOT, compute TPS",
    )

    engine = parser.add_argument_group(
        "engine args consumed by perf test (also forwarded to server)"
    )
    engine.add_argument("--dp_size", type=int, default=1)
    engine.add_argument("--max_seq_len", type=int, default=8192)
    engine.add_argument("--concurrency_limit", type=int, default=64)

    args, remaining = parser.parse_known_args()
    return args, remaining


def _replace_cli_value(argv: List[str], key: str, new_value: str) -> None:
    flag = f"--{key}"
    prefix = f"--{key}="
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            argv[i + 1] = new_value
            return
        if arg.startswith(prefix):
            argv[i] = prefix + new_value
            return
    raise ValueError(
        f"perf_test: missing {flag} in argv, cannot replace with local path"
    )


def resolve_perf_engine_paths(remaining: List[str]) -> List[str]:
    """Resolve Hub links / repo ids to local paths before forwarding to engine."""
    out = list(remaining)
    resolved_cache: Dict[str, str] = {}
    for k in ("checkpoint_path", "tokenizer_path"):
        val = extract_arg(out, k)
        if not val or not needs_perf_hub_resolve(val):
            continue
        if val not in resolved_cache:
            local = resolve_checkpoint_or_tokenizer_for_perf(val)
            logging.info(f"perf_test: resolved --{k} -> {local}")
            resolved_cache[val] = local
        _replace_cli_value(out, k, resolved_cache[val])
    return out


def prepare_config(args: argparse.Namespace, remaining: List[str]) -> PerfTestConfig:
    """Build a unified PerfTestConfig from CLI args."""
    batch_size_explicit = any(a.startswith("--batch_size") for a in sys.argv[1:])
    distribution_mode = (
        args.dataset_name or args.dataset_path or args.dataset or args.test_json
    )

    if distribution_mode:
        if args.partial == 2:
            raise ValueError(
                "Distribution mode only supports decode (--partial 1). "
                "Prefill testing (--partial 2) is only available in grid mode."
            )
        tokenizer_path = (
            extract_arg(remaining, "tokenizer_path")
            or extract_arg(remaining, "checkpoint_path")
            or os.environ.get("TOKENIZER_PATH", "")
        )
        test_config = prepare_distribution_config(
            tokenizer_path=tokenizer_path,
            max_seq_len=args.max_seq_len,
            max_concurrency=args.concurrency_limit * args.dp_size,
            result_dir=args.result_dir,
            dataset_name=args.dataset_name,
            dataset_path=args.dataset_path,
            dataset_csv=args.dataset,
            test_json=args.test_json,
        )
        batch_seq_len_map = test_config["batch_seq_len_map"]
        all_seq_lens = sorted(
            set(sl for sls in batch_seq_len_map.values() for sl in sls)
        )
        needed_seq_len = max(all_seq_lens) + args.decode_test_length
        return PerfTestConfig(
            is_distribution=True,
            all_seq_lens=all_seq_lens,
            batch_size_list=[],
            input_len_list=[],
            max_seq_len=max(needed_seq_len, args.max_seq_len),
            max_concurrency=max(int(k) for k in batch_seq_len_map),
            test_config=test_config,
        )

    # Grid mode
    input_len_list = [int(x) for x in args.input_len.split(",")]

    # Prefill mode (partial=2): always BS=1, no need for large BS list
    if args.partial == 2:
        batch_size_list = [1]
    elif batch_size_explicit:
        batch_size_list = [int(x) for x in args.batch_size.split(",")]
    else:
        batch_size_list = auto_generate_bs_list(args.concurrency_limit)

    effective_max_concurrency = max(batch_size_list)
    if args.target_tpot > 0:
        effective_max_concurrency = max(
            effective_max_concurrency, args.concurrency_limit
        )

    return PerfTestConfig(
        is_distribution=False,
        all_seq_lens=input_len_list,
        batch_size_list=batch_size_list,
        input_len_list=input_len_list,
        max_seq_len=max(input_len_list) + args.decode_test_length,
        max_concurrency=effective_max_concurrency,
    )

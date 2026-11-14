"""Resolve model and tokenizer engine arguments for standalone benchmarks."""

import logging
from typing import Dict, List

from rtp_llm.test.perf_test.dataset import extract_arg
from rtp_llm.test.perf_test.hub_download import (
    needs_perf_hub_resolve,
    resolve_checkpoint_or_tokenizer_for_perf,
)


def _replace_cli_value(argv: List[str], key: str, new_value: str) -> None:
    flag = f"--{key}"
    prefix = f"--{key}="
    for index, arg in enumerate(argv):
        if arg == flag and index + 1 < len(argv):
            argv[index + 1] = new_value
            return
        if arg.startswith(prefix):
            argv[index] = prefix + new_value
            return
    raise ValueError(f"missing {flag} in engine arguments")


def resolve_engine_paths(engine_args: List[str]) -> List[str]:
    """Resolve Hub IDs in checkpoint/tokenizer arguments to local paths."""
    resolved_args = list(engine_args)
    resolved_cache: Dict[str, str] = {}
    for key in ("checkpoint_path", "tokenizer_path"):
        value = extract_arg(resolved_args, key)
        if not value or not needs_perf_hub_resolve(value):
            continue
        if value not in resolved_cache:
            resolved_cache[value] = resolve_checkpoint_or_tokenizer_for_perf(value)
        local_path = resolved_cache[value]
        logging.info("resolved --%s -> %s", key, local_path)
        _replace_cli_value(resolved_args, key, local_path)
    return resolved_args

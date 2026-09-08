# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Generate per-kernel `default_config` from one Triton autotune benchmark run
# and write each `default_config` to `configs/{GPU}/{kernel}.json`.
#
# Usage:
#     # Run op benchmark once, compare the picked config vs. the existing
#     # committed JSON without modifying files:
#     python -m rtp_llm.models_py.triton_kernels.autotune_cache.scripts.extract \
#         --init-default --op kda --dry-run
#
#     # Same but actually write:
#     python -m rtp_llm.models_py.triton_kernels.autotune_cache.scripts.extract \
#         --init-default --op kda
#
#     # Extract from an existing Triton cache directory (no benchmark), write existing config:
#     python -m rtp_llm.models_py.triton_kernels.autotune_cache.scripts.extract \
#         --extract-once --triton-cache-dir ~/.triton/cache
#
#     # Just list *.autotune.json files:
#     python -m rtp_llm.models_py.triton_kernels.autotune_cache.scripts.extract \
#         --list-only --triton-cache-dir ~/.triton/cache

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any


def _clear_triton_cache_dir(path: Path) -> None:
    """Remove any prior autotune output so the next run re-benchmarks."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _timing_less(a: Any, b: Any) -> bool:
    """Treat timing as lexicographic tuple of floats (matches Triton schema)."""

    def _t(x: Any) -> tuple[float, ...]:
        if isinstance(x, (int, float)):
            return (float(x),)
        if isinstance(x, (list, tuple)):
            return tuple(float(v) for v in x)
        return (float("inf"),)

    return _t(a) < _t(b)


def _run_benchmark(op: str, cache_dir: Path) -> None:
    """Run the op's generator once into `cache_dir` as TRITON_CACHE_DIR."""
    import torch

    from rtp_llm.models_py.triton_kernels.autotune_cache.scripts.generators import (
        REGISTRY,
        available_ops,
    )
    from rtp_llm.models_py.triton_kernels.fla.utils import device

    if op not in REGISTRY:
        raise ValueError(f"Unsupported op: {op!r}. Available: {available_ops()}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    _clear_triton_cache_dir(cache_dir)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
    REGISTRY[op](device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _collect_winners(cache_dir: Path) -> dict[str, Any]:
    """Scan Triton cache dir and return {kernel_name: fastest winner config}.

    If multiple *.autotune.json exist per kernel (multiple shapes in one run),
    pick the one with the smallest timing.
    """
    from rtp_llm.models_py.triton_kernels.autotune_cache.export import (
        WinnerSample,
        collect_winners,
    )

    best: dict[str, WinnerSample] = {}
    for w in collect_winners(cache_dir):
        prev = best.get(w.kernel_name)
        if prev is None or _timing_less(w.timing, prev.timing):
            best[w.kernel_name] = w
    return {k: v.config for k, v in best.items()}


def init_default(
    op: str,
    output_dir: Path,
    cache_dir: Path,
    dry_run: bool = False,
) -> int:
    """Run one op benchmark and write each kernel's default_config.

    In dry-run, runs benchmark + compares against existing JSON, but does
    not write. Returns 0 on success, 2 if no winners were collected.
    """
    from rtp_llm.models_py.triton_kernels.autotune_cache.cache import KernelConfigFile
    from rtp_llm.models_py.triton_kernels.autotune_cache.export import (
        _normalize_config,
        write_default_config_json,
    )

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[benchmark] running {op} into {cache_dir}")
    _run_benchmark(op, cache_dir)
    winners = _collect_winners(cache_dir)
    if not winners:
        print(
            "ERROR: no *.autotune.json produced under "
            f"{cache_dir}. Check that the generator actually invokes "
            "triton.autotune (TRITON_AUTOTUNE_CACHE_MODE=disabled is forced)."
        )
        return 2

    print("\n" + "=" * 60)
    print(f"Picked winner for {len(winners)} kernel(s)")
    print("=" * 60)

    written_count = 0
    unchanged_count = 0
    matches_existing_count = 0
    differs_from_existing_count = 0
    no_existing_count = 0
    for kernel_name in sorted(winners):
        cfg = winners[kernel_name]
        chosen_norm = _normalize_config(cfg)
        output_file = output_dir / f"{kernel_name}.json"

        existing = (
            KernelConfigFile.from_file(output_file) if output_file.exists() else None
        )
        existing_default = existing.default_config if existing is not None else None
        if existing_default is None:
            cmp_label = "NO_EXISTING"
            no_existing_count += 1
        elif existing_default == chosen_norm:
            cmp_label = "MATCHES_EXISTING"
            matches_existing_count += 1
        else:
            cmp_label = "DIFFERS_FROM_EXISTING"
            differs_from_existing_count += 1

        print(f"\n  kernel: {kernel_name}")
        print(f"    sampled: {chosen_norm}")
        if existing_default is not None:
            print(f"    existing: {existing_default}")
        print(f"    -> {cmp_label}")

        if dry_run:
            print(f"    (dry-run, not writing) {output_file}")
        else:
            status = write_default_config_json(output_file, kernel_name, cfg)
            print(f"    {status}: {output_file}")
            if status == "unchanged":
                unchanged_count += 1
            else:
                written_count += 1

    print("\n" + "=" * 60)
    if dry_run:
        print("DRY RUN — no files written")
        print(f"matches existing        : {matches_existing_count}")
        print(f"differs from existing   : {differs_from_existing_count}")
        print(f"no existing file        : {no_existing_count}")
    else:
        print(f"written                 : {written_count}")
        print(f"unchanged               : {unchanged_count}")
    print("=" * 60)
    if differs_from_existing_count > 0:
        print(
            "\nNOTE: one or more kernels produced a different winner than the\n"
            "committed JSON. A single benchmark sample is inherently noisy and\n"
            "the winner set tends to fall into a small number of numerically\n"
            "equivalent clusters. Before committing, validate by running smoke\n"
            "× N rounds and comparing output against golden data."
        )
    return 0


def extract_once(triton_cache_dir: Path, output_dir: Path) -> None:
    """Single-shot: take the best config from each *.autotune.json in
    `triton_cache_dir` and write it as that kernel's default_config.
    No fresh benchmark run — consumes whatever's already there.
    """
    from rtp_llm.models_py.triton_kernels.autotune_cache.export import (
        collect_winners,
        write_default_config_json,
    )

    if not triton_cache_dir.exists():
        print(
            f"Triton cache directory not found: {triton_cache_dir}. Nothing to extract."
        )
        sys.exit(1)

    winners = collect_winners(triton_cache_dir)
    if not winners:
        print(f"No *.autotune.json files found in {triton_cache_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(winners)} winner config(s). Writing to {output_dir}")
    per_kernel_best: dict[str, Any] = {}
    for w in winners:
        prev = per_kernel_best.get(w.kernel_name)
        if prev is None or _timing_less(w.timing, prev[1]):
            per_kernel_best[w.kernel_name] = (w.config, w.timing)

    for kernel_name, (cfg, _timing) in sorted(per_kernel_best.items()):
        output_file = output_dir / f"{kernel_name}.json"
        status = write_default_config_json(output_file, kernel_name, cfg)
        print(f"  {status}: {output_file}")


def list_autotune_cache_files(triton_cache_dir: Path) -> None:
    if not triton_cache_dir.exists():
        print(f"Triton cache directory not found: {triton_cache_dir}")
        return

    autotune_files = list(triton_cache_dir.rglob("*.autotune.json"))
    print(f"Found {len(autotune_files)} .autotune.json files in {triton_cache_dir}:\n")
    for i, file in enumerate(autotune_files, 1):
        print(f"{i}. {file}")


def _get_default_triton_cache_dir() -> Path:
    """Isolated Triton cache directory for bootstrap runs."""
    from triton.runtime.cache import knobs

    return Path(knobs.cache.dir) / "rtp_llm_autotune_bootstrap"


def _resolve_output_dir(output_dir: str | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)

    from rtp_llm.models_py.triton_kernels.autotune_cache.cache import get_config_dir

    return get_config_dir()


def main() -> int:
    os.environ["TRITON_AUTOTUNE_CACHE_MODE"] = "disabled"

    from rtp_llm.models_py.triton_kernels.autotune_cache.scripts.generators import (
        available_ops,
    )

    parser = argparse.ArgumentParser(
        description="Generate per-kernel default_config via one Triton autotune run.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: $TRITON_AUTOTUNE_CONFIG_DIR or "
        "rtp_llm/models_py/triton_kernels/autotune_cache/configs/{GPU})",
    )
    parser.add_argument(
        "--triton-cache-dir",
        type=str,
        help="Isolated Triton cache directory. "
        "Default: <triton default cache>/rtp_llm_autotune_bootstrap.",
    )
    parser.add_argument(
        "--init-default",
        action="store_true",
        help="Primary entry: run benchmark once, extract winner per kernel, "
        "write default_config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only valid with --init-default: run benchmark and compare "
        "against existing JSON, but do not modify any files.",
    )
    parser.add_argument(
        "--extract-once",
        action="store_true",
        help="Advanced: take existing *.autotune.json under --triton-cache-dir "
        "as a single-shot extract (no fresh benchmark).",
    )
    parser.add_argument(
        "--list-only",
        "-l",
        action="store_true",
        help="Only list *.autotune.json files under --triton-cache-dir.",
    )
    parser.add_argument(
        "--op",
        choices=available_ops(),
        default="kda",
        help=f"Op family to exercise (default: kda; available: {available_ops()}).",
    )
    args = parser.parse_args()

    modes_selected = sum([args.init_default, args.extract_once, args.list_only])
    if modes_selected != 1:
        parser.error(
            "Choose exactly one of --init-default / --extract-once / --list-only."
        )
    if args.dry_run and not args.init_default:
        parser.error("--dry-run is only valid with --init-default.")

    output_dir = _resolve_output_dir(args.output_dir)

    if args.list_only:
        triton_cache_dir = (
            Path(args.triton_cache_dir)
            if args.triton_cache_dir is not None
            else _get_default_triton_cache_dir()
        )
        list_autotune_cache_files(triton_cache_dir)
        return 0

    if args.extract_once:
        triton_cache_dir = (
            Path(args.triton_cache_dir)
            if args.triton_cache_dir is not None
            else _get_default_triton_cache_dir()
        )
        extract_once(triton_cache_dir, output_dir)
        return 0

    # --init-default
    cache_dir = (
        Path(args.triton_cache_dir)
        if args.triton_cache_dir is not None
        else _get_default_triton_cache_dir()
    )
    return init_default(
        op=args.op,
        output_dir=output_dir,
        cache_dir=cache_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())

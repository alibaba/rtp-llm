"""Require every local TP rank's cache transport before issuing PD requests."""

import argparse
import pathlib
import re


def initialized_rdma_ranks(log: str) -> set[int]:
    return {
        int(match.group(1))
        for match in re.finditer(
            r"\[RANK (\d+)\][^\n]*rdma messager init success, server port \d+,"
            r"[^\n]*rdma server port \d+",
            log,
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine_log", type=pathlib.Path)
    parser.add_argument("--ranks", type=int, required=True)
    args = parser.parse_args()
    if args.ranks < 1:
        parser.error("--ranks must be positive")
    if not args.engine_log.is_file():
        return 1
    ready = initialized_rdma_ranks(args.engine_log.read_text(errors="replace"))
    return 0 if set(range(args.ranks)).issubset(ready) else 1


if __name__ == "__main__":
    raise SystemExit(main())

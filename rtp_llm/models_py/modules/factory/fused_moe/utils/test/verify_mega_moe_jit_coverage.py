"""Opt-in coverage audit using the unmodified, pinned DeepGEMM C++ headers.

Run with --deep-gemm-source PATH and the matching backend on PYTHONPATH.
Requires CUDA headers, a C++20 compiler (CXX), and torch; no collective runs.
"""

import argparse
import json
import os
import re
import subprocess
import sysconfig
import tempfile
from pathlib import Path

import deep_gemm
from deep_gemm import mega_fp8
from torch.utils.cpp_extension import CUDA_HOME, include_paths

from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.jit_warmup import (
    DEEP_GEMM_WARMUP_REVISION,
    generate_mega_moe_jit_token_counts,
    mega_moe_config_signature,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep-gemm-source", type=Path, required=True)
    args = parser.parse_args()
    source = args.deep_gemm_source.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision.startswith(DEEP_GEMM_WARMUP_REVISION), revision
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "csrc",
            "deep_gemm/include",
        ],
        check=True,
    )
    # This template argument is outside the heuristic config in BOTH launchers.
    for impl in ("sm100_fp8_fp8_mega_moe.hpp", "sm100_fp8_fp8_mega_moe_legacy.hpp"):
        code = (source / "csrc/jit_kernels/impls" / impl).read_text()
        assert "kSinglePassDispatchMaxRoutedTokens = 32768;" in code
        assert "num_tokens * num_topk <= kSinglePassDispatchMaxRoutedTokens" in code
    with tempfile.TemporaryDirectory(prefix="mega-moe-coverage-") as temp:
        probe = Path(temp) / "probe"
        includes = [
            source,
            source / "deep_gemm/include",
            source / "third-party/deep_jit/include",
            source / "third-party/cutlass/include",
            Path(CUDA_HOME) / "include",
            Path(CUDA_HOME) / "include/cccl",
            sysconfig.get_path("include"),
            *include_paths(),
        ]
        subprocess.run(
            [
                os.environ.get("CXX", "g++"),
                "-std=c++20",
                "-O2",
                str(Path(__file__).with_name("mega_moe_config_probe.cc")),
                *[f"-I{p}" for p in includes],
                f"-L{CUDA_HOME}/lib64/stubs",
                "-lcuda",
                f"-L{CUDA_HOME}/lib64",
                "-lcudart",
                str(
                    Path(sysconfig.get_config_var("LIBDIR"))
                    / f"libpython{sysconfig.get_python_version()}.so"
                ),
                "-ldl",
                "-o",
                str(probe),
            ],
            check=True,
        )
        results = []
        for ranks, experts, topk, hidden, intermediate, cap in (
            (4, 512, 10, 4096, 1024, 32768),
            (8, 256, 6, 7168, 2048, 8192),
            (4, 32, 6, 1024, 128, 8192),
            (2, 256, 6, 16384, 2048, 8192),
        ):
            rows = subprocess.check_output(
                [
                    str(probe),
                    *map(str, (ranks, experts, topk, hidden, intermediate, cap)),
                ],
                text=True,
            ).splitlines()
            configs = [
                re.fullmatch(
                    r"(\d+) (MegaMoEConfig\(.*?\)) (MegaMoEFP8Config\(.*?\)) ([01])",
                    row,
                ).groups()
                for row in rows
            ]
            for fp8 in (False, True):
                get_block_m = (
                    (
                        lambda t: mega_fp8.get_block_m_for_mega_moe_fp8(
                            ranks, experts, 34560, t, topk
                        )
                    )
                    if fp8
                    else (
                        lambda t: deep_gemm.get_block_m_for_mega_moe(
                            ranks, experts, 34560, t, topk, "fp8xfp4"
                        )
                    )
                )
                params = dict(
                    num_ranks=ranks, num_experts=experts, num_topk=topk, fp8_weights=fp8
                )
                full = []
                grouped = {}
                for t, row in enumerate(configs):
                    config = row[2 if fp8 else 1]
                    block_m = get_block_m(t)
                    assert block_m == int(re.search(r"block_m=(\d+)", config)[1])
                    signature = mega_moe_config_signature(
                        **params, num_tokens=t, block_m=block_m
                    )
                    full_config = (config, row[3] if fp8 else "0")
                    # Every RTP bucket must map to ONE full C++ config, including
                    # stages, shared memory, dispatch/epilogue threads and SF tiles.
                    assert grouped.setdefault(signature, full_config) == full_config, (
                        params,
                        t,
                        signature,
                    )
                    full.append(full_config)
                for include_cap in (False, True):
                    reps = generate_mega_moe_jit_token_counts(
                        **params,
                        get_block_m=get_block_m,
                        max_tokens_per_rank=cap,
                        include_cap=include_cap,
                    )
                    assert {full[t] for t in reps} == set(full)
                results.append(
                    dict(
                        shape=[ranks, experts, topk, hidden, intermediate, cap],
                        fp8=fp8,
                        configurations=len(set(full)),
                        representatives=reps,
                    )
                )
        print(json.dumps(dict(revision=revision, coverage=results), indent=2))


if __name__ == "__main__":
    main()

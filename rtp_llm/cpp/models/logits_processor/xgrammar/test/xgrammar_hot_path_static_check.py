"""Static check for the XGrammar verify-only MTP async hot path.

Per design_workspace/xgrammar_verify_only_mtp_async_gpu_plan.md (Static Checks
section), the grammar-active MTP async hot-path files must not introduce new
host-blocking patterns. This test grep-scans those files for forbidden tokens
(``.cpu()``, ``.item<``, ``synchronize()``, ``accept_len_cpu``) and fails CI
when an occurrence appears outside the agreed whitelist.

Whitelist categories:
  * Lines inside ``#if 0 ... #endif`` debug blocks (not currently used; left as
    a hook in case future hot-path debug staging needs it).
  * Lines marked with the comment ``// xgrammar-hot-path-allow:`` followed by a
    short justification (worker output bookkeeping, admission restore, etc.).
"""

from __future__ import annotations

import os
import re
import unittest
from pathlib import Path
from typing import Iterable, List, Tuple


HOT_PATH_FILES = [
    "rtp_llm/cpp/models/logits_processor/xgrammar/XGrammarLogitsProcessor.cc",
    "rtp_llm/cpp/models/logits_processor/xgrammar/XGrammarLogitsProcessor.h",
    "rtp_llm/cpp/models/logits_processor/xgrammar/XGrammarCompilerCache.cc",
    "rtp_llm/cpp/models/logits_processor/xgrammar/XGrammarCompilerCache.h",
    "rtp_llm/cpp/models/logits_processor/xgrammar/XGrammarGpuState.h",
    "rtp_llm/cpp/models/logits_processor/SpecGrammarVerifyHelper.cc",
    "rtp_llm/cpp/models/logits_processor/SpecGrammarVerifyHelper.h",
]

FORBIDDEN_PATTERNS = [
    re.compile(r"\.cpu\("),
    re.compile(r"\.item<"),
    re.compile(r"synchronize\s*\("),
    re.compile(r"accept_len_cpu"),
]

ALLOW_MARKER = "xgrammar-hot-path-allow:"


def _workspace_root() -> Path:
    here = Path(__file__).resolve()
    for ancestor in [here, *here.parents]:
        if (ancestor / "WORKSPACE").exists():
            return ancestor
        runfiles_marker = ancestor / "github-opensource" / "WORKSPACE"
        if runfiles_marker.exists():
            return ancestor / "github-opensource"
    raise RuntimeError(f"could not find WORKSPACE walking up from {here}")


def _resolve(path: str) -> Path:
    candidates = [
        _workspace_root() / path,
        Path(os.environ.get("RUNFILES_DIR", "")) / "rtp_llm" / path,
        Path.cwd() / path,
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return _workspace_root() / path


def _scan(path: Path) -> List[Tuple[int, str, str]]:
    if not path.is_file():
        return []
    findings: List[Tuple[int, str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.rstrip("\n")
            if ALLOW_MARKER in line:
                continue
            for pat in FORBIDDEN_PATTERNS:
                if pat.search(line):
                    findings.append((lineno, pat.pattern, line.strip()))
                    break
    return findings


def _collect(files: Iterable[str]) -> List[str]:
    failures: List[str] = []
    for rel in files:
        path = _resolve(rel)
        for lineno, pattern, snippet in _scan(path):
            failures.append(
                f"{rel}:{lineno}: forbidden pattern {pattern!r} -> {snippet}"
            )
    return failures


class XGrammarHotPathStaticCheck(unittest.TestCase):
    def test_no_forbidden_host_sync_in_hot_path(self) -> None:
        failures = _collect(HOT_PATH_FILES)
        if failures:
            joined = "\n  ".join(failures)
            self.fail(
                "Forbidden host-sync / D2H pattern found in XGrammar hot-path "
                "files. Either remove it or mark the line with the comment "
                f"'// {ALLOW_MARKER} <why>' if it is truly required.\n  "
                + joined
            )


if __name__ == "__main__":
    unittest.main()

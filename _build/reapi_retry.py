"""Shared REAPI retry policy.

`setup.py` (Bazel build wrapper) and `rtp_llm/test/remote_tests/plugin.py`
(pytest-remote-gpu) both retry on the same set of transient REAPI exit codes.
Keeping the constant in one module prevents drift — if Bazel adds another
transient code, both consumers pick it up after a single edit here.
"""

from __future__ import annotations

import os
from typing import FrozenSet

# Bazel + REAPI exit codes that signal *transient* failures worth retrying.
# Source: bazel/src/main/java/com/google/devtools/build/lib/util/ExitCode.java
REAPI_RETRYABLE_EXIT_CODES: FrozenSet[int] = frozenset(
    {
        34,  # UNAVAILABLE — remote executor connection lost
        38,  # LOCAL_ENVIRONMENTAL_ERROR — local env issue during remote exec
    }
)


def reapi_max_retries(env_var: str = "RTP_BAZEL_MAX_RETRIES", default: int = 2) -> int:
    """Read the retry budget from env. Used by both setup.py and pytest plugin."""
    return int(os.environ.get(env_var, str(default)))

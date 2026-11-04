#!/bin/bash
set -euo pipefail

# Build the standalone C++ SDK before RTP-LLM's wheel.  The caller adds the
# generated package to Bazel with --override_repository=navi_sched_cpp=...
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${repo_root}"
exec rtp_llm/flexlb/navi_sched/scripts/build_cpp_package.sh

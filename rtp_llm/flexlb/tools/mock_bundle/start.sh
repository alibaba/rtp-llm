#!/bin/bash
set -eu
export MOCK_BUNDLE_RUN_DIR="${HIPPO_PROC_WORKDIR:-/home/admin/ai-whale}/mock-$(date +%s)-$$"
exec /opt/conda310/bin/python3 /opt/flexlb-mock/tools/mock_bundle/bundle.py

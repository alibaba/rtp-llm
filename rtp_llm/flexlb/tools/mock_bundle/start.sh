#!/bin/bash
set -euo pipefail
export MOCK_BUNDLE_RUN_DIR="${HIPPO_PROC_WORKDIR:-/home/admin/ai-whale}/mock-$(date +%s)-$$"
case "${MOCK_BUNDLE_ROLE:-}" in
  mock) entry=bundle.py ;;
  master) entry=master.py ;;
  *) echo 'release deployment requires MOCK_BUNDLE_ROLE=mock or master' >&2; exit 1 ;;
esac
exec /opt/conda310/bin/python3 "/opt/flexlb-mock/tools/mock_bundle/$entry"

#!/bin/bash
set -eu
export MOCK_BUNDLE_RUN_DIR="${HIPPO_PROC_WORKDIR:-/home/admin/ai-whale}/mock-$(date +%s)-$$"
case "${MOCK_BUNDLE_ROLE:-bundle}" in
    bundle) entry=bundle.py ;;
    mock) entry=standalone_mock.py ;;
    master) entry=standalone_master.py ;;
    *) echo "MOCK_BUNDLE_ROLE must be bundle, mock or master" >&2; exit 1 ;;
esac
exec /opt/conda310/bin/python3 "/opt/flexlb-mock/tools/mock_bundle/$entry"

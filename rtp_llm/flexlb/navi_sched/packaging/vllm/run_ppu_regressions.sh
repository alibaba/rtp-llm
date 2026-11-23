#!/usr/bin/env bash
set -euo pipefail
source_root=${VLLM_SOURCE_ROOT:?}
results_dir=${VLLM_NAVI_RESULTS_DIR:?}
integration_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Execute the extracted integration contracts against the same installed PPU
# image. These tests use the real adapter wheel and native Python modules.
export VLLM_NAVI_SCHED_ENABLE=1 PYTHONHASHSEED=0 NAVI_SCHED_VLLM_PLATFORM=ppu
.venv/bin/python "$integration_dir/verify_install.py" --expected-commit "${NAVI_SCHED_EXPECTED_COMMIT:?}"
mkdir -p tests/v1/engine tests/v1/executor tests/v1/metrics tests/packaging
cp "$integration_dir/test_navi_same_version_upgrade.py" \
  "$integration_dir/test_install_navi_sched_wheel.py" \
  "$integration_dir/install_navi_sched_wheel.sh" tests/packaging/
cp "$source_root/tests/v1/engine/test_navi_transport_contract.py" \
  "$source_root/tests/v1/engine/test_navi_core_stage.py" \
  "$source_root/tests/v1/engine/test_engine_observer.py" tests/v1/engine/
cp "$source_root/tests/v1/executor/test_navi_worker_rpc_timing.py" tests/v1/executor/
cp "$source_root/tests/v1/metrics/test_stats.py" tests/v1/metrics/
cp "$source_root/tests/entrypoints/openai/test_frontend_request_interceptor.py" tests/entrypoints/openai/
.venv/bin/python -m pytest -v -ra --noconftest --import-mode=importlib \
  --junitxml="$results_dir/navi-junit.xml" \
  tests/v1/engine tests/v1/executor tests/v1/metrics tests/packaging \
  tests/entrypoints/openai/test_frontend_request_interceptor.py
.venv/bin/python - "$results_dir/navi-junit.xml" <<'NAVI_RESULTS'
import sys
import xml.etree.ElementTree as ET
cases = list(ET.parse(sys.argv[1]).getroot().iter("testcase"))
assert cases, "No Navi regressions collected"
assert not any(case.find("skipped") is not None for case in cases), "Required Navi regressions skipped"
print(f"Verified {len(cases)} executed Navi regressions, zero skipped")
NAVI_RESULTS

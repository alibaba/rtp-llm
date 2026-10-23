#!/usr/bin/env bash
# Fixed-rate Master matrix only; the unpaced 8192-request burst is a separate test.
set -euo pipefail
cd "$(dirname "$0")/.."
output_dir="${1:-/tmp/flexlb-master-scale-$(date +%Y%m%d-%H%M%S)}"
repeats="${FLEXLB_PERF_REPEATS:-3}"
[[ "$repeats" =~ ^[1-9][0-9]*$ ]] || { echo "FLEXLB_PERF_REPEATS must be positive" >&2; exit 2; }
mkdir "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
# Prevent two invocations of this runner from contaminating each other's measurements.
# Other stress suites must also be stopped before running this latency gate.
exec 9>"${TMPDIR:-/tmp}/flexlb-master-scale-${UID}.lock"
flock -n 9 || { echo "Another Master scale regression is running" >&2; exit 2; }
{
    git rev-parse HEAD
    git status --short
    uname -a
    java -version
} > "$output_dir/environment.txt" 2>&1
source_manifest() {
    rg --files flexlb-*/src pom.xml flexlb-*/pom.xml | LC_ALL=C sort | xargs -d '\n' sha256sum
}
source_manifest > "$output_dir/source-before.sha256"
status=0
report=flexlb-api/target/surefire-reports/TEST-org.flexlb.httpserver.MasterBatchEndToEndPerformanceTest.xml
for ((round=1; round<=repeats; round++)); do
    for mode in BATCH NON_BATCH; do
        decision=FIXED_WINDOW
        if [[ "$mode" == NON_BATCH ]]; then decision=SINGLE; fi
        run_dir="$output_dir/round-$round-$mode"
        mkdir "$run_dir"
        touch "$run_dir/started"
        printf 'Running round %s/%s %s\n' "$round" "$repeats" "$mode"
        run_status=0
        ./mvnw -B -ntp -P 'opensource,!internal,api-performance-regression' \
            -pl flexlb-api -am \
            '-Dtest=MasterBatchEndToEndPerformanceTest#masterMeetsRateSloAcrossEngineScaleMatrix' \
            -Dsurefire.failIfNoSpecifiedTests=false \
            "-Dflexlb.perf.delivery-mode=$mode" \
            "-Dflexlb.perf.decision-mode=$decision" \
            test > "$run_dir/maven.log" 2>&1 || run_status=$?
        # Check fresh XML and the exact 14 executed cases, not only Maven's exit code.
        if [[ -f "$report" && "$report" -nt "$run_dir/started" ]]; then
            cp "$report" "$run_dir/report.xml"
            python3 - "$run_dir/report.xml" <<'PY' || run_status=1
import sys
import xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
cases = root.findall("testcase")
assert len(cases) == 14, f"expected 14 fixed-rate cases, found {len(cases)}"
assert len({case.attrib["name"] for case in cases}) == 14, "duplicate cases"
for case in cases:
    assert case.attrib["name"].startswith("masterMeetsRateSloAcrossEngineScaleMatrix"), case.attrib
    assert not any(case.find(tag) is not None for tag in ("failure", "error", "skipped")), case.attrib
PY
        else
            echo "Missing fresh performance test report" >&2
            run_status=1
        fi
        echo "$run_status" > "$run_dir/exit-code"
        if ((run_status != 0)); then status=1; fi
        # Continue all planned runs after failure; never replace failed runs with retries.
        awk '/FlexLB Master engine-scale E2E:|BUILD SUCCESS|BUILD FAILURE/' "$run_dir/maven.log"
    done
done
source_manifest > "$output_dir/source-after.sha256"
cmp "$output_dir/source-before.sha256" "$output_dir/source-after.sha256" || status=1
printf 'Logs: %s\n' "$output_dir"
exit "$status"

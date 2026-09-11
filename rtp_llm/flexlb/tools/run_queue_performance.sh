#!/usr/bin/env bash
# Loopback Master E2E: 750P/750D, both delivery modes, both offered rates.
set -euo pipefail
cd "$(dirname "$0")/.."
output_dir="${1:-/tmp/flexlb-queue-perf-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$output_dir"
{
    git rev-parse HEAD
    git status --short
    uname -a
    java -version
} > "$output_dir/environment.txt" 2>&1
status=0
for mode in BATCH NON_BATCH; do
    decision=FIXED_WINDOW
    if [[ "$mode" == NON_BATCH ]]; then decision=SINGLE; fi
    if ./mvnw -B -P 'opensource,!internal,api-performance-regression' \
        -pl flexlb-api -am \
        '-Dtest=MasterBatchEndToEndPerformanceTest#masterMeetsRateSloAcrossEngineScaleMatrix' \
        -DfailIfNoTests=false -Dsurefire.failIfNoSpecifiedTests=false \
        "-Dflexlb.perf.heap=${FLEXLB_PERF_HEAP:-8g}" \
        "-Dflexlb.perf.delivery-mode=$mode" \
        "-Dflexlb.perf.decision-mode=$decision" \
        -Dflexlb.perf.engine-matrix-topologies=750x750 \
        "-Dflexlb.perf.engine-matrix-target-qps=${FLEXLB_PERF_TARGET_QPS:-3000,10000}" \
        "-Dflexlb.perf.engine-matrix-duration-ms=${FLEXLB_PERF_DURATION_MS:-10000}" \
        "-Dflexlb.perf.engine-matrix-warmup-ms=${FLEXLB_PERF_WARMUP_MS:-10000}" \
        -Dflexlb.perf.engine-matrix-warmup-requests-per-prefill=16 \
        -Dflexlb.perf.engine-matrix-min-qps-ratio=0.98 \
        test > "$output_dir/$mode.log" 2>&1; then
        echo 0 > "$output_dir/$mode.exit"
    else
        echo "$?" > "$output_dir/$mode.exit"
        status=1
    fi
    # Run the other mode even when one mode fails its throughput/latency gate.
    awk '/FlexLB offered traffic:|FlexLB Master engine-scale E2E:|\[ERROR\]   Master|BUILD SUCCESS|BUILD FAILURE/' \
        "$output_dir/$mode.log"
done
printf 'Logs: %s\n' "$output_dir"
exit "$status"

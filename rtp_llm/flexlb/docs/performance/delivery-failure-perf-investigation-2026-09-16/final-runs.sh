#!/usr/bin/env bash
set -uo pipefail
export JAVA_HOME=/usr/lib/jvm/java-21-alibaba-dragonwell-21.0.11.0.11-1.1.al8.x86_64
export PATH="$JAVA_HOME/bin:$PATH"
perf_root=/data0/luoli.hn/work/rtp_llm_4/flexlb-perf-investigation-20260916
while [[ ! -f "$perf_root/reviewed-ids-2/exit" ]]; do sleep 1; done
mkdir "$perf_root/frozen-current" || exit 2
cd "$perf_root/frozen-current" || exit 2
tar xzf /tmp/flexlb-perf-frozen-20260916.tar.gz || exit 2
mkdir logs
cd source || exit 2
sha256sum -c ../source.sha256 > ../logs/source-before.log || exit 2
cd rtp_llm/flexlb || exit 2
{
 date -Ins
 java -version
 uname -a
 nproc
 cat /sys/fs/cgroup/cpu.max
} > "$perf_root/frozen-current/logs/environment.txt" 2>&1
./mvnw -B -P 'opensource,!internal' -pl flexlb-sync -am test > "$perf_root/frozen-current/logs/unit-tests.log" 2>&1
echo $? > "$perf_root/frozen-current/logs/unit-tests.exit"
mkdir "$perf_root/frozen-current/logs/unit-reports"
find . -path '*/target/surefire-reports/TEST-*.xml' -exec cp '{}' "$perf_root/frozen-current/logs/unit-reports/" \;
./mvnw -B -P 'opensource,!internal' -pl flexlb-api -am '-Dtest=DispatchFailureTest,WorkerStatusSyncTest,MultipleWorkersTest,FileDiscoveryDynamicScaleEndToEndTest,WorkerOfflineTest,GrpcTimeoutTest' -DfailIfNoTests=false -Dsurefire.failIfNoSpecifiedTests=false test > "$perf_root/frozen-current/logs/fixture-tests.log" 2>&1
echo $? > "$perf_root/frozen-current/logs/fixture-tests.exit"
cp -a "$perf_root/baseline-stub" "$perf_root/baseline-final"
for file in org/flexlb/mock/FlexLBMockTestBase.java org/flexlb/httpserver/MasterBatchEndToEndPerformanceTest.java; do
 cp "flexlb-api/src/test/java/$file" "$perf_root/baseline-final/rtp_llm/flexlb/flexlb-api/src/test/java/$file"
done
for spec in current-1 baseline-final-1 baseline-final-2 current-2 current-nonbatch; do
 mode=BATCH
 decision=FIXED_WINDOW
 if [[ "$spec" == current-nonbatch ]]; then mode=NON_BATCH; decision=SINGLE; fi
 if [[ "$spec" == current* ]]; then
  version=frozen-current/source
 else
  version=baseline-final
 fi
 run_dir="$perf_root/$spec"
 mkdir "$run_dir" || exit 2
 cd "$perf_root/$version/rtp_llm/flexlb" || exit 2
 ss -ltn > "$run_dir/listeners-before.txt"
 if awk 'NR>1 {n=split($4,a,":"); if(a[n]>=10001 && a[n]<=11499) busy=1} END{exit !busy}' "$run_dir/listeners-before.txt"; then echo 'Mock port range occupied'; exit 3; fi
 date -Ins > "$run_dir/environment.txt"
 ./mvnw -B -P 'opensource,!internal,api-performance-regression' -pl flexlb-api -am \
 '-Dtest=MasterBatchEndToEndPerformanceTest#masterMeetsRateSloAcrossEngineScaleMatrix' \
 -DfailIfNoTests=false -Dsurefire.failIfNoSpecifiedTests=false \
 -Dflexlb.perf.heap=8g "-Dflexlb.perf.delivery-mode=$mode" "-Dflexlb.perf.decision-mode=$decision" \
 -Dflexlb.perf.engine-matrix-topologies=750x750 -Dflexlb.perf.engine-matrix-target-qps=3000,10000 \
 -Dflexlb.perf.engine-matrix-duration-ms=10000 -Dflexlb.perf.engine-matrix-warmup-ms=10000 \
 -Dflexlb.perf.engine-matrix-warmup-requests-per-prefill=16 -Dflexlb.perf.engine-matrix-min-qps-ratio=0.98 \
 -Dflexlb.perf.engine-matrix-first-prefill-grpc-port=10001 test > "$run_dir/test.log" 2>&1
 echo $? > "$run_dir/exit"
 awk '/FlexLB offered traffic:|FlexLB client latency:|FlexLB Master engine-scale E2E:|\[ERROR\]   Master|BUILD SUCCESS|BUILD FAILURE/' "$run_dir/test.log" > "$run_dir/summary.txt"
 printf 'RUN %s\n' "$spec"
 cat "$run_dir/summary.txt"
done
cd "$perf_root/frozen-current/source" || exit 2
sha256sum -c ../source.sha256 > ../logs/source-after.log || exit 2
cd "$perf_root" || exit 2
tar czf final-results.tar.gz frozen-current/logs frozen-current/source-manifest.json frozen-current/source.sha256 frozen-current/workspace.diff current-1 baseline-final-1 baseline-final-2 current-2 current-nonbatch

#!/usr/bin/env bash
set -uo pipefail
perf_root=$(cd "$(dirname "$0")" && pwd)
export JAVA_HOME=/usr/lib/jvm/java-21-alibaba-dragonwell-21.0.11.0.11-1.1.al8.x86_64
export PATH="$JAVA_HOME/bin:$PATH"
mkdir -p "$perf_root/logs"
cd "$perf_root" || exit 2
tar xzf source.tar.gz || exit 2
cd source || exit 2
sha256sum -c ../source.sha256 > ../logs/source-before.log || exit 2
git init -q || exit 2
git add . || exit 2
git -c user.name='Perf Snapshot' -c user.email='perf-snapshot@localhost' commit -q -m 'Current workspace performance snapshot' || exit 2
cd rtp_llm/flexlb || exit 2
{ date -Ins; java -version; uname -a; nproc; free -h; cat /sys/fs/cgroup/cpu.max; cat /proc/loadavg; } > "$perf_root/logs/environment.txt" 2>&1
printf 'sync
' > "$perf_root/stage"
./mvnw -B -P 'opensource,!internal,sync-performance-regression' -pl flexlb-sync -am '-Dtest=WorkerBatcherPerformanceTest,PrefillAdmissionFailurePerformanceTest' -Dsurefire.failIfNoSpecifiedTests=false test > "$perf_root/logs/sync.log" 2>&1
echo $? > "$perf_root/logs/sync.exit"
mkdir -p "$perf_root/logs/sync-reports"
find . -path '*/target/surefire-reports/TEST-*.xml' -exec cp '{}' "$perf_root/logs/sync-reports/" \;
printf 'queue
' > "$perf_root/stage"
ss -ltn > "$perf_root/logs/listeners-before.txt"
if awk 'NR>1 {n=split($4,a,":"); if(a[n]>=30001 && a[n]<=31499) busy=1} END{exit !busy}' "$perf_root/logs/listeners-before.txt"; then
 echo 'Mock port range occupied' > "$perf_root/logs/queue.error"
 echo 3 > "$perf_root/logs/queue.exit"
else
 MAVEN_CONFIG=-Dflexlb.perf.engine-matrix-first-prefill-grpc-port=30001 bash tools/run_queue_performance.sh "$perf_root/logs/queue" > "$perf_root/logs/queue-summary.log" 2>&1
 echo $? > "$perf_root/logs/queue.exit"
fi
mkdir -p "$perf_root/logs/api-reports"
find flexlb-api -path '*/target/surefire-reports/TEST-*.xml' -exec cp '{}' "$perf_root/logs/api-reports/" \;
cd "$perf_root/source" || exit 2
sha256sum -c ../source.sha256 > ../logs/source-after.log
printf '%s
' "$?" > ../logs/source-after.exit
cd "$perf_root" || exit 2
tar czf results.tar.gz logs run.sh source-manifest.json source.sha256 workspace.diff
printf 'done
' > stage

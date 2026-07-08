#!/opt/homebrew/bin/bash
set -euo pipefail

# Batcher parameter optimization experiments
# Fixed: N_PREFILL=2, N_DECODE=4, only vary batcher params

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Common environment
export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
export PATH="/opt/homebrew/bin:${JAVA_HOME}/bin:${PATH}"

export N_PREFILL=2
export N_DECODE=4
export LOAD_BALANCE_STRATEGY=COST_BASED_PREFILL
export SCHEDULE_MODE=batch
export MAX_CONCURRENCY=1024
export SLA_TTFT_MS=500
export ZERO_OUTPUT_POLICY=one

# Ports to clean between experiments
PORTS="7001,7002,55150,55151,55152,55153,55154,55155,55156,55157"

cleanup_ports() {
  lsof -ti:${PORTS} 2>/dev/null | xargs kill -9 2>/dev/null || true
  sleep 5
}

run_experiment() {
  local name="$1"
  local max_inflight="$2"
  local wait_ms="$3"
  local speed="$4"

  echo ""
  echo "=============================================="
  echo "EXPERIMENT: ${name}"
  echo "  MAX_INFLIGHT_BATCHES=${max_inflight}"
  echo "  WAIT_MS=${wait_ms}"
  echo "  REPLAY_SPEED=${speed}"
  echo "=============================================="
  echo ""

  export REPLAY_SPEED="${speed}"
  export FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES="${max_inflight}"
  export FLEXLB_BATCH_FIXED_WAIT_MS="${wait_ms}"
  export RUN_ID="${name}"

  /opt/homebrew/bin/bash run_online_eval.sh 2>&1 | tee "run/${name}.log"

  echo ""
  echo "EXPERIMENT ${name} COMPLETE"
  echo "Summary at: run/${name}/load_client/summary.json"
  echo ""

  cleanup_ports
}

# Group A: Batcher concurrency slot scan (REPLAY_SPEED=50, WAIT_MS=220)
run_experiment "exp_a1_b4_w220_s50" 4 220 50
run_experiment "exp_a2_b8_w220_s50" 8 220 50

# Group B: Batch wait time scan (REPLAY_SPEED=50, MAX_INFLIGHT_BATCHES=2)
run_experiment "exp_b1_b2_w100_s50" 2 100 50
run_experiment "exp_b2_b2_w50_s50"  2 50  50

# Group C: Optimal combination (REPLAY_SPEED=50)
run_experiment "exp_c1_b4_w100_s50" 4 100 50
run_experiment "exp_c2_b8_w100_s50" 8 100 50

# Group D: Low pressure regression verification
# Using C1 config (4/100) as it's the more conservative effective combination
run_experiment "exp_d1_b4_w100_s10" 4 100 10
run_experiment "exp_d2_b4_w100_s20" 4 100 20

# Final cleanup
cleanup_ports

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=============================================="
echo ""

# Collect all summary.json paths
echo "Summary files:"
for name in exp_a1_b4_w220_s50 exp_a2_b8_w220_s50 exp_b1_b2_w100_s50 exp_b2_b2_w50_s50 exp_c1_b4_w100_s50 exp_c2_b8_w100_s50 exp_d1_b4_w100_s10 exp_d2_b4_w100_s20; do
  if [[ -f "run/${name}/load_client/summary.json" ]]; then
    echo "  run/${name}/load_client/summary.json"
  else
    echo "  MISSING: run/${name}/load_client/summary.json"
  fi
done

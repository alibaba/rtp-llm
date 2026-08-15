#!/bin/bash
# Dependency gate implementation (offline, seconds, no container). The public
# `scripts/rtpcli deps check --all` command owns the user-facing entry point.
# Run all checks first, then summarize; any failure makes the whole run FAIL.
# Online index availability is checked by the repository's published-index validation;
# this offline gate only verifies committed declarations and generated artifacts.
cd "$(dirname "$0")"
# Resolve py3: $PYTHON override first, then this project's baseline interpreter /opt/conda310/bin/python3,
# then python3 on PATH / other conda (`python3` may not be on PATH in this environment; hard-coding it would void the whole gate).
PY="${PYTHON:-}"
[ -z "$PY" ] && for c in /opt/conda310/bin/python3 /opt/conda/bin/python3; do [ -x "$c" ] && PY="$c" && break; done
[ -z "$PY" ] && PY="$(command -v python3 2>/dev/null || true)"
[ -z "$PY" ] && { echo "FAIL: python3 not found (set PYTHON=/path/to/python3)"; exit 1; }

rc=0
# All checks are independent and read-only, so run them concurrently and print
# the captured output in declaration order once each finishes.
GATE_TMP="$(mktemp -d)"
trap 'rm -rf "$GATE_TMP"' EXIT
GATE_N=0
GATE_TITLES=()
GATE_PIDS=()
run() {
  local out="$GATE_TMP/$GATE_N.out"
  GATE_TITLES+=("$1")
  "${@:2}" >"$out" 2>&1 &
  GATE_PIDS+=($!)
  GATE_N=$((GATE_N + 1))
}
summarize() {
  local idx
  for idx in "${!GATE_PIDS[@]}"; do
    wait "${GATE_PIDS[$idx]}" || rc=1
    echo "── ${GATE_TITLES[$idx]}"
    cat "$GATE_TMP/$idx.out"
  done
}

run "check locks are generated from the current requirements (catches editing requirements without recompiling locks)" \
    "$PY" check_lock_freshness.py .
run "check same-version artifact swaps: no cross-lock hash conflicts + mirrored artifacts declaring sha256 pin exactly that artifact" \
    "$PY" check_lock_artifacts.py .
run "check every //rtp_llm:<pkg> reference has a requirement() shim and shim names are unique (either breaks the whole rtp_llm/BUILD load)" \
    "$PY" check_rtp_llm_shims.py ..
run "check requirement() names in BUILD are all in the locks (catches dead references early; the locks are the only supply)" \
    "$PY" check_requirement_subset.py ..
run "check no profile's consumption sites request packages absent in that profile (guard completeness, incl. internal overlay)" \
    "$PY" check_profile_guards.py
run "check bazel/sm.bazelrc is the latest derivation of arch_config/sm_matrix.bzl (SM single source of truth)" \
    "$PY" gen_sm_bazelrc.py --check
run "self-test the flavor criterion table (rules are regexes; first prove they judge real-world shapes correctly)" \
    "$PY" test_flavor_rules.py
run "self-test that lock checkers do not go false-green on missing inputs" \
    "$PY" test_lock_checkers.py
run "check local-version packages in the locks match the arch flavor (anti-cross-flavor allowlist)" \
    "$PY" check_lock_flavor.py .
run "check the open-source dependency declaration surface has no internal hosts (machines without internal DNS can build)" \
    "$PY" check_public_hosts.py ..
run "check the open-source tree has no internal private repo/rpm names (private names live only in the internal_source payload + ppu.json)" \
    "$PY" check_private_names.py ..
run "check torch/aiter repos referenced by arch_select are all defined (prevents dangling build-time references)" \
    "$PY" check_torch_repos.py ..
run "check repo-boundary subtrees do not fall into the main repo's wildcard target surface (REPO.bazel does not make wildcards skip subtrees like WORKSPACE did)" \
    "$PY" check_repo_boundaries.py
run "check MODULE.bazel.lock covers the pip extension and all 6 hubs (Bzlmod-only)" \
    "$PY" check_lock_coverage.py ..
run "check the 6 pip.parse blocks are a faithful projection of deps.json profiles (hub/lock/index/self-supply set)" \
    "$PY" check_module_pip.py ..
run "check .bazelversion and .bazeliskrc versions agree (.bazelversion only exists so IDE/bazelisk resolve latest without network)" \
    bash -c '[ "$(cat ../.bazelversion)" = "$(sed -n "s/^USE_BAZEL_VERSION=//p" ../.bazeliskrc)" ] && echo "OK: bazel version identical in both files ($(cat ../.bazelversion))" || { echo "FAIL: .bazelversion and .bazeliskrc disagree"; exit 1; }'
run "deps.json self-consistency: schema/exceptions/provable excludes + arch_names resolvable in locks + platform_pins/kserve well-formed + absent_map.bzl" \
    "$PY" check_manifest.py
run "check tracked files have no plaintext AccessKey" \
    bash check_no_secrets.sh

summarize

if [ $rc -ne 0 ]; then
  echo "gate failed: see the fix hints of each item above"
else
  echo "gate passed: all offline checks green"
  echo "  still needs online validation: check the published pip index and run the"
  echo "                                 published-index policy check (bucket object existence -- archives carry upstream same-bytes fallback,"
  echo "                                 so a missing object cannot turn the build red and can only be caught here)"
fi
exit $rc

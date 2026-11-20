#!/usr/bin/env bash
# Run the same checked 750P/750D RPC matrix wired into the Maven test phase.
set -euo pipefail
cd "$(dirname "$0")/.."
./mvnw -B -P 'opensource,!internal' -pl flexlb-api -am \
    -Dgroups=route-performance -Dsurefire.failIfNoSpecifiedTests=false test

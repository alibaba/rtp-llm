#!/bin/sh
set -eu

[ "${FLEXLB_MOCK_WHALE:-0}" = "1" ] || {
    echo "This entry point requires FLEXLB_MOCK_WHALE=1" >&2
    exit 2
}
: "${POD_IP:?POD_IP must be the advertised engine Pod address}"
: "${START_PORT:?START_PORT must be the HTTP service port}"
: "${MOCK_PERFORMANCE_CONFIG:?Mount the mock performance configuration}"
: "${MOCK_MASTER_CONFIG:?Mount the matching master performance configuration}"
case "$START_PORT" in ''|*[!0-9]*) echo "Invalid START_PORT" >&2; exit 2;; esac
[ "$START_PORT" -gt 0 ] && [ "$START_PORT" -lt 65535 ] || exit 2
case "${ROLE_TYPE:-}" in
    PREFILL) prefill=1; decode=0;;
    DECODE) prefill=0; decode=1;;
    *) echo "ROLE_TYPE must be PREFILL or DECODE" >&2; exit 2;;
esac
case "${FETCH_OUTPUT_STREAM:-1}" in
    1) automatic=false;;
    0) automatic=true;;
    *) echo "FETCH_OUTPUT_STREAM must be 0 or 1" >&2; exit 2;;
esac
run_dir=${MOCK_RUN_DIR:-/tmp/flexlb-mock}
mkdir -p "$run_dir"
# exec preserves SIGTERM ownership: one engine JVM, no bundled master process.
exec java -jar "${MOCK_ENGINE_JAR:-/opt/flexlb/mock-engine.jar}" \
    --whale true --n-prefill "$prefill" --n-decode "$decode" \
    --host "$POD_IP" --bind-host "${MOCK_BIND_HOST:-0.0.0.0}" \
    --base-grpc-port "$((START_PORT + 1))" --auto-fetch "$automatic" \
    --kmonitor "${MOCK_KMONITOR_ENABLED:-true}" \
    --performance "$MOCK_PERFORMANCE_CONFIG" --master-config "$MOCK_MASTER_CONFIG" \
    --endpoint-file "$run_dir/unused-endpoint.json" --events-file "$run_dir/engine_events.jsonl" \
    "$@"

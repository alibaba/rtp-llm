#!/bin/sh
set -eu
. "$(dirname "$0")/mode_defaults.sh"

[ "${MOCK_RUNTIME_MODE:-$MOCK_MODE_RUNTIME}" = "$MOCK_MODE_RUNTIME" ] || {
    echo "This entry point only supports $MOCK_MODE_RUNTIME" >&2
    exit 2
}

[ "${FLEXLB_MOCK_WHALE:-0}" = "1" ] || {
    echo "This entry point requires FLEXLB_MOCK_WHALE=1" >&2
    exit 2
}
# This minimal Java image supports only the per-engine placement contract.
[ "${MOCK_DEPLOYMENT_MODE:-per_engine}" = per_engine ] || {
    echo "This image only supports MOCK_DEPLOYMENT_MODE=per_engine" >&2; exit 2
}
[ "${MOCK_BUNDLE_ROLE:-mock}" = mock ] && \
[ "${MOCK_REMOTE_DECODE:-1}" = 1 -o "${MOCK_REMOTE_DECODE:-1}" = true ] && \
[ "${MOCK_BUNDLE_FILE_DISCOVERY:-0}" = 0 ] || {
    echo "Legacy placement flags conflict with per_engine" >&2; exit 2
}
[ -z "${MOCK_PREFILL_COUNT+x}${MOCK_DECODE_COUNT+x}${MOCK_BUNDLE_OVERRIDES_YAML+x}${MOCK_BUNDLE_CONFIG_PATH+x}${MOCK_BUNDLE_LEGACY_MASTER+x}" ] || {
    echo "This image uses ROLE_TYPE; bundle configuration and count overrides are unsupported" >&2; exit 2
}
for argument in "$@"; do
    case "$argument" in
        --whale|--whale-bundle|--n-prefill|--n-decode|--host|--unique-engine-ips|--base-grpc-port|--discovery-file|--endpoint-file)
            echo "Placement argument $argument is owned by MOCK_DEPLOYMENT_MODE and ROLE_TYPE" >&2; exit 2;;
    esac
done
run_dir=${MOCK_RUN_DIR:-/tmp/flexlb-mock}
mkdir -p "$run_dir"
# Whale splits cmd on whitespace; keep bootstrap logic in this file, not sh -c.
if [ -z "${POD_IP:-}" ]; then
    POD_IP=$(hostname -i | awk '{print $1}')
fi
: "${POD_IP:?POD_IP must be the advertised engine Pod address}"
: "${START_PORT:?START_PORT must be the HTTP service port}"
if [ -n "${MOCK_PERFORMANCE_CONFIG_JSON:-}" ]; then
    [ -z "${MOCK_PERFORMANCE_CONFIG:-}" ] || { echo "Choose performance path or JSON, not both" >&2; exit 2; }
    MOCK_PERFORMANCE_CONFIG="$run_dir/performance.json"
    printf '%s' "$MOCK_PERFORMANCE_CONFIG_JSON" > "$MOCK_PERFORMANCE_CONFIG"
fi
if [ -n "${MOCK_MASTER_CONFIG_JSON:-}" ]; then
    [ -z "${MOCK_MASTER_CONFIG:-}" ] || { echo "Choose master path or JSON, not both" >&2; exit 2; }
    MOCK_MASTER_CONFIG="$run_dir/master.json"
    printf '%s' "$MOCK_MASTER_CONFIG_JSON" > "$MOCK_MASTER_CONFIG"
fi
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
case "${MOCK_EVENT_LOG_ENABLED:-$MOCK_MODE_JSONL_DEFAULT}" in
    1) set -- --events-file "$run_dir/engine_events.jsonl" "$@";;
    0) ;;
    *) echo "MOCK_EVENT_LOG_ENABLED must be 0 or 1" >&2; exit 2;;
esac
if ! command -v java >/dev/null 2>&1 && [ -x /opt/taobao/java/bin/java ]; then
    PATH="/opt/taobao/java/bin:$PATH"
    export PATH
fi
printf '{"mode":"per_engine","entrypoint":"mock","master_discovery":"vipserver","remote_decode":true,"colocated_master":false,"role":"%s"}\n' "$ROLE_TYPE" > "$run_dir/deployment.json"
cat "$run_dir/deployment.json"
# exec preserves SIGTERM ownership: one engine JVM, no bundled master process.
exec java -jar "${MOCK_ENGINE_JAR:-/opt/flexlb/mock-engine.jar}" \
    --whale true --n-prefill "$prefill" --n-decode "$decode" \
    --host "$POD_IP" --bind-host "${MOCK_BIND_HOST:-0.0.0.0}" \
    --unique-engine-ips false \
    --base-grpc-port "$((START_PORT + 1))" --auto-fetch "$automatic" \
    --kmonitor "${MOCK_KMONITOR_ENABLED:-$MOCK_MODE_KMONITOR_DEFAULT}" \
    --performance "$MOCK_PERFORMANCE_CONFIG" --master-config "$MOCK_MASTER_CONFIG" \
    --endpoint-file "$run_dir/unused-endpoint.json" \
    "$@"

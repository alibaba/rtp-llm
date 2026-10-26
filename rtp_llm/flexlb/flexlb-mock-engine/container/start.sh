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
[ -z "${MOCK_PREFILL_COUNT+x}${MOCK_DECODE_COUNT+x}${MOCK_VIT_COUNT+x}${MOCK_BUNDLE_OVERRIDES_YAML+x}${MOCK_BUNDLE_CONFIG_PATH+x}${MOCK_BUNDLE_LEGACY_MASTER+x}" ] || {
    echo "This image uses ROLE_TYPE; bundle configuration and count overrides are unsupported" >&2; exit 2
}
for argument in "$@"; do
    case "$argument" in
        --mock.whale|--mock.whale_bundle|--mock.n_prefill|--mock.n_decode|--mock.n_vit|--mock.host|--mock.unique_engine_ips|--mock.base_grpc_port|--mock.discovery_file|--mock.endpoint_file|--mock.engine_config|--mock.calibration)
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
[ -z "${MOCK_PERFORMANCE_CONFIG+x}${MOCK_PERFORMANCE_CONFIG_JSON+x}${MOCK_MASTER_CONFIG+x}${MOCK_MASTER_CONFIG_JSON+x}" ] || {
    echo "Removed performance/master inputs; use MOCK_ENGINE_CONFIG and MOCK_CALIBRATION_CONFIG" >&2; exit 2
}
if [ -n "${MOCK_ENGINE_CONFIG_JSON:-}" ]; then
    [ -z "${MOCK_ENGINE_CONFIG:-}" ] || { echo "Choose engine config path or JSON, not both" >&2; exit 2; }
    MOCK_ENGINE_CONFIG="$run_dir/engine-config.json"
    printf '%s' "$MOCK_ENGINE_CONFIG_JSON" > "$MOCK_ENGINE_CONFIG"
fi
if [ -n "${MOCK_CALIBRATION_CONFIG_JSON:-}" ]; then
    [ -z "${MOCK_CALIBRATION_CONFIG:-}" ] || { echo "Choose calibration path or JSON, not both" >&2; exit 2; }
    MOCK_CALIBRATION_CONFIG="$run_dir/calibration.json"
    printf '%s' "$MOCK_CALIBRATION_CONFIG_JSON" > "$MOCK_CALIBRATION_CONFIG"
fi
: "${MOCK_ENGINE_CONFIG:?Mount the effective engine configuration and allocator capacity}"
: "${MOCK_CALIBRATION_CONFIG:?Mount the simulation calibration}"
case "$START_PORT" in ''|*[!0-9]*) echo "Invalid START_PORT" >&2; exit 2;; esac
[ "$START_PORT" -gt 0 ] && [ "$START_PORT" -lt 65535 ] || exit 2
case "${ROLE_TYPE:-}" in
    PREFILL) prefill=1; decode=0; vit=0;;
    DECODE) prefill=0; decode=1; vit=0;;
    VIT) prefill=0; decode=0; vit=1;;
    *) echo "ROLE_TYPE must be PREFILL, DECODE or VIT" >&2; exit 2;;
esac
case "${FETCH_OUTPUT_STREAM:-1}" in
    1) automatic=false;;
    0) automatic=true;;
    *) echo "FETCH_OUTPUT_STREAM must be 0 or 1" >&2; exit 2;;
esac
case "${MOCK_EVENT_LOG_ENABLED:-$MOCK_MODE_JSONL_DEFAULT}" in
    1) set -- --mock.events_file "$run_dir/engine_events.jsonl" "$@";;
    0) ;;
    *) echo "MOCK_EVENT_LOG_ENABLED must be 0 or 1" >&2; exit 2;;
esac
if [ -n "${MOCK_VIT_MODEL_PATH:-}" ]; then
    set -- --mock.vit_model_path "$MOCK_VIT_MODEL_PATH" "$@"
fi
if ! command -v java >/dev/null 2>&1 && [ -x /opt/taobao/java/bin/java ]; then
    PATH="/opt/taobao/java/bin:$PATH"
    export PATH
fi
printf '{"mode":"per_engine","entrypoint":"mock","master_discovery":"vipserver","remote_decode":true,"colocated_master":false,"role":"%s"}\n' "$ROLE_TYPE" > "$run_dir/deployment.json"
cat "$run_dir/deployment.json"
# exec preserves SIGTERM ownership: one engine JVM, no bundled master process.
exec java -jar "${MOCK_ENGINE_JAR:-/opt/flexlb/mock-engine.jar}" \
    --mock.whale true --mock.n_prefill "$prefill" --mock.n_decode "$decode" --mock.n_vit "$vit" \
    --mock.host "$POD_IP" --mock.bind_host "${MOCK_BIND_HOST:-0.0.0.0}" \
    --mock.unique_engine_ips false \
    --mock.base_grpc_port "$((START_PORT + 1))" --mock.auto_fetch "$automatic" \
    --mock.kmonitor "${MOCK_KMONITOR_ENABLED:-$MOCK_MODE_KMONITOR_DEFAULT}" \
    --mock.engine_config "$MOCK_ENGINE_CONFIG" --mock.calibration "$MOCK_CALIBRATION_CONFIG" \
    --mock.endpoint_file "$run_dir/unused-endpoint.json" \
    "$@"

#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} != --inside ]]; then
    if [[ $EUID == 0 || $# == 0 ]]; then
        printf 'Run the PACE namespace wrapper as a non-root Bazel test user.\n' >&2
        exit 2
    fi
    runfiles=${TEST_SRCDIR:?run this wrapper through bazel test}/${TEST_WORKSPACE:?}
    if [[ $1 != --locked ]]; then
        exec "$runfiles/tools/nvidia_wheel_env" "$0" --locked "$@"
    fi
    shift
    output_dir=$(realpath -m -- "${TEST_UNDECLARED_OUTPUTS_DIR:?}/pace")
    mkdir -p -- "$output_dir"
    work_dir=$(mktemp -d /tmp/rtp-pace.XXXXXX)
    trap 'rm -rf -- "$work_dir"' EXIT
    bin_dir=$(realpath -e -- "${PACE_SMOKE_BIN_DIR:-$runfiles/3rdparty/pace_test_runtime}")
    python_path=$(command -v python3)
    [[ $python_path == /* && -x $python_path ]]
    scripts="$runfiles/rtp_llm/test/smoke/scripts"
    transport=${PACE_SMOKE_TRANSPORT:-tcp}
    faults=${PACE_SMOKE_FAULTS:-0}
    rdma_device=${PACE_SMOKE_RDMA_DEVICE:-}
    if [[ $transport != tcp && $rdma_device == auto ]]; then
        status=0
        rdma_device=$("$python_path" "$scripts/start_pace_smoke.py" check-rdma --transport "$transport" \
            --rdma-device auto --rdma-port "${PACE_SMOKE_RDMA_PORT:-1}" 2>&1) || status=$?
        if [[ $status == 77 ]]; then
            printf 'SKIP: %s: %s\n' "$transport" "$rdma_device" | tee "$output_dir/skip.txt"
            exit 0
        elif [[ $status != 0 ]]; then
            printf '%s\n' "$rdma_device" >&2
            exit "$status"
        fi
        printf 'PACE %s using %s port %s\n' "$transport" "$rdma_device" "${PACE_SMOKE_RDMA_PORT:-1}"
    fi
    if [[ ( $transport == aft || $faults == provider-offline ) && -z ${CHECKPOINT_PATH:-} ]]; then
        cache_dir="${PACE_SMOKE_CACHE_DIR:-${TEST_TMPDIR:?}/pace-smoke-model}"
        if ! CHECKPOINT_PATH=$(timeout --kill-after=5s 300s "$python_path" "$scripts/prepare_pace_smoke.py" \
            "$cache_dir" --model); then
            printf 'Could not prepare the pinned PACE model on this worker within 300 seconds.\n' >&2
            exit 1
        fi
        printf 'Prepared pinned PACE model: %s\n' "$CHECKPOINT_PATH"
    fi
    test_groups=$(id -G)
    run_env=()
    for name in PATH HOME USER LOGNAME LANG LC_ALL TMPDIR LD_LIBRARY_PATH LD_PRELOAD \
        PYTHONPATH PYTHONNOUSERSITE PYTHON_RUNFILES TEST_SRCDIR TEST_WORKSPACE TEST_TMPDIR \
        TEST_UNDECLARED_OUTPUTS_DIR TEST_TARGET TEST_TIMEOUT RUNFILES_DIR RUNFILES_MANIFEST_FILE XML_OUTPUT_FILE \
        GPU_COUNT WORLD_SIZE SMOKE_ARGS CUDA_HOME CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER CHECKPOINT_PATH \
        CC CXX NVCC_PREPEND_FLAGS DG_JIT_CPP_STANDARD DG_JIT_CACHE_DIR TRITON_PTXAS_PATH TRITON_CACHE_DIR \
        FLASHINFER_WORKSPACE_BASE RAYON_NUM_THREADS REMOTE_JIT_DIR _JIT_CACHE_PATHS; do
        if [[ ${!name+x} ]]; then run_env+=("$name=${!name}"); fi
    done
    if ! sudo_path=$(command -p -v sudo || command -v sudo); then
        printf 'PACE namespace setup needs noninteractive sudo.\n' >&2
        exit 2
    fi
    "$sudo_path" -n -- unshare --net --mount --ipc --pid --fork --kill-child --mount-proc --propagation private \
        -- "$BASH" "$(realpath -e -- "$0")" --inside \
        "$EUID" "$(id -g)" "$(pwd -P)" "$work_dir" "$output_dir" "$runfiles" "$bin_dir" "$python_path" \
        "$(readlink /proc/self/ns/net)" "$(readlink /proc/self/ns/mnt)" \
        "$transport" "$faults" "$rdma_device" \
        "${PACE_SMOKE_RDMA_PORT:-1}" "${PACE_SMOKE_GID_INDEX:-0}" "${test_groups// /,}" \
        "${#run_env[@]}" "${run_env[@]}" "$@"
    exit
fi

shift
test_uid=$1 test_gid=$2 test_cwd=$3 work_dir=$4 output_dir=$5 runfiles=$6 bin_dir=$7 python_path=$8
parent_net=$9 parent_mount=${10} transport=${11} faults=${12} rdma_device=${13} rdma_port=${14} gid_index=${15}
test_groups=${16} env_count=${17}
shift 17
if [[ $EUID != 0 || $$ != 1 || $test_uid == 0 || -z $parent_net || -z $parent_mount ||
    $(readlink /proc/self/ns/net) == "$parent_net" || $(readlink /proc/self/ns/mnt) == "$parent_mount" ]]; then
    printf 'PACE setup requires a fresh private network, mount and PID namespace.\n' >&2
    exit 2
fi
as_user=(setpriv --reuid "$test_uid" --regid "$test_gid" --groups "$test_groups" --no-new-privs --
         env -i "${@:1:env_count}")
shift "$env_count"
umask 022
trap 'exit 129' HUP
trap 'exit 143' TERM
trap 'exit 130' INT
cd -- "$test_cwd"

launcher="$runfiles/rtp_llm/test/smoke/scripts/start_pace_smoke.py"
fault_helper="$runfiles/rtp_llm/test/smoke/scripts/pace_smoke_faults.py"
consumer_bin_dir="$bin_dir/consumer"
if [[ ! -d $consumer_bin_dir ]]; then consumer_bin_dir=$bin_dir; fi
if [[ ! -x $bin_dir/tair_mempool_meta_service || ! -x $bin_dir/tair_mempool_server ||
    ! -x $consumer_bin_dir/tair_mempool_server ]]; then
    printf 'Incomplete PACE binaries in %s\n' "$bin_dir" >&2
    exit 2
fi
transport_args=(--transport "$transport")
case "$transport" in
    tcp) ;;
    aft|aft-rdma|aft-gdr|rdma|gdr)
        if [[ -z $rdma_device ]]; then printf 'Set PACE_SMOKE_RDMA_DEVICE for %s.\n' "$transport" >&2; exit 2; fi
        if [[ $transport == aft* && ( $rdma_port != 1 || $gid_index != 0 ) ]]; then
            printf 'Custom RDMA port and GID index are only supported by TENT.\n' >&2
            exit 2
        fi
        transport_args+=(--rdma-device "$rdma_device" --rdma-port "$rdma_port" --gid-index "$gid_index")
        if ! ulimit -l unlimited; then
            printf 'PACE RDMA needs an unlimited memlock limit in the test namespace.\n' >&2
            exit 2
        fi
        "${as_user[@]}" "$python_path" "$launcher" check-rdma "${transport_args[@]}"
        ;;
    *) printf 'Unknown PACE transport: %s\n' "$transport" >&2; exit 2 ;;
esac
case "$faults:$transport" in
    0:*|1:aft|provider-offline:tcp) ;;
    1:tcp)
        tc_binary=$(command -v tc || true)
        if [[ ! -x $tc_binary ]] || ! "$tc_binary" -V >/dev/null 2>&1; then tc_binary="$bin_dir/tools/tc"; fi
        if ! "$tc_binary" -V; then printf 'PACE TCP faults need a working tc.\n' >&2; exit 2; fi
        ;;
    *) printf 'PACE_SMOKE_FAULTS must be 0, 1 with tcp or aft, or provider-offline with tcp.\n' >&2; exit 2 ;;
esac

if [[ $transport == tcp && -d /dev/infiniband ]]; then
    mount -t tmpfs -o mode=0755,nosuid,nodev,noexec tmpfs /dev/infiniband
elif [[ $transport == aft* ]]; then
    verbs_device=
    for node in /sys/class/infiniband_verbs/uverbs*; do
        if [[ -r $node/ibdev && $(< "$node/ibdev") == "$rdma_device" &&
            -c /dev/infiniband/${node##*/} ]]; then
            verbs_device=${node##*/}
            break
        fi
    done
    if [[ -z $verbs_device ]]; then printf 'No uverbs device for %s.\n' "$rdma_device" >&2; exit 2; fi
    "${as_user[@]}" mkdir -- "$work_dir/verbs"
    "${as_user[@]}" touch -- "$work_dir/verbs/$verbs_device"
    mount --bind "/dev/infiniband/$verbs_device" "$work_dir/verbs/$verbs_device"
    mount --rbind "$work_dir/verbs" /dev/infiniband
fi
mount -t tmpfs -o size=6g,mode=1777,nosuid,nodev tmpfs /dev/shm
"${as_user[@]}" mkdir -- "$work_dir/meta-state"
mount --bind "$work_dir/meta-state" /var/log
unshare --net --mount --ipc --propagation private -- "$BASH" -ec '
    mount -t tmpfs -o size=4g,mode=1777,nosuid,nodev tmpfs /dev/shm
    touch "$1"
    exec sleep infinity
' bash "$work_dir/provider.ready" > "$output_dir/provider-namespace.log" 2>&1 &
provider_namespace_pid=$!
for attempt in {1..60}; do
    [[ ! -f $work_dir/provider.ready ]] || break
    kill -0 "$provider_namespace_pid" || break
    sleep 0.1
done
if [[ ! -f $work_dir/provider.ready ]]; then
    printf 'Provider namespace did not start. See %s/provider-namespace.log.\n' "$output_dir" >&2
    exit 1
fi

consumer_ip=172.31.250.1
provider_ip=172.31.250.2
provider_net=(nsenter --target "$provider_namespace_pid" --net)
provider_scope=("${provider_net[@]}" --mount --ipc --)
ip link add pace-consumer type veth peer name pace-provider
ip link set pace-provider netns "$provider_namespace_pid"
ip addr add "$consumer_ip/30" dev pace-consumer
ip link set lo up
ip link set pace-consumer up
"${provider_net[@]}" ip addr add "$provider_ip/30" dev pace-provider
"${provider_net[@]}" ip link set lo up
"${provider_net[@]}" ip link set pace-provider up
printf '127.0.0.1 localhost\n::1 localhost ip6-localhost ip6-loopback\n%s %s\n' \
    "$consumer_ip" "$(uname -n)" > "$work_dir/hosts"
mount --bind "$work_dir/hosts" /etc/hosts
printf 'PACE %s namespace: Consumer/RTP %s, Provider %s. Logs: %s\n' "$transport" "$consumer_ip" "$provider_ip" "$output_dir"

wait_http() {
    local pid=$1 url=$2 output=$3 service_log=$4 attempt
    for attempt in {1..60}; do
        kill -0 "$pid" 2>/dev/null || break
        if "${as_user[@]}" curl --noproxy '*' -fsS --max-time 1 "$url" > "$output" 2>/dev/null; then return 0; fi
        sleep 0.5
    done
    printf 'PACE endpoint did not become ready: %s. See %s.\n' "$url" "$service_log" >&2
    tail -n 20 -- "$service_log" >&2
    return 1
}

meta_port=24910
data_port=24914
meta_url="http://$consumer_ip:$meta_port"
provider_status_url="http://$provider_ip:$((data_port + 2))/api/status"
metrics_url="http://127.0.0.1:$((data_port + 3))/pace/metrics"
"${as_user[@]}" "$python_path" "$launcher" meta --bin-dir "$bin_dir" --work-dir "$work_dir/meta" \
    --port-base "$meta_port" --transport "$transport" > "$output_dir/meta.log" 2>&1 &
wait_http "$!" "$meta_url/v1/api/healthy" "$output_dir/meta-health.json" "$output_dir/meta.log"
if ! "${provider_scope[@]}" "${as_user[@]}" curl --noproxy '*' -fsS --max-time 5 "$meta_url/v1/api/healthy" \
    > "$output_dir/provider-meta-health.json"; then
    printf 'Provider cannot reach Meta over the test veth pair (%s -> %s).\n' "$provider_ip" "$consumer_ip" >&2
    ip -s link show dev pace-consumer >&2 || true
    "${provider_net[@]}" ip -s link show dev pace-provider >&2 || true
    exit 1
fi
"${provider_scope[@]}" "${as_user[@]}" "$python_path" "$launcher" provider \
    --bin-dir "$bin_dir" --work-dir "$work_dir/provider" --port-base "$data_port" --meta-url "$meta_url" \
    --interface pace-provider "${transport_args[@]}" > "$output_dir/provider.log" 2>&1 &
provider_pid=$!
wait_http "$provider_pid" "$provider_status_url" "$output_dir/provider-status.json" "$output_dir/provider.log"

consumer_args=()
if [[ $transport == aft || $faults == provider-offline ]]; then
    offline_dir="$output_dir/offline"
    "${as_user[@]}" mkdir -- "$offline_dir"
    consumer_args=(--provider-offline-dir "$offline_dir")
elif [[ $transport == aft-rdma ]]; then
    consumer_args=(--provider-metrics-url "http://$provider_ip:$((data_port + 3))/pace/metrics")
elif [[ $faults == 1 ]]; then
    fault_socket="$work_dir/fault.sock"
    consumer_args=(--fault-socket "$fault_socket")
fi
"${as_user[@]}" "$python_path" "$launcher" consumer --bin-dir "$consumer_bin_dir" \
    --work-dir "$work_dir/consumer" --port-base "$data_port" --meta-url "$meta_url" \
    --provider-status-url "$provider_status_url" --interface pace-consumer \
    "${transport_args[@]}" "${consumer_args[@]}" > "$output_dir/consumer.log" 2>&1 &
wait_http "$!" "$metrics_url" "$output_dir/metrics-before.txt" "$output_dir/consumer.log"

if [[ $transport == tcp && $faults == 1 ]]; then
    "$python_path" -I -u "$fault_helper" "$fault_socket" "$test_uid" "$test_gid" \
        pace-consumer "$consumer_ip" "$provider_ip" "$((data_port + 1))" "$tc_binary" > "$output_dir/faults.log" 2>&1 &
    fault_pid=$!
    for attempt in {1..60}; do
        [[ ! -S $fault_socket ]] || break
        kill -0 "$fault_pid" || break
        sleep 0.1
    done
    if [[ ! -S $fault_socket ]]; then printf 'PACE fault control did not start. See %s/faults.log.\n' "$output_dir" >&2; exit 1; fi
elif [[ $transport == aft || $faults == provider-offline ]]; then
    "${as_user[@]}" "$BASH" -ec '
        provider_pid=$1 offline_dir=$2
        until [[ -f $offline_dir/pause.request ]]; do sleep 0.025; done
        [[ $(readlink "/proc/$provider_pid/exe") == "$(realpath -e -- "$3")" ]]
        offline_us=$(date +%s%6N)
        kill -STOP "$provider_pid"
        printf "%s\n" "$offline_us" > "$offline_dir/paused.next"
        mv -- "$offline_dir/paused.next" "$offline_dir/paused"
        deadline=$((SECONDS + 2))
        until [[ -f $offline_dir/kill.request ]] || (( SECONDS >= deadline )); do sleep 0.025; done
        kill -KILL "$provider_pid"
        date +%s%6N > "$offline_dir/killed.next"
        mv -- "$offline_dir/killed.next" "$offline_dir/killed"
    ' bash "$provider_pid" "$offline_dir" "$bin_dir/tair_mempool_server" > "$output_dir/faults.log" 2>&1 &
fi

fixture="$work_dir/consumer/fixture.json"
cp -- "$fixture" "$output_dir/fixture.json"
result=0
"${as_user[@]}" "KVCM_PACE_FIXTURE=$fixture" \
    "NO_PROXY=127.0.0.1,$consumer_ip,$provider_ip" "no_proxy=127.0.0.1,$consumer_ip,$provider_ip" \
    "$@" || result=$?
"${as_user[@]}" curl --noproxy '*' -fsS --max-time 5 "$metrics_url" > "$output_dir/metrics-after.txt" || true
exit "$result"

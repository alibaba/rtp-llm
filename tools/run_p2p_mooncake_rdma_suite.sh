#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-all}"

BRANCH="${BRANCH:-develop/vin/p2p-connector-3}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/data0/qiongshi.gb/RTP-LLM}"
OPEN_SOURCE_REPO="${OPEN_SOURCE_REPO:-${WORKSPACE_ROOT}/github-opensource}"
INTERNAL_SOURCE_REPO="${INTERNAL_SOURCE_REPO:-${WORKSPACE_ROOT}/internal_source}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/nas1}"
BAZEL_DEPS_ROOT="${BAZEL_DEPS_ROOT:-/data0/qiongshi.gb/bazel_deps}"
BAZEL_CACHE_DIR="${BAZEL_CACHE_DIR:-/data0/qiongshi.gb/.bazel_cache}"
CONTAINER_NAME="${CONTAINER_NAME:-vin_rtp_rdma_test}"
IMAGE="${IMAGE:-hub.docker.alibaba-inc.com/isearch/rtp_llm_dev_gpu_cuda12_9:2025_12_04_19_49_005d702}"
GPU_CONFIG="${GPU_CONFIG:-sm9x}"
RUN_TCP_REFERENCE="${RUN_TCP_REFERENCE:-1}"
RUN_RDMA_SMOKE="${RUN_RDMA_SMOKE:-1}"
RESTORE_INTERNAL_SOURCE="${RESTORE_INTERNAL_SOURCE:-1}"

TCP_PATCH_FILE="${OPEN_SOURCE_REPO}/docs/backend/p2p-mooncake-tcp-smoke.patch"
BUILD_FILE="${INTERNAL_SOURCE_REPO}/rtp_llm/test/smoke/BUILD"
REMOTE_REUSE_ARTIFACT="${INTERNAL_SOURCE_REPO}/rtp_llm/test/smoke/data/model/qwen25/q_r_l20_remote_cache_pd_sep.query_1.json"
TMP_DIR="$(mktemp -d /tmp/p2p_mooncake_rdma_suite.XXXXXX)"
BUILD_BACKUP="${TMP_DIR}/BUILD.backup"
APPLIED_SMOKE_BUILD=0

declare -a BACKED_UP_ARTIFACTS=()

declare -a BAZEL_OVERRIDE_ARGS=(
  "--override_repository=havenask=${BAZEL_DEPS_ROOT}/havenask_3c973500afbd40933eb0a80cfdfb6592274377fb"
  "--override_repository=com_google_absl=${BAZEL_DEPS_ROOT}/com_google_absl_6f9d96a1f41439ac172ee2ef7ccd8edf0e5d068c"
  "--override_repository=cutlass_fa=${BAZEL_DEPS_ROOT}/cutlass_fa_bbe579a9e3beb6ea6626d9227ec32d0dae119a49"
  "--override_repository=cutlass=${BAZEL_DEPS_ROOT}/cutlass_80243e0b8c644f281e2beb0c20fe78cf7b267061"
  "--override_repository=cutlass_h_moe=${BAZEL_DEPS_ROOT}/cutlass_h_moe_19b4c5e065e7e5bbc8082dfc7dbd792bdac850fc"
  "--override_repository=cutlass4.0=${BAZEL_DEPS_ROOT}/cutlass4_0_dc4817921edda44a549197ff3a9dcf5df0636e7b"
  "--override_repository=cutlass3.6=${BAZEL_DEPS_ROOT}/cutlass3_6_cc3c29a81a140f7b97045718fb88eb0664c37bd7"
  "--override_repository=rules_cc=${BAZEL_DEPS_ROOT}/rules_cc_from_devcache"
  "--override_repository=rules_python=${BAZEL_DEPS_ROOT}/rules_python_084b877c98b580839ceab2b071b02fc6768f3de6_patched"
  "--override_repository=flashinfer_cpp=${BAZEL_DEPS_ROOT}/flashinfer_cpp_1c88d650eeec97be3a4dcebe4a9912d7785bc250_patched"
  "--override_repository=flash_attention=${BAZEL_DEPS_ROOT}/flash_attention_6c9e60de566800538fedad2ad5e6b7b55ca7f0c5_patched"
  "--override_repository=rapidjson=${BAZEL_DEPS_ROOT}/rapidjson_f54b0e47a08782a6131cc3d60f94d038fa6e0a51_patched"
  "--override_repository=grpc=${BAZEL_DEPS_ROOT}/grpc_109c570727c3089fef655edcdd0dd02cc5958010_patched"
)

log() {
  printf '[mooncake-rdma-suite] %s\n' "$*"
}

die() {
  printf '[mooncake-rdma-suite] ERROR: %s\n' "$*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "缺少命令: $1"
}

backup_file_if_exists() {
  local file="$1"
  if [[ -f "$file" ]]; then
    local backup="${TMP_DIR}/$(basename "$file").$((${#BACKED_UP_ARTIFACTS[@]} + 1)).bak"
    cp "$file" "$backup"
    BACKED_UP_ARTIFACTS+=("$file:$backup")
  else
    BACKED_UP_ARTIFACTS+=("$file:")
  fi
}

restore_artifacts() {
  local pair file backup
  for pair in "${BACKED_UP_ARTIFACTS[@]}"; do
    file="${pair%%:*}"
    backup="${pair#*:}"
    if [[ -n "$backup" && -f "$backup" ]]; then
      cp "$backup" "$file"
    else
      rm -f "$file"
    fi
  done
}

cleanup() {
  if [[ "$RESTORE_INTERNAL_SOURCE" == "1" && "$APPLIED_SMOKE_BUILD" == "1" && -f "$BUILD_BACKUP" ]]; then
    cp "$BUILD_BACKUP" "$BUILD_FILE"
    restore_artifacts || true
  fi
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

host_precheck() {
  need_cmd git
  need_cmd docker
  need_cmd python3
  [[ -d "$OPEN_SOURCE_REPO" ]] || die "不存在开源仓目录: $OPEN_SOURCE_REPO"
  [[ -d "$INTERNAL_SOURCE_REPO" ]] || die "不存在 internal_source 目录: $INTERNAL_SOURCE_REPO"
  [[ -d "$MODEL_ROOT" ]] || die "不存在模型挂载目录: $MODEL_ROOT"
  mkdir -p "$BAZEL_CACHE_DIR"
}

checkout_branch() {
  log "切换到分支 $BRANCH"
  git -C "$OPEN_SOURCE_REPO" fetch origin "$BRANCH"
  git -C "$OPEN_SOURCE_REPO" checkout "$BRANCH"
  git -C "$OPEN_SOURCE_REPO" pull --ff-only origin "$BRANCH"
  log "当前提交: $(git -C "$OPEN_SOURCE_REPO" rev-parse HEAD)"
}

start_container() {
  log "启动容器 $CONTAINER_NAME"
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker run -d --name "$CONTAINER_NAME" \
    --privileged \
    --network host \
    --ipc host \
    --gpus all \
    --ulimit memlock=-1 \
    --ulimit nofile=655350:655350 \
    -v /data0/qiongshi.gb:/data0/qiongshi.gb \
    -v "$MODEL_ROOT":"$MODEL_ROOT" \
    -v /dev/infiniband:/dev/infiniband \
    -v /sys/class/infiniband:/sys/class/infiniband \
    -v /sys/class/net:/sys/class/net \
    -v "$BAZEL_CACHE_DIR":/root/.cache/bazel \
    -w "$OPEN_SOURCE_REPO" \
    "$IMAGE" \
    bash -lc 'sleep infinity' >/dev/null
}

container_precheck() {
  docker exec "$CONTAINER_NAME" bash -lc "git config --global --add safe.directory '$OPEN_SOURCE_REPO'"
  log "容器内 GPU / RDMA 状态"
  docker exec "$CONTAINER_NAME" bash -lc 'nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -n 8 || true'
  docker exec "$CONTAINER_NAME" bash -lc 'ls -l /dev/infiniband || true'
  docker exec "$CONTAINER_NAME" bash -lc 'rdma link show || true'
}

docker_bazel_test() {
  local target="$1"
  shift
  log "运行测试: $target"
  docker exec -w "$OPEN_SOURCE_REPO" "$CONTAINER_NAME" \
    bazelisk test "$target" \
    --cache_test_results=no \
    --test_output=errors \
    --config=cuda12_9 \
    "--config=${GPU_CONFIG}" \
    "${BAZEL_OVERRIDE_ARGS[@]}" \
    "$@"
}

ensure_build_backup() {
  if [[ ! -f "$BUILD_BACKUP" ]]; then
    cp "$BUILD_FILE" "$BUILD_BACKUP"
    backup_file_if_exists "${INTERNAL_SOURCE_REPO}/rtp_llm/test/smoke/data/model/qwen25/q_r_prefill_and_decode_reuse.query_0.json"
    backup_file_if_exists "${INTERNAL_SOURCE_REPO}/rtp_llm/test/smoke/data/model/qwen25/q_r_prefill_and_decode_reuse.query_1.json"
    backup_file_if_exists "${INTERNAL_SOURCE_REPO}/rtp_llm/test/smoke/data/model/qwen25/q_r_prefill_and_decode_reuse.query_2.json"
    backup_file_if_exists "${INTERNAL_SOURCE_REPO}/rtp_llm/test/smoke/data/model/qwen25/q_r_l20_remote_cache_pd_sep.query_0.json"
    backup_file_if_exists "${INTERNAL_SOURCE_REPO}/rtp_llm/test/smoke/data/model/qwen25/q_r_l20_remote_cache_pd_sep.query_1.json"
  fi
}

apply_tcp_smoke_patch() {
  ensure_build_backup
  if grep -q 'qwen25_05b_base_openai_remote_cache_pd_sep_mooncake_tcp' "$BUILD_FILE"; then
    log "TCP smoke target 已存在，跳过 patch"
    return
  fi
  log "应用 Mooncake TCP smoke patch"
  (cd "$WORKSPACE_ROOT" && git apply --check "$TCP_PATCH_FILE" && git apply "$TCP_PATCH_FILE")
  APPLIED_SMOKE_BUILD=1
}

ensure_rdma_smoke_targets() {
  ensure_build_backup
  apply_tcp_smoke_patch
  if grep -q 'qwen25_05b_base_openai_remote_cache_pd_sep_mooncake_rdma' "$BUILD_FILE"; then
    log "RDMA smoke target 已存在，跳过生成"
    return
  fi
  log "生成 Mooncake RDMA smoke target"
  BUILD_FILE="$BUILD_FILE" python3 - <<'PY'
from pathlib import Path
import os
build = Path(os.environ['BUILD_FILE'])
text = build.read_text()
blocks = [
    (
        '''        smoke_test(
            name="pd_seperation_prefill_decode_reuse_cache_mooncake_tcp",
            task_info="data/model/qwen25/q_r_prefill_and_decode_reuse.json",
            smoke_args= {
                "prefill": "--load_python_model 1 --warm_up 0 --reuse_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 1024 --write_cache_sync 1 --seq_size_per_block 8 --act_type FP16 --cache_store_rdma_mode 0 --cache_store_mooncake_mode 1 --cache_store_mooncake_transport tcp --cache_store_mooncake_ip_or_host_name 127.0.0.1 --cache_store_mooncake_rpc_port 23545 --use_local 1 --role_type PREFILL --reserver_runtime_mem_mb 4002",
                "decode": "--load_python_model 1 --warm_up 0 --reuse_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 1024 --write_cache_sync 1 --seq_size_per_block 8 --act_type FP16 --cache_store_rdma_mode 0 --cache_store_mooncake_mode 1 --cache_store_mooncake_transport tcp --cache_store_mooncake_ip_or_host_name 127.0.0.1 --cache_store_mooncake_rpc_port 23546 --use_local 1 --role_type DECODE --reserver_runtime_mem_mb 4002"
            },
            gpu_type=["H20"]
        ),
''',
        '''        smoke_test(
            name="pd_seperation_prefill_decode_reuse_cache_mooncake_rdma",
            task_info="data/model/qwen25/q_r_prefill_and_decode_reuse.json",
            smoke_args= {
                "prefill": "--load_python_model 1 --warm_up 0 --reuse_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 1024 --write_cache_sync 1 --seq_size_per_block 8 --act_type FP16 --cache_store_rdma_mode 0 --cache_store_mooncake_mode 1 --cache_store_mooncake_transport rdma --cache_store_mooncake_ip_or_host_name 127.0.0.1 --cache_store_mooncake_rpc_port 23645 --use_local 1 --role_type PREFILL --reserver_runtime_mem_mb 4002",
                "decode": "--load_python_model 1 --warm_up 0 --reuse_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 1024 --write_cache_sync 1 --seq_size_per_block 8 --act_type FP16 --cache_store_rdma_mode 0 --cache_store_mooncake_mode 1 --cache_store_mooncake_transport rdma --cache_store_mooncake_ip_or_host_name 127.0.0.1 --cache_store_mooncake_rpc_port 23646 --use_local 1 --role_type DECODE --reserver_runtime_mem_mb 4002"
            },
            gpu_type=["H20"]
        ),
'''
    ),
    (
        '''        smoke_test(
            name = "qwen25_05b_base_openai_remote_cache_pd_sep_mooncake_tcp",
            data = ["@remote_kv_cache_manager_server//:bin/kv_cache_manager_bin"],
            gpu_type = ["H20"],
            kvcm_envs = ["KVCM_LOG_LEVEL=DEBUG"],
            sleep_time_qr = 20,
            smoke_args = {
                "prefill": "--load_python_model 1 --warm_up 0  --reuse_cache 1 --role_type PREFILL --act_type FP16 --seq_size_per_block 8 --enable_remote_cache true --enable_device_cache 0 --deterministic_attn 1 --reco_put_timeout_ms 12000 --reco_get_timeout_ms 12000 --reco_get_broadcast_timeout 15000 --reco_put_broadcast_timeout 15000 --cache_store_rdma_mode 0 --cache_store_mooncake_mode 1 --cache_store_mooncake_transport tcp --cache_store_mooncake_ip_or_host_name 127.0.0.1 --cache_store_mooncake_rpc_port 23547",
                "decode": "--load_python_model 1 --warm_up 0  --reuse_cache 1 --role_type DECODE --act_type FP16 --seq_size_per_block 8 --enable_remote_cache true --enable_device_cache 0 --deterministic_attn 1 --reco_put_timeout_ms 12000 --reco_get_timeout_ms 12000 --reco_get_broadcast_timeout 15000 --reco_put_broadcast_timeout 15000 --cache_store_rdma_mode 0 --cache_store_mooncake_mode 1 --cache_store_mooncake_transport tcp --cache_store_mooncake_ip_or_host_name 127.0.0.1 --cache_store_mooncake_rpc_port 23548",
            },
            task_info = "data/model/qwen25/q_r_l20_remote_cache_pd_sep.json",
        ),
''',
        '''        smoke_test(
            name = "qwen25_05b_base_openai_remote_cache_pd_sep_mooncake_rdma",
            data = ["@remote_kv_cache_manager_server//:bin/kv_cache_manager_bin"],
            gpu_type = ["H20"],
            kvcm_envs = ["KVCM_LOG_LEVEL=DEBUG"],
            sleep_time_qr = 20,
            smoke_args = {
                "prefill": "--load_python_model 1 --warm_up 0  --reuse_cache 1 --role_type PREFILL --act_type FP16 --seq_size_per_block 8 --enable_remote_cache true --enable_device_cache 0 --deterministic_attn 1 --reco_put_timeout_ms 12000 --reco_get_timeout_ms 12000 --reco_get_broadcast_timeout 15000 --reco_put_broadcast_timeout 15000 --cache_store_rdma_mode 0 --cache_store_mooncake_mode 1 --cache_store_mooncake_transport rdma --cache_store_mooncake_ip_or_host_name 127.0.0.1 --cache_store_mooncake_rpc_port 23647",
                "decode": "--load_python_model 1 --warm_up 0  --reuse_cache 1 --role_type DECODE --act_type FP16 --seq_size_per_block 8 --enable_remote_cache true --enable_device_cache 0 --deterministic_attn 1 --reco_put_timeout_ms 12000 --reco_get_timeout_ms 12000 --reco_get_broadcast_timeout 15000 --reco_put_broadcast_timeout 15000 --cache_store_rdma_mode 0 --cache_store_mooncake_mode 1 --cache_store_mooncake_transport rdma --cache_store_mooncake_ip_or_host_name 127.0.0.1 --cache_store_mooncake_rpc_port 23648",
            },
            task_info = "data/model/qwen25/q_r_l20_remote_cache_pd_sep.json",
        ),
'''
    ),
]
for tcp_block, rdma_block in blocks:
    if tcp_block not in text:
        raise SystemExit('tcp smoke block not found, please apply tcp patch first')
    text = text.replace(tcp_block, tcp_block + rdma_block, 1)
build.write_text(text)
PY
  APPLIED_SMOKE_BUILD=1
}

print_remote_reuse_artifact() {
  if [[ ! -f "$REMOTE_REUSE_ARTIFACT" ]]; then
    log "未找到 artifact: $REMOTE_REUSE_ARTIFACT"
    return
  fi
  log "读取 remote_reuse artifact: $REMOTE_REUSE_ARTIFACT"
  python3 - <<PY
import json
with open("$REMOTE_REUSE_ARTIFACT") as f:
    data = json.load(f)
aux = data.get("aux_info", {})
print(json.dumps({
    "reuse_len": aux.get("reuse_len"),
    "local_reuse_len": aux.get("local_reuse_len"),
    "remote_reuse_len": aux.get("remote_reuse_len"),
    "prefill_remote_reuse_len": aux.get("prefill_remote_reuse_len"),
    "decode_remote_reuse_len": aux.get("decode_remote_reuse_len"),
}, ensure_ascii=False, indent=2))
PY
}

run_common_tests() {
  docker_bazel_test //rtp_llm/server/server_args/test:server_args_test
  docker_bazel_test //rtp_llm/cpp/cache/connector/p2p/test:p2p_connector_config_test
  docker_bazel_test //rtp_llm/cpp/cache/connector/p2p/transfer/test:transfer_backend_config_test
  docker_bazel_test //rtp_llm/cpp/cache/connector/p2p/transfer/tcp/test:tcp_sender_receiver_test
  docker_bazel_test //rtp_llm/cpp/cache/connector/p2p/transfer/mooncake:mooncake_backend_stub_test --define=enable_mooncake_te=true
}

run_tcp_smokes() {
  apply_tcp_smoke_patch
  docker_bazel_test //rtp_llm/test/smoke:pd_seperation_prefill_decode_reuse_cache_mooncake_tcp
  docker_bazel_test //rtp_llm/test/smoke:qwen25_05b_base_openai_remote_cache_pd_sep_mooncake_tcp
  print_remote_reuse_artifact
}

run_rdma_smokes() {
  ensure_rdma_smoke_targets
  docker_bazel_test //rtp_llm/test/smoke:pd_seperation_prefill_decode_reuse_cache_mooncake_rdma
  docker_bazel_test //rtp_llm/test/smoke:qwen25_05b_base_openai_remote_cache_pd_sep_mooncake_rdma
  print_remote_reuse_artifact
}

usage() {
  cat <<USAGE
用法:
  $(basename "$0") [prepare|test|smoke-tcp|smoke-rdma|all]

环境变量:
  BRANCH                     默认: develop/vin/p2p-connector-3
  WORKSPACE_ROOT             默认: /data0/qiongshi.gb/RTP-LLM
  CONTAINER_NAME             默认: vin_rtp_rdma_test
  IMAGE                      默认: $IMAGE
  GPU_CONFIG                 默认: sm9x
  RUN_TCP_REFERENCE          默认: 1
  RUN_RDMA_SMOKE             默认: 1
  RESTORE_INTERNAL_SOURCE    默认: 1，测试结束后恢复 internal_source/rtp_llm/test/smoke/BUILD
USAGE
}

main() {
  host_precheck
  case "$ACTION" in
    prepare)
      checkout_branch
      start_container
      container_precheck
      ;;
    test)
      container_precheck
      run_common_tests
      if [[ "$RUN_TCP_REFERENCE" == "1" ]]; then
        run_tcp_smokes
      fi
      if [[ "$RUN_RDMA_SMOKE" == "1" ]]; then
        run_rdma_smokes
      fi
      ;;
    smoke-tcp)
      container_precheck
      run_tcp_smokes
      ;;
    smoke-rdma)
      container_precheck
      run_rdma_smokes
      ;;
    all)
      checkout_branch
      start_container
      container_precheck
      run_common_tests
      if [[ "$RUN_TCP_REFERENCE" == "1" ]]; then
        run_tcp_smokes
      fi
      if [[ "$RUN_RDMA_SMOKE" == "1" ]]; then
        run_rdma_smokes
      fi
      ;;
    -h|--help|help)
      usage
      return 0
      ;;
    *)
      usage
      die "未知动作: $ACTION"
      ;;
  esac
  log "全部阶段执行完成"
}

main "$@"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:-all}"

usage() {
  cat <<USAGE
用法:
  $(basename "$0") [prepare|test|smoke-tcp|smoke-rdma|all]

说明:
  prepare     只准备容器和运行环境
  test        只执行公共测试与 smoke
  smoke-tcp   只执行 Mooncake TCP smoke
  smoke-rdma  只执行 Mooncake RDMA smoke
  all         先 prepare，再 test

推荐直接使用:
  tools/setup_p2p_mooncake_rdma_env.sh
  tools/run_p2p_mooncake_rdma_tests.sh
USAGE
}

case "$ACTION" in
  prepare)
    exec "$SCRIPT_DIR/setup_p2p_mooncake_rdma_env.sh" prepare
    ;;
  test|smoke-tcp|smoke-rdma)
    exec "$SCRIPT_DIR/run_p2p_mooncake_rdma_tests.sh" "$ACTION"
    ;;
  all)
    "$SCRIPT_DIR/setup_p2p_mooncake_rdma_env.sh" prepare
    "$SCRIPT_DIR/run_p2p_mooncake_rdma_tests.sh" test
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    printf '[mooncake-rdma-suite] ERROR: 未知动作: %s\n' "$ACTION" >&2
    exit 1
    ;;
esac

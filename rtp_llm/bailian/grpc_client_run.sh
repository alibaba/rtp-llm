#!/usr/bin/env bash
# Bailian gRPC 客户端压测 / 单次调用（wire: predict_v2.proto）
#
# 路径：rtp_llm/bailian/grpc_client_run.sh（与 BUILD 中 grpc_client_scripts 一致）
#
# 必须用 bash 运行（不要用 sh：数组 / [[ ]] 会报错）。
# 脚本会自动把仓库根加入 PYTHONPATH（含 rtp_llm 包）。
#
# 示例：只改 top_k（与「export TOP_K=13 && sh …」同类写法，请把 sh 换成 bash）
#   export TOP_K=13 && bash grpc_client_run.sh
#
# 示例：一次性设置全部采样相关环境变量（按需删改；未 export 的用脚本内默认值；示例值均不为 0）
#   export TOP_K=13 \
#     TOP_P=0.95 \
#     TEMPERATURE=0.8 \
#     MAX_NEW_TOKENS=256 \
#     NUM_RETURN_SEQUENCES=1 \
#     MIN_NEW_TOKENS=1 \
#     REPETITION_PENALTY=1.05 \
#     FREQUENCY_PENALTY=0.1 \
#     PRESENCE_PENALTY=0.1
#   export SEED=42                    # 可选；不设则不传 --seed
#   export STOP_TOKEN_IDS="101,102"   # 可选，逗号分隔；不设则不传 --stop_token_ids
#   export RETURN_INPUT_IDS=1         # 可选；1/true/yes 时传 --return_input_ids
#   bash grpc_client_run.sh
#
# 非采样（可选覆盖）：PYTHON GRPC_ADDR CKPT_PATH MODEL_TYPE PROMPT
# 压测循环：GRPC_CLIENT_LOOPS GRPC_CLIENT_DELAY_SEC
#   GRPC_CLIENT_LOOPS=200 GRPC_CLIENT_DELAY_SEC=0.02 bash grpc_client_run.sh

if [ -z "${BASH_VERSION:-}" ]; then
  echo "请使用 bash 运行: bash $0  （不要用 sh）" >&2
  exit 1
fi

set -euo pipefail

# 定位仓库根（含 rtp_llm/）：本文件在 rtp_llm/bailian/ 下
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
if [[ -d "$_SCRIPT_DIR/rtp_llm" ]]; then
  _REPO_ROOT="$_SCRIPT_DIR"
elif [[ -d "$(cd "$_SCRIPT_DIR/../.." && pwd)/rtp_llm" ]]; then
  _REPO_ROOT="$(cd "$_SCRIPT_DIR/../.." && pwd)"
else
  echo "error: 找不到 rtp_llm 包。请在仓库根或 rtp_llm/bailian 下运行本脚本。" >&2
  exit 1
fi
export PYTHONPATH="$_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-/opt/conda310/bin/python3}"
GRPC_ADDR="${GRPC_ADDR:-127.0.0.1:8096}"
CKPT_PATH="${CKPT_PATH:-/home/xinfei.sxf/work/Qwen2-0.5B}"
MODEL_TYPE="${MODEL_TYPE:-qwen_2}"
# PROMPT="${PROMPT:-hello, what is your name}"
PROMPT="${PROMPT:-今天中午吃什么？}"
TOP_K="${TOP_K:-1}"
TOP_P="${TOP_P:-1.0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-0}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
FREQUENCY_PENALTY="${FREQUENCY_PENALTY:-0.0}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0.0}"
# SEED= 不设则不传 --seed；STOP_TOKEN_IDS 同理
SEED="${SEED:-}"
STOP_TOKEN_IDS="${STOP_TOKEN_IDS:-}"
RETURN_INPUT_IDS="${RETURN_INPUT_IDS:-}"
GRPC_CLIENT_LOOPS="${GRPC_CLIENT_LOOPS:-1}"
GRPC_CLIENT_DELAY_SEC="${GRPC_CLIENT_DELAY_SEC:-0}"

run_once() {
  local idx="$1"
  # 单数组 + 条件 +=，避免 set -u 下空数组 "${seed_args[@]}" 报 unbound
  local -a cmd=(
    "$PYTHON" -m rtp_llm.bailian.bailian_grpc_client
    --grpc_addr "$GRPC_ADDR"
    --ckpt_path "$CKPT_PATH"
    --model_type "$MODEL_TYPE"
    --prompt "$PROMPT"
    --request_id "bailian_grpc_client_${idx}_$$"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --num_return_sequences "$NUM_RETURN_SEQUENCES"
    --top_p "$TOP_P"
    --top_k "$TOP_K"
    --temperature "$TEMPERATURE"
    --min_new_tokens "$MIN_NEW_TOKENS"
    --repetition_penalty "$REPETITION_PENALTY"
    --frequency_penalty "$FREQUENCY_PENALTY"
    --presence_penalty "$PRESENCE_PENALTY"
  )
  [[ -n "$SEED" ]] && cmd+=(--seed "$SEED")
  [[ -n "$STOP_TOKEN_IDS" ]] && cmd+=(--stop_token_ids "$STOP_TOKEN_IDS")
  case "${RETURN_INPUT_IDS}" in
    1|true|TRUE|yes|YES) cmd+=(--return_input_ids) ;;
  esac
  "${cmd[@]}"
}

for ((i = 1; i <= GRPC_CLIENT_LOOPS; i++)); do
  echo "=== grpc_client_run.sh call ${i}/${GRPC_CLIENT_LOOPS} ==="
  run_once "$i"
  if ((i < GRPC_CLIENT_LOOPS)); then
    sleep "$GRPC_CLIENT_DELAY_SEC"
  fi
done

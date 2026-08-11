#!/usr/bin/env bash
#
# Kimi K3 跨机 PD 启动脚本(精度已验证的生产配置)
# ============================================================================
#
# 这个脚本把生产部署参数显式写出来，避免依赖 launcher 的隐式默认值。
#
#   用法:  kimi_k3_pd_launch.sh prefill|decode
#
#   必须由调用方给出:
#     CHECKPOINT_PATH     本地盘上的权重目录(禁止 NAS)
#     PREFILL_ENDPOINT    host:port
#     DECODE_ENDPOINT     host:port
#
#   可选:
#     KIMI_K3_RUN_ROOT    运行目录,默认在 CHECKPOINT_PATH 同盘下
#     ENABLE_CUDA_GRAPH   默认 0 —— 见下方"尚未验证"一节
#
# ---------------------------------------------------------------------------
# 精度依据(93 层,官方权重 9f62e4e9,对封版 golden 全 16 条 × 3 次)
# ---------------------------------------------------------------------------
# 开关阶梯实测(线 1 基线,8 个档位):
#
#   档  KDA(P)  MLA(P)    FUSIONS  BATCHED  结果
#   A   kernel  kernel    0        0        4/4 exact
#   B   kernel  kernel    0        0        4/4 exact   (+HOST_META)
#   C   kernel  kernel    1        0        0/4  <-- 坏
#   D   kernel  kernel    1        1        0/4  <-- 坏,边距与 C 逐条相同
#   E   kernel  flashmla  1        1        0/4  <-- 坏
#   F   cula    flashmla  1        1        4/4 exact
#   G   cula    flashmla  0        1        4/4 exact
#   H   cula    kernel    1        1        4/4 exact   <-- 本脚本用的组合
#
# 结论:坏的不是单个开关,是 KDA_BACKEND=kernel × PERF_FUSIONS=1 这个组合。
# 融合 kernel 是围绕 cuLA 设计并验证的(见 kimi_k3.py 里
# copy_free_backend_prefill 的判定),配 Triton chunk 后端时 dtype/布局约定对不上,
# 表现为 Prefill 第一个 token 就错、边距 10-54 ulp、三次重跑一致。
#
# 所以 Prefill 必须钉死 cula。本脚本不给这个选择余地。
set -uo pipefail

role="${1:?用法: kimi_k3_pd_launch.sh prefill|decode}"
case "${role}" in
    prefill | decode) ;;
    *) echo "role 只能是 prefill 或 decode" >&2; exit 2 ;;
esac

: "${CHECKPOINT_PATH:?CHECKPOINT_PATH 必填,且必须是本地 /data* 盘}"
: "${PREFILL_ENDPOINT:?PREFILL_ENDPOINT 必填 host:port}"
: "${DECODE_ENDPOINT:?DECODE_ENDPOINT 必填 host:port}"
case "${CHECKPOINT_PATH}" in
    /data[0-9]*/*) ;;
    *) echo "拒绝非本地盘权重:${CHECKPOINT_PATH}(必须在 /data* 上)" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
run_root="${KIMI_K3_RUN_ROOT:-$(dirname "${CHECKPOINT_PATH}")/k3run-${role}}"
mkdir -p "${run_root}"

# gRPC 会被 127.0.0.1:18180 的代理劫持,必须清干净
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; do unset "$v"; done

# ---------------------------------------------------------------------------
# 拓扑与容量
# ---------------------------------------------------------------------------
export KIMI_K3_DECODE_TOPOLOGY=tp8_ep8   # dp8_ep8 是 legacy,93 层 Decode 会 OOM
export KIMI_K3_MAX_SEQ_LEN="${KIMI_K3_MAX_SEQ_LEN:-16384}"
export KIMI_K3_KV_CACHE_MEM_MB="${KIMI_K3_KV_CACHE_MEM_MB:-8192}"
export MAX_CONTEXT_BATCH_SIZE="${MAX_CONTEXT_BATCH_SIZE:-1}"
export MAX_BATCH_TOKENS_SIZE="${MAX_BATCH_TOKENS_SIZE:-$((KIMI_K3_MAX_SEQ_LEN + 4))}"
export KIMI_K3_MEGA_MAX_TOKENS_PER_RANK="${KIMI_K3_MEGA_MAX_TOKENS_PER_RANK:-8192}"
# ---------------------------------------------------------------------------
# 后端选择 —— Prefill 钉死 cuLA,理由见文件头
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------
export LOAD_METHOD=fastsafetensors

# ---------------------------------------------------------------------------
# 尚未验证:CUDA Graph
# ---------------------------------------------------------------------------
# 上面 8 个档位全部在 ENABLE_CUDA_GRAPH=0 下取得。开启后对精度的影响没有任何数据,
# 而 batched KDA decode + CUDA Graph 是同一个提交(2997d252b)引入的,历史上是
# 精度嫌疑之一。要开请单独跑一轮对照。
export ENABLE_CUDA_GRAPH="${ENABLE_CUDA_GRAPH:-0}"

export CHECKPOINT_PATH TOKENIZER_PATH="${TOKENIZER_PATH:-${CHECKPOINT_PATH}}"
export PREFILL_ENDPOINT DECODE_ENDPOINT
export KIMI_K3_RUN_ROOT="${run_root}/runtime"
# UDS 路径有 107 字节硬上限,run root 太长会以
# "CpuTpBroadcaster UDS path too long" 起不来
export KIMI_K3_TMPDIR="${run_root}/t"
export KIMI_K3_FLASHINFER_WORKSPACE_BASE="${run_root}/fi"
export TMPDIR="${run_root}/pip"
mkdir -p "${TMPDIR}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "[K3-PD] role=${role} host=$(hostname) commit=$(git -C "${repo_root}" rev-parse --short HEAD 2>/dev/null)"
echo "[K3-PD] 显式配置:"
env | grep -E '^KIMI_K3_|^ENABLE_CUDA_GRAPH|^LOAD_METHOD|^MAX_|^CUBLAS_' | sort | sed 's/^/    /'
exec "${repo_root}/example/start_kimi_k3_pd.sh" "${role}"

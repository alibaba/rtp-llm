#!/bin/bash
# G4: index 真实可解析。每 arch 用 uv dry-run 校验 lock 里所有包在 pip 源上真实存在且 hash 匹配。
# 只查元数据不安装。包名拼错 / 私有 OSS 漏传 wheel / 版本不存在，都在这里暴露。
# 用法: bash deps/verify_resolvable.sh [arch...]   缺省全部
set -e
cd "$(dirname "$0")"

command -v uv >/dev/null || { echo "FAIL: uv 未安装 (pip install uv)"; exit 1; }

# unsafe-best-match: 复现 pip/pip-compile 的「合并所有 index 取最优版本」语义。
# uv 默认 first-index-match（防依赖混淆），会拒绝跨 index，与现有 lock 的产出方式不符。
# 私有 OSS index 的防覆盖+local version 治理（T6）是此处放宽的补偿性管控。
IDX=(--index-url https://mirrors.aliyun.com/pypi/simple/
     --extra-index-url https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/rtp_llm/simple/
     --index-strategy unsafe-best-match)

declare -A PLAT=(
  [cuda12]="requirements_lock_torch_gpu_cuda12.txt x86_64-unknown-linux-gnu"
  [cuda12_9]="requirements_lock_torch_gpu_cuda12_9.txt x86_64-unknown-linux-gnu"
  [rocm]="requirements_lock_rocm.txt x86_64-unknown-linux-gnu"
  [cpu]="requirements_lock_torch_cpu.txt x86_64-unknown-linux-gnu"
  [arm]="requirements_lock_torch_arm.txt aarch64-unknown-linux-gnu"
  [cuda12_arm]="requirements_lock_cuda12_arm.txt aarch64-unknown-linux-gnu"
)

ARCHES=("$@")
[ ${#ARCHES[@]} -eq 0 ] && ARCHES=(cuda12 cuda12_9 rocm cpu arm cuda12_arm)

fail=0
for a in "${ARCHES[@]}"; do
  read -r lock plat <<< "${PLAT[$a]}"
  [ -f "$lock" ] || { echo "SKIP $a: $lock 不存在"; continue; }
  # --require-hashes: lock 由 --generate-hashes 产出，每项都带 hash。
  # 加此旗标后 uv 必须选中 hash 与 lock 匹配的产物，否则失败——
  # 这才让 G4 真正校验「index 上的内容 == lock 锁定的 hash」，而不只是「版本可解析」。
  # --system: uv pip install 需要一个目标解释器做基准；--dry-run 保证不真正安装。
  if uv pip install --dry-run --system --require-hashes -r "$lock" \
       --python-version 3.10 --python-platform "$plat" "${IDX[@]}" >/dev/null 2>"/tmp/verify_$a.err"; then
    echo "OK $a"
  else
    echo "FAIL $a: $(tail -3 /tmp/verify_$a.err | head -1)"
    fail=1
  fi
done
exit $fail

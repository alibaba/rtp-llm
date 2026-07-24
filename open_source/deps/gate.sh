#!/bin/bash
# 依赖门禁（离线，秒级，无容器）——CI 与本地提交前的单一入口。
# 三项检查全部跑完再汇总，任一失败则整体 FAIL。
# 需要联网的「lock 里的包在 pip 源上真实存在且 hash 匹配」检查见 deps/verify_resolvable.sh。
cd "$(dirname "$0")"
# 解析 py3：优先 $PYTHON 覆盖，其次本项目基准解释器 /opt/conda310/bin/python3，
# 再退到 PATH 上的 python3 / 其它 conda；治痛点 10①（本环境 `python3` 不在 PATH，硬用会让门禁整体失效）。
PY="${PYTHON:-}"
[ -z "$PY" ] && for c in /opt/conda310/bin/python3 /opt/conda/bin/python3; do [ -x "$c" ] && PY="$c" && break; done
[ -z "$PY" ] && PY="$(command -v python3 2>/dev/null || true)"
[ -z "$PY" ] && { echo "FAIL: 找不到 python3（设 PYTHON=/path/to/python3）"; exit 1; }

rc=0
run() { echo "── $1"; "${@:2}" || rc=1; }

run "检查 lock 是否由当前 requirements 生成（防改了 requirements 忘重编 lock）" \
    "$PY" check_lock_freshness.py .
run "检查 wheel_requires.txt 是 base+extra 的最新派生且版本与 lock 一致（治痛点 8）" \
    "$PY" gen_wheel_requires.py --check
run "检查同平台多份 lock 的同一包 hash 不冲突（防私有 OSS 上同版本被覆盖）" \
    "$PY" check_lock_consistency.py .
run "检查 BUILD 的 requirement() 均在 lock∪bazel桶 中（治痛点 10②，提前拦无效引用）" \
    "$PY" check_requirement_subset.py ..
run "检查 bazel 桶自洽：桶∩lock=∅ 无双供给（治痛点 9 版本项）" \
    "$PY" check_bazel_bucket.py ..

if [ $rc -ne 0 ]; then
  echo "门禁未通过：见上方各项的修复提示"
else
  echo "门禁通过：离线检查全绿（联网校验请跑 deps/verify_resolvable.sh）"
fi
exit $rc

# 功能测试

功能测试验证协议返回、状态转换、取消、边界值和确定性竞态。它使用少量可控请求；任一前置步骤失败会阻断依赖步骤。

先完成[编译与运行底座](build-and-runtime.md)，再从 `rtp_llm/flexlb` 执行。

## 选择实例

```bash
python3 tools/online_eval/scenario_runner.py \
  --source tools/online_eval/scenarios \
  --suite functional --list-json > /tmp/flexlb-functional.json
```

实例 ID 形如 `request_completion::immediate::batch-window`。按 category 选一组，或用 `--instances` 精确选择。精确选择可避免一次运行混入无关场景。

## 预览资源计划

```bash
python3 tools/online_eval/test_runner.py \
  --suite functional \
  --instances 'request_completion::immediate::batch-window' \
  --parallel 1 --dry-run
```

确认 profile、lane 数、Master 端口和 Mock 端口窗口后去掉 `--dry-run`。

## 执行

```bash
OUT=/path/to/new-output/functional
python3 tools/online_eval/test_runner.py \
  --suite functional \
  --profile batch-window \
  --parallel 4 \
  --out-dir "$OUT" \
  --json "$OUT/aggregate.json"
```

常用选择：

- `--categories status,cancel`：运行 category 子集。
- `--instances id1,id2`：运行精确实例，优先级高于广泛分类。
- `--profile` 或 `--master-mode`：选择一种运行形态。
- `--parallel 1`：排查共享状态、时序或单例失败。
- `--archive /path/to/result.zip`：生成可搬运证据包。

## 判定

1. 查看顶层 `aggregate.json` 的总体状态。
2. 对失败实例读取其 `result.json`，区分 `FAIL`、`ERROR`、`TIMEOUT` 和 `BLOCKED`。
3. `FAIL` 表示检查得到反例；`ERROR/TIMEOUT` 表示运行或证据不完整；`BLOCKED` 是前序失败的结果。
4. YAML 中显式登记的 finding 只解释指定检查，不能吞掉异常、超时或清理失败。

功能测试 PASS 只证明相应合同，不证明性能健康。产物字段见[结果与指标](../reference/results.md)。

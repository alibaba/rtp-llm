# 新增 case

## 只改变数据或阈值

在 `config/scenarios/` 下按 case 名新增或修改 schema v2 YAML，不创建分类子目录。YAML 保存拓扑、profile、请求数据、时间预算、阈值和参数约束，不写步骤或条件分支。

## 新增业务流程

在 `src/cases/programs/` 增加 Python program，并在显式目录中注册。Python 通过 `CaseBuilder` 读取 YAML 数据，声明步骤、输出引用和检查；公共动作不足时才在 `src/scenario/actions/` 增加有类型的 handler。

## 分类

- 少量确定请求验证返回码、状态或边界：`functional`。
- 持续负载、故障、扩缩容、恢复或阶段曲线：`workload`。

在 YAML 顶层 `test` 中声明公共默认值，或在 `variants[].test` 中逐实例覆盖：

```yaml
test:
  kind: functional
  description: 请求完成和状态恢复
  collection: diagnostic
```

`kind` 可选 functional / workload；`collection` 可选 aggregate / request / diagnostic。持续负载可以通过 `test.monitoring` 覆盖采样间隔等监控参数。每个实例必须获得完整的性质、说明和采集档位，缺失会报错。

只有需要列入 CI 必跑时，才在 `config/suites.yaml` 的 `ci_suites` 中增加 `case::variant`。文件位置、业务 category 和 kind 均不隐含 CI 必跑。

## 验证

```bash
python3 tools/online_eval/scripts/commands/list_cases.py \
  --source tools/online_eval/config/scenarios --profile batch-window --suite functional --list-json

python3 tools/online_eval/scripts/commands/run_cases.py \
  --instances '<exact-instance-id>' --parallel 1 --dry-run

cd tools/online_eval
PYTHONPATH="$PWD/src:$PWD" python3 -m unittest discover -s tests -p 'test_*.py'
```

新增检查必须保存实际值、期望值和原始证据。扩大拓扑或负载后需重新验证阈值，不能复制旧 band 后直接宣称场景有效。

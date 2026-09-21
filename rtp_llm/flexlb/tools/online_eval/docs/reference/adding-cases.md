# 新增 case

## 只改变数据或阈值

在 `config/scenarios/` 对应 category 中新增或修改 schema v2 YAML。YAML 保存拓扑、profile、请求数据、时间预算、阈值和参数约束，不写步骤或条件分支。

## 新增业务流程

在 `flexlb_test_framework/case_programs/` 增加 Python program，并在显式目录中注册。Python 通过 `CaseBuilder` 读取 YAML 数据，声明步骤、输出引用和检查；公共动作不足时才在 `scenario/actions/` 增加有类型的 handler。

## 分类

- 少量确定请求验证返回码、状态或边界：`functional`。
- 持续负载、故障、扩缩容、恢复或阶段曲线：`workload`。

在 `config/suites.yaml` 中登记分类和采集档位。分类不改变业务 category，也不改变 Master profile。

## 验证

```bash
python3 tools/online_eval/scripts/scenario_runner.py \
  --source tools/online_eval/config/scenarios --list-json

python3 tools/online_eval/scripts/parallel_runner.py \
  --instances '<exact-instance-id>' --parallel 1 --dry-run

python3 -m unittest discover -s tools/online_eval/tests -p 'test_*.py'
```

新增检查必须保存实际值、期望值和原始证据。扩大拓扑或负载后需重新验证阈值，不能复制旧 band 后直接宣称场景有效。

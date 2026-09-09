# FlexLB YAML case 测试入口

本目录是 `codex/ft-case-framework` 的整合工作区。YAML 场景已经实现，文件在
[`rtp_llm/flexlb/tools/online_eval/scenarios/`](rtp_llm/flexlb/tools/online_eval/scenarios/README.md)。
旧 Python case 仍保留作契约对照；默认不带 `--source yaml` 的旧命令不会选择新 YAML。
框架代码目录为 `flexlb_test_framework/`（FlexLB 测试框架），Python 导入也使用该名称。

## 使用文档

- [框架设计](rtp_llm/flexlb/tools/online_eval/docs/framework-design.md)：分层、编译与执行、资源生命周期、结果和迁移边界。
- [如何添加新 case](rtp_llm/flexlb/tools/online_eval/docs/adding-cases.md)：完整 YAML 示例、variant、校验运行、Python action 扩展与测试。

## 先查看 YAML 和编译结果

在仓库根目录执行（只加载、校验和列出计划，不启动 Java）：

```sh
python3 rtp_llm/flexlb/tools/online_eval/scenario_runner.py \
  --source rtp_llm/flexlb/tools/online_eval/scenarios \
  --list-json
```

只看一个文件、一个 profile：

```sh
python3 rtp_llm/flexlb/tools/online_eval/scenario_runner.py \
  --source rtp_llm/flexlb/tools/online_eval/scenarios/priority/priority_queue.yaml \
  --profile single-nonbatch --list-json
```

实例 ID 是 `scenario_id::variant_id::profile`。清单中的 `source_path` 指向对应 YAML。
YAML 加载需要 PyYAML；以上列清单操作不需要 JAR 或运行环境租约。

## 从哪里读实现

| 路径 | 内容 |
|---|---|
| [`scenarios/`](rtp_llm/flexlb/tools/online_eval/scenarios/README.md) | 环境、请求、故障、观测窗口及断言的 YAML 流程 |
| [`scenario_runner.py`](rtp_llm/flexlb/tools/online_eval/scenario_runner.py) | 编译、列实例、执行单实例 |
| [`parallel_runner.py`](rtp_llm/flexlb/tools/online_eval/parallel_runner.py) | 通过 `--source yaml --case-dir ...` 选择新入口，`--parallel 1` 串行，多 lane 并行 |
| [`flexlb_test_framework/scenario/`](rtp_llm/flexlb/tools/online_eval/flexlb_test_framework/scenario/README.md) | 编译器、执行器、资源管理及显式注册的 Python actions |
| [`migration/`](rtp_llm/flexlb/tools/online_eval/migration/README.md) | 旧 case 覆盖映射、原始运行结果与修复复验队列 |

执行真实 Java 需要 Linux 开发容器、两份 JAR 及父 runner 的端口/租约预检；
按 [父 runner 执行说明](rtp_llm/flexlb/tools/online_eval/README.md#structured-parent-entry-integration)
配置环境。`--list-json` 成功仅代表编译成功，不能当作真实 Java 测试通过。

## 实现与验收边界

当前完整候选清单包含 31 个执行定义、183 个 variant、385 个实例。
它们覆盖 29 个目标场景族、139 个旧 case、371 个旧 case/profile 组合。
这些数字分别计数，不能用实例数替代场景族数。

所有旧 case 都保留，未完成旧新版配对验收。运行结果以
[`migration/full_matrix_3e160.json`](rtp_llm/flexlb/tools/online_eval/migration/full_matrix_3e160.json)
为准；后续修复不覆盖冻结版本的原始 FAIL、ERROR 或 TIMEOUT。

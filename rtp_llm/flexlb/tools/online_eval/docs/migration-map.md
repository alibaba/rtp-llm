# 文档迁移映射

本表覆盖重组前 `online_eval/docs/`、压测 README 和两份 Whale 组件 README。`当前`表示仍是规范的一部分；`历史`表示只保留追溯价值。

| 旧位置 | 新位置 | 状态与处理 |
|---|---|---|
| `online_eval/README.md` | 原位置 | 当前；改为四类功能的唯一入口 |
| `online_eval/stress/README.md` | `docs/development/stress.md` | 当前流程重写；原长文移入 `archive/legacy-20260921/stress-README.md` |
| `docs/adding-cases.md` | `docs/reference/adding-cases.md` | 当前内容重写；原文归档 |
| `docs/framework-design.md` | `docs/reference/architecture.md` | 当前结构收敛；原完整设计稿归档 |
| `docs/test-suites.md` | `docs/development/functional.md`、`scenario.md` | 按功能拆分；原文归档 |
| `docs/monitoring-pipeline.md` | `docs/reference/results.md` | 当前口径合并；原文归档 |
| `docs/reporting-pipeline.md` | `docs/reference/results.md` | 当前口径合并；原文归档 |
| `docs/shared-test-runtime.md` | `docs/development/build-and-runtime.md`、`docs/reference/architecture.md` | 共享底座重写；原文归档 |
| `docs/unified-mock-framework-design.md` | `docs/reference/architecture.md`、`docs/whale/README.md` | 开发机与 Whale 分开；原文归档 |
| `docs/framework-validation-20260911.md` | `docs/archive/legacy-20260921/` | 历史；阶段验证记录 |
| `docs/functional-scenario-test-review.md` | `docs/archive/legacy-20260921/` | 历史；审查快照，不再承担入口职责 |
| `docs/case-consolidation-analysis.md` | `docs/archive/legacy-20260921/` | 历史；未实施方案 |
| `docs/cache-hotspot-storm.md` | `docs/reference/cases/cache-hotspot-storm.md` | 当前；特定 case 契约 |
| `docs/cache-scale-in-gate.md` | `docs/reference/cases/cache-scale-in-gate.md` | 当前；特定 case 契约 |
| `docs/decode-scale-out-protection.md` | `docs/reference/cases/decode-scale-out-protection.md` | 当前；特定 case 契约 |
| `docs/engine-removal-contract.md` | `docs/reference/cases/engine-removal-contract.md` | 当前；特定 case 契约 |
| `docs/ha-kv-checkpoints.md` | `docs/reference/cases/ha-kv-checkpoints.md` | 当前；特定 case 契约 |
| `docs/mock-pd-fetch-lifecycle.md` | `docs/reference/concepts/mock-pd-fetch-lifecycle.md` | 当前；协议语义 |
| `docs/traffic-sources-consolidation.md` | `docs/reference/concepts/traffic-sources-consolidation.md` | 当前；流量源语义 |
| `docs/validation/*.md` | `docs/archive/legacy-20260921/validation/` | 历史；按日期保留验证证据 |
| `flexlb-mock-engine/whale/README.md` | `docs/whale/README.md`、`configuration.md` | 当前流程集中到 Whale 线；原长文归档 |
| `tools/whale_mock/README.md` | `docs/whale/README.md`、`configuration.md` | 当前流程集中到 Whale 线；原长文归档 |

## Skill 侧迁移

| 旧内容 | 新归属 |
|---|---|
| 压测命令、档位和默认负载 | `docs/development/stress.md`、`docs/reference/parameters.md` |
| 编译、测试和启动步骤 | `docs/development/build-and-runtime.md` |
| 产物定位、有效性和出图 | `docs/reference/results.md` |
| 功能与场景测试 | `docs/development/functional.md`、`scenario.md` |
| Whale bundle 与独立 Pod | `docs/whale/` |
| 远端连接、租约、同步和作业动词 | skill，仅保留连接说明，不进入项目 runbook |

兼容脚本仍在仓库或 skill 中保留，本次重组不删除脚本；它们不再承担流程文档职责。

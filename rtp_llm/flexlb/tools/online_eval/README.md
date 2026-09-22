# FlexLB Mock 测试

按目标选择[压测](docs/development/stress.md)、[功能测试](docs/development/functional.md)、[场景测试](docs/development/scenario.md)或[Whale 部署](docs/whale/README.md)。

## 目录

- `src/`：组件源码，按 `monitoring/`、`traffic/`、`runtime/`、`cases/`、`scenario/`、`workload/`、`analysis/`、`reporting/`、`artifacts/` 组件组织。
- `scripts/commands/`：日常入口；`scripts/pipeline/`：内部执行与证据加工；`scripts/probes/`：低频验收探针。
- `config/`：场景 YAML、运行模式、suite、规模实验、报告视图与性能配置。
- `data/`：受版本控制的流量样本及性能输入；输出数据写到忽略的 `run/`。
- `docs/`：当前 runbook 与稳定的口径说明。
- `tests/`：代码回归测试与小型 fixture。

各组件直接作为 `src` 下的 Python 包导入；从项目根目录运行测试时使用 `PYTHONPATH=src:.`。旧的三个根层转发入口已经退役；使用 `scripts/commands/` 中的命令。Whale 镜像沿用根目录的 `flexlb_cfg.py`、`mode_profiles.py` 和 `mode_profiles.yaml` 固定拷贝路径，`config/mode_profiles.yaml` 链接到同一份模式表。

目录内容分别见 [config](config/README.md) 和 [data](data/README.md)。更多说明见[文档导航](docs/README.md)。

合成输入与真实捕获的独立对比工具：[`compare_traffic.py` 使用与口径](docs/reference/concepts/synthetic-fidelity.md)。

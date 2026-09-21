# FlexLB Mock 测试

按目标选择[压测](docs/development/stress.md)、[功能测试](docs/development/functional.md)、[场景测试](docs/development/scenario.md)或[Whale 部署](docs/whale/README.md)。

## 目录

- `src/online_eval/`：监控、流量、报告与归档等公共组件。
- `src/flexlb_test_framework/`：场景编译、执行、资源管理、case program 和断言。
- `src/stress/`：压测专用聚合、比较、报告渲染。
- `scripts/`：常用命令入口；`scripts/stress/` 含压测 shell 入口。
- `config/`：场景 YAML、运行模式、suite、规模实验、报告视图与性能配置。
- `data/`：受版本控制的流量样本及性能输入；输出数据写到忽略的 `run/`。
- `docs/`：当前 runbook 与稳定的口径说明。
- `tests/`：代码回归测试与小型 fixture。

Python 包名 `online_eval`、`flexlb_test_framework`、`stress` 保持不变；从项目根目录运行测试时使用 `PYTHONPATH=src:.`。原有三个顶层 Python 入口保留轻量兼容转发，实现在 `scripts/`。Whale 镜像沿用根目录的 `flexlb_cfg.py`、`mode_profiles.py` 和 `mode_profiles.yaml` 固定拷贝路径，`config/mode_profiles.yaml` 链接到同一份模式表。

更多说明见[文档导航](docs/README.md)。

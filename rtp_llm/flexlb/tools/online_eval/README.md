# FlexLB 测试工具

日常功能测试使用 **YAML 配置 + Python case**。默认入口是 `parallel_runner.py`，
默认读取本目录的 `scenarios/`，使用 `batch-window` profile、4 个并行 lane。
Python 定义步骤和断言，YAML 传入 P/D 规模等数据。

## 先看哪里

| 目录 / 文件 | 用途 |
|---|---|
| [scenarios/](scenarios/README.md) | 当前 YAML 配置 |
| [flexlb_test_framework/case_programs/](flexlb_test_framework/case_programs/) | 当前 Python 业务流程；仍为 31 个程序，未合并为 9 个 |
| [flexlb_test_framework/scenario/](flexlb_test_framework/scenario/README.md) | 加载、编译、action、执行与资源清理 |
| [flexlb_cfg.py](flexlb_cfg.py)、[框架公共模块](flexlb_test_framework/README.md) | 配置、进程、RPC 和上下文能力 |
| [tests/](tests/) | 框架及压测工具的回归测试 |
| [stress/](stress/README.md) | 性能压测、指标采集、A/B 对比和报告工具 |
| [data/](data/) | 性能预设和流量输入数据，属于有效输入 |

## 运行新版 case

以下命令从本目录执行。`--case-dir` 省略时使用内置配置，路径不依赖当前工作目录。

```bash
# 列出全部 profile 的实例；不启动服务
python3 scenario_runner.py --source scenarios --list-json

# 预览一个实例的资源与执行计划
python3 parallel_runner.py \
  --instances 'request_completion::immediate::batch-window' \
  --parallel 1 --dry-run
```

真实执行使用已分配的远端资源和新的输出目录，去掉 `--dry-run`。
端口、租约、构建和完整示例见 [添加新 case](docs/adding-cases.md)。
`--source yaml` 可以省略；实例选择统一使用 `--instances`。

## 文档

- [框架设计、术语与分层图](docs/framework-design.md)
- [添加配置与 Python case](docs/adding-cases.md)
- [多 P 缓存热点溢出探针、指标与校准](docs/cache-hotspot-storm.md)
- [9 个业务入口的收缩分析，尚未实施](docs/case-consolidation-analysis.md)

## 压测入口

性能压测从 `stress/` 进入。根目录同名压测脚本是相对符号链接，指向唯一实现。
功能测试统一使用上述 Python case 入口。

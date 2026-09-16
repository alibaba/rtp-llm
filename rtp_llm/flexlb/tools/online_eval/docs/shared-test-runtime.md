# 压测与 case 的公共测试底座

本文记录初次公共模块抽取的边界。当前功能 / 复杂场景两类执行策略、连续采样、报告和阶段对齐比较见 [测试分类与执行](test-suites.md)；以下“本次”指初次抽取提交，不代表当前完整实现状态。

## 边界

压测与 case 共用底层能力，保留各自的编排方式。case 的依赖步骤在前置条件失败后仍然阻断；压测的连续采样不能因为某个观测结论失败就丢失后续证据。合并底层模块不意味着统一两种执行策略。

配置值仍由场景 YAML 提供。公共模块只描述数据结构、协议和执行方法，不增加拓扑、阈值、节奏参数或变体默认值。

```
scenarios/*.yaml → Python case program → scenario/backend
                                              ↓
stress/lib_load_client.sh ────────────→ online_eval 公共模块
                                              ↓
                                 mock 引擎 / JavaLoadClient
```

## 本次完成的抽取

| 模块 | 职责 | 使用方 |
| --- | --- | --- |
| `online_eval/requests.py` | 逐请求记录、请求执行、取消、终态与窗口完整性 | scenario backend；elastic 保留兼容导出 |
| `online_eval/metrics.py` | Prometheus 文本解析 | EngineOps 与 BalanceSampler |
| `online_eval/load_client_env.txt` | JavaLoadClient 读取的环境变量名称清单 | Python ClientOps 与 Bash 压测启动器 |
| `online_eval/load_client.py` | 读取共享清单 | Python 启动器 |

底层请求模块不再导入任何具体 case。现有 `elastic.ClientRecords` 等导入保持兼容，避免外部扩展和存量 case 一次性改名。指标解析保持原有跳过规则、标签限制、时间戳处理方式；不借搬迁调整采样或断言。

环境名称清单不包含配置值。两个启动器先清空清单中的环境变量，再应用调用方显式参数。此前 Python 缺少 `PRIORITY`、`FORCE_PRIORITY`、`RAMP_UP_SECONDS`、`REPLAY_UNIQUE_PREFIX`，Shell 缺少 `LIVE_CLIENT_EVENTS`；现在使用同一清单，避免机器环境改变测试。测试校验清单覆盖 JavaLoadClient 的环境读取，同时验证旧 symlink 和 stress 入口一致。

## 本次没有改变的行为

`EngineOps.run_one_request` 当前已经要求 `snap.completed`，有输出但没有业务 finished 不会返回成功。不能把旧分析中的问题当作当前代码缺陷再修一遍。共享请求成功口径同样要求业务 finished、无业务错误、Schedule/stream 成功、没有客户端取消；请求消费者是否退出由完整性口径单独记录。

根目录的压测脚本链接是兼容入口，不是重复实现。本次没有删除这些链接，也没有修改 Java、master、mock 引擎逻辑或 YAML 阈值。

## 持续流量与 ABA 的后续接入边界

现有持续流量、采集器和 `stress/compare_ab.py` 应复用。完整 ABA 是两个独立 master 共享引擎池上的 A-old → B → A-new 客户端切换，不应描述成 leader 选举。

要把 ABA 接成统一报表，需要另一个行为变更：把 run、阶段、请求 attempt、执行 batch、master/engine generation 的关联落到同一份证据，再由该证据计算逐请求断言、阶段指标与 A/B 曲线。不能把 Python monotonic 秒直接拼到压测 epoch 毫秒上；需要运行内的时间锚点，跨版本按同名阶段对齐，缺失阶段或样本必须显式显示。

这次只抽取已验证的公共底层，没有声称完成 ABA 持续套件迁移、统一报表、曲线变化排序或所有历史文件清理。它们会改变运行/产物契约，应在公共层稳定后按纵向场景单独验证，不能仅搬目录并宣布完成。

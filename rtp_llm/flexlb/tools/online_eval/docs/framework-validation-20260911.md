# 功能与持续负载框架验证（2026-09-11）

本轮从 `3d47e5c5f860b4fa3a32b2ee2ae08fadba1a4ef8` 修复采集、证据、报告和持续场景执行问题。改动限于 `online_eval` 与 `flexlb-mock-engine`（含测试客户端），没有修改 master 或生产引擎。原始失败结果保留，未降低成功率、命中率、分布或恢复阈值。

## 设计与使用

[框架设计](framework-design.md)、[添加 case](adding-cases.md)、[套件分类](test-suites.md) 分别说明术语、Python/YAML 分工和运行入口。当前实例清单为 386 项：功能 320 项、持续负载/复杂场景 66 项。分类按验证机制选择，不要求有限请求的功能测试凑压测窗口。

主要变化：

- 每个 mock 地址只有一个采集者。断言、报告和对比读取同一份采样，避免破坏性指标被重复读取。内存历史有界，落后消费者从原始日志重放。
- 每个环境与每个 master 独立保存采集序列、有效时间范围、进程实例和缺口。指标暂未出现与整个采集源掉线分开处理。
- 失败路径保留原始请求。HA 提前终止时可读取实时 journal，保留终态与未完成请求；缺少完整收尾仍标为 INVALID，不伪造完整性。
- INVALID 不能产生 PASS 或 finding 裁决；原始 FAIL/TIMEOUT 不覆盖。功能断言、执行证据有效性、性能比较是三个独立结论。
- 复用既有 stress 聚合及 HTML 渲染，贯通请求、重试、实际目标、batch 与进程实例。产品未提供的 endpoint generation 保留 null，mock 进程 UUID 不冒充产品代际。
- Java 并发许可覆盖整个 Schedule/Fetch 请求；Schedule 阶段另有独立计数。Python 并发变更使用有界任务池，移除与添加在声明的负载期内交错。
- wraparound 扩大为 4P8D、420 秒、三轮恢复；并发弹性为 4P8D、30 秒；新增 12 个窗口、600 个请求的持续混合负载。配置在 YAML，编排在 Python。

## 验证版本与证据

完整原始产物位于本地 `~/code/case-refactor-reports/2026-09-11/goal-framework/`。`SOURCE-*.json` 记录每次冻结的产品 SHA、测试文件哈希和符号链接。运行中不修改远端源码。

| 验证 | 结果及解释 |
| --- | --- |
| r11 Python 全量 | 910 项通过，538.822 秒 |
| r12 Python 全量 | 912 项中 910 通过、2 失败；两个默认端口单测受租约环境变量影响，未进入 A/B 阶段 |
| r13 修复验证 | 在同样租约端口变量下，23 项相关测试通过；仅隔离单测默认端口，不修改正式租约配置 |
| r14 KV 单测 | 11 项通过 |
| 格式整理后 | 41 项采集/证据/聚合/风暴单测通过；去除 import 后 AST 与 r14 相同 |
| Java | Maven 构建及 53 项选定客户端、路由、控制接口测试通过 |
| 失败证据 | 真实 wraparound 中恢复 1,676 条记录：1,672 终态、4 未完成，仍标为不完整 |

r11 完整矩阵已结束，8 lane 并行、四个 profile 全覆盖：**386 项 = 347 PASS + 27 FAIL + 6 TIMEOUT + 5 FINDING-CONFIRMED + 1 FINDING-RESOLVED**。原始退出码为 1，不能称为全绿。后续定向修复的结果单列，不覆盖这份全量结果。

27 个 FAIL 的原始分组如下；这些是观测结论，不等同于已证实的生产缺陷：

| 分组 | 数量 | 原始失败信号 |
| --- | --- | --- |
| 弹性生命周期 | 10 | 摘机阶段零错误要求未达标（8），KV 倾斜样本各有 1 次失败（2） |
| 原有 leader spill | 4 | 命中率或复制数量合同未达标 |
| 状态协议 / late completion | 6 | zombie scheduler 未清理（4），TTL 事件为 0（1），late scheduler 残留 1（1） |
| wraparound | 4 | 非 batch 两例切换请求失败；batch 两例轮转检测仍按 2P 校验 |
| 并发变更 | 2 | batch failfast 收敛要求未达标 |
| mixed holders | 1 | 非 batch 路由 max_share=1.0，要求不超过 0.75 |

两个非 batch wraparound 因失败提前中止而证据不完整，保留 FAIL/INVALID；r13 已验证能够追回实时请求记录，但不会把未完成请求改成完成。其他 workload 失败证据为 VALID。`ack_drop::batch-window` 出现 FINDING-RESOLVED，需要复核并决定是否摘除探针标记。

6 个 TIMEOUT 均为下述缓存 wave 自锁；r14 定向验证消除了超时。batch wraparound 的 2P 写死判断已改为当前拓扑 P 数，连续落点上限不变。r15 两例轮转检查均通过，batch-window 为 PASS/VALID；single-batch 功能检查全通过，但一次采样开始于 kill 前约 5 ms、连接在 kill 后被重置，因旧采集日志缺少结束时间被标 ERROR/INVALID。补记采样完成时间后，失败按实际观测完成时刻归属窗口，不增加窗口容差。r16 single-batch 完整复跑为 PASS/VALID，无非预期采集缺口，26 项合同/证据单测通过。

## 真实跨版本 A/B

基线产品为 `6b96f5e540bf43dfc3cce8dbd017b4622efc26e0`，候选为 `3d47e5c5f860b4fa3a32b2ee2ae08fadba1a4ef8`，两者使用相同 r13 测试覆盖层、配置和机器。每侧独立构建、运行两次；每次 4P8D、600 请求。四次均 PASS/VALID，六项成熟统计完整性检查全部通过。

| 产品 | 次数 | 成功/发出 | 全链路 p50 / p95 / p99（ms） |
| --- | --- | --- | --- |
| 基线 | 1 | 600/600 | 251.8 / 255.7 / 258.2 |
| 基线 | 2 | 600/600 | 251.8 / 256.4 / 260.8 |
| 候选 | 1 | 600/600 | 251.7 / 255.7 / 259.6 |
| 候选 | 2 | 600/600 | 251.8 / 255.7 / 260.6 |

这是两次重复的描述性比较，不足以宣称性能提升或回归。`ab-r13-statistics.json`、`final-r13/` 保存数字和实际报告。浏览器已验证最终对比页的两条曲线、数值时间轴和长标题换行。HTML 展示前 12 个面板；全部 19,800 条阶段/指标比较保存在 JSON，缺失值不补零。

## 热点风暴实跑

r13 的 P=2/3/4 均为 FINDING-CONFIRMED/VALID，正常退出，不污染套件退出码。基线和恢复硬门槛通过。

| P | 最低命中率 | 最大热点持有引擎数 | 恢复窗口数 | 成功请求数 |
| --- | --- | --- | --- | --- |
| 2 | 0 | 2 | 3 | 150 |
| 3 | 0 | 3 | 3 | 151 |
| 4 | 0 | 4 | 3 | 152 |

本轮说明扩散范围随 P 增加，但恢复窗口没有变慢；不能据此支持“扇出越多恢复越慢”。沿用各拓扑已有独立校准参数，本轮没有为得到 finding 修改 band。完整窗口数据及校准配置在 `storm-r13-summary.json` 与远端各实例产物中。

## 缓存 wave 自锁修复

`cache_local_index::evict_batch`、`cache_global_holders::release_batch/mixed_batch` 的 wave 原来 deferred，所有 Schedule 返回后才 Fetch；已有租约阻塞后续 Schedule，使 Fetch 无法执行。只把这三个 wave 改为准入后立即启动消费，保留 seed 的 deferred 语义、20 请求、节奏和分布断言。定向覆盖 single-batch 与 batch-window 六个实例：五个 PASS，`mixed_batch::single-batch` 完成请求但 max_share=1.0 超过 0.75，保留 FAIL。六个原始 wave 超时均消除，剩余路由集中不能通过放宽分布阈值掩盖。

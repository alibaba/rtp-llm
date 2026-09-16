# 2026-09-10 case 侧失败修复复核

基线：`49663020a986ca7e902e2d3881cacd4a93064707`，交付分支 `codex/ft-case-framework`。
本轮只修改 Python 测试框架、场景 YAML 和测试文档，不修改 master、引擎或 mock Java。

## 归因与修改

| 项目 | 改前口径 → 改后口径 | 依据与边界 |
| --- | --- | --- |
| FR-1 capacity park FIFO | 晚启动的 drain waiter 返回顺序 → 请求自身 `consumer_exit_s` 顺序 | 原始流完成时刻有序，收割线程返回有微小逆序；保留严格 FIFO，不增加容差，不改变引擎行为。 |
| FR-2 queue deadline | 6 admitted / 2 rejected → 5 admitted / 3 rejected | 1500ms queue timeout 与 3000ms 执行时间下，串行 Schedule 被阻塞，第三、五、七笔超时；不是按一次实测直接改数字。 |
| FR-3 KV full shrink | timeout 分支只接受退休 8510 → 同时接受 YAML 指定的 8209 数据面断链错误 | 同时核对 code、message 和摘除后的终态时刻。drain_ok 保持原契约，不将所有 82xx 放入白名单。 |
| FR-4 all masters down | 稳态窗口延伸至 kill → kill 前 1 秒截止 | 只排除跨 kill 边界的重试，route_share 仍要求 1.0。边界值在 YAML。 |
| FR-5 freeze | replay_speed 默认 2 → YAML 显式 0.5 | 2105 笔/100 秒的供给超过约 9 笔/秒完成速率，finish 在等积压，不是死锁。保持 finish 120 秒，开启逐笔事件归档。现有 straddle 判定不重复修改。 |
| FR-6 priority | incoming 拒绝族漏 8511 → YAML 为两个变体各阶段声明一致族表 | 仅接受指定 typed code；成功/失败标志、victim 与 survivor 契约仍独立校验。纯优先级与 observability 取数在基线已修复。 |
| FR-7 batch spill | 饱和请求延迟 fetch → 提交后立即消费流 | 后续 Schedule 阻塞时，前面请求的 fetch 仍可推进，避免 30 秒 context 超时成为唯一出口。容量、流量形状和 p2/p3/p4 不变。 |
| FR-8 admission gate | 核对后不改错误文本断言 | 最新失败行 Schedule 200、流成功，根本没有待适配的错误 message。不能通过把成功当拒绝修绿。 |

## 诊断先行证据

修改前提取的逐笔数据：`~/code/case-refactor-reports/2026-09-10/case-side-followups/diagnostic-rows.json`。
来源为 `full394-integration` 中相应实例的 `admission-wave`、`admission-drain` 和 HA traffic 日志，原始结果保留。

FR-2 的相对时间线（两个 profile 同形，单位秒，取代表样本）：

| 请求 | 开始 Schedule | 结果 | 耗时 |
| --- | ---: | --- | ---: |
| 1 | 0 | 200 | 0.13 |
| 2 | 0.286 | 200 | 0.02 |
| 3 | 0.459 | 8511 | 1.507 |
| 4 | 2.117 | 200 | 1.024 |
| 5 | 3.292 | 8511 | 1.504 |
| 6 | 4.947 | 200 | 1.192 |
| 7 | 6.290 | 8511 | 1.509 |
| 8 | 7.970 | 200 | 1.222 |

一笔执行、一笔等待。第 3 笔的 1.5 秒期限早于第 1 笔约 3 秒完成；第 4 笔赶上约 3 秒的腾位。后续依此交替，接纳编号为 1、2、4、6、8。旧六笔断言失败后的清理取消不能归为服务本身失败。

FR-5：100 秒发射后共 2105 笔，110 秒完成约 592，120 秒约 684，200 秒约 1412；`OwnedHaClient.finish` 在等客户端进程结束。按约 9.1 笔/秒持续处理，剩余工作超过原收尾预算。把 replay_speed 设为 0.5 后，名义供给约 5.26 笔/秒；不改变执行容量，也不扩大超时。

FR-8：失败 `kv_pool_capacity::sb` 和 `prefill_waiting_cap::bw/sb` 的 probe 均成功；Schedule 等约 2.1 秒后获得释放的容量。`all_error_contains` 因“不是错误”失败，并非旧 token 与新 message 不匹配。需要另行重建能够真正抵达引擎拒绝边界的测试构造，不能从此证据宣称 mock 丢失错误消息。

第一轮额外定位到 `kv_full_shrink` 的恢复构造问题：两个终态分支通过后，恢复 20 笔中 20132、20133、20134 报 `LACK_MEM (602): decode KV growth failed`，成功率 17/20。恢复 helper 默认同时发 10 笔，每笔输入 2048、输出 2，在 1024 token/块边界上最终需要 3 块；缩容后仅剩 24 块池，10×3 超过容量。该变体在 YAML 显式设置恢复并发 7，使 7×3+ceil(24×5%)=23≤24；仍提交 20 笔，成功率仍要求 0.95，容量与其他变体不变。此处参数化只控制测试侧发射并发，不改变 master inflight 或 prefill 执行并发。

## leader spill 阈值复核

两族各 10 块，每个 P 容量 12 块。已有 10 块时只有 2 块空余，另族完整写入 10 块至少替换原族 8 块；之后原族已不满足完整前缀命中。重复跨族路由可以持续消耗缓存，不能从容量推出正的全局命中率下界。两台引擎都保留两族的部分块也是可能的，“至少持有一部分”的复制份数可以达到 2，并不违反每台仅 12 块的物理上限。

但当前这两个旧变体明确将 `saturation_hit.M3` 注册为 finding 健康断言。将 0.4 降至观测到的 0.25，会把已知盲区改报为 resolved，而不是修复构造。因此本轮修 fetch 编排，保留 M3 健康阈值及复制断言原值；不按失败样本制造通过。p2/p3/p4 全部配置保持不变。后续如要将旧变体改为“已知缺陷下的性能基线”，应明确更换测试语义并重做校准，不能与健康探针混报。

第二轮 `leader_spill_nonbatch::single-nonbatch` 实测 M3=0.4375，已超过原 normal=0.4；第一轮同实例为 0.25。这直接否证了“0.4 物理不可达”的前提。第二轮仍因复制均值 2.0>1.75 失败，不能把这个独立健康断言静默改为 PASS。

## 验证

定向执行与完整结果归档在 `~/code/case-refactor-reports/2026-09-10/case-side-followups/`，逐实例对照见该目录 `REPORT.md` 和 `summary.json`。

- 完整 Python 单测：867 PASS；YAML 编译 394 实例，实例 ID 集合不变。
- 110/111/133 三台单 lane 实测，745 个运行文件的 SHA-256 全部一致；没有运行产品改动版本。
- r1：30 实例，25 PASS / 5 FAIL；r2：30 实例，26 PASS / 4 FAIL。FR-1、FR-2、FR-4、FR-5、FR-6、isolation 的目标实例连续两轮通过。
- kv_full_shrink 在 r1 暴露恢复并发算术问题；修正后 r2 与追加守卫轮连续两次整例 PASS。drain_ok、drain_timeout、恢复成功率契约均通过。
- 四个旧 leader_spill 的两轮均完成指标裁决，保留 M3/复制健康断言的 FAIL；batch 不再以 Fetch ERROR/TIMEOUT 中止。
- 第二、第三份附件的 R2.1 优先级期望和首 filler 索引等待已包含于基线。现存三个 R2.1 目标、四个 isolation profile 在本轮重复验证通过；不恢复已退役的 error_code_family。

守卫轮状态与原全量逐项列在归档报告；不能以 runner 退出码或仅“代码未改”宣称每个随机时序结果都与旧全量相同。

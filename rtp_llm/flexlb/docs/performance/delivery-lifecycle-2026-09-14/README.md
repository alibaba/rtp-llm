# Delivery 生命周期重构验证

验证基于 `feature/flexlb_config` 的工作区快照；验证时这些改动尚未提交。

## 改动与边界

- DeliveryClaim 收回 RequestSlot，只暴露异步 `complete(result)`；没有新增 Slot 状态字段。
- 准备和交接使用现有具体事务，删除泛型准备回调和交接 BooleanSupplier。
- BATCH 在最终成员确定后设置预测；NON_BATCH 在一次锁内决策中完成预测更新与确认。
- 合并多层 ACK 包装和确认数据结构，删除重复资格检查、临时 snapshot、无用布尔返回值及 permit 的 Future 成员。
- Slot 显式选择响应，Publisher 不再通过 permit 回调请求决策。响应竞争时点、Endpoint 资源结算与锁外执行边界保留。
- 调用关系见 [架构说明](../../architecture/request-ownership.md)。

## UT

| 范围 | Tests run | Failures | Errors | Skipped |
| --- | ---: | ---: | ---: | ---: |
| 本地完整 UT | 1517 | 0 | 0 | 1 |
| 远端完整 UT | 1517 | 0 | 0 | 1 |
| 最终清理后的远端相关 UT | 260 | 0 | 0 | 0 |

完整 UT 后，收尾删除了无人消费的布尔返回值和重复投递资格检查；最终源码重新运行相关 UT 和以下性能矩阵。
新增回归验证非法 batchId 不触碰 Endpoint、空结果不消费凭据、重复结果不改变确认结果。
现有取消/投递交接、早到 Engine 接受、发布竞争、抢占、部分 BATCH 成员丢失后预测重算测试保留。

## 750P/750D

环境：`luoli.hn@11.163.39.110` 的 `luoli_gpu` 容器，Java 21。
使用独立目录 `/data0/luoli.hn/work/rtp_llm_4/flexlb-delivery-validation-20260914`，保留远端原工作目录。
运行 `tools/run_queue_performance.sh`：BATCH/FIXED_WINDOW、NON_BATCH/SINGLE，3000/10000 QPS，10 秒预热、10 秒测量、8 GB heap。
这是 loopback Master E2E 测试。前后顺序运行，相同测试代码与门槛。

下表均为 **修改前 → 最终修改后**，延迟单位 ms。客户端 avg 来自已有纳秒样本，在计时区间结束后计算；不将预热样本混入测量结果。

| 模式 | 目标 QPS | 实际客户端 QPS | 客户端 avg | 客户端 P50 | 客户端 P99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BATCH | 3000 | 2996.8 → 2996.8 | 22.333 → 20.683 | 10.975 → 10.913 | 249.307 → 250.293 |
| BATCH | 10000 | 9989.1 → 9989.1 | 29.773 → 20.679 | 11.094 → 11.029 | 277.330 → 205.298 |
| NON_BATCH | 3000 | 2999.3 → 2999.7 | 1.602 → 0.826 | 0.485 → 0.464 | 43.605 → 13.288 |
| NON_BATCH | 10000 | 9879.7 → 9996.0 | 4.085 → 1.968 | 0.493 → 0.418 | 85.015 → 50.712 |

Master 指标来自现有毫秒直方图，`<1` 表示低于 1 ms 的桶，与客户端原始纳秒样本口径不同。

| 模式 | 目标 QPS | Master avg | Master P50 | Master P99 |
| --- | ---: | ---: | ---: | ---: |
| BATCH | 3000 | 17.430 → 17.226 | 10 → 10 | 249 → 250 |
| BATCH | 10000 | 14.493 → 13.038 | 10 → 10 | 95 → 69 |
| NON_BATCH | 3000 | 0.063 → 0.030 | <1 → <1 | 2 → 1 |
| NON_BATCH | 10000 | 0.348 → 0.238 | <1 → <1 | 11 → 8 |

模式退出码：修改前 `{'BATCH': '1', 'NON_BATCH': '0'}`；修改后 `{'BATCH': '1', 'NON_BATCH': '0'}`。0 为通过，1 为未通过。

- 修改前：Master batch wait P99 242 ms exceeds ceiling 50 ms。
- 修改前：client E2E P99 277.330 ms exceeds ceiling 250.000 ms。
- 修改后：client E2E P99 250.293 ms exceeds ceiling 250.000 ms。
- 修改后：delivery wait P99 52 ms exceeds ceiling 50 ms。

性能不能标记为全部通过；保留原始门槛，不忽略失败。源码快照间同时包含工作区并行的 Response 防御性复制修改，单轮时序数据不足以将差异归因到 Delivery 重构。
为避免挑选结果，保留清理前一轮修改后数据 [after_previous-summary.log](after_previous-summary.log)；最终结论使用当前源码的重跑结果。

## 证据

- [机器可读指标与 UT 汇总](results.json)
- [修改前输出](before-summary.log)、[最终修改后输出](after-summary.log)
- [源码快照与 SHA-256](source-manifest.json)
- 完整远端日志：上述独立目录中的 `after-ut.log`、`after-final-cleanup-ut.log`、`before-perf-final/`、`after-perf-cleanup/`。
- 本地完整日志：`/tmp/flexlb-delivery-full-ut.log`；最终相关 UT：`/tmp/flexlb-delivery-final-eligibility-ut.log`。

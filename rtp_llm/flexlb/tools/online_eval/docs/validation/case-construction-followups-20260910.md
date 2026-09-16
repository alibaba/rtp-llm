# Case 构造与观测修复（2026-09-10）

基于 `f43708ccbe`。只修改测试侧 Python、YAML 和测试模型；master、真实引擎及 Java mock 均未修改。参数及新增观测预算全部由 YAML 提供。

## 摘机前确实存在 pending

旧实现串行等待每个 Schedule 返回，直到累计四笔请求曾落到目标 P。每 P 仅允许两个在飞 batch，因此后续 Schedule 必须等前一笔完成；历史落点减去当前 running/waiting 不能证明 pending，反而把完成请求误算成 pending。

新实现发六笔异步请求，间隔 0.3 秒，不等待被 park 的 Schedule。两台 P 各两个活动槽，先占四槽，至少一笔 Schedule 必须已调用但未返回。同时检查目标 P 至少两笔已落点、引擎完成计数未增长、目标 P 客户端尚未终态。请求形状和 8 秒 prefill 注入保持不变。末次引擎采样之后重新读取客户端记录，避免使用采样前的 pending 状态。

两个普通变体仍要求目标请求在 40 秒内出现明确终态；`zero_errors` 额外要求全部请求成功，不降阈值。它可以在构造有效后继续 FAIL，不能把这类红灯解释为构造失败。

独立 BW/SB 单测模型中，超出四个槽的 Schedule 真正阻塞到摘机，不再伪造已准入响应。容量绕过、提前完成、错误流协议等负例仍必须失败。

## 崩溃注入确实命中每台目标引擎

`no_resurrect` 原本以 `rpc_counts.enqueue_batch + 1` 配置 crash_after，但该 RPC 总计数不等于 Java mock 真正执行崩溃检查的 enqueueCount；部分提前拒绝只增长前者。再经 master 路由发送触发流量，还可能完全绕开目标或一直等待容量。

用现有 `first_enqueue` 注入模式设置最低触发计数，再逐目标直接发送 P-only EnqueueBatch。只在快照证明全部目标 stopped 后继续。意外准入会取消该探针并使构造报错；仅 UNAVAILABLE 可作为断链响应，其他 RPC 错误不吞掉。重启后旧请求不得复活的断言不变。

## 引擎 admission 与 master 等待分开观测

两笔占位仍经完整 master→P→D 流程发送，必须先观察到 waiting cap=1 或 held_blocks≥16。负向探针改为直连目标 P 的 EnqueueBatch，绕过 master 的容量等待；池容量和并发限制不变，prefill 延长到 8 秒以保持观测期间的占位。

要求返回恰好一个对应 request_id 的 error、零 successes，并核对 YAML 指定的原始 error code：waiting backpressure 为 0（当前 mock ACK 只填 message），KV LACK_MEM 为 602。空 ACK、意外成功、错误 RID、错误 code 和基础设施异常不能冒充合法拒绝。意外成功立即 Cancel。保留原来的 <3 秒拒绝时限、占位完成和容量恢复断言。

KV 原始消息为 `LACK_MEM: insufficient KV cache blocks (...)`。删除 master 包装层才添加的 `enqueuebatch rejected` 字符串要求，仍同时检查 LACK_MEM 与 insufficient KV cache，不用通配错误文本。

## Failover 稳态边界

`standalone_a_to_b` 与此前修好的 all_masters_down 一致：稳态采样截止到 kill 前 1 秒，仍要求 target_share=1。切换窗口和切换后的成功/落点断言均保留。避免按请求发出时间选入稳态的在飞请求，在 kill 后合法切换 B 却被算作稳态错路由。

## 风暴采样不阻塞发射

每笔请求前同步 HTTP snapshot，以及窗口末尾同步 snapshot/JSON 写盘，会占用后续发射间隔。改成单线程、最多一个未完成任务的异步采样器，发射线程只读取最新完成快照。相邻窗口保持原 cadence；drain/改速等阶段间的静默期结束后重新采样，再启动下一阶段时钟。

YAML 声明 `max_sample_age_s: 1`，小于 2 秒窗口，按采样开始时间而非返回时间计算新鲜度。过期、快照异常、真正漏过完整发射间隔、并发上限耗尽仍使构造 ERROR。命中指标仍取引擎请求完成事件；缓存计数和持有量取引擎快照。保留各拓扑原有 band 和 finding 裁决，未通过降低阈值制造通过。

单测证明阻塞的观测任务不会阻塞发射或无界堆积，过期和 RPC 错误会显式失败，真实漏发仍被拒绝。远端另以 6 lane 同时运行 P=2/3/4 和三个回归实例，并与一个既有 lane 共机，验证并行负载下的行为。

## 验证工件

本地归档：`/Users/wangziyi/code/case-refactor-reports/2026-09-10/case-construction-followups/`。最终状态、重复运行、原始失败和源文件 hash 见该目录 REPORT.md。环境失败与 case 判定分开记录。

最终验证：870 项 Python 单测全部通过；场景目录编译 394 实例。最终 27 个定向远端实例为 23 PASS、3 FINDING-CONFIRMED、1 FAIL，无 ERROR/TIMEOUT。唯一 FAIL 为 zero_errors 中摘机后明确的 Schedule 拒绝，6 笔成功 5 笔；原零错误契约保持。P=2/3/4 在额外 6 lane 验证和最终并行复验中均为 finding-confirmed，恢复均为 3 窗口。

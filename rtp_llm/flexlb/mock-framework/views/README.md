# 默认大盘

唯一线上参考是 [Whale-LLM-New【禁止修改】](https://grafana.alibaba-inc.com/d/iZ5Q_81nz)，本次读取版本 514，共 41 个 row、703 个 panel。Master 只参考其中的 `FlexLB · 01` 至 `07`，沿用其问题导向的分区和标题；不再参考独立 FlexLB 旧大盘，也不搬入 `FlexLB · 08` 的兼容旧指标。逐项来源与导出状态见 [coverage.md](coverage.md)。

`default.json` 是原生 Prometheus Grafana JSON，所有 row 默认折叠，子面板实际放入 row.panels，展开后才加载该组查询。没有按某一次实验挑选指标。缺失数据不补零；待埋点标题说明的是源码缺口，不是这次实验流量为零。

## P / D 与 ViT

Prefill 和 Decode 分开提供吞吐/延迟/负载、PD RPC、本地 KV、KV 分池指标和 Mock 模拟质量分区。Decode 展示按步耗时估算的 TPOT，Client 展示按请求实际到达时间计算的 TPOT。Prefill 的缓存复用、输入 TPS 不混入 Decode；Decode 的输出 TPS、生成 Batch、输出长度不混入 Prefill。每条 Engine 查询显式使用 `role="prefill"` 或 `role="decode"`，再叠加全局 Engine 地址筛选。Mock 专有指标单列，不能当作名字相近的 real 指标。

线上 Engine 查询主要使用全局 `hippo_role`，并非每个面板都硬编码了 Prefill/Decode。源码语义、线上 PD 分离分区和局部业务过滤共同决定本地归属；Master 的 `role=PREFILL/DECODE/PDFUSION` 是被观察 Worker 的角色，不是 Master 自己的角色，保留原查询的具体限制，不能替换为 Engine 自报的小写 role。

ViT 已接入共享 vision engine：QPS、成功/错误、图片数、Embedding 长度/请求等待耗时、队列深度/等待时间、Forward/合批大小和缓存命中/未命中/合并等待/淘汰/驻留 Tokens/Bytes。使用 real 的 `py_rtp_vit_*` 名称，耗时单位为毫秒。Forward、就绪排队加 Forward（py_rtp_vit_embedding_rt）和请求总耗时（py_rtp_vit_request_rt）分别记录；缓存命中不制造 Embedding/Forward 样本。真实 URL 图片准备提供总耗时、下载及非下载耗时（ms），成功读取另提供 fetch 耗时（μs）。非下载部分测量读尺寸、内容摘要和几何计算。跳过读取时这些面板无样本。预处理子进程、CUDA Graph、GreenNet 尚未模拟。py_rtp_vit_request_rt 覆盖阻塞媒体处理调用的成功或失败，包括下载和共享结果等待，不含异步提交 ACK、RPC 传输及输出编码。py_rtp_vit_preprocess_queue_size 统计准备尚未完成的媒体（排队＋执行中），不包含等待计算或发布的任务；没有独立 CPU 子进程，因此 dispatch/return 指标保持缺失。

## 已实现与缺口

Engine 的 waiting/running、Device 可用 Tokens、Device/Host 容量 Blocks、Device 不可用比例，以及 P/D TPS、终态 QPS 已接入原 RTP 指标名称及对应原面板。使用率为 0–100；used Blocks=total-free，不可用比例=(total-available)/total，不能混淆。Host 暂缺 available 与其引用分类，相关缺口保留。

真实事件的辅助 Counter/Histogram 使用 rtp*llm*\*，包括接纳/终态、缓存查找/命中、Forward Tokens 和实际耗时分布；这些是本地 Prometheus 增补，不声称 real 已有完全相同的 Counter/Histogram 名称。Python 接口同步消费新名字。计划 Forward、ViT 模拟运行图片数及 hash 预算保留 mock\_ 前缀。实际观测归回 P/D 的 RTP row，Mock 模拟质量各保留实际/理论、偏差、模拟执行时间占比三个面板，不将最后一项当作 GPU 利用率。

TPS 保持 Real 的不同统计合同：P context TPS 是有效执行速率；P context wall TPS 是实际窗口墙钟速率；`generate_tps`、`total_tps` 是报告窗口 token 数，不是每秒速率。D 使用停止截断前、包含已计算取消成员的验收 token（含 bonus）。本地把这两项标题标为“报告窗口 token 数”；真实交付吞吐和一分钟墙钟 TPS 使用累计 Forward token Counter 的 rate。不同报告长度的原生窗口计数不能直接对比。各 priority 的分子共用整批执行时间；View 和 Prefill 门禁先合并同实例 priority，再取实例或时间均值。无完成结果的在途长 Forward 不报假零、不清空 TPS 窗口；终态 QPS 独立更新。ViT embedding TPS 按成功实际计算的 embedding token 累计，不重复统计缓存或 in-flight 消费者。

旧实验没有新导出，需使用其保存的原始大盘；本次不改写旧 TSDB，也不生成旧名字别名来伪造新实验结果。

已补逐请求等待/首 Token/总耗时、输入/输出长度与迭代数、逐批 context/generate Batch、执行 Tokens、最长序列；Decode KV loading 记录真实传输开始至完成。仍缺完整 PD 传输延迟/字节/失败、KV 分池引用/淘汰/Host 命中读写等细分埋点。大盘保留线上相应关键指标的入口，但不拿 running 请求数冒充 batch size，不拿 Device 命中冒充 Host 命中，不拿 Fetch 等待冒充 compute 或传输耗时。实际/理论 Forward 最近值和 Histogram 是不同口径，分开展示。

Master 的已有 Reporter 比 Engine 完整，所以可用曲线多。线上 RTP-LLM 大盘还包含本分支尚未导出的 Master 新指标、Frontend 和 Pod 监控，这些明确标为待埋点；原生 Micrometer JVM 资源另外补充，不能把 KMonitor 的不同指标名当成本地已导出。

## Engine 资源归属

每个 Engine 运行在独立 JVM 中。请求、队列、KV 逻辑容量和 Forward 指标按实例导出。Engine 的 CPU、RSS、堆、GC、线程及 FD 暂未接入本地大盘。GPU 资源指标来自真实设备。

## 筛选、标签与事件

Engine、Master、Client 地址筛选各自作用于对应上报者；普通数量/状态按实例平均，明确的集群速率保留总量，Histogram 分位数合并 bucket，不能平均实例 P99。Master 上报中的 engineIp 是被观察 Worker，不能与上报者 instance 混用。

本地大盘直接使用 Prometheus 的 `instance`（IP:port）作为上报者身份，不在 Java 指标中新增或改写 `host_ip/container_ip`。KMonitor 的 `[[tag_…]]` 图例转换成 Prometheus `{{label}}`，业务维度按查询实际保留。

Prefill、Decode、ViT 各有独立的地址选择器，地址来自保存指标的 role 标签，不按 IP 推测角色；没有该角色数据时隐藏对应选择器。三者分别支持多选：`All · 聚合全部` 按角色聚合；选择一个或多个具体地址则按实例分组；“逐实例全选”显式选中所有地址，仍按实例展示。全选地址与 All 是两种不同意图，不自动互相转换。未选择任何地址时回到聚合全部。Master、Client 的自建面板也支持同样的地址多选切换；平均、总量、成功率和 Histogram 的原有计算口径不变。

隐藏变量 `prefill_group`、`decode_group`、`vit_group` 分别依赖同角色的 `*_instance`：All 的值为 `.*` 时返回 `role`，具体地址返回 `instance`。Engine 查询在 `by(...)` 与图例中使用对应角色的分组变量，保留 le、state 等业务维度。跨 P/D 的 Worker 面板拆成两组独立 target，保持两侧选择互不影响。变量查询仅对 `vector(1)` 操作，不扫描业务数据，不新增上报标签；报告页和独立 Grafana 使用同一规则。Client 自建面板使用 client_group（All=role，具体地址=instance）；Master JVM 补充面板使用 master_group（All=application，具体地址=instance）。JVM 内存先按 instance/application/area 汇总池，再按当前分组平均，不能先跨 JVM 求和。以后迁入面板时必须同时适配查询分组与图例。采集完整性的 up、采集时长、样本数始终逐目标展示，以免平均值隐藏失败目标。

PromQL 中地址变量使用 `${prefill_instance}` / `${decode_instance}` / `${vit_instance}` / `${master_instance}` / `${client_instance}`，由 Prometheus 数据源完成正则与字符串转义。显式 `:regex` 格式化会绕开数据源的字符串转义，含点号的具体 IP 可能产生非法 `\.`；只测试 All 的 `.*` 无法发现这个错误。查看服务也会规范化旧报告中这些受管变量及旧版 engine_instance 的旧写法，原始报告不被修改。

标签沿用物理来源 HIPPO_APP/HIPPO_ROLE/HIPPO_SERVICE_NAME/HIPPO_SLAVE_IP/RequestedIP；本地不冒充 Pod IP。Engine 只支持 DP rank 0。业务指标与 Mock 专有指标保留原名，Prometheus 与线上 KMonitor 查询协议仍需分别适配，不能承诺直接导入同一查询 JSON。

页面上的“事件线”可全选、清空或逐项选择 Step。筛选控制图表标注的显示，完整时间线和门禁结果仍可查看。

## 2026-10-30 Engine 埋点边界复核

来源代码：`cpp/metrics/RtpLLMMetrics.cc` 的 Stream/Executor Reporter、`cpp/engine_base/stream/GenerateStream.cc::reportMetrics`，以及 `multimodal/mm_scheduler.py`、`mm_process_engine.py` 和 `mm_embedding_cache.py`。线上 RTP 大盘仍是 v514。只改新 Mock 生产者和本地大盘，不修改 real。

| 边界              | 对齐指标与含义                                                                                                                                      | 明确限制                                                                                                                                                                                                                                                                         |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 调度选批后        | `rtp_llm_wait_stream_size`、`rtp_llm_running_stream_size`、`rtp_llm_loading_cache_stream_size`                                                      | 区间内按轮次求均值。Waiting max/agg 取各实例轮均值的最大值。KV prepare 为同步操作，原生 KV loading 曲线为 0。                                                                                                                                                                    |
| Forward 组装      | `rtp_llm_context_batch_size` / `generate_batch_size`，`execute_token_size`，`max_seq_len`；含 Context 的条件系列只在 P 上产生                       | Decode 执行 Token 取 verify/step，不拿整段历史输入相加；不把 running 请求数当作本批大小                                                                                                                                                                                          |
| 执行步            | P/D `rtp_llm_step_latency_us`                                                                                                                       | 完成的非空执行步，从调度返回到处理完输出，微秒；包含 Mock 执行和输出处理                                                                                                                                                                                                         |
| 每输出 token 耗时 | D `rtp_llm_sp_estimate_tpot_us`、Client `mock_client_tpot_seconds`                                                                                  | D 按输入组装至计算完成的耗时和本批截断前验收 token 数估算，参与 stream 包含入批后取消但计算完成者。Client 按成功请求首尾 token 到达间隔计算。                                                                                                                                    |
| 请求终态          | `rtp_llm_latency_us`、`wait_latency_us`、`first_token_latency_us`、`input_token_length`、`output_token_length`、`iterate_count`、`query_batch_size` | 如 real，仅 success/cancel/timeout 产生流统计；无首 Token 则不报首 Token 耗时；query batch 是 beam 数，当前为 1                                                                                                                                                                  |
| D Loading         | `rtp_llm_stream_loading_cache_latency_us`                                                                                                           | 从实际 beginTransfer 到 transferCompleted，不含等待 P 计算；其他 loading 阶段不混用                                                                                                                                                                                              |
| ViT 调度          | `py_rtp_vit_embedding_queue_size` / `queue_wait_rt` / `batch_size` / `forward_rt`                                                                   | 队列项是提交的工作 chunk。入队、出队和进入 Forward 时记录剩余队列，准备队列在任务进出时采样。                                                                                                                                                                                    |
| ViT 请求          | `py_rtp_vit_qps` / `success_qps` / `error_qps`，`image_count` / `embedding_length` / `embedding_rt`                                                 | QPS（source=mm_embedding）与成功数按去重后的内部工作统计，含异步提交；缓存和共享消费者不重复计。错误按首次观察计数，也包含入口拒绝。图片数/长度按阻塞请求采样；request_rt 带 source=get_embedding_result、status=success/error。纯 hash 命中不产生计算请求，没有 Tensor 数据搬运 |
| ViT 缓存          | `py_rtp_vit_embedding_cache_{hit,miss,inflight,eviction}_qps`、`tokens` / `bytes`                                                                   | GPU+CPU 逻辑驻留合计，不能作为真实 GPU/JVM 物理占用                                                                                                                                                                                                                              |

事件耗时在 KMonitor 逐样本上报，Prometheus 由累计 `_sum` / `_count` 计算加权均值；无事件时不发新的耗时样本。逐请求 QPS 从终态事件累计量计算速率。它们与 KMonitor 的固定上报周期并非逐点相同。

## 事件指标的权重

事件类均值统一由每个实例的累计 `_sum`、`_count` 计算。All 时分别汇总各实例增量再相除；选择具体实例时保留 instance 分组，用同一公式。禁止先计算各实例均值再 avg，也不使用已经删除的裸事件 Gauge。需要 token 总量或命中率的面板使用 `_sum` 增量，不能把样本数当成 token 数。

原来使用 max 聚合事件窗口均值的面板，明确标为“实例均值最大值”，展示各实例在查询区间内事件均值的最大值；这不是单个事件峰值或 P99。每请求图片数和 Embedding 长度改为请求加权均值，不再把各实例均值相加。状态 Gauge、周期 TPS/QPS、Client Histogram 保持各自口径。线上 KMonitor 大盘不修改。

Forward 平均耗时、输入、计算和输出长度均先汇总各实例的耗时或 token 增量，再除以 Forward 次数增量之和，保留批次权重。原生周期 TPS 的实例均值、集群墙钟 TPS 合计和按批次加权长度是不同统计量，标题和说明必须写清，不能相互替代。

P/D 的 DEVICE 容量指标使用 `test_block_num - 1` 及其驻留占用，不使用统一缓存保留总量。`rtp_llm_kv_cache_item_num` 与 CacheStatus key 列表来自统一索引；设置更大的 `mock.cache.total.blocks` 后，索引条目数可超过 device 总块数。缓存命中仍受 device 驻留额度约束，不模拟分层搬运耗时。

事件 QPS 与 ViT embedding TPS 在 Prometheus 中使用原名加 `_total` 的累计 Counter，View 用 `rate(...[1m])` 显示每秒速率；重复读取不清零。KMonitor 原名与逐事件上报口径保持不变。原生 RTP 窗口 TPS Gauge 不适用此换算。

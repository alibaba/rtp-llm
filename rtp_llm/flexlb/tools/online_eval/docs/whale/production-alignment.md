# Whale Mock 生产配置与监控对齐

先在 Whale 测试部署运行与真实服务对应的虚拟集群，再用同一时间窗的线上指标校准。开发机回放用于可重复诊断，不能替代 Whale 发布验证。生产对象只读；模板、zone、发布和 Pod 调整仅操作获授权的测试对象。

## 发布与生效证据

1. 固定源码 SHA，CI 构建完整 `mock-bundle` 镜像，保存 run、job、最终镜像和 SHA。构建失败或未运行不能继续按成功处理。
2. 读取测试部署实际 `biz_version` 对应的模板和所有 zone overrides。模板当前最新版本不一定是部署生效版本。
3. 新建测试模板版本并更新镜像；保留生产模板不变。清除遗留 `MOCK_BUNDLE_JAR_DIR` 自定义 JAR 覆盖，否则换镜像仍可能执行旧代码。检查启动命令与镜像入口一致。
4. 更新下表各项配置，再发布测试部署。读回 Carbon plan、镜像、角色健康与 frontend readiness；API 接受不等于发布完成。
5. 在 Master Pod 核对运行时 master-source-config、performance、发现文件和实际逻辑引擎数量。保存配置摘要与哈希，避免归档凭据。
6. 单独标记发布、冷启动和预热区间；预热后才开始正式对照。观察实际缓存命中、排队与容量是否稳定，不能仅以固定等待秒数判定预热完成。

### 寄生部署的 frontend 健康检查

Carbon 的 `health_checker_config` 和 VIPServer 域名的 `clusters[].healthChecker` 是两套配置，两处均须核对。寄生模式不注册独立 P/D Pod，frontend 原生 `/health` 仍检查 P/D VIP，可能返回 503；此模式按运行验收约定使用 `/frontend_health`。只修改 Carbon 的 `CHECK_PATH` 不会同步修改 VIPServer 的 `curlPath`，会出现 `HT_ALIVE/WT_READY` 但 `SVT_UNAVAILABLE`、ready 为零、发布一直进行的状态。

VIPServer 调整只限测试部署实际引用的域名，保留其他注册参数。先读回域名配置，确认平台 API 对已有域名的行为后再执行；部署发布白名单不意味着域名注册 API 也在白名单。工具拒绝时保留原配置及具体未执行请求，不绕过确认。探针恢复后仍须独立验证 Master、Mock 240 个逻辑引擎（本次 48P/192D）及完整请求成功率；frontend 存活不等于性能门禁通过。

切换模型时同时核查 zone 的 `MODEL_TYPE`、`TOKENIZER_PATH`、`CHECKPOINT_PATH`。只清理前两项仍可能加载旧模型 config，例如 Flash 使用 GLM checkpoint 导致缺失 `compress_ratios`。清除旧覆盖后，检查最终 Carbon plan 与进程环境实际加载的资源，不仅查看 biz 快照。

## 每次必须对齐的配置

| 项目 | 对齐内容与验证方法 |
|---|---|
| 模型与 tokenizer | frontend 的模型类型、tokenizer、special tokens 和输入长度口径；来自其他模型的 tokenizer 会改变长度、prefix 分块及命中率 |
| Master 版本 | 新 Master / legacy Master、配置 schema、镜像内 pinned commit；预测公式与模拟执行公式分别记录 |
| 调度与组批 | BATCH / NON_BATCH、maxRequests、collection wait、early dispatch、每 P inflight batch、等待队列与超时、路由与缓存亲和策略 |
| 请求路径 | Master 直接向 engine 发 batch 或 frontend 接收地址后发送；`FETCH_OUTPUT_STREAM=1`、frontend Fetch 支持、schedule-only 开关及外部可达 Pod IP；不同路径不能混为同一实验 |
| P/D 规模 | 同时记录计划与 ready 的逻辑 worker 数。Decode 需乘 DP；partition 是同一 DP 组的物理分片时不能再乘一次。最终以 worker 发现和运行态验证 |
| CPU 与线程 | Master / Mock heap、Pod CPU/内存、线程池、端口数量。虚拟 engine 数量不代表实际 CPU 算力增加；记录宿主机节流和 GC |
| block / CP | 引擎物理 block、CP 是否分片、CP size、frontend prefix key 粒度、路由有效 block。不能直接把 frontend block 当成 D 的物理 block |
| Device 容量 | P/D 各自 pool blocks、分池、reserve ratio、最小 free blocks、并发限制；按运行态 pool 总块数确认，不能只由显存容量估算 |
| Memory 容量 | 实际 total blocks、CP/分池换算、读写与拷贝延迟、disk cache 是否开启；MB 与 blocks 之间换算必须有模型布局依据 |
| 驱逐与复用 | Device / Memory prefix tree、retain-on-read、copy lifecycle、state 独立驱逐、D reuse/device cache 开关；实现不支持的策略明确列为差异 |
| P 性能 | 与模型、硬件和 CP 对应的 batch 执行公式、compute/input/reuse token 定义、FIFO token/sequence 上限、force-single、噪声；用 model forward 与 context batch 对照校准 |
| D 性能 | 每 step tokens、step base/并发系数、DP 与并发上限、MTP；不要把 DP 同时乘在实例数和单实例性能上 |
| 输出长度 | 用成功请求输出长度分布校准 EOS，保留 max_new_tokens 和截断语义；失败样本的短输出不能用于拟合正常长度 |
| 流量 | 来源、采集起止、用户群体、输入/输出长度、到达率、prefix 分布。全量采集，播放时按长度过滤；同为 0–32k 不代表用户群体或 prefix 分布一致 |

`MOCK_BUNDLE_OVERRIDES_YAML` 控制规模、物理块和 pool；`MOCK_PERFORMANCE_CONFIG_JSON` 控制性能、FIFO、Memory、EOS 等；Master 配置由 `FLEXLB_CONFIG` 控制。修改后必须查看实际生成配置，不能只检查环境变量存在。

## 监控清单与聚合口径

每次保存 Grafana panel JSON，包括 metric、tags、aggregator、downsample、expression 和 transformation。以下 panel ID 来自 engine dashboard `iZ5Q_81nz`，使用前刷新；Master 补充 dashboard 为 `fv0CDqzDk`。

| 指标 | 面板/口径 | 用途 |
|---|---|---|
| Prefill context TPS | 10379，`rtp_llm_context_tps`，有效 compute token / 对应执行时间 | 主性能指标 |
| Prefill with-cache TPS | 10370，`rtp_llm_context_tps_with_cache`，含复用输入 / 对应执行时间 | 与 context TPS 同时观察，禁止称作单独 cache token TPS |
| Decode generate TPS | 10369，`rtp_llm_generate_tps`；对照真实 emitter 与 Mock 上报窗口 | D 性能；不与 frontend output TPS 混用 |
| TTFT / TPOT | frontend 对应面板，保留单位、均值和可用分位数 | 延迟护栏；不能平均各实例 p99 得出全局 p99 |
| Output length | 成功完成请求的实际输出长度及分布 | 检查 EOS、截断和成功率造成的偏差 |
| 实际缓存命中率 | 10077：同窗 `sum(rtp_llm_kv_cache_reuse_length) / sum(rtp_llm_input_token_length)` | 实际复用；不是 30min key 理论命中率；分母为零时显示无样本 |
| Device / Memory 复用 | 9205 / 9206 对应 expression，分别观察 device / memory reuse 与输入比例 | 区分复用来源，不把旧/new Memory 指标相加 |
| KV 空间占用 | 154，`rtp_llm_kv_cache_used_ratio`；并看 556 total、552 available、553 free 与 Memory total/available | 占用率与实际命中率是两项指标；可驱逐缓存与不可回收引用占用不能混算 |
| Batch size | 150 query、122 context、123 generate；Master batch size 另列 | 区分调度批与实际执行批 |
| P waiting / D waiting | 对应角色 `wait_stream_size` 面板及队列深度 | 区分 engine waiting、Master queue 和 cache loading |
| 成功 / 失败 QPS | frontend 完成口径、Master 按 code 分组分别保留 | Master 调度成功不能代替端到端成功；所有错误码、超时与未完成请求都要计入 |
| Model forward | 9101 中 `rtp_llm_model_forward_us`，按 panel 的 `global_avg` 聚合，统一转 ms | P forward 对照；同时保留逐引擎分布和有效实例数 |
| MTP Decode step | 真实 D 使用 speculative decoding 时核对 `rtp_llm_sp_step_latency_us` 及 `mtp_model_type` 标签 | 普通 D forward 为零不代表执行无耗时，不用零值拟合 D 性能 |
| Context batch size | 122 均值与 10099 max 同看 | 排查组批不足或异常大批，不能只看请求数 |

同一测试/生产窗口保留逐引擎曲线、引擎均值与合理的集群总量。真实 D 保留 dp_rank；寄生 Mock 全在一 Pod，必须按 engine / engine_port 拆分。逐引擎 TPS 与 Pod 总和不可比较；吞吐总量不能由混有 standby 的平均值替代。保留零负载点；监控缺点与零值分开处理。priority 的归并严格按 panel transformation 执行。

## 发现偏差时

先检查流量画像、实际 worker 数、配置生效、指标标签与窗口，再分析调度行为。使用 asish 实时解析测试 Master Pod 并打开终端，定位实际运行目录和启动进程，查看 Master / Mock 日志、已实现的只读 debug/metrics 接口。关联请求 ID、P/D worker、batch、cache 命中与等待时间，不仅依赖汇总曲线推断原因。

每轮只改变已明确的参数，记录前后配置与监控窗口。A/B 仅用于观察差异；绝对门禁独立判断 TPS、延迟护栏及 100% 成功率。流量不一致、未校准的模型参数、指标缺失或回放不达速均应标记待校准/无效，不能宣布生产门禁通过。

### 满缓存后的 Mock 宿主机开销

验收窗口必须覆盖 Memory pool 填满并持续驱逐的阶段。2026-09-22 在 48P/192D、每 P 52,295 个 Memory blocks 的测试部署中，旧实现每次驱逐遍历整个缓存并搜索后代；30 秒 JFR 的主要采样集中在 `MockMemoryBlockCache.hasResidentDescendant -> evictOne -> beginWrite`，Mock JVM 长时间消耗约 30 个 CPU 核。重启后的短暂恢复不能作为校准成功证据。

修复 `30b0b8a387` 用可驱逐叶子的 LRU 索引及前缀子树计数维护候选，避免逐块全表扫描。发布后需要验证实际执行镜像、持续驱逐计数、CPU、成功/失败 QPS 和 TPS，不能以单元测试或 CI 成功替代运行验收。

当前 Mock P 的 `rtp_llm_model_forward_us` 上报性能公式计算的 `executionMs`，context TPS 的计时还包含实际回调与缓存处理开销。因此 forward 均值接近生产、TPS 却显著偏低时，应先排查宿主机 CPU、回调延迟、缓存操作和 GC，不能直接缩短公式耗时来掩盖执行开销。输出长度校准同样需要成功请求分布，不能仅通过缩短 EOS 抬高成功 QPS。

### 2026-09-22 Flash 测试部署校准记录

此记录是一次部署观测，不是生产门禁阈值。测试 deployment `6aa3b6979ffe080d001c3dbb` 使用模板 v10 / biz 170，CI 75163186 构建源码 `30b0b8a387` 的 mock-bundle；legacy Master 仍为 `ab73d2b6931d11dd5bc35993042f74fa53e54814`。48P / 192D（12×DP16），CP4，P/D block 为 512/128，Device pool 为 21553/221484，P Memory pool 为 52295。保持生产 BATCH 配置及 `FETCH_OUTPUT_STREAM=1`。

P 执行公式来自已记录的生产预测公式，`prefill.scale=1.23`；只缩放 Mock 执行耗时，不改 Master 预测公式。D 保持 `tokens_per_step=2.6`、`step_base_ms=19.5`、`step_per_running_ms=0.175`；几何 EOS `mean_tokens=400`、seed 20260922。输出长度分布仍需继续核对，此参数不是生产常量。22:00:39 重启后读回全部 48P 的 scale=1.23、runtime_override=false。

22:10–22:12:59 CST，按 panel 9101 的 global_avg 查询，P forward 时序点均值：生产 380.3 ms，Mock 382.1 ms；frontend 完成 QPS 为 1542.2 / 1535.9。22:02–22:12:59 的 12 个 Mock 错误码序列共 11 个分钟点均为零。22:10 有效 Memory 占用样本均值 98.6%，44 个 P 已发生驱逐，写入拒绝为零。这些是监控观测，不替代固定请求集合的 100% 完成核账。

仍未完全对齐：同窗 P context TPS 约 64893 / 58613，with-cache TPS 126681 / 112470（按 priority 分组后求和的角色均值，不能当集群 TPS）；实际 reuse/input 比例 48.4% / 45.6%，context batch 16.45 / 15.02。生产缺测/standby 实例需保留 coverage，不能填成零掩盖差异。frontend readiness 仍为 0/20，测试 VIP 探针修复未执行，Whale 状态仍 PUBLISHING；不宣布整体部署或绝对门禁通过。

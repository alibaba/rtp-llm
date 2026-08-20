# V3.2 A/B 实验台账（截至 2026-08-16）

## Staging 环准入实现（2026-08-19，session 接续后）
- **实现**（设计文档：staging-ring-admission-design.md）：
  - 引擎 C++（rtp-llm-rdma 工作区 +2 文件新增 +7 文件修改）：`initKVBlock` 封顶分配
    （block0+staging32+tail256+环64=353 块，中段 0 哨兵）→ `loadCache` 主通道只拉常驻位
    → 新增 `loadPrefixViaRing` 分批（64 块/批）环形拉取前缀 → `V32AdmissionStore`
    （引擎所有的 host pinned 镜像 + GPU idxp 池，含 30s 墓地延迟释放）
  - PB：`BroadcastLoadRequestPB.admission_ring_block_ids`（field 15，TP>1 时广播环块）
  - python 侧：v32_ctx.cu 新增 `ctx_admission_open/ctx_adopt/ctx_admission_release`
    （dlopen 引擎 so + dlsym C 导出，adopt 引擎缓冲为张量）；v32_capacity `_entry` 优先
    adopt；`_bookkeep_pool` 对准入即 offload 的行回落双波（主池无其 indexer-K 历史）
  - env：`RTP_KV_ADMIT_RING_BLOCKS=64`（0=关）；runtime rtp-b-offload-20260819-ring（r2）
- **Br-smoke 判决（64 请求，mixed W48 秒拒）**：长 1 ok / 8 fail，短 52/55
  - **链路全通**：19:12:04 admission capped（353 块封顶）→ mirror prepared（4.4GB host +
    0.5GB idxp）→ 环拉取 25s 完成 → python adopted → 解码 3.5min 完成 → release
  - 8 个长失败 = 路由聚簇（9 长全落 rank0）+ 容量（available=326 < need 353+reserve128）
    ——与代码无关，B0m 同型；准入需求已从 ~1030 块降到 353 块
  - **发现并修复 UAF**：引擎 stream 结束即 free 镜像，而 v32_ctx fetch 线程可能还在读
    → server.log 出现 malloc_consolidate 堆损坏 → release 改 30s 墓地延迟（r2）
- **Br0m 全量判决 run**（1000 请求，与 B0m 严格同配置）：20:03 启动，配对 B0m 长 1/128
- **Br0m 终值**：长 1/128（与 B0m 持平）、短 865/872——cap 生效（拒绝日志 need=353）但
  单 rank 空闲底线 150-453 < 353+reserve(128)=481
- **重大发现（dp_rank0 独占）**：decode DP8 实例只有 rank0 接请求（ranks 1-7 全程
  v32_sw reqs=0；instance_status 全部 dp_rank=0；LB decode 端点=27001 单地址）——
  **decode 有效 KV 容量 = 名义 1/8**，回溯解释 B2/B3/B4 全部容量怪象；修路由=8×，独立课题
- **Br3m（keep64+ring32）终值**：长 2/128、短 551/872（321 短 8211）、总 ok 553——
  细粒度准入在单 rank 池里把失败从长请求转嫁给短请求，池总量是硬约束
- 全部判决与结论：staging-ring-admission-design.md

## 深夜对齐 AB（2026-08-20 00:00-04:00，8-rank 分发修复后）
- **路由修复**：rtp_cluster.py `decode_worker_address()` 把 decode 展开成 8 个 per-rank 端点
  （FlexLB 把每 rank 当独立 worker）；`RTP_DECODE_RANK_FANOUT=0` 回退
- **Am8（A+分发）**：总失败 5.7%；**长 85/128 ok（66.4%）**、短 858/872——仅路由修复就让 A 的
  长请求从 1/128 → 85/128（8× 容量兑现）；剩余长失败仍是 8211（40 个）
- **Brm8（B环准入+分发）**：长 4/128、短 651/872、总 ok 655——8211 消失（准入不再是瓶颈），
  但 514 主导：255× CACHE_STORE_PUSH_ITEM_FAILED（并发环拉取打爆 TCP cache store，B6/B7
  失效模式非致命回归）+ 74× drain enqueue failed（无诊断）
- **块感知 master 调度**（FlexLB jar flexlb-api-blockaware-895edc9f）：
  WorkerStatus 本地记账/in-transit 校正按 offload 驻留量（env FLEXLB_DECODE_OFFLOAD_RESIDENT_TOKENS）
  估算 KV 需求；select() 对 DECODE 加"free KV ≥ 需求"软过滤
- **Brm9（+块感知 LB）**：长 6/128、短 619/872——改善甚微；终值 514 构成同 Brm8
  （292 push + 66 drain）。**部署事故**：r3 换 so 复用旧 pip 目录，环限流+诊断（BUILD4）
  实际未部署，Brm9 只测到了块感知 LB 单项
- **r4 部署（干净目录验证 sha）+ Brm10**（03:16 起）：环拉取并发闸门（1/rank，等待超时即败）
  + drain 全路径诊断——针对 push 级联的对症修复，最终判决 run
- **Brm10 终值**：长 3/128、总失败 39.8%——诊断揪出第三个深层 bug：**多 GPU 设备绑定**
  （gRPC 线程 device=0 vs rank r 池在 device r → ranks1-7 drain 全部 invalid argument，
  唯一成功环拉取在 rank0）→ r5 修复（入口按池指针 cudaSetDevice + RAII 恢复）
- **Brm11（全要素，r5）终值**：长 **39/128**、短 780/872、总失败 18.1%；drain 失败=0、
  环拉取 20+/26 完成、全夜零崩溃。对齐判决：Am8（A+分发）943 总 ok / 长 85/128 仍占优——
  63k@W48@8rank 对齐点上 8 池空间足以吃下 A 的全量准入，B 的常数准入优势要到
  128k 级/W96/更高长比区间才兑现
- **最终报告：集群AB对齐实验报告-20260820.md**

## 状态：Scheme A 基线定型；Scheme B 影子模式已部署 .78，验证中

## 配置
- 拓扑：3P(TP2×DP4) + 1D(DP8)，KV-TCP（RDMA MR 与 V3.2 多池不兼容，见下）
- 数据集：/tmp/rym/dataset/trace-0715-mixed5k（3500短<8k + 1500长63k/输出p50=2911），FORMAL_REQUESTS=1000
- decode KV=12288MB/rank（≈2457块，块=64tok；实测 KV=bf16 78KB/token——纸面账勿再用fp8 34KB）
- 全部环境开关在 v32_env.sh；驱动 run_v32_schemeA.sh

## A 基线判决（run v32a2-20260814T070614Z / T051729Z 交叉验证）
- 短：拒0.7%，TPOT 110/131ms，TTFT p50 1.9s
- **长：拒 99%（8211 DECODE_MALLOC_FAILED at PD handoff）**——reuse cache 开/关同结果
- 机制：长请求需 1030 块（80%池），分配不排队不逐出直接拒；短请求见缝插针 → 结构性饥饿
- B 判决指标：长拒绝率 99%→<10%、总 goodput、短 SLO 不回退

## 排障账本（11 修，全部固化）
启动超时3600(RTP_START_TIMEOUT)/renderer透传(RTP_WORKER_EXTRA_ENV)/driver时序/RDMA-MR→TCP(RTP_CACHE_TRANSPORT_TCP)/shm清理/.20假包(.local/rtp_llm.disabled-20260813)/PYTHONPATH解耦/DeepEP分支门控/缩进bug/.20 shm扩容400G/prefill旧分支
- **RDMA MR issue**：register user mr for block pool layout[0] failed（BlockPool.cc:633）——V3.2 多池布局不兼容现有 RDMA 注册，独立修复项

## Scheme B 设计（已评审）
- decode 节点：indexer-K 常驻 GPU(8KB/tok)，主 KV host pinned；每步 top-2048 gather H2D（~1.2MB/层/请求，PCIe 充足）；预取=上步索引
- 实现位置：runtime site-packages rtp_llm/models_py/modules/hybrid/mla_attention.py（纯 python，eager）
- 闸门：KVCacheManager 块释放接口是否暴露 python

## Scheme B 影子模式部署（2026-08-16，节点 .78）
- 源码：本目录 v32_offload.py（host pinned 镜像池 + gather + verify）、v32_offload_hook.py（monkeypatch MlaAttention._run_sparse_indexer）
- 部署：两文件拷至 .78 `/home/admin/rtp-hol/runtime/rtp-rdma-stepmetrics-ae442576d9f6/site-packages/rtp_llm/`；mla_attention.py 末尾追加 `import rtp_llm.v32_offload_hook`
- **恢复**：.78 `.../models_py/modules/hybrid/mla_attention.py.bak-pre-v32offload` 覆盖回去 + 删除两个 v32_offload*.py；或直接 unset V32_OFFLOAD_MODE（默认 off，hook 空转）
- 开关：V32_OFFLOAD_MODE=shadow|off、V32_OFFLOAD_VERIFY=1；仅 decode 形态批次生效（topk 行数==block table 行数，prefill 自动跳过）
- 判决点：63k 长请求 decode 全程 `VERIFY MISMATCH`=0 → B 数据通路正确，进入容量模式改造

## 影子模式判决（2026-08-16 00:56，PASS）
- 单机 .78 影子服务：冒烟 + 27.7k tok（1步）+ 55.4k tok（256步）
- 结果：**verify_ok=15000×2 rank（6/7），verify_fail=0，全 61 层覆盖** → host 镜像 gather 与 GPU 池逐位一致，B 数据通路正确
- 首轮 108 万 MISMATCH 为假阳性：DP 空转 dummy 步 block table 全 0，未镜像即校验；修复=过滤 phys==0 + 越界索引（_valid_rows）
- 影子开销：55k ctx decode ≈565ms/步（vs GPU 基线 ~137ms）——预期内，影子每步全量镜像+同步；容量模式改增量镜像后消除
## 容量模式设计（2026-08-16 定稿，路线=最小 C++ 改造）
源码定位：`/home/admin/project/rtp-llm-rdma` @ b7167df18 + step-metrics 工作区改动（本地与 .78 一致，即 runtime ae442576 的源）。
代码事实（决定设计）：
- `StreamCacheResource::tryReleaseKVBlock` 仅支持全量释放（`CHECK(nums==total)`，"Partial release not supported yet"）；但 `freeBatchBlocks(batch_id, blocks)` 已存在
- decode 准入：`DecodeRpcServer::allocateResource` busy-wait `initKVBlock`，malloc 失败立即 RESOURCE_EXHAUSTED（无重试/排队）
- 内建 memory connector（`reuse_cache && enable_memory_cache`）只做前缀复用，另有 tiered eviction（只逐出 BlockCache 缓存块，不动活跃流）——均不解决活跃长流驻留
- **indexer-K 与主 KV 同块**：kv_scale 区放 indexer 缓存（`SingleConfigCreator.cc:153` `(indexer_dim+indexer_dim/128*4)*spb`），块释放两者一起丢
- pybind 无块释放接口（已核实 compute_ops 全部导出）

方案（B 容量模式，mirror-then-shrink）：
1. **C++ 小补丁**（2 处）：a) `offloadPrefixBlocks(keep_last_n)`：置前缀块为 NULL 哨兵（保持向量长度→incr 记账不变）+ `freeBatchBlocks` 归还；decode 角色、seq>阈值（env RTP_KV_OFFLOAD_MIN_SEQ）的流在 step≥2 自动收缩至 RTP_KV_OFFLOAD_KEEP_BLOCKS；b) 准入 malloc 失败带窗重试（env RTP_DECODE_MALLOC_RETRY_MS）
2. **python 容量路径**（hook 扩展）：step1 全量镜像主 KV→host（已验证）+ 把 indexer-K 拷入 python 持有 GPU 侧张量（8KB/tok，63k≈520MB/req，即 ECHO 的 GPU 驻留成本，~10× 收益）；step≥2 对 offload 流：python 精确复算 indexer 打分+top-k（侧张量，~0.2ms/req/step）→ host gather top-2048 主 KV（~144MB/req/step，预取优化后 ~10%）→ python MLA 稀疏注意力（latent 512+rope 64 两个 matmul+softmax）覆写该行输出
3. 语义无损：indexer 分数与注意力均为精确复算，可用影子模式对拍验证（P-B1 先做，不动 C++）
局限（如实写报告）：准入仍需瞬时全量块（收缩前），并发长请求靠重试窗排队消化；ECHO 原生在传输层直落 host，无此瞬时峰值

执行序：P-B1 影子对拍 python indexer/attention vs kernel → P-B2 C++ 补丁+重建 runtime → P-B3 单机容量验证（长请求正确输出+显存下降）→ P-B4 集群 B 跑 mixed-1k kv12g → A/B 报告

## P-B1 判决（2026-08-16 12:39，PASS，全通路可 python 精确复算）
- **attention**：SparseMlaOp.forward 级复算（gather 全局槽位→qn·kv[:512]+qr·kv[512:]→softmax×scale(0.135234)→probs@kv[:512]），vs kernel cos=1.00000（81 checks）
- **indexer top-k**：score=Σ_h w_h·relu(q_h·k)，vs kernel overlap 0.9995–1.0000（norelu 0.77→relu 为正确公式）
- **关键布局事实**（mla_quant_kernel.cu:122-131）：indexer-K 缓存分段块布局——块内 [64×128B fp8][64×4B **float32** scale]（非 132B/token 交错；scale 非 ue8m0 打包 int）；slot_mapping=全局物理槽位（block×64+off）
- topk 索引=request-local 逻辑位置（fmha 侧 _convert_topk_indices_to_global 转全局）；教训：影子 gather 对拍两侧同映射，无法区分索引空间——须以消费端 kernel 语义为准
- 对拍工具固化在 v32_offload_hook.py：V32_VERIFY_TOPK=1 / V32_VERIFY_ATTN=1 / V32_OFFLOAD_MODE=shadow+V32_OFFLOAD_VERIFY=1

## P-B2 C++ 补丁（2026-08-16，rtp-llm-rdma 工作区，全部有 .bak-pre-v32offload 或 git 可恢复）
- StreamCacheResource.{h,cc}：offloadPrefixBlocks(keep_last_n)——group0 前缀块别名到保留块 0（kernel 读合法内存；python 以 phys==0 识别 offload 区）+ freeBlockList 归还；incrKVBlock 入口按 env 触发一次（RTP_KV_OFFLOAD_KEEP_BLOCKS>0 且已生成≥1 token 且 seq≥RTP_KV_OFFLOAD_MIN_SEQ）
- KVCacheManager/KVCacheAllocator/SingleTypeKVCacheAllocator：freeBlockList API（NULL/0 过滤）；stream-end free 过滤 b<=0 防双释放
- DecodeRpcServer::allocateResource：RTP_DECODE_MALLOC_RETRY_MS 重试窗（失败重建 stream，100ms 间隔），默认 0=原行为
- 恢复：`git -C /home/admin/project/rtp-llm-rdma checkout -- <file>` 或 *.bak-pre-v32offload
- 待验证风险：decode pd_kvcache_ref 是否压着 refcount 使释放不生效（P-B3 观察 freeBlocksNum）

## P-B3 python 容量模块（v32_capacity.py，已写就待联调）
- 请求标识=块表[0]物理块（C++ 保留 block[0] 不释放）；offload 检测=块表中部出现 0
- 每层存储：主 KV host pinned bf16 [cap,576] + indexer-K GPU 侧张量 uint8 [cap,132]（分段布局搬运）
- 时序关键：当前 token 主 KV 在 indexer 后写入——host 侧只存 [0,kvlen-1)，打分取 top-(k-1) 历史，attention 时从 GPU 池现读当前行拼接（近似声明：当前 token 恒被注意，同 ECHO 假设）
- kernel 防护：offload 行的 kernel topk 全部改指 kvlen-1（合法内存），输出行由 python 覆写
- 接线：hook 里 V32_OFFLOAD_MODE=capacity 安装 _cap_topk/_cap_fwd；与 VERIFY 开关互斥
- 打包：package_b_runtime.sh → /home/admin/rtp-hol/runtime/rtp-b-offload-<date>（A 基线 runtime 不动）

## P-B3 判决（2026-08-17 02:40，PASS）
- runtime：rtp-b-offload-20260817（.78；wheel 由 rtp-llm-rdma+补丁构建，A 基线 runtime 未动）
- 证据：engine.log `offloaded 801 prefix blocks to host tier, keep_last=64 total=866`（GPU 驻留 866→65 块，-92.5%）
- 正确性：55.4k 请求贪心 64 token，capacity vs off **逐字一致**（TEXT_MATCH True），capacity errors=0
- 途中修复：a) 打包漏 v32_capacity.py；b) 首 token 来自 prefill——offload 条件改为已生成≥2 token（先让第一个 decode 步完成镜像）
- 性能（未优化）：capacity 806ms/步 vs 基线（待补数）；优化项=跨请求批量化 python 路径+上步索引预取——放 B 跑通后
- P-B4 待办：分发 runtime 至 3 节点；v32_env B 变体（decode 侧 V32_OFFLOAD_MODE=capacity, KEEP=64, MIN_SEQ=16384, RETRY_MS=30000）；mixed-1k kv12g W96 3:1 重跑；判决指标不变

## P-B4 首轮判决（2026-08-17 10:20，FAIL——功能对但性能反噬）
- run: v32b-20260816T185634Z-schemeB-mixed1k-kv12g-offload（results.jsonl 在 operator_runs 对应 3to1/）
- B vs A：长拒绝 100% vs 97.7%（B 更差）；短拒绝 10.4% vs 0.8%；短 decode step p50 2261ms vs 110ms（20×）；ok tokens 257k vs 549k
- 失败构成：129×8211_MALLOC + 88×603_GENERATE_TIMEOUT + 2×514
- 根因：python 容量路径在批次关键路径逐层逐行执行——a) 预 offload 全量镜像（72MB×61 层同步 D2H，一步内完成）；b) serve 路径每层每行多次 CPU 同步(.item/.tolist/fill_)；decode 整体 20× 慢→池排不掉→长请求准入雪崩
- 优化方案（下一轮）：1) C++ 加 RTP_KV_OFFLOAD_AFTER_TOKENS（如 24），给 python 24 步分片镜像窗口；2) 镜像改增量分片+侧流异步 D2H；3) serve 路径跨请求向量化、每层仅一次 CPU 同步；4) host gather 用上一步索引预取重叠
- 结论方向不变：单机已证数据通路正确+92.5% 显存释放；纯性能工程问题

## B 优化轮 round-2（2026-08-17，用户定向：异步镜像+驻留窗直读+不分桶）
- C++：RTP_KV_OFFLOAD_AFTER_TOKENS（默认2，B 跑用 32）——延后收缩给异步镜像留窗口
- v32_capacity.py 重写（round1 备份 .round1）：a) 侧 CUDA 流分片镜像（V32_MIRROR_CHUNK=4096/层/步，D2H 全 non_blocking，事件排序）；b) top-k 按驻留性切分——warm（GPU 块）直读池、只有 cold 走 host+PCIe；c) 每层≤3 次 CPU 同步（kvlens/k01/cold 索引各一批）；d) store 空洞防护（bad 标志拒答）
- 驱动 EXTRA_ENV 追加 AFTER_TOKENS=32；k=2048（DSA index_topk），fetch 量 ~2.3MB/层/请求/步，本轮瓶颈是同步而非带宽（用户判断正确）

## 档位1 CUDA 下沉（2026-08-18 进行中）
- round-8 已实测：单机 step 376ms（基线 250），score 10.5ms（原装 kernel）、build 35ms、drain 10ms；正确性 PREFIX_MATCH ✓
- **kernel #1 `build_indices` 完成并单测通过**：residency+staging 二分+miss 提取+索引拼接一个 kernel；61 层成本 35ms→**1.84ms**；对拍与 torch 参考逐位一致（v32_ops.cu / v32_ops_build.py，.so 在 rtp-hol/v32ops_build）
- round-9 集成：v32_capacity.py 走 _ops.build_indices（无 op 自动回退 torch）；miss 计数移交 worker 线程（主线程零同步）；已部署 4 节点，**待 BA 序列结束后做整机门禁**
- 待办：drain 精简、打分跨请求批量化（公共侧池）、C++ fetch 线程

## B round-8 集群终值（2026-08-18 07:21，FAIL——镜像流阻塞）与 round-10 修复
- 终值：长 0/128，短 fail 9.7%、step p50 2137ms；ok tokens 262k。python 段实测仅 ~60ms/步、errors=0——2s/步的墙钟差在段外
- 根因（日志佐证）：_mirror_chunk 的 wait_event 令主流等待侧流 72MB×61 层 D2H；集群 128 长请求错峰到达→每 rank 常态有人在镜像/追加→主流持续被 D2H 卡死（单机仅 1 请求 14 步镜像完，故 376ms 测不出）
- round-10 修复：事件拆分——ev_idx（indexer-K GPU→GPU 小拷贝，主流仅等它）/ ev（KV D2H，仅 fetch worker 同步）；主流零 D2H 等待
- 另：round-9（build_indices CUDA op，61 层 35→1.84ms）已集成待门禁；集群 miss 率比单机高 1.6×（多请求 LRU 压力）
- 待办：A3 跑完后单机门禁 round-10（含多长请求并发压测复现 2s 场景）→ 重跑 B

## 档位1.5：C++ context 重构（2026-08-18 启动，目标=单机追平 A 的 250ms/步）
- 用户指令：重构到 C++ 直到追平 A，停下给 breakdown，之后再定重跑
- 现状：round-11 单机 338ms（+88 vs 250 基线）；构成=打分逐请求循环 11 + staging 维护 15 + build(已 kernel) 2 + hook 通道 20 + GIL/同步 30 + 隐余 ~10
- 架构：C++ 扩展 v32_ops v2——Context 对象（侧池/staging 映射/host store 指针）+ serve_layer（build+消毒+miss 写 mapped pinned）+ staging_update kernel + 常驻 C++ fetch 线程（pinned gather + cudaMemcpyAsync 专用流，GIL 无关）；deep_gemm 打分调用留 python（单次融合调用）但改公共侧池跨请求批量
- 改动半径=3 个自有文件（v32_ops.cu 扩展 / v32_capacity.py 重写为薄壳 / hook 微调）；引擎 C++、wheel、models_py 全部不动
- 门禁标准不变：PREFIX_MATCH + 并发压测（30k×4 错峰）+ 分段 breakdown；预期 88→≤15ms
- 关键既有事实：build_indices kernel 已单测通过（1.84ms/61层，.so 在 rtp-hol/v32ops_build 与 runtime 内）；A3 判决=长失败 75.8%/排队 17.7min；A(秒拒)=97.7%拒

## r28-r29（2026-08-19，方案一副本池）：单波打分落地 + 身份复用事故
- **架构**：全局 indexer 池（V32_IDX_POOL_BLOCKS=4096 块/层 ≈2.1GB/rank）+ 块分配器 + 每请求水位线；
  准入批量 D2D 前缀、缺步回填、每层批量追加当前 token（132B×B，一个 kernel）+ block0 身份绊线（失配行下步重准入）；
  批稳且全员入池 → 跳过原生第一波，一次 fused 调用出全 batch top-2048；任何不确定当步回落 r27 双波
- **r28 事故（block0 身份复用）**：用块表 block0 物理块号当请求身份，冒烟请求结束后块归还 free list，
  下个请求复用同一 block0 → 继承旧池条目（水位线+旧字节）→ run0 PREFIX=False；run1/2 因三轮 prompt 相同
  继承的恰是正确数据而"侥幸"通过。教训：物理资源 id 不是请求身份
- **修复**：decode 不变量 kvlen 每步恰 +1 → `期望 kvlen = 上次 + 经过步数`，失配即判定换租户重准入；
  同款潜伏 bug 在 _store（host 镜像，两个长请求先后复用 block0 会串数据）一并修复
- **r29 判决（PASS）**：174/167/172ms，PREFIX 3/3，errors=0，单波 232+/256 步全程生效，绊线 0 触发
- **关键发现（L0 微二分）**：V32_HOOK_LEVEL=0（纯透传，C++ 收缩补丁+env 全开）= **143/147ms ≈ 基线 144**
  → C++ 引擎补丁零开销；剩余 ~27ms 全部在 hook 链路，且单波（消灭整个第二波）几乎不省时间
  → 推翻"第二波 GPU 21ms"的算术推断：第二波本来就被发射流水气泡吸收，真正的成本是
  每层插入的 python 调用点 + 我们 GPU 链的串行依赖（indexer→我们→attention）打断 launch-ahead

## 集群 B 首轮（v32b-20260818T180338Z，r29 runtime，03:00 完赛）
- 终值：**短 871/872 ok（99.9%）**；**长 1 ok / 127 fail（99.2% 失败，全部 8211=30s 重试窗耗尽）**；全程 45min（对比 round-8 的数小时+步长 2s）
- 判读：性能/正确性机器已健康（短请求零回退、推进快）；长请求败在准入参数——瞬时需 ~1030 块，收缩 32 token 后才释放，
  W96 下 ~12 长并发排队 >30s。机制上 B 的重试可收敛（持续释放），A 不会——参数问题
- **B2 已于 03:01 自动接力**（守望拦截 A3）：RETRY_MS=600s、AFTER_TOKENS=16；预估收敛账：长请求准入节奏 ~10-15s/个，12 个并发排队 ~150s ≪ 600s 窗
- B2 之后自动接 A3 重跑（同参数对照）；tmux v32-ab2，日志 schemeB2A-driver.log

## 集群 B2 中程诊断（03:45）与 B3 决策
- B2（RETRY=600s, AFTER=16）：长请求仍 ~104 个 8211，且 execute time=600062ms——**窗口生效但等满 10 分钟仍进不来**
- 诊断（.78 decode 日志佐证）：收缩正常（每长释放 696 块，keep_last=256 total=985）；瓶颈是 **KEEP=256 → 驻留长 289 块/个，
  单 rank（2457 块）只容 ~7 个并发长**，后续长要等前面的**整个跑完**（活 6-10 分钟/个）→ 准入串行化，排队深度 16/rank × 完成节奏 ≈ 18min > 10min 窗
- 另发现：集群 decode batch 含 dummy/block0≤0 行 → 单波整步跳过（reqs=0），一直走双波回退（r27 路径）——正确性无虞，
  步长略高，后续可改成按行跳过而非整步放弃
- **B3（03:50 布置）**：KEEP_BLOCKS 256→64（驻留 97 块/长，单 rank 并发容量 ~20 长，排队从"等完成"变"等收缩"），
  其余同 B2；守望拦截 B2 后的 A3，接 B3→A3（tmux v32-ab3，schemeB3A-driver.log）
- **B2 终值（03:59）**：长 6 ok / 122 fail（4.7%，10min 窗只多救了 5 个——证实"等完成"串行化诊断）；短健康；
  **B3 已于 04:00 自动接力**（keep64），集群启动中

## 集群 B3 中程（04:45）：诊断修正 + B4 决策
- B3（keep64）仍 106 个 8211 且等满 600s → keep 缩小无效，**诊断修正：稳态容量挤压而非排队串行化**——
  W96 下短请求常驻 ~1000 块/rank，加驻留长后备用容量长期 < 准入阈值 1030 块，重试窗内永远等不到空窗。
  这就是文档的结构性局限（准入需瞬时全量前缀块；ECHO 原生传输层直落 host 无此峰值）在集群负载下的表现
- **B4（用户点名的杠杆——并发数）**：WORKERS 96→48，短压减半，备用容量周期性越过 1030；
  其余同 B3（keep64/retry600/after16）。A3 同步改 W48 对照（run_v32_schemeA3w48.sh），保证公平配对
- A 侧预期仍结构性失败：A 的长请求全生命周期占 1030 块不释放，单 rank 至多 2 个并发长，且历史 A3@W96 已 75.8% 失败
- 守望拦截 B3 后接 B4→A3w48（tmux v32-ab4，schemeB4A-driver.log）
- **B3 终值（05:01）**：长 12 ok / 116 fail（9.4%；B1→B2→B3 = 0.8%→4.7%→9.4%，keep64 有边际收益但远不解决）；
  **B4（W48）已于 05:02 自动接力**

## 集群 B4-B6（06:24-06:50）
- **B4 终值**：长 39 ok / 89 fail（30.5% 成功；W48 显著改善但不达标）；短 5 失败
- B5（W32）秒退：`run_worker_sweep.py --workers 不支持 32`（档位最小 48）——并发杠杆到底
- **B6（06:50 启动，tmux v32-ab6）**：换用户点名的第二杠杆——数据集长请求比例减半
  （make_thin_dataset.py：1500→753 长，保序隔一去一，/tmp/rym/dataset/trace-0715-mixed5k-long50），
  W48+keep64+retry600 不变；A6w48 同数据集同 W 公平对照，序列自动接力
- 遗留观察：长请求可能被 LOAD_ONLY 路由堆到同 rank（负载指标不计块需求）——与既有"调度均衡性"课题呼应，报告中说明

## 集群 B6 崩溃复盘（07:45）
- 时间线：06:50 启动 → 06:57 `[v32port] warn` 刷屏（glm5port 拷贝引擎告警宏）→ 07:06 decode rank0 死 →
  TCPStore 连锁全 rank → 815×8202_CONNECT_FAILED → run 报废（长 1 ok/72 fail 判读无效）
- 排除项：GPU OOM（dcgm 崩溃时刻 fb_used 122/144GB）、宿主 RAM（960GB，可用 825GB）、kernel OOM（dmesg 无）
- 指向：拷贝引擎告警路径（host pinned staging 增长失败或 sticky CUDA error），触发条件=长比减半后
  准入成功率上升 → 并发镜像长请求数上升 → 每长 host pinned 3.9GB + GPU idxp 447MB 的负载放大
- **结论：B 的高并发 offload 存在稳定性墙；B4 配置（W48+12.8%长比，82min 无崩溃）为当前安全区**
- A6w48（vanilla、long50 数据集）07:09 起跑中——等其出数后定 B7（安全区与收敛区之间折中，如 retry=120s）

## 集群 A6w48 终值（08:46）+ B7 启动
- **A 基线判决（long50 数据集，W48，30min 重试窗）**：长 24 ok / 49 fail = **67% 长失败**；短 ~3 fail（99.7% ok）
  ——参数已尽量宽松（W 减半、长比减半、30min 窗），A 仍结构性饿死 2/3 长请求
- **B7（08:47 启动，tmux v32-ab7）**：long50 + W48 + keep64 + retry=180s（把并发镜像长请求数压回 B4 安全区，
  规避 B6 稳定性墙，同时给准入 6× 于 B1 的窗口）；跑完即可与 A6w48 同数据集同 W 直接配对出报告

## 收官（09:10）：B7 同崩 + 根因定罪 + 实验终止
- B7 ~10min 同款崩溃（813×8202）→ 重试窗排除；**定罪：decode rank0 存活但 KV-TCP cache store 全面
  `CACHE_STORE_PUSH_ITEM_FAILED`**——host pinned 镜像（3.9GB×并发长）与 PD 传输缓存店抢宿主资源，
  B1 即有 1 例偶发，long50 提高并发镜像后升级为永久故障 + 实例级联
- 集群实验终止，4 节点已清场；全部判决与修复路线见《集群AB实验报告-20260819.md》（已同步 app_source）
- 关键数字：A6w48 长请求 67% 失败（最宽松参数）；B 最优安全点 B4 = 长 30.5% ok（更难数据集）+ 短 99.9% + 无崩溃

## r25-r27（2026-08-18 深夜，session 接续后）：T2 收尾 + T1 原生化
- r23/r24 事故定罪：`kh=pend[1].tolist()` 冻结快照破坏块表新鲜度 → 回退活视图，r25 恢复 PREFIX=True（175/180/182ms）
- **r26 = T2a**：build_indices 改产出 request-local 逻辑位置（staging 命中经 s_logical 别名反查块表项 j*64+off）；ctx_serve_full 在 C++ 原地写回 kernel_topk；原生 convert-to-global 承运 → SparseMlaOp.forward hook 整个卸除。171/176/180ms PREFIX=True
- **r27 = T1 用户态形态**：offload 行第二波打分换成 A 同款原生 fused（deep_gemm fp8_paged_mqa_logits + fast_topk_transform_fused）直跑侧池：ctx_append_tok 每层补当前 token 132B → lengths=kvlen 与 A 全同构（finalize 强制附加 cur 退役，cur_pos=-1）；稠密合成块表按容量共享，kvlen 张量/schedule metadata 按请求按步缓存。170/173/173ms PREFIX=True，errors=0
- **breakdown（hb 计时）**：orig（第一波 CPU 发射）4.6ms/步；proc（T1 全路径 CPU 发射）0.7ms/步 → python 胶水税已消。剩余 ~28ms = 第二波 GPU 真实计算（logits~10.5 + fused topk~6 + build/写回~4）串联在每层 indexer→attention 之间 + 结构税
- **结论**：重算的实现效率到顶（同款 kernel），下一刀只能消灭重算本身 = 单波打分：独立 indexer 池覆盖全部 decode 行（普通行 admission 时批量 D2D 前缀 + 每步 132B×B 追加；offload 行沿用镜像），第一波直接喂（our_pool, ibt）。显存代价 ≈1.3GB/rank（全 batch indexer-K 副本）+ offload 溢出部分。可先在 runtime 补丁层实现（models_py 均为 python，无需重建 wheel），C++ BlockPool 第三池留作上游 PR

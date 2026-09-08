# V3.2 长上下文 KV Offload（staging 环准入 + Tier-2 有损第三池）— 迁移交接文档

日期：2026-08-20（环准入）／**2026-08-21 追加 §7：Tier-2 有损第三池**
｜ 源：内部工作区 `rtp-llm-rdma @ b7167df18`（+工作区补丁）｜
目标：本仓库（github alibaba/rtp-llm master @ d4d9bf18b）｜ **全部改动未 commit，工作区状态交付**

## 0. 一页结论

DeepSeek-V3.2 63k 长请求在 PD 分离 decode 侧的准入需要瞬时全量 KV 块（63k=985 块），
在池紧张时结构性饿死。本工作实现 **staging 环准入**：handoff 时只分配
block0+staging32+尾部256+环64=353 块（与请求长度无关），16k 外前缀经 64 块环分批拉取，
直落引擎持有的 host 镜像 + GPU indexer 池，python 侧 serve 链 adopt。

最终对齐判决（mixed-1000、W48、秒拒、8rank 分发、RDMA，报告见 docs/）：

| | A 原版 (Am8r) | B 环准入全家桶 (Brm11r-r9) |
|---|---|---|
| 总失败率 | 4.0% | **3.4%** |
| 长请求 ok | 88/128 (68.8%) | **95/128 (74.2%)** |
| 短请求失败 | 0/872 | 1/872 |
| TPOT p50/p99 | 106/115ms | 147/389ms（offload 伺服税） |
| 崩溃 | 0 | 0 |

B 的收益随请求长度/池压力增大（128k 时 A 需 2060 块将再次饿死，B 不变）。

> **读到这里不要停**：结论按 §7 → §8 → §9 → §10 → §11 逐层修正；**接手者直接读 §12
> （终局设计 + 全项目踩坑与经验），它是自足的**。
> 要点：块粒度热池应废弃（改 token 粒度无损取数，§8.3）；共发现**五个**静默数据污染
> bug（§9.6 汇总）；**引擎本身非逐位确定，验收必须用真实语料 + 噪声底判据**（§9.2）；
> §10 用七档成本阶梯把 C 的开销拆到项，并用步进快路径把 python 热路径下沉 C++。
> 单请求正确性已闭环、延迟在基线噪声带内。§11：引擎级独立 indexer-K 池
> （`V32_INDEPENDENT_IDX_POOL=1`）全链路接通，offload 只放主 KV，python 打分波/影子池
> 整体待命化 —— bug #3/#4/#5 的土壤从根上移除，待端到端验证后删影子池。

## 1. 本仓库中已应用的改动（git status 可见，未 commit）

### 引擎 C++（核心，已尽量对齐上游风格）
| 文件 | 内容 | 应用方式 |
|---|---|---|
| `rtp_llm/cpp/cache/V32AdmissionStore.{h,cc}` | **新增**。准入镜像单例：host pinned 主 KV + GPU idxp，30s 墓地延迟释放（防 fetch 线程 UAF），C 导出 `rtp_v32_admission_lookup/release` 供 python 扩展 dlsym adopt | 新文件拷入 |
| `rtp_llm/cpp/cache/BUILD` | cache 目标加入上述文件 | 补丁 |
| `rtp_llm/cpp/cache/Types.h` | `MallocInfo.init_seq_len_override`（首次 malloc 封顶） | 补丁（干净） |
| `rtp_llm/cpp/cache/KVCacheManager.{h,cc}` | `freeBlockList` + ctor 透传 | 补丁（干净） |
| `rtp_llm/cpp/cache/SingleTypeKVCacheAllocator.{h,cc}` | ① init override 响应；② 哨兵安全 free；③ **insertIntoCache 只插首哨兵前的有效前缀**（reuse 缓存污染修复，B6/B7 级崩溃根因）；④ **BlockPool 转发 use_cuda_malloc_block_pool**（RDMA MR 修复的缺失一半，上游 NormalEngine 已有另一半 `shouldUseCudaMallocKVCacheBacking`） | 补丁（干净） |
| `rtp_llm/cpp/engine_base/stream/StreamCacheResource.{h,cc}` | 准入封顶分配（块表重排+0 哨兵+环块摘出）、`releaseAdmissionRing`、运行中收缩（前代方案）、释放钩子 | 补丁（fuzzy） |
| `rtp_llm/cpp/model_rpc/DecodeRpcServer.{h,cc}` | `loadPrefixViaRing`（分批环拉取+排空+**设备绑定修复**+**并发闸门**）、PB 环块编解码、哨兵跳过、malloc 重试窗 | **手工移植**（.cc 上游漂移大，未编译验证，见 §4） |
| `rtp_llm/cpp/model_rpc/proto/model_rpc_service.proto` | `BroadcastLoadRequestPB.admission_ring_block_ids = 15` | 补丁（fuzzy） |
| 其余（GenerateStream/metrics/pipeline 等） | 前代 v32 仪表/收缩配套 | 补丁（见 git status） |

### 未应用（需手工合并，材料在 `v32_migration/manual_merge/`）
- `NormalEngine.cc`：上游已自带 cudaMalloc 接线（更规范），我们补丁中仅剩步进指标部分未合。
- FlexLB Java（`WorkerStatus.java`/`WeightedCacheLoadBalancer.java`）：**上游 LB 已重构为
  CostBasedDecodeStrategy/DecodeEndpoint 体系**，我们的"块感知调度"（offload 驻留量记账 +
  free-KV 候选过滤，env `FLEXLB_DECODE_OFFLOAD_RESIDENT_TOKENS/MIN_SEQ`）需按新体系重写；
  参考实现与单文件补丁在 manual_merge/ 下。
- `WorkerStatusService.cc`、`QueryConverter.cc`、`NormalExecutor.cc` 等：前代仪表，冲突未合。
- `deps/git.bzl`、`cache_store_service.proto`：内部构建/协议差异。

### 构建覆盖（已回退，不适用公开树）
`v32_migration/build_overlay/internal-rdma-build-overlay.diff`：内部 RDMA 构建需要
（arch_select 指向 internal_source 的 `cache_store_rdma_impl` + RDMA copts）。公开树无
internal_source overlay，默认仍为 TCP 桩实现（`Impl.cpp` throw）。

## 2. 运行时资产（`v32_migration/`）

- `python/`：`v32_ctx.cu`（C++/CUDA torch 扩展：staging/miss-fetch/写回 + `ctx_adopt` dlsym 通道）、
  `v32_capacity.py`（单波/双波打分、adopt、镜像）、`v32_offload_hook.py`（安装钩子）、
  `glm5port/`（分级拷贝引擎）、`v32_ctx_build.py`（构建+单测）。部署=拷入 runtime 的
  site-packages/rtp_llm/ 并在 mla_attention.py 末尾 import hook。
- `harness/`：全部 run 脚本（Am8/Brm* 系列）、`rtp_cluster.py`
  （含 **`decode_worker_address()` 8-rank 分发修复**，`RTP_DECODE_RANK_FANOUT=0` 回退）、
  `br-runtime.env`、打包/验证脚本、指标分析器。
- `docs/`：对齐实验报告（20260820 终版）、环准入设计文档（含全部判决）、实验台账、原始指标。
- FlexLB jar（二进制未入库）：`/home/admin/rtp-hol/flexlb/flexlb-api-blockaware2-173c1a8f.jar`
  （sha256 173c1a8f...d894；env 不设=原版行为）。

## 3. 关键环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `RTP_KV_ADMIT_RING_BLOCKS` | 0=关 | 环准入开关+环大小（实验用 64） |
| `RTP_KV_ADMIT_RING_CONCURRENCY` | 1 | 每 rank 并发环拉取闸门（防 cache store 过载） |
| `RTP_KV_OFFLOAD_KEEP_BLOCKS/MIN_SEQ/STAGING_BLOCKS/AFTER_TOKENS` | — | 驻留窗/阈值/staging/收缩延迟 |
| `RTP_KV_POOL_BACKING` | 自动 | torch/cudamalloc 强制（RDMA 需 cudamalloc：VMM 注册不了 MR） |
| `RTP_DECODE_RANK_FANOUT` | 1 | decode 按 DP rank 展开 LB 端点（**8× 有效容量**） |
| `FLEXLB_DECODE_OFFLOAD_RESIDENT_TOKENS/MIN_SEQ` | 0=关 | LB 块感知调度（jar 支持） |
| `RTP_DECODE_MALLOC_RETRY_MS` | 0 | 准入 malloc 重试窗 |

## 4. 已知事项 / 下一步

1. **本树的 DecodeRpcServer.cc 为手工移植且未编译验证**（内部构建环境不可用于公开树）；
   内部已验证版本在 manual_merge/ 供比对。首次编译请重点检查该文件。
2. 公开树默认无 RDMA cache store 实现（internal overlay），engine 侧 MR 修复
   （cudaMalloc backing 链路）已在但只在 RDMA 模式生效。
3. B 剩余成本：TPOT +39%（offload 伺服链结构税，优化方向：CUDA graph 化/引擎第三池/
   fetch 下沉 attention 准备段）；准入排队 p99 3.9s（环限流代价，可错峰/放宽并发）。
4. 建议补判决：128k 级 trace / W96（B 的优势区间：A need 2060 块将撞墙，B 恒定 353）。
5. 历史根因备查：decode DP8 仅 rank0 接请求（分发修复 28× 改善 A 长请求）、
   reuse 缓存哨兵污染（B6/B7 崩溃）、多 GPU 设备绑定（gRPC 线程 device=0）、
   镜像释放 UAF、环拉取并发打爆 cache store——全部有判决数据，见 docs/ 台账。

## 5. 复现路径（内部环境）

```
构建:   bazelisk build //rtp_llm:rtp_llm --config=cuda12_9   (.78, internal overlay)
打包:   harness/package_b_runtime.sh → runtime 目录（另需 v32_ctx.so + manifest/sha 文件）
python: v32_ctx_build.py（GPU 机）→ v32_ctx.so 拷入 runtime
运行:   harness/run_v32_schemeBrm11r.sh（B 全家桶+RDMA）/ run_v32_schemeAm8r.sh（A 对照）
分析:   harness/analyze_aligned_runs.py
```

## 6. 编译修复记录（2026-08-20 晚，公开树首次编译通过）

`bazelisk build --config=cuda12_9 //rtp_llm:rtp_llm` 已在迁移树上编译通过（wheel 产物
bazel-bin/rtp_llm/rtp_llm-0.2.0-*.whl，libth_transformer.so 含 rtp_v32_admission_* 导出 ×2）。
修了三处迁移失误（本地与 .78 镜像均已应用）：

1. `NormalEngine.h` → **还原为上游版**：迁移时加的 `getLastStepMetrics()` 声明（step-metrics
   仪表，未合并完）与 `cache_store_config` 成员位置搬动（与上游 ctor 初始化顺序冲突，
   -Werror=reorder）都不需要——上游已自带 cudaMalloc 接线
   （`shouldUseCudaMallocKVCacheBacking`，NormalEngine.cc:53/437）。
2. `DecodeRpcServer.h` loadCache 签名对齐上游单参：
   `ErrorInfo loadCache(const LoadKVCacheContext&)`（旧 3 参出参已被上游移除）。
3. `WorkerStatusService.h` + `WorkerStatusServiceTest.cc` → **还原为上游版**：
   step-metrics 字段引用未合并的 ExecutorStepMetrics（前代仪表，上游 LB 已重构，不再需要）。
   注：`rtp_llm/cpp/engine_base/ExecutorStepMetrics.h`（未跟踪文件）现已无引用，可删。

依赖拉取注意：github 直连偶发 "Empty reply from server"，重试即可；编译带
`--jobs=192 --local_cpu_resources=HOST_CPUS`。

## 7. Tier-2 有损第三池（2026-08-21，单机验证）

### 7.1 设计

打分保持精确（单波：自管 GPU indexer 池 + `fp8_paged_mqa_logits` + `fast_topk_transform_fused`），
**注意力有损**：每层 top-2048 里既不在引擎驻留窗（block0 + 尾部 `KEEP_BLOCKS`=256 块=16k token）
也不在该层热池（`STAGING_BLOCKS`=32 块=2k token）的选择直接置 -1 丢弃。

关键取巧：**不需要 attention 钩子、不改引擎块表**。热池命中被重映射成 staging 的
*逻辑* 坐标（`table_pos*64+off`），原生 convert-to-global 用未改动的块表翻译
（`bt[table_pos]*64+off`）正好落到 staging 物理槽。32 个 staging 物理块按层各有含义，
因此一个请求的 32 块 = **每层独立的 32 块热池**（61 层共 32×61 个别名）。

miss 导出零同步：mask kernel 把 miss 块号 + 计数 + step tag 写进 mapped pinned 内存
（`__threadfence_system` 后翻 tag），host 下一步按 tag 消费（≥1 步滞后），
按 FIFO 环挑受害者，整块 H2D（64×576×2B=73.7KB）异步预取，别名更新用
clear-old→copy→set-new 的 kernel 括号，事件同步给计算流。

命中率是**服务系统自己的计数器**（mask kernel 里 atomicAdd 到 pinned int64[4]：
tail/pool/miss/serves），不是 trace 回放。

### 7.2 实测（单机、无 PD 分离、DeepSeek-V3.2-Exp、63k=62841 token、batch=1、128 输出、cuda_graph 关）

| 拓扑 | A | B（有损 offload） | 差距 |
|---|---|---|---|
| tp1/dp8（集群生产拓扑） | 138.0 ms | **141.6 ms** | **+2.6%** |
| tp2/dp4 | 147.0 ms | 153.6 ms | +4.5% |

- **A 未被改坏的证明**：同拓扑把 `mla_attention.py` 末尾的 hook import 注掉跑纯 vanilla
  = 138.45 ms，与带 hook 的 138.01 ms 在噪声内（MODE=off 时只是空壳调用；引擎 offload
  分支由 `RTP_KV_OFFLOAD_KEEP_BLOCKS`=0 关闭）。历史 `p1_results.jsonl` 同脚本 135–137 ms
  （2k–32k）也一致。
- 报告里的 110 ms 是集群混合负载**服务端 decode step p50**（并发 128、PD 分离、1k 短请求为主），
  与单请求 63k TPOT 不可比：同拓扑下 1k 上下文单请求也是 141 ms。
- **warm-pool 命中率 0.88–0.92**（tail 85–90%、热池 2.0–2.6%、miss 8–12%），并发下 0.919。
- 逐步归因（`V32_STEP_TRACE`）：迁移期 15 步已降到 136–140 ms（= A 水平），稳态 136–141 ms。

### 7.3 本轮修掉的 5 个 bug（前 4 个都是真 bug，不是调参）

1. **host 镜像一直是空的**（严重）：`ctx_mirror_d2h` 设了 `direct_pinned_host_segments=true`，
   而该标志只对 H2D 有效——D2H 时 `execStagedMemoryCopy` 直接 `return false` 什么都不拷
   （日志里成千上万条 `[v32port] warn` 就是它）。于是热池预取上来的是垃圾数据。
   已改 false + 返回值 `TORCH_CHECK` 硬校验 + 单测加内容断言。
2. **多设备绑定崩溃**（严重）：`copy_stream` 和 cudaEvent 只在 `ctx_init()` 时按当时的
   当前设备建了一份全局的。单请求时路由总落 rank0（device0）所以从未暴露；并发到 8
   请求分到其他 rank 后跨设备用流 → `CUDA error: an illegal memory access`，服务崩溃。
   已改为**按设备惰性建流 + 事件环 + 入口 `c10::cuda::CUDAGuard`**；修后 conc=8 从 4/8 → 8/8。
   （与 docs 里记过的 "gRPC 线程 device=0" 同类，但那次没覆盖到这套 kernel。）
3. **pinned host 镜像分配 = TPOT +25% 的真凶**：每请求给 61 层各分配 85MB pinned
   （共 5.2GB），实测 **3.2s / 1.56 GB/s**，且 `cudaHostAlloc` 会隐式同步设备。
   改为按 8192-token 分桶的复用池 + 后台预热线程；顺带修了预热本身的 bug
   （逐个 alloc 后立刻 free 只是把同一块反复回收，池里最终只有一块）。
4. **私有 GPU indexer 池分配 ≈ 2.9s**：61 × 9.4MB = 572MB，在显存几乎占满时每次
   cudaMalloc 都要回收缓存段并同步。它只被 dual-wave 回退用到（单波占 127/128 步），
   改为惰性分配；dual-wave 时若无历史则该行本步跳过服务（原有 error 路径语义）。
5. **镜像带宽**：row 粒度 staged copy（gather→staging→D2H→**CPU 逐段 memcpy**，实测 4.8–6 GB/s）
   换成整块直拷（`ctx_mirror_blocks_d2h`，物理连续块合并成段，**17–18 GB/s**），
   并把每层一次的 `.cpu()` 同步收敛成每步一次（块表与层无关，跨 61 层复用）。

### 7.4 三项延迟优化（已完成）

1. 事件括号只在真的发了预取时才记/等（原来 `ev_live` 一旦置位就永久每层 `WaitEvent`）；
   改成 `ev_recorded`（供 `cudaEventQuery` 判活）+ `ev_pending`（同一计算流只需等一次）。
2. 每步只算一次行计划 `_row_plan`（kvlen/块表/offload 判定都与层无关，原来 61 层各算一遍）。
3. 镜像 D2H 改 8 深度事件环流水，只在补完历史的最后一块 `flush`（迁移期 mirror 8→3 ms/step）。

### 7.5 未证实 / 待办（重要）

1. **并发下 B 目前是负收益**，且需要 PD 分离环境才测得准。
   KV 池 10GB/rank、tp2/dp4、63k×12 并发（每副本 3 个）、1024 输出：

   | | 总耗时 | 吞吐 | TPOT 均值 | TPOT 最大 |
   |---|---|---|---|---|
   | A | 606.9 s | 20.2 tok/s | 157.6 ms | 169 ms |
   | B | 690.9 s | 14.8 tok/s | 1259 ms | **12979 ms** |

   已排除 python 侧开销（全程 690s 里 `proc` 仅 8.7s，`hostalloc`=0，`errors`=0，命中率 0.919）。
   嫌疑：① 预取流量按并发线性放大（3 行 × 61 层 × 8 块 = **108 MB/step H2D**，而且计算流要等它）；
   ② per-device copy 互斥锁把并发请求的镜像/预取串起来；③ prefill/decode 交错。
   `PREFETCH=0` 的对照跑不出来——单机 prefill/decode 同机时长 prefill 会让 EP all-to-all
   超时（`DeepEP error: CPU recv timeout`），服务直接挂。**结论：并发容量必须在 PD 分离
   环境（decode 独立实例）复测。**
2. **容量收益只存在于稳态**：offload 发生在 prefill 之后，峰值占用仍是全量 984 块。
   KV 池 4GB（817 块 < 984）时 A/B 都收不进一个 63k 请求。每块跨 61 层约 5.01 MB，
   63k 需 984 块，B 稳态只需 256+32+1=289 块（3.4×）。
3. **并发扩展性隐患**：serve 是"每(行,层)一次 C++ 调用"，batch=1 是 61 次/step，
   batch=16 就是 976 次/step。需要把 mask kernel 改成一次调用处理该层所有行（rows 维度并行）。
4. **质量完全未验证**：命中率是用重复中文填充串测的，locality 不代表真实负载；也没做过
   与 A 的输出质量对比。热池目前只救回约 1/5 的 miss（pool% 仅 2.0–2.6%），
   `STAGING_BLOCKS`（32→128）是最直接的质量旋钮，代价是 H2D 流量。
5. 代码位置是临时形态：python 与 `v32_ctx.cu` 仍在 `v32_migration/` 下（改完拷进 runtime
   site-packages / 独立编 torch extension，为的是迭代速度），方案定型后应合回
   `rtp_llm/` 与 `rtp_llm/cpp/`。

### 7.6 新增环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `V32_LOSSY` | 0 | 有损第三池总开关 |
| `V32_LOSSY_PREFETCH` | 8 | 每层每步预取块数上限（0=纯丢弃） |
| `V32_LOSSY_DIAG` | 0 | 只计数不改写选择（归因用：区分我们的开销与下游 attention 的影响） |
| `V32_MIRROR_BLOCKS` | 1 | 整块镜像快路径 |
| `V32_MIRROR_CHUNK` | 4096 | 每层每步镜像 token 数 |
| `V32_HOST_BUCKET` | 8192 | pinned 镜像容量分桶粒度（复用池命中率） |
| `V32_PREWARM_TOKENS` / `V32_PREWARM_LAYERS` | 73728 / 61 | 后台预热的 pinned 镜像（0=关）。**并发场景需按并发数放大** |
| `V32_STEP_TRACE` | 0 | 记 N 步逐步耗时归因 `total(mirror/serve/proc/alloc/adopt/hostalloc)`（会引入每步一次同步） |

### 7.7 新增文件

- `python/v32_lossy_test.py`：有损路径单测（mask/别名/预取/换出 + 生产规模计时 15–19 µs/层）
- `python/v32_mirror_test.py`：镜像内容校验 + 两种路径带宽对比
- `harness/run_ab.sh`：A/B 服务器启动器（`MODE=A|B`、`TP`/`DP`/`GRAPH`/`KVMB`/`MAXSEQ`/`EXTRA_ENV`）
- `harness/bench_ab.py`：单请求 TPOT 测量；`harness/bench_cap.py`：并发容量（错峰到达 + 占用轮询）
- `harness/stream_probe.py`：逐 token 间隔探针（该版本服务端不支持 `is_streaming`，暂无用）
- 不要 commit：`python/topo.xml`、`python/npu_nic_affinity.json`（工具生成的垃圾文件）

### 7.8 复现

```
编译扩展+回归: cd v32_migration/python && python v32_ctx_build.py
单测:          python v32_lossy_test.py; python v32_mirror_test.py
部署:          cp v32_ctx.so v32_capacity.py v32_offload_hook.py \
                 <runtime>/site-packages/rtp_llm/
A/B 延迟:      MODE=A TP=1 DP=8 KVMB=12288 MAXSEQ=65536 bash harness/run_ab.sh
               python harness/bench_ab.py --ctx 63000 --out 128 --reps 3 --tag A
容量:          python harness/bench_cap.py --ctx 63000 --out 1024 --conc 12 --stagger 8
归因:          EXTRA_ENV="V32_STEP_TRACE=20"，看日志 [v32_steptrace] / [v32_lossy] / [v32_capacity] prof
```

## 8. 真机验证与四个正确性 bug（2026-08-30，DeepSeek-V3.2-Exp FP8，8×H20-141G）

### 8.0 一页结论

**延迟问题结案，正确性未闭环。** 单请求 63k、batch=1、tp1/dp8、out=512：

| 方案 | TPOT | vs A | 每 serve 救回 |
|---|---|---|---|
| A 无 offload | 139.59 ms | — | — |
| **C token 粒度无损**（本轮新增） | **140.54** | **+0.68%** | 313/313 |
| B8 块粒度有损热池（原方案） | 140.84 | +0.90% | 36/614 |
| B0 纯丢弃（无取数） | 141.43※ | +1.31% | 0 |

※B0 该次有 rep 提前结束，均值不干净。

**块粒度热池应废弃**：每救回一个 token 多搬 21 倍字节、多付 26 倍临界路径（§8.3）。
**性能已到硬地板**（§8.4）。**§7 的所有数字都受 §8.1 的 bug 影响，需重新审视。**

### 8.1 四个正确性 bug（修复后"输出与基线一致"从 0/4 → 5/6）

**① staging 驻留误判**（有单测）
引擎 `offloadPrefixBlocks` 只从 `free_from = 1 + STAGING_BLOCKS = 33` 起置 0 哨兵，
**块表位置 1..32 仍指向真实物理块**，而设计上这 32 块的内容已交给 python 当热池/scratch。
mask kernel 按 `bt[j] > 0` 判驻留，就把落在逻辑块 1..32（token 64..2111）的选择
**原样交给 attention，读到的是别的逻辑块的数据**。
修：两个 mask kernel 都排除 staging 范围（`LossyState::stg_lo/stg_hi`，register 时由 jpos 的
min/max 得出）。回归测试 `test_staging_not_resident`：修复前 256 个选择被误判驻留。
只偶发是因为 top-2048 很少落在 token 64..2111（attention 主要看近处 + block 0 sink，
而 block 0 未被用作 staging）。**B0 恰好正确**——prefetch=0 时热池从不写入。

**② 无声退回引擎打分波**
`_bookkeep_pool` 首个早退 `if _stepcache.get("fresh_step") != _step: return` 依赖一个异步
D2H 是否在本步落地。不落地 → `_sw["ready"]=False` → `pre_topk` 返回 None →
**引擎跑自己的原生波**。而 offload 生效后引擎块表全是 0 哨兵、indexer-K 已随块释放
→ 打出垃圾分数 → 选错 token。同时因为跳过了我们的打分，**还变快了**
（这解释了"开 offload 反而比关 offload 快"的悖论：关 offload 时 hook 全程生效 = +18.7%）。
修：改为只要有 kvlens/khead 就继续（1 步陈旧的块表只会延迟发现哨兵，
而那时我们自己的波仍然正确）；所有早退加 `_stats["bail"]` 计数；
新增 `[v32_unsafe]` 告警检测"已 offload 但我们的波退让"这个致命组合。

**③ offload 后重建索引池**
请求复用检测 `kvlen != e["lk"] + (_step - e["ls"])` 用的是**外推的** kvlen（`_bookkeep`
把上一步的值加步差）。外推偏 1 步就误判成新请求 → `_free_req` → 从带哨兵的块表
重新 bulk admit → **池被永久污染**。放宽 ①② 的门限后这个窗口变大。
修：复用检测恢复要求 `fresh_step == _step`；**已 offload 的请求永不释放池条目**
（重建只能读到已释放的 indexer-K）；tripwire 命中时改为把 key 加入 `_no_single`
黑名单（禁用单波、走双波），而不是重建。

**④ admit 静默跳过**（自我隐藏，最难发现）
`bulk_admit_kernel` 里 `if (sb <= 0 || db < 0) return;` —— 无计数、无告警。
遇到哨兵源就跳过该 token，目标槽位保持原样：
**首个请求是零**（打分≈0，永不被选中）、**后续请求是上一个请求的残留**
（prompt 相同 → 恰好正确）。所以这个数据污染**只在冷启动暴露，之后被掩盖**——
实测就是"前两个请求错、之后连续 6 次全对"。
修：计数器 `cnts[4]/[5]`（requested/skipped）+ 接入已有 tripwire（`ok_row` 参数，
跳过即置 0，下一步排空后按 ③ 拉黑单波）。实测修前 `admit skipped 120 of 3833484`，修后 0。

> **教训**：三个静默条件（无计数的早退、无标记的 kernel skip、内容恰好正确的块复用）
> 叠加，让真实的数据污染在大多数运行里看不出来。所有此类路径现在都有计数器。

### 8.2 引擎本身不是逐位确定的 —— 验收方法必须改

同一 prompt、`top_k=1`、固定 seed，**A（完全不加 hook）**：

| prompt | A 的输出 |
|---|---|
| 复读型（一句话重复万遍） | **6/6 完全相同** |
| 有信息量的（固定 seed 生成的变化文本） | **4/4 各不相同** |

贪心解码本身确定，不确定性来自 argmax 之前：EP=8 的 MoE all-to-all 求和顺序随网络
到达顺序变化，浮点加法不满足结合律；加上非确定归约与 top-2048 在分数接近时的 tie-break。
复读型 prompt 上 top-1/top-2 差距大，扰动翻不动 argmax；有信息量的 prompt 上很多位置
接近并列，扰动足以翻转，自回归再放大。

**后果**：逐位比对**只在 A 自身稳定的 prompt 上**才是有效判据。§8.1 的修复是在复读
prompt（A 6/6 稳定）上验证的，提升真实；但**剩余 1/6 的不一致无法归因**——
可能是第 5 个 bug，也可能是引擎噪声偶然显形。

**正确的验收方式**：① 先测基线在**真实语料**上的自身不一致率作噪声底；
② 方案的不一致率不显著高于噪声底才算通过；③ 质量用困惑度/下游指标而非字节相等。
**§7 的"命中率 0.88–0.92"等数字都需要标注是在什么噪声背景下测的。**

### 8.3 块粒度 vs token 粒度：数据经济学

miss 散落：每层约 205–614 个 miss token 分布在约 200 个不同的 64-token 块里。
块粒度为救 1152 B 有用数据要搬整块 73.7 KB —— **64 倍放大**。

| | 每层搬运 | 救回 | 每 token 字节 | 每 token 临界路径 |
|---|---|---|---|---|
| B8（8 块/层/步） | 590 KB | 24 | **24.6 KB** | **0.91 µs** |
| C（token 粒度） | 235 KB | 204 | **1.15 KB** | **0.035 µs** |

这就是线上 `pool%` 只有 2.0–2.6% 的算术根源：32 块热池最多覆盖 200 个块中的 32 个。
**不是调参问题，是粒度选错了。**

另一个发现：B 的"异步"预取**并没有藏住**。61 个 gather kernel 排在同一条拷贝流上，
driver 启动队列填满后 `cudaLaunchKernel` **阻塞 host**，而 host 正是往计算流塞 kernel 的
那个人 —— 拷贝流的带宽**反压**了整个 step。"放到另一条流上"在带宽受限时不管用。

### 8.4 性能已到硬地板

新增探针 `ctx_probe_launch`（只做 device guard + 一次空 kernel 启动）：

```
pybind probe    0.30 µs/层   (dispatch + 锁 + map，无 kernel)
launch floor    2.91 µs/层   (guard + 一次空启动)
B(prefetch=0)   3.63 µs/层   ← 只比地板高 0.72 µs
```

**80% 的成本是 CUDA 启动本身**，且**层间严格串行**（第 L+1 层的 selections 依赖第 L 层
attention 输出），batch=1 下每步 61 次启动**不可约**。0.22 ms/step 是任何 per-layer
干预的物理下限（138 ms 的 0.16%）。

C 的成本完全拆开：`12.27 µs/层 = 5.82（两次 kernel 启动）+ 1.85（执行）+ 4.60（PCIe）`。
PCIe 段按三个 miss 率拟合出 **0.0195 µs/KB = 51 GB/s 边际带宽**，已贴 Gen5 x16 实际上限。

本轮把 B(pf=0) 从 10.57 → 3.63 µs/层（−66%），手段：
counters 从 pinned host 移入显存（每层省 4 次 PCIe 原子往返，是最大一项）、
alias kernel 融进 mask kernel、prefetch=0 时跳过 miss 导出（省 `__threadfence_system`）、
指针算术替代 `select`/`reshape`（每步省 122 次 TensorImpl 堆分配）、
栈数组替代 `std::vector`、`_pool_views` 缓存（每步省 183 次 ATen 调用）、
热路径去掉 `cudaGetLastError`、mask 计数改 warp 归约。

**剩下的杠杆不在代码里**：`KEEP_BLOCKS`（驻留尾部）——C 的成本对 miss 数严格线性
（5%/10%/15% → 9.92/12.27/14.53 µs/层），256→512 块可把 miss 大致减半，
代价是每请求多约 1.28 GB 显存。**这比 `STAGING_BLOCKS` 有意义得多。**

### 8.5 offload 触发时机是脆弱的

引擎的 offload 决策在 `StreamCacheResource::incrKVBlock` 里，而**该函数只在需要新的
64-token 块时才被调用**。`il=62841` 落在第 982 块的第 59 个 token → 下一个块边界在
生成第 8 个 token（还没过 `AFTER_TOKENS=16`，不触发），再下一个在**第 72 个**。

- out=512 → 覆盖 86% 的解码步（测量因此略微低估稳态成本）
- out=128 → 只覆盖 44%
- **out=64 → 一次都不触发**，此时方案"看起来"和基线一样好

**触发时机由 `context_len mod 64` 决定，实际上是任意的。** 新增
`[v32_plan] gen=N key=K OFFLOAD ENGAGED at step=M` 日志；未生效会持续告警。
**任何 offload 测量都必须先确认这条日志存在**，否则测的就是基线。

### 8.6 测量基础设施的三个坑（都已修）

1. **共用 server 实例导致 OOM**：为省 700GB 权重加载让 B0/B8/C 共用一个实例，8 个 63k
   请求后显存到 139.38/139.80 GiB，下一个方案首个请求死在 492 MiB 分配上
   （core dump 栈确认 `torch.OutOfMemoryError`）。已改为每模式独立实例。
2. **counters/prof 跨模式累计**，无法归因。已在 `_refresh_mode` 时重置。
3. **`pkill -f` 自匹配**把驱动自己杀掉（踩了两次）。pattern 必须写成
   `rtp_llm_rank[-]` 这种形式；另外 worker 会把进程名改成 `rtp_llm_rank-N` /
   `rtp_llm_backend_server`，只匹配启动命令会留下 8 张卡不释放。

### 8.7 新增 / 修改

修改：`python/v32_ctx.cu`、`python/v32_capacity.py`、`python/v32_lossy_test.py`、
`harness/bench_ab.py`（固定 prompt 文件 + 输出捕获 + `--varied`）、`harness/run_ab.sh`（`MODE=C`）

新增：
- `python/v32_serve_bench.py`：per-layer 微基准，含启动地板探针、临界路径 vs 总搬运双时钟
- `harness/run_abc.sh`：A/B0/B8/C 驱动（每模式独立实例，`V32_MODE_FILE` 选方案）
- `harness/summarize_abc.py`：汇总 + **拒绝比较不同请求**（prompt digest / input_len 漂移即报错）+ 提前结束检测
- `harness/verify_ckpt.py`：checkpoint 完整性（**按文件大小 vs 头部声明做截断检测**；
  index 的 `total_size` 按 2 字节/元素估算，对 fp8 会翻倍，不能用作完整性判据）

新增环境变量：`V32_LOSSLESS`（方案 C）、`V32_MODE_FILE`（运行时切方案）、
`V32_SEL_TRACE`（选择校验和，pre/post 双份）

### 8.8 复现

```
构建+单测: cd v32_migration/python && python v32_lossy_test.py   # 含 staging 驻留回归
微基准:    python v32_serve_bench.py                            # 含启动地板
部署:      cp v32_capacity.py <runtime>/site-packages/rtp_llm/
           cp /home/admin/rtp-hol/v32ctx_build/v32_ctx.so <runtime>/site-packages/rtp_llm/
四点对比:  SUITE="A B0 B8 C" OUT=512 REPS=2 TP=1 DP=8 KVMB=12288 bash harness/run_abc.sh
验完整性:  python harness/verify_ckpt.py /home/admin/models/DeepSeek-V3.2-Exp
```

**跑之前必查**：`[v32_plan] ... OFFLOAD ENGAGED` 是否出现；`admit skipped` 是否为 0；
`[v32_unsafe]` 是否为空；`stats` 里 `unsafe` 是否为 0。

### 8.9 下一步

1. **用真实语料重建验收基线**：测 A 自身的不一致率作噪声底，再判定残留的 1/6（§8.2）。
2. **并发**：完全未测，需要 PD 分离环境（单机 8 卡装不下两套 640GB 实例）。
3. `KEEP_BLOCKS` 扫描（§8.4）：显存换 C 的取数量，这是当前最有意义的旋钮。
4. 代码仍在 `v32_migration/` 临时位置，方案定型后合回 `rtp_llm/` 与 `rtp_llm/cpp/`。

## 9. 第五个 bug + 用真实语料/噪声底闭环（2026-08-31）

### 9.0 结论修正

§8 的判据（退化 prompt 逐位比对）本身不可靠，§8.2 已指出。本节用**真实语料 +
噪声底**重测，结论有两处修正：

1. **§8 的 +0.68% / +0.90% 建立在一个错误的打分上**（第五个 bug，§9.1），当时被退化
   prompt 掩盖。修正后单请求 63k、offload 生效、真实语料：**C 稳态落在基线自身噪声带内**
   （一致度 0.7035 vs 噪声底 0.7468），TPOT 均值 +2.8%，逐步 total 136–139ms 与 A 同带。
2. **曾出现的 +17% 是隔离实验的伪影**（§9.3），不是方案成本。

### 9.1 第五个 bug：灌池步用外推 kvlen 打分（致命，被退化 prompt 掩盖）

`_bookkeep_pool` 里 `kvlens` 是**外推**的（上一步落地值 + 步差，为零同步）。首次 bulk admit
按外推 kvlen 灌到 `kvlen_est-1`，而 `fp8_paged_mqa_logits` 用的是设备上的**真实**
`kvlen_d`。两者之间的缺口没被灌入 → 读到零 → 恰好是**最近的 token**（通常分数最高）→
那一步的 top-2048 被垃圾占据。**一步错就够了**：它发生在解码第 1–2 步，经自回归污染
整个生成（"第 7 字分叉"）。

定位手段（`V32_WAVE_CMP`，`v32_offload_hook.py::_wave_compare`）：同一步跑两个波，
逐 step 聚合 top-2048 重合度：

```
step=804 mean_overlap=0.0039 admits=61   ← 首次灌池步：错
step=805 mean_overlap=1.0000 admits=61   ← 回填后
step=806 mean_overlap=1.0000 admits=0    ← 稳态全对
```

修（`v32_capacity.py`，`catching_up = bool(admits)`）：**灌池步不接管，交给引擎原生波**。
安全，因为此时前缀尚未 offload（offload 在几十步后才发生，且已 offload 的情况由
§8.1-② 另行守护）。修后我们接管的**每一步重合度都是 1.0000**。

### 9.2 引擎本身非确定 ⇒ 噪声底判据（工具：compare_noise.py）

同一 prompt、`top_k=1`、固定 seed，**A（无 hook）** 在真实语料上 **6 次 6 个不同输出**
（MoE all-to-all 浮点求和顺序）。所以逐位比对无效。改用**成对首次分叉位置 + 前缀一致度**：

| | 一致度 | 首次分叉 |
|---|---|---|
| A vs A（噪声底） | **0.7468** | 第 1102 字 |
| 打分错误时（第五个 bug 未修） | 0.0038 | 第 7 字 |
| **C 修复后 vs A** | **0.7035** | 第 1102 字（回到噪声翻转点） |

判据：方案与基线的一致度 ≥ 基线自身一致度 × 0.9 即"在噪声带内"。C 通过。
**语料用真实源码**（`rtp_llm/cpp`，`bench_ab.py --corpus`）；两种合成 filler 都不合格
——重复串让 token 打分趋同（tie-break 不稳 + 掩盖错误），词沙拉则一致均匀不可预测。

### 9.3 +17% 是隔离实验的伪影（KEEP=0）

为隔离打分正确性，曾用 `KEEP_BLOCKS=0` 关掉引擎 offload、只留打分替换 → 测得 163ms
（+17%）。**但 KEEP=0 使 offload 永不生效，于是每层每步都在跑白做的 `_mirror_chunk`**
（把 KV 拷 host 镜像，为一个永不发生的 offload 做准备）。
正常 KEEP=256、offload 生效时，逐步 `total(mirror/serve/proc)=(0/1/2)ms`，
mirror 稳态归零（offload 后 KV 已不在 GPU），我们的净开销每步仅 ~3ms。

**教训：隔离一个变量时，务必确认所用旋钮本身不改变别的成本项。** KEEP=0 同时关掉了
offload 和"mirror 的正当性"，把一个伪开销算进了打分头上。

### 9.4 打分波本身不慢

`V32_WAVE_PROF`（CUDA event）拆我们波的 GPU 时间：`logits=0.048 + topk=0.045 =
0.093 ms/层`，与引擎原生波的 `0.09 ms/层`（`_wave_compare` 实测）**相同**——本该如此，
算的是同样的东西、同样的数据量。所以打分不是开销来源，§9.3 才是。

### 9.5 本节新增/修改

- `python/v32_offload_hook.py`：`_wave_compare`（双波重合度，`V32_WAVE_CMP`），逐 step/层归因
- `python/v32_capacity.py`：`catching_up` 灌池步 stand-down（§9.1）；`_pre_topk_profiled`
  （`V32_WAVE_PROF`）；`_pool_views` 也用于 `pre_topk`；`_stats["fill"]` 计数
- `harness/bench_ab.py`：`--corpus`（真实源码语料）
- `harness/compare_noise.py`：**新增**。成对首次分叉 + 一致度，与噪声底比较；按位置分桶
- 新增环境变量：`V32_WAVE_CMP`（双波重合度采样步数）、`V32_WAVE_PROF`（event 计时步数）

### 9.6 五个 bug 一览（本项目至此）

| # | bug | 表现 | 为何难发现 |
|---|---|---|---|
| 1 | staging 驻留误判（§8.1-①） | 读到别的逻辑块数据 | top-k 少落在该区 |
| 2 | 无声退回引擎打分波（§8.1-②） | offload 后读已释放 indexer-K | 依赖异步拷贝时序 |
| 3 | offload 后重建索引池（§8.1-③） | 池永久污染 | 外推 kvlen 偏 1 步才触发 |
| 4 | admit 静默跳过（§8.1-④） | 池有洞 | 冷启动才暴露，块复用掩盖 |
| 5 | **灌池步外推 kvlen 打分（§9.1）** | 该步 top-k 全错 | **退化 prompt 完全掩盖** |

共同点：都是**静默数据污染**，且都被"退化 prompt + 逐位比对"这套错误验收掩盖过。
现在每条相关路径都有计数器（`bail/unsafe/fill/admit-skip`）+ 双波重合度可随时复查。

### 9.7 复现（真实语料 + 噪声底，推荐的验收流程）

```
噪声底: MODE=A ... bash run_ab.sh
        bench_ab.py --ctx 63000 --out 512 --reps 6 --corpus <repo>/rtp_llm/cpp --tag Acorpus
方案:   MODE=B ... EXTRA_ENV="V32_MODE_FILE=... "  # echo C > mode 文件
        bench_ab.py ... --corpus <repo>/rtp_llm/cpp --tag Ccorpus
判定:   compare_noise.py --jsonl ... --base Acorpus --scheme Ccorpus
        # 一致度 >= 噪声底 x 0.9 即通过
重合度: EXTRA_ENV="... V32_WAVE_CMP=250"，看 [v32_wavecmp] step= 每步应 1.0000
```

### 9.8 现状与下一步

- **单请求正确性已闭环**：接管步重合度 1.0000，噪声底判据通过。
- **单请求延迟**：offload 生效稳态在基线噪声带内（±2ms，均值 +2.8%）。
- **仍未做**：并发（需 PD 分离环境）；`KEEP_BLOCKS` 显存/取数权衡扫描；
  代码合回 `rtp_llm/` 主树。
- **诊断探针（`V32_WAVE_CMP/WAVE_PROF/SEL_TRACE`）默认关，生产前应移除或永久 gate。**

## 10. 成本阶梯归因 + 步进快路径（2026-09-01）

### 10.0 一页结论

用"逐项旁路 + 独立实例 + 真实语料 + 固定 prompt"的端到端阶梯，把 C 相对 A 的
延迟拆到了项（harness/run_cost_ladder.sh，每档 4 reps、out=512）：

| 档 | 配置 | TPOT | 相邻增量 | 归因 |
|---|---|---|---|---|
| L0 | A 无 offload | 137.90 ms | — | — |
| L1 | + hook 拦截（HOOK_LEVEL=0） | 137.85 | ~0 | hook 零成本 |
| L2 | + 替代打分波（SKIP_PROCESS=1, MIRROR=0） | 138.87 | **+1.03** | 打分波+独立池维护 |
| L3 | + process_layer（MIRROR=0） | 140.28 | **+1.41** | python 热路径 |
| L4 | B0 无镜像（KEEP=256） | 139.16 | −1.12※ | ※B0 丢 miss，attention 变少 |
| L5 | B0 有镜像 | 140.08 | **+0.93** | 镜像追赶摊销 |
| L6 | C 完整 | 140.65 | **+0.57** | PCIe 取数 + 恢复的 attention |

**注意 L4 的负增量**：B0 丢弃 miss 让 sparse attention 少算，所以 C−B0 的 0.57ms
**不是纯 PCIe**，还含"恢复这些 token 后 attention 多算的部分"。想再拆需要
V32_LOSSLESS_DIAG_MODE（0=生产/1=只重映射不取数/2=同 kernel 丢弃），但 mode=1 让
attention 读无效 scratch 会 500，**mode=1 不可用**；mode=2 可用作 B0 等工作量对照。

### 10.1 步进快路径（回收 process_layer 的 1.41ms）

三个版本的教训：
1. **v1（try_serve 放在 _process_layer 内部）只回收 0.10ms** —— 位置太晚，
   python preamble（_entry/_grow/mirror 判定/计时器）还在每层跑。
2. **v2（hook 边界 + 每层完整检查）回收 0.67ms** → Cfast2 = 139.98ms。
3. **v3（步进 arm：ctx_step_plan / ctx_step_serve）**：layer 0 做一次请求级检查并
   arm C++ 计划；其余 60 层每层一次 5 参调用（step, layer, kbt, topk, pool），
   C++ 侧按层缓存 LossyState 指针，消除每层 dict/plan/双重 map 查找。
   pool 直接传原生 [blocks,64,576]（impl 用最后一维定行宽），省掉 flat 视图。

**v3 的三个安全点**（都有回归测试 test_step_fast_path）：
- release/recycle 时 disarm 并清空层指针缓存 —— 复用的 block-0 key 永远拿不到旧请求镜像；
- 未注册层返回 false，注册/过渡步由慢路径全权处理；
- **_purge 跳过 armed key**：快路径不再刷新 seen 水位，>1500 步的长生成否则会被
  purge 误回收活跃请求（Cfast2 也有此隐患，本轮一并修复）。

**生效验证**（hb 'fast' 计数器）：offload 生效后 **~93% 的解码步层调用由 C++ 路径
服务**（fast=14518 ≈ 238 步 × 61 层 / 256-token 请求），残余 slow 为 offload 前
~43 步与 dummy 同步步。**运行间 TPOT 均值抖动约 ±0.7ms**，v2/v3 的差异在噪声内；
带快路径的 C 诚实区间为 **139.98–140.63ms vs A 137.90**（真实语料、out=512、4 reps）。
剩余预算 = 打分波维护(~1.0ms) + 镜像追赶摊销(~0.9ms) + 无损取数(~0.6ms)——
前两项归属 indexer 解耦项目（§10.2），python 侧已无可挤空间。

**跳过稳态镜像续拷是安全的**：offload 只发生一次，非驻留区固定且注册时镜像已
durable 覆盖（注册 gate 在 n ≥ hist−LAG），lossless fetch 永远不会读到镜像未覆盖区。

### 10.2 其余结论

- **CUDA event 对 <10µs kernel 不可信**：一对 event 自身 5–6µs，曾把 mask/append
  高估为 0.54/0.55ms；真值要用"关掉某项测 TPOT 差值"（append 实测仅 0.18ms）。
- **引擎直写 indexer pool 不值得为延迟做**（append 仅 0.18ms），但值得为架构做：
  见子代理规划 —— 用引擎 HybridPool 双 group（default + dsa_indexer_k 独立块表/生命
  周期），offload 只释放主 KV；删除整个影子池（_ipool/_ireq/bulk_admit/batch_append），
  从根上消灭 bug #3/#4/#5 的土壤。涉及 allocator/slot_mapping/PD/reuse/CUDA graph
  + 重建 libth_transformer.so，须独立立项配自己的验证基线（部署 wheel 是 8/20 内部
  构建，换 .so 会使现有测量失效）。
- **真值 kvlen admit 已证伪**（§9 补充）：真值配异步（滞后一步）块表反而读错块，
  确定性偏差在第 8 字；stand-down 是当前架构的正确解而非权宜。

### 10.3 新增

- `harness/run_cost_ladder.sh`：七档归因阶梯；`run_fetch_ladder.sh`（mode=1 档作废）
- `V32_SKIP_PROCESS` / `V32_LOSSLESS_DIAG_MODE` / `bench_ab.py --force-length`
- `ctx_step_plan` / `ctx_step_serve`（v32_ctx.cu），`process_layer_native_fast`（v32_capacity.py）
- `ctx_lossy_register` 增 kv_host 参数（C++ 保留 host mirror 所有权）

## 11. 独立 indexer-K 池：引擎级解耦落地（2026-09-01，#31）

### 11.0 一页结论

引擎侧 HybridPool 双 group（`default` 主 KV + `dsa_indexer_k` 132B/token 独立块表）
已全链路接通，编译通过（allocator 单测 PASS）。开关 `V32_INDEPENDENT_IDX_POOL=1`
（默认关，关闭时行为与部署版完全一致）。打开后：offload 只释放主 KV group，
indexer-K 常驻显存 → 原生打分波对 offloaded/非 offloaded 请求统一正确，python
打分波 + 影子池（≈1.0ms/步）整体 stand down；hook 只剩主 KV host mirror + 无损
取回/重映射。这从根上消灭 §9.6 的 bug #3/#4/#5 土壤。

### 11.1 接线清单（本节全部完成）

- 声明：`deepseek_v2.py _post_build_model_config`（门控 env + is_sparse），
  desc 对齐 DSv4 先例（OPAQUE_KV / EXPLICIT / 64 tok/block / 132B）。
- 主组选择：`OpDefs.h getLayerCache(int)` 识别 dsa_indexer_k 伴随组，把
  `indexer_cache_base/seq_size_per_block/group_id` 挂在主组 LayerKVCache 上。
- 块表下发：`PyWrappedModel::setupKVCacheForAttentionInputs` 把伴随组
  `kernel_block_id{,_device}` 挂到主组 attention inputs（保持单输入快路径）。
- 槽位映射：`SparseMlaParams::fillParams` 用基类 batch_indice/positions 在
  CPU 镜像槽位算术，pinned→单次 H2D（`indexer_slot_mapping`，pybind 只读导出）。
- 写/读切换：`indexer_op.py` 全部 6 个消费点走 `_indexer_write_dest` /
  `_indexer_score_src`（声明池缺映射=硬错误，未声明=回退 legacy kv_scale_base）；
  `hybrid/indexer.py` 三个调用点传 `indexer_slot_mapping`。
- 释放解耦：`freeBlockListByTag`（KVCacheAllocator 默认忽略 tag → SingleType 行为
  不变；Hybrid 解析 tag→gid→`kv_cache_groups_[gid]->free`）；
  `offloadPrefixBlocks`/`releaseAdmissionRing` 只放 `default`。
  注意：Hybrid 此前根本没实现 `freeBlockList`（基类空操作）——不加这条，双池下
  offload 会静默不释放。
- python 待命：`v32_capacity.engine_idx_active()`（首个 kv_cache 探测一次）。
  engine 模式下 pre_topk 保留步进簿记（_step/_purge/_bookkeep）但跳过
  `_bookkeep_pool/_sw/_finish_step_pool/admit/append/单波`；`single` 恒真
  （native topk 即全量正确）；镜像只拷主 KV（`st["idxp"]` 不再分配）；
  `_pool_views` 容忍主组无 scale 区；hook 的 `_SANITIZE_KVLEN` 在 engine 模式
  下禁用（kvlen 必须真实）。

### 11.2 已知边界（打开开关前须知）

- CUDA graph：indexer 槽位映射是 host 计算，capture 会被
  `RTP_LLM_CHECK(!is_cuda_graph)` 明确拒绝（fail loud）。
- PD 分离 staging-ring admission：`StreamCacheResource` 的 admission 分支有
  `groupNums()==1` 守卫，双池下自动失活（回退整段 malloc，不影响单机方案）。
  PD/cache-store 双 group 传输未做。
- 槽位算术假定 indexer 块大小 == 主池 seq_size_per_block（当前都是 64）。
- 影子池代码（_ipool/_ireq/ctx_bulk_admit/ctx_batch_append）**未删除**：engine
  模式下不执行，等端到端验证（噪声底判据）签收后再删。

### 11.3 验证结果（2026-09-01 晚，端到端 PASS）

新 .so 上三轮各起新服务器（63k 真实语料 pinned prompt × 4）：

| | A1 | A2 | C_idxpool (`V32_INDEPENDENT_IDX_POOL=1`) |
|---|---|---|---|
| TPOT | 141.54ms | 142.47ms | **143.57ms（伺服税 +1.6ms / +1.1%）** |
| 一致度 vs A1 | — | 0.8052（噪声底） | **0.7371 ≥ 0.9×0.7744 ⇒ PASS** |

- C 的 4 次输出中 3 个 sha 与 A 基线逐字节相同；每次 bench 都真实触发
  `offloaded 618 prefix blocks`（68% 主 KV 释放）；hook 零异常；
  `[v32_sw] engine=True`（python 打分波确认待命）。
- 对比旧影子池方案 C（+2.1~2.7ms）：**伺服税降约 1ms**，来源正是打分波+影子池
  维护的消失。

集成阶段共修掉 4 个 bug（全部只在双池/offload 组合下暴露）：
1. 基线崩溃：`DeepSeekV2._post_build_model_config` override 在门控关闭时
   把基类默认 `kv_cache_spec_descs` 填充也挡掉 → 引擎断言 size 0 != 61。
   修复：回落 `super()`。
2. OPAQUE_KV 每层张量是 2 维 `[blocks, block_bytes]`，indexer 写内核要求
   legacy 3 维 `[blocks, 64, 132]` → resolver 里带缓存的归一化视图
   （`_indexer_pool_3d`）。
3. `HybridKVCacheAllocator::free` 不过滤 0 哨兵 → 首条 offload 流释放即
   `block:0 decrease zero ref count` 崩溃；同时移植 SingleType 的
   insertIntoCache 首哨兵截断（B6/B7 类 reuse 污染防护）。
4. 双池下主组 `kv_scale_base` 是 **None**（非空张量）→ `_pool_views`
   `.data_ptr()` 每层抛异常，serve 链整体失效（表现为 TPOT 670ms +
   一致度 0.52 的无效轮）。修复后归零。

- 运行时：`/home/admin/rtp-hol/runtime/rtp-idxpool-20260901`（本工作树 wheel +
  修复后 python + 最新 v32_ctx.so）；数据 `logs/idxpool_verify.jsonl`。

### 11.4 影子池删除（同日晚，随验证通过执行）

- **自动门控**：`deepseek_v2._post_build_model_config` 在
  `RTP_KV_OFFLOAD_KEEP_BLOCKS/RTP_KV_ADMIT_RING_BLOCKS>0` 时自动声明双池
  （影子池兜底不复存在，单池 + offload = 打分读已释放显存）；
  `V32_INDEPENDENT_IDX_POOL` 变为显式覆盖（=1 强开 / =0 强关并 ERROR 告警）。
- **v32_capacity.py 1372 → 809 行**：删 `_ipool/_ifree/_ireq/_sw/_bookkeep_pool/
  _finish_step_pool/_pre_topk_profiled/_dual_wave_sel/_idxp/_layerctx/
  offloaded_rows_hint` 及 SINGLE_WAVE/IDXNB/SKIP_APPEND/WAVE_PROF 环境变量；
  `pre_topk` 只剩 layer-0 步进簿记（恒返 None，打分全部原生）；
  `_mirror_chunk` 只镜像主 KV；`single ≡ engine_idx_active`；
  非 engine 且 offloaded ⇒ `_stats["errors"]`（配 `[v32_unsafe]` 告警，
  auto-gate 下不可达）。
- **hook**：删 `_SANITIZE_KVLEN` kvlen 造假、`_WAVE_CMP/_wave_compare` 双波对比。
- **v32_ctx.cu**：删 `batch_append_kernel/bulk_admit_kernel` 及
  `ctx_batch_append/ctx_bulk_admit` 绑定；重建后
  `v32_lossy_test` 全 PASS（lossless/staging/step-fast，max err 0）。
- **验收 PASS**：`C_noshadow` 轮（删除后全链路）TPOT **142.99ms**
  （伺服税 ≈+1.0ms，三轮 C 中最低）；一致度 vs A1 = **0.8024 > 噪声底 0.7744**
  （与 A2 的 0.8052 无法区分，first-diff 中位数 1954 与 A1×A2 相同）；
  4 次 bench 全部真实 offload（618 块/次），hook 异常 0，无 core dump。
- 删码教训：大段切片删除吃掉了函数间的模块级变量（`_faststep/_fast_key/
  _HAS_STEP_SERVE`），语法检查发现不了 NameError——用 AST 未定义名扫描兜底。

### 11.5 纯净主干对照（"有没有改坏基线"，2026-09-01 深夜）

同机同 harness 跑纯净 github 主干 `d4d9bf18b`（runtime 零本项目代码；主干本身
编译不过，需带分支里的 grpc `-Werror` 构建补丁 `0002-Fix-MSG_CTRUNC`）：

| | A_master（纯净主干） | A1/A2（本树，门控关） | C_noshadow |
|---|---|---|---|
| TPOT | **141.16ms** | 141.53 / 142.47ms | 142.99ms |

- 关闭态改动无可测开销（差值 < A1 与 A2 自身的 0.94ms 波动）；A_master 的
  4 个输出 sha 全部与本树各轮重合（逐字节相同）。
- **137.90ms 的旧基线属 8/20 内部 RDMA 构建**，与 github 主干代码基线不同；
  github 主干本身就是 ~141ms。跨构建比较 TPOT 无意义，比较必须同 wheel。
- 纯净树保留在 `/home/admin/workspace/aop_lab/wt-master-baseline`，
  runtime `rtp-master-d4d9bf18b`。
- 待做：PD/cache-store 双 group；CUDA graph 槽位映射图内化；staging-ring
  admission 双池版（`groupNums()==1` 守卫解除）。

## 12. 终局设计 + 全项目踩坑与经验（2026-09-01 收官，供后续开发复用）

本章是自足的总结：只读这一章即可接手。细节出处见 §7–§11。

### 12.0 终局架构（当前代码即此状态，全部未 commit）

**问题本源**：DSA 稀疏注意力每层要对全历史 indexer-K（132B/token）打分选 top-2048。
旧布局里 indexer-K 藏在主 KV 池的 scale 区、共用块表 —— offload 释放主 KV 必然
连带释放 indexer-K，原生打分读已释放显存。前四周的"影子池"（python 自管 GPU 池
+每步 132B 追加+打分波）是对这个布局缺陷的绕行，本身孕育了 5 个静默污染 bug。

**终局方案：缓存拓扑级解耦。**

```
HybridPool（引擎，deepseek_v2._post_build_model_config 声明）
├─ group 0 "default"       主 KV（MLA 576×bf16/tok）  offload 时可释放
└─ group 1 "dsa_indexer_k" indexer-K（132B/tok, OPAQUE_KV, 64 tok/块）常驻

写入：indexer_op._indexer_write_dest → 独立池 + SparseMlaParams.indexer_slot_mapping
打分：indexer_op._indexer_score_src → 独立池 + attention_inputs.indexer_cache_kernel_block_id_device
      （offloaded/常驻请求统一走原生 kernel，无 python 干预）
释放：StreamCacheResource::offloadPrefixBlocks → freeBlockListByTag("default")
      只放主 KV；indexer 组块表/块永不清零
伺服：python hook 只剩两件事 ——
      ① 主 KV 异步镜像到 pinned host（_mirror_chunk，分块摊销）
      ② 原生 top-2048 出来后，对 offloaded 行做无损取回/重映射
         （ctx_step_plan/ctx_step_serve 步进快路径 ~93% 层命中，17µs/层）
```

**门控**：开 offload（`RTP_KV_OFFLOAD_KEEP_BLOCKS>0`）即自动声明双池；
`V32_INDEPENDENT_IDX_POOL=1/0` 仅作显式覆盖（=0 会 ERROR 告警：等于自杀）。

**关键文件**（改动全景 `git status`）：
- 声明/门控：`rtp_llm/models/deepseek_v2.py`
- 主组选择+伴随张量：`rtp_llm/models_py/bindings/OpDefs.h`（getLayerCache）
- 伴随块表下发：`rtp_llm/cpp/models/PyWrappedModel.cc`
- 槽位映射：`rtp_llm/models_py/bindings/cuda/SparseMlaParams.{h,cc}`
- 读写切换：`rtp_llm/models_py/modules/base/cuda/indexer_op.py`（两个 resolver）
- 释放解耦：`KVCacheAllocator.h`/`HybridKVCacheAllocator.{h,cc}`/`KVCacheManager.{h,cc}`/`StreamCacheResource.cc`
- python 伺服：`v32_migration/python/{v32_capacity.py(809行),v32_offload_hook.py,v32_ctx.cu}`

**验收数字**（同机同 wheel，63k 真实语料，判据=噪声底）：
纯净主干 A 141.16 ≈ 本树 A 141.5/142.5 → 方案 C 142.99ms（伺服税 +1.1%，
输出与 A 不可区分且多次逐字节相同）。

### 12.1 踩坑全录（按类别，含本次集成日）

**A. 缓存/生命周期类（最危险：全是静默数据污染或延迟崩溃）**
1. 影子池五 bug（§9.6）：根源都是"两份真相"——python 池与引擎池的一致性
   要靠水位、tripwire、stand-down 维护。解耦后这一类整体消失。
2. Hybrid `freeBlockList` 是基类空操作：不实现则双池 offload **静默不释放**
   （显存不降但也不崩，极难发现）。
3. 0 哨兵进引用计数：offload 把块表清零后，流结束 free 撞
   `block:0 decrease zero ref count`（崩在**首个请求完成时**，不是 offload 时）。
4. reuse 插入污染：0 哨兵进共享前缀缓存 → 后续请求拿到假块
   （SingleType 曾以同样方式崩过 B6/B7，Hybrid 要单独再补一次 ——
   **同一功能在两个 allocator 里要打两份补丁**）。
5. 双池下主组 `kv_scale_base` 是 **None** 而不是空张量；引擎 OPAQUE 池每层
   张量是 2 维 `[blocks, block_bytes]` 而 kernel 约定 3 维 `[blocks,64,132]`
   —— 布局/形状契约必须在边界处显式归一（`_indexer_pool_3d`），不能假设。
6. override 基类钩子必须回落 `super()`：`_post_build_model_config` 早退把
   默认 `kv_cache_spec_descs` 填充也吞了，基线全崩（size 0 != 61）。

**B. 验证方法论类**
7. 引擎非逐位确定（MoE all-to-all 浮点序）：byte-diff 判据无效，必须
   真实语料 + 噪声底（A 对 A 的一致度做地板，方案 ≥0.9×地板 即过）。
8. 跨构建比较 TPOT 无意义：137.90（8/20 内部 RDMA 构建）vs 141.16
   （github 主干）差 3.5ms 与我们无关。**换 .so 必须重测基线**；
   怀疑"改坏了"时，做纯净基点的单变量对照实验（§11.5，1 小时搞定）。
9. 触发必须可观测：offload 的触发点取决于 context_len mod 64，一轮"全绿"
   可能只是没触发。每次验收都要核对 `offloaded N prefix blocks` 日志条数
   = bench 次数、hook 异常计数 = 0、engine=True。
10. CUDA event 对 <10µs kernel 不可信（自身 5–6µs）；小项开销用
    "关掉某项测 TPOT 差值"。运行噪声 ±0.7ms，单次差异 <1ms 不要下结论。

**C. 工程/工具类**
11. 大段切片删代码会吃掉函数之间的模块级变量（`_faststep` NameError，
    语法检查发现不了）：删完跑 **AST 未定义名扫描** + 单测，再上服务器。
12. python 服务器 cwd 若含 `rtp_llm/` 源码目录，sys.path 会让源码树遮蔽
    runtime site-packages（ImportError 或更糟：跑错代码）。启动脚本固定
    `cd` 到中性目录；日志（engine.log/main_*.log）落在 **$PWD/logs**，
    找日志先 `readlink /proc/<rank pid>/cwd`。
13. 服务器清理模式要含 `rtp_llm_frontend`（不匹配则孤儿 frontend 占端口，
    下一轮 EADDRINUSE）；`pkill -f` 的关键字别撞自己命令行。
14. bazel：离线机复用既有 output_base（`--output_base=… @//target`）；
    共享 output_base 下 `bazel-bin` 软链会指向**别的树的产物**，取 wheel
    认时间戳、认 execroot 路径；上游主干可能带 `-Werror` 编译不过
    （grpc MSG_CTRUNC 补丁是纯构建修复，可安全带入对照树）。
15. rank 崩溃 core dump 10–15GB/个 × 8，几轮就是百 GB：调试期
    `ulimit -c 0` 或每轮清理。
16. 后台长任务（>10 分钟）用 `setsid nohup`，配 tail -F 监控关键行；
    qoder Bash 工具的 timeout 上限会杀长命令。

### 12.2 可复用的做法（正面清单）

- **消灭"两份真相"优先于优化同步逻辑**：影子池方向修了五个 bug 也没堵死
  类别；拓扑级解耦一次移除整个 bug 家族，且更快（伺服税 2.1→1.0ms）。
- 新缓存布局跟着 **DSv4 多组先例** 走（OPAQUE_KV/EXPLICIT/tag），
  不发明新机制；kernel 兼容靠边界归一化视图，不改 kernel。
- **fail loud**：声明了池但缺槽位映射=assert；CUDA graph 未支持=显式拒绝；
  不安全组合=ERROR 日志。所有"回退到旧行为"必须是显式分支而非静默兜底。
- 验收流程固定为：每档全新服务器 → 同一 pinned 真实语料 → 4 reps →
  噪声底判据 + TPOT + 触发证据三件套（脚本 `harness/run_idxpool_verify.sh`
  可直接复用）。
- 大改动的顺序：编译门（单目标）→ 单测 → 整包 wheel → 基线重测 →
  方案轮 → 删旧码 → 回归轮 → 纯净对照。每步产物/日志留档（jsonl）。

### 12.3 挂起事项（后续立项）

- CUDA graph：indexer 槽位映射 host 计算 → 图内化（参考
  fillDecodeCudaGraphParams 模式）。
- PD 双组已在 §13 完成；后续重点是 RDMA 链路实测与 cache-store 高并发传输优化。
- 槽位算术假定 indexer 块大小 == 主池 seq_size_per_block（当前都是 64），
  分化时需带上 `indexer_seq_size_per_block`。
- FlexLB 块感知调度需按上游新 CostBased 体系重写（§1 未应用清单）。

### 12.4 产物位置（NAS vs 本地盘）

**NAS（`/home/admin/workspace/aop_lab/`，持久）**：
- 代码：`wt-dsa-offload/`（分支 rym/feat/dsa_offload，全部未 commit）；
  纯净对照树 `wt-master-baseline/`（d4d9bf18b + grpc 构建补丁）。
- 实验数据存档：`wt-dsa-offload/v32_migration/data/`
  （idxpool_verify_20260901.jsonl/.log、cost_ladder_20260901.jsonl、
  部署 wheel 的 sha256）。
- 全部脚本：`v32_migration/harness/`（含收官夜用的
  redo_c_rung.sh / run_a_master.sh / pkg_master.sh）。

**本地盘（机器回收即失，均可从 NAS 重建）**：
- runtime：`/home/admin/rtp-hol/runtime/rtp-idxpool-20260901`（重建：
  bazelisk --output_base=$OB build @//rtp_llm:rtp_llm --config=cuda12_9
  → 仿 harness/package_b_runtime.sh 打包 + 拷 v32 python 模块 + v32_ctx.so
  + mla_attention.py 追加 hook import）、`rtp-master-d4d9bf18b`（pkg_master.sh）。
- v32_ctx.so：`/home/admin/rtp-hol/v32ctx_build/`（重建：python 目录下
  `V32_CTX_BUILD=… python v32_ctx_build.py`，源码在树里）。
- bazel output_base：`~/.cache/bazel/_bazel_admin/d1da4251…`（59GB externals，
  离线机的关键缓存；丢了需要有 github 网络的机器重新 fetch）。

## 13. PD 双 group 传输与并发实测（2026-09-03）

### 13.0 当前数据流

BlockTree 与 cache-store 协议没有改，仍作为 tag-aware 黑盒使用。新增接线只解决
“哪个 group 用哪张物理块表”及 decode 首次分配几何：

1. P 端 `PyWrappedModel` 保留 `{default, dsa_indexer_k}` 两份 tagged
   attention inputs。`default` 仍附带 indexer kernel table 供 DSA 计算；两份
   cache-store inputs 各持有自己的 physical block table。
2. `GenericMoeModel` 只为 `default` 创建 FMHA；默认 MLA writer 发布主 KV，
   每层计算结束后 companion writer 以 `dsa_indexer_k` tag 发布 indexer-K。
   `WriteCacheStoreOp` 收紧为“一 writer 对一 LayerKVCache”，禁止跨池共用块表。
3. D 端 HybridPool 首次分配支持 per-tag target：`default` 只分配
   `block0 + staging32 + keep256 + scratch1 = 290` 块；`dsa_indexer_k` 分配完整
   982 块。随后 default 逻辑表扩回 982，中间以 0 表示 CPU-resident prefix；
   indexer 表始终 dense。
4. decode normal tagged load 直接恢复完整 indexer 和 default 的 resident 部分。
   缺失的 693 个 default blocks 通过 cache-store 分批直接写入 D 端 pinned host
   mirror（默认每批最多1024块；63k是一批），不再先落 GPU ring 后 D2H。
   scratch block 仅作为 admission 标记/布局探针，load 完立即释放。
5. decode 原生 indexer 对完整 GPU indexer-K 打分；top-k 指到已 offload 主 KV 时，
   `ctx_step_serve` 从 host mirror token-granular H2D 到 staging slots。

### 13.1 本次新增/修改

- producer/tag 路由：`PyWrappedModel.cc`、`generic_moe.py`、
  `kvcache_store.py`；同时按已有 `79f82d4e1` 方案修正 DSv4 prefill/DSpark
  的 per-tag writer 配对。
- allocation：`MallocInfo.init_seq_len_by_tag`；Hybrid allocator 的 capacity、
  common malloc、incr malloc、失败诊断统一使用同一 tag target；group-local
  reuse count 替代 aggregate count。
- admission：`StreamCacheResource` 按 tag 识别 V3.2 双池，只裁 default，验证
  indexer dense；offloaded stream 不进入 BlockTree/device/memory/remote reuse。
- transfer：`DecodeRpcServer::loadPrefixViaRing` 改为 default→pinned host；
  TCP host destination 用 `memcpy`，CUDA destination 保持原路径。
- lifecycle：AdmissionStore 删除 GPU idxp shadow，增加 generation；Python fast/slow
  path 都检查 generation，延迟 purge 只能释放相同 generation；FULL group 的
  reference/free 过滤 `<=0` sentinel。
- prefill fast path：非 SparseMlaParams 情况下，用 GPU 上的 positions、batch index、
  indexer block table 向量化生成 companion slot mapping；decode 热路径不变。

### 13.2 两机验证拓扑与环境坑

- P：`11.86.13.78`，TP2×DP4；D：`33.240.36.239`，TP1×DP8；
  两端均 8×H20；TCP cache-store（新增机器无现成 RDMA 配置）。
- 远端没有模型/runtime：从本机复制 643GB checkpoint 和2.6GB runtime；所有
  远端控制均由本机隔离 tmux 的 SSH pane 提交。
- runtime 必须自带 jsonschema/attrs/referencing/rpds；本机全局包会掩盖缺依赖。
- 当前 start_server 的 DP rank 端口 stride 是9（frontend 27001/27010/...，
  RemoteGenerate 再+1），旧 harness 写死的 stride8 会打到 HTTP/错误服务，表现为
  `Method not found` / `Trying to connect an http1.x server`。
- rsync 仅按 size+mtime 会漏掉同尺寸、同时间戳的重建 `.so`；部署二进制必须
  `rsync --checksum` 并核对 SHA256。

### 13.3 正确性与生命周期结果

- A_PD 与 C_PD 的 62,841-token 请求输出 hash 均为
  `5c5579eee8f9100f`（16-token输出），逐字节相同。
- 最终 64-token C smoke：wall 64.24s、host pull 24.41s，成功。
- 连续两条 C 请求复用 block0=1，generation 1→2，均成功；最终复测也无
  hook error / 新 core dump。
- 每条请求实际 geometry：total=982、resident=289、scratch=1、offloaded=693；
  indexer 982 blocks dense。请求结束后两个 pool refcount 回到0。

### 13.4 并发性能（公网 TCP，63k input / 64 output）

| conc | A_PD（无 offload） | C_PD（双池+offload） |
|---:|---|---|
| 1 | 1/1，wall 37.9s，1.7 tok/s | 1/1，wall 57.5–64.2s，0.8–1.1 tok/s |
| 4 | **2/4**，wall 46.9s，2.7 tok/s | **4/4**，wall 95.6s，2.7 tok/s |
| 8 | **4/8**，wall 70.0s，3.7 tok/s | **0/8**，normal tagged load 出现 `CACHE_STORE_LOAD_SEND_REQUEST_FAILED` |

解释：
- A 失败是明确的 default pool 容量不足：单请求982块；同 rank 第二条只剩943块，
  再加96 reserve，shortfall=135。当前负载均衡碰撞下 conc4/8 仅完成一半。
- C 每请求稳定态 default 仅289块（理论每 rank 6条），indexer 982/8191块
  （每 rank 8条）；conc4 即使3条落同一 rank 仍4/4成功，容量收益成立。
- C conc8 失败不是 KV 容量：失败发生在 normal tagged load 的 cache-store
  `SEND_REQUEST_FAILED`，stagger5s 仍0/8。当前公网 TCP 传输先于GPU容量成为瓶颈。
- 端到端单请求 C 比 A 慢约20–26s（+52%~69%），主要是约43GB default KV
  跨机写 pinned host；这个环境不能代表生产 RDMA。原 ring64 串行实现耗时210s；
  direct-host 后降到21–55s（最终 smoke 24.4s）。
- `bench_cap` 的 TPOT 用 `(cost-prefill)/(output_len-1)`，会把一次性 P→D transfer
  平摊进只有64 token的TPOT，不能当纯 decode step 延迟；本地同机稳态结果仍是
  §11 的约+1ms。

数据：`v32_migration/data/concurrency_20260902/`。当前结论是：**PD双组功能和
容量收益成立；公网TCP下综合吞吐受cache-store传输限制，生产性能必须在RDMA链路
重测，不能用本轮结果外推。**

## 13. PD 双 group：两机实现与性能（2026-09-03）

### 13.1 最终数据流

BlockTree 和 cache-store 协议没有改，仍按 tag 当黑盒复用。

- P 端保留 `{default, dsa_indexer_k}` 两份 attention inputs；default 继续承载
  DSA 计算所需的 indexer kernel table，但 cache-store writer 严格按 tag 与各自
  physical block table 配对。default 与 indexer 每层分别发布。
- D 端首次分配按 tag 定目标：default 只占
  `block0 + staging32 + keep256 + scratch1 = 290` 个物理块；indexer 分配完整历史
  （本轮63k为982块）。default 的逻辑表扩回982，中间693个位置是0 sentinel；
  indexer 表始终 dense。
- normal tagged load 直接恢复完整 indexer 和 default 的 resident head/tail；
  693个缺失 default blocks 通过 cache-store 直接写入 D 的 pinned host mirror。
  host transfer 默认每批最多1024 blocks（63k是一批，139k最多两批）。resident
  head/tail 再从GPU D2H补齐，最后释放1个 scratch block。
- decode 在完整 GPU indexer 上原生选 top-k；主KV miss由 `ctx_step_serve`
  token-granular H2D 到 staging slots。

### 13.2 运行中发现并修复的问题

1. **非 sparse prefill fast path 没有 `SparseMlaParams.indexer_slot_mapping`**：
   本地decode测试没覆盖。现由 `positions_d + batch_indice_d + indexer block table`
   在GPU上生成fallback mapping；decode sparse热路径仍用C++预计算值。
2. **手写DP fanout端口错一位**：当前 start_server 的DP8 frontend端口是
   27001/27010/27019/...（stride=9），RemoteGenerate再+1。旧harness的stride=8
   会打到HTTP端口，报 `Method not found` / `Trying to connect an http1.x server`。
3. **ring64公网TCP串行拉取**：693块分11批、每批61个layer request，耗时210s并
   撞120s deadline。改成直接写pinned host后降到21–55s；再以1024块为默认上限，
   防止139k形成真正无界请求。TCP host目的地直接 `memcpy`，不再逐块构造tensor。
4. **block0复用导致旧host指针/UAF**：AdmissionStore增加单调generation；Python
   fast/slow path使用前均校验generation，变化时清全部layer/native/block缓存；
   延迟purge只能释放同generation，不能误删新请求。
5. **失败路径pinned mirror泄漏**：host load完成前由RAII guard持有，任一错误自动
   release；prepare在新分配前先回收过期graveyard。
6. **FULL group引用0 sentinel**：fork/reference/free过滤`<=0`，稀疏逻辑位置保留，
   block0不会进入引用计数。
7. **远端runtime不自足**：本机全局`jsonschema`掩盖依赖；最终runtime已显式携带
   jsonschema/attrs/referencing/rpds。二进制同步必须`rsync --checksum`，仅mtime+size
   曾漏传重建的1.1GB `.so`。

### 13.3 两机验证与性能

拓扑：P=`11.86.13.78`（TP2×DP4），D=`33.240.36.239`（TP1×DP8），
TCP cache-store；远端全部由本机 `tmux -L qoder-v32` 的SSH pane控制。两端运行同一
wheel/runtime，A/C只改decode offload环境。

**正确性**：
- A_PD、C_PD对同一62,841-token prompt的16-token输出hash均为
  `5c5579eee8f9100f`。
- 最终C的64-token smoke输出hash `53dde6a8b0aab8bd`；wall=64.24s，
  host pull=24.41s。
- 连续请求复用block0=1、generation 1→2均成功；无新core。

**并发矩阵（63k input / 64 output，公网TCP）**：

| conc | A_PD | C_PD（direct host，一批） |
|---:|---|---|
| 1 | 1/1；wall 37.9s；1.7 tok/s | 1/1；wall 57.5s；1.1 tok/s |
| 4 | **2/4**；wall 46.9s；2.7 tok/s | **4/4**；wall 95.6s；2.7 tok/s |
| 8 | **4/8**；wall 70.0s；3.7 tok/s | **0/8**；cache-store send failure |
| 8，stagger 5s | 未测 | **0/8**；仍是cache-store send failure |

结论：
- **容量收益成立**：A每条63k占982个default块，单rank第二条短缺135块；C稳定态
  只占289个default块，conc4即使3条落同一rank仍4/4成功。理论上当前pool每rank
  可放6条default请求、8条indexer请求。
- **当前网络吞吐不合格**：C单请求多20–26s（+52%~69%）；conc4成功率翻倍但
  goodput仍约2.7 tok/s，因为跨公网写约43GB host KV抵消了容量收益；conc8先撞
  TCP cache-store发送上限。该结果不能外推生产RDMA。
- `bench_cap` 的TPOT把一次性P→D transfer摊进64个输出token，不是纯decode
  step latency；纯decode税仍参考同机§11（约+1ms）。

证据：`v32_migration/data/concurrency_20260902/`。后续性能工作应直接在RDMA链路
复跑同矩阵，并优先检查cache-store并行发送/背压，不再改KV拓扑。

## 14. 两机内部 RDMA 复测（2026-09-03）

### 14.1 环境与构建

- P=`11.17.131.204`（TP2×DP4，`mlx5_7`），D=`33.240.36.239`
  （TP1×DP8，`mlx5_4`）；eth1均为RoCE v2 IPv6 GID。
- 双向 `ping6` 约0.05–0.11ms；`rping` 10次RDMA CM读写校验成功。
- 原公开runtime只有NoRdma stub，会直接报`rdma mode not supported`。从受信路径
  `/home/admin/RTP-LLM/internal_source`引入`cache_store_rdma_impl`和Barex；内部deps
  的grpc规则补入仓库已有`MSG_CTRUNC`修复后，RDMA wheel构建成功。
- runtime：`/home/admin/rtp-hol/runtime/rtp-pd-dual-rdma-ring-20260903`；两端
  `libth_transformer.so` SHA256=`54b5796f...ab3b552`。
- 模型从D直连复制到P，180个文件的名称+大小manifest SHA256两端均为
  `da278d49...c79b1`。没有使用SSH隧道或公共下载源。

### 14.2 RDMA适配

TCP版直接把缺失default块写入pinned host；动态host mirror没有注册RDMA MR，首次
C请求报`find memory mr ... gpu:0 adopted:1`。最终按transport分支：

- RDMA：64个已注册GPU scratch blocks分批接收，再D2H排空到host mirror；
- TCP：保持direct-host路径。

A0/A1的default/indexer pools均观察到`register user mr success`，实际请求观察到
`rdma connection ... connect success`。C的618个缺失prefix blocks经RDMA ring完成，
host pull两次为897ms/836ms，ring均释放。

### 14.3 正确性与矩阵

固定prompt实际input=58,005 tokens，64-token smoke：

- A0：wall 29.98s（复测），TPOT 114.82ms；
- A1：wall 29.98s，TPOT 114.79ms；
- C：wall 33.14/33.08s，TPOT 161.64/164.72ms；
- 三者output SHA均为`d27af9d9104025da`。

正式矩阵（input约63k、output=128）：

| conc | A0 单池 | A1 双池无offload | C 双池offload |
|---:|---|---|---|
| 1 | 1/1，TPOT119.3ms，2.8 tok/s | 1/1，114.8ms，3.0 tok/s | 1/1，146.3ms，2.8 tok/s |
| 4 | 4/4，115.3ms，7.0 tok/s | 2/4，116.5ms，5.8 tok/s | 4/4，165.4ms，6.4 tok/s |
| 8 | 0/8 | 4/8，115.2ms，7.2 tok/s | 0/8 |

解释：

- A1失败由default pool容量决定；C在conc4把成功率从2/4恢复到4/4，容量收益成立，
  但TPOT增加约42%，aggregate throughput仍低于A0（6.4 vs 7.0 tok/s）。
- A0和C的conc8都在P端`deepgemm_hybrid_executor`触发CUDA illegal address，同时
  Barex连接出现`disconnect while inuse`；这是高并发prefill/transport blocker，不能
  把0/8解释为offload容量上限。A1因容量先拒掉一半，未触发同等并发。
- 目前可下的结论止于conc4：C提高准入成功率，但尚未把容量收益转化成更高吞吐。

证据：`v32_migration/data/rdma_pd_20260903/`。下一步先定向复现并修复P端conc8
illegal address，再继续更高并发；不应调整KV拓扑来掩盖该问题。

## 15. Admission异步化、mirror pool与正确TPOT（2026-09-04）

§14的TPOT使用`(P端总cost - model prefill)/(output-1)`，把D端准入与P→D handoff
摊入每token，因此短输出下高估了C开销。最终实现和口径如下：

- D在资源分配拿到block0后立即`prepareAsync()`；4.5GB `cudaHostAlloc`与P端prefill
  重叠。load阶段仅`waitPrepared()`，实测blocking wait=0。
- AdmissionStore按allocation bytes缓存过安全期的pinned mirror，默认每rank 12GB
  （可配`RTP_KV_ADMISSION_HOST_POOL_GB`）。pool命中实测prepare=0.01ms。
- PD采用engine-owned mirror时禁用Python的重复61-layer host prewarm。
- layer0每步generation检查改为generation-only C ABI，不再反复构造4.5GB
  `from_blob` tensor wrapper。
- PD latency schema v2贯通API：P queue/compute/handoff，D KV load、mirror prepare与
  wait、normal load、ring load、queue、first token、service、transport全部独立汇报。
- steady TPOT定义为`decode_service_us/(output_len-1)`；TTFT、wall、prefill和load均
  独立报告，不再进入TPOT。

固定58,005-token源码prompt、512输出：

| latency phase（均值） | A1 | C |
|---|---:|---:|
| HTTP wall | 81.87s | 84.53s |
| D request→first token | 22.59s | 24.27s |
| P queue | 73ms | 72ms |
| P compute wall | 22.48s | 22.66s |
| handoff total（与P compute重叠） | 22.51s | 24.18s |
| handoff blocking tail | 24ms | 1.52s |
| D KV load | 22.51s | 24.18s |
| normal tagged load | 22.50s | 21.32s |
| mirror prepare | 0 | 1.60s（异步） |
| prepare blocking wait | 0 | 0 |
| RDMA ring load | 0 | 0.84s |
| D queue | 32ms | 6ms |
| D service（512-token response） | 58.92s | 59.96s |
| steady TPOT | 115.31ms | 117.33ms |
| steady差值 | — | +2.02ms/token |
| pool hit prepare | — | 0.01ms |

成本阶梯（旧总cost口径仍可用于同长度差分）：hook+pre_topk≈0；remap约0.66ms/token；
PCIe fetch在源码prompt下低于噪声；剩余约1–2ms来自offload mask/control与运行波动。
因此原先“C比A1慢31.5ms/token”是统计错误，最终稳定差约2ms/token，接近单机结果。

最终runtime：`/home/admin/rtp-hol/runtime/rtp-pd-dual-rdma-opt-20260903`；
`libth_transformer.so` SHA256=`248c482a...ca259887`，`v32_ctx.so`
SHA256=`6c6d9e2e...b31a78f5`。证据在`v32_migration/data/rdma_opt_20260903/`和
`v32_migration/data/rdma_cost_20260903/`。

### 15.1 单rank长请求容量

为排除P端并发prefill崩溃和decode负载均衡碰撞，测试把所有请求固定路由到decode
rank0；prompt实际为62,830 tokens，输出固定2,048 tokens，每25秒提交一条。这样
prefill基本串行，decode请求逐步叠加。

- A1提交2条：第1条完整成功（steady TPOT 117.146ms）；第2条在0.23s内因default
  pool不足拒绝。稳定容量为1条。作为长度敏感性对照，58,005-token请求可同时放2条。
- C最初提交6条仅3条成功，根因不是容量，而是Python `_purge()`按全局step误释放
  仍在等待/运行的engine-owned mirror。修复为：adopted mirror只在C++ generation
  已消失或变化后清理Python视图，Python不再主动释放活跃mirror。
- 修复后再次提交6条：前5条均完整输出2,048 tokens，output SHA全部为
  `45905cac478f3634`；第6条在0.23s内被明确拒绝。allocator证据为default
  `available=386`、新请求`need=353`、reserve=96、shortfall=63；indexer仍有3,251块
  可用。因此瓶颈确实是default pool，不是indexer或host内存。
- C五条steady TPOT为117.448–119.926ms；包含25秒错峰的端到端goodput为
  27.517 tok/s。A1单条goodput为7.695 tok/s。
- C rank0 RSS峰值27.62GiB，节点最低可用内存827.37GiB，本轮未触发host内存压力。

当前可验证结论：**62.8k长请求每decode rank由A1的1条提升到C的5条，即5倍容量。**
DP8按独立rank理论为40条，但尚未做全节点40并发验收，不能写成实测容量。

### 15.2 DP8全节点并发验收

拓扑仍为P TP2×DP4、D TP1×DP8、Barex RDMA，输入62,830 tokens。

第一轮40条、5秒错峰、2,048输出时，P端约4–5个长prefill重叠，在第13条附近触发
`deepgemm_hybrid_executor` CUDA illegal address；0/40，该轮不计入D容量结果。

为保证40条能在首条完成前全部进入D，第二阶段把`max_seq_len`提高到73,728，输出改为
8,192 tokens，20秒错峰：

- 自然LB峰值分布为`[5,4,4,4,4,4,4,3]`，共32条准入；
- rank0第5条生成到69,889 tokens时default池达到1,925/1,925，增量申请1 block失败；
- 另外8条在准入阶段立即拒绝；最终31/40完整成功，D服务保持存活。

这证明长输出场景每rank稳定边界是4条，而不是初始63k状态下的5条。为消除LB偏斜，
最终通过HTTP `generate_config.role_addrs`显式轮询8个decode endpoint，提交32条、
每rank精确4条。结果：

- **32/32完整成功**，每条输出8,192 tokens；每rank峰值4，总峰值32；
- 每rank default `req_ref`峰值1,509–1,535，均未触发malloc失败；结束后全部mirror释放，
  active归零；
- 验收时间窗内无RDMA send failure、disconnect、mirror missing、CUDA illegal memory；
- 总时长1,580.5s，端到端goodput 165.862 tok/s；
- steady TPOT：mean 113.827ms，p50 114.347ms，p95 115.168ms，max 115.655ms；
- TTFT：mean 40.85s，p95 53.15s，max 54.73s；prepare wait最大0.01ms；
- normal load均值26.75s；ring load均值1.145s、最大2.709s；
- D GPU利用率均值79–89%，显存峰119.6–122.0GB；P GPU利用率均值59.5–64.5%，
  显存峰136.0–138.3GB；
- D每rank RSS峰22.7–23.2GiB，节点可用内存最低691.44GiB。

32条中30条output SHA相同，2条不同。项目既有A-vs-A验证已经确认引擎非逐位确定；本轮
所有请求均完整输出8,192 tokens，但不能据此声称bit-exact一致。

最终结论：**62.8k输入+8,192输出已实测稳定32条（4/rank）；40条目标未通过。**
62.8k输入+2,048输出的单rank容量仍为5条，但受P端长prefill并发崩溃限制，尚未完成
40条端到端稳定验收。证据在`v32_migration/data/rdma_dp8_accept_20260904/`。

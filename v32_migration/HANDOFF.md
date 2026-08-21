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

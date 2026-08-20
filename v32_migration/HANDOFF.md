# V3.2 长上下文 KV Offload（staging 环准入）— 迁移交接文档

日期：2026-08-20 ｜ 源：内部工作区 `rtp-llm-rdma @ b7167df18`（+工作区补丁）｜
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

## 6. 编译修复记录（2026-08-20 晚，公开树首次编译通过）
全量 wheel 编译通过（--config=cuda12_9 --jobs=192）。三处迁移失误已修（与驱动机一致）：
1. NormalEngine.h 还原上游（step-metrics 声明+成员搬动不需要；上游自带 cudaMalloc 接线）
2. DecodeRpcServer.h loadCache 签名对齐上游单参
3. WorkerStatusService.h/Test 还原上游（ExecutorStepMetrics 仪表未合并，已无引用）
依赖拉取偶发 github 抖动，重试即可。

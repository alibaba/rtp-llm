# Sparse KV Offload 验收

设计、接口、调用图、配置与测试入口统一见 [Sparse KV Offload](sparse_kv_offload.md)。本文只保存版本和验证证据。

## 当前状态（2026-10-27）

当前为 main `bcd02fdb05` 上的本地修改，尚未提交。下文完整模型结果属于各次实验快照，不能当作当前暂存版本已经重新验收。

| 记录 | 代码身份与结果 | 证据目录 |
|---|---|---|
| 10 月 25 日修复与拆分 | 当时的测试提交见下一节；SDK、回归、模型 smoke 和性能结果保留 | `/home/renyuanming.rym/kvoffload-review-20261025/runs/` |
| 10 月 27 日凌晨精简与压测 | 移除独立 `resident_rows` 配置之前的快照；完整 78 层 TP1/DP4/EP4 对照，身份以各 run manifest 为准 | `/home/renyuanming.rym/rtp-llm-sparse-kv-offload-build/perf-20261026/` |
| 10 月 27 日下午暂存审查 | index tree `bb33ec9eb11198302a19d6081ca388684c93acb9`；Python 23/23、C++ 语法 24/24、GPU runtime 14/14；完整 Bazel 构建未完成 | `/home/renyuanming.rym/sparse-kv-stage-review-20261027-144029/` |

下午审查后，测试 BUILD 修复已另行暂存；本次文档合并没有重新构建或跑模型。FlashMLA 升级及其专属精度回归仍是未暂存的独立修改，本轮暂不处理精度问题。当前暂存使用 main 原有依赖，旧的新版依赖测试结果不能替代它的 off/on 验证。

GLM-5.3 的 TP2/4/8 在已测 FlashMLA 下存在 head 数限制；TP2 容量实验只验证请求分配，未完成 decode。证据单独保存在 `/home/renyuanming.rym/kvoffload-review-20261026-tp2/README.md`。

## 10 月 25 日历史记录

基于 main `bcd02fdb05`，2026-10-25 修复、拆分后重新测量。该次证据目录：
`/home/renyuanming.rym/kvoffload-review-20261025/runs/`，下文日志和 run ID 均相对此目录。
以下直到“2026-10-27 本地精简验证”的分支、提交和性能描述均为 10 月 25 日历史状态。

## 修复与范围

- R2：descriptor 的 `offload_runtime_group` 声明共享关系；mapping、owners、热槽状态和 selection scratch 按 runtime 计费，KV payload、热缓存 KV 和 pack staging 按 storage 计费。
- R3：补齐 MLA 路由 fixture 的三个字段，生产代码不增加兼容分支。
- R6：MoE 前 drain 默认关闭，group 末层 drain 保留。
- R7：覆盖 512/8192 热槽的多 CTA 淘汰、跨 CTA 重复 ID；用不同 default/indexer 表、slot mapping 和 FMHA 验证真实 selection 翻译与 pack 路由。
- FP8 requant、DSV4.1 适配、benchmark/snapshot hook 各自拆出；融合 kernel 未加入。R1 的 P/D load 取消后 block 生命周期问题未修复，按指令在本次报告后单独处理。

## 版本与测试

实际测试提交：FP8 `6afdc12b69`、feature `ec2675fcdc`、DSV `6277cb1b1c`、tools `75eb505dba`。
最终分支仅补入本验收文档并重建父子提交，源码、测试和依赖文件与实际测试提交相同；
映射见 `/home/renyuanming.rym/kvoffload-review-20261025/split/state.json`，测试提交保存在同目录树的 `archive/tested-branches.bundle`。
各分支相对表中基线仅一个提交，未 push。

| 分支 | 基于 | 独立验证 | 证据 |
|---|---|---|---|
| `fix/fp8-ue8m0-requant` | main | 精确 requant 权重测试通过，覆盖矩阵与专家张量，无 skip | `branch-fp8-tests.log`、`branch-fp8-testlog/` |
| `feat/sparse-kv-offload` | main | SDK 构建；14/14 测试目标；GLM-5.3 IndexShare smoke；原生 FP32 scale 回归通过 | `branch-main-build.log`、`branch-main-tests.log`、`branch-main-testlogs/`、`branch-main-glm53-smoke.log`、`branch-main-native-fp32-scales.log` |
| `feat/sparse-kv-offload-dsv41` | feature | 3/3 GPU 回归目标、5 个 cache-tag 测试通过 | `branch-dsv41-tests.log`、`branch-dsv41-testlogs/`、`branch-dsv41-cache-tags.log` |
| `tools/sparse-kv-offload-bench` | feature | 含快照 hook 的压测 SDK 构建通过；只用于压测，不合入 | `tools-build.log` |

14 个目标覆盖容量、pickle、allocator/donor/copy、TP2 版本同步、TCP P/D 混合目标、路由、runtime 和模型 smoke。
GLM FP8/BF16、DSv3.2 P/D smoke 通过；GLM-5.3 smoke 使用 7 层 IndexShare，覆盖 eager/graph × off/on 及关闭 MoE 前 drain。
模型 smoke 使用显式稳定 top-K 和选定 logits 对照；本轮性能请求使用原生 top-K，未保存完整输出精度对照，**不能据此宣称完整模型原生跨分支精度对齐**。
DSV4.1：**HEAD 上未重新做真实模型验证**；本地完整 checkpoint 已清理，单测不能替代该验收。DSV 专用依赖和边界见其分支文档。

当时为验证 GLM 原生非 2 次幂 FP32 KV scale，feature 使用 FlashMLA `49e2000.cu132`，requirements、lock、arch_select 随该实验版本升级；这不是当前暂存区的依赖状态。
同一回归在旧 `9241ae3` 下 relative L2 为 `0.305717`，超过 `0.01` 阈值；换回 `49e2000` 通过。
证据：`native-fp32-old-flashmla-negative-control.log` 与 `branch-main-native-fp32-scales.log`。

## 2026-10-25 性能

完整本地 GLM-5.3，78 层，GPU 1–4，TP1/DP4/EP4，输入 32768、输出 256；总并发 32/64/128 对应每 rank 8/16/32。
CUDA Graph 开启，统一 `PYTORCH_ALLOC_CONF=expandable_segments:True`，每组预热后测三轮，正式计时无 profiler。
主版本为 **feature + tools，未叠加独立 FP8 requant 修复**。
指标为客户端稳定 decode TPOT：计输出计数 2 至 256 的 254 个间隔；排除 prefill、快照恢复和首个 decode 输出，保留调度及 RPC 交付。
先保存 prefill KV，再独立启动 decode；本实验不覆盖 P/D 同时运行时的资源竞争。

| 方案 / TPOT ms | 并发 32 | 并发 64 | 并发 128 |
|---|---:|---:|---:|
| main 不 offload | 36.80 | 41.36 | 47.64 |
| main offload，默认 drain 关闭 | 37.99 (+3.2%) | 43.06 (+4.1%) | 56.69 (+19.0%) |
| GLM 不 offload | 34.76 | 40.62 | 46.87 |
| GLM offload，原配置 | 36.01 (+3.6%) | 44.14 (+8.7%) | 74.53 (+59.0%) |
| GLM offload，shared indexer | 35.28 (+1.5%) | 42.95 (+5.7%) | 57.76 (+23.2%) |
| main offload，drain 开启 | 未测 | 未测 | 58.52 (+22.8%) |

括号为相对同分支不 offload 的开销。三轮逐轮数据、请求事件及复算结果见 `matrix_audit.json` 和各 run 的 `results.json`。
run ID 均以 `20261025-` 开头：`main-hbm`、`main-offload`、`glm-hbm`、`glm-offload`、`glm-offload-shared`、`main-offload-drain`。

### 有效配置与可比范围

| 方案 | indexer 物理 slot | 永久 HBM blocks H | host blocks P | 热缓存行 R | 实际 MoE 前 drain |
|---|---:|---:|---:|---:|---|
| main 不 offload | 21 | 8708 | 0 | 0 | 无 |
| main offload | 21 | 1418 | 10492 | 65536 | 关 |
| GLM 不 offload | 78 | 8734 | 0 | 0 | 无 |
| GLM offload，原配置 | 78 | 2 | 9803 | 65536 | 不执行 |
| GLM offload，shared indexer | 21 | 1422 | 10492 | 65536 | 不执行 |
| main offload，drain 开启 | 21 | 1418 | 10492 | 65536 | 开 |

容量是每 rank 的值，四 rank 一致；block 为 128 行，H 含保留 block 0，R 不含永久 HBM 行。

offload 均为每 rank 16 GiB KV HBM + 最多 64 GiB host，R=65536；不 offload 沿用上次 main 56 GiB、GLM 64 GiB 配置，非严格同预算容量对照。
R2 修复后 main H=1418；用本轮同一预算恢复旧的按 78 个 storage 计元数据，求解 H=1302。热缓存大小、pack 和 KV 的计费方式不变。
GLM 原配置不共享 indexer cache；shared 组设置 `ENABLE_GLM52_SHARED_INDEXER_KV_CACHE=1`，四 rank 日志均确认物理 slot 从 78 降为 21，H 从 2 增为 1422，接近 main 的 1418。
shared 行仍相对原配置的 GLM 不 offload 计算开销，包含共享 indexer 的收益；本轮没有额外测量 shared 配置的不 offload 基线，也没有强制两边 H 完全相同。
GLM 的 MoE 前 drain 只对 MegaMoE 生效；当前 DeepEP 策略下，即使环境变量为 1 也不执行。main 的消融确实切换该等待，group 末层等待保持不变。

两边同用 DeepEP `a56d6156` 的 SM10.3 扩展和 FlashMLA `49e2000`；Torch `2.11.0+cu130`、Triton `3.6.0`。
DeepGEMM 保留各分支锁定版本：main `122e18b`，GLM `6db6ed3`。因此跨分支绝对差值包含模型实现及依赖差异，不能只归因于 offload。
GLM 参考源码为 `15f39f43f1` 加原有压测兼容改动，diff SHA256 为 `5dd21fc74d1cb2fb95dede55cef3727f4007681b225869f9a7e4a35c469a460d`。
各 run manifest 保留源码/diff、实际库哈希、模型 141 个 shard 的 SHA256、有效参数及每 rank graph/capacity 证据；运行前后源码身份一致。
正式计时后用独立 DEBUG 请求验证各 rank 的 capture/replay；这不是正式测量逐步命中率。单进程顺序三轮也不等同多次独立重启的统计结论。

## 128 并发归因

正式计时 main 增加 **9.04 ms / 19.0%**。只开启 MoE 前 drain 后为 58.52 ms，默认关闭降低 **1.84 ms / 3.1%**；组末 drain 始终保留。
GLM 开启 shared indexer 后从 74.53 降为 57.76 ms，与 main 56.69 ms 接近；原配置差距主要受共享缓存和常驻容量影响，不能据此宣称 main 的搬运算法领先。

独立 trace：`20261025-trace-main-hbm`、`20261025-trace-main-offload`。每 rank 采集 32 步，去除最初 2 步，按 30 步计算后取四 rank 均值。

trace 中 GPU 窗口为 **48.40 -> 57.01 ms，增加 8.61 ms**。其时间并集可分为：

| 不重叠的时间分量，ms/step | 不 offload | offload | 增量 |
|---|---:|---:|---:|
| 没有其他 kernel 重叠的 fetch | 0 | 4.31 | +4.31 |
| 其他 kernel 的时间并集 | 45.44 | 49.82 | +4.38 |
| 无 kernel 间隙 | 2.96 | 2.88 | -0.08 |

fetch 累计 **10.50 ms**，其中约 **6.19 ms** 与其他 kernel 重叠。仅报告搬运累计时间会高估它对 step 的直接影响；只看独占 fetch 又会漏掉管理工作及重叠时的计算/通信变化。

| kernel 类别，累计 ms/step | 不 offload | offload | 说明 |
|---|---:|---:|---|
| admission | 0 | 0.71 | 每步 63 次 protect/admit/finish |
| write | 0 | 0.49 | 每步 234 次 ID 处理/写入 |
| MLA attention | 2.78 | 4.39 | +1.61 ms，调用次数均为 156 |
| MoE 通信 | 5.26 | 6.42 | +1.16 ms，调用次数均为 150 |
| 其他矩阵乘 | 8.74 | 9.80 | +1.06 ms，调用次数均为 685 |
| 专家 GEMM | 21.00 | 20.95 | 基本持平，调用次数均为 150 |

新增工作和计算/通信耗时变化同时存在；本实验未分离 SM/带宽竞争、KV 地址布局和 rank 等待各自的贡献。
上述 8.61 ms 来自独立的 30 步 trace 窗口，不能强行拆成正式 TPOT 的 9.04 ms。可复算结果见 `trace_comparison.json`，逐 rank 明细见两个 run 的 `trace_summary.json`。

各类 kernel 累计时间包含跨 stream 重叠，不能相加解释 TPOT；“没有其他 kernel 重叠”是观测到的独占时间，不等于已证明的关键路径。
正式 TPOT 与 profiler 数据属于独立运行，后者只用于归因。原始 trace、分类并集和逐 rank 数据随日志保留。

## 证据与清理

GLM 快照首次生成遇到 RPC keepalive watchdog 超时，失败日志保留在 `20261025-save-glm-failed-keepalive/`。
重试只对 save 阶段设置 keepalive time=1800000 ms、timeout=60000 ms，记录在 invocation/manifest；decode 配置不变。
首次编排漏配历史口径的 expandable allocator，GLM 64 GiB KV 分配 OOM；已统一后重跑全部 decode。
`*-default-allocator` 及 `*-failed-default-allocator` 目录仅保留这次偏离配置的记录，不计入当前表。
GPU 0 的 CUDA 实际执行检查仍报 unavailable，本轮只用通过检查的 GPU 1–4，未执行 reset。

测量服务已退出，GPU 上无本次残留进程。已删除约 **604 GiB** 的本轮产物：main/GLM KV 快照约 453 GiB、临时 smoke 模型约 97 GiB、两套 Bazel 输出及临时构建依赖约 54 GiB。
完整本地 GLM-5.3 checkpoint 保留。只保留结果、manifest、原始 trace、日志及重建记录；大日志无损压缩，并与压缩前 SHA256 核对。
清理前后记录见 `cleanup-before.txt`、`build-cleanup-after.txt`、`disk-before-cleanup.txt`、`disk-after-cleanup.txt`。
两次成功 prefill 的 snapshot manifest 单独保留，KV payload 已删除。测试日志已复制为真实文件，不依赖已删除的 Bazel 输出目录。
drain 首次启动因上一组两个 worker 尚在退出而被空闲检查拦截，未发出测量请求；进程自行退出后原配置重试通过，失败记录单独保留。

## 2026-10-27 本地精简验证

本轮合并 layout 的 page/row/payload 参数，删除始终为真的 `plan.write`、重复的 owner 信息和布局校验。
`layout.py` 从 231 行减到 199 行，保留 MLA/MHA/CSA 的 plane、head stride 和页 padding 支持；未新增测试文件，未提交代码。

main 与 GLM SDK 均已重新构建，14/14 个回归目标通过，含 runtime 20 个用例和 P/D 回归。
GLM-5.3 的 7 层模型 smoke 通过：eager/graph × off/on，加 graph/offload/MoE 前 drain 关闭，五组生成 token 和选定 logits 完全一致；日志确认 capture 和 replay。
该精度检查使用稳定 top-K，不代表完整 78 层原生跨分支精度对齐。

精简前后在同一张 L20D 上交替测六轮，使用 CUDA Graph，每轮 50 次；下表为中位数。
范围仅为**单层 FP8 MLA fetch**（656 bytes/行，全部 miss，含相同的 ID 更新），不是模型 TPOT，也没有 32K prompt。

| 每次搬运行数 | 精简前 ms | 精简后 ms |
|---|---:|---:|
| 16,384 | 0.23779 | 0.23749 |
| 32,768 | 0.46934 | 0.46881 |
| 65,536 | 0.93336 | 0.93157 |

差异均小于 0.2%，本微基准未观察到回退。

### 完整模型性能

GLM-5.3 **78 层**，GPU 1–4，**TP1/DP4/EP4**。每请求输入 **32768 token**、输出 **256 token**；并发为四卡合计，shared indexer 和 CUDA Graph 均开启。
每个并发先预热，再测三轮。先用四卡完成真实 prefill 并保存 KV，再单独启动 decode 重放；TPOT 不包含 prefill、KV 恢复和第一个 decode 输出，也不代表 P/D 同时运行时的性能。

| 方案 | 每卡 KV HBM 预算 | 并发 32 TPOT ms | 并发 64 TPOT ms | 并发 128 TPOT ms |
|---|---:|---:|---:|---:|
| main | 56 GiB | 36.68 | 41.35 | 47.20 |
| main + SparseOffloadBackend | 16 GiB | 38.10 | 43.15 | 56.61 |
| GLM | 56 GiB | 34.63 | 40.72 | 46.85 |
| GLM + 原生 offload | 16 GiB | 35.28 | 43.05 | 58.03 |

main 指本地实现关闭 offload，基线为 main `bcd02fdb05`；GLM 基线为 `15f39f43f1`。两侧 shared indexer 均为 21 个物理 slot。
每个方案的正式请求为 `(32 + 64 + 128) × 3 = 672`，全部完成 256 token；每个请求保留 254 个连续 decode 间隔。
CUDA Graph 通过正式测量后的独立 DEBUG 请求核验：四个 rank 均 capture/replay 每卡 batch 8/16/32。正式计时没有 profiler 或稳定 top-K 测试钩子。

- 相对各自不开 offload，main 的 TPOT 增幅为 **3.9% / 4.4% / 19.9%**，GLM 为 **1.9% / 5.7% / 23.9%**，换取每卡 KV HBM 预算从 56 GiB 降到 16 GiB。
- 不开 offload 时每卡 8708 个 KV block。并发 128 时每卡 32 个请求，输入占 8192 个 block，生成 256 token 后仍放得下。
- offload 时 main/GLM 每卡 HBM block 为 1418/1422，Host block 均为 10492，resident rows 均为 65536。并发 32 的输入已需每卡 2048 个 block，因此实际使用了 Host KV。
- main 不开 offload 时也比 GLM 慢 **5.9% / 1.5% / 0.8%**；跨分支差异不能全部归因于 offload。两侧 DeepGEMM 版本不同，本轮未移植融合优化。main offload 在并发 32 较慢、64 接近、128 较快，不能概括为全面快于 GLM。

### 完整模型精度

配置同上，每组 224 个请求，逐请求记录 256 个生成 token 和每步 15 个选定 logits，均为有限值。同一分支的 offload 开关对照复用相同 prefill KV；这些请求不纳入性能表。

**稳定 top-K 下，main offload 开关完全一致**：并发 32/64/128 共 224/224 条完整序列相同，每一步选定 logits 的差值均为 0。只在 decode 使用测试钩子，稳定候选选择及输出排序；模型、缓存容量和 CUDA Graph 配置保持不变。

原生模式不能直接用逐 token 完全一致验收：不开 offload 的 main 自身重复运行，也只有 1/224 条完整序列一致。以下比较首个 decode 步，此时每对请求的输入前缀相同；数值为 **15 个选定 logits 的 relative L2**，不是全词表误差。

| 对照 | 并发 32 | 并发 64 | 并发 128 |
|---|---:|---:|---:|
| main 原生，不 offload，重复运行 | 0.133 | 0.139 | 0.137 |
| main 原生，offload 开关 | 0.128 | 0.136 | 0.134 |
| GLM 原生，offload 开关 | 0.155 | 0.138 | 0.139 |
| main 稳定 top-K，offload 开关 | 0 | 0 | 0 |
| main / GLM 原生，均不 offload | 0.277 | 0.270 | 0.261 |

原生 top-K 用 atomic 分配输出位置，选择顺序可能变化。重复基线和稳定选择对照说明原生非确定性需要单独处理；本轮没有把全部差异归因于它，也没有宣称原生跨分支精度对齐。当前证据支持稳定选择下 offload 搬运和读写的一致性，不能替代全词表或任务准确率评测。
逐请求差异见 `precision_audit.json`、`precision_controls_audit.json`；原始 token、logits 和首处分歧均保留。

记录目录：`/home/renyuanming.rym/rtp-llm-sparse-kv-offload-build/perf-20261026/`。
`layout_benchmark.json` 保留精简前后源码哈希及逐轮耗时；`test.log`、`testlogs/`、`build.log` 保留验证记录。
`matrix_audit.json` 保留逐轮 TPOT 与容量；各运行的 `run_manifest.json` 保留源码哈希、模型分片哈希、构建依赖及 graph 证据。运行目录前缀沿用 `20261026`，完整模型测量实际执行于 2026-10-27。
已清理约 **475.5 GiB** 的本轮 KV 快照、两套构建产物和临时构建文件；完整本地模型、测试日志及原始请求结果保留。快照 manifest 单独留存，性能日志无损压缩后核对了 SHA256。清理记录见 `cleanup-before.txt`、`cleanup-after.txt` 和 `disk-*-cleanup.txt`。GPU 1–4 无本次残留进程，遥测已停止。
测试服务在 SIGTERM 退出阶段出现段错误，10 月 25 日的模型 smoke 日志也有同样现象；本轮未修复，推理对照通过不代表退出流程正常。

# DSV3.2 Decode KV Offload（Scheme B）：设计、踩坑与系统瓶颈拆解

更新：2026-08-19（r29 终版，单机冻结）｜ **~171ms/步（167/172/174）vs 基线 144ms（1.19×）**，PREFIX 3/3 逐字一致，errors=0
环境：11.86.13.78（8×H20-3e），DeepSeek-V3.2-Exp，TP2×DP4×EP8，55k token 长请求，贪心

---

## 1. 背景与动机（为什么做）

**KV 墙**：decode 池 12GB/rank 时，63k 长请求需 867 块（4.3GB，bf16 78KB/token）。集群判决数据：
- A（原版，秒拒）：长请求拒绝 **97.7%**
- A3（原版+30min 重试，"不拒绝"）：长请求失败 **75.8%**，完成者排队 **17.7 分钟**（TTFT p50）
- 两版短请求都正常（~1% 失败 / 110ms 步长）→ 问题是结构性容量，不是调度

**机会**：V3.2 的 DSA 每层每步只注意 top-2048 个 token。⇒ 每步真正要读的 KV 只有 2.3MB/层；
选择集相邻步重叠 ~94%；indexer-K 只有主 KV 的 1/10 体积。⇒ ECHO 式 host offload 成立。

---

## 2. 当前设计（r27 形态）

```
┌─ C++ 引擎补丁（8 文件，rtp-llm-rdma@b7167df18 工作区）
│   decode 流生成≥32 token 后释放前缀块；保留 block0(身份) + 32 staging 块 + 尾部 256 块(16k 窗)
│   准入 malloc 失败带重试窗（RTP_DECODE_MALLOC_RETRY_MS）
│
├─ 镜像（python 薄壳 + 移植的 staged-copy 引擎）
│   主 KV：分片(4096行/步) staged D2H → host pinned store（gather kernel+单次 D2H，专用流）
│   indexer-K：GPU→GPU 原字节搬进自管分页侧存储（[块,64,132] 与池同构）+ 每层 132B 补当前 token
│
├─ 每层服务（offload 行，零同步）
│   A 同款原生 fused 打分（deep_gemm fp8_paged_mqa_logits，喂侧池+稠密合成块表，范围[0,kvlen)）
│   → fused top-2048 → C++ ctx_serve_wb：build_indices kernel
│   （warm→逻辑位置 / staging 命中→块表项别名 / miss→异步取）→ 原地写回 kernel_topk
│   → 原生 convert-to-global 承运给 flash_mla（SparseMlaOp.forward 无 hook）
│
├─ 异步取数（C++ 常驻线程，无 GIL）
│   miss 索引经 GPU buffer + 事件通知线程 → pinned gather → cudaMemcpyBatchAsync 落 staging
│   LRU 驱逐；有界陈旧：本步 miss 下步可见（实测不影响贪心输出）
│
└─ 零同步 bookkeeping：决策用上一步元数据（kvlen 用 age 校正精确），本步元数据异步 D2H
```
python 每层残留：roi 判定（每步一次）+ append_tok + logits + fused_topk + serve_wb（发射合计 0.7ms/步）。

---

## 3. 踩过的坑（按杀伤力排序）

1. 【史诗级】**寄生的第一代影子代码**：早期 shadow 对拍 hook 的启用条件写成 `MODE != "off"`，
   capacity 模式下它每层偷跑"全量逐块镜像 + `stream.synchronize()`"→ **每步 61 次全流水排空**。
   后果：数周内步长钉死 ~340-540ms，对一切优化免疫；profiler 显示为"EP 通信等待"（假象）；
   期间做的多轮"优化实验"结论全部被污染。修复=一行条件 → 337ms 直落 189ms。
   教训：①对拍/调试钩子必须显式白名单启用；②"加了优化没变化"本身就是强信号，应立刻怀疑
   有隐藏串行点而不是继续堆优化；③profiler 的"通信等待"可能是任何上游延迟的投影。
2. **CPU-GPU 同步破坏发射-执行重叠**：V3.2 eager 基线是发射受限（CPU 提前发射 61 层，GPU 流水追赶），
   每步哪怕 1-2 次 `tolist()/.cpu()` 也会把重叠打成串行。⇒ 全链路零同步（决策用上步数据）。
3. **indexer-K 与主 KV 同物理块**（藏在 kv_scale 区）→ 释放即两者皆失 → 自管侧存储。
4. **缓存块内是分段布局**（64×128B fp8 + 64×4B 裸 float32 scale），不是逐 token 交错、
   scale 不是 ue8m0 打包——读错时 top-k 比随机还差（反相关）。
5. **top-k 索引空间**：indexer 出 request-local 逻辑位；消费端才转全局。对拍两侧同映射会互证互掩。
6. **首 token 由 prefill 产生** → offload 必须等第一个 decode 步之后（镜像窗口）。
7. **torch pinned ≠ device-mapped**：kernel 写 host 指针静默毒化整条 stream。
8. **同步 cudaMemcpy 走 legacy 默认流会同步全设备** → 一切拷贝专用 non-blocking 流。
9. **陈旧元数据 × 批次形变**：零同步后，请求边界步（批宽变化/行漂移）会用旧 kvlen 索引窄块表
   → device assert 且异步上报指向无辜代码。⇒ 宽度守卫 + 绊线 + kernel 内边界检查三层防护。
10. 工程杂项：块释放引用计数（哨兵别名/去重）、双重释放、并发 prefill 激活 OOM（34GB，无 chunked
    prefill）、基础设施抖动（ssh/nvidia-smi 瞬时超时曾两次误杀多小时实验→门禁 3 击容错）。
11. **"等价"改写破坏隐式新鲜度契约**（r23/r24 事故）：把 khead 从 pinned 活视图改成 `.tolist()`
    快照（本意省每层张量索引），offload 检测即刻回归 PREFIX=False——活视图能读到本步异步落地的
    块表收缩，冻结快照读到的是上一步；一步误判 offloaded=False → kernel 扫 NULL 别名块。
    r25 回退活视图后恢复（175-182ms, PREFIX=True）。教训：零同步设计里"张量→list"不是等价重构。

---

## 4. 核心瓶颈拆解：为什么会有"两波打分"（现状 +28ms 的主体）

### 4.1 问题的根（三个环环相扣的事实）

**事实 1：DSA 每层每步都要对全历史打分。**
V3.2 每层 decode 用 indexer 对该请求全部历史 token 算 `score=Σ_h w_h·relu(q_h·k)`，
取 top-2048 给稀疏注意力。选择集每步每层都变，打分不可省略。

**事实 2：打分 kernel 的寻址模型 = 一个池基址 + 一张块表。**
打分是一次 kernel 调用处理整个 batch：
```
score_kernel(池基址, 块表[所有行], q, ...)
             ↑一个指针  ↑每行一排块号
数据地址 = 池基址 + 块表[i][p÷64] × 块大小
```
两个硬性限制：a) 一次调用只能绑一个池基址；b) 块表存的是池内块号，
表达不了"数据在池外某个张量里"。

**事实 3：offload 行的 indexer-K 必然在池外。**
- 留在主池 = 不释放块 = 不省显存（offload 目的落空）；
- 主池按 86KB 整块编址（KV+indexer 同块绑定，见踩坑 3），副本按块塞回主池
  每块只用 1/11，显存等于没省；
- ⇒ 副本只能是池外的紧凑独立张量（indexer-K 全量，主 KV 的 1/10）。

### 4.2 必然推论：两波调用

batch 里既有普通行（数据在主池）又有 offload 行（数据在池外副本），
两个内存体 × 每次调用一扇门 ⇒ 必须两次：

| | 池基址 | 块表 | 有效覆盖 | 无效部分 |
|---|---|---|---|---|
| 第一波（引擎固有） | 主池 | 原块表 | 普通行全历史 ✓ | offload 行陪跑（块表项=哨兵块，输出为垃圾，丢弃） |
| 第二波（我们补的） | 侧池副本 | 稠密合成块表 | offload 行全历史 ✓ | — |

**这不是同一个分数算两遍**：每一行的有效计算只有一次，总 FLOPs ≈ A。
第一波无法只算部分行（batch 级调用，引擎无按行跳过的入口）。

### 4.3 r27 时点的代价账目（其中"第二波 21ms"的归因后被 4.4 实测修正）

| 项 | 每步 | 说明 |
|---|---|---|
| off 基线 | 144ms | ~95% 是 eager 发射开销（P1 结论） |
| **B 现状（r27）** | **~172ms** | PREFIX 逐字一致 |
| 第一波给 offload 行的陪跑 | ~1-2ms | 扫哨兵块，L2 命中，便宜 |
| **第二波 GPU 串行时间** | **~21ms** | logits ~10.5 + fused topk ~6 + build/写回 ~4，串联插在每层 indexer→attention 之间 |
| python 发射（hb: proc） | 0.7ms | T2 下沉后已消（r22 时 ~15ms） |
| mirror/bookkeeping | ~5ms | 摊销 |

注：第二波已用 A 同款原生 fused kernel（r27），实现效率到顶；
剩余成本是"第二波的物理存在"本身。

### 4.4 单波已落地（r28/r29），且实测推翻了 4.2/4.3 的成本推断

方案一（副本池）已实现：全局 indexer 池（默认 4096 块/层 ≈2.1GB）覆盖全部 decode 行
（准入批量 D2D + 每步 132B×B 追加 + 水位线回填 + block0 身份绊线），批稳时跳过原生第一波，
一次 fused 调用出全 batch top-2048。r29 判决：**PREFIX 3/3，单波 232+/256 步全程生效**。

但性能几乎没变（r27 双波 172 → r29 单波 171）。微二分实验给出终审账目：

| 配置 | step/ms | 结论 |
|---|---|---|
| 基线（offload 全关） | 144 | — |
| L0：C++ 收缩补丁开、hook 空转 | 143-147 | 引擎补丁 **+0** |
| D2：单波替换+池维护，无收缩/无 serve | 145-146（PREFIX=True） | 打分替换 **+1**，且正确 |
| r29 全量 B | 167-174 | **serve/mirror/fetch 链 +26** |

**"两波打分"从来不是关键路径成本**——eager 基线是发射受限的（95% 发射开销），
第二波的 GPU 时间本来就落在流水气泡里。4.2 节的 21ms 是算术推断，被实测证伪。

### 4.5 实在优化不了的点（+26ms 的构成与原因）

剩余 26ms 全部是**真正伺服 offload KV 的机器**，发生在 offload 行的每层服务链上：

1. **serve 链 CPU 发射 ~10ms/步**（prof: 0.166ms/层）：drain（staging LRU 重排 ~10 个张量算子）+
   build/finalize kernel + 写回拷贝 + 每 4 层一次 fetch 事件/通知线程。
2. **GPU 串行段 ~16ms/步**：这条链插在每层 indexer→attention 的数据依赖上，
   attention 必须等它，launch-ahead 无法把它藏进气泡（和第二波不同——第二波只是"多算"，
   这条链是"必须等"）。
3. **为什么压不下去**：
   - 它做的是不可省略的语义工作：把 host 上的 KV 变成 attention 可读的槽位（查 staging、
     发 miss、翻译索引）——A 没有这项工作，因为 A 的 KV 全在池里；
   - 零同步约束下已无 python 税可收（CPU 发射 10ms 里大半是 aten 算子固有开销）；
   - 进一步只剩两条路，都超出 runtime 补丁半径：**CUDA graph 化整层链**（形状随 kvlen 变，
     需分桶 pad + 引擎配合捕获）或 **引擎第三池 + fetch 下沉进 attention 准备阶段**（C++ 改造，
     上游 PR 方向）。

**结论：单机冻结在 1.19×（171 vs 144）。这 19% 是"长请求能跑起来"的价格——
对照集群判决：A 方案长请求 97.7% 直接被拒（A3 变体 75.8% 失败 + 17.7 分钟排队）。**

---

## 5. 资产与复现

- 代码：本目录 v32_ctx.cu / v32_capacity.py / v32_offload_hook.py / glm5port/（feat/glm5_cu13_rebase
  移植的批量/分级拷贝引擎，最新 commit 0f4c03b95 已核实）；引擎补丁 rtp-llm-rdma@b7167df18 工作区
- runtime：rtp-b-offload-20260817（A 基线 runtime 从未触碰）；部署=拷 3 个文件
- 验证工具链：影子对拍（V32_VERIFY_TOPK/ATTN）、PREFIX 门禁、分段计时（prof/hb）、
  V32_TORCH_PROF、V32_HOOK_LEVEL 微二分、CUDA_LAUNCH_BLOCKING 定位法
- 判决数据：A 97.7% 拒 / A3 75.8% 失败+17.7min 排队 / B 容量语义已证（-70% 驻留、输出逐字一致）

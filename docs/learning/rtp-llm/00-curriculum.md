# RTP-LLM 代码库系统学习课程

- **owner**: caihaowen.chw
- **started**: 2026-05-22
- **status**: draft → in-progress (B1.1)
- **estimated duration**: 15 节，每节 1-2h，弹性 ±2 节

## 1. 目标

完成本课程后，学习者应成为**合格的推理框架工程师**，能够：

- 在不依赖记忆的情况下，向同事/面试官清晰讲解 RTP-LLM 任一主路径（输入 → 调度 → 模型 forward → kernel → 输出）
- 接到"加一个新模型架构 / 加一个调度策略 / 加一个 metric / 加一个 kernel binding / 改一个量化变体"这类任务时，能定位到正确的扩展点、独立完成开发并通过相应的单元/smoke 测试
- 接到线上 issue（崩溃/性能回退/精度问题）时，能选择正确的工具（gdb / nsys / 日志 / smoke 复现 / git bisect）并独立定位到代码行
- 能阅读 BUILD/`.bzl` 文件并理解多架构 (sm7x/8x/9x) 编译的分流逻辑
- 能在 PR review 时识别"这一改动会破坏 X 不变式"或"这一改动该有 Y 测试"

明确**不在**课程范围内（YAGNI）：

- 编写新 CUDA kernel 源码（学习 kernel 选择 / wire-up，不学习内核数学）
- RL / 训练相关代码（仓库里有 `rtp_llm/RL/` 但不在主推理路径上）
- 多模态前向细节（看 multimodal_processor 的位置和接口即可，不深入 VIT）
- ROCm 分支（默认 CUDA）

## 2. 学习者前置（2026-05-22 时点）

**已熟**：Python frontend（pipeline / batch tokenize / dispatcher）、Java flexlb（master/batch_schedule）、bazel 命令、Docker 容器、smoke 测试运行、`batch_group_id` + `force_batch` 用法

**模糊**（本课程要补的"被问会懵"区）：
- 调度/批处理内部（FIFO、continuous batching、PD 分离的代码细节、cache reuse）
- C++ 引擎主路径（RpcServer / Engine / Executor / Stream 谁拥有谁、主循环骨架）
- Kernel / device 层（attention 多实现、SM 分流、cuda graph、3rdparty kernel wire-up）
- Loader / 量化 / 构建（loader 主流程、十几种量化 weight class、bazel select、torch ABI）

## 3. 整体形状

```
┌─────────────────────────────────────────────────────────┐
│  B1  调度 / 批处理              4 节  (B1.1 - B1.4)       │
│  B2  Forward 主路径 (C++→Py)    3 节  (B2.1 - B2.3)       │
│  B3  Kernel / device 选择       4 节  (B3.1 - B3.4)       │
│  B4  Loader / 量化 / 构建       3 节  (B4.1 - B4.3)       │
│  Capstone  真实小改动 PR        1 节  (15)                 │
└─────────────────────────────────────────────────────────┘
```

**拆法逻辑**：每块内部都遵循"对象 → 主循环骨架 → 决策核心 → 非主路径"的递进顺序（详见 §6 单节模板）。块与块之间遵循"由外向内"：先调度层（你已经在做的领域）→ engine 主路径（C++/Py 交界）→ 再深入 kernel/device 实现 → 最后做离线侧的 loader / build。Capstone 用一次真实改动把全部串起来。

## 4. 决策记录

| # | 决策 | 理由 |
|---|---|---|
| D1 | 走方案 A（自顶向下深度优先），不走广度优先两轮 | "被问会懵"靠每块结束硬验收解决，广度浅扫不解决 |
| D2 | B1.4 仅讲 PD 分离，去掉 `batch_group_id` / `force_batch` | 学习者已熟，不浪费一节 |
| D3 | 每节验收任务为**动手类**（找 bug / 加 metric / 复现 issue），非"画图/口述" | 目标升到"能执行开发任务"，纯讲解记忆不牢 |
| D4 | 加 Capstone 第 15 节（一个真实小改动 PR） | 把 14 节知识在真实工作流（改代码 → 测试 → review → 提交）里走一遍 |
| D5 | 每节产出独立 `NN-<slug>.md`，本课程 doc 只放目录 + 哲学 | 课程 doc 不膨胀，session 文件可作长期参考 + 错题本 |
| D6 | 写到 `docs/learning/rtp-llm/`，superpowers/specs/ 下放一个 pointer | 操作笔记和 spec 设计分离，但保持 superpowers 流程可发现性 |
| D7 | 每节 lesson plan 必须**自包含**：所有代码片段、proto 定义、类声明、关键概念，全部嵌入本节文件，**学习者读这一节不需要打开任何其他文件**（除了验收任务里要求动手改的那个文件） | 学习者反馈"原版谜语人式给路标的写法看不懂"，2026-05-22 立此契约 |

## 5. 学习者-讲师契约

- **每节 1-2 小时**，时间不够先停，下次接着，不强制连贯
- **每节开头**：我先 grounded 探一次代码（不靠记忆），更新该节 lesson plan 文件中的"关键文件 + 行号"
- **每节中段**：导览（我讲）→ 验收任务（你做）→ 我对照代码批改
- **每节结尾**：把"你卡住的点 / 真实工时 / 新发现的疑问"沉淀到该节文件，必要时回写 memory
- **块末抽测**（第 4 / 7 / 11 / 14 节末）：5-8 个跨节混合题，能答出 80% 才走下一块
- **跳出权**：任何一节你觉得"已经够熟了"，可以申请跳过，我会出 3 题快速检测，过了就跳
- **倒回权**：任何一节学完发现某个前置概念没真懂，可以倒回去重做

## 6. 单节模板

```
┌─ Header ──────────────────────────────────┐
│ 节号、主题、前置依赖（哪几节）、预估时长     │
├─ Section 1: 这节后你能答的问题 ────────────┤
│ 3-5 个具体问题，验收时抽                  │
├─ Section 2: 导览路径 ──────────────────────┤
│ ASCII 调用链 + 关键文件清单（含行号）       │
│ 设计意图：为什么不是另一种写法              │
├─ Section 3: 验收任务 ──────────────────────┤
│ 一个动手类任务，预计耗时 20-40min          │
│ 评分标准：什么算通过、什么算还要回去补       │
├─ Section 4: 你的答案 ──────────────────────┤
│ (空，开课时由学习者填)                     │
├─ Section 5: 讲师批注 + 错题本 ──────────────┤
│ (空，开课时由讲师填)                      │
├─ Section 6: 沉淀 ──────────────────────────┤
│ 该节关键事实 / 路径 / 坑                  │
│ 是否需要写入 memory                       │
└──────────────────────────────────────────┘
```

## 7. 课程目录

### B1 — 调度 / 批处理（4 节）

| 节 | 主题 | 一句话验收任务 |
|----|------|----------------|
| **B1.1** | 入口与对象转换：请求从哪里进、变成什么 C++ 对象 | 在 `LocalRpcServer.cc` 里加一行 log 打印新建 stream 的 `request_id` 和 token 数，跑通 |
| **B1.2** | Engine 主循环：Stream 进来后谁推它走（主循环最终会跨到 Python，B2 接） | 把 `NormalEngine::loop` 核心 30-50 行抽出来逐行注释，标 step-local vs cross-step 状态 |
| **B1.3** | FIFOScheduler + continuous batching：怎么决定下一批是谁 | 在 FIFOScheduler 加一个简单 metric（例如"本 step 准入了几个 / 驱逐了几个"），跑 smoke 看输出 |
| **B1.4** | PD 分离：Prefill/Decode 跨节点 + cache_store 传输 | 画两张部署图（合并 vs PD）并把 KV cache 跨节点传输的代码路径列出来 |

**B1 块末抽测**：5 题混合，譬如"continuous batching 的标志性 1 行是什么"、"PD 分离下谁触发 decode 开始"

### B2 — Forward 主路径（C++ → Python）（3 节）

| 节 | 主题 | 一句话验收任务 |
|----|------|----------------|
| **B2.1** | Executor 三件套：Stream batch 怎么变成 model input tensor | 在 `NormalModelInputGatherer.cc` 加一行 assert 检查某个不变式（例：所有 stream 的 batch_size 一致），跑通 |
| **B2.2** | PyWrappedModel：C++↔Python 桥 + GIL 命运 | 用 100 字描述"一个 step 里 GIL 何时拿/放/避开"，对照代码验证 |
| **B2.3** | Qwen3 一遍 forward 看到底：Python 模型本体 | 抽出 `qwen3.py` 的 forward 函数逐行注释，标 "C++ call / kernel call / pure tensor op" |

**B2 块末抽测**：5 题，跨 Executor / PyWrappedModel / Python 模型 三层（例："Stream batch 到 logits 之间至少跨几次语言边界"）

### B3 — Kernel / device 选择层（4 节）

| 节 | 主题 | 一句话验收任务 |
|----|------|----------------|
| **B3.1** | Attention 多实现：causal vs MLA、FA2/FA3/flashinfer 怎么选 | 给 prefill/decode/MLA-prefill/MLA-decode 4 种场景各列出实际被调的 kernel + 选择条件 |
| **B3.2** | Kernel 调用面：Python 到 3rdparty bindings | 从 `causal_attention.py` 的 attention 调用追到 3rdparty 入口（如 flash-attn-3），写出 5-step 调用链 |
| **B3.3** | SM 分流：sm7x/8x/9x 如何在同一份代码里并存 | 找一个 sm9x-only kernel，画出它的 BUILD select 链路，能讲清楚为什么 sm8x 编译时不进 binary |
| **B3.4** | CUDA Graph + MTP speculative 等特化路径 | 列出"普通 decode / cuda graph decode / MTP speculative" 三种 step 在主循环里的分支点 |

**B3 块末抽测**：5 题，覆盖跨节挑战题（例："为什么不所有 step 都跑 cuda graph"）

### B4 — Loader / 量化 / 构建（3 节）

| 节 | 主题 | 一句话验收任务 |
|----|------|----------------|
| **B4.1** | Loader 主流程：weight 从磁盘到 sharded GPU tensor | 画 "model path → state_dict → sharded GPU tensors" 流程图，每步标文件位置 |
| **B4.2** | 量化变体：fp8 / awq / w4a8 等十几种 weight class 的继承与差异 | 选 3 种典型 quant weight class 做对比表（继承、scale 存储、dequant 时机） |
| **B4.3** | 构建系统：bazel + stub_source + torch ABI 全家桶 | 写一份 200 字 "为什么 prefillWarmUp 偶尔 SIGSEGV" 的根因分析（系统化你 memory 里的散点） |

**B1-B4 块末抽测**：8 题混合，跨全部 4 块

### Capstone — 真实小改动 PR（1 节）

| 节 | 主题 | 任务 |
|----|------|------|
| **15** | 用整个课程的知识完成一个真实改动 | 从仓库 issue / TODO / your dispatcher 余项中挑一个 ~100-300 行的改动，独立完成：定位 → 改 → 测试 → 自查 → 提交 |

**判定通过标准**：改动 PR 能 merge（或得到 reviewer 通过），且能口述清楚"这次改动碰到的 5 个关键决策点和理由"。

## 8. 标杆问题清单

学完应能逐条回答的 25 个问题，作为"是否融会贯通"的客观尺。卡壳 ≥ 5 题需回炉对应块。

### 主路径（10 题）

1. 一个 HTTP request 从入口到 token 返回，至少经过几次跨进程/跨语言/跨设备边界？把它们按顺序列出来。
2. `LocalRpcServer.cc` 收到 generate 请求的第一行业务代码做了什么？
3. `GenerateInput` / `GenerateConfig` / `GenerateStream` 三者的拥有/生命周期关系是？
4. `NormalEngine::loop` 的主循环最少几行能讲清楚？讲一遍。
5. continuous batching 在代码里的标志性 1 行是什么？为什么这一行是"continuous"的关键？
6. FIFOScheduler 让一个 waiting stream 进 running 的判定条件有几个？任意 3 个。
7. KV cache reuse hit 是发生在 scheduler 还是 cache_manager？具体在哪个方法？
8. `NormalExecutor` → `PyWrappedModel` → Python `qwen3.py.forward` 这条链上，每跨一段，输入数据的形状/位置怎么变？
9. 一个 step 内 GIL 经历几次拿放？哪一段是 GIL-free 的？
10. `qwen3.py` 里的 attention 调用最终落在哪个 3rdparty kernel？怎么选的？

### 调度与分布（5 题）

11. PD 分离下，prefill 节点 KV cache 传给 decode 节点是同步还是异步？由谁触发？传输用什么协议？
12. `LocalRpcServer` / `PrefillRpcServer` / `DecodeRpcServer` 在同一 binary 里共存还是分别 binary？怎么切换部署模式？
13. `force_batch=true` + 同 `batch_group_id` 的两个 stream 同时进来，FIFOScheduler 多做了什么？
14. BatchDecodeScheduler 和 FIFOScheduler 的核心区别是什么？什么场景用哪个？
15. cache_store 在 PD 分离里负责什么？它和 `cache/connector/p2p/` 是什么关系？

### Kernel 与 device（5 题）

16. MLA attention 和 causal attention 在 Python 里是 if/else 还是不同类？为什么这样设计？
17. flash-attn-3 在 sm9x 下被选中、sm8x 下退回到什么？回退路径在代码哪里决定？
18. sm9x-only 的 kernel（举 1 个）在 sm8x 编译时是怎么被排除出 binary 的？
19. cuda graph capture 在什么条件下会触发？capture 失败的兜底是什么？
20. MTP speculative decoding 的 executor 是怎么挂到主 engine 上的？它共用还是另起 scheduler？

### Loader 与构建（5 题）

21. `model_loader/loader.py` 入口拿到 model path，前 5 步做什么？
22. fp8 per-tensor / per-channel / per-block 三种量化在 weight class 上的差异在哪里？scale 怎么存？
23. 一个模型加载到多卡（TP=8）时，weight 的 sharding 是 loader 做还是别的地方做？在哪个文件？
24. `stub_source` 为什么必须是 symlink 不能直接 git？指向 `open_source` 时会发生什么问题？
25. `bazelisk test --config=cuda12_9 --config=sm9x` 这条命令在 `.bazelrc` 里展开后，关键的 5 个编译标志是什么？分别影响什么？

---

## 9. 进度追踪

详见 `README.md`（每节状态 + 真实工时 + 卡点）。

## 10. 文档维护

- 本 doc 一旦开始执行原则上不动；如设计本身要变更（例如发现某节拆错），单独加 `revisions/YYYY-MM-DD-<reason>.md`
- 每节笔记 `NN-<slug>.md` 由对应节的开课时间命名 slug
- 课程结束后，沉淀的复用资产（架构图、调用链表、量化对比表）汇总到 `summary.md`

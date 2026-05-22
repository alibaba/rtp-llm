# B1.1 — 入口与对象转换

- **session**: 1 / 15
- **block**: B1（调度 / 批处理）
- **prereq**: 无（开课）
- **est duration**: 1.5 h
- **date opened**: 2026-05-22
- **status**: 🟡 in-progress

---

## Section 1 · 这节后你能答的问题

1. 一个 HTTP request 从入口到 C++ 第一行业务代码经过几跳？分别是哪几跳？
2. `GenerateInput` / `GenerateConfig` / `GenerateStream` 三者的拥有关系、生命周期？谁先生谁后亡？
3. proto 类型（`GenerateInputPB`）和内部类型（`GenerateInput`）为什么要分两套？转换在哪里做？
4. `LocalRpcServer` / `PrefillRpcServer` / `DecodeRpcServer` 三套并存，谁负责什么场景？同一个 binary 里全有，还是分别 binary？
5. `StreamState` 一共几个值？为什么 `LOADING_CACHE` 要单独成一个状态而不是合并到 `WAITING`？

---

## Section 2 · 导览路径

### 2.1 完整调用链（端到端）

```
┌── Python frontend 进程 ────────────────────────────────────┐
│                                                            │
│  rtp_llm/pipeline/pipeline.py                              │
│   ├─ Pipeline.__call__                                     │
│   ├─ line 216: token_ids = self.tokenizer.encode(prompt)   │
│   └─ line 228: return self.generate_stream(...)            │
│       └─ line 466: async def generate_stream(...)          │
│           └─ line 495: backend_rpc_server_visitor.enqueue(input) │
│                       │                                    │
│  rtp_llm/cpp/model_rpc/model_rpc_client.py                 │
│   ├─ line 51:  input_pb = GenerateInputPB()                │
│   ├─ line 181: _input_to_protobuf(input_py, input_pb, gc)  │
│   ├─ line 358: class ModelRpcClient                        │
│   └─ line 460: stub.GenerateStreamCall(input_pb)  ◄── gRPC │
│                                                            │
└────────────────────────┬───────────────────────────────────┘
                         │
                    gRPC over TCP
                         │
┌────────────────────────▼───────────────────────────────────┐
│ C++ backend 进程                                            │
│                                                            │
│  rtp_llm/cpp/model_rpc/proto/model_rpc_service.proto       │
│   └─ line 599-625: service RpcService                      │
│       └─ line 602: rpc GenerateStreamCall(GenerateInputPB) │
│                       returns (stream GenerateOutputsPB)   │
│                                                            │
│  rtp_llm/cpp/model_rpc/LocalRpcServer.cc                   │
│   └─ line 158: GenerateStreamCall(ctx, request, writer)    │
│       ├─ line 169: prepareInput(*request, input)           │
│       │            └─ cc:125-126: QueryConverter::transQuery│
│       │                                                    │
│       ├─ line 178-179: ★ engine_->enqueue(input)           │
│       │   ← 返回 shared_ptr<GenerateStream>                │
│       │                                                    │
│       └─ pollStreamOutput(...): 循环把 stream 输出 stream 回 client │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

★ 标记的是**最关键的一行**：跨过这一行，请求就从"协议世界"进入"调度/计算世界"。B1.2 起讲的所有循环都在这一行之后展开。

### 2.2 对象生命周期

```
                 时间 →
                 
Python:
  GenerateInput(py)  ────────────────────────────────────────►  GC
                       │ (复制成 proto)
                       ▼
  GenerateInputPB     ──────►(序列化)──────►(线路上)
                                              │
                                              │ (反序列化)
C++ (server side):                            ▼
  const GenerateInputPB&  ──────►──────►  prepareInput
                                              │
                                              │ QueryConverter::transQuery
                                              ▼
  shared_ptr<GenerateInput>  ──►──►──►─►  engine_->enqueue(input)
                                              │
                                              │ engine 内部 new
                                              ▼
  shared_ptr<GenerateStream>  ◄── 同时被 2 个地方持有：
                                  (a) Scheduler 的 waiting/running 队列
                                  (b) GenerateStreamCall 本地变量（用于 poll 输出）
                                  
                              ──── 主循环每 step 推进 stream 状态 ────
                              
  StreamState: WAITING → LOADING_CACHE → RUNNING → FINISHED
                                              │
                                              │ 当 FINISHED 且 client 收完 output
                                              ▼
                                          shared_ptr 计数归零 → 析构
```

**关键关系**：
- `GenerateInput` 装着**输入数据** + 指向 `GenerateConfig`（生成参数）
- `GenerateStream` 装着 `GenerateInput` + `StreamCacheResource`(KV 块引用) + state 机 + 部分输出累积
- `GenerateConfig` 由 `GenerateInput` 持有；stream 通过 input 间接访问
- **拥有关系**：`Stream ⊇ Input ⊇ Config`（包含关系，shared_ptr）

### 2.3 关键文件清单（带行号锚点）

| 角色 | 文件 | 行号 / 符号 |
|------|------|------------|
| Python 入口 | `rtp_llm/pipeline/pipeline.py` | `:216`(tokenize), `:466`(generate_stream), `:495`(enqueue) |
| Python gRPC client | `rtp_llm/cpp/model_rpc/model_rpc_client.py` | `:358`(class), `:460`(GenerateStreamCall) |
| proto 服务定义 | `rtp_llm/cpp/model_rpc/proto/model_rpc_service.proto` | `:599-625`(service), `:602`(GenerateStreamCall), `:137`(GenerateInputPB), `:42`(GenerateConfigPB) |
| C++ 服务声明 | `rtp_llm/cpp/model_rpc/LocalRpcServer.h` | `:28`(class), `:42-44`(GenerateStreamCall), `:119`(engine_ 成员) |
| **C++ 服务实现** | `rtp_llm/cpp/model_rpc/LocalRpcServer.cc` | `:158`(method), `:169`(prepareInput), **`:179`(engine_->enqueue ★)** |
| proto→内部转换 | `rtp_llm/cpp/model_rpc/QueryConverter.h` | `:12`(class), `:14`(transQuery), `:35`(transGenerateConfig) |
| proto→内部转换 impl | `rtp_llm/cpp/model_rpc/QueryConverter.cc` | 344 行，整体浏览即可 |
| GenerateInput 类型 | `rtp_llm/cpp/engine_base/stream/GenerateTypes.h` | `:16`(class), `:44`(成员), `:60-61`(batch_group_id) |
| 输出类型 | `rtp_llm/cpp/engine_base/stream/GenerateTypes.h` | `:64`(AuxInfo), `:92`(GenerateOutput), `:105`(GenerateOutputs) |
| **状态机** | `rtp_llm/cpp/engine_base/stream/GenerateTypes.h` | **`:111-116`(StreamState)**, `:136-157`(StreamEvents) |
| Stream 类（暂只看头） | `rtp_llm/cpp/engine_base/stream/GenerateStream.h` | 641 行；本节先扫前 50 行 + `:431`(batch_group_id getter) |
| GenerateConfig | `rtp_llm/cpp/engine_base/stream/GenerateConfig.h` | 90+ 行；扫看字段 |
| PD 分离的 RPC server | `rtp_llm/cpp/model_rpc/PrefillRpcServer.{h,cc}`、`DecodeRpcServer.{h,cc}` | 本节只看类名 + 继承关系（详细到 B1.4） |

### 2.4 设计意图（"为什么不是另一种写法"）

| 决策 | 选了 | 备选 | 为什么 |
|------|------|------|--------|
| proto vs 内部类型分两套 | `GenerateInputPB` + `GenerateInput` 各一份 | 直接用 proto 类传整个 backend | proto 类被 protoc 生成，没法塞 `torch::Tensor` / `shared_ptr<GenerateConfig>` / 自定义方法；内部类要承载实际计算用的数据结构（tensor、KV 引用）。proto 只是**线路格式** |
| RpcServer 三件套 | `LocalRpcServer` / `PrefillRpcServer` / `DecodeRpcServer` 并列 | 一个 server 内部 if/else 分支 | 三种部署模式（单机合并、PD prefill 节点、PD decode 节点）的请求来源/输出去向/cache 行为都不同，分类继承比一个 if-else 巨怪更清晰。B1.4 会看 Prefill/Decode 怎么走 |
| StreamState 把 LOADING_CACHE 单列 | 4 个状态：WAITING / LOADING_CACHE / RUNNING / FINISHED | 把 LOADING_CACHE 当 WAITING 的子状态 | LOADING_CACHE = GPU 块已分配但 H2D 拷贝还没完，**资源占用与 WAITING 不同**（已扣 KV 块）但还不能跑 forward。单列让调度器统计 / 准入决策能精准区分 |
| StreamEvents 用 bit flag | 32 位 enum 组合 | 一个事件队列 | 事件是**永久的**（不会自动消费），bit flag 天然支持"曾经发生过" 的语义查询，也便于在状态机里组合多事件触发同一次 `moveToNext` |

---

## Section 3 · 验收任务（动手）

### 任务 A（必做，~15min）：trace + 文字答题

打开 `rtp_llm/cpp/model_rpc/LocalRpcServer.cc`，定位 `GenerateStreamCall` 方法（line 158）。**不需要编译**，做以下 3 件事：

1. **抽出主路径**：从 `:158` 开始到 `:179` 这一段的核心 7-10 行（去掉日志/metrics 装饰），按顺序列出来。每行写一句中文注释。
2. **画一张所有权图**：纵轴是时间，横轴是变量。把 `request`(in)、`input`(shared_ptr<GenerateInput>)、`stream`(shared_ptr<GenerateStream>)、`writer`(gRPC) 这 4 个的生命周期画成 4 条横线，标出诞生 / 转移 / 销毁的时刻。
3. **回答**：
   - **Q1**：`prepareInput` 返回后，`request`（proto）这个对象还需要存活吗？为什么？
   - **Q2**：`engine_->enqueue(input)` 之后，`input` 这个 `shared_ptr` 在这个函数里立即被销毁吗？看 cc:178-200 附近，谁还在持有它？

把答案写到下面 Section 4。

### 任务 B（选做，~20min，需要编译）：加一行 log

在 `GenerateStreamCall` 方法入口（line 158 之后的第一行业务代码处）加一行：

```cpp
RTP_LLM_LOG_INFO("recv generate request, request_id=%ld, input_tokens=%d",
                 request->request_id(), request->token_ids_size());
```

然后：
1. `bazelisk build //rtp_llm/cpp/model_rpc:...` 看能否过编译
2. 拉一个本地 smoke test 或写个 minimal Python client 跑一发请求（可参考 `rtp_llm/cpp/model_rpc/model_rpc_client.py` 用法）
3. 在日志里找到这一行
4. **回答**：如果 client 发了一个 batch=500 的 `BatchGenerateCall`，这行 log 出现几次？为什么？

如果时间不够、build 报错卡很久，**就只做 A**，不强求 B。

### 评分

- **A 全做完**：合格。可以走 B1.2。
- **A 错 ≤ 1 题**：基本合格，我点几句就走。
- **A 错 ≥ 2 题**：回炉，把对应的代码段一起看一遍再答。
- **B 完成**：额外加分，更有"动手手感"。

---

## Section 4 · 你的答案

> （由学习者在开课时填）

### 任务 A.1 — 主路径 7-10 行 + 注释

```
（在这里贴你抽出来的代码 + 注释）
```

### 任务 A.2 — 所有权图

```
（ASCII 或贴个手绘照片链接）
```

### 任务 A.3 — Q1 答案

> 

### 任务 A.3 — Q2 答案

> 

### 任务 B（选做）

>

---

## Section 5 · 讲师批注 + 错题本

> （讲师在 review 学习者答案后填）

---

## Section 6 · 沉淀

### 关键事实
- `engine_->enqueue(input)` 是协议世界 → 调度/计算世界的**唯一入口**（GenerateStream 路径）
- `StreamState` 4 值：WAITING → LOADING_CACHE → RUNNING → FINISHED；LOADING_CACHE 单列因为已占 GPU 块但未可跑
- proto 类型和内部类型必须分两套（torch::Tensor / shared_ptr 无法塞 proto）
- `Stream ⊇ Input ⊇ Config` 包含关系

### 真实工时
- 准备：__ min
- 学习者答题：__ min
- 讲师批改：__ min
- **总计**：__ min

### 卡点（如有）
> 

### 是否写入 memory？
- [ ] StreamState 4 值 + 含义 → 值得写
- [ ] gRPC 入口的 ★ 一行（engine_->enqueue）→ 已经在课程文档里，不重复写 memory
- [ ] 其他：

# B1.1 — 入口与对象转换

- **session**: 1 / 15
- **block**: B1（调度 / 批处理）
- **prereq**: 无（开课）
- **est duration**: 2 h
- **date opened**: 2026-05-22
- **status**: 🟡 in-progress（rewritten 2026-05-22 per D7 self-contained 契约）

---

## 这节学完你要会什么

学完后你能不打开代码、就能讲清楚以下 5 件事：

1. 一个 generate 请求从 Python 入口到 C++ 第一行业务代码，经过几跳，每跳做什么
2. `GenerateInput` / `GenerateConfig` / `GenerateStream` 这三个 C++ 类各装什么、谁拥有谁、生命周期怎么走
3. 为什么有 proto 类型（`GenerateInputPB`）和内部类型（`GenerateInput`）两套？转换在哪里、做了什么
4. `LocalRpcServer` / `PrefillRpcServer` / `DecodeRpcServer` 这三个服务并存的原因，分别对应什么部署
5. C++ 里的 `StreamState` 为什么有 4 个值（不是 3 个），多出来的 `LOADING_CACHE` 解决什么问题

---

## 0. 前置概念（不假设你已经会）

> 如果你已经熟悉 gRPC + protobuf + C++ shared_ptr，可以跳过这节。

### 0.1 gRPC 是什么、与 REST 的区别

gRPC 是 Google 出的一个 RPC 框架：客户端像调本地函数一样调远端服务，框架负责把参数序列化、发送、远端反序列化、调函数、返回值序列化回来。

跟 REST 的差别：
- **接口描述**：REST 是文档+约定（GET /users/123）；gRPC 是用 `.proto` 文件描述（一个 `service` 含多个 `rpc` 方法）
- **序列化**：REST 一般 JSON（文本）；gRPC 用 protobuf（二进制）—— 小、快、有 schema
- **HTTP 版本**：REST 一般 HTTP/1.1；gRPC 用 HTTP/2，原生支持长连接 + 多路复用 + **流式**
- **多语言**：`.proto` 一份，可以生成 Python / C++ / Java / Go 等多种语言的 stub

RTP-LLM 选 gRPC 是因为 Python frontend 和 C++ backend 跨进程 / 跨机器（PD 分离），需要高性能 + 强 schema + 流式（边生成边返回 token）。

### 0.2 proto 是什么，protoc 生成什么

`.proto` 文件描述消息格式和服务接口。例如：

```protobuf
message GenerateInputPB {
    int64 request_id = 1;          // 字段编号是协议的核心
    repeated int32 token_ids = 2;
    GenerateConfigPB generate_config = 4;
}

service RpcService {
    rpc GenerateStreamCall(GenerateInputPB) returns (stream GenerateOutputsPB);
}
```

`protoc` 编译器读这个文件生成代码：
- **Python**：一个 `GenerateInputPB` class（dataclass-like）和一个 `RpcServiceStub`（客户端调用代理）
- **C++**：一个 `GenerateInputPB` C++ class（getter/setter/has_*/序列化方法）和一个 `RpcService::Service` 基类（你继承并实现方法）

**关键约束**：proto 字段只支持 scalar / string / bytes / repeated / 嵌套 message，**不能**放 `torch::Tensor`、`shared_ptr`、自定义对象。这是后面"为什么要内部类型"的根因。

### 0.3 流式 RPC vs 一元 RPC

```protobuf
// 一元 (unary)：客户端发 1 个 request，服务器返 1 个 response
rpc Foo(Req) returns (Resp);

// 服务器流 (server-streaming)：客户端发 1 个，服务器返多个
rpc Bar(Req) returns (stream Resp);
```

LLM 生成天然是流式：客户端发 prompt，服务器每生成一个 token 就 push 一次。所以本节看的 `GenerateStreamCall` 是 server-streaming。

在 C++ 服务端，server-streaming 的 handler 签名是：
```cpp
grpc::Status GenerateStreamCall(
    grpc::ServerContext* context,           // 元数据 / cancel / deadline
    const RequestPB* request,                // 客户端发来的 1 个 request
    grpc::ServerWriter<ResponsePB>* writer); // 你往这个 writer 里 Write() 多次，每次是 1 个 response 帧
```

### 0.4 shared_ptr 与"双持有"

C++ `std::shared_ptr<T>` 是引用计数智能指针：每多一个 `shared_ptr<T>` 指着同一对象，计数 +1；析构 -1；归零就 delete 真实对象。

本节会反复看到：一个 `shared_ptr<GenerateStream>` 同时被两个地方持有——
- (a) Scheduler 的内部队列（waiting/running）
- (b) RpcServer 的 GenerateStreamCall handler 本地变量（用于 poll 输出）

只要任一一方还持有，stream 就不会析构。两方都松手了才真正释放。这种"双持有"模式在异步系统里很常见。

---

## 1. 主链路：从 Python 到 C++ 的逐段精读

下面 5 段按调用顺序走。每段都贴**实际代码** + 解释。

```
Python frontend 进程                          C++ backend 进程
─────────────────                          ────────────────
[1] pipeline.py                              [4] LocalRpcServer.cc
      │ 调用                                      ▲ proto 反序列化 + 调 handler
      ▼                                           │
[2] model_rpc_client.py                      ─ gRPC 网络 ─
      │ Python obj → proto                       ▲
      ▼                                           │
[3] gRPC 客户端 stub.GenerateStreamCall ────────┘
                                              [5] QueryConverter::transQuery
                                                   (proto → 内部 GenerateInput)
```

### 1.1 Python pipeline 怎么发起请求

文件：`rtp_llm/pipeline/pipeline.py`，关键行：

```python
# line 216  —— 先把 prompt tokenize 成 input_ids
token_ids = self.tokenizer.encode(prompt)

# line 228  —— 然后把 token_ids + config 包装成 GenerateInput，调 generate_stream
return self.generate_stream(token_ids, generate_config, ...)
```

`generate_stream` 是个 async 生成器，里面（line 466 起）：

```python
async def generate_stream(self, ...):
    # ... 各种参数处理 ...
    await self.backend_rpc_server_visitor.enqueue(input)   # line 495
```

`backend_rpc_server_visitor` 实际是 `ModelRpcClient`（或它的包装），下一段看它。

**这一段你要记住**：Python pipeline 把 prompt 变 token_ids 后，构造一个 Python 的 `GenerateInput` 对象，**还没序列化**，调 `enqueue(input)`。

### 1.2 Python 客户端怎么把对象变 proto + 发 gRPC

文件：`rtp_llm/cpp/model_rpc/model_rpc_client.py`，关键片段（精简后）：

```python
# line 51 附近 —— 构造空 proto
input_pb = GenerateInputPB()

# line 181 —— 把 Python GenerateInput 的字段填进 proto
_input_to_protobuf(input_py, input_pb, generate_config)
# 这个函数做的事大致是：
#   input_pb.request_id = input_py.request_id
#   input_pb.token_ids.extend(input_py.token_ids)
#   input_pb.generate_config.max_new_tokens = generate_config.max_new_tokens
#   ... 几十个字段挨个 copy

# line 358 —— 客户端类
class ModelRpcClient(object):
    # ... 内部维护 channel pool (gRPC 长连接) ...

# line 460 —— 真正的 gRPC 调用
channel = await self._channel_pool.get(target_address)
stub = RpcServiceStub(channel)
response_iterator = stub.GenerateStreamCall(
    input_pb, timeout=grpc_timeout_seconds
)

# server-streaming —— 异步迭代每个 response 帧
async for response in response_iterator.__aiter__():
    yield trans_output(input_py, response, stream_state)
```

**这一段你要记住**：
- Python 对象 → proto 是**字段挨个 copy**，没有黑魔法
- gRPC 实际调用是 `stub.GenerateStreamCall(input_pb)`，返回一个**异步迭代器**
- 每次 `async for response in ...` 拿到 1 个 `GenerateOutputsPB`，对应服务器 push 的一帧（通常是 1 个或一小批新生成的 token）

### 1.3 proto 服务接口的本体

文件：`rtp_llm/cpp/model_rpc/proto/model_rpc_service.proto`

整个 RPC 服务的定义在末尾这个 service block：

```protobuf
// line 599-625
service RpcService {
    rpc GetWorkerStatus(StatusVersionPB) returns (WorkerStatusPB);           // 健康/状态查询
    rpc GetCacheStatus(CacheVersionPB) returns (CacheStatusPB);              // KV cache 查询
    rpc GenerateStreamCall(GenerateInputPB) returns (stream GenerateOutputsPB);  // ★ 单条流式生成
    rpc BatchGenerateCall(BatchGenerateInputPB) returns (BatchGenerateOutputsPB); // ★ 批量一元生成
    rpc RemoteLoad(BroadcastLoadRequestPB) returns (BroadcastLoadResponsePB); // PD 间 KV 加载
    rpc RemoteGenerate(stream GenerateRequestPB) returns (stream GenerateOutputsPB); // PD 远端生成
    // ... 还有 RemoteFinish / CheckHealth / SetPause / UpdateWeights 等运维 RPC ...
}
```

加 ★ 的两个是本节关心的"接收请求"入口：
- `GenerateStreamCall`：单条请求，server-streaming（每生成一帧 push 一次）
- `BatchGenerateCall`：一批请求打包发送，unary 返回（全部跑完一次返）

`GenerateInputPB` 自己长这样（精简，省略部分字段）：

```protobuf
// line 137
message GenerateInputPB {
    int64 request_id = 1;
    repeated int32 token_ids = 2;                       // tokenize 后的 input ids
    repeated MultimodalInputPB multimodal_inputs = 3;   // 多模态输入（图片等）
    GenerateConfigPB generate_config = 4;               // ★ 所有生成参数（top_k、max_new_tokens...）
    string client_id = 5;
    int64 start_time = 6;
    int32 batch_group_size = 7;
    google.protobuf.Int64Value batch_group_id = 8;      // force_batch 用的 group id
}
```

`GenerateConfigPB`（line 42 起）就更大了，有 ~60 个字段（top_k / top_p / temperature / num_beams / stop_words_list / return_logits / ...）—— 你不需要全记，知道**所有生成相关参数都在这里**就行。

**这一段你要记住**：
- 整个 backend 服务只有一个 `service RpcService`
- 进入"生成"路径的有两个 rpc：`GenerateStreamCall`（单条流式）和 `BatchGenerateCall`（批量）
- payload 是 `GenerateInputPB`，里面装 token_ids + generate_config + 可选 multimodal

### 1.4 ★ C++ 服务端入口：`LocalRpcServer::GenerateStreamCall`

这是**本节最重要的代码**。文件：`rtp_llm/cpp/model_rpc/LocalRpcServer.cc`，函数完整体（line 158-188）：

```cpp
grpc::Status LocalRpcServer::GenerateStreamCall(grpc::ServerContext*                   context,
                                                const GenerateInputPB*                 request,
                                                grpc::ServerWriter<GenerateOutputsPB>* writer) {
    RTP_LLM_PROFILE_SCOPE("rpc.generate_stream_call");                    // [A] profile 开始计时
    AtomicGuard request_guard(onflight_requests_);                        // [B] 在飞计数 +1，析构时 -1
    auto        request_id = request->request_id();                       // [C] 取 request_id
    RTP_LLM_LOG_DEBUG("receive request %ld", request_id);

    auto generate_context =
        GenerateContext(request_id, request->generate_config().timeout_ms(),
                        context, metrics_reporter_, meta_);               // [D] 本次请求的上下文容器
    std::shared_ptr<GenerateInput> input;                                 // [E] 内部 GenerateInput，未填
    {
        auto mm_res = prepareInput(*request, input);                      // [F] ★ proto → 内部 input
        if (!mm_res.ok()) {
            generate_context.error_status = serializeErrorMsg(generate_context.request_key, mm_res);
        }
    }
    CHECK_ERROR_STATUS(generate_context);                                 // [G] 错则提前返回
    RTP_LLM_LOG_DEBUG("request [%ld] trans to stream success", request_id);

    {
        RTP_LLM_PROFILE_SCOPE("rpc.enqueue_engine");
        generate_context.setStream(engine_->enqueue(input));              // [H] ★★ 关键一行：入队 engine
    }
    RTP_LLM_LOG_DEBUG("request [%ld] enqueue success", request_id);

    generate_context.error_status =
        pollStreamOutput(context, generate_context.request_key,
                         writer, generate_context.getStream());           // [I] 循环 poll stream 输出，
                                                                          //     边出边 writer->Write() 给 client
    meta_->dequeue(generate_context.request_id, generate_context.getStream());  // [J] 元数据登记 done
    return generate_context.error_status;
}
```

**逐行解释**：

- **[A]** `RTP_LLM_PROFILE_SCOPE(...)` 是一个 RAII 计时器宏，构造时记开始时间，析构时记结束时间，进 metrics。每个 scope 一个 name，可以嵌套。
- **[B]** `AtomicGuard request_guard(onflight_requests_)` 是个 RAII：构造时 `onflight_requests_++`，析构时 `--`。所以函数返回（无论正常或异常）时计数自动还原。`onflight_requests_` 是 `std::atomic<size_t>`（看 .h:124），用于"当前在飞请求数"的统计。
- **[C]** `request->request_id()` 是 protoc 生成的 getter（小写带下划线变成函数）。
- **[D]** `GenerateContext` 是一个聚合容器（不是 stream，别混淆），装 request_id / timeout / cancel context / metrics / shared_ptr 给后面用，统一管理本次 RPC 的元数据。
- **[E]** 先声明一个空的 `std::shared_ptr<GenerateInput>`，准备让 prepareInput 填。
- **[F]** `prepareInput(*request, input)` 把 proto `request` 翻译成内部 `GenerateInput`（含 multimodal 处理）。失败返 ErrorInfo。
- **[G]** 如果上一步报错，直接返回错误 grpc::Status。
- **[H]** ★★ **整个 B1.1 的核心一行**。`engine_->enqueue(input)` 把 input 喂给 EngineBase（后面 B1.2/1.3 详讲），engine 内部会 new 一个 `GenerateStream` 并把它塞进 scheduler 的 waiting 队列；同时返回这个 stream 的 `shared_ptr` 给我们。`generate_context.setStream(...)` 把这个 stream 存进 context（这样 context 持有 stream 的一份引用）。**跨过这一行，请求就从"协议世界"进入"调度/计算世界"**。
- **[I]** `pollStreamOutput(...)` 是个循环：不停从 stream 取新生成的 token，每取到一帧就 `writer->Write(outputs_pb)` 推给 client。stream FINISHED 或 error 时循环退出。**这个循环阻塞当前线程**，整个 RPC 调用就停在这里直到生成完。
- **[J]** 收尾：通知 meta（运维元数据管理器）这个请求 done 了。

**几个关键观察**：

1. 这个函数本身没有线程，**完全串行**。`engine_->enqueue` 之后真正的 forward 是 engine 在自己线程里跑，本函数只是用 `pollStreamOutput` 在另一头等结果。
2. `input` 这个 `shared_ptr` 在函数末尾析构，但 stream 内部持有 `input`（B1.2 会看到），所以 input 实际不会立即销毁——典型双持有。
3. `request`（proto）只在 prepareInput 里被读，之后整个函数都不再用——但**生命周期**绑在 gRPC 框架，本函数返回前不会被销毁。

### 1.5 proto → 内部对象：`QueryConverter::transQuery`

`prepareInput` 内部其实就一行（LocalRpcServer.cc:125-126）：

```cpp
ErrorInfo LocalRpcServer::prepareInput(const GenerateInputPB& input_pb,
                                       std::shared_ptr<GenerateInput>& output) {
    output = QueryConverter::transQuery(&input_pb);
    // ... 后面还有 multimodal 异步处理（本节不深入）...
}
```

真正的翻译在 `QueryConverter.cc:108`：

```cpp
std::shared_ptr<GenerateInput> QueryConverter::transQuery(const GenerateInputPB* input) {
    std::shared_ptr<GenerateInput> generate_input = std::make_shared<GenerateInput>();   // [a] 堆上 new
    generate_input->request_id    = input->request_id();                                 // [b] scalar 字段直接 copy
    generate_input->begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();     // [c] 打入队时间戳

    if (input->has_generate_config()) {
        generate_input->generate_config = transGenerateConfig(&(input->generate_config())); // [d] 嵌套 config 递归翻译
    }

    generate_input->input_ids =
        torch::from_blob(const_cast<int*>(input->token_ids().data()),                    // [e] ★ token_ids → torch::Tensor
                         {(int64_t)input->token_ids_size()},
                         torch::kInt32)
            .clone();                                                                    //   ★ .clone() 拷出来，不共享 proto buffer

    if (input->multimodal_inputs_size() > 0) {                                           // [f] 多模态输入翻译
        std::vector<MultimodalInput> mm_inputs;
        for (int i = 0; i < input->multimodal_inputs_size(); i++) {
            auto mm_input             = &input->multimodal_inputs(i);
            auto mm_preprocess_config = &mm_input->mm_preprocess_config();
            mm_inputs.emplace_back(mm_input->multimodal_url(),
                                   torch::empty(1),
                                   mm_input->multimodal_type(),
                                   mm_preprocess_config->width(),
                                   /* ... */
                                   mm_preprocess_config->max_frames());
        }
        generate_input->multimodal_inputs = std::move(mm_inputs);
    }

    generate_input->batch_group_size = input->batch_group_size() > 0 ?
                                       input->batch_group_size() : 1;
    if (input->has_batch_group_id()) {
        generate_input->batch_group_id = input->batch_group_id().value();
    }
    return generate_input;
}
```

**逐段解释**：

- **[a]** `make_shared` 在堆上 new 一个 `GenerateInput`，引用计数 = 1。
- **[b][c]** scalar 字段平凡 copy。`begin_time_us` 不是来自 proto 是当场打的"内部入队时间"，后面用于统计 wait_time。
- **[d]** `GenerateConfigPB` → `GenerateConfig` 走另一个函数 `transGenerateConfig`（QueryConverter.cc:14-106），里面是 60+ 个字段挨个 copy 的死代码（你可以扫一眼那个函数体感受一下"字段挨个搬"的工作量）。
- **[e]** ★ **关键**：proto 里 `repeated int32 token_ids` 在 C++ 是 `google::protobuf::RepeatedField<int32>`，内存是连续的。`torch::from_blob(ptr, shape, dtype)` 用这块连续内存**当 view** 包出一个 tensor，但**立即 `.clone()`** —— 这一步把数据拷到 torch 自己管理的内存里。**为什么要 clone？** 因为 proto 对象（`*input`）的生命周期归 gRPC 框架管，本函数返回后可能被回收；而 `generate_input` 要跟着 stream 活很久，绝不能让它持有指向已释放内存的指针。
- **[f]** 多模态输入翻译；多模态特性本课程不深入。

**这一段你要记住**：
- 翻译是**字段挨个搬** + **必要时 deep copy**（tensor 数据要拷）
- proto 不能塞 `torch::Tensor`，所以必然需要这一层
- 内部 `GenerateInput` 一旦构造完，跟 proto 对象**彻底独立**（生命周期、内存都不共享）

---

## 2. 三个核心 C++ 类详解

### 2.1 `GenerateInput` —— 装"输入数据 + 生成参数指针"

文件：`rtp_llm/cpp/engine_base/stream/GenerateTypes.h:16`，整个类（小，60 行）：

```cpp
class GenerateInput {
public:
    int inputLength() {
        RTP_LLM_CHECK(input_ids.dim() == 1);
        return input_ids.size(0);
    }
    int promptLength() {
        return inputLength() - prefix_length;
    }
    std::string debugString() const { /* ... */ }
    void updatePrefix(const std::vector<int>& prefix_prompt) { /* ... */ }

public:
    int64_t                         request_id = 0;
    std::shared_ptr<GenerateConfig> generate_config;     // ★ 持有 config
    torch::Tensor                   input_ids;            // 输入 token ids (1D int32 tensor)
    bool                            need_release_resource = true;
    bool                            fake_query            = false;

    // 多模态（图片/视频）
    std::optional<std::vector<MultimodalInput>> multimodal_inputs;
    std::optional<std::vector<torch::Tensor>>   multimodal_features;
    std::optional<torch::Tensor>                text_tokens_mask;
    std::optional<torch::Tensor>                mm_locs;
    std::optional<std::vector<torch::Tensor>>   mm_position_ids;

    int     prefix_length = 0;            // system prompt / cache reuse 的前缀长度
    int64_t begin_time_us = 0;            // 入队时间戳，用于统计 wait_time

    // batch grouping (force_batch 用)
    int     batch_group_size = 1;
    int64_t batch_group_id   = -1;        // -1 表示不参与强制组批
};
```

**关键观察**：
- 没有 KV cache、没有调度状态——这些是 `GenerateStream` 的事。`GenerateInput` 只描述"这个请求要算什么"。
- `generate_config` 是 `shared_ptr`——可以被多个 stream 共享（用于 batch tokenize 这种场景）。
- `input_ids` 是 torch tensor，cpu 上（这时还没上 GPU）。

### 2.2 `GenerateConfig` —— 装所有生成参数

文件：`rtp_llm/cpp/engine_base/stream/GenerateConfig.h:23`，主要字段（精简）：

```cpp
class GenerateConfig: public autil::legacy::Jsonizable {  // 可 JSON 化（运维用）
public:
    int global_request_id  = -1;
    int max_new_tokens     = 8192;
    int min_new_tokens     = 0;
    int num_validate_token = 0;          // speculative decoding 校验

    // 采样参数
    int                  num_beams            = 1;     // beam search
    std::vector<int>     variable_num_beams;
    int                  num_return_sequences = 1;
    int                  top_k                = 0;
    float                top_p                = 1.0;
    float                temperature          = 1.0;
    float                repetition_penalty   = 1.0;
    float                presence_penalty     = 0.0;
    float                frequency_penalty    = 0.0;
    std::optional<int>   random_seed;
    std::optional<int>   no_repeat_ngram_size;

    // 输出控制
    bool                          return_logits            = false;
    bool                          return_hidden_states     = false;
    bool                          return_incremental       = false;
    bool                          is_streaming             = false;
    int                           timeout_ms               = -1;
    std::vector<std::vector<int>> stop_words_list;

    // 推理特性开关
    bool can_use_pd_separation = true;
    bool reuse_cache           = true;
    bool enable_device_cache   = true;
    bool enable_memory_cache   = true;
    bool enable_remote_cache   = true;

    // batch grouping
    bool force_batch = false;     // 设为 true 时，相同 batch_group_id 的 stream 必须一起 schedule
    std::optional<int> batch_group_timeout;

    // LoRA / 适配器
    std::string adapter_name = "";

    // PD 分离专用
    bool pd_separation = false;

    // ... 其他 ~30 个字段（hidden_states 处理、speculative、profile 等）
};
```

注释里有句关键的设计说明（line 15-21）：

> Params should be split into two parts: (1) per-sampler-different (top_k, top_p, temperature...), (2) per-sampler-same (beam_size, max_seq_len...). For (2), different samplers should be created for different params, so they can't be batched together for now.

意思是：**一批 stream 能不能 batch 跑 sampling**，依赖它们的 `GenerateConfig` 在某些字段上是否一致。这是后面 B1.3 调度器逻辑的伏笔。

### 2.3 `GenerateStream` —— 装"调度 + 计算 + 输出"全套状态

文件：`rtp_llm/cpp/engine_base/stream/GenerateStream.h`。这个类有 641 行，本节**只看类头的关键 alias 和声明**，详细的字段 / 方法等后续节遇到再讲。

```cpp
// line 84
using GenerateStreamPtr = std::shared_ptr<GenerateStream>;

// class GenerateStream（声明，本节不全展开）
//   - 持有 input (GenerateInput)
//   - 持有 stream_cache_resource (StreamCacheResource，KV block 引用)
//   - 持有 state machine (GenerateStateMachine)
//   - 持有 complete_token_ids（已生成的 token 累积）
//   - 提供 nextOutput() / setStop() / 等接口给 RpcServer poll
//   - line 431: batch_group_id getter（你已经知道这个字段）
```

**关键观察**：
- `GenerateInput` 是**输入快照**，`GenerateStream` 是**运行时态**。
- Stream 持有 input 的 shared_ptr，所以 input 至少跟 stream 一样长寿。
- Stream 是 schedule / forward / sample 的中央数据结构，**所有后续节都会回到这个类**。

### 2.4 `StreamState` —— 4 个值，不是 3 个

文件：`rtp_llm/cpp/engine_base/stream/GenerateTypes.h:111`：

```cpp
enum class StreamState {
    WAITING,         // 在 scheduler 等待，未分配任何 GPU 资源
    LOADING_CACHE,   // ★ GPU blocks 已分配，等待 connector H2D 拷贝完成
    RUNNING,         // 正在被 forward
    FINISHED         // 跑完或出错
};
```

**为什么 LOADING_CACHE 要单列**？

考虑这个场景：调度器决定让某个 stream 进入运行，为它分配了 N 个 KV cache block（占住了显存），但 KV 数据本身（来自 cache_store 或 P2P connector）还没拷到 GPU。这时这个 stream：

- 不算 WAITING（已占用 GPU 资源，调度器统计 free blocks 时要算它）
- 也不算 RUNNING（还不能跑 forward）

如果合并到 WAITING，调度器在做"还能塞下几个新 request"决策时会高估可用资源；如果合并到 RUNNING，executor 会试图跑一个 KV 没准备好的 stream 而崩。所以必须单独一个状态。

PD 分离场景下，LOADING_CACHE 更突出：decode 节点要从 prefill 节点 P2P 拷 KV 过来，这个时间可能非平凡（毫秒级），状态必须能表达。

---

## 3. 对象生命周期完整图

```
                                时间 →

Python 进程
─────────
GenerateInput(py)  ─────┐
                        │ (字段 copy)
                        ▼
GenerateInputPB  ─────────────┐
                              │ (序列化 + 发送)
                              ▼
                         ─ gRPC 网络 ─
                              │
═══════════════════════════════════════════════════════════════
C++ 进程
─────────                     ▼
                  const GenerateInputPB& request  ─────────────┐
                  (gRPC 框架持有，函数返回前不销毁)              │
                              │                                │
                              │ prepareInput / QueryConverter  │
                              ▼                                │
                  shared_ptr<GenerateInput> input  ────┐       │
                  (本函数局部变量，refcount=1)          │       │
                              │                       │       │
                              │ engine_->enqueue(input)       │
                              ▼                       │       │
                  engine 内部 new GenerateStream      │       │
                  传入 input → stream 持 input ref   │       │
                              │                       │       │
              ┌───────────────┼──────────────────┐    │       │
              │               │                  │    │       │
              ▼               ▼                  ▼    ▼       │
       Scheduler 队列    GenerateContext      input refcount=2│
       持 stream ref    持 stream ref         (stream内 + 局部)│
       (running)        (本函数中)                             │
              │               │                                │
              │               │ pollStreamOutput 循环           │
              │               │ writer->Write(outputs_pb)       │
              │               │                                │
              │   (期间 stream 状态:                            │
              │    WAITING → LOADING_CACHE → RUNNING → FINISHED)│
              │                                                │
              │               │                                │
              ▼               ▼                                ▼
      FINISHED 后 scheduler   函数返回前 context 析构      函数返回时
      移除 stream ref        stream refcount -1            request_guard 析构,
                                                          onflight_requests -1
                              │
                              │ stream refcount 全归零
                              ▼
                  GenerateStream 析构
                  → stream 内的 input refcount -1
                  → input refcount 归零 → 析构
                  → input.input_ids tensor refcount -1 → 显存/内存释放
```

**用大白话总结**：
1. 请求进来 → 翻译成内部对象 → 喂给 engine → 拿到一个 stream
2. stream 同时被 scheduler 和当前 RPC handler 两个地方持有（双持有）
3. handler 在 pollStreamOutput 里阻塞着等 stream 出结果
4. stream 跑完，scheduler 松手，handler 也松手，对象一起回归宇宙

---

## 4. 三个并存的 RpcServer（Local / Prefill / Decode）

文件存在情况（rtp_llm/cpp/model_rpc/）：

- `LocalRpcServer.{h,cc}`     ← 本节看的，单节点合并部署
- `PrefillRpcServer.{h,cc}`   ← PD 分离的 prefill 节点角色
- `DecodeRpcServer.{h,cc}`    ← PD 分离的 decode 节点角色
- `RemoteRpcServer.{cc}`      ← 复合
- 配套：`LocalRpcServiceImpl.h`、`RemoteRpcServiceImpl.h`、`PrefillGenerateContext.cc`、`DecodeGenerateContext.cc`

**对应的部署模型**：

```
[场景 A] 单节点合并部署 (LocalRpcServer)
─────────────────────────────────────
  Python frontend  ──gRPC──>  C++ LocalRpcServer
                                  │
                                  ├─ engine（含 prefill + decode 在同一进程）
                                  └─ scheduler（FIFO）

[场景 B] PD 分离部署 (PrefillRpcServer + DecodeRpcServer)
─────────────────────────────────────────────────────
  Python frontend  ──gRPC──>  Prefill 节点(PrefillRpcServer)
                                  │
                                  ├─ engine 只做 prefill
                                  ├─ scheduler 只调度 prefill
                                  └─ KV cache 算完后通过 cache_store / P2P
                                                       │
                                                       ▼
                              Decode 节点(DecodeRpcServer)
                                  │
                                  ├─ engine 只做 decode
                                  ├─ scheduler 是 BatchDecodeScheduler
                                  └─ output 直接 stream 回 frontend（或经 prefill 节点中转）
```

为什么不用一个 RpcServer 配 if/else 切换？三种模式的：

- **请求来源** 不同（frontend 直连 / 经其他节点）
- **engine 调度行为** 不同（mixed PD vs decode-only）
- **输出去向** 不同（stream 给谁 / 是否需要先经 prefill 中转）
- **生命周期** 不同（PD 下 decode 节点的 stream 是被远程 trigger 创建的）

每种都用独立类承载，比一个超级 server 里几十个 if-else 干净。本节只看 Local；Prefill / Decode 在 **B1.4** 详细对比。

---

## 5. 设计意图汇总（不再"为什么不是另一种写法"，而是直接解释）

### 5.1 为什么要 proto 类型 + 内部类型两套？

protoc 生成的 C++ 类是受协议格式约束的——字段只能是 scalar / repeated / 嵌套 message，没法塞 `torch::Tensor`、`shared_ptr<某复杂对象>`、`std::function`、甚至连默认值的语义都受限。但内部 backend 要做的事（持有 GPU tensor、注册回调、和调度器/sampler 互动）需要这些 C++ 类型。

所以 proto 类型只用作**线路上的数据格式**，进 C++ 后第一件事就是翻译成"自己的"内部对象。**这种"protocol model vs domain model 分离"是所有用 protobuf/Thrift 的系统的标准做法**，不是 RTP-LLM 独有。

### 5.2 为什么 `GenerateInput` 和 `GenerateStream` 拆开？

- `GenerateInput` 是**只读的输入快照**：一旦构造好就不再改（除了 prefix update 这种特殊情况）。它代表"这个请求要算什么"。
- `GenerateStream` 是**可变的运行时态**：每 step 都在变（状态机、cache 块、累积的输出）。它代表"这个请求正在被怎么算"。

如果合并到一个类，会出现"输入快照部分"和"运行时态部分"在同一个对象里互相污染，且无法多 stream 共享同一份 input（虽然现在还没有这种用法，但是设计预留）。

### 5.3 为什么状态机有 4 个状态（含 LOADING_CACHE）？

见 §2.4，核心是"已占资源但未可跑"是个真实状态，必须能区分。这个区分在调度器统计可用 KV blocks 和 executor 判断"该不该 forward 这个 stream"两处都用到。

### 5.4 为什么有 3 个 RpcServer 类？

见 §4，三种部署的请求来源/调度行为/输出路径都不同，分类继承比 if-else 干净。

---

## 6. 自测题（先自己回答，再做验收任务）

**Q1**：从 Python 调 `tokenizer.encode(prompt)` 开始，到 C++ `engine_->enqueue(input)` 这一行，经过了几次 "对象 → 对象" 的转换？分别是哪几次？

**Q2**：`engine_->enqueue(input)` 返回后，本函数局部变量 `input` 析构时，`GenerateInput` 对象真的被销毁吗？为什么？

**Q3**：假设客户端把 token_ids 数组传过来，C++ 这边在 `QueryConverter::transQuery` 里写了 `.clone()`。如果**漏写 `.clone()`** 会怎样？什么时候出问题？

**Q4**：`StreamState::LOADING_CACHE` 和 `WAITING` 在调度器视角下有什么差异？给一个具体场景。

**Q5**：如果有人想给 RPC 加一个新的运维接口（比如 `DumpAllStreams`），应该改哪些文件？给出 3 个最少需要动的文件。

> 答案后面会给参考。先自己想 5 分钟。

<details>
<summary>参考答案（点开看）</summary>

- **Q1**：3 次。(1) Python `GenerateInput` 对象 → Python `GenerateInputPB` proto；(2) proto 在线路上序列化为 bytes、对端反序列化为 C++ `GenerateInputPB`；(3) C++ `GenerateInputPB` → 内部 `GenerateInput`（QueryConverter::transQuery）。
- **Q2**：不会立即销毁。`engine_->enqueue(input)` 内部会让新建的 `GenerateStream` 持有 `input` 的 shared_ptr 副本，refcount 从 1 增到 2。本函数析构 input 时 refcount 减到 1，stream 还持有；stream 真正销毁时再减到 0，input 才被 delete。
- **Q3**：会出 use-after-free。proto 对象 `*input` 的生命周期是这个 RPC 函数的栈范围内（实际由 gRPC 框架在更上层管理），但 `torch::from_blob` 只是把 proto 的 RepeatedField 内存当 view 包起来——一旦 proto 销毁，这块内存就被回收，但 stream 内的 input_ids tensor 还指向它。**短测试可能侥幸不崩**（因为内存还没被覆盖），但稍微跑久点或在 stress 测试下就会 segfault 或数据错乱。这种 bug 极其难复现。
- **Q4**：差异在"是否占用 GPU KV cache 块"。WAITING = 0 块占用；LOADING_CACHE = N 块已分配但 KV 数据还在拷。调度器计算"还能新准入几个 request"时，free_blocks = total - (RUNNING + LOADING_CACHE)，不能把 LOADING_CACHE 算进可用。具体场景：PD 分离 decode 端，新 stream 从 prefill 节点拉 KV，拉完前是 LOADING_CACHE。
- **Q5**：最少 3 个：(1) `model_rpc_service.proto` 加 rpc 方法 + 加 message；(2) `LocalRpcServer.h` 加 handler 声明；(3) `LocalRpcServer.cc` 加 handler 实现。可能还要动 `LocalRpcServiceImpl.h` 把新方法注册进 gRPC service，但不一定（取决于继承结构）。

</details>

---

## 7. 验收任务（动手）

### 任务 A（必做，~15 min）：抽路径 + 答 Q

打开 `rtp_llm/cpp/model_rpc/LocalRpcServer.cc`，**就这一个文件**。不需要打开任何其他文件——上面所有需要的信息都在本课程文档里。

完成 3 小题：

**A.1** — 从 §1.4 我已经贴出的 `GenerateStreamCall` 完整代码（line 158-188），**抽出"主路径"（不含 profile / log / 错误处理装饰）**的 5-7 行核心代码，每行用中文写一句话注释，说明这一行的"业务意图"（而不是机械翻译代码）。

**A.2** — 把第 3 节给出的对象生命周期图自己**重画一遍**（可以更简洁），但必须明确标出：
- `input` (shared_ptr<GenerateInput>) 的 refcount 在什么时候变化（从 1 变到 2，再变回 1，再到 0）
- `stream` (shared_ptr<GenerateStream>) 的 refcount 在什么时候变化
- 哪个对象先死，哪个后死

**A.3** — 回答下面 2 个问题（基于本课程内容能直接推出，不要查代码）：

- **Q1**：如果 `pollStreamOutput` 因为 client 断连返回了 error，但 stream 还在 scheduler 队列里 RUNNING，会发生什么？谁会最终让 stream 析构？
- **Q2**：假设我们想加一个 metric "请求从入队到第一个 token 出来的延迟"（TTFT），最自然的埋点位置在 `GenerateStreamCall` 的哪两行之间？这两行分别是 §1.4 编号 [A]-[J] 中的哪两个？

把答案贴到 Section 8 / 9（下面）。

### 任务 B（选做，~30 min，需要编译）：加 log 看真行为

在 `LocalRpcServer.cc` 的 `GenerateStreamCall` 入口（line 162 `AtomicGuard` 之后、line 164 已有的 LOG_DEBUG 处）把 DEBUG 改成 INFO，并扩充信息：

```cpp
// 原来：
RTP_LLM_LOG_DEBUG("receive request %ld", request_id);

// 改为：
RTP_LLM_LOG_INFO("recv generate request, request_id=%ld, input_tokens=%d, max_new_tokens=%d",
                 request_id,
                 request->token_ids_size(),
                 request->generate_config().max_new_tokens());
```

然后：
1. `bazelisk build //rtp_llm/cpp/model_rpc:...` 看能否过编译（如果整个 target path 不对，找 `BUILD` 文件看正确 target 名）
2. 起一个本地服务、发一个请求（或跑一个 smoke test）
3. 在日志里找到这一行
4. **回答**：如果 client 用 `BatchGenerateCall` 发了一批 N=5 的请求，这行 log 出现几次？为什么？提示：看 §1.4 的 `BatchGenerateCall` 函数体（line 190-）和 `GenerateStreamCall` 是不是共享 prepareInput 路径。

### 评分

- **A 全做完且 Q 全对**：✅ 合格，可以走 B1.2
- **A 错 ≤ 1 题**：✅ 基本合格，我点几句就走
- **A 错 ≥ 2 题**：⚠️ 回炉，把对应代码段一起再看一遍
- **B 完成**：🌟 加分，有实操手感

---

## 8. 你的答案（学习者填）

### 任务 A.1 — 主路径 5-7 行 + 注释

```cpp
// 在这里贴你抽出来的代码 + 你写的注释
```

### 任务 A.2 — 重画的生命周期图

```
(在这里画图)
```

### 任务 A.3 — Q1

> 

### 任务 A.3 — Q2

> 

### 任务 B（选做）

> 

---

## 9. 讲师批注 + 错题本

> （讲师 review 学习者答案后填）

---

## 10. 沉淀

### 关键事实
- `engine_->enqueue(input)`（LocalRpcServer.cc:179）是协议世界 → 调度/计算世界的**唯一入口**（GenerateStream 路径）
- proto 和内部类型分两套是 protobuf 必然，不是 RTP-LLM 独有设计
- `StreamState` 4 值：WAITING / LOADING_CACHE / RUNNING / FINISHED；多出来的 LOADING_CACHE 区分"已占 GPU 块但未可跑"
- 三个 RpcServer 类对应三种部署模式
- `Stream ⊇ Input ⊇ Config`（包含关系，shared_ptr）
- `transQuery` 里 `.clone()` 必要，避免 use-after-free

### 真实工时
- 准备：__ min
- 学习者答题：__ min
- 讲师批改：__ min
- **总计**：__ min

### 卡点（如有）
> 

### 是否写入 memory？
- [ ] `engine_->enqueue` 这个入口 = LocalRpcServer.cc:179 → 值得（一次记住一辈子用）
- [ ] StreamState 4 值含义 → 值得
- [ ] proto/内部类型分离的设计原则 → 通识，不需要专门 memory
- [ ] 其他：

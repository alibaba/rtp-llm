# B1.1 — 入口与对象转换

> **目标**：读完这节，你能用大白话向同事讲清楚"一个生成请求从 Python 进到 C++ 引擎，中间到底变成了什么"。
> **时长**：1-1.5h（含动手）
> **前置**：会写 Python；C++ 看得懂语法（不要求会写）；知道 gRPC 是"一个跨进程调函数的协议"就够了

---

## §0 一句话讲完整件事

> 用户调 `pipeline.generate(prompt)` → Python 把 prompt 切成 token id → 装进一个 protobuf 消息 → 通过 gRPC 发到 C++ 引擎进程 → C++ 把消息翻译成自己的 C++ 对象（叫 `GenerateInput`）→ 引擎给它配一个"档案袋"（叫 `GenerateStream`，记录生成全过程状态）→ 这个档案袋同时被两方持有：RPC 那一边阻塞等结果，引擎主循环这一边一步步推进生成 → 每出一个 token，主循环把它写进档案袋，RPC 那边立刻拿到一个，stream 回 Python → 直到 EOS 或达到 max_tokens，档案袋销毁。

只记三句话也行：

1. 跨进程靠 **gRPC**，跨进程时数据形态是 **protobuf message**（命名带 PB 后缀）
2. 进了 C++ 立刻翻译成 **C++ 对象**，proto 这层"信封"扔掉
3. 翻译完最关键的一行是 `engine_->enqueue(input)`——它建档案袋、入调度队列、把 shared_ptr 同时给 handler 和 scheduler

剩下都是把这三句话展开。

---

## §1 故事版（不看代码先听一遍）

### 第一幕 · 你按下回车

你在 Python 里写：

```python
result = pipeline.generate("你好，介绍一下自己")
```

`pipeline` 拿到这串字符串，**先用 tokenizer 切成 token id 数组**（比如 `[123, 456, 789, ...]`），然后准备好：

- 一串 token id
- 一坨生成参数（`max_tokens=100, temperature=0.7, top_p=0.9, ...`）
- 一个 request_id（自动生成的）

但是 —— 这些东西在 Python 进程里，**模型在另一个进程**（C++ 进程，跑在 GPU 上）。中间隔着一道"国境线"。要把数据递过去，必须**装到信封里、贴邮票、走通道**。

### 第二幕 · 装信封寄出去

"信封"就是 **protobuf 消息**（简称 pb）。protobuf 是 Google 设计的二进制序列化协议，比 JSON 紧凑十倍、解析快十倍。字段定义写在 `.proto` 文件里，跑 `protoc` 一编译，C++ 和 Python 两边各自得到对得上的 class，两边能拿同一个 PB 类型互通。

"通道"就是 **gRPC**——可以理解为"用 HTTP/2 跑的远程过程调用框架"。客户端调一个像本地函数一样的接口，gRPC 帮你把参数序列化、寄过去、把返回值反序列化回来。

特别的是，这个请求用的是 **server-streaming RPC**：客户端发一个 request，server 可以**回一长串 response**（每个 token 一个 response）。这就是为什么你能看到模型像打字一样一个字一个字往外蹦——底层就是 gRPC 在一条流里 push 多个消息。

### 第三幕 · C++ 把信收下来，立刻"翻译"

C++ 进程那边有个 handler 等着，函数签名长这样（先不看细节）：

```cpp
GenerateStreamCall(input_pb, writer)
```

- `input_pb` 就是寄过来的 protobuf 消息
- `writer` 是一个"出口"，handler 往里写啥，gRPC 就 stream 啥回客户端（先当 stdout 用）

handler 第一件事：**把 pb 翻译成 C++ 自己的对象**，叫 `GenerateInput`。为什么不直接用 pb？因为 pb 是"运输容器"，字段是 proto 自己的类型（`RepeatedField<int32>` 之类），用起来别扭；C++ 引擎内部用的是 `torch::Tensor` 这种正经类型。所以一进门就翻译完，proto 这层信封就扔了。

翻译这步藏了**整节课唯一的"地雷"**，后面 §3.4 详讲。

### 第四幕 · 交给引擎，开两条独立时间线

翻译完，handler 立刻喊一声：

```cpp
auto stream = engine_->enqueue(input);
```

这一行是**整节课的灵魂**。它返回一个 `GenerateStream`——你可以理解为**这次请求的"档案袋"**，里面装着：

- 原始输入（input 那份不可变快照）
- 当前生成到哪一步、KV cache 占了哪些块、已经吐了哪些 token
- 一个"输出队列"（每生成一个 token 就 push 进去）

关键来了：这个档案袋**同时被两方拿着**（C++ 智能指针 `shared_ptr` 的"双持"）：

- **RPC handler**：拿着它，在一个 while 循环里调 `stream->nextOutput()` 阻塞拉 token，拉到就 `writer->Write()` 推回客户端
- **引擎主循环**：拿着它（其实是从 scheduler 的 queue 里拿的），每个 step 把它和别的 stream 一起 batch 起来 forward，forward 完调 `stream->update()` push 新 token 进去

这就解释了为什么这条链能"边生成边吐"——handler 那边永远在等下一个 token，引擎那边在自己的节奏里推进，**两条独立的时间线在档案袋上汇合**。

谁先撒手都不影响另一方：

- 客户端断了，handler 调 `stream->cancel()`，引擎下一个 step 看到 cancel 就把它从 running queue 拿掉
- 引擎让 stream finish 了，handler 下一次 `nextOutput()` 就会返回 finish 信号，跳出 while

最后一方释放 shared_ptr 时，档案袋才真正析构（释放 KV cache 块、清理状态）。

---

## §2 鸟瞰图：一张图记住整条链

```
   Python 进程                                    C++ 引擎进程
 ┌──────────────────────────┐               ┌────────────────────────────────┐
 │  pipeline.generate       │               │ LocalRpcServer::                │
 │    │                     │               │   GenerateStreamCall(pb, writer)│
 │    │ tokenize            │               │     │                           │
 │    ▼                     │               │     │ ① prepareInput            │
 │  ModelRpcClient          │               │     ▼                           │
 │    │                     │               │  QueryConverter::transQuery     │
 │    │ 把 dict 灌进         │               │     │  proto → C++ object       │
 │    │ GenerateInputPB     │   ★ gRPC ★   │     ▼                           │
 │    │                     │  ─────────►   │  GenerateInput (shared_ptr)     │
 │    │ stub.Generate-      │  HTTP/2       │     │                           │
 │    │ StreamCall(pb)      │  protobuf     │     │ ② ★ engine_->enqueue ★    │
 │    ▼                     │               │     ▼                           │
 │  阻塞 async for          │               │  GenerateStream (shared_ptr)    │
 │    │                     │               │    ├──► RPC handler 持一份      │
 │    │                     │               │    └──► Scheduler queue 持一份  │
 │    │ 每收一个 pb         │  ◄─────────   │       │                         │
 │    │ yield 一个 token    │  stream resp  │       │ ③ 引擎主循环每个 step：  │
 │    ▼                     │               │       │   • scheduler 挑 batch  │
 │  用户拿到 token          │               │       │   • executor forward    │
 │                          │               │       │   • stream->update(tok) │
 └──────────────────────────┘               │       ▼                         │
                                            │     handler 这边 nextOutput()   │
                                            │     立刻拿到这个 token          │
                                            └────────────────────────────────┘
```

**5 个关键名字（后面反复出现）：**

- `GenerateInputPB` — proto 消息（信封）
- `GenerateInput` — C++ 不可变快照
- `GenerateStream` — C++ 档案袋（runtime 状态，shared_ptr 双持）
- `LocalRpcServer` — C++ 这边的 gRPC handler
- `engine_->enqueue` — 灵魂 1 行

---

## §3 顺着走，每一步看一小段真代码

每一步只贴关键 5-10 行。**先讲它在干嘛、为什么需要，再贴代码，再点关键行。**

### 3.1 Python 入口（pipeline.py）—— 把字符串变成请求

**它在干嘛**：tokenize + 打包请求参数 + 把请求扔给 gRPC client。

文件：`rtp_llm/pipeline/pipeline.py`

```python
# 行 216：tokenize 字符串 → int 数组
token_ids = self.tokenizer.encode(prompt)

# 行 228：打包后调 generate_stream
return self.generate_stream(...)

# 行 466：generate_stream 函数体
async def generate_stream(self, ...):
    ...
    # 行 495：扔给 gRPC client
    await self.backend_rpc_server_visitor.enqueue(input)
```

**关键点**：

- `backend_rpc_server_visitor` 名字长，本质就是 **gRPC client 的封装**
- 那个 `enqueue(input)` 最终调到 C++ 那个 `GenerateStreamCall`
- **到这一行为止数据全在 Python 进程里**，下一行就要跨界了

### 3.2 跨界一刻：proto 长啥样

**它在干嘛**：定义"Python 和 C++ 两边都同意的合同"。

文件：`rtp_llm/cpp/model_rpc/proto/model_rpc_service.proto`

```proto
// 行 599 附近：服务定义
service RpcService {
  // 单条流式生成：发 1 个，收 N 个 token
  rpc GenerateStreamCall(GenerateInputPB) returns (stream GenerateOutputsPB);

  // 批量非流式：发 1 批，收 1 批最终结果
  rpc BatchGenerateCall(BatchGenerateInputPB) returns (BatchGenerateOutputsPB);
}

// 行 137 附近：请求消息
message GenerateInputPB {
  int64                request_id      = 1;
  repeated int32       token_ids       = 2;   // 数字数组
  GenerateConfigPB     generate_config = 3;
  // ... LoRA / multimodal / embedding ...
}
```

**用大白话**：

- `service RpcService { ... }` 定义 C++ 那边对外接口。`protoc` 编译后，C++ 侧自动得到一个抽象基类，你只要实现 `GenerateStreamCall` 方法
- `stream GenerateOutputsPB` 里 `stream` 关键字 = **server 流式**：一个 request 进来，server 可以连续回多条 response
- `int64 request_id = 1` 中 `= 1` 不是赋值，是 **field tag**（protobuf 二进制编码用它定位字段，**删字段时 tag 不能复用**，类似数据库主键）
- `repeated int32 token_ids` = "int32 数组"

**关键事实**：proto 是 **两边唯一同意的合同**。改字段必须改 proto + 跑 protoc + 两边都重编。

### 3.3 C++ handler 入口

**它在干嘛**：gRPC 收到请求后跑这个函数，函数返回前要把所有 token 流式 Write 出去。

文件：`rtp_llm/cpp/model_rpc/LocalRpcServer.cc`

```cpp
// 行 158
grpc::Status LocalRpcServer::GenerateStreamCall(
        grpc::ServerContext*                     server_context,
        const GenerateInputPB*                   input_pb,
        grpc::ServerWriter<GenerateOutputsPB>*   writer) {

    // [1] 一次性 request 上下文（埋点 / cancel 检测 / 异常捕获）
    GenerateContext generate_context(server_context, writer, metrics_reporter_);

    // [2] proto → C++ 对象
    auto input = prepareInput(input_pb, &generate_context);
    if (!input.ok()) return convertStatus(input.status());

    // [3] ★★★ 灵魂 1 行：交给 engine
    generate_context.setStream(engine_->enqueue(input.value()));

    // [4] engine 拒收（队列满 / 参数非法）
    if (generate_context.stream == nullptr) {
        return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, "...");
    }

    // [5] 边拉边写
    while (!generate_context.stream->finished()) {
        auto output = generate_context.stream->nextOutput();   // 阻塞等下一个 token
        GenerateOutputsPB pb;
        toProto(output, &pb);
        writer->Write(pb);                                     // 推回客户端
        if (server_context->IsCancelled()) {                   // 客户端断了
            generate_context.stream->cancel();
            break;
        }
    }
    return grpc::Status::OK;
}
```

**逐段大白话**：

- **函数签名**：handler 收到 `input_pb`，需要往 `writer` 写若干个 `GenerateOutputsPB`，最后返回 `grpc::Status`
- **[1]** `GenerateContext` 把"客户端 context + writer + 监控上报器"打包，方便后面埋点和异常处理
- **[2]** `prepareInput` 调下面 §3.4 的转换函数，把 pb 翻译成 `shared_ptr<GenerateInput>`。返回 `absl::StatusOr`，所以要校验 `ok()`
- **[3]** `engine_->enqueue` 是**核心 1 行**。这一行之前所有事在 handler 线程；这一行之后，stream 演进被甩给 engine 主循环线程。详见 §3.5
- **[4]** engine 队列满或输入非法（比如 token_num 超 max_seq_len），enqueue 返回 nullptr，立刻给客户端回 `RESOURCE_EXHAUSTED`
- **[5]** handler 在 while 里转，每次 `nextOutput()` 阻塞等下一个 token；拿到就序列化成 pb、`writer->Write` 出去；同时检查客户端断没断（断了调 `stream->cancel()`）

### 3.4 翻译：proto → C++ 对象（藏着唯一的"地雷"）

**它在干嘛**：把 pb 字段一个个抄进 C++ 对象，token_ids 这种数组要转成 `torch::Tensor`。

文件：`rtp_llm/cpp/model_rpc/QueryConverter.cc`

```cpp
// 行 108 附近
std::shared_ptr<GenerateInput> QueryConverter::transQuery(const GenerateInputPB* input_pb) {
    auto input = std::make_shared<GenerateInput>();

    // (a) 基础字段拷贝
    input->request_id      = input_pb->request_id();
    input->generate_config = transGenerateConfig(&input_pb->generate_config());

    // (b) ★★ 关键 3 行：proto int32 数组 → torch::Tensor
    auto token_ids = const_cast<int32_t*>(input_pb->token_ids().data());
    int  token_num = input_pb->token_ids_size();
    input->input_ids = torch::from_blob(token_ids, {token_num}, torch::kInt32).clone();
    //                                                                       ^^^^^^
    //  ☠ 没这个 .clone() 必出偶发 SIGSEGV

    // (c) LoRA / (d) embedding / (e) multimodal ... 同理
    return input;
}
```

**地雷在哪——`.clone()` 为什么必须有：**

`torch::from_blob(指针, shape, dtype)` 是 PyTorch 一个"零拷贝"接口。它建出来的 tensor **不复制底层数据**，只是借用你给它的那块内存。

问题来了：那块内存是 `input_pb->token_ids().data()`，也就是 pb 消息里的字段。**pb 的生命周期只到 RPC handler 返回为止**。一旦 handler 返回，pb 析构，那块内存就被释放或复用。但是 tensor 已经被你存进 `GenerateInput`、传给了 engine，可能在主循环里某个 step 才被访问——到那时候访问的是野指针，**use-after-free，偶发 SIGSEGV**（这种 bug 复现还很难，因为不一定每次都 crash）。

`.clone()` 把数据真正复制到新内存，从此 tensor 和 pb 解绑。**这是新人写转换代码 100% 会踩的坑**——以后改 transQuery 加新字段时，凡是涉及 tensor 的，必须自问"我现在指向的是 pb 内存还是独立内存"。

**另外提一句 `transGenerateConfig`**（同文件行 14 附近）：60+ 字段一个一个手抄。加一个生成参数得 **三处一起改**：proto + GenerateConfig.h + transGenerateConfig。漏改任意一处都会"参数静默失效"。

### 3.5 灵魂 1 行：`engine_->enqueue(input)`

**它在干嘛**：建档案袋 + 入队 + 返回共享指针。

回到 `LocalRpcServer.cc:179` 那一行。它内部干了三件事：

1. **建档案袋**：`new GenerateStream(input)`，把 input 这个不可变快照塞进去，初始化 `state_ = WAITING`，输出队列为空
2. **入队**：把 stream 推进 scheduler 的 `waiting_queue_`
3. **返回 shared_ptr**：把 stream 的 shared_ptr 返回给 handler。**从这一刻起，stream 被两方共同持有**——scheduler 的 queue 持一份，handler 局部变量持一份

为什么用 shared_ptr 而不是 unique_ptr？因为**谁先撒手都不能让 stream 立刻死**：

- 客户端断了，handler 撒手；但 engine 那一份还在，下一个 step 才能优雅 cancel
- engine 出错让 stream finish 了，scheduler 撒手；但 handler 那一份还在，让 handler 能读到 finish 状态、给客户端回完整结尾

**双持的代价**：你不能假设 stream 只有一份引用。改 stream 状态必须线程安全（内部用 mutex）。这是 RTP-LLM 这类异步服务最常见的并发模型。

### 3.6 handler 这边怎么拉 token

```cpp
while (!stream->finished()) {
    auto output = stream->nextOutput();   // 阻塞
    writer->Write(toProto(output));
}
```

`nextOutput()` 内部是个**条件变量**：output_queue 空 + stream 没 finish 时 wait；engine 那边调 `update()` push 进新 token 后 notify，nextOutput 醒来取走。

这就是为什么用户体验是"一字一字蹦"——每个 token 一出来就立刻通过 gRPC stream 推回 Python。

到这里 B1.1 的主链路就走完了。**剩下的事**（engine 主循环怎么转、scheduler 怎么挑 batch、executor 怎么 forward）是 B1.2 和 B2 的内容。

---

## §4 现在正式认识那 3 个 C++ 对象

前面反复出现的 `GenerateInput / GenerateConfig / GenerateStream`，现在贴一遍精简版定义。

### GenerateInput（请求快照，进了 engine 就不许改）

```cpp
// rtp_llm/cpp/engine_base/stream/GenerateTypes.h:16
class GenerateInput {
public:
    int64_t                          request_id;
    torch::Tensor                    input_ids;        // [seq_len]
    std::shared_ptr<GenerateConfig>  generate_config;
    // 可选：multimodal / lora / mm_features / tokens_embeddings ...
    int64_t batch_group_size = 0;
    int64_t batch_group_id   = -1;     // 同 group 强制同 step
};
```

**记忆点**：immutable。Engine 内部多线程读 input 不用加锁。

### GenerateConfig（生成参数容器）

```cpp
// rtp_llm/cpp/engine_base/stream/GenerateConfig.h:23
class GenerateConfig : public autil::legacy::Jsonizable {
public:
    int   max_new_tokens;
    int   num_beams;
    bool  do_sample;
    float top_p;
    int   top_k;
    float temperature;
    // ...
    bool  force_batch = false;
    // ...
};
```

**记忆点**：纯参数，可 JSON 序列化。`Jsonizable` 是 autil 库协议，给一组 register 宏，就能 toJson/fromJson。

### GenerateStream（runtime 档案袋）

```cpp
// rtp_llm/cpp/engine_base/stream/GenerateStream.h
using GenerateStreamPtr = std::shared_ptr<GenerateStream>;

class GenerateStream {
    std::shared_ptr<GenerateInput>  input_;        // 持原始快照
    StreamState                     state_;        // 见下面 enum
    std::vector<int>                output_tokens_;
    KVCacheBlockIds                 kv_blocks_;    // 占了哪几块 KV cache
    std::queue<GenerateOutput>      output_queue_;
    std::mutex                      mu_;           // 双持下必须有
public:
    GenerateOutput nextOutput();    // handler 调，阻塞
    void           update(...);     // executor 调，push 新 token
    bool           finished() const;
    void           cancel();
    StreamState    state() const;
    int64_t        batch_group_id() const;
};
```

### StreamState（4 个状态，不是 3 个）

```cpp
// rtp_llm/cpp/engine_base/stream/GenerateTypes.h:111
enum class StreamState {
    WAITING,         // 在 scheduler 的 waiting queue 里排队
    LOADING_CACHE,   // PD 分离时，decode 节点在拉 prefill 节点的 KV
    RUNNING,         // 进 running batch，每个 step 都会被 forward
    FINISHED,        // EOS / max_tokens 到了 / 被 cancel
};
```

合并部署时永远不会进 `LOADING_CACHE`，直接 WAITING → RUNNING。

---

## §5 三种 RpcServer = 三种部署形态

| 类 | 部署场景 | 干什么 |
|---|---|---|
| `LocalRpcServer`   | **合并部署**（一台机做 prefill + decode） | 上面讲的全部 |
| `PrefillRpcServer` | **PD 分离**：prefill 节点 | 只做到 prefill 完，KV 推给 decode 节点 |
| `DecodeRpcServer`  | **PD 分离**：decode 节点 | 接 KV（LOADING_CACHE），直接进 RUNNING |

三个类都编进同一个 binary，启动参数决定用谁。B1.4 会详细讲 PD 分离。

---

## §6 为什么这么设计（5 个 Q&A）

**Q1 为什么不让 RPC handler 直接 forward 模型？**
forward 要 batch + 调度 + KV 复用。每个 handler 各自 forward 就批不起来了，吞吐爆死。handler 只负责"丢 queue + 拉 output"，调度集中给 engine。

**Q2 为什么 GenerateInput 不可变？**
快照之后 engine 内部就能放心传引用、跨线程读，不用加锁。所有可变状态全部塞 GenerateStream。"不可变 input + 可变 stream"是经典并发设计。

**Q3 为什么 stream 要 shared_ptr 双持？**
handler 和 scheduler 谁先撒手都不能让 stream 立刻死，否则另一方要么读到野指针、要么阻塞永远不醒。

**Q4 为什么 server-streaming RPC？**
要 token-by-token 推。等全部 token 出齐再返回延迟会爆，UX 也丢了"打字效果"。

**Q5 为什么有 Local/Prefill/Decode 三个 RpcServer 类，不写成一个带 if-else？**
代码清晰 + 部署独立。每个 binary 只暴露它当下扮演角色对应的 server，少一类被误调的风险。

---

## §7 5 道自测题（先答再看答案）

<details><summary><b>Q1</b> pipeline.py 拿到 token_ids 之后，request 经过几次"对象形变"才进 engine？列出形变链。</summary>

`dict (Python)` → `GenerateInputPB (proto, Python)` → 跨进程 → `GenerateInputPB (proto, C++)` → `GenerateInput (C++ shared_ptr)` → `GenerateStream (C++ shared_ptr)`。**5 次形变 + 1 次跨进程**。
</details>

<details><summary><b>Q2</b> LocalRpcServer::GenerateStreamCall 整个函数的"灵魂 1 行"是哪行？为什么？</summary>

`generate_context.setStream(engine_->enqueue(input.value()))`（LocalRpcServer.cc:179）。它把同步的 RPC 调用切成两条独立时间线：handler 阻塞读 output，engine 在自己主循环推进 stream。这一行之前所有事在 handler 线程，之后 stream 演进就脱离 handler 了。
</details>

<details><summary><b>Q3</b> 为什么 transQuery 必须写 .clone()？省掉会怎样？</summary>

`torch::from_blob` 不拷贝底层数据，tensor 借用 `input_pb->token_ids()` 的内存。handler 返回后 pb 析构，tensor 就指野指针。engine 主循环后面访问 token_ids 时 use-after-free → 偶发 SIGSEGV。
</details>

<details><summary><b>Q4</b> StreamState 一共几个值？哪个是 PD 分离才有的？</summary>

4 个：`WAITING / LOADING_CACHE / RUNNING / FINISHED`。`LOADING_CACHE` 是 PD 分离 decode 节点在拉 prefill 节点 KV 的状态。
</details>

<details><summary><b>Q5</b> GenerateInput 和 GenerateStream 谁是 immutable，谁是 mutable？为什么这样分？</summary>

`GenerateInput` immutable，`GenerateStream` mutable。这样 engine 内部多线程读 input 不需要加锁；所有 mutation 集中在 stream 上，通过明确的 update() 接口改。
</details>

---

## §8 验收任务

**任务 A（必做，~30 min）** 在 `LocalRpcServer::GenerateStreamCall` 入口（line 158 函数体第一行）加一行 log：

```cpp
RTP_LLM_LOG_INFO("GenerateStreamCall received: request_id=%ld, token_num=%d",
                 input_pb->request_id(), input_pb->token_ids_size());
```

跑一遍 smoke test（用你熟的 Qwen2.5-0.5B smoke），grep 日志确认每个 request 都打了 1 行、`request_id` 和 `token_num` 都对。

完成后在 §9 填：

1. 改动 patch（贴 diff）
2. smoke 命令 + grep 结果片段
3. 你卡住的点（编译错？路径错？格式串错？没卡住就写"无"）

**任务 B（选做）** 同样位置加一段，统计这个 binary 启动以来收到的 request 总数（用 `std::atomic<uint64_t>` 计数）。要点：放在哪里、为什么用 atomic、log 怎么避免高频刷屏。

---

## §9 你的答案

_（开课时由学习者填）_

## §10 讲师批注 + 错题本

_（开课时由讲师填）_

## §11 沉淀

_（节末填关键事实 / 路径 / 坑）_

# Device 重构设计文档

目标：设计硬件实现分离的LLM推理逻辑，让rtp-llm可以扩展支持多种硬件后端。

## 抽象层级
1. 描述层: 描述层负责管理两部分信息：模型信息和请求信息。
2. 逻辑组装层: 这一层负责维护整个计算逻辑。具体来说，需要根据模型信息和当前请求，组装相应的计算流程，并调用合适的硬件op。
3. 硬件实现层: 负责实现业务无关的具体计算逻辑。在`DeviceOps.h`中抽象了框架算子的定义，每一种硬件需要实现相应的计算后端。

## op层级设计
不同硬件的op实现粒度会有所不同。比如，gpu对attention layer有gemm级别的实现，
而有的硬件可能在attention layer/ffn layer这一粒度上实现op。
因此op会有多个层级的设计，最顶层的layer会有一个默认实现，如果硬件支持细粒度op，可以从细粒度开始实现。
而如果硬件希望从粗粒度实现，也可以直接overwrite默认实现。

## KV cache 管理机制
kv cache的管理粒度为block级别。block的结构定义和实现由硬件层完成，
但是每个cache block对应的seq长度由`SEQ_SIZE_PER_BLOCK`环境变量决定。
`kv_cache_blocks` 和 `kv_cache_scales` 保存的均为int64类型指针，shape为[batch_size, block_num]，指针指向对应的block。
硬件还需要提供一个接口获取当前模型和系统的config下一个kv cache block的size，框架会统一分配和管理block，
在请求时根据seq长度将所需的kv cache block一并传入op中。

## 内存复用
llm中各个模块的计算逻辑是顺序进行，所以计算用到的同一块buffer可以在layer间复用。
在layer内部，细粒度实现之间（如连续的多个矩阵乘），也可能希望复用同一块buffer。
因此，我们在device层面设计一个buffer manager，op计算中所需要的内存均需要通过buffer manager分配。
buffer manager的申请接口参数除了基本的size信息之外，还要提供额外信息：
1. device/host
2. buffer 根据生命周期长短分为两类：
 - `SHORT`: 短生命周期，仅在一个独立的计算逻辑（如gemm）中使用，用完后马上释放
 - `LONG`: 长生命周期，可能会跨多个生命周期
3. buffer 预期的空间复杂度：
 - `CONST`: 常数空间复杂度
 - `LINEAR`: 线性复杂度
 - `QUADRATIC`: 平方复杂度
 - `UNKNOWN`: 默认选项，实现者可以不填
4. buffer 名称，用于标记复用的内存块

长短生命周期的buffer会从两个池子里进行复用，并且对于长生命周期的buffer，manager会记录其分配和释放的顺序，
可供打印timeline并计算碎片复用策略。

## 基础对象

### Buffer
Buffer 在这里只记录一块内存指针及where、type、shape这些metadata。Buffer本身不持有内存，也不负责内存释放。
BufferManager分配的内存放置在`std::shared_ptr<Buffer>`中，配备了自定义的deleter，在shared_ptr析构时标记释放。

### Allocator
Allocator 封装了硬件的基础内存分配逻辑。
每种device配套两个allocator，一个用来分配和释放device内存，另一个对应host内存。两个allocator可以使用同一个。


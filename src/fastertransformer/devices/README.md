# Device 重构设计文档

目标：设计硬件实现分离的LLM推理逻辑，让rtp-llm可以扩展支持多种硬件后端。

## 抽象层级：
1. 描述层: 描述层负责管理两部分信息：模型信息和请求信息。
2. 逻辑组装层: 这一层负责维护整个计算逻辑。具体来说，需要根据模型信息和当前请求，组装相应的计算流程，并调用合适的硬件op。
3. 硬件实现层: 负责实现业务无关的具体计算逻辑。在`DeviceOps.h`中抽象了框架算子的定义，每一种硬件需要实现相应的计算后端。

## KV cache
`kv_cache_blocks` 和 `kv_cache_scales` 保存的均为int64类型指针，shape为[batch_size, block_num]，
其中一个kv cache block的结构由硬件实现层自行定义，但是每个cache block对应的seq长度由`SEQ_SIZE_PER_BLOCK`环境变量决定。
硬件还需要实现分配和释放kv cache 逻辑，框架会对kv cache block做一定的管理

## Buffers
llm中各个模块的计算逻辑是顺序进行，所以计算用到的同一块buffer可以在layer间复用。
但是每种硬件需要的buffer又不完全相同，因此buffer分配逻辑也抽象为硬件op的一部分。
在`Buffers.h`中，仅提供最基础的Buffer虚接口，硬件需要自行实现Buffer结构和分配逻辑。


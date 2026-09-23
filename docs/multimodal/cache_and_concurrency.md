# 多模态迁移范围

参考仓库：/home/xieshui.yyx/RTP-LLM/github-opensource，feat/minimax_m3_0802，a2cb9b5e5c6。

## 保留的参考实现

- MMProcessEngine 内部异步线程池、准入计数、排队期限及请求所有权。普通 embedding 路径直接调用预处理与 MMScheduler；async_submit/get_embedding_result 使用线程池。
- VIT_CONCURRENCY、VIT_MAX_QUEUE_SIZE，以及 GPU/CPU embedding 缓存和独立行哈希缓存的字节预算。GPU/CPU 预算均为零时不共享在途计算。
- FeatureHash CPU/CUDA 实现、token span 工具，以及参考库的单个 multimodal_feature_hash 张量和协议版本 1。
- vit_app 内的 /mm_cache/keys 和 /mm_cache/metadata。
- gRPC pinned 张量解码使用参考库的 pbToPinnedTorch 接口。
- 连接超时重试 2 次；内部 OSS 地址转换按参考库实现执行。

当前分支已有的调度、预处理指标、构建边界和 RDMA 传输结构保留；不整体覆盖成参考分支。

## 撤回的自行扩展

- MMAsyncExecutor 类、缓存关闭时的额外去重、消费者取消到 MMScheduler 的传递、额外异步指标。
- 自定行哈希协议版本 2 及 repeated feature_hashes 表达。
- 自写 SelectVit gRPC 接入及配套 frontend/FlexLB generation 校验。frontend/FlexLB 恢复迁移前路由，因此此次不保留这套缓存感知路由接入。
- 自写 Barex pinned 接收池、多槽 pinned 拼接及 receive_memory/receive_pool_bytes/grpc_pinned_memory 配置。RDMA 恢复迁移前实现。
- 自加的下载共享超时预算和 MM_OSS_USE_GROUP_INTRANET 开关。
- 对应测试入口和独立 mm_cache_routes 包装模块。

Qwen3.5 专用的 NVDEC、视频预处理、ViT CUDA graph 及相关测试保留。迁移开始前两处 MoE 改动保留。

本轮回退改变了 protobuf 和 C++ 接口；旧构建产物不能作为当前源码的验证结果。压测保持暂停。

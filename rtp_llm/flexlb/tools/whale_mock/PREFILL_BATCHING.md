# NON_BATCH 输入的引擎侧自动组批

生产参照：模板 7 的 engine `5d478bb3c`，`FIFOScheduler::evaluateWaitingStreams` / `fitsPrefillTokenLimits`；master 保持 `96890a107`，不修改 real FLEXLB_CONFIG 或性能公式。

之前 Whale profile 的 direct_batch_size_max=1、max_batch_requests=1 实际关闭了已存在的合批。新 profile 使用 prefill.fifo，旧 case 未声明此对象时保持旧预算行为。

## 配置与执行

NON_BATCH 只表示 master 每次下发一个请求。P 空闲时立即执行；忙时请求排队，上批完成后扫描等待队列组下一批，不人为增加凑批等待时间。每批只调用一次现有性能公式，输入为本批所有请求，各请求分别保留完成、取消和 P→D 转交身份。

`glm53-inner-calibration.json` 的 prefill.fifo 声明所有批预算：

| 字段 | 值 | 含义 |
| --- | ---: | --- |
| max_requests | 64 | CONCURRENCY_LIMIT，即 max_generate_batch_size；包含边界 |
| max_batch_tokens | 512000 | 未命中部分的计算 token 总和；严格小于 |
| max_batch_kv_len | 3145728 | 包含命中前缀的完整逻辑 token 总和；严格小于 |
| max_seq_len | 1048576 | 首个请求的独立长度边界 |
| cp_size | 8 | TP=8；逐请求计算 token 补齐到 2×CP |
| force_single | false | CP_FORCE_SINGLE_PREFILL=0 |
| max_batch_tokens_without_cache | 0 | 另一个停止配额，0 表示关闭 |
| max_waiting_requests | 256 | 保留原测试队列容量，不随批大小相乘 |
| max_inited_kv_streams | 128 | 已初始化 P KV 的流上限 |

MAX_CONTEXT_BATCH_SIZE=1 仅用于未显式配置 max_batch_tokens_size 时的默认预算推导，不能当作请求数上限。这里已经显式配置了 512000。

选入候选前检查预算。装不下的请求留在原队列，继续查看后续较小请求；请求数限制不会越界。首个合法请求允许单独超过批 token 预算，避免大请求饿死。max_batch_kv_len=0 时改用完整 token 总和及 max_length×batch_size 的矩形预算。计算长度与完整 KV 长度不能混用。

## 容量准入与保真边界

执行预算不等于物理 KV 块池容量。新 FIFO 的 NON_BATCH 路径在候选选入时申请 P lease：重新匹配前缀，再检查本批预算、128 个 initialized-stream 限额，以及块池容量/水位。只有成功申请的请求占用本批预算。暂时申请不到的请求留队，后续较小请求仍可参加本批。P lease 释放或转入 LRU 后触发一次异步调度唤醒，不靠超时兜底或周期轮询。池子永久装不下的请求在入口直接拒绝。

排队请求不预占 P KV；D 的提前预留仍保留。取消与选入串行核对终态，排队取消不申请 P KV。FIFO 的块需求按完整 input_len 向上取整，不再拿可能未覆盖完整输入的 hash-key 数量代替容量。复用命中在执行选择时重新计算，保留原请求输出长度。

尚未模拟 chunked prefill、分层 KV 初始化和真实 GPU 执行；BATCH 入口的 P lease 时点保留原有行为。这里只对齐当前 Whale 的 NON_BATCH 自动组批及其 P 容量准入。

混合 BATCH/NON_BATCH 输入仍沿用已有 direct 队列优先级，未改成两种入口共享的全局 FIFO；当前 Whale 为纯 NON_BATCH，不受此限制影响。

## 验证

MockPrefillBatchPolicyTest 覆盖候选预检查、严格边界、命中/计算双预算、CP padding、singleton、矩形预算和请求数上限。DirectPrefillCoalescingTest 使用独立 GenerateStream 请求验证 [1]、[60,39]、[40] 的批次划分与全部收尾；5 块小池测试验证三个各需 4 块的请求依次执行（每批保留 1 块水位）、暂时缺容量不提前拒绝，以及排队取消后块池全部归还。其余原有直接组批、等待队列、取消和命中率测试一同回归。

部署验收必须另行核对实际 batch_size>1、completed 增量、等待队列、错误率及 Fetch RPC=0；编译或单测成功不能代替线上验收。

2026-09-14 远端 134 / Java 21 验证：租约 `20260914_025007.`，作业 `fifo-final`，43 tests、0 failures、0 errors、0 skipped，耗时 27.046 秒。使用 `opensource,!internal` profile；未改 master/生产引擎源码。

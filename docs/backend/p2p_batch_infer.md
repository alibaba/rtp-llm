# P2PConnector batch infer

## 实现

复用现有 HTTP `/batch_infer`、`BatchGenerateCall`、`EnqueueBatch`、`getScheduler().enqueueGroup()` 和 batch collector，不新增 RPC、生成配置或引擎接口。

`Frontend → Decode.BatchGenerateCall → Prefill.EnqueueBatch → P2P KV/sidechannel → Decode 分组推理 → 整批响应`

Frontend 为一批选择同一 PDFUSION 或同一组 P/D。Decode 复用单请求的参数准备、TP/CP 信息、取消注册；Master 已入队的子项通过 `master_enqueued_<request_id>` 接入。不适合 PD 的普通 batch（如单 token、beam search、多返回序列）整批交给 Prefill 原有 `BatchGenerateCall` 完整生成。

`return_prompt_logits` 沿用已有请求开关。DSv4.1 的 prompt scoring 只返回 1 token，关闭流式和 PD；此类 batch 走 Prefill 原有 `BatchGenerateCall` 回退路径。普通 P2P batch 仍返回完整多 token，分别测试两条路径。

sidechannel 保留评分交接能力，字段使用 `PromptLogitsOutputPB prompt_logits = 14`，保留 DSv4.1 的 `multimodal_lengths = 13`。复用 `TensorPbConvert` 传递 float32 top-k 分数、int32 token ID、可选 target 分数和位置范围。Prefill 保存独立 CPU 快照，Decode 可恢复到首步输出；该协议能力不改变上述 prompt scoring 入口限制。空切片仍允许没有评分。

部分入队、ACK 丢失、超时和取消，复用 `PDCancelRegistry` 关闭已尝试提交的生命周期。KV 资源释放仍受调度器和 P2P lease 保护。

## 与主干对照

本次适配基于 DSv4.1 固定基准 `d2675e02bab62c1154247cfb93bf4205a0e3f6bb`；来源提交原先对照 `origin/main` 的 `ea60e44019a1023b686b52b8eaab0fe0642fe247`。

保持 HTTP 格式、非流式整批返回、输入顺序、任一子项错误导致整批失败、逐项超时、调用方配置不变。合部继续使用原有实现。HTTP 为子项预留共享 ID 序列；仍依赖原有每毫秒 4096 序列的容量约束。

明确的差异和限制：

- PD batch 仅单 DP、同一 P/D 组合；调度器可拆分执行，不保证同一 GPU step。
- HTTP batch 通过服务发现固定 P/D，不做 Master 逐条 admission；可接入地址一致的已入队请求。
- 部分 Master 入队、同时含不适合 PD 的子项时拒绝，避免重复生成。普通 HTTP batch 共用生成配置，不产生这种混合。
- PD 失败会尽快取消其余子项，错误返回时机可能早于主干合部串行收集。
- 请求和服务均未配置正超时时，沿用分支的一小时回退；较新主干允许无限等待。

## 来源提交的验证与审阅记录

在 `11.163.39.111 / yzh / yanzhan.yzh`，经允许使用 CUDA 13、独立目录/缓存及 GPU 0。

完整通过：components 134、stream cache 53、P2P RPC 67、GPU KV smoke 13、路由 31 个用例；前端 17 个中 6 个按条件跳过。另两个 Python 套件按 batch 过滤，共 12 个用例通过。

GPU smoke 使用真实 KV 传输和模拟模型。实模型精度、完整 HTTP E2E、跨机 RDMA 未验证。完整 Python 套件的既有 gRPC 遥测用例报错/等待，已中断；不能视作全量通过。

审阅 commit 仅含本次增量，基于修改前工作区快照 `9bea68fbd1d6ea929a1b5d7bf0332a5e39e42ccd`，原 PR 分支和其他工作区改动保留。

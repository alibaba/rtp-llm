# P2P 阶段时延指标

所有时延单位为 μs。

## 入口

复用 `rtp_llm_rpc_prepare_generate_context_rt_us`（PD 准备）、`rtp_llm_rpc_remote_generate_rt_us`（Prefill RPC 发起）及 `rtp_llm_tp_sync_input_us`。

省略前缀 `rtp_llm_p2p_connector_`。

已有 `prefill_worker_store_store_wait_done_time_us`：store 到 event 就绪及发布 buffer；补异常漏报。

## Decode 编排

后缀加 `decode_schedule_`：

| 后缀 | 计时范围 |
|---|---|
| plan_cost_time_us | 计划、digest 和 routes |
| kickoff_queue_time_us | 提交任务到线程池开始执行 |
| server_submit_time_us | 发起 StartLoad，同步部分 |
| broadcast_submit_time_us | 发起 READ，同步部分 |
| server_call_cost_time_us | StartLoad RPC 及响应解析完成 |
| tp_sync_cost_time_us | 所有 READ 完成 |
| lease_query_time_us | 累计查询时间 |
| lease_hold_time_us | 保护到所有 rank 停止 |

`decode_schedule_cost_time_us` 计逻辑完成；保护完成另计。

## Prefill 编排

后缀加 `prefill_scheduler_`：

| 后缀 | 计时范围 |
|---|---|
| resource_register_time_us | addResource |
| check_plan_time_us | 会合前计划校验 |
| resource_wait_time_us | 等待并取得资源 |
| plan_cost_time_us | 计划及 routes |
| broadcast_submit_time_us | 发起 HANDLE_READ |
| broadcast_wait_time_us | 等广播、超时或取消 |
| side_channel_wait_time_us | 等待首 token/MTP |
| side_channel_fill_time_us | 填充响应 |
| process_read_time_us | StartLoad handler，含等待 GenerateStream 请求登记 |

## 数据面

后缀加 `prefill_worker_write_`：

| 后缀 | 计时范围 |
|---|---|
| add_buffer_time_us | 获取 computed buffer |
| dispatch_time_us | 逐层等待与分发 |
| sender_queue_time_us | 单个发送任务排队 |
| send_submit_time_us | sender.send 同步部分 |
| send_complete_time_us | 进入 send 到完成回调 |
| callback_wait_time_us | 分发后等待剩余回调 |

后缀加 `decode_worker_`：`prepare_time_us` 为注册 recv；`recv_wait_time_us` 是注册后等待；`recv_task_time_us` 为单 recv 到回调，含等 Prefill 数据。

## 口径修正

- Prefill 原 `first_layer_wait_time_us` / `last_layer_wait_time_us`：sendKVCache 进入到首个/全部待传 `(layer, tag)` 就绪。
- Decode 原 `first_layer_wait_time_us`：首个 `(layer, tag)` 的全部 route 成功完成才有样本。
- 未执行阶段不报；全命中无 READ/数据面样本；新增阶段保留有效 0 值。
- 单任务完成用 `success=true/false` 标签，不重复计请求 QPS；其余阶段包含失败路径。
- 阶段重叠，不能相加；不跨机器相减，也不加 request/key/layer/peer 标签。

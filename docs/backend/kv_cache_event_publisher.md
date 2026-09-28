# KV cache 事件上报

RTP 可将完整的可复用 HBM 前缀 key 上报给 KVCM。默认 `none` 不创建队列、线程或连接；`kvcm` 启用 reporter。

## 状态语义

事件使用逻辑 cache key。参与前缀复用的 DEVICE group 均完整且可读、没有未完成 transfer 时才发布；任一必需 group 失效会删除 key。重复 put 和 LRU touch 不产生事件。

推理线程只做有界非阻塞入队，网络请求在后台执行。队列溢出、请求失败、心跳失败、定期对账以及服务端 `snapshot_required` 会安排完整快照；publisher 失败不会阻塞推理、分配或驱逐。

当前 emitter 仅追踪完整 DEVICE 前缀链。虽然新版协议可以表达 FULL／LINEAR 组件，当前快照没有完整 tail-state／HOST／DISK 状态，所以 tail-sparse 和非 DEVICE 可复用组仍禁用。多个 FULL DEVICE group 聚合成一个 HBM spec。

只在 `pp_size=1`、`tp_rank=0`、非 CP 分片时发布。每个 DP replica 的 host identity 必须唯一；未填写时由 server IP 和 rank 端口派生。同一 host identity 的快照会替换该 reporter 的全部 medium 状态。

## 协议与生命周期

使用 `ST_EVENT_REPORT_L1P5`、medium `hbm` 和含聚合字节 size 的 `event_report://host/hbm?size=...` URI。Instance 显式注册为 prefix 模式。服务端必须配置对应的 L1P5 event storage，并将其加入 Instance Group 的 `event_report_storage_candidates`。

启动注册 Instance／node，提交完整快照，再发送 ADD／DELETE 和独立心跳。失败的快照 payload 保留用于重试。成功或失败响应的 `snapshot_required` 触发对账，`retry_after_ms` 推迟后续数据事件请求（默认最多 5 分钟），心跳独立刷新存活；快照限流保留注册状态，Instance／节点缺失和 leader 错误重新注册。停止时结束工作线程，再尽力发送最终 HOST_DOWN。

## 配置

参数有对应的大写环境变量。

| 参数 | 默认 | 语义 |
|---|---|---|
| `kv_cache_event_publisher_type` | `none` | `none`／`kvcm` |
| `kv_cache_event_manager_endpoint` | 空 | KVCM Meta HTTP endpoint |
| `kv_cache_event_instance_group` | 空 | 回退到 `kvcm_instance_group` |
| `kv_cache_event_instance_id` | 空 | 与注册配置一致的稳定 Instance ID |
| `kv_cache_event_host_ip_port` | 自动派生 | 每个 DP replica 唯一的稳定 endpoint |

配置非法只禁用 publisher。endpoint 须已解析；当前 reporter 不新增服务发现或自动 leader 切换策略。

配套制品、pickle 兼容和验收边界见 [KVCM remote cache](../kvcm_remote_cache.md)。

# P2P KV 生命周期收敛方案

## 1. 目标

Prefill TP 采用“rank 0 请求级持有、各 rank 逐层描述”的生命周期模型：

```text
rank 0: ResourceStore -> request KVCacheResource -> allocator connector ref
rank 0..N: writeByLayer -> LayerCacheBuffer -> 本 rank block 描述
```

`P2PConnectorResourceStore` 仅在 Prefill rank 0 初始化，负责 `StartLoad`、请求级 KV 引用和 side channel。逐层链路不持有 allocator 引用。

## 2. 所有权

各 rank 拥有独立 KV 显存池；相同 `block_id` 表示各本地池中的相同槽位。rank 0 分配槽位，`tpSyncModelInputs()` 将编号广播给其他 rank，只有 rank 0 allocator 管理占用和复用。

`addResource()` 保存 `incrKVCacheRef()` 产生的 `KVCacheResourcePtr`。`StartLoad` 后 entry 转交 `processRead()`，覆盖 `broadcastPerRank()` 及所有 worker 返回，最后析构并降低引用计数。

各 rank 的 `LayerCacheBuffer` 只保存 block 标识；event 和异步发送状态由逐层队列管理。持有链为：

```text
StoreWaitContext -> ComputedLayerCacheBufferStore
-> AsyncSendTaskState -> sender request
```

该链路保证描述和异步任务存活；rank 0 请求级引用防止 block 被复用。

## 3. 修改

### ResourceStore 与 Prefill

- 仅 `tp_rank == 0` 创建 ResourceStore 和 Scheduler；所有 rank 仍创建 Worker。
- `registerResource()`、side-channel、`processRead()` 及 ResourceStore 终态仅在 rank 0 执行。
- 保留 `P2PConnectorResourceEntry::kv_cache_resource`。
- 非零 rank 路径删除 ResourceStore 清理调用。

### 逐层链路

- `writeP2PLayer()` 不再调用 `incrKVCacheRef()`；所有 rank 只创建当前层的地址描述。
- `LayerCacheBuffer` 删除 `KVCacheResourcePtr` 及 `setKVCacheResource()`；仅保留发送必需的 layer/tag/block 映射。
- `P2PConnectorAcceptedWriteContext` 不再持有 `KVCacheResourcePtr`，只表达逐层任务已被接收。
- 现有逐层队列和异步任务继续管理描述对象。
- `sendKVCache()` 删除 resource 参数和整请求转换。`processRead()` 的 `resource_entry` 覆盖 Scheduler 等待所有 rank 返回的同步调用。
- route 校验失败时调用 worker `cancelRequest()`：删除该 rank 的逐层描述并记录 `request_id` 终态，拒绝晚到 layer 和 StartLoad，避免重新创建空 buffer 等待。

## 4. 释放边界

- 正常路径：所有 rank 的 `HANDLE_READ` 返回，且各 worker 已完成发送 callback 后，才允许释放请求级引用。
- StartLoad 未到达：ResourceStore 按资源等待期限回收 entry。
- no-transfer：所有 rank 完成 no-transfer 处理后释放。
- 取消或超时的物理传输 lease 由独立方案处理，不属于本次修改。

## 5. 验证

- 验证只有 Prefill rank 0 访问 ResourceStore。
- 验证 rank 1..N 不调用 `incrKVCacheRef()`，仍能生成本地地址。
- 单测覆盖正常发送、StartLoad 未到达、no-transfer、空 route 和 cancel 后晚到请求。

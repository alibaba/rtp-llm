# 已知问题与验收边界

## Master HA：batch-window 切换期间的请求错误

CI run `78808193`，内层提交 `42c1e7065d2457dd7de336ad65d4c280aa8a97a7`，
`master_ha_failover::default::batch-window`：切换窗口 1976 个请求中 1 个非 OK，
全程 9311 个请求中 2 个非 OK；恢复窗口 `late_errors=0`，A/B survivor target share 均为 1.0。
切换窗口属于全程，两个计数不能相加成 3 个错误。

`handover_errors` 和 `rolling_errors` 的门槛仍为 0；场景明确将 batch-window
配置为 advisory，因此两项为 WARNING，最终 gate 为 PASS。single-nonbatch 没有该豁免。
这证明当前可用性和恢复合同通过，不证明切换全程零错误，也不证明每一时刻仅 Leader 处理请求。

后续处理：关联客户端重试与 Master 的 enqueue/fetch 上下文生命周期，定位切换时的
NOT_FOUND / active-context 重复拒绝；补针对性回归，再决定是否移除 advisory。
未定位根因前不降低阈值、不将 WARNING 描述为零错误，不通过增大重试预算掩盖问题。

[完整实验及六份交互报告](https://alidocs.dingtalk.com/i/nodes/r1R7q3QmWew5lo02f6rjrXLAJxkXOEP2)。
后续运行应独立记录版本、窗口、错误计数和清理状态，不覆盖本轮证据。

## CodeQL：临时端口探测的 wildcard bind

`runtime.network.port_in_use` 为启动前占用检测。JVM 可能绑定全网卡，因此只探测
loopback 会漏掉其他接口上的冲突。探测 socket 不调用 listen/accept、不读取或发送
数据，并在 with 退出时立即关闭；它不创建对外服务。

在 bind 行使用 `py/bind-socket-all-network-interfaces` 的单行告警说明，
仅适用于上述探测，不适用于 HTTP/gRPC 服务的监听地址。服务暴露范围应由各服务
启动配置审查；不能把本说明扩展为允许所有服务无条件监听全网卡。

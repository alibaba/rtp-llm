# PD Decode 入口与同版本发布说明

本版本统一采用 Decode 入口：Frontend 把请求发到 Decode；Prefill 计算首 token，并通过 StartLoad 交付首 token、采样状态和 KV 传输信息；后续生成和流式输出由 Decode 完成。`DECODE_ENTRANCE` / `--decode_entrance` 已移除，不再用于切换入口。

## 发布契约

Frontend、FlexLB master、Prefill、Decode 必须来自同一发布版本，使用配套的外源 commit、内源 commit 和构建产物。不能把“服务启动成功”当作协议兼容证明。

本次切换不支持这四部分跨新旧版本混跑：

- 旧 P→D 的 RemoteGenerate / RemoteLoad / RemoteFinish 不再承载请求。
- 新请求从 Decode.GenerateStreamCall 获取输出；旧 EnqueueGroup / FetchResponse 不提供过渡实现。
- 新 Decode 要求 Prefill 的 GetPeerInfo / StartLoad 与当前分组 KV 布局一致。KV 传输、远端缓存和 host-cache 对端仍须遵守同构建协议契约。

当前没有自动版本握手或混版本降级。版本配套和路由分组由发布编排保证，不能依赖运行时自动识别并修复误配。

## 切换步骤

1. 记录同一轮完整功能 CI 使用的内外源 commit，准备该版本配套的四类产物。CI 未通过前，不视为可发布。
2. 建立独立的新版本路由组，配置正确的 Prefill、Decode 和前端地址。旧版和新版不能共享会随机选取两代实例的路由池；新组未齐备时不接流量。
3. 核对各角色实际版本、TP/CP、模型及 KV 布局。验证新组的普通请求、Master 预入队、流式取消、TCP/RDMA 传输和真实 smoke 输出。
4. 流量按完整路由组切换到新版本，确保每个请求的 FE→Master→D→P 都留在同一版本组。禁止逐个替换在线 P/D、让另一侧继续使用旧协议。
5. 旧组停止接收新请求后，等待在途请求、物理传输和资源持有全部结束，再回收旧实例。取消 ACK、RPC 超时或 Decode 本地终态都不能单独证明 Prefill 侧也已清理完成。

新 Decode 发布取消终态的条件是本地生成结束、活动读取结束且 P2P 资源持有已释放；Prefill 不可达时，远端取消记录独立重试，并在本地安全终态后按现有 10 分钟保留期退出。到期只删除轻量控制记录，不把 Prefill 标为已清理；本地读取或资源持有未结束时不适用该回收规则。不得据此强制释放仍在传输的 KV。

取消注册表对活跃记录、保留记录和提前取消栅栏设共享上限 1,048,576 条；满载时新增请求返回并发容量错误，无法存入新取消栅栏时 Cancel 返回 gRPC RESOURCE_EXHAUSTED。已有请求的取消和重复取消仍可执行。到期清理按批推进，不淘汰尚未完成本地物理清理的请求。

## 回滚

保留旧版本完整路由组。需要回滚时将流量切回该组，先排空新组再回收；不能只回滚其中一个角色。新旧组使用独立 KV 资源，不跨协议版本复用在途 KV 或缓存状态。

本文说明发布要求，不表示上述 CI 或部署验证已经完成；以对应准确 commit 的完整 CI 和部署证据为准。

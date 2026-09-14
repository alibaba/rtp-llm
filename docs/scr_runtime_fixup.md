# SCR 服务前模板的运行时修正

适用范围：单 Pod 内的 ranks 共同 dump 服务前模板，Prefill/Decode 从模板恢复，FlexLB 持续运行。每个 role 使用固定的服务发现域名。SCR 控制面负责 checkpoint/restore，RTP-LLM 负责资源注册、同步屏障，以及屏障返回后的本地初始化。

## Checkpoint 边界

- Backend：准备模型、KV 内存和 executor → checkpoint → 启动 engine loop、CacheStore 和 RPC/HTTP 服务。
- Frontend/Dash：创建客户端对象和请求处理器 → checkpoint → 开启监听 → 请求首次创建业务 gRPC channel。
- Master 服务发现：模板构造阶段不查询 VIP、不启动探测线程；release 后按原启动流程首次启动。
- Grammar：模板构造阶段不启动 sandbox 子进程；首次 grammar 校验创建 worker pool。
- Python/native Kmonitor：保留指标注册，外部上报和 Python 报告线程延迟到 release 后首次启动。

因此，模板不包含旧的业务 channel、VIP 实例缓存、master 探测记录、grammar worker 或 Kmonitor sink。不需要对它们做关闭、缓存清理、线程重启等恢复操作。同一模板重复恢复仍从这些资源未启动的状态开始。

## 必要的 fixup 与统一接口

checkpoint 前已经保存的 Pod IP、请求来源身份、日志/监控标签和 worker 地址仍须修正。流程为：

```text
prepare → Epsilon 同步屏障 → 读取本轮环境/发现 Pod IP
        → Logger/Hippo 身份更新 → 各组件 restore_fixup(context)
        → release → 首次启动延迟的资源与服务
```

`RestoreContext` 提供 `generation`、本轮 `pod_ip`、每轮惰性读取一次的 `endpoint_manifest` 和共用的 `resolve_world_info(...)` 校验。相同模板多次恢复也创建新 context，manifest 缺失时同一轮不重复读取。

只修正状态的组件实现 `restore_fixup(context)`，使用 `TemplateLifecycle.register_fixup(name, component)` 注册；仅需在 release 启动的组件使用 `CallbackHook(release=...)`。注册在 prepare 前完成，全部 fixup 成功后才能 release。屏障或 fixup 失败时不启动服务或 reporter。

当前参与者：

| 组件 | 屏障返回后处理 |
| --- | --- |
| Runtime identity | 读取当前环境，更新 RequestedIP、Logger 和 Hippo 身份 |
| ServerConfig | 更新已保存的 `ip` |
| BackendManager | 更新 Python/native 保存的 worker 广播地址，供首次 CacheStore 初始化使用 |
| Backend visitor | 更新本机 RPC 地址列表和请求来源 IP |
| DashScInferenceServicer | 更新请求 ID 生成器使用的 IP |
| Kmonitor | 以本轮身份首次启动 reporter，保留之前注册的指标句柄 |
| MasterService | release 后首次启动服务发现和探测线程 |

## 单 Pod 地址与服务发现

`RTPLLM_ENABLE_SCR` 开启、`SCR_PHASE` 为 `checkpoint`/`restore`、`WORLD_SIZE == LOCAL_WORLD_SIZE > 0` 且单节点时，TCPStore、NCCL rendezvous、rank 注册和本机 RPC 自动使用 loopback。跨 Pod 的 KV 广播地址使用本轮真实 Pod IP；这是单机模板的必要地址转换。

各 role 的服务发现域名不变，`MODEL_SERVICE_CONFIG` 保持启动时配置。平台在原域名下发布恢复后的实例，RTP-LLM 首次发现时获取实例列表。restore 不重写路由配置，也不增加 worker 到 FlexLB 的注册 RPC。

FlexLB 沿用原有服务发现和 worker 管理流程，本次 SCR 接入不修改其 cache 索引和版本处理逻辑。

单 Pod P/D 的 CacheStore 和外部 RPC 尚未初始化，因此不要求提供额外 endpoint manifest 或 `transport.ready` 来证明这些连接已重建。已有多节点 manifest/readiness 校验继续保留；这不代表多节点 GPU collective 恢复已验证。RTP-LLM 不重写 NCCL/GLOO/LD_PRELOAD，不注入 SCR 通信库，也不增加 transport components/rebuilt 协议。

若显式提供已有 endpoint manifest，仍校验 generation、rank 拓扑和端口。模板保存的本 rank 监听端口不能只靠广播地址变更；本地端口布局变化时拒绝发布，避免连接到未监听的端口。

## 恢复环境输入

每次屏障成功返回后读取 `/etc/scr/envs.json`，接受 JSON 对象，例如：

```json
{
  "RequestedIP": "192.0.2.20",
  "HIPPO_SLAVE_IP": "192.0.2.200",
  "HIPPO_ROLE": "restored-role",
  "HIPPO_ROLE_SHORT_NAME": null
}
```

支持 RequestedIP、HIPPO_SLAVE_IP、HIPPO_ROLE、HIPPO_ROLE_SHORT_NAME、HIPPO_APP、HIPPO_SERVICE_NAME、kmonitorSinkAddress、kmonitorPort。`null` 删除 seed 值；未出现字段保留原值。完整环境文件中的其他键被忽略，包括 `MODEL_SERVICE_CONFIG`。

文件不存在时，重新发现当前 Pod IP；显式提供 RequestedIP 时使用该值。Pod IP 必须为非 loopback、非 unspecified、非 multicast 的 IPv4。文件损坏、允许字段非法或读取失败时阻止 release。没有提供 HIPPO_SLAVE_IP 时，无法从 Pod IP 推断宿主机 IP，保留原值并记录其新鲜度未验证。

平台负责原子发布目标容器本轮文件；重新读取文件不代替文件新鲜度保证。显式 `restore_env` 或自定义 provider 仍可用于已有接入和测试，优先级为显式参数 → provider → 默认文件。

## 验证与打包

UT 覆盖模板边界无业务连接/发现线程、release 后首次启动、grammar worker 延迟创建、监控首次启动及旧指标句柄身份更新、单 Pod 无 manifest 的恢复、loopback/KV 地址分离、环境读取、失败阻止 release。

Python/native 必须一起打包。ARM wheel 目标为 `//rtp_llm:rtp_llm_aarch64`，配置为 `--config=cuda13_arm`。

单元测试和本机构建不能代替真实 SCR/GPU 验收。打包后验证：模板完成 dump；P/D 从模板恢复；原 role 域名发现新实例；日志/请求 ID/监控身份正确；实际 P/D 请求完成 KV 传输。FlexLB JVM 自身恢复和运行中进程的再次 dump 不在本轮范围。

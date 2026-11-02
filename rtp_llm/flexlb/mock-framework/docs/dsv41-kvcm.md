# DSV4.1、KVCM 与 Mock Engine 接线

本分支基于 DSV4.1 `df6feebad330f3a0b5d954108f3e6ceb5869d9b1`，保留其 Frontend、ViT 路由、PD 协议和 Engine 实现。Mock 框架基线为 `096780512b12571686e90202c5c1e344edc9719e`。KVCM MetaService 查询契约取自 MU `5bd6e6cdb3e017427ee776222c07b952f31e8deb`，没有整体替换 MU Master 或导入其调度体系。

## 模式和职责

`FLEXLB_CONFIG.cacheMatching.type` 缺省为 `LOCAL_SYNC`。`KVCM` 模式在 DSV 现有 CacheAwareService 查询入口调用 MetaService；Frontend 提供的 key 原样传入，返回的 local 连续命中 block 数继续由 DSV 转成 token 数。限定当前活跃候选地址，忽略其他部署、已离线或无法对应的 host。单 Engine 接受对应的 `@0` 身份，不将其他 engine index 当成同一实例。

KVCM 查询失败时回退到原有同步 Engine 缓存索引，日志明确记录回退；失败延迟也记录到已有缓存查询指标。这里复用真实缓存索引，不新增 Local Standby 预测索引、管理接口或后台状态机。配置在启动时固定，更改模式需重启 Master。所有主备 Master 使用相同模式。

只查询本地 HBM residency；不请求 global/p2p 估算，不启用远端 KV payload 加载。MetaService 字段 7/8 不发送，以避免两种已知 KVCM schema 中这些字段的含义差异。返回 global 命中不会被当作本地命中。非 Full Attention、多个逻辑 engine 共用一个端点等契约尚不在本批支持范围。

KVCM 服务 bootstrap 使用 `MODEL_SERVICE_CONFIG.kvcm.address/port`，通过现有 ServiceDiscovery 获取地址；无发现结果时按 DNS/IP 地址连接。通过原生 GetClusterInfo 获取 leader，再调用 GetHostCacheState，共用一次查询的 deadline。错误使下一次查询重新发现 leader，旧 channel 会清理；不新增发现服务或定时器。配置 namespace 必填，不从不确定的部署名推断。

## 同一镜像的开关两套部署

[开启配置](../deployments/kvcm-on.json)和[关闭配置](../deployments/kvcm-off.json)列出每个 role 的 ENV。它们是供平台配置的环境变量清单，不是新的启动协议，也不由启动脚本解码。将各 role 的配置分别放到 Whale 对应角色，沿用现有 real 镜像、发现、探针和启动方式。Master 单独部署；切换 Master 不重启 Engine/KVCM。

部署前需要替换示例中的发现域名、Meta HTTP 地址、模型目录和资源值，确认实际实例组存在且允许原生注册。KVCM gRPC bootstrap 与事件 HTTP 是两种不同端口，不能互换。示例不创建或修改 Whale 部署。

### Engine 事件上报的服务发现

与本分支 real Engine 使用相同的原生属性和 ENV 绑定：`kvcm_enable_vipserver` 对应 `RECO_ENABLE_VIPSERVER`，`kvcm_vipserver_domain` 对应 `RECO_VIPSERVER_DOMAIN`，`kvcm_server_address` 对应 `RECO_SERVER_ADDRESS`。Case 的配置使用原生属性名；Whale 使用这些 ENV，不另造 Mock 发现参数。

没有 `KV_CACHE_EVENT_MANAGER_ENDPOINT` 覆盖时，通过 VIPServer SDK 获取健康 RPC 端点，或使用 `RECO_SERVER_ADDRESS` 的 RPC 地址，然后调用原生 `GetClusterInfo(instance_id)`，校验 leader 身份并读取其 `meta_http_port`。VIP 域名不交给系统 DNS 解析。VIP 注册端口必须是 KVCM Meta RPC 端口（默认 6381），HTTP 事件发送到服务返回的 Meta HTTP 端口（默认 6382），不是 Admin HTTP 6492。VIP 探针使用 KVCM 原生健康入口，与注册端口分别配置。

发现与上报由 `shared/cache` 的 KvcmDiscovery/KvcmReporter 负责，RTP 只映射 native 配置和装配。按候选列表逐个查询 bootstrap，单个地址失败不掩盖后续可用地址。成功的 leader 一直缓存，不做每秒刷新；HTTP 传输失败或 SERVER_NOT_LEADER 后使对应地址失效，再通过原有上报重试恢复注册和全量快照。旧地址的迟到失败不会清除新地址，不发送到猜测地址。

网络查询不占用 Scheduler/Compute 或缓存锁。VIP 查询有等待上限，关闭可取消 VIP 查询与 gRPC；不持有覆盖网络调用的关闭锁。正常关闭仅向已缓存的 endpoint 尝试 HOST_DOWN，不为退出重新发现；Reporter 统一释放 resolver 和传输资源。没有新增发现进程、定时器或模块。原生 HTTP 直连覆盖仍优先，适用于本地测试或明确提供的稳定服务地址；Whale 示例使用发现，不写 Pod IP。

内部镜像复用已有 `vipserver` 组件，通过 Java 标准 ServiceLoader 接入，不复制 SDK 或增加发现进程。`package-engine.py` 在内部源码目录存在时启用 `internal-discovery` Maven profile，并检查最终 JAR 的 SDK、provider 类和服务声明；普通本地构建不需要内部依赖。开源包启用 VIP 配置会明确拒绝启动，不静默改用 DNS。

- 开启：Master `KVCM`；P/D `KV_CACHE_EVENT_PUBLISHER_TYPE=kvcm`。
- 关闭：Master `LOCAL_SYNC`；P/D `KV_CACHE_EVENT_PUBLISHER_TYPE=none`、`ENABLE_REMOTE_CACHE=false`，不注册也不发送控制心跳。KVCM 服务可以不部署；无效的 KVCM topology 不应触发连接。
- Frontend 和 ViT 不增加 KVCM 开关，保持两组相同。ViT 默认执行图片读取，Whale 不使用本地假 URL 跳过模式。
- 两组的模型、输入、seed、block size、资源、流量和本地缓存配置一致。示例关闭 Host cache，因为当前事件上报仅覆盖 HBM，而 LOCAL_SYNC 还能看到 Host cache；不把不同介质可见范围当成查询模式差异。

## 缓存空间与 Engine 身份

P/D 同一缓存空间使用同一个 `KV_CACHE_EVENT_INSTANCE_ID`，必须等于 Master 查询的 `<namespace>_<cacheKeyBlockSize>`，示例为 `dsv41-kvcm_128`。每个 Engine 在该空间内以 host 区分。删除原来自动生成的每节点 UUID instance 默认值；启用注册时没有显式 instance_id 会启动失败。

默认事件 host 与 DSV WorkerAddressService 一致，为发布的 `IP:(gRPC端口-1)`；Whale 标准 START_PORT=8088 对应身份 IP:8088、gRPC IP:8089。它不是任意控制 API 的地址。非标准发现可以显式设置 `KV_CACHE_EVENT_HOST_IP_PORT`，但必须与 Master 看到的身份一致。实际 Pod IP 由平台/发布地址提供，不编造独立指标 tag。

共享 CacheMetadata 是 Engine 本地缓存事实的唯一来源；Master RPC 与 Reporter 消费同一缓存状态。Reporter 的全量快照附带 key_count，包括空快照；计算和调度不等待上报。更换 Master 不改变 Engine 缓存空间或事件 host。

## Frontend、Client 与 Engine key

真实 Frontend 保留 DSV 的 canonical token + 多模态 cache span 算法。Mock Client 与 Engine 共用 CacheSequence：先滚入每个 block 的语义 token，再滚入该 block 的图片区间身份。完整 block 才作为可复用 key；不足一块不虚构 key。Master/KVCM 不从 hash 反推输入或重新生成另一套 key。

Mock Client 配置 `seq_size_per_block`，与 P/D 一致；不配置时不声称支持缓存路由。Client 和 P/D 使用相同 `checkpoint_path` 和 `MODEL_TYPE`。DSV4.1 的模型 namespace seed 与真实代码一致：full 为 `0x4453563431494631`，bounded 为 `0x4453563431494231`，由相同的 `DSV41_SWA_BOUNDED_REPLAY` 启动环境决定。DSV4.1 按真实模型初始化规则计算 seed；其他模型的 config.json 显式提供 `cache_key_hash_seed` 时使用该值。effective-config 输出最终 seed 便于核对。这里只对齐 key namespace，不宣称实现 DSV 的物理混合池或 bounded replay 计算。

本批原生端到端配置使用 CP=1、单 DP。真实 Frontend 的 Page-RR 是抽取已有物理 block key，再使用对应虚拟 block 粒度；不能把 block size 放大后重新哈希。跨语言测试覆盖抽样规则，但尚未证明 Mock 的 CP>1/KVCM 注册、虚拟 block 空间和容量语义闭环，不将它列为可用部署形态。

## 验证范围

- Java 测试覆盖原生 leader 查询、signed int64 key 原样传输、namespace、候选过滤、local/global 区分、故障重发现、开关和回退，以及原有 ViT/PD 路径。
- `tests/test_cache_key_contract.py` 用本分支真实 Frontend 函数和 C++ HashUtil.h 对照 Java CacheSequence，覆盖纯文本、负数、图片跨 block、图片变化、多图片、末尾不足 block、DSV full/bounded seed。
- `test_master.py` 的 `kvcm-cold-warm` 使用独立真实 DSV Master、Mock V/P/D 和原生协议 Meta 测试进程，Engine 通过原生 bootstrap RPC 发现 Meta HTTP 地址，检查首次请求产出缓存、HTTP 事件进入查询空间、随后 gRPC 查询命中并完成多模态生成；`vit-local-sync` 使用相同输入验证关闭模式。Meta 测试进程不能代替真实 KVCM 的部署验收。

Whale 首次验收还必须核对实际镜像 SHA、发现身份、Meta 服务版本、有效配置、真实 ViT URL、请求结果及实际缓存命中。仅有请求成功不表示 KVCM 在生效，回退成功也不能当作 KVCM 查询成功。

### 本地验证结果

缓存 key 的跨语言验证覆盖真实 Frontend/C++ 与 Java；发现、缓存、上报及 Client/ViT 的 47 个 Java 定向测试通过。独立进程的普通 LOCAL_SYNC、ViT LOCAL_SYNC、ViT KVCM 场景均通过。ViT KVCM 场景完成两次请求，Meta 测试进程收到 2 次查询、2 次正命中；关闭场景不启动 Meta 服务，两次生成同样完成。两种多模态场景使用同一输入、模型设置和模拟计算参数。这些结果是本地协议与功能验证，不代表线上 KVCM、真实图片下载或容量验证。

构建框架后，可在框架目录运行：

```sh
uv run pytest tests/test_cache_key_contract.py -q
uv run pytest tests/test_master.py -k 'kvcm-cold-warm or vit-local-sync' -q
```

Java 使用 JDK 21；key 对照测试还需要本机 C++ 编译器。测试的 Meta 服务仅存在于测试类路径中，不打入生产入口，也不被 Whale 启动。

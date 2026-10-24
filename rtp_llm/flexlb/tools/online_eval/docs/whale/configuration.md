# Whale Mock 配置与环境变量

## 部署模式

`MOCK_DEPLOYMENT_MODE` 是部署拓扑的唯一选择。它与 Master 调度配置、
性能模型、Fetch 和监控相互独立。

| 模式 | Mock JVM 放置 | Master 发现 Mock | P/D 接续 |
|---|---|---|---|
| `embedded` | 寄生 Master Pod；一个 Mock JVM 承载全部 P/D | 动态文件发现 | JVM 内接续 |
| `single_pod` | 独立一个 Mock Pod；一个 JVM 承载全部 P/D | VipServer | JVM 内接续 |
| `per_engine` | 独立多个 Mock Pod；每个 P 或 D 独占一个 Pod/JVM | VipServer | 跨 Pod RPC |

`mock-bundle` 镜像固定入口 `sh /home/admin/whale_start.sh` 支持三个模式，
均要求 `RTP_LLM_MOCK_BUNDLE=1`。只有 `embedded` 启动 Master JVM。
独立 Master 必须使用包含 VipServer adapter 的正式 Master 镜像；bundle 配方打包的
`master.jar` 使用文件发现，含 `legacy-master.jar` 的配方也用于寄生测试，不用于独立 Master。

`mock-engine` 精简 Java 镜像的入口 `sh /opt/flexlb/start.sh` 只支持
`per_engine`，要求 `FLEXLB_MOCK_WHALE=1`。它支持下面列出的挂载文件配置，
不读取 bundle YAML。两种镜像遵守相同的单引擎 Pod 和 VipServer 契约。

启动输出并写入运行目录的 `deployment.json` 记录解析后的模式、发现方式与
接续方式；bundle 镜像还记录 P/D 数量，独立 Mock 记录待注册的 gRPC 地址。
这是启动配置记录，不能作为平台注册成功或运行健康的证明。

### 模式维度与版本支持

`MOCK_DEPLOYMENT_MODE` 描述 Pod/JVM 放置和发现方式。
`master_mode=sb/sn/wb/wn` 描述 Master 的决策与 dispatcher，不描述部署。
`mode_profiles` 中的 `whale_embedded` / `whale_independent` 是底层地址与观测
预设；`--whale` / `--whale-bundle` 是 launcher 使用的 Java 兼容参数。
它们不构成第四种部署模式，平台配置不再手工组合这些内部参数。

支持本契约的 bundle 启动脚本调用 `deployment.py`。既有镜像须根据实际打包
源码与脚本核对能力；环境变量出现在模板中，不证明旧镜像识别它。
legacy JAR 是否存在取决于 CI 配方，见[构建与产物](README.md#ci-构建与产物)。

### 兼容变量与冲突

新配置只设置 `MOCK_DEPLOYMENT_MODE`。bundle 镜像在未设置它时保留
`MOCK_BUNDLE_ROLE=bundle`（默认）的 `embedded` 行为；`MOCK_BUNDLE_ROLE=mock`
映射到 `single_pod`，再配 `MOCK_REMOTE_DECODE=1` 则映射到 `per_engine`。
直接执行 `standalone_mock.py` 时未指定模式仍默认 `single_pod`。
旧变量会打印弃用提示；与所选模式冲突时，在启动 JVM 前报错。

`MOCK_BUNDLE_FILE_DISCOVERY` 不再独立控制发现方式：`embedded` 自动启用文件
发现，包含 legacy Master；独立模式依赖 VipServer。显式保留该旧变量时，值
必须与模式一致。legacy 文件发现不再同时注入静态 `DOMAIN_ADDRESS:*` 快照。

`FETCH_OUTPUT_STREAM` 只选择 frontend 是否取输出，不选择 Pod 拓扑。
`MOCK_BUNDLE_LEGACY_MASTER` 只选择寄生模式的 Master 版本。
独立模式拒绝 bundle YAML、bundle overrides 和 legacy Master 开关。

## 寄生 bundle

默认配置在 `tools/mock_bundle/bundle.yaml`。一个 Pod 内有 Master JVM 和 Mock JVM，Mock JVM 管理多个逻辑 P/D 引擎。

| 变量 | 作用 |
|---|---|
| `RTP_LLM_MOCK_BUNDLE=1` | 必填的 test-only 开关 |
| `START_PORT` | Master HTTP 端口；management 与 Mock control 从配置偏移派生 |
| `POD_IP` | 对外通告的 Pod 地址；不能是 loopback 或 `0.0.0.0` |
| `FLEXLB_CONFIG` | 传给 Master 的完整配置 |
| `MOCK_BUNDLE_OVERRIDES_YAML` | 仅覆盖逻辑 P/D 数与 heap |
| `MOCK_ENGINE_CONFIG_JSON` | real 有效行为与容量快照；缺省使用显式 synthetic 测试文件 |
| `MOCK_CALIBRATION_CONFIG_JSON` | 独立的耗时、EOS 与模拟参数 |
| `MOCK_EOS_CONFIG_JSON` | 可选 EOS 对象；设置后覆盖 performance 内 EOS |
| `FETCH_OUTPUT_STREAM` | `0` 自动接续、不等 frontend Fetch；`1` 等待完整 FetchResponse |
| `MOCK_EVENT_LOG_ENABLED` | 是否保存请求级 JSONL；生产式观测默认依赖 KMonitor |
| `MOCK_PREFILL_HIPPO_ROLE` / `MOCK_DECODE_HIPPO_ROLE` | 显式监控 role 别名，值按原样使用 |
| `MOCK_BUNDLE_LEGACY_MASTER=1` | 使用镜像内 pin 的 legacy Master |
| `MOCK_PREFILL_COUNT` / `MOCK_DECODE_COUNT` | 可覆盖 YAML 中 P/D 数；与 overrides 同时设置时必须一致 |

`MOCK_BUNDLE_OVERRIDES_YAML` 只接受实现列出的字段，数量必须为正整数；heap 接受正整数字节数或 JVM 容量字符串（如 `512m`、`8g`）。P/D 数量变化后必须同时检查 Pod CPU/内存、端口数、Mock heap、每引擎 KV 容量和 D 并发；只增逻辑实例不会自动增加 Pod 资源。

行为、校准、allocator 容量与支持边界统一见[引擎配置](../architecture/engine-configuration.md)。

## 独立 Mock：bundle 镜像

两个独立模式共享 `standalone_mock.py`。`single_pod` 使用
`MOCK_PREFILL_COUNT` / `MOCK_DECODE_COUNT`（各默认 1），允许某角色为 0，
总数必须大于 0。`per_engine` 设置 `ROLE_TYPE=PREFILL` 或 `DECODE` 自动推导
1P0D 或 0P1D，也可显式给出相同数量；每 Pod 总数必须严格为 1。

| 变量 | 默认值 / 作用 |
|---|---|
| `MOCK_GRPC_BASE_PORT` | 默认 29130；全部 P 在前、D 在后连续分配 gRPC 端口 |
| `START_PORT` | 可选 HTTP 控制/健康端口；设置后 gRPC 从 `START_PORT+1` 开始，显式 gRPC 基端口必须一致 |
| `POD_IP` | 必须是可通告的 IPv4 Pod 地址；缺省解析 hostname |
| `MOCK_EVENT_LOOP_THREADS` / `MOCK_COMPLETION_THREADS` | 各 4 |
| `MOCK_JAVA_OPTS` | `-Xms512m -Xmx2g`；支持引号，不执行 shell |
| `FETCH_OUTPUT_STREAM` | 默认 1；0 开启自动接续 |
| `MOCK_KMONITOR_ENABLED` | true |
| `MOCK_CACHE_STATUS_COMPACT_ENCODING` | false；显式启用紧凑 CacheStatus 编码 |
| `MOCK_ENGINE_CONFIG_JSON` | 必填，角色行为与 allocator 容量 |
| `MOCK_CALIBRATION_CONFIG_JSON` | 校准 JSON，缺省读取镜像内文件 |

| `MOCK_BUNDLE_RUN_DIR` | 固定脚本为每次启动建立独立运行目录 |
| `MOCK_BUNDLE_JAR_DIR` | 镜像内 tools/mock_bundle/jars，从中读取 mock.jar |
| `JAVA_HOME` | `/opt/taobao/java` |

例如单独一个 Mock Pod：

```text
RTP_LLM_MOCK_BUNDLE=1
MOCK_DEPLOYMENT_MODE=single_pod
MOCK_PREFILL_COUNT=2
MOCK_DECODE_COUNT=3
FETCH_OUTPUT_STREAM=1
MOCK_ENGINE_CONFIG_JSON=<有效引擎与容量 JSON>
```

按引擎拆分时，P 角色和 D 角色分别配置：

```text
RTP_LLM_MOCK_BUNDLE=1
MOCK_DEPLOYMENT_MODE=per_engine
ROLE_TYPE=PREFILL
START_PORT=7001
FETCH_OUTPUT_STREAM=1
MOCK_ENGINE_CONFIG_JSON=<有效引擎与容量 JSON>
```

D 角色使用 `ROLE_TYPE=DECODE`；P/D 的数量由 Whale 各角色副本数决定。

### VipServer 与健康检查

入口不注册或注销 VIP。部署必须将每个逻辑 P/D 的地址注册到对应角色的 VIP
域，Master 的服务路由引用这些域。对 `protocol=GRPC` 的路由，注册 gRPC 端口；
Master 的同步与 gRPC resolver 均按协议换算端口。协议与注册端口必须配套。

`single_pod` 一个 Pod 有多个逻辑端点，不能只注册一个 Pod 默认端口。
按 `deployment.json` 中的 `vip_grpc_endpoints` 分别注册 P/D 地址；平台模板
必须能表达同一 IP 的多个端口以及 P/D 域归属。若模板只支持一 Pod 一地址，
应使用 `per_engine`，不能把单 Pod 模式当作已经发现全部引擎。
健康检查只访问 Mock 控制口（gRPC 基端口减一），它不等于所有逻辑引擎都有独立
HTTP 健康监听。single_pod 产生的本地 discovery.json 仅供诊断，不能挂给独立 Master。

## 独立 P/D Pod：精简 Java 镜像

| 变量 | 作用 |
|---|---|
| `FLEXLB_MOCK_WHALE=1` | 必填的独立 Pod 开关 |
| `ROLE_TYPE` | `PREFILL` 或 `DECODE` |
| `POD_IP` | 平台通告地址；缺省时入口尝试取 Pod 内首个地址 |
| `START_PORT` | HTTP 健康/控制端口；gRPC 为 `START_PORT+1` |
| `MOCK_ENGINE_CONFIG` / `_JSON` | 有效引擎配置路径或完整 JSON，二选一，必填 |
| `MOCK_CALIBRATION_CONFIG` / `_JSON` | 校准路径或完整 JSON，二选一，必填 |
| `FETCH_OUTPUT_STREAM` | `1` 等待 Fetch；`0` 自动接续 |
| `MOCK_KMONITOR_ENABLED` | Whale 默认为 true；缺少真实 adapter 时启动失败 |
| `MOCK_EVENT_LOG_ENABLED` | 可选请求级事件日志 |
| `MOCK_RUN_DIR` | 写入运行时配置和可选日志的目录 |
| `MOCK_BIND_HOST` | gRPC 监听地址，默认 `0.0.0.0` |

Whale 启动命令应为 `sh /opt/flexlb/start.sh`。配置 JSON 放在环境变量或挂载文件中，不要把带引号的多段 shell 填入平台 `cmd`。

## 独立 ViT 与 Master

P/D 的 `per_engine` 模式不负责启动 Master 或 ViT。每个 ViT Pod 通过
`python3 /opt/flexlb-mock/tools/mock_bundle/vit.py` 启动一个 Java Mock ViT，
要求 `RTP_LLM_MOCK_BUNDLE=1`、`ROLE_TYPE=VIT`，并提供真实模型的
`CHECKPOINT_PATH` 或 `TOKENIZER_PATH`。`START_PORT` 是 HTTP 健康口，
gRPC 是其加一。`MOCK_ENGINE_CONFIG_JSON` 必须包含 `vit` 角色；
并发、两层队列与缓存预算来自该角色。该入口不加载 Torch，也不生成模型 metadata。

独立 Master 使用包含真实 VipServer adapter/client 的 JAR。CI 配方支持时，
构建参数 `master_vipserver=true` 打包并证明该能力；运行入口为
`python3 /opt/flexlb-mock/tools/mock_bundle/master_vip.py`，要求显式 Mock 开关，
并验证镜像构建清单。`MODEL_SERVICE_CONFIG` 分别引用 P、D、V 的 VIP 域，
一条成员对应一个 worker，不使用逻辑端口展开或本地 discovery 文件。
Master 的生命周期 hook 在进程准备后执行一次。

V 的健康检查、VIP 成员和真实 embedding RPC 分别验收。图片请求携带已选择的
V 地址到 P；P 向该 V 取 feature，D 只负责 Decode 接续。每 Pod 的缓存、并发、
取消和资源释放独立，Pod 副本数不能替代模型的图片 token 数。

## Whale 模板字段

- `per_engine` 的 P/D 使用独立角色、资源模板、VIP 域和副本数；`single_pod` 在一个 Mock 角色内声明多个逻辑 P/D 地址；`embedded` 不注册独立 P/D VIP。
- `resource_plan.meta_tag_list` 必须初始化为空列表。
- 最终镜像由顶层 `image_infos` 选择；资源槽位里的 package 信息可能被覆盖。
- 健康检查、容忍配置和资源池必须与目标池匹配，即使 Mock 不申请 GPU。
- 以最终 Carbon plan 为准核对镜像、环境、端口、健康检查和角色数量。

## 监控标签

Mock 指标使用引擎兼容的 `rtp_llm_*` 名称，并保留 `hippo_app`、`hippo_role`、`host_ip`、`container_ip`、`engine`、`engine_port`、`role`、`generation` 和 `backend=mock`。寄生 bundle 的多个逻辑实例必须按 `engine` 或 `engine_port` 拆分，不能把 Pod 总和当作单引擎值。

Prefill `context_tps` 按有效计算 token / 对应 batch execution 时间统计；`context_tps_with_cache` 用包含复用的输入 token 及其对应执行时间。它们不是墙钟吞吐；墙钟指标是 `context_wall_tps` / `context_wall_tps_with_cache`。Mock 数值用于验证配置与调度，不代表实际 GPU kernel 性能。

真实规模的部署校准必须执行 [生产配置与监控对齐](production-alignment.md)，不能只修改 P/D 数量或复制一段性能公式。

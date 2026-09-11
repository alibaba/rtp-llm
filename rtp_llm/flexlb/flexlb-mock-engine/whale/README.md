# Whale 单引擎 Pod

此入口只用于显式启用的 Whale mock 部署。每个 Pod 一个 P 或 D JVM，镜像不包含 master 启动程序。frontend、master 保留各自角色和真实 RPC；P 使用 master 在 `role_addrs` 中选择的 D 地址，通过 `RemoteGenerate` 完成 ALLOCATE、LOAD、GENERATE 和输出回传。

LOAD 模拟 KV 字节传输，缓存容量、预留、排队、生成与取消使用现有 mock 状态机。它验证协议和资源生命周期，不模拟 GPU/RDMA 性能。

## 构建与依赖

离线入口和默认 Maven profile 保持不变。开源环境可执行：

```sh
bash mvnw -P'opensource,!internal' -pl flexlb-mock-engine -am package
```

公司环境先按 master CI 的顺序构建 common，再在 `internal_source/java` 执行 `mvn -pl kmonitor -am install -DskipTests`（不构建 mock 不需要的 VipServer），最后执行 `-Pinternal` Maven package。该 profile 为 mock jar 加入现有 KMonitor 适配包；没有私有包的开源构建不引用其类型。将 `target/flexlb-mock-engine-1.0.0-SNAPSHOT-all.jar` 复制为 `whale/mock-engine.jar`，以 `whale/` 为 Docker build context。内部 CI 通过 `JAVA_BASE_IMAGE` 使用 master 同源的内部 Java 21 基础镜像，覆盖其入口为 mock 启动脚本，不启动 master；开源构建默认使用 Temurin。

`--kmonitor true` 只允许与 `--whale true` 同时启用。启用后若私有适配缺失或退化为 NoOp，启动失败，避免把未上报指标误当成上线成功。开源网络联调显式设置 `MOCK_KMONITOR_ENABLED=false`。

## Whale 角色配置

为 P、D 分别配置 CPU 资源模板、独立 VIP 域和各自 Pod 数；两者使用同一 mock 镜像。master 与 frontend 使用原有独立角色镜像。不要把 mock 镜像放入 master 角色，不要将 P/D 行数设为零后借用 master Pod。

下列环境由角色配置或平台注入：

| 变量 | 语义 |
| --- | --- |
| `FLEXLB_MOCK_WHALE=1` | 必填；缺失时启动包装拒绝运行 |
| `ROLE_TYPE` | `PREFILL` 或 `DECODE` |
| `POD_IP` | 平台通告的 Pod 地址；未注入时取 `hostname -i` 的首个地址，不能用物理机 IP 或 `0.0.0.0` |
| `START_PORT` | HTTP 健康/控制端口，gRPC 固定为其加一 |
| `MOCK_PERFORMANCE_CONFIG` | 挂载的性能配置路径 |
| `MOCK_MASTER_CONFIG` | 与 master 性能模型对应的配置路径 |
| `MOCK_PERFORMANCE_CONFIG_JSON`、`MOCK_MASTER_CONFIG_JSON` | 无挂载时传入完整 JSON；由启动脚本写入运行目录，与对应路径变量互斥 |
| `FETCH_OUTPUT_STREAM` | 默认 1；0 时 P 计算完成直接接续 D，不等 Fetch 或超时 |
| `MOCK_KMONITOR_ENABLED` | Whale 包装默认 true；开源联调设 false |
| `MOCK_RUN_DIR` | 事件日志目录，默认 `/tmp/flexlb-mock` |
| `FLEXLB_MONITOR_SERVICE_NAME`、`FLEXLB_MONITOR_TENANT_NAME` | 复用 master 内部监控适配的环境配置 |

平台启动命令使用 `sh /opt/flexlb/start.sh`，不要把带引号的 `sh -c` 脚本放进 `cmd`：Whale 会按空格拆分该字段。配置 JSON 通过上述环境变量传递，内容不会作为 shell 命令求值。

Whale CPU inference 模板必须初始化 `resource_plan.meta_tag_list: []`，否则 PD gang 信息填充可能空指针。镜像实际由顶层 `image_infos` 选择，资源槽位的 `package_infos` 会被覆盖；frontend 的独立镜像保持在 `front_app_zone_plan`。资源池的容忍配置也必须与目标池匹配，即使 GPU 请求为 0。以最终 Carbon 计划确认这些字段，而非只检查提交模板。

Whale 健康探测使用 `START_PORT` 的 `/health`；停止/排空状态返回 503。平台应把健康 P/D HTTP 地址注册到各自 VIP 域，master 使用已有 VipServerDiscovery 消费这些地址，并按 HTTP+1 访问 gRPC。mock P 不查询 VIP，也不自行选择 D，因此不需要引入 VipServer 客户端依赖。部署时需核实最终 Carbon launch plan 的地址、端口和健康检查配置，不能向生产 master 注入本地 `discovery.json`。

`--bind-host` 只控制 gRPC 监听地址，默认 `0.0.0.0`；`--host` 是通告地址。Whale 模式禁止 `/add_engine`，扩容应创建新 Pod。

## 监控和验证边界

Whale KMonitor 上报累计 context/generate token、KV tokens、waiting/running requests、completed/cancelled 数，以及按实际时间差计算的 TPS。所有指标带 engine、role、进程 generation、backend=mock 标签。累计 token 读数不被 HTTP 抓取消耗。

本地跨进程测试使用不同回环 IP 模拟 Pod，覆盖两个 D 同端口的地址路由、Fetch、显式无 Fetch、P 进程死亡和 D 进程死亡。真实 Whale 验收仍需核对镜像来源、平台注册、KMonitor 数据和 frontend→master→P→D→frontend；单测或镜像编译成功不能代替已部署验收。

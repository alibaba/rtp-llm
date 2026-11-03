# Whale Engine 镜像与启动

P/D/V 共用一个 Mock Engine 镜像，每 Pod 一个 JVM。Master、Frontend 继续使用各自正常镜像；镜像内不带 Case Runner、本地 Prometheus/Grafana、Master bundle 或代码覆盖程序。

## 构建

在框架目录执行：

```sh
JAVA_HOME=/path/to/jdk21 python3 scripts/package-engine.py
```

脚本调用现有 Maven 构建入口，编译 `java/host` 及其依赖，生成 `target/image`。该目录只包含 Dockerfile、固定启动脚本、Engine JAR 和 build-manifest.json。manifest 记录源码 SHA、CI 配方 SHA（本地未提供时为空）、JAR SHA256；打包默认跳过单测执行，并如实记录 tests_executed=false。构建完成后以隔离 ENV 和临时目录启动实际 JAR，检查所属子进程的启动记录及健康接口，通过后记录 service_startup_verified=true。修改工作区后做的本地打包不能仅凭 HEAD SHA 作为干净源码构建的证明。

```sh
docker build -t mock-engine:local target/image
```

Dockerfile 采用与当前 real FlexLB 生产镜像相同的固定 CPU/JDK 基础镜像；不安装 GPU 栈，不在容器启动时编译、下载或覆盖代码。继承镜像的 Master 入口被显式替换，JVM 前台运行。

内部仓 `.aoneci/image.yaml` 增加显式选择的 `mock-engine` 构建项。选择 `pre-jobs: [mock-engine]`，`github_commit` 必须是包含新框架的完整源码 SHA；`github_source_repo` 按标准生产流程指定源码仓。CI 沿用 Java 21、build-image、DADI 转换及区域镜像同步流程，不改已有角色的默认构建项。

CI 执行 package-engine.py 时传入 EXPECTED_SOURCE_SHA 与 CI_RECIPE_SHA。源码不匹配则在编译前失败。普通镜像仓库为 `hub.docker.alibaba-inc.com/isearch/flexlb_mock_engine`，标签遵循生产配方的 release/weights/version 组合；Whale 需要的 accelerated 镜像以对应转换任务的实际结果为准。

## 启动

镜像入口为 `/home/admin/whale_start.sh`，只设置默认监听地址并 exec Java。JVM 参数直接使用标准 `JAVA_TOOL_OPTIONS`，不解析另一套 JVM 参数字符串，不使用 eval。

```sh
docker run --rm \
  -e ROLE_TYPE=PREFILL -e START_PORT=8088 \
  -e SEQ_SIZE_PER_BLOCK=128 -e MAX_SEQ_LEN=32768 -e TEST_BLOCK_NUM=1025 \
  -e MOCK_COMPUTE_FORWARD_EXPRESSION='1 + batchSize' \
  -e JAVA_TOOL_OPTIONS='-Xmx1g' \
  -p 8088:8088 -p 8089:8089 mock-engine:local
```

V 使用 `ROLE_TYPE=VIT`；native 和 Mock 参数遵循[配置参考](component-reference.md#engine-配置规则)。未提供 MOCK_NODE_ID 时，身份使用 role 与本机发布地址；需要指定发布地址时设置 MOCK_ADVERTISE_HOST。共享主机上的实例物理资源仍由平台/JVM 统计，不伪造每 Engine 的 CPU/内存。

HTTP 为 START_PORT，gRPC 为 START_PORT+1；平台探针访问 HTTP `/health`，指标访问 `/metrics`。注册端口必须与 Master 配置的发现协议一致：GRPC 发现发布实际 gRPC 端口，HTTP 发现发布 HTTP 端口；当前 real WorkerAddressService 会据此转换地址，不能把任意控制端口注册进去。

服务不读取本地 topology，不创建实验目录、ready 或 endpoints 文件。固定配置、实际端点在启动日志中记录。正常 TERM 关闭实例；内部 Loop/Scheduler 不可恢复故障使健康检查失败，进程随后非零退出，平台负责重拉。同一工作目录重启不受旧实验文件影响。

## 部署接线边界

这些文件提供可构建的镜像和服务入口，不自动修改 Whale 模板、发布角色或注册 VIP。接入目标部署时，使用其现有角色发现、探针、资源和停止宽限设置。Master 仍是独立 JVM，不起本地 ZK 或 Mock 集群管理器。

公共构建通过 `/metrics` 提供 Prometheus 导出；内部镜像使用 KMonitor 上报，具体选择见下文。端点存在或 SDK 启动成功都不能代替实际采集验收。

首次平台验收应检查实际镜像与生效配置、P/D/V 注册地址、真实请求、TERM 与故障重拉、指标采集。本批不部署到 Whale。

## KVCM 开关

两套部署使用相同镜像、模型、输入和计算配置，只在 Whale 中调整下列原生配置；本地运行直接配置 Case，不另外维护完整部署配置副本。

| Role | 开启 | 关闭 |
| --- | --- | --- |
| Master（全部主备） | `FLEXLB_CONFIG.cacheMatching.type=KVCM`，配置 `MODEL_SERVICE_CONFIG.kvcm` 的服务地址、端口和 namespace | `cacheMatching.type=LOCAL_SYNC` |
| P/D | `KV_CACHE_EVENT_PUBLISHER_TYPE=kvcm`，配置原生 `RECO_*` 发现参数及共享 `KV_CACHE_EVENT_INSTANCE_ID` | `KV_CACHE_EVENT_PUBLISHER_TYPE=none`、`ENABLE_REMOTE_CACHE=false` |
| Frontend、ViT | 保持原有链路和配置 | 相同 |

这里只启用缓存元信息查询和上报，未实现远端 KV 数据加载，开启组的 `ENABLE_REMOTE_CACHE` 也保持 false。Engine 的共享 instance_id 必须等于 Master 查询的 `<namespace>_<cacheKeyBlockSize>`，host 身份必须与 Master 发现结果一致。更改 Master 查询模式需要重启。


## 监控自动选择

Engine 只启用一个指标后端，不需要额外部署开关。公共构建没有 `MetricReporter` SPI 实现，P/D/V 提供 Prometheus `/metrics`；CI 发现原生 `internal_source/java/kmonitor/pom.xml` 时自动启用 `internal-monitoring` profile，将内部 KMonitor reporter、SDK 和 SPI 文件打入 Engine JAR。启动时发现该实现就使用 KMonitor，不注册 `/metrics`，endpoints 的 `metrics_backend` 为 `kmonitor`、`metrics` 为空字符串。内部包在本地启动也会选择 KMonitor；本地 Case 使用公共构建。

内部 reporter 读取 real 的 `kmonitorSinkAddress`（默认 `HIPPO_SLAVE_IP`，再回退 127.0.0.1）、`kmonitorPort`（4141）、`kmonitorServiceName`、`kmonitorTenant`（default）和 `kmonitorNormalSamplePeriod`（1 秒）。`kmonitorTags` 使用原生 `key^value@key^value` 格式，合并到现有 Hippo 身份标签，平台显式标签优先。没有使用 Master 的 `FLEXLB_MONITOR_*` 配置或 `whale-lb` 默认服务名。

两种后端消费同一份结构化快照，各自只有一个采集入口，不解析 Prometheus 文本，不重复清空窗口。快照里的 QPS 已是速率、counter 是累计值，KMonitor 按 Gauge 上报，避免再次转换成 QPS；Histogram bucket/sum/count 保留为同名数值序列，不冒充原始延迟样本。SDK 的 JVM 内置资源采集关闭，避免当成每个模拟 Engine 的资源。指标实例在关闭前停止其采集线程；SDK 为 JVM 共享，在进程退出时关闭。

缺少 provider 时使用 Prometheus；provider 存在但 SDK 加载或初始化失败则启动失败，不静默回退。运行期间上报异常记录错误并在后续周期继续尝试。打包会检查 reporter、SDK、SPI 及服务启动后 `/metrics` 的互斥行为；本地公共测试不能替代内部 CI 的 SDK 编译、配置测试，以及 Whale 上实际指标验收。内部配置测试命令为 `scripts/build.sh -pl java/host -am -Pinternal-monitoring -Dtest=KMonitorReporterTest -Dsurefire.failIfNoSpecifiedTests=false`。

# Whale Engine 镜像与启动

P/D/V 共用一个 Mock Engine 镜像，每 Pod 一个 JVM。Master、Frontend 继续使用各自正常镜像；镜像内不带 Case Runner、本地 Prometheus/Grafana、Master bundle 或代码覆盖程序。

Whale 拆分部署的角色类型使用 `PY_PREFILL`、`PY_DECODE`、`PY_VIT`、`PY_MASTER`、`PY_FRONT_APP`、`PY_KVCM`。不要把 Prefill、Decode 都定义成 `PY_INFERENCE` 再靠 ENV 或手改域名区分；平台会把它们识别为融合角色。P/D/V、Master、Frontend、KVCM 使用各自部署独立的发现域名，发布前回读角色类型与实际 Carbon 注册计划。

## 构建

在框架目录执行：

```sh
JAVA_HOME=/path/to/jdk21 python3 scripts/package-whale-engine.py
```

脚本调用现有 Maven 构建入口，编译 `java/engine` 及其依赖，生成 `target/image`。该目录只包含 Dockerfile、固定启动脚本、Engine JAR 和 build-manifest.json。manifest 记录源码 SHA、CI 配方 SHA（本地未提供时为空）、JAR SHA256；打包默认跳过单测执行，并如实记录 tests_executed=false。修改工作区后做的本地打包不能仅凭 HEAD SHA 作为干净源码构建的证明。

```sh
docker build -t mock-engine:local target/image
```

源码配方 `docker/whale-engine.Dockerfile` 在打包时复制为 `target/image/Dockerfile`。配方采用与当前 real FlexLB 生产镜像相同的固定 CPU/JDK 基础镜像；不安装 GPU 栈，不在容器启动时编译、下载或覆盖代码。继承镜像的 Master 入口被显式替换，JVM 前台运行。

内部仓 `.aoneci/image.yaml` 增加显式选择的 `mock-engine` 构建项。选择 `pre-jobs: [mock-engine]`，`github_commit` 必须是包含新框架的完整源码 SHA；`github_source_repo` 按标准生产流程指定源码仓。CI 沿用 Java 21、build-image、DADI 转换及区域镜像同步流程，不改已有角色的默认构建项。

CI 执行 package-whale-engine.py 时传入 EXPECTED_SOURCE_SHA 与 CI_RECIPE_SHA。源码不匹配则在编译前失败。普通镜像仓库为 `hub.docker.alibaba-inc.com/isearch/flexlb_mock_engine`，标签遵循生产配方的 release/weights/version 组合；Whale 需要的 accelerated 镜像以对应转换任务的实际结果为准。

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

多模态 V 显式设置 `PATCH_SIZE`、`DOWNSAMPLE_RATIO`、`HIDDEN_SIZE`、`MIN_PIXELS` 和 `MAX_IMAGE_TOKENS`。默认异步 GET 图片 URL 并读取尺寸，不依赖 Frontend 提供宽高；准备执行器和 Forward 队列分离。`VIT_CONCURRENCY` 限制异步媒体工作线程，`VIT_MAX_QUEUE_SIZE` 提供额外等待额度，`VIT_GPU_MAX_BATCH_SIZE`/`VIT_GPU_BATCH_WAIT_MS` 控制就绪工作的合批。`MM_TIMEOUT_MS`、`MM_IMAGE_MAX_FILE_SIZE_KB`、`DOWNLOAD_HEADERS` 使用 real 名称与单位。支持的 URL/图片格式和模拟边界见组件参考。

本地假 URL 回放设置 `mock.vit.skip.download: true`（ENV 为 `MOCK_VIT_SKIP_DOWNLOAD=true`），并提供图片宽高；它只跳过读取，仍经过相同的准入、执行器、就绪队列和结果发布流程。Whale 使用默认 false；跳过读取不能用于评估真实下载吞吐。

HTTP 为 START_PORT，gRPC 为 START_PORT+1；平台探针访问 HTTP `/health`，指标访问 `/metrics`。注册端口必须与 Master 配置的发现协议一致：GRPC 发现发布实际 gRPC 端口，HTTP 发现发布 HTTP 端口；当前 real WorkerAddressService 会据此转换地址，不能把任意控制端口注册进去。

服务不读取本地 topology，不创建实验目录、ready 或 endpoints 文件。固定配置、实际端点在启动日志中记录。正常 TERM 关闭实例；内部 Loop/Scheduler 不可恢复故障使健康检查失败，进程随后非零退出，平台负责重拉。同一工作目录重启不受旧实验文件影响。

## 部署接线边界

Mock Engine 启动不读取 checkpoint 或请求 C2 Fuse 挂载。P/D 使用显式的 `MAX_SEQ_LEN`、`IMAGE_TOKEN_ID`（多模态）和 `CACHE_KEY_HASH_SEED`（与真实 Frontend 匹配时）。Seed 使用有符号十进制 int64。部署迁移时将有效参数加入角色配置后再切换镜像。

这些文件提供可构建的镜像和服务入口，不自动修改 Whale 模板、发布角色或注册 VIP。接入目标部署时，使用其现有角色发现、探针、资源和停止宽限设置。Master 仍是独立 JVM，不起本地 ZK 或 Mock 集群管理器。

`/metrics` 是 Prometheus 导出；目标部署必须配置实际采集链路。它不等于已经接入 KMonitor，原生指标命名也不代表采集器已经抓到数据。KMonitor SDK/平台采集接线需另行验证，不能以端点存在代替验收。

首次平台验收应检查实际镜像与生效配置、P/D/V 注册地址、真实请求、TERM 与故障重拉、指标采集。本批不部署到 Whale。

## KVCM 开关

两套部署使用相同镜像、模型、输入和计算配置，只在 Whale 中调整下列原生配置；本地运行直接配置 Case，不另外维护完整部署配置副本。

| Role               | 开启                                                                                                   | 关闭                                 |
| ------------------ | ------------------------------------------------------------------------------------------------------ | ------------------------------------ |
| Master（全部主备） | `FLEXLB_CONFIG.cacheMatching.type=KVCM`，配置 `MODEL_SERVICE_CONFIG.kvcm` 的服务地址、端口和 namespace | `cacheMatching.type=LOCAL_SYNC`      |
| P/D                | `KV_CACHE_EVENT_PUBLISHER_TYPE=kvcm`，配置原生 `RECO_*` 发现参数及共享 `KV_CACHE_EVENT_INSTANCE_ID`    | `KV_CACHE_EVENT_PUBLISHER_TYPE=none` |
| Frontend、ViT      | 保持原有链路和配置                                                                                     | 相同                                 |

这里只启用缓存元信息查询和上报，未实现远端 KV 数据加载。Engine 的共享 instance*id 必须等于 Master 查询的 `<namespace>*<cacheKeyBlockSize>`，host 身份必须与 Master 发现结果一致。更改 Master 查询模式需要重启。

当前 KVCM 查询只使用本地 HBM 命中，Master 的 cacheMatching 同时显式配置 `medium: [hbm]`、`globalKvsHostCount: 0`、`enableP2p: false`。仍需核对实际 MetaService descriptor；不能将 global 命中或本地回退成功当成 KVCM 本地查询成功。协议及回退边界见[组件参考](component-reference.md#dsv41-engine--kvcm-契约)。

## 监控自动选择

Engine 只启用一个指标后端，不需要额外部署开关。公共构建没有 `MetricReporter` SPI 实现，P/D/V 提供 Prometheus `/metrics`；CI 发现原生 `internal_source/java/kmonitor/pom.xml` 时自动启用 `internal-monitoring` profile，将内部 KMonitor reporter、SDK 和 SPI 文件打入 Engine JAR。启动时发现该实现就使用 KMonitor，不注册 `/metrics`，endpoints 的 `metrics_backend` 为 `kmonitor`、`metrics` 为空字符串。内部包在本地启动也会选择 KMonitor；本地 Case 使用公共构建。

内部 reporter 读取 real 的 `kmonitorSinkAddress`（默认 `HIPPO_SLAVE_IP`，再回退 127.0.0.1）、`kmonitorPort`（4141）、`kmonitorServiceName`、`kmonitorTenant`（default）和 `kmonitorNormalSamplePeriod`（1 秒）。`kmonitorTags` 使用原生 `key^value@key^value` 格式，合并到现有 Hippo 身份标签，平台显式标签优先。没有使用 Master 的 `FLEXLB_MONITOR_*` 配置或 `whale-lb` 默认服务名。

两种后端消费同一份结构化快照，各自只有一个采集入口，不解析 Prometheus 文本，不重复清空窗口。快照里的 QPS 已是速率、counter 是累计值，KMonitor 按 Gauge 上报，避免再次转换成 QPS；Histogram bucket/sum/count 保留为同名数值序列，不冒充原始延迟样本。SDK 的 JVM 内置资源采集关闭，避免当成每个模拟 Engine 的资源。指标实例在关闭前停止其采集线程；SDK 为 JVM 共享，在进程退出时关闭。

缺少 provider 时使用 Prometheus；provider 存在但 SDK 加载或初始化失败则启动失败，不静默回退。运行期间上报异常记录错误并在后续周期继续尝试。打包会检查 reporter、SDK、SPI 及服务启动后 `/metrics` 的互斥行为；本地公共测试不能替代内部 CI 的 SDK 编译、配置测试，以及 Whale 上实际指标验收。内部配置测试命令为 `scripts/build-java.sh -pl java/engine -am -Pinternal-monitoring -Dtest=KMonitorReporterTest -Dsurefire.failIfNoSpecifiedTests=false`。

### 配置初始化边界

文件配置和 Whale ENV 都在 `RtpConfig` / `VitConfig` 中完成默认值、派生值和校验，随后才创建组件。
`EngineConfig` 负责输入转换和模型文件读取，`LLMMockEngine` / `VitNode` 负责装配与生命周期；Scheduler 使用最终值，不重新解释自动配置。
模型元信息每次加载只读一次，缓存 key namespace 由共享 `CacheSequence` 解析，Client 与 Engine 共用。
ENV 路径始终使用入口传入的环境，包括模型下载和 DSV4.1 namespace，不混入另一份进程 ENV。

- `max_batch_tokens_size` 缺省或为 0 时，使用 `max_context_batch_size × max_seq_len`；context batch 默认 1。显式非零预算保持不变，超出支持的整数范围会在启动前报错。
- `seq_size_per_block` 缺省或为 0 时使用普通设备路径的 64。Mock 不根据宿主硬件推断 ROCm/XPU；需要这些布局时显式设置原生参数。
- `max_seq_len` 缺省或为 0 时使用 8192；Mock 不从 checkpoint 推导长度，模拟长上下文时显式配置。
- CP 开启时，`cp_force_single_prefill` 默认 true；设为 false 才允许同轮接纳多个 Prefill。uncached token 配额按 real 的 `2 × CP size` 对齐计数。
- WorkerStatus 和 KVCM 注册保留单 DP 的协议元信息。DP size 固定为 1，WorkerStatus 的 rank 固定为 0，TP size 在 P 上取 CP size，在 D 上取 1。
- ViT 将 `vit_concurrency` 归一为至少 1，`vit_max_queue_size` 归一为至少 0；不额外限制一小时超时上限。

容量只保留两个入口：`test_block_num` 必填，表示包含 null block 的真实 device 块数（至少 2）；可选 `mock.cache.total.blocks` 表示统一缓存保留容量，不含 null block，默认 `test_block_num - 1` 且不能更小。ENV 分别为 `TEST_BLOCK_NUM`、`MOCK_CACHE_TOTAL_BLOCKS`。`reserve_block_ratio` 只按 device 容量计算。

例如 `TEST_BLOCK_NUM=1025 MOCK_CACHE_TOTAL_BLOCKS=4096` 表示 device 最多容纳 1024 个不同的活动 KV block，统一索引最多保留 4096 个 block。缓存命中同样需要 device 额度，初次容量不足等待重试，Decode 增长不足失败；不模拟 Host/device 搬运耗时。

`device_blocks`、`reserve_blocks` 仅为派生输出。旧 `mock.cache.memory.blocks`、`capacity.*` 和 `device_eviction_policy` 不再接受，LRU 固定；不要为扩大统一池配置 `enable_memory_cache`。`memory_cache_size_mb` 及真实 GPU 显存预算不用于推算 block 数。Mock Case 应只保留实际消费的参数。

### 真实 Frontend 的 CP 缓存布局

Whale 的真实 Frontend 按原生配置开启分片。对于基础块为 128 的 CP4 布局，Mock P 的缓存配置如下。

```text
SEQ_SIZE_PER_BLOCK=128
PREFILL_CP_SIZE=4
```

Mock P 上报 K4、K8 等 canonical keys，CacheStatus 的逻辑块为 512。哈希和 KVCM 注册的基础块大小为 128。Frontend 与 Mock P 使用相同的 seed、基础块和抽样步长。

迁移旧 Mock 部署时，直接填写所需的 `PREFILL_CP_SIZE`。旧的 `TP_SIZE`、`DP_SIZE`、`CP_ROTATE_METHOD` 和 `PREFILL_CP_KV_CACHE_SHARDED` 已退出 Mock 配置。原来只配置了 CP size 的部署，现在会同时应用对应的 Prefill 准入规则。

D 的本地 FULL KV 按基础块管理。P→D 传完整语义输入和媒体身份，不增加 CP 协议字段。本地 Client/Case 可保持不分片，无需把 CP 写进数据集。具体范围见 [Prefill CP](component-reference.md#prefill-cp)。

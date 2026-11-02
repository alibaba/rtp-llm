# Whale Engine 镜像与启动

P/D/V 共用一个 Mock Engine 镜像，每 Pod 一个 JVM。Master、Frontend 继续使用各自正常镜像；镜像内不带 Case Runner、本地 Prometheus/Grafana、Master bundle 或代码覆盖程序。

Whale 拆分部署的角色类型使用 `PY_PREFILL`、`PY_DECODE`、`PY_VIT`、`PY_MASTER`、`PY_FRONT_APP`、`PY_KVCM`。不要把 Prefill、Decode 都定义成 `PY_INFERENCE` 再靠 ENV 或手改域名区分；平台会把它们识别为融合角色。P/D/V、Master、Frontend、KVCM 使用各自部署独立的发现域名，发布前回读角色类型与实际 Carbon 注册计划。

## 构建

在框架目录执行：

```sh
JAVA_HOME=/path/to/jdk21 python3 scripts/package-engine.py
```

脚本调用现有 Maven 构建入口，编译 `java/host` 及其依赖，生成 `target/image`。该目录只包含 Dockerfile、固定启动脚本、Engine JAR 和 build-manifest.json。manifest 记录源码 SHA、CI 配方 SHA（本地未提供时为空）、JAR SHA256；打包默认跳过单测执行，并如实记录 tests_executed=false。修改工作区后做的本地打包不能仅凭 HEAD SHA 作为干净源码构建的证明。

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

多模态 V 设置 `CHECKPOINT_PATH` 指向可读的模型目录，仅读取 config.json。默认异步 GET 图片 URL 并读取尺寸，不依赖 Frontend 提供宽高；准备执行器和 Forward 队列分离。`VIT_CONCURRENCY` 限制异步媒体工作线程，`VIT_MAX_QUEUE_SIZE` 提供额外等待额度，`VIT_GPU_MAX_BATCH_SIZE`/`VIT_GPU_BATCH_WAIT_MS` 控制就绪工作的合批。`MM_TIMEOUT_MS`、`MM_IMAGE_MAX_FILE_SIZE_KB`、`DOWNLOAD_HEADERS` 使用 real 名称与单位。支持的 URL/图片格式和模拟边界见组件参考。

本地假 URL 回放设置 `mock.vit.skip.download: true`（ENV 为 `MOCK_VIT_SKIP_DOWNLOAD=true`），并提供图片宽高；它只跳过读取，仍经过相同的准入、执行器、就绪队列和结果发布流程。Whale 使用默认 false；跳过读取不能用于评估真实下载吞吐。


HTTP 为 START_PORT，gRPC 为 START_PORT+1；平台探针访问 HTTP `/health`，指标访问 `/metrics`。注册端口必须与 Master 配置的发现协议一致：GRPC 发现发布实际 gRPC 端口，HTTP 发现发布 HTTP 端口；当前 real WorkerAddressService 会据此转换地址，不能把任意控制端口注册进去。

服务不读取本地 topology，不创建实验目录、ready 或 endpoints 文件。固定配置、实际端点在启动日志中记录。正常 TERM 关闭实例；内部 Loop/Scheduler 不可恢复故障使健康检查失败，进程随后非零退出，平台负责重拉。同一工作目录重启不受旧实验文件影响。

## 部署接线边界

`CHECKPOINT_PATH` 可以使用平台注入的 OSS 模型目录；Java 启动时通过 C2 Fuse 的原生只读挂载协议解析为本地目录，只读取 config.json。与 real RTP 一致，host Fuse 使用 HIPPO_FUSE_PORT、HIPPO_SLAVE_IP 和 HIPPO_APP_WORKDIR，sidecar Fuse 使用 28006 端口及 /mnt/fuse。模板须提供 /mnt/fuse 的共享卷和 HostToContainer 传播。有效配置记录本地挂载路径，避免输出带凭据的 OSS URI；不在部署 ENV 中注入下载程序。

这些文件提供可构建的镜像和服务入口，不自动修改 Whale 模板、发布角色或注册 VIP。接入目标部署时，使用其现有角色发现、探针、资源和停止宽限设置。Master 仍是独立 JVM，不起本地 ZK 或 Mock 集群管理器。

`/metrics` 是 Prometheus 导出；目标部署必须配置实际采集链路。它不等于已经接入 KMonitor，原生指标命名也不代表采集器已经抓到数据。KMonitor SDK/平台采集接线需另行验证，不能以端点存在代替验收。

首次平台验收应检查实际镜像与生效配置、P/D/V 注册地址、真实请求、TERM 与故障重拉、指标采集。本批不部署到 Whale。

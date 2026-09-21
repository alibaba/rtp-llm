# Whale Mock CI 与部署

Whale 运行不复用开发机 runner。代码通过 CI 生成镜像，Whale 负责实例、环境变量、健康检查和滚动更新，KMonitor 保存运行证据。

## 选择拓扑与 CI 产物

| 目的 | CI `pre-jobs` | 镜像 | Whale 拓扑 |
|---|---|---|---|
| 新建或完整更新寄生部署 | `mock-bundle` | `rtp_llm_mock_bundle` | 一个 CPU Pod 内运行 Master 和多个逻辑 P/D |
| 只更新已部署 bundle 的 Java Mock | `mock-refresh` | `rtp_llm_mock_bundle` | 保留固定基础镜像中的 legacy Master 与 Python，只替换并验证 Mock 相关文件 |
| P/D 分成独立 Pod | `mock-engine` | `flexlb-mock-engine` | 每 Pod 一个 PREFILL 或 DECODE JVM；Master/frontend 使用各自镜像 |

CI 定义位于外层仓库 `.aoneci/image.yaml`。`mock-bundle` 和 `mock-refresh` 使用 Maven profile `opensource,!internal,whale-bundle`，先安装公共模块与内部 KMonitor，再构建测试 JAR。`mock-engine` 使用 internal profile，并校验产物确实含 KMonitor 类。

## 构建链

### 完整 bundle

1. 构建 `flexlb-api` 与 `flexlb-mock-engine`，执行 Whale/RemoteDecode 相关测试。
2. 复制 `master.jar`、`mock.jar`、`tools/whale_mock`、`flexlb_cfg.py`、模式表和性能文件到镜像上下文。
3. 若存在 legacy master spec，按 pin 构建 `legacy-master.jar`。
4. 以标准 engine 镜像提供 Python，以标准 FlexLB 镜像提供 Java/runtime，生成 `rtp_llm_mock_bundle:<source-version>`。

### mock-refresh

1. 在当前源码上构建并运行更完整的 Mock 单测集合。
2. 重新生成 Mock JAR、兼容层、bundle 入口与模式表。
3. 以 CI 中明确 pin 的 `MOCK_BASE_IMAGE` 为基础生成新镜像。该 pin 决定保留的 Master 与运行时，不能只看当前源码 commit。

### 独立 Mock Engine

1. 构建带内部 KMonitor 的 shaded Mock JAR。
2. 校验 JAR 中存在 monitoring factory 与 KMonitor 类。
3. 将 JAR 放入 `flexlb-mock-engine/whale` context，生成 `flexlb-mock-engine:<source-version>`。

CI 成功后记录源码 commit、job 类型、完整镜像地址和 digest。镜像构建成功不等于部署成功。

## Whale 配置与发布

1. 选择与拓扑对应的镜像，不要把独立 engine 镜像放进 Master 角色，也不要把 bundle 镜像放进 P/D 独立角色。
2. 按[配置与环境变量](configuration.md)填写必需变量、资源、端口、健康检查和角色数量。
3. 生成新模板版本并发布到目标部署。模板版本、biz version 和最终部署引用必须来自同一次变更。
4. 等待所有目标角色 ready；发布状态仍在进行或 frontend 未 ready 时不能进入验收。

## 运行验收

### 寄生 bundle

- frontend 健康检查使用原生 `/frontend_health`。
- Master `/health` 正常。
- Mock 控制口 `/health` 返回 `healthy`，引擎数与配置一致。
- 调度接受数、Decode 完成数、失败数和输出 token 能闭合。
- no-Fetch 模式确认 Fetch RPC 为 0；Fetch 模式确认完整输出链路。
- 停流后队列、running 和应释放 KV 回到预期稳态。

### 独立 P/D Pod

- Whale 健康检查访问每 Pod `START_PORT` 的 `/health`。
- Master 通过平台注册发现 P/D，并按 HTTP 端口加一访问 gRPC。
- P/D Pod 数、角色、注册地址和 KMonitor `hippo_role` 与部署配置一致。
- frontend → Master → P → D → frontend 完成真实请求。

Schedule ACK 只证明请求被接受，不能代替 Decode 完成或端到端成功。单测、镜像、发布 API 成功和 Pod ready 是不同阶段的证据。

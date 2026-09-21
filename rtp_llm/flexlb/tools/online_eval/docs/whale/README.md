# Whale Mock CI 与部署

Whale 运行不复用开发机 runner。代码通过 CI 生成镜像，Whale 负责实例、环境变量、健康检查和滚动更新，KMonitor 保存运行证据。

## 选择拓扑与 CI 产物

| 目的 | CI `pre-jobs` | 镜像 | Whale 拓扑 |
|---|---|---|---|
| 构建或更新寄生部署 | `mock-bundle` | `rtp_llm_mock_bundle` | 一个 CPU Pod 内运行 Master 和多个逻辑 P/D |
| P/D 分成独立 Pod | `mock-engine` | `flexlb-mock-engine` | 每 Pod 一个 PREFILL 或 DECODE JVM；Master/frontend 使用各自镜像 |

CI 定义位于外层仓库 `.aoneci/image.yaml`。`mock-bundle` 使用 Maven profile `opensource,!internal,whale-bundle`，先安装公共模块与内部 KMonitor，再构建测试 JAR。`mock-engine` 使用 internal profile，并校验产物确实含 KMonitor 类。

## 构建链

### 寄生 bundle

1. 构建 `flexlb-api` 与 `flexlb-mock-engine`，执行 Whale、RemoteDecode、缓存、扩缩容和取消相关测试，并执行 `tools/whale_mock` Python 测试。
2. 复制 `master.jar`、`mock.jar`、`tools/whale_mock`、`flexlb_cfg.py`、模式表和性能文件到镜像上下文。
3. 按 pin 构建 `legacy-master.jar`，并生成 `tools/whale_mock/bundle-manifest.sha256` 记录完整上下文文件摘要。
4. 直接使用固定版本的 CPU `rtp_llm_root_base`，仅安装运行时依赖 PyYAML，生成 `rtp_llm_mock_bundle:<source-version>`。

`mock-bundle` 只静态依赖 `info`，不会触发 CUDA 或 FlexLB 镜像 job。当前
`master.jar`、`mock.jar`、legacy Master 和所有 Python/配置文件均来自本次
checkout，不在历史 bundle 上做部分覆盖。CPU 基座通过流水线
`vars.mock_runtime_base_image` 固定版本；升级 Python/JDK 基座必须显式修改该变量。

```bash
python3 /Users/wangziyi/.agents/skills/ci-image-build/scripts/ci_image_build.py \
  --pipeline-branch feat/dsv4-master-bundle-codex \
  --build-ref origin/codex/ft-case-framework \
  --jobs mock-bundle --poll
```

旧 `mock-refresh` 已移除：它不更新当前 `master.jar`，而且白名单覆盖会让镜像
混入旧基座内容。新构建统一使用 `mock-bundle`；历史 CI run 仍可按 run ID 查询。

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

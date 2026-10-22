# Whale Mock CI 与部署

先在[配置契约](configuration.md)选择 `MOCK_DEPLOYMENT_MODE`，再选能够承载该
模式的镜像与 CI 配方。Whale 负责角色、实例、注册、健康检查和滚动更新；
开发机 runner 不负责 Whale 部署。

## 文档与工具分工

| 内容 | 维护入口 |
|---|---|
| 三种模式、配置作用域、兼容规则、端口和 VIP 要求 | [配置契约](configuration.md) |
| 模式入口与 supervisor | `tools/mock_bundle/deployment.py`、`bundle.py`、`standalone_mock.py` |
| 构建计划、配方检查、产物证明 | `ci-image-build` skill；读取所选配方的 contract |
| 平台模板、部署、镜像及发布 | `whale-biz-operation` skill 与 Whale MCP |
| 开发机租约、同步、作业、产物读取 | `flexlb-mock-engine-test` skill 与 FlexLB MCP |
| 性能参数与监控口径对齐 | [生产对齐](production-alignment.md) |

仓库维护运行契约，skill 负责入口选择，MCP 提供基础能力。
不从镜像名、job 名或旧模式别名推导部署拓扑。

## CI 构建与产物

`mock-bundle` 是镜像产物，不等于 `embedded` 模式；支持统一入口的 bundle
镜像可以运行三种模式。`mock-engine` 精简镜像只运行 `per_engine`。
job 是否可用、打包组件和依赖以所选 CI 配方的 contract 为准。

Mock bundle 构建使用 `ci-image-build` 的版本化 plan，分别固定配方、Master 和
Mock 源码。先从该 skill 的示例填写实际 checkout、ref、jobs 和参数，再执行：

```bash
python3 ~/.agents/skills/ci-image-build/scripts/ci_image_build.py \
  --plan '<build.plan.json>' --check-only --plan-output '<checked.plan.json>'

python3 ~/.agents/skills/ci-image-build/scripts/ci_image_build.py \
  --plan '<checked.plan.json>' --plan-output '<submitted.plan.json>' --poll \
  --manifest-output '<ci-manifest.json>' --evidence-output '<ci-evidence.json>'
```

不使用旧 `--pipeline-branch / --build-ref / --jobs mock-bundle` 组合新建构建。
`--check-only` 只检查，不触发 CI。普通镜像构建按配方默认跳过单测执行；
需要测试验证时显式设置 plan 的 `run_tests=true`。跳过单测时交付写明 tests skipped。

不同配方可能只包含 `master.jar`、`mock.jar`，也可能额外包含
`legacy-master.jar`；不能仅凭镜像名称判断支持 legacy Master。
检查实际构建证明、JAR 清单、启动命令和 `MOCK_BUNDLE_JAR_DIR` 覆盖。
独立 Master 还需确认其自身包含 VipServer adapter。

镜像内的 launcher 必须来自支持所选模式的源码。检查模式解析入口和启动脚本，
不能给旧镜像增加 `MOCK_DEPLOYMENT_MODE` 后就认定它生效。
CI 成功后保存 run、实际组件 SHA、镜像地址、可用的 digest 和构建证明。
构建成功不证明镜像已经部署。

## Whale 配置与发布

1. 读取目标部署实际版本的模板快照，核对模式、角色、命令、环境、资源与注册端口。
2. 镜像替换可用 Whale MCP `template_image_plan`，它保留原配置，不执行模式迁移。
   切换模式时单独列出角色、环境、发现方式、VIP 和健康检查的变化。
3. 按既有授权准备精确写请求，生成模板版本，再单独发布至指定部署。
4. 只读核对实际版本和 Carbon plan；等待所有目标角色 ready，再验收请求链路。

本地源码支持、镜像打包支持、平台注册配置和运行健康是不同证据。
`deployment.json` 只记录 launcher 的解析结果，不代替 VIP 成员和真实请求验证。

## 运行验收

- `embedded`：Master 使用动态发现文件；legacy Master 也须走文件适配，不能退回静态地址快照。
- `single_pod`：按 P/D 域核对全部逻辑端点的 VIP 成员，不能只检查 Pod 默认端口。
- `per_engine`：核对每 Pod 仅一个 P 或 D，P/D 规模由各角色副本数决定。
- frontend、Master 和每个 Mock 控制口的健康检查分别通过；frontend 探针及
  Carbon/VIP 两处检查规则见[生产对齐](production-alignment.md)。
- frontend → Master → P → D → frontend 完成请求；Fetch 模式核对完整输出，
  自动接续模式核对完成与释放。停流后队列、running 和 KV 回到预期稳态。
- 多逻辑引擎共享一个 JVM 时按 engine / engine_port 拆分监控，不能用 Pod 总量
  代替单引擎指标。

Schedule ACK 只证明请求被接受，不能代替 Decode 完成或端到端成功。

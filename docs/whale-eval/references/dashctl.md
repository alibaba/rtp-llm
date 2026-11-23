# dashctl 与快照

需要本机已安装且能访问生产 ModelBoard 的 `dashctl`，以及登录可用的 Whale、Spectrum、大眼猴。按仓库安装和配置 Token。配置快照和回执时使用 `umask 077` 的临时文件，日志只输出必要字段。Checkpoint URI 只输出无查询参数的路径。 [dashctl 仓库](https://code.alibaba-inc.com/dashscope/dashctl)

## 从 Whale 锁定版本

- 打开给定的实际部署运维页，在「发布历史」以目标版本和发布记录找到部署快照。默认取当前生效或最新成功发布的对应快照。用户明确要求正在发布的版本时，以不可变快照校验并明确该发布状态。排除已取消或失败的记录。快照列表里的 13 位时间戳和服务版本号用来定位行。打开该行的「部署快照」读取 JSON 中的 24 位 `id`。
- 核对快照的 `deployment_id`、`template_info`、主模型/Decode/SP checkpoint、Frontend 自定义镜像、Master 配置。对照用户要求锁定，不从服务名称或时间戳猜版本。
- 使用已配置的 `dashctl` 校验快照，读取 `.frozen_coordinates.snapshot.{id,sha256,source_biz_id,source_deployment_id}` 和 Model/Mint/CI-Test 已解析的 repo、path、完整 commit。按 `.subtasks[]` 核对类型及 suite。命名模式预检通常取默认套件。对显式 nightly/weekly 用独立验证的精确 CI-Test 路径和 commit，保持模型、目标组件、卡型和上下文桶相符。

```sh
dashctl create cirun "$MODEL" --snapshot "$SNAPSHOT_ID" \
  --test func,effective,stab --region "$VERIFY_REGION" \
  --dry-run=server --baseline-version -o json
```

## 冻结可执行基线

同源且试跑三元组完全一致时可用原生 `--snapshot` 流程。实际提交会先复制 develop 基线到独立分支，所以遇到 copy 已创建、后续预检失败的情况，应保留返回的分支和 commit 并核对是否已有作业。

自动选机房时优先使用带精确 Model 路径和 commit 的基线。若原生快照提交因 source service 不同被拒绝，使用已验证与源服务匹配的个人基线副本，也可从精确的兼容 Model 基线拷一份。对仍无匹配副本的服务，可用官方 `--service "$WHALE_URL"` 新建参考模式做只读服务端预检。只有服务端实际支持新建、预检明确确认源快照与三元组，才采用该模式，并按同一资源核算规则逐机房预检实际参数。若提示仅支持借用，或无法固定用户要求的快照，报告缺少兼容基线或后端能力。避免强行覆盖已有基线的源服务身份。

```sh
umask 077
dashctl create cibaseline --baseline-path "$MODEL_PATH" --ref "$COMPATIBLE_MODEL_COMMIT" \
  --name "$BASELINE_NAME" --request-id "$PREPARE_ID" -o json > "$BASELINE_EXPORT"
```

编辑的是完整 CIBaseline 导出，保留外层 ID、名称和 commit，并确认 `.document.spec.components[]` 中仅有一个 `.metadata.role == "whale"` 的对象。核对其 `.spec._meta.sourceBizId` 与目标快照所属服务一致，并确认资源规格兼容。同一服务的新部署可使用快照中实际的 `.sourceDeploymentId`。更新这些字段。

- `.spec._meta.sourceDeploymentId`
- `.spec._meta.sourceDeploymentVersionSnapshotId`
- `.spec._meta.sourceDeploymentVersionSnapshotSha256`

若用户指定不同 CI-Test 套件，在 `.document.metadata.ci_test_baseline[]["ci-test"]` 使用已解析的 repo、path 和带完整 SHA 的 revision，其他 Model/Mint 配置继承。保存个人版本，核对返回的 source、snapshot、`.resolved_refs` 和完整 commit。完整导出用于更新，脱敏的 preview 内容只用于核对。

```sh
dashctl update cibaseline "$BASELINE_ID" -f "$EDITED_EXPORT" \
  --request-id "$SAVE_ID" -o json > "$SAVED_EXPORT"
```

在选定机房再用 `--dry-run=server --baseline-version` 核对保存后实际冻结的 Model/Mint/CI-Test、源快照 ID 和要求的角色行数。已验证链路通过 `--baseline-path "$MODEL_PATH" --ref "$SAVED_COMMIT"` 明确指定区域。当前 CLI 的 `--ci-baseline NAME@version` 组合可能不支持 `--region`。

## 路由和提交

在本 Skill 目录执行下面的只读预检。Model 基线应固定 Mint 和 CI-Test 依赖的完整 commit。

```sh
python3 scripts/choose_region.py \
  --baseline-path "$MODEL_PATH" --ref "$SAVED_COMMIT" \
  --suite "$CI_TEST_PATH" --suite-commit "$CI_TEST_COMMIT" \
  --mint-path "$MINT_PATH" --mint-commit "$MINT_COMMIT"
```

可选 `--gpu-type L20D`、`--replicas p=1,d=1,frontend=20`、`--test func,effective`，仅按用户要求或已确认的配置传入。重复 `--region` 限制候选。Python 3.9+、`dashctl` 在 PATH 即可运行。脚本只读。每个预检会按子任务汇总同一地区、池和单位的需求，重复可用量只取最小值。GPU 和 CPU 实例的可用量各自比较。多个可行候选按 Whale GPU 瓶颈剩余卡数排序。退出码 2 表示没有已验证可行的候选。若预检错误的详情影响恢复，重跑该机房同参数的官方 dry-run 获取诊断。

对 `submit_argv` 所示参数做最终快照与角色核对后执行提交。追加 `--request-id "$RUN_ID" -o json`，保留回执和提交参数。保存/拷贝/作业提交各自管理请求 ID。收到已接受回执后使用 job ID 跟踪。写请求结果不确定时，先查本次作业记录，必要时以同一请求 ID 和同样参数恢复。迁到别的机房只在用户授权或原作业明确没有接受、已取消或终止后进行。

## 场景核验

`--test` 只选模块。对指定场景的请求先查锁定的 CI-Test 配置。ModelBoard 的已认证只读接口 `GET /api/v1/cicd/scheduler/test-module-groups` 接收 `repo`、`revision`、`path`，应传入冻结三元组。认证用当前受保护的 ModelBoard Bearer，避免将 Token 放进 URL。该接口返回声明的场景组，实际大眼猴可能展开成多个用例。因此按任务展开核对真正执行的用例、类型和数量。若声明范围有多余项，使用有明确选择功能的官方手动运行界面和受支持的配置，或者说明 CLI 无法完成精确筛选。不要把全模块当作指定子集提交。

GLM5.2 nightly 曾配置 Prefill 单点、Decode 单点、Decode 过载反压 Prefill 三个 `stab` 场景组。每次以锁定的套件版本为准。

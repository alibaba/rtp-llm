# 在 AoneCI 运行 Mock Case

用 `a1` CLI 在内部仓 `foundation_models/RTP-LLM` 触发流水线。有兼容的 Case 环境镜像时，直接跳到第 2 步。环境镜像可以复用，每次功能测试都会从指定的源码提交构建 Java 运行产物。

`RECIPE_REF` 是**内部仓**的 CI 配方分支。换分支时，先确认该分支的 `.aoneci/image-base.yaml` 有 `mock-ci-image`，且 `.aoneci/main.yaml` 有 `mock-ci-case` 入口。`SOURCE_SHA` 是已推送到 `SOURCE_REPO` 的完整 40 位源码提交 SHA。

```sh
CI_REPO=foundation_models/RTP-LLM
RECIPE_REF=feat/mock-case-ci-20261103-wzy
SOURCE_REPO=git@github.com:alibaba/rtp-llm.git
# 在已推送的源码 checkout 中执行，或手动填写要测的完整 SHA。
SOURCE_SHA=$(git rev-parse HEAD)
```

## 1. 准备 Case 环境镜像

环境镜像提供 Java 21、Prometheus 和 Maven、uv 依赖缓存。首次使用或构建依赖变更时，用 **base/dev 镜像**流水线 `3828` 构建。下面只选 `mock-ci-image` Job。

```sh
a1 ci pipeline run 3828 --repo "$CI_REPO" --branch "$RECIPE_REF" \
  --param 'mock-ci-image=true' \
  --param 'ppu=false' --param 'amd=false' --param 'cuda12_9=false' \
  --param 'cuda13=false' --param 'cuda13_arm=false' --param 'flexlb=false' \
  --param "github_commit=${SOURCE_SHA:?}" --param "github_source_repo=$SOURCE_REPO" \
  --param 'cancel-in-progress=false' -f json
```

将触发返回的 Run ID 填入 `IMAGE_RUN_ID`。确认运行成功后，从 Job 列表找到 `mock-ci-image` 的 ID，填入 `IMAGE_JOB_ID`。

```sh
a1 ci run get "$IMAGE_RUN_ID" --repo "$CI_REPO" -f json
a1 ci job list --run "$IMAGE_RUN_ID" --repo "$CI_REPO" -f json
a1 ci job summary "$IMAGE_JOB_ID" --run "$IMAGE_RUN_ID" --repo "$CI_REPO" -f json
```

从摘要复制 `hub.docker.alibaba-inc.com/isearch/rtp_llm_mock_case_ci` 的完整镜像地址（带 digest 或 tag），设为 `CASE_IMAGE`。

## 2. 运行一个或多个 Case

用**功能测试**流水线 `1346`。将 `SOURCE_SHA` 设为这次要验证的源码提交，并把兼容的环境镜像地址填入 `CASE_IMAGE`。

| 参数                                  | 填法                                                                                                                                       |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `--branch`                            | 内部仓 CI 配方分支。                                                                                                                       |
| `github_source_repo`、`github_commit` | 源码仓地址和已推送的 40 位提交 SHA。                                                                                                       |
| `pre-jobs`                            | 设为 `["mock-ci-case"]`，选择 Mock Case 测试范围。                                                                                         |
| `mock-cases`                          | 选择 Case 和 runner，默认 `["basic"]`。每项可填 `<id>`、`<id>@8-32Gi` 或 `<id>@32-256Gi`。省略规格时用 `8-32Gi`，同一 Case ID 只能选一次。 |
| `mock-scrape-interval`                | 本次运行的 Prometheus 采集间隔，默认 `1s`。单独跑 HA 或 perf 时可传 `4s`。                                                                 |
| `mock-case-image`                     | 第 1 步产出的完整环境镜像地址，或兼容的已有镜像。                                                                                          |
| `cancel-in-progress`                  | 设为 `false`，避免这次触发取消同一配方分支上已在运行的 CI。                                                                                |

以下命令让 `basic` 用 `8-32Gi`，HA 用 `32-256Gi`，两者并行执行。单独跑性能回归时，把 `mock-cases` 改为 `["performance_regression@32-256Gi"]`。示例中的参数文件按 `a1 --params-file` 接受的字符串格式传值，AoneCI 会将数组和布尔参数解析为对应类型。

```sh
PARAMS_FILE=$(mktemp)
cat > "$PARAMS_FILE" <<EOF
{
  "pre-jobs": "[\"mock-ci-case\"]",
  "mock-cases": "[\"basic\", \"master_ha_failover@32-256Gi\"]",
  "github_source_repo": "${SOURCE_REPO:?}",
  "github_commit": "${SOURCE_SHA:?}",
  "mock-case-image": "${CASE_IMAGE:?}",
  "cancel-in-progress": "false"
}
EOF
a1 ci pipeline run 1346 --repo "$CI_REPO" --branch "$RECIPE_REF" \
  --params-file "$PARAMS_FILE" -f json
```

同一次运行会先执行一次 `mock-ci-case-build`。共享构建验证 Case 文件并按 runner 规格分组，随后为每个 Case 展开独立 Job。各 Job 可以并行，独立上传报告。新增 Case 后直接传它的 ID 即可。

Case Job 把 Java 运行包解到结果目录旁的 `<输出目录>.runtime`，包内 classpath 使用相对路径。启动配置生成时通过 `--runtime-root` 指定解包位置，并通过 `--scrape-interval` 设置采样间隔。运行包留在 Job 工作区。报告目录保存 `exit-code`、生成的 `launch.yaml` 和 Case 的 `result/` 目录。

## 3. 下载和查看 Case 报告

将触发返回的 Run ID 填入 `RUN_ID`。先用 `run get` 回读源码 SHA、Case 列表和运行状态，再用 `job list` 查看共享构建及各 Case 的状态。同一资源组的 Job 名相同，`run-case` 日志首行会打印对应的 Case ID。

```sh
a1 ci run get "$RUN_ID" --repo "$CI_REPO" -f json
a1 ci job list --run "$RUN_ID" --repo "$CI_REPO" -f json
a1 ci artifact list --run "$RUN_ID" --repo "$CI_REPO" -f json
```

从制品列表复制目标 Case 的完整 `name`，填入 `ARTIFACT_NAME`。报告名称格式为 `mock-case-<Case ID>-<配方提交>`，`mock-case-runtime-` 开头的是共享 Java 运行包。Case 执行失败时也可下载已上传的报告。

下载文件是 AoneCI 的 ZIP 归档，内含 `mock-case-result.tar.gz`。每个 Case 解包到自己的目录。

```sh
REPORT_DIR=$(mktemp -d)
a1 ci artifact download "$ARTIFACT_NAME" --run "$RUN_ID" --repo "$CI_REPO" \
  --output "$REPORT_DIR/artifact.zip"
unzip -p "$REPORT_DIR/artifact.zip" mock-case-result.tar.gz | tar -xzf - -C "$REPORT_DIR"

RESULT_DIR="$REPORT_DIR/mock-case-result/result"
cat "$REPORT_DIR/mock-case-result/exit-code"
cat "$RESULT_DIR/result.json"
```

- `result.json` 是 Case 总结，包含运行状态和错误。若在 Case 启动前失败，先看 Job 的 `run-case` 日志和报告里的 `exit-code`。
- `result/configuration.json` 保存编译后的 Case 和实际使用的启动配置。
- `gates.jsonl` 记录每项门禁的实际值和判定，`events.jsonl` 记录运行时间线。
- 如需回看曲线，保留完整的 `result/` 目录，将 `RESULT_DIR` 传给 [View 指南](view-guide.md) 中的 `mock-view`。曲线依赖报告中保留的 Prometheus 数据。

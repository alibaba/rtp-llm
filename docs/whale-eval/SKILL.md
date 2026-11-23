---
name: whale-eval
description: 基于用户给出的 Whale 基准部署，使用 dashctl 为指定模型一键拉起 ModelBoard 功能、精度和稳定性评测。锁定源快照，比较中卫、乌兰察布和上海的实际预检余量，跟踪评测启动并返回 Whale、大眼猴和 ModelBoard 链接。用于拉起或接续核验这类评测。
---

# Whale 一键评测

用户给 Whale 部署链接和目标模型即可启动。已有请求中的版本、nightly/weekly、机房、GPU 型号、行数和场景限制继续有效。默认选该模型的标准 CI-Test 套件，跑 `func,effective,stab`。模型以用户指定值为准，部署名称只用于关联。没有明确行数时沿用基线，另有要求时预检对应覆盖值。

## 执行

1. **锁定源。** 使用当前环境的 `dashctl`、ModelBoard Token、Whale/Spectrum/大眼猴权限。按照 [dashctl 与快照](references/dashctl.md) 从源部署发布历史取得确切部署快照，核对版本、主模型与 Decode/SP checkpoint、Frontend 镜像覆盖和 Master 配置。固定快照、Model/Mint/CI-Test 版本。源部署只读。
2. **选择资源。** 对同一冻结配置和完整测试组合执行 [机房预检脚本](scripts/choose_region.py)。默认并行比较中卫 `cn-zhongwei`、乌兰察布 `cn-wulanchabu`、上海 `cn-shanghai`。用户指定机房或机型时限制候选。选择能满足全部子任务、Whale GPU 瓶颈余量最大的候选。池名以当次预检为准。容量是查询时的观测值，实际分配看调度。没有可用候选时，报告需求、已知余量和无法验证的候选。
3. **提交。** 用刚通过预检的精确参数及唯一 `--request-id` 提交。常见情况是一次提交生成两个子任务，功能与精度一组、稳定性一组。以服务端预检出的分组为准。如果平台要求分开提交，核对两组的完整需求再分别启动。指定稳定性场景时，按 [场景核验](references/dashctl.md#场景核验) 确认范围。
4. **跟到实际启动。** 根据作业和子任务 ID 跟踪资源分配、部署就绪和大眼猴子任务。 `RUNNING/ApplyForResources` 仍在申请资源，`TestExecute` 仍可能在等待 P/D 可用。观察到对应大眼猴执行及下游用例后才报评测已启动。遇到已接受的提交超时或排队，先查原任务及事件，避免重复提交或替换已有任务。
5. **交付链接。** 按 [链接验证](references/links.md) 给每组一行，包含实际 Whale 部署、大眼猴执行、ModelBoard 调度详情。无调度详情时，给命中该任务的 ModelBoard 历史执行记录。尚未生成的链接标明当前阻塞。用户另要 Mint 时一并给出已核对的部署链接。

除非用户要求调整，只验证新部署继承的 Master、镜像、checkpoint 和角色配置。旧评测、源部署和生产基线均应通过各自的显式操作另行处理。

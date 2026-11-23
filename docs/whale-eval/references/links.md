# 启动和链接验证

```sh
dashctl describe cirun "$JOB_ID" --events=none --results --resources=false -o json
dashctl describe cirun "$JOB_ID" --events --results=false --resources=false -o json
dashctl logs cirun "$JOB_ID" --subtask "$SUBTASK_ID" --stage deploy --tail 100 -o json
```

读取 `.execution.subtasks[]`，用 `subtask_id` 关联 `.deployments.links[]` 和 `.results.subtasks[]`。每组同时记录 job ID、`pipeline_run_id`、测试类型和实际地区。阶段还未生成部署或报告时，`describe` 可能非零退出，结构化回执仍可提供其他已知链接。 `RUNNING` 只表示作业活跃。最终确认大眼猴执行 ID 与流水线匹配，至少一个可见的下游用例已执行或结束，再报评测启动。指定场景时核对所有子项的实际名称、类型和数量。

## Whale 与 Mint

- `kind=whale_service` 的 `name` 是 Whale 服务 ID，有些版本将这个 ID 拼进了部署 URL。打开服务详情 `https://whale.online.alibaba-inc.com/inference/online/service/operate/<service-id>/detail`，在部署列表找到和该子任务的地域、池、Spectrum 应用一致的部署。复制真实部署行的 `/inference/online/deployment/operate/<deployment-id>/operation`，打开核对服务名、行数和状态。部署名随业务变化。
- 浏览器以外可使用授权的 Whale CLI 或 `whale-biz-operation` 技能中已文档化的服务部署列表查询。由真实部署对象的 ID 和所属服务验证关联。
- Mint 优先用流水线或部署日志返回的精确 Spectrum deployment 路径。只有应用地址可验证时，把它标为 Mint 应用入口。

## 大眼猴

- 从同一子任务的 `.results.subtasks[].source_ids.{req_id,curl_task_id}` 或 `.execution.subtasks[].chimp_url` 取得两个 ID。旧地址的 `reqId` 与 `id` 对应大眼猴 3.0 的 `https://chimp3.alibaba-inc.com/test-requirements/requirementManage?reqId=<req-id>&currentPage=1&id=<curl-task-id>`。
- 打开后核对执行行 ID、关联部署/流水线、测试类型和执行状态，避免同一需求下点到历史执行。再展开查看用例或确认实际启动。需求 ID、本次执行 ID 和单用例 ID 要分别使用。

## ModelBoard

- 优先使用同一子任务的 `report_url` 或经过核对的数字 `schedule_run_id`。调度页形如 `https://modelboard.alibaba-inc.com/cicd/report/schedule/<schedule-run-id>`。打开并核对页面中的 CI job 和 Spectrum Run。
- 调度页未生成时，打开 [ModelBoard CI 历史查询](https://modelboard.alibaba-inc.com/cicd/v2/report/history)，按完整 job ID 查询并确认有该任务和流水线的执行记录。分享界面生成的结果 URL（带 `taskId` 与覆盖触发时间的时间范围），标为执行记录。
- `cijob-...`、Demand ID 和 Spectrum Run ID 均不能代替 `schedule_run_id`。普通网页访问也不能替代带 Bearer Token 的状态 API。若只查到 API 记录，就报真实阶段及缺少的页面。

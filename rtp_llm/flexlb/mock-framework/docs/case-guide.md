# 编写 Case 与扩展接口

启动已有 Case 见 [README](../README.md#跑第一个-case)。本指南用于新增或评审 Case，以及在现有动作不足时扩展接口。

## 新增 Case

1. 在 `cases/<case_id>/` 添加 `case.yaml`。基础收发参考 [Basic](../cases/basic/case.yaml)，故障恢复参考 [Master HA](../cases/master_ha_failover/case.yaml)，吞吐和延迟门禁参考 [性能回归](../cases/performance_regression/case.yaml)。
2. 把本 Case 的角色配置放在 `configs/`，用 `config: configs/master.yaml` 等本地相对路径引用。借用其他 Case 的配置时先复制到本目录。少量参数可内联，空配置用 `config: {}`。
3. 流量数据放在共享的 `data/traffic/`，在 `client/start.args.requests_file` 引用 `.mckidx`。以 `data/` 开头的路径按共享根解析，默认是框架目录，可用 `--shared-root` 指定。其他相对路径从 Case 目录解析。Client 共用同一份数据，各自按索引读取。[回放格式](architecture.md#共享索引回放输入)
4. 按实验目的划分 Phase。用已有接口描述操作和可观测的通过条件。使用默认大盘时省略 `view`，自定义大盘才显式声明路径。[默认大盘](../views/README.md)
5. 提交前在框架目录运行 `uv run --locked mock-case compile cases/<case_id>/case.yaml`。编译检查 YAML、组件配置、接口参数和引用。再按 README 运行完整 Case，核对门禁和请求证据。

## YAML 规范

### Case、Phase 和 Step

| 层次 | 必填字段 | 可选字段 |
| --- | --- | --- |
| Case | `id`、`describe`、`environment`、`phases` | `name`、`view` |
| Phase | `id`、`describe`、`timeout`、`steps` | |
| Step | `id`、`describe`、`interface` | `args`、`expect`、`within`、`wait_after`、`background` |

`environment` 按角色声明 `master`、`prefill`、`decode`、`vit`、`client`。P/D 必须写 `engine: rtp`。`deployment.replicas` 默认 1，表示独立 JVM 数。Client 的 `qps`、`total_concurrency` 是所有副本的合计，由 `client/start` 的参数指定。本地启动器会拒绝当前无法落实的 `resources` 和 `heap` 限制。

`config` 文件的内容就是角色配置 mapping。Master 的 `FLEXLB_CONFIG` 和 `MODEL_SERVICE_CONFIG` 使用原生 JSON 字符串。Mock 点分键由编译器展开。重复键、YAML merge 和点分路径冲突会报错。[组件参数](component-reference.md#engine-配置规则)

ID 使用小写 `snake_case`，Case、Phase 和 Step 的 ID 在同一个 Case 中全局唯一。Case ID 与目录同名，Phase ID 描述阶段。Step ID 写成“动作 + 对象”，例如 `start_client`、`get_master_a`、`check_prefill_tps`、`check_client_success_rate_after_drain`。阈值放进 `expect`，场景意图放进 `describe`。改名时同步修改引用。

### 引用、断言与时间预算

`{ref: steps.<id>}` 引用前置 Step 的完整返回值，`{ref: steps.<id>.<field>}` 引用它的字段。实时目标列表可用 `{ref: globals.masters}`、`globals.prefill`、`globals.decode` 或 `globals.vit`。全局列表在当前 Step 取值，已保存的返回值仍指向原地址。支持 `targets` 的接口通常在省略时选择操作域内的全部实例，显式空列表、重复或无效地址按接口报错。

`expect.eq` 比较结构和值，列表顺序参与比较。`count` 检查列表长度。`gte` 和 `lte` 检查数值，可组合成上下限。省略 `expect` 时只执行接口并保存返回值。

```yaml
phases:
  - id: startup
    describe: 确认主 Master 并记录它发现的 Prefill
    timeout: 15s
    steps:
      - id: check_master_leader
        describe: 确认唯一主节点，等待发现传播
        interface: master/get
        args: {is_leader: true}
        expect: {count: 1}
        within: 10s
        wait_after: 2s
      - id: get_leader_prefill_engines
        describe: 读取主 Master 看到的 Prefill 地址
        interface: master/engine/get
        args:
          targets: {ref: steps.check_master_leader}
          role: prefill
```

- `timeout` 是整个 Phase 的预算，`within` 是单步上限，取其与 Phase 剩余预算的较小值。两者都使用正的带单位时长。
- `wait_after` 在接口与断言成功后执行，默认 `0s`，计入 Step 和 Phase 预算。它不刷新结果。要验证等待后的状态，另写一个读取 Step。
- Step 默认按顺序执行。`background: true` 允许与同 Phase 的其他 Step 并行，Phase 结束前会收齐或取消这些任务。后台 Step 的结果不可用于 `ref`。

## 用现有接口写场景

`client/start` 开始整轮回放，固定 `qps` 和 `replay_speed` 二选一。需要缓存复用时在 start 设置 `reuse_cache: true`，同时启用 Engine 的缓存复用。性能 Case 先预热，再检查预热后的指标区间。`client/stop` 收集整轮结果，随后可用 `client/requests` 检查逐请求记录。[Client 回放与请求记录](architecture.md#client-如何回放)

`engine/scale` 通过 `direction: out` 扩容或 `direction: in` 缩容，返回实际变更的实例地址。缩容可用 `targets` 指定实例。需要验证发现收敛时，另用 `master/engine/get` 查询 Master 的实际视图。

Master 故障用 `master/stop`，默认正常停止，需要强制退出时写 `mode: kill`。`master/restart` 在原地址恢复已停止的实例，`master/start` 新增实例并返回新地址。HA Case 先保存实际选举出的主备地址，按地址注入故障，再查询主节点和流量成功率。操作回执只说明动作已经完成，发现、选举和业务恢复分别用后续 Step 验证。

### 指标门禁

优先用语义接口和绝对阈值。指标范围相对当前 Step 固定的查询时刻，例如 `range: [-60s, 0s]`。多数接口省略范围时查询最近 60 秒。`client/success_rate` 显式选择 `range` 或引用 `client/stop` 的 `playback_result`。

- 默认 `interval` 返回整个区间的一个值，Case 省略该字段。命中率和成功率用区间内分子、分母的总增量相除。吞吐和延迟分别按接口定义计算。
- `mode: series` 返回按采集周期排列的数值，`gte` 和 `lte` 要求每个点都满足。Counter 派生指标每点的滚动长度由实现确定。
- 缺失样本、目标缺数据和零分母属于查询错误。自定义诊断可用 `prometheus/query`，返回原始 instant query 的样本，数值门禁优先用现成指标接口。

| 接口 | 返回口径 |
| --- | --- |
| `client/qps` | 已发请求速率，requests/s。 |
| `engine/prefill/tps`、`engine/prefill/tps_with_cache` | Prefill 有效执行速率，分别只计实际计算 token、计入复用 token，单位为 tokens/s。 |
| `engine/decode/tps` | Decode 已提交输出 token 的墙钟速率，单位为 tokens/s。 |
| `engine/cache/hit_rate` | Prefill 复用 token / 输入 token，比率为 0–1。 |
| `client/success_rate` | 成功终态 / 全部终态，比率为 0–1。整轮模式要求没有未发请求。 |
| `client/ttft`、`client/tpot` | Client 实际收到首 token 的耗时，以及成功请求首 token 之后的平均每 token 耗时。两者都支持 `mean`（默认）、`p95` 和 `p99`，返回单位为秒，阈值可写 `lte: 200ms`。 |

## 新增 Case 接口

用现有接口组合完成的场景留在 YAML。只有缺少可独立复用的操作或查询时，才在 `python/mock_framework/execution/step/` 新增动作。

1. 在动作文件定义带类型注解的参数 dataclass 和异步函数。参数默认值写在 dataclass，跨字段约束写在 `__post_init__`。用 `@register_step` 声明稳定的接口名、`effect="read"` 或 `effect="write"`，数值接口声明单位。
2. 将函数加入 `execution/step/__init__.py` 的 `standard_actions()`。现有 `Step` 负责解析引用、校验参数和结果、执行断言与等待。动作通过 `StepContext` 使用共享查询、角色管理和剩余预算。
3. 在动作或角色管理层处理目标选择和业务状态。Runner 负责时间预算及整场收尾，Java 端点提供实例能力。写操作发出后未取得有效回执，Step 记录 `outcome_unknown`，禁止自动重放。
4. 返回值供 YAML 引用和断言使用。单值指标返回数值，逐点指标返回数值列表，实例查询保留地址或按地址组织的结果。在动作旁的测试验证参数、返回值和失败语义，用实际 Case 验证接线。

[现有动作示例 `master/get`](../python/mock_framework/execution/step/master_get.py) · [Step 注册和类型校验](../python/mock_framework/execution/step/step_registry.py) · [Case 执行架构](architecture.md#case-如何执行)

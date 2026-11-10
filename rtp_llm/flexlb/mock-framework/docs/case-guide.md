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

| 层次  | 必填字段                                  | 可选字段                                               |
| ----- | ----------------------------------------- | ------------------------------------------------------ |
| Case  | `id`、`describe`、`environment`、`phases` | `name`、`view`                                         |
| Phase | `id`、`describe`、`timeout`、`steps`      |                                                        |
| Step  | `id`、`describe`、`interface`             | `args`、`expect`、`within`、`wait_after`、`background` |

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
        args: { is_leader: true }
        expect: { count: 1 }
        within: 10s
        wait_after: 2s
      - id: get_leader_prefill_engines
        describe: 读取主 Master 看到的 Prefill 地址
        interface: master/engine/get
        args:
          targets: { ref: steps.check_master_leader }
          role: prefill
```

- `timeout` 是整个 Phase 的预算，`within` 是单步上限，取其与 Phase 剩余预算的较小值。两者都使用正的带单位时长。
- `wait_after` 在接口与断言成功后执行，默认 `0s`，计入 Step 和 Phase 预算。它不刷新结果。要验证等待后的状态，另写一个读取 Step。
- Step 默认按顺序执行。`background: true` 允许与同 Phase 的其他 Step 并行，Phase 结束前会收齐或取消这些任务。后台 Step 的结果不可用于 `ref`。

## 用现有接口写场景

先从同类 Case 复用步骤。基础收发在回放结束后检查逐请求结果。性能回归在预热后检查绝对阈值。故障恢复先保存实际选举出的主备地址，再按地址执行故障步骤。操作回执之后，用独立的读取 Step 验证发现、选举和流量是否恢复。[Basic](../cases/basic/case.yaml) · [性能回归](../cases/performance_regression/case.yaml) · [Master HA](../cases/master_ha_failover/case.yaml)

接口名以 `standard_actions()` 注册的动作列表为准。每个动作的参数类型、默认值、返回类型和单位由函数签名及 `@register_step` 声明。门禁记录在 `gates.jsonl` 中保存当次检查的单位。[已注册的 Step](../python/mock_framework/execution/step/__init__.py)

### 指标门禁

优先选择有明确单位和聚合方式的指标动作，用 `expect` 写绝对阈值。查询范围相对当前 Step 的固定查询时刻。比如 `range: [-60s, 0s]` 表示截至该时刻的最近 60 秒。安排预热和等待时，把所选窗口放在需要观察的阶段。

支持 `mode` 的指标动作默认汇总整个区间。比例类指标分别汇总分子和分母，再相除。需要每个采样点都达标时，使用 `mode: series`，数值上下限会逐点检查。缺少样本或分母为零按查询错误处理。具体接口支持的参数和单位查动作源码，指标采样边界见 [指标覆盖](../views/coverage.md#指标口径)。

## 新增 Case 接口

用现有接口组合完成的场景留在 YAML。只有缺少可独立复用的操作或查询时，才在 `python/mock_framework/execution/step/` 新增动作。

1. 在动作文件定义带类型注解的参数 dataclass 和异步函数。参数默认值写在 dataclass，跨字段约束写在 `__post_init__`。用 `@register_step` 声明稳定的接口名、`effect="read"` 或 `effect="write"`，数值接口声明单位。
2. 将函数加入 `execution/step/__init__.py` 的 `standard_actions()`。现有 `Step` 负责解析引用、校验参数和结果、执行断言与等待。动作通过 `StepContext` 使用共享查询、角色管理和剩余预算。
3. 在动作或角色管理层处理目标选择和业务状态。Runner 负责时间预算及整场收尾，Java 端点提供实例能力。写操作发出后未取得有效回执，Step 记录 `outcome_unknown`，禁止自动重放。
4. 返回值供 YAML 引用和断言使用。单值指标返回数值，逐点指标返回数值列表，实例查询保留地址或按地址组织的结果。在动作旁的测试验证参数、返回值和失败语义，用实际 Case 验证接线。

[现有动作示例 `master/get`](../python/mock_framework/execution/step/master_get.py) · [Step 注册和类型校验](../python/mock_framework/execution/step/step_registry.py) · [Case 执行架构](architecture.md#case-如何执行)

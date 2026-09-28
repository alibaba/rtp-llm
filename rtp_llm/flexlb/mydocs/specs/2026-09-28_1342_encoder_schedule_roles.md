# Spec: FlexLB Encoder 按角色选点与分阶段请求生命周期

## Goal

- `Schedule` 接受 `schedule_roles`，只为指定角色返回选点；未传时仍按已配置角色决策。
- 新增独立的 PAI-vLLM `ENCODER` 角色及基于并发、可用 KV cache 的选点策略，保留 RTP-LLM `VIT` 的现有随机策略。
- 同一个业务 `request_id` 的 Encoder 阶段和 Generation 阶段在 FlexLB 内分别记录、查询与取消。

## Done Contract

- gRPC 协议、配置角色、路由、WorkerStatus 同步与请求生命周期均支持 Encoder；现有 Generation 调用在不传新字段时保持原行为。
- 测试先复现角色过滤、Encoder 选点、两个阶段同 ID 共存与生命周期变化，再通过相关模块测试和全仓 Spotless 检查。

## Scope

- In: `rtp_llm/flexlb` 内的协议、Java 实现、测试、必要的架构文档和接口 Javadoc。
- Out: Frontend、PAI-vLLM WorkerStatus 上报实现、引擎 Cancel RPC、Encoder 批量派发、Encoder 抢占。
- 用户已切分的任务单元: 协议与角色选择、Encoder 策略、分阶段生命周期。
- 轻量评估: 涉及多个 FlexLB 模块和状态机；仍在这一个明确功能范围内。

## Facts / Constraints

- 文档协议：`ScheduleRolePB` 包含 `UNSPECIFIED=0`、`ENCODER=1`、`PREFILL=2`、`DECODE=3`、`PDFUSION=4`；`FlexlbScheduleRequestPB.schedule_roles=17`。响应沿用 `server_status=4` 的角色端点列表，Encoder 角色字符串为 `ENCODER`。
- 显式角色列表按集合处理并去重，实际处理顺序沿用模型已配置的角色顺序。未配置的显式角色、`UNSPECIFIED` 和未知枚举值返回 `INVALID_REQUEST`。
- Encoder 使用常规 `GetWorkerStatus`；仅健康且 `available_kv_cache >= 0` 的 Worker 入围。先选 `running_query_len + waiting_query_len` 最小者，并列时选可用 KV cache 最大者。无候选返回 Encoder 资源不足错误。
- 选中后、WorkerStatus 尚未反映请求前，FlexLB 要把本地待观察请求计入 Encoder 并发；观察到运行、完成或失活时对账，避免持续重复计数。
- Encoder 只走 DIRECT，FlexLB 返回端点，Frontend 执行请求。`VIT` 的随机策略保持原样。
- Encoder 生命周期遵循现有 DIRECT 请求状态变化：端点接管请求时进入 `DISPATCHING`，确认路由成功响应发布时进入 `ACKNOWLEDGED`，随后处理 WorkerStatus 中运行/完成/失败及失活兜底。`running_task_info` 与 `finished_task_list` 是 Worker 事实来源。
- 请求记录以业务 `request_id` 加阶段区分。`Cancel` 与 `GetRequestState` 新增 phase；未传或默认值指向 `GENERATION`，显式 `ENCODER` 指向 Encoder。Encoder Cancel 只更新 FlexLB 本地状态，不发引擎取消 RPC。Encoder 不填抢占 ID。
- 当前使用流程是先单独请求 Encoder，处理完后再请求 Generation；已有 Generation 路径为 Prefill、Prefill + Decode、PDFusion。文档和 Javadoc 说明该流程，不新增角色组合白名单，也不对混合 Encoder/Generation 请求显式抛异常。
- 无 `schedule_roles` 时仍按协议尝试所有已配置角色；不为混合阶段调用承诺额外的端到端语义。
- DashLLM 的 `flexlb-batch-qwen-3-8-flash-multimodal` 分支已有 Encoder 场景的 WorkerStatus 负载和任务测试。本次对齐后，PAI-vLLM EPD 的 `is_vit_node` 上报 `RoleType.ENCODER`；WorkerStatus 的 `RoleTypePB` 新增 `ROLE_TYPE_ENCODER = 5`，字符串与 typed enum 一致。RTP-LLM 的 VIT 仍是独立角色。
- `RoleAddrPB.RoleType` 在现有 0-4 值后追加 `ENCODER=5`，Java 转换器可对 Encoder 双写枚举和 `role_str`；现有角色的 wire 编号保持不变。RTP-LLM C++ 当前只消费原有角色，Encoder 地址仍只由支持该枚举的客户端消费。

## Open Questions

- 无阻塞项。实现时按现有 protobuf 与 Java 模型命名习惯决定 phase 枚举和 Encoder 资源不足错误的具体常量名及编号。

## Restated Understanding

- 我理解当前任务是：FlexLB 新增 Encoder 角色和策略，让客户端按 `schedule_roles` 请求选点，按阶段维护相同业务 ID 的任务。
- 当前核心目标是：Encoder 与 Generation 的决策和生命周期互不覆盖，原有 VIT 及 Generation 行为兼容。
- 当前边界是：只改 FlexLB；Encoder 使用直接返回端点的调度方式。
- 暂不处理：Frontend 与 Worker 实现、混合阶段调用的完整语义、Encoder 批量派发及抢占。

## Goal Alignment Check

- 当前动作是否仍服务于核心目标：是，先固定已确认契约及测试边界。
- 模型当前路径是否仍在用户边界内：是，仅覆盖 FlexLB。
- 是否出现更适合代码地形的水流路径：暂无；实现中优先复用现有 DIRECT 状态机。
- 若否，偏差在哪里：无。
- 是否需要调整本轮目标或范围：否。

## Checkpoint Summary

- 当前任务理解：按角色选点，新增 Encoder，并以 phase 区分同一业务 ID 的生命周期。
- 当前核心目标：FlexLB 支持分两次请求 Encoder 与 Generation，且保持现有调用兼容。
- 当前进度：FlexLB 实现、单元测试、架构文档、Javadoc、全量测试和代码评审已完成。
- 下一步 1: 将 FlexLB 的新 proto/角色读取能力先于 DashLLM Encoder 上报能力部署。
- 下一步 2: 接入真实 Encoder 节点后验证端到端选点和生命周期。
- 涉及文件 / 模块：`flexlb-grpc`、`flexlb-common`、`flexlb-sync`、`flexlb-api` 及 `docs/architecture/`。
- 风险：现有 RequestRegistry 只以 request ID 索引；现有 RouteAdmission 依赖 Prefill，Encoder 不能直接复用这一前提。
- 验证方式：针对性 JUnit 测试、相关 Maven reactor 测试、全仓 `./mvnw spotless:check -Pspotless-check`。
- Execution Approval: `Approved`（用户于 2026-09-28 回复“开始实施”）

## Change Log

- 2026-09-28: 记录逐项确认的协议、策略、生命周期与测试边界；用户批准实施。
- 2026-09-28: 完成 FlexLB 代码与测试；根据用户反馈改为无全局 Encoder 选点锁、ConcurrentHashMap 记录在途请求，并删除旧的测试专用 DefaultRouter 构造方法。核对 DashLLM 和 PAI-vLLM 分支的 WorkerStatus 实现。
- 2026-09-28: 按用户确认的 PAI-vLLM EPD 角色语义，追加 `ROLE_TYPE_ENCODER = 5`，并让 DashLLM 的 Encoder 节点双写一致的字符串和 typed enum；FlexLB 转换器与跨语言 wire 测试同步更新。
- 2026-09-28: 按 DashLLM Encoder 容量语义允许 `available_kv_cache = 0` 入围，继续过滤负数；保留并发优先、可用 KV cache 并列决胜。
- 2026-09-28: 对齐 Prefill DIRECT 生命周期，将 Encoder 端点接管与成功响应发布拆开；两者分别推进 `DISPATCHING` 和 `ACKNOWLEDGED`，并验证中间取消不会发布成功响应。
- 2026-09-28: 应用户要求，补齐 `RoleAddrPB.RoleType.ENCODER=5` 与 Java 双向转换；验证现有角色 wire 编号不变及 Encoder 新值往返。

## Validation

- Self-check: 独立 Encoder 角色、DIRECT 生命周期、Generation 默认 phase 和 VIT 不变均已按规格落地；混合阶段调用沿用现有选点路径，未定义完整混合生命周期。
- Static checks: `git diff --check` 与全量 `./mvnw spotless:check -Pspotless-check` 均通过。
- Runtime / Test: 最终并发容器改动后全量 `./mvnw test` 通过；Mock Engine 396 项测试全部通过。代码评审未发现阻断项。
- Human confirmation: 已确认实施。
- 结果汇总：FlexLB 实现、测试、架构文档和静态检查均完成。
- 核心目标是否已由证据证明完成：FlexLB 内部行为及 DashLLM Encoder role wire 对齐已由自动化测试证明；真实跨仓库端到端行为仍待部署验证。
- 若未完成，当前剩余差距：真实部署验证。
- 剩余风险：旧版 FlexLB 不识别 typed enum 5，部署时应先升级 FlexLB；真实 Encoder 节点的选点与生命周期仍需部署验证。
- `RoleAddrPB` 后续补齐：先观察到 Encoder 转换测试因缺少枚举而失败；追加 `ENCODER=5` 后，定向 `RoleAddrProtocolCompatibilityTest`、全量 `./mvnw test`、全仓 `./mvnw spotless:check -Pspotless-check` 与 `git diff --check` 均通过。旧角色 0-4 的编号有固定值断言；旧版读取器仍不能消费新的 Encoder 地址。

## Resume / Handoff

- 当前状态：FlexLB 范围已完成。
- 当前卡点：无。
- 下一步唯一动作：交付跨仓库 role 对齐结果与剩余容量字段验证条件。
- 下一轮核心目标：真实部署时验证独立 Encoder WorkerStatus。

## Project Sync Candidates

- 是否发现可复用项目事实：Yes。
- 候选事实：Encoder 与 VIT 分属不同引擎角色；Encoder 使用两阶段请求和 DIRECT 决策。
- 建议同步位置：`docs/architecture/` 对应稳态架构文档及接口 Javadoc。
- 同步状态：已更新 `docs/architecture/` 对应文档。

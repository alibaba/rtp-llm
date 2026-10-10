# FlexLB 调度重构文档

**当前开发入口：[请求调度与交付责任：类图、成员、API 和实现约束](target-class-api-review.md)（2026-10-03）。** 本轮目标是删除 RequestLifecycle，保留四方法 RequestScheduler 和 Direct/Queued 实现，将身份索引、单请求裁决、交付清理及资源账本划清边界，并补齐“确定不会 fetch 必须及时取消 Prefill”的协议。该稿是待实施契约，不是已验收实现。

[此前的设计改进评审稿](round-design-review.md)保留作为历史背景。其 PlacementStrategy、DirectPlacementCoordinator、GlobalQueueCoordinator 和不使用抽象基类的建议已经被后续讨论替代，不作为当前请求调度的开发依据。

整体设计先读[架构边界与设计约束](architecture-contract.md)：这是待落实的系统级提案，区分接入、应用编排、业务内核、技术适配及装配，规定依赖和所有权。类级交接不能替代这些约束。

类与接口评审见[目标类图和 API](target-class-api-review.md)，包含核心成员、scope、协作方法及拟收拢的协议入口。

此前请求内部迁移见[FlexLB 请求职责重构开发交接](request-ownership-handoff.md)。它基于 2026 年 10 月 2 日工作区，记录历史字段迁移和行为约束；当前迁移去向、实现批次和新增取消协议以当前开发入口为准。

本轮保留当前 `BalanceContext` 和七阶段状态机。与本轮冲突的旧类名、五阶段迁移及拆分建议不作为开发依据。[此前的整体目标方案](final-design.md)保留供历史背景查阅，不能直接据此恢复已删除对象。

历史评审附录：

- [类时序与状态转移](sequence-flows.md)：按类展开五个关键交接，图描述目标职责。
- [复审前的 15 张线程链路图](thread-model-detailed-flows.md)：原图及文字完整保留；首页标注已否决的旧边，不能直接据此实施。
- [枚举现状审查快照](appendix-enum-inventory.md)：旧源码基线上的枚举分类与所有者清单，仅作迁移核对；目标取舍以最终方案为准。

局部实施与性能记录归档在 [evidence](evidence/)；它们记录已运行的验证及未通过项，不是新设计版本。

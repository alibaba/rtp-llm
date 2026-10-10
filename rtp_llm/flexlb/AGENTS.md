# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## 可读性：核心开发与审核要求

可读性是 FlexLB 的核心质量要求，适用于生产代码、测试、注释和文档。
判断标准是：第一次接触这段代码、了解基本业务的读者，能直接看懂变量存了什么、
状态表示什么、方法做了什么，以及为什么需要这个分支，而不必依赖作者口头解释或追踪多层实现。
代码能运行、测试通过，不代表满足可读性要求。

- **用具体、常见的词表达用途。** 名称优先说明业务对象和实际动作，不使用显得高级但含义
  模糊的词汇。不要单独用 ownership、terminal、settlement、proof 等抽象词概括不同职责。
  必须使用协议或领域术语时，在首次出现处用简单语言说明它对应哪个字段、对象或行为。
- **变量名要说明保存的内容，状态名要说明正在发生的事。** 布尔值应能读成一个明确判断，
  例如“是否正在等待引擎的 Finished 上报”。相较于 `awaitingTerminal`，
  `waitingForWorkerFinishedReport` 更直接地说明等待谁、等待什么。
  状态字段还应说明何时设置、何时清除，以及对调度、容量统计和请求清理的影响。
- **方法名必须与实际效果一致。** 只清除容量记账状态的方法，不能用
  `clearRequestOwnership` 这样的名字暗示整个请求的关联都被清除。
  反过来，名称只表示修改一个字段的方法，也不应隐含删除记录、释放资源或通知调度器等操作。
  优先修正命名和职责，不用长篇注释解释一个具有误导性的名字。
- **不同含义分开表达。** “引擎当前是否在执行”“是否占用容量”“是否收到完成反馈”
  “是否保留请求关联”是不同问题，不能用一个含糊的状态或多个难以理解的布尔组合混在一起。
  状态转换应能从相邻代码中看清条件和结果，避免为了显得抽象而增加跳转层次。
- **注释解释业务原因和约束。** 用具体对象描述保留什么、释放什么、等待什么；必要时给一个
  简短请求示例。避免用另一组抽象术语重复代码，也不要依赖修改历史才能理解当前行为。
- **控制流代码分行写。** 新增或修改 Java 代码时，`if`、`for`、`while` 等控制流的条件、
  执行语句和闭合花括号分别占行；即使只有一条 `return` 或 `continue`，也不写成
  `if (condition) { return value; }`。审核当前改动时清除这种单行分支，不批量重排无关代码。
- **审核时逐项检查可读性。** 对新增或修改的名称、状态和方法，检查读者能否仅根据命名及
  附近代码准确判断用途和影响。需要反复追问“这个变量是干什么的”或“这个方法到底清掉什么”
  才能理解时，应调整命名或结构后再提交审核。规则适用于当前改动，不要求顺带重构无关代码。

## Project Overview

FlexLB is a high-performance, intelligent load balancer for AI model inference workloads
(part of RTP-LLM). Multi-module Maven project on Java 21 / Spring Boot 2.7.18 (WebFlux
reactive architecture).

Modules: `flexlb-api` (web layer), `flexlb-common` (shared models/config), `flexlb-grpc`
(gRPC stubs), `flexlb-sync` (core load balancing logic), `flexlb-cache` (KV cache
management).

## Architecture Docs

架构设计文档在 [docs/architecture/](docs/architecture/00-overview.md)（稳态架构文档，
描述当前代码长什么样；架构变了必须同步更新）：

| 文档 | 内容 |
|---|---|
| [00-overview](docs/architecture/00-overview.md) | 模块划分、技术栈、请求主链路、核心不变量 |
| [01-routing-and-balancing](docs/architecture/01-routing-and-balancing.md) | Router / LoadBalancer、角色多阶段路由、回滚、策略 |
| [02-queue-scheduling](docs/architecture/02-queue-scheduling.md) | PriorityScheduler / WorkerBatcher 调度、请求生命周期 |
| [03-resource-management](docs/architecture/03-resource-management.md) | EndpointRegistry、worker 可用性与本地资源预留 |
| [04-worker-sync-and-cache](docs/architecture/04-worker-sync-and-cache.md) | Worker 状态同步、KV cache 索引 |
| [05-lifecycle-and-consistency](docs/architecture/05-lifecycle-and-consistency.md) | 优雅上下线 Hook、ZooKeeper 主选举 |
| [06-configuration-and-observability](docs/architecture/06-configuration-and-observability.md) | 环境变量配置、HTTP 端点、监控指标 |

## Build Commands

从 `rtp_llm/flexlb` 目录使用 Maven Wrapper：

```bash
# Build entire project (skipping tests)
./mvnw clean package -DskipTests

# Build a specific module
./mvnw clean package -pl flexlb-sync -DskipTests

# Build a module with its dependencies
./mvnw clean package -pl flexlb-api -am -DskipTests

# Full build with tests
./mvnw clean install

# Run all tests
./mvnw test

# Run tests for a specific module
./mvnw test -pl flexlb-sync -am

# Run a single test class
./mvnw test -Dtest=DefaultRouterTest

# Run a single test method
./mvnw test -Dtest=DefaultRouterTest#testRouteSuccess

# Check code formatting
./mvnw spotless:check -Pspotless-check

# Auto-format code
./mvnw spotless:apply -Pspotless-check
```

## Run Application

```bash
java -jar flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar \
  --server.port=7002 \
  --management.server.port=8804 \
  --spring.profiles.active=test
```

运行模型服务时需要 `MODEL_SERVICE_CONFIG`；`FLEXLB_CONFIG` 可省略并使用有效默认值。
ZooKeeper 主选举通过 `FLEXLB_CONFIG.consistency` 配置；不存在独立的一致性行为环境变量。字段说明见
[06-configuration-and-observability](docs/architecture/06-configuration-and-observability.md)。

JVM args required for Java 21 module system（见 pom.xml spring-boot-maven-plugin 配置）。

## Maven Profiles

- **opensource**（默认）：无内部依赖，日常开发使用。
- **internal**：当 `../../../internal_source` 存在时自动激活，启用 KMonitor 与
  VipServer 集成。

## Git Conventions

Commit message 遵循 Conventional Commits：

```
<type>[optional scope]: <description>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`

示例：

- `feat(router): add cache-aware routing strategy`
- `fix(grpc): handle connection timeout gracefully`
- `refactor(LoadBalancer): rename method getLoadBalanceStrategy to getLoadBalancer`

## Java Javadoc 格式

新增或修改 Javadoc 时，统一使用多行格式，即使只有一句说明也不压缩成单行。
`/**` 和 `*/` 各自独占一行，说明文字放在中间以 ` * ` 开头的行上。

```java
/**
 * Exposes whether application warm-up has completed.
 */
```

## Testing Strategy

- 单元测试用 JUnit 5 + Mockito 5.20.0（Java 21 下无需 PowerMock）。
- 测试类结构镜像源码结构（如 `DefaultRouterTest` 对应 `DefaultRouter`）。
- Mock 外部依赖（gRPC 客户端、cache manager、config service）。
- 重点覆盖：路由逻辑、策略选择、错误处理、回滚行为。

## Important Reminders

1. Do what is asked; no more, no less.
2. Don't keep reading the file back and forth. If you need to make changes, do it quickly.
   Do not repeatedly read the same file multiple times — once you have sufficient context,
   proceed to edit directly.
3. Always prefer editing existing files over creating new ones.
4. Do not proactively create documentation files (*.md) or README files unless explicitly
   requested.
5. When fixing issues in code, make the code appear as if the problem never existed in the
   first place. Do not write comments explaining why a solution was used to fix a problem —
   readers should not wonder about a problem X they weren't aware of. Bad example:

   ```java
   // Request queue (using configured capacity parameter to control queue size, avoiding race conditions)
   private final BlockingDeque<BalanceContext> queue;
   ```
6. Before considering any code change complete, run the full-repository
   `./mvnw spotless:check -Pspotless-check` from `rtp_llm/flexlb` and ensure the entire
   Maven reactor passes. A module-only Spotless result is not sufficient.

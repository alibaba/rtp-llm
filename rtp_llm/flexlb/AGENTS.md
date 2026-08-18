# AGENTS.md

This file provides guidance to agents when working with code in this repository.

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
| [02-queue-scheduling](docs/architecture/02-queue-scheduling.md) | QueueManager / RequestScheduler 排队调度、请求生命周期 |
| [03-resource-management](docs/architecture/03-resource-management.md) | DynamicWorkerManager 动态容量、资源水位 |
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

必需环境变量：`FLEXLB_CONFIG`、`MODEL_SERVICE_CONFIG`；可选：
`FLEXLB_SYNC_CONSISTENCY_CONFIG`（ZooKeeper 主选举）。字段说明见
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

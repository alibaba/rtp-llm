# CLAUDE.md

这个文件为Claude Code (claude.ai/code) 提供指导，在这个FlexLB项目仓库中工作时使用。

## 项目概述

FlexLB是一个基于Java的AI模型推理负载均衡服务，使用Spring Boot 2.7.1和Java 8构建。它为AI/ML工作负载提供智能负载均衡、缓存和请求批处理功能。

## 模块架构

项目包含5个Maven模块：

- **flexlb-api**: Web层，提供HTTP端点和响应式Web服务
- **flexlb-common**: 共享工具、数据模型、异常处理和通用配置
- **flexlb-grpc**: 用于模型服务通信的gRPC客户端实现
- **flexlb-sync**: 核心负载均衡逻辑、调度策略和工作节点状态同步
- **flexlb-cache**: 分布式缓存管理模块，提供双层哈希表架构的KV缓存服务

### 关键架构组件
- 负载均衡策略（轮询、最低并发、最短TTFT）
- 请求批处理和缓存机制
- 工作节点健康监控和状态同步
- 使用ZooKeeper的主节点选举服务
- 支持后端服务的gRPC和HTTP协议

### FlexLB-Cache模块详细说明

flexlb-cache模块是一个独立的缓存管理服务，为FlexLB负载均衡器提供高性能的分布式缓存功能：

#### 核心架构
- **双层哈希表设计**: GlobalCacheIndex + EngineLocalView
- **智能缓存匹配**: 基于负载均衡的缓存选择策略
- **Worker状态集成**: 与flex-sync模块集成，接收worker_status更新缓存
- **性能优化**: 内存管理、查询优化、热点识别

#### 主要组件
- `FlexCacheManager`: 对外统一接口，封装所有缓存服务
- `KvCacheManager`: 核心缓存管理器
- `GlobalCacheIndex`: 全局缓存索引
- `EngineLocalView`: 引擎本地视图
- `CacheMatchingService`: 智能缓存匹配服务
- `EngineService`: 引擎管理服务
- `PerformanceOptimizer`: 性能优化器

#### 对外接口
flex-sync模块通过`FlexCacheManager`调用缓存服务：
- Worker状态缓存更新 (`updateWorkerCache`, `batchUpdateWorkerCache`)
- 缓存匹配和查询 (`smartMatch`, `findMatchingEngines`)
- 引擎管理和状态监控 (`getEngineStatistics`)
- 性能优化管理 (`getOptimizationStatistics`)
- 系统健康检查 (`checkSystemHealth`)

#### 同步机制说明
flexlb-cache模块不再包含独立的同步机制，而是复用flex-sync模块的同步能力：
- flex-sync负责定时获取worker状态
- flex-sync调用FlexCacheManager.updateWorkerCache()更新缓存
- flexlb-cache专注于缓存数据管理和查询优化

## 常用开发命令

### 构建和打包
```bash
mvn clean package -DskipTests
```

### 本地运行应用
```bash
bash start-master.sh
```
这会在端口7002启动应用，管理端点在8804。

### 测试命令
基于README.md示例：

测试同步LB状态：
```bash
curl --location 'http://127.0.0.1:7002/load-balance/prefill/consistency/syncStatus' \
--header 'Authorization: Bearer M4TTTOGSZS' \
--header 'Content-Type: application/json' \
--data '{"roleId": "test_model_nm125_L40S_2TP_stream.inference_part0"}'
```

测试主节点通知：
```bash
curl --location 'http://127.0.0.1:7002/load-balance/prefill/consistency/notifyMaster' \
--header 'Authorization: Bearer M4TTTOGSZS' \
--header 'Content-Type: application/json' \
--data '{"reqIp": "1.2.3.4", "roleId": "test_model_nm125_L40S_2TP_stream.inference_part0"}'
```

### flexlb-cache模块测试
```bash
# 运行缓存模块测试
mvn test -pl flexlb-cache

# 生成测试覆盖率报告
mvn jacoco:report -pl flexlb-cache
```

### 配置

应用使用Spring profiles（本地开发设置为"test"）和环境变量进行配置：

- `WHALE_MASTER_CONFIG`: 负载均衡和缓存配置
- `WHALE_SYNC_LB_CONSISTENCY_CONFIG`: ZooKeeper和一致性设置
- `MODEL_SERVICE_CONFIG`: 后端服务端点

关键配置文件：
- `antx.properties`: 应用角色配置
- `application.yml` / `application-test.yml`: Spring配置，包含占位符值

## 关键技术

- Spring Boot 2.7.1 (WebFlux用于响应式编程)
- gRPC 1.50.2用于服务通信
- Netty 4.1.77用于网络处理
- ZooKeeper (Curator 4.2.0)用于主节点选举
- Redis用于缓存和状态管理
- Prometheus/Micrometer用于指标收集

## 重要指令提醒
做所要求的事情；不多不少。
除非绝对必要，否则不要创建文件。
始终优先编辑现有文件而不是创建新文件。
不要主动创建文档文件(*.md)或README文件。只有在用户明确要求时才创建文档文件。

      
重要：这个上下文可能与您的任务相关，也可能不相关。除非与您的任务高度相关，否则您不应该回应这个上下文。
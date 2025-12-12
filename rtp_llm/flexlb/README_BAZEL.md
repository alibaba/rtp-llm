# FlexLB Bazel Build Guide

本文档说明如何使用 Bazel 构建 FlexLB 项目。

## 构建配置完成情况

已经完成将 FlexLB 的所有 Maven 模块转换为 Bazel targets：

### 已配置的 Targets

1. **Proto/gRPC targets**
   - `//rtp_llm/flexlb:engine_rpc_proto` - Protocol Buffer 定义
   - `//rtp_llm/flexlb:engine_rpc_java_proto` - Java Proto 库
   - `//rtp_llm/flexlb:engine_rpc_java_grpc` - Java gRPC 库

2. **Java 库 targets**
   - `//rtp_llm/flexlb:flexlb-common` - 基础通用库
   - `//rtp_llm/flexlb:flexlb-cache` - 缓存管理模块
   - `//rtp_llm/flexlb:flexlb-grpc` - gRPC 客户端实现
   - `//rtp_llm/flexlb:flexlb-sync` - 核心负载均衡逻辑
   - `//rtp_llm/flexlb:flexlb-api` - Web API 层

3. **测试 targets**
   - `//rtp_llm/flexlb:flexlb-common-test`
   - `//rtp_llm/flexlb:flexlb-cache-test`
   - `//rtp_llm/flexlb:flexlb-sync-test`

4. **可执行 targets**
   - `//rtp_llm/flexlb:flexlb-api-executable` - 可执行的 JAR
   - `//rtp_llm/flexlb:flexlb-app` - 部署用的 uber JAR

## 构建命令

### 构建所有库
```bash
bazel build //rtp_llm/flexlb:flexlb-api
```

### 构建 gRPC 相关代码
```bash
bazel build //rtp_llm/flexlb:engine_rpc_java_grpc
```

### 构建可执行 JAR
```bash
# 构建包含所有依赖的 uber JAR
bazel build //rtp_llm/flexlb:flexlb-app_deploy.jar

# 运行应用
bazel run //rtp_llm/flexlb:flexlb-app
```

### 运行测试
```bash
# 运行所有测试
bazel test //rtp_llm/flexlb:all

# 运行特定模块的测试
bazel test //rtp_llm/flexlb:flexlb-common-test
bazel test //rtp_llm/flexlb:flexlb-cache-test
bazel test //rtp_llm/flexlb:flexlb-sync-test
```

## 依赖管理

所有 Maven 依赖都通过 `deps.bzl` 文件管理，并在 WORKSPACE 中通过 `flexlb_maven` 仓库引入。

### 添加新依赖
1. 编辑 `rtp_llm/flexlb/deps.bzl`
2. 在 `flexlb_maven_init` 函数的 `artifacts` 列表中添加新依赖
3. 运行 `bazel sync` 更新依赖

## 注意事项

1. Java 版本：项目使用 Java 21，确保你的环境中配置了正确的 Java 版本
2. 所有模块间的依赖关系已经正确配置
3. 测试使用 JUnit 5 框架
4. 主应用入口是 `org.flexlb.Application`

## 与 Maven 构建的对比

| Maven 命令 | Bazel 命令 |
|-----------|-----------|
| `mvn clean package` | `bazel build //rtp_llm/flexlb:flexlb-app_deploy.jar` |
| `mvn test` | `bazel test //rtp_llm/flexlb:all` |
| `mvn spring-boot:run` | `bazel run //rtp_llm/flexlb:flexlb-app` |
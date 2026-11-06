# main 上的 DSV4.1 Mock 迁移审查

基线为 main `6a1eca9a62894c688669935693515cfe36966935`。
Mock 来源为 `codex/mock-engine-on-vit-kvcm-align@fb64aacfde`。
DSV 行为参考 `feature/dsv41-dev-main-202161013@9ca78c35f6`。

完成一项时，记录实际验证命令和结果。源码检查、编译和运行验证分别记录，待验证项保持未勾选。

## 旧实现及引用

- [x] 删除 flexlb-mock-engine 和 tools/online_eval，移除父 POM 的旧模块入口。
- [x] 将生产性能测试需要的 sample_access.json 和 trace_30min.jsonl 原样迁入 flexlb-api 测试资源，修改数据定位。
- [x] 删除旧 TERM harness 入口和旧性能操作文档，保留其他 TERM 单测和生产测试替身。
- [x] 旧 TERM harness 不再保留对旧 Mock 的调用，现有生产 TERM 单测保留。其 late、queued、engine 三个整包场景未迁入新框架，不计入本次覆盖。
- [x] 复查 CI、镜像、脚本、文档和反射/SPI 入口，跟踪源码搜索无旧运行入口。KVCM VIP SPI 对齐 main 的 ServiceDiscoveryProvider/Endpoint；内部 SDK 打包与真实 VIP 仍待部署验证。
- [x] 检查打包后的 Engine/Client JAR 和 wheel，无 org/flexlb/mockengine、flexlb_ft 或临时迁移脚本，wheel 包含 View 页面。

## 生产代码差异

- [x] main KVCM、连接池、配置、回退与反馈使用原实现，旧迁移代码不再导入。生产 cache/KVCM 目录无迁移差异。
- [x] Mock 对本地 HBM、远端加载和部署形态的限制只在 Mock 层校验。
- [x] 每一处 Mock 目录外的新增或修改按下文逐项记录职责和验证；生产性能数据只迁移位置，内容保持不变。
- [x] Worker 代次、异步回调、退役和取消清理保留 main 已有约束，原生全量 Java 回归通过。
- [x] 无整目录生产代码覆盖、无无关格式修改、无已撤销的 DSV 补丁。

## DSV 链路

- [x] Mock Client → main Master → ViT → P → D 已完成实际请求验证。真实 Frontend/Renderer 保持 main，未整目录移植 feature；真实 Frontend 联调不在此项验证范围内。
- [x] main 既有协议字段保留，新增 PD 区间与 ViT 负载字段无编号冲突。保留 block_size=23、lookahead=24、group=25、rollback=26、step=27，ViT 负载使用可缺省的 field 28。
- [x] ViT 网络准备、调度、计算、结果发布、负载状态与健康查询边界保持来源实现，Java ViT 回归及实际多模态请求通过。
- [x] P/D 缓存 key、复用长度、计算区间及 KV handoff 由共享协议组件实现；原生 C++ 哈希对照、Java Scheduler 和 V/P/D 冷热请求通过。
- [x] 本地原生 KVCM 协议 fixture 验证 namespace、host 和媒体 key 与 main Master 对接，产生 1 次实际查询、2 个匹配 host。真实服务待联调。

## 配置、部署和产物

- [x] 对齐的 real 配置使用 real 名称，Mock 简化模型参数保留小写点分名；配置和启动边界测试通过。
- [x] Master 不再读取旧 FLEXLB_SYNC_CONSISTENCY_CONFIG；配置、文档和测试使用原生 consistency。未增加旧接口兼容层。
- [x] 本地 Master 独立 JVM，绑定与选举地址一致，本地产物留在 Mock Master。双 Master 实际运行因本机缺少回环别名被跳过，不计入运行验证。
- [ ] Whale 启动和镜像遵循真实入口，Case 的产物与发现写入留在运行器。
- [x] 保留 P/D/V 独立筛选、聚合和事件线；View 测试通过，Basic View 实际加载成功。

## 验证

- [x] FlexLB 全 reactor Spotless 检查通过。命令：`cd rtp_llm/flexlb && ./mvnw spotless:check -Pspotless-check`。
- [x] Java 全 reactor verify 通过；后续改动的原生 gRPC、ViT 状态、KVCM 发现均定向回归通过。
- [x] Python 全量首次 422 通过、5 失败、5 跳过；修正并定向复验 Master 文件为 27 通过、2 跳过，新增身份配置边界文件为 15 通过。未重复整套长实验。
- [x] 纯文本 P/D、多模态 V/P/D 和 KVCM 开关场景通过。
- [ ] 多 Master HA、Engine 重启/替换、取消、超时及输出积压场景通过。
- [x] 独立新目录按 CLI compile/launch/run 执行 Basic：PASS、QPS=2.0、20/20 成功、清理无错误；View API 返回 PASS、3 个门禁、metrics=ready，无警告。
- [x] 最终生产差异和旧实现引用再审一遍；从 main 建立独立线性分支，无 feature 分支 merge。

## 基线候选检查

远端 DSV/V4.1 相关分支已检查。近期 dev、release、zsp、yemu 和 MU 对齐分支与最新 main 的共同祖先仍为 bcd02fdb05，未包含 main 的 KVCM 合入 #1502。
feature/dsv41-main-migration-20261009 是 #1488 的早期迁移来源，采用类型化 V4.1 输入，不包含后续 URL/PD 多模态改造。
feat/diag-v41-base 包含 main KVCM，但只是在较近 main 上增加诊断注释。
因此当前没有找到同时具备最新 main KVCM 和后续 DSV 多模态链路、可以直接替代本次迁移的远端分支。

## 生产差异的职责

- `model_rpc_service.proto`：增加 DSV PD 的媒体缓存区间、已计算 KV 末端，以及 ViT 在途负载；保留 main 的 typed V4.1 输入和其他字段。
- `WorkerStatus` / `EngineStatusConverter`：在原子状态快照中保留可缺省的 ViT 负载，未上报与显式零分开。
- `VitCacheSelector` / `VitCacheDirectory`：ViT 元数据路由支持在途负载选择；目录、缓存归属和第二次路由校验统一使用 main 的逻辑 Worker 身份。选择结果仍校验代次。
- `RoutingConfig` / validator：只增加 ViT 路由方式和报告有效期；其他路由、KVCM、连接池保持 main 实现。
- `FlexlbGrpcServer` / 两个选举服务：遵守 Spring `server.address`，使独立 Master 可以使用不同监听 IP、同一协议端口；默认行为不变。本地产物和发现文件由 Mock Master 负责。
- `cache_key_routing.py`：保留 main typed V4.1 哈希，增加 DSV row-hash 区间输入，两种媒体身份不能混用；滚动算法仍调用原生 C++ HashUtil。未整目录覆盖 Frontend、Renderer 或 GPU Engine。

## 已执行验证

- Java：`scripts/build-java.sh` 全 reactor verify 通过；ViT 逻辑身份修正后单独重编译，`VitCacheDirectoryTest`、`EngineStatusConverterTest` 共 29 项通过。
- Python：配置与 View 共 151 项、Master 纯配置 22 项、Master 生命周期单元测试 9 项通过。
- 缓存 key：18 组文本/媒体区间、符号边界和 namespace seed 对照原生 C++ HashUtil 与 Java Client/Engine。
- 旧引用：跟踪源码中搜索 `flexlb-mock-engine`、`flexlb_ft`、`tools/online_eval` 无活动入口；普通生产单元测试替身保留。

真实 Whale、内部 VIPServer SDK 和真实 GPU/Frontend 完整链路不由上述本地结果证明。

## 本地发现的问题与修正

- main 的监控 provider 缺省为 noop。Mock Master 显式选择原生 micrometer provider，并用 `flexlb_` 前缀导出全部业务指标；不改动生产默认配置。
- main 的 ViT 目录索引为 `ip:port@engineIndex`，此前物理地址校验会误报 VIT_ROUTE_STALE。目录和校验改用逻辑身份，保留代次检查。
- KVCM 查询缓存可以让两次生成只产生一次原生查询。测试检查实际正向匹配，不要求固定 RPC 次数；测试统计中的命中数量按匹配 host 计数。
- Basic Case 初次运行的 4s 窗口含冷启动后的补发：请求 2→3 间隔 1.562s，随后请求 3→4 间隔 2ms，短窗口观测为 2.674 QPS。将采样前等待从 6s 延至 8s，保持 QPS=2、门禁 [1.5, 2.5] 和全部请求成功条件。

最终 Basic 产物：`/Users/wangziyi/.codex/outputs/mock-main-dsv41-20261106.ckOtt1/result-final`。
首次失败产物保存在同一目录的 `result`，未覆盖。

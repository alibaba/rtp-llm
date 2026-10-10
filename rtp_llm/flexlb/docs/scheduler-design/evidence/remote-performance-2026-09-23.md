# 远端 FlexLB 性能测试（2026-09-23）

在 `luoli.hn@11.163.39.110` 的 `luoli_gpu` 容器执行，目录为
`/data0/luoli.hn/work/rtp_llm_4/github-opensource/rtp_llm/flexlb`。
分支 `main`，开始与结束提交均为 `017648a37d529ce56ce6d8feaba5354adaa11272`。
使用 Dragonwell Java 21.0.11、Maven 3.9.9，API profile 原有固定 2 GB 堆。
本轮没有修改远端源码、配置或测试门槛；没有同步本地未提交变更。

## 结果

- Sync：3/3 通过。稳定队列捕获分配为 0 B/op；等待快照为 488 B/op。
- 准入失败分配量：单独补跑 1/1 通过；1/64/512/1024 worker 分别为 328/328/368/368 B/op。
- API：16 个用例，15 通过、1 失败，没有 error 或 skip。
- 失败用例 `batchScheduleRemainsFastAcrossRealGrpcBoundaries`：8192 请求突发实测客户端 4041.3 QPS，低于 5000 QPS 门槛，首先在吞吐断言失败。日志同时记录客户端 P99=865.396 ms、Master P99=862 ms；不能声称后续断言已执行。
- 突发阶段 P99：gRPC 排队 16 ms，route submit 589 ms，batch wait 626 ms，dispatch ACK 36 ms，ACK response 2 ms。各指标不是可直接相加的独立区间，目前不能据此确定根因。

以下为已通过的定速规模场景：

| 规模 | 目标 QPS | 客户端实际 QPS | Master P99 |
| --- | ---: | ---: | ---: |
| 1P/2D | 1000 | 996.7 | 14ms |
| 1P/2D | 2000 | 1982.3 | 31ms |
| 1P/2D | 5000 | 4901.3 | 17ms |
| 1P/2D | 10000 | 9721.3 | 13ms |
| 2P/4D | 2000 | 1967.0 | 12ms |
| 4P/8D | 1000 | 995.3 | 12ms |
| 4P/8D | 2000 | 1965.9 | 11ms |
| 4P/8D | 5000 | 4898.5 | 20ms |
| 4P/8D | 10000 | 9330.1 | 40ms |
| 8P/16D | 2000 | 1961.7 | 11ms |
| 16P/32D | 1000 | 990.9 | 11ms |
| 16P/32D | 2000 | 1961.9 | 20ms |
| 16P/32D | 5000 | 4897.0 | 11ms |
| 16P/32D | 10000 | 9791.5 | 26ms |

## 复跑命令

```sh
export JAVA_HOME=/usr/lib/jvm/java-21
export PATH="$JAVA_HOME/bin:$PATH"
./mvnw clean test -P '!internal,sync-performance-regression' -pl flexlb-sync -am
./mvnw clean test -P '!internal,api-performance-regression' -pl flexlb-api -am
./mvnw test -P '!internal,sync-performance-regression' -pl flexlb-sync -am -Dtest=PrefillAdmissionFailurePerformanceTest
```

三组顺序执行。准入失败性能类未在当前默认 Sync 性能 profile 的 include 中，故显式补跑。

## 证据与限制

- 容器原始日志：`/tmp/flexlb-perf-main-8PxGV3bx/{environment,sync,api,admission}.log`。
- 本地日志副本：`/tmp/flexlb-remote-perf-017648a3/`。
- 首轮 `/tmp/flexlb-perf-cdZaUXD6/` 启动于旧提交，期间用户切换分支，编译失败；不纳入性能结果。随后对新提交执行 clean 构建。
- 这是远端当前提交的单轮测试，不是本地改动前后的性能对照。远端与本地版本及机器不同，不能将数值差异归因于某个改动。
- API 用例通过真实 gRPC 边界和测试中的 Mock Engine；不是 GPU 模型推理吞吐测试。未执行全量功能 UT，未定位突发失败根因。

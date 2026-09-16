# RequestSlot 接口整理后的远端验证

参考 [2026-09-14 验证流程](../request-terminal-2026-09-14/README.md)，本次同步整个工作区的构建输入，包含未提交修改和新增 DecodeState 文件。未改性能脚本或测试门槛。

## 源码与环境

- 快照时间：`2026-09-16T18:14:51.028156+08:00`，本地 HEAD：`8e82cc4bd8534c7371ce9e436b617bd1ae55de81`；实际测试内容以 [逐文件 SHA-256](source.sha256) 为准，不能仅用 HEAD 代表。
- 475 个构建输入文件，包含 Maven 配置、测试数据、配置示例和两个 proto；不包含本地 target。
- 远端：`luoli.hn@11.163.39.110`，容器 `luoli_gpu`，Dragonwell Java 21.0.11.0.11。
- 独立目录：`/data0/luoli.hn/work/rtp_llm_4/flexlb-request-slot-validation-20260916-181450`。目录中的 Git 提交仅标识测试快照，没有提交本地业务工作区。
- 执行前后均通过全部 475 个文件的 SHA-256 校验。

测试完成后核对本地源码，与本次快照一致。

## 功能回归

执行 `./mvnw -B -P 'opensource,!internal' -pl flexlb-sync -am test`。

112 个测试套件，1173 项测试，0 failures、0 errors、0 skipped。`EndpointCleanupDeadlockTest` 的 6 项测试全部通过，包括 Decode 清理与 Worker 结束交错、定时器关闭与 Slot 锁交错。测试通过不能证明所有可能交错均无死锁，但没有削弱或跳过这些用例。

[各套件结果](unit-results.json)。完整 XML 和 Maven 日志见归档。

## 性能回归

执行原有 `tools/run_queue_performance.sh`：750P/750D，8 GiB heap，各档预热 10 秒、测量 10 秒，BATCH/FIXED_WINDOW 与 NON_BATCH/SINGLE 顺序执行。

默认端口段存在占用，通过 Wrapper 支持的 `MAVEN_CONFIG=-Dflexlb.perf.engine-matrix-first-prefill-grpc-port=10001` 传入现有端口参数。其余脚本参数保持原样。

| 模式 | 目标 QPS | Client QPS | Master QPS | Client P99 | Master P99 | Delivery wait P99 | Batch wait P99 | 结果 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BATCH | 3000 | 2954.9 | 2958.7 | 122.062ms | 31ms | 11ms | 14ms | 通过 |
| BATCH | 10000 | 9988.6 | 10003.9 | 86.756ms | 45ms | 37ms | 37ms | 通过 |
| NON_BATCH | 3000 | 2999.1 | 3002.0 | 57.412ms | 4ms | 1ms | N/A | 通过 |
| NON_BATCH | 10000 | 9984.0 | 9987.9 | 176.144ms | 75ms | 1ms | N/A | 通过 |

原有门槛：Client/Master 吞吐不低于目标 98%，Client/Master P99 ≤ 250 ms，Delivery wait P99 ≤ 50 ms，BATCH 的 Batch wait P99 ≤ 50 ms。模式退出码：`{'BATCH': 0, 'NON_BATCH': 0}`。

本轮四档均通过原门槛。本次未做同机配对基线，不能把与历史数据的差值直接解释为本次重构带来的性能变化。这是 loopback 调度测试，不是 GPU 推理吞吐测试。

## 首轮及复现材料

首轮 18:08 快照的 1163 项功能测试全部通过，但 `MAVEN_ARGS` 未被当前 Wrapper 传递，BATCH 仍使用默认端口并在 23477 绑定失败，没有产生有效 BATCH 性能数据。初次创建测试目录还遇到 Git 目录所有权检查；已通过仅对测试进程设置 safe.directory 解决。

之后重新固定 18:14 的当前工作区快照，纳入期间变化的三个文件，并修正参数传入方式，完整重跑功能和性能测试。保留 [首轮日志](initial-remote-logs.tar.gz) 与 [首轮结果](initial-results.json)，不将启动失败算成吞吐测量。

[本轮机器可读结果](results.json)；[源码清单](source-manifest.json)；[完整远端日志、测试 XML、运行脚本、退出码和源码校验](remote-logs.tar.gz)。

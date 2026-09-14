# 请求终止清理与命名调整验证

验证对象为 `abef01a1534f75dbde6d5945c3b99df772287a23` 上的当前工作区快照，
包括请求终止清理、资源账本 UT，以及 terminal record / REQUEST_FENCED 配套命名调整。
远程独立目录内的 Git 提交仅标识验证快照；源码逐文件 SHA-256 在日志归档的 `source.sha256` 中。

## 功能回归

- 资源清理调整后：相关 flexlb-sync 用例 324 个、上游 RequestTest 3 个全部通过。
- 后续命名调整后：请求与抢占相关用例 161 个、上游 RequestTest 3 个全部通过。
- 覆盖 inflight、Prefill 记账、Decode KV reservation、running 容量、旧身份回调、锁外清理和 BATCH 不提前释放。

## 远程性能

环境：`luoli.hn@11.163.39.110` 的 `luoli_gpu` 容器，Dragonwell Java 21.0.11.0.11。
独立目录：`/data0/luoli.hn/work/rtp_llm_4/flexlb-request-terminal-validation-20260914-2205`。
执行原有 `tools/run_queue_performance.sh`，750P/750D、固定 8 GiB heap，
各档预热 10 秒、测量 10 秒，BATCH 和 NON_BATCH 顺序运行。

| 模式 | 目标 QPS | Client QPS | Master QPS | Client P99 ms | Master P99 ms | Delivery wait P99 ms | 结果 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BATCH | 3000 | 2996.8 | 3000.8 | 253.894 | 253 | 58 | 未通过 |
| BATCH | 10000 | 9975.4 | 10010.5 | 235.187 | 137 | 71 | 未通过 |
| NON_BATCH | 3000 | 2999.7 | 3000.6 | 2.735 | <1 | 1 | 通过 |
| NON_BATCH | 10000 | 9995.6 | 10002.5 | 58.713 | 10 | 1 | 通过 |

原始门槛保持不变：完成吞吐至少为目标的 98%，client/server P99 不超过 250 ms，
delivery wait P99 不超过 50 ms。BATCH 两档均未通过延迟门槛，不能记为性能验收通过。
本次没有进行同机当前 HEAD 的配对基线测量，不能仅凭历史失败将结果归因于环境或排除改动影响。
用户在收到上述失败结果后明确要求提交代码。

这是 loopback Master/client/mock 调度性能测试，不是 GPU 推理性能测试。
[机器可读结果](results.json)；[完整远程日志、退出码和源码校验](remote-logs.tar.gz)。

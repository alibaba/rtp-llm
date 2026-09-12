# DeepSeek V4 Pro KV Offload 实测

模型已下载完成，代码基于 main，在 `rym/feat/dsv4_dsa_kvoffload` 分支保存，并已在本地四卡跑通。结论是：同 batch 的额外开销较小；可以用更大的 batch 提高 goodput，但还没有达到“batch 翻倍且 TPOT 不变”。

存储划分、每层执行流程、Graph 依赖，以及单请求 KV / CPU 池容量 / prefill 峰值的区别，见 [设计与显存口径](DESIGN.zh-CN.md)。

## 实现

- CSA 的 Indexer Key、HCA、SWA 和 compressor state 全部留在 GPU。
- CSA KV 在 CPU 保留完整副本，GPU 预算内尽量驻留，超出部分从 CPU 读取。
- 每个请求、每个 CSA 层独立保留 2048 个压缩 KV 条目的热缓存，支持跨步复用。每个条目对应 4 个原始 token；模型每步选 1024 个条目。
- TopK 出来后启动异步回填，与当前层主 compressor 重叠；attention 前等待回填和边界 KV 更新完成。支持 CUDA Graph。

## 主对照

同样的 4 张 L20D，TP4 / EP4 / DP1 / CP1，开启 CUDA Graph。每条输入严格为 131072 token；输出 145 token，排除首 token 和 16 个预热 token，测量其后 128 token。每卡 KV 总预算均为 12 GiB，offload 主方案按 CSA GPU 区 6 GiB、其他必驻留池 6 GiB 分配。

| Batch | Vanilla TPOT ms | Offload TPOT ms | Vanilla goodput token/s | Offload goodput token/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 18.48 | 18.74 | 54.10 | 53.35 |
| 8 | 24.05 | 24.38 | 331.86 | 325.92 |
| 16 | 29.32 | 30.57 | 545.18 | 522.92 |
| 24 | KV 容量不足 | 34.76 | 不适用 | 689.59 |
| 32 | KV 容量不足 | 38.53 | 不适用 | 829.36 |

这里 goodput 的门槛是每请求平均 TPOT ≤45 ms，只计算 decode，不含 prefill、排队或 TTFT。B16 同 batch 开销约 4.3%；offload B32 对比 vanilla B16，goodput 增加 52.1%，TPOT 增加 31.4%。

## 更低延迟的配置

总预算仍为 12 GiB，改为 CSA GPU 区 8 GiB、其他池 4 GiB，并单独捕获 B18/B20 的 Graph：

| Batch | TPOT ms | Goodput @45ms token/s | 相对 vanilla B16 |
| ---: | ---: | ---: | --- |
| 18 | 30.82 | 583.49 | TPOT +5.1%，goodput +7.0% |
| 20 | 32.36 | 616.03 | TPOT +10.4%，goodput +13.0% |

若要求 TPOT 至多比 vanilla B16 增加 10%，这轮测量中 B18 更合适。放宽到 +20%，主方案的 B24 可达到 goodput +26.5%。不能把 45 ms 门槛下的全部吞吐直接当成更严格门槛下的 goodput。

## 验证与边界

- 原生构建成功，最终 18 项针对性回归通过。
- 真实模型 B16/128K 连续 40 个 decode step，在四卡全部 30 个 CSA 层逐字节检查 GPU 将读取的 KV 与 CPU 副本，无不一致；检查在 Graph 重放中生效。该诊断另跑，不混入性能数据。
- B1 的 145 个输出 token 完全一致。B8/B16 的部分请求生成过程中出现输出分叉，两次 vanilla B8 运行之间也存在类似分叉；原因尚未隔离，不能据此宣称模型质量完全等价。
- Vanilla B24/B32 是本测试要求“整批同时准入”时收到的 KV 配额不足错误，不是物理 CUDA OOM，也不是把 TPOT 超标计为失败。普通服务可以排队或缩小 batch。结论限于指定预算，未测出整机的绝对容量上限。
- 曾用 42 GiB 配置在权重重排启动阶段遇到真实 CUDA OOM，但它不属于这张对照表的运行时容量证据。
- 本轮为单次测量，尚未覆盖线上混合 prefill/decode、完整质量评测、CP、PD、MTP 或前缀复用。两种方法都使用相同的 TP4 适配，包括 FlashMLA 的 32→64 head 补齐兼容路径。

详细结果见 [RESULTS.md](RESULTS.md)，复现命令见 [REPRODUCE.md](REPRODUCE.md)，原始 JSON 和多 SLO 表格在 `results/`。归档中的 `source/` 是代码快照，`dsv4-main-offload.patch` 可应用到记录的 main 提交。

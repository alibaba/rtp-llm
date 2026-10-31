# GLM MTP 与 sparse KV offload 验证（2026-10-29）

在 `feat/sparse-kv-offload`、基线 `7e43d7d0088fad46dc08fb4b89df9472d4decdfa` 上实现 GLM NextN；权重映射和 EAGLE 执行方式参考 `6cbc2810302fb46d60430de050fc4ff554d69b8e`，接入当前 grouped cache API。保留原有按请求分配 HBM 的改动。

实现包括完整/独立 MTP checkpoint 加载、draft embedding/hidden 融合、因果多 query indexer、target verify/draft extend 的 sparse MLA 路由、target/draft 分层缓存存储和容量计费、block 版本同步，以及 CUDA Graph 的 draft extend 标记与共享 runtime 生命周期。不启用跨 MTP 迭代 index 复用优化。

## 配置与结果

使用本地完整 GLM-5.3 checkpoint：`/home/renyuanming.rym/models/GLM-5.3-ms-4ed051d0`。模型 config/index 与 `/mnt/nas1/hf/GLM-5.3-ms-4ed051d0` 的 SHA256 相同。实际有 8 张可用卡，采用 4P/4D，每端 TP1/DP4/EP4；P 使用 DeepEP normal，D 使用 DeepEP low latency。BF16 activation、FP8 KV，target 78 层、draft 1 层、`gen_num_per_cycle=3`。

每种模式包含 3 个串行请求和 8 个并发请求，各生成 32 tokens；输入长度为 5 和 2384 tokens。包含重复请求与 block 复用。下表迭代数取串行请求，包含 prefill。

| 模式 | 短提示迭代数 | 长提示迭代数 | 串行输出一致 | 全部长请求一致 |
|---|---:|---:|---|---|
| 关闭 MTP | 32 | 32 | 基线 | 基线 |
| MTP eager | 14 | 9 | 是 | 是 |
| MTP eager + offload | 14 | 9 | 是 | 是 |
| MTP CUDA Graph | 14 | 9 | 是 | 是 |
| MTP CUDA Graph + offload | 14 | 9 | 是 | 是 |

短提示 13 个 draft round 的逐位置接受计数为 `[9, 6, 3]`，长提示 8 个 round 为 `[8, 8, 8]`，覆盖部分接受、拒绝和全接受。两个 graph 模式中，4 个 D rank 都完成 target verify、draft decode、draft extend 捕获，并通过真实请求日志确认 graph 回放。

offload 对照强制每卡 `H=2/P=80/N=82`，长请求必须使用 Host KV。选取容量为 `R=S=65536`，由最大 batch 8、每轮 4 个 query、top-K 2048 推导；GPU KV 总计费约 3.30 GiB，包含热缓存、正式 HBM KV、indexer group 和元数据，Host KV 约 0.99 GiB。该配置用于验证实际 Host 路径，不是容量或吞吐基准。

## 精度范围

7 层 target 加真实 NextN 层的两卡对照，5 种模式的 25 个主要请求全部逐 token 相同，另完成 2 个 graph 回放探针。裁剪 target 的 draft 接受数为 0，因此不据此判断完整模型的接受率。

完整模型完成 55 个主要请求与 2 个 graph 回放探针。每个 MTP 模式都有 7/11 请求与关闭 MTP 的输出完全相同，包含全部串行请求和全部长请求；4 个并发短提示存在续写差异。关闭 MTP 的基线本身在串行和并发时也出现同类变化，首个差异可早至第 3 个输出 token。因此完整模型的严格全请求比较曾失败，后续显式使用 `RTP_GLM_MTP_COMPARE_IDS=record` 收集其余组合，并独立断言全部串行及长请求一致；不宣称多卡 FP8 在不同 batch/rank 路由下逐 token 完全确定。

精度对照使用测试专用稳定 top-K，以隔离原子 top-K 输出顺序造成的 attention reduction 差异。另有原生 GPU indexer 数值测试与参考计算逐 query 对照。这里没有测量稳定吞吐加速比，也没有验证 TP2/4/8；迭代数下降不能直接换算成相同比例的速度提升。

## 回归与证据

13 个 Bazel 回归 target 全部通过，覆盖 KV 配置/容量/存储、attention 路由、GPU offload、index sharing、hidden capture、引擎和 MTP 执行器。新增 GLM 单测共 7 项，包含真实 GPU 的多请求、不同 query 数、因果前缀对照，以及 graph wrapper 初始化不重建已捕获 runtime 的检查。最终再次运行 GLM target 与 CUDA Graph 回放重试集成 target，两项均通过。

全部日志、HTTP 原始结果和对照脚本保存在 `/home/renyuanming.rym/glm-mtp-offload-20261029/`：`verification.json`、`small/glm-mtp-summary.json`、`full-4p4d/glm-mtp-summary.json`、`test-regression-13-pass.log`、`test.log`、`native-small-graph-driver.log`、`native-full-remaining-driver.log`。`mtp-only.patch` 仅包含本次 MTP 适配，排除先前的 request routing 改动。

停止服务时，部分进程在 backend 已停止后打印 SIGSEGV，关闭 MTP 的基线也出现；本次未修改停机处理。推理阶段无 CUDA 错误或 OOM，实验结束已释放所有测试服务和 GPU。

使用参数及限制见 [sparse KV offload 说明](sparse_kv_offload.md)。

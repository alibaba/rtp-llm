# 93 层 FP8 从 3FS 加载时的并发停顿

2026-09-29 在 114/115 用集成分支 `86d93fbfe46df6697fca28efe4020b994ed10d7e` 做了加载诊断。两端以个人账号在同镜像的 GPU 容器内分别用 `--config=cuda13 --config=sm10x` 编译，Bazel 日志均报 `Build completed successfully, 23714 total actions`。完整 target 直接读取 `/mnt/hf3fs/3fs/models/kimi/kimi-k3`，MTP shard 从个人目录的 3FS 链接读取；加载器仍明确选用 FastSafetensors。没有发 smoke 请求，也没有性能采样。

首次双机 PD 启动使用任务进程内的 `libparallel_3fs_pread_20260929.so`，每 rank 64 个读取线程。两端均没有完成 target 加载，约 9 分钟后停止任务进程。随后在 115 上保持同一代码、权重和八卡 Decode 配置，临时给 Bazel runfile 的 loader 副本加逐张量日志，做了有时限的单端对照；原源码未改，诊断结束后 runfile 符号链接已恢复。

| 115 对照 | FastSafetensors 路径 | 每 rank 任务读取线程 | 八 rank 的最远加载事件 | 观察结果 |
| --- | --- | ---: | --- | --- |
| `r2` | SHM | 64 | 有些 rank 卡在第 18 个张量的 `f_b_proj`，其余刚完成它 | 未完成加载 |
| `r3` | NOGDS | 64 | 第 18 个张量 | 180 秒无后续事件 |
| `r4` | SHM 原生读取 | 无 helper | 所有 rank 至少第 51 个张量，一个 rank 到第 512 个 | 86.1 秒时达到诊断目标后主动停止 |
| `r5` | SHM | 32 | 所有 rank 到第 21 个张量 | 64.1 秒时达到诊断目标后主动停止 |
| `r6` | SHM | 48 | 第 18 个张量 | 180 秒无后续事件 |
| `r7` | SHM | 40 | 第 18 个张量 | 180 秒无后续事件 |

`r4` 和 `r5` 的停机是诊断脚本达到“八 rank 均越过第 20 个张量”后主动执行，不能写成完整 93 层加载成功。`r2/r3/r6/r7` 在限时内没有通过首层，不能写成 FastSafetensors、3FS 或 FP8 算子单独导致的死锁。首层 `f_b_proj` 取自完整 target；它在单 GPU 和八 GPU 的独立 FP8 分块量化复现中都完成，说明单独的量化计算不足以复现加载停顿。`r3` 改用 NOGDS 仍停住，不能据此认定 SHM copier 是唯一原因。当前只知道 40/48/64 路 helper 在这条八 rank 加载路径上没有推进，32 路和原生读取越过了同一位置；**32 路是已短时验证的最高档位，还没有完成全程加载与吞吐验证**。

该 helper 只通过本次任务进程的 `LD_PRELOAD` 注入，环境变量 `K3_3FS_PREAD_THREADS` 设为表中档位；没有修改 FUSE、110–115 其他机器或集群配置。回滚是在启动脚本中去掉 `libparallel_3fs_pread_20260929.so:` 和该变量，保留 GCC13 `libstdc++`。后续完整模型应先试 32 路全程加载；如果仍停顿，则用原生 FastSafetensors 直接读 3FS，并保留逐 rank 日志。不能以单 shard 的 64 路速度结果推断八 rank 全模型也更快。

四层检查点与本次完整 target 的同名张量**内容不同**。[抽样报告](weight_identity_115.json)对四个张量比较了相同形状、相同 dtype 的 512 B 或 64 KiB 前缀，四项 SHA256 全部不同。四层 `extraction_manifest.json` 记录来源为 `/data5/kimi-k3`；当前机器上尚未找到该来源的完整副本。四层权重的三方时间线仍可用于同一权重、同一形状的候选筛选，但不能证明这份 3FS 完整 target 的数值正确性。完整 target 的精确上游修订号仍需核实。

`evidence/` 保留两端首次启动、115 各档位的原始日志压缩包、逐 rank 结果、复现脚本和此前 64 路 helper 的源文件。对照期间只停止本任务的进程组；结束后 115 的八卡显存已释放，runfile 链接恢复，工作树源码保持原样。此次没有通过 93 层 FP8 双机 PD smoke，性能锚点和峰值激活锚点也尚未完成。

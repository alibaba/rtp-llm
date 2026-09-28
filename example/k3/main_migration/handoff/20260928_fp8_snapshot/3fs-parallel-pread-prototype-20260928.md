# Kimi K3 3FS 直读并发原型记录

时间：2026-09-28。只改任务进程，不改 3FS 共享挂载、集群配置或权重文件。原始 SHM 加载保持 `FastSafetensors` 与 `direct_io=True`；并发层仅在 `/mnt/hf3fs/3fs/models/kimi/*.safetensors` 的大块 `pread64` 上生效。

## 原因与做法

113 上对四层 shard 2 的系统调用跟踪记录到 510 次 `pread64`，其中 499 次是 32 MiB。3FS 的 SHM ioctl 返回 `EINVAL` 后，二进制扩展沿同一线程依次发起这些读取；`max_threads` 只适用于另一个 nogds copier，现有 SHM 配置不能增加这段读取的并发。跟踪原始文件：`strace-fss-shm-113.log`。

任务级 `parallel_3fs_pread.c` 编译成 `libparallel_3fs_pread.so` 后，通过启动进程的 `LD_PRELOAD` 注入。第一次大块读取某个 K3 shard 时，64 个线程按 4 MiB 不重叠块直接从原 3FS 文件读取到匿名内存；随后把所请求的字节返回给原 FastSafetensors 调用。打开下一文件时按设备、inode、大小检查并释放旧缓存；关闭文件也释放缓存。失败时退回原 `pread64`。每 rank 可能临时占用一个 shard 大小的主机内存；四层 shard 约 17 GB，完整模型八 rank 同时加载时需要继续观察内存与 3FS 侧负载。

## 单 shard 结果与校验

113、同一 16,990,911,504 字节 shard、同一 FastSafetensors SHM 接口，交错顺序为原路径／64 线程／64 线程／原路径。两条路径均加载 5,404 个张量，首张量抽样相同。

| 路径 | 两次完整加载 | FastSafetensors 内部 `read` | `copyToDevice` |
| --- | ---: | ---: | ---: |
| 原路径 | 18.910、18.878 秒 | 17.063、16.876 秒 | 0.795、0.954 秒 |
| 64 线程任务层 | 4.413、5.185 秒 | 2.459、2.413 秒 | 0.461、0.561 秒 |

单 shard 整次加载中位数由 18.894 秒降到 4.799 秒，约 3.94 倍；内部读取中位数由 16.970 秒降到 2.436 秒，约 6.97 倍。另一轮分别算出的全部张量摘要完全一致：`3db636f3a1b79d26e40dd082c59698d9e711f38790e88e7d98d91e02da3719fd`。用同一个 reader 连读 shard 2、3（两文件大小相同），原路径用时 16.758/16.691 秒，并发路径 3.804/3.374 秒；两文件各自的首、中、末张量抽样摘要逐项与原路径相同，且两文件之间不同。日志：`fss-parallel-abba-*.log`、`fss-parallel-digest-*.log`、`fss-parallel-two-shards-*.log`。

这是开发诊断，113 的 GPU7 有另一用户的空闲进程，未满足正式独占性能比较要求。该进程可能影响总加载时间；读取阶段和全张量一致性结论仍保留。64 线程达到的增益不能直接外推到八 rank 同时加载，也不能证明 3FS 服务端的持续吞吐。

## 111 实际 TP8 启用

原 r41 Prefill 进程的 `LD_PRELOAD` 只包含项目要求的 GCC13 `libstdc++`。r41b 只在 111 个人任务容器的 Prefill 启动时，将相同 ABI、SHA256 `99ae7c24fbda0e681e6876833e4cdb1f5cf0f3f1c6c0c3db8332e1e1a8435771` 的并发库**前置**于该 `libstdc++`，并设置 `K3_3FS_PREAD_THREADS=64`。启动脚本：`run_fp8_3fs_parallel_r41b_prefill_111.sh`；与 r41 的配置、权重路径、代码二进制、端口一致。112 的 Decode 未调整、未重启。

111 的启动输出有 14 次 `task 3FS parallel pread ... threads=64 ready`，rank0 target/MTP 的 `load weights took` 为 12.33/7.91 秒；同机器原 r41 为 24.31/9.03 秒。r41b 各 rank 两次权重加载均完成，target 范围约 8.81–16.61 秒，MTP 范围约 4.79–10.26 秒。启动日志经 `weight_loader_guard.py verify-log` 检查为 PASS，仍明确选中 `fastsafetensors`，无 scratch/fallback 证据。完整服务健康、正式正确性和长期稳定性须另行记录，不能用一次加载结果代替。

## 回滚

只在实际读权重的任务进程上移除 `libparallel_3fs_pread.so:` 这一段 `LD_PRELOAD`，并取消 `K3_3FS_PREAD_THREADS=64`；保留原 GCC13 `libstdc++` 路径，再按原 r41 启动脚本重启即可。没有调整 FUSE、机器共享配置或集群参数，也没有权重文件变更。任务进程退出后这份匿名缓存随进程释放。

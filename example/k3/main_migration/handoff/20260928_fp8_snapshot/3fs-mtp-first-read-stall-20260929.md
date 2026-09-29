# 2026-09-29 Kimi K3 MTP shard 首次读取阻塞

9 月 29 日约 09:27（北京时间），准备在 114 Prefill、115 Decode 上启动四层 FP8 + MTP PD 功能检查。两端使用个人账号 `luohaocheng.lhc`，`lhc_GPU_k3_rdma_20260929` 容器中的 `LOAD_METHOD=fastsafetensors` guard。四层 target `/mnt/hf3fs/3fs/models/kimi/kimi-k3-4layers` 在两端均通过索引、shard header 和显式加载方式检查。**这个检查没有读取完整 target 权重，不能证明 target 数据吞吐正常。**

MTP checkpoint 视图在各主机的 `/data0/luohaocheng.lhc/models/kimi-k3-mtp-3fs-view-20260927`，配置与索引在本地，九个 shard 是指向 `/mnt/hf3fs/3fs/models/kimi/kimi-k3-mtp/` 的符号链接。两端的 guard 随后同时卡在 `mtp-experts-rank00-of-08.safetensors`。在 09:32:07 读取进程已分别等待 4 分 25 秒以上，状态均为 `D`、等待点均为 `folio_wait_bit_common`；`/proc/<pid>/fd/3` 指向该 shard，`fdinfo/3` 的位置仍为 0，`/proc/<pid>/syscall` 显示正在从 fd 3 读取 `0x100000`（1 MiB）。当时 114、115 的 `fuse.hf3fs` 挂载仍在；112 的 `/mnt/hf3fs/3fs` 已退回本地 ext4 路径，Kimi 权重不可见，且 `lhc_GPU` 容器消失。

| 主机 | 容器内 guard PID | 09:32:07 状态 | 文件 inode | 处理 |
| --- | ---: | --- | ---: | --- |
| 114 | 27530 | `D`, `folio_wait_bit_common`, 04:25 | 3146980 | 向本任务进程发 `SIGTERM`，命令最终退出 143 |
| 115 | 24599 | `D`, `folio_wait_bit_common`, 04:26 | 3146980 | 向本任务进程发 `SIGTERM`，命令最终退出 143 |

这是 shard 首个 1 MiB 读取的等待，尚未进入 FastSafetensors 的大块权重加载或 GPU 实验。此次 guard 没有注入任务级 64 线程 `pread` 库，因此没有测量并发加载时的吞吐；但单次 1 MiB 读取超过四分钟，不能据此归因于 64 线程上限。此前真实 FastSafetensors SHM 读取在 64、128、256 线程下约为 2.1–2.2 秒，继续增线程未见可确认收益，见 [并发原型记录](3fs-parallel-pread-prototype-20260928.md) 与 [存储侧读取报告](3fs-owner-read-report-20260928.md) 第 14 项。

已暂停 114/115 的新服务启动，没有修改共享 FUSE、集群配置或权重文件，也没有触碰受保护的 `/3fs-data/3fs/mtp_test`。请存储侧重点查看该 shard 对应的服务端对象与副本、09:27–09:32 的 RPC/重试/限流及两个客户端的 FUSE 请求，同时确认 112 挂载消失是否为预期维护。客户端恢复后，再重做两端 guard 和正式加载日志检查；只有实际加载完成才计入后续 smoke。

## 09:41 后的定位补充

114、115 此时均无外部 GPU 进程，满足独占测量的机器条件；112 仍无 3FS 挂载。115 再读同一 MTP rank00 shard 的首个 1 MiB，约 21 秒后仍停在 `folio_wait_bit_common`、fd 3 的位置仍为 0，已向本任务探针发 `SIGTERM`，探针退出 143。作为对照，在同一 115 容器内，`kimi-k3-4layers/model-00001-of-000007.safetensors` 的首个 1 MiB 用时 0.005 秒，`kimi-k3-mtp/mtp-experts-rank01-of-08.safetensors` 用时 0.007 秒。这两次只是小块热读，不能据此估计整文件吞吐；它们说明本次并非同一客户端所有 3FS 文件都无法读取。

尝试读取 rank00 文件偏移 4 MiB 时，探针在 `openat` 阶段就进入 `D` 状态，尚未获得 fd，因此**不能推断 4 MiB 数据块是否可读**。容器内 PID 25038 在至少 1 分 24 秒后仍处于 `folio_wait_bit_common`，`SIGTERM` 已待处理；观察时该诊断进程尚未退出，没有启动任何 GPU 服务，也不会继续向这个 shard 增加读请求。请存储侧连同 inode 3146980 的 open/read 等待一起核查，尤其与相邻 inode 3146981 的正常首读作对照。

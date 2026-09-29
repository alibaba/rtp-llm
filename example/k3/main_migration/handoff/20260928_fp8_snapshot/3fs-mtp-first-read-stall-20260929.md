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

09:47 复查：PID 25038 最终响应终止信号并退出 143；重跑一次 MTP guard，仍在 rank00 的 fd 3、偏移 0、1 MiB `read` 上进入同一内核等待，容器内 PID 25310；已终止，退出 143。此前两个独立客户端 114/115 同时出现相同症状，115 上相邻 MTP shard 与四层 target shard 的小块读取均正常，因此暂不另起任务私有 FUSE 或继续增加读取线程，避免把特定文件的等待扩大为更多并发请求。此判断仍需存储侧确认服务端根因。

完整 target 96 个 shard 的元数据统计为 1453.74 GiB、最大单 shard 15.82 GiB，没有文件超过任务级并发读取原型的 32 GiB 文件上限。115 当时可用主机内存约 3.8 TiB；八 rank 各缓存一个最大 shard 的粗略上界约 126.6 GiB。这只排除了明显的文件大小上限和当前主机容量缺口，不能代替完整模型加载时的内存与吞吐测量。当前阻断服务启动的是 MTP rank00 的 `open/read` 等待。

## 10:00 的四层 target 复查

为准备固定 vLLM 版本的四层 PD 对照，114、115 已拉到相同镜像 digest `sha256:dfaab3570be5b1f66c21e60c60f1616ad3a0143f9899b8738257004f289979fd`，两端镜像 ID 都是 `sha256:ac8e1e42fbe7d8e3f50c103d6f811354d4aaadd1ef776ca1f81b381d04a31755`。10:00 左右的 GPU 独占复查选中 114/115，各 rank 空闲显存约 268.6 GiB，GPU 利用率为 0%；主机间 ping 成功，两端 InfiniBand 端口为 Active/LinkUp。尚未启动 vLLM 服务或任何 GPU 请求。

启动前在两端各执行一次 `sha256sum`，参数依次为四层 target 的 `config.json` 和 `model.safetensors.index.json`。两端读取进程约 40 秒后仍未产生第一行哈希，均处于 `D` 状态、等待点 `folio_wait_bit_common`。已只向这两个本任务进程发送 `SIGTERM`；两端命令均结束，没有修改权重或其他人的进程。这表明阻塞已不局限于此前的 MTP rank00：连四层 target 的小型配置文件读取也出现等待。但当前证据只能定位到客户端读等待，不能断定服务端根因或判断所有 3FS 文件都受影响。

两端 3FS 仍以 `fuse.hf3fs` 挂载；共享的 `hf3fs-fuse` 容器处于 running。FUSE 连接的 `max_background` 等参数为 root-only，本次个人账号无法读取，也没有修改由其他用户管理的 FUSE 容器、挂载或集群配置。先前 64、128、256 线程的真实 FastSafetensors 读取耗时接近；当前连小文件首读都等待，继续增加本任务的并发读请求没有可验证的收益，反而会累积挂起请求。待存储侧排查 10:00 左右 114/115 对四层 target 配置文件的 FUSE/RPC 等待，以及此前 MTP rank00 的 inode 3146980 后，再复查同路径读取和完整模型加载。

### 114 的 FUSE 错误与 112 存储端口

随后以个人 UID 只读检查 114 的 FUSE 错误日志，找到了与本任务 MTP 首读对应的服务端结果：09:32:42，UID 19357313 的 `batchRead` 对 `ChainId(90000007)`、chunk `00000000-00000030-04E40000-00000000` 重试约 300 秒后返回 `StorageClient::NotAvailable(7005)`；紧接着 `hf3fs_read` 报 inode `0x3004e4`（十进制 3146980）、偏移 0、大小 131072 字节、错误 `-7005`。这是此前内核等待的直接错误记录。日志在 09:17–09:18 还反复报告连向 `11.163.39.112:8001` 的 TCP/RDMA 连接被拒绝；目前不能仅凭这些行断言该 chunk 的唯一副本就在 112。

10:04 后从 114 和 115 分别连接 `11.163.39.112:8001`，两次 `connect_ex` 都立即返回 `111`（Connection refused）。112 上没有该端口监听，Docker daemon 也不可连接，`/mnt/hf3fs` 退回本地 ext4。两端 target 小文件读取等待与 MTP 的 `NotAvailable` 一起说明当前应先恢复/核查 3FS 存储服务及路由，不宜通过增加读取线程或修改共享 FUSE 配置掩盖故障。此检查只读取了日志和端口状态，未启动或重启任何共享服务。

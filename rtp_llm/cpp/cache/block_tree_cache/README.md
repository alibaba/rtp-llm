# BlockTreeCache

RTP-LLM 与 vLLM 调用同一个原生 `BlockTreeCache`。它拥有 `BlockTree` 和每个缓存组的资源记录，并调用原有的 loader、storer、evictor 和 transfer dispatcher。模型请求的块表和设备内存由各引擎管理。

RTP 工厂创建实例后由 coordinator 直接调用 C++。vLLM 通过绑定调同一份 C++。下图中的路径相对于本目录。

```text
RTP  → integration/rtp_llm/BlockTreeCacheFactory.cc → BlockTreeCache.cc
vLLM → integration/vllm/btc_vllm/ → bindings/python/ → BlockTreeCacheC*.cc → BlockTreeCache.cc

BlockTreeCache.cc
├── load/BlockTreeLoader.cc     匹配前缀，安排加载
├── store/BlockTreeStorer.cc    发布计算完成的页
└── evict/BlockTreeEvictor.cc   按组选择副本降层或回收
    三条路径共享 BlockTree.cc、group_set/ 和 block_pool/ 的资源记录。
    本地搬运走 transfer/，远端查询和读写走 storage_backend/。
```

## 从入口看代码

| 路径 | 职责 |
|---|---|
| `BlockTreeCache.h/.cc`、`BlockTree.h/.cc`、`TreeNode.h` | 两端共有的 C++ 对象、前缀树、快照和生命周期。原生 RTP 的整数键和 vLLM 的完整块哈希分别保留原始身份。|
| `group_set/`、`block_pool/` | FULL、LINEAR、SWA 的命中规则、每组物理页和引用。设备、Host 与可选 Disk 节点在原生组资源中记录。|
| `load/`、`store/`、`evict/` | 匹配、等待目标块分配、加入在途加载、发布、写回以及按池回收。|
| `transfer/MultiRankBlockTransferEngine.cc` | 按一次操作预期的参与者合并逐页结果。RTP 在 `integration/rtp_llm/RtpRankBroadcast.cc` 收取 gRPC 结果，vLLM 在 `transfer/PolledTransfer.cc` 交付调度回执。|
| `transfer/PerRankBlockTransferEngine.cc`、`transfer/TransferWorkLoop.cc` | 两种 worker 实际运行的分批、排队、到期检查、完成和关停代码。RTP 接本地设备拷贝和 CRC。TPU 扩展通过 `include/block_tree_cache_worker.h` 调同一库，由 PJRT 的设备事件报告批次完成。|
| `storage_backend/kvcm/` | 两端共享 `SdkClient.cc` 的 SDK 注册、载入和存储，以及 `SdkWriteSession.h` 的写入会话与分片完成判定。RTP 提供本地块池，TPU 提供已注册的 Host 共享内存。TPU 对旧版 SDK 的校验和查询及提交适配也在这个目录。|
| `integration/rtp_llm/` | 将 RTP 的工厂、coordinator、真实块池、搬运与监控装配到原生对象。|
| `BlockTreeCacheC.cc`、`BlockTreeCacheCMatch.cc`、`bindings/python/`、`integration/vllm/` | C 句柄持有原生对象。Python 通过同一套 C 接口驱动它；vLLM 适配器把匹配返回的位置绑定到请求块表，TorchTPU 提供 PJRT 页搬运。|

### 一次请求准入怎样完成匹配

RTP 的 coordinator 调一次 `BlockTreeCache::match`，在远端结果回调中接着分配设备块。vLLM 的 Device/Host 路径调一次 `btc_match`。两者都由 `load/BlockTreeLoader.cc` 按组选择可复用的前缀。vLLM 通过以下步骤把结果交给自己的调度器。

1. `prepareHostAdmission` 在树锁内选出本地 Device、Host 页并持有引用。需要访问 KVCM 时，C 接口保留这次准入并返回 `PENDING`。有加载 token 上限的请求在首次匹配时就传入上限，已就绪的设备前缀仍可全部复用。
2. 调度器在后续轮次调 `btc_poll_match(request_id)`。KVCM 安全写入 Host 后，`completeHostAdmission` 按这次选中的组、位置及物理页校验原计划，补入远端查到的页。多个请求可共用一笔远端读取，各自领取自己的计划。远端读取失败或超时，可以交付仍有效的本地部分。
3. vLLM 安装请求块表，为缺失位置分配设备页，然后调 `btc_bind`。BTC 归还已由请求接管的设备引用。各 worker 通过 PerRank、WorkLoop 和 PJRT 执行拷贝，安全回执决定何时归还搬运引用。设备池分配失败或请求取消时，适配器撤销本次准入，后续重试会创建新的准入编号。

## vLLM 的分层迁移

vLLM 发布新算出的 KV 时，BTC 只在树上登记设备页。每个前缀节点的每个缓存组独立迁移。设备池达到 90% 后，公共淘汰器选冷页搬到 Host，目标是把设备占用降到 82%。容量取自 vLLM 的真实设备块池，并扣除保留块。只有 BTC 单独持有、当前步骤和其他请求均未占用的页才参与。请求分配设备块已经遇到短缺时，仍可直接回收符合条件的设备页，该页以后再用时可能需要重算。

Host 水位按实际的 `HostBlockPool` 计算，共用一个池的各组一起参与。Host 达到 94% 时向 90% 回落。有 KVCM 时，旧页写入远端、提交成功且缓冲区安全后才归还槽位。设备和 Host 之间则要等全部参与 worker 报告成功且缓冲区安全，并重新确认页仍有效，才把树中的位置换到目标层。搬运期间两端同时占用是正常的。直接调用 C 接口的客户端仍可用 `btc_demote(..., remove_device=0)` 显式保留两份。RTP 原有的 Device 发布写 KVCM 路径由其配置决定。

冷请求先查询 KVCM。若要读回 4 页而 Host 只空 1 槽，BTC 会优先为这批读安排 3 个槽的回收。已经在写回的页可以抵扣要新发的工作，真实空槽凑齐后才开始读。请求用到的页在等候期间得到保护。KVCM 到 Host 的读取完成后，调度器绑定具体的设备目标，全部搬运完成才归还 Host。设备目标失效时仍可从 Host 恢复。需求超过池容量时按所需缓存组共同允许的边界缩短前缀。

调度端使用 C ABI 2 和 `block-tree-cache` 2.1.0。vLLM 通过 `tiered_eviction=True` 启用两层水位，并调用 `btc_balance_device` 提交设备池快照。ABI 1 调用方需要随绑定一起升级。逐 rank worker 接口仍为版本 1。

## Host 槽位与 KV 数据

`block_pool/HostBlockPool.cc` 让 RTP 和 vLLM 用同一套空闲槽、引用计数、代际和行寻址。原生块号从 1 开始。在 RTP 中，`integration/rtp_llm/RtpHostBlockPool.cc` 申请锁页内存，保留第 0 行，块 1 使用第 1 行。在 vLLM 中，C 接口把块 1 导出为该池的第一个公开槽。使用全局共享池时，这就是槽 0，正好指向已有共享文件的第一行。

两种入口都直接创建 `HostBlockPool`。`BlockTreeCacheC.cc` 记录各组公开槽号的起点。启用 `release_local_pages` 时，最后一个树引用结束后，C 接口把该页放入 `RELEASE` 工作队列；执行端成功确认缓冲区安全后，池校验代际并回收。默认配置直接按原生引用计数回收。可选的本地磁盘文件由 `block_pool/DiskBlockIO.cc` 中的 `DiskFiles` 管理。

普通 TPU worker 和调度器打开同一份 `SharedOffloadRegion`。调度器把已有映射、行距、每行有效字节数传给 `Cache(..., host_region=...)`。KVCM 通过 `cache.host_slot_buffer(group, slot)` 取得公共池算出的行，worker 根据收到的槽号在自己的映射中发起设备拷贝。槽还有在途读写时，缓存保持引用；关闭成功后，绑定才释放借用的内存。[C 接口](include/block_tree_cache.h) · [共享 Host 池](block_pool/HostBlockPool.cc)

分片部署中，调度器的同一个 `HostBlockPool` 管全局槽位，每个 rank 的 worker 和 KVCM 用该进程现有的内存文件保存自己的分片。跨进程消息携带槽号。各 rank 按本地布局解析这一槽对应的数据范围，某个组在该 rank 上可以是零字节。[普通 TPU worker](integration/vllm/btc_vllm/torchtpu/worker.py) · [分片 worker](integration/vllm/btc_vllm/torchtpu/sharded_worker.py)

## 构建

在本目录或解压后的源码发行包中执行以下命令。CMake 和 RTP 的 Bazel 目标用同一份 `native_sources.bzl` 选择公共源文件。源码发行包附带原生头文件依赖的轻量值类型和计时工具。核心构建需要 C++17 与线程库；各引擎构建自己的设备适配与可选 SDK 后端。

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake --install build --prefix "$PWD/build/install"
python -m pip wheel --no-deps --wheel-dir dist .
```

C 调用方通过 `#include <block_tree_cache.h>` 使用公开 ABI，C++ 提供 `<block_tree_cache.hpp>` 轻量封装。Python wheel 携带同源构建的 `libblock_tree_cache`。[公共接口](include/block_tree_cache.h) · [RTP 适配](integration/rtp_llm/) · [vLLM 适配](integration/vllm/)

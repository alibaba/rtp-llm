# Sparse KV Offload 设计

Sparse KV offload 的核心分工是：**cache group 描述 KV 怎么存，模型决定本步读写哪些 KV，offload runtime 负责把选中行准备到 GPU。**

请求继续使用原 block table 和 block ID。一个 offload group 的正式 KV 可以同时位于 HBM 和 pinned Host；GPU 热缓存只保存最近被稀疏 attention 访问到的 Host 行，不参与请求 block 分配或 P/D 传输。

## 1. 目标与范围

- 只对 decode 阶段稀疏访问的正式 KV 做 offload；prefill、prefix reuse 和 decode reuse 保持原路径。
- `indexer_kv`、HCA、SWA 和 compressor state 仍在 HBM，不进入本机制。
- 不引入第二套对外 KV 地址。Host 行在 attention 前被映射到热缓存槽，原 sparse attention kernel 继续消费 GPU tensor。
- 当前接入 GLM/DSA sparse MLA 和 DeepSeek V4 FP8 CSA。MHA 已覆盖存储和 runtime，但还没有完整模型 attention 适配。

生产 decode 使用 `--reuse_cache 0`。这里关闭的是 decode prefix reuse；P 侧仍可单独启用 prefill prefix reuse。

开启 offload 的 D 侧不支持普通 context prefill，也不支持 `multi_task_prompt`；配置和运行时会提前拒绝这些路径。Target verify 和 MTP draft extend 使用独立的轮次标志，仍可运行。异步 prepare 在 hidden states 尚未生成时也使用同一标志。

真实请求只支持单序列：`num_return_sequences=1`、`num_beams=1`，且 `variable_num_beams` 为空。多返回序列和 beam search 会在调度、D 侧 KV 分配前返回请求级 `INVALID_PARAMS`（D 侧 RPC 为 `INVALID_ARGUMENT`），不会占用热缓存或进入 attention。此限制同时适用于 `static` 和 `dynamic` 热缓存模式；内部 batch warmup 不受影响。

支持默认的 `--warm_up 1`。预热使用最小的真实 tiered cache（2 个 HBM block 和 1 个 Host block），selection 上限和写入 staging 仍按部署配置分配，并执行 offload attention 路径。预热完成后释放临时 cache、plan 和 staging，再根据峰值显存建立生产 cache、重新绑定并捕获 graph。临时 KV backing 不计入运行期峰值；由容量公式单独预留的 offload buffer 也从测量值中扣除，避免重复计费。

P 侧 sparse MLA prefill warmup 使用最小的普通 GPU KV cache，使 indexer 和 sparse attention 都参与预热；预热完成后解除临时绑定，再创建生产 cache。P 侧不启用 offload。

## 2. 分层与调用链

```mermaid
flowchart TD
    M["模型层<br/>KVCacheSpecDesc / top-K or CSA selection"] --> C["存储配置<br/>CacheConfigCreator"]
    C --> P["DeviceBlockPoolConfigHelper<br/>DeviceBlockPool"]
    P --> S["OffloadStorage<br/>pinned Host + HBM resident planes"]
    S --> K["KVCacheManager / pybind KVCache"]
    M --> B["build_offload_plans"]
    K --> B
    B --> R["SparseOffloadRuntime<br/>mapping / admission / prefetch"]
    R --> A["MLA / V4 attention<br/>write / acquire"]
```

| 层 | 职责 | 主要代码入口 |
|---|---|---|
| 存储层 | 把模型声明转成每个 tag/layer 的 Host/HBM block、native planes、热缓存容量和 generation | [KVCacheSpecDesc.h](../rtp_llm/cpp/cache/KVCacheSpecDesc.h)、[CacheConfigCreator.cc](../rtp_llm/cpp/cache/CacheConfigCreator.cc)、[DeviceBlockPool.cc](../rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.cc)、[BufferTypes.h](../rtp_llm/cpp/cache/BufferTypes.h) |
| runtime 层 | 把 backing ID 转成 resident 索引，完成去重、热槽替换、异步 fetch、写回和版本检查 | [plan.py](../rtp_llm/models_py/modules/offload/plan.py)、[runtime.py](../rtp_llm/models_py/modules/offload/runtime.py)、[layout.py](../rtp_llm/models_py/modules/offload/layout.py) |
| 模型层 | 声明可 offload 的 group，产生选择结果，调用 `begin/write/acquire/drain` 后进入原 attention kernel | [deepseek_v2.py](../rtp_llm/models/deepseek_v2.py)、[deepseek_v4.py](../rtp_llm/models/deepseek_v4.py)、[mla_attention.py](../rtp_llm/models_py/modules/hybrid/mla_attention.py)、[V4 attention](../rtp_llm/models_py/modules/dsv4/fp8/attention.py) |
| 生命周期 | block 重新分配、P/D copy/load 或 speculative cache swap 后发布 generation，使旧热副本失效 | [DecodeRpcServer.cc](../rtp_llm/cpp/model_rpc/DecodeRpcServer.cc)、[NormalExecutor.cc](../rtp_llm/cpp/normal_engine/NormalExecutor.cc)、[MtpExecutor.cc](../rtp_llm/cpp/normal_engine/speculative/MtpExecutor.cc) |

层间交接有五个稳定接口：

| 接口 | 交接内容 |
|---|---|
| `KVCacheSpecDesc.offload` | 是否开启 offload，以及一次 selection 的行数上限 `max_selected_rows` |
| `KVCacheSpecDesc.offload_runtime_group` | 同一 tag 下哪些层共享选择结果和 runtime 元数据；未设置时每层独立 |
| `KVCacheSpec.entryLayout()` | 每个 native page 的 entry 数和各 plane 的 dtype、shape、offset、stride |
| `OffloadStorage` | Host/resident tensor views、block 几何、选择上限和所有层共享的 `generations` |
| `LayerOffloadPlan` | 某层对应哪个 runtime/storage，以及该层是否负责组首 `begin` 和组末 `drain` |

## 3. 存储层

### 3.1 block 与 plane 布局

一个 offload group 有 `N = H + P` 个正式 block：`[0, H)` 位于 HBM，`[H, N)` 位于 pinned Host。GPU 热缓存另外占用 resident plane 的尾部空间，不改变正式 block 编号。

`entryLayout()` 描述一次可选择的 entry，而不是假设所有模型都使用连续 MLA KV：

| 布局 | 一个 entry | plane |
|---|---|---|
| MLA | 一个 token 的压缩 KV | packed KV |
| sparse MHA | 一个 token 的 K/V | K、V，量化时另有 scales |
| V4 CSA | 一个压缩 KV entry | 576 B packed KV、8 B scale |

`PagedRowLayout` 直接使用 tensor 的 shape/stride 计算页内偏移，保留原生 K/V/scale 布局和页 padding。这样存储层无需把模型 tensor 重新打包成统一 dense 格式。

### 3.2 配置与分配

初始化链路为：

```text
模型 KVCacheSpecDesc
  -> CacheConfigCreator：按 tag 计算容量、布局、owner layer、runtime 数
  -> DeviceBlockPoolConfigHelper：生成 pool 配置和 native page geometry
  -> DeviceBlockPool::initializeOffloadStorage()：创建 pinned Host 和 resident CUDA planes
  -> OffloadStorage：暴露给 CacheManager、pybind 和 Python runtime
```

模型只声明策略，不直接分配 buffer。例如 sparse MLA 会把 `default` group 标记为 offload，并把同一 indexer 选择源的层设置为同一个 runtime group：

```python
policy = CacheOffloadPolicy()
policy.enabled = True
policy.max_selected_rows = concurrency * max(1, gen_num_per_cycle + 1) * hot_rows
policy.max_requests = concurrency
desc.offload = policy
desc.offload_runtime_group = runtime_groups[layer_id]
```

`hot_rows` 由 `--sparse_kv_offload_hot_rows` 配置，默认每请求、每个 query 2048 个 KV 位置；总量随并发和 speculative query 数扩展。该值必须为正且不小于模型 top-K，不改变模型自身的稀疏选择宽度。`max_selected_rows` 同时限定一次 `begin` 的 flat selection 上限；热缓存容量按 page 向上取整：

```text
resident_rows = ceil(max_selected_rows / rows_per_page) * rows_per_page
```

`OffloadStorage` 保存 `total_blocks`、`hbm_blocks`、`rows_per_block`、`rows_per_page`、`max_selected_rows`、`host_planes`、`resident_planes` 和 `generations`。C++ 持有正式 KV 和热缓存内存，Python 只持有同一块内存的 tensor view。

热缓存支持两种启动模式，使用 `--sparse_kv_offload_hot_cache_mode static|dynamic` 或环境变量 `SPARSE_KV_OFFLOAD_HOT_CACHE_MODE` 选择，默认 `static`。两种模式都共享一个热槽池；各请求没有固定独享的槽位范围，GPU gather 直接读取 pinned Host 的路径相同。

| 模式 | HBM 分配 | 开销与适用条件 |
|---|---|---|
| `static`（默认） | 启动时预分配最大并发对应的全部热槽，空闲也保留 | 容量固定，无扩缩容同步和物理分配版本检查；适合显存预算充足、重视延迟稳定性的部署 |
| `dynamic` | 按实际执行的活跃请求数映射，空闲释放热缓存独占物理页 | 减少低并发时的物理占用；并发变化有同步和映射开销，查找还需检查槽位分配版本 |

静态容量为 `resident_rows`；动态模式每个请求的容量为 `hot_rows × max(1, gen_num_per_cycle + 1)`，按 native page 对齐。请求减少时缩小，物理映射以 2 MiB 为单位；与正式 HBM KV 共用物理页的尾部空间会保留。静态模式不创建动态容量标量或物理分配版本数组，Triton 编译时去掉相关检查。两种模式的 HBM 预算都保留最大并发上限，防止活跃请求增加时显存不足。

动态模式下，`OffloadDeviceBuffer` 通过 CUDA VMM 预留最大虚拟地址范围，仅为正式 HBM KV 和当前活跃热缓存前缀建立物理映射。因此 resident tensor 的最大 shape、stride 和地址不随扩缩容改变，同一张 CUDA Graph 可以继续重放。`active_hot_rows` 是 GPU 容量标量；`hot_cache_epoch` 为每个槽保存物理分配版本。释放的槽在重新映射后必须重新 fetch，保留槽的缓存命中继续有效。metadata 和 selection scratch 仍按配置上限分配；HBM 预算也保留最大并发的容量上限，防止活跃请求增加时没有显存可用。

动态模式的扩缩容只在引擎的 forward 边界执行，先等待此前计算及 offload 搬运完成。并发不变时不做映射操作或 CPU 同步。并发变化时的驱动分配/释放有额外延迟，随需映射的物理页数增加。TP 各 rank 使用同步后的实际请求数，graph padding 和空闲 fake stream 不增加配额。MTP 内部 draft 子调用只能增加容量，target 调用负责按完整请求批次缩减，避免 rejection 后的紧凑 draft batch 提前归还仍在使用的配额。普通 decode 和 MTP 的空闲分支都会释放热缓存物理页。

默认 warmup 仍开启。warmup 期间热缓存增长所占的 driver 显存从 runtime 峰值测量中扣除，由 offload 的峰值容量预算统一计费。

模型构造器会填写 `max_requests`。动态模式下，旧版序列化 policy 缺少此字段时保留 `max_requests=0`：有请求就使用旧版完整 batch 配额，空闲时释放，以兼容无法推导单请求配额的旧描述。

同一 tag 的层共享 `generations`，但每层仍有自己的 `OffloadStorage` 和正式 KV。共享的是 block 版本与 runtime 管理状态，不是 KV 数据。

### 3.3 数据版本

block 被重新分配或覆盖后，`DeviceBlockPool` 增加该 block 的 generation：

```text
block allocation / overwrite
  -> DeviceBlockPool::onAllocatedNoLock()
  -> takeOffloadVersionUpdates()
  -> NormalExecutor / MtpExecutor 做 TP broadcast
  -> publishOffloadVersions()
  -> OffloadStorage.generations
```

`DecodeRpcServer` 在 P/D load 覆盖目标 block 后调用 `markOffloadOverwrite()`。Runtime 的热槽保存 `(block, generation)`；下一次 `begin` 发现 generation 不一致时重新 fetch，避免 request block 复用后读到旧热副本。

## 4. Runtime 层

`SparseOffloadRuntime` 按 `(tag, offload_runtime_group)` 创建。构造时校验所有 storage 的 block 几何、plane 形状和 `generations` 地址完全一致，然后分配：

- `mapping`：正式 backing row 到 resident 槽的映射。
- `owners/tags/versions/protected`：去重、替换保护、槽归属和版本。
- `_slots/_misses/_write_slots`：selection、miss 和写回 scratch。
- transfer stream 与每层 ready event：异步搬运和计算流依赖。

一次 selection 的执行接口如下：

| 接口 | 调用位置 | 行为 |
|---|---|---|
| `begin(backing_ids)` | MLA top-K 转成 `default` backing ID 后；V4 把 CSA 压缩 ID 转成全局 slot 后 | 检查 flat 行数，去重 HBM/Host 行，保护当前热槽，安排 miss 行从 Host 搬到每层 resident plane |
| `write(storage_index, KVWrite)` | MLA RoPE/pack 后；V4 compressor 写完 striped staging 后 | 等待本层 fetch，写入正式 HBM/Host backing，并同步仍有效的热副本 |
| `acquire(storage_index)` | 进入原 sparse attention kernel 前 | 让当前计算流等待该层 fetch，返回 resident planes 和重映射后的索引 |
| `drain()` | 共享组末层、下一次 `begin` 前、graph 生命周期边界 | 等待组内剩余搬运完成 |
| `reset()` | `reset_sparse_kv_offload()`，graph capture 外 | 清空 mapping/热槽/事件状态，保留正式 KV 和 block generation |

`acquire` 是 CUDA stream 依赖，不是锁。`begin` 发起整组各层的 fetch，后续层搬运可与当前层 attention 重叠。

```mermaid
sequenceDiagram
    participant M as 模型 attention
    participant R as SparseOffloadRuntime
    participant T as Transfer Stream
    participant A as Attention Kernel
    M->>M: top-K / CSA selection，翻译为 backing IDs
    M->>R: begin(backing_ids)
    R->>T: 各组层安排 miss rows 的 Host -> GPU fetch
    loop 组内各层
        M->>M: Q/KV 计算、RoPE、pack
        M->>R: write(storage_index, KVWrite)
        M->>R: acquire(storage_index)
        R-->>M: KVReadView(resident_planes, indices)
        M->>A: 原 sparse attention kernel
    end
    M->>R: drain()
```

`begin` 本身会先调用 `drain()`，因此一个 runtime 同时只有一份活动 selection。这个约束依赖引擎串行执行 forward 并保证 backing block 的生命周期。

## 5. 模型层

模型层只负责“选择谁”和“何时调用 runtime”，不直接处理 Host 地址或热槽替换。

| 模型路径 | runtime 分组 | 调用方式 |
|---|---|---|
| GLM IndexShare / 共享选择结果的 sparse MLA | 同 tag、同 `offload_runtime_group` 的层共用一个 runtime | 组首层 `begin`，各层独立 `write/acquire`，组末层 `drain` |
| 不共享选择结果的 DSA MLA | 每层独立 runtime | 每层产生自己的 top-K，再执行 `begin/write/acquire/drain` |
| DeepSeek V4 FP8 CSA | 每层独立 runtime，只 offload `csa_kv` | 压缩 ID 经 block table 翻译后 `begin`，compressor staging 经 `write` 写入，attention 用 `acquire` 返回的新索引读取 |

GLM 的各层 KV 仍独立。共享 runtime 只共享“本步选择了哪些历史行”和 mapping/热槽元数据，避免组内每层重复做一次相同的 selection admission。

MLA 的桥接点是 `MlaAttention.configure_offload()` 和 `SparseMlaImpl.forward()`：前者把 `LayerOffloadPlan` 和 writer 装到模型层，后者在 RoPE 后调用 `write()`，在 attention 前调用 `acquire()`。

V4 的桥接点是 `DeepSeekV4Model._initialize_sparse_offload()`、`configure_sparse_offload()` 和 `compressor.write_offloaded_kv()`。CSA 不能直接使用 token ID，必须使用压缩 entry ID；SWA、HCA 和 compressor state 仍走各自的原生 group。

## 6. 生命周期与 SWA

正常 decode 的调用顺序是：

1. C++ 完成 block 分配、copy/load，并在必要时更新 generation。
2. Executor 将 generation update 通过 TP 同步后发布到 GPU tensor。
3. 模型产生 top-K/CSA selection，调用 `begin`。
4. runtime 完成 Host 行 admission 和 fetch，模型写新 KV 并读取 resident KV。
5. 组末 `drain`，backing block 保持由原 cache manager 管理。

SWA 清理还有一个独立的单位约束：`active_tail_blocks` 是 block 数，`reserve_step` 是 token 数。保留区必须换算成 block：

```text
reserve_blocks = ceil(reserve_step / seq_size_per_block)
```

`SWACacheManager::removeSkippedBlocks()` 只释放 active tail 和 reserve blocks 之前的历史 block。Coordinator 端到端测试
`CompleteTokenIdsReserveStepReclaimsSwaRollingWindow` 从 `CompleteTokenIds` 读取 `reserve_step`，经过真实 allocator 路径，检查最终有效 block 数，避免再次混用 token/block 单位。

## 7. 部署

最小 D 侧增量配置：

```bash
--role_type DECODE
--enable_sparse_kv_offload 1
--sparse_kv_offload_host_mb 53248
--sparse_kv_offload_hot_rows 2048
--sparse_kv_offload_hot_cache_mode static
--kv_cache_mem_mb 0
--reuse_cache 0
```

`--kv_cache_mem_mb` 是所有 HBM cache 和热缓存的总预算；设为 0 时由部署脚本或固定 pool 参数控制容量。`--sparse_kv_offload_host_mb` 是每个 rank 的 Host cache 总预算，包含 offload backing 和其他放在 Host 上的 cache pool。开启 `--dsv4_fixed_pool_use_memory 1` 时，固定池的显式容量和随总 block 数增长的容量也计入该预算；预算连最小布局都无法容纳时，在分配前报错。热缓存容量由 `--concurrency_limit`、`--sparse_kv_offload_hot_rows` 和 MTP/DSpark 的每轮 token 数共同决定；也可通过环境变量 `SPARSE_KV_OFFLOAD_HOT_ROWS` 配置每请求容量。

启用 `--use_batch_decode_scheduler` 时，`--batch_decode_scheduler_batch_size` 必须在 `1..concurrency_limit` 范围内；启动和运行时 `UpdateSchedulerInfo` 都检查这一限制。超限更新返回 `INVALID_ARGUMENT`，保留原调度配置。

P/D 两端必须使用一致的模型和 speculative 配置。offload 只在 D 打开，P 不打开。本次验证的 V4 四卡部署为：

- P：`CP2 / EP2`，GPU 0/1。
- D：`TP1 / DP2 / EP2`，GPU 2/3。
- DSpark `gen_num_per_cycle=3`，因此实验使用 `reserve_step=9`。
- DSpark 与 D 侧 CUDA Graph 开启。

## 8. 实验与结果

以下吞吐来自此前的部署配置，不能直接作为本次修订的性能验收。当前默认 `hot_rows=2048`；V4 top-K 为 512 时，热缓存及相关 selection scratch 的容量上限是按 512 配置时的 4 倍，正式 Host KV 容量不因此改变。部署时必须重新核算 HBM 预算，再在相同请求长度、并发和采样设置下测量。

以每个 D rank 并发 192、gamma=3 为例，43 层 V4 Flash target 有 21 个 CSA 层，每 entry 为 584 B；仅 target 热缓存 plane 从 512 调到 2048 就增加约 `192 × 4 × (2048-512) × 584 × 21 = 13.47 GiB/rank`，尚不含 draft 和管理 scratch。V4 可显式配置 `--sparse_kv_offload_hot_rows 512` 来匹配此前的容量设置；这也减少跨步保留的热行，需同时复测 Host 读取量与吞吐。GLM/DSA top-K=2048 的模型仍要求至少 2048。

实验环境为 DeepSeek-V4-Flash-0731、4 张 L20D，并按 L20A 有效 HBM 预算 `189470 MiB/GPU` 校准。D 卡纯模型权重为 `82.105 GiB`，KV/state HBM pool 为 `66.38/66.17 GiB`，Host KV pool 分别为 `32/52 GiB`。

数据集是固定 synthetic 长文本：同一份重复技术文档经模型 tokenizer 编为恰好 `131072` tokens，所有请求复用该输入。采样设置为 `top_k=1`、`random_seed=42`、`ignore_eos=false`，输出到自然 EOS 或 1M context limit。测量口径为 D 侧 egress、P 转发前，排除 prefill 和 KV load。

两档均通过资源审计：

| 设置 | 并发/D | Host KV | 成功请求 | 聚合 decode TPS | TPOT P50 | TPOT P99 | CSA 读放大 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Offload-150 | 150 | 32 GiB | 300/300 | 9789.4 | 27.995 ms | 30.841 ms | 51.91 KB/token |
| Offload-192 | 192 | 52 GiB | 384/384 | **10377.4** | 34.965 ms | 39.130 ms | 129.01 KB/token |

150 档是干净复现交叉验证。192 档没有漏测：干净环境下它是本次两档中的最高聚合吞吐点，但 TPOT 和 Host 读放大也更高。192 档早期的 `9281.3 TPS` 来自 GPU 污染窗口，不作为结果。

最终功能回归沿用同一模型、同一 `131072`-token 输入和上述 2P2D 部署（P：GPU 0/1，D：GPU 2/3），同时提交 4 个请求，Host KV 为 `32 GiB`，decode reuse 为 0。4/4 请求自然 EOS，输出长度分别为 1344、1510、1523、1765，每个 D rank 各接收 2 个请求；D 侧聚合 decode 为 `587.70 token/s`，TPOT P50/P99 为 `5.962/6.022 ms`。该点用于 128K 输入和 SWA rolling-window 修复的端到端功能验收，不与 150/192 容量吞吐点做性能排名。

## 9. 验证与边界

关键回归入口：

```bash
bazel test --config=cuda13 \
  //rtp_llm/cpp/cache/test:hybrid_pool_kv_cache_allocator_test \
  //rtp_llm/cpp/cache/test:tiered_cache_config_test \
  //rtp_llm/config/test:kv_cache_config_pickle_test \
  //rtp_llm/config/test:server_config_setup_test \
  //rtp_llm/cpp/cache/block_tree_cache/test:device_swa_cache_manager_test

bazel test --config=cuda13 \
  //rtp_llm/models_py/modules/offload/test:dsv4_offload_gpu_test \
  //rtp_llm/models_py/modules/offload/test:dsv4_offload_mega_gpu_test_x86 \
  //rtp_llm/models_py/modules/offload/test:sparse_kv_offload_gpu_test
```

C++ 回归覆盖 SWA reserve 单位、pool sizing、storage/version 和 coordinator 路径；GPU 入口覆盖 Host 更新后的 graph replay、compressor 写回和多 token verify。

真实模型 P/D smoke 使用默认 warmup，覆盖 offload 开关、eager/CUDA graph、长输入及 block 复用；MTP smoke 还覆盖真实 NextN draft 轮次。offload 启动日志必须包含 warmup 峰值记录：

```bash
bazel test --config=cuda13 \
  //rtp_llm/test/smoke:sparse_kv_offload_model_test \
  //rtp_llm/test/smoke:glm5_mtp_offload_model_test
```

普通 smoke 使用 GLM-5 和 DeepSeek V3.2 四层 checkpoint。MTP smoke 默认使用 `GLM-5-FP8-4layer` 作 target、`GLM-5-BF16-4layer` 中的真实 NextN 权重作 draft，通过正常加载器按 `FP8_PER_BLOCK` 量化 draft，使单卡 graph 的 FP8 MoE backend 同时适用于 target 和 draft。可通过 `RTP_GLM_MTP_CHECKPOINT` 和 `RTP_GLM_MTP_DRAFT_CHECKPOINT` 指定本地路径。裁剪 checkpoint 保留原始 NextN 层号时，测试生成独立的 draft 配置来匹配权重偏移。上述默认命令不加载 GLM-5.3。

GLM/DSA 独立精度 smoke 同时比较完整 greedy token 序列和最后一步全词表 logits（`rtol=atol=1e-5`），覆盖超过 2048 token 的输入、重复请求导致的 block 复用、2 个 HBM block 之外的 Host KV、eager/graph 和 offload 开关。比较前先检查无 offload 基线自身可重复，避免把基线波动当成 offload 误差。该入口使用默认的 MoE drain 策略，不注入稳定 top-k 测试 hook。V4 DSpark API 返回接受后的 token IDs，V4 smoke 严格比较这些 IDs；attention tensor 的数值对照由 V4 GPU 测试覆盖。

这三个精度入口默认验证 `static`；使用 `--test_env=RTP_TEST_SPARSE_KV_OFFLOAD_HOT_CACHE_MODE=dynamic` 可对动态模式运行相同用例和精度断言。动态模式的 GLM/DSA 和 MTP 用例还检查低于最大并发的配额以及空闲释放事件。

```bash
bazel test --config=cuda13 \
  //rtp_llm/test/smoke:sparse_kv_offload_accuracy_model_test

# V4 入口要求提前准备本地 checkpoint；CP overlap 和真实 DSpark 轮次开启。
bazel test --config=cuda13 \
  --test_env=RTP_DSV4_TEST_CHECKPOINT=/local/models/DeepSeek-V4-Flash-0731 \
  //rtp_llm/test/smoke:dsv4_sparse_kv_offload_accuracy_model_test
```

原生 fast top-k 的边界同分候选按较小索引优先选择，输出按索引排列，保证 attention 的输入集合和归约顺序可重复。正常长行路径继续使用原 radix 筛选，仅排序选出的 K 个索引；短行或边界同分需要截断时使用稳定 compaction。粗筛候选超过 shared-memory 缓冲容量时，对完整行执行 radix refinement，避免截断候选后选错 top-K。各路径复用原有 shared memory，不增加持久 GPU buffer。GPU 回归覆盖同分、正负零、64K 近似分数导致的候选溢出、paged/ragged 转换，以及修改输入后的 CUDA Graph replay。

本次在 L20D 上以同一进程、同一随机输入交替运行修复前后 kernel，两个版本均以 NVCC `-O3`、SM103 独立扩展编译；当前 Bazel 产物另行复测，耗时相近。CUDA Graph 内每次包含 10 次调用，取 5 组计时的中位数。以下是 top-k 本身的耗时，不是整模型吞吐：

| 行数 | 候选数 | K | 修复前 | 修复后 |
|---:|---:|---:|---:|---:|
| 1 | 526 | 512 | 5.34 µs | 6.36 µs |
| 1 | 4096 | 2048 | 3.30 µs | 9.43 µs |
| 1 | 131072 | 2048 | 51.82 µs | 61.76 µs |
| 128 | 131072 | 2048 | 55.36 µs | 65.58 µs |
| 1052 | 4096 | 2048 | 32.68 µs | 71.24 µs |

稳定选择和输出顺序增加了计算成本，尤其是大量短行的 prefill。该开销不能用历史 offload TPS 抵消或推算；当前修订的部署吞吐需要在空闲设备、相同模型和请求分布下重新测量。runtime 的 decode 热路径继续使用 GPU admission 和逐层 CUDA event 依赖，没有加入逐步 `.cpu()`/`.item()` 或 CPU synchronize；默认不在每层 MoE 前 drain，仍保留跨层 prefetch overlap。

另测 65536 个近似分数（`1.001 + (index % 128) * 1e-6`）触发候选溢出回退：单行 43.15 → 118.31 µs，128 行 44.55 → 121.57 µs。修复前该分布的结果不是真正 top-K，旧耗时只能用于说明修复成本。回退通过范围检查跳过共同高位字节；全同分输入直接走快捷路径，单行 46.21 → 32.95 µs。溢出分布使用每 graph 3 次调用、3 组各 10 次 replay 的中位数。

MTP 的 host 状态路径在 rejection 后压缩 draft extend 行数，异步 prepare 必须使用压缩后的输入。DSpARK 和保留 device 行数的异步路径仍可提前 prepare；没有为修复精度而关闭整个异步执行。

MTP smoke 使用同一 MTP 配置的无 offload 结果作为严格参考，比较完整串行/并发 token 序列及 graph 结果；串行请求还严格比较非 MTP 控制组。并发的 MTP/非 MTP 差异单独写入 `non_mtp_mismatches`，不作为 offload 的精度基线。已通过独立 attention 回放确认：当前 FlashMLA 会按 query batch 形状改变 split-K；同一 query 和逐字节一致的所选 KV，在 32/11 splits 下输出最多相差 `6.1035e-5`，四层 checkpoint 的后续生成可因此产生不同 token。本次不承诺跨 batch 逐 token 一致。

- 一个 runtime 当前只支持一份活动 selection，依赖引擎串行 forward 和 backing block 生命周期管理。
- 已观察到 P/D load 取消后的在途回调问题，本分支未修复。
- 性能数字来自 L20D 上的 V4 Flash FP8 KV，不外推到 V4.1、BF16 KV 或其他 attention layout。
- 本次只有 natural-EOS primary 批次，没有 profiler 批次的组件级 GPU 空闲拆分。

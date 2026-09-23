# MiniMax-M3.1 NVFP4 KV/idxK Native Fused Kernel 设计

## 0. SGLang Q8KV4 demo 对照结论（2026-09-23）

本轮只复用 SGLang demo 已验证的数据流和算子语义，不复制它的 cache
指针公式。两边的核心差异如下：

| 项目 | SGLang demo | RTP-LLM 当前 ABI | 处理结论 |
| --- | --- | --- | --- |
| Q / idxQ | scale-1 E4M3，即 Q8 | phase-1 reader 保留 BF16 Q；旧实验另有 Q4 | 新增 Q8 路径；Q4 不能冒充 M3.1 Q8KV4 |
| packed K/V | 独立 slot-major NHD tensor | 每物理 page 一行，K/V 共用 `kv_cache_base` | 保留 RTP page-row ABI，用显式 plane view/stride |
| K/V scale | 独立 slot-major compact block-16 scale | `kv_scale_base` 前半段，且每 page 有额外 side stride | 不能复制 SGLang `page*128+row` 指针公式 |
| idxK | 独立 slot-major packed tensor + scale | 位于 `kv_scale_base` 的 main-scale 之后 | 必须由 `NVFP4CacheLayout.indexer()` 导出 offset |
| PD | demo 未验证 P/D scale 传输 | 两个 opaque block 同生命周期传输 | 暂不拆四 block，也不改变已跑通的 PD ABI |
| compute path | 选中页预反量化到紧凑 E4M3 | RTP native kernel 直接读 packed cache | prefill/decode/target-verify 只允许 packed FP4 原生路径；BF16 working-page runtime 已删除 |
| CUDA Graph | demo Triton 路径有专用 graph 合同 | RTP graph bucket/workspace 所有权不同 | target-only DP8 已完成 E2E capture/replay；PD/DSpARK 仍需独立门禁 |
| prefill / CP | demo 是自身 page table 和调度器 | RTP 有 CP page-RR、prefix gather、compact prefill | 不把 decode kernel 误接到 CP prefill；单独设计/验证 |

`M3_NVFP4_DECODE_BACKEND` / `M3_NVFP4_PREFILL_BACKEND` 已删除。
`NVFP4_KV_CACHE=1` 的唯一运行时合同是：Q/idxQ 使用 Q8 计算语义，writer
直接写 RTP packed E2M1 values + E4M3 scales，prefill、普通 decode 和 target
verify/DSpARK 均由原生 FP4 reader 消费；不得构造 BF16 working pages。

### 需要 reviewer 明确确认的决策

1. **发布时点**：CUDA Graph、target verify/DSpARK 和同 ID 精度矩阵是发布
   gate；失败时应修复原生路径，而不是回退 BF16 working pages。
2. **数值目标**：生产 kernel 是要求与 SGLang/fmha_sm100 bit-exact，还是允许
   在同 ID 模型精度通过后使用数学等价但 softmax reduction 顺序不同的 RTP
   Triton kernel。当前原型属于后者，不能宣称 bit-exact。
3. **CUDA Graph scratch ABI**：当前使用 Q8KV4 独立 workspace，按 device、
   batch、heads、max-blocks、top-k 和 chunk 数缓存固定地址，并已通过单算子
   capture/replay。需要 reviewer 确认不同 graph runner 是否保证串行 replay；
   若允许并发 stream，同 shape workspace key 还必须加入 runner/stream owner。
4. **Prefill/CP 路线**：长 prefill 的 idxK 可以像 SGLang 一样先转 E4M3 紧凑
   页以复用多 query tile；主 K/V 只转换 top-k 页。CP page-RR 的 owner 和
   prefix restore 必须在转换前完成，不能直接套 SGLang slot-major helper。
5. **PD block 数**：第一阶段继续传 `kv + kv_scale` 两个 opaque block；四 block
   只保留为后续可选 ABI，不应为了 kernel pointer 方便提前破坏 cache key 和
   publication 原子性。

### 当前不能直接复制的 SGLang 代码

- `_store_fp4`：RTP 已有显式 RNE、scale 上下界和 positive-zero canonicalization，
  且 writer 要处理 CP/DSpARK physical slot；SGLang writer 没有这些 RTP 合同。
- slot-major pool 类：会破坏 C++ cache sizing、prefix reuse 和 PD descriptor。
- page-table/CSR 构造：SGLang 的 request/page ownership 与 RTP CP/PD 不同。
- 训练兼容 backend 路由：RTP 还要覆盖 target verify、DSpARK 和分角色 CUDA
  Graph，不能用一个全局 `training_attention` 分支替代。

可以复用/移植的是：Q8 cast 语义、packed E2M1 到 E4M3 的读取方式、decode
idxK 单 tile 直接解包、长 prefill referenced-page 预反量化策略，以及只处理
top-k main K/V pages 的工作集边界。

## 1. 文档目的

MiniMax-M3.1 的 KV4/idxK4 由 attention/indexer 融合算子直接读取 paged FP4 cache 以及对应的 E4M3 scale，并在算子内部完成解包、反量化和计算。运行时 BF16 working-page 路径已删除；离线 reference 仅用于单算子对拍，不能成为服务 fallback。

本文固定以下内容：

- FP4 cache 的目标物理布局；
- Main K/V 和 idxK 的 scale 存放方式；
- Prefill/Decode/CP/PD 分离的数据流；
- native fused kernel 所需的 ABI 和 offset；
- E2M1/E4M3 直接读取、解包、反量化方法；
- 性能、对齐、CUDA Graph 和 page boundary 约束；
- 离线 BF16 reference 与生产运行时的隔离边界；
- 分阶段实现计划和验收标准。

## 2. 设计结论

当前“两块物理内存 + 明确的 FP4/scale planes”方向可以保留：

```text
kv_cache_base   : packed Main K/V FP4 values
kv_scale_base   : Main K/V scales + idxK FP4 values + idxK scales
```

但它必须从“Python 侧的隐式 side region 约定”升级为 C++、Python、CUDA/Triton 共用的固定 layout ABI。不能让 native kernel 依赖诸如“idxK 恰好位于 scale region 后面”这种未显式导出的约定。

PD 传输格式不需要做数值转换：Prefill 发送原始 FP4 value bytes 和 scale bytes，Decode 原样恢复后由 native kernel 直接消费。PD 层不做反量化、不传 BF16 工作页、不重排 idxK。

可选的后续目标态可以将当前的两个 opaque block 进一步拆成四个逻辑 block：

```text
1. main_kv_fp4       Main K/V packed values
2. main_kv_scale      Main K/V E4M3 scales
3. idx_k_fp4          idxK packed values
4. idx_k_scale        idxK E4M3 scales
```

四个 block 必须被视为同一个 layer/page 的不可分割 cache bundle，而不是四个可以独立命中、独立复用的 cache object。这样既能让 native kernel 获得四个明确 pointer，也能保证 PD 恢复时不会把 value 和 scale 配错。四-block 不是第一阶段 native kernel 的前置条件。

## 3. 当前实现和最终目标的关系

### 3.1 当前所处阶段

生产路径已经完成从 phase-1 BF16 working-page 方案到 packed-FP4 reader 的
切换。`NVFP4_KV_CACHE=1` 时不存在 backend 选择开关，也不允许静默回退：

```text
BF16 K/V、idxK
    -> fused writer 写 E2M1 packed values + per-16 E4M3 scales
    -> kv_cache_base / kv_scale_base 两块持久化与 PD opaque 传输

CP Prefill
    -> fmha_sm100 FP4 IndexScore
    -> production TopK
    -> fmha_sm100 packed-FP4 sparse attention

Decode / target verify
    -> Q/idxQ cast 到 scale-1 E4M3
    -> packed idxK IndexScore
    -> production TopK
    -> packed main K/V sparse attention
```

同一份 cache 同时被 IndexScore 和 attention 读取。writer、reader 和 PD
恢复必须共享 `NVFP4CacheLayout` 导出的 value/scale plane 与 byte offset；
不能用孤立算子对不同 scale layout 的结果冒充整链验证。

M3.1 的真实 shape 是 4 个 Indexer-Q heads、4 个 KV heads、TopK=16，
不存在 2:1 Indexer head 合并导致的 TopK=32。正式性能数据必须使用这一
shape；8-head 只保留作泛化单测。

当前两个物理 block 在 kernel 接口暴露四个逻辑 view：

```text
main_kv_fp4_view
main_kv_scale_view
idx_k_fp4_view
idx_k_scale_view
```

这样可以同时满足：

- 不修改当前 PD 两块传输协议；
- native kernel 不依赖模糊的 side-region 约定；
- 将来可以无语义变化地切换为四个物理 block。

BF16 解包代码只允许用于离线单算子 oracle/格式检查。它不能分配在服务热
路径、不能成为不支持 geometry 的安全降级，也不能作为 PD 传输格式。

当前仍需独立关闭的发布 gate 是：真实 PD4+4 的 CUDA Graph on、DSpARK
完整链、任务级精度矩阵，以及不同 graph runner/stream 是否会并发复用同
shape workspace。2026-09-24 已补充的 target-only DP8 CG-on profiling 只
证明普通 decode 图路径，不能替代上述 PD/DSpARK gate。

## 4. FP4 数值格式

### 4.1 E2M1 value

每个值占 4 bit，每两个值打包到一个 byte。正值幅度码为：

| magnitude code | 数值 |
|---:|---:|
| 0 | 0.0 |
| 1 | 0.5 |
| 2 | 1.0 |
| 3 | 1.5 |
| 4 | 2.0 |
| 5 | 3.0 |
| 6 | 4.0 |
| 7 | 6.0 |

最高 bit 表示符号。零值必须强制编码为正零 `0000`，禁止产生负零 bit pattern。

### 4.2 E4M3 scale

Main K/V 和 idxK 均采用每 16 个连续元素一个 scale：

```text
group_size = 16
scale[g] = E4M3(amax(group) / 6.0)
```

scale 限制为 E4M3 可表达范围：

```text
1 / 512 <= scale <= 448
```

反量化为：

```text
value_bf16 = decode_e2m1(code) * decode_e4m3(scale)
```

### 4.3 RNE ladder

量化不能使用硬件默认 round-half-up。E2M1 的 RNE 阈值如下：

```text
if   a > 5.0:   magnitude = 7  // 6.0
elif a >= 3.5:  magnitude = 6  // 4.0
elif a > 2.5:   magnitude = 5  // 3.0
elif a >= 1.75: magnitude = 4  // 2.0
elif a > 1.25:  magnitude = 3  // 1.5
elif a >= 0.75: magnitude = 2  // 1.0
elif a > 0.25:  magnitude = 1  // 0.5
else:           magnitude = 0  // 0.0
```

在 native reader 和 reference reader 中必须使用相同的 decode 规则；否则会出现 FP4 cache 和 BF16 working page 对拍不一致。

## 5. 目标物理布局

每个 layer、每个 physical page 的布局固定为两个 byte region。

### 5.1 `kv_cache_base`

```text
kv_cache_base[physical_block]
├── main_k_fp4  [kv_head, token, head_dim / 2]
└── main_v_fp4  [kv_head, token, head_dim / 2]
```

Main K 和 Main V 都是 E2M1 packed byte。每两个 head-dimension 元素占一个 byte。

Main value block 的大小为：

```text
main_kv_bytes =
    2 * kv_head_num * page_size * head_dim / 2
```

即：

```text
main_kv_bytes = kv_head_num * page_size * head_dim
```

### 5.2 当前两-region 兼容布局：`kv_scale_base`

```text
kv_scale_base[physical_block]
├── main_k_scale [kv_head, token, head_dim / 16]  E4M3
├── main_v_scale [kv_head, token, head_dim / 16]  E4M3
├── idx_k_fp4    [token, idx_head_dim / 2]         E2M1 packed
└── idx_k_scale  [token, idx_head_dim / 16]        E4M3
```

Main K/V scale 区域大小：

```text
main_scale_bytes =
    2 * kv_head_num * page_size * (head_dim / 16)
```

idxK value 和 scale 区域大小：

```text
idxk_value_bytes = page_size * idx_head_dim / 2
idxk_scale_bytes = page_size * (idx_head_dim / 16)
```

最终每个 block 的 scale-side stride 为：

```text
kv_scale_stride_bytes =
    main_scale_bytes
    + idxk_value_bytes
    + idxk_scale_bytes
```

这是当前实现以及第一阶段 native kernel 推荐采用的布局：idxK values/scales 复用 `kv_scale_base` 的尾部。native kernel 通过显式 offset 直接读取，不需要先拆成四个物理 block。四个独立 logical plane/physical block 是后续可选演进，不应阻塞第一阶段融合 kernel。

### 5.3 可选的四-block 目标布局

如果后续需要更清晰的物理 pointer、独立对齐或独立 cache policy，可以把上述布局表达为四个独立的 per-block allocation/transfer region：

```text
main_kv_fp4_block:
    [main_k_fp4][main_v_fp4]

main_kv_scale_block:
    [main_k_scale][main_v_scale]

idx_k_fp4_block:
    [idx_k_fp4]

idx_k_scale_block:
    [idx_k_scale]
```

对应的 block stride 为：

```text
main_kv_block_stride   = main_kv_bytes
main_scale_block_stride = main_scale_bytes
idx_k_block_stride      = idxk_value_bytes
idx_k_scale_stride      = idxk_scale_bytes
```

四个 block 都使用相同的 physical block id 和 page table。算子接口可直接接收：

```text
main_kv_fp4_ptr
main_kv_scale_ptr
idx_k_fp4_ptr
idx_k_scale_ptr
```

此时不需要在 kernel 内部计算 `idx_k_offset` 和 `idx_k_scale_offset`，但仍必须传入四个 block 的 byte stride。

四-block 方案的优点：

- native kernel 获得四个语义明确、类型一致的 byte pointer；
- Main KV、idxK value、idxK scale 可以独立做对齐和 vector load；
- idxK 不再借用名为 `kv_scale_base` 的 side region，减少误用；
- 未来可以针对 idxK 单独做压缩、预取或 cache policy；
- PD 调试可以分别校验四种 byte pattern。

四-block 方案的代价：

- cache allocator 从两块 backing storage 扩展为四块；
- cache registration/RDMA MR 需要注册四个 region；
- cache-store/RPC descriptor 不能再只使用 `BlockInfoPair {kv, kv_scale}`；
- PD 元数据和 transfer request 的 descriptor 数量增加；
- 四个 block 必须保持相同的生命周期、引用计数、释放顺序和 cache key；
- CP compact/page-RR 下必须同时映射四个 region，不能只重排 Main KV；
- CUDA Graph 和 kernel input contract 要从两个 pointer 扩展为四个 pointer。

建议所有 plane 起始 offset 按至少 16B 对齐，整个 block stride 按 128B 或 256B 对齐，以减少 native kernel 的非合并访问和跨 cache-line 读取。

### 5.4 必须显式导出的 offset/pointer

不能只导出总 stride。C++ layout 应明确生成：

```cpp
struct NVFP4CacheLayout {
    size_t kv_block_stride_bytes;
    size_t scale_block_stride_bytes;

    size_t main_k_offset;
    size_t main_v_offset;
    size_t main_k_scale_offset;
    size_t main_v_scale_offset;
    size_t idx_k_offset;
    size_t idx_k_scale_offset;
};
```

如果采用四-block 目标布局，结构可以改为：

```cpp
struct NVFP4CacheBlocks {
    BlockInfo main_kv_fp4;
    BlockInfo main_kv_scale;
    BlockInfo idx_k_fp4;
    BlockInfo idx_k_scale;
};
```

`NVFP4CacheBlocks` 必须带有同一个 `layer_id`、`physical_block_id` 和 bundle identity。不能把四个 block 当作四个独立 cache key，否则会出现 value 已命中但 scale 未命中、或 idxK 来自另一个请求的错误组合。

Python、C++、CUDA、Triton 共享同一套定义。offset 应由 C++ cache layout 计算并透传给算子，Python attention 不应自行复制一份偏移推导逻辑。

不建议在每个 page 前增加 header：

- header 会增加 cache 和 PD 传输开销；
- header 对每个 block 都重复；
- CUDA Graph 下会增加不必要的 layout 元数据访问；
- 所有 geometry 已经可以通过固定 config 和 stride 表达。

四-block 方案应把元数据放在 transfer descriptor/config 中，而不是写进每个 page 的数据区。

## 6. Native Main K/V 读取

假设 block table 给出物理 page，token 位于 page 内的 `token_offset`，head-dimension 为 `d`。

### 6.1 packed value 位置

```cpp
physical_block = block_table[logical_block];

byte_offset =
    physical_block * kv_block_stride_bytes
    + plane_offset
    + head * page_size * head_dim / 2
    + token_offset * head_dim / 2
    + d / 2;
```

读取：

```cpp
uint8_t packed = kv_cache_base[byte_offset];

uint8_t code = (d % 2 == 0)
             ? (packed & 0x0f)
             : (packed >> 4);

float fp4_value = decode_e2m1(code);
```

nibble 顺序必须固定，并由 quantize kernel、reference reader、native reader 共用。不能让 Main K 和 idxK 使用不同的隐式 nibble 约定。

### 6.2 scale 位置

```cpp
group = d / 16;

scale_offset =
    physical_block * kv_scale_stride_bytes
    + scale_plane_offset
    + head * page_size * (head_dim / 16)
    + token_offset * (head_dim / 16)
    + group;
```

```cpp
float scale = decode_e4m3(kv_scale_base[scale_offset]);
float value = fp4_value * scale;
```

一个 16-dim group 的 scale 应该在一个 warp/tile 内加载一次，并在对应的 16 个 FP4 value 上复用。不能对每个标量 value 重复加载和解码 scale。

## 7. Native idxK/indexer 读取

idxK 不应先完整转换为 BF16 再做 index score。native indexer kernel 应直接完成：

```text
idxK FP4 load
→ E2M1 unpack
→ idxK E4M3 scale load/decode
→ dequant
→ Q_indexer · K_indexer
→ score/top-k
```

idxK value 起始位置：

```cpp
idx_k_value_base =
    physical_block * kv_scale_stride_bytes
    + idx_k_offset;
```

idxK scale 起始位置：

```cpp
idx_k_scale_base =
    physical_block * kv_scale_stride_bytes
    + idx_k_scale_offset;
```

单个 idxK value 的读取：

```cpp
packed = idx_k_value_base[
    token_offset * idx_head_dim / 2
    + d / 2
];

code = (d % 2 == 0)
      ? (packed & 0x0f)
      : (packed >> 4);

scale = idx_k_scale_base[
    token_offset * (idx_head_dim / 16)
    + d / 16
];

idxk_value = decode_e2m1(code) * decode_e4m3(scale);
```

idxK scale 以 token/group 维度连续，适合 indexer kernel 以 token tile 方式加载。top-k 逻辑必须在 native kernel 和 BF16 reference 之间进行数值对拍，尤其关注：

- 分数接近 top-k 边界时的排序；
- page boundary；
- CP rank-local block table；
- padding token；
- 长序列下的累积误差。

## 8. Native attention kernel 接口

native attention operator 至少需要接收：

```text
Q
kv_cache_base:       uint8*
kv_scale_base:       uint8*
block_table
seq_lens

kv_block_stride_bytes
kv_scale_stride_bytes

main_k_offset
main_v_offset
main_k_scale_offset
main_v_scale_offset

page_size
head_dim
kv_head_num
```

native indexer operator 需要额外接收：

```text
idx_k_offset
idx_k_scale_offset
idx_head_dim
```

`kv_scale_base` 在算子 ABI 中应按 `uint8*` 处理。即使上层历史接口把它暴露成 FP32 tensor，也必须在 native path 中使用原始 byte pointer 和 byte stride，不能依赖 FP32 element stride 或错误的 dtype cast。

## 9. PD 分离数据流

### 9.1 Prefill

```text
BF16 K/V、BF16 idxK
        │
        ├── quantize Main K/V
        │       ├── E2M1 packed values → kv_cache_base
        │       └── E4M3 scales        → kv_scale_base
        │
        └── quantize idxK
                ├── E2M1 packed values → kv_scale_base[idx_k_offset]
                └── E4M3 scales        → kv_scale_base[idx_k_scale_offset]
```

### 9.2 PD transfer

PD 使用 opaque whole-block transfer：

```text
发送 kv_cache_base 原始 bytes
发送 kv_scale_base 原始 bytes
```

不做：

- FP4 → BF16；
- E4M3 → FP32 展开；
- idxK 单独重排；
- K/V 或 scale 的二次量化；
- BF16 working page 传输。

当前 `MemoryLayoutStrategy::convertIndexToBuffer()` 对 MSA/opaque block 返回完整 KV block 和完整 scale block，必须继续保持这一语义。

#### 9.2.1 四-block PD transfer 目标态

如果采用四个传输 block，PD 请求应表达为一个 bundle：

```text
CacheBundle(request_id, layer_id, physical_block_id)
├── main_kv_fp4
├── main_kv_scale
├── idx_k_fp4
└── idx_k_scale
```

发送和接收顺序可以固定为上述顺序，但正确性不能依赖裸数组顺序；每个 descriptor 应携带 region type、byte size、stride 和 bundle identity。

Prefill 注册时：

```text
register(bundle_key, main_kv_fp4)
register(bundle_key, main_kv_scale)
register(bundle_key, idx_k_fp4)
register(bundle_key, idx_k_scale)
```

Decode 加载时必须完成 bundle-level admission：四个 region 全部可用后才把该 page 标记为 ready。不能在只收到 Main KV 或只收到 Main scale 时让 native attention 开始计算。

当前代码的 `BlockInfoPair {kv, kv_scale}`、`MemoryLayoutStrategy::convertIndexToAddr()` 和 `createBasicBlockInfo()` 都按两块设计，直接把返回 vector 改成四块会破坏旧路径。因此建议分两步迁移：

1. 先引入 `NVFP4CacheBlocks`/bundle descriptor，并让 native op 可以拿到四个逻辑 pointer；
2. 保持旧 `kv + kv_scale` transfer ABI 的兼容适配，将现有 side region 拆成四个子 view；
3. 等 cache-store/RDMA/RPC 支持 bundle 后，再切换为四个物理 backing block；
4. 在内源仓和外源仓都完成 descriptor、allocator、registration、load/restore 和测试后，才移除两块兼容路径。

四-block 物理传输不是 native kernel 的必要条件，也不是第一阶段的实现目标。第一阶段固定使用两块物理存储和四个逻辑 plane；只有在 native kernel 稳定后，确实需要独立对齐、独立传输或独立 cache policy 时，再切换到四个物理 block。

### 9.3 Decode

```text
收到 kv_cache_base + kv_scale_base
        │
        ├── 原样恢复到本地 paged cache
        ├── native Main attention 直接读取
        └── native idxK/indexer 直接读取
```

PD 的正确性条件是：

- source 和 destination 的 `kv_block_stride_bytes` 一致；
- source 和 destination 的 `kv_scale_stride_bytes` 一致；
- plane offset 一致；
- page size 和 head geometry 一致；
- block table 对应相同的逻辑 page；
- 原始 byte pattern round-trip 一致。

## 10. Prefill、Decode、CP 和 DSpARK 的统一要求

native layout 不应按阶段设计多套格式。

### Prefill

- 写入 paged FP4 cache；
- native prefill kernel 直接读历史 page；
- 不能把 prefill-only BF16 scratch 误当作持久化格式。

### Decode

- native decode kernel 直接读取 paged FP4；
- scale 按 16-dim group 缓存到 register/shared memory；
- 维持 CUDA Graph 下固定 stride、固定 pointer 约束。

### CP

- CP 只改变 block ownership 和 block table；
- 不改变 FP4 page 内部布局；
- page-RR、rank-local compact block table 必须由 native kernel 正确解释；
- CP prefix restore 必须同时恢复 value region 和 scale region。

### DSpARK

- draft attention 和 target verify 使用同一套 FP4 reader；
- DSpARK 的 query block 不应重新创建独立的 FP4 cache layout；
- proposal/commit 使用的 draft cache 也要遵守相同的 value/scale ABI。

## 11. 为什么不能直接复用现有 FMHA/FlashInfer

现有 BF16/FP8 attention kernel 通常假设：

- K/V 是完整 element，而不是两个 nibble 共用一个 byte；
- K/V 的 scale 不需要单独按 group 查找；
- cache pointer 和 scale pointer 的 dtype/stride 已固定；
- idxK 不是复用 Main KV scale region 的 packed sidecar；
- kernel 可以直接使用既有 paged layout。

当前 FP4 cache 不满足这些假设，因此不能只修改 dtype enum 就复用现有 kernel。需要：

- 专用 FP4 reader；
- 专用 Main attention kernel；
- 专用 idxK/indexer kernel；
- 明确的 C++ op binding；
- Triton/CUDA reference 对拍。

## 12. 性能设计要求

### 12.1 Main K/V

- 两个 FP4 value 一次 byte load；
- 16-dim group 共享一个 E4M3 scale；
- scale 解码后放 register 或 shared memory；
- K/V plane 保持连续，避免按 value/scale 交错导致大量小内存访问；
- warp tile 与 head-dim 维度对齐。

### 12.2 idxK

- idxK value 和 idxK scale 分离连续；
- 按 token tile 加载；
- 直接在 dot-product 前解包，不产生完整 BF16 idxK workspace；
- top-k 前尽量在 shared/register 中完成 scale 复用。

### 12.3 对齐

建议：

- 每个 plane 起始地址至少 16B 对齐；
- block stride 至少 128B 对齐，必要时 256B 对齐；
- `head_dim` 和 `idx_head_dim` 必须是 16 的整数倍；
- packed byte plane 的行跨度保持连续；
- scale plane 不使用非连续 view 作为 native kernel 的实际输入。

### 12.4 CUDA Graph

native path 必须在 graph capture/replay 下保持：

- cache pointer 稳定；
- scale pointer 稳定；
- block table tensor 地址稳定；
- stride/offset 不在 replay 期间改变；
- prefill/decode 的 kernel key 分开管理；
- fallback 和 native path 不共用错误的 graph contract。

## 13. 正确性验证

### 13.1 Reader 对拍

对同一组随机 BF16 输入：

```text
BF16 → reference quantize → packed FP4 cache
packed FP4 cache → reference decode
packed FP4 cache → native reader decode
```

验证：

- 每个 nibble 的 code；
- 正零而非负零；
- E4M3 scale byte；
- native/reference decoded value；
- page boundary；
- odd/even dim；
- 最大值、最小值和全零 group。

### 13.2 Attention 对拍

分别对比：

```text
BF16 direct attention
FP4 cache + BF16 working-page attention
FP4 cache + native attention
```

需要固定：

- 相同 Q/K/V 输入；
- 相同 block table；
- 相同 sequence length；
- 相同 CP rank 和 padding；
- 相同 softmax 累积精度。

### 13.3 idxK/top-k 对拍

验证：

- idxK dequant 数值；
- 每个候选 token 的 index score；
- top-k token 集合；
- top-k 排序；
- 边界相同分数的 tie 行为。

### 13.4 PD round-trip

验证：

```text
Prefill device cache
→ export kv_cache_base + kv_scale_base bytes
→ PD transport
→ Decode cache restore
→ native reader
```

要求：

- value region byte-for-byte 一致；
- Main K/V scale byte-for-byte 一致；
- idxK value/scale byte-for-byte 一致；
- native attention 输出与 Prefill 本地 cache 输出一致。

## 14. 历史分阶段实现计划

以下阶段记录设计演进，不代表当前 runtime 可以回退到中间态；当前状态以
第 3 节和第 18 节为准。

### Phase 0：冻结 ABI

- C++ 定义 `NVFP4CacheLayout`；
- 导出所有 plane offset 和 stride；
- 固定 nibble 顺序；
- 固定 E2M1/E4M3 decode 规则；
- 增加 layout size/alignment 单测。

### Phase 1：统一 reader

- CPU/reference reader；
- Triton reader；
- CUDA device reader；
- Main K/V 和 idxK 共用 decode 语义；
- 通过随机、边界、全零和 PD byte round-trip 测试。

### Phase 2：native idxK/indexer

- 直接读取 idxK FP4 和 idxK scale；
- 融合 dequant、dot-product 和 top-k；
- 与 BF16 indexer 对拍；
- 先接 Decode，再扩展 Prefill/CP。

### Phase 3：native Decode attention

- 直接读取 Main K/V FP4 和 scale；
- 用离线 BF16 oracle 对拍，不提供服务 fallback；
- 做单步 decode correctness 和 microbenchmark；
- 验证 CUDA Graph、padding、page boundary 和 CP block table。

### Phase 4：native Prefill/CP attention

- 接入 paged prefill；
- 接入 CP page-RR；
- 验证长上下文、CP4、PD decode restore；
- 对比 BF16 working-page 的带宽和 kernel timeline。

### Phase 5：DSpARK 集成

- draft attention 使用同一 reader；
- proposal/commit cache 统一 layout；
- target verify 统一 native path；
- 做 acceptance、吞吐和 graph 稳定性验证。

### Phase 6：降低 BF16 fallback 优先级

- native path 覆盖稳定 geometry 后默认启用；
- fallback 仅用于不支持 geometry、调试和对拍；
- 不能在没有 native kernel 性能证据前宣称“FP4 attention 已完成”。

## 15. 当前代码需要提前调整的点

1. 将 `kv_scale_base` 的 byte layout 从注释约定提升为正式 ABI。
2. C++ 侧导出所有 plane offset，而不是只导出 `kv_scale_stride_bytes`。
3. Python attention 不再重复推导 offset，统一使用 C++ layout metadata。
4. native op 以 `uint8*` 和 byte stride 读取 scale region。
5. 保证 Main K/V 和 idxK 使用统一的 nibble 顺序和 E2M1 decode。
6. 为 scale plane 增加对齐和 stride 校验。
7. 增加 native reader 与 BF16 working-page 的对拍测试。
8. 增加 PD 原始 byte pattern round-trip 测试。
9. 对 CP rank-local、padding、page boundary 明确测试覆盖。
10. 删除 M3.1 NVFP4 的 BF16 working-page runtime；BF16 仅允许存在于离线
    correctness oracle，不能由服务 backend selector 或异常回退触发。
11. 评估并实现四-block bundle descriptor：`main_kv_fp4`、`main_kv_scale`、`idx_k_fp4`、`idx_k_scale`。
12. 扩展 `BlockInfoPair`、cache allocator、MR registration、PD RPC/cache-store descriptor，使四个 region 具有一致的 bundle identity 和生命周期。
13. 在四-block ABI 完成前，提供两块物理存储到四个逻辑 pointer 的兼容适配，避免 native kernel 依赖 side-region 隐式 offset。
14. 增加“四个 region 缺一个不能 ready”的 PD 原子性测试，以及乱序到达、重复到达、超时释放测试。

## 16. 最终判断

当前 FP4 cache 布局可以演进到 native fused kernel，不需要推倒重做。PD 的数值传输语义也不需要改变；但如果采用四-block 目标态，需要扩展 PD/cache-store 的 descriptor 和 bundle 管理协议。关键工作是把现有的隐式 byte sidecar 约定固化成跨 C++/Python/CUDA/Triton 的 layout ABI，然后实现能够直接读取：

```text
packed E2M1 value + per-16 E4M3 scale
```

的 Main attention 和 idxK/indexer 融合算子。

最终目标应明确为：

```text
4-bit 持久化存储
+ Main KV / Main scale / idxK / idxK scale 明确分区
+ 原始 bytes PD bundle 传输
+ native kernel 直接读取
+ 算子内部完成 dequant 和 attention/indexer
```

而不是：

```text
4-bit 持久化存储
+ 每次先恢复成 BF16 工作页
+ 继续使用旧 attention kernel
```

## 17. Phase-1 native decode 单算子实现结果

当前实现已经接入生产 dispatch：

| 环节 | 实现 | 当前结论 |
|---|---|---|
| Decode cache 写入 | fused Triton K/V/Indexer-K writer | 需要优化；已从三次 launch 合并为一次，支持 compact/MMA scale |
| IndexScore | Triton BF16-Q/K4 MMAv5；CUDA WMMA reference | 需要优化；Q4/K4 因 TopK 精度下降已拒绝 |
| TopK | 现有 MiniMax TopK | 与 FP4 无关，本阶段不重写 |
| Decode metadata | direct Triton CSR/schedule builder | 需要优化；不再扫描全部历史 block |
| Sparse attention | CuTe SM100 BF16-Q/K4/V4 | 需要优化；直接读取 FP4 Main K/V，无 gather |
| Main/Indexer gather | Triton fallback/oracle | 不优化，不进入目标 Decode 热路径 |
| MoE activation pack | fused Triton two-level NVFP4 packer | 需要优化；保留 DeepGEMM MegaMoE ABI |
| MoE GEMM/dispatch | DeepGEMM NVFP4 MegaMoE | 已是原生后端，先 profile 再决定是否修改 |
| Weight transform | DeepGEMM startup transform | 非逐 token 热路径，不列为 Decode 优化目标 |

Native Decode 的目标数据流为：

```text
fused FP4 writer
    -> BF16-Q / FP4-Indexer-K score
    -> TopK16
    -> direct decode metadata
    -> CuTe BF16-Q / FP4-Main-KV sparse attention
```

这条链路不生成 Main K/V 或 Indexer-K BF16 working pages。Triton attention
保留为 correctness oracle 和不支持 CuTe 时的 fallback；SM100 性能主实现是
CuTe kernel。

已完成的正确性和稳定性覆盖包括：

- compact/MMA scale 的 IndexScore 结果一致；
- fused writer 输出与分离 writer bitwise 一致；
- CuTe attention 对反量化 oracle，最大绝对误差门限 `0.02`；
- CUDA Graph 预分配输出和输入更新 replay；
- 长度 `1/127/128/129/79999/80000/1000000`；
- duplicate/invalid TopK、ragged tail 和乱序 physical page；
- 1M 场景实际 value/side byte offset 超过 `2^31`。

GPU 忙碌期间得到的 timing 只能用于选择实现，不能作为正式性能结论。
正式结果由 `run_nvfp4_80k_matrix.py` 在 GPU 4-7 空闲时生成，固定
`BS=4/8/16 per GPU`、`global BS=16/32/64`、`seq_len=80000`，同时报告
四卡 min/median/max、BF16 对照、精度和 source identity。

## 18. 2026-09-24 BF16 runtime 删除与 PD4+4 证据

M3.1 NVFP4 不再提供 `M3_NVFP4_PREFILL_BACKEND` 或
`M3_NVFP4_DECODE_BACKEND`。运行时固定为：

```text
CP4 Prefill: packed writer -> packed FP4 prefix/suffix working set
             -> fmha_sm100 FP4 IndexScore + sparse attention
DP4 Decode:  fused packed writer -> Q8KV4 IndexScore -> production TopK
             -> fused packed-FP4 sparse attention
MTP verify:  target 使用上述 M3.1 Q8KV4 路径；draft 保持 MiniMax-M3
             MXFP8 MoE + FP8 KV，不继承 target NVFP4 配置
```

旧 MiniMax-M3 的 BF16/FP8 通用 MSA scratch 仍保留，这是 M3/MTP 的兼容
合同，不是 M3.1 NVFP4 fallback。M3.1 若绕过 native CP prefill、paged decode
或 target-verify 路由进入该通用路径，会立即报错，防止静默恢复全历史 BF16
materialization。

DP idle fake stream 只分配 `1 + propose_step` 个 block，可能小于
`topk_blocks=16`。生产 `minimax_decode_topk` 已支持该短序列合同：有效 block
之后写 `-1`。Q8KV4 wrapper 删除了错误的 `block_table_width >= topk` 限制，
并新增短表单测；没有扩大 fake cache，也没有为 fake stream 增加 BF16 分支。

验证结果：

- Q8KV4 单测（普通 decode、CUDA Graph replay、短 fake block table）与 packed
  writer 物理平面测试共 3/3 通过；
- MiniMax-M3 MTP 配置/兼容性测试 31/31 通过；
- `FORCE_CPU_LOAD_WEIGHTS=0`、`LOAD_METHOD=fastsafetensors`、Prefill CP4
  KV-sharded、Decode DP4/EP4、CG off 的 PD E2E 跑通；首轮 JIT 后同一算术
  请求连续 5/5 输出 `84`，多个 DP rank 实际完成请求；
- CG on 在 `PyWrappedModel`/graph 初始化期间四个 rank 同时 SIGABRT，尚未到
  HTTP ready。该项是独立未完成 gate，不能用 CG-off 证据替代，也不能通过
  恢复 BF16 working-page 绕过。

补充边界：同日 target-only DP8、每 DP rank BS16、输入 81920、CG on 已完成
128/128 请求并采集 8-rank timeline，证明普通 decode 的 native Q8KV4 writer、
IndexScore、TopK 和 sparse attention 均在 graph 路径执行。它不改变上一条
PD4+4 CG-on 的未完成结论，也不证明 DSpARK graph。

# GLM-5.4 压缩 Indexer 与缓存设计

状态：`feat/glm5_4` 上的首版基础设施实现。由于 GLM-5.4 模型文件和正式的
`config.json` 尚未发布，本文会明确标注所有可能影响数值行为的假设。

## 目标

预计 GLM-5.4 会混合使用 MLA 层和 KDA/线性注意力层。其稀疏 MLA Indexer
预计会把相邻 4 个原始 token 压缩为一个索引 key，选出 512 个压缩分组，再展开为
2048 个原始 MLA token 位置。

本设计**不压缩主 MLA KV cache**。DSV4 CSA 仅作为 learned compressor 和类型化
cache pool 的参考，不复用 DSV4 的压缩主注意力 KV 和 SWA 注意力路径。

```text
hidden states
  -> 压缩比为 4 的 learned Indexer compressor
  -> INDEXER_KV（每 4 个原始 token 存一个条目）

query -> 对压缩 key 计算分数 -> 选出 512 个分组 id
      -> 将分组 g 展开为 [4g, 4g+1, 4g+2, 4g+3]
      -> 通过 MLA block table 映射 2048 个请求内原始位置
      -> 在原始逐 token MLA cache 上执行稀疏 FlashMLA
```

## 暂定配置字段

`AttentionConfigs` 已实现以下字段：

| 字段 | 首版取值 | 含义 |
| --- | ---: | --- |
| `indexer_topk` | 512 | Indexer 选出的压缩分组数 |
| `indexer_compress_ratio` | 4 | 一个 Indexer key 代表的原始 token 数 |
| `sparse_attention_topk` | 2048 | 稀疏 MLA 消费的原始 token 数 |
| `indexer_compressor_overlap` | 1 | compressor 窗口额外包含的前序分组数 |
| `indexer_layer_ids` | 可选 | 拥有 Indexer cache pool 的 MLA 层编号，从 0 开始 |

`DeepSeekV2.from_huggingface()` 当前会识别暂定 JSON 字段
`indexer_compress_ratio`（或 `index_compress_ratio`）、
`indexer_compressor_overlap`、`sparse_attention_topk` 和
`indexer_layer_ids`。模型发布后必须根据正式配置检查这些名称。字段缺失时会保持
GLM-5.2 行为，即 `ratio=1`。

首版要求 `sparse_attention_topk == indexer_topk * indexer_compress_ratio`。
在参考实现明确不完整当前分组的处理方法前，暂不额外加入 local-window 配额。

## Cache 布局

`GLM54CacheConfigHelper` 会在常规 hybrid cache 布局上追加两个类型化区域：

| 区域 | 所有者 | 分配方式 | 首版条目大小 |
| --- | --- | --- | --- |
| `DEFAULT` | MLA 层 | 现有 FULL pool | 现有逐 token MLA KV |
| `DEFAULT` | KDA 层 | 现有 LINEAR pool | 现有 KDA recurrent/conv state |
| `INDEXER_KV` | 带 Indexer 的 MLA 层 | FULL paged pool | FP8：132 B；BF16：256 B |
| `INDEXER_STATE` | 同一批 MLA 层 | fixed/SWA 风格 pool | FP32 projected KV 与 score+APE |

当 kernel block 包含 128 个 token、压缩比为 4 时，`INDEXER_KV` 每个 block 有
32 个条目。block table 仍使用原始 token 坐标，以保持 prefix cache identity 和
MLA block 所有权一致。

初版 state 布局沿用 DSV4 CSA 假设：

```text
projection 分支数 = 1 + overlap = 2
state 宽度        = 2 * 分支数 * indexer_head_dim
                  = 2 * 2 * 128 = 512 个 FP32 值
state ring 条目数 = even_ceil(分支数 * ratio + gen_num_per_cycle)
                  = 8（无推测解码时）
```

`INDEXER_STATE` 当前复用 DSV4 的 fixed-pool 容量参数和 region 语义。如果
GLM-5.4 需要不同的生命周期或 PD 传输规则，应对现有能力重命名并做通用化，而不是
再增加一个私有 cache allocator。

当 `indexer_layer_ids` 缺失时，所有非 LINEAR 的全注意力层都会拥有两个 Indexer
区域。这只是兜底行为。拿到实际 IndexShare 排布后，loader 必须只为具有完整
Indexer 权重的层填入编号；共享层应复用分组 id，不应单独分配 compressor cache。

## Compressor 参考实现

`indexer_compressor.py` 包含一个 CPU/PyTorch 数值参考实现。首版假设其计算方式与
DSV4 CSA compressor 一致：

1. `wkv` 和 `wgate` 分别投影出两个 128 维分支。
2. 对满足 `(p + 1) % 4 == 0` 的原始位置 `p`，收集位置 `p-7..p`。
3. 前 4 个位置使用投影分支 0，当前 4 个位置使用投影分支 1。
4. 将 `ape[position % 4]` 加到 score 上，按通道在 8 个位置上执行 softmax，
   然后归约 projected KV。
5. 执行 RMSNorm，存储一个 128 维 Indexer key。

实现这一假设是为了在拿到 checkpoint/参考实现后有明确的对齐基线，并不表示这就是
GLM-5.4 最终公式。

当 `indexer_compress_ratio != 1` 时，优化模型路径目前会主动抛出
`NotImplementedError`。Cache 分配和稀疏 MLA 分组展开已经接通，但在 compressor
权重绑定和 pool 元数据确认前，模型必须明确失败，不能静默复用旧的逐 token
`wk + k_norm` 路径。

## 压缩分组与原始 token 坐标

Indexer 输出保持为请求内压缩分组坐标，形状为 `[..., 512]`。在稀疏 MLA cache
寻址边界处执行展开：

```text
分组 id [..., 512]
  -> 按 score 顺序、组内 lane 顺序展开为原始 id [..., 2048]
  -> 通过 DEFAULT MLA block table 转换
  -> 物理 MLA slot
```

负分组 id 会展开为 4 个 `-1`。参考 helper 还可以根据传入的原始序列长度屏蔽越界
lane。CP 路径会在自身 block-table 转换前执行同样的展开。

当前实现保持“分组 score 顺序优先、组内 lane 顺序次之”。如果改为对所有原始位置
做全局排序，可能改变 FlashMLA 的归约顺序，因此必须以参考数值结果为依据。

## 等待模型文件确认的事项

解除运行时保护前，必须确认以下细节：

1. 正式模型 architecture 名称，以及 MLA/KDA 层排布字段。
2. Compressor checkpoint 名称、tensor layout、dtype，以及公式是否确实与 DSV4
   CSA 一致。
3. 压缩分组是否为不重叠的 `[4g, ..., 4g+3]`。
4. 是否只有完整分组可见；当前参考实现不会为末尾 1～3 个不完整 token 生成 key。
5. 除 top-512 分组外，是否强制加入 local/current group。
6. 准确的 IndexShare/完整 Indexer 层排布。
7. MTP reject/rollback 时 compressor state 的处理语义。
8. CP 所有权以及 state ring 的 PD 传输要求。

## 测试

当前单测覆盖：

- 旧的逐 token top-2048 和新的压缩 512×4 几何关系；
- 确定性的分组展开、无效项处理和不完整尾部屏蔽；
- compressor 参考实现的分支/窗口行为和完整分组策略；
- FP8 Indexer 条目大小、每 block 压缩条目数和 state-ring 容量；
- MLA/KDA cache region 所有权、显式 Indexer 层排布和 MTP state 冗余空间。

模型发布后，应补充逐层对齐测试：对比官方实现生成的压缩 key、选中分组 id、展开后
原始 id 和稀疏注意力输出，并覆盖 prefill、单 token decode、MTP verify、prefix
复用以及序列长度对 4 取模的各种情况。

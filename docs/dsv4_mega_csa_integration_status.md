# DSV4 Mega CSA TP1 接入状态

更新日期：2026-08-12

## 1. 当前结论

RTP 开源分支 `dsv4-mega` 已重新对齐 `upstream/feat/dsv4_on_dev@bbf66a5f8`。
当前 `rtp_llm` 代码相对上游为零差异，生产 decode 没有 Mega 开关、backend 或选路逻辑。

此前从 `8d75c0666` 开始的五个实验提交已从分支历史撤回，并保存在本地备份分支：

```text
backup/dsv4-mega-pre-minimal-20260812 @ 18cba4448
```

备份只用于查阅旧实验，不应继续作为生产接入基础。旧实现混合了必要性未证实的 cache ABI、
Mega 专用 metadata、旧 FlashMLA fused-query contract 和未进入生产路由的旁路 backend。

## 2. 为什么最终没有保留框架代码

### 2.1 Cache geometry 不需要新增 ABI

算子需要两个量：

```text
entries_per_block
block_stride_bytes
```

当前框架已经把每层 typed pool 暴露为二维 tensor：

```text
[kernel_block_count, physical_stride_elements]
```

因此物理 stride 可以直接获得：

```python
block_stride_bytes = pool.stride(0) * pool.element_size()
```

现有 DSV4 约定从物理 stride 和已知 entry bytes 得到 entry 数：

```python
entries_per_block = block_stride_bytes // bytes_per_entry
```

FP8 MLA pool 的 entry 为 584B，TMA 对齐为 576B；补齐量始终小于一个 entry，因此当前
支持的 geometry 下整数除法不会产生额外 phantom entry。例如：

| Pool | 物理 stride | entry bytes | 推导结果 |
| --- | ---: | ---: | ---: |
| CSA KV | 19008B | 584B | 32 |
| HCA KV | 1152B | 584B | 1 |
| Indexer KV | 4224B | 132B | 32 |
| SWA KV | 74880B | 584B | 128 |

FULL pool 在 C++ memory layout 中已经展开为 kernel-block 粒度；prefill CP 的 SWA byte-slice
也已有专用逻辑恢复全 ring entry 数。因此新增以下字段没有提供当前代码无法推导的信息：

```text
logical_entries_per_block()
group_entries_per_block
LayerKVCache.kv_entries_per_block
LayerKVCache.kv_block_stride_bytes
```

这组 ABI 改动已撤销。只有未来出现 `padding >= bytes_per_entry`、非二维 pool view，或者
entry payload 与物理 record 不再一一对应时，才需要重新引入显式 descriptor。

### 2.2 现有 metadata 已有正确性所需字段

上游 `DSv4DecodeAttnMetadataFP8` 已经提供：

```text
pool_block_tables
pool_write_slot_mappings
compressor_state_slot_mappings
position_ids / position_ids_long
compressed_lens
topk_buffer_compressed
swa_global_slots
```

旧提交新增的 `mega_*_i32` mirror 只是适配当时 CUDA kernel 的 int32 ABI，并避免每层 cast；
它不是新的框架语义。最终 kernel ABI 尚未稳定，现在保留会提前固化重复 buffer 和通用
metadata 结构，因此已撤销。

MQA schedule epoch/cache 同样应先由 Mega adapter 私有管理。只有确认多个 layer 必须共享，且
CUDA graph capture/replay 生命周期无法由 adapter 管理时，才提升到通用 metadata。

### 2.3 旧 FlashMLA contract 已失效

旧提交给通用 FlashMLA wrapper 增加了：

```text
q_rms_sum_sq
q_rope_cos
q_rope_sin
q_rms_eps
```

最新 Wuda 改为 FlashMLA 前完成 query RMSNorm/RoPE route，再调用原生 FlashMLA wheel。
旧 fused-query 参数不再是目标 ABI，相关框架修改和运行时 `TypeError` 探测已全部撤销。

### 2.4 旧 Mega backend 不是生产框架实现

旧 `mega_csa_backend.py` 从未被 `Block.forward_decode` 选择，并明确标记
`full_pipeline_ready = False`。它还依赖旧 FlashMLA contract、Mega int32 mirror 和旧
CUDA Extension 快照。继续维护只会制造“已经接入”的错觉，因此不保留在当前分支。

## 3. 当前真实链路

每层 decode 仍执行：

```text
attn_hc.pre
-> attention RMSNorm
-> AttentionFP8.forward_decode
-> output projection
-> attn_hc.post
-> FFN sublayer
```

入口位于：

```text
rtp_llm/models_py/modules/dsv4/block.py::Block.forward_decode
```

Mega 最终应接管整个 attention sublayer，而不是只替换
`AttentionFP8.forward_decode`：

```text
attention MHC pre
-> front / WQ_B / Indexer / MQA / TopK
-> native FlashMLA
-> output projection
-> attention MHC post
```

FFN MHC 和层循环后的 model head MHC 不属于替换范围。

## 4. 算子仓库状态

| 仓库 | 当前基线 | 说明 |
| --- | --- | --- |
| Wuda | `main@d9e53a0` | 最新优化源；包含 query route、PDL overlap 和正在演进的 TP1/TP2 逻辑 |
| CUDA Extension | `dsv4_megakernel@b2f07a8` | 较早 RTP 交付快照；包含 FP8 indexer producer 和 FP8 MQA/main compressor |
| RTP 开源 | `dsv4-mega@bbf66a5f8` | 与上游完全一致，仅增加本文档提交后产生一个 docs commit |
| RTP 内源 | `develop/wangyin_ds_v4_20260424` | 最终承接开源版本和内源制品依赖 |

Wuda 与 CUDA Extension 的重构和迁移方案见
`docs/dsv4_mega_cuda_extension_refactor.md`。

## 5. 后续最小框架接入面

必须先固定 CUDA Extension 和原生 FlashMLA 的最终 TP1 ABI，再修改 RTP。届时框架侧只允许
增加以下三类改动。

### 5.1 配置和生命周期

- 默认关闭、模型构造期固定的 `enable_mega_csa`；
- 只支持 FP8 KV cache、TP1、`q_len == 1` 和已验证 CSA geometry；
- 只为满足条件的 layer 构造 adapter、权重布局和 graph-stable workspace；
- capability check、JIT 和 workspace 分配必须在 cache write/CUDA graph capture 前完成。

### 5.2 薄 adapter

adapter 只负责：

- 将现有 framework pool tensor、block table、slot mapping 和 position metadata 传给算子；
- 从 tensor stride 取得 runtime page geometry；
- 管理 Mega 私有 schedule/workspace；
- 调用原生 FlashMLA wheel；
- cache write 前完成全部 ABI 校验，写入后失败时禁止回退普通 attention。

不要把算子 pipeline 重新用 Python 拼成 1000 行 backend，也不要把尚未共享的 schedule 和
dtype mirror 提前放入通用 metadata。

### 5.3 Attention sublayer switch

`Block.forward_decode` 在 attention sublayer 入口选择：

```python
if mega_adapter is not None and q_len == 1:
    x = mega_adapter.forward_attention_sublayer(x, metadata, kv_cache)
else:
    x = existing_attention_sublayer(x, metadata, kv_cache)
```

两条路径在 FFN 前汇合。普通 `AttentionFP8.forward_decode` 必须继续服务开关关闭、SWA-only、
HCA、target verify、prefill 和不支持的 geometry。

## 6. 重新增加框架字段的门槛

未来任何框架字段都要同时满足：

1. 最终算子 ABI 已固定；
2. 现有 pool/metadata 无法正确或无额外 launch 地提供该信息；
3. 字段有明确 eager、capture、replay 生命周期；
4. 开关关闭时不分配、不更新、不增加 kernel；
5. 有独立正确性测试和普通路径性能 A/B。

不满足这些条件的内容留在 adapter 或算子仓库，不修改通用 RTP 框架。

## 7. 验证顺序

1. Wuda TP1 `--fullrank`，不启用 TP2；
2. CUDA Extension 单算子正确性和性能 A/B；
3. 原生 FlashMLA adapter contract test；
4. RTP eager TP1、batch=1、`q_len=1`；
5. normal prefill -> Mega decode -> normal target verify -> Mega decode；
6. slot reuse、非平凡 block id、compression boundary；
7. CUDA graph capture/replay；
8. 普通 FP8 与 Mega FP8 的整段 attention、整模型性能 A/B。

当前阶段不应提前修改 `Block.forward_decode` 或通用 metadata；先完成 Wuda -> CUDA
Extension 的稳定 TP1 迁移。

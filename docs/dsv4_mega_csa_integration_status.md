# DSV4 Mega CSA TP1 接入状态与后续方案

更新日期：2026-08-15

## 1. 当前结论

DSV4 Mega CSA 的开源框架适配已经进入生产 decode 层循环。使用本地
`cuda_extension@e1d1c985` 本地构建，TP1 单层真实 RTP attention sublayer 的数值对照、eager、
CUDA Graph 和 slot reuse 已通过；但还不能称为 RTP-LLM 整模型端到端已跑通。框架当前锁定的
CUDA13 `rtp-kernel` wheel 仍不含 Mega 扩展，必须先发布对应制品并完成真实 allocator、
prefill/decode 切换和整模型验证。

当前实现遵循“完整 attention sublayer 单独选路”，没有逐个替换普通算子：

```text
Block.forward_decode
  ├─ CSA Mega: adapter 已挂载 && q_len == 1
  │    mHC pre + attention RMSNorm
  │    -> front mixed GEMM
  │    -> WQ-B + indexer compressor + SWA write
  │    -> FP8 MQA + main compressor + query RMS/RoPE
  │    -> RTP persistent TopK
  │    -> RTP 原生 FlashMLA 路径
  │    -> CUDA inverse-RoPE + FP8 quant
  │    -> 现有 wo_a / wo_b output projection
  │    -> mHC post
  └─ 原路径: 其他所有情况
       attn_hc.pre -> AttentionFP8.forward_decode
       -> output projection -> attn_hc.post
```

FFN sublayer、FFN mHC、model head mHC 不在 Mega 替换范围内。

## 2. 支持边界

| 项目 | 当前支持 | 处理方式 |
| --- | --- | --- |
| 硬件 | Blackwell `sm_100a/sm_103a` | 首次执行前强校验 |
| 并行 | TP1、单卡 | `tp_size != 1` 初始化失败 |
| KV cache | FP8 | 非 FP8 初始化失败 |
| 层类型 | `compress_ratio == 4` 的 CSA 层 | 只给这些层挂 adapter |
| 请求形态 | decode、`q_len == 1`、batch 1..128 | 其余形态走现有路径 |
| 进程角色 | `DECODE` 和单卡 `PDFUSION` | 由 `forward_decode` 限制实际执行 |
| 开关 | `DSV4_MEGA_CSA=1` | 默认关闭，模型构造期固定 |

下列场景保持现有实现：prefill、SWA-only、HCA、target verify (`q_len > 1`)、MTP、TP2/DP2。
MTP 是独立模型且当前 `compress_ratio == 0`，不会挂载 CSA adapter。

`is_decode_role=False` 同时覆盖 `PDFUSION` 和专用 PREFILL，框架目前没有更细的构造参数。
因此 `DSV4_MEGA_CSA=1` 只应配置在 `DECODE/PDFUSION` 进程；误配到专用 PREFILL
不会执行 Mega decode，但会产生不必要的 fused-weight 重排和显存占用。

## 3. 已完成的框架适配

### 3.1 文件与职责

| 文件 | 修改 |
| --- | --- |
| `dsv4/transformer.py` | 解析开关；校验 FP8 KV/TP1；创建模型级 runtime；给 CSA 层挂 adapter |
| `dsv4/decode/forward.py` | 在生产 layer loop 前推进一次 Mega decode step |
| `dsv4/block.py` | 在 attention sublayer 入口选择完整 Mega 路径；FFN 前重新汇合 |
| `fp8/decode/mega_csa_weights.py` | 校验 checkpoint tensor 并构造算子要求的 TP1 fused layout |
| `fp8/decode/mega_csa_runtime.py` | 共享 workspace、logits、MQA schedule 和 RoPE table；校验并透传框架 slot tensor |
| `fp8/decode/mega_csa_adapter.py` | 绑定现有 cache/metadata，编排 Mega 算子、TopK、原生 FlashMLA 和 o-proj |
| `fp8/test/test_mega_csa_adapter.py` | 覆盖选路、PDFUSION、权重布局、ABI 和 runtime 生命周期 |
| `fp8/test/test_mega_csa_rtp_eager.py` | 用真实 `AttentionFP8`/`KVCache` 对照原 attention 子层，并覆盖 eager、graph、cache/state 和性能 |

### 3.2 权重

`MegaCSAWeights` 在模型初始化时从原 checkpoint tensor 构造以下连续布局：

```text
front_fp8 = [wq_a; wkv]
front_sf = [wq_a_scale; wkv_scale]
front_bf16 = [main_wkv; main_wgate; index_wkv; index_wgate; index_weight_proj]
wq_b_fp8 = [index_wq_b; main_wq_b]
wq_b_sf = [index_wq_b_scale; main_wq_b_scale]
```

FP8 权重和 UE8M0 scale 不做数值反量化/再量化。Indexer score 的两个归一化因子在初始化时
折入 `index_weight_proj`，与现有 `IndexerFP8` 语义一致。

当前每个 CSA 层约增加 158 MiB 连续权重副本，21 个 CSA 层约 3.3 GiB。它不影响单步
kernel 时间，但影响模型初始化和常驻显存；在保留普通 target-verify 路径时不能直接释放原权重。
后续可评估 loader 直接产出 fused layout，或者调整 kernel 接受分段权重，避免重复存储。

### 3.3 模型级 runtime

所有 CSA 层顺序复用同一批按 `(device, batch, split)` 缓存的 workspace，不按层重复分配。
runtime 还负责：

- 每个模型 decode step 只生成一次 MQA schedule；
- 在 WQ-B 提交前准备 schedule，保持 WQ-B 到 MQA 的 PDL 顺序；
- 校验框架五组 slot mapping 为连续 CUDA int64 tensor，并将原 tensor 直接传给算子；
- 保留 capture 期间生成的 schedule tensor，避免 graph 中悬空指针；
- 缓存从 `freqs_cis` 拆出的连续 cos/sin table。

`cuda_extension@e1d1c985` 已把 FP8 CSA 的五组 slot ABI 改为 int64，与 RTP metadata 对齐。
runtime 不再分配 int32 mirror，不执行 `copy_`，也不按 eager/graph metadata 缓存 slot 副本；
CUDA Graph 捕获期间直接使用框架 tensor 的稳定地址。position、block table、context length 和
schedule metadata 等其他 ABI 均未扩大为 int64。

### 3.4 Cache 与 FlashMLA

没有增加通用 cache ABI。adapter 直接使用现有：

```text
pool_block_tables
pool_write_slot_mappings
compressor_state_slot_mappings
compressed_lens
topk_buffer_compressed
position_ids / position_ids_long
swa_global_slots
```

`entries_per_block` 和 `block_stride_bytes` 继续从 typed pool 的现有 view/stride 推导。

FlashMLA wrapper 没有修改，也不依赖 Wuda 的改造版 FlashMLA。adapter 在写 cache 前检查
现有 FlashMLA metadata 和 backend，再通过 `AttentionFP8._forward_decode_compressed` 调用
RTP 当前原生 FlashMLA wheel。进入 Mega 且发生 cache write 后，任何错误直接上抛，禁止回退
普通 attention，避免同一步重复写 cache。

Wuda `origin/main@6818258` 新增的 MLA output inverse-RoPE + FP8 quant CUDA producer 已迁入
`rtp-kernel`。Mega runtime 为它提供模型级复用的 graph-stable FP8/scale workspace；adapter 直接
传框架 int64 position 和已有的 FP32 cos/sin table，不再先执行 `freqs_cis.index_select`。producer
输出继续交给 RTP 现有 `_wo_a_einsum_from_fp8` 和 `wo_b`。普通 attention 路径仍使用原 Triton
producer，没有修改通用 output-projection 选路。

### 3.5 普通路径影响

开关关闭时不构造 fused weights、runtime 或 workspace，也不新增 CUDA kernel。
普通路径保留原有 tensor 和 cache ABI。代码层只增加一次 model-step runtime presence check，以及
每层一次 `adapter is not None` 的 Python 分支；是否可测必须由 normal FP8 A/B 给出，不能只凭
静态分析宣称零下降。

## 4. 算子与制品状态

CUDA Extension 已完成 Wuda 最新 TP1（不含 TPDP）迁移并推送：

```text
repo:   /root/work/cuda_extension
branch: origin/dsv4_megakernel
base:   origin/main@3bc0ca4
source: Wuda origin/main@6818258
commit: e1d1c985 feat(dsv4): fuse MLA output inverse RoPE quant
```

当前已生成但尚未发布的本地 CUDA13.2 wheel：

```text
/root/work/cuda_extension/dist/
  rtp_kernel-0.1.0+e1d1c985.cu132-cp310-cp310-linux_x86_64.whl
sha256: 175b34b2e08bb6234dd50b35d83c48abd1f54fd363863203b24ec7088add5bd1
```

wheel 已确认包含：

```text
rtp_kernel/dsv4_mega.py
rtp_ops_dsv4_mega.cpython-310-x86_64-linux-gnu.so
```

RTP 当前 CUDA13 lock 仍解析到：

```text
rtp-kernel 0.1.0+cu13.4a1a7e3
```

该 RTP lock 中的旧 wheel 没有 `dsv4_mega`。当前验证使用 `e1d1c985` 本地 wheel，通过未跟踪的
本地 CUDA13 lock 让 Bazel 正常解析，没有替换 Bazel external cache。必须发布 wheel 后再更新
公共 requirements 和 lock，不能把本地绝对路径或临时 URL 写进提交。

adapter 另外依赖当前 DeepGEMM 的：

```text
tf32_hc_prenorm_gemm
get_paged_mqa_logits_metadata
get_num_sms
```

首次真实执行会同时检查 GPU capability、`rtp_kernel.dsv4_mega` 函数签名和固定 geometry，
避免“有同名旧符号但 ABI 不兼容”时进入 cache write。

## 5. 已完成验证

以下 CPU/静态回归已通过：

```text
//rtp_llm/models_py/modules/dsv4/fp8/test:test_mega_csa_adapter
//rtp_llm/models_py/modules/dsv4/fp8/test:test_attention_csa_overlap
//rtp_llm/models_py/modules/dsv4/fp8/test:test_decode_topk_length
//rtp_llm/models_py/modules/dsv4/decode/test:decode_fmha_impl_test
```

以下完整编译已通过：

```bash
bazelisk build //rtp_llm:rtp_llm \
  --verbose_failures \
  --config=cuda13 \
  --test_output=errors \
  --test_env="LOG_LEVEL=INFO" \
  --jobs=64
```

使用本地 `cuda_extension@e1d1c985` 和 Bazel CUDA13 依赖路径执行 adapter ABI 检查已通过，
geometry 为 main `65536`、index `8192`、merged `73728`、main heads `128`、index heads `64`，
slot ABI 为 int64。

新增 SM100 单卡测试：

```text
//rtp_llm/models_py/modules/dsv4/fp8/test:test_mega_csa_rtp_eager
```

该测试显式固定 Wuda `origin/config.json` 中的 DSV4 Pro attention geometry：

```text
dim=7168, n_heads=128, q_lora_rank=1536
head_dim=512, rope_head_dim=64
o_groups=16, o_lora_rank=1024
window_size=128, compress_ratio=4 (CSA)
index_n_heads=64, index_head_dim=128, index_topk=1024
original_seq_len=65536, max_seq_len=65536
rope_theta=10000, rope_factor=16, beta_fast=32, beta_slow=1
compress_rope_theta=160000, hc_mult=4, hc_sinkhorn_iters=20
FP8 indexer, TP1, RTP persistent TopK, RTP wo_a/wo_b, official FlashMLA
```

Mega 和 reference 都调用 RTP 现有 persistent TopK；没有迁移或选择 Wuda TopK。2026-08-15
纠正：提交 `792fd721d` 中的合成测试误用了 `index_topk=512`、`o_groups=8`，该提交记录的
性能数字不是 Pro 配置，已全部作废并由下表替换。

测试使用一个真实 `AttentionFP8` 层、确定性合成权重和两套相同初态的 RTP pybind `KVCache`。
reference 严格执行 `Block.forward_decode` 的原 attention 分支：

```text
attn_hc.pre -> attn_norm -> AttentionFP8.forward_decode -> attn_hc.post
```

Mega 与 reference 分别写独立 cache，连续执行 position `0..3` 到首个 CSA compression boundary。
结果如下：

| 对照项 | `calc_diff` / 结果 | 门限 |
| --- | ---: | ---: |
| 最终 attention sublayer 输出 | `1.135427e-05` | `< 1e-3` |
| CSA KV（解量化） | `1.261505e-05` | `< 1e-3` |
| Indexer KV（解量化） | `3.661147e-04` | `< 1e-3` |
| SWA KV（解量化） | `4.985002e-07` | `< 1e-3` |
| CSA state | `5.116385e-11` | `< 1e-4` |
| Indexer state | `5.772682e-11` | `< 1e-4` |
| TopK | int32 全量一致 | 精确一致 |
| CUDA Graph replay | bitwise 一致 | 精确一致 |

另在 position `4095` 预填充 1024 个随机有效 FP8 packed CSA/Indexer cache entry，从 1024 个
候选中选择 Top-1024：Mega/reference 有效 TopK overlap 为 `1024/1024`，最终输出
`calc_diff=3.094866e-09`。

reference 使用 RTP 默认 TileLang mHC。切换到 int64 slot ABI 后重新测量 B128/64K；首次 JIT、
metadata 构造和每个 model step 只调用一次的 `runtime.begin_decode` 不计入单层时间：

| Batch | Context | 口径 | 原路径 | Mega | 变化 |
| ---: | ---: | --- | ---: | ---: | ---: |
| 128 | 65536 | 生产 CUDA Event | `384.72 us` | `289.91 us` | `-24.6%` |
| 128 | 65536 | 预绑定纯算子链 | `357.17 us` | `261.22 us` | `-26.9%` |

Mega 生产 graph 的 profiler kernel envelope 为 `290.47 us`。与 `261.22 us` 的纯算子链相比，
仍相差约 `29.3 us`，主要是动态 FlashMLA metadata planner；新的 O-proj producer 直接消费
position/cos/sin，不再需要此前的 `freqs_cis.index_select` 输入复制。五个 slot conversion kernel
仍未出现在 timeline。因此约 `290 us` 是 RTP 当前生产 graph，约 `261 us` 是与 Wuda `graph`
列对应的预绑定算子链；二者差额不能归因于 RTP TopK。

此前 B1/8/16 和 B128 的生产数据使用旧 int32 mirror ABI，不再作为当前性能结论；完整 batch
grid 需要在新 wheel/最终依赖环境下重测。

B128/64K 使用两套相同的随机有效 FP8 packed CSA/Indexer/SWA cache。最终 attention sublayer
输出 `calc_diff=2.684947e-07`、最大绝对误差 `2.954102e-02`、cosine `0.999789596`。每个请求
从 16384 个 compressed 候选中选择 Top-1024，有效
overlap min/mean/max 为 `1000/1023.7/1024`。测试门限为每个请求至少 97% 有效 overlap；差异
集中在 TopK 截断边界，最终输出仍满足数值门限。ctx=2048 时只有 512 个有效 compressed 候选，
因此固定宽度 1024 的 TopK buffer 表现为 512 个有效索引和 512 个 padding；这三个 case 的有效
overlap 均为 `512/512`。

B128/64K 同步校验本步写入内容，而不只校验最终输出：CSA KV、Indexer KV、SWA KV 的
`calc_diff` 分别为 `2.986297e-05`、`3.371339e-04`、`2.818445e-07`，CSA state 和 Indexer state
分别为 `6.308681e-10`、`7.094991e-10`，均通过各自数值门限。

性能模式通过 `--test_env=DSV4_MEGA_RUN_PERF=1` 显式开启，并对每个 batch 设置不高于原路径
`1.05x` 的 eager、生产 CUDA Graph 和预绑定算子链回归门。它覆盖真实 RTP 单层算子链，但
typed pool/block table 仍由测试按生产 geometry 构造，不是 `KVCacheManager` 分配；也未使用
真实 checkpoint，不能替代整模型端到端验证。

## 6. 端到端剩余缺口

按阻塞顺序还需要：

1. 发布 `e1d1c985` 对应的 CUDA13 x86_64 wheel，更新开源/内源实际使用的依赖入口和 lock；
2. 增加由真实 `KVCacheManager` 创建 typed pools/block tables 的集成测试，替代手工 pool fixture；
3. 校验 normal prefill -> Mega decode -> normal target verify -> Mega decode；
4. 跑完整 TP1 模型正确性；
5. 测量开关关闭时普通 FP8 整模型路径，确认新增 Python 分支不可测；
6. 对 normal FP8 与 Mega FP8 做真实模型、代表性长上下文和完整 batch grid 性能 A/B。

性能报告至少应单列：

- 框架 int64 slot 直传，并确认 timeline 中没有隐式 conversion/copy；
- mHC pre 到 front、WQ-B 到 MQA 的 PDL 收益；
- MQA schedule 生成；
- TopK + 原生 FlashMLA；
- 完整 attention sublayer；
- 开关关闭的普通 FP8 路径；
- eager 与 CUDA Graph；
- batch 1/8/16/32/64/128 和代表性 context length。

## 7. 内源合入方案

目标内源分支为 `develop/wangyin_ds_v4_20260424`。在开源提交稳定后：

1. 将目标内源 worktree 对齐远端分支，保留现有用户修改和 gitlink；
2. 迁移本分支的 adapter、runtime、weights、选路及测试文档改动，不迁移 Wuda TPDP 或改造版
   FlashMLA 逻辑；
3. 新 wheel 发布后，同时更新内源 CUDA13 requirements lock 和实际 Bazel 依赖选择；
4. 先跑与开源相同的 CPU tests 和 `//rtp_llm:rtp_llm` 完整编译；
5. 再在内源服务配置中只对 TP1 FP8 `DECODE/PDFUSION` 打开 `DSV4_MEGA_CSA=1`；
6. 完成第 6 节 GPU 矩阵后，才能把开关从实验配置提升为默认配置。

不需要修改 HCA、SWA、prefill、TP2/DP2 或 FlashMLA 通用接口。若后续接入这些场景，应分别
新增受支持的完整 sublayer adapter，不能放宽当前 CSA TP1 adapter 的 geometry 检查。

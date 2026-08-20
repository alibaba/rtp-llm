# DSV4 Mega CSA/HCA TP1 接入状态与后续方案

更新日期：2026-08-20

## 1. 当前结论

DSV4 Mega CSA 的开源框架适配已经进入生产 decode 层循环。TP1 单层真实 RTP attention
sublayer 的数值对照、eager、CUDA Graph 和 slot reuse 已通过。2026-08-18 起本地整模型
serving 端到端已跑通（裁层 DSV4-Pro）；2026-08-19 起 **全量 DeepSeek-V4-Flash（43 层，
单卡）端到端跑通**，baseline 与 Mega（CSA+HCA 双开）输出语义等价，仅在近平局 token 处
出现 greedy 分岔（与框架 smoke 对不同拓扑使用各自 golden 的既有现象同类）。框架公共
CUDA13 `rtp-kernel` wheel 仍不含 Mega 扩展；当前所有验证使用本地 wheel + 未提交的本地
lock 补丁，发布制品后才能推平公共依赖。

2026-08-19 起，同一 extension 二进制内同时编译 **DSV4-Pro 与 DSV4-Flash 两套 CSA 几何**
（Pro: dim 7168 / q_lora 1536 / 128 heads / o_groups 16；Flash: 4096 / 1024 / 64 / 8），
python wrapper 按张量形状 dispatch；HCA 算子自始就带双几何。RTP 侧 weights/adapter/runtime
以 `CSAGeometry` profile 按 `attn.dim` 选择，两种模型共用全部代码路径。

2026-08-18 起，HCA（`compress_ratio == 128`）层的同型接入也已完成并通过单层对照，由独立
开关 `DSV4_MEGA_HCA` 控制。HCA 没有 indexer/TopK/MQA 阶段：opA/opB 两个融合 GEMM 覆盖
front 投影、state 环 FRONT-EMIT、mHC post/comb tail、q_b 投影、128-token 边界压缩写
HCA_KV 和 window 写 SWA_KV；query RMSNorm+RoPE 保持框架 `fused_rmsnorm_rope`，稠密
compressed index 直接复用每 step 已构建的 `topk_total_by_ratio[128]`。CSA+HCA 同开时
Mega 覆盖 DSV4-Pro 全部 61 个 attention 层（30 CSA + 31 HCA）。

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
  ├─ HCA Mega: adapter 已挂载 && q_len == 1
  │    mHC pre + attention RMSNorm
  │    -> front mixed GEMM（FRONT-EMIT 写 kv|gate state 环 + mHC post/comb tail）
  │    -> WQ-B + 边界 compressor 写 HCA_KV + window 写 SWA_KV
  │    -> 框架 fused_rmsnorm_rope（q RMSNorm + 部分 RoPE，原地）
  │    -> metadata 稠密 compressed index（topk_total_by_ratio[128]）
  │    -> RTP 原生 FlashMLA 路径（SWA_KV + HCA_KV）
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
| 层类型 | `compress_ratio == 4` 的 CSA 层（`DSV4_MEGA_CSA`）；`compress_ratio == 128` 的 HCA 层（`DSV4_MEGA_HCA`） | 按 ratio 分别挂 adapter |
| 模型几何 | DSV4-Pro（dim 7168）与 DSV4-Flash（dim 4096） | `GEOMETRY_BY_DIM` 按 `attn.dim` 选 profile，其他 dim 初始化失败 |
| 请求形态 | decode、`q_len == 1`、batch 1..128 | 其余形态走现有路径 |
| 进程角色 | `DECODE` 和单卡 `PDFUSION` | 由 `forward_decode` 限制实际执行 |
| 开关 | `DSV4_MEGA_CSA=1` / `DSV4_MEGA_HCA=1` | 各自默认关闭，模型构造期固定，共享一个模型级 runtime |

下列场景保持现有实现：prefill、SWA-only、target verify (`q_len > 1`)、MTP、TP2/DP2。
MTP 是独立模型且当前 `compress_ratio == 0`，不会挂载 CSA adapter。

`is_decode_role=False` 同时覆盖 `PDFUSION` 和专用 PREFILL，框架目前没有更细的构造参数。
因此两个 Mega 开关都只应配置在 `DECODE/PDFUSION` 进程；误配到专用 PREFILL
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
| `fp8/decode/mega_hca_weights.py` | HCA 层 fused 布局：`front_fp8=[wq_a;wkv]`、`front_bf16=[comp_wkv;comp_wgate]`、`wq_b`，约 130 MiB/层 × 31 层 ≈ 4 GiB |
| `fp8/decode/mega_hca_adapter.py` | HCA 编排：front/WQ-B 两个融合 GEMM、框架 `fused_rmsnorm_rope`、稠密 idx、原生 FlashMLA、共享 o-proj producer |
| `fp8/decode/mega_csa_runtime.py`（扩展） | 新增 HCA workspace 缓存与 HCA 三组 slot（HCA_STATE/HCA_KV/SWA_KV）int64 直传校验，`begin_decode`/rope 表与 CSA 共享 |
| `dsv4/block.py`（扩展） | `enable_mega_hca`（仅 ratio==128），`forward_decode` 统一 `_mega_csa_adapter or _mega_hca_adapter` 选路 |
| `fp8/test/test_mega_hca_adapter.py` | HCA 选路、双开关、权重布局、geometry/ABI、runtime slot 生命周期 |
| `fp8/test/test_mega_hca_rtp_eager.py` | 真实 `AttentionFP8(ratio=128)` 对照原 `_forward_decode_hca`：输出、边界压缩 HCA_KV、SWA、state 环、长上下文、graph、性能 |
| `mega_csa_weights.py`（Flash 化） | `CSAGeometry` profile（PRO/FLASH），打包形状全部由 profile 派生；模块级 Pro 常量保留为别名 |
| `mega_csa_adapter.py` / `mega_hca_adapter.py`（Flash 化） | `_validate_geometry` 按 `attn.dim` 选 profile 并 fail-fast；ABI 探针改为按本层几何的子集校验（extension 广告双形状） |
| `mega_csa_runtime.py`（Flash 化） | CSA/HCA workspace 尺寸按 profile 分配并以 dim 入 key；`num_hc_splits` 接受 hidden 宽度 |

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

`cuda_extension@e1d1c985` 已把 FP8 CSA 的五组 slot ABI 改为 int64，与 RTP metadata 对齐；
`cuda_extension@b93e0761` 把 HCA 的三组 slot（state/window/compressed destinations）同样升级
为 int64，并新增 `geometry_hca()` 供 fail-fast ABI 探针（HCA front 的 PDL 按算子契约保持关闭）。
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
branch: dsv4_megakernel
base:   origin/main@3bc0ca4
source: Wuda origin/main@6818258 + origin/flash@ce0b82b（Flash CSA 几何）
commit: 9f4c3fe fix(dsv4): accept 64-head query rows in fused/standalone query RMS+RoPE
        aac0948 feat(dsv4): compile CSA ops for Flash geometry
        b93e0761 feat(dsv4): consume int64 slots in HCA decode ops
        e1d1c985 feat(dsv4): fuse MLA output inverse RoPE quant
```

Flash 支持的实现方式：CSA 链（front/wq_b/hc_reduce）在同一源码上以
`DSV4_FLASH_CSA` 编译出第二组 TU（`*_flash.cu`），所有受宏影响的 namespace/kernel
符号重命名以避免 ODR 合并；pybind 注册 `*_flash` 入口，python wrapper 按输入形状
一行 dispatch。`geometry_csa()`/`geometry_hca()` 同时广告两组形状。随迁的几何无关
优化：wq_b BM16 小批模板（M<=16）、qnorm warp 数模板化、mqa fp4/fp8 与 standalone
`query_rms_rope_out` 的 `output_heads` 运行时化（64/128）。

当前已生成但尚未发布的本地 CUDA13.2 wheel：

```text
/root/work/cuda_extension/dist/
  rtp_kernel-0.1.0+aac0948b.20260819174054-cp310-cp310-linux_x86_64.whl
sha256: ebba2d0a9bcbbfd92d6773ce3bc6bec0cc45e31a82646b09c5f2c6a9f2c4cde7
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

CSA adapter 另外依赖当前 DeepGEMM 的：

```text
tf32_hc_prenorm_gemm
get_paged_mqa_logits_metadata
get_num_sms
```

HCA adapter 只依赖其中的 `tf32_hc_prenorm_gemm`（无 MQA schedule）。

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

### 2026-08-18：Mega HCA TP1 接入与验证

HCA 接入提交：开源 `c558d9b27`（本仓）+ `cuda_extension@b93e0761`。关键几何均取自
`DSV4CacheConfigHelper.cc`：HCA_KV 为 `tokens_per_block/128 = 2` entries/block；HCA_STATE
ring 为 `computeStateRing(128, kHcaOverlap=0, gen)`，非 MTP 时恰为 128（注意 `kHcaOverlap`
是 0，不是 CSA 的 1）；state 行为 `kv(512)|gate(512)` 交错 fp32，算子以两个 stride-1024
view 直接写框架池。

新增测试：

```text
//rtp_llm/models_py/modules/dsv4/fp8/test:test_mega_hca_adapter
//rtp_llm/models_py/modules/dsv4/fp8/test:test_mega_hca_rtp_eager
```

单层对照结果（真实 `AttentionFP8(compress_ratio=128, indexer=None)`，对照原
`_forward_decode_hca` 分支，B300/sm_103a）：

| 对照项 | `calc_diff` / 结果 | 门限 |
| --- | ---: | ---: |
| 最终 attention sublayer 输出（pos 0..3） | `2.140570e-05` | `< 1e-3` |
| 边界压缩步输出（pos 127，随机 state 环） | `7.721056e-07` | `< 1e-3` |
| 边界压缩写入 HCA_KV（解量化） | `1.533674e-05` | `< 1e-3` |
| SWA KV（解量化） | `0.0`（bitwise） | `< 1e-3` |
| HCA state 环 | `1.1e-06` | `< 1e-4` |
| 长上下文（pos 4095，随机 cache+state） | `4.038074e-06` | `< 1e-3` |
| CUDA Graph replay | bitwise 一致 | 精确一致 |

CSA 回归在新 wheel 与共享 runtime/block/transformer 改动下重跑，数值与 2026-08-15 基线
逐位一致。

性能（生产 CUDA Graph 口径；测量时 GPU 带约 15% 外部负载，绝对值有噪声）：

| Batch | Context | 步型 | 原路径 | Mega | 变化 |
| ---: | ---: | --- | ---: | ---: | ---: |
| 8 | 2000 | 非边界 | `176.0 us` | `140.8 us` | `-20.0%` |
| 8 | 2048 | 压缩边界 | `316.5 us` | `141.6 us` | `-55.3%` |
| 16 | 2048 | 压缩边界 | `315.3 us` | `145.2 us` | `-53.9%` |
| 128 | 4096 | 压缩边界 | `445.6 us` | `253.2 us` | `-43.2%` |

原路径的边界压缩是独立 Triton kernel，Mega 把它并进 WQ-B GEMM，因此边界步收益远大于
非边界步；按每 128 步一次边界加权约 `-20%`/层。两点已知说明：

1. 边界步会立即稠密读回本步刚压缩的 HCA_KV entry；两条实现在该 entry 上的 bf16/FP8
   舍入差（约 `1.5e-05`）在稠密短候选下的 softmax 占比大，把输出差放大到最坏
   `2.1e-04`（仍远低于 `1e-3` 主门限）。因此 HCA 性能用例的 cosine 门限为 `0.995`
   （非边界步实测 ~`0.99998`），并额外强制 per-request `calc_diff < 1e-3`。
2. HCA state 环对照为 `1e-06` 量级（CSA 为 `1e-11`）：extension FRONT-EMIT 的 bf16
   round 语义与框架 Triton compressor 存在实现差，后续可对齐；不阻塞当前结论。

### 2026-08-18：裁层 DSV4-Pro 本地 serving 端到端

用 `truncate_dsv4_pro.py` 从 `/mnt/nas1/nanjun.cp/DeepSeek-V4-Pro` 裁出 4 层
（`compress_ratios=[128,128,4,128]`，59 GB，`num_nextn_predict_layers=0`，需带 `encoding/`）。
单卡 `rtp_llm.start_server`（BF16 act、FP8 KV、`seq_size_per_block=256`）baseline 与
Mega（CSA+HCA 双开）各完成 3 条 greedy 请求（含跨 128-token 压缩边界的 200-token 生成），
无 crash/NaN。token 前缀 7~12 个一致后分岔；logits 对照 `calc_diff=5.6e-04/7.5e-04`、
`top1_same=1`、baseline top1 margin 仅 `0.0036/0.0040`——4 层裁层模型 argmax 病态敏感，
分岔不构成 Mega 缺陷证据；正确性定论需健康模型（见下）。

### 2026-08-19/20：Flash 双几何与全量 Flash 端到端

extension 侧（`aac0948` + `9f4c3fe`）：

- 新增 `tests/test_dsv4_mega_flash_csa.py`：hc_reduce（DIM 4096）、front（K 4096/N 4160）、
  wq_b（K 1024/64 heads，覆盖 BM16 与全部 32-row 模板）对照 torch 参考全过
  （GEMM `calc_diff < 1e-5`，fp4 链 byte-match）；另含 64-head 奇数 batch 的
  `query_rms_rope_out` 回归用例。
- Pro CSA 6/6、HCA 4/4 pytest 回归全绿（含 BM16 新路径）。
- 修复：`output_heads` 运行时化漏掉的三处 128-head 行数前置校验
  （mqa fp8/fp4 `numel % (128*512)`、standalone query_rms_rope 的 host 校验与
  kernel `>>7`）。Pro 偶数头批次永远不触发，Flash 首步 decode 即 abort；`9f4c3fe`
  修复并补测试。

RTP 侧（`5a393bedda` CSA、`da03d2b19c` HCA）：bazel
`test_mega_{csa,hca}_rtp_eager` + `test_mega_{csa,hca}_adapter` 全绿（Pro 回归口径）。

全量 Flash 端到端（`/mnt/nas1/hf/DeepSeek-V4-Flash-0731`，43 层 156 GB，单卡，
`max_seq_len 4096`）：baseline 与 Mega（CSA+HCA 双开，SWA-only 前两层走原生路径）
各完成 3 条 greedy 请求。两侧输出均语言连贯且关键答案一致（"Paris"、"2+2=4"）；
200-token 长生成前约 160 字符逐字相同后在近平局 token 处分岔，之后各自连贯。
0/3 文本逐字一致——与框架 smoke 对 cp2/cp4/tp1 各用 golden 的既有现象同类，
文本级验收应采用 per-配置 golden；logits 级定量对照在排队等空卡（`watch_and_run_logits.sh`）。

## 6. 端到端剩余缺口

按阻塞顺序还需要：

1. 发布 `9f4c3fe` 对应的 CUDA13 x86_64 wheel，更新开源/内源实际使用的依赖入口和 lock；
2. 增加由真实 `KVCacheManager` 创建 typed pools/block tables 的集成测试，替代手工 pool
   fixture（本地 serving e2e 已实际走真实 allocator，但缺 bazel 内可回归的形式）；
3. ~~校验 normal prefill -> Mega decode~~ 已在裁层 Pro 与全量 Flash serving 中覆盖
   （target verify / MTP 场景仍未覆盖）；
4. 整模型正确性收口：为 Mega 配置生成 per-配置 golden（框架 smoke 惯例），并在健康模型
   上完成 logits 级对照（Flash 对照排队中）；建议同时把 4 层 Pro 裁层 checkpoint 上传 NAS
   并新增 `v4_pro_4layer_tp1` / `..._mega` smoke case；
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

目标内源分支为 `develop/wangyin_ds_v4_20260424`。在开源提交稳定后（迁移清单现含 CSA 与
HCA 两组 adapter/runtime/weights/测试文件）：

1. 将目标内源 worktree 对齐远端分支，保留现有用户修改和 gitlink；
2. 迁移本分支的 adapter、runtime、weights、选路及测试文档改动，不迁移 Wuda TPDP 或改造版
   FlashMLA 逻辑；
3. 新 wheel 发布后，同时更新内源 CUDA13 requirements lock 和实际 Bazel 依赖选择；
4. 先跑与开源相同的 CPU tests 和 `//rtp_llm:rtp_llm` 完整编译；
5. 再在内源服务配置中只对 TP1 FP8 `DECODE/PDFUSION` 打开开关，按 `DSV4_MEGA_CSA` →
   `DSV4_MEGA_HCA` → 双开的顺序分阶段验证；双开后 Mega 覆盖全部 61 个 attention 层；
6. 完成第 6 节 GPU 矩阵后，才能把开关从实验配置提升为默认配置。

HCA 已按同样的“完整 sublayer adapter”模式接入（`MegaHCAAdapter`，独立 geometry 检查）。
SWA-only、prefill、TP2/DP2 与 FlashMLA 通用接口仍不修改；若后续接入这些场景，应分别
新增受支持的完整 sublayer adapter，不能放宽现有 CSA/HCA TP1 adapter 的 geometry 检查。

## 8. 开发操作手册（分支 / 编译 / 运行 / 测试 / benchmark）

以下均为本文档验证所实际使用的流程，路径以开发容器为准（`/root/work` 与
`/data3/<user>/root-work` 为同一目录）。

### 8.1 仓库与分支

| 仓库 | 地址 | 分支 | 角色 |
| --- | --- | --- | --- |
| Wuda（算子上游） | `git@github.com:guluguluhhhh/wuda.git` | `main`（Pro TP1）、`flash`（Flash CSA 几何源，`ce0b82b`） | 只读迁移源，不直接部署 |
| cuda_extension | `git@gitlab.alibaba-inc.com:foundation_models/cuda_extension.git` | `dsv4_megakernel`（HEAD `9f4c3fe`） | Mega 算子生产载体，出 `rtp-kernel` wheel |
| RTP 开源 fork | `git@github.com:guluguluhhhh/rtp-llm.git` | `dsv4-mega`（HEAD `da03d2b19c`） | 框架适配（adapter/runtime/weights/测试/本文档） |
| RTP 内源 | gitlab `foundation_models/RTP-LLM` | `develop/wangyin_ds_v4_20260424` | 内源载体；子模块 `github-opensource` 指向上一行的 fork 分支 |

本地布局：内源主仓检出后执行 `scripts/create_symlinks.sh`；`.worktrees/dsv4-mega`
是同一内源仓的 worktree（保有 Bazel 暖缓存，GPU bazel 测试在这里跑）；主树
`github-opensource` 做提交，测试前将改动文件同步进 worktree 的同名路径。

### 8.2 编译

CUDA Extension（venv 需 torch cu130，已备 `~/root-work/.venvs/rtpllm-dsv4-mega-*`）：

```bash
cd /root/work/cuda_extension
python build.py          # pip wheel 全量构建，约 13 分钟，产物在 dist/
pip install --force-reinstall --no-deps dist/rtp_kernel-*.whl
```

冒烟：`python -c "from rtp_kernel import dsv4_mega; print(dsv4_mega.geometry_csa())"`
应同时出现 Pro 与 `*_flash` 两组形状。

RTP Bazel 依赖本地 wheel：修改两树
`internal_source/deps/requirements_lock_torch_gpu_cuda13.txt` 中 `rtp-kernel` 行为
`file:///root/work/cuda_extension/dist/<wheel>` 并更新其 `--hash=sha256:`。该补丁
**不提交**；wheel 重建后（文件名含时间戳）必须同步刷新。完整编译：

```bash
cd .worktrees/dsv4-mega/github-opensource
bazelisk build //rtp_llm:rtp_llm --config=cuda13 --jobs=64 --verbose_failures
```

### 8.3 运行（本地 serving 端到端）

serving venv 为 `~/root-work/.venvs/rtpllm-e2e`（`rtp_llm` 以 site-packages 拷贝安装：
改完框架代码需把改动文件拷入该 venv，或重装）。可用 checkpoint：

```text
/root/work/checkpoints/DeepSeek-V4-Pro-4layer   59 GB 裁层（truncate_dsv4_pro.py 产物）
/root/work/checkpoints/DeepSeek-V4-Pro-6layer   87 GB 裁层
/mnt/nas1/hf/DeepSeek-V4-Flash-0731            156 GB 全量 43 层（单卡可跑；NAS 冷读约 40 min）
```

一键对照脚本（`/root/work/scripts_e2e/`，环境变量 `E2E_CKPT`/`E2E_GPU` 可覆盖）：

```bash
python run_e2e_compare.py                       # baseline + mega 两轮 serving + 文本对照
python run_e2e_logits.py baseline|mega|compare  # 三步式 logits 级对照（return_logits）
./watch_and_run_logits.sh                       # 轮询空卡（>=200GB），自动跑上面三步
```

脚本封装的关键运行要素（手工起 server 时同样必需）：
`MODEL_TYPE=deepseek_v4`、`CHECKPOINT_PATH`/`TOKENIZER_PATH`、`START_PORT`；
`--load_method scratch --act_type BF16 --fp8_kv_cache 1 --seq_size_per_block 256`
（必须为 128 的倍数且 >=128）；容器内 `/tmp/rtp-llm` 可能属他人，需预设 8 个 JIT
cache 环境变量（`FLASHINFER_WORKSPACE_BASE`、`DG_JIT_CACHE_DIR`、`TRTLLM_DG_CACHE_DIR`、
`TILELANG_CACHE_DIR`、`TORCH_EXTENSIONS_DIR`、`TVM_FFI_CACHE_DIR`、`CUTE_DSL_CACHE_DIR`、
`TRITON_CACHE_DIR`）到自有目录；`DG_JIT_CPP_STANDARD=20`。Mega 开关：
`DSV4_MEGA_CSA=1`、`DSV4_MEGA_HCA=1`（默认全关，即 baseline）。

### 8.4 测试

CUDA Extension（pytest，单卡 GPU，全套约 1 分钟）：

```bash
cd /root/work/cuda_extension
CUDA_VISIBLE_DEVICES=<idx> python -m pytest   tests/test_dsv4_mega_front_gemm_csa.py tests/test_dsv4_mega_wq_b_csa.py   tests/test_dsv4_mega_hc_fused.py tests/test_dsv4_mega_mqa_logits.py   tests/test_dsv4_mega_idx_post.py tests/test_dsv4_mega_mla_o_quant.py   tests/test_dsv4_mega_front_gemm_hca.py tests/test_dsv4_mega_wq_b_hca.py   tests/test_dsv4_mega_hca_chain.py tests/test_dsv4_mega_state_pool.py   tests/test_dsv4_mega_flash_csa.py -x -q
# tests/test_dsv4_mega_hca_e2e.py 需要 RTP_OPENSOURCE_ROOT 指向开源树
```

RTP（bazel，GPU 目标在 worktree 跑）：

```bash
cd .worktrees/dsv4-mega/github-opensource
bazelisk test --config=cuda13 --jobs=64 --test_output=summary   --test_env=CUDA_VISIBLE_DEVICES=<idx> --nocache_test_results   //rtp_llm/models_py/modules/dsv4/fp8/test:test_mega_csa_adapter   //rtp_llm/models_py/modules/dsv4/fp8/test:test_mega_csa_rtp_eager   //rtp_llm/models_py/modules/dsv4/fp8/test:test_mega_hca_adapter   //rtp_llm/models_py/modules/dsv4/fp8/test:test_mega_hca_rtp_eager
```

CPU/静态回归与端到端见第 5 节与 8.3。

### 8.5 Benchmark

1. **RTP 单层生产口径**（最有对比价值）：eager 测试内建性能模式，测同一真实
   sublayer 的原路径 vs Mega，三口径（eager / 生产 CUDA Graph / 预绑定算子链）
   并带 `1.05x` 回归门：

   ```bash
   bazelisk test ... //rtp_llm/models_py/modules/dsv4/fp8/test:test_mega_csa_rtp_eager      --test_env=DSV4_MEGA_RUN_PERF=1      --test_env=DSV4_MEGA_PERF_CASES="1:2048,8:2048,32:8192,128:65536"
   # HCA 同理换 test_mega_hca_rtp_eager；边界步用 ctx 为 128 的倍数（如 8:2048）
   ```

   已有基线：CSA B128/64K `-24.6%`（§5 表）；HCA 非边界 `-20%`、压缩边界 `-54%`、
   B128/4K `-43%`（§5 HCA 表）。
2. **端到端粗口径**：`run_e2e_compare.py` 输出每请求 `cost_time`/`iter_count`
   （aux_info），可对 baseline/mega 两轮做同 prompt 对比；正式吞吐 A/B 见第 6 节
   缺口 6，需要专用整机窗口。
3. **extension 侧微基准**：`tests/benchmark_dsv4_mega_flash_segments.py`（Flash HCA
   decode 链冷-L2 分段基准：opA→opB→Q norm/RoPE→FlashMLA→O-proj），
   以及 Wuda 仓 `dsv4_megakernel/megakernel/test/` 下的原始 bench（`test_e2e_decode.py`
   等，含 cuBLAS 对照）——后者在 Wuda 环境跑，用于算子级归因，不是 RTP 生产口径。

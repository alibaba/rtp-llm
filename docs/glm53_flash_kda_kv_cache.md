# 对照代码理解 GLM53-Flash 的 KDA 与状态缓存

本文按“一个 token 如何计算 → 历史如何进入状态 → 状态如何存取”的顺序解释。代码基线为当前工作树 HEAD `a05e23d7573aa431ebd5fc2c8e904b2e967bcd18`，整理日期 2026-09-15。以下代码片段注明了是源码节选还是解释用伪代码；公式由本地 recurrent kernel 推导。本次没有运行模型或 GPU 测试。

当前 prefill 默认使用 cuLA，Triton 为可选对照后端。为解释数学原理，先看本仓库中逐 token 的 Triton decode kernel；然后说明 prefill 如何计算同一种状态并保存检查点。这样不必先理解分块矩阵优化。

## 1. 先明确：我们要缓存的是什么

普通 softmax attention 通常保存此前每个 token 的 K/V。新 query 到来，再用它查询这串历史。

KDA 的做法是：**把历史不断写进一个固定大小的矩阵，新 query 直接读取这个矩阵。** 每个 head 各有一个矩阵，每层各有自己的矩阵，每个请求也各有自己的矩阵。

它还有一段短卷积，所以需要两种历史：

| 缓存 | 保存内容 | 下一 token 为什么需要 |
|---|---|---|
| `ssm_states` | 处理完已有序列后的 KDA 递推矩阵 | 用旧记忆更新新记忆、计算 attention 输出 |
| `conv_states` | 最近 `W-1` 个 token 的**卷积前投影 QKV** | 计算宽度为 `W` 的因果短卷积 |

这里的 `ssm` 是框架对递推状态的命名；它具体就是下面要解释的矩阵，不是另一份独立的传统 KV 列表。

**一个可恢复的 KDA 检查点必须同时有这两部分，而且对应同一个序列位置。** 只有矩阵没有卷积历史，下一 token 的 q/k/v 就算错；只有卷积历史没有矩阵，则丢失更早的记忆。

GLM53-Flash 只在 `linear_attention` 层使用 KDA；`deepseek_sparse_attention` 层仍走 MLA/indexer。配置入口见 [glm5_3_flash.py](../rtp_llm/models/glm5_3_flash.py) 的 `parse_glm53_flash_config`，层选择见 [kimi_linear.py](../rtp_llm/models_py/model_desc/kimi_linear.py) 的 `KimiLinearDecoderLayer.__init__`。本文只追 KDA 分支。

## 2. 先固定符号，避免把几种 gate 和 state 混起来

先只考虑“一层、一个请求、一个 head、一个 token”。

| 符号 | 含义 | 形状 |
|---|---|---|
| `x_t` | 当前层收到的 hidden state | `[hidden_size]` |
| `z_t` | 当前 token 的卷积前投影 QKV | `[3D]`，单 head 简化 |
| `q_t, k_t, v_t` | 短卷积之后的 query/key/value | 各 `[D]` |
| `beta_t` | 本次写入的强度 | 每 head 一个标量 |
| `g_t` | 对旧状态的对数衰减 | `[D]`，每个 key 维度一个 |
| `a_t = exp(g_t)` | 旧记忆保留比例 | `[D]` |
| `M_t` | 处理完 token t 后的记忆矩阵 | `[V,K]`，当前 `V=K=D` |
| `o_t` | KDA 核心读取输出 | `[V]` |

本文采用 `M[V,K]`，因为 decode kernel 的 `b_h` 正是这个方向：每一列是一个 key 方向，每一行是一个 value 分量。

**物理缓存默认按 `S[K,V] = Mᵀ` 的地址方向存放。** 第 8 节会对照地址公式解释。先用 `M` 推数学，避免中途反复转置。

## 3. 第一步：同一个 hidden state 产生四条分支

打开 [kimi_linear.py](../rtp_llm/models_py/model_desc/kimi_linear.py)，看 `KimiLinearKDA.forward` 的非 sharded 分支。源码节选：

```python
projected_qkv = self.in_proj_qkv(hidden_states)
beta_input = self.in_proj_b(hidden_states)
forget_gate = self.f_b_proj(self.f_a_proj(hidden_states))
g_proj = self.g_b_proj(self.g_a_proj(hidden_states))
```

四条分支各有一个用途：

```text
hidden state
  ├─ in_proj_qkv → 短卷积 → q、k、v → 读写记忆矩阵
  ├─ in_proj_b   → sigmoid → beta  → 控制这次写入多少
  ├─ f_a/f_b     → gate 激活 → g   → 控制旧矩阵忘掉多少
  └─ g_a/g_b     → sigmoid         → 控制最终输出多少
```

容易混淆的是名字：**`forget_gate` 后面变成递推公式里的 `g`；`g_proj` 是 output gate，不参与矩阵更新。** 两个低秩投影的名字看着相似，作用却不同。

多 head 时 `projected_qkv` 的最后一维是：

```text
[所有 local heads 的 Q | 所有 local heads 的 K | 所有 local heads 的 V]
宽度 = 2 * Hk * K + Hv * V
```

`KimiLinearKDABase.__init__` 的 `qkv_size` 就是这个公式。GLM53 配置令 K/V head 数相等、head dim 相等，所以简化为 `3HD`。

## 4. 第二步：短卷积为什么必须有 Conv cache

打开 `KimiLinearKDADecode._conv1d`。源码节选：

```python
conv_states = self._get_conv_states(kv_cache_tensor)
out = causal_conv1d_update(
    mixed_qkv,
    conv_states.transpose(1, 2),
    self.conv_weights,
    bias=None,
    activation="silu",
    ...,
)
```

假设卷积宽度 `W=4`，对某一个通道 `c`，其计算原理是：

```text
u_t[c] = SiLU(w0[c]*z_(t-3)[c]
            + w1[c]*z_(t-2)[c]
            + w2[c]*z_(t-1)[c]
            + w3[c]*z_t[c])
```

这是解释用公式：实际 kernel 把所有通道并行处理。卷积沿 token 维做，每个通道用自己的权重，不在这个操作里混合 Q/K/V 通道。

处理前后缓存内容为：

```text
处理 t 之前：Conv cache = [z_(t-3), z_(t-2), z_(t-1)]
本次输入：               z_t
处理 t 之后：Conv cache = [z_(t-2), z_(t-1), z_t]
```

然后把卷积输出 `u_t` 分成 q、k、v，进入 KDA。

所以 Conv cache 存的是 `z`，即投影后、卷积前的输入。它不存 `SiLU` 后的结果，也不存门控分支。新请求没有历史时，缺失的卷积历史按零处理。

对应实现见 [causal_conv1d.py](../rtp_llm/models_py/triton_kernels/causal_conv1d/causal_conv1d.py) 的 prefill/update kernels。当前还有可选 `GLM53_KDA_DECODE_FUSION=1`，由 [glm53_short_conv.py](../rtp_llm/models_py/triton_kernels/kimi_kda/glm53_short_conv.py) 融合处理满足条件的 decode 短卷积；需要保留的历史内容没有改变。

## 5. 第三步：对照 kernel，看矩阵怎么“忘、改、读”

打开 [fused_recurrent.py](../rtp_llm/models_py/triton_kernels/kimi_kda/fused_recurrent.py)，函数 `fused_recurrent_kda_fwd_kernel`。下面分解它的实际执行顺序。

### 5.1 载入旧矩阵

源码先创建 FP32 寄存器 tile：

```python
b_h = tl.zeros([BV, BK], dtype=tl.float32)
...
b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
```

`b_h` 是单个 head 的 `M_prev` 的一块。`BV` 是本次 program 处理的 value 行数，`BK` 覆盖 key 维；大矩阵可以由多个 program 分别处理不同 value 行。

后续同一个 tile 在寄存器里完成衰减、误差计算和更新，再写回 cache。

### 5.2 归一化 q/k

源码：

```python
b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
b_q = b_q * scale
```

调用方开启了 `use_qk_l2norm_in_kernel=True`。默认 `scale=1/sqrt(K)`。

归一化后的 k 表示“写入哪个方向”；归一化并缩放后的 q 表示“沿哪个方向读”。后文公式中的 q/k 均指处理后的向量，不再每次重复写归一化符号。

### 5.3 把 forget projection 变成衰减比例

源码节选：

```python
b_g = b_g + b_bias
if USE_LOWER_BOUND:
    b_gk = lower_bound * tl.sigmoid(exp(b_A) * b_g)
else:
    b_gk = -exp(b_A) * softplus(b_g)
```

这里 `b_A` 来自 `A_log`，`b_bias` 来自 `dt_bias`。实际作用在状态上的比例是 `exp(b_gk)`。

若配置 `lower_bound=-5`，则 `g` 在 `(-5,0)` 内，对应保留比例约在 `(0.0067,1)` 内。每个 key 维度可以有不同的保留比例。

接下来这行才真正修改旧状态：

```python
b_h *= exp(b_gk[None, :])
```

因为 `b_h` 是 `[V,K]`，`[None,:]` 把 gate 沿 value 行广播：**同一个 key 列里的所有 value 分量乘相同的衰减。**

数学上：

```text
M_decay = M_prev · diag(a)
a = exp(g)
```

这一步回答了“旧记忆如何遗忘”：先按 key 方向衰减旧矩阵，然后才写入当前 token。

### 5.4 用当前 k 查询旧矩阵，算出它已经记住了多少

源码：

```python
b_v -= tl.sum(b_h * b_k[None, :], 1)
```

这里 `sum(...,1)` 沿 key 列求和，得到一个 value 向量：

```text
v_pred = M_decay · k
delta  = v - v_pred
```

`v_pred` 是旧矩阵“看到当前 key 时，会返回什么 value”。`delta` 是当前 value 与已有记忆之间的差。

**这就是 delta rule 的关键：写入误差，不是每次把完整 v 再加一遍。** 如果旧状态已经准确表达了当前 `(k,v)`，误差接近零，就不必重复强化同一份内容。

此时变量 `b_v` 已经不再表示原始 value，而是误差向量。继续读 kernel 时要记住这一点。

### 5.5 用 beta 控制本次修正强度，再做外积写入

源码主干：

```python
b_v *= b_beta
b_h += b_v[:, None] * b_k[None, :]
```

GLM53 的 beta 是每 head 一个标量，由 `sigmoid(beta_input)` 得到。普通路径在 Python 中计算 sigmoid；融合路径可以把 sigmoid 放进 kernel，数学含义相同。

上面两行对应：

```text
correction = beta * (v - M_decay · k)
M_new = M_decay + correction · kᵀ
```

`correction[:,None] * k[None,:]` 是外积，生成 `[V,K]` 矩阵；不是向量点积。

忽略归一化里的小 epsilon、认为 `kᵀk=1`，更新后再次用同一个 k 查询：

```text
M_new · k
= M_decay · k + beta * (v - M_decay · k)
= (1-beta)*v_pred + beta*v
```

于是 beta 的含义非常直接：

- beta 接近 0：基本不吸收新 value，但前面的遗忘仍会发生。
- beta 接近 1：当前 key 方向的读取结果接近新 value。
- 中间值：将已有预测向当前 value 推进一部分。

不同 key 不一定正交，写入一个方向也可能影响另一个方向。这是固定矩阵存储历史的表达方式，不能把它理解成无损保存全部历史 K/V。

### 5.6 用当前 q 读取已经更新的矩阵

源码：

```python
b_o = tl.sum(b_h * b_q[None, :], 1)
```

对应：

```text
o = M_new · q
```

注意读的是 `M_new`，因此当前 token 的写入会影响自己的输出。这仍是因果计算：只用了旧状态和当前输入，没有用未来 token。

随后把 `b_h` 写回缓存。下一 token 从这个更新后的矩阵继续。

### 5.7 将整个 kernel 压缩成可读伪代码

以下是解释用伪代码，不是独立可运行的模型实现：

```python
# M: 一个 head 的旧状态，数学方向 [V,K]
q = l2norm(q_after_conv) / sqrt(K)
k = l2norm(k_after_conv)
v = v_after_conv
beta = sigmoid(beta_input)
a = exp(activate_forget_gate(forget_gate, A_log, dt_bias))

M = M * a[None, :]                  # 忘：每个 key 列衰减
prediction = M @ k                 # 看看已记住什么
correction = beta * (v - prediction)
M = M + correction[:, None] * k[None, :]  # 改：只写误差
o = M @ q                          # 读：输出 value 向量
# 保存新的 M，供下一 token 使用
```

## 6. 用一个 2×2 例子实际算两步

只看单个 head。为便于手算，直接给出归一化后的 key，省略短卷积、gate 投影和归一化 epsilon；这不是完整模型输入。令 `K=V=2`，beta 为 0.5。

### 第一个 token

初始矩阵为零，当前 `k=[1,0]`、`v=[2,4]`：

```text
M_prev = [[0,0],       prediction = [0,0]
          [0,0]]      correction = 0.5 * ([2,4]-[0,0]) = [1,2]

外积 correction * kᵀ = [[1,0],
                        [2,0]]

M_1 = [[1,0],
       [2,0]]
```

含义是：矩阵的第一个 key 方向现在记住了一半的 value `[1,2]`。若用 `q=[1,0]/sqrt(2)` 读取，得到 `[1,2]/sqrt(2)`。

### 第二个 token

这次两个 key 维度都保留一半旧记忆，仍然 `k=[1,0]`，但新 `v=[4,2]`：

```text
① 遗忘
M_decay = 0.5 * M_1 = [[0.5,0],
                       [1.0,0]]

② 当前 key 查询旧记忆
prediction = M_decay @ [1,0] = [0.5,1.0]

③ 只修正误差
correction = 0.5 * ([4,2]-[0.5,1]) = [1.75,0.5]

④ 写入
M_2 = [[0.5,0], + [[1.75,0], = [[2.25,0],
       [1.0,0]]    [0.50,0]]    [1.50,0]]

⑤ 读取
q = [1,0]/sqrt(2)
o = [2.25,1.50]/sqrt(2) ≈ [1.59099,1.06066]
```

与 kernel 的逐行对应：

| 手算步骤 | kernel |
|---|---|
| ① 遗忘 | `b_h *= exp(b_gk[None, :])` |
| ②、③ 的差值 | `b_v -= tl.sum(b_h * b_k[None, :], 1)` |
| ③ 缩放误差 | `b_v *= b_beta` |
| ④ 外积写入 | `b_h += b_v[:, None] * b_k[None, :]` |
| ⑤ 读取 | `b_o = tl.sum(b_h * b_q[None, :], 1)` |

如果误写成“直接把 `beta*v` 加进去”，第二步第一列会变成 `[2.5,2.0]`，与正确的 `[2.25,1.5]` 不同。原文只有公式时不容易看出的区别就在这里。

处理第三个 token 时，只需要 `M_2` 和 Conv history，不需要重新读取第一个、第二个 token 的独立 K/V。

## 7. 读出以后还没结束：Output gate 与 cache 无关

回到 `KimiLinearKDA.forward`。核心输出随后经过：

```text
每 head RMSNorm(o) × sigmoid(g_proj)
  → 拼回所有 heads
  → out_proj
  → 按并行配置进行归约
```

`g_proj` 只控制送入后续网络的输出。矩阵已经在前面的 recurrent kernel 中更新完毕，output gate 不会反过来修改已保存的 SSM。

这也解释了为什么 cache 里只需要 SSM + Conv，没有必要跨 token 缓存 `beta_input`、`forget_gate`、`g_proj`：这些量由当前 token 的 hidden state 当场产生。

KDA 被称为线性注意力，是因为固定 head 维度时，可以沿长度为 T 的序列递推，核心计算约为 `O(T*H*K*V)`，无需构造 `T×T` attention 矩阵。这不表示它与 softmax attention 完全等价，也不表示它在所有长度和硬件上都更快。

## 8. 数学矩阵如何对应真实 KV Cache 字节

### 8.1 一个物理块装一份完整状态

[C++ LinearKVCacheSpec.h](../rtp_llm/cpp/cache/LinearKVCacheSpec.h) 定义：

```text
ssm_state_size  = Hv * K * V
qkv_size        = 2 * Hk * K + Hv * V
conv_state_size = (W-1) * qkv_size

block_bytes = ssm_state_size * sizeof(ssm_dtype)
            + conv_state_size * sizeof(conv_dtype)
```

布局为：

```text
某层、某请求位置对应的物理块
┌────────────────────────────────────────────────────┐
│ 所有 local heads 的 SSM 状态，FP32                  │
├────────────────────────────────────────────────────┤
│ history 0: 全 Q | 全 K | 全 V                       │
│ history 1: 全 Q | 全 K | 全 V                       │
│ ... 共 W-1 行，精度为 conv_state_dtype              │
└────────────────────────────────────────────────────┘
```

GLM53 的 `Glm53FlashModelConfig.init_linear_attention_cache_precision` 强制 SSM 为 FP32；Conv 跟随模型计算 dtype，BF16 模型中为 BF16。MLA 使用 FP8 并不会把这里的矩阵也变成 FP8。

C++ 把 SSM 叫 `k_block_size`、Conv 叫 `v_block_size`，是为了复用通用 KV 管理接口。这两个名字不表示此处物理保存了传统 attention 的 K 矩阵和 V 矩阵。

### 8.2 Python view 并没有复制数据

[typed_storage_view.py](../rtp_llm/models_py/utils/typed_storage_view.py) 的 `LinearCacheConverter` 用同一个 `untyped_storage()` 建立不同 dtype 的 view：

```text
SSM 起点 = block_base
Conv 起点 = block_base + SSM_bytes
下一块起点 = block_base + 实际 block_stride_bytes
```

因此两个 view 指向同一物理块中的不同区域。不能先把整块 `.to(FP32)` 再切片；那会把混合存储载体当成一种数值数组进行转换，破坏原始字节。

### 8.3 `b_h[V,K]` 与 cache `[K,V]` 的对应

recurrent kernel 默认 `state_v_first=False`。源码地址计算：

```python
p_h0 = p_h0 + o_k[None, :] * V + o_v[:, None]
```

对寄存器中的元素 `M[v,k]`，读取的物理偏移是 `k*V+v`。所以物理存放的是 `S[k,v]=M[v,k]`，即 `S=Mᵀ`。写回也使用同样的偏移。

例如上面的 `M_2`：

```text
数学 M_2[V,K] = [[2.25,0],
                 [1.50,0]]

物理 S_2[K,V] = [[2.25,1.50],
                 [0,   0  ]]

连续 FP32 字节对应数值次序：2.25, 1.50, 0, 0
```

converter 的外部 shape 命名为 `[num_blocks,Hv,V,K]`，但当前要求 `V=K`，与 `[num_blocks,Hv,K,V]` 的数值尺寸一致。**相同 shape 不代表可以把内容随便转置。** dump、checkpoint 和异构传输时，以这个地址公式及具体后端的状态方向约定为准。

## 9. `seq_size_per_block` 为什么没有乘进状态大小

设 `B=seq_size_per_block=128`。

普通 paged KV 的一页通常装 128 个 token 的 KV。KDA 的一个物理块只装**一份矩阵和一份卷积历史**，表示某个序列位置的累计状态。

```text
slot 0：处理到前 128-token 区间内某位置时的状态
slot 1：处理到第 129～256 token 区间内某位置时的状态
...
```

未完成区间的尾块会随 decode 更新；完成区间且被保留的块可以作为边界快照。

所以：

- `B` 决定逻辑位置与检查点粒度；
- `H/K/V/W/dtype` 决定每份状态有多少字节；
- allocator 保留多少份状态决定实际物理占用。

三者不能混为一谈。

## 10. 普通 decode：完整走一遍缓存寻址

进入 `KimiLinearKDADecode.forward` 后，代码从 `kv_cache_base` 建立 Conv 和 SSM view。两条 kernel 使用同一张当前 LINEAR group 的 `block_map`。

设本次输入 token 是序列第 n 个 token，处理完后共有 n 个 token 的状态。`sequence_lengths_plus_1_d` 按这个累计长度语义传入。kernel 的 helper 为：

```python
cal_block_idx(x, B) = (x - 1) // B
```

对应读写位置：

```text
read_slot  = cal_block_idx(n-1, B) = (n-2)//B
write_slot = cal_block_idx(n,   B) = (n-1)//B
read_id    = block_map[request, read_slot]
write_id   = block_map[request, write_slot]
```

这是已有 prefix state 后的 decode 路径；新请求从零初态走 prefill，不应把 `n=1` 直接套进这个读取旧状态的公式。

例如 `B=128`，请求块表为 `slot0→物理块7，slot1→物理块12`：

| 本次处理 token | 读取 | 写入 | 状态含义 |
|---|---|---|---|
| 128 | 块7：截至127 | 块7：截至128 | 当前块更新到边界 |
| 129 | 块7：截至128 | 块12：截至129 | 读旧边界，写新尾块 |
| 130 | 块12：截至129 | 块12：截至130 | 新尾块继续更新 |

处理 token129 时，执行的是：

1. 从块7 的 Conv 区读投影历史，和新投影做卷积；将新历史写到块12 的 Conv 区。
2. 从块7 的 SSM 区读旧矩阵，执行第5节的遗忘、误差修正和读取；将新矩阵写到块12 的 SSM 区。
3. 两部分都完成后，块12 才是一份截至 token129 的可用状态。

`inplace_final_state=True` 表示写回同一整层 cache 存储，不保证每轮读取与写入的物理块相同。跨块时保留旧尾块，就是为了不提前释放下一步要读的状态。

## 11. Prefill：计算同一种递推状态，但批量处理多个 token

数学上，prefill 等价于把第5节的递推连续应用 L 次。实现上不必启动 L 次单 token kernel：可以把块内计算组织成矩阵运算，再串联块之间的状态。

Prefill 与 decode 要衔接，关键不是内部用了多少临时张量，而是：**最后保存的状态必须表示处理完同一个前缀之后的同一种矩阵和卷积历史。**

### 11.1 恢复初始状态

`KimiLinearKDAPrefill._fla` 调用 [block.py](../rtp_llm/models_py/triton_kernels/fla/block.py) 的 `load_initial_state_from_block_map`：

```text
已有前缀长度 P=0：initial state 为零
已有前缀长度 P>0：读取 block_map[(P-1)//B] 的矩阵
```

卷积也从同一前缀位置读历史。多请求通过 `cu_seqlens` 分隔输入，各自恢复各自的 initial state，不能在拼接的 batch token 流上串联状态。

### 11.2 当前默认路径：cuLA checkpoints

`KimiLinearKDAPrefill._backend()` 读取：

```python
os.environ.get("GLM5_KDA_PREFILL_BACKEND", "cula")
```

默认 cuLA 路径由 `_cula_checkpoint_prefill` 调用外部 `cula.kda.chunk_kda`，传入 FP32 initial states，并要求向指定 buffer 输出 checkpoint；当前接口设置 `output_final_state=False`，不从独立 `final_state` 返回值取状态。

本仓库负责的 checkpoint 落块逻辑见 [checkpoint.py](../rtp_llm/models_py/triton_kernels/kimi_kda/checkpoint.py)。令本次新输入长度为 L：

```text
检查点数量 = ceil(L/B)
第 j 个检查点对应的新输入末端 = min((j+1)*B, L)
目标逻辑 slot = (P + 新输入末端 - 1)//B
```

例如 `P=128,L=200,B=128`：

| 本次检查点 | 累计已处理长度 | 目标 slot |
|---|---:|---:|
| 新输入第128个 token 后 | 256 | 1 |
| 新输入第200个 token 后 | 328 | 2 |

最后的部分块也保存当前末尾状态，供 decode 继续。scatter kernel 只向合法、已分配的物理 block 写入；没有 materialize 的逻辑位置跳过。

此路径明确要求 `P` 是非负且按 page size 对齐。外部 cuLA 的内部优化不在本文逐行审计范围内；这里确认的是本仓库传参、checkpoint buffer 及 scatter 的合同。

### 11.3 可选路径：本地 Triton chunk

设置 `GLM5_KDA_PREFILL_BACKEND=triton` 时：

```text
chunk_kda → 输出本次 token 的 attention 结果
          → 返回中间 chunk 入口状态 h
          → 返回序列最终状态 final_state
store_ssm_state_to_block_map → 将所需边界状态写入物理 cache
```

[chunk.py](../rtp_llm/models_py/triton_kernels/kimi_kda/chunk.py) 的 `KDA_CHUNK_SIZE` 支持 64、128、256，默认64。**计算 chunk 大小与缓存 page size 是两件事。** 当前调用方会检查 chunk size 与 cache alignment 的整除关系。

在 [chunk_fwd.py](../rtp_llm/models_py/triton_kernels/kimi_kda/chunk_fwd.py) 中，可以按功能看三个阶段：gate 的块内累计、`chunk_kda_fwd_intra` 的块内处理、状态传播及输出计算。临时的 `w/u/h` 等是批量实现递推的工作区，不是新增的持久 KV 类型。

当前 [block.py](../rtp_llm/models_py/triton_kernels/fla/block.py) 在非末尾 chunk 到达 page 边界时，从下一 chunk 的入口状态取快照；末尾使用 `final_state`。旧文档提到的 `chunk>0` 条件在当前版本已不存在，不能继续据此判断第一个边界会漏存。

无论选择哪个后端，Conv cache 都在短卷积时保存边界/末尾历史；SSM cache 在 attention 状态计算后保存。两者都对应相同累计位置，prefill→decode 才能衔接。

## 12. 为了 MTP，为什么还要保存多个候选状态

`is_target_verify=True` 时，KDA 进入 decode 分支，把输入整理成 `[batch,候选token数,...]`。同一个 kernel 连续递推多个 token，但为每一步分别保存状态。

源码写入公式：

```python
write_block_offset = cal_block_idx(sequence_length, B) + i_t
```

所以候选位置不是按普通页内偏移堆到同一份矩阵里，而是：

```text
旧状态 → 候选1状态 → 候选2状态 → 候选3状态
            slot j      slot j+1     slot j+2
```

即使这些 token 在同一普通128-token区间，也必须有不同物理快照。如果只保留最后一份，候选2被拒绝时就不能直接找回候选1之后的状态。

验证结束后，[GenerateStream.cc](../rtp_llm/cpp/engine_base/stream/GenerateStream.cc) 根据实际接受长度调用交换工具，[StreamCacheResource.cc](../rtp_llm/cpp/engine_base/stream/StreamCacheResource.cc) 的 `swapLinearBlocks` 调整 LINEAR 组映射。SSM 和 Conv 在同一个块里，因此一起选中正确版本。

这是通过已保存快照恢复，不是把被拒绝 token 从矩阵里“减掉”。被拒绝步骤发生了遗忘和误差修正，不能按传统 KV 删除几行的方式处理。

## 13. 为什么有块表，却不必给历史每一页都分配矩阵

计算下一 token 只需要最新状态。历史快照主要服务于前缀复用、保留跨块输入、MTP 恢复。

[LinearKVCacheGroup.cc](../rtp_llm/cpp/cache/LinearKVCacheGroup.cc) 因此允许：

```text
逻辑 slot: 0   1   2   3   4   5
物理 ID:   ·   ·   ·   ·   7  12
```

空洞不代表丢了下一步计算需要的信息；截至前面的历史已经被尾状态概括。空洞必须仍占逻辑位置，不能把数组压缩后改变 token→slot 关系。

当前有两种策略：

| 模式 | 历史状态保留方式 |
|---|---|
| 旧模式 | 尾部状态 + 开启复用时每 `linear_step` 个块的快照，可用 `linear_fixed_cap` 限制较旧快照 |
| `ENABLE_LINEAR_ATTN_REQUEST_CACHE=1` | 工作尾部 + speculative reserves + 最近合法对齐的复用候选；池按并发预算 |

开启 request-cache 后，[CacheConfig.h](../rtp_llm/cpp/cache/CacheConfig.h) 对 decode 的自动 LINEAR 池预算主要为：

```text
max_generate_batch_size × (2 + speculative_reserve_blocks)
```

这与“一个状态矩阵固定大小”是两个层次：数学决定单份矩阵大小，allocator 决定保留几份、给多少并发预留空间。

前缀命中也不能只检查 MLA 页。必须在同一边界找到完整 KDA 状态，新请求才能跳过前缀计算。当前 whole-request memory connector 已提供整状态保存/恢复路径及相应限制，详细适配见 [Host KV Cache 方案](glm53_flash_host_kv_cache_design.md)。

## 14. 用真实尺寸把“状态”和“缓存池”区分开

以下尺寸来自 [glm5_3_flash_config_test.py](../rtp_llm/test/glm5_3_flash_config_test.py) 的测试配置，不是本次加载的 checkpoint：64 heads、D=128、W=4。

Attention TP=1，Conv 为 BF16：

```text
单层 SSM  = 64*128*128*4 bytes = 4 MiB
单层 Conv = 3*(3*64*128)*2 bytes = 144 KiB
单层一份完整状态 = 4.140625 MiB
```

测试的层安排有34个 KDA 层：

```text
所有 KDA 层各一份状态：140.78125 MiB
所有 KDA 层各两份状态：281.5625 MiB
```

attention TP=8 时 local heads=8，每 rank 的对应有效载荷除以8；实际应使用 `get_attn_tp_size()`，不要把 CP/DP 的进程数直接当成 head 切分数。

总内存还要乘实际请求/候选/缓存快照份数，并加空闲预分配块、临时 workspace。MLA/indexer、权重、激活另计。

把活跃 KDA 状态放在 host 并每 token 读回/写回，会搬整份矩阵；MLA 按需拉取的是稀疏 token 行，二者的传输单位完全不同。这也是为什么 KDA 更适合把活跃状态留在 GPU，只在恢复或保存检查点时传输完整状态。

## 15. 按这个顺序读代码

1. [kimi_linear.py](../rtp_llm/models_py/model_desc/kimi_linear.py)：先看 `KimiLinearKDA.forward` 的四条投影，再看 `KimiLinearKDADecode._conv1d` / `_fla`。
2. [fused_recurrent.py](../rtp_llm/models_py/triton_kernels/kimi_kda/fused_recurrent.py)：对照第5、6节，逐行看 `b_h`、`b_v` 如何变化。这是理解原理最关键的一段。
3. [typed_storage_view.py](../rtp_llm/models_py/utils/typed_storage_view.py) 与 [LinearKVCacheSpec.h](../rtp_llm/cpp/cache/LinearKVCacheSpec.h)：核对矩阵/卷积历史对应的 byte offset、dtype 和 stride。
4. [checkpoint.py](../rtp_llm/models_py/triton_kernels/kimi_kda/checkpoint.py)：看当前默认 prefill 的检查点如何写进同一缓存。
5. [LinearKVCacheGroup.cc](../rtp_llm/cpp/cache/LinearKVCacheGroup.cc)：最后再看哪些快照要保留、何时回收。

判断是否理解了这条链路，可以检查三个问题：下一 token 为什么同时需要 SSM 和 Conv？`b_v -= ...` 为什么不能省？一个128-token逻辑块为什么只保存一份状态？它们分别对应局部卷积历史、delta误差修正、固定大小递推记忆这三个核心原理。

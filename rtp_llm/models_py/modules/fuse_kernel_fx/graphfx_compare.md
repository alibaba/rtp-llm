# GraphFX 融合方案对比:Qwen3.5 vs DSV4(初学者版)

> 写给完全没接触过 `torch.compile` 的同学。从最基础的概念讲起,逐步解释这套架构在做什么、为什么这么做。

---

## 第一部分:三个新概念,从零开始

### 1.1 我们要解决的问题

模型里有大量"两次连着的小算子",比如:

```python
# 一段典型的 transformer 代码
hidden = rmsnorm(x, weight)              # 第 1 次 kernel launch
fp8, scale = quantize(hidden)            # 第 2 次 kernel launch
```

GPU 每次启动一个 kernel 都有固定的开销(几微秒到几十微秒,叫 "launch overhead")。如果一个 layer 里有 N 个小算子,每次 forward 就是 N 次 launch,加起来可能比真正算的时间还长。

**解决办法**:写一个 "fused kernel",把这两步合并成一次:

```python
fp8, scale = fused_rmsnorm_quant(x, weight)  # 只有 1 次 kernel launch,而且省一次内存读写
```

但问题来了:模型里有几十个这种 "两步合并" 的机会,每加一个 fused kernel,**就要去改模型代码**:

```python
# 模型代码原来
hidden = rmsnorm(x, weight)
fp8, scale = quantize(hidden)

# 加 fused kernel 后变成
if self._fuse_rmsnorm_quant:
    fp8, scale = fused_rmsnorm_quant(x, weight)
else:
    hidden = rmsnorm(x, weight)
    fp8, scale = quantize(hidden)
```

5 个 model 文件 × 10 处这种 if 分支 = 50 处散落的"双路径"代码。**有没有办法让模型代码保持干净,让"用不用 fused kernel"这个决定自动化做出来?**

`torch.compile` + GraphFX 就是这个自动化机制的基础。

---

### 1.2 `torch.compile` 是什么

它是 PyTorch 2.0 开始提供的一个**函数包装器**。给它一个普通的 Python 函数,它返回一个"看起来一样、但跑起来不一样"的函数:

```python
def my_func(x):
    y = x + 2
    z = torch.relu(y)
    return z

# 这是原函数
result1 = my_func(torch.randn(4))

# 包一下
compiled = torch.compile(my_func)

# 用法完全一样
result2 = compiled(torch.randn(4))
# 结果跟 my_func 一模一样,但中间发生的事情完全不同
```

**默认情况下,`torch.compile` 内部会**:
1. 第一次调用时,**捕获**这个函数实际在干什么(`x + 2`、`torch.relu`)
2. 把这些操作**重新编译成一个高效的 CUDA kernel**(默认用 Inductor 这个组件来做)
3. 后续调用直接跑编译好的版本

听起来很美好。但我们 **不用它默认的编译路径**。我们只用它的"捕获"能力,后面解释为什么。

`torch.compile` 有几个关键参数:

| 参数 | 含义 |
|---|---|
| `backend=` | 捕获到的图给谁处理。默认 `"inductor"`(自动生成 Triton kernel);**我们换成自己写的函数** |
| `fullgraph=True/False` | 捕获不了的地方是报错(True),还是分段捕获(False) |
| `dynamic=True/False` | shape 变化时要不要重新编译 |

---

### 1.3 Dynamo 是什么

Dynamo 是 `torch.compile` 背后真正负责"捕获"的组件。

**它干什么**:

当你调用 `compiled(torch.randn(4))` 时,Dynamo 接管,**逐条 Python 字节码符号执行** `my_func` 的代码。"符号执行" 的意思是:它不真的把 `x + 2` 算出来,而是 **记录"这里发生了一次加法,左操作数是参数 x,右操作数是常数 2"**。

形象点说:

```python
# 真实执行(eager 模式)
y = x + 2    # GPU 真的算了一次加法,y 里有真数据

# Dynamo 符号执行
y = x + 2    # GPU 不算,Dynamo 只记下:y = add(x, 2)
```

Dynamo 一路记录,直到把整个函数走完,产出一个**节点列表**:

```
node 1: x 是输入
node 2: y = add(x, 2)
node 3: z = relu(y)
node 4: 返回 z
```

这个节点列表的容器,叫 **FX Graph**(下面讲)。

**遇到不能符号执行的地方怎么办**:有些 Python 代码 Dynamo 处理不了,比如:

```python
if x.item() > 0:   # 需要真的算出 x 的值才能判断
    ...
```

这种情况下 Dynamo 做 **graph break**(图断裂):
- 把"到这一行为止"的节点打包成一张 FX Graph
- 这一行 `x.item()` 退回去用 Python eager 模式真的执行
- 然后开始抓下一张图

所以一个复杂函数,可能被 Dynamo 拆成 **3-5 张图,中间夹杂 eager 代码**。

**Dynamo 还有一个 cache**:第二次调用 `compiled(torch.randn(4))`,Dynamo 检查输入 shape / dtype 跟上次是否一样,**一样的话直接跑上次的编译结果,不重新抓图**。

---

### 1.4 FX Graph 是什么

FX Graph 是 Dynamo 抓出来的"操作清单"的官方数据结构。

继续上面的例子:

```python
def my_func(x):
    y = x + 2
    z = torch.relu(y)
    return z
```

Dynamo 抓出来的 FX Graph 长这样(伪代码表示):

```
placeholder    name=x                          # 函数的输入
call_function  target=operator.add  args=(x, 2)     # y = x + 2
call_function  target=torch.relu    args=(y,)       # z = relu(y)
output         args=(z,)                            # 返回值
```

每个节点(node)有这几个字段:

| 字段 | 含义 |
|---|---|
| `op` | 节点种类:`placeholder`(输入)/ `call_function`(调用一个普通函数)/ `call_method`(调用一个方法,如 `.contiguous()`)/ `call_module`(调用一个子模块)/ `output`(返回) |
| `target` | 调用的具体对象(`operator.add`、`torch.relu`、`"contiguous"` 等) |
| `args` / `kwargs` | 输入参数(可以是常数,也可以是别的 node) |

整张 FX Graph 包在一个 `GraphModule` 对象里,它本身就是个 `nn.Module`,可以 `.code` 看反序列化出来的 Python 代码,可以 `.recompile()` 让它重新跑。

**关键能力**:我们可以写 Python 代码去 **遍历这些 node、增/删/改**。这就是"FX pass"。

举例:写一个 pass 把 `add(x, 2)` 替换成 `add(x, 3)`:

```python
def my_pass(gm):
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == operator.add:
            if node.args[1] == 2:
                node.args = (node.args[0], 3)   # 把常数 2 改成 3
    gm.recompile()
    return gm
```

跑完这个 pass 后,`compiled` 调用时算出来的就是 `(x+3).relu()` 而不是 `(x+2).relu()`。

---

### 1.5 三者关系:一张图

```
你写的代码                      def my_func(x):
                                    y = x + 2
                                    z = torch.relu(y)
                                    return z

                                compiled = torch.compile(my_func, backend=my_backend)
                                compiled(torch.randn(4))    ← 调用
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────┐
│ torch.compile 包装层                                       │
│   - 第一次调用 → 启动 Dynamo                              │
│   - 后续调用 → 查 cache,命中就直接跑编译结果              │
└─────────────────────────────────────────────────────────┘
                                          │
                                          ▼ 第一次
┌─────────────────────────────────────────────────────────┐
│ Dynamo:逐条 Python 字节码符号执行                          │
│   - 看到 `x + 2`     → 记下一个 add 节点                   │
│   - 看到 `relu(y)`   → 记下一个 relu 节点                  │
│   - 看到 graph break → 把当前积累的节点打包成 FX Graph     │
│                                                          │
│   产物:一个或多个 FX GraphModule                          │
└─────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────┐
│ my_backend(gm, example_inputs)                           │
│   ← 我们在这里写 FX pass,改写图节点                       │
│   ← 返回一个可调用的东西(通常就是改写后的 gm.forward)   │
└─────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                                  存进 Dynamo cache
                                          │
                                          ▼
                                  执行,返回结果
```

---

## 第二部分:为什么我们要用这套机制(而不是直接改模型代码)

回到一开始的问题:"模型里散落了 50 处 `if self._fuse_*:`,有没有办法让它自动化"?

GraphFX 的思路是:

1. **模型代码保持干净**,只写最朴素的"两步分开"版本
   ```python
   hidden = rmsnorm(x, weight)
   fp8, scale = quantize(hidden)
   ```
2. 用 `torch.compile(model.forward, backend=my_backend)` 包一下
3. `my_backend` 拿到 FX Graph,**自动识别**"哪两个节点是 RMSNorm + quant 这种可以合并的模式"
4. 把这两个节点**替换**成一个 `fused_rmsnorm_quant` 节点
5. 后面 PyTorch 跑改写后的图,就只 launch 一个 kernel 了

**模型代码作者不需要知道 fused kernel 的存在**。新加一个 fused kernel,只要写一个 FX pass。

这就是 DSV4 和我们 Qwen3.5 这次重构都在做的事。

---

## 第三部分:DSV4 怎么做的

DSV4(DeepSeek V4)是 RTP-LLM 里第一个采用这套架构的模型。它的特点:**模型代码天生就是干净的**,直接走 GraphFX 路线。

### 3.1 模块布局

```
rtp_llm/models_py/modules/dsv4/fusions/
    fusion_registry.py             ← 框架核心:backend、Dynamo 配置、pass 注册表
    graphfx_injector.py            ← 安装入口
    indexed_rope_pass.py           ← FX pass 1:删多余的 gather
    rmsnorm_fp8_quant_pass.py      ← FX pass 2:RMSNorm + quant 合并
    rmsnorm_bf16_fp8_quant_pass.py ← FX pass 3:同上但带 BF16 输出
    kv_rope_fp8_quant_pass.py      ← FX pass 4:KV path 的 quant 合并
```

### 3.2 安装时机

模型加载完之后(`BaseModel.load()` 末尾),如果环境变量 `DSV4_GRAPHFX_FUSION=1`:

```python
# 简化版伪代码
if env("DSV4_GRAPHFX_FUSION") == "1":
    # 把 py_model.forward 替换成 torch.compile 包装版
    py_model.forward = torch.compile(
        py_model.forward,
        backend=dsv4_fusion_backend,   ← 我们的 backend
        fullgraph=False,                ← 允许 graph break
        dynamic=True,                   ← shape 变化不重 trace
    )
```

之后 C++ 推理引擎每次调 `py_model.forward(inputs)`,实际跑的是 `torch.compile` 包装后的版本。

### 3.3 backend 干什么

```python
def dsv4_fusion_backend(gm, example_inputs):
    # gm 就是 Dynamo 抓出来的 FX GraphModule
    for pass_fn in registered_passes:    # 按 priority 顺序跑 4 个 pass
        gm = pass_fn(gm)
    gm.recompile()                       # 让改写后的图重新生效
    return gm.forward                    # 还给 PyTorch 当作正常 callable
```

### 3.4 4 条 pass 各做什么

| 优先级 | 名字 | 看见什么模式 | 改成什么 |
|---|---|---|---|
| 5 | `indexed_rope_fx` | `freqs_cis.index_select(...).contiguous() → fused_rmsnorm_rope` | 删掉中间的 gather,直接把 `freqs_cis` 和 `position_ids` 喂给一个新的 indexed CUDA kernel |
| 10 | `rmsnorm_bf16_fp8_quant_fx` | RMSNorm 的输出 **同时** 被 BF16 算子和 FP8 quant 用 | 替换成一个三输出 fused kernel(BF16 + FP8 + scale) |
| 11 | `rmsnorm_fp8_quant_fx` | RMSNorm 的输出 **只** 被 FP8 quant 用 | 替换成单输出 fused kernel |
| 20 | `kv_rope_fp8_quant_fx` | KV 处理后挂的孤立 FP8 quant | 类似上面,用 provenance 机制(下面讲) |

### 3.5 跨图融合:provenance 机制

前面说过,Dynamo 遇到 graph break 会把代码切成多张 FX Graph。问题是:**RMSNorm 可能在图 A,quant 可能在图 B**,这时单个 pass 只看见图 A 或图 B 里的一半,没法合并。

DSV4 的解法叫 **provenance(出处)**:

```
图 A:                                          图 B(几行 Python eager 之后):
  hidden = rmsnorm(x, weight)                    fp8, scale = quantize(hidden)

FX pass 改写图 A:                              FX pass 改写图 B:
  hidden = producer_token(x, weight)             fp8, scale = from_provenance(hidden)
  # 跑原 rmsnorm + 把 (x, weight, hidden 指纹)    # 查表看 hidden 是谁的输出
  # 记到一张全局表里                              # 找到 → 跑 fused kernel
                                                  # 找不到 → 回退到普通 quantize
```

这个全局表用 **`weakref`(弱引用)**,以三种方式记 hidden tensor 的"身份":
- Python `id(tensor)`
- `(data_ptr, shape, stride, dtype, device)` 元组
- `(data_ptr, numel, last_dim, dtype, device)` 元组

任意一种命中就认为是同一个 tensor。这是为了应对 Dynamo 可能把 tensor 包成 view、改下 stride 之类的情况。

**这套机制是 eager 模式做不到的** —— eager 下你想合并跨函数的两个 op,得手工把它们的封装拆掉。

### 3.6 让 Dynamo "看得见、走得过" 的辅助工具

要让 GraphFX 工作,还有几个工程细节:

1. **`torch.library.custom_op`**:有些 fused kernel 是 pybind 暴露的 C++ 函数(比如 `rtp_llm_ops.rmsnorm`),Dynamo 默认处理不了(会因为 FakeTensor 阶段读 raw 指针而崩)。我们用 `torch.library.custom_op` 包一下,告诉 Dynamo "这是一个会修改第 1 个参数的不透明 op",让 FX 把它当成一个普通节点

2. **`torch._dynamo.allow_in_graph`**:Triton kernel(如 `fused_rmsnorm_rope`)默认会被 Dynamo 一路 inline 到底层 ptr 操作。我们告诉 Dynamo "这个函数当成一个原子节点,别拆",这样 FX pass 才能稳定按名字匹配

3. **`torch.compiler.disable`**:模型里很多"非计算"的辅助函数(metadata builder、KV cache 写入、debug dump),让 Dynamo 去 trace 它们只会让图变脏。我们给这些函数加 `@torch.compiler.disable`,Dynamo 看到就直接 graph break,这部分留 eager 跑。DSV4 disable 了约 30 个这种函数

4. **`torch._dynamo.mark_dynamic(tensor, dim)`**:告诉 Dynamo "这个 tensor 的第 0 维是动态的(batch size、token 数变化)",避免每个 batch size 都重新 trace

---

## 第四部分:Qwen3.5 怎么做的(本次重构后)

跟 DSV4 框架几乎一样,但起点不同 —— Qwen3.5 的模型代码 **不是天生干净** 的,所以我们额外做了一步"模型代码减法"。

### 4.1 模块��局

```
rtp_llm/models_py/modules/fuse_kernel_fx/
    fusion_registry.py             ← 跟 DSV4 同形态
    graphfx_injector.py            ← 同上
    _pass_helpers.py               ← 4 个 pass 共用的小工具
    add_rmsnorm_fp8_quant_pass.py  + add_rmsnorm_runtime.py
    rmsnorm_gated_fp8_quant_pass.py
    silu_and_mul_fp8_quant_pass.py
    sigmoid_mul_fp8_quant_pass.py
```

### 4.2 4 条 pass

| 优先级 | 名字 | 看见什么模式 | 改成什么 |
|---|---|---|---|
| 10 | `add_rmsnorm_fp8_quant_fx` | `RMSResNorm(h, r) + sgl_per_token_group_quant_fp8` | `fused_add_rmsnorm_fp8_quant`(单输出 / 三输出根据下游) |
| 15 | `rmsnorm_gated_fp8_quant_fx` | `layer_norm_fwd(x, w, b, eps, z=gate, is_rms_norm=True) + sgl quant` | `fused_rmsnorm_gated_fp8_quant` |
| 20 | `silu_and_mul_fp8_quant_fx` | `silu_and_mul(up) + sgl quant` | `silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd` |
| 25 | `sigmoid_mul_fp8_quant_fx` | `attn * torch.sigmoid(gate) + sgl quant` **或** `sigmoid_mul_inplace_triton(a, g) + sgl quant` | `sigmoid_mul_fp8_quant_fwd` |

### 4.3 跟 DSV4 不同点:**先做模型代码减法**

Qwen3.5 的模型代码 refactor 前散落着这种东西:

```python
# Qwen3NextDecoderLayer.forward(refactor 前,简化)
if self._fuse_input_norm_quant and hidden_states.dim() == 2:
    bf16_hs, fp8_hs, scale = fused_add_rmsnorm_fp8_quant_with_bf16_output(...)
    hidden_states = self.self_attn(bf16_hs, x_fp8=fp8_hs, x_scale=scale, ...)
elif self._fuse_input_norm_quant_linear and hidden_states.dim() == 2:
    ... 另一套 ...
else:
    hidden_states, residual = self.input_layernorm(hidden_states, residual)
    hidden_states = self.self_attn(hidden_states, ...)

if self._fuse_post_norm_quant and hidden_states.dim() == 2:
    ...
elif self._fuse_post_norm_quant_moe and hidden_states.dim() == 2:
    ...
else:
    ...
```

70+ 行嵌套 if/elif/else。每个分支都是"如果用 fused kernel 走这边,否则走那边"。

**refactor 之后,变成 7 行**:

```python
def forward(self, hidden_states, residual, fmha_impl, kv_cache=None, ...):
    hidden_states, residual = self.input_layernorm(hidden_states, residual)
    hidden_states = self.self_attn(hidden_states=hidden_states, ...)
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual
```

所有 fused 决策都搬到 FX pass 里。这就是"DSV4 风格"的精髓。

我们一共改了 5 个文件:`causal_attention.py`、`dense_mlp.py`、`qwen3_next.py`(两处类)、`generic_moe.py`。

### 4.4 跟"全局开关"配合

Qwen3.5 之前有个全局开关 `HWKernelConfig.enable_fuse_kernels`(默认 True),`=False` 时所有 eager fused 路径都被绕过,走 baseline。这是用来做"精度对比调试"的。

refactor 后这个开关仍然保留语义:

```python
# 简化的安装钩子
if env("QWEN35_GRAPHFX_FUSION") == "1":
    if hw_kernel_config.enable_fuse_kernels:    ← 两个开关都打开才装
        install_graphfx(py_model)
    else:
        log("用户要 pure baseline,跳过 GraphFX")
```

这样:
- 生产:`enable_fuse_kernels=True` + `QWEN35_GRAPHFX_FUSION=1` → 走 fused 路径
- 调试:`enable_fuse_kernels=False` → 走 pure baseline,bit-exact 可调

---

## 第五部分:对比

| 维度 | DSV4 | Qwen3.5(refactor 后) |
|---|---|---|
| 编译边界(`torch.compile` 包的函数) | `DeepSeekV4Model.forward` | `Qwen3NextModel.forward` 或 `GenericMoeModel.forward` |
| 主开���环境变量 | `DSV4_GRAPHFX_FUSION` | `QWEN35_GRAPHFX_FUSION` + `HWKernelConfig.enable_fuse_kernels` 双开关 |
| FX pass 数量 | 4 | 4 |
| 跨图 provenance | 1 套(rmsnorm + quant) | 1 套(add+rmsnorm + quant) |
| 精算模式 | 有 | 有 |
| 关掉的"非计算"辅助函数 | ~30 个 | ~25 个 |
| **eager 模型代码是否需要 strip** | **不需要**(本来就干净) | **需要**(本次 refactor 删了 5 个文件里的 if/else 分支) |
| GraphFX 关掉时 eager 走什么 | 完整模型(DSV4 模型本身就需要 fuse 才能跑动) | 纯 unfused baseline(慢但可调试) |

**整体形态高度一致**,主要差异:
- DSV4 起点已经干净,GraphFX 直接接入
- Qwen3.5 起点是双路径,先剥光再接入

---

## 第六部分:这样做的好处

### 6.1 模型代码与融合策略**完全解耦**

| | 旧做法(eager 直接调 fused) | 新做法(GraphFX) |
|---|---|---|
| 加一个新 fused kernel | 改 5+ 个 model 文件,塞 if/else | 只写一个 FX pass 文件 |
| 删一个 fused kernel | 反向再改一遍 5 个文件 | 删一个 FX pass 文件 |
| 形状/dtype 校验逻辑 | 散落在每个 callsite | 集中在 pass 的契约检查里 |
| 看模型 forward 逻辑 | 70 行嵌套 if/else | 7 行干净 baseline |

### 6.2 单个 pass 可以 **独立灰度**

每条 pass 都有自己的 env 开关:

```
QWEN35_FUSED_ADD_RMSNORM_FP8_QUANT=1     ← 单独开/关 add+rmsnorm fusion
QWEN35_FUSED_SILU_AND_MUL_FP8_QUANT=1    ← 单独开/关 silu fusion
...
```

某条 pass 出问题,生产线上 unset 这个变量就关掉了,不需要发新版本。

### 6.3 **集中的可观测性**

- `QWEN35_GRAPHFX_MISS_LOG=1`:每条 pass 没命中是因为什么(`quant_contract_mismatch`、`unsupported_fixed_hidden_dim`...)。eager 散落的 `if self._fuse_*:` 永远不会告诉你它为什么没进 if
- `QWEN35_GRAPHFX_COMPILE_STATS=1`:每张 FX 图被 trace 了几次、签名是什么 → 排查"为什么这个 shape 一直在重 trace"
- atexit 自动 dump 汇总,独立日志文件

### 6.4 跨图融合 —— eager 写不出来的优化

```
Layer 里的代码:
    hidden = rmsnorm(x, weight)      ← 在 layer.forward 里
    output = fp8_linear(hidden)       ← linear.forward 内部才有 quant
```

eager 模式下想合并这两步,你得**破坏 Linear 模块的封装**(把它内部的 quant 拎出来跟 rmsnorm 摆在一起)。GraphFX 用 provenance 机制可以做到 —— Linear 内部的 quant 看到 `hidden`,查表知道它是某 RMSNorm 的输出,跑融合 kernel 直接返回 fp8。模型代码完全不动。

### 6.5 跟 CUDA Graph、TP、DeepGEMM 这些都兼容

- FX 改写只增减节点,不动 host 同步 → CUDA Graph capture 不受影响
- 难 trace 的地方(metadata、KV write)自动 graph break,留 eager 跑,**两边都不阻塞**
- 默认开 `FALLBACK_UNFUSED=1`,任何 trace 错误回退 eager,生产路径不会因为 GraphFX 炸掉

---

## 第七部分:`torch.compile` 在这套架构里到底起什么作用

**老实说,`torch.compile` 不带来直接的性能提升**。性能完全来自:① 手写的 fused Triton/CUDA kernel,② 我们 4 条 FX pass 把这些 kernel 替换进图里。

`torch.compile` 在这里贡献的是 **基础设施**,具体四点:

### 7.1 比静态 trace 工具能 cover 的场景多

PyTorch 还有别的"把代码变成图"的工具:

| 工具 | 局限 |
|---|---|
| `torch.fx.symbolic_trace(fn)` | 遇到 Python `if`、`.item()`、读普通 Python 对象属性 → 直接挂 |
| `torch.jit.trace(fn, example_inputs)` | 同上,而且只支持 tensor 输入 |
| `torch._dynamo.export(fn)` | 要求 `fullgraph=True`,任何 graph break 都报错 |

我们的模型 forward 里到处是 metadata 读取、控制流、pybind 调用 —— 上面三个工具全都用不了。**`torch.compile(..., fullgraph=False, dynamic=True)` 是唯一现实选择**:能 trace 的部分给 backend,不能的自动 graph break 走 eager。

### 7.2 Dynamo cache 让多次调用复用编译结果

第二次同 shape signature 进来,Dynamo 查 cache → 命中就直接跑改写后的 GraphModule,**不重 trace、不重跑 pass**。

我们设的 `cache_size_limit=128`,配合 `mark_dynamic` 标记动态轴(token 数、batch),可以让"T=1 到 T=8192 都共享同一份编译结果",避免每个 batch size 都重新 trace 一次。

### 7.3 `backend=` 是 PyTorch **官方扩展点**

```python
torch.compile(fn, backend=my_backend)
```

`my_backend` 是个普通 Python 函数,签名是 `(gm: GraphModule, example_inputs) -> Callable`。返回的 callable 给 PyTorch 当正常函数跑。

这个 API 是稳定的扩展点,**不需要 hack 任何 PyTorch 内部细节**,torch 升版本不会立刻挂。

### 7.4 跟 `torch.library.custom_op` / `allow_in_graph` 配套

pybind 暴露的 mutating op(如 `rtp_llm_ops.fused_add_rmsnorm`),直接被 Dynamo trace 会在 FakeTensor 阶段炸(因为它会读 raw 指针)。`torch.library.custom_op(..., mutates_args=("hidden", "residual"))` + `register_fake` 让 Dynamo 把它当成一个"不透明的、会修改前两个参数的节点",产生稳定的 target 名字供 FX pass 匹配。

这套 API **只有跟 Dynamo / torch.compile 配套才有意义** —— 它是为这套机制设计的。

---

## 第八部分:局限,不该过度神化的地方

1. **首次 trace 成本高**:第一次调用要跑 Dynamo + 4 条 pass + `gm.recompile()`,几百 ms。生产服务要先用各种 shape 暖一遍
2. **silent miss**:pattern 不命中通常没有任何用户可见反馈。要养成"开发新功能时打开 `MISS_LOG=1` 看 reason"的习惯
3. **`enable_fuse_kernels=False` 路径变慢**:本次 refactor 后,模型代码只剩 unfused baseline,关掉 GraphFX = 零融合。这个组合只适合做"精度调试 baseline",生产不能这么开
4. **没有 FX pass 覆盖的 fuse kernel 还是 eager 在管**:MLA 的 strided rmsnorm、Indexer 的 logits gate、flashmla 的 BMM —— 这三块本次没动 pass,模型代码里仍是 always-on 的 eager fast-path
5. **单元测试只验证 FX 节点替换**:不验证 fused kernel 输出与 unfused 数值等价。精度等价靠现有 smoke / perf gate 兜底

---

## 第九部分:一句话总结

> **`torch.compile` 提供"把模型 forward 自动抓成 FX 图"的能力,Dynamo 是抓图的执行引擎,FX Graph 是抓出来的中间产物。**
>
> **GraphFX 这套架构在这个基础上做的事是:用 FX pass 自动识别"该融合的 op 对",替换成 fused kernel 调用,把"用不用 fused"这个决定从模型代码里彻底搬走。**
>
> **DSV4 的模型代码天生就是干净的,直接走 GraphFX。Qwen3.5 之前散落着 50+ 处 fused/unfused 双路径,本次 refactor 把它们全删了,让模型 forward 回到 7 行 baseline,fused 决策完全交给 GraphFX。**

# GLM5 Mega MoE 可疑问题汇总

日期：2026-05-23

范围：静态代码对比。本文只汇总代码上的可疑点，没有修改代码，也没有做 GPU / distributed 运行验证。

## 总结

GLM5 和 DeepSeekV4 不走同一套 Python 侧 Mega MoE 实现。两者最终都会调用 `deep_gemm.fp8_fp4_mega_moe`，但上层封装、策略选择、buffer 管理、token budget、weight layout 处理都不同。

- GLM5：`glm_5` 注册到 `DeepSeekV2`，再创建 `GenericMoeModel`；当 `moe_config.moe_strategy == "mega_moe"` 时，`GenericMoeLayer` 直接使用 `MegaMoeFusedWrapper`，再进入 `GLM5MegaMoE`。
  - 入口：[deepseek_v2.py:989](../rtp_llm/models/deepseek_v2.py#L989)
  - Python model：[deepseek_v2.py:562](../rtp_llm/models/deepseek_v2.py#L562)
  - Mega wrapper 选择：[generic_moe.py:87](../rtp_llm/models_py/model_desc/generic_moe.py#L87)
  - DeepGEMM 调用：[mega_moe.py:492](../rtp_llm/models_py/modules/glm5_mega_moe/mega_moe.py#L492)
- DeepSeekV4：`deepseek_v4` 注册到 `DeepSeekV4`，创建 `DeepSeekV4Model`，每层 `MoE` 通过 `select_strategy` 选择策略；EP 场景默认要求 `MegaMoEStrategy`。
  - 入口：[deepseek_v4.py:820](../rtp_llm/models/deepseek_v4.py#L820)
  - Python model：[deepseek_v4.py:510](../rtp_llm/models/deepseek_v4.py#L510)
  - MoE strategy 选择：[moe_layer.py:246](../rtp_llm/models_py/modules/dsv4/moe/moe_layer.py#L246)
  - EP 强制 Mega：[base.py:227](../rtp_llm/models_py/modules/dsv4/moe/strategies/base.py#L227)
  - DeepGEMM 调用：[mega.py:404](../rtp_llm/models_py/modules/dsv4/moe/strategies/mega.py#L404)

## 主要实现差异

| 维度 | GLM5 Mega MoE | DeepSeekV4 Mega MoE |
| --- | --- | --- |
| Python 实现 | `rtp_llm/models_py/modules/glm5_mega_moe/*` | `rtp_llm/models_py/modules/dsv4/moe/strategies/mega.py` |
| 策略选择 | `moe_strategy == "mega_moe"` 时直接创建 wrapper | `select_strategy`，EP>1 默认要求 Mega，否则抛错 |
| 权重组织 | 复用通用 `W.moe_w1/W.moe_w2`，`moe_w1` 内部拼 gate/up | V4 独立加载 `w1/w2/w3`，strategy 内显式拼 `w1/w3` |
| enable / capability gate | 定义了 `mega_moe_enabled()`，但当前 GLM5 核心路径没有调用 | `MegaMoEStrategy.can_handle()` 检查 EP、SM100、dist、DeepGEMM 可用性 |
| output buffer | 按 `cfg.max_tokens_per_rank` 分配 | 使用 `_mega_output_capacity()` 覆盖 DeepGEMM 内部对齐容量 |
| collective 同步 | pack 后直接调用 kernel | kernel 前有可选 barrier 和 warmup rank sync |
| token budget | `max(8192, max_seq_len, ll_num_max_token)` | CP-aware，并有 chunked MoE / decode 角色约束 |
| input packer 校验 | 只校验 CUDA、维度、`D % 128` | 额外校验 `weights/indices` shape、dtype、`out_sf` shape |

## 可疑问题

### 1. 高风险：GLM5 `moe_w1` 的 gate/up 顺序可能反了

可疑点：GLM5 权重描述里 `W.moe_w1` 的 ckpt 顺序是 `up_proj` 再 `gate_proj`，但通用拼接函数和 Mega wrapper 都假定前半是 gate、后半是 up。

代码依据：

- GLM5 / DeepSeekV2 权重描述把 `up_proj` 放在前面，`gate_proj` 放在后面：[deepseek_v2.py:375-L389](../rtp_llm/models/deepseek_v2.py#L375-L389)
- `stack_moe_w1()` 把输入 list 前半命名为 `gate`，后半命名为 `up`，最终拼成 `[gate || up]`：[model_weight.py:427-L435](../rtp_llm/utils/model_weight.py#L427-L435)
- loader 的 GPU preallocate 快路径按 `ckpt_weights` 顺序写入 first half / second half，没有额外交换：[ffn_weight.py:491-L503](../rtp_llm/model_loader/ffn_weight.py#L491-L503)
- loader 注释也写明 `stack_moe_w1: gate + up`：[ffn_weight.py:478-L480](../rtp_llm/model_loader/ffn_weight.py#L478-L480)
- GLM5 wrapper 把 `w1[:, :half_n]` 当作 `w1_gate`，`w1[:, half_n:]` 当作 `w1_up`：[fused_moe_wrapper.py:96-L107](../rtp_llm/models_py/modules/glm5_mega_moe/fused_moe_wrapper.py#L96-L107)、[fused_moe_wrapper.py:116-L127](../rtp_llm/models_py/modules/glm5_mega_moe/fused_moe_wrapper.py#L116-L127)
- GLM5 FP4 入口注释也假定 `w1_w` 是 `gate+up`：[mega_moe.py:123-L130](../rtp_llm/models_py/modules/glm5_mega_moe/mega_moe.py#L123-L130)
- DSV4 的本地参考路径是 `silu(gate) * up`，交换 gate/up 不等价：[local_loop.py:414-L426](../rtp_llm/models_py/modules/dsv4/moe/strategies/local_loop.py#L414-L426)

影响：如果没有隐藏的 checkpoint 命名约定或 loader 重排，GLM5 Mega MoE 会把 `up_proj` 当 gate，把 `gate_proj` 当 up，实际计算会变成近似 `silu(up) * gate`。SwiGLU 里乘法本身可交换，但 SiLU 只作用在 gate 上，所以该交换会导致语义错误。

建议优先验证：用一个小维度确定性权重构造语义测试，比较 `silu(gate) * up` 与当前 GLM5 Mega 输出；或者直接用真实 GLM5 layer weights 对比非 Mega / torch reference。

### 2. 高风险：GLM5 output buffer 没覆盖 DeepGEMM 内部 token 对齐容量

可疑点：DSV4 明确认为 output rows 需要覆盖 DeepGEMM 内部对齐后的 token capacity，并用 `_mega_output_capacity()` 处理；GLM5 只按 `cfg.max_tokens_per_rank` 分配 `_mega_y`。

代码依据：

- DSV4 `_mega_output_capacity()` 取 `max(requested_capacity, buf.num_max_tokens_per_rank)`：[mega.py:45-L51](../rtp_llm/models_py/modules/dsv4/moe/strategies/mega.py#L45-L51)
- DSV4 注释说明 DeepGEMM 会把 `num_max_tokens_per_rank` 内部对齐到 token alignment：[mega.py:205-L209](../rtp_llm/models_py/modules/dsv4/moe/strategies/mega.py#L205-L209)
- DSV4 分配 `_mega_y` 时使用 `_mega_output_capacity()`：[mega.py:228-L235](../rtp_llm/models_py/modules/dsv4/moe/strategies/mega.py#L228-L235)
- GLM5 分配 `_mega_y` 只使用 `max(cfg.max_tokens_per_rank, 1)`：[mega_moe.py:354-L359](../rtp_llm/models_py/modules/glm5_mega_moe/mega_moe.py#L354-L359)

影响：如果 DeepGEMM kernel 或 symmetric buffer 按对齐后的 `buf.num_max_tokens_per_rank` 访问 output buffer，GLM5 的 `_mega_y` 可能行数不足，轻则运行时报错，重则越界写或 silent corruption。DSV4 已经专门补了这个保护，GLM5 缺失值得优先排查。

### 3. 中高风险：GLM5 定义了 `mega_moe_enabled()`，但核心路径没有使用

可疑点：GLM5 的 `mega_buf.py` 有完整的可用性检查和 `GLM5_USE_MEGA_MOE=0` 开关，但 `GenericMoeLayer -> MegaMoeFusedWrapper -> GLM5MegaMoE` 路径没有调用它。

代码依据：

- `mega_moe_enabled()` 会检查 `GLM5_USE_MEGA_MOE=0`，并依赖 DeepGEMM、distributed、world size、CUDA、SM100：[mega_buf.py:144-L186](../rtp_llm/models_py/modules/glm5_mega_moe/mega_buf.py#L144-L186)
- `GLM5MegaMoE` 只 import 了 `mega_moe_enabled`，未实际调用：[mega_moe.py:33-L37](../rtp_llm/models_py/modules/glm5_mega_moe/mega_moe.py#L33-L37)
- GLM5 只要 `moe_strategy == "mega_moe"` 就创建 Mega wrapper：[generic_moe.py:87-L94](../rtp_llm/models_py/model_desc/generic_moe.py#L87-L94)
- DSV4 的 `MegaMoEStrategy.can_handle()` 会检查 `ep_size > 1` 和 `_mega_moe_enabled()`：[mega.py:105-L109](../rtp_llm/models_py/modules/dsv4/moe/strategies/mega.py#L105-L109)
- DSV4 EP 场景如果 Mega 不可用会早期抛出带原因的错误：[base.py:227-L244](../rtp_llm/models_py/modules/dsv4/moe/strategies/base.py#L227-L244)

影响：`GLM5_USE_MEGA_MOE=0` 可能无法禁用已选中的 `mega_moe`；EP=1、distributed 未初始化、非 SM100、DeepGEMM 缺失等场景会更晚失败，且错误信息不如 DSV4 明确。

### 4. 中风险：GLM5 token budget / CP / chunked MoE 处理弱于 DSV4

可疑点：GLM5 wrapper 用 `max(8192, max_seq_len, ll_num_max_token)` 估算 `max_tokens_per_rank`，forward 里超过就直接报错；DSV4 对 CP、chunked MoE、decode 角色有更细的预算和执行约束。

代码依据：

- GLM5 budget 来源：[fused_moe_wrapper.py:56-L72](../rtp_llm/models_py/modules/glm5_mega_moe/fused_moe_wrapper.py#L56-L72)
- GLM5 forward 超过 `buf.num_max_tokens_per_rank` 直接报错：[mega_moe.py:480-L486](../rtp_llm/models_py/modules/glm5_mega_moe/mega_moe.py#L480-L486)
- DSV4 CP-aware buffer sizing：[deepseek_v4_model.py:319-L349](../rtp_llm/models_py/model_desc/deepseek_v4_model.py#L319-L349)
- DSV4 chunk / decode 约束：[moe_layer.py:254-L266](../rtp_llm/models_py/modules/dsv4/moe/moe_layer.py#L254-L266)

影响：长上下文或 CP 场景下，GLM5 可能过度分配 symmetric buffer 导致 OOM；也可能遇到实际 tokens 超预算时没有 chunk fallback，只能运行时报错。

### 5. 中风险：GLM5 缺少 DSV4 kernel 前的 collective 同步保护

可疑点：DSV4 在调用 `fp8_fp4_mega_moe` 前专门处理 rank 同步，并用注释说明该 kernel 是 peer-symmetric NVLink collective；GLM5 pack 后直接调用 kernel。

代码依据：

- DSV4 注释强调所有 rank 都必须一起进入 kernel，包括 `T == 0` 的 rank，否则可能 silent wrong output 或 NVLink barrier timeout：[mega.py:375-L395](../rtp_llm/models_py/modules/dsv4/moe/strategies/mega.py#L375-L395)
- DSV4 kernel 前执行 `_maybe_pre_kernel_barrier()` 和 `sync_cuda_graph_warmup_ranks()`：[mega.py:396-L401](../rtp_llm/models_py/modules/dsv4/moe/strategies/mega.py#L396-L401)
- GLM5 pack 后直接 `fp8_fp4_mega_moe`：[mega_moe.py:488-L503](../rtp_llm/models_py/modules/glm5_mega_moe/mega_moe.py#L488-L503)

影响：如果 GLM5 的 EP/CP/CUDA graph warmup 场景也会出现 rank 间阶段错位，可能复现 DSV4 注释中的超时、SIGTRAP、`CUDA_ERROR_LAUNCH_FAILED` 或 silent wrong output。

### 6. 中低风险：GLM5 没有从 model config 传递 `swiglu_limit`

可疑点：GLM5 `GLM5MegaMoE.from_params()` 默认 `swiglu_limit=0.0`，wrapper 创建时没有传配置值；DSV4 会从 config 读取并一路下发。

代码依据：

- GLM5 默认值：[mega_moe.py:91-L114](../rtp_llm/models_py/modules/glm5_mega_moe/mega_moe.py#L91-L114)
- GLM5 wrapper 创建时没有传 `swiglu_limit`：[fused_moe_wrapper.py:63-L72](../rtp_llm/models_py/modules/glm5_mega_moe/fused_moe_wrapper.py#L63-L72)
- DeepSeekV4 从 config 读取 `swiglu_limit`：[deepseek_v4.py:639-L644](../rtp_llm/models/deepseek_v4.py#L639-L644)
- DeepSeekV4 下发到 args / cfg：[deepseek_v4_model.py:210-L223](../rtp_llm/models_py/model_desc/deepseek_v4_model.py#L210-L223)、[moe_layer.py:232-L244](../rtp_llm/models_py/modules/dsv4/moe/moe_layer.py#L232-L244)

影响：如果 GLM5 checkpoint / config 也期望 SwiGLU clamp，Mega 路径会少做 clamp。若 GLM5 确认不需要 clamp，则这点可以降级为实现差异。

### 7. 中低风险：GLM5 fused input packer 校验比 DSV4 少

可疑点：GLM5 packer 只校验 CUDA、`x.dim()` 和 `D % 128`；DSV4 额外校验 router weights / indices shape、dtype 和 `out_sf` shape。

代码依据：

- GLM5 校验范围：[input_packer_triton.py:130-L143](../rtp_llm/models_py/modules/glm5_mega_moe/input_packer_triton.py#L130-L143)
- DSV4 额外校验：[_mega_input_pack_triton.py:180-L205](../rtp_llm/models_py/modules/dsv4/moe/_mega_input_pack_triton.py#L180-L205)

影响：shape 或 dtype 不符合预期时，GLM5 更容易进入 Triton kernel 后才失败，或者产生更难定位的错误。这里不像 gate/up 顺序那样直接指向精度 bug，但会增加线上问题排查成本。

### 8. 中风险：现有 GLM5 Mega MoE 测试不足以捕获语义错误

可疑点：GLM5 Mega MoE 单测主要检查 shape、dtype、NaN；FP8->FP4 测试对比的是 GLM5 自身转换逻辑，不能证明 gate/up 语义或 Mega 输出正确。

代码依据：

- GLM5 BF16 Mega 测试只断言 shape、dtype、NaN：[test_mega_moe.py:115-L123](../rtp_llm/models_py/modules/glm5_mega_moe/test_mega_moe.py#L115-L123)
- FP8->FP4 测试的 reference 注释写明匹配 `GLM5MegaMoE.setup_weights_from_fp8` 自身逻辑：[test_mega_moe_fp8_to_fp4.py:35-L37](../rtp_llm/model_loader/test/test_mega_moe_fp8_to_fp4.py#L35-L37)
- 同一测试里也假定 `stack_moe_w1` 产物是 `[gate || up]`：[test_mega_moe_fp8_to_fp4.py:123-L130](../rtp_llm/model_loader/test/test_mega_moe_fp8_to_fp4.py#L123-L130)

影响：如果 GLM5 `moe_w1` 顺序错、`swiglu_limit` 漏传、output buffer 对齐存在问题，当前测试很可能不会提前暴露。

## 建议验证顺序

1. 先验证 gate/up 顺序。构造小维度确定性权重，分别计算 `silu(gate) * up` 和 `silu(up) * gate`，对比 GLM5 Mega 输出；最好再用真实 GLM5 layer weights 对比非 Mega 路径或 torch reference。
2. 对齐 DSV4 的 `_mega_output_capacity()` 思路，加一个只测 buffer rows 的单测，确认 `buf.num_max_tokens_per_rank` 大于请求值时 GLM5 当前 `_mega_y` 是否不足。
3. 验证 `GLM5_USE_MEGA_MOE=0`、EP=1、distributed 未初始化、非 SM100 等场景的行为是否符合预期。
4. 明确 GLM5 是否需要 `swiglu_limit`。如果 config 中可能出现该字段，应补一条配置传递测试。
5. 补一个语义级 Mega MoE output test，而不是只做 shape / dtype / NaN 检查。

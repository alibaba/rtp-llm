# GLM5 thinking budget 强制关闭验证记录

## 验证目的

验证以下假设：当 reasoning 因 thinking budget 被提前截断后，将原始 prompt、已生成的 reasoning 和 `</think>` 重新作为输入，并关闭 thinking，模型是否会直接生成最终答案。

该实验用于判断异常 content 是 RTP-LLM 的 thinking 状态没有关闭，还是模型在语义上仍沿着未完成的 reasoning 继续生成。

## 验证环境与方式

- 日期：2026-09-19
- 分支：`feat/glm5_cu13_rebase`
- 模型：`/data4/GLM-5.2-FP8`
- Smoke target：`//internal_source/rtp_llm/test/smoke:mla_mtp_mega_moe_cudagraph_pd_full_ckpt_page_rr_cp4_bpk8`
- PD 模式：8 GPU，MTP sampling step 为 5
- 环境：Prefill 和 Decode 均按 smoke target 的服务配置启动
- 请求接口：`/__dash_sc_grpc__`
- 测试执行：

```bash
bazelisk test \
  //internal_source/rtp_llm/test/smoke:mla_mtp_mega_moe_cudagraph_pd_full_ckpt_page_rr_cp4_bpk8 \
  --config=cuda13 \
  --test_output=all \
  --test_timeout=7200 \
  --nocache_test_results
```

测试通过，约 947 秒完成；包含服务启动和退出的总耗时约 1002 秒。实际请求从 `00:10:08.521` 到 `00:10:58.316`，约 49.8 秒。

## 输入构造

原始请求 prompt token IDs 为：

```text
[154822,154824,154826,25062,287,29905,371,25,7487,154827,99032,109611,105087,220,99457,98845,103767,220,154828,154841]
```

待拼入的短 reasoning 及结束标记为：

```text
[154841,16,13,3070,99158,103424,5122,1019,262,154842]
```

原始 prompt 已以 `154841`（think start）结尾，因此最终输入只保留一个 think start。构造后的 29 个输入 IDs 为：

```text
[154822,154824,154826,25062,287,29905,371,25,7487,154827,99032,109611,105087,220,99457,98845,103767,220,154828,154841,16,13,3070,99158,103424,5122,1019,262,154842]
```

其中 `154842` 是 `</think>`。请求明确设置：

```json
{
  "enable_thinking": false,
  "generate_config": {
    "max_new_tokens": 500,
    "num_return_sequences": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "repetition_penalty": 1,
    "temperature": 1,
    "top_k": 0,
    "top_p": 0.95,
    "min_new_tokens": 0,
    "return_logprobs": false,
    "top_logprobs": 0,
    "random_seed": 42
  }
}
```

请求未设置 `max_new_think_tokens`，并检查响应 parameters 中不得出现 `generate_think_token_num`，以确认引擎侧 thinking 已关闭。完整请求保存在：

`artifacts/disable_think_prefilled_reasoning_experiment/request_fixture.json`

## 实际输出

响应统计：

| 项目 | 结果 |
| --- | ---: |
| prompt token 数 | 29 |
| generated token 数 | 500 |
| reasoning token 数 | 0 |
| content token 数 | 500 |
| finish reason | length（内部值 1） |
| `generate_think_token_num` | 不存在 |

`reasoning_content` 为空，但 `content` 明显继续了未完成的分析过程：

```text
    *   **主题：** Spring (春天)
    *   **长度：** 大约100个字符 (100字)
    *   **格式：** 散文/作文 (作文)

2. **头脑风暴春天的意象：**
    *   融化的冰雪 (冰雪融化)
    *   发芽的柳树 (柳树发芽)
    *   盛开的花朵 (百花齐放 / 迎春花)
    *   温暖的微风 (春风)
    *   燕子归来 (燕子归来)
    *   整体感觉：充满活力、希望、绿色 (生机勃勃, 充满希望)

3. **起草 - 尝试 1（侧重于意象）：**
    春天来了，冰雪融化，万物复苏。春风轻轻吹过，像妈妈的手抚摸着大地……
    *字数检查（心算）：* 春(1)天(2)来(3)了(4)，冰(5)雪(6)融(7)化(8)……
```

随后仍持续输出逐字计数内容，直到耗尽 500 个 token。完整响应文本、`generated_ids`、`content_ids`、parameters 和 aux info 保存在：

`artifacts/disable_think_prefilled_reasoning_experiment/actual.json`

## 结论

1. 引擎确实关闭了 thinking：`reasoning_content` 为空，响应没有 `generate_think_token_num`，全部输出均被归入 content。
2. 模型并没有直接转入最终答案，而是继续完成 reasoning。`enable_thinking=false` 只改变引擎的 token 约束与输出归类，不能消除输入 KV 中由未完成 reasoning 形成的语义状态。
3. 单独补 `</think>` 不能可靠处理很小 thinking budget。预算在一个未完成的分析步骤中截断时，模型仍可能在标签之后继续分析，只是这些 token 会被暴露为 content。
4. 因此，“拼接原始 prompt、全部 reasoning、`</think>` 后重新 prefill，并关闭 thinking”的方案既有重复 prefill 的性能损失，也不能保证输出正确。
5. 最终修复不能保留残缺 reasoning。参考内源 DashLLM 的 GLM 路径，第二阶段应从原始 prompt 加空 `<think></think>` 重新开始；这样仍有第二次调度/prefill，但不会重复计算已生成的 reasoning。

## 内源 vLLM 与 DashLLM 的实现现状

### 版本边界

本次检查的内源 vLLM 提交为：

```text
94e8ae5d37184500d75a2f2803f742a8f297c1fc
[vllmgen] ppu kvs support
vLLM version: 0.11.1
```

该提交本身只改了 11 个 PPU 890P 的 vllmgen 配置文件，没有修改 thinking、采样器、reasoning parser 或请求协议。该版本的 vLLM 源码中也不存在 `thinking_budget`、`thinking_token_budget`、`max_new_think_tokens` 或引擎内强制 `</think>` 的实现。

真正的 thinking budget 控制在独立的 DashLLM 适配层中。vLLM 的构建脚本 `ci-tools/setup_env.sh:104-107` 会另外 clone `dashscope/dashllm` 并以 editable 方式安装。因此，只提供 vLLM commit 不能唯一确定镜像中的 DashLLM 版本；要做到镜像级精确复现，还需要该镜像内的 DashLLM commit 或 wheel 版本。本节静态分析使用本机已有的 DashLLM checkout：

```text
/data0/yangchengjun.ycj/work/RTP-LLM/dashllm
commit b74f3d9c49d17b4464a3b945c2c05c140322343a
```

### 参数进入路径

DashLLM frontend 在 `dashllm/core/frontend/processor.py:1036-1042` 中完成映射：

```text
thinking_budget
  -> thinking_token_budget
  -> max_new_think_tokens
```

`enable_thinking=false` 时会直接把 `max_new_think_tokens` 设为 0。vLLM backend 在 `dashllm/core/backend/_backend_vllm.py:1238-1242` 中把它临时写成 `max_think_tokens`，随后由 DashLLM 的 `_LLMBackend4Think` 包装器取走。也就是说，该预算没有进入 vLLM 0.11.1 的原生 SamplingParams 和 sampler。

### 达到 budget 后的两阶段处理

第一阶段仍是正常 reasoning decode。`dashllm/think/core.py:163-189` 累加实际输出 token；达到预算后会：

1. 把当前批次裁到预算边界。
2. 标记 `aborted_by_length=true`。
3. 在对外 token 流末尾补齐 `</think>`。
4. 停止第一阶段 engine request。
5. 将本次响应暂时保持为 `streaming`，开始第二阶段 content request。

第二阶段不是续用第一阶段请求，而是一个新 request：

- engine-side 路径使用 `{request_uuid}-2`，见 `dashllm/core/backend/engine/_think.py:715-735`；
- NEW-PD client policy 使用 `{request_id}-2` 发起一轮新的 `transport.generate()`，见 `dashllm/client/policy.py:300-357`；
- NEW-PD 会异步取消第一阶段 decode stream，避免第一阶段在第二阶段期间继续占用 decode slot。

### GLM 与 Qwen 的关键差异

DashLLM 并非对所有模型都把 reasoning 拼回第二阶段。`dashllm/core/backend/engine/_think.py:186-197` 只对 Qwen3、Qwen3-MoE、Qwen3-VL 和 MiMo v2 设置 `phase2_feed_reasoning=true`。

GLM 不在这个集合中。对 `glm_moe_dsa`，DashLLM 在 `dashllm/core/backend/engine/_think.py:79-81` 定义的空 thinking 块是：

```text
<think></think>
```

`dashllm/think/core.py:221-225` 和 `dashllm/client/policy.py:375-415` 的实际构造规则为：

```text
GLM phase 2 input = 原始 prompt（移除第一阶段附加的 think BOS）+ <think></think>
```

第一阶段被截断的 reasoning 不会进入第二阶段 prompt。第二阶段同时移除 `max_new_think_tokens` 和 `enable_thinking`，从空 thinking 块后的 content 状态重新生成答案。

这正是内源 vLLM + DashLLM 在 `thinking_budget=1` 时仍能得到正常 content 的核心原因：它没有要求模型沿着一个只生成了 1 个 token 的残缺 reasoning 继续作答，而是抛弃这段 reasoning，从原始问题的“无 thinking”状态重新生成。

这也解释了它与本次 RTP 实验的差异：本次 RTP 实验把短 reasoning 和 `</think>` 一起保留在输入中，模型语义状态仍处于未完成的分析过程；DashLLM 的 GLM 路径不会保留这段语义状态。

## 两阶段方案的性能影响

### TTFT 会不会升高

会。需要先区分两个口径：

- **首个流式 token 的 TTFT**：第一阶段 reasoning token 可以正常流出，这个 TTFT 主要仍是第一次 prefill 加一次 decode。
- **首个 content token 的 TTFT**：必须等第一阶段生成到 thinking budget、终止旧请求、完成第二阶段调度和 prefill，再生成第一个 content token。这个值必然升高。

首个 content token 的近似耗时为：

```text
T_content_first
  = T_prefill_phase1
  + thinking_budget × reasoning_TPOT
  + T_abort_and_reschedule
  + T_prefill_phase2
  + T_first_content_decode
```

其中 `thinking_budget × reasoning_TPOT` 是用户选择该预算本来就需要付出的 reasoning 成本；两阶段方案额外引入的是停止/取消、重新排队、第二次 prefill，以及 PD 模式下的新一轮路由、握手和可能的 KV 传输。

### reasoning 很长时是否会全部重算

对 GLM，**不会重算被截断的长 reasoning**。假设原始 prompt 长度为 `P`，实际 reasoning 达到预算 `R`：

| 方案 | 第二阶段输入长度 | 是否重新处理 `R` 个 reasoning token |
| --- | ---: | --- |
| RTP 已否决的 reasoning 回灌方案 | 约 `P + R + transition + </think>` | 是 |
| 内源 DashLLM 的 GLM 路径 | 约 `P + <think></think>` | 否 |

因此，用户担心的“reasoning 已经很长，到预算后又把全部 reasoning prefill 一次”确实适用于当前 RTP 方案，但不适用于 DashLLM 的 GLM 路径。DashLLM 会丢弃 reasoning，只重新处理原始 prompt 和很短的空 thinking 后缀。

### 第二次 prefill 的实际成本

第二次 prefill 仍然存在，成本取决于 prefix cache 是否命中：

1. **同一 engine 且 prefix cache 命中**：第二阶段和第一阶段共享原始 prompt 前缀。vLLM 按完整 cache block 复用，通常只需重新计算最后一个未对齐 block 和 `<think></think>` 后缀，而不是整个 `P`。当前 DashLLM 的默认 CUDA block size 在该路径中是 128，配置也可以覆盖；`enable_prefix_caching` 会透传给 vLLM。
2. **prefix cache 未启用、条目被驱逐或第二阶段落到无法访问该缓存的实例**：需要重新 prefill 约 `P` 个 token，长 prompt 会明显拉高首个 content token 延迟。
3. **旧 PD engine-side 路径**：Decode 侧第二阶段被标成 `D_PREFILL_DECODE`，在 Decode 节点本地执行 prefill + decode，仍会产生一次本地 prefill；能复用多少取决于该节点的 prefix cache。
4. **NEW-PD external-router 路径**：第二阶段是完整的新 PD round，会重新经历路由、Prefill、握手、Decode 和 KV 协调。若只依赖单实例本地 prefix cache，第二阶段被路由到其他 Prefill 实例时命中没有保证；若配置了跨实例/远端缓存，才能稳定减少第二次 prefill。

因此这套方案的性能特征是：避免了 `O(R)` 的 reasoning 重放，但仍保留最多 `O(P)` 的第二次 prefill 和一次完整调度边界。对于短 prompt、长 reasoning，它比当前 RTP 方案明显便宜；对于超长 prompt 且缓存 miss，它仍可能显著增加首个 content token 延迟。

### 当前指标容易掩盖第二阶段成本

DashLLM 会保留第一阶段的 `prompt_cached_token_num`，并覆盖第二阶段返回的同名字段：

- engine-side：`dashllm/core/backend/engine/_think.py:753-756`
- NEW-PD policy：`dashllm/client/policy.py:336-350`

所以最终响应中的 `prompt_cached_token_num` 不能直接证明第二阶段 cache hit。engine-side 在 `dashllm/core/backend/engine/_think.py:740-743` 有一条 `think second request` 日志，它在覆盖前记录第二阶段原始 `prompt_cached_token_num`；性能验证应使用这条日志，并同时记录 phase 1 abort 到 phase 2 首 token 的间隔。

## 对 RTP 修复方案的参考结论

内源实现给出的可参考点是“GLM 被强制截断后不要保留残缺 reasoning”，而不是照搬当前 RTP 的 reasoning 回灌方案。

可选方向按性能从好到差排序如下：

1. **同一请求内切换到无 thinking 语义状态**：理想情况下无需新 prefill，但必须找到模型训练协议支持的 token 序列，并证明它在 budget=1、MTP=5、流式和 PD 下都稳定。仅强制 `</think>` 已被本次实验否定。
2. **参考 DashLLM 的 GLM 两阶段方案**：停止第一阶段，第二阶段使用 `原始 prompt + <think></think>`，不携带 reasoning。它能避开异常输出，也不会重放长 reasoning；代价是第二次请求和最多一次原始 prompt prefill。
3. **已否决的 RTP 方案**：`原始 prompt + 全部 reasoning + transition + </think>`。它既重复处理全部 reasoning，又不能保证模型转入 content，因此已从代码中删除。

如果采用方向 2，RTP 还需要增加可观测性：单独记录 phase 2 的输入长度、cached token 数、重新计算 token 数、重新排队耗时、prefill 耗时和首个 content token 时间，否则无法评估它在线上长 prompt 流量中的真实代价。


## RTP 最终修复

最终实现采用方向 2，并与内源 DashLLM 的 GLM 规则对齐：

1. 引擎在 thinking budget 强制生成结束标签时设置 `forced_think_end`，该标记通过 RPC aux info 传到 DashSc；自然生成的 `</think>` 不设置该标记。
2. DashSc 收到强制关闭标记后，保留第一阶段已经返回给用户的 reasoning 及强制 `</think>`，停止第一阶段请求。
3. 第二阶段输入固定为 `原始 prompt（移除尾部 think BOS）+ 空 <think></think>`，并关闭 thinking。被截断的 reasoning、同一 MTP packet 中结束标签后的 token、英文过渡指令都不进入第二阶段输入。
4. `RTP_LLM_MAX_TOKENS_EXCLUDE_THINKING=1` 时，第二阶段保留完整 content 预算；为 0 时，从总预算扣除第一阶段 reasoning 与结束标签。两阶段共享原请求的总超时。
5. 对外 usage 仍报告第一阶段原始 prompt；第二阶段内部 prompt 长度和 cache 命中应通过 phase 日志/指标单独观察，避免把第二次 prefill 成本隐藏在业务 usage 中。

这个实现把第二阶段 prefill 长度从 `P + R + transition` 降到 `P + empty_think`。因此 reasoning 越长，和旧方案相比节省越明显；它不会把已经生成的长 reasoning 再跑一遍。首个 content token 仍需等待第二次调度与 prefill，prefix cache miss 时最多重算原始 prompt，所以 content TTFT 仍可能升高，但不再随 reasoning 长度线性增加。

## 修复验证

- DashSc frontend 回归：通过。
- ThinkMode logits processor：通过。
- Reasoning + grammar logits processor：通过。
- Model RPC aux info 透传：通过。
- GenerateStream 新增强制关闭/自然关闭隔离用例：通过。
- GenerateStream 全量测试仍有该分支已有的 4 个顺序相关失败；本次新增用例隔离运行通过。
- 8 卡真实 GLM PD smoke 已按 Prefill/Decode `RTP_LLM_MAX_TOKENS_EXCLUDE_THINKING=1`、MTP=5、thinking budget=10、content budget=500 启动，但 Decode 在请求前的 FP8→FP4 权重在线转换阶段 OOM：GPU 3 申请 6.00 GiB 时仅剩 5.46 GiB。该次 smoke 没有进入请求阶段，因此不能作为模型输出验证结果；临时 smoke 配置已恢复。

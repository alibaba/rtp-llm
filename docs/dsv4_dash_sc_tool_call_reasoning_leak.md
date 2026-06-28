# DSV4 dash_sc Tool Call 泄漏到 Reasoning 的问题记录

## 背景

DeepSeek V4 在 thinking 模式下，有时不会先生成 `</think>`，而是直接开始生成 DSML tool-call 标记：

```text
<｜DSML｜tool_calls>
```

normal OpenAI frontend 走 `DeepseekV4Renderer`，renderer 会在 decoded text 层处理这个情况：一旦看到 `<｜DSML｜tool_calls>` 先于 `</think>` 出现，就把这个 DSML marker 当成隐式的 reasoning 结束边界，marker 之前仍然是 `reasoning_content`，marker 及其后面的内容交给 DSV4 tool detector 解析成 `tool_calls`。

`dash_sc` 链路不走 `DeepseekV4Renderer` 的输出解析。它收到的是 backend 流式生成的 token ids，并通过 `generate_think_token_num` 告诉下游 reasoning 在哪里结束。因此 dash_sc 需要在发 chunk 前自己识别 DSV4 tool-call marker，否则 DSML 标记会被当成 reasoning token 一起流出去，最后泄漏到 `reasoning_content`。

## 现象

同一模型权重、同一部署，只是请求链路不同：

- normal frontend：输出经过 `DeepseekV4Renderer._process_reasoning_and_tool_calls`，tool-call marker 不进入 `reasoning_content`。
- dash_sc frontend + xgrammar：输出没有经过 renderer 的文本级 reasoning/tool-call 拆分，tool-call marker 可能进入 reasoning。

核心差异不是模型，而是 frontend 侧的输出后处理不同。

```text
normal frontend
  generated token ids
      |
      v
  decode text delta
      |
      v
  DeepseekV4Renderer
      |
      +-- reasoning_content
      +-- content
      +-- tool_calls

dash_sc frontend
  generated token ids
      |
      v
  ModelStreamInfer response
      |
      v
  下游依赖 generate_think_token_num 切 reasoning/content
```

## 为什么不能按固定 token 序列匹配

一开始最容易想到的是匹配：

```text
tokenizer.encode("<｜DSML｜tool_calls>")
```

但这个方案不可靠。DSV4 tokenizer 的 BPE 切分和上下文有关，marker 文本稳定，但 marker 对应的 token 序列不稳定。

更准确地说，`<｜DSML｜tool_calls>` 不是一个可以直接拦截的单一 special token，而是一组 token 片段序列。常见 standalone 形态类似：

```text
[30, 128825, 72461, 4941, 12548, 32]
  |    |       |      |     |      |
  <  ｜DSML｜ tool   _c    alls   >
```

这组片段本身就不是语义完整的 protocol 边界；真正的 protocol 边界是这些片段 decode 后拼出的文本 marker。而 BPE 又会让片段边界和前后文合并，所以“tool-call tag 的 token 序列”不是一个稳定对象。

真实 tokenizer 中观察到的例子：

```text
<｜DSML｜tool_calls>        -> [30, 128825, 72461, 4941, 12548, 32]
<｜DSML｜tool_calls>\n      -> [30, 128825, 72461, 4941, 12548, 1018]
<｜DSML｜tool_calls>{       -> [30, 128825, 72461, 4941, 12548, 31923]
<｜DSML｜tool_calls><｜...  -> [30, 128825, 72461, 4941, 12548, 5451, ...]
 <｜DSML｜tool_calls>       -> [818, 128825, 72461, 4941, 12548, 32]
.<｜DSML｜tool_calls>       -> [32334, 128825, 72461, 4941, 12548, 32]
```

也就是说：

- 末尾的 `>` 可能和后一个字符合并，比如 `>\n`、`>{`、`><`。
- 开头的 `<` 可能和前一个字符合并，比如 `" <"`、`".<"`。
- 不能枚举一个固定 token list，也不能只匹配 standalone marker 的 encode 结果。

```text
单独 encode:

  <｜DSML｜tool_calls>
  |                    |
  稳定的是文本 marker     encode(...) 只是其中一种 token 序列

真实流式输出:

  previous text + <｜DSML｜tool_calls> + next text
                ^                    ^
                |                    |
                左侧可能合并          右侧可能合并

结论:

  protocol 边界是 decoded text 中的 <｜DSML｜tool_calls>，
  不是某个固定 token 序列。
```

## normal frontend 里的解决办法

normal frontend 的处理在：

```text
rtp_llm/openai/renderers/deepseekv4_renderer.py
```

关键路径是：

```text
DeepseekV4Renderer._process_reasoning_and_tool_calls
  -> DeepseekV4Renderer._extract_reasoning_content
     -> DeepseekV4Renderer._extract_streaming_reasoning_content
  -> DeepseekV4Renderer._extract_tool_calls_content
     -> DeepSeekV4Detector
```

streaming 时，renderer 不是按 token id 匹配 DSML，而是先把 token ids decode 成文本 delta，再用文本 buffer 处理三类边界：

```text
<think>
</think>
<｜DSML｜tool_calls>
```

处理逻辑可以概括为：

```text
decoded text delta
      |
      v
追加到 renderer/detector 的文本 buffer
      |
      +-- 如果 </think> 先出现:
      |     </think> 之前是 reasoning
      |     </think> 之后是 normal content/tool-call text
      |
      +-- 如果 <｜DSML｜tool_calls> 先出现:
      |     DSML 之前是 reasoning
      |     DSML marker 及之后的文本交给 tool detector
      |
      +-- 如果当前 chunk 只包含边界前缀:
            暂存在 buffer，等下一个 chunk 补齐
```

这个方案能抗 BPE merge，因为 split 发生在 decoded text 的字符边界上。比如某个 token decode 出 `" <"`，renderer 可以在文本层把空格留在普通文本里，把 `<｜DSML｜tool_calls>` 从 `<` 开始交给 tool detector。

打开 renderer debug 后，如果日志里出现下面的信号，说明 normal frontend 确实触发了“DSML marker 先于 `</think>`，把它当隐式 reasoning 结束”的逻辑：

```text
[DeepSeekV4RendererDebug] implicit_think_end_before_dsml
```

相关 debug 开关是环境变量：

```text
RTP_LLM_DSV4_RENDERER_DEBUG=1
```

## 探索过但不成立的方案

这些方案不可行，根因都和两个事实有关：

- DSML tool-call tag 是一组 token 片段序列，不是单 token 边界。
- BPE 会根据上下文把 tag 的首尾片段和相邻文本合并，导致 encode/decode 不对称。

### 只依赖 xgrammar 不行

xgrammar / `structural_tag` 能约束“接下来应该按什么 grammar 生成”，但它不能替代 frontend 的 reasoning/tool-call 拆分。

这个问题发生在 phase-1 thinking 输出已经开始流出的时候：模型在 reasoning 中先吐出了 `<｜DSML｜tool_calls>`，而 dash_sc 还没有把 reasoning 边界切出来。xgrammar 本身并不知道“这个 DSML marker 应该结束 reasoning”，也不会自动设置 `generate_think_token_num`。

如果强行把 tool-call grammar 加到 phase-1 thinking 上，会破坏自由推理；如果只在 phase-2 使用 grammar，又必须先正确发现 phase-1 的结束边界。这个边界不能靠固定 token 片段序列判断，因为 tag 的首尾 token 可能被 BPE 合并。

所以 xgrammar 是 phase-2 生成 tool-call 的约束工具，不是 reasoning 泄漏的根因修复。

更细地说，xgrammar 方案要分两种：

```text
可行:
  已经确认 reasoning 结束
      |
      v
  phase-2 从干净边界开始
      |
      v
  使用 structural_tag / xgrammar 约束 tool-call 格式

不可行:
  phase-1 reasoning 自由生成中
      |
      v
  试图靠 xgrammar 判断 <｜DSML｜tool_calls> 是否出现
      |
      v
  再由 xgrammar 触发 reasoning 结束
```

第一种可行，因为 xgrammar 接管的是“从某个明确位置开始，后续文本必须满足 tool-call grammar”。这正是当前 phase-2 的用途。

第二种不适合作为主方案，原因是：

- reasoning 阶段不能全程挂 tool-call grammar，否则会限制正常思考内容。
- 如果只想让 grammar 监听 DSML marker，本质上仍然需要识别 `<｜DSML｜tool_calls>` 的起点；但这个 tag 是 token 片段序列，且首尾可能被 BPE 和上下文合并，不存在稳定 token-id DFA。
- 即使 grammar 约束成功，也不会自动告诉 dash_sc 下游 `generate_think_token_num` 应该是多少；reasoning/content 的边界仍然要由 frontend 明确设置。
- 如果把“隐式 thinking 结束”塞进 grammar logits processor，还要处理异步调度、MTP/speculative draft token accept/rollback、grammar matcher 状态回滚、reasoning 状态迁移的一致性，复杂度高且收益不直接。

因此 xgrammar 的合理位置是 phase-2 tool-call 生成，不是 phase-1 DSML marker 边界识别。

### 靠 mask logits 不行

在采样阶段 mask 掉 `<｜DSML｜tool_calls>` 相关 token 也不可靠：

- marker 没有固定 token 序列，standalone encode 出来的 token id 不覆盖真实上下文里的左右合并形态。
- tool-call tag 本身由多个 token 片段拼成，mask 单个片段并不等价于 mask 整个 tag。
- 如果扩大 mask 范围，会误伤 `<`、`>`、`tool`、`_c`、`alls` 这类普通 token 或其他 DSML 结构。
- tool-call marker 本身是合法且需要的输出，只是它不能被归到 reasoning 里。把它 mask 掉会让模型无法正常发起 tool call。

因此问题不应该在 logits 层解决，而应该在输出后处理层识别边界。

### 简单插入 `</think>` 不行

“看到 DSML 后直接往同一条流里插入 `</think>`”也不是通用解法。

normal renderer 的做法不是向模型输出里补一个真实 `</think>` 文本，而是在解析层把 DSML marker 当作隐式边界：marker 之前给 `reasoning_content`，marker 之后给 tool parser。它不会把额外的 `</think>` 暴露给用户或 tool detector。

dash_sc 当前可以在一种受控场景下插入 synthetic `</think>`：当检测到 DSML marker，且存在匹配的 DSV4 `structural_tag` grammar 并决定切到 phase-2 时，phase-1 会在 marker 前截断，然后向下游发一个 synthetic `</think>` close，让下游知道 reasoning 已结束，再用 phase-2 重新生成 grammar-constrained tool call。

但这和“在原始同流 DSML 前简单插入 `</think>`”不是一回事。后者可能导致：

- 下游看到额外 close token，语义和 normal renderer 不一致。
- tool detector 的输入被污染。
- 仍然解决不了 BPE 左右合并导致的 token 边界不确定。
- 如果 DSML 起点和前一个字符合并在同一个 token 里，单纯插入 `</think>` 也无法表达“这个 token 的前半段属于 reasoning、后半段属于 tool tag”。

所以 synthetic `</think>` 只能作为 phase-2 切换协议的一部分使用，不能作为普通 same-stream 场景的补丁。

## dash_sc 当前接入方案

dash_sc 的修复思路是把 renderer 的核心策略迁移过来：不要按固定 token list 匹配，而是在小窗口内 decode token ids，然后搜索稳定的文本 marker。

代码里的共享常量：

```text
rtp_llm/utils/deepseekv4_constants.py

DSML_PREFIX = "<｜DSML｜"
DSML_TOOL_CALLS_MARKER = f"{DSML_PREFIX}tool_calls>"
```

处理流程：

```text
backend generated_ids chunk
      |
      v
pending_tail_ids + current_chunk_ids
      |
      v
decode 小 token buffer
      |
      v
搜索 decoded text 中的 <｜DSML｜tool_calls>
      |
      +-- 没找到:
      |     如果 decoded text 末尾是 marker 的部分前缀
      |       暂存对应 tail token，等待下一个 chunk
      |     其余 token 正常发出
      |
      +-- 找到:
            把 marker 的字符 offset 映射回 token offset
            设置 generate_think_token_num
            清空 pending tail
```

这样做和 renderer 一样，稳定边界是 decoded text 中的 DSML marker；不同点是 dash_sc 最终还必须把字符边界映射回 token offset，因为下游协议当前依赖 token 级的 `generate_think_token_num`。

## Phase-2 行为

检测到 `<｜DSML｜tool_calls>` 后，不一定都会进入 phase-2。

```text
decoded DSML marker found
      |
      v
是否存在匹配的 DSV4 structural_tag grammar，且 phase-2 可用？
      |
      +-- 是:
      |     phase-1 在 marker token offset 前截断
      |     发 synthetic </think>
      |     关闭 phase-1 stream
      |     phase-2 使用 structural_tag / xgrammar 重新生成 tool call
      |
      +-- 否:
            不切 phase-2
            同一条流继续
            设置 generate_think_token_num
            DSML marker 及后续内容交给下游 DSV4 tool parser
```

也就是说：

- `structural_tag` 命中时，phase-2 是为了让 xgrammar 接管 tool-call 生成。
- 没有 `structural_tag` 时，行为应对齐 normal frontend：同流继续，reasoning 边界在 DSML marker 之前，后续 DSML 交给 tool parser。
- 一旦 `generate_think_token_num` 被设置，后面即使再出现 `</think>`，也不能重新定义 reasoning 边界。

## 方案边界

normal renderer 可以在字符级拆分 decoded text；dash_sc 当前协议只能用 token ids 和 `generate_think_token_num` 表达边界。因此如果某个 token decode 后同时包含 reasoning 尾部字符和 DSML 起始字符，dash_sc 不能表达“半个 token 属于 reasoning，半个 token 属于 tool-call”。

保守策略是把包含 DSML 起点的整个 token 放到非 reasoning 侧，优先保证 DSML 不泄漏进 `reasoning_content`。这和 renderer 的字符级精度有差异，但符合 dash_sc 当前 token 级协议的能力边界。

## 验证

已跑过的单测：

```bash
env PYTHONPATH=. /opt/conda310/bin/python rtp_llm/dash_sc/test/inference/servicer_test.py
env PYTHONPATH=. /opt/conda310/bin/python rtp_llm/test/deepseekv4_renderer_test.py
```

额外用真实 DeepSeek-V4 tokenizer 做过 marker 形态检查，覆盖了：

- standalone marker
- `>\n` 右合并
- `>{` 右合并
- `><｜DSML...` 右合并
- `" <"` 左合并
- `".<"` 左合并

这些 case 说明固定 token 序列方案不可靠，也说明 decoded text 检测是当前更稳的方案。

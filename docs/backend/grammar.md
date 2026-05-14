# Grammar / Constrained Decoding (XGrammar)

RTP-LLM uses [XGrammar](https://github.com/mlc-ai/xgrammar) as its default
constrained decoding backend. With grammar enabled, the engine masks the
token logits at every decoding step so that only tokens compatible with the
caller-supplied schema (JSON Schema / regex / EBNF) can be sampled. The
final output is byte-for-byte conformant — no post-hoc retry / repair.

For reasoning models that emit `<think>...</think>` content before the
schema-conformant payload, an additional **reasoner wrapper** lets the
matcher run in "free-accept" mode during the thinking segment and snap onto
the schema only after `</think>` is observed.

## Overview

Two orthogonal switches control the feature:

| Switch | Purpose |
|---|---|
| `--grammar_backend` | Picks the backend implementation. Default `xgrammar`. Set to `none` to disable. |
| `--reasoning_parser` | Empty by default. Pass a sglang-compatible detector key (e.g. `qwen3`) to wrap the backend with reasoner gating. |

Constrained generation is requested **per-request** via
`response_format` (OpenAI-compatible) or the equivalent fields on the
native API. The server-side switches above only need to be set once at
startup.

## Server Arguments

All flags live in the `Grammar Configuration` argument group. Defaults are
production-safe; override only when needed.

| Argument | Env | Default | Description |
|---|---|---|---|
| `--grammar_backend` | `GRAMMAR_BACKEND` | `xgrammar` | Backend impl. `xgrammar` or `none`. |
| `--constrained_json_disable_any_whitespace` | `CONSTRAINED_JSON_DISABLE_ANY_WHITESPACE` | `False` | Disables XGrammar's "any-whitespace" mode for JSON schemas. Strict mode rejects any whitespace not explicitly allowed by the schema — useful when downstream parsers are fragile. |
| `--reasoning_parser` | `REASONING_PARSER` | `""` (disabled) | sglang-compatible detector key. See [Reasoning Parser](#reasoning-parser). |
| `--grammar_compile_timeout_ms` | `GRAMMAR_COMPILE_TIMEOUT_MS` | `60000` | Wall-clock timeout per compile request inside the GrammarManager queue. Raise under sustained queue pressure; lower to fail-fast on huge schemas. |
| `--grammar_num_workers` | `GRAMMAR_NUM_WORKERS` | `32` | Size of the C++ compile worker pool. A pathological schema (recursive `$ref`, ReDoS regex) can hang one worker indefinitely; with N workers, the system survives N-1 concurrent hangs. |

## Sending Constrained Requests

Use the OpenAI `/v1/chat/completions` endpoint with the `response_format`
field. Three constraint modes are supported.

### JSON Schema

```json
{
  "model": "any",
  "messages": [{"role": "user", "content": "Generate a person profile."}],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "person",
      "schema": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "age":  {"type": "integer"},
          "city": {"type": "string"}
        },
        "required": ["name", "age", "city"]
      }
    }
  }
}
```

The `name` field is required by XGrammar's compile cache key; pick any
string that uniquely identifies the schema shape.

### Regex

```json
{
  "messages": [{"role": "user", "content": "Pick a hex color."}],
  "response_format": {
    "type": "regex",
    "pattern": "#[0-9a-fA-F]{6}"
  }
}
```

### EBNF

```json
{
  "messages": [{"role": "user", "content": "Emit two integers."}],
  "response_format": {
    "type": "ebnf",
    "grammar": "root ::= [0-9]+ \" \" [0-9]+"
  }
}
```

### Disabling per-request

Omit `response_format` entirely. The backend short-circuits and decoding
runs without grammar masking.

## Reasoning Parser

### Why it exists

Reasoning models (DeepSeek-R1, Qwen3 thinking, GLM4.5, Kimi K2, …) emit
their answer in **two segments**:

```
<think> arbitrary chain-of-thought reasoning ... </think>{schema-conformant output}
```

A vanilla grammar matcher would reject `<think>` itself (it isn't valid
JSON / regex / EBNF), so the model would derail at the very first token.
The reasoner wrapper solves this by:

1. Letting any token through during the thinking segment ("free-accept").
2. Watching for the model-specific `</think>` sentinel token.
3. Switching to strict schema enforcement the moment that token is
   sampled.

### How `--reasoning_parser` works

You pass a **detector key** that names which sglang detector class to use:

```
--reasoning_parser qwen3
```

At engine startup the bootstrap path:

1. Looks up `ReasoningParser.DetectorMap[<key>]` to find the detector
   class.
2. Reads its `think_end_token` string (e.g. `</think>` or `◁/think▷`).
3. Encodes that string with the live HF tokenizer.
4. If it encodes to **exactly one** token id, stashes it as
   `GrammarConfig.think_end_id` and the C++ engine builds an
   `XGrammarBackendCpp` in reasoner mode. Otherwise logs a warning and
   falls back to plain xgrammar.

If the key is **not** in `DetectorMap`, the engine logs a warning and
disables the reasoner — grammar still works but the model will derail on
the `<think>` token.

### Available Detector Keys

These keys are wired into `ReasoningParser.DetectorMap` (sglang-compatible
vocabulary). Pick the row matching your model.

| Key | `think_end_token` | Models |
|---|---|---|
| `qwen3` | `</think>` | Qwen3-* (8B / 32B / Next / 235B …), Qwen3.5-* MoE/Dense |
| `qwen3-thinking` | `</think>` | Qwen3-Thinking-* (forces reasoning on regardless of `enable_thinking`) |
| `deepseek-r1` | `</think>` | DeepSeek-R1, DeepSeek-R1-0528 |
| `deepseek-v3` | `</think>` | DeepSeek-V3, DeepSeek-V3.1, DeepSeek-V3.2 (reasoning variants) |
| `glm45` | `</think>` | ChatGLM-4.5 (and other GLM thinking variants) |
| `kimi` | `◁/think▷` | Kimi Thinking (base) |
| `kimi_k2` | `</think>` | Kimi-K2 (uses Qwen3-style think tokens) |
| `step3` | `</think>` | StepFun Step3 |

> **Note.** Several keys map to the same `</think>` literal but the
> tokenizer-side encoded id is **per-model**: the same string can resolve
> to a single special-token id (Qwen3 → 151668), to a multi-token
> sequence (some untrained tokenizers), or fail to encode at all. The
> bootstrap path re-encodes per startup so you do not need to know the
> id manually — but if encoding produces more than one token, the
> reasoner is auto-disabled with a warning.

### How to Pick the Right Key

1. Find your model family in the table above.
2. If the model card says "thinking mode supported", use that family's
   key.
3. If your model is a derivative (LoRA / SFT) of a listed family and
   keeps the same `<think>...</think>` token, reuse the parent family's
   key — the resolver only needs the token literal, not the model
   architecture.

If your model isn't listed:

- **First option (preferred).** Add an entry to
  `rtp_llm/openai/renderers/sglang_helpers/reasoning_parser.py` —
  introduce a new `*Detector` class with the right
  `think_start_token` / `think_end_token`, register it in
  `DetectorMap`. Send a PR.
- **Second option (interim).** If the new model uses one of the existing
  literals (e.g. `</think>`), just reuse the closest matching key
  (`qwen3` is the most generic).

### Combining with `--think_mode`

`--reasoning_parser` toggles the **grammar-side** wrapper only. To get
full reasoning behaviour you usually want both:

| Flag | Effect |
|---|---|
| `--think_mode 1` | Engine-side: marks `generate_config.in_think_mode=true`, which sets `require_reasoning=true` at the GrammarManager level. Without this, the reasoner wrapper is built but never engaged per-request. |
| `--reasoning_parser <key>` | Resolves `think_end_id` and wraps the backend in `ReasonerGrammarBackend`. Required for the matcher to know where the reasoning segment ends. |
| `chat_template_kwargs.enable_thinking=true` (per-request) | Tokenizer-side: drives Qwen3's chat template into thinking mode. Required for the model to actually emit the `<think>` opening. Other model families have their own switches; consult their chat template. |

### Worked Example: Qwen3-8B with JSON Schema + Reasoning

**Server start.**

```bash
python3 -m rtp_llm.start_server \
    --checkpoint_path /path/to/Qwen3-8B \
    --model_type qwen_3 \
    --act_type BF16 \
    --think_mode 1 \
    --reasoning_parser qwen3
```

Startup log to verify:

```
grammar bootstrap: vocab=151936, json=12345B, think_end_id=151668
XGrammarBackendCpp constructed: think_end_id=151668, ...
```

`think_end_id != -1` confirms the reasoner is active.

**Per-request payload.**

```json
{
  "messages": [
    {"role": "user", "content": "Generate a person profile as JSON."}
  ],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "person",
      "schema": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "age":  {"type": "integer"},
          "city": {"type": "string"}
        },
        "required": ["name", "age", "city"]
      }
    }
  },
  "extra_configs": {
    "chat_template_kwargs": {"enable_thinking": true}
  }
}
```

**Response shape.**

```
<think>
The user wants a person object with name/age/city ...
</think>
{"name": "Alice", "age": 30, "city": "Beijing"}
```

The text inside `<think>...</think>` is **not** schema-constrained (the
matcher is in free-accept). The text after `</think>` is byte-for-byte
JSON-schema-conformant.

### Worked Example: DeepSeek-R1

DeepSeek-R1 always reasons (no opt-in switch needed):

```bash
python3 -m rtp_llm.start_server \
    --checkpoint_path /path/to/DeepSeek-R1 \
    --model_type deepseek_v3 \
    --think_mode 1 \
    --reasoning_parser deepseek-r1
```

R1 sometimes omits the opening `<think>` tag and only emits the closing
`</think>`. The `DeepSeekR1Detector` sets `force_reasoning=True` so the
matcher starts in free-accept regardless.

### Common Pitfalls

- **`reasoner_grammar: think_end_token=... encoded to N tokens; reasoner
  disabled`** — your tokenizer doesn't have the closing think tag as a
  single token. Add it as a special token via the tokenizer's
  `added_tokens.json`, or use a different detector key whose literal
  does encode atomically.
- **`reasoner_grammar: think_end_id resolve failed (Unsupported model
  type: foo)`** — the key you passed isn't in `DetectorMap`. Check
  spelling against the table above.
- **Reasoner enabled but model still derails on `<think>`** — you forgot
  `--think_mode 1`, or the request didn't set
  `chat_template_kwargs.enable_thinking=true`. Both are required for
  Qwen3 family.
- **JSON output has stray whitespace your downstream parser hates** —
  pass `--constrained_json_disable_any_whitespace 1`.
- **Compile worker hangs forever on a recursive `$ref` schema** —
  expected, that's why `--grammar_num_workers` defaults to 32. Lower
  `--grammar_compile_timeout_ms` if you want tighter fail-fast at the
  queue level (the running compile itself isn't interrupted).

## Limitations

- Grammar masking adds per-step bitmask compute; expect ~1-3% TPS hit
  on dense models, more on small MoE.
- The OpenAI `tools` / function-calling path uses its own renderer
  (`reasoning_tool_base_renderer.py`) which is independent of
  `--reasoning_parser`. Tool-calling models pick the detector via the
  renderer; the CLI flag governs only the grammar wrapper.
- Reasoner mode requires the closing think token to be a **single**
  tokenizer id. Models that split it across multiple tokens are
  silently downgraded to plain grammar (with a warning at startup).

## See Also

- `rtp_llm/cpp/engine_base/grammar/XGrammarBackendCpp.h` — C++ backend
  implementation.
- `rtp_llm/openai/renderers/sglang_helpers/reasoning_parser.py` —
  detector definitions and `DetectorMap`.
- `rtp_llm/async_decoder_engine/xgrammar_bootstrap.py` — engine-init
  bootstrap that resolves `think_end_id`.
- [XGrammar documentation](https://xgrammar.mlc.ai/) — upstream grammar
  syntax reference.

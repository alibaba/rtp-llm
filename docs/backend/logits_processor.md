### LogitsProcessor

#### Overview
LogitsProcessor is the preprocessing chain that constrains/masks logits before sampling, enabling hard-constrained decoding or special modes (e.g., think mode). In RTP-LLM, LogitsProcessors work together with the Sampler: they first mask the vocabulary distribution, then regular sampling strategies (top-k, top-p, temperature, etc.) are applied.

#### Built-in processors
- **MultiSeqLogitsProcessor**: For finished sequences, only keeps the `eos_token` valid to prevent further generation.
- **ThinkModeLogitsProcessor**: Think-mode control. Combined with templates/end markers, it constrains tokens within the “thinking” segment by set max_thinking_tokens (internally uses a string-contain DFA).
- **TreeLogitsProcessor (highlight)**: Enforces hard decoding constraints based on a prefix → candidate-token-set mapping (tree-structured DFA). Only tokens within the candidate set are allowed.

---

### Tree Decode: How it works
Tree Decode maintains a tree-shaped DFA (`TreeDFA<std::string, int>`) at runtime, driven by a mapping of “prefix → candidate tokens”:

- **State representation**: a string key concatenating the start token id and the generated token ids with a separator, e.g., `"225_64000_64001"`.
- **Candidate set**: for the current `status`, lookup the allowed candidate tokens from the mapping; the logits processor builds a vocabulary mask from this set and sets non-candidate positions to `-inf`.
- **Fallback**: if the mapping does not contain the current key, only `end_token_id` is allowed (forces early termination), ensuring safety and validity of constraints.
- **Beam/multi-seq**: supports beam search and `num_return_sequences`. Each beam/sequence maintains its own DFA state and updates it after each generated token.

Simplified steps:
1) Initialize DFA: `status = str(start_token_id)`;
2) At each step, get candidates by `status` → build mask → set non-candidates to `-inf`;
3) After sampling a new token `t`, update DFA: `status = status + sep + str(t)`;
4) If no candidates exist for the new status, fallback to `[end_token_id]`.

Note: Ensure that the token_id corresponding to the last token of the input is the root of the tree. If using a prompt string, make sure the last token of the prompt is the tree's root. If using messages, this must be ensured via the chat template.

---

### JSON configuration format
The prefix mapping for Tree Decode is provided via a JSON file:

```json
{
  "start_token_id": 225,
  "end_token_id": 2,
  "sep": "_",
  "prefix_dict": {
    "225_64000": [64001, 64002],
    "225_64000_64001": [2]
  }
}
```

- **start_token_id**: start token id.
- **end_token_id**: end token id.
- **sep**: separator for keys, default `_`.
- **prefix_dict**: key is the concatenation string of `start` and already generated token ids; value is the allowed candidate token list under that prefix.

Note: Keys should start from `start_token_id`. For example, the key for the first-step candidates is `"{start}_{t0}"`, the second step is `"{start}_{t0}_{t1}"`, and so on.

---

### Enabling Tree Decode
Tree Decode is enabled when the model loads a valid config file (global effect; disabled if not configured):

- **Service arg / environment variable** (recommended):
  - Arg: `--tree_decode_config <file_name>`
  - Env: `TREE_DECODE_CONFIG=<file_name>`
  - Resolution rule: the file path is resolved as `<ckpt_path>/<file_name>`.

- **C++ direct call (dev/test)**:
  - `PrefixToCandidateTokens::instance()->reloadPrefixDict("/abs/path/to/tree.json");`

If loaded successfully, you will see: `PrefixToCandidateTokens load [path] successfully` in logs. If not configured or failed to load, Tree Decode is disabled.

---

### Design and best practices
- **Vocab alignment**: `start_token_id`/`end_token_id` must be valid ids of the current model.
- **Key convention**: keys start with `start_token_id`; the separator is controlled by `sep` (default `_`).
- **Performance**: keep candidate sets as small as possible to reduce masking overhead; build large mappings offline and load per scenario when necessary.
- **Composition**: Tree Decode masks first, then regular sampling (top-k/top-p/temperature) applies; they are compatible.

---

### Related source files
- Entry: `rtp_llm/cpp/engine_base/stream/GenerateStream.cc` (build and register logits processors)
- Tree Decode: `rtp_llm/cpp/models/logits_processor/TreeLogitsProcessor.{h,cc}`
- DFA/Mapping: `rtp_llm/cpp/models/logits_processor/DFAUtil.h`, `PrefixToCandidateTokens.h`

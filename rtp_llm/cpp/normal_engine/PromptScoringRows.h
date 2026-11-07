#pragma once

#include <algorithm>
#include <utility>

namespace rtp_llm {

// Row range [a, b) — in the stream's own execute-token coordinates — whose full-vocab logits any
// output consumer can read during this forward step:
//   return_prompt_logits -> the clamped window [start, end)
//   calculate_loss       -> [0, token_size - 1)
//   both                 -> their union (contiguous, because the loss range starts at 0)
//   neither              -> empty
// NormalModelInputGatherer uses this to build the scoring row set (all_logits_row_indexes) and
// NormalOutputDispatcher uses the identical rule to locate each stream's slice inside the
// produced scoring outputs (legacy all_logits at tp==1, sharded partial rows at tp>1). The
// clamping must stay byte-for-byte identical to the dispatcher's window logic.
inline std::pair<int, int> promptScoringRowRange(
    int token_size, bool calculate_loss, bool return_prompt_logits, int prompt_logits_start, int prompt_logits_end) {
    int a = 0, b = 0;
    if (return_prompt_logits) {
        const int s = std::clamp(prompt_logits_start >= 0 ? prompt_logits_start : 0, 0, token_size);
        const int e = std::clamp(prompt_logits_end >= 0 ? prompt_logits_end : token_size, s, token_size);
        a           = s;
        b           = e;
    }
    if (calculate_loss) {
        a = 0;
        b = std::max(b, token_size - 1);
    }
    return {a, b};
}

}  // namespace rtp_llm

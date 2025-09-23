#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace rtp_llm {
void calculatePaddingOffset(torch_ext::PyAttentionInputs& py_attn_inputs);

void getPaddingOffset(
    int32_t* padding_offset, int32_t* input_lengths, int32_t* prefix_length, int32_t batch_size, int32_t max_seq_len);
}  // namespace rtp_llm

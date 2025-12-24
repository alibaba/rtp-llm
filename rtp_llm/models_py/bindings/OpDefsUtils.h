#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <cstdint>

namespace rtp_llm {

inline void getPaddingOffset(
    int32_t* padding_offset, int32_t* input_lengths, int32_t* prefix_length, int32_t batch_size, int32_t max_seq_len) {
    // do cumulated sum
    int32_t cum_offset = 0;
    int32_t index      = 0;
    for (int32_t i = 0; i < batch_size; i++) {
        int32_t seq_len = input_lengths[i];
        if (prefix_length) {
            seq_len += prefix_length[i];
        }
        if (padding_offset) {
            for (int32_t j = 0; j < seq_len; j++) {
                padding_offset[index] = cum_offset;
                index++;
            }
        }
        cum_offset += max_seq_len - seq_len;
    }
}

// for `FusedRopKVCache` kernel
inline void calculatePaddingOffset(torch_ext::PyAttentionInputs& py_attn_inputs) {
    // check input_lengths and prefix_lengths is host tensor
    RTP_LLM_CHECK_WITH_INFO(py_attn_inputs.input_lengths.device().is_cpu(), "input_lengths must be a host tensor");
    RTP_LLM_CHECK_WITH_INFO(py_attn_inputs.prefix_lengths.device().is_cpu(), "prefix_lengths must be a host tensor");

    int     batch_size   = py_attn_inputs.input_lengths.size(0);
    int32_t total_tokens = py_attn_inputs.total_tokens;

    // inputs_length:  [1,2,1,1] ,total_tokens = 5
    // padding_offsets: [0,1,1,1,2]
    int  max_seq_len         = py_attn_inputs.input_lengths.max().item<int32_t>();
    auto padding_offset_host = torch::zeros({total_tokens}, torch::TensorOptions(torch::kInt32).device(torch::kCPU));

    if (total_tokens > 0) {
        getPaddingOffset(padding_offset_host.data_ptr<int32_t>(),
                         py_attn_inputs.input_lengths.data_ptr<int32_t>(),
                         nullptr,
                         batch_size,
                         max_seq_len);
    }

    py_attn_inputs.padding_offset = padding_offset_host.cuda();
}

}  // namespace rtp_llm
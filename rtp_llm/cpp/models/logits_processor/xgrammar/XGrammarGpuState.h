#pragma once

#include <torch/all.h>

namespace rtp_llm {

struct XGrammarDeviceState {
    torch::Tensor state_id;                // [batch], int32, cuda
    torch::Tensor terminated;              // [batch], bool/int8, cuda
    torch::Tensor dead;                    // [batch], bool/int8, cuda
    torch::Tensor consumed_seq_len;         // [batch], int64, cuda
    torch::Tensor grammar_accept_len_cap;   // [batch], int32, cuda
};

}  // namespace rtp_llm

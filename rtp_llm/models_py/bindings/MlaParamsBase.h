#pragma once
#include <memory>
#include <torch/extension.h>
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace rtp_llm {

class MlaParamsBase {
public:
    virtual ~MlaParamsBase() = default;
    torch_ext::MlaParams fillParams(torch::Tensor t_prefix_lengths,
                                    torch::Tensor t_sequence_lengths,
                                    torch::Tensor t_input_lengths,
                                    torch::Tensor t_kv_cache_block_id_host,
                                    int           seq_size_per_block);
};

}  // namespace rtp_llm
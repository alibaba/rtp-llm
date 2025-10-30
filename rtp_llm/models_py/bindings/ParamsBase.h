#pragma once
#include <memory>
#include <torch/extension.h>

namespace rtp_llm {

using ParamsPtr = std::shared_ptr<void>;
class ParamsBase {
public:
    virtual ~ParamsBase() = default;
    virtual void fillParams(torch::Tensor sequence_lengths,
                            torch::Tensor input_lengths,
                            torch::Tensor kv_cache_block_id_host,
                            int           batch_size,
                            int           seq_size_per_block) {};

    virtual void update() {};
    // check whether the parmas can be recycled automatically.
    virtual bool check_recycle() {
        return false;
    };

    // Public member variables
    torch::Tensor sequence_lengths_;
    torch::Tensor input_lengths_;
    torch::Tensor kv_cache_block_id_host_;
    int           batch_size_;
    int           seq_size_per_block_;
    torch::Tensor cu_seqlens_q_;
    torch::Tensor cu_seqlens_k_;
    torch::Tensor seq_lens_;
    torch::Tensor kv_cache_block_id_device_;
    // bool          is_prefill_{true};
    int max_seq_len_{0};
    int max_seqlen_q_{0};
    int max_seqlen_k_{0};
};
using ParamsBasePtr = std::shared_ptr<ParamsBase>;

}  // namespace rtp_llm
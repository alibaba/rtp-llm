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
    // reuse `this` params, so we can give it back to memory cache if it's supported.
    virtual void recycleParams() {};
};
using ParamsBasePtr = std::shared_ptr<ParamsBase>;

}  // namespace rtp_llm
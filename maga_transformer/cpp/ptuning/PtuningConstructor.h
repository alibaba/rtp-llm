#pragma once
#include <assert.h>
#include <cstdint>
#include <mutex>
#include <string>
#include <optional>
#include <vector>
#include <torch/torch.h>
#include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/cache/CacheManager.h"
#include "maga_transformer/cpp/ptuning/Ptuning.h"

namespace rtp_llm {

// class PtuningConstructor {
// public:
//     static PrefixParams createPtuningV2Params(const CacheConfig& config,
//                                               std::shared_ptr<CacheManager> cache_manager,
//                                               const torch::Tensor& prefix_prompt) {
//         int prefix_seq_length = prefix_prompt.size(-2);
        
//         // Assume the Tensor class has methods for reshaping and permuting dimensions
//         torch::Tensor reshaped_prefix_prompt = prefix_prompt.reshape(config.layer_num, 2, prefix_prompt.size(1), prefix_prompt.size(2), prefix_prompt.size(3))
//                                                         .permute({1, 0, 3, 2, 4})
//                                                         .makeContiguous();

//         int prefixBlocks = (prefix_seq_length - 1) / config.seq_size_per_block + 1;
//         std::vector<int> prefix_block_indice = cache_manager->mallocIndex(prefixBlocks);
//         setKvPrefixBlock(config, cache_manager, reshaped_prefix_prompt, prefix_block_indice);

//         // Assuming constructor for PrefixParams exists and takes these arguments
//         return PrefixParams(PrefixType::PTuningV2, prefix_seq_length, prefixBlockIndice, nullptr); // `nullptr` used in place of an actual tensor
//     }



// };

} // namespace rtp_llm

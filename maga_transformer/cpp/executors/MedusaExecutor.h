#pragma once

#include "maga_transformer/cpp/executors/NormalExecutor.h"
#include "maga_transformer/cpp/dataclass/MagaInitParameter.h"
#include "maga_transformer/cpp/dataclass/MergedQuery.h"

namespace rtp_llm {

class MedusaModelExecutor: public NormalExecutor {
public:
    explicit MedusaModelExecutor(const MagaInitParams&                                                   params,
                                 const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layer_weights,
                                 const std::unordered_map<std::string, ft::ConstBufferPtr>&              weights);
};

}  // namespace rtp_llm

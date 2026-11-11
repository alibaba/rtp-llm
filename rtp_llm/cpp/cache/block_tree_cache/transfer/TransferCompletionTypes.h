#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTaskRegistry.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

struct TransferCompletionRankResult {
    size_t                      rank{0};
    std::string                 server_address;
    TransferTaskRegistry::State state{TransferTaskRegistry::State::UNKNOWN};
    std::optional<int>           grpc_code;
    std::string                 error;
    bool                        cq_received{false};
    bool                        cq_ok{false};
};

struct TransferCompletionQueryResult {
    TransferExecutionTrace                    trace_id{uint64_t{0}};
    std::vector<TransferCompletionRankResult> ranks;
    bool                                      all_safe{false};
    std::string                               error;
};

}  // namespace rtp_llm

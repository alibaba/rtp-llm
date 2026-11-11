#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTaskRegistry.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

inline constexpr size_t kTransferCompletionForgetBatchSize = 1024;

// Routing only: cleanup must not retain a FrozenTask, pool or business owner.
struct TransferCompletionPeer {
    size_t      rank{0};
    std::string address;
};

struct TransferCompletionTarget {
    bool                                remote{false};
    std::vector<TransferCompletionPeer> peers;
};

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

struct TransferCompletionForgetRankResult {
    size_t                           rank{0};
    std::string                      server_address;
    TransferTaskRegistry::ForgetResult result{TransferTaskRegistry::ForgetResult::FAILED};
    std::optional<int>                grpc_code;
    std::string                      error;
    bool                             cq_received{false};
    bool                             cq_ok{false};
};

struct TransferCompletionForgetEntryResult {
    TransferExecutionTrace                         trace_id{uint64_t{0}};
    std::vector<TransferCompletionForgetRankResult> ranks;
    bool                                           all_done{false};
};

struct TransferCompletionForgetResult {
    std::vector<TransferCompletionForgetEntryResult> entries;
    std::string                                     error;
};

}  // namespace rtp_llm

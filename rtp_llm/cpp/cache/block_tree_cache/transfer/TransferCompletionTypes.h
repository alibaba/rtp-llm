#pragma once

#include <cstddef>
#include <cstdint>
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

struct TransferCompletionIdentity {
    RegistryType registry_type{RegistryType::BLOCK_TRANSFER};
    std::string trace_id;
    std::optional<int64_t>              request_id;
    bool                                remote{false};
    size_t                              expected_rank_count{0};
    std::vector<TransferCompletionPeer> expected_peers;
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
    bool                        admission_closed{false};

    bool bufferSafe() const {
        return state == TransferTaskRegistry::State::BUFFER_SAFE
               || (state == TransferTaskRegistry::State::UNKNOWN && admission_closed);
    }
};

struct TransferCompletionQueryResult {
    RegistryType registry_type{RegistryType::BLOCK_TRANSFER};
    std::string trace_id;
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
    RegistryType registry_type{RegistryType::BLOCK_TRANSFER};
    std::string trace_id;
    std::vector<TransferCompletionForgetRankResult> ranks;
    bool                                           all_done{false};
};

struct TransferCompletionForgetResult {
    std::vector<TransferCompletionForgetEntryResult> entries;
    std::string                                     error;
};

}  // namespace rtp_llm

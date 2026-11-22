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

struct TransferCompletionIdentity {
    TransferCompletionKey key;
    std::optional<int64_t>              request_id;
    bool                                remote{false};
    size_t                              expected_rank_count{0};
};

struct TransferCompletionTarget {
    bool   remote{false};
    size_t rank_count{0};
};

struct TransferCompletionRankResult {
    size_t                      rank{0};
    TransferTaskRegistry::State state{TransferTaskRegistry::State::UNKNOWN};
    std::optional<int>           grpc_code;
    std::string                 error;
    bool                        admission_closed{false};

    bool bufferSafe() const {
        return state == TransferTaskRegistry::State::BUFFER_SAFE
               || (state == TransferTaskRegistry::State::UNKNOWN && admission_closed);
    }
};

struct TransferCompletionQueryResult {
    std::string trace_id;
    std::vector<TransferCompletionRankResult> ranks;
    bool                                      all_safe{false};
    std::string                               error;
};

struct TransferCompletionForgetRankResult {
    size_t                           rank{0};
    TransferTaskRegistry::ForgetResult result{TransferTaskRegistry::ForgetResult::FAILED};
    std::optional<int>                grpc_code;
    std::string                      error;
};

struct TransferCompletionForgetEntryResult {
    std::string trace_id;
    std::vector<TransferCompletionForgetRankResult> ranks;
    bool                                           all_done{false};
};

struct TransferCompletionForgetResult {
    std::vector<TransferCompletionForgetEntryResult> entries;
    std::string                                     error;
};

}  // namespace rtp_llm

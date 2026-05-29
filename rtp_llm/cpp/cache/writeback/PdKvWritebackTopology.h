#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"

namespace rtp_llm {

enum class PdKvWritebackTopologyMode {
    TpEqual,
};

struct PdKvWritebackTopologyInput {
    int32_t local_tp_size               = 1;
    int32_t source_partition_count      = 1;
    int32_t destination_partition_count = 1;
    bool    prefill_cp_enabled          = false;

    std::vector<std::string> decode_grpc_addrs;
    std::vector<std::string> prefill_grpc_addrs;
    std::vector<std::string> prefill_worker_addrs;
};

struct PdKvWritebackRankMapping {
    int32_t decode_rank  = 0;
    int32_t prefill_rank = 0;

    std::string decode_grpc_addr;
    std::string prefill_grpc_addr;
    std::string prefill_worker_addr;

    int32_t local_partition_count  = 1;
    int32_t local_partition_id     = 0;
    int32_t remote_partition_count = 1;
    int32_t remote_partition_id    = 0;
};

struct PdKvWritebackTopologyPlan {
    PdKvWritebackTopologyMode              mode = PdKvWritebackTopologyMode::TpEqual;
    std::vector<PdKvWritebackRankMapping> mappings;
};

absl::StatusOr<PdKvWritebackTopologyPlan> buildPdKvWritebackTopology(const PdKvWritebackTopologyInput& input);

}  // namespace rtp_llm

#include "rtp_llm/cpp/cache/writeback/PdKvWritebackTopology.h"

#include <utility>

#include "absl/status/status.h"

namespace rtp_llm {

absl::StatusOr<PdKvWritebackTopologyPlan> buildPdKvWritebackTopology(const PdKvWritebackTopologyInput& input) {
    if (input.local_tp_size <= 0) {
        return absl::FailedPreconditionError("local_tp_size must be positive");
    }
    if (input.source_partition_count <= 0 || input.destination_partition_count <= 0) {
        return absl::FailedPreconditionError("partition_count must be positive");
    }
    if (input.prefill_grpc_addrs.size() != input.prefill_worker_addrs.size()) {
        return absl::FailedPreconditionError("prefill address count mismatch");
    }
    if (input.prefill_cp_enabled) {
        return absl::UnimplementedError("unsupported_topology: prefill cp writeback is not supported in phase 1");
    }

    const auto local_tp_size = static_cast<size_t>(input.local_tp_size);
    if (input.decode_grpc_addrs.size() != local_tp_size) {
        return absl::FailedPreconditionError("decode grpc address count mismatch");
    }
    if (input.prefill_grpc_addrs.size() != local_tp_size) {
        return absl::UnimplementedError("unsupported_topology: prefill tp must equal decode tp in phase 1");
    }
    if (input.source_partition_count != input.destination_partition_count
        || input.source_partition_count != input.local_tp_size) {
        return absl::UnimplementedError("unsupported_topology: source/destination partition counts must equal local tp");
    }

    PdKvWritebackTopologyPlan plan;
    plan.mode = PdKvWritebackTopologyMode::TpEqual;
    plan.mappings.reserve(local_tp_size);
    for (size_t i = 0; i < local_tp_size; ++i) {
        PdKvWritebackRankMapping mapping;
        mapping.decode_rank            = static_cast<int32_t>(i);
        mapping.prefill_rank           = static_cast<int32_t>(i);
        mapping.decode_grpc_addr       = input.decode_grpc_addrs[i];
        mapping.prefill_grpc_addr      = input.prefill_grpc_addrs[i];
        mapping.prefill_worker_addr    = input.prefill_worker_addrs[i];
        mapping.local_partition_count  = 1;
        mapping.local_partition_id     = 0;
        mapping.remote_partition_count = 1;
        mapping.remote_partition_id    = 0;
        plan.mappings.push_back(std::move(mapping));
    }
    return plan;
}

}  // namespace rtp_llm

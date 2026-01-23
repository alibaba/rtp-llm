#include "rtp_llm/cpp/cache/connector/p2p/AsymmetricTpUtil.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

AsymmetricTpUtil::AsymmetricTpUtil(const ParallelismConfig& parallelism_config):
    parallelism_config_(parallelism_config) {}

AsymmetricTpUtil::~AsymmetricTpUtil() = default;

std::vector<AsymmetricTPContext>
AsymmetricTpUtil::handleAsymmetricTP(const std::vector<std::pair<std::string, uint32_t>>& decode_transfer_servers) {
    if (parallelism_config_.tp_size > static_cast<int64_t>(decode_transfer_servers.size())) {
        return handleNP1D(decode_transfer_servers);
    }
    return handleND1P(decode_transfer_servers);
}

std::vector<AsymmetricTPContext>
AsymmetricTpUtil::handleNP1D(const std::vector<std::pair<std::string, uint32_t>>& decode_transfer_servers) {
    if (parallelism_config_.tp_size % static_cast<int64_t>(decode_transfer_servers.size()) != 0) {
        RTP_LLM_LOG_ERROR(
            "AsymmetricTpUtil handleNP1D: tp_size %ld is not divisible by decode_transfer_servers size %zu",
            parallelism_config_.tp_size,
            decode_transfer_servers.size());
        return std::vector<AsymmetricTPContext>();
    }

    // prefill is more than decode, prefill should send to one decode partially
    auto  local_partition_count  = 1;
    auto  local_partition_id     = 0;
    auto  remote_partition_count = static_cast<int>(parallelism_config_.tp_size / decode_transfer_servers.size());
    auto  remote_partition_id    = static_cast<int>(parallelism_config_.tp_rank % remote_partition_count);
    auto& decode_transfer_server =
        decode_transfer_servers[static_cast<size_t>(parallelism_config_.tp_rank / remote_partition_count)];

    std::vector<AsymmetricTPContext> asymmetric_tp_contexts{{decode_transfer_server.first,
                                                             decode_transfer_server.second,
                                                             local_partition_count,
                                                             local_partition_id,
                                                             remote_partition_count,
                                                             remote_partition_id}};
    return asymmetric_tp_contexts;
}

std::vector<AsymmetricTPContext>
AsymmetricTpUtil::handleND1P(const std::vector<std::pair<std::string, uint32_t>>& decode_transfer_servers) {
    if (decode_transfer_servers.size() % static_cast<size_t>(parallelism_config_.tp_size) != 0) {
        RTP_LLM_LOG_ERROR(
            "AsymmetricTpUtil handleND1P: decode_transfer_servers size %zu is not divisible by tp_size %ld",
            decode_transfer_servers.size(),
            parallelism_config_.tp_size);
        return std::vector<AsymmetricTPContext>();
    }

    std::vector<AsymmetricTPContext> asymmetric_tp_contexts;
    // decode is more than prefill, one prefill should send to multiple decode
    auto remote_partition_count = 1;
    auto remote_partition_id    = 0;
    auto local_partition_count  = static_cast<int>(decode_transfer_servers.size() / parallelism_config_.tp_size);
    for (int i = 0; i < local_partition_count; i++) {
        auto decode_transfer_server =
            decode_transfer_servers[static_cast<size_t>(parallelism_config_.tp_rank * local_partition_count + i)];
        auto local_partition_id = i;
        asymmetric_tp_contexts.emplace_back(decode_transfer_server.first,
                                            decode_transfer_server.second,
                                            local_partition_count,
                                            local_partition_id,
                                            remote_partition_count,
                                            remote_partition_id);
    }
    return asymmetric_tp_contexts;
}

}  // namespace rtp_llm

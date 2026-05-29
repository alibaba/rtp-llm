#include "rtp_llm/cpp/cache/writeback/PdKvWritebackTransfer.h"

#include <limits>
#include <utility>

#include "absl/status/status.h"

namespace rtp_llm {

namespace {

const std::vector<BlockIndicesType>& selectBlockIds(const PdKvWritebackTransferPlan& plan,
                                                    PdKvWritebackBlockSide           side) {
    return side == PdKvWritebackBlockSide::DecodeSource ? plan.decode_group_block_ids : plan.prefill_group_block_ids;
}

absl::StatusOr<std::pair<std::string, uint32_t>> parsePdKvWritebackTransferServer(const std::string& worker_addr) {
    const auto first_colon = worker_addr.find(':');
    if (first_colon == std::string::npos || first_colon == 0) {
        return absl::InvalidArgumentError("invalid worker addr");
    }
    const auto second_colon = worker_addr.find(':', first_colon + 1);
    const auto port_end     = second_colon == std::string::npos ? worker_addr.size() : second_colon;
    try {
        auto port = std::stoul(worker_addr.substr(first_colon + 1, port_end - first_colon - 1));
        if (second_colon != std::string::npos) {
            ++port;
        }
        if (port > std::numeric_limits<uint32_t>::max()) {
            return absl::InvalidArgumentError("worker addr port out of range");
        }
        return std::make_pair(worker_addr.substr(0, first_colon), static_cast<uint32_t>(port));
    } catch (...) {
        return absl::InvalidArgumentError("invalid worker addr port");
    }
}

}  // namespace

absl::StatusOr<KVCacheResourcePtr> buildPdKvWritebackResource(const PdKvWritebackTransferPlan& plan,
                                                              PdKvWritebackBlockSide           side) {
    const auto& group_block_ids = selectBlockIds(plan, side);
    if (plan.cache_keys.empty()) {
        return absl::InvalidArgumentError("cache_keys is empty");
    }
    if (group_block_ids.empty()) {
        return absl::InvalidArgumentError("group_block_ids is empty");
    }

    const int group_count = plan.group_count > 0 ? plan.group_count : static_cast<int>(group_block_ids.size());
    const int layer_count = plan.layer_count > 0 ? plan.layer_count : group_count;
    if (group_count != static_cast<int>(group_block_ids.size())) {
        return absl::InvalidArgumentError("group_count does not match group_block_ids size");
    }
    if (layer_count <= 0) {
        return absl::InvalidArgumentError("layer_count must be positive");
    }

    std::vector<int> layer_to_group_id;
    if (!plan.layer_to_group_id.empty()) {
        if (plan.layer_to_group_id.size() < static_cast<size_t>(layer_count)) {
            return absl::InvalidArgumentError("layer_to_group_id shorter than layer_count");
        }
        layer_to_group_id.assign(plan.layer_to_group_id.begin(), plan.layer_to_group_id.begin() + layer_count);
    }

    auto resource = std::make_shared<KVCacheResource>();
    resource->initGroups(group_count, layer_count, layer_to_group_id);
    resource->cacheKeys() = plan.cache_keys;
    for (int gid = 0; gid < group_count; ++gid) {
        resource->mutableBlockIds(gid).assign(group_block_ids[static_cast<size_t>(gid)]);
    }
    return resource;
}

std::vector<BlockIndicesType> extractPdKvWritebackGroupBlockIds(const BatchKVCacheResourcePtr& resource) {
    std::vector<BlockIndicesType> group_block_ids;
    if (!resource || resource->batchSize() == 0) {
        return group_block_ids;
    }
    const int group_count = resource->groupNums();
    group_block_ids.reserve(static_cast<size_t>(group_count));
    for (int gid = 0; gid < group_count; ++gid) {
        group_block_ids.push_back(resource->blocks(0, gid));
    }
    return group_block_ids;
}

std::vector<std::pair<std::string, uint32_t>>
parsePdKvWritebackTransferServers(const std::vector<std::string>& worker_addrs) {
    std::vector<std::pair<std::string, uint32_t>> servers;
    servers.reserve(worker_addrs.size());
    for (const auto& addr : worker_addrs) {
        auto server = parsePdKvWritebackTransferServer(addr);
        if (server.ok()) {
            servers.push_back(server.value());
        }
    }
    return servers;
}

absl::StatusOr<PdKvWritebackTransferTarget> parsePdKvWritebackTransferTarget(const std::string& worker_addr,
                                                                             int32_t            local_partition_count,
                                                                             int32_t            local_partition_id,
                                                                             int32_t            remote_partition_count,
                                                                             int32_t            remote_partition_id,
                                                                             int32_t            decode_rank,
                                                                             int32_t            prefill_rank) {
    auto server = parsePdKvWritebackTransferServer(worker_addr);
    if (!server.ok()) {
        return server.status();
    }
    PdKvWritebackTransferTarget target;
    target.ip                     = server->first;
    target.port                   = server->second;
    target.local_partition_count  = local_partition_count;
    target.local_partition_id     = local_partition_id;
    target.remote_partition_count = remote_partition_count;
    target.remote_partition_id    = remote_partition_id;
    target.decode_rank            = decode_rank;
    target.prefill_rank           = prefill_rank;
    return target;
}

}  // namespace rtp_llm

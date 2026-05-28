#include "rtp_llm/cpp/cache/writeback/PdKvWritebackManager.h"

#include "absl/status/status.h"

namespace rtp_llm {

absl::Status validatePdKvWritebackCompatibility(const PdKvWritebackCompatibility& source,
                                                const PdKvWritebackCompatibility& destination) {
    if (source.seq_size_per_block <= 0 || destination.seq_size_per_block <= 0) {
        return absl::FailedPreconditionError("seq_size_per_block must be positive");
    }
    if (source.seq_size_per_block != destination.seq_size_per_block) {
        return absl::FailedPreconditionError("seq_size_per_block mismatch");
    }
    if (source.layer_count <= 0 || destination.layer_count <= 0) {
        return absl::FailedPreconditionError("layer_count must be positive");
    }
    if (source.layer_count != destination.layer_count) {
        return absl::FailedPreconditionError("layer_count mismatch");
    }
    if (source.group_count <= 0 || destination.group_count <= 0) {
        return absl::FailedPreconditionError("group_count must be positive");
    }
    if (source.group_count != destination.group_count) {
        return absl::FailedPreconditionError("group_count mismatch");
    }
    if (source.partition_count <= 0 || destination.partition_count <= 0) {
        return absl::FailedPreconditionError("partition_count must be positive");
    }
    if (source.partition_count != destination.partition_count) {
        return absl::FailedPreconditionError("partition_count mismatch");
    }
    if (source.layer_to_group_id != destination.layer_to_group_id) {
        return absl::FailedPreconditionError("layer_to_group_id mismatch");
    }
    if (source.group_types != destination.group_types) {
        return absl::FailedPreconditionError("group_types mismatch");
    }
    return absl::OkStatus();
}

PdKvWritebackManager::PdKvWritebackManager(const PDSepConfig& pd_config, PdKvWritebackCacheWriter* cache_writer):
    pd_config_(pd_config), cache_writer_(cache_writer) {}

PdKvWritebackManager::PdKvWritebackManager(const PDSepConfig&           pd_config,
                                           PdKvWritebackCacheWriter*    cache_writer,
                                           PdKvWritebackTransferClient* transfer_client):
    pd_config_(pd_config), cache_writer_(cache_writer), transfer_client_(transfer_client) {}

PdKvWritebackLaunchResult PdKvWritebackManager::launchFromDecode(const PdKvWritebackLaunchRequest& request) const {
    if (!pd_config_.enable_pd_kv_cache_writeback) {
        return {PdKvWritebackLaunchStatus::Skipped, "disabled"};
    }
    if (request.manifest.reusable_block_count == 0) {
        return {PdKvWritebackLaunchStatus::Skipped, "empty_manifest"};
    }
    if (request.source_prefill_grpc_addrs.empty()) {
        return {PdKvWritebackLaunchStatus::Skipped, "missing_source_prefill_grpc_addrs"};
    }

    auto status = validatePdKvWritebackCompatibility(request.source, request.destination);
    if (!status.ok()) {
        return {PdKvWritebackLaunchStatus::Skipped, std::string(status.message())};
    }
    return {PdKvWritebackLaunchStatus::Started, "started"};
}

absl::Status PdKvWritebackManager::receiveOnPrefill(const PdKvWritebackLaunchRequest& request,
                                                    const BatchKVCacheResourcePtr&    destination_resource) {
    if (!pd_config_.enable_pd_kv_cache_writeback) {
        return absl::FailedPreconditionError("disabled");
    }
    if (!cache_writer_) {
        return absl::FailedPreconditionError("cache_writer is null");
    }
    if (!transfer_client_) {
        return absl::FailedPreconditionError("transfer_client is null");
    }
    if (request.manifest.reusable_block_count == 0) {
        return absl::OkStatus();
    }
    if (request.manifest.cache_keys.size() < static_cast<size_t>(request.manifest.reusable_block_count)) {
        return absl::InvalidArgumentError("cache_keys shorter than reusable_block_count");
    }
    auto compatibility_status = validatePdKvWritebackCompatibility(request.source, request.destination);
    if (!compatibility_status.ok()) {
        return compatibility_status;
    }

    auto status = cache_writer_->mallocWritebackBlocks(destination_resource,
                                                       static_cast<size_t>(request.manifest.reusable_block_count));
    if (!status.ok()) {
        return status;
    }
    auto transfer_status = transfer_client_->transfer(buildTransferPlan(request, destination_resource));
    if (!transfer_status.ok()) {
        cache_writer_->freeWritebackBlocks(destination_resource);
        return transfer_status;
    }
    cache_writer_->commitWritebackBlocks(destination_resource, request.manifest.cache_keys, false);
    return absl::OkStatus();
}

PdKvWritebackTransferPlan
PdKvWritebackManager::buildTransferPlan(const PdKvWritebackLaunchRequest& request,
                                        const BatchKVCacheResourcePtr&    destination_resource) const {
    PdKvWritebackTransferPlan plan;
    plan.request_id               = request.manifest.request_id;
    plan.request_key              = request.manifest.request_key;
    plan.deadline_ms              = request.deadline_ms;
    plan.layer_count              = request.destination.layer_count;
    plan.group_count              = request.destination.group_count;
    plan.remote_tp_size           = request.source.partition_count;
    plan.cache_keys               = request.manifest.cache_keys;
    plan.decode_group_block_ids   = request.manifest.group_block_ids;
    plan.prefill_group_block_ids  = extractPdKvWritebackGroupBlockIds(destination_resource);
    plan.layer_to_group_id        = request.destination.layer_to_group_id;
    plan.prefill_transfer_servers = parsePdKvWritebackTransferServers(request.prefill_worker_addrs);
    return plan;
}

}  // namespace rtp_llm

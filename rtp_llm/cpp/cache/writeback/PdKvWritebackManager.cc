#include "rtp_llm/cpp/cache/writeback/PdKvWritebackManager.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/cache/writeback/PdKvWritebackMetrics.h"
#include "rtp_llm/cpp/cache/writeback/PdKvWritebackTopology.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
namespace {

std::string pdKvRoleName(RoleType role_type) {
    if (role_type == RoleType::PREFILL) {
        return "prefill";
    }
    if (role_type == RoleType::DECODE) {
        return "decode";
    }
    return "unknown";
}

std::string pdKvBackendName(const PDSepConfig& pd_config) {
    return pd_config.cache_store_rdma_mode ? "rdma" : "tcp";
}

PdKvWritebackMetricExtraTags pdKvBackendTags(const PDSepConfig& pd_config) {
    return {{"backend", pdKvBackendName(pd_config)}};
}

int64_t buildPdKvWritebackDeadlineMs(const PDSepConfig& pd_config) {
    static constexpr int64_t kDefaultWritebackTimeoutMs = 5000;
    const int64_t            timeout_ms =
        pd_config.load_cache_timeout_ms > 0 ? pd_config.load_cache_timeout_ms : kDefaultWritebackTimeoutMs;
    return currentTimeMs() + timeout_ms;
}

PdKvWritebackTopologyInput buildTopologyInput(const PdKvWritebackLaunchRequest& request) {
    PdKvWritebackTopologyInput input;
    input.local_tp_size               = request.destination.partition_count;
    input.source_partition_count      = request.source.partition_count;
    input.destination_partition_count = request.destination.partition_count;
    input.decode_grpc_addrs           = request.decode_worker_addrs;
    input.prefill_grpc_addrs          = request.source_prefill_grpc_addrs;
    input.prefill_worker_addrs        = request.prefill_worker_addrs;
    return input;
}

absl::Status keepLocalTpRankMappings(PdKvWritebackTopologyPlan& topology, int32_t local_tp_rank) {
    if (local_tp_rank < 0) {
        return absl::FailedPreconditionError("local tp rank must be non-negative");
    }
    topology.mappings.erase(std::remove_if(topology.mappings.begin(),
                                           topology.mappings.end(),
                                           [local_tp_rank](const PdKvWritebackRankMapping& mapping) {
                                               return mapping.decode_rank != local_tp_rank;
                                           }),
                            topology.mappings.end());
    if (topology.mappings.empty()) {
        return absl::FailedPreconditionError("local tp rank has no writeback mapping");
    }
    return absl::OkStatus();
}

void dropLocalDecodeRankMappings(PdKvWritebackTopologyPlan& topology, int32_t local_tp_rank) {
    topology.mappings.erase(std::remove_if(topology.mappings.begin(),
                                           topology.mappings.end(),
                                           [local_tp_rank](const PdKvWritebackRankMapping& mapping) {
                                               return mapping.decode_rank == local_tp_rank;
                                           }),
                            topology.mappings.end());
}

absl::StatusOr<BatchKVCacheResourcePtr> buildLocalDecodeSourceResource(const PdKvWritebackLaunchRequest& request) {
    const int group_count = request.source.group_count > 0 ? request.source.group_count :
                                                             static_cast<int>(request.manifest.group_block_ids.size());
    if (group_count <= 0 || group_count != static_cast<int>(request.manifest.group_block_ids.size())) {
        return absl::InvalidArgumentError("invalid writeback source group count");
    }
    if (request.source.layer_count <= 0) {
        return absl::InvalidArgumentError("invalid writeback source layer count");
    }

    std::vector<int> layer_to_group_id;
    layer_to_group_id.assign(request.source.layer_to_group_id.begin(), request.source.layer_to_group_id.end());

    auto source_resource = std::make_shared<BatchKVCacheResource>();
    source_resource->resetBatchSize(1);
    source_resource->initBatchGroups(0, group_count, request.source.layer_count, layer_to_group_id);
    for (int group_id = 0; group_id < group_count; ++group_id) {
        source_resource->setBatchBlocks(0, group_id, request.manifest.group_block_ids[static_cast<size_t>(group_id)]);
    }
    source_resource->setBatchCacheKeys(0, request.manifest.cache_keys);
    return source_resource;
}

int groupIdForLayer(const PdKvWritebackCompatibility& compatibility, int layer_id) {
    if (compatibility.layer_to_group_id.empty()) {
        return 0;
    }
    return compatibility.layer_to_group_id[static_cast<size_t>(layer_id)];
}

std::string compactMissingLayers(const std::vector<int>& missing_layers) {
    std::string             result;
    static constexpr size_t kMaxLoggedLayers = 16;
    const size_t            limit            = std::min(missing_layers.size(), kMaxLoggedLayers);
    for (size_t i = 0; i < limit; ++i) {
        if (!result.empty()) {
            result += ",";
        }
        result += std::to_string(missing_layers[i]);
    }
    if (missing_layers.size() > limit) {
        result += ",...";
    }
    return result;
}

std::string pendingReceiveKey(const PdKvWritebackLaunchRequest& request) {
    return std::to_string(request.manifest.request_id) + ":" + request.manifest.request_key;
}

absl::Status validateWritebackSourceComplete(const PdKvWritebackLaunchRequest& request,
                                             const BatchKVCacheResourcePtr&    source_resource) {
    if (!source_resource || source_resource->batchSize() != 1) {
        return absl::FailedPreconditionError("source_incomplete: source_resource_invalid");
    }
    if (request.manifest.reusable_block_count <= 0) {
        return absl::OkStatus();
    }
    if (request.manifest.cache_keys.size() < static_cast<size_t>(request.manifest.reusable_block_count)) {
        return absl::FailedPreconditionError("source_incomplete: cache_keys_shorter_than_reusable_blocks");
    }
    if (request.source.layer_count <= 0 || request.source.group_count <= 0) {
        return absl::FailedPreconditionError("source_incomplete: invalid_source_layout");
    }
    if (!request.source.layer_to_group_id.empty()
        && request.source.layer_to_group_id.size() < static_cast<size_t>(request.source.layer_count)) {
        return absl::FailedPreconditionError("source_incomplete: layer_to_group_id_shorter_than_layer_count");
    }
    if (source_resource->groupNums() != request.source.group_count) {
        return absl::FailedPreconditionError("source_incomplete: group_count_mismatch");
    }

    std::vector<int> missing_layers;
    missing_layers.reserve(static_cast<size_t>(request.source.layer_count));
    const auto reusable_block_count = static_cast<size_t>(request.manifest.reusable_block_count);
    for (int layer_id = 0; layer_id < request.source.layer_count; ++layer_id) {
        const int group_id = groupIdForLayer(request.source, layer_id);
        if (group_id < 0 || group_id >= request.source.group_count) {
            return absl::FailedPreconditionError("source_incomplete: invalid_layer_group_mapping");
        }
        const auto& blocks = source_resource->blocks(0, group_id);
        if (blocks.size() < reusable_block_count) {
            missing_layers.push_back(layer_id);
            continue;
        }
        const bool has_missing_block = std::any_of(blocks.begin(),
                                                   blocks.begin() + reusable_block_count,
                                                   [](BlockIdxType block_id) { return isNullBlockIdx(block_id); });
        if (has_missing_block) {
            missing_layers.push_back(layer_id);
        }
    }
    if (!missing_layers.empty()) {
        RTP_LLM_LOG_WARNING(
            "PD KV writeback source incomplete, request_id=%ld, reusable_blocks=%ld, missing_layers=%zu, layers=%s",
            request.manifest.request_id,
            request.manifest.reusable_block_count,
            missing_layers.size(),
            compactMissingLayers(missing_layers).c_str());
        return absl::FailedPreconditionError("source_incomplete: missing source layer blocks");
    }

    RTP_LLM_LOG_INFO(
        "PD KV writeback source complete, request_id=%ld, reusable_blocks=%ld, layers=%d, groups=%d, expected_transfers_per_target=%d",
        request.manifest.request_id,
        request.manifest.reusable_block_count,
        request.source.layer_count,
        request.source.group_count,
        request.source.layer_count);
    return absl::OkStatus();
}

absl::Status fillExplicitTransferTargets(PdKvWritebackTransferPlan& plan, const PdKvWritebackTopologyPlan& topology) {
    plan.prefill_transfer_targets.clear();
    plan.prefill_transfer_targets.reserve(topology.mappings.size());
    for (const auto& mapping : topology.mappings) {
        auto target = parsePdKvWritebackTransferTarget(mapping.prefill_worker_addr,
                                                       mapping.local_partition_count,
                                                       mapping.local_partition_id,
                                                       mapping.remote_partition_count,
                                                       mapping.remote_partition_id,
                                                       mapping.decode_rank,
                                                       mapping.prefill_rank);
        if (!target.ok()) {
            return target.status();
        }
        plan.prefill_transfer_targets.push_back(target.value());
    }
    if (plan.prefill_transfer_targets.empty()) {
        return absl::FailedPreconditionError("writeback transfer targets are empty");
    }
    return absl::OkStatus();
}

absl::StatusOr<PdKvWritebackTransferPlan>
buildDecodeTransferPlanFromRequest(const PdKvWritebackLaunchRequest& request,
                                   const PdKvWritebackTopologyPlan*  topology) {
    PdKvWritebackTransferPlan plan;
    plan.request_id               = request.manifest.request_id;
    plan.request_key              = request.manifest.request_key;
    plan.deadline_ms              = request.deadline_ms;
    plan.layer_count              = request.source.layer_count;
    plan.group_count              = request.source.group_count;
    plan.remote_tp_size           = request.destination.partition_count;
    plan.cache_keys               = request.manifest.cache_keys;
    plan.decode_group_block_ids   = request.manifest.group_block_ids;
    plan.layer_to_group_id        = request.source.layer_to_group_id;
    plan.prefill_transfer_servers = parsePdKvWritebackTransferServers(request.prefill_worker_addrs);
    if (topology) {
        auto status = fillExplicitTransferTargets(plan, *topology);
        if (!status.ok()) {
            return status;
        }
    }
    return plan;
}

absl::Status sendOnDecodeWithClient(const PDSepConfig&                 pd_config,
                                    PdKvWritebackTransferClient*       transfer_client,
                                    const kmonitor::MetricsReporterPtr metrics_reporter,
                                    const PdKvWritebackLaunchRequest&  request,
                                    const BatchKVCacheResourcePtr&     source_resource,
                                    const PdKvWritebackTopologyPlan*   topology = nullptr) {
    const int64_t                 send_begin_us = currentTimeUs();
    PdKvWritebackMetricsCollector collector;
    collector.transfer_qps = true;
    collector.block_count  = request.manifest.reusable_block_count;
    collector.token_count  = request.manifest.final_token_count;
    auto report_send       = [&](const std::string& status, const std::string& reason) {
        collector.transfer_latency_us = currentTimeUs() - send_begin_us;
        if (status == "failed") {
            collector.transfer_failed_qps = true;
        }
        reportPdKvWritebackMetric(metrics_reporter,
                                  collector,
                                  "decode_send",
                                  status,
                                  reason,
                                  pdKvRoleName(pd_config.role_type),
                                  request.source.partition_count,
                                  "tp_equal",
                                  pdKvBackendTags(pd_config));
    };

    if (!pd_config.enable_pd_kv_cache_writeback) {
        report_send("failed", "disabled");
        return absl::FailedPreconditionError("disabled");
    }
    if (!transfer_client) {
        report_send("failed", "transfer_client_null");
        return absl::FailedPreconditionError("transfer_client is null");
    }
    if (!source_resource || source_resource->batchSize() != 1) {
        report_send("failed", "source_resource_invalid");
        return absl::FailedPreconditionError("source_resource is invalid");
    }
    if (request.manifest.reusable_block_count == 0) {
        report_send("skipped", "empty_manifest");
        return absl::OkStatus();
    }
    if (request.manifest.cache_keys.size() < static_cast<size_t>(request.manifest.reusable_block_count)) {
        report_send("failed", "cache_keys_shorter_than_reusable_blocks");
        return absl::InvalidArgumentError("cache_keys shorter than reusable_block_count");
    }
    auto compatibility_status = validatePdKvWritebackCompatibility(request.source, request.destination);
    if (!compatibility_status.ok()) {
        report_send("failed", "compatibility_mismatch");
        return compatibility_status;
    }
    auto source_complete_status = validateWritebackSourceComplete(request, source_resource);
    if (!source_complete_status.ok()) {
        report_send("skipped", "source_incomplete");
        return source_complete_status;
    }

    auto plan_status = buildDecodeTransferPlanFromRequest(request, topology);
    if (!plan_status.ok()) {
        report_send("failed", "transfer_target_invalid");
        return plan_status.status();
    }
    auto plan                   = std::move(plan_status).value();
    plan.decode_group_block_ids = extractPdKvWritebackGroupBlockIds(source_resource);
    if (plan.decode_group_block_ids.empty()) {
        report_send("failed", "source_blocks_empty");
        return absl::FailedPreconditionError("source blocks are empty");
    }

    RTP_LLM_LOG_INFO(
        "PD KV writeback decode send start, request_id=%ld, reusable_blocks=%ld, cache_keys=%zu, backend=%s, explicit_targets=%zu, fallback_servers=%zu",
        request.manifest.request_id,
        request.manifest.reusable_block_count,
        request.manifest.cache_keys.size(),
        pdKvBackendName(pd_config).c_str(),
        plan.prefill_transfer_targets.size(),
        plan.prefill_transfer_servers.size());
    auto status = transfer_client->transfer(plan);
    if (!status.ok()) {
        report_send("failed", "transfer_failed");
        return status;
    }
    RTP_LLM_LOG_INFO("PD KV writeback decode send done, request_id=%ld, reusable_blocks=%ld",
                     request.manifest.request_id,
                     request.manifest.reusable_block_count);
    report_send("success", "ok");
    return absl::OkStatus();
}

}  // namespace

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

PdKvWritebackManager::PdKvWritebackManager(const PDSepConfig&                           pd_config,
                                           PdKvWritebackCacheWriter*                    cache_writer,
                                           std::shared_ptr<PdKvWritebackTransferClient> transfer_client,
                                           std::shared_ptr<PdKvWritebackRpcClient>      rpc_client):
    PdKvWritebackManager(pd_config, cache_writer, std::move(transfer_client), std::move(rpc_client), {}, nullptr) {}

PdKvWritebackManager::PdKvWritebackManager(const PDSepConfig&                           pd_config,
                                           PdKvWritebackCacheWriter*                    cache_writer,
                                           std::shared_ptr<PdKvWritebackTransferClient> transfer_client,
                                           std::shared_ptr<PdKvWritebackRpcClient>      rpc_client,
                                           std::vector<std::string>                     decode_worker_grpc_addrs,
                                           kmonitor::MetricsReporterPtr                 metrics_reporter):
    pd_config_(pd_config),
    cache_writer_(cache_writer),
    transfer_client_(transfer_client.get()),
    rpc_client_(rpc_client.get()),
    owned_transfer_client_(std::move(transfer_client)),
    owned_rpc_client_(std::move(rpc_client)),
    decode_worker_grpc_addrs_(std::move(decode_worker_grpc_addrs)),
    metrics_reporter_(std::move(metrics_reporter)) {}

PdKvWritebackLaunchResult PdKvWritebackManager::launchFromDecode(const PdKvWritebackLaunchRequest& request) const {
    const int64_t launch_begin_us = currentTimeUs();
    auto          report_launch   = [&](PdKvWritebackLaunchStatus status, const std::string& reason) {
        PdKvWritebackMetricsCollector collector;
        collector.launch_qps        = true;
        collector.launch_latency_us = currentTimeUs() - launch_begin_us;
        collector.block_count       = request.manifest.reusable_block_count;
        collector.token_count       = request.manifest.final_token_count;
        collector.launch_rate_valid = reason != "disabled" && reason != "empty_manifest";
        collector.launch_rate       = status == PdKvWritebackLaunchStatus::Started ? 1.0 : 0.0;
        if (status == PdKvWritebackLaunchStatus::Skipped) {
            collector.launch_skipped_qps = true;
        } else if (status == PdKvWritebackLaunchStatus::Failed) {
            collector.launch_failed_qps = true;
        }
        reportPdKvWritebackMetric(metrics_reporter_,
                                  collector,
                                  "launch",
                                  status == PdKvWritebackLaunchStatus::Started ? "started" :
                                             status == PdKvWritebackLaunchStatus::Skipped ? "skipped" :
                                                                                            "failed",
                                  reason,
                                  pdKvRoleName(pd_config_.role_type),
                                  request.destination.partition_count,
                                  "tp_equal",
                                  pdKvBackendTags(pd_config_));
    };

    if (!pd_config_.enable_pd_kv_cache_writeback) {
        report_launch(PdKvWritebackLaunchStatus::Skipped, "disabled");
        return {PdKvWritebackLaunchStatus::Skipped, "disabled"};
    }
    if (request.manifest.reusable_block_count == 0) {
        report_launch(PdKvWritebackLaunchStatus::Skipped, "empty_manifest");
        return {PdKvWritebackLaunchStatus::Skipped, "empty_manifest"};
    }

    auto status = validatePdKvWritebackCompatibility(request.source, request.destination);
    if (!status.ok()) {
        report_launch(PdKvWritebackLaunchStatus::Skipped, "compatibility_mismatch");
        return {PdKvWritebackLaunchStatus::Skipped, std::string(status.message())};
    }
    auto source_resource_status = buildLocalDecodeSourceResource(request);
    if (!source_resource_status.ok()) {
        report_launch(PdKvWritebackLaunchStatus::Skipped, "source_incomplete");
        return {PdKvWritebackLaunchStatus::Skipped, "source_incomplete"};
    }
    auto source_complete_status = validateWritebackSourceComplete(request, source_resource_status.value());
    if (!source_complete_status.ok()) {
        report_launch(PdKvWritebackLaunchStatus::Skipped, "source_incomplete");
        return {PdKvWritebackLaunchStatus::Skipped, "source_incomplete"};
    }
    auto request_for_topology = request;
    if (request_for_topology.decode_worker_addrs.empty()) {
        request_for_topology.decode_worker_addrs = decode_worker_grpc_addrs_;
    }
    auto topology_status = buildPdKvWritebackTopology(buildTopologyInput(request_for_topology));
    if (!topology_status.ok()) {
        const std::string reason = topology_status.status().code() == absl::StatusCode::kUnimplemented ?
                                       "unsupported_topology" :
                                       "topology_mismatch";
        report_launch(PdKvWritebackLaunchStatus::Skipped, reason);
        return {PdKvWritebackLaunchStatus::Skipped, std::string(topology_status.status().message())};
    }

    auto* rpc_client = owned_rpc_client_ ? owned_rpc_client_.get() : rpc_client_;
    if (!rpc_client) {
        report_launch(PdKvWritebackLaunchStatus::Failed, "rpc_client_null");
        return {PdKvWritebackLaunchStatus::Failed, "rpc_client_null"};
    }

    auto request_copy         = std::move(request_for_topology);
    request_copy.deadline_ms  = buildPdKvWritebackDeadlineMs(pd_config_);
    auto topology             = std::move(topology_status).value();
    auto local_topology       = topology;
    auto local_mapping_status = keepLocalTpRankMappings(local_topology, request_copy.local_tp_rank);
    if (!local_mapping_status.ok()) {
        report_launch(PdKvWritebackLaunchStatus::Skipped, "topology_mismatch");
        return {PdKvWritebackLaunchStatus::Skipped, std::string(local_mapping_status.message())};
    }
    auto remote_decode_topology = topology;
    dropLocalDecodeRankMappings(remote_decode_topology, request_copy.local_tp_rank);
    auto* transfer_client = owned_transfer_client_ ? owned_transfer_client_.get() : transfer_client_;
    if (!transfer_client) {
        report_launch(PdKvWritebackLaunchStatus::Failed, "transfer_client_null");
        return {PdKvWritebackLaunchStatus::Failed, "transfer_client_null"};
    }
    auto rpc_client_owner      = owned_rpc_client_;
    auto transfer_client_owner = owned_transfer_client_;
    auto pd_config             = pd_config_;
    auto metrics_reporter      = metrics_reporter_;
    {
        std::lock_guard<std::mutex> lock(writeback_tasks_mutex_);
        pruneCompletedWritebackTasksLocked();
        writeback_tasks_.push_back(std::async(
            std::launch::async,
            [request_copy,
             topology,
             local_topology,
             remote_decode_topology,
             rpc_client,
             rpc_client_owner,
             transfer_client,
             transfer_client_owner,
             pd_config,
             metrics_reporter]() {
                auto receive_task =
                    std::async(std::launch::async, [request_copy, topology, rpc_client, rpc_client_owner]() {
                        return rpc_client->requestPrefillReceive(request_copy, topology);
                    });
                const bool                has_remote_decode_mappings = !remote_decode_topology.mappings.empty();
                std::future<absl::Status> remote_decode_task;
                if (has_remote_decode_mappings) {
                    remote_decode_task = std::async(
                        std::launch::async, [request_copy, remote_decode_topology, rpc_client, rpc_client_owner]() {
                            return rpc_client->requestDecodeSend(request_copy, remote_decode_topology);
                        });
                }
                auto source_resource      = buildLocalDecodeSourceResource(request_copy);
                auto send_status          = source_resource.ok() ? sendOnDecodeWithClient(pd_config,
                                                                                 transfer_client,
                                                                                 metrics_reporter,
                                                                                 request_copy,
                                                                                 source_resource.value(),
                                                                                 &local_topology) :
                                                                   source_resource.status();
                auto receive_status       = receive_task.get();
                auto remote_decode_status = has_remote_decode_mappings ? remote_decode_task.get() : absl::OkStatus();
                if (!receive_status.ok()) {
                    RTP_LLM_LOG_WARNING("PD KV writeback prefill receive fanout failed, request_id=%ld, error=%s",
                                        request_copy.manifest.request_id,
                                        receive_status.ToString().c_str());
                }
                if (!remote_decode_status.ok()) {
                    RTP_LLM_LOG_WARNING("PD KV writeback remote decode send fanout failed, request_id=%ld, error=%s",
                                        request_copy.manifest.request_id,
                                        remote_decode_status.ToString().c_str());
                }
                if (!send_status.ok()) {
                    RTP_LLM_LOG_WARNING("PD KV writeback decode send fanout failed, request_id=%ld, error=%s",
                                        request_copy.manifest.request_id,
                                        send_status.ToString().c_str());
                }
                if (receive_status.ok() && remote_decode_status.ok() && send_status.ok()) {
                    auto commit_status = rpc_client->requestPrefillCommit(request_copy, topology);
                    if (!commit_status.ok()) {
                        RTP_LLM_LOG_WARNING("PD KV writeback prefill commit fanout failed, request_id=%ld, error=%s",
                                            request_copy.manifest.request_id,
                                            commit_status.ToString().c_str());
                    }
                } else {
                    auto abort_status = rpc_client->requestPrefillAbort(request_copy, topology);
                    if (!abort_status.ok()) {
                        RTP_LLM_LOG_WARNING("PD KV writeback prefill abort fanout failed, request_id=%ld, error=%s",
                                            request_copy.manifest.request_id,
                                            abort_status.ToString().c_str());
                    }
                }
            }));
    }
    report_launch(PdKvWritebackLaunchStatus::Started, "started");
    return {PdKvWritebackLaunchStatus::Started, "started"};
}

void PdKvWritebackManager::waitForWritebackTasksForTest() const {
    std::vector<std::future<void>> tasks;
    {
        std::lock_guard<std::mutex> lock(writeback_tasks_mutex_);
        tasks.swap(writeback_tasks_);
    }
    for (auto& task : tasks) {
        task.get();
    }
}

void PdKvWritebackManager::pruneCompletedWritebackTasksLocked() const {
    writeback_tasks_.erase(std::remove_if(writeback_tasks_.begin(),
                                          writeback_tasks_.end(),
                                          [](std::future<void>& task) {
                                              return task.valid()
                                                     && task.wait_for(std::chrono::milliseconds(0))
                                                            == std::future_status::ready;
                                          }),
                           writeback_tasks_.end());
}

size_t PdKvWritebackManager::trackedWritebackTaskCountForTest() const {
    std::lock_guard<std::mutex> lock(writeback_tasks_mutex_);
    return writeback_tasks_.size();
}

size_t PdKvWritebackManager::completedWritebackTaskCountForTest() const {
    std::lock_guard<std::mutex> lock(writeback_tasks_mutex_);
    return std::count_if(writeback_tasks_.begin(), writeback_tasks_.end(), [](std::future<void>& task) {
        return task.valid() && task.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready;
    });
}

absl::Status PdKvWritebackManager::receiveOnPrefill(const PdKvWritebackLaunchRequest& request,
                                                    const BatchKVCacheResourcePtr&    destination_resource) {
    auto prepare_status = prepareReceiveOnPrefill(request, destination_resource);
    if (!prepare_status.ok()) {
        return prepare_status;
    }
    auto commit_status = commitReceiveOnPrefill(request);
    if (!commit_status.ok()) {
        abortReceiveOnPrefill(request).IgnoreError();
    }
    return commit_status;
}

absl::Status PdKvWritebackManager::prepareReceiveOnPrefill(const PdKvWritebackLaunchRequest& request,
                                                           const BatchKVCacheResourcePtr&    destination_resource) {
    const int64_t                 receive_begin_us = currentTimeUs();
    PdKvWritebackMetricsCollector collector;
    collector.receive_qps = true;
    collector.block_count = request.manifest.reusable_block_count;
    collector.token_count = request.manifest.final_token_count;
    auto report_receive   = [&](const std::string& status, const std::string& reason) {
        collector.receive_latency_us = currentTimeUs() - receive_begin_us;
        if (status == "failed") {
            collector.receive_failed_qps = true;
        }
        reportPdKvWritebackMetric(metrics_reporter_,
                                  collector,
                                  "prefill_receive",
                                  status,
                                  reason,
                                  pdKvRoleName(pd_config_.role_type),
                                  request.destination.partition_count,
                                  "tp_equal",
                                  pdKvBackendTags(pd_config_));
    };

    if (!pd_config_.enable_pd_kv_cache_writeback) {
        report_receive("failed", "disabled");
        return absl::FailedPreconditionError("disabled");
    }
    if (!cache_writer_) {
        report_receive("failed", "cache_writer_null");
        return absl::FailedPreconditionError("cache_writer is null");
    }
    if (!transfer_client_) {
        report_receive("failed", "transfer_client_null");
        return absl::FailedPreconditionError("transfer_client is null");
    }
    if (request.manifest.reusable_block_count == 0) {
        report_receive("skipped", "empty_manifest");
        return absl::OkStatus();
    }
    if (request.manifest.cache_keys.size() < static_cast<size_t>(request.manifest.reusable_block_count)) {
        report_receive("failed", "cache_keys_shorter_than_reusable_blocks");
        return absl::InvalidArgumentError("cache_keys shorter than reusable_block_count");
    }
    auto compatibility_status = validatePdKvWritebackCompatibility(request.source, request.destination);
    if (!compatibility_status.ok()) {
        report_receive("failed", "compatibility_mismatch");
        return compatibility_status;
    }

    RTP_LLM_LOG_INFO("PD KV writeback receive start, request_id=%ld, reusable_blocks=%ld, cache_keys=%zu, backend=%s",
                     request.manifest.request_id,
                     request.manifest.reusable_block_count,
                     request.manifest.cache_keys.size(),
                     pdKvBackendName(pd_config_).c_str());

    const int64_t malloc_begin_us = currentTimeUs();
    auto          status          = cache_writer_->mallocWritebackBlocks(destination_resource,
                                                       static_cast<size_t>(request.manifest.reusable_block_count));
    collector.malloc_latency_us   = currentTimeUs() - malloc_begin_us;
    if (!status.ok()) {
        report_receive("failed", "malloc_failed");
        return status;
    }
    const int64_t transfer_begin_us = currentTimeUs();
    collector.transfer_qps          = true;
    auto transfer_status            = transfer_client_->transfer(buildTransferPlan(request, destination_resource));
    collector.transfer_latency_us   = currentTimeUs() - transfer_begin_us;
    if (!transfer_status.ok()) {
        collector.transfer_failed_qps = true;
        cache_writer_->freeWritebackBlocks(destination_resource);
        report_receive("failed", "transfer_failed");
        return transfer_status;
    }

    const auto pending_key = pendingReceiveKey(request);
    {
        std::lock_guard<std::mutex> lock(pending_receives_mutex_);
        auto inserted = pending_receives_.emplace(pending_key, PendingReceive{request, destination_resource});
        if (!inserted.second) {
            cache_writer_->freeWritebackBlocks(destination_resource);
            report_receive("failed", "duplicate_pending_receive");
            return absl::AlreadyExistsError("duplicate pending writeback receive");
        }
    }
    RTP_LLM_LOG_INFO("PD KV writeback receive prepared, request_id=%ld, reusable_blocks=%ld",
                     request.manifest.request_id,
                     request.manifest.reusable_block_count);
    report_receive("success", "ok");
    return absl::OkStatus();
}

absl::Status PdKvWritebackManager::commitReceiveOnPrefill(const PdKvWritebackLaunchRequest& request) {
    if (!pd_config_.enable_pd_kv_cache_writeback) {
        return absl::FailedPreconditionError("disabled");
    }
    if (!cache_writer_) {
        return absl::FailedPreconditionError("cache_writer is null");
    }

    PendingReceive pending;
    const auto     pending_key = pendingReceiveKey(request);
    {
        std::lock_guard<std::mutex> lock(pending_receives_mutex_);
        auto                        it = pending_receives_.find(pending_key);
        if (it == pending_receives_.end()) {
            return absl::NotFoundError("pending writeback receive not found");
        }
        pending = std::move(it->second);
        pending_receives_.erase(it);
    }

    const int64_t commit_begin_us = currentTimeUs();
    cache_writer_->commitWritebackBlocks(pending.destination_resource, pending.request.manifest.cache_keys, false);
    cache_writer_->freeWritebackBlocks(pending.destination_resource);
    RTP_LLM_LOG_INFO("PD KV writeback receive commit, request_id=%ld, reusable_blocks=%ld, cost_us=%ld",
                     pending.request.manifest.request_id,
                     pending.request.manifest.reusable_block_count,
                     currentTimeUs() - commit_begin_us);
    return absl::OkStatus();
}

absl::Status PdKvWritebackManager::abortReceiveOnPrefill(const PdKvWritebackLaunchRequest& request) {
    if (!cache_writer_) {
        return absl::FailedPreconditionError("cache_writer is null");
    }
    PendingReceive pending;
    const auto     pending_key = pendingReceiveKey(request);
    {
        std::lock_guard<std::mutex> lock(pending_receives_mutex_);
        auto                        it = pending_receives_.find(pending_key);
        if (it == pending_receives_.end()) {
            return absl::OkStatus();
        }
        pending = std::move(it->second);
        pending_receives_.erase(it);
    }
    cache_writer_->freeWritebackBlocks(pending.destination_resource);
    RTP_LLM_LOG_INFO("PD KV writeback receive abort, request_id=%ld, reusable_blocks=%ld",
                     pending.request.manifest.request_id,
                     pending.request.manifest.reusable_block_count);
    return absl::OkStatus();
}

absl::Status PdKvWritebackManager::sendOnDecode(const PdKvWritebackLaunchRequest& request,
                                                const BatchKVCacheResourcePtr&    source_resource) const {
    auto* transfer_client      = owned_transfer_client_ ? owned_transfer_client_.get() : transfer_client_;
    auto  request_for_topology = request;
    if (request_for_topology.decode_worker_addrs.empty()) {
        request_for_topology.decode_worker_addrs = decode_worker_grpc_addrs_;
    }
    auto topology_status = buildPdKvWritebackTopology(buildTopologyInput(request_for_topology));
    if (!topology_status.ok()) {
        return topology_status.status();
    }
    auto local_topology = std::move(topology_status).value();
    auto status         = keepLocalTpRankMappings(local_topology, request_for_topology.local_tp_rank);
    if (!status.ok()) {
        return status;
    }
    return sendOnDecodeWithClient(
        pd_config_, transfer_client, metrics_reporter_, request_for_topology, source_resource, &local_topology);
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

#include "rtp_llm/cpp/cache/writeback/PdKvWritebackManager.h"

#include <future>
#include <utility>

#include "absl/status/status.h"
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
    auto report_launch = [&](PdKvWritebackLaunchStatus status, const std::string& reason) {
        PdKvWritebackMetricsCollector collector;
        collector.launch_qps        = true;
        collector.launch_latency_us = currentTimeUs() - launch_begin_us;
        collector.block_count       = request.manifest.reusable_block_count;
        collector.token_count       = request.manifest.final_token_count;
        if (status == PdKvWritebackLaunchStatus::Skipped) {
            collector.launch_skipped_qps = true;
        } else if (status == PdKvWritebackLaunchStatus::Failed) {
            collector.launch_failed_qps = true;
        }
        reportPdKvWritebackMetric(metrics_reporter_,
                                  collector,
                                  "launch",
                                  status == PdKvWritebackLaunchStatus::Started ? "started" :
                                      status == PdKvWritebackLaunchStatus::Skipped ? "skipped" : "failed",
                                  reason,
                                  pdKvRoleName(pd_config_.role_type),
                                  request.destination.partition_count,
                                  "tp_equal");
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

    auto request_copy     = std::move(request_for_topology);
    auto topology         = std::move(topology_status).value();
    auto rpc_client_owner = owned_rpc_client_;
    {
        std::lock_guard<std::mutex> lock(writeback_tasks_mutex_);
        writeback_tasks_.push_back(std::async(std::launch::async,
                                              [request_copy, topology, rpc_client, rpc_client_owner]() {
                                                  auto receive_status =
                                                      rpc_client->requestPrefillReceive(request_copy, topology);
                                                  auto send_status = rpc_client->requestDecodeSend(request_copy, topology);
                                                  if (!receive_status.ok()) {
                                                      RTP_LLM_LOG_WARNING(
                                                          "PD KV writeback prefill receive fanout failed, request_id=%ld, error=%s",
                                                          request_copy.manifest.request_id,
                                                          receive_status.ToString().c_str());
                                                  }
                                                  if (!send_status.ok()) {
                                                      RTP_LLM_LOG_WARNING(
                                                          "PD KV writeback decode send fanout failed, request_id=%ld, error=%s",
                                                          request_copy.manifest.request_id,
                                                          send_status.ToString().c_str());
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

    RTP_LLM_LOG_INFO("PD KV writeback receive start, request_id=%ld, reusable_blocks=%ld, cache_keys=%zu",
                     request.manifest.request_id,
                     request.manifest.reusable_block_count,
                     request.manifest.cache_keys.size());

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
    cache_writer_->freeWritebackBlocks(destination_resource);
    RTP_LLM_LOG_INFO("PD KV writeback receive commit, request_id=%ld, reusable_blocks=%ld",
                     request.manifest.request_id,
                     request.manifest.reusable_block_count);
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

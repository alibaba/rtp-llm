#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorSchedulerDecode.h"

#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "autil/LockFreeThreadPool.h"
#include <algorithm>
#include <memory>
#include <optional>

namespace rtp_llm {
namespace {
// Keep kickoff bounded: connection acquisition/reconnection may block, while
// completion polling remains isolated on P2PConnectorAsyncReadContextChecker.
constexpr size_t kAsyncReadThreadCount = 4;
constexpr size_t kAsyncReadQueueSize   = 1024;
}  // namespace

P2PConnectorSchedulerDecode::P2PConnectorSchedulerDecode(
    P2PConnectorSchedulerConfig                config,
    const kmonitor::MetricsReporterPtr&        metrics_reporter,
    const std::shared_ptr<P2PBroadcastClient>& tp_broadcast_client):
    config_(std::move(config)), metrics_reporter_(metrics_reporter), tp_broadcast_client_(tp_broadcast_client) {}

P2PConnectorSchedulerDecode::~P2PConnectorSchedulerDecode() {
    if (async_read_pool_) {
        async_read_pool_->stop(autil::ThreadPool::STOP_AFTER_QUEUE_EMPTY);
        async_read_pool_->join();
        async_read_pool_.reset();
    }
    if (checker_) {
        checker_->stop();
    }
}

bool P2PConnectorSchedulerDecode::init(const std::string& process_id) {
    server_caller_ = std::make_shared<PrefillLoadCaller>(config_.worker_addrs);

    auto async_read_pool = std::make_shared<autil::LockFreeThreadPool>(
        kAsyncReadThreadCount, kAsyncReadQueueSize, nullptr, "P2PAsyncReadKickoff");
    if (!async_read_pool->start()) {
        RTP_LLM_LOG_ERROR("P2PConnectorSchedulerDecode init failed: async read pool start failed");
        return false;
    }
    async_read_pool_ = std::move(async_read_pool);

    checker_ = std::make_shared<P2PConnectorAsyncReadContextChecker>();
    if (!checker_->init(metrics_reporter_, tp_broadcast_client_)) {
        RTP_LLM_LOG_ERROR("P2PConnectorSchedulerDecode init failed: checker init failed");
        async_read_pool_->stop();
        async_read_pool_.reset();
        return false;
    }

    return true;
}

void P2PConnectorSchedulerDecode::stopChecker() {
    if (checker_) {
        checker_->stop();
    }
}

void P2PConnectorSchedulerDecode::cancel(const std::shared_ptr<P2PConnectorAsyncReadContext>& context) {
    if (context) {
        context->cancel(tp_broadcast_client_);
    }
}

P2PConnectorSchedulerDecode::AsyncReadResult P2PConnectorSchedulerDecode::asyncRead(
    const KVCacheResourcePtr& resource,
    const std::shared_ptr<Meta>& meta,
    const std::pair<int, int>& block_range,
    bool no_transfer) {
    if (!meta || !resource) {
        RTP_LLM_LOG_WARNING("asyncRead: meta or resource is null");
        return {nullptr, ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "meta or resource is null")};
    }

    // Extract routing from Meta::p2pRouting()
    auto routing = meta->p2pRouting();
    if (!routing.has_value()) {
        RTP_LLM_LOG_WARNING("asyncRead: meta->p2pRouting() returned nullopt");
        return {
            nullptr,
            ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "meta->p2pRouting() returned nullopt")};
    }

    const int64_t     request_id          = routing->request_id;
    const std::string unique_key          = routing->unique_key;
    const int64_t     request_deadline_ms = routing->deadline_ms;
    const int64_t     now_ms              = currentTimeMs();
    if (request_deadline_ms <= 0 || now_ms >= request_deadline_ms) {
        RTP_LLM_LOG_WARNING("asyncRead: request deadline expired, unique_key: %s", unique_key.c_str());
        return {nullptr, ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "P2P request deadline expired")};
    }
    // StartLoad carries both deadlines: deadline_ms bounds this physical
    // transfer, while request_deadline_ms only bounds request-level terminal
    // state such as Prefill resource tombstones.
    const int64_t     transfer_deadline_ms =
        std::min(request_deadline_ms, now_ms + config_.p2p_max_transfer_deadline_ms);
    const auto& prefill_addr    = routing->prefill_addr;
    const int   prefill_tp_size = routing->prefill_tp_size;
    const int   prefill_cp_size = routing->prefill_cp_size;

    if (unique_key.empty()) {
        RTP_LLM_LOG_WARNING("asyncRead: unique_key is empty");
        return {nullptr, ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "unique_key is empty")};
    }

    if (prefill_addr.first.empty() || prefill_addr.second == 0) {
        RTP_LLM_LOG_WARNING("asyncRead: prefill_ip is empty or prefill_port is 0");
        return {nullptr,
                ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED,
                          "prefill_ip is empty or prefill_port is 0")};
    }

    auto collector = std::make_shared<DecodeSchedulerMetricsCollector>(metrics_reporter_);
    if (!no_transfer && !config_.topology) {
        collector->success = false;
        return {nullptr,
                ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED, "cache topology is null")};
    }
    if (!no_transfer && prefill_cp_size != config_.cp_size) {
        RTP_LLM_LOG_WARNING("asyncRead: source/target CP layout mismatch, prefill_cp_size=%d, decode_cp_size=%d",
                            prefill_cp_size,
                            config_.cp_size);
        collector->success = false;
        return {nullptr,
                ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED,
                          "source and target CP sizes must match")};
    }
    P2PBroadcastClient::RankLayerCacheBuffers rank_layer_cache_buffers;
    if (!no_transfer) {
        const size_t worker_num = config_.worker_grpc_addrs.size();
        if (config_.cp_size <= 0 || worker_num == 0 || worker_num % static_cast<size_t>(config_.cp_size) != 0) {
            RTP_LLM_LOG_WARNING(
                "asyncRead: invalid CP layout, worker_num=%zu, cp_size=%d", worker_num, config_.cp_size);
            collector->success = false;
            return {nullptr,
                    ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED,
                              "worker count is incompatible with CP size")};
        }

        // Scheduler only runs on rank 0. Build every worker's local view here;
        // allocator block ids are rank-synchronized before asyncRead starts.
        rank_layer_cache_buffers.reserve(worker_num);
        for (size_t worker_rank = 0; worker_rank < worker_num; ++worker_rank) {
            const int cp_rank = static_cast<int>(worker_rank % static_cast<size_t>(config_.cp_size));
            rank_layer_cache_buffers.push_back(LayerCacheBufferUtil::convert(*resource,
                                                                             *config_.topology,
                                                                             block_range.first,
                                                                             block_range.second,
                                                                             cp_rank,
                                                                             config_.cp_size));
        }
    }
    const bool all_rank_buffers_empty =
        std::all_of(rank_layer_cache_buffers.begin(), rank_layer_cache_buffers.end(), [](const auto& buffers) {
            return buffers.empty();
        });
    if (!no_transfer && all_rank_buffers_empty) {
        RTP_LLM_LOG_WARNING("asyncRead: all rank layer_cache_buffers are empty");
        collector->success = false;
        return {nullptr,
                ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED,
                          "all rank layer_cache_buffers are empty")};
    }

    auto async_context = std::make_shared<P2PConnectorAsyncReadContext>(resource,
                                                                        unique_key,
                                                                        collector,
                                                                        config_.p2p_transfer_not_done_resource_hold_ms,
                                                                        no_transfer,
                                                                        request_deadline_ms);

    auto submit_result = async_read_pool_->pushTask(
        [this,
         request_id,
         prefill_ip = prefill_addr.first,
         prefill_port = prefill_addr.second,
         unique_key,
         request_deadline_ms,
         transfer_deadline_ms,
         rank_layer_cache_buffers = std::move(rank_layer_cache_buffers),
         collector,
         async_context,
         prefill_tp_size,
         no_transfer]() mutable {
            if (!async_context->beginKickoff()) {
                return;
            }

            ErrorInfo start_error;
            auto      async_calls = startAsyncReadCalls(request_id,
                                                        prefill_ip,
                                                        prefill_port,
                                                        unique_key,
                                                        request_deadline_ms,
                                                        transfer_deadline_ms,
                                                        rank_layer_cache_buffers,
                                                        collector,
                                                        start_error,
                                                        prefill_tp_size,
                                                        no_transfer);
            if (!async_calls) {
                async_context->markStartFailed(start_error);
                return;
            }

            const bool cancel_requested =
                async_context->setCallResults(async_calls->tp_sync_result, async_calls->server_call_result);
            if (cancel_requested) {
                async_context->cancel(tp_broadcast_client_);
            }
        },
        false);
    if (submit_result != autil::ThreadPoolBase::ERROR_NONE) {
        collector->success = false;
        return {nullptr,
                ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED,
                          "failed to submit P2P async read kickoff task")};
    }

    // addContext only publishes the pending context. The potentially slow RPC
    // kickoff runs in async_read_pool_ and never holds async_contexts_mutex_.
    const int64_t add_context_start_us = currentTimeUs();
    checker_->addContext(async_context);
    const int64_t add_context_cost_us = currentTimeUs() - add_context_start_us;
    if (add_context_cost_us >= 100000) {
        RTP_LLM_LOG_WARNING("[PD-DIAG] P2PConnectorSchedulerDecode::asyncRead slow addContext, "
                            "unique_key=%s, cost_us=%ld",
                            unique_key.c_str(),
                            add_context_cost_us);
    }

    return {async_context, ErrorInfo::OkStatus()};
}

std::optional<P2PConnectorSchedulerDecode::AsyncReadCallResults> P2PConnectorSchedulerDecode::startAsyncReadCalls(
    int64_t                                                 request_id,
    const std::string&                                      prefill_ip,
    uint32_t                                                prefill_port,
    const std::string&                                      unique_key,
    int64_t                                                 request_deadline_ms,
    int64_t                                                 transfer_deadline_ms,
    const P2PBroadcastClient::RankLayerCacheBuffers&        rank_layer_cache_buffers,
    const std::shared_ptr<DecodeSchedulerMetricsCollector>& collector,
    ErrorInfo&                                              out_error,
    int                                                     prefill_tp_size,
    bool                                                    no_transfer) {

    const int64_t entry_us = currentTimeUs();
    RTP_LLM_LOG_DEBUG("[PD-DIAG] startAsyncReadCalls entry, unique_key=%s, prefill=%s:%u, timestamp_us=%ld",
                     unique_key.c_str(),
                     prefill_ip.c_str(),
                     prefill_port,
                     entry_us);

    // [PD-DIAG] Sub-stage timing. server_caller_->load eventually hits
    // RpcPool::getConnection (a pool-wide mutex + potentially synchronous
    // gRPC channel reconnection). tp_broadcast_client_->broadcast does
    // per-worker gRPC AsyncExecuteFunction. Either can be the source of
    // 18-22s asyncReadAfterMatch stalls observed in production.
    const int64_t server_load_start_us = currentTimeUs();
    auto server_call_result = server_caller_->load(request_id,
                                                   prefill_ip,
                                                   prefill_port,
                                                   unique_key,
                                                   request_deadline_ms,
                                                   transfer_deadline_ms,
                                                   no_transfer);
    const int64_t server_load_cost_us = currentTimeUs() - server_load_start_us;
    if (server_load_cost_us >= 100000) {
        RTP_LLM_LOG_WARNING("[PD-DIAG] startAsyncReadCalls slow server_caller->load, "
                            "unique_key=%s, prefill=%s:%u, cost_us=%ld",
                            unique_key.c_str(),
                            prefill_ip.c_str(),
                            prefill_port,
                            server_load_cost_us);
    }
    if (!server_call_result) {
        RTP_LLM_LOG_WARNING("asyncRead: server_caller load failed, unique_key: %s", unique_key.c_str());
        collector->success = false;
        out_error          = ErrorInfo(ErrorCode::P2P_CONNECTOR_LOAD_FROM_PREFILL_FAILED,
                              "server_caller load failed: failed to start async StartLoad RPC to prefill");
        return std::nullopt;
    }

    const int64_t broadcast_start_us = currentTimeUs();
    std::shared_ptr<P2PBroadcastClient::Result> tp_sync_result;
    if (no_transfer) {
        tp_sync_result = std::make_shared<P2PBroadcastClient::Result>(unique_key);
    } else {
        tp_sync_result = tp_broadcast_client_->broadcastPerRank(request_id,
                                                                rank_layer_cache_buffers,
                                                                {},
                                                                unique_key,
                                                                transfer_deadline_ms,
                                                                P2PConnectorBroadcastType::READ,
                                                                prefill_tp_size,
                                                                request_deadline_ms);
    }
    const int64_t broadcast_cost_us = currentTimeUs() - broadcast_start_us;
    if (broadcast_cost_us >= 100000) {
        RTP_LLM_LOG_WARNING(
            "[PD-DIAG] startAsyncReadCalls slow tp_broadcast_client->broadcast, unique_key=%s, cost_us=%ld",
            unique_key.c_str(),
            broadcast_cost_us);
    }
    if (!tp_sync_result) {
        collector->success = false;
        RTP_LLM_LOG_WARNING("asyncRead: broadcast failed");
        out_error = ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "broadcast failed");
        server_call_result->cancel();
        return std::nullopt;
    }

    const int64_t total_cost_us = currentTimeUs() - entry_us;
    if (total_cost_us >= 100000) {
        RTP_LLM_LOG_WARNING(
            "[PD-DIAG] startAsyncReadCalls slow total, unique_key=%s, total_us=%ld, server_load_us=%ld, broadcast_us=%ld",
            unique_key.c_str(),
            total_cost_us,
            server_load_cost_us,
            broadcast_cost_us);
    }
    return AsyncReadCallResults{server_call_result, tp_sync_result};
}

}  // namespace rtp_llm

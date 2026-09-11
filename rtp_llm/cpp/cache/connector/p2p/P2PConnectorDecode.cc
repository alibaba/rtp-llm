#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorDecode.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnector.h"

#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBuffer.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PBroadcastClient.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorBackend.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PSchedulerDecodeRead.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PWorkerDecodeRead.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PWorkerDecodeWrite.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/RouteCodec.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "autil/NetUtil.h"
#include <algorithm>
#include <limits>
#include <set>
#include <unistd.h>
#include <utility>

namespace rtp_llm {

P2PConnectorDecode::P2PConnectorDecode(P2PConnectorConfig                          config,
                                       const std::shared_ptr<LayerBlockConverter>& layer_block_converter,
                                       const kmonitor::MetricsReporterPtr&         metrics_reporter):
    config_(std::move(config)),
    layer_block_converter_(layer_block_converter),
    metrics_reporter_(metrics_reporter) {}

P2PConnectorDecode::~P2PConnectorDecode() = default;

bool P2PConnectorDecode::init() {
    if (config_.tp_rank == 0) {
        tp_broadcast_client_ = std::make_shared<P2PBroadcastClient>(
            config_.scheduler_config.worker_grpc_addrs, config_.scheduler_config.p2p_cancel_broadcast_timeout_ms);
        if (!tp_broadcast_client_->init()) {
            RTP_LLM_LOG_ERROR("decode connector init failed: tp_broadcast_client init failed");
            return false;
        }
        scheduler_ =
            std::make_unique<P2PSchedulerDecodeRead>(config_.scheduler_config, metrics_reporter_, tp_broadcast_client_);
        std::string process_id = autil::NetUtil::getBindIp() + "_pid_" + std::to_string(getpid()) + "_timestamp_"
                                 + std::to_string(currentTimeUs());
        if (!scheduler_->init(process_id)) {
            RTP_LLM_LOG_ERROR("decode connector init failed: scheduler init failed");
            return false;
        }
    }

    auto [sender, receiver] =
        p2p_internal::createAndRegisterTransferBackend(config_.worker_config, layer_block_converter_, metrics_reporter_);
    if (!sender || !receiver) {
        RTP_LLM_LOG_ERROR("decode connector init failed: transfer backend init failed");
        return false;
    }
    worker_ = std::make_unique<P2PWorkerDecodeRead>(
        config_.worker_config, layer_block_converter_, metrics_reporter_, receiver);
    if (!worker_->initialized()) {
        RTP_LLM_LOG_ERROR("decode connector init failed: worker init failed");
        return false;
    }
    if (config_.p2p_writeback_enable) {
        write_worker_ = std::make_unique<P2PWorkerDecodeWrite>(config_.worker_config, layer_block_converter_, sender);
        if (!write_worker_->init()) {
            RTP_LLM_LOG_ERROR("decode connector init failed: write worker init failed");
            return false;
        }
    }
    return true;
}

bool P2PConnectorDecode::writePerRank(const P2PConnectorBroadcastTpRequestPB& request, FunctionResponsePB& response) {
    response.mutable_p2p_response()->Clear();
    const auto reject = [&](const std::string& message) {
        P2PConnector::setP2PResponse(response, ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, message));
        return false;
    };
    if (!write_worker_) {
        return reject("decode writeback disabled");
    }
    switch (request.write_operation()) {
        case WRITE_START: {
            P2PWorkerRoutePlan plan;
            plan.plan_digest = request.plan_digest();
            for (const auto& pb : request.routes()) {
                if (pb.partition_count() != 1 || pb.partition_id() != 0
                    || pb.slice_mode() != static_cast<int>(CpBlockSliceMode::NONE) || pb.slice_count() != 1
                    || pb.slice_index() != 0) {
                    return reject("writeback currently requires symmetric whole-block routes");
                }
                const auto     local = RouteCodec::decode(pb);
                P2PWorkerRoute route;
                route.route_id  = local.route_id;
                route.cache_tag = local.cache_tag;
                route.partition = local.partition;
                route.slice     = local.slice;
                if (local.peer_index < 0 || local.peer_index >= request.peer_workers_size()) {
                    return reject("write route peer index out of range");
                }
                const auto& peer = request.peer_workers(local.peer_index);
                if (peer.ip().empty() || peer.cache_store_port() <= 0 || peer.cache_store_port() > 65535) {
                    return reject("write route has invalid peer endpoint");
                }
                route.dst_ip   = peer.ip();
                route.dst_port = peer.cache_store_port();
                for (const auto& layer : pb.layer_blocks()) {
                    if (layer.layer_id() > static_cast<uint32_t>(std::numeric_limits<int>::max())
                        || layer.cache_keys().empty() || layer.cache_keys_size() != layer.block_ids_size()) {
                        return reject("write route has invalid layer or key/block pairs");
                    }
                    auto              buffer = std::make_shared<LayerCacheBuffer>(layer.layer_id(), layer.cache_tag());
                    std::set<int64_t> keys;
                    for (int i = 0; i < layer.cache_keys_size(); ++i) {
                        if (layer.block_ids(i) == 0
                            || layer.block_ids(i) > static_cast<uint32_t>(std::numeric_limits<int>::max())
                            || !keys.insert(layer.cache_keys(i)).second) {
                            return reject("write request contains invalid block or duplicate key");
                        }
                        buffer->addBlockId(layer.cache_keys(i), layer.block_ids(i));
                    }
                    route.layer_buffers.push_back(std::move(buffer));
                }
                plan.routes.push_back(std::move(route));
            }
            const auto error =
                write_worker_->write(request.request_id(), request.unique_key(), request.deadline_ms(), plan);
            if (error.hasError()) {
                WriteTaskStatus status;
                if (write_worker_->queryWriteStatus(request.unique_key(), status)) {
                    P2PConnector::fillWriteResponse(response, status);
                }
                P2PConnector::setP2PResponse(response, error);
                return false;
            }
            break;
        }
        case WRITE_CANCEL:
            if (!write_worker_->cancelWrite(request.unique_key(), request.deadline_ms())) {
                return reject("invalid write key");
            }
            break;
        case WRITE_QUERY:
            break;
        default:
            return reject("invalid write operation");
    }
    WriteTaskStatus status;
    if (!write_worker_->queryWriteStatus(request.unique_key(), status)) {
        return reject("unknown write task");
    }
    P2PConnector::fillWriteResponse(response, status);
    return true;
}

std::shared_ptr<AsyncContext> P2PConnectorDecode::read(const KVCacheResourcePtr&    resource,
                                                       const std::shared_ptr<Meta>& meta,
                                                       int                          start_read_block_index,
                                                       int                          read_block_num) {
    if (!meta || !resource || !meta->generateStream()) {
        RTP_LLM_LOG_WARNING("asyncRead failed, meta, resource, or generate_stream is null");
        return nullptr;
    }

    std::pair<int, int> block_range{start_read_block_index, read_block_num};
    const bool          no_transfer = read_block_num == 0;
    const auto          make_failed_context = [&](const ErrorInfo& error_info) {
        auto failed_context = std::make_shared<P2PConnectorAsyncReadContext>(
            resource,
            meta->p2pRouting().value_or(Meta::P2PRoutingContext{}).unique_key,
            nullptr,
            config_.scheduler_config.p2p_transfer_not_done_resource_hold_ms);
        failed_context->markStartFailed(error_info);
        return failed_context;
    };
    if (scheduler_ == nullptr) {
        ErrorInfo error_info(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "P2P scheduler is not ready");
        RTP_LLM_LOG_WARNING("asyncRead failed: %s", error_info.ToString().c_str());
        return make_failed_context(error_info);
    }
    auto result = scheduler_->asyncRead(resource, meta, block_range, no_transfer);
    if (!result.ok()) {
        RTP_LLM_LOG_WARNING("asyncRead failed, unique_key: %s, error: %s",
                            meta->p2pRouting().value_or(Meta::P2PRoutingContext{}).unique_key.c_str(),
                            result.error_info.ToString().c_str());
        return make_failed_context(result.error_info);
    }
    return result.context;
}

void P2PConnectorDecode::cancelRead(const std::shared_ptr<AsyncContext>& context) {
    if (!scheduler_) {
        return;
    }
    scheduler_->cancel(std::dynamic_pointer_cast<P2PConnectorAsyncReadContext>(context));
}

bool P2PConnectorDecode::readPerRank(int64_t                                 request_id,
                                     const std::string&                      unique_key,
                                     int64_t                                 deadline_ms,
                                     const P2PConnectorBroadcastTpRequestPB& p2p_request,
                                     FunctionResponsePB&                     response) {
    auto reject_read = [&response](const std::string& message) {
        ErrorInfo error_info(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, message);
        P2PConnector::setP2PResponse(response, error_info);
        return false;
    };
    if (unique_key.empty()) {
        return reject_read("READ request has empty unique_key");
    }
    if (deadline_ms <= currentTimeMs()) {
        return reject_read("READ request deadline has expired");
    }

    // 「routes 为空」是权威的「本 worker 无任务」信号，取代了 allow_empty_projection。
    if (p2p_request.routes_size() == 0) {
        RTP_LLM_LOG_DEBUG("executeRead: no routes for this worker, request_id=%ld, unique_key=%s",
                          request_id,
                          unique_key.c_str());
        P2PConnector::setP2PResponse(response);
        return true;
    }

    P2PWorkerRoutePlan worker_plan;
    worker_plan.plan_digest = p2p_request.plan_digest();
    std::set<std::string> route_layer_tag_keys;
    for (const auto& route_pb : p2p_request.routes()) {
        const auto     local = RouteCodec::decode(route_pb);
        P2PWorkerRoute worker_route;
        worker_route.route_id  = local.route_id;
        worker_route.cache_tag = local.cache_tag;
        worker_route.partition = local.partition;
        worker_route.slice     = local.slice;

        for (const auto& layer_block_pb : route_pb.layer_blocks()) {
        auto layer_id           = layer_block_pb.layer_id();
        auto cache_keys         = layer_block_pb.cache_keys();
        auto block_ids          = layer_block_pb.block_ids();
        if (cache_keys.size() != block_ids.size()) {
            const std::string error_message = "cache_keys size " + std::to_string(cache_keys.size())
                                              + " != block_ids size " + std::to_string(block_ids.size())
                                              + " for unique_key=" + unique_key
                                              + ", layer_id=" + std::to_string(layer_id);
            RTP_LLM_LOG_WARNING("executeRead rejected malformed request, request_id=%ld, unique_key=%s, layer_id=%d, "
                                "cache_keys=%d, block_ids=%d",
                                request_id,
                                unique_key.c_str(),
                                layer_id,
                                cache_keys.size(),
                                block_ids.size());
            return reject_read(error_message);
        }
        if (cache_keys.empty() || layer_id < 0 || !config_.worker_config.topology
            || static_cast<size_t>(layer_id) >= config_.worker_config.topology->layers().size()) {
            return reject_read("READ request has empty blocks or invalid layer");
        }
        const auto& layer = config_.worker_config.topology->layer(layer_id);
        if (std::find(layer.group_tags.begin(), layer.group_tags.end(), layer_block_pb.cache_tag())
            == layer.group_tags.end()) {
            return reject_read("READ request cache tag is not owned by layer");
        }
        // route 内的 (layer, tag) 唯一；跨 route 允许重复（同一层不同 route 写不同字节区间）。
        const std::string layer_tag_key = std::to_string(local.route_id) + "@" + std::to_string(layer_id) + ":"
                                          + layer_block_pb.cache_tag();
        if (!route_layer_tag_keys.insert(layer_tag_key).second) {
            return reject_read("READ request contains duplicate layer/tag within one route");
        }
        for (const auto block_id : block_ids) {
            if (block_id < 0) {
                return reject_read("READ request contains invalid block id");
            }
        }
        const std::set<int64_t> distinct_keys(cache_keys.begin(), cache_keys.end());
        if (distinct_keys.size() != static_cast<size_t>(cache_keys.size())) {
            return reject_read("READ request contains duplicate cache key");
        }
        auto layer_cache_buffer = std::make_shared<LayerCacheBuffer>(layer_id, layer_block_pb.cache_tag());
        for (size_t i = 0; i < cache_keys.size(); i++) {
            layer_cache_buffer->addBlockId(cache_keys[i], block_ids[i]);
        }
            worker_route.layer_buffers.push_back(layer_cache_buffer);
        }
        if (worker_route.layer_buffers.empty()) {
            return reject_read("READ route has no layer blocks");
        }
        worker_plan.routes.push_back(std::move(worker_route));
    }
    ErrorInfo error_info = worker_->read(request_id, unique_key, deadline_ms, worker_plan);
    if (error_info.hasError()) {
        RTP_LLM_LOG_WARNING("executeRead failed, request_id: %ld, unique_key: %s, error: %s",
                            request_id,
                            unique_key.c_str(),
                            error_info.ToString().c_str());
    }
    P2PConnector::setP2PResponse(response, error_info);
    return error_info.ok();
}

bool P2PConnectorDecode::cancelReadPerRank(const std::string& unique_key,
                                           int64_t            request_deadline_ms,
                                           FunctionResponsePB& response) {
    bool ret = worker_->cancelRead(unique_key, request_deadline_ms);
    P2PConnector::setP2PResponse(response);
    return ret;
}

bool P2PConnectorDecode::queryLeaseStatusPerRank(const std::string& unique_key, FunctionResponsePB& response) {
    bool sealed       = true;
    int  started_ops  = 0;
    int  finished_ops = 0;
    bool stopped      = true;

    worker_->queryLeaseStatus(unique_key, sealed, started_ops, finished_ops, stopped);

    auto* p2p_response = response.mutable_p2p_response();
    p2p_response->set_error_code(ErrorCodePB::NONE_ERROR);
    auto* lease_status = p2p_response->mutable_lease_status();
    lease_status->set_sealed(sealed);
    lease_status->set_started_ops(started_ops);
    lease_status->set_finished_ops(finished_ops);
    lease_status->set_stopped(stopped);

    RTP_LLM_LOG_DEBUG("executeQueryLeaseStatus: unique_key=%s sealed=%d started=%d finished=%d stopped=%d",
                      unique_key.c_str(),
                      sealed,
                      started_ops,
                      finished_ops,
                      stopped);
    return true;
}

}  // namespace rtp_llm

#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorPrefill.h"

#include "rtp_llm/cpp/cache/connector/KVCacheConnectorLayerContext.h"
#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBuffer.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PBroadcastClient.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorBackend.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorResourceStore.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorSchedulerPrefill.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorkerPrefill.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/RouteCodec.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>
#include <chrono>
#include <utility>

namespace rtp_llm {

P2PConnectorPrefill::P2PConnectorPrefill(P2PConnectorConfig                          config,
                                         const std::shared_ptr<LayerBlockConverter>& layer_block_converter,
                                         const kmonitor::MetricsReporterPtr&         metrics_reporter):
    config_(std::move(config)),
    layer_block_converter_(layer_block_converter),
    metrics_reporter_(metrics_reporter) {}

P2PConnectorPrefill::~P2PConnectorPrefill() = default;

bool P2PConnectorPrefill::init() {
    if (config_.tp_rank == 0) {
        tp_broadcast_client_ = std::make_shared<P2PBroadcastClient>(
            config_.scheduler_config.worker_grpc_addrs, config_.scheduler_config.p2p_cancel_broadcast_timeout_ms);
        if (!tp_broadcast_client_->init()) {
            RTP_LLM_LOG_ERROR("prefill connector init failed: tp_broadcast_client init failed");
            return false;
        }
        scheduler_ = std::make_unique<P2PConnectorSchedulerPrefill>(
            config_.scheduler_config, metrics_reporter_, tp_broadcast_client_);
    }

    auto [sender, receiver] =
        p2p_internal::createAndRegisterTransferBackend(config_.worker_config, layer_block_converter_, metrics_reporter_);
    if (!sender || !receiver) {
        RTP_LLM_LOG_ERROR("prefill connector init failed: transfer backend init failed");
        return false;
    }
    worker_ = std::make_shared<P2PConnectorWorkerPrefill>(
        config_.worker_config, layer_block_converter_, metrics_reporter_, sender);
    if (!worker_->init(10 * 1000)) {
        RTP_LLM_LOG_ERROR("prefill connector init failed: worker init failed");
        return false;
    }

    if (config_.tp_rank == 0) {
        stream_store_ = std::make_shared<P2PConnectorResourceStore>(
            metrics_reporter_,
            config_.scheduler_config.p2p_resource_store_timeout_check_interval_ms,
            config_.scheduler_config.p2p_prefill_resource_hold_ms,
            config_.scheduler_config.p2p_cancelled_keys_ttl_ms);
        stream_store_->setOnRequestReleased([computed_buffers = worker_->getComputedBuffersStore()](
                                                int64_t request_id, int64_t request_deadline_ms) {
            if (computed_buffers) {
                computed_buffers->removeBuffer(request_id, request_deadline_ms);
            }
        });
        if (!stream_store_->init()) {
            RTP_LLM_LOG_ERROR("prefill connector init failed: stream_store init failed");
            return false;
        }
    }
    return true;
}

std::shared_ptr<AsyncContext> P2PConnectorPrefill::registerResource(const KVCacheResourcePtr&    resource,
                                                                    const std::shared_ptr<Meta>& meta) {
    if (config_.tp_rank != 0 || !stream_store_) {
        RTP_LLM_LOG_WARNING("asyncRead failed, resource registration is only available on prefill tp_rank 0");
        return nullptr;
    }
    if (!meta || !resource || !meta->generateStream()) {
        RTP_LLM_LOG_WARNING("asyncRead failed, meta, resource, or generate_stream is null");
        return nullptr;
    }
    if (!stream_store_->addResource(meta, resource)) {
        RTP_LLM_LOG_WARNING("asyncRead failed, stream_store add resource failed");
        return nullptr;
    }
    return std::make_shared<CompletedAsyncContext>(ErrorInfo::OkStatus());
}

std::shared_ptr<AsyncContext> P2PConnectorPrefill::asyncWriteByLayer(
    int layer_id, const std::shared_ptr<KVCacheConnectorLayerContext>& layer_context) {
    if (!worker_ || !layer_context) {
        RTP_LLM_LOG_WARNING("asyncWriteByLayer failed, worker or layer context is null, layer_id=%d", layer_id);
        return nullptr;
    }
    auto resource = std::make_shared<KVCacheResource>(layer_context->kvCacheResource());
    if (!worker_->writeByLayer(layer_id,
                               resource,
                               layer_context->requestId(),
                               layer_context->attentionEvent(),
                               layer_context->deadlineMs())) {
        RTP_LLM_LOG_WARNING("asyncWriteByLayer failed to schedule P2P write, layer_id=%d, request_id=%ld",
                            layer_id,
                            layer_context->requestId());
        return nullptr;
    }
    return std::make_shared<P2PConnectorAcceptedWriteContext>();
}

bool P2PConnectorPrefill::writeByLayerTag(int                                   layer_id,
                                          const std::string&                    tag,
                                          const KVCacheResourcePtr&             resource,
                                          int64_t                               request_id,
                                          const std::shared_ptr<c10::Event>& event,
                                          int64_t                               deadline_ms) {
    if (!worker_ || !resource) {
        RTP_LLM_LOG_WARNING("writeByLayerTag failed, worker or resource is null, request_id=%ld layer_id=%d tag=%s",
                            request_id,
                            layer_id,
                            tag.c_str());
        return false;
    }
    return worker_->writeByLayerTag(layer_id, tag, resource, request_id, event, deadline_ms);
}

void P2PConnectorPrefill::processRead(const P2PConnectorStartLoadRequestPB& request,
                                      P2PConnectorStartLoadResponsePB&      response,
                                      std::function<bool()>                 is_cancelled) {
    if (scheduler_ == nullptr) {
        RTP_LLM_LOG_WARNING("handleRead failed, scheduler not initialized (only tp_rank 0 has scheduler)");
        response.set_error_code(transErrorCodeToRPC(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED));
        response.set_error_message("scheduler not initialized");
        return;
    }

    const std::string& unique_key           = request.unique_key();
    const int64_t      transfer_deadline_ms = request.deadline_ms();
    const int64_t      request_deadline_ms  = request.request_deadline_ms();
    const int64_t      now_ms               = currentTimeMs();
    if (unique_key.empty()) {
        response.set_error_code(transErrorCodeToRPC(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED));
        response.set_error_message("invalid StartLoad unique_key");
        return;
    }
    if (request_deadline_ms <= 0 || transfer_deadline_ms <= 0 || transfer_deadline_ms > request_deadline_ms) {
        if (!unique_key.empty() && request_deadline_ms > 0) {
            stream_store_->markTerminal(unique_key, request_deadline_ms);
            stream_store_->clearSideChannelData(unique_key);
        }
        response.set_error_code(transErrorCodeToRPC(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED));
        response.set_error_message("invalid StartLoad deadlines");
        return;
    }
    if (now_ms >= transfer_deadline_ms) {
        if (!unique_key.empty()) {
            stream_store_->markTerminal(unique_key, request_deadline_ms);
            stream_store_->clearSideChannelData(unique_key);
        }
        response.set_error_code(transErrorCodeToRPC(ErrorCode::GENERATE_TIMEOUT));
        response.set_error_message("transfer deadline expired before handleRead");
        return;
    }
    auto handle_read_start_us = currentTimeUs();

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    for (const auto& worker : request.workers()) {
        decode_transfer_servers.emplace_back(worker.ip(), worker.cache_store_port());
    }

    RTP_LLM_LOG_DEBUG("[PD-DIAG] handleRead start, unique_key=%s, deadline_ms=%ld, timestamp_us=%ld",
                     unique_key.c_str(),
                     transfer_deadline_ms,
                     handle_read_start_us);

    std::shared_ptr<P2PConnectorResourceEntry> resource_entry = nullptr;
    grpc::Status wait_status = waitForResourceEntry(
        unique_key, request_deadline_ms, transfer_deadline_ms, is_cancelled, resource_entry);
    auto         wait_resource_cost_us = currentTimeUs() - handle_read_start_us;
    if (!wait_status.ok()) {
        RTP_LLM_LOG_WARNING("[PD-DIAG] handleRead waitForResourceEntry failed, unique_key=%s, cost_us=%ld, status=%s",
                            unique_key.c_str(),
                            wait_resource_cost_us,
                            wait_status.error_message().c_str());
        ErrorCode error_code = ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED;
        if (wait_status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED) {
            error_code = ErrorCode::GENERATE_TIMEOUT;
        } else if (wait_status.error_code() == grpc::StatusCode::CANCELLED) {
            error_code = ErrorCode::CANCELLED;
        }
        response.set_error_code(transErrorCodeToRPC(error_code));
        response.set_error_message("waitForResourceEntry failed: " + wait_status.error_message());
        return;
    }

    // The resource-hold deadline only applies while waiting in the store.
    // Once StartLoad has consumed the entry, response/side-channel waiting is
    // governed by this transfer's absolute deadline.
    resource_entry->deadline_ms = transfer_deadline_ms;

    int64_t request_id = resource_entry->request_id;
    // Downgrade noisy entry log: total_cost_us in the "handleRead complete" log
    // below already accounts for wait_resource_cost_us; keep this as DEBUG for
    // ad-hoc tracing without dominating the log file (~1150 entries per 4h).
    RTP_LLM_LOG_DEBUG("[PD-DIAG] handleRead resource ready, unique_key=%s, request_id=%ld, wait_cost_us=%ld",
                      unique_key.c_str(),
                      request_id,
                      wait_resource_cost_us);
    // Wrap is_cancelled to also directly cancel the local worker's send
    // when the gRPC client disconnects — bypassing the CANCEL_HANDLE_READ
    // broadcast RPC hop. Remote workers (other TP ranks) still receive
    // the broadcast; this only accelerates rank-0's own worker.
    auto direct_cancel = [is_cancelled,
                          worker = worker_,
                          request_id,
                          unique_key,
                          transfer_deadline_ms,
                          request_deadline_ms]() -> bool {
        if (is_cancelled && is_cancelled()) {
            worker->cancelRequest(request_id, unique_key, transfer_deadline_ms, request_deadline_ms);
            return true;
        }
        return false;
    };
    auto      send_start_us = currentTimeUs();
    ErrorInfo error_info = scheduler_->sendKVCache(unique_key,
                                                   request_id,
                                                   decode_transfer_servers,
                                                   transfer_deadline_ms,
                                                   direct_cancel,
                                                   request.no_transfer(),
                                                   request_deadline_ms);
    auto send_cost_us = currentTimeUs() - send_start_us;
    if (error_info.hasError()) {
        RTP_LLM_LOG_ERROR("[PD-DIAG] handleRead sendKVCache failed, unique_key=%s, send_cost_us=%ld, error=%s",
                          unique_key.c_str(),
                          send_cost_us,
                          error_info.ToString().c_str());
        // Seal the consumed request before clearing its side-channel entry.
        // Otherwise a late first-token notification can recreate the entry
        // with the request-level deadline after this handler returns.
        stream_store_->markTerminal(unique_key, request_deadline_ms);
        stream_store_->clearSideChannelData(unique_key);
        response.set_error_code(transErrorCodeToRPC(error_info.code()));
        response.set_error_message(error_info.ToString());
        return;
    }

    // send_cost_us is already implied by total_cost_us - wait_resource_cost_us
    // in the complete log; demote to DEBUG to reduce log volume.
    RTP_LLM_LOG_DEBUG(
        "[PD-DIAG] handleRead sendKVCache done, unique_key=%s, send_cost_us=%ld", unique_key.c_str(), send_cost_us);
    waitAndFillResponse(resource_entry, response, is_cancelled);
    // waitAndFillResponse clears the currently visible payload. Mark terminal
    // and clear once more to close the race with a notification arriving
    // between its final clear and this handler returning.
    stream_store_->markTerminal(unique_key, request_deadline_ms);
    stream_store_->clearSideChannelData(unique_key);
    RTP_LLM_LOG_DEBUG("[PD-DIAG] handleRead complete, unique_key=%s, request_id=%ld, "
                     "total_cost_us=%ld, wait_resource_us=%ld, send_us=%ld",
                     unique_key.c_str(),
                     request_id,
                     currentTimeUs() - handle_read_start_us,
                     wait_resource_cost_us,
                     send_cost_us);
}

void P2PConnectorPrefill::waitAndFillResponse(const std::shared_ptr<P2PConnectorResourceEntry>& resource_entry,
                                              P2PConnectorStartLoadResponsePB&                  response,
                                              std::function<bool()>                             is_cancelled) {
    // Wait for side-channel data to be ready (notified by prefill engine when first token / SP data is produced)
    // Note: resource_entry has already been stolen from resource_map_, so we wait on it directly
    const std::string& unique_key  = resource_entry->unique_key;
    int64_t            deadline_ms = resource_entry->deadline_ms;

    std::unique_lock<std::mutex> lock(resource_entry->side_channel_mutex);
    const int64_t                remaining_us = deadline_ms * 1000 - currentTimeUs();
    if (remaining_us <= 0) {
        RTP_LLM_LOG_WARNING("waitAndFillResponse: past deadline, unique_key: %s", unique_key.c_str());
        stream_store_->clearSideChannelData(unique_key);
        response.set_error_code(transErrorCodeToRPC(ErrorCode::P2P_CONNECTOR_SCHEDULER_FILL_RESPONSE_FAILED));
        response.set_error_message("waitAndFillResponse: past deadline");
        return;
    }

    const auto timeout_tp = std::chrono::system_clock::now() + std::chrono::microseconds(remaining_us);
    while (!resource_entry->side_channel_ready) {
        if (is_cancelled && is_cancelled()) {
            RTP_LLM_LOG_DEBUG("waitAndFillResponse: cancelled, unique_key: %s", unique_key.c_str());
            stream_store_->clearSideChannelData(unique_key);
            response.set_error_code(transErrorCodeToRPC(ErrorCode::P2P_CONNECTOR_SCHEDULER_FILL_RESPONSE_FAILED));
            response.set_error_message("waitAndFillResponse: cancelled");
            return;
        }

        P2PConnectorResourceEntry::SideChannelData side_channel_data;
        if (stream_store_->consumeSideChannelData(unique_key, side_channel_data)) {
            resource_entry->side_channel_data  = std::move(side_channel_data);
            resource_entry->side_channel_ready = true;
            break;
        }

        auto next_wake_tp = std::min(timeout_tp, std::chrono::system_clock::now() + std::chrono::milliseconds(10));
        resource_entry->side_channel_cv.wait_until(lock, next_wake_tp);
        if (!resource_entry->side_channel_ready) {
            if (stream_store_->consumeSideChannelData(unique_key, side_channel_data)) {
                resource_entry->side_channel_data  = std::move(side_channel_data);
                resource_entry->side_channel_ready = true;
                break;
            }
            if (std::chrono::system_clock::now() >= timeout_tp) {
                RTP_LLM_LOG_WARNING("waitAndFillResponse: timeout, unique_key: %s", unique_key.c_str());
                stream_store_->clearSideChannelData(unique_key);
                response.set_error_code(transErrorCodeToRPC(ErrorCode::P2P_CONNECTOR_SCHEDULER_FILL_RESPONSE_FAILED));
                response.set_error_message("waitAndFillResponse: timeout");
                return;
            }
        }
    }

    // Release lock before calling fillResponseWithStreamInfo to avoid deadlock
    // (fillResponseWithStreamInfo acquires side_channel_mutex internally)
    lock.unlock();

    grpc::Status fill_status = fillResponseWithStreamInfo(resource_entry, response);
    if (!fill_status.ok()) {
        RTP_LLM_LOG_WARNING("waitAndFillResponse failed, unique_key: %s, error: %s",
                            resource_entry->unique_key.c_str(),
                            fill_status.error_message().c_str());
        stream_store_->clearSideChannelData(unique_key);
        response.set_error_code(transErrorCodeToRPC(ErrorCode::P2P_CONNECTOR_SCHEDULER_FILL_RESPONSE_FAILED));
        response.set_error_message("fillResponseWithStreamInfo failed: " + fill_status.error_message());
        return;
    }

    stream_store_->clearSideChannelData(unique_key);
    response.set_error_code(ErrorCodePB::NONE_ERROR);
}

namespace {

void setP2PResponse(FunctionResponsePB& response, const ErrorInfo& error_info) {
    auto* p2p_response = response.mutable_p2p_response();
    if (error_info.hasError()) {
        p2p_response->set_error_code(transErrorCodeToRPC(error_info.code()));
        p2p_response->set_error_message(error_info.ToString());
    } else {
        p2p_response->set_error_code(ErrorCodePB::NONE_ERROR);
        p2p_response->set_error_message("");
    }
}

void setP2PResponseOk(FunctionResponsePB& response) {
    auto* p2p_response = response.mutable_p2p_response();
    p2p_response->set_error_code(ErrorCodePB::NONE_ERROR);
    p2p_response->set_error_message("");
}

}  // namespace

bool P2PConnectorPrefill::processReadPerRank(int64_t                                 request_id,
                                             const std::string&                      unique_key,
                                             int64_t                                 deadline_ms,
                                             const P2PConnectorBroadcastTpRequestPB& p2p_request,
                                             FunctionResponsePB&                     response) {
    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    for (const auto& peer_worker : p2p_request.peer_workers()) {
        decode_transfer_servers.emplace_back(peer_worker.ip(), peer_worker.cache_store_port());
    }
    // 解出本 prefill worker 自己那份 route。peer_index 在此解析成具体端点，worker 不再自选目标。
    P2PWorkerRoutePlan worker_plan;
    worker_plan.plan_digest = p2p_request.plan_digest();
    for (const auto& route_pb : p2p_request.routes()) {
        const auto     local = RouteCodec::decode(route_pb);
        P2PWorkerRoute worker_route;
        worker_route.route_id  = local.route_id;
        worker_route.cache_tag = local.cache_tag;
        worker_route.partition = local.partition;
        worker_route.slice     = local.slice;
        if (local.peer_index < 0 || static_cast<size_t>(local.peer_index) >= decode_transfer_servers.size()) {
            ErrorInfo error_info(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED,
                                 "HANDLE_READ route peer_index out of range: " + std::to_string(local.peer_index));
            RTP_LLM_LOG_WARNING("executeHandleRead rejected: %s", error_info.ToString().c_str());
            worker_->cancelRequest(
                request_id, unique_key, deadline_ms, p2p_request.request_deadline_ms());
            setP2PResponse(response, error_info);
            return false;
        }
        worker_route.dst_ip   = decode_transfer_servers[local.peer_index].first;
        worker_route.dst_port = decode_transfer_servers[local.peer_index].second;
        worker_plan.routes.push_back(std::move(worker_route));
    }

    ErrorInfo error_info = worker_->sendKVCache(
        request_id, unique_key, deadline_ms, worker_plan, p2p_request.request_deadline_ms());
    if (error_info.hasError()) {
        RTP_LLM_LOG_WARNING("executeHandleRead failed, request_id: %ld, unique_key: %s, error: %s",
                            request_id,
                            unique_key.c_str(),
                            error_info.ToString().c_str());
    }
    setP2PResponse(response, error_info);
    return error_info.ok();
}

bool P2PConnectorPrefill::processNoTransferPerRank(
    int64_t                                 request_id,
    const std::string& /* unique_key */,
    int64_t                                 deadline_ms,
    const P2PConnectorBroadcastTpRequestPB& p2p_request,
    FunctionResponsePB&                     response) {
    worker_->completeNoTransfer(request_id, deadline_ms, p2p_request.request_deadline_ms());
    setP2PResponseOk(response);
    return true;
}

bool P2PConnectorPrefill::cancelProcessReadPerRank(int64_t             request_id,
                                                   const std::string& unique_key,
                                                   int64_t             deadline_ms,
                                                   int64_t             request_deadline_ms,
                                                   FunctionResponsePB& response) {
    bool ret = worker_->cancelRequest(request_id, unique_key, deadline_ms, request_deadline_ms);
    setP2PResponseOk(response);
    return ret;
}

grpc::Status P2PConnectorPrefill::waitForResourceEntry(
    const std::string&                          unique_key,
    int64_t                                     request_deadline_ms,
    int64_t                                     transfer_deadline_ms,
    std::function<bool()>                       is_cancelled,
    std::shared_ptr<P2PConnectorResourceEntry>& resource_entry) {
    if (currentTimeMs() >= request_deadline_ms) {
        RTP_LLM_LOG_WARNING("waiting for resource deadline exceeded, unique_key: %s", unique_key.c_str());
        return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "resource wait deadline exceeded");
    }
    resource_entry = stream_store_->waitAndStealResource(unique_key, transfer_deadline_ms, is_cancelled);
    if (resource_entry) {
        return grpc::Status::OK;
    }
    if (is_cancelled && is_cancelled()) {
        // The decode-side gRPC was cancelled while we were waiting for the resource.
        // The resource may not be in the store yet (prefill is still computing), or it
        // may arrive shortly after this handler exits. Mark the key as cancelled so that
        // addResource() rejects it immediately on arrival instead of pinning KV blocks
        // until the next checkTimeout() cycle (avoiding LACK MEM under sustained overload).
        stream_store_->markCancelled(unique_key, request_deadline_ms);
        RTP_LLM_LOG_WARNING("waiting for resource cancelled, unique_key: %s", unique_key.c_str());
        return grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled");
    }
    RTP_LLM_LOG_WARNING("resource not found, unique_key: %s", unique_key.c_str());
    if (stream_store_->isMarkedCancelled(unique_key)) {
        return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "resource expired: prefill hold time exceeded");
    }
    // The transfer wait window ended before the resource arrived. This
    // StartLoad attempt is terminal; retain that state until the original
    // request deadline so duplicate or delayed calls cannot wait again.
    stream_store_->markCancelled(unique_key, request_deadline_ms);
    return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "resource wait transfer deadline exceeded");
}

grpc::Status P2PConnectorPrefill::fillResponseWithStreamInfo(
    const std::shared_ptr<P2PConnectorResourceEntry>& resource_entry,
    P2PConnectorStartLoadResponsePB&                  response) {
    // Read side-channel data from entry (filled by notifySideChannelReady)
    P2PConnectorResourceEntry::SideChannelData data;
    {
        std::lock_guard<std::mutex> lock(resource_entry->side_channel_mutex);
        if (!resource_entry->side_channel_ready) {
            return grpc::Status(grpc::StatusCode::INTERNAL, "side-channel data not ready");
        }
        data = resource_entry->side_channel_data;
    }

    // Fill response proto from side-channel data
    auto* payload = response.mutable_payload();
    payload->set_has_first_generate_token(data.has_first_token);
    if (data.has_first_token) {
        payload->set_first_generate_token_id(data.first_token_id);
    }
    payload->set_total_reuse_len(data.total_reuse_len);
    payload->set_local_reuse_len(data.local_reuse_len);
    payload->set_remote_reuse_len(data.remote_reuse_len);
    payload->set_memory_reuse_len(data.memory_reuse_len);
    payload->set_disk_reuse_len(data.disk_reuse_len);

    if (!data.propose_tokens.empty()) {
        auto& propose_tensor = (*payload->mutable_tensors())["propose_tokens"];
        auto* tokens_pb      = propose_tensor.mutable_tensor();
        tokens_pb->set_data_type(TensorPB::INT32);
        tokens_pb->add_shape(data.propose_tokens.size());
        std::vector<int32_t> int32_tokens(data.propose_tokens.begin(), data.propose_tokens.end());
        tokens_pb->set_int32_data(int32_tokens.data(), int32_tokens.size() * sizeof(int32_t));
    }
    if (data.propose_probs.data_type() != TensorPB::FP32 && data.propose_probs.fp16_data().empty()
        && data.propose_probs.bf16_data().empty() && data.propose_probs.fp32_data().empty()) {
        // propose_probs is empty but that's OK — skip
    } else {
        auto& probs_tensor = (*payload->mutable_tensors())["propose_probs"];
        probs_tensor.mutable_tensor()->CopyFrom(data.propose_probs);
    }
    if (data.propose_hidden.data_type() != TensorPB::FP32 && data.propose_hidden.fp16_data().empty()
        && data.propose_hidden.bf16_data().empty() && data.propose_hidden.fp32_data().empty()) {
        // propose_hidden is empty but that's OK — skip
    } else {
        auto& hidden_tensor = (*payload->mutable_tensors())["propose_hidden"];
        hidden_tensor.mutable_tensor()->CopyFrom(data.propose_hidden);
    }
    if (!data.position_ids.empty()) {
        auto& pos_tensor = (*payload->mutable_tensors())["position_ids"];
        auto* pos_pb     = pos_tensor.mutable_tensor();
        pos_pb->set_data_type(TensorPB::INT32);
        pos_pb->add_shape(data.position_ids.size());
        pos_pb->set_int32_data(data.position_ids.data(), data.position_ids.size() * sizeof(int32_t));
    }

    RTP_LLM_LOG_DEBUG("fill response from entry: first_token: %ld, total_reuse: %d, local: %d, remote: %d, "
                      "memory: %d, disk: %d",
                      data.first_token_id,
                      data.total_reuse_len,
                      data.local_reuse_len,
                      data.remote_reuse_len,
                      data.memory_reuse_len,
                      data.disk_reuse_len);

    return grpc::Status::OK;
}

}  // namespace rtp_llm

#include "rtp_llm/cpp/cache/connector/p2p/P2PConnector.h"

#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorDecode.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorPrefill.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <utility>

namespace rtp_llm {

P2PConnector::P2PConnector(P2PConnectorConfig                          config,
                           const std::shared_ptr<LayerBlockConverter>& layer_block_converter,
                           const kmonitor::MetricsReporterPtr&         metrics_reporter):
    config_(std::move(config)), layer_block_converter_(layer_block_converter), metrics_reporter_(metrics_reporter) {}

P2PConnector::~P2PConnector() = default;

bool P2PConnector::init() {
    if (config_.role_type == RoleType::PREFILL) {
        prefill_ = std::make_unique<P2PConnectorPrefill>(config_, layer_block_converter_, metrics_reporter_);
        if (!prefill_->init()) {
            RTP_LLM_LOG_ERROR("init failed: prefill connector init failed");
            return false;
        }
        return true;
    }

    if (config_.role_type == RoleType::DECODE) {
        decode_ = std::make_unique<P2PConnectorDecode>(config_, layer_block_converter_, metrics_reporter_);
        if (!decode_->init()) {
            RTP_LLM_LOG_ERROR("init failed: decode connector init failed");
            return false;
        }
        return true;
    }

    RTP_LLM_LOG_ERROR("init failed: unsupported role type %d", config_.role_type);
    return false;
}

std::shared_ptr<PrefillResultStore> P2PConnector::resultStore() const {
    return prefill_ ? prefill_->resultStore() : nullptr;
}

std::shared_ptr<P2PConnectorResourceStore> P2PConnector::streamStore() const {
    return prefill_ ? prefill_->resourceStore() : nullptr;
}

std::shared_ptr<AsyncContext> P2PConnector::asyncRead(const KVCacheResourcePtr&    resource,
                                                      const std::shared_ptr<Meta>& meta,
                                                      int                          start_read_block_index,
                                                      int                          read_block_num) {
    if (prefill_) {
        return prefill_->registerResource(resource, meta);
    }
    if (decode_) {
        return decode_->read(resource, meta, start_read_block_index, read_block_num);
    }
    RTP_LLM_LOG_WARNING("asyncRead failed, connector not initialized");
    return nullptr;
}

void P2PConnector::cancelRead(const std::shared_ptr<AsyncContext>& context) {
    if (decode_) {
        decode_->cancelRead(context);
    }
}

std::shared_ptr<AsyncContext>
P2PConnector::asyncWriteByLayer(int layer_id, const std::shared_ptr<KVCacheConnectorLayerContext>& layer_context) {
    return prefill_ ? prefill_->asyncWriteByLayer(layer_id, layer_context) : nullptr;
}

bool P2PConnector::writeByLayerTag(int                                   layer_id,
                                   const std::string&                    tag,
                                   const KVCacheResourcePtr&             resource,
                                   int64_t                               request_id,
                                   const std::shared_ptr<c10::Event>& event,
                                   int64_t                               deadline_ms) {
    return prefill_ ? prefill_->writeByLayerTag(layer_id, tag, resource, request_id, event, deadline_ms) : false;
}

void P2PConnector::handleRead(const P2PConnectorStartLoadRequestPB& request,
                              P2PConnectorStartLoadResponsePB&      response,
                              std::function<bool()>                 is_cancelled) {
    if (!prefill_) {
        RTP_LLM_LOG_WARNING("handleRead failed, Prefill connector is not initialized");
        response.set_error_code(transErrorCodeToRPC(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED));
        response.set_error_message("Prefill connector is not initialized");
        return;
    }
    prefill_->processRead(request, response, std::move(is_cancelled));
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

}  // namespace

bool P2PConnector::executeFunction(const FunctionRequestPB& request, FunctionResponsePB& response) {
    if (!prefill_ && !decode_) {
        RTP_LLM_LOG_WARNING("executeFunction failed, connector not initialized");
        return false;
    }

    if (!request.has_p2p_request()) {
        RTP_LLM_LOG_WARNING("executeFunction failed, no p2p_request in FunctionRequestPB");
        return false;
    }

    const auto& p2p_request = request.p2p_request();
    int64_t     request_id  = p2p_request.request_id();
    std::string unique_key  = p2p_request.unique_key();
    int64_t     deadline_ms = p2p_request.deadline_ms();

    const auto reject_role = [&response, &p2p_request]() {
        ErrorInfo error_info(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED,
                             "P2P request type does not match connector role: "
                                 + std::to_string(p2p_request.type()));
        setP2PResponse(response, error_info);
        return false;
    };

    switch (p2p_request.type()) {
        case P2PConnectorBroadcastType::HANDLE_READ:
            return prefill_ ?
                       prefill_->processReadPerRank(request_id, unique_key, deadline_ms, p2p_request, response) :
                       reject_role();
        case P2PConnectorBroadcastType::HANDLE_READ_NO_TRANSFER:
            return prefill_ ?
                       prefill_->processNoTransferPerRank(
                           request_id, unique_key, deadline_ms, p2p_request, response) :
                       reject_role();
        case P2PConnectorBroadcastType::READ:
            return decode_ ? decode_->readPerRank(request_id, unique_key, deadline_ms, p2p_request, response) :
                             reject_role();
        case P2PConnectorBroadcastType::CANCEL_READ:
            return decode_ ? decode_->cancelReadPerRank(unique_key, p2p_request.request_deadline_ms(), response) :
                             reject_role();
        case P2PConnectorBroadcastType::CANCEL_HANDLE_READ:
            return prefill_ ? prefill_->cancelProcessReadPerRank(unique_key, response) : reject_role();
        case P2PConnectorBroadcastType::QUERY_LEASE_STATUS:
            return decode_ ? decode_->queryLeaseStatusPerRank(unique_key, response) : reject_role();
        default:
            RTP_LLM_LOG_WARNING("executeFunction failed, unsupported p2p_request type %d", p2p_request.type());
            auto* p2p_response = response.mutable_p2p_response();
            p2p_response->set_error_code(transErrorCodeToRPC(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED));
            p2p_response->set_error_message("unsupported p2p_request type");
            return false;
    }
}

}  // namespace rtp_llm

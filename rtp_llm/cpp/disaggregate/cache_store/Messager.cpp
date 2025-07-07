#include "rtp_llm/cpp/disaggregate/cache_store/Messager.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheTransferServiceClosure.h"

#include "autil/NetUtil.h"

namespace rtp_llm {

CacheLoadRequest* Messager::makeLoadRequest(const std::shared_ptr<LoadRequest>& request) {
    auto blocks = request->request_block_buffer->getBlocks();

    auto load_request = new CacheLoadRequest;
    load_request->set_timeout_ms(request->timeout_ms - 10);
    load_request->set_requestid(request->request_block_buffer->getRequestId());
    load_request->set_client_ip(autil::NetUtil::getBindIp());
    load_request->set_request_send_start_time_us(currentTimeUs());
    load_request->set_partition_count(request->partition_count);
    load_request->set_partition_id(request->partition_id);

    for (auto& [key, block] : blocks) {
        auto block_msg = load_request->add_blocks();
        block_msg->set_key(block->key);
        block_msg->set_len(block->len);
    }
    return load_request;
}

void Messager::transfer(const std::shared_ptr<TransferRequest>& request) {
    RTP_LLM_LOG_DEBUG("transfer engine start to transfer, ip %s:%u", request->ip.c_str(), request->port);

    auto channel = tcp_client_->getChannel(request->ip, request->port);
    if (!channel) {
        RTP_LLM_LOG_WARNING("messager client read get channel failed, request %s, ip %s:%u",
                            request->request_id.c_str(),
                            request->ip.c_str(),
                            request->port);
        request->callback(false, CacheStoreErrorCode::LoadConnectFailed, request->buffer_pairs);
        return;
    }

    auto transfer_request = makeTransferRequest(request);
    if (transfer_request == nullptr) {
        RTP_LLM_LOG_WARNING("messager client generate read request failed, request %s", request->request_id.c_str());
        request->callback(false, CacheStoreErrorCode::LoadSendRequestFailed, request->buffer_pairs);
        return;
    }

    auto                     transfer_response = new CacheTransferResponse;
    arpc::ANetRPCController* controller        = new arpc::ANetRPCController();
    controller->SetExpireTime(request->timeout_ms);
    auto closure = new CacheTransferServiceClosure(request, transfer_request, transfer_response, controller);

    KvCacheStoreService_Stub stub((::google::protobuf::RpcChannel*)(channel.get()),
                                  ::google::protobuf::Service::STUB_DOESNT_OWN_CHANNEL);
    stub.transfer(controller, transfer_request, transfer_response, closure);
}

CacheTransferRequest* Messager::makeTransferRequest(const std::shared_ptr<TransferRequest>& request) {
    auto transfer_request = new CacheTransferRequest;

    auto transfer_info = transfer_request->mutable_transfer_info();

    transfer_info->set_partition_count(request->remote_partition_count);
    transfer_info->set_partition_id(request->remote_partition_id);

    for (auto& [local_key, remote_key] : request->buffer_pairs) {
        transfer_info->add_local_keys(local_key);
        transfer_info->add_remote_keys(remote_key);
    }

    for (auto& [local_key, remote_key] : request->buffer_pairs) {
        auto block_buffer = request_block_buffer_store_->findUserBuffer(local_key);
        if (block_buffer == nullptr || block_buffer->len % request->local_partition_count != 0) {
            RTP_LLM_LOG_WARNING("messager client find user buffer failed or len not match, local key %s, len %d",
                                local_key.c_str(),
                                block_buffer->len);
            delete transfer_request;
            return nullptr;
        }

        auto block_info = transfer_request->add_blocks();
        if (!generateBlockInfo(block_info, block_buffer, request->local_partition_count, request->local_partition_id)) {
            delete transfer_request;
            return nullptr;
        }
    }

    transfer_request->set_request_id(request->request_id);
    transfer_request->set_client_ip(autil::NetUtil::getBindIp());
    transfer_request->set_client_port(init_params_.server_port);
    transfer_request->set_rdma_client_port(init_params_.rdma_server_port);
    transfer_request->set_timeout_ms(request->timeout_ms - 10);
    return transfer_request;
}

}  // namespace rtp_llm
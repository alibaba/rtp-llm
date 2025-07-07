#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreServiceImpl.h"

#include "rtp_llm/cpp/utils/Logger.h"

#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <unistd.h>

namespace rtp_llm {

CacheStoreServiceImpl::CacheStoreServiceImpl(
    const std::shared_ptr<MemoryUtil>&               memory_util,
    const std::shared_ptr<RequestBlockBufferStore>&  request_block_buffer_store,
    const kmonitor::MetricsReporterPtr&              metrics_reporter,
    const std::shared_ptr<TimerManager>&             timer_manager,
    const std::shared_ptr<LockedBlockBufferManager>& locked_block_buffer_manager):
    memory_util_(memory_util),
    request_block_buffer_store_(request_block_buffer_store),
    metrics_reporter_(metrics_reporter),
    timer_manager_(timer_manager),
    locked_block_buffer_manager_(locked_block_buffer_manager) {}

void CacheStoreServiceImpl::load(::google::protobuf::RpcController* controller,
                                 const ::CacheLoadRequest*          request,
                                 ::CacheLoadResponse*               response,
                                 ::google::protobuf::Closure*       done) {
    if (request_block_buffer_store_ == nullptr) {
        RTP_LLM_LOG_WARNING(
            "cache store service has no block cache store, request failed, request from [%s], request id [%s]",
            request->client_ip().c_str(),
            request->requestid().c_str());
        response->set_error_code(KvCacheStoreServiceErrorCode::EC_FAILED_INTERNAL);
        done->Run();
        return;
    }
    loadImpl(controller, request, response, done);
}

void CacheStoreServiceImpl::transfer(::google::protobuf::RpcController* controller,
                                     const ::CacheTransferRequest*      request,
                                     CacheTransferResponse*             response,
                                     ::google::protobuf::Closure*       done) {
    RTP_LLM_LOG_DEBUG("recv transfer request %s", request->ShortDebugString().c_str());

    // get peer block infos
    std::map<std::string, std::shared_ptr<BlockBufferInfo>> remote_block_infos;
    for (auto i = 0; i < request->blocks_size(); i++) {
        auto block_info                       = std::make_shared<BlockBufferInfo>(request->blocks(i));
        remote_block_infos[block_info->key()] = block_info;
    }

    // get local block and peer block pair
    std::vector<std::shared_ptr<BlockBuffer>>     local_blocks;
    std::vector<std::shared_ptr<BlockBufferInfo>> remote_blocks;

    auto& request_id    = request->request_id();
    auto& client_ip     = request->client_ip();
    auto& transfer_info = request->transfer_info();

    for (int i = 0; i < transfer_info.local_keys_size(); i++) {
        // revserse on client to server
        auto& local_key  = transfer_info.remote_keys(i);
        auto& remote_key = transfer_info.local_keys(i);

        auto local_block = request_block_buffer_store_->findUserBuffer(local_key);
        if (local_block == nullptr) {
            RTP_LLM_LOG_WARNING(
                "cache store service transfer get local block %s failed, request id is %s, request from %s",
                local_key.c_str(),
                request_id.c_str(),
                client_ip.c_str());
            response->set_error_code(KvCacheStoreServiceErrorCode::EC_FAILED_LOAD_BUFFER);
            done->Run();
            return;
        }

        auto iter = remote_block_infos.find(remote_key);
        if (iter == remote_block_infos.end()) {
            RTP_LLM_LOG_WARNING("cache store service get remote block failed, request id is %s, request from %s",
                                remote_key.c_str(),
                                request_id.c_str(),
                                client_ip.c_str());
            response->set_error_code(KvCacheStoreServiceErrorCode::EC_FAILED_LOAD_BUFFER);
            done->Run();
            return;
        }

        // verify length
        if (transfer_info.partition_count() == 0 || local_block->len % transfer_info.partition_count() != 0
            || local_block->len / transfer_info.partition_count() != iter->second->len()) {
            RTP_LLM_LOG_WARNING(
                "cache store service verify local block %s failed, len %d, partition count %d, remote block len %d,  request id is %s, request from %s",
                local_key.c_str(),
                local_block->len,
                transfer_info.partition_count(),
                iter->second->len(),
                request_id.c_str(),
                client_ip.c_str());
            response->set_error_code(KvCacheStoreServiceErrorCode::EC_FAILED_LOAD_BUFFER);
            done->Run();
            return;
        }

        if (transfer_info.partition_count() == 1) {
            local_blocks.emplace_back(local_block);
        } else {
            auto new_block  = std::make_shared<BlockBuffer>(*local_block);
            new_block->len  = local_block->len / transfer_info.partition_count();
            new_block->addr = std::shared_ptr<void>(
                (void*)((int64_t)(local_block->addr.get()) + transfer_info.partition_id() * new_block->len),
                [](void*) {});
            local_blocks.push_back(new_block);
        }
        remote_blocks.emplace_back(iter->second);
    }
    transferImpl(controller, request, response, done, local_blocks, remote_blocks);
}

void CacheStoreServiceImpl::blockRead(::google::protobuf::RpcController* controller,
                                      const ::BlockReadRequest*          request,
                                      ::BlockReadResponse*               response,
                                      ::google::protobuf::Closure*       done) {
    blockReadImpl(controller, request, response, done);
}

}  // namespace rtp_llm

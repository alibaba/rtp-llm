#include "rtp_llm/cpp/disaggregate/cache_store_new/CacheStoreClientService.h"

namespace rtp_llm {

CacheStoreClientService::CacheStoreClientService(const std::shared_ptr<LoadContextStore>& load_context_store):
    load_context_store_(load_context_store) {}

CacheStoreClientService::~CacheStoreClientService() {}

void CacheStoreClientService::transfer(::google::protobuf::RpcController* controller,
                                       const ::TransferRequest*           request,
                                       ::TransferResponse*                response,
                                       ::google::protobuf::Closure*       done) {
    auto load_context = load_context_store_->getLoadContext(request->context_id());
    if (!load_context) {
        response->set_success(false);
        response->set_info("load context not found");
        done->Run();
        return;
    }

    if (load_context->isDone()) {
        RTP_LLM_LOG_WARNING("load context is done, context id: %lld", request->context_id());
        response->set_success(false);
        response->set_info("load context already done");
        done->Run();
        return;
    }

    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    for (auto& layer_block_info : request->layer_blocks()) {
        auto layer_cache_buffer = load_context->getLayerCacheBuffer(layer_block_info.layer_id());
        if (!layer_cache_buffer) {
            RTP_LLM_LOG_WARNING("layer cache buffers not found, layer id: %d", layer_block_info.layer_id());
            response->set_success(false);
            response->set_info("layer cache buffers not found");
            done->Run();
            return;
        }
        layer_cache_buffers.push_back(layer_cache_buffer);
    }

    loadLayerBlocks(layer_cache_buffers, request, response, done);
}

void CacheStoreClientService::loadLayerBlocks(const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                                              const ::TransferRequest*                              request,
                                              ::TransferResponse*                                   response,
                                              ::google::protobuf::Closure*                          done) {
    // TODO: TCP Load : copy content to buffer
    // TODO: RDMA Load : call read
}

}  // namespace rtp_llm
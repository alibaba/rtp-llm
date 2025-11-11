#include "rtp_llm/cpp/cache_new/p2p_connector/CacheStoreClientService.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace cache_store {

CacheStoreClientService::CacheStoreClientService(const std::shared_ptr<LoadContextStore>& load_context_store,
                                                 rtp_llm::DeviceBase*                     device):
    load_context_store_(load_context_store), device_(device) {}

CacheStoreClientService::~CacheStoreClientService() {}

void CacheStoreClientService::transfer(::google::protobuf::RpcController*                  controller,
                                       const cache_store_proto::LayerBlockTransferRequest* request,
                                       cache_store_proto::LayerBlockTransferResponse*      response,
                                       ::google::protobuf::Closure*                        done) {
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
        if (!loadLayerBlocks(layer_cache_buffer, layer_block_info)) {
            response->set_success(false);
            response->set_info("load layer blocks failed");
            done->Run();
            return;
        }
        load_context->notifyLayerLoadDone(layer_block_info.layer_id());
    }
    response->set_success(true);
    done->Run();
}

bool CacheStoreClientService::loadLayerBlocks(const std::shared_ptr<LayerCacheBuffer>&       layer_cache_buffer,
                                              const cache_store_proto::LayerBlockBufferInfo& layer_block_info) {
    for (auto& block_info : layer_block_info.blocks()) {
        auto key          = block_info.key();
        auto block_buffer = layer_cache_buffer->getBlockCacheBuffer(key);

        auto& blocks = block_info.blocks();
        if (block_buffer == nullptr) {
            RTP_LLM_LOG_WARNING("block buffer not found, key: %lld", key);
            return false;
        }
        if (block_buffer->buffer2 != nullptr && blocks.size() != 2) {
            RTP_LLM_LOG_WARNING("block buffer size is not equal to 2, key: %lld", key);
            return false;
        }
        copyBlockBuffer(block_buffer->buffer1, blocks[0]);
        if (block_buffer->buffer2 != nullptr) {
            copyBlockBuffer(block_buffer->buffer2, blocks[1]);
        }
    }
    return true;
}

void CacheStoreClientService::copyBlockBuffer(const BufferPtr&                          block_buffer,
                                              const cache_store_proto::BlockBufferInfo& block_buffer_info) {
    auto src_buffer = rtp_llm::Buffer(rtp_llm::MemoryType::MEMORY_CPU,
                                      rtp_llm::DataType::TYPE_UINT8,
                                      {block_buffer_info.len()},
                                      block_buffer_info.content().data());
    device_->noBlockCopy({*block_buffer, src_buffer});
}
}  // namespace cache_store
}  // namespace rtp_llm
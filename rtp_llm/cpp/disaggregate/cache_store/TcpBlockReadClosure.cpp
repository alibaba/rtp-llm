#include "rtp_llm/cpp/disaggregate/cache_store/TcpBlockReadClosure.h"

#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreUtil.h"

namespace rtp_llm {

TcpBlockReadClosure::TcpBlockReadClosure(const std::vector<std::shared_ptr<BlockBuffer>>&     local_blocks,
                                         const std::vector<std::shared_ptr<BlockBufferInfo>>& remote_blocks,
                                         TransferConnection::ReadDoneCallback                 callback,
                                         BlockReadRequest*                                    request,
                                         BlockReadResponse*                                   response,
                                         arpc::ANetRPCController*                             controller):
    local_blocks_(local_blocks),
    remote_blocks_(remote_blocks),
    callback_(callback),
    request_(request),
    response_(response),
    controller_(controller),
    device_(rtp_llm::DeviceFactory::getDefaultDevice()) {}

TcpBlockReadClosure::~TcpBlockReadClosure() {
    delete request_;
    delete response_;
    delete controller_;
}

void TcpBlockReadClosure::Run() {
    if (controller_->Failed()) {
        RTP_LLM_LOG_WARNING("tcp transfer connection read failed, error is %s", controller_->ErrorText().c_str());
        end(false, CacheStoreUtil::fromArpcErrorCode(controller_->GetErrorCode()));
        return;
    }

    if (response_->error_code() != KvCacheStoreServiceErrorCode::EC_SUCCESS) {
        RTP_LLM_LOG_WARNING("tcp transfer connection read failed, error code is %d", response_->error_code());
        end(false, CacheStoreUtil::fromKvCacheStoreErrorCode(response_->error_code()));
        return;
    }

    if (response_->blocks_size() != local_blocks_.size()) {
        RTP_LLM_LOG_WARNING(
            "tcp transfer connection read failed, block size is not equal, local block size is %d, remote block size is %d",
            local_blocks_.size(),
            response_->blocks_size());
        end(false, CacheStoreErrorCode::LoadBufferTimeout);
        return;
    }

    for (int i = 0; i < response_->blocks_size(); ++i) {
        const auto& block        = response_->blocks(i);
        auto&       unload_block = local_blocks_[i];

        if (unload_block == nullptr || block.len() != unload_block->len || block.len() != block.content().size()) {
            RTP_LLM_LOG_WARNING(
                "can not find match block %s from response %s", block.key().c_str(), unload_block->key.c_str());
            end(false, CacheStoreErrorCode::LoadBufferTimeout);
            return;
        }

        auto dst_buffer = unload_block->toDeviceBuffer();
        auto src_buffer = rtp_llm::Buffer(rtp_llm::MemoryType::MEMORY_CPU,
                                          rtp_llm::DataType::TYPE_UINT8,
                                          {unload_block->len},
                                          block.content().data());
        device_->noBlockCopy({dst_buffer, src_buffer});
    }
    end(true, CacheStoreErrorCode::None);
}

void TcpBlockReadClosure::end(bool success, CacheStoreErrorCode error_code) {
    callback_(success, error_code, local_blocks_);
    delete this;
}

}  // namespace rtp_llm
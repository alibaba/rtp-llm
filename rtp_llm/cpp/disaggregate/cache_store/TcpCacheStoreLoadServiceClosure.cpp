#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreLoadServiceClosure.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreDevicePin.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include <torch/torch.h>
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

TcpCacheStoreLoadServiceClosure::~TcpCacheStoreLoadServiceClosure() {
    if (controller_) {
        delete controller_;
    }
    if (request_) {
        delete request_;
    }
    if (response_) {
        delete response_;
    }
}

void TcpCacheStoreLoadServiceClosure::Run() {
    collector_->markRequestCallEnd(currentTimeUs() - response_->response_send_start_time_us());
    if (!tryPinThreadDevice(device_id_, "cache load request")) {
        end(false, CacheStoreErrorCode::LoadErrorUnknown);
        return;
    }

    if (controller_->Failed()) {
        RTP_LLM_LOG_WARNING("cache load request failed, controller err is %d", controller_->GetErrorCode());
        end(false, CacheStoreUtil::fromArpcErrorCode(controller_->GetErrorCode()));
        return;
    }

    if (response_->error_code() != KvCacheStoreServiceErrorCode::EC_SUCCESS) {
        RTP_LLM_LOG_WARNING("cache load request failed, response err is %d", response_->error_code());
        end(false, CacheStoreUtil::fromKvCacheStoreErrorCode(response_->error_code()));
        return;
    }

    CacheStoreErrorCode copy_error = CacheStoreErrorCode::None;
    bool                copy_ran   = false;
    try {
        copy_ran = copy_fence_ ? copy_fence_->runIfOpen([&]() { copy_error = copyResponseBlocks(); })
                               : (copy_error = copyResponseBlocks(), true);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("cache load response copy failed: %s", e.what());
        end(false, CacheStoreErrorCode::LoadErrorUnknown);
        return;
    } catch (...) {
        RTP_LLM_LOG_WARNING("cache load response copy failed with unknown exception");
        end(false, CacheStoreErrorCode::LoadErrorUnknown);
        return;
    }
    if (!copy_ran) {
        end(false, CacheStoreErrorCode::LoadBufferTimeout);
        return;
    }

    // The fence permit is released before callback execution. This lock order
    // prevents deadlock with waitDone(), which closes the fence after dropping
    // the SyncContext mutex.
    end(copy_error == CacheStoreErrorCode::None, copy_error);
}

CacheStoreErrorCode TcpCacheStoreLoadServiceClosure::copyResponseBlocks() {
    if (response_->blocks_size() != request_block_buffer_->getBlocksCount()) {
        RTP_LLM_LOG_WARNING("cache load response block count not equal to request block buffer");
        return CacheStoreErrorCode::LoadBufferTimeout;
    }

    for (int i = 0; i < response_->blocks_size(); i++) {
        const auto& block        = response_->blocks(i);
        auto        unload_block = request_block_buffer_->getBlock(block.key());

        if (unload_block == nullptr || block.len() != unload_block->len) {
            RTP_LLM_LOG_WARNING("can not find match block %s from response, request is %s",
                                block.key().c_str(),
                                request_block_buffer_->getRequestId().c_str());
            return CacheStoreErrorCode::LoadBufferTimeout;
        }

        auto dst_tensor = torch::from_blob(
            unload_block->addr.get(),
            {(int64_t)unload_block->len},
            torch::TensorOptions().dtype(torch::kUInt8).device(unload_block->gpu_mem ? torch::kCUDA : torch::kCPU));
        auto src_tensor = torch::from_blob(const_cast<char*>(block.content().data()),
                                           {(int64_t)block.len()},
                                           torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
        execNoBlockCopy({dst_tensor, src_tensor});
    }
    return CacheStoreErrorCode::None;
}

void TcpCacheStoreLoadServiceClosure::end(bool success, CacheStoreErrorCode ec) {
    collector_->markEnd(success);
    callback_(success, ec);
    delete this;
}

}  // namespace rtp_llm

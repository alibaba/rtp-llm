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
    runWithCopy([](const BlockBuffer& dst, const BlockBufferInfo& block) {
        const auto options = torch::TensorOptions().dtype(torch::kUInt8);
        auto dst_tensor = torch::from_blob(
            dst.addr.get(), {int64_t(dst.len)}, options.device(dst.gpu_mem ? torch::kCUDA : torch::kCPU));
        auto src_tensor = torch::from_blob(
            const_cast<char*>(block.content().data()), {int64_t(block.len())}, options.device(torch::kCPU));
        execNoBlockCopy({dst_tensor, src_tensor});
    });
}

void TcpCacheStoreLoadServiceClosure::runWithCopy(CopyBlock copy) {
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

    auto targets = request_block_buffer_->getBlocks();
    if (response_->blocks_size() != targets.size()) {
        RTP_LLM_LOG_WARNING("cache load response block count not equal to request block buffer");
        end(false, CacheStoreErrorCode::LoadBufferTimeout);
        return;
    }

    // Validate the entire reply before writing any destination. The declared
    // byte count is not proof that the protobuf actually contains those bytes.
    std::vector<std::shared_ptr<BlockBuffer>> destinations;
    destinations.reserve(targets.size());
    for (const auto& block : response_->blocks()) {
        auto it = targets.find(block.key());
        if (it == targets.end() || !it->second || block.len() != it->second->len
            || block.content().size() != block.len()) {
            RTP_LLM_LOG_WARNING("invalid TCP cache block %s from response, request is %s",
                                block.key().c_str(),
                                request_block_buffer_->getRequestId().c_str());
            end(false, CacheStoreErrorCode::LoadBufferTimeout);
            return;
        }
        destinations.push_back(std::move(it->second));
        targets.erase(it);
    }
    try {
        for (int i = 0; i < response_->blocks_size(); ++i) {
            copy(*destinations[i], response_->blocks(i));
        }
    } catch (const std::exception& e) {
        // Do not let a failed transfer escape the ANet IO worker or report
        // a partially copied cache as usable.
        RTP_LLM_LOG_WARNING(
            "TCP cache load copy failed, request %s: %s", request_block_buffer_->getRequestId().c_str(), e.what());
        end(false, CacheStoreErrorCode::LoadErrorUnknown);
        return;
    }
    end(true, CacheStoreErrorCode::None);
}

void TcpCacheStoreLoadServiceClosure::end(bool success, CacheStoreErrorCode ec) {
    collector_->markEnd(success);
    callback_(success, ec);
    delete this;
}

}  // namespace rtp_llm

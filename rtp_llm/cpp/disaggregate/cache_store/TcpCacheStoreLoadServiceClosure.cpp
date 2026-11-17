#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreLoadServiceClosure.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreDevicePin.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include <torch/torch.h>
#include <cstdlib>
#include <cstring>
#if USING_ROCM
#include <unordered_map>
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include <hip/hip_runtime.h>
#endif
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
#if USING_ROCM
namespace {
struct TcpHipCopyStream {
    explicit TcpHipCopyStream(int device): device_(device) {
        const auto error = hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking);
        TORCH_CHECK(error == hipSuccess, "Failed to create TCP HIP copy stream: ", hipGetErrorString(error));
    }
    ~TcpHipCopyStream() {
        int previous = -1;
        if (hipGetDevice(&previous) == hipSuccess) {
            if (hipSetDevice(device_) == hipSuccess) {
                (void)hipStreamDestroy(stream_);
            }
            (void)hipSetDevice(previous);
        }
    }
    TcpHipCopyStream(const TcpHipCopyStream&) = delete;
    TcpHipCopyStream& operator=(const TcpHipCopyStream&) = delete;
    const int device_;
    hipStream_t stream_{nullptr};
};

hipStream_t tcpHipCopyStream(int device) {
    // A dedicated stream per callback thread/device never aliases the Torch
    // stream pool used by inference and asynchronous MTP preparation.
    static thread_local std::unordered_map<int, std::unique_ptr<TcpHipCopyStream>> streams;
    auto& owner = streams[device];
    if (!owner) owner = std::make_unique<TcpHipCopyStream>(device);
    return owner->stream_;
}
}  // namespace
#endif

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

    // TCP Mode 下需要Copy数据
    if (response_->blocks_size() != request_block_buffer_->getBlocksCount()) {
        RTP_LLM_LOG_WARNING("cache load response block count not equal to request block buffer");
        end(false, CacheStoreErrorCode::LoadBufferTimeout);
        return;
    }

#if USING_ROCM
    const char* split_mode = std::getenv("ATOM_RTP_SPLIT_SSM_STORAGE");
    // Request-owned SSM does not invoke the legacy ATOM environment setup.
    const bool isolated_copy = isolated_rocm_copy_stream_ || (split_mode != nullptr && std::strcmp(split_mode, "1") == 0);
    MultiCopyParams copies;
    bool has_gpu_copy = false;
#endif
    for (int i = 0; i < response_->blocks_size(); i++) {
        const auto& block        = response_->blocks(i);
        auto        unload_block = request_block_buffer_->getBlock(block.key());

        if (unload_block == nullptr || block.len() != unload_block->len) {
            RTP_LLM_LOG_WARNING("can not find match block %s from response, request is %s",
                                block.key().c_str(),
                                request_block_buffer_->getRequestId().c_str());
            end(false, CacheStoreErrorCode::LoadBufferTimeout);
            return;
        }

        auto dst_tensor = torch::from_blob(
            unload_block->addr.get(),
            {(int64_t)unload_block->len},
            torch::TensorOptions().dtype(torch::kUInt8).device(unload_block->gpu_mem ? torch::kCUDA : torch::kCPU));
        auto src_tensor = torch::from_blob(const_cast<char*>(block.content().data()),
                                           {(int64_t)block.len()},
                                           torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
#if USING_ROCM
        if (isolated_copy) {
            has_gpu_copy = has_gpu_copy || dst_tensor.is_cuda();
            copies.multi_dst.push_back(std::move(dst_tensor));
            copies.multi_src.push_back(std::move(src_tensor));
        } else
#endif
        {
            execNoBlockCopy({dst_tensor, src_tensor});
        }
    }
#if USING_ROCM
    if (isolated_copy) {
        if (!has_gpu_copy) {
            for (size_t i = 0; i < copies.multi_dst.size(); ++i) {
                copies.multi_dst[i].copy_(copies.multi_src[i]);
            }
            end(true, CacheStoreErrorCode::None);
            return;
        }
        // Keep TCP callbacks off the inference stream. The response owns the
        // source bytes, so completion must precede callback_ and delete this.
        const auto copy_stream = tcpHipCopyStream(device_id_);
        hipError_t copy_status = hipSuccess;
        for (size_t i = 0; i < copies.multi_dst.size(); ++i) {
            const auto& dst = copies.multi_dst[i];
            const auto& src = copies.multi_src[i];
            if (!dst.is_cuda()) {
                dst.copy_(src);
                continue;
            }
            copy_status = hipMemcpyAsync(dst.data_ptr(), src.data_ptr(), src.nbytes(),
                                         hipMemcpyHostToDevice, copy_stream);
            if (copy_status != hipSuccess) break;
        }
        // Also drain successful earlier submissions on an enqueue failure.
        const auto sync_status = hipStreamSynchronize(copy_stream);
        if (copy_status != hipSuccess || sync_status != hipSuccess) {
            RTP_LLM_LOG_WARNING("ROCm TCP cache copy failed: enqueue=%s sync=%s",
                                hipGetErrorString(copy_status), hipGetErrorString(sync_status));
            end(false, CacheStoreErrorCode::LoadErrorUnknown);
            return;
        }
        end(true, CacheStoreErrorCode::None);
        return;
    }
#endif
    end(true, CacheStoreErrorCode::None);
}

void TcpCacheStoreLoadServiceClosure::end(bool success, CacheStoreErrorCode ec) {
    collector_->markEnd(success);
    callback_(success, ec);
    delete this;
}

}  // namespace rtp_llm

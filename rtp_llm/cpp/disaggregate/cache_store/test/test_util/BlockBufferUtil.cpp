#include "rtp_llm/cpp/disaggregate/cache_store/test/test_util/BlockBufferUtil.h"

#include "rtp_llm/cpp/utils/Logger.h"

#include <atomic>
#include <cuda.h>
#include <cuda_runtime.h>

#include <atomic>

namespace rtp_llm {

BlockBufferUtil::BlockBufferUtil(const std::shared_ptr<MemoryUtil>& memory_util,
                                 const std::shared_ptr<DeviceUtil>& device_util):
    memory_util_(memory_util), device_util_(device_util) {}

BlockBufferUtil::~BlockBufferUtil() {}

std::shared_ptr<BlockBuffer>
BlockBufferUtil::makeBlockBuffer(const std::string& key, uint32_t len, char val, bool gpu) {
    void* buffer = gpu ? device_util_->mallocGPU(len) : device_util_->mallocCPU(len);
    if (buffer == nullptr) {
        RTP_LLM_LOG_WARNING("block buffer util malloc failed");
        return nullptr;
    }

    std::shared_ptr<void> shared_buffer(buffer, [memory_util = memory_util_, device_util = device_util_, gpu](void* p) {
        memory_util->deregUserMr(p, true);
        if (gpu) {
            device_util->freeGPU(p);
        } else {
            device_util->freeCPU(p);
        }
    });

    if (!memory_util_->regUserMr(buffer, len, true)) {
        RTP_LLM_LOG_WARNING("block buffer reg user mr failed");
        return nullptr;
    }

    if (gpu) {
        if (!device_util_->memsetGPU(buffer, val, len)) {
            RTP_LLM_LOG_WARNING("block buffer memset gpu failed");
            return nullptr;
        }
        // memsetGPU sync
        device_util_->device_->syncAndCheck();
    } else {
        device_util_->memsetCPU(buffer, val, len);
    }
    return std::make_shared<BlockBuffer>(key, shared_buffer, len, gpu, true);
}

std::vector<std::shared_ptr<RequestBlockBuffer>> BlockBufferUtil::makeRequestBlockBufferVec(
    const std::string& requestid, int layer_num, int block_num, uint32_t block_size) {

    size_t mem_len = layer_num * block_num * block_size * 2;  // for k/v block

    void* mem;
    auto  ret = cudaMalloc(&mem, mem_len);
    if (ret != cudaSuccess) {
        RTP_LLM_LOG_WARNING("block buffer util malloc gpu failed, len is %lu, ret is %d", mem_len, ret);
        return {};
    }

    if (!memory_util_->regUserMr(mem, mem_len, true)) {
        RTP_LLM_LOG_WARNING("block buffer util reg user mr failed, len is %lu", mem_len);
        return {};
    }

    cudaMemset(mem, 'a', mem_len);

    uint64_t k_block_start = reinterpret_cast<uint64_t>(mem);
    uint64_t v_block_start = reinterpret_cast<uint64_t>(mem) + mem_len / 2;

    std::shared_ptr<std::atomic_int> ref_count(new std::atomic_int(0));

    auto del_func = [mem, ref_count, memory_util = memory_util_, device_util = device_util_](void* p) {
        (*ref_count)--;
        if (ref_count->load() == 0) {
            memory_util->deregUserMr(mem, true);
            cudaFree(mem);
        }
    };

    std::vector<std::shared_ptr<RequestBlockBuffer>> request_block_buffer_vec;
    for (int i = 0; i < layer_num; i++) {
        auto request_block_buffer = std::make_shared<RequestBlockBuffer>(requestid);
        for (int j = 0; j < block_num; j++) {
            std::string cache_key = "token_id_str_" + std::to_string(j) + "_layer_id_" + std::to_string(i);

            (*ref_count)++;
            std::shared_ptr<void> k_block(reinterpret_cast<void*>(k_block_start + (i * block_num + j) * block_size),
                                          del_func);
            request_block_buffer->addBlock("k_" + cache_key, k_block, block_size, true, true);

            (*ref_count)++;
            std::shared_ptr<void> v_block(reinterpret_cast<void*>(v_block_start + (i * block_num + j) * block_size),
                                          del_func);
            request_block_buffer->addBlock("v_" + cache_key, v_block, block_size, true, true);
        }
        request_block_buffer_vec.push_back(request_block_buffer);
    }
    return request_block_buffer_vec;
}

bool BlockBufferUtil::verifyBlock(
    const std::shared_ptr<BlockBuffer>& block, const std::string& key, uint32_t len, bool gpu_mem, char val) {
    if (block == nullptr) {
        return false;
    }

    if (key != block->key || len != block->len || gpu_mem != block->gpu_mem) {
        return false;
    }

    if (len == 0) {
        return true;
    }

    if (!gpu_mem) {
        return val == ((char*)(block->addr.get()))[0];
    }

    auto buf = device_util_->mallocCPU(len);
    if (buf == nullptr) {
        return false;
    }

    auto ret = device_util_->memcopy(buf, false, block->addr.get(), block->gpu_mem, len);
    if (!ret) {
        device_util_->freeCPU(buf);
        return false;
    }

    ret = val == ((char*)(buf))[0];
    device_util_->freeCPU(buf);
    return ret;
}

}  // namespace rtp_llm
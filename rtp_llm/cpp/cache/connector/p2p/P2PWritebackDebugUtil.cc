#include "rtp_llm/cpp/cache/connector/p2p/P2PWritebackDebugUtil.h"

#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/CudaCopyUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace rtp_llm {
namespace {

constexpr size_t   kChecksumWindowBytes = 256;
constexpr uint64_t kFnvOffsetBasis      = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime            = 1099511628211ULL;

struct SampleWindow {
    size_t offset = 0;
    size_t size   = 0;
};

uint64_t updateChecksum(uint64_t checksum, const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
        checksum ^= bytes[i];
        checksum *= kFnvPrime;
    }
    return checksum;
}

uint64_t updateChecksumWithIntegral(uint64_t checksum, uint64_t value) {
    return updateChecksum(checksum, &value, sizeof(value));
}

std::vector<SampleWindow> sampleWindows(size_t size_bytes) {
    if (size_bytes == 0) {
        return {};
    }
    if (size_bytes <= kChecksumWindowBytes * 3) {
        return {{0, size_bytes}};
    }

    const size_t middle_offset = (size_bytes / 2) - (kChecksumWindowBytes / 2);
    return {
        {0, kChecksumWindowBytes},
        {middle_offset, kChecksumWindowBytes},
        {size_bytes - kChecksumWindowBytes, kChecksumWindowBytes},
    };
}

bool copySampleToHost(const BlockInfo& block, const SampleWindow& window, std::vector<char>& host) {
    if (window.size == 0 || block.addr == nullptr) {
        return false;
    }
    host.resize(window.size);
    auto* src = static_cast<char*>(block.addr) + window.offset;
    if (!block.is_cuda) {
        std::memcpy(host.data(), src, window.size);
        return true;
    }

    transfer::tcp::CopyTask              task{src, window.size, host.data()};
    std::vector<transfer::tcp::CopyTask> tasks{task};
    transfer::tcp::CudaCopyUtil          copy_util;
    return copy_util.batchCopyToHost(tasks);
}

struct BlockChecksum {
    uint64_t checksum            = kFnvOffsetBasis;
    size_t   sampled_bytes       = 0;
    size_t   total_bytes         = 0;
    int      physical_block_num  = 0;
    bool     copied_successfully = true;
};

BlockChecksum checksumBlocks(const std::vector<BlockInfo>& blocks) {
    BlockChecksum     result;
    std::vector<char> host;
    for (const auto& block : blocks) {
        result.total_bytes += block.size_bytes;
        ++result.physical_block_num;
        result.checksum = updateChecksumWithIntegral(result.checksum, block.size_bytes);
        for (const auto& window : sampleWindows(block.size_bytes)) {
            result.checksum = updateChecksumWithIntegral(result.checksum, window.offset);
            result.checksum = updateChecksumWithIntegral(result.checksum, window.size);
            if (!copySampleToHost(block, window, host)) {
                result.copied_successfully = false;
                continue;
            }
            result.sampled_bytes += host.size();
            result.checksum = updateChecksum(result.checksum, host.data(), host.size());
        }
    }
    return result;
}

}  // namespace

bool pdKvWritebackChecksumDebugEnabled() {
    static const bool enabled = autil::EnvUtil::getEnv("PD_KV_WRITEBACK_DEBUG_CHECKSUM", false);
    return enabled;
}

void logPdKvWritebackChecksum(const std::string&                          stage,
                              int64_t                                     request_id,
                              const std::string&                          unique_key,
                              const std::shared_ptr<LayerBlockConverter>& converter,
                              const std::shared_ptr<LayerCacheBuffer>&    layer_cache_buffer,
                              int                                         partition_count,
                              int                                         partition_id) {
    if (!pdKvWritebackChecksumDebugEnabled() || !converter || !layer_cache_buffer) {
        return;
    }

    auto key_block_infos =
        LayerCacheBufferUtil::buildKeyBlockInfos(converter, layer_cache_buffer, partition_count, partition_id);
    for (const auto& [cache_key, key_block_info] : key_block_infos) {
        if (!key_block_info) {
            continue;
        }
        const auto checksum = checksumBlocks(key_block_info->blocks);
        RTP_LLM_LOG_INFO(
            "PD KV writeback checksum stage=%s request_id=%ld unique_key=%s layer_id=%d partition=%d/%d cache_key=%ld logical_block_id=%d physical_blocks=%d total_bytes=%zu sampled_bytes=%zu checksum=%llu copy_ok=%d",
            stage.c_str(),
            request_id,
            unique_key.c_str(),
            layer_cache_buffer->getLayerId(),
            partition_id,
            partition_count,
            cache_key,
            layer_cache_buffer->blockIdMap().at(cache_key),
            checksum.physical_block_num,
            checksum.total_bytes,
            checksum.sampled_bytes,
            static_cast<unsigned long long>(checksum.checksum),
            checksum.copied_successfully);
    }
}

void logPdKvWritebackChecksum(const std::string&                                    stage,
                              int64_t                                               request_id,
                              const std::string&                                    unique_key,
                              const std::shared_ptr<LayerBlockConverter>&           converter,
                              const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                              int                                                   partition_count) {
    if (!pdKvWritebackChecksumDebugEnabled()) {
        return;
    }
    for (const auto& layer_cache_buffer : layer_cache_buffers) {
        for (int partition_id = 0; partition_id < partition_count; ++partition_id) {
            logPdKvWritebackChecksum(
                stage, request_id, unique_key, converter, layer_cache_buffer, partition_count, partition_id);
        }
    }
}

}  // namespace rtp_llm

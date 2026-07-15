#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>

#include "rtp_llm/cpp/cache/block_tree_cache/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/device_group/DeviceSWAKVCacheGroup.h"

namespace rtp_llm {
namespace test {
namespace {

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value): name_(name) {
        const char* old_value = std::getenv(name_);
        if (old_value != nullptr) {
            old_value_ = old_value;
            had_value_ = true;
        }
        setenv(name_, value, 1);
    }

    ~ScopedEnvVar() {
        if (had_value_) {
            setenv(name_, old_value_.c_str(), 1);
        } else {
            unsetenv(name_);
        }
    }

private:
    const char* name_;
    std::string old_value_;
    bool        had_value_ = false;
};

DeviceBlockPoolConfig makeDeviceBlockPoolConfig() {
    constexpr uint32_t kLayerNum        = 1;
    constexpr uint32_t kBlockNum        = 4;
    constexpr size_t   kKvBlockStride   = 1024;
    constexpr size_t   kHalfBlockStride = kKvBlockStride / 2;

    MemoryLayoutConfig layout;
    layout.layer_num                = kLayerNum;
    layout.block_num                = kBlockNum;
    layout.dtype                    = DataType::TYPE_FP16;
    layout.kv_cache_offset_bytes    = 0;
    layout.kv_scale_offset_bytes    = kLayerNum * kBlockNum * kKvBlockStride;
    layout.kv_block_stride_bytes    = kKvBlockStride;
    layout.k_block_stride_bytes     = kHalfBlockStride;
    layout.v_block_stride_bytes     = kHalfBlockStride;
    layout.kv_block_pool_size_bytes = kLayerNum * kBlockNum * kKvBlockStride;
    layout.kv_scale_pool_size_bytes = 0;
    layout.total_size_bytes         = layout.kv_block_pool_size_bytes;

    DeviceBlockPoolConfig config;
    config.pool_type               = BlockPoolType::DEVICE;
    config.pool_name               = "swa_malloc_range_test";
    config.physical_block_count    = kBlockNum;
    config.total_size_bytes        = layout.total_size_bytes;
    config.memory_layouts          = {layout};
    config.allocation_type         = AllocationType::DEVICE;
    config.use_cuda_malloc_backing = false;
    return config;
}

// DeviceSWAKVCacheGroup only accepts a DeviceBlockPoolPtr, and DeviceBlockPool's init() rejects any
// allocation_type other than DEVICE. This test exercises only the SWA tail-block placement
// pattern (which slots are NULL vs REAL) and the free path, both memory-medium-agnostic.
DeviceBlockPoolPtr createDeviceBlockPool() {
    auto device_config = std::make_shared<DeviceBlockPoolConfig>(makeDeviceBlockPoolConfig());

    std::shared_ptr<const DeviceBlockPoolConfig> const_config = device_config;
    auto                                         block_pool   = std::make_shared<DeviceBlockPool>(const_config);
    RTP_LLM_CHECK_WITH_INFO(block_pool->init(), "init device block pool failed");
    return block_pool;
}

std::shared_ptr<MHAKVCacheSpec> makeMHASpec(int seq_size_per_block) {
    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = seq_size_per_block;
    return spec;
}

}  // namespace

TEST(DeviceSWAKVCacheGroupMallocRangeTest, EmptyBlockIdsKeepTailBlocksForSeqLenUpTo1M) {
    constexpr int kSeqSizePerBlock = 256;
    constexpr int kMaxSeqLen       = 1000000;

    ScopedEnvVar          disable_pin_host_pool("RTP_LLM_PIN_HOST_BLOCK_POOL", "0");
    auto                  block_pool = createDeviceBlockPool();
    DeviceSWAKVCacheGroup group({}, makeMHASpec(kSeqSizePerBlock), block_pool, 0);

    auto check_seq_len = [&](int seq_len) {
        BlockIds block_ids;
        ASSERT_EQ(block_ids.blocksNum(), 0u) << "seq_len=" << seq_len;

        ASSERT_TRUE(group.malloc(block_ids, seq_len, /*enable_reuse_cache=*/false, /*reserve_step=*/0))
            << "seq_len=" << seq_len;

        const auto& blocks = block_ids.blocks();
        ASSERT_EQ(blocks.size(), static_cast<size_t>((seq_len + kSeqSizePerBlock - 1) / kSeqSizePerBlock))
            << "seq_len=" << seq_len;
        if (blocks.size() == 1) {
            EXPECT_FALSE(isNullBlockIdx(blocks[0])) << "seq_len=" << seq_len;
        } else {
            EXPECT_FALSE(isNullBlockIdx(blocks[blocks.size() - 2])) << "seq_len=" << seq_len;
            EXPECT_FALSE(isNullBlockIdx(blocks[blocks.size() - 1])) << "seq_len=" << seq_len;
        }

        group.free(blocks);
    };

    // SWA malloc depends on seq_slots=ceil(seq_len / block_size). The first
    // and last seq_len in each slot cover all behavior classes from 1..1M.
    const int max_seq_slots = (kMaxSeqLen + kSeqSizePerBlock - 1) / kSeqSizePerBlock;
    for (int seq_slots = 1; seq_slots <= max_seq_slots; ++seq_slots) {
        const int first_seq_len = (seq_slots - 1) * kSeqSizePerBlock + 1;
        const int last_seq_len  = std::min(seq_slots * kSeqSizePerBlock, kMaxSeqLen);
        check_seq_len(first_seq_len);
        if (last_seq_len != first_seq_len) {
            check_seq_len(last_seq_len);
        }
    }
}

}  // namespace test
}  // namespace rtp_llm

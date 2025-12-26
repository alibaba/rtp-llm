#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <thread>

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {

class KVCacheManagerTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        device_ = createDevice();
        ASSERT_NE(device_, nullptr);
    }

protected:
    rtp_llm::DeviceBase* device_ = nullptr;
};

static void assertBlockBytesEq(rtp_llm::DeviceBase*                            device,
                               const std::shared_ptr<rtp_llm::KVCacheManager>& cache_manager,
                               int                                             layer_id,
                               int                                             block_id,
                               const std::vector<int8_t>&                      expected) {
    auto addr_info = cache_manager->convertIndexToAddr(block_id, layer_id);
    ASSERT_NE(addr_info.kv_addr, nullptr);
    rtp_llm::Buffer dev_view(rtp_llm::MemoryType::MEMORY_GPU,
                             rtp_llm::DataType::TYPE_INT8,
                             {expected.size()},
                             static_cast<int8_t*>(addr_info.kv_addr));
    auto            host_buf = device->clone({dev_view, rtp_llm::AllocationType::HOST});
    ASSERT_NE(host_buf, nullptr);
    ASSERT_EQ(host_buf->sizeBytes(), expected.size());
    const auto* ptr = host_buf->data<int8_t>();
    for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_EQ(ptr[i], expected[i]) << "mismatch at byte " << i << " layer=" << layer_id << " block=" << block_id;
    }
}

static void assertScaleEq(rtp_llm::DeviceBase*                            device,
                          const std::shared_ptr<rtp_llm::KVCacheManager>& cache_manager,
                          int                                             layer_id,
                          int                                             block_id,
                          const std::vector<float>&                       expected_k,
                          const std::vector<float>&                       expected_v) {
    auto addr_info = cache_manager->convertIndexToAddr(block_id, layer_id);
    ASSERT_NE(addr_info.kv_scale_addr, nullptr);
    ASSERT_EQ(expected_k.size(), expected_v.size());

    // kv_scale_addr points to K-scale, and V-scale follows by kv_scale_block_bytes (= kv_scale_stride_bytes / 2).
    const size_t kv_scale_stride_bytes = cache_manager->cacheConfig().kv_scale_stride_bytes;
    ASSERT_GT(kv_scale_stride_bytes, 0u);
    const size_t kv_scale_block_bytes = kv_scale_stride_bytes / 2;
    void*        v_scale_addr = static_cast<void*>(static_cast<char*>(addr_info.kv_scale_addr) + kv_scale_block_bytes);

    rtp_llm::Buffer dev_k(
        rtp_llm::MemoryType::MEMORY_GPU, rtp_llm::DataType::TYPE_FP32, {expected_k.size()}, addr_info.kv_scale_addr);
    rtp_llm::Buffer dev_v(
        rtp_llm::MemoryType::MEMORY_GPU, rtp_llm::DataType::TYPE_FP32, {expected_v.size()}, v_scale_addr);

    auto host_k = device->clone({dev_k, rtp_llm::AllocationType::HOST});
    auto host_v = device->clone({dev_v, rtp_llm::AllocationType::HOST});
    ASSERT_NE(host_k, nullptr);
    ASSERT_NE(host_v, nullptr);

    const float* k_ptr = host_k->data<float>();
    const float* v_ptr = host_v->data<float>();
    for (size_t i = 0; i < expected_k.size(); ++i) {
        ASSERT_FLOAT_EQ(k_ptr[i], expected_k[i])
            << "k scale mismatch i=" << i << " layer=" << layer_id << " block=" << block_id;
        ASSERT_FLOAT_EQ(v_ptr[i], expected_v[i])
            << "v scale mismatch i=" << i << " layer=" << layer_id << " block=" << block_id;
    }
}

TEST_F(KVCacheManagerTest, WarmupConfigSmoke) {
    auto cache_config = makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/4, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);

    auto cache_manager = std::make_shared<KVCacheManager>(cache_config, device_, /*warmup=*/true);
    ASSERT_TRUE(cache_manager->init());

    EXPECT_EQ(cache_manager->cacheConfig().block_num, 1);

    EXPECT_EQ(cache_manager->totalBlocksNum(), 0);
    EXPECT_EQ(cache_manager->freeBlocksNum(), 0);
}

TEST_F(KVCacheManagerTest, MetricsThreadSmoke) {
    auto cache_config = makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/4, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);

    auto kmon_tags = kmonitor::MetricsTags();
    auto reporter  = std::make_shared<kmonitor::MetricsReporter>("", "", kmon_tags);

    auto cache_manager = std::make_shared<KVCacheManager>(cache_config, device_, /*warmup=*/true, reporter);

    ASSERT_TRUE(cache_manager->init());
    EXPECT_TRUE(cache_manager->metrics_reporter_thread_.joinable());
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));

    cache_manager.reset();
}

TEST_F(KVCacheManagerTest, SetKVBlockValueAndBlockCopy) {
    // Use non-warmup config so we have usable blocks (block 0 is reserved in BlockPool).
    auto cache_config = makeSimpleMhaCacheConfig(
        /*layer_num=*/2, /*block_num=*/6, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);
    auto cache_manager = std::make_shared<KVCacheManager>(cache_config, device_, /*warmup=*/false);
    ASSERT_TRUE(cache_manager->init());

    auto&        spec    = cache_manager->cacheConfig().cache_specs[0];
    const size_t k_bytes = spec->k_block_size_bytes();
    const size_t v_bytes = spec->v_block_size_bytes();
    ASSERT_GT(k_bytes, 0u);
    ASSERT_GT(v_bytes, 0u);

    const int block_src = 1;
    const int block_dst = 3;

    std::vector<int8_t> k_vec(k_bytes, 7);
    std::vector<int8_t> v_vec(v_bytes, 9);
    auto                k_buf = rtp_llm::vector2Buffer(k_vec);
    auto                v_buf = rtp_llm::vector2Buffer(v_vec);

    ASSERT_TRUE(cache_manager->setKVBlockValue(block_src, *k_buf, *v_buf));

    std::vector<int8_t> expected_block(k_bytes + v_bytes, 0);
    std::fill(expected_block.begin(), expected_block.begin() + k_bytes, 7);
    std::fill(expected_block.begin() + k_bytes, expected_block.end(), 9);

    // Check both layers in source block
    assertBlockBytesEq(device_, cache_manager, /*layer_id=*/0, block_src, expected_block);
    assertBlockBytesEq(device_, cache_manager, /*layer_id=*/1, block_src, expected_block);

    // Copy src -> dst and validate
    cache_manager->blockCopy(block_src, block_dst);
    assertBlockBytesEq(device_, cache_manager, /*layer_id=*/0, block_dst, expected_block);
    assertBlockBytesEq(device_, cache_manager, /*layer_id=*/1, block_dst, expected_block);

    // Now overwrite only layer 0 on dst block; layer 1 should remain unchanged.
    std::vector<int8_t> k2_vec(k_bytes, 1);
    std::vector<int8_t> v2_vec(v_bytes, 2);
    auto                k2_buf = rtp_llm::vector2Buffer(k2_vec);
    auto                v2_buf = rtp_llm::vector2Buffer(v2_vec);
    ASSERT_TRUE(cache_manager->setKVBlockValue(block_dst, /*layer_id=*/0, *k2_buf, *v2_buf));

    std::vector<int8_t> expected_layer0(k_bytes + v_bytes, 0);
    std::fill(expected_layer0.begin(), expected_layer0.begin() + k_bytes, 1);
    std::fill(expected_layer0.begin() + k_bytes, expected_layer0.end(), 2);
    assertBlockBytesEq(device_, cache_manager, /*layer_id=*/0, block_dst, expected_layer0);
    assertBlockBytesEq(device_, cache_manager, /*layer_id=*/1, block_dst, expected_block);
}

TEST_F(KVCacheManagerTest, BlockCopyAlsoCopiesScaleWhenQuantized) {
    auto cache_config = makeSimpleMhaCacheConfig(
        /*layer_num=*/2, /*block_num=*/6, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);
    auto cache_manager = std::make_shared<KVCacheManager>(cache_config, device_, /*warmup=*/false);
    ASSERT_TRUE(cache_manager->init());

    auto kv_buf = cache_manager->kvCacheBuffer();
    ASSERT_NE(kv_buf.kv_scale_blocks, nullptr);

    const int    block_src   = 1;
    const int    block_dst   = 4;
    const size_t scale_elems = 2;  // local_head_num_kv(=1) * tokens_per_block(=2)

    std::vector<float> src_k = {0.5f, 0.6f};
    std::vector<float> src_v = {1.5f, 1.6f};
    ASSERT_EQ(src_k.size(), scale_elems);
    ASSERT_EQ(src_v.size(), scale_elems);

    for (int layer_id = 0; layer_id < 2; ++layer_id) {
        auto addr = cache_manager->convertIndexToAddr(block_src, layer_id);
        ASSERT_NE(addr.kv_scale_addr, nullptr);

        auto host_k = rtp_llm::vector2Buffer(src_k);
        auto host_v = rtp_llm::vector2Buffer(src_v);

        const size_t kv_scale_stride_bytes = cache_manager->cacheConfig().kv_scale_stride_bytes;
        ASSERT_GT(kv_scale_stride_bytes, 0u);
        const size_t kv_scale_block_bytes = kv_scale_stride_bytes / 2;
        void*        v_scale_addr = static_cast<void*>(static_cast<char*>(addr.kv_scale_addr) + kv_scale_block_bytes);

        rtp_llm::Buffer dst_k(
            rtp_llm::MemoryType::MEMORY_GPU, rtp_llm::DataType::TYPE_FP32, {scale_elems}, addr.kv_scale_addr);
        rtp_llm::Buffer dst_v(
            rtp_llm::MemoryType::MEMORY_GPU, rtp_llm::DataType::TYPE_FP32, {scale_elems}, v_scale_addr);

        rtp_llm::Buffer src_k_view(host_k->where(), host_k->type(), {scale_elems}, host_k->data());
        rtp_llm::Buffer src_v_view(host_v->where(), host_v->type(), {scale_elems}, host_v->data());

        device_->copy({dst_k, src_k_view});
        device_->copy({dst_v, src_v_view});
    }
    device_->syncAndCheck();

    // Copy should include both K/V scales.
    cache_manager->blockCopy(block_src, block_dst);
    device_->syncAndCheck();

    for (int layer_id = 0; layer_id < 2; ++layer_id) {
        assertScaleEq(device_, cache_manager, layer_id, block_dst, src_k, src_v);
    }
}

TEST_F(KVCacheManagerTest, BlockBatchCopy) {
    auto cache_config = makeSimpleMhaCacheConfig(
        /*layer_num=*/2, /*block_num=*/10, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);
    auto cache_manager = std::make_shared<KVCacheManager>(cache_config, device_, /*warmup=*/false);
    ASSERT_TRUE(cache_manager->init());

    auto&        spec    = cache_manager->cacheConfig().cache_specs[0];
    const size_t k_bytes = spec->k_block_size_bytes();
    const size_t v_bytes = spec->v_block_size_bytes();

    const int src_blocks_num = 2;
    const int dst_blocks_num = 4;

    // Initialize src blocks with distinct patterns.
    for (int i = 0; i < src_blocks_num; ++i) {
        const int           block_id = 1 + i;
        std::vector<int8_t> k_vec(k_bytes, static_cast<int8_t>(block_id));
        std::vector<int8_t> v_vec(v_bytes, static_cast<int8_t>(block_id + 10));
        auto                k_buf = rtp_llm::vector2Buffer(k_vec);
        auto                v_buf = rtp_llm::vector2Buffer(v_vec);
        ASSERT_TRUE(cache_manager->setKVBlockValue(block_id, *k_buf, *v_buf));
    }

    std::vector<BlockIdPair> mapping;
    mapping.reserve(dst_blocks_num);
    for (int j = 0; j < dst_blocks_num; ++j) {
        const int dst_block = 1 + src_blocks_num + j;
        const int src_block = 1 + (j % src_blocks_num);
        mapping.push_back({src_block, dst_block});
    }

    cache_manager->blockBatchCopy(mapping);

    // Validate copied blocks for both layers.
    for (int j = 0; j < dst_blocks_num; ++j) {
        const int dst_block = 1 + src_blocks_num + j;
        const int src_block = 1 + (j % src_blocks_num);

        std::vector<int8_t> expected(k_bytes + v_bytes, 0);
        std::fill(expected.begin(), expected.begin() + k_bytes, static_cast<int8_t>(src_block));
        std::fill(expected.begin() + k_bytes, expected.end(), static_cast<int8_t>(src_block + 10));

        assertBlockBytesEq(device_, cache_manager, /*layer_id=*/0, dst_block, expected);
        assertBlockBytesEq(device_, cache_manager, /*layer_id=*/1, dst_block, expected);
    }
}

}  // namespace test
}  // namespace rtp_llm

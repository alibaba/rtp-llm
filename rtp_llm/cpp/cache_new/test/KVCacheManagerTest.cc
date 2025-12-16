#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <thread>

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache_new/KVCacheManager.h"
#include "rtp_llm/cpp/cache_new/KVCacheMemoryConnector.h"
#include "rtp_llm/cpp/cache_new/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {

static CacheConfig
makeSimpleMhaCacheConfig(int layer_num, int block_num, size_t tokens_per_block, rtp_llm::DataType dtype) {
    CacheConfig config;
    config.layer_num          = layer_num;
    config.block_num          = block_num;
    config.seq_size_per_block = tokens_per_block;

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->type               = KVCacheType::MultiHeadAttention;
    spec->dtype              = dtype;
    spec->seq_size_per_block = static_cast<uint32_t>(tokens_per_block);
    spec->layer_num          = static_cast<uint32_t>(layer_num);
    spec->block_nums         = static_cast<uint32_t>(block_num);
    spec->local_head_num_kv  = 1;
    spec->size_per_head      = 1;
    config.cache_specs.push_back(spec);

    std::vector<int> layer_ids(layer_num);
    for (int i = 0; i < layer_num; ++i) {
        layer_ids[i] = i;
    }
    config.layer_ids.push_back(layer_ids);

    config.block_stride    = static_cast<int>(spec->block_size());
    config.block_size      = static_cast<int>(spec->block_size() * spec->layer_num);
    config.k_block_stride  = spec->block_size();
    config.v_block_stride  = 0;
    config.kv_block_stride = spec->block_size();
    return config;
}

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
    auto buf_info = cache_manager->allocator_->convertIndexToBuffer(layer_id, block_id);
    ASSERT_NE(buf_info.k_addr, nullptr);
    auto host_buf = device->clone({*buf_info.k_addr, rtp_llm::AllocationType::HOST});
    ASSERT_NE(host_buf, nullptr);
    ASSERT_EQ(host_buf->sizeBytes(), expected.size());
    const auto* ptr = host_buf->data<int8_t>();
    for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_EQ(ptr[i], expected[i]) << "mismatch at byte " << i << " layer=" << layer_id << " block=" << block_id;
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

    auto cache_manager =
        std::make_shared<KVCacheManager>(cache_config, device_, /*warmup=*/true, reporter, rtp_llm::GptInitParameter{});

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
    const size_t k_bytes = spec->k_block_size();
    const size_t v_bytes = spec->v_block_size();
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

TEST_F(KVCacheManagerTest, BlockBatchCopy) {
    auto cache_config = makeSimpleMhaCacheConfig(
        /*layer_num=*/2, /*block_num=*/10, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);
    auto cache_manager = std::make_shared<KVCacheManager>(cache_config, device_, /*warmup=*/false);
    ASSERT_TRUE(cache_manager->init());

    auto&        spec    = cache_manager->cacheConfig().cache_specs[0];
    const size_t k_bytes = spec->k_block_size();
    const size_t v_bytes = spec->v_block_size();

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

// class MockKVCacheCoordinator: public KVCacheConnectorCoordinator {
// public:
//     MOCK_METHOD(bool, copyCache, (const CopyCacheRequestPB&, CopyCacheResponsePB&), (override));
//     MOCK_METHOD(void, clearCache, (), (override));
// };

// class KVCacheManagerTest: public DeviceTestBase {
// protected:
//     void SetUp() override {
//         // Use default cache config
//         CacheConfig config;
//         kv_cache_manager_ = std::make_shared<KVCacheManager>(config, device_);
//     }

//     std::shared_ptr<KVCacheManager> kv_cache_manager_;
// };

// TEST_F(KVCacheManagerTest, CopyCache_ReturnFalse_WhenNoMemRequest) {
//     CopyCacheRequestPB  request;
//     CopyCacheResponsePB response;

//     // Request has no mem_request
//     // Should return false and log warning
//     EXPECT_FALSE(kv_cache_manager_->copyCache(request, response));
// }

// TEST_F(KVCacheManagerTest, CopyCache_ReturnFalse_WhenMemoryConnectorIsNull) {
//     CopyCacheRequestPB request;
//     request.mutable_mem_request();  // Add mem_request
//     CopyCacheResponsePB response;

//     // memory_connector_ is null by default
//     kv_cache_manager_->memory_connector_ = nullptr;

//     EXPECT_FALSE(kv_cache_manager_->copyCache(request, response));
//     EXPECT_FALSE(response.mem_response().success());
// }
    // EXPECT_FALSE(kv_cache_manager_->copyCache(request, response));
    // EXPECT_FALSE(response.mem_response().success());
}

// TEST_F(KVCacheManagerTest, CopyCache_DelegatesToMemoryConnector_AndReturnsTrue) {
//     CopyCacheRequestPB request;
//     request.mutable_mem_request();
//     CopyCacheResponsePB response;

//     // Create mock connector
//     CacheConfig              config;
//     std::vector<std::string> tp_addrs;
//     auto mock_connector = std::make_shared<MockKVCacheMemoryConnector>(config, nullptr, device_, tp_addrs);

//     // Inject mock
//     kv_cache_manager_->memory_connector_ = mock_connector;

//     // Expect call
//     EXPECT_CALL(*mock_connector, copyCache(testing::_, testing::_))
//         .WillOnce(testing::Invoke([](const MemoryCopyCacheRequestPB& req, MemoryCopyCacheResponsePB& resp) {
//             resp.set_success(true);
//             return true;
//         }));
    // auto mock_coordinator = std::make_shared<MockKVCacheCoordinator>();
    // kv_cache_manager_->setConnectorCoordinatorForTest(mock_coordinator);

    // // Expect call
    // EXPECT_CALL(*mock_coordinator, copyCache(testing::_, testing::_))
    //     .WillOnce(testing::Invoke([](const CopyCacheRequestPB& req, CopyCacheResponsePB& resp) {
    //         resp.mutable_mem_response()->set_success(true);
    //         return true;
    //     }));

//     EXPECT_TRUE(kv_cache_manager_->copyCache(request, response));
//     EXPECT_TRUE(response.mem_response().success());
// }

// TEST_F(KVCacheManagerTest, CopyCache_DelegatesToMemoryConnector_AndReturnsFalse) {
//     CopyCacheRequestPB request;
//     request.mutable_mem_request();
//     CopyCacheResponsePB response;

//     // Create mock connector
//     CacheConfig              config;
//     std::vector<std::string> tp_addrs;
//     auto mock_connector = std::make_shared<MockKVCacheMemoryConnector>(config, nullptr, device_, tp_addrs);

//     // Inject mock
//     kv_cache_manager_->memory_connector_ = mock_connector;

//     // Expect call
//     EXPECT_CALL(*mock_connector, copyCache(testing::_, testing::_))
//         .WillOnce(testing::Invoke([](const MemoryCopyCacheRequestPB& req, MemoryCopyCacheResponsePB& resp) {
//             resp.set_success(false);
//             return false;
//         }));
    // auto mock_coordinator = std::make_shared<MockKVCacheCoordinator>();
    // kv_cache_manager_->setConnectorCoordinatorForTest(mock_coordinator);

    // // Expect call
    // EXPECT_CALL(*mock_coordinator, copyCache(testing::_, testing::_))
    //     .WillOnce(testing::Invoke([](const CopyCacheRequestPB& req, CopyCacheResponsePB& resp) {
    //         resp.mutable_mem_response()->set_success(false);
    //         return false;
    //     }));

//     EXPECT_FALSE(kv_cache_manager_->copyCache(request, response));
//     EXPECT_FALSE(response.mem_response().success());
// }

}  // namespace test
}  // namespace rtp_llm

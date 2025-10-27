#include "rtp_llm/cpp/cache2/ProxyCacheManager.h"
#include "rtp_llm/cpp/cache2/ProxyDistKvCache.h"
#include "rtp_llm/cpp/cache2/kvcm_client_wrapper/KVCMClientWrapper.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/cache2/test/MockKVCMClient.h"
#include "rtp_llm/cpp/cache2/test/MockProxyDistKVCache.h"

using namespace kv_cache_manager;
using namespace ::testing;

namespace rtp_llm {

class ProxyCacheManagerTest: public DeviceTestBase {
public:
private:
    CacheConfig initConfig() {
        // layer_num, block_nums, local_head_num_kv, size_per_head, seq_size_per_block, dtype
        CacheConfig config(KVCacheParam({1, 4, 1, 1, 1, rtp_llm::TYPE_INT8}));
        return config;
    }

    std::vector<int64_t> constructCacheKey(CacheManager& cache_manager, const std::vector<int>& token_ids) {
        auto                 seq_size_per_block = cache_manager.config_.seq_size_per_block;
        auto                 total_blocks       = token_ids.size() / seq_size_per_block;
        std::vector<int64_t> cache_keys;
        int64_t              hash = 0;
        for (int index = 0; index < total_blocks; index++) {
            auto start_pos = token_ids.begin() + index * seq_size_per_block;
            hash           = std::accumulate(start_pos, start_pos + seq_size_per_block, hash, std::plus<int>());
            cache_keys.push_back(hash);
        }
        return cache_keys;
    }
};

TEST_F(ProxyCacheManagerTest, testMatchInKvcmSuccess) {
    auto cache_config               = initConfig();
    cache_config.block_nums         = 10;
    cache_config.seq_size_per_block = 1;
    GptInitParameter param;
    param.kv_cache_config.enable_dist_kvcache = true;
    ProxyCacheManager cache_manager(cache_config, device_, false, nullptr, param);

    auto mock_proxy_dist_kv_cache_ptr = std::make_shared<MockProxyDistKVCache>(&cache_manager, param, nullptr);
    auto kvcm_client_wrapper_ptr      = std::make_shared<KVCMClientWrapper>();
    auto mock_meta_client_ptr         = std::make_shared<MockMetaClient>();
    auto mock_transfer_client_ptr     = std::make_unique<MockTransferClient>();

    kvcm_client_wrapper_ptr->meta_client_map_[""]      = mock_meta_client_ptr;
    kvcm_client_wrapper_ptr->config_map_[""]           = std::make_shared<KVCMClientWrapperConfig>();
    KVCMClientWrapper::transfer_client_                = std::move(mock_transfer_client_ptr);
    mock_proxy_dist_kv_cache_ptr->kvcm_client_wrapper_ = kvcm_client_wrapper_ptr;
    mock_proxy_dist_kv_cache_ptr->wait_match_thread_pool_ =
        std::make_unique<autil::LockFreeThreadPool>(1, 2000, nullptr, "MockWaitMatchThreadPool");
    ASSERT_TRUE(mock_proxy_dist_kv_cache_ptr->wait_match_thread_pool_->start());

    cache_manager.dist_kvcache_ = mock_proxy_dist_kv_cache_ptr;

    std::vector<int>                 token_ids  = {10, 20, 30, 40, 50};
    auto                             cache_keys = constructCacheKey(cache_manager, token_ids);
    CacheManager::AdvancedMallocInfo malloc_info(1, token_ids, cache_keys, {});
    BlockCache::MatchResult          match_result;

    // Allocate the two locally matched blocks for this same request to remove them from the free list
    KVCacheAllocator::SimpleMallocInfo block_malloc_info(/*request_id=*/6, 2);
    auto [success, resource] = cache_manager.malloc(block_malloc_info);
    ASSERT_TRUE(success);
    ASSERT_EQ(resource.block_id.size(), 2u);
    match_result.block_indices = resource.block_id;

    using MatchLocationReturnType = std::pair<ClientErrorCode, LocationsMap>;
    using VariantType             = std::variant<std::vector<bool>, size_t>;
    EXPECT_CALL(*mock_meta_client_ptr, MatchLocation(_, _, _, _, Eq(VariantType(static_cast<size_t>(2))), _, _))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_OK, {{"tp0", {"l3", "l4"}}}})));

    EXPECT_CALL(*mock_proxy_dist_kv_cache_ptr,
                getForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillOnce(Invoke([=](const std::vector<int64_t>&         actual_cache_keys,
                             const std::vector<int32_t>&         block_indices,
                             const DistKvCache::LocationsMapPtr& locations_map_ptr,
                             size_t                              ignore_block_num,
                             int64_t                             request_id,
                             std::map<std::string, std::string>  extra_metas) -> bool {
            EXPECT_EQ(actual_cache_keys.size(), 4);
            EXPECT_EQ(actual_cache_keys, (std::vector<int64_t>(cache_keys.begin(), cache_keys.begin() + 4)));
            // remote allocation should fill the next free blocks {3,4}
            EXPECT_EQ(block_indices.size(), 2);
            EXPECT_EQ(block_indices, (std::vector<int32_t>{3, 4}));
            EXPECT_EQ(ignore_block_num, match_result.block_indices.size());
            EXPECT_EQ(request_id, malloc_info.request_id);
            EXPECT_EQ(1, locations_map_ptr->size());
            EXPECT_EQ(2, locations_map_ptr->begin()->second.size());
            return true;
        }));

    cache_manager.matchInDistKvCache(malloc_info, match_result);
    ASSERT_EQ(match_result.block_indices.size(), 4u);
    // first two indices remain the original local ones
    ASSERT_EQ(match_result.block_indices, std::vector<int>({1, 2, 3, 4}));
}

}  // namespace rtp_llm
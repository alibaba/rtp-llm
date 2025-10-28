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

    std::vector<int64_t> constructCacheKey(const std::vector<int>& token_ids) {
        auto                 seq_size_per_block = cache_manager_->config_.seq_size_per_block;
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

    void init(uint32_t cache_config_block_nums = 10, const std::string& adapter = "") {
        auto cache_config               = initConfig();
        cache_config.block_nums         = cache_config_block_nums;
        cache_config.seq_size_per_block = 1;
        GptInitParameter param;
        param.kv_cache_config.enable_dist_kvcache = true;
        cache_manager_ = std::make_unique<ProxyCacheManager>(cache_config, device_, false, nullptr, param);

        mock_proxy_dist_kv_cache_ptr_ = std::make_shared<MockProxyDistKVCache>(cache_manager_.get(), param, nullptr);
        auto kvcm_client_wrapper_ptr  = std::make_shared<KVCMClientWrapper>();
        mock_meta_client_ptr_         = std::make_shared<MockMetaClient>();
        auto mock_transfer_client_ptr = std::make_unique<MockTransferClient>();
        mock_transfer_client_raw_ptr_ = mock_transfer_client_ptr.get();

        kvcm_client_wrapper_ptr->meta_client_map_[adapter]  = mock_meta_client_ptr_;
        kvcm_client_wrapper_ptr->config_map_[adapter]       = std::make_shared<KVCMClientWrapperConfig>();
        kvcm_client_wrapper_ptr->transfer_client_           = std::move(mock_transfer_client_ptr);
        mock_proxy_dist_kv_cache_ptr_->kvcm_client_wrapper_ = kvcm_client_wrapper_ptr;
        mock_proxy_dist_kv_cache_ptr_->wait_match_thread_pool_ =
            std::make_unique<autil::LockFreeThreadPool>(1, 2000, nullptr, "MockWaitMatchThreadPool");
        ASSERT_TRUE(mock_proxy_dist_kv_cache_ptr_->wait_match_thread_pool_->start());

        cache_manager_->dist_kvcache_ = mock_proxy_dist_kv_cache_ptr_;
    }

    std::unique_ptr<ProxyCacheManager>    cache_manager_;
    std::shared_ptr<MockMetaClient>       mock_meta_client_ptr_;
    MockTransferClient*                   mock_transfer_client_raw_ptr_ = nullptr;
    std::shared_ptr<MockProxyDistKVCache> mock_proxy_dist_kv_cache_ptr_;
};

using MatchLocationReturnType = std::pair<ClientErrorCode, LocationsMap>;
using VariantType             = std::variant<std::vector<bool>, size_t>;
using ExtraMeta               = std::map<std::string, std::string>;

// loss present -> do NOT put to dist
TEST_F(ProxyCacheManagerTest, testInsertIntoCache_LossNotEmpty_PutToBlockCache_NotPutToDistKvCache) {
    init();
    auto [ok, idx] = cache_manager_->mallocIndex({0, 2});
    ASSERT_TRUE(ok);
    std::vector<int>   token_ids  = {400, 401, 402};  // block_len = 2
    auto               cache_keys = constructCacheKey(token_ids);
    std::vector<float> loss       = {0.1f, 0.2f};

    EXPECT_CALL(*mock_proxy_dist_kv_cache_ptr_, putForAllRank(_, _, _, _, _)).Times(0);

    CacheManager::FreeInfo free_info(
        0, token_ids, cache_keys, idx, loss, /*adapter_name=*/"test_adapter_name", /*enable_3fs=*/true);
    cache_manager_->insertResidentCache(free_info);
}

// loss empty and dist enabled -> put to dist called with prefix block_len
TEST_F(ProxyCacheManagerTest, testInsertIntoCache_PutToDistKvCache) {
    std::string adapter_name = "test_adapter_name";
    init(10, adapter_name);

    int64_t request_id = 1;
    auto [ok, idx]     = cache_manager_->mallocIndex({request_id, 2});
    ASSERT_TRUE(ok);

    std::vector<int> token_ids  = {300, 301, 302};  // block_len = 2
    auto             cache_keys = constructCacheKey(token_ids);
    ASSERT_EQ(3, cache_keys.size());
    std::vector<int64_t> actual_keys(cache_keys.begin(), cache_keys.begin() + 2);
    ExtraMeta            expected_extra_meta{{"LORA_ADAPTER_NAME", adapter_name}};
    EXPECT_CALL(*mock_proxy_dist_kv_cache_ptr_,
                putForAllRank(                        //
                    ContainerEq(actual_keys),         // cache_keys
                    ContainerEq(idx),                 // block_indices
                    Eq(0),                            // ignore_block_num
                    Eq(request_id),                   // request_id
                    ContainerEq(expected_extra_meta)  // extra_metass
                    ))
        .WillOnce(Return(true));

    CacheManager::FreeInfo free_info(
        request_id, token_ids, cache_keys, idx, /*loss=*/{}, adapter_name, /*enable_3fs=*/true);
    cache_manager_->insertResidentCache(free_info);
}

// remote not match => early return
TEST_F(ProxyCacheManagerTest, testMatchInDistKvCache_remoteNotMatch) {
    init();

    std::vector<int>                 token_ids  = {10, 20, 30};
    auto                             cache_keys = constructCacheKey(token_ids);
    CacheManager::AdvancedMallocInfo malloc_info(/*request_id=*/2, token_ids, cache_keys, /*mm_bounds=*/{});
    BlockCache::MatchResult          match_result;
    match_result.block_indices = {5};
    cache_manager_->incrRefCounter(match_result.block_indices);

    LocationsMap expected_locations_map({{"tp0", {}}});
    EXPECT_CALL(*mock_meta_client_ptr_,
                MatchLocation(                                //
                    _,                                        // trace_id
                    _,                                        // query_type
                    ContainerEq(cache_keys),                  // keys
                    _,                                        // tokens
                    Eq(VariantType(static_cast<size_t>(1))),  // block_mask
                    _,                                        // sw_size
                    _                                         // location_spec_names
                    ))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_OK, expected_locations_map})));

    cache_manager_->matchInDistKvCache(malloc_info, match_result);
    ASSERT_EQ(match_result.block_indices.size(), 1u);
    ASSERT_EQ(match_result.block_indices, std::vector<int>({5}));
}

// getForAllRank failed => free allocated blocks and return
TEST_F(ProxyCacheManagerTest, testMatchInDistKvCache_GetFailed) {
    init();

    std::vector<int>                 token_ids  = {10, 20, 30, 40};
    auto                             cache_keys = constructCacheKey(token_ids);
    CacheManager::AdvancedMallocInfo malloc_info(/*request_id=*/5, token_ids, cache_keys, /*mm_bounds=*/{});
    BlockCache::MatchResult          match_result;
    match_result.block_indices = {7};
    cache_manager_->incrRefCounter(match_result.block_indices);

    LocationsMap expected_locations_map({{"tp0", {"l2", "l3"}}});
    EXPECT_CALL(*mock_meta_client_ptr_,
                MatchLocation(                                //
                    _,                                        // trace_id
                    _,                                        // query_type
                    ContainerEq(cache_keys),                  // keys
                    _,                                        // tokens
                    Eq(VariantType(static_cast<size_t>(1))),  // block_mask
                    _,                                        // sw_size
                    _                                         // location_spec_names
                    ))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_OK, expected_locations_map})));

    EXPECT_CALL(*mock_proxy_dist_kv_cache_ptr_, getForAllRank(_, _, _, _, _, _)).WillOnce(::testing::Return(false));

    auto free_before = cache_manager_->freeBlockNums();
    cache_manager_->matchInDistKvCache(malloc_info, match_result);
    auto free_after = cache_manager_->freeBlockNums();

    ASSERT_EQ(match_result.block_indices.size(), 1u);
    ASSERT_EQ(free_before, free_after);  // allocated blocks were freed
    ASSERT_EQ(match_result.block_indices, std::vector<int>({7}));
}

// success path => update matched length and append blocks
TEST_F(ProxyCacheManagerTest, testMatchInDistKvCache_Success) {
    init();
    std::vector<int>                 token_ids  = {10, 20, 30, 40, 50};
    auto                             cache_keys = constructCacheKey(token_ids);
    CacheManager::AdvancedMallocInfo malloc_info(1, token_ids, cache_keys, {});
    BlockCache::MatchResult          match_result;

    // Allocate the two locally matched blocks for this same request to remove them from the free list
    KVCacheAllocator::SimpleMallocInfo block_malloc_info(6, 2);
    auto [success, resource] = cache_manager_->malloc(block_malloc_info);
    ASSERT_TRUE(success);
    ASSERT_EQ(resource.block_id.size(), 2u);
    match_result.block_indices = resource.block_id;

    LocationsMap expected_locations_map({{"tp0", {"l3", "l4"}}});
    EXPECT_CALL(*mock_meta_client_ptr_,
                MatchLocation(                                //
                    _,                                        // trace_id
                    _,                                        // query_type
                    ContainerEq(cache_keys),                  // keys
                    _,                                        // tokens
                    Eq(VariantType(static_cast<size_t>(2))),  // block_mask
                    _,                                        // sw_size
                    _                                         // location_spec_names
                    ))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_OK, expected_locations_map})));

    std::vector<int64_t> expected_keys(cache_keys.begin(), cache_keys.begin() + 4);
    // remote allocation should fill the next free blocks {3,4}
    EXPECT_CALL(*mock_proxy_dist_kv_cache_ptr_,
                getForAllRank(                                                       //
                    ContainerEq(expected_keys),                                      // cache_keys
                    ContainerEq((std::vector<int32_t>{3, 4})),                       // block_indices
                    AllOf(NotNull(), Pointee(ContainerEq(expected_locations_map))),  // locations_map_ptr
                    Eq(2),                                                           // ignore_block_num
                    Eq(malloc_info.request_id),                                      // request_id
                    _                                                                // extra_metas
                    ))
        .WillOnce(Return(true));

    cache_manager_->matchInDistKvCache(malloc_info, match_result);
    ASSERT_EQ(match_result.block_indices.size(), 4u);
    // first two indices remain the original local ones
    ASSERT_EQ(match_result.block_indices, std::vector<int>({1, 2, 3, 4}));
}

// empty keys => return true, do not call dist_kvcache_
TEST_F(ProxyCacheManagerTest, testPutToDistKvCache_EmptyKeys_ReturnsTrue_NoCall) {
    init();

    std::vector<int64_t> cache_keys{};  // empty
    std::vector<int32_t> block_indices{1, 2, 3};
    size_t               ignore_block_num = 0;
    int64_t              request_id       = 100;
    std::string          adapter_name     = "test_adapter_name";

    EXPECT_CALL(*mock_proxy_dist_kv_cache_ptr_, putForAllRank(_, _, _, _, _)).Times(0);

    bool ok = cache_manager_->putToDistKvCache(cache_keys, block_indices, ignore_block_num, request_id, adapter_name);
    ASSERT_TRUE(ok);
}

// size mismatch => return false, do not call dist_kvcache_
TEST_F(ProxyCacheManagerTest, testPutToDistKvCache_SizeMismatch_ReturnsFalse_NoCall) {
    init();

    std::vector<int64_t> cache_keys{1, 2};
    std::vector<int32_t> block_indices{1};  // mismatch
    size_t               ignore_block_num = 0;
    int64_t              request_id       = 101;
    std::string          adapter_name     = "test_adapter_name";

    EXPECT_CALL(*mock_proxy_dist_kv_cache_ptr_, putForAllRank(_, _, _, _, _)).Times(0);

    bool ok = cache_manager_->putToDistKvCache(cache_keys, block_indices, ignore_block_num, request_id, adapter_name);
    ASSERT_FALSE(ok);
}

// failure path => returns false when dist returns false
TEST_F(ProxyCacheManagerTest, testPutToDistKvCache_CallsAndFails) {
    init();

    std::vector<int64_t> cache_keys{7, 8};
    std::vector<int32_t> block_indices{9, 10};
    size_t               ignore_block_num = 0;
    int64_t              request_id       = 104;
    std::string          adapter_name     = "test_adapter_name";

    EXPECT_CALL(*mock_proxy_dist_kv_cache_ptr_, putForAllRank(_, _, _, _, _)).WillOnce(Return(false));

    bool ok = cache_manager_->putToDistKvCache(cache_keys, block_indices, ignore_block_num, request_id, adapter_name);
    ASSERT_FALSE(ok);
}

// success path => calls dist_kvcache_->putForAllRank and returns true
TEST_F(ProxyCacheManagerTest, testPutToDistKvCache_CallsAndSucceeds) {
    init();

    std::vector<int64_t> cache_keys{100, 300, 600};
    std::vector<int32_t> block_indices{3, 4, 5};
    size_t               ignore_block_num = 1;
    int64_t              request_id       = 103;
    std::string          adapter_name     = "test_adapter_name";

    EXPECT_CALL(*mock_proxy_dist_kv_cache_ptr_,
                putForAllRank(                                            //
                    ContainerEq(cache_keys),                              // cache_keys
                    ContainerEq(block_indices),                           // block_indices
                    Eq(ignore_block_num),                                 // ignore_block_num
                    Eq(request_id),                                       // request_is
                    Eq(ExtraMeta({{"LORA_ADAPTER_NAME", adapter_name}}))  // extra_metas
                    ))
        .WillOnce(Return(true));

    bool ok = cache_manager_->putToDistKvCache(cache_keys, block_indices, ignore_block_num, request_id, adapter_name);
    ASSERT_TRUE(ok);
}

}  // namespace rtp_llm
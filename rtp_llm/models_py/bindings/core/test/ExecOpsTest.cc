#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include <gtest/gtest.h>

using namespace rtp_llm;

class MockCacheStore: public rtp_llm::CacheStore {
public:
    struct StoreRecord {
        std::string request_id;
        size_t      block_count{0};
    };
    std::vector<StoreRecord>                                  records;
    std::vector<std::shared_ptr<rtp_llm::RequestBlockBuffer>> stored_requests;

    void store(const std::shared_ptr<rtp_llm::RequestBlockBuffer>& buf,
               rtp_llm::CacheStoreStoreDoneCallback                cb) override {
        records.push_back({buf->getRequestId(), buf->getBlocksCount()});
        stored_requests.push_back(buf);
        if (cb) {
            cb(true, rtp_llm::CacheStoreErrorCode::None);
        }
    }

    void load(const std::shared_ptr<rtp_llm::RequestBlockBuffer>&,
              rtp_llm::CacheStoreLoadDoneCallback,
              const std::string&,
              uint32_t,
              uint32_t,
              uint32_t,
              int,
              int) override {}

    std::shared_ptr<rtp_llm::LoadContext> loadBuffers(const std::vector<std::shared_ptr<rtp_llm::RequestBlockBuffer>>&,
                                                      const std::string&,
                                                      uint32_t,
                                                      uint32_t,
                                                      int64_t,
                                                      rtp_llm::LoadContext::CheckCancelFunc,
                                                      int,
                                                      int) override {
        return nullptr;
    }

    std::shared_ptr<rtp_llm::StoreContext>
    storeBuffers(const std::vector<std::shared_ptr<rtp_llm::RequestBlockBuffer>>&, int64_t) override {
        return nullptr;
    }

    std::shared_ptr<rtp_llm::RemoteStoreTask>
    submitRemoteStoreTask(const std::shared_ptr<rtp_llm::RemoteStoreRequest>&,
                          const std::shared_ptr<rtp_llm::CacheStoreRemoteStoreMetricsCollector>&,
                          rtp_llm::RemoteStoreTask::CheckCancelFunc) override {
        return nullptr;
    }

    void releaseRemoteStoreTask(const std::shared_ptr<rtp_llm::RemoteStoreTask>&) override {}

    bool regUserBuffers(const std::vector<std::shared_ptr<rtp_llm::BlockBuffer>>&) override {
        return true;
    }

    std::shared_ptr<rtp_llm::BlockBuffer> findUserBuffer(const std::string&) override {
        return nullptr;
    }

    const std::shared_ptr<rtp_llm::MemoryUtil>& getMemoryUtil() const override {
        return null_util_;
    }

    void debugInfo() override {}

private:
    std::shared_ptr<rtp_llm::MemoryUtil> null_util_;
};

static rtp_llm::CacheStoreInputs makeHybridInputs(int layer_id) {
    rtp_llm::CacheStoreInputs p;
    p.pd_separation         = true;
    p.warmup                = false;
    p.context_batch_size    = 1;
    p.decoder_batch_size    = 0;
    p.tokens_per_block      = 2;
    p.kv_block_stride_bytes = 64;
    p.layer_id              = layer_id;
    p.model_id              = 0;

    p.kv_cache_group_types_host    = torch::tensor({0, 1}, torch::kInt32);
    p.kv_cache_layer_to_group_host = torch::tensor({0, 1}, torch::kInt32);

    p.input_lengths_host  = torch::tensor({6}, torch::kInt32);
    p.prefix_lengths_host = torch::tensor({0}, torch::kInt32);

    p.host_kv_cache_offset    = torch::zeros({2, 1, 3}, torch::kInt32);
    p.host_kv_cache_offset[1] = torch::ones({1, 3}, torch::kInt32);

    p.request_id            = torch::tensor({int64_t(42)}, torch::kInt64);
    p.request_pd_separation = torch::tensor({true}, torch::kBool);

    p.cache_keys        = {"blk0", "blk1", "blk2"};
    p.pre_created_event = runtimeCreateEvent();
    return p;
}

static rtp_llm::CacheStoreInputs makeHybridInputs2D(int layer_id) {
    rtp_llm::CacheStoreInputs p = makeHybridInputs(layer_id);
    p.host_kv_cache_offset      = torch::tensor({{0, 1, 2}}, torch::kInt32);
    return p;
}

class ExecOpsTest: public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        initRuntime(/*device_id=*/0,
                    /*trace_memory=*/false,
                    /*enable_comm_overlap=*/false,
                    MlaOpsType::AUTO);
    }
};

TEST_F(ExecOpsTest, testInitRuntimeIdempotent) {
    // Second call should be a no-op (already initialized).
    auto mla = initRuntime(0, false, false, MlaOpsType::AUTO);
    (void)mla;
    ASSERT_TRUE(isRuntimeInitialized());
}

TEST_F(ExecOpsTest, testGetEnableCommOverlap) {
    // Default DeviceResourceConfig has enable_comm_overlap = some value;
    // just verify the accessor works.
    (void)getEnableCommOverlap();
}

TEST_F(ExecOpsTest, testRuntimeSyncAndCheck) {
    ASSERT_NO_THROW(runtimeSyncAndCheck());
}

TEST_F(ExecOpsTest, testRuntimeCreateEvent) {
    auto event = runtimeCreateEvent();
    ASSERT_NE(event, nullptr);
    ASSERT_NO_THROW(event->synchronize());
}

TEST_F(ExecOpsTest, testCopyD2D) {
    auto       src = torch::randn({16}, torch::kCUDA);
    auto       dst = torch::empty({16}, torch::kCUDA);
    CopyParams params{dst, src};
    ASSERT_NO_THROW(runtimeCopy(params));
    runtimeSyncAndCheck();
    ASSERT_TRUE(torch::equal(src, dst));
}

TEST_F(ExecOpsTest, testCopyH2D) {
    auto       src = torch::randn({16}, torch::kCPU);
    auto       dst = torch::empty({16}, torch::kCUDA);
    CopyParams params{dst, src};
    ASSERT_NO_THROW(runtimeCopy(params));
    runtimeSyncAndCheck();
    ASSERT_TRUE(torch::equal(src, dst.cpu()));
}

TEST_F(ExecOpsTest, testCopyD2H) {
    auto       src = torch::randn({16}, torch::kCUDA);
    auto       dst = torch::empty({16}, torch::kCPU);
    CopyParams params{dst, src};
    ASSERT_NO_THROW(runtimeCopy(params));
    ASSERT_TRUE(torch::equal(src.cpu(), dst));
}

TEST_F(ExecOpsTest, testNoBlockCopy) {
    auto       src = torch::randn({32}, torch::kCUDA);
    auto       dst = torch::empty({32}, torch::kCUDA);
    CopyParams params{dst, src};
    ASSERT_NO_THROW(execNoBlockCopy(params));
    runtimeSyncAndCheck();
    ASSERT_TRUE(torch::equal(src, dst));
}

TEST_F(ExecOpsTest, testBatchCopyD2D) {
    auto src1 = torch::randn({8}, torch::kCUDA);
    auto src2 = torch::randn({16}, torch::kCUDA);
    auto dst1 = torch::empty({8}, torch::kCUDA);
    auto dst2 = torch::empty({16}, torch::kCUDA);

    BatchCopyParams params;
    auto&           d2d = params.copy_buffers[BatchCopyParams::D2D];
    d2d.src_ptr.push_back(src1.data_ptr());
    d2d.dst_ptr.push_back(dst1.data_ptr());
    d2d.sizes.push_back(src1.nbytes());
    d2d.src_ptr.push_back(src2.data_ptr());
    d2d.dst_ptr.push_back(dst2.data_ptr());
    d2d.sizes.push_back(src2.nbytes());

    ASSERT_NO_THROW(execBatchCopy(params));
    runtimeSyncAndCheck();
    ASSERT_TRUE(torch::equal(src1, dst1));
    ASSERT_TRUE(torch::equal(src2, dst2));
}

TEST_F(ExecOpsTest, testBatchCopyH2D) {
    auto src = torch::randn({8}, torch::kCPU);
    auto dst = torch::empty({8}, torch::kCUDA);

    BatchCopyParams params;
    auto&           h2d = params.copy_buffers[BatchCopyParams::H2D];
    h2d.src_ptr.push_back(src.data_ptr());
    h2d.dst_ptr.push_back(dst.data_ptr());
    h2d.sizes.push_back(src.nbytes());

    ASSERT_NO_THROW(execBatchCopy(params));
    runtimeSyncAndCheck();
    ASSERT_TRUE(torch::equal(src, dst.cpu()));
}

TEST_F(ExecOpsTest, testBatchCopyD2H) {
    auto src = torch::randn({8}, torch::kCUDA);
    auto dst = torch::empty({8}, torch::kCPU);

    BatchCopyParams params;
    auto&           d2h = params.copy_buffers[BatchCopyParams::D2H];
    d2h.src_ptr.push_back(src.data_ptr());
    d2h.dst_ptr.push_back(dst.data_ptr());
    d2h.sizes.push_back(src.nbytes());

    ASSERT_NO_THROW(execBatchCopy(params));
    ASSERT_TRUE(torch::equal(src.cpu(), dst));
}

TEST_F(ExecOpsTest, testGetGpuExecStatus) {
    auto status = getGpuExecStatus();
    ASSERT_GT(status.device_memory_status.free_bytes, 0u);
    ASSERT_GT(status.device_memory_status.available_bytes, 0u);
}

TEST_F(ExecOpsTest, testRuntimeMaskLogits) {
    auto logits = torch::randn({2, 8}, torch::kCUDA);
    auto mask   = torch::zeros({2, 8}, torch::TensorOptions(torch::kBool).device(torch::kCUDA));
    mask[0][0]  = true;
    mask[1][3]  = true;

    ASSERT_NO_THROW(runtimeMaskLogits(logits, mask));
    runtimeSyncAndCheck();
}

TEST_F(ExecOpsTest, testWriteCacheStoreGid_LinearGroup) {
    auto cache_store = std::make_shared<MockCacheStore>();
    auto param       = makeHybridInputs(/*layer_id=*/0);

    rtp_llm::KvCacheInfo kv;
    kv.layer_num       = 2;
    kv.kv_cache_buffer = torch::zeros({192}, torch::kByte);

    ASSERT_NO_THROW(rtp_llm::runtimeWriteCacheStore(param, kv, /*mla_kvcache=*/false, cache_store));

    ASSERT_EQ(cache_store->records.size(), 1u);
    EXPECT_EQ(cache_store->records[0].block_count, 2u);
}

TEST_F(ExecOpsTest, testWriteCacheStoreGid_FullGroup) {
    auto cache_store = std::make_shared<MockCacheStore>();
    auto param       = makeHybridInputs(/*layer_id=*/1);

    rtp_llm::KvCacheInfo kv;
    kv.layer_num       = 2;
    kv.kv_cache_buffer = torch::zeros({192}, torch::kByte);

    ASSERT_NO_THROW(rtp_llm::runtimeWriteCacheStore(param, kv, /*mla_kvcache=*/false, cache_store));

    ASSERT_EQ(cache_store->records.size(), 1u);
    EXPECT_EQ(cache_store->records[0].block_count, 3u);
}

TEST_F(ExecOpsTest, testWriteCacheStoreGid_3DOffset_NonZeroGroup) {
    auto cache_store = std::make_shared<MockCacheStore>();
    auto param       = makeHybridInputs(/*layer_id=*/1);

    rtp_llm::KvCacheInfo kv;
    kv.layer_num       = 2;
    kv.kv_cache_buffer = torch::zeros({128}, torch::kByte);

    ASSERT_NO_THROW(rtp_llm::runtimeWriteCacheStore(param, kv, /*mla_kvcache=*/false, cache_store));

    ASSERT_EQ(cache_store->records.size(), 1u);
    EXPECT_EQ(cache_store->records[0].block_count, 3u);
}

TEST_F(ExecOpsTest, testWriteCacheStoreGid_2DOffset_NonZeroGroup) {
    auto cache_store = std::make_shared<MockCacheStore>();
    auto param       = makeHybridInputs2D(/*layer_id=*/1);

    rtp_llm::KvCacheInfo kv;
    kv.layer_num       = 2;
    kv.kv_cache_buffer = torch::zeros({192}, torch::kByte);

    ASSERT_NO_THROW(rtp_llm::runtimeWriteCacheStore(param, kv, /*mla_kvcache=*/false, cache_store));

    ASSERT_EQ(cache_store->records.size(), 1u);
    EXPECT_EQ(cache_store->records[0].block_count, 3u);
}

TEST_F(ExecOpsTest, testRuntimeWriteCacheStoreUsesLayerAttnGroupAndLinearLastTwoBlocks) {
    constexpr int64_t layer_id          = 5;
    constexpr size_t  max_blocks        = 4;
    constexpr size_t  kv_stride_bytes   = 8;
    const auto        region_name       = KVCacheRegionName::SWA_KV;
    const int64_t     region_name_index = static_cast<int64_t>(region_name);

    CacheStoreInputs inputs;
    inputs.input_lengths_host           = torch::tensor({static_cast<int>(max_blocks)}, torch::kInt32);
    inputs.prefix_lengths_host          = torch::tensor({0}, torch::kInt32);
    inputs.host_kv_cache_offset         = torch::empty({2, 1, static_cast<int64_t>(max_blocks)}, torch::kInt32);
    auto* offset_data                   = inputs.host_kv_cache_offset.data_ptr<int32_t>();
    offset_data[0]                      = 100;
    offset_data[1]                      = 101;
    offset_data[2]                      = 102;
    offset_data[3]                      = 103;
    offset_data[4]                      = 200;
    offset_data[5]                      = 201;
    offset_data[6]                      = 202;
    offset_data[7]                      = 203;
    inputs.kv_cache_layer_to_group_host = torch::zeros({layer_id + 1}, torch::kInt32);
    inputs.kv_cache_layer_region_to_group_host =
        torch::full({layer_id + 1, static_cast<int64_t>(KVCacheRegionName::REGION_COUNT)}, -1, torch::kInt32);
    inputs.kv_cache_layer_region_to_group_host.index_put_({layer_id, region_name_index}, 1);
    inputs.kv_cache_group_types_host = torch::tensor(
        {static_cast<int32_t>(CacheGroupType::FULL), static_cast<int32_t>(CacheGroupType::LINEAR)}, torch::kInt32);
    inputs.context_batch_size    = 1;
    inputs.decoder_batch_size    = 0;
    inputs.request_id            = torch::tensor({1234}, torch::kInt64);
    inputs.request_pd_separation = torch::tensor({true}, torch::kBool);
    inputs.cache_keys            = {"10", "11", "12", "13"};
    inputs.tokens_per_block      = 1;
    inputs.kv_block_stride_bytes = kv_stride_bytes;
    inputs.kv_scale_stride_bytes = 0;
    inputs.pd_separation         = true;
    inputs.model_id              = 7;
    inputs.decode_entrance       = false;
    inputs.warmup                = false;
    inputs.layer_id              = static_cast<int>(layer_id);
    inputs.region_name           = region_name;
    inputs.pre_created_event     = runtimeCreateEvent();

    KvCacheInfo kv_cache;
    kv_cache.kv_cache_buffer = torch::empty({256, static_cast<int64_t>(kv_stride_bytes)}, torch::kUInt8);

    auto cache_store = std::make_shared<MockCacheStore>();
    runtimeWriteCacheStore(inputs, kv_cache, /*mla_kvcache=*/false, cache_store);

    ASSERT_EQ(cache_store->stored_requests.size(), 1u);
    const auto blocks = cache_store->stored_requests[0]->getBlocks();
    ASSERT_EQ(blocks.size(), 2u);

    const auto key2 = "kv_" + makeCacheKey(inputs.model_id, "12", layer_id, region_name);
    const auto key3 = "kv_" + makeCacheKey(inputs.model_id, "13", layer_id, region_name);
    ASSERT_TRUE(blocks.count(key2));
    ASSERT_TRUE(blocks.count(key3));
    EXPECT_EQ(blocks.at(key2)->len, kv_stride_bytes);
    EXPECT_EQ(blocks.at(key3)->len, kv_stride_bytes);

    const auto* base = static_cast<const int8_t*>(kv_cache.kv_cache_buffer.data_ptr());
    EXPECT_EQ(blocks.at(key2)->addr.get(), static_cast<const void*>(base + 202 * kv_stride_bytes));
    EXPECT_EQ(blocks.at(key3)->addr.get(), static_cast<const void*>(base + 203 * kv_stride_bytes));
}

TEST_F(ExecOpsTest, testRuntimeWriteCacheStoreSkipsNullSecondLastLinearBlock) {
    constexpr int64_t layer_id          = 5;
    constexpr size_t  max_blocks        = 4;
    constexpr size_t  kv_stride_bytes   = 8;
    const auto        region_name       = KVCacheRegionName::SWA_KV;
    const int64_t     region_name_index = static_cast<int64_t>(region_name);

    CacheStoreInputs inputs;
    inputs.input_lengths_host           = torch::tensor({static_cast<int>(max_blocks)}, torch::kInt32);
    inputs.prefix_lengths_host          = torch::tensor({0}, torch::kInt32);
    inputs.host_kv_cache_offset         = torch::empty({2, 1, static_cast<int64_t>(max_blocks)}, torch::kInt32);
    auto* offset_data                   = inputs.host_kv_cache_offset.data_ptr<int32_t>();
    offset_data[0]                      = 100;
    offset_data[1]                      = 101;
    offset_data[2]                      = 102;
    offset_data[3]                      = 103;
    offset_data[4]                      = 200;
    offset_data[5]                      = 201;
    offset_data[6]                      = NULL_BLOCK_IDX;
    offset_data[7]                      = 203;
    inputs.kv_cache_layer_to_group_host = torch::zeros({layer_id + 1}, torch::kInt32);
    inputs.kv_cache_layer_region_to_group_host =
        torch::full({layer_id + 1, static_cast<int64_t>(KVCacheRegionName::REGION_COUNT)}, -1, torch::kInt32);
    inputs.kv_cache_layer_region_to_group_host.index_put_({layer_id, region_name_index}, 1);
    inputs.kv_cache_group_types_host = torch::tensor(
        {static_cast<int32_t>(CacheGroupType::FULL), static_cast<int32_t>(CacheGroupType::LINEAR)}, torch::kInt32);
    inputs.context_batch_size    = 1;
    inputs.decoder_batch_size    = 0;
    inputs.request_id            = torch::tensor({1234}, torch::kInt64);
    inputs.request_pd_separation = torch::tensor({true}, torch::kBool);
    inputs.cache_keys            = {"10", "11", "12", "13"};
    inputs.tokens_per_block      = 1;
    inputs.kv_block_stride_bytes = kv_stride_bytes;
    inputs.kv_scale_stride_bytes = 0;
    inputs.pd_separation         = true;
    inputs.model_id              = 7;
    inputs.decode_entrance       = false;
    inputs.warmup                = false;
    inputs.layer_id              = static_cast<int>(layer_id);
    inputs.region_name           = region_name;
    inputs.pre_created_event     = runtimeCreateEvent();

    KvCacheInfo kv_cache;
    kv_cache.kv_cache_buffer = torch::empty({256, static_cast<int64_t>(kv_stride_bytes)}, torch::kUInt8);

    auto cache_store = std::make_shared<MockCacheStore>();
    runtimeWriteCacheStore(inputs, kv_cache, /*mla_kvcache=*/false, cache_store);

    ASSERT_EQ(cache_store->stored_requests.size(), 1u);
    const auto blocks = cache_store->stored_requests[0]->getBlocks();
    ASSERT_EQ(blocks.size(), 1u);

    const auto skipped_key = "kv_" + makeCacheKey(inputs.model_id, "12", layer_id, region_name);
    const auto stored_key  = "kv_" + makeCacheKey(inputs.model_id, "13", layer_id, region_name);
    EXPECT_FALSE(blocks.count(skipped_key));
    ASSERT_TRUE(blocks.count(stored_key));

    const auto* base = static_cast<const int8_t*>(kv_cache.kv_cache_buffer.data_ptr());
    EXPECT_EQ(blocks.at(stored_key)->addr.get(), static_cast<const void*>(base + 203 * kv_stride_bytes));
}

TEST_F(ExecOpsTest, testRuntimeWriteCacheStoreSupportsMultipleRegionsInOneLayer) {
    constexpr int64_t layer_id        = 3;
    constexpr size_t  max_blocks      = 4;
    constexpr size_t  kv_stride_bytes = 8;

    CacheStoreInputs inputs;
    inputs.input_lengths_host           = torch::tensor({static_cast<int>(max_blocks)}, torch::kInt32);
    inputs.prefix_lengths_host          = torch::tensor({0}, torch::kInt32);
    inputs.host_kv_cache_offset         = torch::empty({2, 1, static_cast<int64_t>(max_blocks)}, torch::kInt32);
    auto* offset_data                   = inputs.host_kv_cache_offset.data_ptr<int32_t>();
    offset_data[0]                      = 10;
    offset_data[1]                      = 11;
    offset_data[2]                      = 12;
    offset_data[3]                      = 13;
    offset_data[4]                      = 20;
    offset_data[5]                      = 21;
    offset_data[6]                      = 22;
    offset_data[7]                      = 23;
    inputs.kv_cache_layer_to_group_host = torch::zeros({layer_id + 1}, torch::kInt32);
    inputs.kv_cache_layer_region_to_group_host =
        torch::full({layer_id + 1, static_cast<int64_t>(KVCacheRegionName::REGION_COUNT)}, -1, torch::kInt32);
    inputs.kv_cache_layer_region_to_group_host.index_put_({layer_id, static_cast<int64_t>(KVCacheRegionName::CSA_KV)},
                                                          0);
    inputs.kv_cache_layer_region_to_group_host.index_put_({layer_id, static_cast<int64_t>(KVCacheRegionName::SWA_KV)},
                                                          1);
    inputs.kv_cache_group_types_host = torch::tensor(
        {static_cast<int32_t>(CacheGroupType::FULL), static_cast<int32_t>(CacheGroupType::LINEAR)}, torch::kInt32);
    inputs.context_batch_size    = 1;
    inputs.decoder_batch_size    = 0;
    inputs.request_id            = torch::tensor({4321}, torch::kInt64);
    inputs.request_pd_separation = torch::tensor({true}, torch::kBool);
    inputs.cache_keys            = {"0", "1", "2", "3"};
    inputs.tokens_per_block      = 1;
    inputs.kv_block_stride_bytes = kv_stride_bytes;
    inputs.kv_scale_stride_bytes = 0;
    inputs.pd_separation         = true;
    inputs.model_id              = 9;
    inputs.decode_entrance       = false;
    inputs.warmup                = false;
    inputs.layer_id              = static_cast<int>(layer_id);
    inputs.pre_created_event     = runtimeCreateEvent();

    KvCacheInfo kv_cache;
    kv_cache.kv_cache_buffer = torch::empty({32, static_cast<int64_t>(kv_stride_bytes)}, torch::kUInt8);

    auto cache_store   = std::make_shared<MockCacheStore>();
    inputs.region_name = KVCacheRegionName::CSA_KV;
    runtimeWriteCacheStore(inputs, kv_cache, /*mla_kvcache=*/false, cache_store);
    inputs.region_name = KVCacheRegionName::SWA_KV;
    runtimeWriteCacheStore(inputs, kv_cache, /*mla_kvcache=*/false, cache_store);

    ASSERT_EQ(cache_store->stored_requests.size(), 2u);
    const auto full_blocks   = cache_store->stored_requests[0]->getBlocks();
    const auto linear_blocks = cache_store->stored_requests[1]->getBlocks();
    ASSERT_EQ(full_blocks.size(), 4u);
    ASSERT_EQ(linear_blocks.size(), 2u);

    const auto full_key0 = "kv_" + makeCacheKey(inputs.model_id, "0", layer_id, KVCacheRegionName::CSA_KV);
    const auto full_key3 = "kv_" + makeCacheKey(inputs.model_id, "3", layer_id, KVCacheRegionName::CSA_KV);
    const auto lin_key2  = "kv_" + makeCacheKey(inputs.model_id, "2", layer_id, KVCacheRegionName::SWA_KV);
    const auto lin_key3  = "kv_" + makeCacheKey(inputs.model_id, "3", layer_id, KVCacheRegionName::SWA_KV);
    ASSERT_TRUE(full_blocks.count(full_key0));
    ASSERT_TRUE(full_blocks.count(full_key3));
    ASSERT_TRUE(linear_blocks.count(lin_key2));
    ASSERT_TRUE(linear_blocks.count(lin_key3));

    const auto* base = static_cast<const int8_t*>(kv_cache.kv_cache_buffer.data_ptr());
    EXPECT_EQ(full_blocks.at(full_key0)->addr.get(), static_cast<const void*>(base + 10 * kv_stride_bytes));
    EXPECT_EQ(full_blocks.at(full_key3)->addr.get(), static_cast<const void*>(base + 13 * kv_stride_bytes));
    EXPECT_EQ(linear_blocks.at(lin_key2)->addr.get(), static_cast<const void*>(base + 22 * kv_stride_bytes));
    EXPECT_EQ(linear_blocks.at(lin_key3)->addr.get(), static_cast<const void*>(base + 23 * kv_stride_bytes));
}

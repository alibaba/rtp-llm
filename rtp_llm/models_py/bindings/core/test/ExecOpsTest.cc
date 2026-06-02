#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include <gtest/gtest.h>

using namespace rtp_llm;

// MockCacheStore: captures every store() call (request-id + block count).
class MockCacheStore: public rtp_llm::CacheStore {
public:
    struct StoreRecord {
        std::string request_id;
        size_t      block_count{0};
    };
    std::vector<StoreRecord> records;

    void store(const std::shared_ptr<rtp_llm::RequestBlockBuffer>& buf,
               rtp_llm::CacheStoreStoreDoneCallback                cb) override {
        records.push_back({buf->getRequestId(), buf->getBlocksCount()});
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

// Build a CacheStoreInputs for a 2-group hybrid scenario:
//   group 0 = LINEAR (type 0),  group 1 = FULL (type 1)
//   layer 0 → group 0,          layer 1 → group 1
// batch_size = 1, tokens_per_block = 2, input_length = 6  → total_blocks = 3
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

    // group types: [LINEAR=0, FULL=1]
    p.kv_cache_group_types_host = torch::tensor({0, 1}, torch::kInt32);
    // layer-to-group mapping: layer 0 → group 0, layer 1 → group 1
    p.kv_cache_layer_to_group_host = torch::tensor({0, 1}, torch::kInt32);

    // input_lengths[decoder_batch_size + context_batch_size] = [6]
    p.input_lengths_host = torch::tensor({6}, torch::kInt32);
    // prefix_lengths[context_batch_size] = [0]  (no reuse blocks)
    p.prefix_lengths_host = torch::tensor({0}, torch::kInt32);

    // 3-D offset: [num_groups, batch_size, max_blocks_per_batch] = [2, 1, 3]
    // Group 0 offsets = 0, group 1 offsets = 1.
    p.host_kv_cache_offset    = torch::zeros({2, 1, 3}, torch::kInt32);
    p.host_kv_cache_offset[1] = torch::ones({1, 3}, torch::kInt32);

    p.request_id            = torch::tensor({int64_t(42)}, torch::kInt64);
    p.request_pd_separation = torch::tensor({true}, torch::kBool);

    // cache_keys: context_batch_size * max_blocks_per_batch = 3 strings
    p.cache_keys = {"blk0", "blk1", "blk2"};

    // Match current models_py API: pre_created_event is std::shared_ptr<c10::Event>.
    p.pre_created_event = runtimeCreateEvent();
    return p;
}

// 2-D compatibility path: host_kv_cache_offset = [batch, max_blocks].
// Block IDs 0, 1, 2 map to valid offsets inside a 192-byte kv_cache_buffer.
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

// Layer 0 maps to group 0 (LINEAR).
// CacheGroupType::LINEAR means only the last block is transferred;
// so exactly 1 block should be recorded in the mock store.
TEST_F(ExecOpsTest, testWriteCacheStoreGid_LinearGroup) {
    auto cache_store = std::make_shared<MockCacheStore>();
    auto param       = makeHybridInputs(/*layer_id=*/0);

    // 3 blocks × 64 bytes each; CPU tensor, pointer arithmetic only.
    rtp_llm::KvCacheInfo kv;
    kv.layer_num       = 2;
    kv.kv_cache_buffer = torch::zeros({192}, torch::kByte);

    ASSERT_NO_THROW(rtp_llm::runtimeWriteCacheStore(param, kv, /*mla_kvcache=*/false, cache_store));

    ASSERT_EQ(cache_store->records.size(), 1u) << "Expected exactly one store() call for the single request";
    EXPECT_EQ(cache_store->records[0].block_count, 1u)
        << "Layer 0 → group 0 (LINEAR): only the last block should be stored";
}

// Layer 1 maps to group 1 (FULL).
// CacheGroupType::FULL means all blocks are transferred;
// with total_blocks = 3, exactly 3 blocks should reach the mock store.
TEST_F(ExecOpsTest, testWriteCacheStoreGid_FullGroup) {
    auto cache_store = std::make_shared<MockCacheStore>();
    auto param       = makeHybridInputs(/*layer_id=*/1);

    rtp_llm::KvCacheInfo kv;
    kv.layer_num       = 2;
    kv.kv_cache_buffer = torch::zeros({192}, torch::kByte);

    ASSERT_NO_THROW(rtp_llm::runtimeWriteCacheStore(param, kv, /*mla_kvcache=*/false, cache_store));

    ASSERT_EQ(cache_store->records.size(), 1u);
    EXPECT_EQ(cache_store->records[0].block_count, 3u) << "Layer 1 → group 1 (FULL): all 3 blocks should be stored";
}

// 3-D block-table path: layer 1 → group 1 (FULL).
// Verifies that gid resolution is also correct when host_kv_cache_offset
// has a third group dimension (the case that originally triggered the fix).
TEST_F(ExecOpsTest, testWriteCacheStoreGid_3DOffset_NonZeroGroup) {
    auto cache_store = std::make_shared<MockCacheStore>();
    auto param       = makeHybridInputs(/*layer_id=*/1);

    // Group 1 uses block_id=1; max offset = 1 × 64 = 64 bytes.
    rtp_llm::KvCacheInfo kv;
    kv.layer_num       = 2;
    kv.kv_cache_buffer = torch::zeros({128}, torch::kByte);

    ASSERT_NO_THROW(rtp_llm::runtimeWriteCacheStore(param, kv, /*mla_kvcache=*/false, cache_store));

    ASSERT_EQ(cache_store->records.size(), 1u);
    EXPECT_EQ(cache_store->records[0].block_count, 3u)
        << "3-D path, layer 1 → group 1 (FULL): all 3 blocks should be stored";
}

// 2-D block-table compatibility path: layer 1 → group 1 (FULL).
// Verifies the legacy [batch, max_blocks] layout still resolves gid correctly.
TEST_F(ExecOpsTest, testWriteCacheStoreGid_2DOffset_NonZeroGroup) {
    auto cache_store = std::make_shared<MockCacheStore>();
    auto param       = makeHybridInputs2D(/*layer_id=*/1);

    rtp_llm::KvCacheInfo kv;
    kv.layer_num       = 2;
    kv.kv_cache_buffer = torch::zeros({192}, torch::kByte);

    ASSERT_NO_THROW(rtp_llm::runtimeWriteCacheStore(param, kv, /*mla_kvcache=*/false, cache_store));

    ASSERT_EQ(cache_store->records.size(), 1u);
    EXPECT_EQ(cache_store->records[0].block_count, 3u)
        << "2-D path, layer 1 → group 1 (FULL): all 3 blocks should be stored";
}

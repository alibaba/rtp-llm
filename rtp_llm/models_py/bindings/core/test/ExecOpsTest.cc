#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include <gtest/gtest.h>
#include <future>
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"

using namespace rtp_llm;

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

// Retain published blocks past the callback, as the real remote-read store does.
class RetainingCacheStore: public CacheStore {
public:
    std::vector<std::shared_ptr<RequestBlockBuffer>> published;
    std::shared_ptr<MemoryUtil> memory;
    bool callback_success = true;
    void load(const std::shared_ptr<RequestBlockBuffer>&, CacheStoreLoadDoneCallback,
              const std::string&, uint32_t, uint32_t, uint32_t, int, int) override { FAIL(); }
    std::shared_ptr<LoadContext> loadBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>&,
        const std::string&, uint32_t, uint32_t, int64_t, LoadContext::CheckCancelFunc, int, int) override {
        ADD_FAILURE(); return nullptr;
    }
    std::shared_ptr<StoreContext> storeBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>&,
                                              int64_t) override { ADD_FAILURE(); return nullptr; }
    std::shared_ptr<RemoteStoreTask> submitRemoteStoreTask(const std::shared_ptr<RemoteStoreRequest>&,
        const std::shared_ptr<CacheStoreRemoteStoreMetricsCollector>&,
        RemoteStoreTask::CheckCancelFunc) override { ADD_FAILURE(); return nullptr; }
    void releaseRemoteStoreTask(const std::shared_ptr<RemoteStoreTask>&) override { FAIL(); }
    bool regUserBuffers(const std::vector<std::shared_ptr<BlockBuffer>>&) override { ADD_FAILURE(); return false; }
    std::shared_ptr<BlockBuffer> findUserBuffer(const std::string&) override { ADD_FAILURE(); return nullptr; }
    const std::shared_ptr<MemoryUtil>& getMemoryUtil() const override { return memory; }
    void debugInfo() override {}

    void store(const std::shared_ptr<RequestBlockBuffer>& blocks,
               CacheStoreStoreDoneCallback callback) override {
        published.push_back(blocks);
        callback(callback_success, callback_success ? CacheStoreErrorCode::None : CacheStoreErrorCode::StoreFailed);
    }
};

TEST_F(ExecOpsTest, terminalSsmTransferOwnsWidenedStorage) {
    for (const bool bf16 : {false, true}) {
        for (const bool async : {false, true}) {
            const auto dtype = bf16 ? torch::kBFloat16 : torch::kFloat32;
            auto source = torch::randn({5, 24}, torch::TensorOptions().device(torch::kCUDA).dtype(dtype));
            auto expected = source[3].slice(0, 0, 16).to(torch::kFloat32).clone();
            auto conv = source[3].slice(0, 16, 24).clone();
            KvCacheInfo cache{};
            cache.kv_cache_buffer = source;
            cache.linear_cache_segment_sizes = {16 * source.element_size(), 8 * source.element_size()};
            cache.linear_ssm_bf16_to_fp32 = bf16;
            CacheStoreInputs inputs{};
            inputs.warmup = false;
            inputs.pd_separation = true;
            inputs.context_batch_size = 1;
            inputs.tokens_per_block = 64;
            inputs.kv_block_stride_bytes = 24 * source.element_size();
            inputs.input_lengths_host = torch::tensor({128}, torch::kInt32);
            inputs.prefix_lengths_host = torch::tensor({0}, torch::kInt32);
            inputs.host_kv_cache_offset = torch::tensor({1, 3}, torch::kInt32).reshape({1, 2});
            inputs.kv_cache_group_types_host = torch::tensor({static_cast<int>(CacheGroupType::LINEAR)}, torch::kInt32);
            inputs.request_id = torch::tensor({42}, torch::kInt64);
            inputs.request_pd_separation = torch::tensor({true}, torch::kBool);
            inputs.cache_keys = {"first", "terminal"};
            inputs.pre_created_event = runtimeCreateEvent();
            auto store = std::make_shared<RetainingCacheStore>();
            auto publish = [&] { runtimeWriteCacheStore(inputs, cache, true, store); };
            if (async) {
                std::async(std::launch::async, publish).get();
            } else {
                publish();
            }
            ASSERT_EQ(store->published.size(), 1);
            auto publication = store->published.front();
            if (bf16) {
                EXPECT_NE(publication->getEvent(), inputs.pre_created_event.get());
            }
            publication->getEvent()->synchronize();
            ASSERT_EQ(publication->getBlocksCount(), 2);
            const auto key = makeCacheKey(0, "terminal", 0);
            auto ssm = publication->getBlock(makeLinearCacheSegmentKey(0, key));
            auto history = publication->getBlock(makeLinearCacheSegmentKey(1, key));
            ASSERT_NE(ssm, nullptr);
            ASSERT_NE(history, nullptr);
            ASSERT_EQ(ssm->len, 16 * sizeof(float));
            ASSERT_EQ(history->len, conv.nbytes());
            EXPECT_EQ(history->addr.get(), source[3].slice(0, 16, 24).data_ptr());
            if (bf16) {
                EXPECT_NE(ssm->addr.get(), source[3].data_ptr());
                // The pool can be reused after conversion, without changing the
                // snapshot retained for a later remote read.
                source[3].slice(0, 0, 16).zero_();
            } else {
                EXPECT_EQ(ssm->addr.get(), source[3].data_ptr());
            }
            auto received = torch::from_blob(ssm->addr.get(), {16}, source.options().dtype(torch::kFloat32));
            EXPECT_TRUE(torch::equal(received, expected));
            auto received_conv = torch::from_blob(history->addr.get(), {8}, source.options());
            EXPECT_TRUE(torch::equal(received_conv, conv));
            std::weak_ptr<void> owner = ssm->addr;
            store->published.clear();
            publication.reset();
            EXPECT_FALSE(owner.expired());
            ssm.reset();
            history.reset();
            EXPECT_TRUE(owner.expired());
        }
    }
}

// A failed publication can still have an outstanding remote reader. A retry
// of the same request/key must not mutate or take ownership of its snapshot.
TEST_F(ExecOpsTest, failedTerminalSsmRetryKeepsIndependentSnapshots) {
    for (const bool async : {false, true}) {
        auto source = torch::full({2, 24}, 1.003,
                                  torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16));
        KvCacheInfo cache{};
        cache.kv_cache_buffer = source;
        cache.linear_cache_segment_sizes = {16 * sizeof(at::BFloat16), 8 * sizeof(at::BFloat16)};
        cache.linear_ssm_bf16_to_fp32 = true;
        CacheStoreInputs inputs{};
        inputs.warmup = false;
        inputs.pd_separation = true;
        inputs.context_batch_size = 1;
        inputs.tokens_per_block = 64;
        inputs.kv_block_stride_bytes = 24 * sizeof(at::BFloat16);
        inputs.input_lengths_host = torch::tensor({64}, torch::kInt32);
        inputs.prefix_lengths_host = torch::tensor({0}, torch::kInt32);
        inputs.host_kv_cache_offset = torch::tensor({1}, torch::kInt32).reshape({1, 1});
        inputs.kv_cache_group_types_host = torch::tensor({static_cast<int>(CacheGroupType::LINEAR)}, torch::kInt32);
        inputs.request_id = torch::tensor({43}, torch::kInt64);
        inputs.request_pd_separation = torch::tensor({true}, torch::kBool);
        inputs.cache_keys = {"terminal"};
        auto store = std::make_shared<RetainingCacheStore>();
        auto publish = [&] {
            inputs.pre_created_event = runtimeCreateEvent();
            auto write = [&] { runtimeWriteCacheStore(inputs, cache, true, store); };
            if (async) {
                std::async(std::launch::async, write).get();
            } else {
                write();
            }
            store->published.back()->getEvent()->synchronize();
        };
        const auto key = makeLinearCacheSegmentKey(0, makeCacheKey(0, "terminal", 0));
        auto first_expected = source[1].slice(0, 0, 16).to(torch::kFloat32).clone();
        store->callback_success = false;
        publish();
        ASSERT_EQ(store->published.size(), 1);
        auto late_reader = store->published.front()->getBlock(key);
        ASSERT_NE(late_reader, nullptr);
        std::weak_ptr<void> first_owner = late_reader->addr;

        source[1].slice(0, 0, 16).fill_(2.007);
        auto retry_expected = source[1].slice(0, 0, 16).to(torch::kFloat32).clone();
        store->callback_success = true;
        publish();
        ASSERT_EQ(store->published.size(), 2);
        auto retry_reader = store->published.back()->getBlock(key);
        ASSERT_NE(retry_reader, nullptr);
        EXPECT_NE(late_reader->addr.get(), retry_reader->addr.get());
        std::weak_ptr<void> retry_owner = retry_reader->addr;
        source[1].slice(0, 0, 16).zero_();

        // Model request cleanup while both readers still hold the blocks.
        store->published.clear();
        EXPECT_FALSE(first_owner.expired());
        EXPECT_FALSE(retry_owner.expired());
        auto options = source.options().dtype(torch::kFloat32);
        EXPECT_TRUE(torch::equal(torch::from_blob(late_reader->addr.get(), {16}, options), first_expected));
        EXPECT_TRUE(torch::equal(torch::from_blob(retry_reader->addr.get(), {16}, options), retry_expected));
        late_reader.reset();
        EXPECT_TRUE(first_owner.expired());
        EXPECT_FALSE(retry_owner.expired());
        retry_reader.reset();
        EXPECT_TRUE(retry_owner.expired());
    }
}

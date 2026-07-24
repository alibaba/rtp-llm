#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/common/WriteCacheStoreOp.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/testing/TestLogCapture.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <unordered_map>

using namespace rtp_llm;

// MockCacheStore: captures every store() call (request-id + block count).
class MockCacheStore: public rtp_llm::CacheStore {
public:
    struct BlockRecord {
        void*    addr{nullptr};
        uint32_t len{0};
    };

    struct StoreRecord {
        std::string                                  request_id;
        size_t                                       block_count{0};
        std::unordered_map<std::string, BlockRecord> blocks;
        std::vector<std::string>                     block_keys;
    };
    std::vector<StoreRecord> records;
    bool                     store_success = true;
    CacheStoreErrorCode      store_error   = CacheStoreErrorCode::None;
    bool                     load_success  = true;
    CacheStoreErrorCode      load_error    = CacheStoreErrorCode::None;

    void store(const std::shared_ptr<rtp_llm::RequestBlockBuffer>& buf,
               rtp_llm::CacheStoreStoreDoneCallback                cb) override {
        StoreRecord record;
        record.request_id  = buf->getRequestId();
        record.block_count = buf->getBlocksCount();
        for (const auto& [key, block] : buf->getBlocks()) {
            record.blocks.emplace(key, BlockRecord{block->addr.get(), block->len});
            record.block_keys.push_back(key);
        }
        records.push_back(std::move(record));
        if (cb) {
            cb(store_success, store_error);
        }
    }

    void load(const std::shared_ptr<rtp_llm::RequestBlockBuffer>&,
              rtp_llm::CacheStoreLoadDoneCallback callback,
              const std::string&,
              uint32_t,
              uint32_t,
              uint32_t,
              int,
              int) override {
        callback(load_success, load_error);
    }

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

static size_t countKeyPrefix(const std::vector<std::string>& keys, const std::string& prefix) {
    return static_cast<size_t>(
        std::count_if(keys.begin(), keys.end(), [&](const auto& key) { return key.rfind(prefix, 0) == 0; }));
}

// Build tag-local CacheStoreInputs for a 2-group hybrid scenario.
// batch_size = 1, tokens_per_block = 2, input_length = 6  → total_blocks = 3
static rtp_llm::CacheStoreInputs makeHybridInputs(int layer_id) {
    rtp_llm::CacheStoreInputs p;
    p.pd_separation                = true;
    p.warmup                       = false;
    p.context_batch_size           = 1;
    p.decoder_batch_size           = 0;
    p.tokens_per_block             = 2;
    p.kv_block_stride_bytes        = 64;
    p.layer_id                     = layer_id;
    p.tag                          = layer_id == 0 ? "linear" : "full";
    p.model_id                     = 0;
    p.use_opaque_kv_cache_store    = false;
    p.kv_cache_group_policies      = {{"linear", defaultCacheGroupPolicy(CacheGroupType::LINEAR)},
                                      {"full", defaultCacheGroupPolicy(CacheGroupType::FULL)}};
    p.tokens_per_block_by_tag      = {{"linear", 2}, {"full", 2}};
    p.kv_block_stride_bytes_by_tag = {{"linear", 64}, {"full", 64}};
    p.kv_scale_stride_bytes_by_tag = {{"linear", 0}, {"full", 0}};

    // input_lengths[decoder_batch_size + context_batch_size] = [6]
    p.input_lengths_host = torch::tensor({6}, torch::kInt32);
    // prefix_lengths[context_batch_size] = [0]  (no reuse blocks)
    p.prefix_lengths_host = torch::tensor({0}, torch::kInt32);

    // The caller has already selected the layer tag, so the table is group-local.
    p.host_kv_cache_offset = layer_id == 0 ? torch::zeros({1, 3}, torch::kInt32) : torch::ones({1, 3}, torch::kInt32);

    p.request_id            = torch::tensor({int64_t(42)}, torch::kInt64);
    p.request_pd_separation = torch::tensor({true}, torch::kBool);

    // cache_keys: context_batch_size * max_blocks_per_batch = 3 strings
    p.cache_keys = {"blk0", "blk1", "blk2"};

    // Match current models_py API: pre_created_event is std::shared_ptr<c10::Event>.
    p.pre_created_event = runtimeCreateEvent();
    return p;
}

static torch_ext::PyCacheStoreInputs makePyCacheStoreInputs(const std::shared_ptr<MockCacheStore>& cache_store,
                                                            size_t                                 tokens_per_block,
                                                            size_t                                 kv_stride,
                                                            size_t                                 scale_stride,
                                                            size_t                                 block_num,
                                                            bool                                   mla_kvcache) {
    torch_ext::PyCacheStoreInputs inputs;
    inputs.context_batch_size             = 1;
    inputs.decoder_batch_size             = 0;
    inputs.request_id                     = torch::tensor({int64_t(42)}, torch::kInt64);
    inputs.request_pd_separation          = torch::tensor({true}, torch::kBool);
    inputs.kv_cache_group_policies        = {{"default", defaultCacheGroupPolicy(CacheGroupType::FULL)}};
    inputs.tokens_per_block_by_tag        = {{"default", tokens_per_block}};
    inputs.kv_block_stride_bytes_by_tag   = {{"default", kv_stride}};
    inputs.kv_scale_stride_bytes_by_tag   = {{"default", scale_stride}};
    inputs.kv_block_transfer_bytes_by_tag = {{"default", kv_stride}};
    inputs.kv_scale_transfer_bytes_by_tag = {{"default", scale_stride}};
    for (size_t i = 0; i < block_num; ++i) {
        inputs.cache_keys.push_back("block_" + std::to_string(i));
    }
    inputs.tokens_per_block      = tokens_per_block;
    inputs.kv_block_stride_bytes = kv_stride;
    inputs.kv_scale_stride_bytes = scale_stride;
    inputs.pd_separation         = true;
    inputs.mla_kvcache           = mla_kvcache;
    inputs.cache_store           = cache_store;
    return inputs;
}

static void expectMlaPhysicalViewUsesExplicitStride(const torch::Tensor& kv_cache_base) {
    constexpr size_t physical_block_num        = 4;
    constexpr size_t physical_tokens_per_block = 8;
    constexpr int    kernel_tokens_per_block   = 2;

    ASSERT_EQ(static_cast<size_t>(kv_cache_base.size(0)), physical_block_num);
    const size_t explicit_stride = static_cast<size_t>(kv_cache_base.nbytes()) / physical_block_num;
    auto         cache_store     = std::make_shared<MockCacheStore>();
    auto         inputs          = makePyCacheStoreInputs(cache_store,
                                         physical_tokens_per_block,
                                         explicit_stride,
                                         /*scale_stride=*/0,
                                         physical_block_num,
                                         /*mla_kvcache=*/true);

    torch_ext::LayerKVCache layer_cache;
    layer_cache.kv_cache_base      = kv_cache_base;
    layer_cache.seq_size_per_block = kernel_tokens_per_block;
    layer_cache.layer_id           = 0;
    layer_cache.tag                = "default";

    auto input_lengths =
        torch::tensor({static_cast<int32_t>(physical_block_num * physical_tokens_per_block)}, torch::kInt32);
    auto prefix_lengths = torch::tensor({0}, torch::kInt32);
    auto block_ids      = torch::arange(static_cast<int64_t>(physical_block_num), torch::kInt32).reshape({1, -1});

    ASSERT_NO_THROW(WriteCacheStoreOp(
        input_lengths, prefix_lengths, block_ids, std::make_optional(inputs), std::make_optional(layer_cache)));

    ASSERT_EQ(cache_store->records.size(), 1u);
    const auto& record = cache_store->records.front();
    ASSERT_EQ(record.block_count, physical_block_num);
    ASSERT_EQ(record.blocks.size(), physical_block_num);

    const auto base_addr   = reinterpret_cast<uintptr_t>(kv_cache_base.data_ptr());
    const auto storage_end = base_addr + static_cast<size_t>(kv_cache_base.nbytes());
    for (size_t block_id = 0; block_id < physical_block_num; ++block_id) {
        const std::string key = "kv_" + makeCacheKey(0, inputs.cache_keys[block_id], 0);
        const auto        it  = record.blocks.find(key);
        ASSERT_NE(it, record.blocks.end()) << "missing block " << key;
        const auto addr = reinterpret_cast<uintptr_t>(it->second.addr);
        EXPECT_EQ(addr, base_addr + block_id * explicit_stride);
        EXPECT_EQ(it->second.len, explicit_stride);
        EXPECT_LE(addr + it->second.len, storage_end);
    }
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

TEST_F(ExecOpsTest, testRuntimeApplyPackedMaskLogitsUsesCompactRowMapping) {
    constexpr int64_t vocab_size        = 35;
    constexpr int64_t logits_columns    = 40;
    auto              packed_allow_mask = torch::tensor({1, 4, 2, 2}, torch::kInt32).reshape({2, 2});
    auto              row_indices       = torch::tensor({1, 3}, torch::kInt32);
#if USING_CUDA
    packed_allow_mask = packed_allow_mask.to(torch::kCUDA);
    row_indices       = row_indices.to(torch::kCUDA);
#endif

    for (const auto dtype : {torch::kFloat32, torch::kFloat16, torch::kBFloat16}) {
        auto logits = torch::ones({4, logits_columns}, torch::TensorOptions(dtype).device(torch::kCUDA));
        ASSERT_NO_THROW(runtimeApplyPackedMaskLogits(logits, packed_allow_mask, row_indices, vocab_size));
        runtimeSyncAndCheck();

        auto result = logits.to(torch::kFloat32).cpu().contiguous();
        for (int64_t row = 0; row < result.size(0); ++row) {
            for (int64_t token = 0; token < result.size(1); ++token) {
                const bool allowed = row == 0 || row == 2 || token >= vocab_size
                                     || (row == 1 && (token == 0 || token == 34))
                                     || (row == 3 && (token == 1 || token == 33));
                if (allowed) {
                    EXPECT_FLOAT_EQ(result[row][token].item<float>(), 1.0f);
                } else {
                    EXPECT_TRUE(std::isinf(result[row][token].item<float>()));
                    EXPECT_LT(result[row][token].item<float>(), 0.0f);
                }
            }
        }
    }
}

TEST_F(ExecOpsTest, testRuntimeApplyPackedMaskLogitsSupportsSingleRowIdentityMapping) {
    constexpr int64_t vocab_size = 35;
    auto              logits = torch::ones({vocab_size}, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
    auto              packed_allow_mask = torch::tensor({1, 4}, torch::kInt32).reshape({1, 2});
#if USING_CUDA
    packed_allow_mask = packed_allow_mask.to(torch::kCUDA);
#endif

    ASSERT_NO_THROW(runtimeApplyPackedMaskLogits(logits, packed_allow_mask, vocab_size));
    runtimeSyncAndCheck();

    auto result = logits.cpu().contiguous();
    for (int64_t token = 0; token < vocab_size; ++token) {
        if (token == 0 || token == 34) {
            EXPECT_FLOAT_EQ(result[token].item<float>(), 1.0f);
        } else {
            EXPECT_TRUE(std::isinf(result[token].item<float>()));
            EXPECT_LT(result[token].item<float>(), 0.0f);
        }
    }
}

TEST_F(ExecOpsTest, testRuntimeApplyPackedMaskLogitsSkipsOutOfRangeRows) {
    constexpr int64_t vocab_size = 4;
    auto              logits = torch::ones({3, vocab_size}, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
    auto              packed_allow_mask = torch::zeros({3, 1}, torch::kInt32);
    auto              row_indices       = torch::tensor({-1, 1, 3}, torch::kInt32);
#if USING_CUDA
    packed_allow_mask = packed_allow_mask.to(torch::kCUDA);
    row_indices       = row_indices.to(torch::kCUDA);
#endif

    ASSERT_NO_THROW(runtimeApplyPackedMaskLogits(logits, packed_allow_mask, row_indices, vocab_size));
    runtimeSyncAndCheck();

    auto result = logits.cpu().to(torch::kFloat32).contiguous();
    for (int64_t token = 0; token < vocab_size; ++token) {
        EXPECT_FLOAT_EQ(result[0][token].item<float>(), 1.0f);
        EXPECT_TRUE(std::isinf(result[1][token].item<float>()));
        EXPECT_LT(result[1][token].item<float>(), 0.0f);
        EXPECT_FLOAT_EQ(result[2][token].item<float>(), 1.0f);
    }
}

TEST_F(ExecOpsTest, testRuntimeApplyPackedMaskLogitsCopiesBackToNonContiguousInput) {
    constexpr int64_t vocab_size = 4;
    auto backing = torch::ones({2, vocab_size + 2}, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
    auto logits  = backing.narrow(/*dim=*/1, /*start=*/1, /*length=*/vocab_size);
    ASSERT_FALSE(logits.is_contiguous());
    ASSERT_EQ(logits.stride(0), vocab_size + 2);

    auto packed_allow_mask = torch::tensor({1, 8}, torch::kInt32).reshape({2, 1});
#if USING_CUDA
    packed_allow_mask = packed_allow_mask.to(torch::kCUDA);
#endif

    ASSERT_NO_THROW(runtimeApplyPackedMaskLogits(logits, packed_allow_mask, vocab_size));
    runtimeSyncAndCheck();

    auto result = backing.cpu().to(torch::kFloat32).contiguous();
    for (int64_t row = 0; row < result.size(0); ++row) {
        EXPECT_FLOAT_EQ(result[row][0].item<float>(), 1.0f);
        EXPECT_FLOAT_EQ(result[row][vocab_size + 1].item<float>(), 1.0f);
        for (int64_t token = 0; token < vocab_size; ++token) {
            const bool allowed = (row == 0 && token == 0) || (row == 1 && token == 3);
            const auto value   = result[row][token + 1].item<float>();
            if (allowed) {
                EXPECT_FLOAT_EQ(value, 1.0f);
            } else {
                EXPECT_TRUE(std::isinf(value));
                EXPECT_LT(value, 0.0f);
            }
        }
    }
}

TEST_F(ExecOpsTest, testWriteCacheStoreMlaBf16PhysicalViewUsesExplicitStride) {
    // Four physical blocks, each containing four kernel blocks. The old shape heuristic treated the leading
    // dimension as kernel-block count and inflated the physical stride by 4x.
    auto kv_cache_base = torch::zeros({4, 8, 16}, torch::kBFloat16);
    expectMlaPhysicalViewUsesExplicitStride(kv_cache_base);
}

TEST_F(ExecOpsTest, testWriteCacheStoreMlaFp8PackedPhysicalViewUsesExplicitStride) {
    // Packed FP8 MLA storage contains FP8 NoPE, BF16 RoPE, and scale bytes in the same physical block.
    auto kv_cache_base = torch::zeros({4, 8, 73}, torch::kUInt8);
    expectMlaPhysicalViewUsesExplicitStride(kv_cache_base);
}

TEST_F(ExecOpsTest, testWriteCacheStoreMhaKernelViewKeepsExplicitKvAndScaleStrides) {
    constexpr size_t physical_block_num         = 4;
    constexpr size_t physical_tokens_per_block  = 8;
    constexpr size_t kernel_tokens_per_block    = 2;
    constexpr size_t kernel_blocks_per_physical = physical_tokens_per_block / kernel_tokens_per_block;

    auto physical_kv = torch::zeros(
        {static_cast<int64_t>(physical_block_num), 2, 1, static_cast<int64_t>(physical_tokens_per_block), 4},
        torch::kUInt8);
    auto kernel_kv      = physical_kv.reshape({static_cast<int64_t>(physical_block_num * kernel_blocks_per_physical),
                                               2,
                                               1,
                                               static_cast<int64_t>(kernel_tokens_per_block),
                                               4});
    auto physical_scale = torch::zeros({static_cast<int64_t>(physical_block_num), 32}, torch::kUInt8);
    auto kernel_scale =
        physical_scale.reshape({static_cast<int64_t>(physical_block_num * kernel_blocks_per_physical), 8});

    const size_t kv_stride    = static_cast<size_t>(physical_kv.nbytes()) / physical_block_num;
    const size_t scale_stride = static_cast<size_t>(physical_scale.nbytes()) / physical_block_num;
    auto         cache_store  = std::make_shared<MockCacheStore>();
    auto         inputs       = makePyCacheStoreInputs(cache_store,
                                         physical_tokens_per_block,
                                         kv_stride,
                                         scale_stride,
                                         physical_block_num,
                                         /*mla_kvcache=*/false);

    torch_ext::LayerKVCache layer_cache;
    layer_cache.kv_cache_base      = kernel_kv;
    layer_cache.kv_scale_base      = kernel_scale;
    layer_cache.seq_size_per_block = kernel_tokens_per_block;
    layer_cache.layer_id           = 0;
    layer_cache.tag                = "default";

    auto input_lengths =
        torch::tensor({static_cast<int32_t>(physical_block_num * physical_tokens_per_block)}, torch::kInt32);
    auto prefix_lengths = torch::tensor({0}, torch::kInt32);
    auto block_ids      = torch::arange(static_cast<int64_t>(physical_block_num), torch::kInt32).reshape({1, -1});

    ASSERT_NO_THROW(WriteCacheStoreOp(
        input_lengths, prefix_lengths, block_ids, std::make_optional(inputs), std::make_optional(layer_cache)));

    ASSERT_EQ(cache_store->records.size(), 1u);
    const auto& record = cache_store->records.front();
    ASSERT_EQ(record.block_count, physical_block_num * 4);
    for (size_t block_id = 0; block_id < physical_block_num; ++block_id) {
        const auto cache_key  = makeCacheKey(0, inputs.cache_keys[block_id], 0);
        const auto kv_base    = reinterpret_cast<uintptr_t>(physical_kv.data_ptr()) + block_id * kv_stride;
        const auto scale_base = reinterpret_cast<uintptr_t>(physical_scale.data_ptr()) + block_id * scale_stride;
        const std::vector<std::tuple<std::string, uintptr_t, uint32_t>> expected = {
            {"k_" + cache_key, kv_base, static_cast<uint32_t>(kv_stride / 2)},
            {"v_" + cache_key, kv_base + kv_stride / 2, static_cast<uint32_t>(kv_stride / 2)},
            {"k_scale_" + cache_key, scale_base, static_cast<uint32_t>(scale_stride / 2)},
            {"v_scale_" + cache_key, scale_base + scale_stride / 2, static_cast<uint32_t>(scale_stride / 2)},
        };
        for (const auto& [key, addr, len] : expected) {
            const auto it = record.blocks.find(key);
            ASSERT_NE(it, record.blocks.end()) << "missing block " << key;
            EXPECT_EQ(reinterpret_cast<uintptr_t>(it->second.addr), addr);
            EXPECT_EQ(it->second.len, len);
        }
    }
}

TEST_F(ExecOpsTest, testWriteCacheStoreSharedPoolUsesPhysicalBlockStrideInsteadOfLayerViewStride) {
    constexpr size_t physical_block_num        = 4;
    constexpr size_t physical_tokens_per_block = 8;
    constexpr size_t pool_block_stride         = 256;
    constexpr size_t layer_view_stride         = 64;
    auto physical_kv = torch::zeros({static_cast<int64_t>(physical_block_num), static_cast<int64_t>(pool_block_stride)},
                                    torch::kUInt8);
    auto kv_cache_base =
        physical_kv.as_strided({static_cast<int64_t>(physical_block_num), static_cast<int64_t>(layer_view_stride)},
                               {static_cast<int64_t>(layer_view_stride), 1});
    auto cache_store                    = std::make_shared<MockCacheStore>();
    auto inputs                         = makePyCacheStoreInputs(cache_store,
                                         physical_tokens_per_block,
                                         pool_block_stride,
                                         /*scale_stride=*/0,
                                         physical_block_num,
                                         /*mla_kvcache=*/false);
    inputs.kv_cache_group_policies      = {{"full", defaultCacheGroupPolicy(CacheGroupType::FULL)},
                                           {"linear", defaultCacheGroupPolicy(CacheGroupType::LINEAR)}};
    inputs.tokens_per_block_by_tag      = {{"full", physical_tokens_per_block}, {"linear", physical_tokens_per_block}};
    inputs.kv_block_stride_bytes_by_tag = {{"full", pool_block_stride}, {"linear", pool_block_stride}};
    inputs.kv_scale_stride_bytes_by_tag = {{"full", 0}, {"linear", 0}};
    inputs.kv_block_transfer_bytes_by_tag = {{"full", pool_block_stride}, {"linear", layer_view_stride}};
    inputs.kv_scale_transfer_bytes_by_tag = {{"full", 0}, {"linear", 0}};

    torch_ext::LayerKVCache layer_cache;
    layer_cache.kv_cache_base      = kv_cache_base;
    layer_cache.seq_size_per_block = physical_tokens_per_block;
    layer_cache.layer_id           = 1;
    layer_cache.tag                = "linear";

    auto input_lengths =
        torch::tensor({static_cast<int32_t>(physical_block_num * physical_tokens_per_block)}, torch::kInt32);
    auto prefix_lengths = torch::tensor({0}, torch::kInt32);
    auto group1_ids     = torch::arange(static_cast<int64_t>(physical_block_num), torch::kInt32).reshape({1, -1});
    auto block_ids      = group1_ids;

    ASSERT_NO_THROW(WriteCacheStoreOp(
        input_lengths, prefix_lengths, block_ids, std::make_optional(inputs), std::make_optional(layer_cache)));

    ASSERT_EQ(cache_store->records.size(), 1u);
    const auto& record = cache_store->records.front();
    ASSERT_EQ(record.block_count, 1u);
    const std::string key = "kv_" + makeCacheKey(0, inputs.cache_keys.back(), 1, "linear");
    const auto        it  = record.blocks.find(key);
    ASSERT_NE(it, record.blocks.end());
    EXPECT_EQ(reinterpret_cast<uintptr_t>(it->second.addr),
              reinterpret_cast<uintptr_t>(kv_cache_base.data_ptr()) + (physical_block_num - 1) * pool_block_stride);
    EXPECT_EQ(it->second.len, layer_view_stride);
    EXPECT_LE(reinterpret_cast<uintptr_t>(it->second.addr) + it->second.len,
              reinterpret_cast<uintptr_t>(physical_kv.data_ptr()) + physical_kv.nbytes());
}

TEST_F(ExecOpsTest, testWriteCacheStoreCpStateSendsCompleteRankLocalRow) {
    constexpr size_t canonical_tokens_per_block = 4;
    constexpr size_t physical_row_stride        = 40;
    constexpr size_t canonical_block_num        = 4;

    auto cache_store                    = std::make_shared<MockCacheStore>();
    auto inputs                         = makePyCacheStoreInputs(cache_store,
                                         canonical_tokens_per_block,
                                         physical_row_stride,
                                         /*scale_stride=*/0,
                                         canonical_block_num,
                                         /*mla_kvcache=*/false);
    auto state_policy                   = defaultCacheGroupPolicy(CacheGroupType::SWA);
    state_policy.cp_slice               = CpBlockSliceMode::PAYLOAD_BYTES;
    inputs.kv_cache_group_policies      = {{"state", state_policy}};
    inputs.tokens_per_block_by_tag      = {{"state", canonical_tokens_per_block}};
    inputs.kv_block_stride_bytes_by_tag = {{"state", physical_row_stride}};
    inputs.kv_scale_stride_bytes_by_tag = {{"state", 0}};
    inputs.use_opaque_kv_cache_store    = true;
    inputs.cp_rank                      = 1;
    inputs.cp_size                      = 2;

    torch_ext::LayerKVCache layer_cache;
    layer_cache.kv_cache_base      = torch::zeros({2, static_cast<int64_t>(physical_row_stride)}, torch::kUInt8);
    layer_cache.seq_size_per_block = canonical_tokens_per_block;
    layer_cache.layer_id           = 2;
    layer_cache.tag                = "state";

    // The global key namespace uses 2-token canonical blocks under CP. A
    // 2-token reused prefix is therefore valid even though the rank-local
    // physical row spans 4 tokens.
    auto input_lengths  = torch::tensor({6}, torch::kInt32);
    auto prefix_lengths = torch::tensor({2}, torch::kInt32);
    auto block_ids      = torch::tensor({{0, 1}}, torch::kInt32);

    ASSERT_NO_THROW(WriteCacheStoreOp(
        input_lengths, prefix_lengths, block_ids, std::make_optional(inputs), std::make_optional(layer_cache)));

    ASSERT_EQ(cache_store->records.size(), 1u);
    const auto& record = cache_store->records.front();
    ASSERT_EQ(record.blocks.size(), 2u);
    const auto base_addr = reinterpret_cast<uintptr_t>(layer_cache.kv_cache_base.data_ptr());
    for (size_t local_block = 0; local_block < 2; ++local_block) {
        const size_t key_index = 1 + local_block * 2;
        const auto   key = "kv_" + makeCacheKey(0, inputs.cache_keys[key_index], layer_cache.layer_id, layer_cache.tag);
        const auto   it  = record.blocks.find(key);
        ASSERT_NE(it, record.blocks.end()) << "missing block " << key;
        EXPECT_EQ(reinterpret_cast<uintptr_t>(it->second.addr), base_addr + local_block * physical_row_stride);
        EXPECT_EQ(it->second.len, physical_row_stride);
    }
}

TEST_F(ExecOpsTest, testWriteCacheStoreCpRoundRobinUsesCanonicalKeyCount) {
    constexpr size_t physical_tokens_per_block = 4;
    constexpr size_t physical_row_stride       = 16;
    constexpr size_t canonical_block_num       = 11;
    constexpr size_t local_block_num           = 6;

    auto cache_store = std::make_shared<MockCacheStore>();
    auto inputs      = makePyCacheStoreInputs(cache_store,
                                         physical_tokens_per_block,
                                         physical_row_stride,
                                         /*scale_stride=*/0,
                                         canonical_block_num,
                                         /*mla_kvcache=*/true);
    inputs.cp_rank   = 0;
    inputs.cp_size   = 2;

    torch_ext::LayerKVCache layer_cache;
    layer_cache.kv_cache_base =
        torch::zeros({static_cast<int64_t>(local_block_num), static_cast<int64_t>(physical_row_stride)}, torch::kUInt8);
    layer_cache.seq_size_per_block = physical_tokens_per_block;
    layer_cache.layer_id           = 0;
    layer_cache.tag                = "default";

    auto input_lengths  = torch::tensor({22}, torch::kInt32);
    auto prefix_lengths = torch::tensor({0}, torch::kInt32);
    auto block_ids      = torch::arange(static_cast<int64_t>(local_block_num), torch::kInt32).reshape({1, -1});

    ASSERT_NO_THROW(WriteCacheStoreOp(
        input_lengths, prefix_lengths, block_ids, std::make_optional(inputs), std::make_optional(layer_cache)));

    ASSERT_EQ(cache_store->records.size(), 1u);
    const auto& record = cache_store->records.front();
    ASSERT_EQ(record.blocks.size(), local_block_num);
    const auto base_addr = reinterpret_cast<uintptr_t>(layer_cache.kv_cache_base.data_ptr());
    for (size_t local_block = 0; local_block < local_block_num; ++local_block) {
        const size_t key_index = local_block * static_cast<size_t>(inputs.cp_size);
        const auto   key = "kv_" + makeCacheKey(0, inputs.cache_keys[key_index], layer_cache.layer_id, layer_cache.tag);
        const auto   it  = record.blocks.find(key);
        ASSERT_NE(it, record.blocks.end()) << "missing block " << key;
        EXPECT_EQ(reinterpret_cast<uintptr_t>(it->second.addr), base_addr + local_block * physical_row_stride);
        EXPECT_EQ(it->second.len, physical_row_stride);
    }
}

TEST_F(ExecOpsTest, testWriteCacheStoreFailureBufferContainsEveryBlockKey) {
    rtp_llm::test::TestLogCapture log_capture("write_cache_failure");
    constexpr size_t              block_num        = 2;
    constexpr size_t              tokens_per_block = 4;
    constexpr size_t              kv_stride        = 16;
    auto                          cache_store      = std::make_shared<MockCacheStore>();
    cache_store->store_success                     = false;
    cache_store->store_error                       = CacheStoreErrorCode::StoreFailed;
    auto inputs                                    = makePyCacheStoreInputs(
        cache_store, tokens_per_block, kv_stride, /*scale_stride=*/0, block_num, /*mla_kvcache=*/true);

    torch_ext::LayerKVCache layer_cache;
    layer_cache.kv_cache_base      = torch::zeros({2, 16}, torch::kUInt8);
    layer_cache.seq_size_per_block = tokens_per_block;
    layer_cache.layer_id           = 0;
    layer_cache.tag                = "default";

    ASSERT_NO_THROW(WriteCacheStoreOp(torch::tensor({8}, torch::kInt32),
                                      torch::tensor({0}, torch::kInt32),
                                      torch::tensor({{0, 1}}, torch::kInt32),
                                      std::make_optional(inputs),
                                      std::make_optional(layer_cache)));

    ASSERT_EQ(cache_store->records.size(), 1u);
    ASSERT_EQ(cache_store->records[0].blocks.size(), block_num);
    const auto log_content = log_capture.content();
    EXPECT_NE(log_content.find("PD_CACHE_KEY_WRITE_FAILED"), std::string::npos);
    for (size_t block_id = 0; block_id < block_num; ++block_id) {
        const auto key = "kv_" + makeCacheKey(0, inputs.cache_keys[block_id], 0);
        EXPECT_NE(cache_store->records[0].blocks.find(key), cache_store->records[0].blocks.end());
        EXPECT_NE(log_content.find(key), std::string::npos);
    }
}

TEST_F(ExecOpsTest, testWriteCacheStoreSuccessDoesNotLogBlockKeys) {
    rtp_llm::test::TestLogCapture log_capture("write_cache_success");
    constexpr size_t              block_num        = 2;
    constexpr size_t              tokens_per_block = 4;
    auto                          cache_store      = std::make_shared<MockCacheStore>();
    auto                          inputs           = makePyCacheStoreInputs(
        cache_store, tokens_per_block, /*kv_stride=*/16, /*scale_stride=*/0, block_num, /*mla_kvcache=*/true);

    torch_ext::LayerKVCache layer_cache;
    layer_cache.kv_cache_base      = torch::zeros({2, 16}, torch::kUInt8);
    layer_cache.seq_size_per_block = tokens_per_block;
    layer_cache.layer_id           = 0;
    layer_cache.group_id           = 0;
    layer_cache.tag                = "default";

    ASSERT_NO_THROW(WriteCacheStoreOp(torch::tensor({8}, torch::kInt32),
                                      torch::tensor({0}, torch::kInt32),
                                      torch::tensor({{0, 1}}, torch::kInt32),
                                      std::make_optional(inputs),
                                      std::make_optional(layer_cache)));

    const auto log_content = log_capture.content();
    EXPECT_EQ(log_content.find("PD_CACHE_KEY_WRITE_FAILED"), std::string::npos);
    for (const auto& cache_key : inputs.cache_keys) {
        EXPECT_EQ(log_content.find(makeCacheKey(0, cache_key, 0)), std::string::npos);
    }
}

TEST_F(ExecOpsTest, testLoadContextFailureDebugInfoContainsEveryBlockKey) {
    auto cache_store          = std::make_shared<MockCacheStore>();
    cache_store->load_success = false;
    cache_store->load_error   = CacheStoreErrorCode::LoadConnectFailed;

    auto request_buffer = std::make_shared<RequestBlockBuffer>("request", "request-0");
    request_buffer->addBlock(
        "kv_key_0", std::shared_ptr<void>(reinterpret_cast<void*>(1), [](void*) {}), 1, true, true);
    request_buffer->addBlock(
        "kv_key_1", std::shared_ptr<void>(reinterpret_cast<void*>(2), [](void*) {}), 1, true, true);

    auto load_context = std::make_shared<LoadContext>(cache_store, /*combine_load=*/false);
    load_context->load(
        {request_buffer},
        "127.0.0.1",
        /*port=*/1,
        /*rdma_port=*/2,
        /*timeout_ms=*/1000,
        []() { return false; },
        /*partition_count=*/1,
        /*partition_id=*/0);
    load_context->waitDone();

    ASSERT_FALSE(load_context->success());
    const auto debug_infos = load_context->failedBlockDebugInfos();
    ASSERT_EQ(debug_infos.size(), 1u);
    EXPECT_NE(debug_infos[0].find("kv_key_0"), std::string::npos);
    EXPECT_NE(debug_infos[0].find("kv_key_1"), std::string::npos);
}

// Layer 0 maps to group 0 (LINEAR).
// CacheGroupType::LINEAR means only the last block is transferred;
// so exactly 1 block should be recorded in the mock store.
TEST_F(ExecOpsTest, testWriteCacheStoreTag_LinearGroup) {
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
    ASSERT_EQ(cache_store->records[0].block_keys.size(), 1u);
    EXPECT_EQ(cache_store->records[0].block_keys[0].rfind("kv_", 0), 0u)
        << "Hybrid cache-store must write opaque kv_ keys even when use_opaque_kv_cache_store=false";
}

// Layer 1 maps to group 1 (FULL).
// CacheGroupType::FULL means all blocks are transferred;
// with total_blocks = 3, exactly 3 opaque kv entries should reach the mock store.
TEST_F(ExecOpsTest, testWriteCacheStoreTag_FullGroup) {
    auto cache_store = std::make_shared<MockCacheStore>();
    auto param       = makeHybridInputs(/*layer_id=*/1);

    rtp_llm::KvCacheInfo kv;
    kv.layer_num       = 2;
    kv.kv_cache_buffer = torch::zeros({192}, torch::kByte);

    ASSERT_NO_THROW(rtp_llm::runtimeWriteCacheStore(param, kv, /*mla_kvcache=*/false, cache_store));

    ASSERT_EQ(cache_store->records.size(), 1u);
    EXPECT_EQ(cache_store->records[0].block_count, 3u)
        << "Layer 1 -> group 1 (FULL): all 3 blocks should be stored as opaque kv entries";
    EXPECT_EQ(countKeyPrefix(cache_store->records[0].block_keys, "kv_"), 3u);
}

TEST_F(ExecOpsTest, testWriteCacheStoreRejectsNonLocalBlockTable) {
    auto cache_store = std::make_shared<MockCacheStore>();
    auto param       = makeHybridInputs(/*layer_id=*/1);

    param.host_kv_cache_offset = torch::ones({2, 1, 3}, torch::kInt32);
    rtp_llm::KvCacheInfo kv;
    kv.layer_num       = 2;
    kv.kv_cache_buffer = torch::zeros({128}, torch::kByte);

    EXPECT_ANY_THROW(rtp_llm::runtimeWriteCacheStore(param, kv, /*mla_kvcache=*/false, cache_store));
}

TEST_F(ExecOpsTest, testWriteCacheStoreTag_LocalOffset) {
    auto cache_store           = std::make_shared<MockCacheStore>();
    auto param                 = makeHybridInputs(/*layer_id=*/1);
    param.host_kv_cache_offset = torch::tensor({{0, 1, 2}}, torch::kInt32);

    rtp_llm::KvCacheInfo kv;
    kv.layer_num       = 2;
    kv.kv_cache_buffer = torch::zeros({192}, torch::kByte);

    ASSERT_NO_THROW(rtp_llm::runtimeWriteCacheStore(param, kv, /*mla_kvcache=*/false, cache_store));

    ASSERT_EQ(cache_store->records.size(), 1u);
    EXPECT_EQ(cache_store->records[0].block_count, 3u)
        << "2-D path, layer 1 -> group 1 (FULL): all 3 blocks should be stored as opaque kv entries";
    EXPECT_EQ(countKeyPrefix(cache_store->records[0].block_keys, "kv_"), 3u);
}

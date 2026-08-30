#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/testing/TestLogCapture.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#if USING_CUDA
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
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

static torch_ext::PyCacheStoreInputs makePyCacheStoreInputs(size_t tokens_per_block, size_t block_num) {
    torch_ext::PyCacheStoreInputs inputs;
    inputs.input_lengths_host    = torch::tensor({static_cast<int32_t>(tokens_per_block * block_num)}, torch::kInt32);
    inputs.prefix_lengths_host   = torch::tensor({0}, torch::kInt32);
    inputs.host_kv_cache_offset  = torch::arange(static_cast<int64_t>(block_num), torch::kInt32).reshape({1, -1});
    inputs.request_id            = torch::tensor({int64_t(42)}, torch::kInt64);
    inputs.request_pd_separation = torch::tensor({true}, torch::kBool);
    inputs.cache_keys = torch::arange(int64_t(100), int64_t(100 + block_num), torch::kInt64).reshape({1, -1});
    return inputs;
}

static torch::Tensor makeNonContiguousVector(const torch::Tensor& values, int64_t gap_value) {
    auto storage = torch::empty({values.numel(), 2}, values.options());
    storage.select(1, 0).copy_(values);
    storage.select(1, 1).fill_(gap_value);
    return storage.select(1, 0);
}

static torch::Tensor makeNonContiguousMatrix(const torch::Tensor& values, int64_t gap_value) {
    auto storage = torch::full({values.size(0), values.size(1) * 2}, gap_value, values.options());
    auto view    = storage.slice(/*dim=*/1, /*start=*/0, /*end=*/values.size(1) * 2, /*step=*/2);
    view.copy_(values);
    return view;
}

static std::shared_ptr<KVCacheSpec> makeTestSpec(const std::string& tag, size_t tokens_per_block, bool mla_cache) {
    std::shared_ptr<KVCacheSpec> spec = mla_cache ?
                                            std::static_pointer_cast<KVCacheSpec>(std::make_shared<MLAKVCacheSpec>()) :
                                            std::static_pointer_cast<KVCacheSpec>(std::make_shared<MHAKVCacheSpec>());
    spec->seq_size_per_block          = static_cast<uint32_t>(tokens_per_block);
    return spec;
}

static CacheConfig makeCacheConfig(size_t             tokens_per_block,
                                   size_t             physical_kv_stride,
                                   size_t             physical_scale_stride,
                                   size_t             block_num,
                                   const std::string& tag               = "default",
                                   int                layer_id          = 0,
                                   CacheGroupPolicy   policy            = defaultCacheGroupPolicy(CacheGroupType::FULL),
                                   bool               add_dummy_group   = false,
                                   bool               mla_cache         = false,
                                   bool               independent_pools = false,
                                   size_t             transfer_kv_bytes = 0,
                                   size_t             transfer_scale_bytes = 0,
                                   bool               opaque_store         = false) {
    CacheConfig config;
    config.layer_num                   = static_cast<uint32_t>(layer_id + 1);
    config.layer_all_num               = config.layer_num;
    config.block_num                   = static_cast<uint32_t>(block_num);
    config.seq_size_per_block          = tokens_per_block;
    config.kernel_seq_size_per_block   = tokens_per_block;
    config.kv_block_stride_bytes       = physical_kv_stride;
    config.kv_scale_stride_bytes       = physical_scale_stride;
    config.use_independent_block_pools = independent_pools;
    config.use_opaque_kv_cache_store   = opaque_store;

    GroupBase target_group;
    target_group.tag                       = tag;
    target_group.spec                      = makeTestSpec(tag, tokens_per_block, mla_cache);
    target_group.policy                    = policy;
    target_group.block_num                 = static_cast<uint32_t>(block_num);
    target_group.seq_size_per_block        = tokens_per_block;
    target_group.kernel_seq_size_per_block = tokens_per_block;
    target_group.kv_block_stride_bytes     = transfer_kv_bytes == 0 ? physical_kv_stride : transfer_kv_bytes;
    target_group.kv_scale_stride_bytes     = transfer_scale_bytes == 0 ? physical_scale_stride : transfer_scale_bytes;

    std::vector<LayerBase> layers(static_cast<size_t>(layer_id + 1));
    for (size_t i = 0; i < layers.size(); ++i) {
        layers[i].layer_id = static_cast<int>(i);
    }

    std::vector<GroupBase> groups;
    if (add_dummy_group) {
        GroupBase dummy_group;
        dummy_group.tag                       = tag == "full" ? "linear" : "full";
        dummy_group.spec                      = makeTestSpec(dummy_group.tag, tokens_per_block, false);
        dummy_group.policy                    = defaultCacheGroupPolicy(CacheGroupType::FULL);
        dummy_group.block_num                 = static_cast<uint32_t>(block_num);
        dummy_group.seq_size_per_block        = tokens_per_block;
        dummy_group.kernel_seq_size_per_block = tokens_per_block;
        dummy_group.kv_block_stride_bytes     = physical_kv_stride;
        dummy_group.kv_scale_stride_bytes     = physical_scale_stride;
        for (int i = 0; i < layer_id; ++i) {
            dummy_group.layer_ids.push_back(i);
            layers[static_cast<size_t>(i)].group_tags = {dummy_group.tag};
        }
        groups.push_back(std::move(dummy_group));
    } else {
        for (int i = 0; i < layer_id; ++i) {
            target_group.layer_ids.push_back(i);
            layers[static_cast<size_t>(i)].group_tags = {tag};
        }
    }
    target_group.layer_ids.push_back(layer_id);
    layers[static_cast<size_t>(layer_id)].group_tags = {tag};
    groups.push_back(std::move(target_group));
    config.setTopology(std::move(groups), std::move(layers));
    return config;
}

static std::string cacheKeyAt(const torch_ext::PyCacheStoreInputs& inputs,
                              size_t                               index,
                              int                                  layer_id,
                              const std::string&                   tag      = "default",
                              size_t                               model_id = 0) {
    return makeCacheKey(model_id, std::to_string(inputs.cache_keys.data_ptr<int64_t>()[index]), layer_id, tag);
}

static void expectMlaPhysicalViewUsesExplicitStride(const torch::Tensor& kv_cache_base, size_t blocks_to_write = 4) {
    constexpr size_t physical_block_num        = 4;
    constexpr size_t physical_tokens_per_block = 8;
    constexpr int    kernel_tokens_per_block   = 2;

    ASSERT_EQ(static_cast<size_t>(kv_cache_base.size(0)), physical_block_num);
    const size_t explicit_stride = static_cast<size_t>(kv_cache_base.nbytes()) / physical_block_num;
    auto         cache_store     = std::make_shared<MockCacheStore>();
    auto         inputs          = makePyCacheStoreInputs(physical_tokens_per_block, physical_block_num);
    inputs.input_lengths_host.fill_(static_cast<int64_t>(physical_tokens_per_block * blocks_to_write));
    auto config = makeCacheConfig(physical_tokens_per_block,
                                  explicit_stride,
                                  /*physical_scale_stride=*/0,
                                  physical_block_num,
                                  "default",
                                  /*layer_id=*/0,
                                  defaultCacheGroupPolicy(CacheGroupType::FULL),
                                  /*add_dummy_group=*/false,
                                  /*mla_cache=*/true);

    torch_ext::LayerKVCache layer_cache;
    layer_cache.kv_cache_base      = kv_cache_base;
    layer_cache.seq_size_per_block = kernel_tokens_per_block;
    layer_cache.layer_id           = 0;
    layer_cache.tag                = "default";

    ASSERT_NO_THROW(runtimeWriteCacheStore(
        inputs, layer_cache, config, cache_store, /*cache_model_id=*/0, /*cp_rank=*/0, /*cp_size=*/1, nullptr));

    ASSERT_EQ(cache_store->records.size(), 1u);
    const auto& record = cache_store->records.front();
    ASSERT_EQ(record.block_count, blocks_to_write);
    ASSERT_EQ(record.blocks.size(), blocks_to_write);

    const auto base_addr   = reinterpret_cast<uintptr_t>(kv_cache_base.data_ptr());
    const auto storage_end = base_addr + static_cast<size_t>(kv_cache_base.nbytes());
    for (size_t block_id = 0; block_id < blocks_to_write; ++block_id) {
        const std::string key = "kv_" + cacheKeyAt(inputs, block_id, 0);
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
    auto logits = torch::ones({2, 8}, torch::kCUDA);
    auto mask   = torch::zeros({2, 8}, torch::TensorOptions(torch::kBool).device(torch::kCUDA));
    mask[0][0]  = true;
    mask[1][3]  = true;

    ASSERT_NO_THROW(runtimeMaskLogits(logits, mask));
    runtimeSyncAndCheck();

    auto result = logits.cpu();
    EXPECT_TRUE(std::isinf(result[0][0].item<float>()));
    EXPECT_LT(result[0][0].item<float>(), 0.0f);
    EXPECT_TRUE(std::isinf(result[1][3].item<float>()));
    EXPECT_LT(result[1][3].item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(result[0][1].item<float>(), 1.0f);
    EXPECT_FLOAT_EQ(result[1][2].item<float>(), 1.0f);
}

TEST_F(ExecOpsTest, testRuntimeApplyPackedMaskLogitsUsesCompactRowMapping) {
    constexpr int64_t vocab_size        = 35;
    constexpr int64_t logits_columns    = 40;
    auto              packed_allow_mask = torch::tensor({1, 4, 2, 2}, torch::kInt32).reshape({2, 2}).to(torch::kCUDA);
    auto              row_indices       = torch::tensor({1, 3}, torch::kInt32).to(torch::kCUDA);

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
                } else if (dtype == torch::kFloat32) {
                    EXPECT_FLOAT_EQ(result[row][token].item<float>(), -std::numeric_limits<float>::max());
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
    auto              packed_allow_mask = torch::tensor({1, 4}, torch::kInt32).reshape({1, 2}).to(torch::kCUDA);

    ASSERT_NO_THROW(runtimeApplyPackedMaskLogits(logits, packed_allow_mask, vocab_size));
    runtimeSyncAndCheck();

    auto result = logits.cpu().contiguous();
    for (int64_t token = 0; token < vocab_size; ++token) {
        if (token == 0 || token == 34) {
            EXPECT_FLOAT_EQ(result[token].item<float>(), 1.0f);
        } else {
            EXPECT_FLOAT_EQ(result[token].item<float>(), -std::numeric_limits<float>::max());
        }
    }
}

TEST_F(ExecOpsTest, testRuntimeApplyPackedMaskLogitsSkipsOutOfRangeRows) {
    constexpr int64_t vocab_size = 4;
    auto              logits = torch::ones({3, vocab_size}, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
    auto packed_allow_mask   = torch::zeros({3, 1}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
    auto row_indices         = torch::tensor({-1, 1, 3}, torch::kInt32).to(torch::kCUDA);

    ASSERT_NO_THROW(runtimeApplyPackedMaskLogits(logits, packed_allow_mask, row_indices, vocab_size));
    runtimeSyncAndCheck();

    auto result = logits.cpu().to(torch::kFloat32).contiguous();
    for (int64_t token = 0; token < vocab_size; ++token) {
        EXPECT_FLOAT_EQ(result[0][token].item<float>(), 1.0f);
        EXPECT_FLOAT_EQ(result[1][token].item<float>(), -std::numeric_limits<float>::max());
        EXPECT_FLOAT_EQ(result[2][token].item<float>(), 1.0f);
    }
}

TEST_F(ExecOpsTest, testRuntimeApplyPackedMaskLogitsCopiesBackToNonContiguousInput) {
    constexpr int64_t vocab_size = 4;
    auto backing = torch::ones({2, vocab_size + 2}, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
    auto logits  = backing.narrow(/*dim=*/1, /*start=*/1, /*length=*/vocab_size);
    ASSERT_FALSE(logits.is_contiguous());
    ASSERT_EQ(logits.stride(0), vocab_size + 2);

    auto packed_allow_mask = torch::tensor({1, 8}, torch::kInt32).reshape({2, 1}).to(torch::kCUDA);

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
                EXPECT_FLOAT_EQ(value, -std::numeric_limits<float>::max());
            }
        }
    }
}

TEST_F(ExecOpsTest, testWriteCacheStoreRejectsUndefinedRequestId) {
    auto inputs       = makePyCacheStoreInputs(/*tokens_per_block=*/2, /*block_num=*/1);
    inputs.request_id = torch::Tensor();
    auto config       = makeCacheConfig(/*tokens_per_block=*/2,
                                  /*physical_kv_stride=*/64,
                                  /*physical_scale_stride=*/0,
                                  /*block_num=*/1,
                                  "default",
                                  /*layer_id=*/0,
                                  defaultCacheGroupPolicy(CacheGroupType::FULL),
                                  /*add_dummy_group=*/false,
                                  /*mla_cache=*/true);

    torch_ext::LayerKVCache layer_cache;
    layer_cache.kv_cache_base      = torch::zeros({1, 64}, torch::kUInt8);
    layer_cache.seq_size_per_block = 2;
    layer_cache.layer_id           = 0;
    layer_cache.tag                = "default";

    try {
        runtimeWriteCacheStore(inputs,
                               layer_cache,
                               config,
                               std::make_shared<MockCacheStore>(),
                               /*cache_model_id=*/0,
                               /*cp_rank=*/0,
                               /*cp_size=*/1,
                               nullptr);
        FAIL() << "expected an undefined request_id to fail";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("request_id must be defined"), std::string::npos);
    }
}

TEST_F(ExecOpsTest, testWriteCacheStoreMlaBf16PhysicalViewUsesExplicitStride) {
    // Four physical blocks, each containing four kernel blocks. The old shape heuristic treated the leading
    // dimension as kernel-block count and inflated the physical stride by 4x. Writing two physical blocks also
    // distinguishes the physical page size from the smaller kernel page exposed by LayerKVCache.
    auto kv_cache_base = torch::zeros({4, 8, 16}, torch::kBFloat16);
    expectMlaPhysicalViewUsesExplicitStride(kv_cache_base, /*blocks_to_write=*/2);
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
    auto         inputs       = makePyCacheStoreInputs(physical_tokens_per_block, physical_block_num);
    auto         config       = makeCacheConfig(physical_tokens_per_block, kv_stride, scale_stride, physical_block_num);

    torch_ext::LayerKVCache layer_cache;
    layer_cache.kv_cache_base      = kernel_kv;
    layer_cache.kv_scale_base      = kernel_scale;
    layer_cache.seq_size_per_block = kernel_tokens_per_block;
    layer_cache.layer_id           = 0;
    layer_cache.tag                = "default";

    ASSERT_NO_THROW(runtimeWriteCacheStore(
        inputs, layer_cache, config, cache_store, /*cache_model_id=*/0, /*cp_rank=*/0, /*cp_size=*/1, nullptr));

    ASSERT_EQ(cache_store->records.size(), 1u);
    const auto& record = cache_store->records.front();
    ASSERT_EQ(record.block_count, physical_block_num * 4);
    for (size_t block_id = 0; block_id < physical_block_num; ++block_id) {
        const auto cache_key  = cacheKeyAt(inputs, block_id, 0);
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
    auto cache_store = std::make_shared<MockCacheStore>();
    auto inputs      = makePyCacheStoreInputs(physical_tokens_per_block, physical_block_num);
    auto config      = makeCacheConfig(physical_tokens_per_block,
                                  pool_block_stride,
                                  /*physical_scale_stride=*/0,
                                  physical_block_num,
                                  "linear",
                                  /*layer_id=*/1,
                                  defaultCacheGroupPolicy(CacheGroupType::LINEAR),
                                  /*add_dummy_group=*/true,
                                  /*mla_cache=*/false,
                                  /*independent_pools=*/false,
                                  /*transfer_kv_bytes=*/layer_view_stride);

    torch_ext::LayerKVCache layer_cache;
    layer_cache.kv_cache_base      = kv_cache_base;
    layer_cache.seq_size_per_block = physical_tokens_per_block;
    layer_cache.layer_id           = 1;
    layer_cache.tag                = "linear";

    ASSERT_NO_THROW(runtimeWriteCacheStore(
        inputs, layer_cache, config, cache_store, /*cache_model_id=*/0, /*cp_rank=*/0, /*cp_size=*/1, nullptr));

    ASSERT_EQ(cache_store->records.size(), 1u);
    const auto& record = cache_store->records.front();
    ASSERT_EQ(record.block_count, 1u);
    const std::string key = "kv_" + cacheKeyAt(inputs, physical_block_num - 1, 1, "linear");
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

    auto cache_store      = std::make_shared<MockCacheStore>();
    auto inputs           = makePyCacheStoreInputs(canonical_tokens_per_block, canonical_block_num);
    auto state_policy     = defaultCacheGroupPolicy(CacheGroupType::SWA);
    state_policy.cp_slice = CpBlockSliceMode::PAYLOAD_BYTES;
    auto config           = makeCacheConfig(canonical_tokens_per_block,
                                  physical_row_stride,
                                  /*physical_scale_stride=*/0,
                                  /*block_num=*/2,
                                  "state",
                                  /*layer_id=*/2,
                                  state_policy,
                                  /*add_dummy_group=*/false,
                                  /*mla_cache=*/false,
                                  /*independent_pools=*/false,
                                  /*transfer_kv_bytes=*/physical_row_stride,
                                  /*transfer_scale_bytes=*/0,
                                  /*opaque_store=*/true);

    torch_ext::LayerKVCache layer_cache;
    layer_cache.kv_cache_base      = torch::zeros({2, static_cast<int64_t>(physical_row_stride)}, torch::kUInt8);
    layer_cache.seq_size_per_block = canonical_tokens_per_block;
    layer_cache.layer_id           = 2;
    layer_cache.tag                = "state";

    // The global key namespace uses 2-token canonical blocks under CP. A
    // 2-token reused prefix is therefore valid even though the rank-local
    // physical row spans 4 tokens.
    inputs.input_lengths_host   = torch::tensor({6}, torch::kInt32);
    inputs.prefix_lengths_host  = torch::tensor({2}, torch::kInt32);
    inputs.host_kv_cache_offset = torch::tensor({{0, 1}}, torch::kInt32);

    ASSERT_NO_THROW(runtimeWriteCacheStore(
        inputs, layer_cache, config, cache_store, /*cache_model_id=*/0, /*cp_rank=*/1, /*cp_size=*/2, nullptr));

    ASSERT_EQ(cache_store->records.size(), 1u);
    const auto& record = cache_store->records.front();
    ASSERT_EQ(record.blocks.size(), 2u);
    const auto base_addr = reinterpret_cast<uintptr_t>(layer_cache.kv_cache_base.data_ptr());
    for (size_t local_block = 0; local_block < 2; ++local_block) {
        const size_t key_index = 1 + local_block * 2;
        const auto   key       = "kv_" + cacheKeyAt(inputs, key_index, layer_cache.layer_id, layer_cache.tag);
        const auto   it        = record.blocks.find(key);
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
    auto inputs      = makePyCacheStoreInputs(physical_tokens_per_block, canonical_block_num);
    auto config      = makeCacheConfig(physical_tokens_per_block,
                                  physical_row_stride,
                                  /*physical_scale_stride=*/0,
                                  local_block_num,
                                  "default",
                                  /*layer_id=*/0,
                                  defaultCacheGroupPolicy(CacheGroupType::FULL),
                                  /*add_dummy_group=*/false,
                                  /*mla_cache=*/true);

    torch_ext::LayerKVCache layer_cache;
    layer_cache.kv_cache_base =
        torch::zeros({static_cast<int64_t>(local_block_num), static_cast<int64_t>(physical_row_stride)}, torch::kUInt8);
    layer_cache.seq_size_per_block = physical_tokens_per_block;
    layer_cache.layer_id           = 0;
    layer_cache.tag                = "default";

    inputs.input_lengths_host   = torch::tensor({22}, torch::kInt32);
    inputs.host_kv_cache_offset = torch::arange(static_cast<int64_t>(local_block_num), torch::kInt32).reshape({1, -1});

    ASSERT_NO_THROW(runtimeWriteCacheStore(
        inputs, layer_cache, config, cache_store, /*cache_model_id=*/0, /*cp_rank=*/0, /*cp_size=*/2, nullptr));

    ASSERT_EQ(cache_store->records.size(), 1u);
    const auto& record = cache_store->records.front();
    ASSERT_EQ(record.blocks.size(), local_block_num);
    const auto base_addr = reinterpret_cast<uintptr_t>(layer_cache.kv_cache_base.data_ptr());
    for (size_t local_block = 0; local_block < local_block_num; ++local_block) {
        const size_t key_index = local_block * 2;
        const auto   key       = "kv_" + cacheKeyAt(inputs, key_index, layer_cache.layer_id, layer_cache.tag);
        const auto   it        = record.blocks.find(key);
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
    auto inputs                                    = makePyCacheStoreInputs(tokens_per_block, block_num);
    auto config                                    = makeCacheConfig(tokens_per_block,
                                  kv_stride,
                                  /*physical_scale_stride=*/0,
                                  block_num,
                                  "default",
                                  /*layer_id=*/0,
                                  defaultCacheGroupPolicy(CacheGroupType::FULL),
                                  /*add_dummy_group=*/false,
                                  /*mla_cache=*/true);

    torch_ext::LayerKVCache layer_cache;
    layer_cache.kv_cache_base      = torch::zeros({2, 16}, torch::kUInt8);
    layer_cache.seq_size_per_block = tokens_per_block;
    layer_cache.layer_id           = 0;
    layer_cache.tag                = "default";

    ASSERT_NO_THROW(runtimeWriteCacheStore(
        inputs, layer_cache, config, cache_store, /*cache_model_id=*/0, /*cp_rank=*/0, /*cp_size=*/1, nullptr));

    ASSERT_EQ(cache_store->records.size(), 1u);
    ASSERT_EQ(cache_store->records[0].blocks.size(), block_num);
    const auto log_content = log_capture.content();
    EXPECT_NE(log_content.find("PD_CACHE_KEY_WRITE_FAILED"), std::string::npos);
    for (size_t block_id = 0; block_id < block_num; ++block_id) {
        const auto key = "kv_" + cacheKeyAt(inputs, block_id, 0);
        EXPECT_NE(cache_store->records[0].blocks.find(key), cache_store->records[0].blocks.end());
        EXPECT_NE(log_content.find(key), std::string::npos);
    }
}

TEST_F(ExecOpsTest, testWriteCacheStoreSuccessDoesNotLogBlockKeys) {
    rtp_llm::test::TestLogCapture log_capture("write_cache_success");
    constexpr size_t              block_num        = 2;
    constexpr size_t              tokens_per_block = 4;
    auto                          cache_store      = std::make_shared<MockCacheStore>();
    auto                          inputs           = makePyCacheStoreInputs(tokens_per_block, block_num);
    auto                          config           = makeCacheConfig(tokens_per_block,
                                  /*physical_kv_stride=*/16,
                                  /*physical_scale_stride=*/0,
                                  block_num,
                                  "default",
                                  /*layer_id=*/0,
                                  defaultCacheGroupPolicy(CacheGroupType::FULL),
                                  /*add_dummy_group=*/false,
                                  /*mla_cache=*/true);

    torch_ext::LayerKVCache layer_cache;
    layer_cache.kv_cache_base      = torch::zeros({2, 16}, torch::kUInt8);
    layer_cache.seq_size_per_block = tokens_per_block;
    layer_cache.layer_id           = 0;
    layer_cache.tag                = "default";

    ASSERT_NO_THROW(runtimeWriteCacheStore(
        inputs, layer_cache, config, cache_store, /*cache_model_id=*/0, /*cp_rank=*/0, /*cp_size=*/1, nullptr));

    const auto log_content = log_capture.content();
    EXPECT_EQ(log_content.find("PD_CACHE_KEY_WRITE_FAILED"), std::string::npos);
    for (size_t block_id = 0; block_id < block_num; ++block_id) {
        EXPECT_EQ(log_content.find(cacheKeyAt(inputs, block_id, 0)), std::string::npos);
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

TEST_F(ExecOpsTest, testWriteCacheStoreTag_LinearGroup) {
    auto                    cache_store = std::make_shared<MockCacheStore>();
    auto                    inputs      = makePyCacheStoreInputs(/*tokens_per_block=*/2, /*block_num=*/3);
    auto                    config      = makeCacheConfig(/*tokens_per_block=*/2,
                                  /*physical_kv_stride=*/64,
                                  /*physical_scale_stride=*/0,
                                  /*block_num=*/3,
                                  "linear",
                                  /*layer_id=*/0,
                                  defaultCacheGroupPolicy(CacheGroupType::LINEAR),
                                  /*add_dummy_group=*/true);
    torch_ext::LayerKVCache layer_cache;
    layer_cache.kv_cache_base      = torch::zeros({3, 64}, torch::kUInt8);
    layer_cache.seq_size_per_block = 2;
    layer_cache.layer_id           = 0;
    layer_cache.tag                = "linear";

    ASSERT_NO_THROW(runtimeWriteCacheStore(
        inputs, layer_cache, config, cache_store, /*cache_model_id=*/0, /*cp_rank=*/0, /*cp_size=*/1, nullptr));

    ASSERT_EQ(cache_store->records.size(), 1u) << "Expected exactly one store() call for the single request";
    EXPECT_EQ(cache_store->records[0].block_count, 1u) << "LINEAR policy should store only its active tail block";
    ASSERT_EQ(cache_store->records[0].block_keys.size(), 1u);
    EXPECT_EQ(cache_store->records[0].block_keys[0].rfind("kv_", 0), 0u)
        << "Hybrid cache-store must write opaque kv_ keys even when use_opaque_kv_cache_store=false";
}

TEST_F(ExecOpsTest, testWriteCacheStoreTag_FullGroup) {
    auto                    cache_store = std::make_shared<MockCacheStore>();
    auto                    inputs      = makePyCacheStoreInputs(/*tokens_per_block=*/2, /*block_num=*/3);
    auto                    config      = makeCacheConfig(/*tokens_per_block=*/2,
                                  /*physical_kv_stride=*/64,
                                  /*physical_scale_stride=*/0,
                                  /*block_num=*/3,
                                  "full",
                                  /*layer_id=*/1,
                                  defaultCacheGroupPolicy(CacheGroupType::FULL),
                                  /*add_dummy_group=*/true);
    torch_ext::LayerKVCache layer_cache;
    layer_cache.kv_cache_base      = torch::zeros({3, 64}, torch::kUInt8);
    layer_cache.seq_size_per_block = 2;
    layer_cache.layer_id           = 1;
    layer_cache.tag                = "full";

    ASSERT_NO_THROW(runtimeWriteCacheStore(
        inputs, layer_cache, config, cache_store, /*cache_model_id=*/0, /*cp_rank=*/0, /*cp_size=*/1, nullptr));

    ASSERT_EQ(cache_store->records.size(), 1u);
    EXPECT_EQ(cache_store->records[0].block_count, 3u)
        << "Layer 1 -> group 1 (FULL): all 3 blocks should be stored as opaque kv entries";
    EXPECT_EQ(countKeyPrefix(cache_store->records[0].block_keys, "kv_"), 3u);
}

TEST_F(ExecOpsTest, testWriteCacheStoreSameLayerRoutesByTag) {
    constexpr size_t tokens_per_block = 2;
    constexpr size_t block_num        = 3;
    constexpr size_t block_stride     = 64;

    CacheConfig config;
    config.layer_num                 = 1;
    config.layer_all_num             = 1;
    config.block_num                 = block_num;
    config.seq_size_per_block        = tokens_per_block;
    config.kernel_seq_size_per_block = tokens_per_block;
    config.kv_block_stride_bytes     = block_stride;

    auto make_group = [](const std::string& tag, CacheGroupType type) {
        GroupBase group;
        group.tag                       = tag;
        group.spec                      = makeTestSpec(tag, tokens_per_block, false);
        group.policy                    = defaultCacheGroupPolicy(type);
        group.layer_ids                 = {0};
        group.block_num                 = block_num;
        group.seq_size_per_block        = tokens_per_block;
        group.kernel_seq_size_per_block = tokens_per_block;
        group.kv_block_stride_bytes     = block_stride;
        return group;
    };
    config.setTopology({make_group("linear", CacheGroupType::LINEAR), make_group("full", CacheGroupType::FULL)},
                       {{0, {"linear", "full"}}});

    auto                    inputs = makePyCacheStoreInputs(tokens_per_block, block_num);
    torch_ext::LayerKVCache layer_cache;
    layer_cache.kv_cache_base      = torch::zeros({3, 64}, torch::kUInt8);
    layer_cache.seq_size_per_block = tokens_per_block;
    layer_cache.layer_id           = 0;

    auto linear_store = std::make_shared<MockCacheStore>();
    layer_cache.tag   = "linear";
    ASSERT_NO_THROW(runtimeWriteCacheStore(
        inputs, layer_cache, config, linear_store, /*cache_model_id=*/0, /*cp_rank=*/0, /*cp_size=*/1, nullptr));
    ASSERT_EQ(linear_store->records.size(), 1u);
    EXPECT_EQ(linear_store->records[0].block_count, 1u);

    auto full_store = std::make_shared<MockCacheStore>();
    layer_cache.tag = "full";
    ASSERT_NO_THROW(runtimeWriteCacheStore(
        inputs, layer_cache, config, full_store, /*cache_model_id=*/0, /*cp_rank=*/0, /*cp_size=*/1, nullptr));
    ASSERT_EQ(full_store->records.size(), 1u);
    EXPECT_EQ(full_store->records[0].block_count, block_num);
}

TEST_F(ExecOpsTest, testWriteCacheStoreRejectsNonLocalBlockTable) {
    auto cache_store               = std::make_shared<MockCacheStore>();
    auto inputs                    = makePyCacheStoreInputs(/*tokens_per_block=*/2, /*block_num=*/3);
    inputs.host_kv_cache_offset    = torch::ones({2, 1, 3}, torch::kInt32);
    auto                    config = makeCacheConfig(/*tokens_per_block=*/2, /*physical_kv_stride=*/64, 0, 3);
    torch_ext::LayerKVCache layer_cache;
    layer_cache.kv_cache_base      = torch::zeros({3, 64}, torch::kUInt8);
    layer_cache.seq_size_per_block = 2;
    layer_cache.layer_id           = 0;
    layer_cache.tag                = "default";

    EXPECT_ANY_THROW(runtimeWriteCacheStore(
        inputs, layer_cache, config, cache_store, /*cache_model_id=*/0, /*cp_rank=*/0, /*cp_size=*/1, nullptr));
}

TEST_F(ExecOpsTest, testWriteCacheStoreTag_LocalOffset) {
    auto                    cache_store = std::make_shared<MockCacheStore>();
    auto                    inputs      = makePyCacheStoreInputs(/*tokens_per_block=*/2, /*block_num=*/3);
    auto                    config      = makeCacheConfig(/*tokens_per_block=*/2,
                                  /*physical_kv_stride=*/64,
                                  /*physical_scale_stride=*/0,
                                  /*block_num=*/3,
                                  "full",
                                  /*layer_id=*/1,
                                  defaultCacheGroupPolicy(CacheGroupType::FULL),
                                  /*add_dummy_group=*/true);
    torch_ext::LayerKVCache layer_cache;
    layer_cache.kv_cache_base      = torch::zeros({3, 64}, torch::kUInt8);
    layer_cache.seq_size_per_block = 2;
    layer_cache.layer_id           = 1;
    layer_cache.tag                = "full";

    ASSERT_NO_THROW(runtimeWriteCacheStore(
        inputs, layer_cache, config, cache_store, /*cache_model_id=*/0, /*cp_rank=*/0, /*cp_size=*/1, nullptr));

    ASSERT_EQ(cache_store->records.size(), 1u);
    EXPECT_EQ(cache_store->records[0].block_count, 3u)
        << "2-D path, layer 1 -> group 1 (FULL): all 3 blocks should be stored as opaque kv entries";
    EXPECT_EQ(countKeyPrefix(cache_store->records[0].block_keys, "kv_"), 3u);
}

TEST_F(ExecOpsTest, testWriteCacheStoreSkipsNullPhysicalBlocks) {
    auto cache_store               = std::make_shared<MockCacheStore>();
    auto inputs                    = makePyCacheStoreInputs(/*tokens_per_block=*/2, /*block_num=*/3);
    inputs.host_kv_cache_offset    = torch::tensor({{0, -1, 1}}, torch::kInt32);
    auto                    config = makeCacheConfig(/*tokens_per_block=*/2,
                                  /*physical_kv_stride=*/64,
                                  /*physical_scale_stride=*/0,
                                  /*block_num=*/2,
                                  "default",
                                  /*layer_id=*/0,
                                  defaultCacheGroupPolicy(CacheGroupType::FULL),
                                  /*add_dummy_group=*/false,
                                  /*mla_cache=*/true);
    torch_ext::LayerKVCache layer_cache;
    layer_cache.kv_cache_base      = torch::zeros({2, 64}, torch::kUInt8);
    layer_cache.seq_size_per_block = 2;
    layer_cache.layer_id           = 0;
    layer_cache.tag                = "default";

    ASSERT_NO_THROW(runtimeWriteCacheStore(
        inputs, layer_cache, config, cache_store, /*cache_model_id=*/0, /*cp_rank=*/0, /*cp_size=*/1, nullptr));

    ASSERT_EQ(cache_store->records.size(), 1u);
    EXPECT_EQ(cache_store->records[0].block_count, 2u);
    EXPECT_NE(cache_store->records[0].blocks.find("kv_" + cacheKeyAt(inputs, 0, 0)),
              cache_store->records[0].blocks.end());
    EXPECT_EQ(cache_store->records[0].blocks.find("kv_" + cacheKeyAt(inputs, 1, 0)),
              cache_store->records[0].blocks.end());
    EXPECT_NE(cache_store->records[0].blocks.find("kv_" + cacheKeyAt(inputs, 2, 0)),
              cache_store->records[0].blocks.end());
}

TEST_F(ExecOpsTest, testWriteCacheStoreUsesDecoderOffsetForContextBlockRow) {
    auto cache_store               = std::make_shared<MockCacheStore>();
    auto inputs                    = makePyCacheStoreInputs(/*tokens_per_block=*/2, /*block_num=*/2);
    inputs.input_lengths_host      = torch::tensor({1, 4}, torch::kInt32);
    inputs.host_kv_cache_offset    = torch::tensor({{1, 1}, {0, 1}}, torch::kInt32);
    auto                    config = makeCacheConfig(/*tokens_per_block=*/2,
                                  /*physical_kv_stride=*/64,
                                  /*physical_scale_stride=*/0,
                                  /*block_num=*/2,
                                  "default",
                                  /*layer_id=*/0,
                                  defaultCacheGroupPolicy(CacheGroupType::FULL),
                                  /*add_dummy_group=*/false,
                                  /*mla_cache=*/true);
    torch_ext::LayerKVCache layer_cache;
    layer_cache.kv_cache_base      = torch::zeros({2, 64}, torch::kUInt8);
    layer_cache.seq_size_per_block = 2;
    layer_cache.layer_id           = 0;
    layer_cache.tag                = "default";

    ASSERT_NO_THROW(runtimeWriteCacheStore(
        inputs, layer_cache, config, cache_store, /*cache_model_id=*/0, /*cp_rank=*/0, /*cp_size=*/1, nullptr));

    ASSERT_EQ(cache_store->records.size(), 1u);
    const auto second_key = "kv_" + cacheKeyAt(inputs, 1, 0);
    ASSERT_NE(cache_store->records[0].blocks.find(second_key), cache_store->records[0].blocks.end());
    EXPECT_EQ(reinterpret_cast<uintptr_t>(cache_store->records[0].blocks.at(second_key).addr),
              reinterpret_cast<uintptr_t>(layer_cache.kv_cache_base.data_ptr()) + 64);
}

TEST_F(ExecOpsTest, testWriteCacheStoreUsesCacheModelIdInKeyNamespace) {
    constexpr size_t        cache_model_id = 23;
    auto                    cache_store    = std::make_shared<MockCacheStore>();
    auto                    inputs         = makePyCacheStoreInputs(/*tokens_per_block=*/2, /*block_num=*/1);
    auto                    config         = makeCacheConfig(/*tokens_per_block=*/2,
                                  /*physical_kv_stride=*/64,
                                  0,
                                  1,
                                  "default",
                                  0,
                                  defaultCacheGroupPolicy(CacheGroupType::FULL),
                                  false,
                                  true);
    torch_ext::LayerKVCache layer_cache;
    layer_cache.kv_cache_base      = torch::zeros({1, 64}, torch::kUInt8);
    layer_cache.seq_size_per_block = 2;
    layer_cache.layer_id           = 0;
    layer_cache.tag                = "default";

    ASSERT_NO_THROW(runtimeWriteCacheStore(
        inputs, layer_cache, config, cache_store, cache_model_id, /*cp_rank=*/0, /*cp_size=*/1, nullptr));

    ASSERT_EQ(cache_store->records.size(), 1u);
    EXPECT_NE(cache_store->records[0].blocks.find("kv_" + cacheKeyAt(inputs, 0, 0, "default", cache_model_id)),
              cache_store->records[0].blocks.end());
}

TEST_F(ExecOpsTest, testWriteCacheStoreReadsNonContiguousHostMetadata) {
    auto cache_store             = std::make_shared<MockCacheStore>();
    auto inputs                  = makePyCacheStoreInputs(/*tokens_per_block=*/2, /*block_num=*/1);
    inputs.input_lengths_host    = torch::tensor({2, 2}, torch::kInt32);
    inputs.prefix_lengths_host   = torch::tensor({0, 0}, torch::kInt32);
    inputs.host_kv_cache_offset  = torch::tensor({{0}, {1}}, torch::kInt32);
    inputs.request_id            = torch::tensor({int64_t(42), int64_t(43)}, torch::kInt64);
    inputs.request_pd_separation = torch::tensor({true, true}, torch::kBool);
    inputs.cache_keys            = torch::tensor({{int64_t(100)}, {int64_t(200)}}, torch::kInt64);

    inputs.host_kv_cache_offset  = makeNonContiguousMatrix(inputs.host_kv_cache_offset, /*gap_value=*/99);
    inputs.input_lengths_host    = makeNonContiguousVector(inputs.input_lengths_host, /*gap_value=*/0);
    inputs.prefix_lengths_host   = makeNonContiguousVector(inputs.prefix_lengths_host, /*gap_value=*/1);
    inputs.request_id            = makeNonContiguousVector(inputs.request_id, /*gap_value=*/999);
    inputs.request_pd_separation = makeNonContiguousVector(inputs.request_pd_separation, /*gap_value=*/false);
    inputs.cache_keys            = makeNonContiguousMatrix(inputs.cache_keys, /*gap_value=*/999);
    ASSERT_FALSE(inputs.host_kv_cache_offset.is_contiguous());
    ASSERT_FALSE(inputs.input_lengths_host.is_contiguous());
    ASSERT_FALSE(inputs.prefix_lengths_host.is_contiguous());
    ASSERT_FALSE(inputs.request_id.is_contiguous());
    ASSERT_FALSE(inputs.request_pd_separation.is_contiguous());
    ASSERT_FALSE(inputs.cache_keys.is_contiguous());

    auto                    config = makeCacheConfig(/*tokens_per_block=*/2,
                                  /*physical_kv_stride=*/64,
                                  /*physical_scale_stride=*/0,
                                  /*block_num=*/2,
                                  "default",
                                  /*layer_id=*/0,
                                  defaultCacheGroupPolicy(CacheGroupType::FULL),
                                  /*add_dummy_group=*/false,
                                  /*mla_cache=*/true);
    torch_ext::LayerKVCache layer_cache;
    layer_cache.kv_cache_base      = torch::zeros({2, 64}, torch::kUInt8);
    layer_cache.seq_size_per_block = 2;
    layer_cache.layer_id           = 0;
    layer_cache.tag                = "default";

    ASSERT_NO_THROW(runtimeWriteCacheStore(
        inputs, layer_cache, config, cache_store, /*cache_model_id=*/0, /*cp_rank=*/0, /*cp_size=*/1, nullptr));

    ASSERT_EQ(cache_store->records.size(), 2u);
    const auto first_key  = "kv_" + makeCacheKey(/*model_id=*/0, "100", /*layer_id=*/0, "default");
    const auto second_key = "kv_" + makeCacheKey(/*model_id=*/0, "200", /*layer_id=*/0, "default");
    EXPECT_NE(cache_store->records[0].blocks.find(first_key), cache_store->records[0].blocks.end());
    EXPECT_NE(cache_store->records[1].blocks.find(second_key), cache_store->records[1].blocks.end());
}

#if USING_CUDA
TEST_F(ExecOpsTest, testSampleFromProbsHandlesSingleAndMultiBlockVocab) {
    auto forced_probs  = torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto forced_output = execSampleFromProbs(forced_probs);
    EXPECT_TRUE(torch::equal(forced_output.cpu(), torch::arange(4, torch::kInt32)));

    auto multi_block_probs     = torch::zeros({2, 2051}, forced_probs.options());
    multi_block_probs[0][2048] = 1.0f;
    multi_block_probs[1][1024] = 1.0f;
    auto multi_block_output    = execSampleFromProbs(multi_block_probs);
    EXPECT_TRUE(torch::equal(multi_block_output.cpu(), torch::tensor({2048, 1024}, torch::kInt32)));
}

TEST_F(ExecOpsTest, testSampleFromProbsUsesDefaultGenerator) {
    auto probabilities =
        torch::full({64, 16}, 1.0f / 16.0f, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto generator = at::cuda::detail::getDefaultCUDAGenerator();
    generator.set_current_seed(17);
    auto first = execSampleFromProbs(probabilities);
    generator.set_current_seed(17);
    auto second = execSampleFromProbs(probabilities);
    EXPECT_TRUE(torch::equal(first, second));
}

TEST_F(ExecOpsTest, testSampleFromProbsMatchesDistribution) {
    constexpr int64_t distribution_rows = 2048;
    auto              generator         = at::cuda::detail::getDefaultCUDAGenerator();
    generator.set_current_seed(23);
    auto distribution_probs = torch::softmax(torch::tensor({1.0f, 0.0f, -1.0f}, torch::kFloat32), -1)
                                  .repeat({distribution_rows, 1})
                                  .to(torch::kCUDA);
    auto distribution_output = execSampleFromProbs(distribution_probs);
    auto frequencies         = torch::bincount(distribution_output.to(torch::kLong), {}, 3).to(torch::kFloat32)
                       / static_cast<float>(distribution_rows);
    auto expected_frequencies = torch::softmax(torch::tensor({1.0f, 0.0f, -1.0f}), -1);
    EXPECT_TRUE(torch::allclose(frequencies.cpu(), expected_frequencies, 0.05, 0.02)) << frequencies.cpu();
}
#endif

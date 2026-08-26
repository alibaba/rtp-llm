#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <torch/torch.h>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyStrategy.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"

namespace rtp_llm {
namespace {

using block_transfer_engine_test::TempDirGuard;
using block_transfer_engine_test::DirectAlignmentDiskBlockIO;
using block_transfer_engine_test::StatusDiskBlockIO;
using block_transfer_engine_test::expectStatus;
using block_transfer_engine_test::makeDescriptor;
using block_transfer_engine_test::makeDiskPool;
using block_transfer_engine_test::makeHostPool;
using block_transfer_engine_test::makeTestGroupBase;
using block_transfer_engine_test::makeTestGroupSet;
using block_transfer_engine_test::makeTestTopology;
using block_transfer_engine_test::poolMalloc;
using block_transfer_engine_test::releasePoolBlock;
using block_transfer_engine_test::submitSucceeded;

struct DeviceLayerBufferSpec {
    size_t kv_bytes{0};
    size_t scale_bytes{0};
};

static DeviceBlockPoolPtr
makeDevicePool(const std::vector<DeviceLayerBufferSpec>& specs, size_t usable_count, const std::string& pool_name) {
    const auto physical_block_count = usable_count + 1;

    auto config                     = std::make_shared<DeviceBlockPoolConfig>();
    config->pool_type               = BlockPoolType::DEVICE;
    config->pool_name               = pool_name;
    config->physical_block_count    = physical_block_count;
    config->use_cuda_malloc_backing = false;

    size_t offset = 0;
    for (const auto& spec : specs) {
        MemoryLayoutConfig layout;
        layout.layer_num                = 1;
        layout.block_num                = static_cast<uint32_t>(physical_block_count);
        layout.dtype                    = TYPE_INT8;
        layout.kv_cache_offset_bytes    = offset;
        layout.kv_block_stride_bytes    = spec.kv_bytes;
        layout.kv_block_pool_size_bytes = physical_block_count * spec.kv_bytes;
        layout.block_stride_bytes       = spec.kv_bytes + spec.scale_bytes;
        layout.total_size_bytes         = layout.kv_block_pool_size_bytes;
        offset += layout.kv_block_pool_size_bytes;

        if (spec.scale_bytes > 0) {
            layout.enable_kv_scale          = true;
            layout.kv_scale_offset_bytes    = offset;
            layout.kv_scale_stride_bytes    = spec.scale_bytes;
            layout.kv_scale_pool_size_bytes = physical_block_count * spec.scale_bytes;
            layout.total_size_bytes += layout.kv_scale_pool_size_bytes;
            offset += layout.kv_scale_pool_size_bytes;
        }

        layout.local_head_num_kv          = 1;
        layout.seq_size_per_block         = 1;
        layout.kernel_blocks_per_kv_block = 1;
        config->memory_layouts.push_back(layout);
    }
    config->total_size_bytes = offset;

    auto pool = std::make_shared<DeviceBlockPool>(config);
    RTP_LLM_CHECK(pool->init());
    return pool;
}

// DeviceBlockPool backing is always CUDA, so the byte view is unconditionally a CUDA tensor.
static torch::Tensor makePoolByteTensor(void* addr, size_t bytes) {
    return torch::from_blob(
        addr, {static_cast<int64_t>(bytes)}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
}

static void fillDeviceLayer(const DeviceBlockPoolPtr&   pool,
                            int                         layer_id,
                            BlockIdxType                block,
                            const std::vector<uint8_t>& patterns) {
    auto buffers = pool->convertIndexToBuffer(layer_id, block);
    ASSERT_EQ(buffers.size(), patterns.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
        makePoolByteTensor(buffers[i].addr, buffers[i].size_bytes).fill_(patterns[i]);
    }
    if (pool->where() == MemoryType::MEMORY_GPU) {
        cudaDeviceSynchronize();
    }
}

static void fillDeviceLayerSequential(const DeviceBlockPoolPtr& pool, int layer_id, BlockIdxType block) {
    auto buffers = pool->convertIndexToBuffer(layer_id, block);
    for (const auto& buffer : buffers) {
        std::vector<uint8_t> cpu_data(buffer.size_bytes);
        for (size_t i = 0; i < buffer.size_bytes; ++i) {
            cpu_data[i] = static_cast<uint8_t>(i & 0xFF);
        }
        auto cpu_tensor =
            torch::from_blob(cpu_data.data(), {static_cast<int64_t>(buffer.size_bytes)}, torch::kUInt8).clone();
        makePoolByteTensor(buffer.addr, buffer.size_bytes).copy_(cpu_tensor);
    }
    if (pool->where() == MemoryType::MEMORY_GPU) {
        cudaDeviceSynchronize();
    }
}

static std::vector<uint8_t> readDeviceLayer(const DeviceBlockPoolPtr& pool, int layer_id, BlockIdxType block) {
    if (pool->where() == MemoryType::MEMORY_GPU) {
        cudaDeviceSynchronize();
    }

    std::vector<uint8_t> out;
    auto                 buffers = pool->convertIndexToBuffer(layer_id, block);
    for (const auto& buffer : buffers) {
        auto  tensor = makePoolByteTensor(buffer.addr, buffer.size_bytes);
        auto  cpu    = tensor.cpu();
        auto* ptr    = cpu.data_ptr<uint8_t>();
        out.insert(out.end(), ptr, ptr + buffer.size_bytes);
    }
    return out;
}

static GroupBase makeGroupBase(std::vector<int> layer_ids, size_t kv_bytes, size_t scale_bytes = 0) {
    auto policy                = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.enable_prefix_reuse = true;
    return makeTestGroupBase(std::move(policy), std::move(layer_ids), kv_bytes, scale_bytes);
}

static GroupBase
makeGroupBase(CacheGroupType group_type, std::vector<int> layer_ids, size_t kv_bytes, size_t scale_bytes = 0) {
    auto policy                = defaultCacheGroupPolicy(group_type);
    policy.enable_prefix_reuse = true;
    if (group_type == CacheGroupType::SWA) {
        policy.sliding_window_size = 2;
    }
    return makeTestGroupBase(std::move(policy), std::move(layer_ids), kv_bytes, scale_bytes);
}

static GroupSetPtr makeDeviceHostGroup(size_t                                  group_set_id,
                                       std::vector<DeviceBlockPoolPtr>         device_pools,
                                       std::shared_ptr<HostBlockPool>          host_pool,
                                       std::vector<GroupBase>                  groups,
                                       std::shared_ptr<BlockTreeDiskBlockPool> disk_pool = nullptr) {
    auto                topology = makeTestTopology(std::move(groups));
    std::vector<size_t> group_ids(device_pools.size());
    std::iota(group_ids.begin(), group_ids.end(), 0);
    auto group = makeTestGroupSet(group_set_id,
                                  std::move(topology),
                                  std::move(group_ids),
                                  std::move(device_pools),
                                  std::move(host_pool),
                                  std::move(disk_pool));
    return group;
}

static std::shared_ptr<PerRankBlockTransferEngine> makeEngine(std::vector<GroupSetPtr> groups,
                                                              DeviceHostCopyOptions    options = {}) {
    return std::make_shared<PerRankBlockTransferEngine>(std::move(groups), std::move(options));
}

struct StrategyCounters {
    int attempts{0};
    int done{0};
    int not_applicable{0};
    int failed{0};
};

class RecordingStrategy: public DeviceHostCopyStrategy {
public:
    RecordingStrategy(std::unique_ptr<DeviceHostCopyStrategy> delegate, StrategyCounters* counters):
        delegate_(std::move(delegate)), counters_(counters) {}

    StrategyResult tryExecute(const DeviceHostCopyPlan& plan, const DeviceHostCopyOptions& options) override {
        ++counters_->attempts;
        auto result = delegate_->tryExecute(plan, options);
        switch (result.status) {
            case StrategyStatus::DONE:
                ++counters_->done;
                break;
            case StrategyStatus::NOT_APPLICABLE:
                ++counters_->not_applicable;
                break;
            case StrategyStatus::FAILED:
                ++counters_->failed;
                break;
        }
        return result;
    }

private:
    std::unique_ptr<DeviceHostCopyStrategy> delegate_;
    StrategyCounters*                       counters_;
};

class FailingStrategy: public DeviceHostCopyStrategy {
public:
    StrategyResult tryExecute(const DeviceHostCopyPlan&, const DeviceHostCopyOptions&) override {
        return StrategyResult::failed(TransferStatus::INVALID_ARGS);
    }
};

class BlockingStrategy: public DeviceHostCopyStrategy {
public:
    StrategyResult tryExecute(const DeviceHostCopyPlan&, const DeviceHostCopyOptions&) override {
        std::unique_lock<std::mutex> lock(mutex_);
        ++entered_count_;
        cv_.notify_all();
        cv_.wait(lock, [this] { return released_; });
        return StrategyResult::done();
    }

    bool waitUntilEntered(size_t count, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, timeout, [this, count] { return entered_count_ >= count; });
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        cv_.notify_all();
    }

private:
    std::mutex              mutex_;
    std::condition_variable cv_;
    size_t                  entered_count_{0};
    bool                    released_{false};
};

static void installStrategyRecorders(DeviceHostTransferExecutor& executor, std::array<StrategyCounters, 3>& counters) {
    RTP_LLM_CHECK(executor.strategies_.size() == counters.size());
    for (size_t i = 0; i < counters.size(); ++i) {
        executor.strategies_[i] = std::make_unique<RecordingStrategy>(std::move(executor.strategies_[i]), &counters[i]);
    }
}

static bool expectCudaBatchStrategyDone() {
    int        runtime_version       = 0;
    const auto runtime_version_error = cudaRuntimeGetVersion(&runtime_version);
    return CUDART_VERSION >= 12080 && runtime_version_error == cudaSuccess && runtime_version >= 12080
           && (CUDART_VERSION >= 13000) == (runtime_version >= 13000);
}

TEST(DeviceHostTransferExecutorConfigTest, PrefersCudaBatchThenStagedSmThenGeneric) {
    DeviceHostTransferExecutor executor;
    EXPECT_TRUE(executor.options_.cuda_batch_copy_enabled);
    EXPECT_TRUE(executor.options_.staged_sm_copy_enabled);
    ASSERT_EQ(executor.strategies_.size(), 3u);
    EXPECT_NE(dynamic_cast<CudaBatchDeviceHostCopyStrategy*>(executor.strategies_[0].get()), nullptr);
    EXPECT_NE(dynamic_cast<StagedSmDeviceHostCopyStrategy*>(executor.strategies_[1].get()), nullptr);
    EXPECT_NE(dynamic_cast<GenericMultiCopyDeviceHostCopyStrategy*>(executor.strategies_[2].get()), nullptr);
}

// ---- PerRankBlockTransferEngine submit() tests (real CUDA) ----

class PerRankBlockTransferEngineTest: public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";

        layer_bytes_     = {100, 100, 100};
        host_block_size_ = 300;

        // Create host pool — 10 usable blocks
        host_pool_ = makeHostPool(host_block_size_, 10, true);

        device_pool_  = makeDevicePool({{100, 0}, {200, 0}, {150, 0}}, 10, "per_rank_transfer_engine_device");
        device_block_ = poolMalloc(*device_pool_);
        ASSERT_NE(device_block_, NULL_BLOCK_IDX);
        device_blocks_ = {device_block_};

        group_set_ = makeDeviceHostGroup(0, {device_pool_}, host_pool_, {makeGroupBase({0, 1, 2}, 100)});
        ASSERT_EQ(group_set_->payloadBytes(), host_block_size_);
        per_rank_transfer_engine_ = makeEngine({group_set_});
    }

    std::vector<size_t>                         layer_bytes_;
    size_t                                      host_block_size_;
    std::shared_ptr<HostBlockPool>              host_pool_;
    std::shared_ptr<PerRankBlockTransferEngine> per_rank_transfer_engine_;
    DeviceBlockPoolPtr                          device_pool_;
    BlockIdxType                                device_block_;
    std::vector<BlockIdxType>                   device_blocks_;
    GroupSetPtr                                 group_set_;
};

TEST_F(PerRankBlockTransferEngineTest, SubmitDeviceHostRoundTripPreservesLayout) {
    fillDeviceLayerSequential(device_pool_, 0, device_block_);
    fillDeviceLayer(device_pool_, 1, device_block_, {0x5A});
    fillDeviceLayerSequential(device_pool_, 2, device_block_);

    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    auto d2h_desc = makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block);
    ASSERT_TRUE(submitSucceeded(per_rank_transfer_engine_, d2h_desc));

    const uint8_t* host_data = static_cast<const uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    for (size_t i = 0; i < layer_bytes_[0]; ++i)
        EXPECT_EQ(host_data[i], static_cast<uint8_t>(i & 0xFF));
    for (size_t i = 0; i < layer_bytes_[1]; ++i)
        EXPECT_EQ(host_data[layer_bytes_[0] + i], 0x5A);
    const size_t off2 = layer_bytes_[0] + layer_bytes_[1];
    for (size_t i = 0; i < layer_bytes_[2]; ++i)
        EXPECT_EQ(host_data[off2 + i], static_cast<uint8_t>(i & 0xFF));

    fillDeviceLayer(device_pool_, 0, device_block_, {0x00});
    fillDeviceLayer(device_pool_, 1, device_block_, {0x00});
    fillDeviceLayer(device_pool_, 2, device_block_, {0x00});
    auto h2d_desc = makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, host_block);
    ASSERT_TRUE(submitSucceeded(per_rank_transfer_engine_, h2d_desc));

    auto d0 = readDeviceLayer(device_pool_, 0, device_block_);
    auto d1 = readDeviceLayer(device_pool_, 1, device_block_);
    auto d2 = readDeviceLayer(device_pool_, 2, device_block_);
    for (size_t i = 0; i < layer_bytes_[0]; ++i)
        EXPECT_EQ(d0[i], static_cast<uint8_t>(i & 0xFF));
    for (size_t i = 0; i < layer_bytes_[1]; ++i)
        EXPECT_EQ(d1[i], 0x5A);
    for (size_t i = layer_bytes_[1]; i < d1.size(); ++i)
        EXPECT_EQ(d1[i], 0x00);
    for (size_t i = 0; i < layer_bytes_[2]; ++i)
        EXPECT_EQ(d2[i], static_cast<uint8_t>(i & 0xFF));
    for (size_t i = layer_bytes_[2]; i < d2.size(); ++i)
        EXPECT_EQ(d2[i], 0x00);

    releasePoolBlock(*host_pool_, host_block);
}

TEST_F(PerRankBlockTransferEngineTest, ExecutorDerivesDirectionFromDescriptorTargetTier) {
    DeviceHostTransferExecutor executor;
    fillDeviceLayer(device_pool_, 0, device_block_, {0xA5});

    const BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    const HostBlockBuffer host_buffer = host_pool_->blockBuffer(host_block);
    const HostBufferView  host{host_buffer.addr, host_buffer.payload_bytes, host_buffer.stride_bytes};

    const auto d2h_desc = makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block);
    ASSERT_EQ(executor.execute({host}, {d2h_desc}, {group_set_.get()}), TransferStatus::OK);
    const auto* host_data = static_cast<const uint8_t*>(host.base);
    for (size_t i = 0; i < layer_bytes_[0]; ++i) {
        EXPECT_EQ(host_data[i], 0xA5);
    }

    fillDeviceLayer(device_pool_, 0, device_block_, {0x00});
    const auto h2d_desc = makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, host_block);
    ASSERT_EQ(executor.execute({host}, {h2d_desc}, {group_set_.get()}), TransferStatus::OK);
    const auto device_data = readDeviceLayer(device_pool_, 0, device_block_);
    for (size_t i = 0; i < layer_bytes_[0]; ++i) {
        EXPECT_EQ(device_data[i], 0xA5);
    }

    releasePoolBlock(*host_pool_, host_block);
}

TEST_F(PerRankBlockTransferEngineTest, SharedDevicePoolGroupsIsolateByBlockId) {
    auto shared_pool = makeDevicePool({{64, 0}, {32, 0}}, 4, "per_rank_transfer_engine_shared_pool");
    auto host_pool   = makeHostPool(128, 4, true);
    auto group       = makeDeviceHostGroup(
        0, {shared_pool, shared_pool}, host_pool, {makeGroupBase({0, 1}, 32), makeGroupBase({0, 1}, 32)});
    auto engine = makeEngine({group});

    const BlockIdxType block_a = poolMalloc(*shared_pool);
    const BlockIdxType block_b = poolMalloc(*shared_pool);
    ASSERT_NE(block_a, NULL_BLOCK_IDX);
    ASSERT_NE(block_b, NULL_BLOCK_IDX);
    ASSERT_NE(block_a, block_b);

    fillDeviceLayer(shared_pool, 0, block_a, {0xA0});
    fillDeviceLayer(shared_pool, 1, block_a, {0xA1});
    fillDeviceLayer(shared_pool, 0, block_b, {0xB0});
    fillDeviceLayer(shared_pool, 1, block_b, {0xB1});

    const BlockIdxType host_block = poolMalloc(*host_pool);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    ASSERT_TRUE(submitSucceeded(engine, makeDescriptor(Tier::DEVICE, Tier::HOST, {block_a, block_b}, host_block)));

    const uint8_t* host_data = static_cast<const uint8_t*>(host_pool->blockBuffer(host_block).addr);
    for (size_t i = 0; i < 32; ++i)
        EXPECT_EQ(host_data[i], 0xA0);
    for (size_t i = 0; i < 32; ++i)
        EXPECT_EQ(host_data[32 + i], 0xA1);
    for (size_t i = 0; i < 32; ++i)
        EXPECT_EQ(host_data[64 + i], 0xB0);
    for (size_t i = 0; i < 32; ++i)
        EXPECT_EQ(host_data[96 + i], 0xB1);

    fillDeviceLayer(shared_pool, 0, block_a, {0x00});
    fillDeviceLayer(shared_pool, 1, block_a, {0x00});
    fillDeviceLayer(shared_pool, 0, block_b, {0x00});
    fillDeviceLayer(shared_pool, 1, block_b, {0x00});
    ASSERT_TRUE(submitSucceeded(engine, makeDescriptor(Tier::HOST, Tier::DEVICE, {block_a, block_b}, host_block)));

    const auto a0 = readDeviceLayer(shared_pool, 0, block_a);
    for (size_t i = 0; i < 32; ++i)
        EXPECT_EQ(a0[i], 0xA0);
    for (size_t i = 32; i < a0.size(); ++i)
        EXPECT_EQ(a0[i], 0x00);
    for (auto byte : readDeviceLayer(shared_pool, 1, block_a))
        EXPECT_EQ(byte, 0xA1);
    const auto b0 = readDeviceLayer(shared_pool, 0, block_b);
    for (size_t i = 0; i < 32; ++i)
        EXPECT_EQ(b0[i], 0xB0);
    for (size_t i = 32; i < b0.size(); ++i)
        EXPECT_EQ(b0[i], 0x00);
    for (auto byte : readDeviceLayer(shared_pool, 1, block_b))
        EXPECT_EQ(byte, 0xB1);

    releasePoolBlock(*host_pool, host_block);
    shared_pool->incRef(block_a);
    shared_pool->decRef(block_a);
    shared_pool->incRef(block_b);
    shared_pool->decRef(block_b);
}

TEST_F(PerRankBlockTransferEngineTest, SubmitAcceptsValidUnallocatedHostBlock) {
    constexpr BlockIdxType unallocated_host_block = 1;
    expectStatus(per_rank_transfer_engine_,
                 makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, unallocated_host_block),
                 TransferStatus::OK);
    expectStatus(per_rank_transfer_engine_,
                 makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, unallocated_host_block),
                 TransferStatus::OK);
}

TEST_F(PerRankBlockTransferEngineTest, SubmitAcceptsValidUnallocatedDeviceBlock) {
    // Worker transfers may use a valid logical block ID without local allocator ownership.
    BlockIdxType freed_device_block = poolMalloc(*device_pool_);
    ASSERT_NE(freed_device_block, NULL_BLOCK_IDX);
    device_pool_->incRef(freed_device_block);
    device_pool_->decRef(freed_device_block);
    std::vector<BlockIdxType> unallocated_device_blocks = {freed_device_block};

    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    expectStatus(per_rank_transfer_engine_,
                 makeDescriptor(Tier::DEVICE, Tier::HOST, unallocated_device_blocks, host_block),
                 TransferStatus::OK);
    expectStatus(per_rank_transfer_engine_,
                 makeDescriptor(Tier::HOST, Tier::DEVICE, unallocated_device_blocks, host_block),
                 TransferStatus::OK);

    releasePoolBlock(*host_pool_, host_block);
}

TEST_F(PerRankBlockTransferEngineTest, SubmitReportsFinalStatusAfterWait) {
    fillDeviceLayer(device_pool_, 0, device_block_, {0xAA});
    fillDeviceLayer(device_pool_, 1, device_block_, {0xBB});
    fillDeviceLayer(device_pool_, 2, device_block_, {0xCC});

    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    const auto desc = makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block);

    auto context = per_rank_transfer_engine_->submit({desc});
    ASSERT_NE(context, nullptr);
    context->waitDone();
    EXPECT_TRUE(context->success());
    EXPECT_TRUE(context->errorInfo().ok());

    auto& strategies = per_rank_transfer_engine_->device_host_executor_->strategies_;
    strategies.clear();
    strategies.push_back(std::make_unique<FailingStrategy>());
    auto failed_context = per_rank_transfer_engine_->submit({desc});
    ASSERT_NE(failed_context, nullptr);
    failed_context->waitDone();
    EXPECT_FALSE(failed_context->success());
    EXPECT_EQ(failed_context->errorInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_FALSE(failed_context->errorInfo().ToString().empty());

    releasePoolBlock(*host_pool_, host_block);
}

TEST_F(PerRankBlockTransferEngineTest, SubmitHostToDeviceIndependentDescriptors) {
    BlockIdxType second_device_block = poolMalloc(*device_pool_);
    ASSERT_NE(second_device_block, NULL_BLOCK_IDX);
    std::vector<BlockIdxType> second_device_blocks = {second_device_block};

    BlockIdxType host_block_1 = poolMalloc(*host_pool_);
    auto*        host_data_1  = static_cast<uint8_t*>(host_pool_->blockBuffer(host_block_1).addr);
    std::memset(host_data_1, 0x12, layer_bytes_[0]);
    std::memset(host_data_1 + layer_bytes_[0], 0x34, layer_bytes_[1]);
    std::memset(host_data_1 + layer_bytes_[0] + layer_bytes_[1], 0x9A, layer_bytes_[2]);

    BlockIdxType host_block_2 = poolMalloc(*host_pool_);
    auto*        host_data_2  = static_cast<uint8_t*>(host_pool_->blockBuffer(host_block_2).addr);
    std::memset(host_data_2, 0x56, layer_bytes_[0]);
    std::memset(host_data_2 + layer_bytes_[0], 0x78, layer_bytes_[1]);
    std::memset(host_data_2 + layer_bytes_[0] + layer_bytes_[1], 0xBC, layer_bytes_[2]);

    auto context = per_rank_transfer_engine_->submit(
        {makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, host_block_1),
         makeDescriptor(Tier::HOST, Tier::DEVICE, second_device_blocks, host_block_2)});
    context->waitDone();
    ASSERT_TRUE(context->success());

    const auto first_layer0  = readDeviceLayer(device_pool_, 0, device_block_);
    const auto first_layer1  = readDeviceLayer(device_pool_, 1, device_block_);
    const auto first_layer2  = readDeviceLayer(device_pool_, 2, device_block_);
    const auto second_layer0 = readDeviceLayer(device_pool_, 0, second_device_block);
    const auto second_layer1 = readDeviceLayer(device_pool_, 1, second_device_block);
    const auto second_layer2 = readDeviceLayer(device_pool_, 2, second_device_block);
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_EQ(first_layer0[i], 0x12);
        EXPECT_EQ(first_layer1[i], 0x34);
        EXPECT_EQ(first_layer2[i], 0x9A);
        EXPECT_EQ(second_layer0[i], 0x56);
        EXPECT_EQ(second_layer1[i], 0x78);
        EXPECT_EQ(second_layer2[i], 0xBC);
    }

    releasePoolBlock(*host_pool_, host_block_1);
    releasePoolBlock(*host_pool_, host_block_2);
    device_pool_->incRef(second_device_block);
    device_pool_->decRef(second_device_block);
}

TEST_F(PerRankBlockTransferEngineTest, SameDirectionDeviceToHostTasksMayUseSharedWorkers) {
    const BlockIdxType second_device_block = poolMalloc(*device_pool_);
    const BlockIdxType first_host_block     = poolMalloc(*host_pool_);
    const BlockIdxType second_host_block    = poolMalloc(*host_pool_);
    ASSERT_NE(second_device_block, NULL_BLOCK_IDX);
    ASSERT_NE(first_host_block, NULL_BLOCK_IDX);
    ASSERT_NE(second_host_block, NULL_BLOCK_IDX);

    auto blocking_strategy = std::make_unique<BlockingStrategy>();
    auto* blocker          = blocking_strategy.get();
    per_rank_transfer_engine_->device_host_executor_->strategies_.clear();
    per_rank_transfer_engine_->device_host_executor_->strategies_.push_back(std::move(blocking_strategy));

    auto first = per_rank_transfer_engine_->submit(
        {makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, first_host_block)});
    ASSERT_TRUE(blocker->waitUntilEntered(1, std::chrono::seconds(5)));
    auto second = per_rank_transfer_engine_->submit(
        {makeDescriptor(Tier::DEVICE, Tier::HOST, {second_device_block}, second_host_block)});
    const bool second_started_before_release = blocker->waitUntilEntered(2, std::chrono::milliseconds(200));

    blocker->release();
    first->waitDone();
    second->waitDone();
    EXPECT_TRUE(second_started_before_release);
    EXPECT_TRUE(first->success());
    EXPECT_TRUE(second->success());

    releasePoolBlock(*host_pool_, first_host_block);
    releasePoolBlock(*host_pool_, second_host_block);
    device_pool_->incRef(second_device_block);
    device_pool_->decRef(second_device_block);
}

TEST_F(PerRankBlockTransferEngineTest, SameDirectionHostToDeviceTasksMayUseSharedWorkers) {
    const BlockIdxType second_device_block = poolMalloc(*device_pool_);
    const BlockIdxType first_host_block     = poolMalloc(*host_pool_);
    const BlockIdxType second_host_block    = poolMalloc(*host_pool_);
    ASSERT_NE(second_device_block, NULL_BLOCK_IDX);
    ASSERT_NE(first_host_block, NULL_BLOCK_IDX);
    ASSERT_NE(second_host_block, NULL_BLOCK_IDX);

    auto blocking_strategy = std::make_unique<BlockingStrategy>();
    auto* blocker          = blocking_strategy.get();
    per_rank_transfer_engine_->device_host_executor_->strategies_.clear();
    per_rank_transfer_engine_->device_host_executor_->strategies_.push_back(std::move(blocking_strategy));

    auto first = per_rank_transfer_engine_->submit(
        {makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, first_host_block)});
    ASSERT_TRUE(blocker->waitUntilEntered(1, std::chrono::seconds(5)));
    auto second = per_rank_transfer_engine_->submit(
        {makeDescriptor(Tier::HOST, Tier::DEVICE, {second_device_block}, second_host_block)});
    const bool second_started_before_release = blocker->waitUntilEntered(2, std::chrono::milliseconds(200));

    blocker->release();
    first->waitDone();
    second->waitDone();
    EXPECT_TRUE(second_started_before_release);
    EXPECT_TRUE(first->success());
    EXPECT_TRUE(second->success());

    releasePoolBlock(*host_pool_, first_host_block);
    releasePoolBlock(*host_pool_, second_host_block);
    device_pool_->incRef(second_device_block);
    device_pool_->decRef(second_device_block);
}

TEST_F(PerRankBlockTransferEngineTest, TransferWorkerCountIsSharedAcrossDirections) {
    constexpr size_t kWorkerCount = 4;
    per_rank_transfer_engine_ = std::make_shared<PerRankBlockTransferEngine>(
        std::vector<GroupSetPtr>{group_set_}, DeviceHostCopyOptions{}, 4, 64, kWorkerCount);

    auto blocking_strategy = std::make_unique<BlockingStrategy>();
    auto* blocker          = blocking_strategy.get();
    per_rank_transfer_engine_->device_host_executor_->strategies_.clear();
    per_rank_transfer_engine_->device_host_executor_->strategies_.push_back(std::move(blocking_strategy));

    std::vector<std::shared_ptr<AsyncContext>> contexts;
    for (size_t index = 0; index < kWorkerCount; ++index) {
        contexts.push_back(per_rank_transfer_engine_->submit(
            {makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, /*host_block=*/1)}));
        contexts.push_back(per_rank_transfer_engine_->submit(
            {makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, /*host_block=*/1)}));
    }

    ASSERT_TRUE(blocker->waitUntilEntered(kWorkerCount, std::chrono::seconds(5)));
    EXPECT_FALSE(blocker->waitUntilEntered(kWorkerCount + 1, std::chrono::milliseconds(200)));

    blocker->release();
    for (const auto& context : contexts) {
        context->waitDone();
        EXPECT_TRUE(context->success());
    }
}

TEST_F(PerRankBlockTransferEngineTest, BatchAllowsSharedReadEndpoint) {
    fillDeviceLayer(device_pool_, 0, device_block_, {0x4A});
    const BlockIdxType first_host_block  = poolMalloc(*host_pool_);
    const BlockIdxType second_host_block = poolMalloc(*host_pool_);
    ASSERT_NE(first_host_block, NULL_BLOCK_IDX);
    ASSERT_NE(second_host_block, NULL_BLOCK_IDX);

    auto context = per_rank_transfer_engine_->submit(
        {makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, first_host_block),
         makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, second_host_block)});
    context->waitDone();
    EXPECT_TRUE(context->success());

    releasePoolBlock(*host_pool_, first_host_block);
    releasePoolBlock(*host_pool_, second_host_block);
}

class PerRankBlockTransferEngineMultiMemberTest: public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
        host_pool_ = makeHostPool(240, 4, true);
        disk_pool_ = makeDiskPool(240, 4, temp_dir_.path);
        pools_     = {
            makeDevicePool({{64, 16}}, 4, "per_rank_transfer_engine_multi_member_0"),
            makeDevicePool({{64, 16}}, 4, "per_rank_transfer_engine_multi_member_1"),
            makeDevicePool({{64, 16}}, 4, "per_rank_transfer_engine_multi_member_2"),
        };
        for (const auto& pool : pools_) {
            blocks_.push_back(poolMalloc(*pool));
            ASSERT_NE(blocks_.back(), NULL_BLOCK_IDX);
        }

        GroupSetPtr group =
            makeDeviceHostGroup(0,
                                pools_,
                                host_pool_,
                                {makeGroupBase({0}, 64, 16), makeGroupBase({0}, 64, 16), makeGroupBase({0}, 64, 16)},
                                disk_pool_);
        engine_     = makeEngine({group});
        host_block_ = poolMalloc(*host_pool_);
        disk_block_ = poolMalloc(*disk_pool_);
        ASSERT_NE(host_block_, NULL_BLOCK_IDX);
        ASSERT_NE(disk_block_, NULL_BLOCK_IDX);
    }

    TempDirGuard                                temp_dir_{"per_rank_transfer_engine_multi_member"};
    std::shared_ptr<HostBlockPool>              host_pool_;
    BlockTreeDiskBlockPoolPtr                   disk_pool_;
    std::vector<DeviceBlockPoolPtr>             pools_;
    std::vector<BlockIdxType>                   blocks_;
    std::shared_ptr<PerRankBlockTransferEngine> engine_;
    BlockIdxType                                host_block_{NULL_BLOCK_IDX};
    BlockIdxType                                disk_block_{NULL_BLOCK_IDX};
};

TEST_F(PerRankBlockTransferEngineMultiMemberTest, CompleteMultiMemberThreeTierRoundTripPreservesOffsets) {
    fillDeviceLayer(pools_[0], 0, blocks_[0], {0xA1, 0xA2});
    fillDeviceLayer(pools_[1], 0, blocks_[1], {0xB1, 0xB2});
    fillDeviceLayer(pools_[2], 0, blocks_[2], {0xC1, 0xC2});
    auto* host_data = static_cast<uint8_t*>(host_pool_->blockBuffer(host_block_).addr);
    std::memset(host_data, 0xFF, 240);

    const auto expect_host_payload = [host_data]() {
        for (size_t i = 0; i < 64; ++i)
            EXPECT_EQ(host_data[i], 0xA1);
        for (size_t i = 64; i < 80; ++i)
            EXPECT_EQ(host_data[i], 0xA2);
        for (size_t i = 80; i < 144; ++i)
            EXPECT_EQ(host_data[i], 0xB1);
        for (size_t i = 144; i < 160; ++i)
            EXPECT_EQ(host_data[i], 0xB2);
        for (size_t i = 160; i < 224; ++i)
            EXPECT_EQ(host_data[i], 0xC1);
        for (size_t i = 224; i < 240; ++i)
            EXPECT_EQ(host_data[i], 0xC2);
    };

    expectStatus(engine_,
                 makeDescriptor(Tier::DEVICE, Tier::HOST, {blocks_[0], blocks_[1], blocks_[2]}, host_block_),
                 TransferStatus::OK);
    expect_host_payload();

    expectStatus(engine_, makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block_, disk_block_), TransferStatus::OK);
    std::memset(host_data, 0, 240);
    expectStatus(engine_, makeDescriptor(Tier::DISK, Tier::HOST, {}, host_block_, disk_block_), TransferStatus::OK);
    expect_host_payload();

    fillDeviceLayer(pools_[0], 0, blocks_[0], {0x00, 0x00});
    fillDeviceLayer(pools_[1], 0, blocks_[1], {0x00, 0x00});
    fillDeviceLayer(pools_[2], 0, blocks_[2], {0x00, 0x00});
    expectStatus(engine_,
                 makeDescriptor(Tier::HOST, Tier::DEVICE, {blocks_[0], blocks_[1], blocks_[2]}, host_block_),
                 TransferStatus::OK);

    const auto comp0 = readDeviceLayer(pools_[0], 0, blocks_[0]);
    const auto comp1 = readDeviceLayer(pools_[1], 0, blocks_[1]);
    const auto comp2 = readDeviceLayer(pools_[2], 0, blocks_[2]);
    for (size_t i = 0; i < 64; ++i) {
        EXPECT_EQ(comp0[i], 0xA1);
        EXPECT_EQ(comp1[i], 0xB1);
        EXPECT_EQ(comp2[i], 0xC1);
    }
    for (size_t i = 64; i < 80; ++i) {
        EXPECT_EQ(comp0[i], 0xA2);
        EXPECT_EQ(comp1[i], 0xB2);
        EXPECT_EQ(comp2[i], 0xC2);
    }
}

TEST(PerRankBlockTransferEngineIntegrationTest, DeviceHostDiskHostDeviceRoundTrip) {
    ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
    TempDirGuard     temp_dir("per_rank_transfer_engine_three_tier");
    constexpr size_t payload_bytes = 80;
    auto             host_pool     = makeHostPool(payload_bytes, 2, true);
    auto             disk_pool     = makeDiskPool(payload_bytes, 2, temp_dir.path);
    auto             device_pool   = makeDevicePool({{64, 16}}, 2, "per_rank_transfer_engine_three_tier_device");
    auto             device_block  = poolMalloc(*device_pool);
    auto             host_block    = poolMalloc(*host_pool);
    auto             disk_block    = poolMalloc(*disk_pool);
    ASSERT_NE(device_block, NULL_BLOCK_IDX);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);

    auto group  = makeDeviceHostGroup(0, {device_pool}, host_pool, {makeGroupBase({0}, 64, 16)}, disk_pool);
    auto engine = makeEngine({group});
    fillDeviceLayer(device_pool, 0, device_block, {0x6A, 0xD3});
    const auto expected = readDeviceLayer(device_pool, 0, device_block);

    expectStatus(engine,
                 makeDescriptor(Tier::DEVICE, Tier::HOST, {device_block}, host_block, NULL_BLOCK_IDX, 0),
                 TransferStatus::OK);
    const auto* host_data = static_cast<const uint8_t*>(host_pool->blockBuffer(host_block).addr);
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), host_data));

    expectStatus(engine, makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, disk_block, 0), TransferStatus::OK);
    std::memset(host_pool->blockBuffer(host_block).addr, 0, payload_bytes);
    fillDeviceLayer(device_pool, 0, device_block, {0x00, 0x00});

    expectStatus(engine, makeDescriptor(Tier::DISK, Tier::HOST, {}, host_block, disk_block, 0), TransferStatus::OK);
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), host_data));

    fillDeviceLayer(device_pool, 0, device_block, {0x00, 0x00});
    expectStatus(engine,
                 makeDescriptor(Tier::HOST, Tier::DEVICE, {device_block}, host_block, NULL_BLOCK_IDX, 0),
                 TransferStatus::OK);
    EXPECT_EQ(readDeviceLayer(device_pool, 0, device_block), expected);
}

class RecordingReadBatchDiskBlockIO: public StatusDiskBlockIO {
public:
    RecordingReadBatchDiskBlockIO(): StatusDiskBlockIO(DiskBlockIOStatus::OK) {}

    DiskBlockIOStatus read(const std::vector<DiskRead>& reads) override {
        batch_sizes.push_back(reads.size());
        return StatusDiskBlockIO::read(reads);
    }

    std::vector<size_t> batch_sizes;
};

class BlockingDiskBlockIO: public DiskBlockIO {
public:
    enum class BlockOn {
        READ,
        WRITE,
    };

    explicit BlockingDiskBlockIO(BlockOn block_on): block_on_(block_on) {}

    DiskBlockIOStatus openAndPreallocate(const std::string&, size_t bytes, bool) override {
        std::lock_guard<std::mutex> lock(data_mutex_);
        data_.resize(bytes, 0);
        return DiskBlockIOStatus::OK;
    }
    DiskBlockIOStatus read(uint64_t offset, void* dst, size_t bytes) override {
        if (block_on_ == BlockOn::READ) {
            blockUntilReleased();
        }
        std::lock_guard<std::mutex> lock(data_mutex_);
        if (offset + bytes > data_.size()) {
            return DiskBlockIOStatus::INVALID_SIZE;
        }
        std::memcpy(dst, data_.data() + offset, bytes);
        return DiskBlockIOStatus::OK;
    }
    DiskBlockIOStatus write(uint64_t offset, const void* src, size_t bytes) override {
        if (block_on_ == BlockOn::WRITE) {
            blockUntilReleased();
        }
        std::lock_guard<std::mutex> lock(data_mutex_);
        if (offset + bytes > data_.size()) {
            data_.resize(offset + bytes, 0);
        }
        std::memcpy(data_.data() + offset, src, bytes);
        return DiskBlockIOStatus::OK;
    }
    DiskBlockIOStatus read(const std::vector<DiskRead>& reads) override {
        for (const auto& item : reads) {
            const auto status = read(item.offset, item.buffer, item.bytes);
            if (status != DiskBlockIOStatus::OK) {
                return status;
            }
        }
        return DiskBlockIOStatus::OK;
    }
    DiskBlockIOStatus write(const std::vector<DiskWrite>& writes) override {
        for (const auto& item : writes) {
            const auto status = write(item.offset, item.buffer, item.bytes);
            if (status != DiskBlockIOStatus::OK) {
                return status;
            }
        }
        return DiskBlockIOStatus::OK;
    }
    void        close() override {}
    std::string debugString() const override {
        return "BlockingDiskBlockIO";
    }

    bool waitUntilBlocked(std::chrono::milliseconds timeout) {
        return waitUntilBlocked(1, timeout);
    }
    bool waitUntilBlocked(size_t count, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, timeout, [this, count] { return blocked_count_ >= count; });
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        cv_.notify_all();
    }

private:
    void blockUntilReleased() {
        std::unique_lock<std::mutex> lock(mutex_);
        ++blocked_count_;
        cv_.notify_all();
        cv_.wait(lock, [this] { return released_; });
    }

    const BlockOn           block_on_;
    std::mutex              mutex_;
    std::condition_variable cv_;
    size_t                  blocked_count_{0};
    bool                    released_{false};
    std::mutex              data_mutex_;
    std::vector<char>       data_;
};

class BlockingIOGuard {
public:
    BlockingIOGuard(BlockingDiskBlockIO& io, std::thread& thread): io_(&io), thread_(&thread) {}
    ~BlockingIOGuard() {
        releaseAndJoin();
    }

    void releaseAndJoin() {
        if (io_ == nullptr) {
            return;
        }
        io_->release();
        if (thread_->joinable()) {
            thread_->join();
        }
        io_ = nullptr;
    }

private:
    BlockingDiskBlockIO* io_;
    std::thread*         thread_;
};

TEST(PerRankBlockTransferEngineIntegrationTest, DeviceDiskDirectRoundTripWithoutHostPool) {
    ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
    TempDirGuard     temp_dir("per_rank_direct_round_trip");
    constexpr size_t payload_bytes = 80;
    auto             owned_io      = std::make_unique<DirectAlignmentDiskBlockIO>();
    auto*            direct_io     = owned_io.get();
    auto disk_pool    = makeDiskPool(payload_bytes, 2, temp_dir.path, std::move(owned_io), "device_disk_direct", false);
    auto device_pool  = makeDevicePool({{64, 16}}, 2, "per_rank_direct_round_trip_device");
    auto device_block = poolMalloc(*device_pool);
    auto disk_block   = poolMalloc(*disk_pool);
    ASSERT_NE(device_block, NULL_BLOCK_IDX);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);

    auto group = makeDeviceHostGroup(0, {device_pool}, nullptr, {makeGroupBase({0}, 64, 16)}, disk_pool);
    ASSERT_EQ(group->hostPool(), nullptr);
    auto engine = makeEngine({group});

    fillDeviceLayer(device_pool, 0, device_block, {0x6A, 0xD3});
    const auto expected = readDeviceLayer(device_pool, 0, device_block);

    expectStatus(engine,
                 makeDescriptor(Tier::DEVICE, Tier::DISK, {device_block}, NULL_BLOCK_IDX, disk_block, 0),
                 TransferStatus::OK);
    EXPECT_FALSE(direct_io->bufferedIo());
    EXPECT_EQ(direct_io->lastWriteBytes(), disk_pool->strideBytes());
    fillDeviceLayer(device_pool, 0, device_block, {0x00, 0x00});
    expectStatus(engine,
                 makeDescriptor(Tier::DISK, Tier::DEVICE, {device_block}, NULL_BLOCK_IDX, disk_block, 0),
                 TransferStatus::OK);
    EXPECT_EQ(readDeviceLayer(device_pool, 0, device_block), expected);
    EXPECT_EQ(direct_io->lastReadBytes(), disk_pool->strideBytes());
}

TEST(PerRankBlockTransferEngineIntegrationTest, DeviceToDiskRejectsMultipleDescriptors) {
    ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
    TempDirGuard     temp_dir("per_rank_device_to_disk_batch");
    constexpr size_t payload_bytes = 80;
    auto             owned_io      = std::make_unique<DirectAlignmentDiskBlockIO>();
    auto*            direct_io     = owned_io.get();
    auto disk_pool = makeDiskPool(payload_bytes, 2, temp_dir.path, std::move(owned_io), "device_to_disk_batch", false);
    auto device_pool = makeDevicePool({{64, 16}}, 2, "per_rank_device_to_disk_batch");
    auto group = makeDeviceHostGroup(0, {device_pool}, nullptr, {makeGroupBase({0}, 64, 16)}, disk_pool);
    auto engine = makeEngine({group});

    std::vector<TransferDescriptor> descriptors;
    for (size_t index = 0; index < 2; ++index) {
        descriptors.push_back(makeDescriptor(Tier::DEVICE,
                                             Tier::DISK,
                                             {poolMalloc(*device_pool)},
                                             NULL_BLOCK_IDX,
                                             poolMalloc(*disk_pool),
                                             0));
    }
    auto context = engine->submit(descriptors);
    ASSERT_NE(context, nullptr);
    context->waitDone();
    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(direct_io->lastWriteBytes(), 0u);
}

TEST(PerRankBlockTransferEngineIntegrationTest, DeviceToDiskFailureReleasesStaging) {
    ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
    TempDirGuard     temp_dir("per_rank_device_to_disk_failure");
    constexpr size_t payload_bytes = 80;
    auto             failing_io    = std::make_unique<StatusDiskBlockIO>(DiskBlockIOStatus::IO_ERROR);
    auto*            status_io     = failing_io.get();
    auto disk_pool  = makeDiskPool(payload_bytes, 1, temp_dir.path, std::move(failing_io));
    auto device_pool = makeDevicePool({{64, 16}}, 1, "per_rank_device_to_disk_failure");
    auto descriptor = makeDescriptor(Tier::DEVICE,
                                     Tier::DISK,
                                     {poolMalloc(*device_pool)},
                                     NULL_BLOCK_IDX,
                                     poolMalloc(*disk_pool),
                                     0);
    auto group = makeDeviceHostGroup(0, {device_pool}, nullptr, {makeGroupBase({0}, 64, 16)}, disk_pool);
    auto engine = makeEngine({group});

    expectStatus(engine, descriptor, TransferStatus::DISK_IO_ERROR);
    status_io->setStatus(DiskBlockIOStatus::OK);
    expectStatus(engine, descriptor, TransferStatus::OK);
}

TEST(PerRankBlockTransferEngineIntegrationTest, DiskDeviceLaneCapacityControlsPhysicalBatchSize) {
    ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
    TempDirGuard temp_dir("per_rank_disk_device_lane_capacity");

    auto full_io = std::make_unique<RecordingReadBatchDiskBlockIO>();
    auto* full_io_ptr = full_io.get();
    auto swa_io = std::make_unique<RecordingReadBatchDiskBlockIO>();
    auto* swa_io_ptr = swa_io.get();
    auto full_disk = makeDiskPool(8192, 5, temp_dir.path, std::move(full_io), "disk_device_full");
    auto swa_disk = makeDiskPool(4096, 5, temp_dir.path, std::move(swa_io), "disk_device_swa");
    auto full_device = makeDevicePool({{8192, 0}}, 5, "disk_device_full");
    auto swa_device = makeDevicePool({{4096, 0}}, 5, "disk_device_swa");
    auto full_group = makeDeviceHostGroup(
        0, {full_device}, nullptr, {makeGroupBase(CacheGroupType::FULL, {0}, 8192)}, full_disk);
    auto swa_group = makeDeviceHostGroup(
        1, {swa_device}, nullptr, {makeGroupBase(CacheGroupType::SWA, {0}, 4096)}, swa_disk);
    auto engine = std::make_shared<PerRankBlockTransferEngine>(
        std::vector<GroupSetPtr>{full_group, swa_group}, DeviceHostCopyOptions{}, 4);

    EXPECT_EQ(engine->device_disk_executor_->full_batch_capacity_, 2u);
    EXPECT_EQ(engine->device_disk_executor_->swa_batch_capacity_, 4u);

    std::vector<TransferDescriptor> full_descriptors;
    std::vector<TransferDescriptor> swa_descriptors;
    for (size_t index = 0; index < 5; ++index) {
        full_descriptors.push_back(makeDescriptor(Tier::DISK,
                                                  Tier::DEVICE,
                                                  {poolMalloc(*full_device)},
                                                  NULL_BLOCK_IDX,
                                                  poolMalloc(*full_disk),
                                                  0));
        swa_descriptors.push_back(makeDescriptor(Tier::DISK,
                                                 Tier::DEVICE,
                                                 {poolMalloc(*swa_device)},
                                                 NULL_BLOCK_IDX,
                                                 poolMalloc(*swa_disk),
                                                 1));
    }

    auto full_context = engine->submit(full_descriptors);
    full_context->waitDone();
    ASSERT_TRUE(full_context->success());
    auto swa_context = engine->submit(swa_descriptors);
    swa_context->waitDone();
    ASSERT_TRUE(swa_context->success());
    EXPECT_EQ(full_io_ptr->batch_sizes, (std::vector<size_t>{2, 2, 1}));
    EXPECT_EQ(swa_io_ptr->batch_sizes, (std::vector<size_t>{4, 1}));
}

TEST(PerRankBlockTransferEngineIntegrationTest, DiskToDeviceReturnsPendingContext) {
    ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
    TempDirGuard     temp_dir("per_rank_disk_to_device_async");
    constexpr size_t payload_bytes = 80;
    auto             blocking_io   = std::make_unique<BlockingDiskBlockIO>(BlockingDiskBlockIO::BlockOn::READ);
    auto*            io            = blocking_io.get();
    auto disk_pool  = makeDiskPool(payload_bytes, 1, temp_dir.path, std::move(blocking_io));
    auto device_pool = makeDevicePool({{64, 16}}, 1, "per_rank_disk_to_device_async");
    auto device_block = poolMalloc(*device_pool);
    auto disk_block   = poolMalloc(*disk_pool);
    auto group = makeDeviceHostGroup(0, {device_pool}, nullptr, {makeGroupBase({0}, 64, 16)}, disk_pool);
    auto engine = makeEngine({group});
    [[maybe_unused]] auto release_guard = std::shared_ptr<void>(nullptr, [io](void*) { io->release(); });
    std::vector<uint8_t> disk_data(disk_pool->strideBytes(), 0x5A);
    ASSERT_EQ(disk_pool->write(disk_block, disk_data.data(), disk_data.size()), BlockIOStatus::OK);

    auto context = engine->submit(
        {makeDescriptor(Tier::DISK, Tier::DEVICE, {device_block}, NULL_BLOCK_IDX, disk_block, 0)});
    ASSERT_NE(context, nullptr);
    ASSERT_TRUE(io->waitUntilBlocked(std::chrono::seconds(5)));
    EXPECT_FALSE(context->done());
    io->release();
    context->waitDone();
    EXPECT_TRUE(context->success());
}

TEST(PerRankBlockTransferEngineIntegrationTest, DiskDeviceStageFailureShortCircuitsAndReleasesStaging) {
    ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
    TempDirGuard     temp_dir("per_rank_direct_stage_failure");
    constexpr size_t payload_bytes = 80;
    auto             failing_io    = std::make_unique<StatusDiskBlockIO>(DiskBlockIOStatus::IO_ERROR);
    auto*            status_io     = failing_io.get();
    auto             disk_pool     = makeDiskPool(payload_bytes, 2, temp_dir.path, std::move(failing_io));
    auto             device_pool   = makeDevicePool({{64, 16}}, 2, "per_rank_direct_stage_failure_device");
    auto             device_block  = poolMalloc(*device_pool);
    auto             disk_block    = poolMalloc(*disk_pool);
    ASSERT_NE(device_block, NULL_BLOCK_IDX);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);

    auto group  = makeDeviceHostGroup(0, {device_pool}, nullptr, {makeGroupBase({0}, 64, 16)}, disk_pool);
    auto engine = makeEngine({group});

    fillDeviceLayer(device_pool, 0, device_block, {0xAB, 0xCD});
    const auto expected = readDeviceLayer(device_pool, 0, device_block);

    expectStatus(engine,
                 makeDescriptor(Tier::DISK, Tier::DEVICE, {device_block}, NULL_BLOCK_IDX, disk_block, 0),
                 TransferStatus::DISK_IO_ERROR);
    EXPECT_EQ(readDeviceLayer(device_pool, 0, device_block), expected);
    status_io->setStatus(DiskBlockIOStatus::OK);
    expectStatus(engine,
                 makeDescriptor(Tier::DISK, Tier::DEVICE, {device_block}, NULL_BLOCK_IDX, disk_block, 0),
                 TransferStatus::OK);
}

TEST(PerRankBlockTransferEngineIntegrationTest, DiskDeviceStageTwoFailureReleasesStaging) {
    ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
    TempDirGuard temp_dir("per_rank_stage_two_failure");
    constexpr size_t payload_bytes = 80;
    auto disk_pool = makeDiskPool(payload_bytes, 2, temp_dir.path,
                                  std::make_unique<StatusDiskBlockIO>(DiskBlockIOStatus::OK));
    auto device_pool = makeDevicePool({{64, 16}}, 2, "per_rank_stage_two_failure_device");
    const auto device_block = poolMalloc(*device_pool);
    const auto disk_block = poolMalloc(*disk_pool);
    auto group = makeDeviceHostGroup(0, {device_pool}, nullptr, {makeGroupBase({0}, 64, 16)}, disk_pool);
    auto engine = makeEngine({group});
    auto& strategies = engine->device_host_executor_->strategies_;
    strategies.clear();
    strategies.push_back(std::make_unique<FailingStrategy>());
    const auto descriptor =
        makeDescriptor(Tier::DISK, Tier::DEVICE, {device_block}, NULL_BLOCK_IDX, disk_block, 0);

    auto failed = engine->submit({descriptor});
    failed->waitDone();
    EXPECT_FALSE(failed->success());

    strategies.clear();
    strategies.push_back(std::make_unique<GenericMultiCopyDeviceHostCopyStrategy>());
    auto retry = engine->submit({descriptor});
    retry->waitDone();
    EXPECT_TRUE(retry->success());
}

TEST(PerRankBlockTransferEngineIntegrationTest, DiskDeviceWaitingForStagingDoesNotOccupyTransferWorker) {
    ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
    TempDirGuard temp_dir("per_rank_staging_admission");
    constexpr size_t payload_bytes = 80;
    auto disk_pool = makeDiskPool(payload_bytes, 1, temp_dir.path,
                                  std::make_unique<StatusDiskBlockIO>(DiskBlockIOStatus::OK));
    auto host_pool   = makeHostPool(payload_bytes, 1, true);
    auto device_pool = makeDevicePool({{64, 16}}, 2, "per_rank_staging_admission_device");
    auto group = makeDeviceHostGroup(
        0, {device_pool}, host_pool, {makeGroupBase(CacheGroupType::FULL, {0}, 64, 16)}, disk_pool);
    auto engine = std::make_shared<PerRankBlockTransferEngine>(
        std::vector<GroupSetPtr>{group}, DeviceHostCopyOptions{}, 2, 64, 1);

    auto held_staging = engine->device_disk_executor_->full_staging_pool_->tryMallocBatch(1);
    ASSERT_TRUE(held_staging.has_value());

    auto blocking_strategy = std::make_unique<BlockingStrategy>();
    auto* blocker          = blocking_strategy.get();
    engine->device_host_executor_->strategies_.clear();
    engine->device_host_executor_->strategies_.push_back(std::move(blocking_strategy));

    auto disk_to_device = engine->submit({makeDescriptor(Tier::DISK,
                                                         Tier::DEVICE,
                                                         {poolMalloc(*device_pool)},
                                                         NULL_BLOCK_IDX,
                                                         poolMalloc(*disk_pool),
                                                         0)});
    auto host_to_device = engine->submit({makeDescriptor(Tier::HOST,
                                                         Tier::DEVICE,
                                                         {poolMalloc(*device_pool)},
                                                         poolMalloc(*host_pool),
                                                         NULL_BLOCK_IDX,
                                                         0)});

    EXPECT_TRUE(blocker->waitUntilEntered(1, std::chrono::milliseconds(300)));
    EXPECT_FALSE(disk_to_device->done());
    blocker->release();
    host_to_device->waitDone();
    EXPECT_TRUE(host_to_device->success());

    held_staging.reset();
    disk_to_device->waitDone();
    EXPECT_TRUE(disk_to_device->success());
}

TEST(PerRankBlockTransferEngineIntegrationTest, DeviceDiskWaitingForStagingDoesNotOccupyTransferWorker) {
    ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
    TempDirGuard temp_dir("per_rank_device_disk_staging_admission");
    constexpr size_t payload_bytes = 80;
    auto disk_pool = makeDiskPool(payload_bytes, 1, temp_dir.path,
                                  std::make_unique<StatusDiskBlockIO>(DiskBlockIOStatus::OK));
    auto host_pool   = makeHostPool(payload_bytes, 1, true);
    auto device_pool = makeDevicePool({{64, 16}}, 2, "per_rank_device_disk_staging_admission_device");
    auto group = makeDeviceHostGroup(
        0, {device_pool}, host_pool, {makeGroupBase(CacheGroupType::FULL, {0}, 64, 16)}, disk_pool);
    auto engine = std::make_shared<PerRankBlockTransferEngine>(
        std::vector<GroupSetPtr>{group}, DeviceHostCopyOptions{}, 2, 64, 1);

    auto held_staging = engine->device_disk_executor_->full_staging_pool_->tryMallocBatch(1);
    ASSERT_TRUE(held_staging.has_value());

    auto blocking_strategy = std::make_unique<BlockingStrategy>();
    auto* blocker          = blocking_strategy.get();
    engine->device_host_executor_->strategies_.clear();
    engine->device_host_executor_->strategies_.push_back(std::move(blocking_strategy));

    auto device_to_disk = engine->submit({makeDescriptor(Tier::DEVICE,
                                                         Tier::DISK,
                                                         {poolMalloc(*device_pool)},
                                                         NULL_BLOCK_IDX,
                                                         poolMalloc(*disk_pool),
                                                         0)});
    auto host_to_device = engine->submit({makeDescriptor(Tier::HOST,
                                                         Tier::DEVICE,
                                                         {poolMalloc(*device_pool)},
                                                         poolMalloc(*host_pool),
                                                         NULL_BLOCK_IDX,
                                                         0)});

    EXPECT_TRUE(blocker->waitUntilEntered(1, std::chrono::milliseconds(300)));
    EXPECT_FALSE(device_to_disk->done());
    blocker->release();
    host_to_device->waitDone();
    EXPECT_TRUE(host_to_device->success());

    held_staging.reset();
    device_to_disk->waitDone();
    EXPECT_TRUE(device_to_disk->success());
}

TEST(PerRankBlockTransferEngineIntegrationTest, DiskDeviceFullAndSwaStagingMayOverlapWithSharedWorkers) {
    ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
    TempDirGuard temp_dir("per_rank_disk_device_lane_overlap");
    constexpr size_t payload_bytes = 80;
    auto full_io = std::make_unique<BlockingDiskBlockIO>(BlockingDiskBlockIO::BlockOn::READ);
    auto* full_io_ptr = full_io.get();
    auto swa_io = std::make_unique<BlockingDiskBlockIO>(BlockingDiskBlockIO::BlockOn::READ);
    auto* swa_io_ptr = swa_io.get();
    auto full_disk = makeDiskPool(payload_bytes, 1, temp_dir.path, std::move(full_io), "lane_overlap_full");
    auto swa_disk = makeDiskPool(payload_bytes, 1, temp_dir.path, std::move(swa_io), "lane_overlap_swa");
    auto full_device = makeDevicePool({{64, 16}}, 1, "lane_overlap_full");
    auto swa_device = makeDevicePool({{64, 16}}, 1, "lane_overlap_swa");
    auto full_group = makeDeviceHostGroup(
        0, {full_device}, nullptr, {makeGroupBase(CacheGroupType::FULL, {0}, 64, 16)}, full_disk);
    auto swa_group = makeDeviceHostGroup(
        1, {swa_device}, nullptr, {makeGroupBase(CacheGroupType::SWA, {0}, 64, 16)}, swa_disk);
    auto engine = std::make_shared<PerRankBlockTransferEngine>(
        std::vector<GroupSetPtr>{full_group, swa_group}, DeviceHostCopyOptions{}, 4, 64, 2);
    [[maybe_unused]] auto release_guard = std::shared_ptr<void>(nullptr, [full_io_ptr, swa_io_ptr](void*) {
        full_io_ptr->release();
        swa_io_ptr->release();
    });

    auto full_context = engine->submit({makeDescriptor(Tier::DISK,
                                                       Tier::DEVICE,
                                                       {poolMalloc(*full_device)},
                                                       NULL_BLOCK_IDX,
                                                       poolMalloc(*full_disk),
                                                       0)});
    ASSERT_TRUE(full_io_ptr->waitUntilBlocked(std::chrono::seconds(5)));
    auto swa_context = engine->submit({makeDescriptor(Tier::DISK,
                                                      Tier::DEVICE,
                                                      {poolMalloc(*swa_device)},
                                                      NULL_BLOCK_IDX,
                                                      poolMalloc(*swa_disk),
                                                      1)});
    ASSERT_TRUE(swa_io_ptr->waitUntilBlocked(std::chrono::milliseconds(500)));

    full_io_ptr->release();
    swa_io_ptr->release();
    full_context->waitDone();
    swa_context->waitDone();
    EXPECT_TRUE(full_context->success());
    EXPECT_TRUE(swa_context->success());
}

TEST(PerRankBlockTransferEngineIntegrationTest, DiskDeviceSameLaneTasksMayUseAvailableStagingAndWorkers) {
    ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
    for (const CacheGroupType group_type : {CacheGroupType::FULL, CacheGroupType::SWA}) {
        SCOPED_TRACE(group_type == CacheGroupType::FULL ? "FULL" : "SWA");
        TempDirGuard temp_dir(group_type == CacheGroupType::FULL ? "per_rank_disk_device_same_lane_full" :
                                                                        "per_rank_disk_device_same_lane_swa");
        constexpr size_t payload_bytes = 80;
        auto blocking_io = std::make_unique<BlockingDiskBlockIO>(BlockingDiskBlockIO::BlockOn::READ);
        auto* io = blocking_io.get();
        const std::string pool_name = group_type == CacheGroupType::FULL ? "same_lane_full" : "same_lane_swa";
        auto disk_pool = makeDiskPool(payload_bytes, 2, temp_dir.path, std::move(blocking_io), pool_name);
        auto device_pool = makeDevicePool({{64, 16}}, 2, pool_name);
        auto group = makeDeviceHostGroup(
            0, {device_pool}, nullptr, {makeGroupBase(group_type, {0}, 64, 16)}, disk_pool);
        auto engine = std::make_shared<PerRankBlockTransferEngine>(
            std::vector<GroupSetPtr>{group}, DeviceHostCopyOptions{}, 4);
        [[maybe_unused]] auto release_guard = std::shared_ptr<void>(nullptr, [io](void*) { io->release(); });

        auto first = engine->submit({makeDescriptor(Tier::DISK,
                                                    Tier::DEVICE,
                                                    {poolMalloc(*device_pool)},
                                                    NULL_BLOCK_IDX,
                                                    poolMalloc(*disk_pool),
                                                    0)});
        ASSERT_TRUE(io->waitUntilBlocked(std::chrono::seconds(5)));
        auto second = engine->submit({makeDescriptor(Tier::DISK,
                                                     Tier::DEVICE,
                                                     {poolMalloc(*device_pool)},
                                                     NULL_BLOCK_IDX,
                                                     poolMalloc(*disk_pool),
                                                     0)});
        const bool second_started_before_release = io->waitUntilBlocked(2, std::chrono::milliseconds(200));

        io->release();
        first->waitDone();
        second->waitDone();
        EXPECT_TRUE(second_started_before_release);
        EXPECT_TRUE(first->success());
        EXPECT_TRUE(second->success());
    }
}

// ---- Strategy chain tests ----

class PerRankBlockTransferEngineStrategyTest: public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";

        layer_bytes_     = {128, 128};
        host_block_size_ = 256;

        host_pool_    = makeHostPool(host_block_size_, 10, true);
        device_pool_  = makeDevicePool({{128, 0}, {256, 0}}, 10, "strategy_test_device");
        device_block_ = poolMalloc(*device_pool_);
        ASSERT_NE(device_block_, NULL_BLOCK_IDX);
        device_blocks_ = {device_block_};

        group_set_ = makeDeviceHostGroup(0, {device_pool_}, host_pool_, {makeGroupBase({0, 1}, 128)});
    }

    std::shared_ptr<PerRankBlockTransferEngine> makePerRankBlockTransferEngine(DeviceHostCopyOptions options = {}) {
        return makeEngine({group_set_}, std::move(options));
    }

    std::vector<size_t>            layer_bytes_;
    size_t                         host_block_size_;
    std::shared_ptr<HostBlockPool> host_pool_;
    DeviceBlockPoolPtr             device_pool_;
    BlockIdxType                   device_block_;
    std::vector<BlockIdxType>      device_blocks_;
    GroupSetPtr                    group_set_;
};

TEST_F(PerRankBlockTransferEngineStrategyTest, GenericStrategyRoundTrip) {
    DeviceHostCopyOptions options;
    options.cuda_batch_copy_enabled                          = false;
    auto                            per_rank_transfer_engine = makePerRankBlockTransferEngine(options);
    std::array<StrategyCounters, 3> counters;
    installStrategyRecorders(*per_rank_transfer_engine->device_host_executor_, counters);

    fillDeviceLayer(device_pool_, 0, device_block_, {0xAA});
    fillDeviceLayer(device_pool_, 1, device_block_, {0xBB});

    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    auto d2h = makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block);
    ASSERT_TRUE(submitSucceeded(per_rank_transfer_engine, d2h));

    const auto* host_data = static_cast<const uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    for (size_t i = 0; i < 128; ++i)
        EXPECT_EQ(host_data[i], 0xAA);
    for (size_t i = 128; i < 256; ++i)
        EXPECT_EQ(host_data[i], 0xBB);

    fillDeviceLayer(device_pool_, 0, device_block_, {0x00});
    fillDeviceLayer(device_pool_, 1, device_block_, {0x00});

    auto h2d = makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, host_block);
    ASSERT_TRUE(submitSucceeded(per_rank_transfer_engine, h2d));

    auto d0 = readDeviceLayer(device_pool_, 0, device_block_);
    auto d1 = readDeviceLayer(device_pool_, 1, device_block_);
    for (auto b : d0)
        EXPECT_EQ(b, 0xAA);
    for (size_t i = 0; i < 128; ++i)
        EXPECT_EQ(d1[i], 0xBB);
    for (size_t i = 128; i < d1.size(); ++i)
        EXPECT_EQ(d1[i], 0x00);
    EXPECT_EQ(counters[0].not_applicable, 2);
    EXPECT_EQ(counters[1].not_applicable, 2);
    EXPECT_EQ(counters[2].done, 2);

    releasePoolBlock(*host_pool_, host_block);
}

TEST_F(PerRankBlockTransferEngineStrategyTest, BatchStrategyExecutesWhenSupportedOtherwiseFallsBack) {
    const bool expect_batch_done = expectCudaBatchStrategyDone();

    DeviceHostCopyOptions options;
    options.cuda_batch_copy_enabled                          = true;
    auto                            per_rank_transfer_engine = makePerRankBlockTransferEngine(options);
    std::array<StrategyCounters, 3> counters;
    installStrategyRecorders(*per_rank_transfer_engine->device_host_executor_, counters);

    fillDeviceLayer(device_pool_, 0, device_block_, {0x11});
    fillDeviceLayer(device_pool_, 1, device_block_, {0x22});

    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    auto d2h = makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block);
    ASSERT_TRUE(submitSucceeded(per_rank_transfer_engine, d2h));

    const auto* host_data = static_cast<const uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    for (size_t i = 0; i < 128; ++i)
        EXPECT_EQ(host_data[i], 0x11);
    for (size_t i = 128; i < 256; ++i)
        EXPECT_EQ(host_data[i], 0x22);

    fillDeviceLayer(device_pool_, 0, device_block_, {0x00});
    fillDeviceLayer(device_pool_, 1, device_block_, {0x00});

    auto h2d = makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, host_block);
    ASSERT_TRUE(submitSucceeded(per_rank_transfer_engine, h2d));

    auto d0 = readDeviceLayer(device_pool_, 0, device_block_);
    auto d1 = readDeviceLayer(device_pool_, 1, device_block_);
    for (auto b : d0)
        EXPECT_EQ(b, 0x11);
    for (size_t i = 0; i < 128; ++i)
        EXPECT_EQ(d1[i], 0x22);
    for (size_t i = 128; i < d1.size(); ++i)
        EXPECT_EQ(d1[i], 0x00);

    EXPECT_EQ(counters[0].attempts, 2);
    EXPECT_EQ(counters[0].done, expect_batch_done ? 2 : 0);
    EXPECT_EQ(counters[0].not_applicable, expect_batch_done ? 0 : 2);
    EXPECT_EQ(counters[0].failed, 0);

    EXPECT_EQ(counters[1].attempts, expect_batch_done ? 0 : 2);
    EXPECT_EQ(counters[1].done, 0);
    EXPECT_EQ(counters[1].not_applicable, expect_batch_done ? 0 : 2);
    EXPECT_EQ(counters[1].failed, 0);

    EXPECT_EQ(counters[2].attempts, expect_batch_done ? 0 : 2);
    EXPECT_EQ(counters[2].done, expect_batch_done ? 0 : 2);
    EXPECT_EQ(counters[2].not_applicable, 0);
    EXPECT_EQ(counters[2].failed, 0);

    releasePoolBlock(*host_pool_, host_block);
}

TEST_F(PerRankBlockTransferEngineStrategyTest, StagedEnabledBelowThresholdFallsBackToGeneric) {
    DeviceHostCopyOptions options;
    options.staged_sm_copy_enabled                           = true;
    options.staged_sm_min_tile_count                         = 100;
    options.staged_sm_min_bytes                              = 0;
    options.cuda_batch_copy_enabled                          = false;
    auto                            per_rank_transfer_engine = makePerRankBlockTransferEngine(options);
    std::array<StrategyCounters, 3> counters;
    installStrategyRecorders(*per_rank_transfer_engine->device_host_executor_, counters);

    fillDeviceLayer(device_pool_, 0, device_block_, {0xCC});
    fillDeviceLayer(device_pool_, 1, device_block_, {0xDD});

    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    auto d2h = makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block);
    ASSERT_TRUE(submitSucceeded(per_rank_transfer_engine, d2h));

    const auto* host_data = static_cast<const uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    for (size_t i = 0; i < 128; ++i)
        EXPECT_EQ(host_data[i], 0xCC);
    for (size_t i = 128; i < 256; ++i)
        EXPECT_EQ(host_data[i], 0xDD);
    EXPECT_EQ(counters[0].not_applicable, 1);
    EXPECT_EQ(counters[1].not_applicable, 1);
    EXPECT_EQ(counters[2].done, 1);

    releasePoolBlock(*host_pool_, host_block);
}

TEST_F(PerRankBlockTransferEngineStrategyTest, StagedStrategyAboveThresholdRoundTrip) {
    DeviceHostCopyOptions options;
    options.staged_sm_copy_enabled                           = true;
    options.staged_sm_min_tile_count                         = 1;
    options.staged_sm_min_bytes                              = 1;
    options.cuda_batch_copy_enabled                          = false;
    auto                            per_rank_transfer_engine = makePerRankBlockTransferEngine(options);
    std::array<StrategyCounters, 3> counters;
    installStrategyRecorders(*per_rank_transfer_engine->device_host_executor_, counters);

    fillDeviceLayer(device_pool_, 0, device_block_, {0x31});
    fillDeviceLayer(device_pool_, 1, device_block_, {0x42});

    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    auto d2h = makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block);
    ASSERT_TRUE(submitSucceeded(per_rank_transfer_engine, d2h));
    const auto* host_data = static_cast<const uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    for (size_t i = 0; i < 128; ++i)
        EXPECT_EQ(host_data[i], 0x31);
    for (size_t i = 128; i < 256; ++i)
        EXPECT_EQ(host_data[i], 0x42);

    fillDeviceLayer(device_pool_, 0, device_block_, {0x00});
    fillDeviceLayer(device_pool_, 1, device_block_, {0x00});
    auto h2d = makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, host_block);
    ASSERT_TRUE(submitSucceeded(per_rank_transfer_engine, h2d));
    for (auto byte : readDeviceLayer(device_pool_, 0, device_block_))
        EXPECT_EQ(byte, 0x31);
    const auto staged_layer1 = readDeviceLayer(device_pool_, 1, device_block_);
    for (size_t i = 0; i < 128; ++i)
        EXPECT_EQ(staged_layer1[i], 0x42);
    for (size_t i = 128; i < staged_layer1.size(); ++i)
        EXPECT_EQ(staged_layer1[i], 0x00);
    EXPECT_EQ(counters[0].not_applicable, 2);
    EXPECT_EQ(counters[1].done, 2);
    EXPECT_EQ(counters[2].attempts, 0);

    releasePoolBlock(*host_pool_, host_block);
}

TEST_F(PerRankBlockTransferEngineStrategyTest, CudaBatchTakesPrecedenceWhenBothStrategiesAreEligible) {
    const bool expect_batch_done = expectCudaBatchStrategyDone();
    DeviceHostCopyOptions options;
    options.staged_sm_copy_enabled                           = true;
    options.staged_sm_min_tile_count                         = 1;
    options.staged_sm_min_bytes                              = 1;
    options.cuda_batch_copy_enabled                          = true;
    auto                            per_rank_transfer_engine = makePerRankBlockTransferEngine(options);
    std::array<StrategyCounters, 3> counters;
    installStrategyRecorders(*per_rank_transfer_engine->device_host_executor_, counters);

    fillDeviceLayer(device_pool_, 0, device_block_, {0x5C});
    fillDeviceLayer(device_pool_, 1, device_block_, {0x6D});
    auto host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    expectStatus(per_rank_transfer_engine,
                 makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block),
                 TransferStatus::OK);
    EXPECT_EQ(counters[0].done, expect_batch_done ? 1 : 0);
    EXPECT_EQ(counters[0].not_applicable, expect_batch_done ? 0 : 1);
    EXPECT_EQ(counters[1].attempts, expect_batch_done ? 0 : 1);
    EXPECT_EQ(counters[1].done, expect_batch_done ? 0 : 1);
    EXPECT_EQ(counters[2].attempts, 0);

    releasePoolBlock(*host_pool_, host_block);
}

}  // namespace
}  // namespace rtp_llm

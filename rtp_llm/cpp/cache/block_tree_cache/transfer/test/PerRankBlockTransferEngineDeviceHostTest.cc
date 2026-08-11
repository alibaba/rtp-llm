#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <limits>
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
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceDiskTransferExecutor.h"
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

class CapturingDoneStrategy: public DeviceHostCopyStrategy {
public:
    StrategyResult tryExecute(const DeviceHostCopyPlan& plan, const DeviceHostCopyOptions&) override {
        plans.push_back(plan);
        return StrategyResult::done();
    }

    std::vector<DeviceHostCopyPlan> plans;
};

class CapturingFailingStrategy: public DeviceHostCopyStrategy {
public:
    StrategyResult tryExecute(const DeviceHostCopyPlan& plan, const DeviceHostCopyOptions&) override {
        plans.push_back(plan);
        return StrategyResult::failed(TransferStatus::DEVICE_IO_ERROR);
    }

    std::vector<DeviceHostCopyPlan> plans;
};

class ThrowingDeviceHostCopyStrategy: public DeviceHostCopyStrategy {
public:
    StrategyResult tryExecute(const DeviceHostCopyPlan&, const DeviceHostCopyOptions&) override {
        throw std::runtime_error("injected device host copy exception");
    }
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

    host_pool_->free(host_block);
}

TEST_F(PerRankBlockTransferEngineTest, SubmitVectorExecutesDeviceHostDirectionsAsLogicalBatches) {
    const BlockIdxType second_device_block = poolMalloc(*device_pool_);
    const BlockIdxType first_host_block    = poolMalloc(*host_pool_);
    const BlockIdxType second_host_block   = poolMalloc(*host_pool_);
    ASSERT_NE(second_device_block, NULL_BLOCK_IDX);
    ASSERT_NE(first_host_block, NULL_BLOCK_IDX);
    ASSERT_NE(second_host_block, NULL_BLOCK_IDX);

    for (size_t layer = 0; layer < layer_bytes_.size(); ++layer) {
        fillDeviceLayer(device_pool_, layer, device_block_, {0x31});
        fillDeviceLayer(device_pool_, layer, second_device_block, {0x52});
    }

    auto context = per_rank_transfer_engine_->submit(std::vector<TransferDescriptor>{
        makeDescriptor(Tier::DEVICE, Tier::HOST, {device_block_}, first_host_block),
        makeDescriptor(Tier::DEVICE, Tier::HOST, {second_device_block}, second_host_block),
    });
    ASSERT_NE(context, nullptr);
    context->waitDone();
    ASSERT_TRUE(context->success()) << context->errorInfo().ToString();

    auto* first_host_data  = static_cast<uint8_t*>(host_pool_->blockBuffer(first_host_block).addr);
    auto* second_host_data = static_cast<uint8_t*>(host_pool_->blockBuffer(second_host_block).addr);
    EXPECT_EQ(first_host_data[0], 0x31);
    EXPECT_EQ(second_host_data[0], 0x52);

    std::memset(first_host_data, 0x63, host_block_size_);
    std::memset(second_host_data, 0x74, host_block_size_);
    context = per_rank_transfer_engine_->submit(std::vector<TransferDescriptor>{
        makeDescriptor(Tier::HOST, Tier::DEVICE, {device_block_}, first_host_block),
        makeDescriptor(Tier::HOST, Tier::DEVICE, {second_device_block}, second_host_block),
    });
    ASSERT_NE(context, nullptr);
    context->waitDone();
    ASSERT_TRUE(context->success()) << context->errorInfo().ToString();

    EXPECT_EQ(readDeviceLayer(device_pool_, 0, device_block_).front(), 0x63);
    EXPECT_EQ(readDeviceLayer(device_pool_, 0, second_device_block).front(), 0x74);

    host_pool_->free(first_host_block);
    host_pool_->free(second_host_block);
    device_pool_->free(second_device_block);
}

TEST_F(PerRankBlockTransferEngineTest, SubmitVectorRejectsDuplicateDeviceTargetBeforeIo) {
    const BlockIdxType first_host_block  = poolMalloc(*host_pool_);
    const BlockIdxType second_host_block = poolMalloc(*host_pool_);
    ASSERT_NE(first_host_block, NULL_BLOCK_IDX);
    ASSERT_NE(second_host_block, NULL_BLOCK_IDX);

    auto  strategy     = std::make_unique<CapturingDoneStrategy>();
    auto* strategy_ptr = strategy.get();
    per_rank_transfer_engine_->device_host_executor_->strategies_.clear();
    per_rank_transfer_engine_->device_host_executor_->strategies_.push_back(std::move(strategy));

    auto context = per_rank_transfer_engine_->submit(std::vector<TransferDescriptor>{
        makeDescriptor(Tier::HOST, Tier::DEVICE, {device_block_}, first_host_block),
        makeDescriptor(Tier::HOST, Tier::DEVICE, {device_block_}, second_host_block),
    });
    ASSERT_NE(context, nullptr);
    context->waitDone();

    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_NE(context->errorInfo().ToString().find("descriptor_index=1"), std::string::npos);
    EXPECT_TRUE(strategy_ptr->plans.empty());

    host_pool_->free(first_host_block);
    host_pool_->free(second_host_block);
}

TEST_F(PerRankBlockTransferEngineTest, SubmitVectorRejectsDuplicateHostTargetBeforeIo) {
    const BlockIdxType second_device_block = poolMalloc(*device_pool_);
    const BlockIdxType host_block          = poolMalloc(*host_pool_);
    ASSERT_NE(second_device_block, NULL_BLOCK_IDX);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    auto  strategy     = std::make_unique<CapturingDoneStrategy>();
    auto* strategy_ptr = strategy.get();
    per_rank_transfer_engine_->device_host_executor_->strategies_.clear();
    per_rank_transfer_engine_->device_host_executor_->strategies_.push_back(std::move(strategy));

    auto context = per_rank_transfer_engine_->submit(std::vector<TransferDescriptor>{
        makeDescriptor(Tier::DEVICE, Tier::HOST, {device_block_}, host_block),
        makeDescriptor(Tier::DEVICE, Tier::HOST, {second_device_block}, host_block),
    });
    ASSERT_NE(context, nullptr);
    context->waitDone();

    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_TRUE(strategy_ptr->plans.empty());

    host_pool_->free(host_block);
    device_pool_->free(second_device_block);
}

TEST(PerRankBlockTransferEngineEndpointTest, SameHostBlockIdInDifferentPhysicalPoolsDoesNotConflict) {
    ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";

    auto       first_host_pool     = makeHostPool(80, 2, true);
    auto       second_host_pool    = makeHostPool(80, 2, true);
    auto       first_device_pool   = makeDevicePool({{64, 16}}, 2, "endpoint_first_device");
    auto       second_device_pool  = makeDevicePool({{64, 16}}, 2, "endpoint_second_device");
    const auto first_host_block    = poolMalloc(*first_host_pool);
    const auto second_host_block   = poolMalloc(*second_host_pool);
    const auto first_device_block  = poolMalloc(*first_device_pool);
    const auto second_device_block = poolMalloc(*second_device_pool);
    ASSERT_EQ(first_host_block, second_host_block);

    auto first_group  = makeDeviceHostGroup(0, {first_device_pool}, first_host_pool, {makeGroupBase({0}, 64, 16)});
    auto second_group = makeDeviceHostGroup(1, {second_device_pool}, second_host_pool, {makeGroupBase({0}, 64, 16)});
    auto engine       = makeEngine({first_group, second_group});

    auto  strategy     = std::make_unique<CapturingDoneStrategy>();
    auto* strategy_ptr = strategy.get();
    engine->device_host_executor_->strategies_.clear();
    engine->device_host_executor_->strategies_.push_back(std::move(strategy));

    auto context = engine->submit(std::vector<TransferDescriptor>{
        makeDescriptor(Tier::DEVICE, Tier::HOST, {first_device_block}, first_host_block, NULL_BLOCK_IDX, 0),
        makeDescriptor(Tier::DEVICE, Tier::HOST, {second_device_block}, second_host_block, NULL_BLOCK_IDX, 1),
    });
    ASSERT_NE(context, nullptr);
    context->waitDone();

    EXPECT_TRUE(context->success()) << context->errorInfo().ToString();
    EXPECT_EQ(strategy_ptr->plans.size(), 1);

    first_host_pool->free(first_host_block);
    second_host_pool->free(second_host_block);
    first_device_pool->free(first_device_block);
    second_device_pool->free(second_device_block);
}

TEST(PerRankBlockTransferEngineEndpointTest, SameHostBlockInSharedPhysicalPoolConflictsAcrossGroupSets) {
    ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";

    auto       host_pool           = makeHostPool(80, 2, true);
    auto       first_device_pool   = makeDevicePool({{64, 16}}, 2, "endpoint_shared_first_device");
    auto       second_device_pool  = makeDevicePool({{64, 16}}, 2, "endpoint_shared_second_device");
    const auto host_block          = poolMalloc(*host_pool);
    const auto first_device_block  = poolMalloc(*first_device_pool);
    const auto second_device_block = poolMalloc(*second_device_pool);

    auto first_group  = makeDeviceHostGroup(0, {first_device_pool}, host_pool, {makeGroupBase({0}, 64, 16)});
    auto second_group = makeDeviceHostGroup(1, {second_device_pool}, host_pool, {makeGroupBase({0}, 64, 16)});
    auto engine       = makeEngine({first_group, second_group});

    auto  strategy     = std::make_unique<CapturingDoneStrategy>();
    auto* strategy_ptr = strategy.get();
    engine->device_host_executor_->strategies_.clear();
    engine->device_host_executor_->strategies_.push_back(std::move(strategy));

    auto context = engine->submit(std::vector<TransferDescriptor>{
        makeDescriptor(Tier::DEVICE, Tier::HOST, {first_device_block}, host_block, NULL_BLOCK_IDX, 0),
        makeDescriptor(Tier::DEVICE, Tier::HOST, {second_device_block}, host_block, NULL_BLOCK_IDX, 1),
    });
    ASSERT_NE(context, nullptr);
    context->waitDone();

    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_TRUE(strategy_ptr->plans.empty());

    host_pool->free(host_block);
    first_device_pool->free(first_device_block);
    second_device_pool->free(second_device_block);
}

TEST_F(PerRankBlockTransferEngineTest, SubmitVectorCompletesLogicalBatchWhenExecutorThrows) {
    const BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    per_rank_transfer_engine_->device_host_executor_->strategies_.clear();
    per_rank_transfer_engine_->device_host_executor_->strategies_.push_back(
        std::make_unique<ThrowingDeviceHostCopyStrategy>());

    auto context = per_rank_transfer_engine_->submit(std::vector<TransferDescriptor>{
        makeDescriptor(Tier::DEVICE, Tier::HOST, {device_block_}, host_block),
    });
    ASSERT_NE(context, nullptr);

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!context->done() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    ASSERT_TRUE(context->done()) << "executor exception left logical batch incomplete";
    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::EXECUTION_EXCEPTION);
    EXPECT_NE(context->errorInfo().ToString().find("injected device host copy exception"), std::string::npos);

    host_pool_->free(host_block);
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

    host_pool->free(host_block);
    shared_pool->free(block_a);
    shared_pool->free(block_b);
}

TEST_F(PerRankBlockTransferEngineTest, SubmitRejectsMissingRequiredBlocks) {
    expectStatus(per_rank_transfer_engine_,
                 makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, NULL_BLOCK_IDX),
                 TransferStatus::INVALID_ARGS);
    expectStatus(per_rank_transfer_engine_,
                 makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, NULL_BLOCK_IDX),
                 TransferStatus::INVALID_ARGS);
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

TEST_F(PerRankBlockTransferEngineTest, SubmitRejectsOutOfRangeHostBlock) {
    const BlockIdxType out_of_range = static_cast<BlockIdxType>(host_pool_->totalBlocksNum() + 1);
    expectStatus(per_rank_transfer_engine_,
                 makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, out_of_range),
                 TransferStatus::INVALID_ARGS);
    expectStatus(per_rank_transfer_engine_,
                 makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, out_of_range),
                 TransferStatus::INVALID_ARGS);
}

TEST_F(PerRankBlockTransferEngineTest, SubmitAcceptsValidUnallocatedDeviceBlock) {
    // Worker transfers may use a valid logical block ID without local allocator ownership.
    BlockIdxType freed_device_block = poolMalloc(*device_pool_);
    ASSERT_NE(freed_device_block, NULL_BLOCK_IDX);
    device_pool_->free(freed_device_block);
    std::vector<BlockIdxType> unallocated_device_blocks = {freed_device_block};

    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    expectStatus(per_rank_transfer_engine_,
                 makeDescriptor(Tier::DEVICE, Tier::HOST, unallocated_device_blocks, host_block),
                 TransferStatus::OK);
    expectStatus(per_rank_transfer_engine_,
                 makeDescriptor(Tier::HOST, Tier::DEVICE, unallocated_device_blocks, host_block),
                 TransferStatus::OK);

    host_pool_->free(host_block);
}

TEST_F(PerRankBlockTransferEngineTest, SubmitRejectsOutOfRangeDeviceBlock) {
    const BlockIdxType              out_of_range  = static_cast<BlockIdxType>(device_pool_->totalBlocksNum() + 1);
    const std::vector<BlockIdxType> device_blocks = {out_of_range};
    const BlockIdxType              host_block    = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    expectStatus(per_rank_transfer_engine_,
                 makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks, host_block),
                 TransferStatus::INVALID_ARGS);
    expectStatus(per_rank_transfer_engine_,
                 makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks, host_block),
                 TransferStatus::INVALID_ARGS);

    host_pool_->free(host_block);
}

TEST_F(PerRankBlockTransferEngineTest, SubmitRejectsMismatchedDeviceBlockCount) {
    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    const std::array<std::vector<BlockIdxType>, 2> wrong_blocks = {
        std::vector<BlockIdxType>{},
        std::vector<BlockIdxType>{device_block_, device_block_},
    };
    for (const auto& blocks : wrong_blocks) {
        SCOPED_TRACE(::testing::Message() << "block_count=" << blocks.size());
        expectStatus(per_rank_transfer_engine_,
                     makeDescriptor(Tier::DEVICE, Tier::HOST, blocks, host_block),
                     TransferStatus::INVALID_ARGS);
        expectStatus(per_rank_transfer_engine_,
                     makeDescriptor(Tier::HOST, Tier::DEVICE, blocks, host_block),
                     TransferStatus::INVALID_ARGS);
    }
    host_pool_->free(host_block);
}

TEST_F(PerRankBlockTransferEngineTest, SubmitRejectsInvalidGroupSetId) {
    for (size_t group_id : {std::numeric_limits<size_t>::max(), size_t{99}}) {
        SCOPED_TRACE(::testing::Message() << "group_id=" << group_id);
        expectStatus(per_rank_transfer_engine_,
                     makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, NULL_BLOCK_IDX, NULL_BLOCK_IDX, group_id),
                     TransferStatus::INVALID_ARGS);
    }
}

TEST_F(PerRankBlockTransferEngineTest, SubmitRejectsInvalidTierPairs) {
    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    const std::array<std::pair<Tier, Tier>, 8> invalid_pairs = {
        std::pair{Tier::NONE, Tier::HOST},
        std::pair{Tier::DEVICE, Tier::NONE},
        std::pair{Tier::DEVICE, Tier::DEVICE},
        std::pair{Tier::HOST, Tier::HOST},
        std::pair{Tier::DISK, Tier::DISK},
        std::pair{Tier::DEVICE, Tier::DISK},
        std::pair{Tier::DISK, Tier::DEVICE},
        std::pair{Tier::REMOTE, Tier::HOST},
    };
    for (const auto& [source, target] : invalid_pairs) {
        SCOPED_TRACE(::testing::Message() << tierName(source) << "->" << tierName(target));
        expectStatus(per_rank_transfer_engine_,
                     makeDescriptor(source, target, device_blocks_, host_block),
                     TransferStatus::INVALID_ARGS);
    }
    host_pool_->free(host_block);
}

TEST_F(PerRankBlockTransferEngineTest, SubmitRejectsIncompleteDeviceHostLayout) {
    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    const auto desc = makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block);

    auto missing_host_group  = makeDeviceHostGroup(0, {device_pool_}, nullptr, {makeGroupBase({0, 1, 2}, 100)});
    auto missing_host_engine = makeEngine({missing_host_group});
    expectStatus(missing_host_engine, desc, TransferStatus::INVALID_ARGS);
    host_pool_->free(host_block);
}

TEST_F(PerRankBlockTransferEngineTest, SubmitReturnsCompletedContextWithFinalStatus) {
    fillDeviceLayer(device_pool_, 0, device_block_, {0xAA});
    fillDeviceLayer(device_pool_, 1, device_block_, {0xBB});
    fillDeviceLayer(device_pool_, 2, device_block_, {0xCC});

    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    const std::array<std::pair<TransferDescriptor, TransferStatus>, 2> cases = {
        std::pair{makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block), TransferStatus::OK},
        std::pair{makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block, NULL_BLOCK_IDX, 99),
                  TransferStatus::INVALID_ARGS},
    };

    for (const auto& [desc, expected] : cases) {
        auto context = per_rank_transfer_engine_->submit(desc);
        ASSERT_NE(context, nullptr);
        EXPECT_TRUE(context->done());
        context->waitDone();
        EXPECT_EQ(context->success(), expected == TransferStatus::OK);
        if (expected == TransferStatus::OK) {
            EXPECT_TRUE(context->errorInfo().ok());
        } else {
            EXPECT_FALSE(context->errorInfo().ok());
            EXPECT_EQ(context->errorInfo().code(), ErrorCode::INVALID_PARAMS);
            EXPECT_FALSE(context->errorInfo().ToString().empty());
        }
    }

    host_pool_->free(host_block);
}

TEST_F(PerRankBlockTransferEngineTest, SubmitRejectsAllNullDeviceBlocks) {
    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    auto* host_data = static_cast<uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    std::memset(host_data, 0xA5, host_block_size_);

    const std::vector<BlockIdxType> all_null = {NULL_BLOCK_IDX};
    expectStatus(per_rank_transfer_engine_,
                 makeDescriptor(Tier::DEVICE, Tier::HOST, all_null, host_block),
                 TransferStatus::INVALID_ARGS);
    for (size_t i = 0; i < host_block_size_; ++i)
        EXPECT_EQ(host_data[i], 0xA5);

    expectStatus(per_rank_transfer_engine_,
                 makeDescriptor(Tier::HOST, Tier::DEVICE, all_null, host_block),
                 TransferStatus::INVALID_ARGS);
    host_pool_->free(host_block);
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

    ASSERT_TRUE(submitSucceeded(per_rank_transfer_engine_,
                                makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, host_block_1)));

    ASSERT_TRUE(submitSucceeded(per_rank_transfer_engine_,
                                makeDescriptor(Tier::HOST, Tier::DEVICE, second_device_blocks, host_block_2)));

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

    host_pool_->free(host_block_1);
    host_pool_->free(host_block_2);
    device_pool_->free(second_device_block);
}

class PerRankBlockTransferEngineMultiMemberTest: public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
        host_pool_ = makeHostPool(240, 4, true);
        pools_     = {
            makeDevicePool({{64, 16}}, 4, "per_rank_transfer_engine_multi_member_0"),
            makeDevicePool({{64, 16}}, 4, "per_rank_transfer_engine_multi_member_1"),
            makeDevicePool({{64, 16}}, 4, "per_rank_transfer_engine_multi_member_2"),
        };
        for (const auto& pool : pools_) {
            blocks_.push_back(poolMalloc(*pool));
            ASSERT_NE(blocks_.back(), NULL_BLOCK_IDX);
        }

        auto group =
            makeDeviceHostGroup(0,
                                pools_,
                                host_pool_,
                                {makeGroupBase({0}, 64, 16), makeGroupBase({0}, 64, 16), makeGroupBase({0}, 64, 16)});
        engine_     = makeEngine({group});
        host_block_ = poolMalloc(*host_pool_);
        ASSERT_NE(host_block_, NULL_BLOCK_IDX);
    }

    std::shared_ptr<HostBlockPool>              host_pool_;
    std::vector<DeviceBlockPoolPtr>             pools_;
    std::vector<BlockIdxType>                   blocks_;
    std::shared_ptr<PerRankBlockTransferEngine> engine_;
    BlockIdxType                                host_block_{NULL_BLOCK_IDX};
};

TEST_F(PerRankBlockTransferEngineMultiMemberTest, CompleteMultiMemberRoundTripPreservesOffsets) {
    fillDeviceLayer(pools_[0], 0, blocks_[0], {0xA1, 0xA2});
    fillDeviceLayer(pools_[1], 0, blocks_[1], {0xB1, 0xB2});
    fillDeviceLayer(pools_[2], 0, blocks_[2], {0xC1, 0xC2});
    auto* host_data = static_cast<uint8_t*>(host_pool_->blockBuffer(host_block_).addr);
    std::memset(host_data, 0xFF, 240);

    expectStatus(engine_,
                 makeDescriptor(Tier::DEVICE, Tier::HOST, {blocks_[0], blocks_[1], blocks_[2]}, host_block_),
                 TransferStatus::OK);
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

// Blocks the selected operation until released; payloads land in memory so blocked
// transfers can still be verified after release.
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
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, timeout, [this] { return blocked_; });
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        cv_.notify_all();
    }

private:
    void blockUntilReleased() {
        std::unique_lock<std::mutex> lock(mutex_);
        blocked_ = true;
        cv_.notify_all();
        cv_.wait(lock, [this] { return released_; });
    }

    const BlockOn           block_on_;
    std::mutex              mutex_;
    std::condition_variable cv_;
    bool                    blocked_{false};
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

class BlockingFirstDeviceHostCopyStrategy: public DeviceHostCopyStrategy {
public:
    StrategyResult tryExecute(const DeviceHostCopyPlan& plan, const DeviceHostCopyOptions& options) override {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            host_addresses_.push_back(plan.copy_tiles.front().host_addr);
            ++call_count_;
            cv_.notify_all();
            if (call_count_ == 1) {
                first_call_blocked_ = true;
                cv_.wait(lock, [this] { return released_; });
            }
        }
        return delegate_.tryExecute(plan, options);
    }

    bool waitUntilFirstCallBlocked(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, timeout, [this] { return first_call_blocked_; });
    }

    bool waitForCallCount(size_t expected, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, timeout, [this, expected] { return call_count_ >= expected; });
    }

    void release() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            released_ = true;
        }
        cv_.notify_all();
    }

    size_t callCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return call_count_;
    }

    std::vector<void*> hostAddresses() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return host_addresses_;
    }

private:
    GenericMultiCopyDeviceHostCopyStrategy delegate_;
    mutable std::mutex                     mutex_;
    std::condition_variable                cv_;
    size_t                                 call_count_{0};
    bool                                   first_call_blocked_{false};
    bool                                   released_{false};
    std::vector<void*>                     host_addresses_;
};

class FailingDeviceHostCopyStrategy: public DeviceHostCopyStrategy {
public:
    StrategyResult tryExecute(const DeviceHostCopyPlan&, const DeviceHostCopyOptions&) override {
        ++call_count_;
        return StrategyResult::failed(TransferStatus::DEVICE_IO_ERROR);
    }

    size_t callCount() const {
        return call_count_.load();
    }

private:
    std::atomic<size_t> call_count_{0};
};

TEST_F(PerRankBlockTransferEngineTest, DeviceToHostRunsTwoLogicalBatchesConcurrently) {
    using namespace std::chrono_literals;

    const BlockIdxType second_device_block = poolMalloc(*device_pool_);
    const BlockIdxType first_host_block    = poolMalloc(*host_pool_);
    const BlockIdxType second_host_block   = poolMalloc(*host_pool_);
    ASSERT_NE(second_device_block, NULL_BLOCK_IDX);
    ASSERT_NE(first_host_block, NULL_BLOCK_IDX);
    ASSERT_NE(second_host_block, NULL_BLOCK_IDX);

    auto  blocking_strategy = std::make_unique<BlockingFirstDeviceHostCopyStrategy>();
    auto* strategy          = blocking_strategy.get();
    per_rank_transfer_engine_->device_host_executor_->strategies_.clear();
    per_rank_transfer_engine_->device_host_executor_->strategies_.push_back(std::move(blocking_strategy));

    auto first = per_rank_transfer_engine_->submit(
        std::vector<TransferDescriptor>{makeDescriptor(Tier::DEVICE, Tier::HOST, {device_block_}, first_host_block)});
    ASSERT_TRUE(strategy->waitUntilFirstCallBlocked(5s));
    auto       second                      = per_rank_transfer_engine_->submit(std::vector<TransferDescriptor>{
        makeDescriptor(Tier::DEVICE, Tier::HOST, {second_device_block}, second_host_block)});
    const bool both_entered_before_release = strategy->waitForCallCount(2, 1s);

    strategy->release();
    first->waitDone();
    second->waitDone();

    EXPECT_TRUE(both_entered_before_release);
    EXPECT_TRUE(first->success()) << first->errorInfo().ToString();
    EXPECT_TRUE(second->success()) << second->errorInfo().ToString();

    device_pool_->free(second_device_block);
    host_pool_->free(first_host_block);
    host_pool_->free(second_host_block);
}

TEST_F(PerRankBlockTransferEngineTest, HostToDeviceRunsTwoLogicalBatchesConcurrently) {
    using namespace std::chrono_literals;

    const BlockIdxType second_device_block = poolMalloc(*device_pool_);
    const BlockIdxType first_host_block    = poolMalloc(*host_pool_);
    const BlockIdxType second_host_block   = poolMalloc(*host_pool_);
    ASSERT_NE(second_device_block, NULL_BLOCK_IDX);
    ASSERT_NE(first_host_block, NULL_BLOCK_IDX);
    ASSERT_NE(second_host_block, NULL_BLOCK_IDX);

    auto  blocking_strategy = std::make_unique<BlockingFirstDeviceHostCopyStrategy>();
    auto* strategy          = blocking_strategy.get();
    per_rank_transfer_engine_->device_host_executor_->strategies_.clear();
    per_rank_transfer_engine_->device_host_executor_->strategies_.push_back(std::move(blocking_strategy));

    auto first = per_rank_transfer_engine_->submit(
        std::vector<TransferDescriptor>{makeDescriptor(Tier::HOST, Tier::DEVICE, {device_block_}, first_host_block)});
    ASSERT_TRUE(strategy->waitUntilFirstCallBlocked(5s));
    auto       second                      = per_rank_transfer_engine_->submit(std::vector<TransferDescriptor>{
        makeDescriptor(Tier::HOST, Tier::DEVICE, {second_device_block}, second_host_block)});
    const bool both_entered_before_release = strategy->waitForCallCount(2, 1s);

    strategy->release();
    first->waitDone();
    second->waitDone();

    EXPECT_TRUE(both_entered_before_release);
    EXPECT_TRUE(first->success()) << first->errorInfo().ToString();
    EXPECT_TRUE(second->success()) << second->errorInfo().ToString();

    device_pool_->free(second_device_block);
    host_pool_->free(first_host_block);
    host_pool_->free(second_host_block);
}

TEST_F(PerRankBlockTransferEngineTest, InFlightWriteConflictIsRejectedAndReleasedAfterCompletion) {
    const BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    auto  blocking_strategy = std::make_unique<BlockingFirstDeviceHostCopyStrategy>();
    auto* strategy          = blocking_strategy.get();
    per_rank_transfer_engine_->device_host_executor_->strategies_.clear();
    per_rank_transfer_engine_->device_host_executor_->strategies_.push_back(std::move(blocking_strategy));

    const auto descriptor = makeDescriptor(Tier::HOST, Tier::DEVICE, {device_block_}, host_block);
    auto       first      = per_rank_transfer_engine_->submit(std::vector<TransferDescriptor>{descriptor});
    ASSERT_TRUE(strategy->waitUntilFirstCallBlocked(std::chrono::seconds(5)));

    auto       second = per_rank_transfer_engine_->submit(std::vector<TransferDescriptor>{descriptor});
    const bool second_completed_before_release = second->done();

    strategy->release();
    first->waitDone();
    second->waitDone();

    EXPECT_TRUE(first->success()) << first->errorInfo().ToString();
    EXPECT_TRUE(second_completed_before_release);
    EXPECT_FALSE(second->success());
    EXPECT_EQ(second->errorInfo().code(), ErrorCode::EXECUTION_EXCEPTION);
    EXPECT_NE(second->errorInfo().ToString().find("RESOURCE_EXHAUSTED: transfer endpoint conflict"), std::string::npos);
    EXPECT_NE(second->errorInfo().ToString().find("direction=HOST->DEVICE"), std::string::npos);

    auto third = per_rank_transfer_engine_->submit(std::vector<TransferDescriptor>{descriptor});
    third->waitDone();
    EXPECT_TRUE(third->success()) << third->errorInfo().ToString();

    host_pool_->free(host_block);
}

TEST_F(PerRankBlockTransferEngineTest, InFlightReadReadWithDistinctTargetsIsAllowed) {
    const BlockIdxType first_host_block  = poolMalloc(*host_pool_);
    const BlockIdxType second_host_block = poolMalloc(*host_pool_);
    ASSERT_NE(first_host_block, NULL_BLOCK_IDX);
    ASSERT_NE(second_host_block, NULL_BLOCK_IDX);

    auto  blocking_strategy = std::make_unique<BlockingFirstDeviceHostCopyStrategy>();
    auto* strategy          = blocking_strategy.get();
    per_rank_transfer_engine_->device_host_executor_->strategies_.clear();
    per_rank_transfer_engine_->device_host_executor_->strategies_.push_back(std::move(blocking_strategy));

    auto first = per_rank_transfer_engine_->submit(
        std::vector<TransferDescriptor>{makeDescriptor(Tier::DEVICE, Tier::HOST, {device_block_}, first_host_block)});
    ASSERT_TRUE(strategy->waitUntilFirstCallBlocked(std::chrono::seconds(5)));
    auto second = per_rank_transfer_engine_->submit(
        std::vector<TransferDescriptor>{makeDescriptor(Tier::DEVICE, Tier::HOST, {device_block_}, second_host_block)});

    strategy->release();
    first->waitDone();
    second->waitDone();

    EXPECT_TRUE(first->success()) << first->errorInfo().ToString();
    EXPECT_TRUE(second->success()) << second->errorInfo().ToString();

    host_pool_->free(first_host_block);
    host_pool_->free(second_host_block);
}

TEST_F(PerRankBlockTransferEngineTest, InFlightReadPreventsWriteToSameEndpoint) {
    const BlockIdxType read_target_host  = poolMalloc(*host_pool_);
    const BlockIdxType write_source_host = poolMalloc(*host_pool_);
    ASSERT_NE(read_target_host, NULL_BLOCK_IDX);
    ASSERT_NE(write_source_host, NULL_BLOCK_IDX);

    auto  blocking_strategy = std::make_unique<BlockingFirstDeviceHostCopyStrategy>();
    auto* strategy          = blocking_strategy.get();
    per_rank_transfer_engine_->device_host_executor_->strategies_.clear();
    per_rank_transfer_engine_->device_host_executor_->strategies_.push_back(std::move(blocking_strategy));

    auto reader = per_rank_transfer_engine_->submit(
        std::vector<TransferDescriptor>{makeDescriptor(Tier::DEVICE, Tier::HOST, {device_block_}, read_target_host)});
    ASSERT_TRUE(strategy->waitUntilFirstCallBlocked(std::chrono::seconds(5)));
    auto writer = per_rank_transfer_engine_->submit(
        std::vector<TransferDescriptor>{makeDescriptor(Tier::HOST, Tier::DEVICE, {device_block_}, write_source_host)});
    const bool writer_completed_before_release = writer->done();

    strategy->release();
    reader->waitDone();
    writer->waitDone();

    EXPECT_TRUE(reader->success()) << reader->errorInfo().ToString();
    EXPECT_TRUE(writer_completed_before_release);
    EXPECT_FALSE(writer->success());
    EXPECT_NE(writer->errorInfo().ToString().find("RESOURCE_EXHAUSTED: transfer endpoint conflict"), std::string::npos);

    host_pool_->free(read_target_host);
    host_pool_->free(write_source_host);
}

TEST_F(PerRankBlockTransferEngineTest, InFlightWritePreventsReadFromSameEndpoint) {
    const BlockIdxType write_source_host = poolMalloc(*host_pool_);
    const BlockIdxType read_target_host  = poolMalloc(*host_pool_);
    ASSERT_NE(write_source_host, NULL_BLOCK_IDX);
    ASSERT_NE(read_target_host, NULL_BLOCK_IDX);

    auto  blocking_strategy = std::make_unique<BlockingFirstDeviceHostCopyStrategy>();
    auto* strategy          = blocking_strategy.get();
    per_rank_transfer_engine_->device_host_executor_->strategies_.clear();
    per_rank_transfer_engine_->device_host_executor_->strategies_.push_back(std::move(blocking_strategy));

    auto writer = per_rank_transfer_engine_->submit(
        std::vector<TransferDescriptor>{makeDescriptor(Tier::HOST, Tier::DEVICE, {device_block_}, write_source_host)});
    ASSERT_TRUE(strategy->waitUntilFirstCallBlocked(std::chrono::seconds(5)));
    auto reader = per_rank_transfer_engine_->submit(
        std::vector<TransferDescriptor>{makeDescriptor(Tier::DEVICE, Tier::HOST, {device_block_}, read_target_host)});
    const bool reader_completed_before_release = reader->done();

    strategy->release();
    writer->waitDone();
    reader->waitDone();

    EXPECT_TRUE(writer->success()) << writer->errorInfo().ToString();
    EXPECT_TRUE(reader_completed_before_release);
    EXPECT_FALSE(reader->success());
    EXPECT_NE(reader->errorInfo().ToString().find("RESOURCE_EXHAUSTED: transfer endpoint conflict"), std::string::npos);

    host_pool_->free(write_source_host);
    host_pool_->free(read_target_host);
}

TEST_F(PerRankBlockTransferEngineTest, ReservationIsReusableAfterTransferFailure) {
    const BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    const auto descriptor = makeDescriptor(Tier::HOST, Tier::DEVICE, {device_block_}, host_block);

    per_rank_transfer_engine_->device_host_executor_->strategies_.clear();
    per_rank_transfer_engine_->device_host_executor_->strategies_.push_back(
        std::make_unique<CapturingFailingStrategy>());
    auto failed = per_rank_transfer_engine_->submit(std::vector<TransferDescriptor>{descriptor});
    failed->waitDone();
    ASSERT_FALSE(failed->success());

    per_rank_transfer_engine_->device_host_executor_->strategies_.clear();
    per_rank_transfer_engine_->device_host_executor_->strategies_.push_back(std::make_unique<CapturingDoneStrategy>());
    auto retried = per_rank_transfer_engine_->submit(std::vector<TransferDescriptor>{descriptor});
    retried->waitDone();
    EXPECT_TRUE(retried->success()) << retried->errorInfo().ToString();

    host_pool_->free(host_block);
}

TEST(PerRankBlockTransferEngineIntegrationTest, HostAndDiskLoadsCannotWriteSameDeviceTarget) {
    ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
    TempDirGuard     temp_dir("per_rank_host_disk_same_device_target");
    constexpr size_t payload_bytes = 80;
    auto             host_pool     = makeHostPool(payload_bytes, 2, true);
    auto             disk_pool     = makeDiskPool(payload_bytes, 2, temp_dir.path);
    auto             device_pool   = makeDevicePool({{64, 16}}, 2, "host_disk_same_device_target");
    const auto       host_block    = poolMalloc(*host_pool);
    const auto       disk_block    = poolMalloc(*disk_pool);
    const auto       device_block  = poolMalloc(*device_pool);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);
    ASSERT_NE(device_block, NULL_BLOCK_IDX);

    auto group  = makeDeviceHostGroup(0, {device_pool}, host_pool, {makeGroupBase({0}, 64, 16)}, disk_pool);
    auto engine = makeEngine({group});
    ASSERT_TRUE(submitSucceeded(engine, makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, disk_block)));

    auto  blocking_strategy = std::make_unique<BlockingFirstDeviceHostCopyStrategy>();
    auto* strategy          = blocking_strategy.get();
    engine->device_host_executor_->strategies_.clear();
    engine->device_host_executor_->strategies_.push_back(std::move(blocking_strategy));

    const auto host_descriptor = makeDescriptor(Tier::HOST, Tier::DEVICE, {device_block}, host_block, NULL_BLOCK_IDX);
    const auto disk_descriptor = makeDescriptor(Tier::DISK, Tier::DEVICE, {device_block}, NULL_BLOCK_IDX, disk_block);
    auto       host_load       = engine->submit(std::vector<TransferDescriptor>{host_descriptor});
    ASSERT_TRUE(strategy->waitUntilFirstCallBlocked(std::chrono::seconds(5)));
    auto       disk_load                          = engine->submit(std::vector<TransferDescriptor>{disk_descriptor});
    const bool disk_load_completed_before_release = disk_load->done();

    strategy->release();
    host_load->waitDone();
    disk_load->waitDone();

    EXPECT_TRUE(host_load->success()) << host_load->errorInfo().ToString();
    EXPECT_TRUE(disk_load_completed_before_release);
    EXPECT_FALSE(disk_load->success());
    EXPECT_NE(disk_load->errorInfo().ToString().find("RESOURCE_EXHAUSTED: transfer endpoint conflict"),
              std::string::npos);

    auto completed_disk_load = engine->submit(std::vector<TransferDescriptor>{disk_descriptor});
    completed_disk_load->waitDone();
    ASSERT_TRUE(completed_disk_load->success()) << completed_disk_load->errorInfo().ToString();
    auto reused_target = engine->submit(std::vector<TransferDescriptor>{host_descriptor});
    reused_target->waitDone();
    EXPECT_TRUE(reused_target->success()) << reused_target->errorInfo().ToString();

    host_pool->free(host_block);
    disk_pool->free(disk_block);
    device_pool->free(device_block);
}

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

TEST(PerRankBlockTransferEngineIntegrationTest, DiskToDeviceVectorSlicesByOneOfTwoStagingPools) {
    ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
    TempDirGuard     temp_dir("per_rank_disk_to_device_vector");
    constexpr size_t payload_bytes = 80;
    auto             owned_io      = std::make_unique<DirectAlignmentDiskBlockIO>();
    auto*            direct_io     = owned_io.get();
    auto disk_pool = makeDiskPool(payload_bytes, 5, temp_dir.path, std::move(owned_io), "disk_to_device_vector", false);
    auto device_pool = makeDevicePool({{64, 16}}, 5, "per_rank_disk_to_device_vector_device");
    auto group       = makeDeviceHostGroup(0, {device_pool}, nullptr, {makeGroupBase({0}, 64, 16)}, disk_pool);
    auto engine      = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group},
                                                               DeviceHostCopyOptions{},
                                                               /*device_disk_staging_block_count=*/4,
                                                               /*max_descriptors_per_transfer_batch=*/1);

    std::vector<BlockIdxType>         device_blocks;
    std::vector<BlockIdxType>         disk_blocks;
    std::vector<std::vector<uint8_t>> expected;
    std::vector<TransferDescriptor>   descriptors;
    for (size_t index = 0; index < 5; ++index) {
        const BlockIdxType device_block = poolMalloc(*device_pool);
        const BlockIdxType disk_block   = poolMalloc(*disk_pool);
        ASSERT_NE(device_block, NULL_BLOCK_IDX);
        ASSERT_NE(disk_block, NULL_BLOCK_IDX);
        device_blocks.push_back(device_block);
        disk_blocks.push_back(disk_block);
        fillDeviceLayer(
            device_pool, 0, device_block, {static_cast<uint8_t>(0x20 + index), static_cast<uint8_t>(0x70 + index)});
        expected.push_back(readDeviceLayer(device_pool, 0, device_block));
        ASSERT_TRUE(submitSucceeded(
            engine, makeDescriptor(Tier::DEVICE, Tier::DISK, {device_block}, NULL_BLOCK_IDX, disk_block, 0)));
        fillDeviceLayer(device_pool, 0, device_block, {0x00, 0x00});
        descriptors.push_back(makeDescriptor(Tier::DISK, Tier::DEVICE, {device_block}, NULL_BLOCK_IDX, disk_block, 0));
    }

    const std::shared_ptr<AsyncContext> context = engine->submit(descriptors);
    ASSERT_NE(context, nullptr);
    context->waitDone();
    ASSERT_TRUE(context->success()) << context->errorInfo().ToString();
    EXPECT_EQ(direct_io->readBatchSizes(), (std::vector<size_t>{2, 2, 1}));
    for (size_t index = 0; index < device_blocks.size(); ++index) {
        EXPECT_EQ(readDeviceLayer(device_pool, 0, device_blocks[index]), expected[index]);
    }
}

TEST(PerRankBlockTransferEngineIntegrationTest, DiskToDeviceOverlapsNextStageOneWithCurrentStageTwo) {
    ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
    TempDirGuard     temp_dir("per_rank_disk_to_device_overlap");
    constexpr size_t payload_bytes = 80;
    auto             owned_io      = std::make_unique<DirectAlignmentDiskBlockIO>();
    auto*            direct_io     = owned_io.get();
    auto             disk_pool =
        makeDiskPool(payload_bytes, 4, temp_dir.path, std::move(owned_io), "disk_to_device_overlap", false);
    auto device_pool = makeDevicePool({{64, 16}}, 4, "per_rank_disk_to_device_overlap_device");
    auto group       = makeDeviceHostGroup(0, {device_pool}, nullptr, {makeGroupBase({0}, 64, 16)}, disk_pool);
    auto engine      = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group},
                                                               DeviceHostCopyOptions{},
                                                               /*device_disk_staging_block_count=*/4);

    std::vector<TransferDescriptor> descriptors;
    for (size_t index = 0; index < 4; ++index) {
        const BlockIdxType device_block = poolMalloc(*device_pool);
        const BlockIdxType disk_block   = poolMalloc(*disk_pool);
        ASSERT_NE(device_block, NULL_BLOCK_IDX);
        ASSERT_NE(disk_block, NULL_BLOCK_IDX);
        fillDeviceLayer(
            device_pool, 0, device_block, {static_cast<uint8_t>(0x30 + index), static_cast<uint8_t>(0x60 + index)});
        ASSERT_TRUE(submitSucceeded(
            engine, makeDescriptor(Tier::DEVICE, Tier::DISK, {device_block}, NULL_BLOCK_IDX, disk_block, 0)));
        fillDeviceLayer(device_pool, 0, device_block, {0x00, 0x00});
        descriptors.push_back(makeDescriptor(Tier::DISK, Tier::DEVICE, {device_block}, NULL_BLOCK_IDX, disk_block, 0));
    }

    auto  blocking_strategy = std::make_unique<BlockingFirstDeviceHostCopyStrategy>();
    auto* strategy          = blocking_strategy.get();
    engine->device_host_executor_->strategies_.clear();
    engine->device_host_executor_->strategies_.push_back(std::move(blocking_strategy));

    const std::shared_ptr<AsyncContext> context = engine->submit(descriptors);
    ASSERT_TRUE(strategy->waitUntilFirstCallBlocked(std::chrono::seconds(5)));
    EXPECT_TRUE(direct_io->waitForReadBatchCount(2, std::chrono::seconds(5)))
        << "slice 2 stage1 should run while slice 1 stage2 is blocked";
    strategy->release();
    context->waitDone();
    ASSERT_TRUE(context->success()) << context->errorInfo().ToString();
    EXPECT_EQ(direct_io->readBatchSizes(), (std::vector<size_t>{2, 2}));
    ASSERT_EQ(strategy->callCount(), 2u);
    const std::vector<void*> addresses = strategy->hostAddresses();
    ASSERT_EQ(addresses.size(), 2u);
    EXPECT_NE(addresses[0], addresses[1]);
}

TEST(PerRankBlockTransferEngineIntegrationTest, DiskToDeviceStageOneFailureFailsLogicalBatchAndStopsLaterSlices) {
    ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
    TempDirGuard     temp_dir("per_rank_disk_to_device_stage1_failure");
    constexpr size_t payload_bytes = 80;
    auto             owned_io      = std::make_unique<DirectAlignmentDiskBlockIO>();
    auto*            direct_io     = owned_io.get();
    auto             disk_pool =
        makeDiskPool(payload_bytes, 5, temp_dir.path, std::move(owned_io), "disk_to_device_stage1_fail", false);
    auto device_pool = makeDevicePool({{64, 16}}, 5, "per_rank_disk_to_device_stage1_fail_device");
    auto group       = makeDeviceHostGroup(0, {device_pool}, nullptr, {makeGroupBase({0}, 64, 16)}, disk_pool);
    auto engine      = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group},
                                                               DeviceHostCopyOptions{},
                                                               /*device_disk_staging_block_count=*/4);

    std::vector<TransferDescriptor> descriptors;
    for (size_t index = 0; index < 5; ++index) {
        const BlockIdxType device_block = poolMalloc(*device_pool);
        const BlockIdxType disk_block   = poolMalloc(*disk_pool);
        ASSERT_NE(device_block, NULL_BLOCK_IDX);
        ASSERT_NE(disk_block, NULL_BLOCK_IDX);
        fillDeviceLayer(
            device_pool, 0, device_block, {static_cast<uint8_t>(0x40 + index), static_cast<uint8_t>(0x50 + index)});
        ASSERT_TRUE(submitSucceeded(
            engine, makeDescriptor(Tier::DEVICE, Tier::DISK, {device_block}, NULL_BLOCK_IDX, disk_block, 0)));
        descriptors.push_back(makeDescriptor(Tier::DISK, Tier::DEVICE, {device_block}, NULL_BLOCK_IDX, disk_block, 0));
    }
    direct_io->failReadBatch(2);

    const std::shared_ptr<AsyncContext> context = engine->submit(descriptors);
    context->waitDone();
    EXPECT_FALSE(context->success());
    EXPECT_NE(context->errorInfo().ToString().find("descriptor_range=[2,4)"), std::string::npos);
    EXPECT_EQ(direct_io->readBatchSizes(), (std::vector<size_t>{2, 2}));
}

TEST(PerRankBlockTransferEngineIntegrationTest, DiskToDeviceStageTwoFailureFailsLogicalBatchAndSkipsQueuedStageTwo) {
    ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
    TempDirGuard     temp_dir("per_rank_disk_to_device_stage2_failure");
    constexpr size_t payload_bytes = 80;
    auto             owned_io      = std::make_unique<DirectAlignmentDiskBlockIO>();
    auto*            direct_io     = owned_io.get();
    auto             disk_pool =
        makeDiskPool(payload_bytes, 4, temp_dir.path, std::move(owned_io), "disk_to_device_stage2_fail", false);
    auto device_pool = makeDevicePool({{64, 16}}, 4, "per_rank_disk_to_device_stage2_fail_device");
    auto group       = makeDeviceHostGroup(0, {device_pool}, nullptr, {makeGroupBase({0}, 64, 16)}, disk_pool);
    auto engine      = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group},
                                                               DeviceHostCopyOptions{},
                                                               /*device_disk_staging_block_count=*/4);

    std::vector<TransferDescriptor> descriptors;
    for (size_t index = 0; index < 4; ++index) {
        const BlockIdxType device_block = poolMalloc(*device_pool);
        const BlockIdxType disk_block   = poolMalloc(*disk_pool);
        ASSERT_NE(device_block, NULL_BLOCK_IDX);
        ASSERT_NE(disk_block, NULL_BLOCK_IDX);
        fillDeviceLayer(
            device_pool, 0, device_block, {static_cast<uint8_t>(0x10 + index), static_cast<uint8_t>(0x20 + index)});
        ASSERT_TRUE(submitSucceeded(
            engine, makeDescriptor(Tier::DEVICE, Tier::DISK, {device_block}, NULL_BLOCK_IDX, disk_block, 0)));
        descriptors.push_back(makeDescriptor(Tier::DISK, Tier::DEVICE, {device_block}, NULL_BLOCK_IDX, disk_block, 0));
    }

    auto  failing_strategy = std::make_unique<FailingDeviceHostCopyStrategy>();
    auto* strategy         = failing_strategy.get();
    engine->device_host_executor_->strategies_.clear();
    engine->device_host_executor_->strategies_.push_back(std::move(failing_strategy));

    const std::shared_ptr<AsyncContext> context = engine->submit(descriptors);
    context->waitDone();
    EXPECT_FALSE(context->success());
    EXPECT_NE(context->errorInfo().ToString().find("descriptor_range=[0,2)"), std::string::npos);
    EXPECT_EQ(strategy->callCount(), 1u);
    EXPECT_LE(direct_io->readBatchSizes().size(), 2u);
}

TEST(PerRankBlockTransferEngineIntegrationTest, DiskToDeviceTimesOutOnlyWaitingForAFreeStagingSlice) {
    ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
    TempDirGuard     temp_dir("per_rank_disk_to_device_staging_timeout");
    constexpr size_t payload_bytes = 80;
    auto             owned_io      = std::make_unique<DirectAlignmentDiskBlockIO>();
    auto*            direct_io     = owned_io.get();
    auto             disk_pool =
        makeDiskPool(payload_bytes, 6, temp_dir.path, std::move(owned_io), "disk_to_device_timeout", false);
    auto device_pool = makeDevicePool({{64, 16}}, 6, "per_rank_disk_to_device_timeout_device");
    auto group       = makeDeviceHostGroup(0, {device_pool}, nullptr, {makeGroupBase({0}, 64, 16)}, disk_pool);
    auto engine      = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group},
                                                               DeviceHostCopyOptions{},
                                                               /*device_disk_staging_block_count=*/4);

    std::vector<TransferDescriptor> descriptors;
    for (size_t index = 0; index < 6; ++index) {
        const BlockIdxType device_block = poolMalloc(*device_pool);
        const BlockIdxType disk_block   = poolMalloc(*disk_pool);
        ASSERT_NE(device_block, NULL_BLOCK_IDX);
        ASSERT_NE(disk_block, NULL_BLOCK_IDX);
        fillDeviceLayer(
            device_pool, 0, device_block, {static_cast<uint8_t>(0x50 + index), static_cast<uint8_t>(0x70 + index)});
        ASSERT_TRUE(submitSucceeded(
            engine, makeDescriptor(Tier::DEVICE, Tier::DISK, {device_block}, NULL_BLOCK_IDX, disk_block, 0)));
        descriptors.push_back(makeDescriptor(Tier::DISK, Tier::DEVICE, {device_block}, NULL_BLOCK_IDX, disk_block, 0));
    }

    auto  blocking_strategy = std::make_unique<BlockingFirstDeviceHostCopyStrategy>();
    auto* strategy          = blocking_strategy.get();
    engine->device_host_executor_->strategies_.clear();
    engine->device_host_executor_->strategies_.push_back(std::move(blocking_strategy));

    const std::shared_ptr<AsyncContext> context = engine->submit(descriptors);
    ASSERT_TRUE(strategy->waitUntilFirstCallBlocked(std::chrono::seconds(5)));
    ASSERT_TRUE(direct_io->waitForReadBatchCount(2, std::chrono::seconds(5)));
    std::this_thread::sleep_for(std::chrono::milliseconds(1200));
    strategy->release();
    context->waitDone();
    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::DEADLINE_EXCEEDED);
    EXPECT_EQ(direct_io->readBatchSizes(), (std::vector<size_t>{2, 2}));
}

TEST(PerRankBlockTransferEngineIntegrationTest, DiskToDeviceStageOneQueueRejectionCompletesImmediately) {
    ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
    TempDirGuard     temp_dir("per_rank_disk_to_device_stage1_queue_rejection");
    constexpr size_t payload_bytes = 80;
    auto disk_pool   = makeDiskPool(payload_bytes, 1, temp_dir.path, std::make_unique<DirectAlignmentDiskBlockIO>());
    auto device_pool = makeDevicePool({{64, 16}}, 1, "per_rank_disk_to_device_stage1_queue_device");
    const BlockIdxType device_block = poolMalloc(*device_pool);
    const BlockIdxType disk_block   = poolMalloc(*disk_pool);
    ASSERT_NE(device_block, NULL_BLOCK_IDX);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);
    auto group  = makeDeviceHostGroup(0, {device_pool}, nullptr, {makeGroupBase({0}, 64, 16)}, disk_pool);
    auto engine = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group},
                                                               DeviceHostCopyOptions{},
                                                               /*device_disk_staging_block_count=*/4);
    engine->device_disk_executor_->disk_to_staging_task_pool_->shutdown();

    const auto context = engine->submit(std::vector<TransferDescriptor>{
        makeDescriptor(Tier::DISK, Tier::DEVICE, {device_block}, NULL_BLOCK_IDX, disk_block, 0)});
    ASSERT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_NE(context->errorInfo().ToString().find("RESOURCE_EXHAUSTED"), std::string::npos);
}

TEST(PerRankBlockTransferEngineIntegrationTest, DiskToDeviceStageTwoQueueRejectionFailsLogicalBatch) {
    ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
    TempDirGuard     temp_dir("per_rank_disk_to_device_stage2_queue_rejection");
    constexpr size_t payload_bytes = 80;
    auto disk_pool   = makeDiskPool(payload_bytes, 1, temp_dir.path, std::make_unique<DirectAlignmentDiskBlockIO>());
    auto device_pool = makeDevicePool({{64, 16}}, 1, "per_rank_disk_to_device_stage2_queue_device");
    const BlockIdxType device_block = poolMalloc(*device_pool);
    const BlockIdxType disk_block   = poolMalloc(*disk_pool);
    ASSERT_NE(device_block, NULL_BLOCK_IDX);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);
    auto group  = makeDeviceHostGroup(0, {device_pool}, nullptr, {makeGroupBase({0}, 64, 16)}, disk_pool);
    auto engine = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group},
                                                               DeviceHostCopyOptions{},
                                                               /*device_disk_staging_block_count=*/4);
    engine->device_disk_executor_->staging_to_device_task_pool_->shutdown();

    const auto context = engine->submit(std::vector<TransferDescriptor>{
        makeDescriptor(Tier::DISK, Tier::DEVICE, {device_block}, NULL_BLOCK_IDX, disk_block, 0)});
    context->waitDone();
    EXPECT_FALSE(context->success());
    EXPECT_NE(context->errorInfo().ToString().find("RESOURCE_EXHAUSTED"), std::string::npos);
}

TEST(TransientHostStagingPoolTest, PageableBackingServesLeasesWhenPinningDisabled) {
    using Pool = DeviceDiskTransferExecutor::TransientHostStagingPool;
    Pool pool(1, 4096, /*try_pin_memory=*/false);

    auto lease = pool.tryAcquire();
    ASSERT_TRUE(lease.has_value());
    const auto view = lease->view(64);
    ASSERT_NE(view.base, nullptr);
    EXPECT_EQ(view.payload_bytes, 64u);
    EXPECT_EQ(view.capacity_bytes, 4096u);
    std::memset(view.base, 0xAB, view.payload_bytes);

    EXPECT_FALSE(pool.tryAcquire().has_value());
    lease.reset();
    EXPECT_TRUE(pool.tryAcquire().has_value());
}

TEST(PerRankBlockTransferEngineIntegrationTest, DeviceDiskStageFailureShortCircuitsAndReleasesStaging) {
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

    expectStatus(engine,
                 makeDescriptor(Tier::DEVICE, Tier::DISK, {device_block}, NULL_BLOCK_IDX, disk_block, 0),
                 TransferStatus::DISK_IO_ERROR);
    status_io->setStatus(DiskBlockIOStatus::OK);
    expectStatus(engine,
                 makeDescriptor(Tier::DEVICE, Tier::DISK, {device_block}, NULL_BLOCK_IDX, disk_block, 0),
                 TransferStatus::OK);
}

TEST(PerRankBlockTransferEngineIntegrationTest, DeviceDiskStagingTransientExhaustionWaitsAndSucceeds) {
    ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
    TempDirGuard     temp_dir("per_rank_transient_exhaustion");
    constexpr size_t payload_bytes = 80;
    auto             blocking_io   = std::make_unique<BlockingDiskBlockIO>(BlockingDiskBlockIO::BlockOn::WRITE);
    auto*            io            = blocking_io.get();
    auto             disk_pool     = makeDiskPool(payload_bytes, 2, temp_dir.path, std::move(blocking_io));
    auto             device_pool   = makeDevicePool({{64, 16}}, 2, "per_rank_transient_exhaustion_device");
    auto             first_block   = poolMalloc(*device_pool);
    auto             second_block  = poolMalloc(*device_pool);
    auto             first_slot    = poolMalloc(*disk_pool);
    auto             second_slot   = poolMalloc(*disk_pool);
    ASSERT_NE(first_block, NULL_BLOCK_IDX);
    ASSERT_NE(second_block, NULL_BLOCK_IDX);
    ASSERT_NE(first_slot, NULL_BLOCK_IDX);
    ASSERT_NE(second_slot, NULL_BLOCK_IDX);

    auto group  = makeDeviceHostGroup(0, {device_pool}, nullptr, {makeGroupBase({0}, 64, 16)}, disk_pool);
    auto engine = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group},
                                                               DeviceHostCopyOptions{},
                                                               /*device_disk_staging_block_count=*/1);

    std::atomic<bool> first_submit_ok{false};
    std::thread       writer([&] {
        first_submit_ok = submitSucceeded(
            engine, makeDescriptor(Tier::DEVICE, Tier::DISK, {first_block}, NULL_BLOCK_IDX, first_slot, 0));
    });
    BlockingIOGuard   writer_guard(*io, writer);
    ASSERT_TRUE(io->waitUntilBlocked(std::chrono::seconds(5)));

    // The first transfer holds the only lease until the releaser fires, so the second
    // transfer must wait for it instead of failing fast.
    std::thread releaser([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        io->release();
    });
    expectStatus(engine,
                 makeDescriptor(Tier::DEVICE, Tier::DISK, {second_block}, NULL_BLOCK_IDX, second_slot, 0),
                 TransferStatus::OK);

    releaser.join();
    writer_guard.releaseAndJoin();
    EXPECT_TRUE(first_submit_ok.load());
}

TEST(PerRankBlockTransferEngineIntegrationTest, DeviceDiskStagingPersistentExhaustionTimesOut) {
    ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
    TempDirGuard     temp_dir("per_rank_persistent_exhaustion");
    constexpr size_t payload_bytes = 80;
    auto             blocking_io   = std::make_unique<BlockingDiskBlockIO>(BlockingDiskBlockIO::BlockOn::WRITE);
    auto*            io            = blocking_io.get();
    auto             disk_pool     = makeDiskPool(payload_bytes, 2, temp_dir.path, std::move(blocking_io));
    auto             device_pool   = makeDevicePool({{64, 16}}, 2, "per_rank_persistent_exhaustion_device");
    auto             first_block   = poolMalloc(*device_pool);
    auto             second_block  = poolMalloc(*device_pool);
    auto             first_slot    = poolMalloc(*disk_pool);
    auto             second_slot   = poolMalloc(*disk_pool);
    ASSERT_NE(first_block, NULL_BLOCK_IDX);
    ASSERT_NE(first_slot, NULL_BLOCK_IDX);
    ASSERT_NE(second_block, NULL_BLOCK_IDX);
    ASSERT_NE(second_slot, NULL_BLOCK_IDX);

    auto group  = makeDeviceHostGroup(0, {device_pool}, nullptr, {makeGroupBase({0}, 64, 16)}, disk_pool);
    auto engine = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group},
                                                               DeviceHostCopyOptions{},
                                                               /*device_disk_staging_block_count=*/1);

    std::atomic<bool> first_submit_ok{false};
    std::thread       writer([&] {
        first_submit_ok = submitSucceeded(
            engine, makeDescriptor(Tier::DEVICE, Tier::DISK, {first_block}, NULL_BLOCK_IDX, first_slot, 0));
    });
    BlockingIOGuard   writer_guard(*io, writer);
    ASSERT_TRUE(io->waitUntilBlocked(std::chrono::seconds(5)));

    const auto wait_start = std::chrono::steady_clock::now();
    expectStatus(engine,
                 makeDescriptor(Tier::DEVICE, Tier::DISK, {second_block}, NULL_BLOCK_IDX, second_slot, 0),
                 TransferStatus::RESOURCE_EXHAUSTED);
    const auto elapsed = std::chrono::steady_clock::now() - wait_start;
    // Internal staging acquire budget is fixed at 1000 ms.
    EXPECT_GE(elapsed, std::chrono::milliseconds(900));
    EXPECT_LT(elapsed, std::chrono::seconds(5));

    writer_guard.releaseAndJoin();
    EXPECT_TRUE(first_submit_ok.load());
    // Timeout must not leak the lease.
    expectStatus(engine,
                 makeDescriptor(Tier::DEVICE, Tier::DISK, {second_block}, NULL_BLOCK_IDX, second_slot, 0),
                 TransferStatus::OK);
}

TEST(PerRankBlockTransferEngineIntegrationTest, DiskDeviceStagingTransientExhaustionWaitsAndDeliversPayload) {
    ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
    TempDirGuard     temp_dir("per_rank_disk2d_transient_exhaustion");
    constexpr size_t payload_bytes = 80;
    auto             blocking_io   = std::make_unique<BlockingDiskBlockIO>(BlockingDiskBlockIO::BlockOn::READ);
    auto*            io            = blocking_io.get();
    auto             disk_pool     = makeDiskPool(payload_bytes, 2, temp_dir.path, std::move(blocking_io));
    auto             device_pool   = makeDevicePool({{64, 16}}, 2, "per_rank_disk2d_transient_device");
    auto             first_block   = poolMalloc(*device_pool);
    auto             second_block  = poolMalloc(*device_pool);
    auto             first_slot    = poolMalloc(*disk_pool);
    auto             second_slot   = poolMalloc(*disk_pool);
    ASSERT_NE(first_block, NULL_BLOCK_IDX);
    ASSERT_NE(first_slot, NULL_BLOCK_IDX);
    ASSERT_NE(second_block, NULL_BLOCK_IDX);
    ASSERT_NE(second_slot, NULL_BLOCK_IDX);

    auto group  = makeDeviceHostGroup(0, {device_pool}, nullptr, {makeGroupBase({0}, 64, 16)}, disk_pool);
    auto engine = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group},
                                                               DeviceHostCopyOptions{},
                                                               /*device_disk_staging_block_count=*/1);

    fillDeviceLayer(device_pool, 0, first_block, {0x6A, 0xD3});
    fillDeviceLayer(device_pool, 0, second_block, {0x4B, 0xE1});
    const auto expected_first  = readDeviceLayer(device_pool, 0, first_block);
    const auto expected_second = readDeviceLayer(device_pool, 0, second_block);

    expectStatus(engine,
                 makeDescriptor(Tier::DEVICE, Tier::DISK, {first_block}, NULL_BLOCK_IDX, first_slot, 0),
                 TransferStatus::OK);
    expectStatus(engine,
                 makeDescriptor(Tier::DEVICE, Tier::DISK, {second_block}, NULL_BLOCK_IDX, second_slot, 0),
                 TransferStatus::OK);
    fillDeviceLayer(device_pool, 0, first_block, {0x00, 0x00});
    fillDeviceLayer(device_pool, 0, second_block, {0x00, 0x00});

    std::atomic<bool> first_submit_ok{false};
    std::thread       reader([&] {
        first_submit_ok = submitSucceeded(
            engine, makeDescriptor(Tier::DISK, Tier::DEVICE, {first_block}, NULL_BLOCK_IDX, first_slot, 0));
    });
    BlockingIOGuard   reader_guard(*io, reader);
    ASSERT_TRUE(io->waitUntilBlocked(std::chrono::seconds(5)));

    std::thread releaser([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        io->release();
    });
    expectStatus(engine,
                 makeDescriptor(Tier::DISK, Tier::DEVICE, {second_block}, NULL_BLOCK_IDX, second_slot, 0),
                 TransferStatus::OK);

    releaser.join();
    reader_guard.releaseAndJoin();
    EXPECT_TRUE(first_submit_ok.load());
    // Both waited transfers must land the real payload, not stale staging data.
    EXPECT_EQ(readDeviceLayer(device_pool, 0, first_block), expected_first);
    EXPECT_EQ(readDeviceLayer(device_pool, 0, second_block), expected_second);
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

    host_pool_->free(host_block);
}

TEST_F(PerRankBlockTransferEngineStrategyTest, MultipleDescriptorsUseOnePlanPerDirectionAndDevice) {
    const BlockIdxType second_device_block = poolMalloc(*device_pool_);
    ASSERT_NE(second_device_block, NULL_BLOCK_IDX);
    const BlockIdxType first_host_block  = poolMalloc(*host_pool_);
    const BlockIdxType second_host_block = poolMalloc(*host_pool_);
    ASSERT_NE(first_host_block, NULL_BLOCK_IDX);
    ASSERT_NE(second_host_block, NULL_BLOCK_IDX);

    DeviceHostTransferExecutor executor;
    auto                       capturing_strategy = std::make_unique<CapturingDoneStrategy>();
    auto*                      capture            = capturing_strategy.get();
    executor.strategies_.clear();
    executor.strategies_.push_back(std::move(capturing_strategy));

    const auto                        first_host_buffer  = host_pool_->blockBuffer(first_host_block);
    const auto                        second_host_buffer = host_pool_->blockBuffer(second_host_block);
    const std::vector<HostBufferView> hosts              = {
        {first_host_buffer.addr, first_host_buffer.payload_bytes, first_host_buffer.stride_bytes},
        {second_host_buffer.addr, second_host_buffer.payload_bytes, second_host_buffer.stride_bytes},
    };
    const std::vector<const GroupSet*> groups = {group_set_.get(), group_set_.get()};

    const std::vector<TransferDescriptor> d2h_descriptors = {
        makeDescriptor(Tier::DEVICE, Tier::HOST, {device_block_}, first_host_block),
        makeDescriptor(Tier::DEVICE, Tier::HOST, {second_device_block}, second_host_block),
    };
    EXPECT_EQ(executor.deviceToHost(d2h_descriptors, groups, hosts), TransferStatus::OK);
    ASSERT_EQ(capture->plans.size(), 1u);
    EXPECT_TRUE(capture->plans[0].device_to_host);
    ASSERT_EQ(capture->plans[0].copy_tiles.size(), 4u);
    EXPECT_EQ(capture->plans[0].copy_tiles[0].host_addr, first_host_buffer.addr);
    EXPECT_EQ(capture->plans[0].copy_tiles[1].host_addr,
              static_cast<uint8_t*>(first_host_buffer.addr) + layer_bytes_[0]);
    EXPECT_EQ(capture->plans[0].copy_tiles[2].host_addr, second_host_buffer.addr);
    EXPECT_EQ(capture->plans[0].copy_tiles[3].host_addr,
              static_cast<uint8_t*>(second_host_buffer.addr) + layer_bytes_[0]);

    const std::vector<TransferDescriptor> h2d_descriptors = {
        makeDescriptor(Tier::HOST, Tier::DEVICE, {device_block_}, first_host_block),
        makeDescriptor(Tier::HOST, Tier::DEVICE, {second_device_block}, second_host_block),
    };
    EXPECT_EQ(executor.hostToDevice(hosts, h2d_descriptors, groups), TransferStatus::OK);
    ASSERT_EQ(capture->plans.size(), 2u);
    EXPECT_FALSE(capture->plans[1].device_to_host);
    ASSERT_EQ(capture->plans[1].copy_tiles.size(), 4u);
    EXPECT_EQ(capture->plans[1].copy_tiles[0].host_addr, first_host_buffer.addr);
    EXPECT_EQ(capture->plans[1].copy_tiles[1].host_addr,
              static_cast<uint8_t*>(first_host_buffer.addr) + layer_bytes_[0]);
    EXPECT_EQ(capture->plans[1].copy_tiles[2].host_addr, second_host_buffer.addr);
    EXPECT_EQ(capture->plans[1].copy_tiles[3].host_addr,
              static_cast<uint8_t*>(second_host_buffer.addr) + layer_bytes_[0]);

    host_pool_->free(first_host_block);
    host_pool_->free(second_host_block);
    device_pool_->free(second_device_block);
}

TEST_F(PerRankBlockTransferEngineStrategyTest, MultipleDescriptorsSplitIntoOnePlanPerDevice) {
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count < 2) {
        GTEST_SKIP() << "per-device split test requires two CUDA devices";
    }

    const int first_device_index  = device_pool_->deviceIndex();
    const int second_device_index = (first_device_index + 1) % device_count;
    ASSERT_EQ(cudaSetDevice(second_device_index), cudaSuccess);
    auto               second_device_pool  = makeDevicePool({{128, 0}, {256, 0}}, 4, "strategy_test_second_device");
    const BlockIdxType second_device_block = poolMalloc(*second_device_pool);
    ASSERT_EQ(cudaSetDevice(first_device_index), cudaSuccess);
    ASSERT_NE(second_device_block, NULL_BLOCK_IDX);
    ASSERT_NE(second_device_pool->deviceIndex(), first_device_index);

    auto second_group = makeDeviceHostGroup(1, {second_device_pool}, host_pool_, {makeGroupBase({0, 1}, 128)});
    const BlockIdxType first_host_block  = poolMalloc(*host_pool_);
    const BlockIdxType second_host_block = poolMalloc(*host_pool_);
    ASSERT_NE(first_host_block, NULL_BLOCK_IDX);
    ASSERT_NE(second_host_block, NULL_BLOCK_IDX);

    DeviceHostTransferExecutor executor;
    auto                       capturing_strategy = std::make_unique<CapturingDoneStrategy>();
    auto*                      capture            = capturing_strategy.get();
    executor.strategies_.clear();
    executor.strategies_.push_back(std::move(capturing_strategy));

    const auto                        first_host_buffer  = host_pool_->blockBuffer(first_host_block);
    const auto                        second_host_buffer = host_pool_->blockBuffer(second_host_block);
    const std::vector<HostBufferView> hosts              = {
        {first_host_buffer.addr, first_host_buffer.payload_bytes, first_host_buffer.stride_bytes},
        {second_host_buffer.addr, second_host_buffer.payload_bytes, second_host_buffer.stride_bytes},
    };
    const std::vector<const GroupSet*> groups = {group_set_.get(), second_group.get()};

    const auto verify_plans = [&](bool device_to_host) {
        ASSERT_EQ(capture->plans.size(), 2u);
        for (const auto& plan : capture->plans) {
            EXPECT_EQ(plan.device_to_host, device_to_host);
            ASSERT_EQ(plan.copy_tiles.size(), 2u);
            const int plan_device = plan.copy_tiles.front().device_index;
            EXPECT_TRUE(plan_device == first_device_index || plan_device == second_device_index);
            for (const auto& tile : plan.copy_tiles) {
                EXPECT_EQ(tile.device_index, plan_device);
            }
            EXPECT_EQ(plan.copy_tiles.front().host_addr,
                      plan_device == first_device_index ? first_host_buffer.addr : second_host_buffer.addr);
        }
    };

    const std::vector<TransferDescriptor> d2h_descriptors = {
        makeDescriptor(Tier::DEVICE, Tier::HOST, {device_block_}, first_host_block, NULL_BLOCK_IDX, 0),
        makeDescriptor(Tier::DEVICE, Tier::HOST, {second_device_block}, second_host_block, NULL_BLOCK_IDX, 1),
    };
    EXPECT_EQ(executor.deviceToHost(d2h_descriptors, groups, hosts), TransferStatus::OK);
    verify_plans(true);

    capture->plans.clear();
    const std::vector<TransferDescriptor> h2d_descriptors = {
        makeDescriptor(Tier::HOST, Tier::DEVICE, {device_block_}, first_host_block, NULL_BLOCK_IDX, 0),
        makeDescriptor(Tier::HOST, Tier::DEVICE, {second_device_block}, second_host_block, NULL_BLOCK_IDX, 1),
    };
    EXPECT_EQ(executor.hostToDevice(hosts, h2d_descriptors, groups), TransferStatus::OK);
    verify_plans(false);

    host_pool_->free(first_host_block);
    host_pool_->free(second_host_block);
    second_device_pool->free(second_device_block);
}

TEST_F(PerRankBlockTransferEngineStrategyTest, BatchStrategyFailureFailsWholeExecutorCall) {
    const BlockIdxType second_device_block = poolMalloc(*device_pool_);
    ASSERT_NE(second_device_block, NULL_BLOCK_IDX);
    const BlockIdxType first_host_block  = poolMalloc(*host_pool_);
    const BlockIdxType second_host_block = poolMalloc(*host_pool_);
    ASSERT_NE(first_host_block, NULL_BLOCK_IDX);
    ASSERT_NE(second_host_block, NULL_BLOCK_IDX);

    DeviceHostTransferExecutor executor;
    auto                       failing_strategy = std::make_unique<CapturingFailingStrategy>();
    auto*                      capture          = failing_strategy.get();
    executor.strategies_.clear();
    executor.strategies_.push_back(std::move(failing_strategy));

    const auto                        first_host_buffer  = host_pool_->blockBuffer(first_host_block);
    const auto                        second_host_buffer = host_pool_->blockBuffer(second_host_block);
    const std::vector<HostBufferView> hosts              = {
        {first_host_buffer.addr, first_host_buffer.payload_bytes, first_host_buffer.stride_bytes},
        {second_host_buffer.addr, second_host_buffer.payload_bytes, second_host_buffer.stride_bytes},
    };
    const std::vector<const GroupSet*>    groups      = {group_set_.get(), group_set_.get()};
    const std::vector<TransferDescriptor> descriptors = {
        makeDescriptor(Tier::DEVICE, Tier::HOST, {device_block_}, first_host_block),
        makeDescriptor(Tier::DEVICE, Tier::HOST, {second_device_block}, second_host_block),
    };

    EXPECT_EQ(executor.deviceToHost(descriptors, groups, hosts), TransferStatus::DEVICE_IO_ERROR);
    ASSERT_EQ(capture->plans.size(), 1u);
    ASSERT_EQ(capture->plans.front().copy_tiles.size(), 4u);
    EXPECT_EQ(capture->plans.front().copy_tiles[0].host_addr, first_host_buffer.addr);
    EXPECT_EQ(capture->plans.front().copy_tiles[2].host_addr, second_host_buffer.addr);

    host_pool_->free(first_host_block);
    host_pool_->free(second_host_block);
    device_pool_->free(second_device_block);
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
    EXPECT_EQ(counters[0].done, 0);
    EXPECT_EQ(counters[0].not_applicable, 2);
    EXPECT_EQ(counters[0].failed, 0);

    EXPECT_EQ(counters[1].attempts, 2);
    EXPECT_EQ(counters[1].done, expect_batch_done ? 2 : 0);
    EXPECT_EQ(counters[1].not_applicable, expect_batch_done ? 0 : 2);
    EXPECT_EQ(counters[1].failed, 0);

    EXPECT_EQ(counters[2].attempts, expect_batch_done ? 0 : 2);
    EXPECT_EQ(counters[2].done, expect_batch_done ? 0 : 2);
    EXPECT_EQ(counters[2].not_applicable, 0);
    EXPECT_EQ(counters[2].failed, 0);

    host_pool_->free(host_block);
}

TEST_F(PerRankBlockTransferEngineStrategyTest, BatchNotApplicableFallsBackToGeneric) {
    DeviceHostCopyOptions options;
    options.cuda_batch_copy_enabled = true;
    DeviceHostTransferExecutor      executor(options);
    std::array<StrategyCounters, 3> counters;
    installStrategyRecorders(executor, counters);

    fillDeviceLayer(device_pool_, 0, device_block_, {0x71});
    auto device_buffer = device_pool_->convertIndexToBuffer(0, device_block_).front();
    auto host_block    = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    auto* host_data = static_cast<uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    std::memset(host_data, 0, device_buffer.size_bytes);

    DeviceHostCopyPlan plan;
    plan.device_to_host = true;
    plan.single_device  = true;
    plan.group_set_id   = 0;
    plan.host           = {host_data, device_buffer.size_bytes};
    plan.copy_tiles.push_back(DeviceHostCopyTile{host_data, device_buffer.addr, 0, device_buffer.size_bytes, -1, 0, 0});
    EXPECT_EQ(executor.executeStrategies(plan), TransferStatus::OK);
    for (size_t i = 0; i < device_buffer.size_bytes; ++i)
        EXPECT_EQ(host_data[i], 0x71);
    EXPECT_EQ(counters[1].not_applicable, 1);
    EXPECT_EQ(counters[2].done, 1);
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

    host_pool_->free(host_block);
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
    EXPECT_EQ(counters[0].done, 2);
    EXPECT_EQ(counters[1].attempts, 0);
    EXPECT_EQ(counters[2].attempts, 0);

    host_pool_->free(host_block);
}

TEST_F(PerRankBlockTransferEngineStrategyTest, StagedStrategyTakesPrecedenceWhenEligible) {
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
    EXPECT_EQ(counters[0].done, 1);
    EXPECT_EQ(counters[1].attempts, 0);
    EXPECT_EQ(counters[2].attempts, 0);

    host_pool_->free(host_block);
}

}  // namespace
}  // namespace rtp_llm

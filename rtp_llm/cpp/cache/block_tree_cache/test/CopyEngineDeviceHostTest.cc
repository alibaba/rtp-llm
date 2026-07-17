#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <torch/torch.h>

#include "rtp_llm/cpp/cache/block_tree_cache/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/CopyEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/DeviceHostCopyStrategy.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/DeviceHostTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/CopyEngineTestUtils.h"

namespace rtp_llm {
namespace {

using copy_engine_test::TempDirGuard;
using copy_engine_test::expectStatus;
using copy_engine_test::makeDescriptor;
using copy_engine_test::makeDiskPool;
using copy_engine_test::makeHostPool;
using copy_engine_test::poolMalloc;

struct DeviceLayerBufferSpec {
    size_t kv_bytes{0};
    size_t scale_bytes{0};
};

// Build a DeviceBlockPool with the given per-layer layout.
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

static Component makeComponent(int                                         component_id,
                               int                                         component_group_id,
                               const std::vector<MemoryBlockLayerTagSlot>& slots,
                               int                                         device_pool_index = 0) {
    Component component;
    component.component_id                 = component_id;
    component.component_group_id           = component_group_id;
    component.type                         = CacheGroupType::FULL;
    component.memory_block_layer_tag_slots = slots;
    component.device_pool_index            = device_pool_index;
    return component;
}

static ComponentGroupPtr makeDeviceHostGroup(int                             group_id,
                                             std::vector<int>                component_indices,
                                             std::vector<DeviceBlockPoolPtr> device_pools,
                                             std::shared_ptr<HostBlockPool>  host_pool,
                                             std::shared_ptr<DiskBlockPool>  disk_pool = nullptr) {
    auto group                = std::make_shared<FullComponentGroup>();
    group->component_group_id = group_id;
    group->group_type         = CacheGroupType::FULL;
    group->component_indices  = std::move(component_indices);
    group->setDevicePools(std::move(device_pools));
    group->setHostPool(std::move(host_pool));
    group->setDiskPool(std::move(disk_pool));
    return group;
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

static void installStrategyRecorders(DeviceHostTransferExecutor& executor, std::array<StrategyCounters, 3>& counters) {
    RTP_LLM_CHECK(executor.strategies_.size() == counters.size());
    for (size_t i = 0; i < counters.size(); ++i) {
        executor.strategies_[i] = std::make_unique<RecordingStrategy>(std::move(executor.strategies_[i]), &counters[i]);
    }
}

// ---- CopyEngine submit() tests (real CUDA) ----

class CopyEngineTest: public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";

        // 3 layers with different strides: 100, 200, 150 bytes
        slots_ = {
            {0, "layer_0", 100},
            {1, "layer_1", 200},
            {2, "layer_2", 150},
        };
        host_block_size_ = CopyEngine::computeHostBlockSize(slots_);  // 450

        // Create host pool — 10 usable blocks
        host_pool_ = makeHostPool(host_block_size_, 10, true);

        device_pool_  = makeDevicePool({{100, 0}, {200, 0}, {150, 0}}, 10, "copy_engine_device");
        device_block_ = poolMalloc(*device_pool_);
        ASSERT_NE(device_block_, NULL_BLOCK_IDX);
        device_blocks_ = {device_block_};

        component_       = makeComponent(0, 0, slots_);
        component_group_ = makeDeviceHostGroup(0, {0}, {device_pool_}, host_pool_);
        copy_engine_     = std::make_shared<CopyEngine>(std::vector<ComponentGroupPtr>{component_group_},
                                                    std::vector<Component>{component_});
    }

    std::vector<MemoryBlockLayerTagSlot> slots_;
    size_t                               host_block_size_;
    std::shared_ptr<HostBlockPool>       host_pool_;
    std::shared_ptr<CopyEngine>          copy_engine_;
    DeviceBlockPoolPtr                   device_pool_;
    BlockIdxType                         device_block_;
    std::vector<BlockIdxType>            device_blocks_;
    Component                            component_;
    ComponentGroupPtr                    component_group_;
};

TEST_F(CopyEngineTest, SubmitDeviceHostRoundTripPreservesLayout) {
    fillDeviceLayerSequential(device_pool_, 0, device_block_);
    fillDeviceLayer(device_pool_, 1, device_block_, {0x5A});
    fillDeviceLayerSequential(device_pool_, 2, device_block_);

    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    auto d2h_desc = makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block);
    ASSERT_TRUE(copy_engine_->submit(d2h_desc).ok());

    const uint8_t* host_data = static_cast<const uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    for (size_t i = 0; i < slots_[0].stride_bytes; ++i)
        EXPECT_EQ(host_data[i], static_cast<uint8_t>(i & 0xFF));
    for (size_t i = 0; i < slots_[1].stride_bytes; ++i)
        EXPECT_EQ(host_data[slots_[0].stride_bytes + i], 0x5A);
    const size_t off2 = slots_[0].stride_bytes + slots_[1].stride_bytes;
    for (size_t i = 0; i < slots_[2].stride_bytes; ++i)
        EXPECT_EQ(host_data[off2 + i], static_cast<uint8_t>(i & 0xFF));

    fillDeviceLayer(device_pool_, 0, device_block_, {0x00});
    fillDeviceLayer(device_pool_, 1, device_block_, {0x00});
    fillDeviceLayer(device_pool_, 2, device_block_, {0x00});
    auto h2d_desc = makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, host_block);
    ASSERT_TRUE(copy_engine_->submit(h2d_desc).ok());

    auto d0 = readDeviceLayer(device_pool_, 0, device_block_);
    auto d1 = readDeviceLayer(device_pool_, 1, device_block_);
    auto d2 = readDeviceLayer(device_pool_, 2, device_block_);
    for (size_t i = 0; i < slots_[0].stride_bytes; ++i)
        EXPECT_EQ(d0[i], static_cast<uint8_t>(i & 0xFF));
    for (auto byte : d1)
        EXPECT_EQ(byte, 0x5A);
    for (size_t i = 0; i < slots_[2].stride_bytes; ++i)
        EXPECT_EQ(d2[i], static_cast<uint8_t>(i & 0xFF));

    host_pool_->free(host_block);
}

TEST_F(CopyEngineTest, SubmitRejectsMissingRequiredBlocks) {
    expectStatus(copy_engine_,
                 makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, NULL_BLOCK_IDX),
                 CopyStatus::INVALID_ARGS);
    expectStatus(copy_engine_,
                 makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, NULL_BLOCK_IDX),
                 CopyStatus::INVALID_ARGS);
}

TEST_F(CopyEngineTest, SubmitAcceptsValidUnallocatedHostBlock) {
    constexpr BlockIdxType unallocated_host_block = 1;
    expectStatus(
        copy_engine_, makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, unallocated_host_block), CopyStatus::OK);
    expectStatus(
        copy_engine_, makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, unallocated_host_block), CopyStatus::OK);
}

TEST_F(CopyEngineTest, SubmitRejectsOutOfRangeHostBlock) {
    const BlockIdxType out_of_range = static_cast<BlockIdxType>(host_pool_->totalBlocksNum() + 1);
    expectStatus(
        copy_engine_, makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, out_of_range), CopyStatus::INVALID_ARGS);
    expectStatus(
        copy_engine_, makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, out_of_range), CopyStatus::INVALID_ARGS);
}

TEST_F(CopyEngineTest, SubmitAcceptsValidUnallocatedDeviceBlock) {
    // Worker transfers may use a valid logical block ID without local allocator ownership.
    BlockIdxType freed_device_block = poolMalloc(*device_pool_);
    ASSERT_NE(freed_device_block, NULL_BLOCK_IDX);
    device_pool_->free(freed_device_block);
    std::vector<BlockIdxType> unallocated_device_blocks = {freed_device_block};

    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    expectStatus(
        copy_engine_, makeDescriptor(Tier::DEVICE, Tier::HOST, unallocated_device_blocks, host_block), CopyStatus::OK);
    expectStatus(
        copy_engine_, makeDescriptor(Tier::HOST, Tier::DEVICE, unallocated_device_blocks, host_block), CopyStatus::OK);

    host_pool_->free(host_block);
}

TEST_F(CopyEngineTest, SubmitRejectsOutOfRangeDeviceBlock) {
    const BlockIdxType              out_of_range  = static_cast<BlockIdxType>(device_pool_->totalBlocksNum() + 1);
    const std::vector<BlockIdxType> device_blocks = {out_of_range};
    const BlockIdxType              host_block    = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    expectStatus(
        copy_engine_, makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks, host_block), CopyStatus::INVALID_ARGS);
    expectStatus(
        copy_engine_, makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks, host_block), CopyStatus::INVALID_ARGS);

    host_pool_->free(host_block);
}

TEST_F(CopyEngineTest, SubmitRejectsMismatchedDeviceBlockCount) {
    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    const std::array<std::vector<BlockIdxType>, 2> wrong_blocks = {
        std::vector<BlockIdxType>{},
        std::vector<BlockIdxType>{device_block_, device_block_},
    };
    for (const auto& blocks : wrong_blocks) {
        SCOPED_TRACE(::testing::Message() << "block_count=" << blocks.size());
        expectStatus(
            copy_engine_, makeDescriptor(Tier::DEVICE, Tier::HOST, blocks, host_block), CopyStatus::INVALID_ARGS);
        expectStatus(
            copy_engine_, makeDescriptor(Tier::HOST, Tier::DEVICE, blocks, host_block), CopyStatus::INVALID_ARGS);
    }
    host_pool_->free(host_block);
}

TEST_F(CopyEngineTest, SubmitRejectsInvalidComponentGroupId) {
    for (int group_id : {-1, 99}) {
        SCOPED_TRACE(::testing::Message() << "group_id=" << group_id);
        expectStatus(copy_engine_,
                     makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, NULL_BLOCK_IDX, NULL_BLOCK_IDX, group_id),
                     CopyStatus::INVALID_ARGS);
    }
}

TEST_F(CopyEngineTest, SubmitRejectsInvalidTierPairs) {
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
        expectStatus(
            copy_engine_, makeDescriptor(source, target, device_blocks_, host_block), CopyStatus::INVALID_ARGS);
    }
    host_pool_->free(host_block);
}

TEST_F(CopyEngineTest, SubmitRejectsInvalidComponentIndex) {
    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    for (int component_index : {-1, 1}) {
        SCOPED_TRACE(::testing::Message() << "component_index=" << component_index);
        auto group = makeDeviceHostGroup(0, {component_index}, {device_pool_}, host_pool_);
        auto engine =
            std::make_shared<CopyEngine>(std::vector<ComponentGroupPtr>{group}, std::vector<Component>{component_});
        expectStatus(
            engine, makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block), CopyStatus::INVALID_ARGS);
    }
    host_pool_->free(host_block);
}

TEST_F(CopyEngineTest, SubmitRejectsComponentFromDifferentGroup) {
    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    auto wrong_component               = component_;
    wrong_component.component_group_id = 1;
    auto engine                        = std::make_shared<CopyEngine>(std::vector<ComponentGroupPtr>{component_group_},
                                               std::vector<Component>{wrong_component});
    expectStatus(
        engine, makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block), CopyStatus::INVALID_ARGS);
    host_pool_->free(host_block);
}

TEST_F(CopyEngineTest, SubmitRejectsInvalidDevicePoolIndex) {
    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    for (int pool_index : {-1, 1}) {
        SCOPED_TRACE(::testing::Message() << "device_pool_index=" << pool_index);
        auto component              = component_;
        component.device_pool_index = pool_index;
        auto engine                 = std::make_shared<CopyEngine>(std::vector<ComponentGroupPtr>{component_group_},
                                                   std::vector<Component>{component});
        expectStatus(
            engine, makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block), CopyStatus::INVALID_ARGS);
    }

    auto null_pool_group  = makeDeviceHostGroup(0, {0}, {nullptr}, host_pool_);
    auto null_pool_engine = std::make_shared<CopyEngine>(std::vector<ComponentGroupPtr>{null_pool_group},
                                                         std::vector<Component>{component_});
    expectStatus(null_pool_engine,
                 makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block),
                 CopyStatus::INVALID_ARGS);
    host_pool_->free(host_block);
}

TEST_F(CopyEngineTest, SubmitRejectsIncompleteDeviceHostLayout) {
    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    const auto desc = makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block);

    auto missing_host_group  = makeDeviceHostGroup(0, {0}, {device_pool_}, nullptr);
    auto missing_host_engine = std::make_shared<CopyEngine>(std::vector<ComponentGroupPtr>{missing_host_group},
                                                            std::vector<Component>{component_});
    expectStatus(missing_host_engine, desc, CopyStatus::INVALID_ARGS);

    auto empty_group = makeDeviceHostGroup(0, {}, {device_pool_}, host_pool_);
    auto empty_engine =
        std::make_shared<CopyEngine>(std::vector<ComponentGroupPtr>{empty_group}, std::vector<Component>{});
    expectStatus(empty_engine, desc, CopyStatus::INVALID_ARGS);

    auto empty_component    = makeComponent(0, 0, {});
    auto empty_slots_group  = makeDeviceHostGroup(0, {0}, {device_pool_}, host_pool_);
    auto empty_slots_engine = std::make_shared<CopyEngine>(std::vector<ComponentGroupPtr>{empty_slots_group},
                                                           std::vector<Component>{empty_component});
    expectStatus(empty_slots_engine, desc, CopyStatus::INVALID_ARGS);
    host_pool_->free(host_block);
}

TEST_F(CopyEngineTest, SubmitRejectsInvalidLayerSlotLayout) {
    {
        const std::vector<MemoryBlockLayerTagSlot> zero_stride_slots = {
            {0, "valid", 64},
            {1, "zero", 0},
        };
        auto host_pool   = makeHostPool(64, 2, true);
        auto device_pool = makeDevicePool({{64, 0}, {16, 0}}, 2, "copy_engine_zero_stride");
        auto block       = poolMalloc(*device_pool);
        auto host_block  = poolMalloc(*host_pool);
        auto component   = makeComponent(0, 0, zero_stride_slots);
        auto group       = makeDeviceHostGroup(0, {0}, {device_pool}, host_pool);
        auto engine =
            std::make_shared<CopyEngine>(std::vector<ComponentGroupPtr>{group}, std::vector<Component>{component});
        expectStatus(engine, makeDescriptor(Tier::DEVICE, Tier::HOST, {block}, host_block), CopyStatus::INVALID_ARGS);
    }

    {
        const std::vector<MemoryBlockLayerTagSlot> mismatched_slots = {{0, "mismatch", 65}};
        auto                                       host_pool        = makeHostPool(65, 2, true);
        auto device_pool = makeDevicePool({{64, 0}}, 2, "copy_engine_slot_mismatch");
        auto block       = poolMalloc(*device_pool);
        auto host_block  = poolMalloc(*host_pool);
        auto component   = makeComponent(0, 0, mismatched_slots);
        auto group       = makeDeviceHostGroup(0, {0}, {device_pool}, host_pool);
        auto engine =
            std::make_shared<CopyEngine>(std::vector<ComponentGroupPtr>{group}, std::vector<Component>{component});
        expectStatus(engine, makeDescriptor(Tier::DEVICE, Tier::HOST, {block}, host_block), CopyStatus::INVALID_ARGS);
        expectStatus(engine, makeDescriptor(Tier::HOST, Tier::DEVICE, {block}, host_block), CopyStatus::INVALID_ARGS);
    }
}

TEST_F(CopyEngineTest, SubmitRejectsHostLayoutPayloadMismatch) {
    auto host_pool  = makeHostPool(host_block_size_ + 1, 2, true);
    auto host_block = poolMalloc(*host_pool);
    auto group      = makeDeviceHostGroup(0, {0}, {device_pool_}, host_pool);
    auto engine =
        std::make_shared<CopyEngine>(std::vector<ComponentGroupPtr>{group}, std::vector<Component>{component_});
    expectStatus(
        engine, makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block), CopyStatus::INVALID_ARGS);
    expectStatus(
        engine, makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, host_block), CopyStatus::INVALID_ARGS);
}

TEST_F(CopyEngineTest, UnusableCopyBufferReturnsDeviceIoError) {
    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    device_pool_->layout_strategies_[0]->config_.kv_block_stride_bytes = 0;
    expectStatus(copy_engine_,
                 makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block),
                 CopyStatus::DEVICE_IO_ERROR);
    expectStatus(copy_engine_,
                 makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, host_block),
                 CopyStatus::DEVICE_IO_ERROR);
    host_pool_->free(host_block);
}

TEST_F(CopyEngineTest, SubmitReturnsCompletedHandleWithFinalStatus) {
    fillDeviceLayer(device_pool_, 0, device_block_, {0xAA});
    fillDeviceLayer(device_pool_, 1, device_block_, {0xBB});
    fillDeviceLayer(device_pool_, 2, device_block_, {0xCC});

    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    const std::array<std::pair<TransferDescriptor, CopyStatus>, 2> cases = {
        std::pair{makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block), CopyStatus::OK},
        std::pair{makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block, NULL_BLOCK_IDX, 99),
                  CopyStatus::INVALID_ARGS},
    };

    uint64_t previous_request_id = 0;
    for (const auto& [desc, expected] : cases) {
        auto handle = copy_engine_->submit(desc);
        ASSERT_TRUE(handle.valid());
        EXPECT_TRUE(handle.done());
        handle.wait();
        EXPECT_EQ(handle.status(), expected);
        EXPECT_EQ(handle.ok(), expected == CopyStatus::OK);
        EXPECT_GT(handle.requestId(), previous_request_id);
        previous_request_id = handle.requestId();

        bool callback_called = false;
        handle.onComplete([&](CopyStatus status) {
            callback_called = true;
            EXPECT_EQ(status, expected);
        });
        EXPECT_TRUE(callback_called);
    }

    host_pool_->free(host_block);
}

TEST_F(CopyEngineTest, SubmitRejectsAllNullDeviceBlocks) {
    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    auto* host_data = static_cast<uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    std::memset(host_data, 0xA5, host_block_size_);

    const std::vector<BlockIdxType> all_null = {NULL_BLOCK_IDX};
    expectStatus(
        copy_engine_, makeDescriptor(Tier::DEVICE, Tier::HOST, all_null, host_block), CopyStatus::INVALID_ARGS);
    for (size_t i = 0; i < host_block_size_; ++i)
        EXPECT_EQ(host_data[i], 0xA5);

    expectStatus(
        copy_engine_, makeDescriptor(Tier::HOST, Tier::DEVICE, all_null, host_block), CopyStatus::INVALID_ARGS);
    host_pool_->free(host_block);
}

TEST_F(CopyEngineTest, SubmitHostToDeviceIndependentDescriptors) {
    BlockIdxType second_device_block = poolMalloc(*device_pool_);
    ASSERT_NE(second_device_block, NULL_BLOCK_IDX);
    std::vector<BlockIdxType> second_device_blocks = {second_device_block};

    BlockIdxType host_block_1 = poolMalloc(*host_pool_);
    auto*        host_data_1  = static_cast<uint8_t*>(host_pool_->blockBuffer(host_block_1).addr);
    std::memset(host_data_1, 0x12, slots_[0].stride_bytes);
    std::memset(host_data_1 + slots_[0].stride_bytes, 0x34, slots_[1].stride_bytes);
    std::memset(host_data_1 + slots_[0].stride_bytes + slots_[1].stride_bytes, 0x9A, slots_[2].stride_bytes);

    BlockIdxType host_block_2 = poolMalloc(*host_pool_);
    auto*        host_data_2  = static_cast<uint8_t*>(host_pool_->blockBuffer(host_block_2).addr);
    std::memset(host_data_2, 0x56, slots_[0].stride_bytes);
    std::memset(host_data_2 + slots_[0].stride_bytes, 0x78, slots_[1].stride_bytes);
    std::memset(host_data_2 + slots_[0].stride_bytes + slots_[1].stride_bytes, 0xBC, slots_[2].stride_bytes);

    auto result_1 = copy_engine_->submit(makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, host_block_1));
    ASSERT_TRUE(result_1.ok());

    auto result_2 = copy_engine_->submit(makeDescriptor(Tier::HOST, Tier::DEVICE, second_device_blocks, host_block_2));
    ASSERT_TRUE(result_2.ok());

    for (auto byte : readDeviceLayer(device_pool_, 0, device_block_))
        EXPECT_EQ(byte, 0x12);
    for (auto byte : readDeviceLayer(device_pool_, 1, device_block_))
        EXPECT_EQ(byte, 0x34);
    for (auto byte : readDeviceLayer(device_pool_, 2, device_block_))
        EXPECT_EQ(byte, 0x9A);
    for (auto byte : readDeviceLayer(device_pool_, 0, second_device_block))
        EXPECT_EQ(byte, 0x56);
    for (auto byte : readDeviceLayer(device_pool_, 1, second_device_block))
        EXPECT_EQ(byte, 0x78);
    for (auto byte : readDeviceLayer(device_pool_, 2, second_device_block))
        EXPECT_EQ(byte, 0xBC);

    host_pool_->free(host_block_1);
    host_pool_->free(host_block_2);
    device_pool_->free(second_device_block);
}

class CopyEngineMixedNullTest: public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
        host_pool_ = makeHostPool(168, 4, true);
        pools_     = {
            makeDevicePool({{64, 16}}, 4, "copy_engine_mixed_0"),
            makeDevicePool({{48, 0}}, 4, "copy_engine_mixed_1"),
            makeDevicePool({{32, 8}}, 4, "copy_engine_mixed_2"),
        };
        for (const auto& pool : pools_) {
            blocks_.push_back(poolMalloc(*pool));
            ASSERT_NE(blocks_.back(), NULL_BLOCK_IDX);
        }

        std::vector<Component> components = {
            makeComponent(0, 0, {{0, "kv_scale_0", 80}}, 0),
            makeComponent(1, 0, {{0, "missing", 48}}, 1),
            makeComponent(2, 0, {{0, "kv_scale_2", 40}}, 2),
        };
        auto group  = makeDeviceHostGroup(0, {0, 1, 2}, pools_, host_pool_);
        engine_     = std::make_shared<CopyEngine>(std::vector<ComponentGroupPtr>{group}, components);
        host_block_ = poolMalloc(*host_pool_);
        ASSERT_NE(host_block_, NULL_BLOCK_IDX);
    }

    std::shared_ptr<HostBlockPool>  host_pool_;
    std::vector<DeviceBlockPoolPtr> pools_;
    std::vector<BlockIdxType>       blocks_;
    std::shared_ptr<CopyEngine>     engine_;
    BlockIdxType                    host_block_{NULL_BLOCK_IDX};
};

TEST_F(CopyEngineMixedNullTest, DeviceToHostMixedNullComponentsPreserveOffsets) {
    fillDeviceLayer(pools_[0], 0, blocks_[0], {0xA1, 0xA2});
    fillDeviceLayer(pools_[1], 0, blocks_[1], {0xB1});
    fillDeviceLayer(pools_[2], 0, blocks_[2], {0xC1, 0xC2});
    auto* host_data = static_cast<uint8_t*>(host_pool_->blockBuffer(host_block_).addr);
    std::memset(host_data, 0xFF, 168);

    expectStatus(engine_,
                 makeDescriptor(Tier::DEVICE, Tier::HOST, {blocks_[0], NULL_BLOCK_IDX, blocks_[2]}, host_block_),
                 CopyStatus::OK);
    for (size_t i = 0; i < 64; ++i)
        EXPECT_EQ(host_data[i], 0xA1);
    for (size_t i = 64; i < 80; ++i)
        EXPECT_EQ(host_data[i], 0xA2);
    for (size_t i = 80; i < 128; ++i)
        EXPECT_EQ(host_data[i], 0x00);
    for (size_t i = 128; i < 160; ++i)
        EXPECT_EQ(host_data[i], 0xC1);
    for (size_t i = 160; i < 168; ++i)
        EXPECT_EQ(host_data[i], 0xC2);
}

TEST_F(CopyEngineMixedNullTest, HostToDeviceMixedNullComponentsPreserveOffsets) {
    auto* host_data = static_cast<uint8_t*>(host_pool_->blockBuffer(host_block_).addr);
    std::memset(host_data, 0x11, 64);
    std::memset(host_data + 64, 0x12, 16);
    std::memset(host_data + 80, 0x22, 48);
    std::memset(host_data + 128, 0x31, 32);
    std::memset(host_data + 160, 0x32, 8);
    fillDeviceLayer(pools_[0], 0, blocks_[0], {0x00, 0x00});
    fillDeviceLayer(pools_[1], 0, blocks_[1], {0xEE});
    fillDeviceLayer(pools_[2], 0, blocks_[2], {0x00, 0x00});

    expectStatus(engine_,
                 makeDescriptor(Tier::HOST, Tier::DEVICE, {blocks_[0], NULL_BLOCK_IDX, blocks_[2]}, host_block_),
                 CopyStatus::OK);

    const auto comp0 = readDeviceLayer(pools_[0], 0, blocks_[0]);
    const auto comp1 = readDeviceLayer(pools_[1], 0, blocks_[1]);
    const auto comp2 = readDeviceLayer(pools_[2], 0, blocks_[2]);
    for (size_t i = 0; i < 64; ++i)
        EXPECT_EQ(comp0[i], 0x11);
    for (size_t i = 64; i < 80; ++i)
        EXPECT_EQ(comp0[i], 0x12);
    for (auto byte : comp1)
        EXPECT_EQ(byte, 0xEE);
    for (size_t i = 0; i < 32; ++i)
        EXPECT_EQ(comp2[i], 0x31);
    for (size_t i = 32; i < 40; ++i)
        EXPECT_EQ(comp2[i], 0x32);
}

TEST(CopyEngineIntegrationTest, DeviceHostDiskHostDeviceRoundTrip) {
    ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
    TempDirGuard     temp_dir("copy_engine_three_tier");
    constexpr size_t payload_bytes = 80;
    auto             host_pool     = makeHostPool(payload_bytes, 2, true);
    auto             disk_pool     = makeDiskPool(payload_bytes, 2, temp_dir.path);
    auto             device_pool   = makeDevicePool({{64, 16}}, 2, "copy_engine_three_tier_device");
    auto             device_block  = poolMalloc(*device_pool);
    auto             host_block    = poolMalloc(*host_pool);
    auto             disk_block    = poolMalloc(*disk_pool);
    ASSERT_NE(device_block, NULL_BLOCK_IDX);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);

    auto component = makeComponent(0, 0, {{0, "kv_scale", payload_bytes}});
    auto group     = makeDeviceHostGroup(0, {0}, {device_pool}, host_pool, disk_pool);
    auto engine =
        std::make_shared<CopyEngine>(std::vector<ComponentGroupPtr>{group}, std::vector<Component>{component});
    fillDeviceLayer(device_pool, 0, device_block, {0x6A, 0xD3});
    const auto expected = readDeviceLayer(device_pool, 0, device_block);

    expectStatus(engine,
                 makeDescriptor(Tier::DEVICE, Tier::HOST, {device_block}, host_block, NULL_BLOCK_IDX, 0),
                 CopyStatus::OK);
    const auto* host_data = static_cast<const uint8_t*>(host_pool->blockBuffer(host_block).addr);
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), host_data));

    expectStatus(engine, makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, disk_block, 0), CopyStatus::OK);
    std::memset(host_pool->blockBuffer(host_block).addr, 0, payload_bytes);
    fillDeviceLayer(device_pool, 0, device_block, {0x00, 0x00});

    expectStatus(engine, makeDescriptor(Tier::DISK, Tier::HOST, {}, host_block, disk_block, 0), CopyStatus::OK);
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), host_data));

    // Clear Device again after the disk read so the final bytes can only come from the full path.
    fillDeviceLayer(device_pool, 0, device_block, {0x00, 0x00});
    expectStatus(engine,
                 makeDescriptor(Tier::HOST, Tier::DEVICE, {device_block}, host_block, NULL_BLOCK_IDX, 0),
                 CopyStatus::OK);
    EXPECT_EQ(readDeviceLayer(device_pool, 0, device_block), expected);
}

// ---- Strategy chain tests ----

class CopyEngineStrategyTest: public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";

        slots_ = {
            {0, "layer_0", 128},
            {1, "layer_1", 256},
        };
        host_block_size_ = CopyEngine::computeHostBlockSize(slots_);

        host_pool_    = makeHostPool(host_block_size_, 10, true);
        device_pool_  = makeDevicePool({{128, 0}, {256, 0}}, 10, "strategy_test_device");
        device_block_ = poolMalloc(*device_pool_);
        ASSERT_NE(device_block_, NULL_BLOCK_IDX);
        device_blocks_ = {device_block_};

        component_       = makeComponent(0, 0, slots_);
        component_group_ = makeDeviceHostGroup(0, {0}, {device_pool_}, host_pool_);
    }

    std::shared_ptr<CopyEngine> makeCopyEngine(DeviceHostCopyOptions options = {}) {
        return std::make_shared<CopyEngine>(
            std::vector<ComponentGroupPtr>{component_group_}, std::vector<Component>{component_}, std::move(options));
    }

    std::vector<MemoryBlockLayerTagSlot> slots_;
    size_t                               host_block_size_;
    std::shared_ptr<HostBlockPool>       host_pool_;
    DeviceBlockPoolPtr                   device_pool_;
    BlockIdxType                         device_block_;
    std::vector<BlockIdxType>            device_blocks_;
    Component                            component_;
    ComponentGroupPtr                    component_group_;
};

TEST_F(CopyEngineStrategyTest, GenericStrategyRoundTrip) {
    DeviceHostCopyOptions options;
    options.cuda_batch_copy_enabled             = false;
    auto                            copy_engine = makeCopyEngine(options);
    std::array<StrategyCounters, 3> counters;
    installStrategyRecorders(*copy_engine->device_host_executor_, counters);

    fillDeviceLayer(device_pool_, 0, device_block_, {0xAA});
    fillDeviceLayer(device_pool_, 1, device_block_, {0xBB});

    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    auto d2h = makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block);
    ASSERT_TRUE(copy_engine->submit(d2h).ok());

    const auto* host_data = static_cast<const uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    for (size_t i = 0; i < 128; ++i)
        EXPECT_EQ(host_data[i], 0xAA);
    for (size_t i = 128; i < 384; ++i)
        EXPECT_EQ(host_data[i], 0xBB);

    fillDeviceLayer(device_pool_, 0, device_block_, {0x00});
    fillDeviceLayer(device_pool_, 1, device_block_, {0x00});

    auto h2d = makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, host_block);
    ASSERT_TRUE(copy_engine->submit(h2d).ok());

    auto d0 = readDeviceLayer(device_pool_, 0, device_block_);
    auto d1 = readDeviceLayer(device_pool_, 1, device_block_);
    for (auto b : d0)
        EXPECT_EQ(b, 0xAA);
    for (auto b : d1)
        EXPECT_EQ(b, 0xBB);
    EXPECT_EQ(counters[0].not_applicable, 2);
    EXPECT_EQ(counters[1].not_applicable, 2);
    EXPECT_EQ(counters[2].done, 2);

    host_pool_->free(host_block);
}

TEST_F(CopyEngineStrategyTest, BatchStrategyExecutesWhenSupportedOtherwiseFallsBack) {
    DeviceHostCopyOptions options;
    options.cuda_batch_copy_enabled             = true;
    auto                            copy_engine = makeCopyEngine(options);
    std::array<StrategyCounters, 3> counters;
    installStrategyRecorders(*copy_engine->device_host_executor_, counters);

    fillDeviceLayer(device_pool_, 0, device_block_, {0x11});
    fillDeviceLayer(device_pool_, 1, device_block_, {0x22});

    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    auto d2h = makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block);
    ASSERT_TRUE(copy_engine->submit(d2h).ok());

    const auto* host_data = static_cast<const uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    for (size_t i = 0; i < 128; ++i)
        EXPECT_EQ(host_data[i], 0x11);
    for (size_t i = 128; i < 384; ++i)
        EXPECT_EQ(host_data[i], 0x22);

    fillDeviceLayer(device_pool_, 0, device_block_, {0x00});
    fillDeviceLayer(device_pool_, 1, device_block_, {0x00});

    auto h2d = makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, host_block);
    ASSERT_TRUE(copy_engine->submit(h2d).ok());

    auto d0 = readDeviceLayer(device_pool_, 0, device_block_);
    auto d1 = readDeviceLayer(device_pool_, 1, device_block_);
    for (auto b : d0)
        EXPECT_EQ(b, 0x11);
    for (auto b : d1)
        EXPECT_EQ(b, 0x22);
    EXPECT_EQ(counters[0].not_applicable, 2);
#if CUDART_VERSION >= 12080
    EXPECT_EQ(counters[1].done, 2);
    EXPECT_EQ(counters[1].not_applicable, 0);
    EXPECT_EQ(counters[2].attempts, 0);
#else
    EXPECT_EQ(counters[1].done, 0);
    EXPECT_EQ(counters[1].not_applicable, 2);
    EXPECT_EQ(counters[2].attempts, 2);
    EXPECT_EQ(counters[2].done, 2);
#endif

    host_pool_->free(host_block);
}

TEST_F(CopyEngineStrategyTest, BatchNotApplicableFallsBackToGeneric) {
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
    plan.device_to_host     = true;
    plan.single_device      = true;
    plan.component_group_id = 0;
    plan.host               = {host_data, device_buffer.size_bytes};
    plan.copy_tiles.push_back(
        DeviceHostCopyTile{host_data, device_buffer.addr, 0, device_buffer.size_bytes, -1, 0, 0});
    EXPECT_EQ(executor.executeStrategies(plan), CopyStatus::OK);
    for (size_t i = 0; i < device_buffer.size_bytes; ++i)
        EXPECT_EQ(host_data[i], 0x71);
    EXPECT_EQ(counters[1].not_applicable, 1);
    EXPECT_EQ(counters[2].done, 1);
}

TEST_F(CopyEngineStrategyTest, StagedEnabledBelowThresholdFallsBackToGeneric) {
    DeviceHostCopyOptions options;
    options.staged_sm_copy_enabled              = true;
    options.staged_sm_min_tile_count            = 100;
    options.staged_sm_min_bytes                 = 0;
    options.cuda_batch_copy_enabled             = false;
    auto                            copy_engine = makeCopyEngine(options);
    std::array<StrategyCounters, 3> counters;
    installStrategyRecorders(*copy_engine->device_host_executor_, counters);

    fillDeviceLayer(device_pool_, 0, device_block_, {0xCC});
    fillDeviceLayer(device_pool_, 1, device_block_, {0xDD});

    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    auto d2h = makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block);
    ASSERT_TRUE(copy_engine->submit(d2h).ok());

    const auto* host_data = static_cast<const uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    for (size_t i = 0; i < 128; ++i)
        EXPECT_EQ(host_data[i], 0xCC);
    for (size_t i = 128; i < 384; ++i)
        EXPECT_EQ(host_data[i], 0xDD);
    EXPECT_EQ(counters[0].not_applicable, 1);
    EXPECT_EQ(counters[1].not_applicable, 1);
    EXPECT_EQ(counters[2].done, 1);

    host_pool_->free(host_block);
}

TEST_F(CopyEngineStrategyTest, StagedStrategyAboveThresholdRoundTrip) {
    DeviceHostCopyOptions options;
    options.staged_sm_copy_enabled              = true;
    options.staged_sm_min_tile_count            = 1;
    options.staged_sm_min_bytes                 = 1;
    options.cuda_batch_copy_enabled             = false;
    auto                            copy_engine = makeCopyEngine(options);
    std::array<StrategyCounters, 3> counters;
    installStrategyRecorders(*copy_engine->device_host_executor_, counters);

    fillDeviceLayer(device_pool_, 0, device_block_, {0x31});
    fillDeviceLayer(device_pool_, 1, device_block_, {0x42});

    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    auto d2h = makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block);
    ASSERT_TRUE(copy_engine->submit(d2h).ok());
    const auto* host_data = static_cast<const uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    for (size_t i = 0; i < 128; ++i)
        EXPECT_EQ(host_data[i], 0x31);
    for (size_t i = 128; i < 384; ++i)
        EXPECT_EQ(host_data[i], 0x42);

    fillDeviceLayer(device_pool_, 0, device_block_, {0x00});
    fillDeviceLayer(device_pool_, 1, device_block_, {0x00});
    auto h2d = makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, host_block);
    ASSERT_TRUE(copy_engine->submit(h2d).ok());
    for (auto byte : readDeviceLayer(device_pool_, 0, device_block_))
        EXPECT_EQ(byte, 0x31);
    for (auto byte : readDeviceLayer(device_pool_, 1, device_block_))
        EXPECT_EQ(byte, 0x42);
    EXPECT_EQ(counters[0].done, 2);
    EXPECT_EQ(counters[1].attempts, 0);
    EXPECT_EQ(counters[2].attempts, 0);

    host_pool_->free(host_block);
}

TEST_F(CopyEngineStrategyTest, StagedStrategyTakesPrecedenceWhenEligible) {
    DeviceHostCopyOptions options;
    options.staged_sm_copy_enabled              = true;
    options.staged_sm_min_tile_count            = 1;
    options.staged_sm_min_bytes                 = 1;
    options.cuda_batch_copy_enabled             = true;
    auto                            copy_engine = makeCopyEngine(options);
    std::array<StrategyCounters, 3> counters;
    installStrategyRecorders(*copy_engine->device_host_executor_, counters);

    fillDeviceLayer(device_pool_, 0, device_block_, {0x5C});
    fillDeviceLayer(device_pool_, 1, device_block_, {0x6D});
    auto host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    expectStatus(copy_engine, makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block), CopyStatus::OK);
    EXPECT_EQ(counters[0].done, 1);
    EXPECT_EQ(counters[1].attempts, 0);
    EXPECT_EQ(counters[2].attempts, 0);

    host_pool_->free(host_block);
}

}  // namespace
}  // namespace rtp_llm

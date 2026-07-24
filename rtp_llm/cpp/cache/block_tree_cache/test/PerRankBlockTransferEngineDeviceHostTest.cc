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
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyStrategy.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/PerRankBlockTransferEngineTestUtils.h"

namespace rtp_llm {
namespace {

using block_transfer_engine_test::TempDirGuard;
using block_transfer_engine_test::expectStatus;
using block_transfer_engine_test::makeDescriptor;
using block_transfer_engine_test::makeDiskPool;
using block_transfer_engine_test::makeHostPool;
using block_transfer_engine_test::poolMalloc;

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

static Component makeComponent(int                        component_id,
                               int                        component_group_id,
                               const std::vector<size_t>& layer_bytes,
                               const std::string&         tag             = "",
                               const std::vector<int>&    model_layer_ids = {}) {
    return block_transfer_engine_test::makeSchemaComponent(component_id,
                                                           component_group_id,
                                                           tag.empty() ? ("comp_" + std::to_string(component_id)) : tag,
                                                           layer_bytes,
                                                           model_layer_ids);
}

static ComponentGroupPtr makeDeviceHostGroup(int                                     group_id,
                                             std::vector<int>                        component_indices,
                                             std::vector<DeviceBlockPoolPtr>         device_pools,
                                             std::shared_ptr<HostBlockPool>          host_pool,
                                             const std::vector<Component>&           components,
                                             std::shared_ptr<BlockTreeDiskBlockPool> disk_pool = nullptr) {
    auto group                = std::make_shared<FullComponentGroup>();
    group->component_group_id = group_id;
    group->group_type         = CacheGroupType::FULL;
    group->setDevicePools(std::move(device_pools));
    group->setHostPool(std::move(host_pool));
    group->setDiskPool(std::move(disk_pool));
    (void)group->finalizeLayout(std::move(component_indices), components);
    return group;
}

static std::shared_ptr<PerRankBlockTransferEngine> makeEngine(std::vector<ComponentGroupPtr> groups,
                                                              std::vector<Component>         components,
                                                              DeviceHostCopyOptions          options = {}) {
    return std::make_shared<PerRankBlockTransferEngine>(
        std::move(groups),
        block_transfer_engine_test::makeComponentRegistry(std::move(components)),
        std::move(options));
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

        layer_bytes_     = {100, 200, 150};
        host_block_size_ = 450;

        // Create host pool — 10 usable blocks
        host_pool_ = makeHostPool(host_block_size_, 10, true);

        device_pool_  = makeDevicePool({{100, 0}, {200, 0}, {150, 0}}, 10, "per_rank_transfer_engine_device");
        device_block_ = poolMalloc(*device_pool_);
        ASSERT_NE(device_block_, NULL_BLOCK_IDX);
        device_blocks_ = {device_block_};

        component_       = makeComponent(0, 0, layer_bytes_);
        component_group_ = makeDeviceHostGroup(0, {0}, {device_pool_}, host_pool_, {component_});
        ASSERT_TRUE(component_group_->hasLayout());
        ASSERT_EQ(component_group_->layout().payloadBytes(), host_block_size_);
        per_rank_transfer_engine_ = makeEngine({component_group_}, {component_});
    }

    std::vector<size_t>                         layer_bytes_;
    size_t                                      host_block_size_;
    std::shared_ptr<HostBlockPool>              host_pool_;
    std::shared_ptr<PerRankBlockTransferEngine> per_rank_transfer_engine_;
    DeviceBlockPoolPtr                          device_pool_;
    BlockIdxType                                device_block_;
    std::vector<BlockIdxType>                   device_blocks_;
    Component                                   component_;
    ComponentGroupPtr                           component_group_;
};

TEST_F(PerRankBlockTransferEngineTest, SubmitDeviceHostRoundTripPreservesLayout) {
    fillDeviceLayerSequential(device_pool_, 0, device_block_);
    fillDeviceLayer(device_pool_, 1, device_block_, {0x5A});
    fillDeviceLayerSequential(device_pool_, 2, device_block_);

    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    auto d2h_desc = makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block);
    ASSERT_TRUE(per_rank_transfer_engine_->submit(d2h_desc).ok());

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
    ASSERT_TRUE(per_rank_transfer_engine_->submit(h2d_desc).ok());

    auto d0 = readDeviceLayer(device_pool_, 0, device_block_);
    auto d1 = readDeviceLayer(device_pool_, 1, device_block_);
    auto d2 = readDeviceLayer(device_pool_, 2, device_block_);
    for (size_t i = 0; i < layer_bytes_[0]; ++i)
        EXPECT_EQ(d0[i], static_cast<uint8_t>(i & 0xFF));
    for (auto byte : d1)
        EXPECT_EQ(byte, 0x5A);
    for (size_t i = 0; i < layer_bytes_[2]; ++i)
        EXPECT_EQ(d2[i], static_cast<uint8_t>(i & 0xFF));

    host_pool_->free(host_block);
}

TEST_F(PerRankBlockTransferEngineTest, SharedDevicePoolComponentsIsolateByBlockId) {
    auto shared_pool = makeDevicePool({{64, 0}, {32, 0}}, 4, "per_rank_transfer_engine_shared_pool");
    auto host_pool   = makeHostPool(192, 4, true);
    // Non-identity model layer ids: any path that leaked them into pool lookups
    // would address out-of-range slots instead of descriptor-local {0, 1}.
    auto comp_a = makeComponent(0, 0, {64, 32}, "tag_a", {2, 5});
    auto comp_b = makeComponent(1, 0, {64, 32}, "tag_b", {3, 7});
    auto group  = makeDeviceHostGroup(0, {0, 1}, {shared_pool, shared_pool}, host_pool, {comp_a, comp_b});
    auto engine = makeEngine({group}, {comp_a, comp_b});

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
    ASSERT_TRUE(engine->submit(makeDescriptor(Tier::DEVICE, Tier::HOST, {block_a, block_b}, host_block)).ok());

    const uint8_t* host_data = static_cast<const uint8_t*>(host_pool->blockBuffer(host_block).addr);
    for (size_t i = 0; i < 64; ++i)
        EXPECT_EQ(host_data[i], 0xA0);
    for (size_t i = 0; i < 32; ++i)
        EXPECT_EQ(host_data[64 + i], 0xA1);
    for (size_t i = 0; i < 64; ++i)
        EXPECT_EQ(host_data[96 + i], 0xB0);
    for (size_t i = 0; i < 32; ++i)
        EXPECT_EQ(host_data[160 + i], 0xB1);

    fillDeviceLayer(shared_pool, 0, block_a, {0x00});
    fillDeviceLayer(shared_pool, 1, block_a, {0x00});
    fillDeviceLayer(shared_pool, 0, block_b, {0x00});
    fillDeviceLayer(shared_pool, 1, block_b, {0x00});
    ASSERT_TRUE(engine->submit(makeDescriptor(Tier::HOST, Tier::DEVICE, {block_a, block_b}, host_block)).ok());

    for (auto byte : readDeviceLayer(shared_pool, 0, block_a))
        EXPECT_EQ(byte, 0xA0);
    for (auto byte : readDeviceLayer(shared_pool, 1, block_a))
        EXPECT_EQ(byte, 0xA1);
    for (auto byte : readDeviceLayer(shared_pool, 0, block_b))
        EXPECT_EQ(byte, 0xB0);
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

TEST_F(PerRankBlockTransferEngineTest, SubmitRejectsInvalidComponentGroupId) {
    for (int group_id : {-1, 99}) {
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

    auto missing_host_group  = makeDeviceHostGroup(0, {0}, {device_pool_}, nullptr, {component_});
    auto missing_host_engine = makeEngine({missing_host_group}, {component_});
    expectStatus(missing_host_engine, desc, TransferStatus::INVALID_ARGS);

    auto empty_group  = makeDeviceHostGroup(0, {}, {device_pool_}, host_pool_, {});
    auto empty_engine = makeEngine({empty_group}, {});
    expectStatus(empty_engine, desc, TransferStatus::INVALID_ARGS);

    auto empty_component    = makeComponent(0, 0, std::vector<size_t>{});
    auto empty_slots_group  = makeDeviceHostGroup(0, {0}, {device_pool_}, host_pool_, {empty_component});
    auto empty_slots_engine = makeEngine({empty_slots_group}, {empty_component});
    expectStatus(empty_slots_engine, desc, TransferStatus::INVALID_ARGS);
    host_pool_->free(host_block);
}

TEST_F(PerRankBlockTransferEngineTest, SubmitRejectsInvalidLayerSlotLayout) {
    {
        auto host_pool   = makeHostPool(64, 2, true);
        auto device_pool = makeDevicePool({{64, 0}, {16, 0}}, 2, "per_rank_transfer_engine_zero_stride");
        auto block       = poolMalloc(*device_pool);
        auto host_block  = poolMalloc(*host_pool);
        auto component   = makeComponent(0, 0, {64, 0});
        auto group       = makeDeviceHostGroup(0, {0}, {device_pool}, host_pool, {component});
        EXPECT_FALSE(group->hasLayout());
        auto engine = makeEngine({group}, {component});
        expectStatus(
            engine, makeDescriptor(Tier::DEVICE, Tier::HOST, {block}, host_block), TransferStatus::INVALID_ARGS);
    }

    {
        auto host_pool   = makeHostPool(65, 2, true);
        auto device_pool = makeDevicePool({{64, 0}}, 2, "per_rank_transfer_engine_slot_mismatch");
        auto block       = poolMalloc(*device_pool);
        auto host_block  = poolMalloc(*host_pool);
        auto component   = makeComponent(0, 0, {65});
        auto group       = makeDeviceHostGroup(0, {0}, {device_pool}, host_pool, {component});
        auto engine      = makeEngine({group}, {component});
        expectStatus(
            engine, makeDescriptor(Tier::DEVICE, Tier::HOST, {block}, host_block), TransferStatus::INVALID_ARGS);
        expectStatus(
            engine, makeDescriptor(Tier::HOST, Tier::DEVICE, {block}, host_block), TransferStatus::INVALID_ARGS);
    }
}

TEST_F(PerRankBlockTransferEngineTest, UnusableCopyBufferReturnsDeviceIoError) {
    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    device_pool_->layout_strategies_[0]->config_.kv_block_stride_bytes = 0;
    expectStatus(per_rank_transfer_engine_,
                 makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block),
                 TransferStatus::DEVICE_IO_ERROR);
    expectStatus(per_rank_transfer_engine_,
                 makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, host_block),
                 TransferStatus::DEVICE_IO_ERROR);
    host_pool_->free(host_block);
}

TEST_F(PerRankBlockTransferEngineTest, SubmitReturnsCompletedHandleWithFinalStatus) {
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

    uint64_t previous_request_id = 0;
    for (const auto& [desc, expected] : cases) {
        auto handle = per_rank_transfer_engine_->submit(desc);
        ASSERT_TRUE(handle.valid());
        EXPECT_TRUE(handle.done());
        handle.wait();
        EXPECT_EQ(handle.status(), expected);
        EXPECT_EQ(handle.ok(), expected == TransferStatus::OK);
        EXPECT_GT(handle.requestId(), previous_request_id);
        previous_request_id = handle.requestId();

        bool callback_called = false;
        handle.onComplete([&](TransferStatus status) {
            callback_called = true;
            EXPECT_EQ(status, expected);
        });
        EXPECT_TRUE(callback_called);
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

    auto result_1 =
        per_rank_transfer_engine_->submit(makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, host_block_1));
    ASSERT_TRUE(result_1.ok());

    auto result_2 =
        per_rank_transfer_engine_->submit(makeDescriptor(Tier::HOST, Tier::DEVICE, second_device_blocks, host_block_2));
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

class PerRankBlockTransferEngineMixedNullTest: public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
        host_pool_ = makeHostPool(168, 4, true);
        pools_     = {
            makeDevicePool({{64, 16}}, 4, "per_rank_transfer_engine_mixed_0"),
            makeDevicePool({{48, 0}}, 4, "per_rank_transfer_engine_mixed_1"),
            makeDevicePool({{32, 8}}, 4, "per_rank_transfer_engine_mixed_2"),
        };
        for (const auto& pool : pools_) {
            blocks_.push_back(poolMalloc(*pool));
            ASSERT_NE(blocks_.back(), NULL_BLOCK_IDX);
        }

        std::vector<Component> components = {
            makeComponent(0, 0, {80}, "kv_scale_0"),
            makeComponent(1, 0, {48}, "missing"),
            makeComponent(2, 0, {40}, "kv_scale_2"),
        };
        // Pool bindings remain concrete in the declarative topology. The middle
        // descriptor block is NULL, so lowering must skip it without touching its pool.
        auto group  = makeDeviceHostGroup(0, {0, 1, 2}, pools_, host_pool_, components);
        engine_     = makeEngine({group}, components);
        host_block_ = poolMalloc(*host_pool_);
        ASSERT_NE(host_block_, NULL_BLOCK_IDX);
    }

    std::shared_ptr<HostBlockPool>              host_pool_;
    std::vector<DeviceBlockPoolPtr>             pools_;
    std::vector<BlockIdxType>                   blocks_;
    std::shared_ptr<PerRankBlockTransferEngine> engine_;
    BlockIdxType                                host_block_{NULL_BLOCK_IDX};
};

TEST_F(PerRankBlockTransferEngineMixedNullTest, DeviceToHostMixedNullComponentsPreserveOffsets) {
    fillDeviceLayer(pools_[0], 0, blocks_[0], {0xA1, 0xA2});
    fillDeviceLayer(pools_[1], 0, blocks_[1], {0xB1});
    fillDeviceLayer(pools_[2], 0, blocks_[2], {0xC1, 0xC2});
    auto* host_data = static_cast<uint8_t*>(host_pool_->blockBuffer(host_block_).addr);
    std::memset(host_data, 0xFF, 168);

    expectStatus(engine_,
                 makeDescriptor(Tier::DEVICE, Tier::HOST, {blocks_[0], NULL_BLOCK_IDX, blocks_[2]}, host_block_),
                 TransferStatus::OK);
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

TEST_F(PerRankBlockTransferEngineMixedNullTest, HostToDeviceMixedNullComponentsPreserveOffsets) {
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
                 TransferStatus::OK);

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

    auto component = makeComponent(0, 0, {payload_bytes}, "kv_scale");
    auto group     = makeDeviceHostGroup(0, {0}, {device_pool}, host_pool, {component}, disk_pool);
    auto engine    = makeEngine({group}, {component});
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

    // Clear Device again after the disk read so the final bytes can only come from the full path.
    fillDeviceLayer(device_pool, 0, device_block, {0x00, 0x00});
    expectStatus(engine,
                 makeDescriptor(Tier::HOST, Tier::DEVICE, {device_block}, host_block, NULL_BLOCK_IDX, 0),
                 TransferStatus::OK);
    EXPECT_EQ(readDeviceLayer(device_pool, 0, device_block), expected);
}

// ---- Strategy chain tests ----

class PerRankBlockTransferEngineStrategyTest: public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";

        layer_bytes_     = {128, 256};
        host_block_size_ = 384;

        host_pool_    = makeHostPool(host_block_size_, 10, true);
        device_pool_  = makeDevicePool({{128, 0}, {256, 0}}, 10, "strategy_test_device");
        device_block_ = poolMalloc(*device_pool_);
        ASSERT_NE(device_block_, NULL_BLOCK_IDX);
        device_blocks_ = {device_block_};

        component_       = makeComponent(0, 0, layer_bytes_);
        component_group_ = makeDeviceHostGroup(0, {0}, {device_pool_}, host_pool_, {component_});
        ASSERT_TRUE(component_group_->hasLayout());
    }

    std::shared_ptr<PerRankBlockTransferEngine> makePerRankBlockTransferEngine(DeviceHostCopyOptions options = {}) {
        return makeEngine({component_group_}, {component_}, std::move(options));
    }

    std::vector<size_t>            layer_bytes_;
    size_t                         host_block_size_;
    std::shared_ptr<HostBlockPool> host_pool_;
    DeviceBlockPoolPtr             device_pool_;
    BlockIdxType                   device_block_;
    std::vector<BlockIdxType>      device_blocks_;
    Component                      component_;
    ComponentGroupPtr              component_group_;
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
    ASSERT_TRUE(per_rank_transfer_engine->submit(d2h).ok());

    const auto* host_data = static_cast<const uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    for (size_t i = 0; i < 128; ++i)
        EXPECT_EQ(host_data[i], 0xAA);
    for (size_t i = 128; i < 384; ++i)
        EXPECT_EQ(host_data[i], 0xBB);

    fillDeviceLayer(device_pool_, 0, device_block_, {0x00});
    fillDeviceLayer(device_pool_, 1, device_block_, {0x00});

    auto h2d = makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, host_block);
    ASSERT_TRUE(per_rank_transfer_engine->submit(h2d).ok());

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
    ASSERT_TRUE(per_rank_transfer_engine->submit(d2h).ok());

    const auto* host_data = static_cast<const uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    for (size_t i = 0; i < 128; ++i)
        EXPECT_EQ(host_data[i], 0x11);
    for (size_t i = 128; i < 384; ++i)
        EXPECT_EQ(host_data[i], 0x22);

    fillDeviceLayer(device_pool_, 0, device_block_, {0x00});
    fillDeviceLayer(device_pool_, 1, device_block_, {0x00});

    auto h2d = makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, host_block);
    ASSERT_TRUE(per_rank_transfer_engine->submit(h2d).ok());

    auto d0 = readDeviceLayer(device_pool_, 0, device_block_);
    auto d1 = readDeviceLayer(device_pool_, 1, device_block_);
    for (auto b : d0)
        EXPECT_EQ(b, 0x11);
    for (auto b : d1)
        EXPECT_EQ(b, 0x22);

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
    plan.device_to_host     = true;
    plan.single_device      = true;
    plan.component_group_id = 0;
    plan.host               = {host_data, device_buffer.size_bytes};
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
    ASSERT_TRUE(per_rank_transfer_engine->submit(d2h).ok());

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
    ASSERT_TRUE(per_rank_transfer_engine->submit(d2h).ok());
    const auto* host_data = static_cast<const uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    for (size_t i = 0; i < 128; ++i)
        EXPECT_EQ(host_data[i], 0x31);
    for (size_t i = 128; i < 384; ++i)
        EXPECT_EQ(host_data[i], 0x42);

    fillDeviceLayer(device_pool_, 0, device_block_, {0x00});
    fillDeviceLayer(device_pool_, 1, device_block_, {0x00});
    auto h2d = makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, host_block);
    ASSERT_TRUE(per_rank_transfer_engine->submit(h2d).ok());
    for (auto byte : readDeviceLayer(device_pool_, 0, device_block_))
        EXPECT_EQ(byte, 0x31);
    for (auto byte : readDeviceLayer(device_pool_, 1, device_block_))
        EXPECT_EQ(byte, 0x42);
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

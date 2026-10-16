#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <functional>
#include <c10/cuda/CUDAGuard.h>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"

namespace rtp_llm {
namespace {
using namespace block_transfer_engine_test;

DeviceBlockPoolPtr regularPool(size_t width, size_t layers) {
    auto config                       = std::make_shared<DeviceBlockPoolConfig>();
    config->pool_type                 = BlockPoolType::DEVICE;
    config->pool_name                 = "template-test";
    config->physical_block_count      = 5;
    config->use_device_malloc_backing = true;
    MemoryLayoutConfig layout;
    layout.layer_num                = layers;
    layout.block_num                = 5;
    layout.dtype                    = TYPE_INT8;
    layout.kv_block_stride_bytes    = width;
    layout.block_stride_bytes       = width;
    layout.kv_block_pool_size_bytes = width * layers * 5;
    layout.total_size_bytes         = layout.kv_block_pool_size_bytes;
    config->total_size_bytes        = layout.total_size_bytes;
    config->memory_layouts          = {layout};
    auto pool                       = std::make_shared<DeviceBlockPool>(config);
    EXPECT_TRUE(pool->init());
    return pool;
}

GroupSetPtr templateGroup(CacheGroupType type) {
    auto policy                = defaultCacheGroupPolicy(type);
    policy.sliding_window_size = type == CacheGroupType::SWA ? 2 : 0;
    auto topology              = makeTestTopology({makeTestGroupBase(policy, {0, 1}, 4),
                                                   makeTestGroupBase(policy, {0, 1, 2}, 8),
                                                   makeTestGroupBase(policy, {0}, 16)});
    return makeTestGroupSet(
        0, topology, {"group0", "group1", "group2"}, {regularPool(4, 2), regularPool(8, 3), regularPool(16, 1)});
}

TEST(DeviceHostCopyTemplateTest, CoversEveryLegacyTileForFullAndSwa) {
    BlockTreeTaskPool          tasks(1, 8, "TemplateTest");
    DeviceHostTransferExecutor executor(tasks, 8);
    for (auto type : {CacheGroupType::FULL, CacheGroupType::SWA}) {
        auto        group     = templateGroup(type);
        const auto& templates = group->copy3DTemplates();
        ASSERT_EQ(templates.size(), 3u);
        EXPECT_EQ(templates[0].host_offset, 0u);
        EXPECT_EQ(templates[1].host_offset, 8u);
        EXPECT_EQ(templates[2].host_offset, 32u);
        EXPECT_EQ(templates[0].device_pitch, 20u);
        EXPECT_EQ(templates[1].device_pitch, 40u);
        EXPECT_EQ(templates[2].device_pitch, 80u);
        std::vector<uint8_t> host(48);
        HostBufferView       view{host.data(), 48, 48};
        // Independent member allocations deliberately have different block IDs.
        const std::vector<BlockIdxType> blocks{1, 2, 3};
        for (bool d2h : {false, true}) {
            auto desc =
                d2h ? TransferDescriptor::deviceToHost(0, blocks, 1) : TransferDescriptor::hostToDevice(0, 1, blocks);
            auto [status, plans] = executor.generatePlan({view}, {desc}, {group.get()});
            ASSERT_EQ(status, TransferStatus::OK);
            ASSERT_EQ(plans.size(), 1u);
            ASSERT_EQ(plans[0].copy_tiles.size(), 6u);
            size_t index = 0;
            for (const auto& t : templates) {
                for (size_t layer = 0; layer < t.layer_count; ++layer) {
                    const auto& tile = plans[0].copy_tiles[index++];
                    EXPECT_EQ(tile.host_addr, host.data() + t.host_offset + layer * t.width_bytes);
                    EXPECT_EQ(tile.device_addr,
                              static_cast<uint8_t*>(t.device_base) + blocks[t.member_index] * t.width_bytes
                                  + layer * t.device_pitch);
                    EXPECT_EQ(tile.bytes, t.width_bytes);
                }
            }
        }
    }
}

TEST(DeviceHostCopyTemplateTest, ReusesTemplatesUntilTopologyChanges) {
    auto        group     = templateGroup(CacheGroupType::FULL);
    const auto* templates = group->copy3DTemplates().data();
    group->initialize(group->groupSetId(), group->topologyPtr(), group->groupTags());
    EXPECT_EQ(group->copy3DTemplates().data(), templates);
    auto prefix = makeTestTopology({makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0, 1}, 2),
                                    makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0, 1, 2}, 8),
                                    makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, 16)});
    group->initialize(0, prefix, {"group0", "group1", "group2"});
    EXPECT_TRUE(group->copy3DTemplates().empty());
}

TEST(DeviceHostCopyTemplateTest, RejectsWidthMismatchAndIndependentScaleTemplates) {
    auto pool  = regularPool(16, 2);
    auto group = makeTestGroupSet(
        0,
        makeTestTopology({makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0, 1}, 8)}),
        {"group0"},
        {pool});
    EXPECT_TRUE(group->copy3DTemplates().empty());
    auto scale_pool = makeTestDevicePool({{16, 4}}, 4, "scale-fallback");
    auto scaled     = makeTestGroupSet(
        0,
        makeTestTopology({makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, 16, 4)}),
        {"group0"},
        {scale_pool});
    EXPECT_TRUE(scaled->copy3DTemplates().empty());
}

TEST(DeviceHostCopyTemplateTest, ReinitializationUpdatesCrcBackingLayout) {
    auto group = templateGroup(CacheGroupType::FULL);
    group->initialize(0, group->topologyPtr(), group->groupTags(), BackingLayout{0, 64, 64, true});
    EXPECT_TRUE(group->crcEnabled());
    EXPECT_EQ(group->payloadBytes(), 48u);
    EXPECT_EQ(group->storageBytes(), 64u);
    group->initialize(0, group->topologyPtr(), group->groupTags());
    EXPECT_FALSE(group->crcEnabled());
    EXPECT_EQ(group->storageBytes(), 48u);
}

TEST(DeviceHostCopyTemplateTest, RealCudaRoundTripPreservesOtherBlocks) {
    auto                            group = templateGroup(CacheGroupType::FULL);
    const std::vector<BlockIdxType> first_blocks{1, 2, 3}, second_blocks{3, 1, 2};
    void*                           raw = nullptr;
    ASSERT_EQ(cudaMallocHost(&raw, 192), cudaSuccess);
    std::unique_ptr<void, decltype(&cudaFreeHost)> storage(raw, cudaFreeHost);
    auto*                                          input  = static_cast<uint8_t*>(raw);
    auto*                                          output = input + 96;
    for (size_t i = 0; i < 96; ++i) {
        input[i]  = static_cast<uint8_t>(i + 1);
        output[i] = 0;
    }
    for (const auto& pool : group->devicePools()) {
        ASSERT_EQ(cudaMemset(pool->getBaseAddress(), 0xA5, pool->getTotalSizeBytes()), cudaSuccess);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    Cuda3DBatchDeviceHostCopyStrategy strategy;
    auto                              result = strategy.tryExecute(
        {{input, 48, 48}, {input + 48, 48, 48}},
        {TransferDescriptor::hostToDevice(0, 1, first_blocks), TransferDescriptor::hostToDevice(0, 2, second_blocks)},
        {group.get(), group.get()});
    if (result.status == StrategyStatus::NOT_APPLICABLE) {
        GTEST_SKIP() << "Installed CUDA runtime/driver does not support 3D batch";
    }
    ASSERT_EQ(result.status, StrategyStatus::DONE);
    ASSERT_EQ(strategy
                  .tryExecute({{output, 48, 48}, {output + 48, 48, 48}},
                              {TransferDescriptor::deviceToHost(0, first_blocks, 1),
                               TransferDescriptor::deviceToHost(0, second_blocks, 2)},
                              {group.get(), group.get()})
                  .status,
              StrategyStatus::DONE);
    EXPECT_TRUE(std::equal(input, input + 96, output));
    // Compare the complete physical pools, including every unselected block.
    for (size_t member = 0; member < group->devicePools().size(); ++member) {
        const auto&          pool = group->devicePools()[member];
        std::vector<uint8_t> expected(pool->getTotalSizeBytes(), 0xA5), actual(expected.size());
        const auto&          t = group->copy3DTemplates()[member];
        for (size_t layer = 0; layer < t.layer_count; ++layer) {
            std::copy_n(input + t.host_offset + layer * t.width_bytes,
                        t.width_bytes,
                        expected.data() + layer * t.device_pitch + first_blocks[member] * t.width_bytes);
            std::copy_n(input + 48 + t.host_offset + layer * t.width_bytes,
                        t.width_bytes,
                        expected.data() + layer * t.device_pitch + second_blocks[member] * t.width_bytes);
        }
        ASSERT_EQ(cudaMemcpy(actual.data(), pool->getBaseAddress(), actual.size(), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(actual, expected);
    }
}

class BatchProbe: public Cuda3DBatchDeviceHostCopyStrategy {
public:
    explicit BatchProbe(std::function<StrategyResult()> action): action_(std::move(action)) {}
    StrategyResult tryExecute(const std::vector<HostBufferView>&,
                              const std::vector<TransferDescriptor>&,
                              const std::vector<const GroupSet*>&) override {
        return action_();
    }

private:
    std::function<StrategyResult()> action_;
};

class TileProbe: public DeviceHostCopyStrategy {
public:
    explicit TileProbe(std::function<StrategyResult(const DeviceHostCopyPlan&)> action): action_(std::move(action)) {}
    StrategyResult tryExecute(const DeviceHostCopyPlan& plan, const DeviceHostCopyOptions&) override {
        return action_(plan);
    }

private:
    std::function<StrategyResult(const DeviceHostCopyPlan&)> action_;
};

TEST(DeviceHostCopyTemplateTest, RejectsMismatchedCopyWidthBeforeLegacyDispatch) {
    BlockTreeTaskPool tasks(1, 8, "WidthMismatchTest");
    for (const auto [physical_width, copy_width] :
         {std::pair<size_t, size_t>{16, 8}, std::pair<size_t, size_t>{8, 16}}) {
        SCOPED_TRACE(::testing::Message() << "physical=" << physical_width << " copy=" << copy_width);
        auto pool  = regularPool(physical_width, 1);
        auto group = makeTestGroupSet(
            0,
            makeTestTopology({makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, copy_width)}),
            {"group0"},
            {pool});
        std::vector<uint8_t>       host(copy_width);
        DeviceHostTransferExecutor executor(tasks, 8);
        executor.strategies_.clear();
        executor.strategies_.push_back(
            std::make_unique<TileProbe>([](const DeviceHostCopyPlan&) { return StrategyResult::done(); }));
        EXPECT_EQ(executor
                      .executeBatch({{host.data(), copy_width, copy_width}},
                                    {TransferDescriptor::hostToDevice(0, 1, {1})},
                                    {group.get()})
                      .copy_error,
                  CacheCopyError::INVALID_REQUEST);
    }
}

TEST(DeviceHostCopyTemplateTest, RejectsMixedDirectionBeforeStrategyDispatch) {
    BlockTreeTaskPool          tasks(1, 8, "MixedDirectionTest");
    DeviceHostTransferExecutor executor(tasks, 8);
    auto                       group = templateGroup(CacheGroupType::FULL);
    std::vector<uint8_t>       host(96);
    executor.strategies_.clear();
    executor.strategies_.push_back(std::make_unique<BatchProbe>([] { return StrategyResult::done(); }));
    EXPECT_EQ(executor
                  .executeBatch({{host.data(), 48, 48}, {host.data() + 48, 48, 48}},
                                {TransferDescriptor::hostToDevice(0, 1, {1, 2, 3}),
                                 TransferDescriptor::deviceToHost(0, {1, 2, 3}, 2)},
                                {group.get(), group.get()})
                  .copy_error,
              CacheCopyError::INVALID_REQUEST);
}

TEST(DeviceHostCopyTemplateTest, CrcBackingsDoNotDispatchPlainCopyStrategies) {
    BlockTreeTaskPool          tasks(1, 8, "CrcDispatchTest");
    DeviceHostTransferExecutor executor(tasks, 8);
    auto                       group = templateGroup(CacheGroupType::FULL);
    group->initialize(0, group->topologyPtr(), group->groupTags(), BackingLayout{0, 64, 64, true});
    size_t attempts = 0;
    executor.strategies_.clear();
    executor.strategies_.push_back(std::make_unique<BatchProbe>([&] {
        ++attempts;
        return StrategyResult::done();
    }));
    std::vector<uint8_t> host(64);
    EXPECT_EQ(
        executor
            .executeBatch({{host.data(), 48, 64}}, {TransferDescriptor::hostToDevice(0, 1, {1, 2, 3})}, {group.get()})
            .copy_error,
        CacheCopyError::INVALID_REQUEST);
    EXPECT_EQ(attempts, 0u);
}

TEST(DeviceHostCopyTemplateTest, RejectsMismatchedMemberBlocksBeforeStrategyDispatch) {
    BlockTreeTaskPool          tasks(1, 8, "MemberBlockCountTest");
    DeviceHostTransferExecutor executor(tasks, 8);
    auto                       group = templateGroup(CacheGroupType::FULL);
    std::vector<uint8_t>       host(96);
    int                        attempts = 0;
    executor.strategies_.clear();
    executor.strategies_.push_back(std::make_unique<BatchProbe>([&] {
        ++attempts;
        return StrategyResult::done();
    }));
    EXPECT_EQ(executor
                  .executeBatch({{host.data(), 48, 48}, {host.data() + 48, 48, 48}},
                                {TransferDescriptor::hostToDevice(0, 1, {1, 2, 3}),
                                 TransferDescriptor::hostToDevice(0, 2, {1, 2})},
                                {group.get(), group.get()})
                  .copy_error,
              CacheCopyError::INVALID_REQUEST);
    EXPECT_EQ(attempts, 0);
}

TEST(DeviceHostCopyTemplateTest, Disabled3DBatchFallsBackToLegacyStrategy) {
    BlockTreeTaskPool     tasks(1, 8, "Disabled3DBatchTest");
    DeviceHostCopyOptions options;
    options.cuda_3d_batch_copy_enabled = false;
    DeviceHostTransferExecutor executor(tasks, 8, options);
    auto                       group = templateGroup(CacheGroupType::FULL);
    std::vector<uint8_t>       host(48);
    int                        three_d_attempts = 0;
    int                        legacy_attempts  = 0;
    executor.strategies_.clear();
    executor.strategies_.push_back(std::make_unique<BatchProbe>([&] {
        ++three_d_attempts;
        return StrategyResult::done();
    }));
    executor.strategies_.push_back(std::make_unique<TileProbe>([&](const DeviceHostCopyPlan&) {
        ++legacy_attempts;
        return StrategyResult::done();
    }));
    EXPECT_TRUE(
        executor
            .executeBatch({{host.data(), 48, 48}}, {TransferDescriptor::hostToDevice(0, 1, {1, 2, 3})}, {group.get()})
            .ok());
    EXPECT_EQ(three_d_attempts, 0);
    EXPECT_EQ(legacy_attempts, 1);
}

TEST(DeviceHostCopyTemplateTest, NotApplicableFallsBackButFailedStops) {
    BlockTreeTaskPool    tasks(1, 8, "FallbackTest");
    auto                 group = templateGroup(CacheGroupType::FULL);
    std::vector<uint8_t> host(48);
    for (bool failed : {false, true}) {
        DeviceHostTransferExecutor executor(tasks, 8);
        executor.strategies_.clear();
        int legacy_attempts = 0;
        executor.strategies_.push_back(std::make_unique<BatchProbe>([&] {
            return failed ? StrategyResult::failed(TransferStatus::DEVICE_IO_ERROR) : StrategyResult::notApplicable();
        }));
        executor.strategies_.push_back(std::make_unique<TileProbe>([&](const DeviceHostCopyPlan& plan) {
            ++legacy_attempts;
            EXPECT_EQ(plan.copy_tiles.size(), 6u);
            return StrategyResult::done();
        }));
        EXPECT_EQ(executor
                      .executeBatch(
                          {{host.data(), 48, 48}}, {TransferDescriptor::hostToDevice(0, 1, {1, 2, 3})}, {group.get()})
                      .copy_error,
                  failed ? CacheCopyError::COPY_FAILED : CacheCopyError::NONE);
        EXPECT_EQ(legacy_attempts, failed ? 0 : 1);
    }
}

TEST(DeviceHostCopyTemplateTest, RejectsInvalid3DAddressesBeforeSubmission) {
    auto                              group = templateGroup(CacheGroupType::FULL);
    std::vector<uint8_t>              host(48);
    HostBufferView                    view{host.data(), 48, 48};
    auto                              desc = TransferDescriptor::hostToDevice(0, 1, {1, 2, 3});
    Cuda3DBatchDeviceHostCopyStrategy strategy;
    auto                              short_view = view;
    short_view.capacity_bytes                    = 47;
    auto result                                  = strategy.tryExecute({short_view}, {desc}, {group.get()});
    if (result.status == StrategyStatus::NOT_APPLICABLE) {
        GTEST_SKIP() << "Installed CUDA runtime/driver does not support 3D batch";
    }
    EXPECT_EQ(result.status, StrategyStatus::FAILED);
    EXPECT_EQ(result.copy_status, TransferStatus::INVALID_ARGS);
    for (const std::vector<BlockIdxType>& blocks :
         {std::vector<BlockIdxType>{1, 2, 5}, std::vector<BlockIdxType>{1, 2, -1}}) {
        auto invalid          = desc;
        invalid.target_blocks = blocks;
        result                = strategy.tryExecute({view, view}, {desc, invalid}, {group.get(), group.get()});
        EXPECT_EQ(result.status, StrategyStatus::FAILED);
        EXPECT_EQ(result.copy_status, TransferStatus::INVALID_ARGS);
    }
}

TEST(DeviceHostCopyTemplateTest, SeparatesLayoutsWithNonzeroOffsets) {
    auto config                       = std::make_shared<DeviceBlockPoolConfig>();
    config->pool_type                 = BlockPoolType::DEVICE;
    config->pool_name                 = "split-layout-test";
    config->physical_block_count      = 5;
    config->use_device_malloc_backing = true;
    MemoryLayoutConfig first;
    first.layer_num             = 2;
    first.block_num             = 5;
    first.dtype                 = TYPE_INT8;
    first.kv_block_stride_bytes = first.block_stride_bytes = 8;
    first.kv_cache_offset_bytes                            = 16;
    first.kv_block_pool_size_bytes = first.total_size_bytes = 80;
    auto second                                             = first;
    second.layer_num                                        = 1;
    second.kv_cache_offset_bytes                            = 128;
    second.kv_block_pool_size_bytes = second.total_size_bytes = 40;
    config->memory_layouts                                    = {first, second};
    config->total_size_bytes                                  = 168;
    auto pool                                                 = std::make_shared<DeviceBlockPool>(config);
    ASSERT_TRUE(pool->init());
    auto group = makeTestGroupSet(
        0,
        makeTestTopology({makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0, 1, 2}, 8)}),
        {"group0"},
        {pool});
    const auto& templates = group->copy3DTemplates();
    ASSERT_EQ(templates.size(), 2u);
    EXPECT_EQ(templates[0].layer_count, 2u);
    EXPECT_EQ(templates[1].layer_count, 1u);
    EXPECT_EQ(templates[0].device_base, static_cast<uint8_t*>(pool->getBaseAddress()) + 16);
    EXPECT_EQ(templates[1].device_base, static_cast<uint8_t*>(pool->getBaseAddress()) + 128);
    EXPECT_EQ(templates[1].host_offset, 16u);
    BlockTreeTaskPool          tasks(1, 8, "SplitLayoutTest");
    DeviceHostTransferExecutor executor(tasks, 8);
    std::vector<uint8_t>       host(24);
    auto [status, plans] =
        executor.generatePlan({{host.data(), 24, 24}}, {TransferDescriptor::hostToDevice(0, 1, {3})}, {group.get()});
    ASSERT_EQ(status, TransferStatus::OK);
    ASSERT_EQ(plans.size(), 1u);
    ASSERT_EQ(plans[0].copy_tiles.size(), 3u);
    size_t row = 0;
    for (const auto& t : templates) {
        for (size_t layer = 0; layer < t.layer_count; ++layer) {
            EXPECT_EQ(plans[0].copy_tiles[row].device_addr,
                      static_cast<uint8_t*>(t.device_base) + 3 * t.width_bytes + layer * t.device_pitch);
            EXPECT_EQ(plans[0].copy_tiles[row++].host_addr, host.data() + t.host_offset + layer * t.width_bytes);
        }
    }
}

TEST(DeviceHostCopyTemplateTest, MixedDeviceBatchFallsBackToPerDevicePlans) {
    int count = 0, original = 0;
    ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
    if (count < 2) {
        GTEST_SKIP() << "Requires two GPUs";
    }
    ASSERT_EQ(cudaGetDevice(&original), cudaSuccess);
    auto        first = templateGroup(CacheGroupType::FULL);
    GroupSetPtr second;
    {
        c10::cuda::CUDAGuard guard((original + 1) % count);
        second = templateGroup(CacheGroupType::FULL);
    }
    BlockTreeTaskPool          tasks(1, 8, "MixedDevicesTest");
    DeviceHostTransferExecutor executor(tasks, 8);
    executor.strategies_.clear();
    executor.strategies_.push_back(std::make_unique<Cuda3DBatchDeviceHostCopyStrategy>());
    std::vector<int> planned_devices;
    executor.strategies_.push_back(std::make_unique<TileProbe>([&](const DeviceHostCopyPlan& plan) {
        planned_devices.push_back(plan.copy_tiles.front().device_index);
        return StrategyResult::done();
    }));
    std::vector<uint8_t> host(96);
    HostBufferView       view{host.data(), 48, 48}, other{host.data() + 48, 48, 48};
    auto                 desc = TransferDescriptor::hostToDevice(0, 1, {1, 2, 3});
    EXPECT_TRUE(executor.executeBatch({view, other}, {desc, desc}, {first.get(), second.get()}).ok());
    ASSERT_EQ(planned_devices.size(), 2u);
    EXPECT_NE(planned_devices[0], planned_devices[1]);
}

}  // namespace
}  // namespace rtp_llm

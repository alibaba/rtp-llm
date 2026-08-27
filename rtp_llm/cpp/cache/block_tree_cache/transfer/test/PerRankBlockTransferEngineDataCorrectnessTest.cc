#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <torch/torch.h>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"

namespace rtp_llm {
namespace {

using block_transfer_engine_test::TempDirGuard;
using block_transfer_engine_test::makeDescriptor;
using block_transfer_engine_test::makeDiskPool;
using block_transfer_engine_test::makeHostPool;
using block_transfer_engine_test::makeTestDevicePool;
using block_transfer_engine_test::makeTestGroupBase;
using block_transfer_engine_test::makeTestGroupSet;
using block_transfer_engine_test::makeTestTopology;
using block_transfer_engine_test::poolMalloc;

constexpr size_t kLayerCount = 2;
constexpr size_t kKvBytes    = 256;
constexpr size_t kScaleBytes = 32;
constexpr size_t kPayloadBytes = kLayerCount * (kKvBytes + kScaleBytes);

struct AllocatedBlocks {
    BlockIdxType              before{NULL_BLOCK_IDX};
    std::vector<BlockIdxType> data;
    BlockIdxType              after{NULL_BLOCK_IDX};
};

std::string groupTypeName(CacheGroupType group_type) {
    return group_type == CacheGroupType::FULL ? "Full" : "Swa";
}

std::vector<uint8_t> makePattern(size_t pattern_id) {
    std::vector<uint8_t> bytes(kPayloadBytes);
    for (size_t offset = 0; offset < bytes.size(); ++offset) {
        bytes[offset] = static_cast<uint8_t>(((pattern_id + 1) * 53 + offset * 29 + (offset / 7) * 11) % 251 + 1);
    }
    return bytes;
}

AllocatedBlocks allocateBlocks(IBlockPool& pool, size_t count) {
    AllocatedBlocks blocks;
    const auto allocate = [&pool]() {
        const BlockIdxType block = poolMalloc(pool);
        if (block == NULL_BLOCK_IDX) {
            throw std::runtime_error("test block pool exhausted");
        }
        return block;
    };
    blocks.before = allocate();
    blocks.data.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        blocks.data.push_back(allocate());
    }
    blocks.after = allocate();
    return blocks;
}

std::vector<BlockInfo> deviceBuffers(const DeviceBlockPoolPtr& pool, BlockIdxType block) {
    std::vector<BlockInfo> buffers;
    for (int layer_id = 0; layer_id < static_cast<int>(kLayerCount); ++layer_id) {
        auto layer_buffers = pool->convertIndexToBuffer(layer_id, block);
        buffers.insert(buffers.end(), layer_buffers.begin(), layer_buffers.end());
    }
    return buffers;
}

void writeDevicePayload(const DeviceBlockPoolPtr& pool, BlockIdxType block, const std::vector<uint8_t>& bytes) {
    ASSERT_EQ(bytes.size(), kPayloadBytes);
    size_t offset = 0;
    for (const BlockInfo& buffer : deviceBuffers(pool, block)) {
        ASSERT_LE(offset + buffer.size_bytes, bytes.size());
        ASSERT_EQ(cudaMemcpy(buffer.addr, bytes.data() + offset, buffer.size_bytes, cudaMemcpyHostToDevice),
                  cudaSuccess);
        offset += buffer.size_bytes;
    }
    ASSERT_EQ(offset, bytes.size());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

std::vector<uint8_t> readDevicePayload(const DeviceBlockPoolPtr& pool, BlockIdxType block) {
    std::vector<uint8_t> bytes(kPayloadBytes);
    size_t               offset = 0;
    for (const BlockInfo& buffer : deviceBuffers(pool, block)) {
        EXPECT_LE(offset + buffer.size_bytes, bytes.size());
        if (offset + buffer.size_bytes > bytes.size()) {
            return {};
        }
        const cudaError_t error =
            cudaMemcpy(bytes.data() + offset, buffer.addr, buffer.size_bytes, cudaMemcpyDeviceToHost);
        EXPECT_EQ(error, cudaSuccess);
        if (error != cudaSuccess) {
            return {};
        }
        offset += buffer.size_bytes;
    }
    EXPECT_EQ(offset, bytes.size());
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    return bytes;
}

void writeHostPayload(const std::shared_ptr<HostBlockPool>& pool,
                      BlockIdxType                          block,
                      const std::vector<uint8_t>&          bytes) {
    ASSERT_EQ(bytes.size(), pool->payloadBytes());
    std::memcpy(pool->blockBuffer(block).addr, bytes.data(), bytes.size());
}

std::vector<uint8_t> readHostPayload(const std::shared_ptr<HostBlockPool>& pool, BlockIdxType block) {
    const HostBlockBuffer buffer = pool->blockBuffer(block);
    const auto*           begin  = static_cast<const uint8_t*>(buffer.addr);
    return std::vector<uint8_t>(begin, begin + buffer.payload_bytes);
}

void writeHostStride(const std::shared_ptr<HostBlockPool>& pool, BlockIdxType block, uint8_t value) {
    const HostBlockBuffer buffer = pool->blockBuffer(block);
    std::memset(buffer.addr, value, buffer.stride_bytes);
}

std::vector<uint8_t> readHostStride(const std::shared_ptr<HostBlockPool>& pool, BlockIdxType block) {
    const HostBlockBuffer buffer = pool->blockBuffer(block);
    const auto*           begin  = static_cast<const uint8_t*>(buffer.addr);
    return std::vector<uint8_t>(begin, begin + buffer.stride_bytes);
}

void writeDiskPayload(const BlockTreeDiskBlockPoolPtr& pool,
                      BlockIdxType                     block,
                      const std::vector<uint8_t>&     bytes,
                      uint8_t                          padding) {
    ASSERT_EQ(bytes.size(), pool->payloadBytes());
    std::vector<uint8_t> stride(pool->strideBytes(), padding);
    std::copy(bytes.begin(), bytes.end(), stride.begin());
    ASSERT_EQ(pool->write(block, stride.data(), stride.size()), BlockIOStatus::OK);
}

void writeDiskStride(const BlockTreeDiskBlockPoolPtr& pool, BlockIdxType block, uint8_t value) {
    std::vector<uint8_t> stride(pool->strideBytes(), value);
    ASSERT_EQ(pool->write(block, stride.data(), stride.size()), BlockIOStatus::OK);
}

std::vector<uint8_t> readDiskStride(const BlockTreeDiskBlockPoolPtr& pool, BlockIdxType block) {
    std::vector<uint8_t> stride(pool->strideBytes());
    EXPECT_EQ(pool->read(block, stride.data(), stride.size()), BlockIOStatus::OK);
    return stride;
}

std::vector<uint8_t> readDiskPayload(const BlockTreeDiskBlockPoolPtr& pool, BlockIdxType block) {
    auto bytes = readDiskStride(pool, block);
    bytes.resize(pool->payloadBytes());
    return bytes;
}

void expectTransferSuccess(const std::shared_ptr<PerRankBlockTransferEngine>& engine,
                           const std::vector<TransferDescriptor>&             descriptors) {
    auto context = engine->submit(descriptors);
    ASSERT_NE(context, nullptr);
    context->waitDone();
    ASSERT_TRUE(context->success()) << context->errorInfo().ToString();
}

struct TransferFixture {
    TransferFixture(CacheGroupType group_type, size_t data_block_count, const std::string& name):
        temp_dir(name.c_str()) {
        const size_t usable_count = data_block_count + 2;
        auto policy                = defaultCacheGroupPolicy(group_type);
        policy.enable_prefix_reuse = true;
        if (group_type == CacheGroupType::SWA) {
            policy.sliding_window_size = 2;
        }

        auto topology = makeTestTopology(
            {makeTestGroupBase(std::move(policy), {0, 1}, kKvBytes, kScaleBytes)});
        device_pool = makeTestDevicePool(
            {{kKvBytes, kScaleBytes}, {kKvBytes, kScaleBytes}}, usable_count, name + "_device");
        host_pool = makeHostPool(kPayloadBytes, usable_count, true);
        disk_pool = makeDiskPool(kPayloadBytes, usable_count, temp_dir.path, nullptr, name + "_disk");
        group_set = makeTestGroupSet(
            0, std::move(topology), {0}, {device_pool}, host_pool, disk_pool);
        engine = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group_set});
    }

    TempDirGuard                                temp_dir;
    DeviceBlockPoolPtr                          device_pool;
    std::shared_ptr<HostBlockPool>              host_pool;
    BlockTreeDiskBlockPoolPtr                   disk_pool;
    GroupSetPtr                                 group_set;
    std::shared_ptr<PerRankBlockTransferEngine> engine;
};

class PerRankBlockTransferEngineDataCorrectnessTest: public ::testing::TestWithParam<CacheGroupType> {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run transfer correctness tests";
    }
};

TEST_P(PerRankBlockTransferEngineDataCorrectnessTest, DeviceHostRoundTripsNineDistinctBlocksExactly) {
    constexpr size_t kDescriptorCount = 9;
    TransferFixture env(GetParam(), kDescriptorCount, "data_correctness_device_host_" + groupTypeName(GetParam()));
    const auto       device_blocks = allocateBlocks(*env.device_pool, kDescriptorCount);
    const auto       host_blocks   = allocateBlocks(*env.host_pool, kDescriptorCount);
    const auto       device_before = makePattern(500);
    const auto       device_after  = makePattern(501);
    const auto       host_before   = std::vector<uint8_t>(env.host_pool->strideBytes(), 0xD1);
    const auto       host_after    = std::vector<uint8_t>(env.host_pool->strideBytes(), 0xD2);
    writeDevicePayload(env.device_pool, device_blocks.before, device_before);
    writeDevicePayload(env.device_pool, device_blocks.after, device_after);
    writeHostStride(env.host_pool, host_blocks.before, 0xD1);
    writeHostStride(env.host_pool, host_blocks.after, 0xD2);

    std::vector<std::vector<uint8_t>> h2d_expected;
    std::vector<TransferDescriptor>   h2d;
    for (size_t index = 0; index < kDescriptorCount; ++index) {
        h2d_expected.push_back(makePattern(index));
        writeHostPayload(env.host_pool, host_blocks.data[index], h2d_expected.back());
        writeDevicePayload(env.device_pool, device_blocks.data[index], makePattern(100 + index));
        h2d.push_back(makeDescriptor(
            Tier::HOST, Tier::DEVICE, {device_blocks.data[index]}, host_blocks.data[index]));
    }
    expectTransferSuccess(env.engine, h2d);
    for (size_t index = 0; index < kDescriptorCount; ++index) {
        EXPECT_EQ(readDevicePayload(env.device_pool, device_blocks.data[index]), h2d_expected[index]);
        EXPECT_EQ(readHostPayload(env.host_pool, host_blocks.data[index]), h2d_expected[index]);
    }

    std::vector<std::vector<uint8_t>> d2h_expected;
    std::vector<TransferDescriptor>   d2h;
    for (size_t index = 0; index < kDescriptorCount; ++index) {
        d2h_expected.push_back(makePattern(200 + index));
        writeDevicePayload(env.device_pool, device_blocks.data[index], d2h_expected.back());
        writeHostPayload(env.host_pool, host_blocks.data[index], makePattern(300 + index));
        d2h.push_back(makeDescriptor(
            Tier::DEVICE, Tier::HOST, {device_blocks.data[index]}, host_blocks.data[index]));
    }
    expectTransferSuccess(env.engine, d2h);
    for (size_t index = 0; index < kDescriptorCount; ++index) {
        EXPECT_EQ(readHostPayload(env.host_pool, host_blocks.data[index]), d2h_expected[index]);
        EXPECT_EQ(readDevicePayload(env.device_pool, device_blocks.data[index]), d2h_expected[index]);
    }

    EXPECT_EQ(readDevicePayload(env.device_pool, device_blocks.before), device_before);
    EXPECT_EQ(readDevicePayload(env.device_pool, device_blocks.after), device_after);
    EXPECT_EQ(readHostStride(env.host_pool, host_blocks.before), host_before);
    EXPECT_EQ(readHostStride(env.host_pool, host_blocks.after), host_after);
}

TEST_P(PerRankBlockTransferEngineDataCorrectnessTest, HostDiskRoundTripsDistinctBlocksExactly) {
    constexpr size_t kDescriptorCount = 3;
    TransferFixture env(GetParam(), kDescriptorCount, "data_correctness_host_disk_" + groupTypeName(GetParam()));
    const auto       host_blocks = allocateBlocks(*env.host_pool, kDescriptorCount);
    const auto       disk_blocks = allocateBlocks(*env.disk_pool, kDescriptorCount);
    writeHostStride(env.host_pool, host_blocks.before, 0xC1);
    writeHostStride(env.host_pool, host_blocks.after, 0xC2);
    writeDiskStride(env.disk_pool, disk_blocks.before, 0xB1);
    writeDiskStride(env.disk_pool, disk_blocks.after, 0xB2);
    const auto host_before = readHostStride(env.host_pool, host_blocks.before);
    const auto host_after  = readHostStride(env.host_pool, host_blocks.after);
    const auto disk_before = readDiskStride(env.disk_pool, disk_blocks.before);
    const auto disk_after  = readDiskStride(env.disk_pool, disk_blocks.after);

    std::vector<std::vector<uint8_t>> h2disk_expected;
    std::vector<TransferDescriptor>   h2disk;
    for (size_t index = 0; index < kDescriptorCount; ++index) {
        h2disk_expected.push_back(makePattern(400 + index));
        writeHostPayload(env.host_pool, host_blocks.data[index], h2disk_expected.back());
        writeDiskPayload(env.disk_pool, disk_blocks.data[index], makePattern(450 + index), 0xEE);
        h2disk.push_back(makeDescriptor(
            Tier::HOST, Tier::DISK, {}, host_blocks.data[index], disk_blocks.data[index]));
    }
    expectTransferSuccess(env.engine, h2disk);
    for (size_t index = 0; index < kDescriptorCount; ++index) {
        EXPECT_EQ(readDiskPayload(env.disk_pool, disk_blocks.data[index]), h2disk_expected[index]);
        EXPECT_EQ(readHostPayload(env.host_pool, host_blocks.data[index]), h2disk_expected[index]);
    }

    std::vector<std::vector<uint8_t>> disk2h_expected;
    std::vector<TransferDescriptor>   disk2h;
    for (size_t index = 0; index < kDescriptorCount; ++index) {
        disk2h_expected.push_back(makePattern(500 + index));
        writeDiskPayload(env.disk_pool, disk_blocks.data[index], disk2h_expected.back(), 0xA5);
        writeHostPayload(env.host_pool, host_blocks.data[index], makePattern(550 + index));
        disk2h.push_back(makeDescriptor(
            Tier::DISK, Tier::HOST, {}, host_blocks.data[index], disk_blocks.data[index]));
    }
    expectTransferSuccess(env.engine, disk2h);
    for (size_t index = 0; index < kDescriptorCount; ++index) {
        EXPECT_EQ(readHostPayload(env.host_pool, host_blocks.data[index]), disk2h_expected[index]);
        EXPECT_EQ(readDiskPayload(env.disk_pool, disk_blocks.data[index]), disk2h_expected[index]);
    }

    EXPECT_EQ(readHostStride(env.host_pool, host_blocks.before), host_before);
    EXPECT_EQ(readHostStride(env.host_pool, host_blocks.after), host_after);
    EXPECT_EQ(readDiskStride(env.disk_pool, disk_blocks.before), disk_before);
    EXPECT_EQ(readDiskStride(env.disk_pool, disk_blocks.after), disk_after);
}

TEST_P(PerRankBlockTransferEngineDataCorrectnessTest, DeviceDiskRoundTripsDistinctBlocksWithoutStaleStagingData) {
    constexpr size_t kDescriptorCount = 3;
    TransferFixture env(GetParam(), kDescriptorCount, "data_correctness_device_disk_" + groupTypeName(GetParam()));
    const auto       device_blocks = allocateBlocks(*env.device_pool, kDescriptorCount);
    const auto       disk_blocks   = allocateBlocks(*env.disk_pool, kDescriptorCount);
    writeDevicePayload(env.device_pool, device_blocks.before, makePattern(600));
    writeDevicePayload(env.device_pool, device_blocks.after, makePattern(601));
    writeDiskStride(env.disk_pool, disk_blocks.before, 0x91);
    writeDiskStride(env.disk_pool, disk_blocks.after, 0x92);
    const auto device_before = readDevicePayload(env.device_pool, device_blocks.before);
    const auto device_after  = readDevicePayload(env.device_pool, device_blocks.after);
    const auto disk_before   = readDiskStride(env.disk_pool, disk_blocks.before);
    const auto disk_after    = readDiskStride(env.disk_pool, disk_blocks.after);

    std::vector<std::vector<uint8_t>> d2disk_expected;
    for (size_t index = 0; index < kDescriptorCount; ++index) {
        d2disk_expected.push_back(makePattern(700 + index));
        writeDevicePayload(env.device_pool, device_blocks.data[index], d2disk_expected.back());
        writeDiskPayload(env.disk_pool, disk_blocks.data[index], makePattern(750 + index), 0xF0);
        expectTransferSuccess(env.engine,
                              {makeDescriptor(Tier::DEVICE,
                                              Tier::DISK,
                                              {device_blocks.data[index]},
                                              NULL_BLOCK_IDX,
                                              disk_blocks.data[index])});
    }
    for (size_t index = 0; index < kDescriptorCount; ++index) {
        EXPECT_EQ(readDiskPayload(env.disk_pool, disk_blocks.data[index]), d2disk_expected[index]);
        EXPECT_EQ(readDevicePayload(env.device_pool, device_blocks.data[index]), d2disk_expected[index]);
    }

    std::vector<std::vector<uint8_t>> disk2d_expected;
    std::vector<TransferDescriptor>   disk2d;
    for (size_t index = 0; index < kDescriptorCount; ++index) {
        disk2d_expected.push_back(makePattern(800 + index));
        writeDiskPayload(env.disk_pool, disk_blocks.data[index], disk2d_expected.back(), 0xA0 + index);
        writeDevicePayload(env.device_pool, device_blocks.data[index], makePattern(850 + index));
        disk2d.push_back(makeDescriptor(Tier::DISK,
                                       Tier::DEVICE,
                                       {device_blocks.data[index]},
                                       NULL_BLOCK_IDX,
                                       disk_blocks.data[index]));
    }
    expectTransferSuccess(env.engine, disk2d);
    for (size_t index = 0; index < kDescriptorCount; ++index) {
        EXPECT_EQ(readDevicePayload(env.device_pool, device_blocks.data[index]), disk2d_expected[index]);
        EXPECT_EQ(readDiskPayload(env.disk_pool, disk_blocks.data[index]), disk2d_expected[index]);
    }

    EXPECT_EQ(readDevicePayload(env.device_pool, device_blocks.before), device_before);
    EXPECT_EQ(readDevicePayload(env.device_pool, device_blocks.after), device_after);
    EXPECT_EQ(readDiskStride(env.disk_pool, disk_blocks.before), disk_before);
    EXPECT_EQ(readDiskStride(env.disk_pool, disk_blocks.after), disk_after);
}

INSTANTIATE_TEST_SUITE_P(
    FullAndSwa,
    PerRankBlockTransferEngineDataCorrectnessTest,
    ::testing::Values(CacheGroupType::FULL, CacheGroupType::SWA),
    [](const ::testing::TestParamInfo<CacheGroupType>& info) { return groupTypeName(info.param); });

}  // namespace
}  // namespace rtp_llm

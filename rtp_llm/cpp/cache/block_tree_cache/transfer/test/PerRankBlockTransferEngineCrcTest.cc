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

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferRequestConverter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"

namespace rtp_llm {
namespace {

using namespace block_transfer_engine_test;

// Independent reflected Castagnoli oracle. No production CRC helper is used.
uint32_t softwareCrc32c(const uint8_t* data, size_t size) {
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ ((crc & 1u) ? 0x82f63b78u : 0u);
        }
    }
    return ~crc;
}

class PerRankBlockTransferEngineCrcTest: public ::testing::Test {
protected:
    void SetUp() override {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
            GTEST_SKIP() << "requires a CUDA13 GPU worker";
        }
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    }

    void TearDown() override {
        engine_.reset();
        for (auto& [pool, block] : owned_) {
            releasePoolBlock(*pool, block);
        }
        owned_.clear();
        group_.reset();
        devices_.clear();
        host_.reset();
        disk_.reset();
        temp_dir_.reset();
    }

    BlockIdxType allocate(const std::shared_ptr<IBlockPool>& pool) {
        const auto block = poolMalloc(*pool);
        owned_.emplace_back(pool, block);
        return block;
    }

    void initialize(bool   page_boundary   = false,
                    bool   undersized_host = false,
                    size_t item_count      = 3,
                    size_t kv_multiplier   = 1) {
        // KV strides remain unaligned to 16 bytes, exercising ragged tile copies.
        // Both scale strides and their bases must be float-aligned: TestUtils
        // places scale storage immediately after physical_block_count * KV bytes.
        specs_  = page_boundary ?
                      std::vector<std::pair<size_t, size_t>>{{4096, 0}} :
                      std::vector<std::pair<size_t, size_t>>{{20 * kv_multiplier, 4}, {28 * kv_multiplier, 8}};
        layers_ = page_boundary ? 1 : 2;
        std::vector<GroupBase> groups;
        std::vector<size_t>    ids;
        payload_ = 0;
        for (size_t member = 0; member < specs_.size(); ++member) {
            const auto [kv, scale] = specs_[member];
            std::vector<int> layer_ids;
            for (size_t layer = 0; layer < layers_; ++layer) {
                layer_ids.push_back(static_cast<int>(layer));
            }
            groups.push_back(makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), layer_ids, kv, scale));
            devices_.push_back(makeTestDevicePool(std::vector<std::pair<size_t, size_t>>(layers_, specs_[member]),
                                                  std::max<size_t>(16, 2 * item_count),
                                                  "crc_member_" + std::to_string(member)));
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "device pool initialization, member " << member;
            ids.push_back(member);
            payload_ += layers_ * (kv + scale);
        }
        encoded_  = ((payload_ + 4 + 15) / 16) * 16;
        host_     = makeHostPool(payload_, std::max<size_t>(8, item_count), !undersized_host);
        temp_dir_ = std::make_unique<TempDirGuard>("block_transfer_crc");
        disk_ =
            makeDiskPool(payload_, std::max<size_t>(8, item_count), temp_dir_->path, nullptr, "crc_disk", true, true);
        group_ = makeTestGroupSet(0, makeTestTopology(std::move(groups)), ids, devices_, host_, disk_, true);
        // Keep every corruption case in one executor batch, including the
        // large-backing fixture that exercises segmented CRC through the engine.
        engine_ = std::make_shared<PerRankBlockTransferEngine>(
            std::vector<GroupSetPtr>{group_}, true, DeviceHostCopyOptions{}, 8, std::max<size_t>(8, item_count), 1);
        for (size_t item = 0; item < item_count; ++item) {
            std::vector<BlockIdxType> source, target;
            for (const auto& device : devices_) {
                source.push_back(allocate(device));
                target.push_back(allocate(device));
            }
            sources_.push_back(source);
            targets_.push_back(target);
            hosts_.push_back(allocate(host_));
            disks_.push_back(allocate(disk_));
            std::vector<uint8_t> expected;
            for (size_t member = 0; member < devices_.size(); ++member) {
                for (size_t layer = 0; layer < layers_; ++layer) {
                    const auto buffers = devices_[member]->convertIndexToBuffer(layer, source[member]);
                    for (size_t kind = 0; kind < buffers.size(); ++kind) {
                        std::vector<uint8_t> bytes(buffers[kind].size_bytes);
                        for (size_t i = 0; i < bytes.size(); ++i) {
                            bytes[i] =
                                static_cast<uint8_t>((i * 29 + item * 71 + member * 43 + layer * 17 + kind * 13) & 255);
                        }
                        ASSERT_EQ(cudaMemcpy(buffers[kind].addr, bytes.data(), bytes.size(), cudaMemcpyHostToDevice),
                                  cudaSuccess);
                        expected.insert(expected.end(), bytes.begin(), bytes.end());
                    }
                }
            }
            expected_.push_back(std::move(expected));
        }
        fillTargets(0xa5);
    }

    void fillTargets(uint8_t value) {
        for (const auto& target : targets_) {
            for (size_t member = 0; member < devices_.size(); ++member) {
                for (size_t layer = 0; layer < layers_; ++layer) {
                    for (const auto& buffer : devices_[member]->convertIndexToBuffer(layer, target[member])) {
                        ASSERT_EQ(cudaMemset(buffer.addr, value, buffer.size_bytes), cudaSuccess);
                    }
                }
            }
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }

    std::vector<uint8_t> readTarget(size_t item) {
        std::vector<uint8_t> result;
        for (size_t member = 0; member < devices_.size(); ++member) {
            for (size_t layer = 0; layer < layers_; ++layer) {
                for (const auto& buffer : devices_[member]->convertIndexToBuffer(layer, targets_[item][member])) {
                    std::vector<uint8_t> bytes(buffer.size_bytes);
                    EXPECT_EQ(cudaMemcpy(bytes.data(), buffer.addr, bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
                    result.insert(result.end(), bytes.begin(), bytes.end());
                }
            }
        }
        return result;
    }

    uint8_t* hostData(size_t item) {
        return static_cast<uint8_t*>(host_->blockBuffer(hosts_[item]).addr);
    }

    std::shared_ptr<AsyncContext> execute(Tier from, Tier to) {
        std::vector<TransferDescriptor> descriptors;
        for (size_t item = 0; item < sources_.size(); ++item) {
            descriptors.push_back(makeDescriptor(
                from, to, from == Tier::DEVICE ? sources_[item] : targets_[item], hosts_[item], disks_[item]));
        }
        // DEVICE->DISK stages one backing per task; keep load directions batched
        // so corruption in the final backing still exercises all-item validation.
        if (from == Tier::DEVICE && to == Tier::DISK) {
            std::shared_ptr<AsyncContext> context;
            for (auto& descriptor : descriptors) {
                context = engine_->execute(makeTransferTask({std::move(descriptor)}));
                context->waitDone();
                EXPECT_TRUE(context->done());
                if (!context->success()) {
                    ADD_FAILURE() << "DEVICE->DISK fixture setup failed: " << context->errorInfo().ToString();
                    return context;
                }
            }
            return context;
        }
        auto context = engine_->execute(makeTransferTask(std::move(descriptors)));
        context->waitDone();
        EXPECT_TRUE(context->done());
        return context;
    }

    void expectIntegrityFailure(const std::shared_ptr<AsyncContext>& context) {
        ASSERT_FALSE(context->success());
        EXPECT_EQ(context->errorInfo().code(), ErrorCode::CACHE_INTEGRITY_ERROR);
    }

    void expectUntouchedTargets() {
        for (size_t item = 0; item < targets_.size(); ++item) {
            EXPECT_EQ(readTarget(item), std::vector<uint8_t>(payload_, 0xa5)) << "item " << item;
        }
    }

    std::unique_ptr<TempDirGuard>                                     temp_dir_;
    std::shared_ptr<HostBlockPool>                                    host_;
    std::shared_ptr<BlockTreeDiskBlockPool>                           disk_;
    std::vector<DeviceBlockPoolPtr>                                   devices_;
    GroupSetPtr                                                       group_;
    std::shared_ptr<PerRankBlockTransferEngine>                       engine_;
    std::vector<std::pair<std::shared_ptr<IBlockPool>, BlockIdxType>> owned_;
    std::vector<std::pair<size_t, size_t>>                            specs_;
    std::vector<std::vector<BlockIdxType>>                            sources_, targets_;
    std::vector<BlockIdxType>                                         hosts_, disks_;
    std::vector<std::vector<uint8_t>>                                 expected_;
    size_t                                                            payload_{0}, encoded_{0}, layers_{0};
};

TEST_F(PerRankBlockTransferEngineCrcTest, MultiBackingMemberLayerScaleRoundTripMatchesIndependentCrc) {
    ASSERT_NO_FATAL_FAILURE(initialize());
    ASSERT_TRUE(group_->crcEnabled());
    ASSERT_EQ(group_->storageBytes(), encoded_);
    ASSERT_TRUE(execute(Tier::DEVICE, Tier::HOST)->success());
    for (size_t item = 0; item < hosts_.size(); ++item) {
        ASSERT_EQ(std::vector<uint8_t>(hostData(item), hostData(item) + payload_), expected_[item]);
        uint32_t footer = 0;
        std::memcpy(&footer, hostData(item) + encoded_ - sizeof(footer), sizeof(footer));
        EXPECT_EQ(footer, softwareCrc32c(expected_[item].data(), expected_[item].size()));
    }
    ASSERT_TRUE(execute(Tier::HOST, Tier::DEVICE)->success());
    for (size_t item = 0; item < targets_.size(); ++item) {
        EXPECT_EQ(readTarget(item), expected_[item]);
    }
}

TEST_F(PerRankBlockTransferEngineCrcTest, RpcWorkerUsesValidDeviceIndicesWithoutLocalAllocation) {
    ASSERT_NO_FATAL_FAILURE(initialize());
    // RPC workers use indices owned by the coordinator; their local allocator
    // bitmap need not contain those allocations. Pool backing remains resident.
    for (auto it = owned_.begin(); it != owned_.end();) {
        const bool device_owned = std::any_of(
            devices_.begin(), devices_.end(), [&](const auto& pool) { return pool.get() == it->first.get(); });
        if (device_owned) {
            releasePoolBlock(*it->first, it->second);
            EXPECT_TRUE(it->first->validBlock(it->second));
            EXPECT_FALSE(it->first->isAllocated(it->second));
            it = owned_.erase(it);
        } else {
            ++it;
        }
    }
    auto rpcExecute = [&](Tier from, Tier to) {
        std::vector<TransferDescriptor> descriptors;
        for (size_t item = 0; item < sources_.size(); ++item) {
            descriptors.push_back(makeDescriptor(
                from, to, from == Tier::DEVICE ? sources_[item] : targets_[item], hosts_[item], disks_[item]));
        }
        MemoryOperationRequestPB request;
        EXPECT_TRUE(
            BlockTransferRequestConverter::encodeTransfer(request, makeTransferTask(std::move(descriptors)), {group_}));
        std::vector<TransferDescriptor> decoded;
        EXPECT_TRUE(BlockTransferRequestConverter::decodeTransfer(request, decoded, {group_}));
        auto context = engine_->execute(makeTransferTask(std::move(decoded)));
        context->waitDone();
        return context;
    };
    auto stored = rpcExecute(Tier::DEVICE, Tier::HOST);
    ASSERT_TRUE(stored->success()) << stored->errorInfo().ToString();
    for (size_t item = 0; item < hosts_.size(); ++item) {
        EXPECT_EQ(std::vector<uint8_t>(hostData(item), hostData(item) + payload_), expected_[item]);
        uint32_t footer = 0;
        std::memcpy(&footer, hostData(item) + encoded_ - sizeof(footer), sizeof(footer));
        EXPECT_EQ(footer, softwareCrc32c(expected_[item].data(), expected_[item].size()));
    }
    auto loaded = rpcExecute(Tier::HOST, Tier::DEVICE);
    ASSERT_TRUE(loaded->success()) << loaded->errorInfo().ToString();
    for (size_t item = 0; item < targets_.size(); ++item) {
        EXPECT_EQ(readTarget(item), expected_[item]);
    }
    fillTargets(0xa5);
    hostData(2)[payload_ - 1] ^= 1;
    expectIntegrityFailure(rpcExecute(Tier::HOST, Tier::DEVICE));
    expectUntouchedTargets();
}

TEST_F(PerRankBlockTransferEngineCrcTest, InvalidDeviceIndicesAndMemberCountsNeverWriteTargets) {
    ASSERT_NO_FATAL_FAILURE(initialize());
    ASSERT_TRUE(execute(Tier::DEVICE, Tier::HOST)->success());
    const std::vector<uint8_t>                   sealed(hostData(0), hostData(0) + host_->strideBytes());
    const std::vector<std::vector<BlockIdxType>> invalid{
        {0, targets_[0][1]},
        {NULL_BLOCK_IDX, targets_[0][1]},
        {17, targets_[0][1]},  // 16 usable blocks plus reserved block zero
        {targets_[0][0]},
        {targets_[0][0], targets_[0][1], targets_[0][0]},
    };
    for (const auto& blocks : invalid) {
        for (const auto direction :
             {std::make_pair(Tier::DEVICE, Tier::HOST), std::make_pair(Tier::HOST, Tier::DEVICE)}) {
            std::memset(hostData(0), 0x6b, host_->strideBytes());
            if (direction.first == Tier::HOST) {
                std::memcpy(hostData(0), sealed.data(), sealed.size());
            }
            const std::vector<uint8_t> before(hostData(0), hostData(0) + host_->strideBytes());
            fillTargets(0xa5);
            auto descriptor = makeDescriptor(direction.first, direction.second, blocks, hosts_[0], disks_[0]);
            auto context    = engine_->execute(makeTransferTask({descriptor}));
            context->waitDone();
            ASSERT_TRUE(context->done());
            EXPECT_FALSE(context->success());
            EXPECT_EQ(std::vector<uint8_t>(hostData(0), hostData(0) + host_->strideBytes()), before);
            expectUntouchedTargets();
        }
    }
}

TEST_F(PerRankBlockTransferEngineCrcTest, LastBackingPayloadCorruptionRejectsWholeBatchBeforeScatter) {
    ASSERT_NO_FATAL_FAILURE(initialize());
    ASSERT_TRUE(execute(Tier::DEVICE, Tier::HOST)->success());
    hostData(2)[payload_ - 1] ^= 0x80;
    expectIntegrityFailure(execute(Tier::HOST, Tier::DEVICE));
    expectUntouchedTargets();
}

TEST_F(PerRankBlockTransferEngineCrcTest, LastBackingFooterCorruptionRejectsWholeBatchBeforeScatter) {
    ASSERT_NO_FATAL_FAILURE(initialize());
    ASSERT_TRUE(execute(Tier::DEVICE, Tier::HOST)->success());
    hostData(2)[encoded_ - 1] ^= 1;
    expectIntegrityFailure(execute(Tier::HOST, Tier::DEVICE));
    expectUntouchedTargets();
}

TEST_F(PerRankBlockTransferEngineCrcTest, EncodedAndDiskPaddingAreOutsideChecksum) {
    ASSERT_NO_FATAL_FAILURE(initialize());
    ASSERT_TRUE(execute(Tier::DEVICE, Tier::HOST)->success());
    ASSERT_LT(payload_, encoded_ - 4);
    for (size_t item = 0; item < hosts_.size(); ++item) {
        std::memset(hostData(item) + payload_, 0x71, encoded_ - 4 - payload_);
        std::memset(hostData(item) + encoded_, 0x92, host_->strideBytes() - encoded_);
    }
    ASSERT_TRUE(execute(Tier::HOST, Tier::DEVICE)->success());
    for (size_t item = 0; item < targets_.size(); ++item) {
        EXPECT_EQ(readTarget(item), expected_[item]);
    }
}

TEST_F(PerRankBlockTransferEngineCrcTest, HostDiskPreservesFooterAndRefusesToResealCorruptSource) {
    ASSERT_NO_FATAL_FAILURE(initialize());
    ASSERT_TRUE(execute(Tier::DEVICE, Tier::HOST)->success());
    std::vector<std::vector<uint8_t>> sealed;
    for (size_t item = 0; item < hosts_.size(); ++item) {
        sealed.emplace_back(hostData(item), hostData(item) + encoded_);
    }
    ASSERT_TRUE(execute(Tier::HOST, Tier::DISK)->success());
    for (size_t item = 0; item < hosts_.size(); ++item) {
        std::memset(hostData(item), 0xcc, host_->strideBytes());
    }
    ASSERT_TRUE(execute(Tier::DISK, Tier::HOST)->success());
    for (size_t item = 0; item < hosts_.size(); ++item) {
        EXPECT_EQ(std::vector<uint8_t>(hostData(item), hostData(item) + encoded_), sealed[item]);
    }
    hostData(2)[0] ^= 1;
    expectIntegrityFailure(execute(Tier::HOST, Tier::DISK));
    // A failed H2DISK must not turn the corrupt payload into a valid new block.
    ASSERT_TRUE(execute(Tier::DISK, Tier::HOST)->success());
    EXPECT_EQ(std::vector<uint8_t>(hostData(2), hostData(2) + encoded_), sealed[2]);
}

TEST_F(PerRankBlockTransferEngineCrcTest, DiskCorruptionIsRejectedWithoutDeviceScatter) {
    ASSERT_NO_FATAL_FAILURE(initialize());
    ASSERT_TRUE(execute(Tier::DEVICE, Tier::DISK)->success());
    std::vector<uint8_t> raw(disk_->strideBytes());
    ASSERT_EQ(disk_->read(disks_[2], raw.data(), raw.size()), BlockIOStatus::OK);
    raw[payload_ / 2] ^= 0x40;
    ASSERT_EQ(disk_->write(disks_[2], raw.data(), raw.size()), BlockIOStatus::OK);
    expectIntegrityFailure(execute(Tier::DISK, Tier::HOST));
    expectIntegrityFailure(execute(Tier::DISK, Tier::DEVICE));
    expectUntouchedTargets();
}

TEST_F(PerRankBlockTransferEngineCrcTest, PageBoundaryPayloadUsesExtraPageAndDeviceDiskRoundTrips) {
    ASSERT_NO_FATAL_FAILURE(initialize(true));
    ASSERT_EQ(payload_, 4096u);
    ASSERT_EQ(encoded_, 4112u);
    EXPECT_EQ(host_->strideBytes(), 8192u);
    EXPECT_EQ(disk_->strideBytes(), 8192u);
    ASSERT_TRUE(execute(Tier::DEVICE, Tier::DISK)->success());
    ASSERT_TRUE(execute(Tier::DISK, Tier::DEVICE)->success());
    for (size_t item = 0; item < targets_.size(); ++item) {
        EXPECT_EQ(readTarget(item), expected_[item]);
    }
}

TEST_F(PerRankBlockTransferEngineCrcTest, PayloadSizedHostCapacityCannotHoldFooter) {
    EXPECT_THROW(initialize(true, true), std::invalid_argument);
    ASSERT_NE(host_, nullptr);
    EXPECT_EQ(host_->strideBytes(), 4096u);
    EXPECT_EQ(engine_, nullptr);
}

TEST_F(PerRankBlockTransferEngineCrcTest, SegmentedBatchRoundTripRejectsCorruptionBeforeAnyScatter) {
    ASSERT_NO_FATAL_FAILURE(initialize(false, false, 32, 256));
    ASSERT_EQ(hosts_.size(), 32u);
    ASSERT_GT(payload_, 16384u);
    ASSERT_NE(payload_ % 16384, 0u);
    ASSERT_TRUE(execute(Tier::DEVICE, Tier::HOST)->success());
    std::vector<std::vector<uint8_t>> records;
    for (size_t item = 0; item < hosts_.size(); ++item) {
        records.emplace_back(hostData(item), hostData(item) + encoded_);
        EXPECT_EQ(std::vector<uint8_t>(hostData(item), hostData(item) + payload_), expected_[item]);
        uint32_t footer = 0;
        std::memcpy(&footer, hostData(item) + encoded_ - 4, sizeof(footer));
        EXPECT_EQ(footer, softwareCrc32c(expected_[item].data(), payload_));
    }
    ASSERT_TRUE(execute(Tier::HOST, Tier::DEVICE)->success());
    for (size_t item = 0; item < targets_.size(); ++item) {
        EXPECT_EQ(readTarget(item), expected_[item]);
    }
    // Corrupt both a segment boundary and the stored footer. Even the last
    // backing's failure must protect the first backing's destination.
    for (size_t offset : {size_t(0), size_t(16384), payload_ - 1, encoded_ - 1}) {
        SCOPED_TRACE(offset);
        hostData(31)[offset] ^= 0x40;
        fillTargets(0xa5);
        expectIntegrityFailure(execute(Tier::HOST, Tier::DEVICE));
        expectUntouchedTargets();
        hostData(31)[offset] ^= 0x40;
        for (size_t item = 0; item < hosts_.size(); ++item) {
            EXPECT_EQ(std::vector<uint8_t>(hostData(item), hostData(item) + encoded_), records[item]);
        }
        ASSERT_TRUE(execute(Tier::HOST, Tier::DEVICE)->success());
        for (size_t item = 0; item < targets_.size(); ++item) {
            EXPECT_EQ(readTarget(item), expected_[item]);
        }
    }
}

TEST_F(PerRankBlockTransferEngineCrcTest, SegmentedHostDiskValidationDoesNotResealCorruptSource) {
    ASSERT_NO_FATAL_FAILURE(initialize(false, false, 32, 256));
    ASSERT_TRUE(execute(Tier::DEVICE, Tier::HOST)->success());
    std::vector<std::vector<uint8_t>> records;
    for (size_t item = 0; item < hosts_.size(); ++item) {
        records.emplace_back(hostData(item), hostData(item) + encoded_);
    }
    ASSERT_TRUE(execute(Tier::HOST, Tier::DISK)->success());
    hostData(31)[16384] ^= 0x80;
    const std::vector<uint8_t> corrupt(hostData(31), hostData(31) + encoded_);
    expectIntegrityFailure(execute(Tier::HOST, Tier::DISK));
    EXPECT_EQ(std::vector<uint8_t>(hostData(31), hostData(31) + encoded_), corrupt);
    for (size_t item = 0; item < hosts_.size(); ++item) {
        std::memset(hostData(item), 0xcc, host_->strideBytes());
    }
    ASSERT_TRUE(execute(Tier::DISK, Tier::HOST)->success());
    for (size_t item = 0; item < hosts_.size(); ++item) {
        EXPECT_EQ(std::vector<uint8_t>(hostData(item), hostData(item) + encoded_), records[item]);
    }
    fillTargets(0xa5);
    ASSERT_TRUE(execute(Tier::HOST, Tier::DEVICE)->success());
    for (size_t item = 0; item < targets_.size(); ++item) {
        EXPECT_EQ(readTarget(item), expected_[item]);
    }
}

}  // namespace
}  // namespace rtp_llm

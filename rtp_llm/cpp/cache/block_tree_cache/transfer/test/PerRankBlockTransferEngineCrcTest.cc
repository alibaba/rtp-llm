#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferRequestConverter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"
#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcTransferService.h"

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
        if (!CrcBlockCopyBatch::available()) {
            GTEST_SKIP() << "CUDA13 CRC backend unavailable";
        }
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

    void initialize(bool   page_boundary     = false,
                    bool   undersized_host   = false,
                    size_t item_count        = 3,
                    size_t kv_multiplier     = 1,
                    bool   physical_geometry = false) {
        // KV strides remain unaligned to 16 bytes, exercising ragged tile copies.
        // Both scale strides and their bases must be float-aligned: TestUtils
        // places scale storage immediately after physical_block_count * KV bytes.
        specs_  = page_boundary ?
                      std::vector<std::pair<size_t, size_t>>{{4096, 0}} :
                      std::vector<std::pair<size_t, size_t>>{{20 * kv_multiplier, 4}, {28 * kv_multiplier, 8}};
        layers_ = page_boundary ? 1 : 2;
        if (physical_geometry) {
            // Logical specs describe only the first layer. The actual MTP-like
            // pools vary both KV strides and scale presence by local layer.
            specs_  = {{20, 0}, {28, 8}};
            layers_ = 3;
        }
        std::vector<TestGroupConfig> groups;
        std::vector<std::string>     tags;
        payload_ = 0;
        for (size_t member = 0; member < specs_.size(); ++member) {
            const auto [kv, scale] = specs_[member];
            std::vector<int> layer_ids;
            for (size_t layer = 0; layer < layers_; ++layer) {
                layer_ids.push_back(static_cast<int>(physical_geometry ? 2 * layer + member : layer));
            }
            groups.push_back(makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), layer_ids, kv, scale));
            auto layer_bytes = std::vector<std::pair<size_t, size_t>>(layers_, specs_[member]);
            if (physical_geometry) {
                layer_bytes = member == 0 ? std::vector<std::pair<size_t, size_t>>{{20, 0}, {44, 4}, {68, 8}} :
                                            std::vector<std::pair<size_t, size_t>>{{28, 8}, {52, 0}, {76, 4}};
            }
            devices_.push_back(makeTestDevicePool(
                layer_bytes, std::max<size_t>(16, 2 * item_count), "crc_member_" + std::to_string(member)));
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "device pool initialization, member " << member;
            tags.push_back("group" + std::to_string(member));
            for (const auto& [physical_kv, physical_scale] : layer_bytes) {
                payload_ += physical_kv + physical_scale;
            }
        }
        if (physical_geometry) {
            // Membership order is independent from the topology's tag order.
            std::reverse(tags.begin(), tags.end());
            std::reverse(devices_.begin(), devices_.end());
        }
        encoded_  = ((payload_ + 4 + 15) / 16) * 16;
        host_     = makeHostPool(payload_, std::max<size_t>(8, item_count), !undersized_host);
        temp_dir_ = std::make_unique<TempDirGuard>("block_transfer_crc");
        disk_ =
            makeDiskPool(payload_, std::max<size_t>(8, item_count), temp_dir_->path, nullptr, "crc_disk", true, true);
        group_ = makeTestGroupSet(0,
                                  makeTestTopology(std::move(groups)),
                                  tags,
                                  devices_,
                                  host_,
                                  disk_,
                                  true,
                                  physical_geometry ? payload_ : 0);
        // Keep every corruption case in one executor batch, including the
        // segmented CRC fixture. Disk staging reserves half its slots for FULL.
        engine_ = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group_},
                                                               true,
                                                               DeviceHostCopyOptions{},
                                                               std::max<size_t>(8, 2 * item_count),
                                                               std::max<size_t>(8, item_count),
                                                               1);
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
        last_descriptors_ = descriptors;
        auto context      = engine_->execute(makeTransferTask(std::move(descriptors)));
        context->waitDone();
        EXPECT_TRUE(context->done());
        return context;
    }

    void expectIntegrityFailure(const std::shared_ptr<AsyncContext>& context) {
        ASSERT_FALSE(context->success());
        EXPECT_EQ(context->errorInfo().code(), ErrorCode::EXECUTION_EXCEPTION);
    }

    void expectUntouchedTargets() {
        for (size_t item = 0; item < targets_.size(); ++item) {
            EXPECT_EQ(readTarget(item), std::vector<uint8_t>(payload_, 0xa5)) << "item " << item;
        }
    }

    std::filesystem::path captureDumps(const char* name) {
        const auto root                                            = std::filesystem::path(temp_dir_->path) / name;
        engine_->device_host_executor_->crc_service_->dump_writer_ = std::make_shared<CrcDumpWriter>(0, root.string());
        return root / "rank_0";
    }

    void expectDump(const std::filesystem::path&    root,
                    const std::vector<uint8_t>&     record,
                    const std::vector<std::string>& fields) {
        ASSERT_TRUE(std::filesystem::is_directory(root));
        std::vector<std::filesystem::path> dumps;
        for (const auto& entry : std::filesystem::directory_iterator(root)) {
            if (entry.is_directory()) {
                dumps.push_back(entry.path());
            }
        }
        ASSERT_EQ(dumps.size(), 1u);
        std::ifstream     file(dumps[0] / "manifest.json");
        const std::string manifest((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        for (const auto& field : fields) {
            EXPECT_NE(manifest.find(field), std::string::npos) << field << "\n" << manifest;
        }
        const auto read_bytes = [](const std::filesystem::path& path) {
            std::ifstream input(path, std::ios::binary);
            return std::vector<uint8_t>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
        };
        ASSERT_GE(record.size(), encoded_);
        EXPECT_EQ(read_bytes(dumps[0] / "cpu.bin"), std::vector<uint8_t>(record.begin(), record.begin() + encoded_));
        EXPECT_EQ(read_bytes(dumps[0] / "gpu_staging.bin"),
                  std::vector<uint8_t>(record.begin(), record.begin() + payload_));
        const std::vector<uint8_t> footer(record.begin() + encoded_ - 4, record.begin() + encoded_);
        EXPECT_EQ(read_bytes(dumps[0] / "cpu_footer.bin"), footer);
        EXPECT_EQ(read_bytes(dumps[0] / "gpu_staging_footer.bin"), footer);
        EXPECT_EQ(read_bytes(dumps[0] / "checked_footer.bin"), footer);
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
    std::vector<TransferDescriptor>                                   last_descriptors_;
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

TEST_F(PerRankBlockTransferEngineCrcTest, MixedRpcBatchUsesLocalCrcAndDoesNotScatterOnMismatch) {
    ASSERT_NO_FATAL_FAILURE(initialize(true, false, 2));
    ASSERT_TRUE(execute(Tier::DEVICE, Tier::HOST)->success());
    engine_.reset();
    auto topology = makeTestTopology({makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, payload_),
                                     makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {1}, payload_)});
    group_->initialize(0, topology, {"group0"}, makeTestBackingLayout(payload_, true));
    auto ordinary = makeTestGroupSet(1, topology, {"group1"}, devices_, host_);
    const std::vector<GroupSetPtr> groups{group_, ordinary};
    engine_ = std::make_shared<PerRankBlockTransferEngine>(groups, false, DeviceHostCopyOptions{}, 8, 8, 1);

    MemoryOperationRequestPB request;
    ASSERT_TRUE(BlockTransferRequestConverter::encodeTransfer(
        request,
        makeTransferTask({TransferDescriptor::hostToDevice(0, hosts_[0], targets_[0]),
                          TransferDescriptor::hostToDevice(1, hosts_[1], targets_[1])}),
        groups));
    EXPECT_EQ(request.copy_direction(), MemoryOperationRequestPB::H2D);
    // The ordinary GroupSet uses only its payload, even when the pool has room for a footer.
    hostData(1)[encoded_ - 1] ^= 1;
    hostData(0)[1] ^= 1;
    std::vector<TransferDescriptor> decoded;
    ASSERT_TRUE(BlockTransferRequestConverter::decodeTransfer(request, decoded, groups));
    auto failed = engine_->execute(makeTransferTask(decoded));
    failed->waitDone();
    expectIntegrityFailure(failed);
    EXPECT_TRUE(decoded[0].corrupted());
    EXPECT_FALSE(decoded[1].corrupted());
    expectUntouchedTargets();

    hostData(0)[1] ^= 1;
    decoded.clear();
    ASSERT_TRUE(BlockTransferRequestConverter::decodeTransfer(request, decoded, groups));
    auto recovered = engine_->execute(makeTransferTask(decoded));
    recovered->waitDone();
    ASSERT_TRUE(recovered->success()) << recovered->errorInfo().ToString();
    EXPECT_EQ(readTarget(0), expected_[0]);
    EXPECT_EQ(readTarget(1), expected_[1]);
}

TEST_F(PerRankBlockTransferEngineCrcTest, CorruptionDumpIncludesPhysicalLayoutAndKeepsFailureWhenIoIsUnavailable) {
    ASSERT_NO_FATAL_FAILURE(initialize(false, false, 3, 1, true));
    ASSERT_TRUE(execute(Tier::DEVICE, Tier::HOST)->success());
    hostData(0)[1] ^= 1;
    const auto         root = std::filesystem::path(temp_dir_->path) / "diagnostics";
    CrcTransferService service({group_}, 3, 1, 17);
    service.dump_writer_ = std::make_shared<CrcDumpWriter>(17, root.string());
    const HostBufferView host{hostData(0), payload_, host_->strideBytes()};
    auto                 descriptor = TransferDescriptor::hostToDevice(0, hosts_[0], targets_[0]);
    EXPECT_EQ(service.copy({host}, {descriptor}, {group_.get()}), TransferStatus::CRC_MISMATCH);
    EXPECT_TRUE(descriptor.corrupted());
    EXPECT_FALSE(descriptor.crcComputeFailed());
    expectUntouchedTargets();
    std::vector<std::filesystem::path> dumps;
    for (const auto& entry : std::filesystem::directory_iterator(root / "rank_17"))
        if (entry.is_directory())
            dumps.push_back(entry.path());
    ASSERT_EQ(dumps.size(), 1u);
    std::ifstream     file(dumps[0] / "manifest.json");
    const std::string manifest((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    for (const auto* field : {"\"rank\": \"17\"",
                              "\"device\": \"0\"",
                              "\"kind\": \"FULL\"",
                              "\"layout\": \"physical_pool_geometry\"",
                              "\"layout_tile_count\": \"10\"",
                              "\"member_0_tag\": \"group1\"",
                              "\"layout_tile_0_layer_id\": \"1\"",
                              "\"staging_captured\": \"true\"",
                              "\"output_written\": \"false\""})
        EXPECT_NE(manifest.find(field), std::string::npos) << field << "\n" << manifest;
    EXPECT_EQ(std::filesystem::file_size(dumps[0] / "cpu.bin"), encoded_);
    EXPECT_EQ(std::filesystem::file_size(dumps[0] / "gpu_staging.bin"), payload_);
    EXPECT_EQ(std::filesystem::file_size(dumps[0] / "checked_footer.bin"), 4u);

    // Exercise rotation in the service binary, which links libtorch and its
    // filesystem symbols. The standalone writer test has no Torch dependency.
    const auto outside = std::filesystem::path(temp_dir_->path) / "unrelated";
    std::filesystem::create_directory(outside);
    std::ofstream(outside / "keep.bin") << "keep";
    const auto linked = root / "rank_17" / "crc_symlink";
    std::filesystem::create_directory_symlink(outside, linked);
    const auto old_dump       = dumps.front();
    const auto one_dump_quota = 2 * encoded_ + CrcDumpWriter::kMetadataBytes + 12;
    service.dump_writer_      = std::make_shared<CrcDumpWriter>(17, root.string(), one_dump_quota);
    auto rotated              = TransferDescriptor::hostToDevice(0, hosts_[0], targets_[0]);
    EXPECT_EQ(service.copy({host}, {rotated}, {group_.get()}), TransferStatus::CRC_MISMATCH);
    EXPECT_TRUE(rotated.corrupted());
    expectUntouchedTargets();
    EXPECT_FALSE(std::filesystem::exists(old_dump));
    EXPECT_TRUE(std::filesystem::is_symlink(linked));
    EXPECT_EQ(std::filesystem::file_size(outside / "keep.bin"), 4u);
    dumps.clear();
    for (const auto& entry : std::filesystem::directory_iterator(root / "rank_17"))
        if (std::filesystem::is_directory(entry.symlink_status()))
            dumps.push_back(entry.path());
    ASSERT_EQ(dumps.size(), 1u);
    EXPECT_TRUE(std::filesystem::exists(dumps[0] / "manifest.json"));
    EXPECT_EQ(std::filesystem::file_size(dumps[0] / "checked_footer.bin"), 4u);
    const auto read_bytes = [](const std::filesystem::path& path) {
        std::ifstream input(path, std::ios::binary);
        return std::vector<uint8_t>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    };
    EXPECT_EQ(read_bytes(dumps[0] / "cpu.bin"), std::vector<uint8_t>(hostData(0), hostData(0) + encoded_));
    EXPECT_EQ(read_bytes(dumps[0] / "gpu_staging.bin"), std::vector<uint8_t>(hostData(0), hostData(0) + payload_));

    // A real admission I/O error must preserve the record-level mismatch flag,
    // leave the device untouched and release the slot for the next operation.
    const auto blocked = std::filesystem::path(temp_dir_->path) / "blocked_diagnostics";
    std::ofstream(blocked) << "not a directory";
    service.dump_writer_ = std::make_shared<CrcDumpWriter>(17, blocked.string());
    auto next            = TransferDescriptor::hostToDevice(0, hosts_[0], targets_[0]);
    EXPECT_EQ(service.copy({host}, {next}, {group_.get()}), TransferStatus::CRC_MISMATCH);
    EXPECT_TRUE(next.corrupted());
    EXPECT_FALSE(next.crcComputeFailed());
    expectUntouchedTargets();
    hostData(0)[1] ^= 1;
    auto healthy = TransferDescriptor::hostToDevice(0, hosts_[0], targets_[0]);
    EXPECT_EQ(service.copy({host}, {healthy}, {group_.get()}), TransferStatus::OK);
    EXPECT_FALSE(healthy.corrupted());
    EXPECT_EQ(readTarget(0), expected_[0]);
}

TEST_F(PerRankBlockTransferEngineCrcTest, Dsv4Cp16ShardedSwaRoundtripPreservesOpaqueBytes) {
    constexpr size_t cp_size = 16;
    constexpr size_t entries = 144;  // SWA window 128, gen_num_per_cycle 5, aligned to CP16.
    payload_                 = 5256;
    encoded_                 = 5264;
    layers_                  = 1;
    ASSERT_EQ(payload_ * cp_size, entries * 584);

    auto policy                = defaultCacheGroupPolicy(CacheGroupType::SWA);
    policy.sliding_window_size = 128;
    devices_                   = {makeTestDevicePool({{payload_, 0}}, 2, "cp16_swa")};
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    host_  = makeHostPool(payload_, 1, true);
    group_ = makeTestGroupSet(0,
                              makeTestTopology({makeTestGroupBase(policy, {0}, payload_, 0)}),
                              {"group0"},
                              devices_,
                              host_,
                              nullptr,
                              /*enable_crc=*/true,
                              /*physical_payload_bytes=*/payload_);
    ASSERT_TRUE(group_->crcEnabled());
    ASSERT_TRUE(group_->usesPhysicalPayloadGeometry());
    ASSERT_EQ(group_->storageBytes(), encoded_);
    engine_  = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group_});
    sources_ = {{allocate(devices_[0])}};
    targets_ = {{allocate(devices_[0])}};
    hosts_   = {allocate(host_)};
    disks_   = {NULL_BLOCK_IDX};

    // A rank owns a contiguous slice of the global data-then-scales layout,
    // not an independently quantized block. Rank 15 contains the scale tail.
    std::vector<uint8_t> expected(entries * 584);
    for (size_t byte = 0; byte < expected.size(); ++byte) {
        expected[byte] =
            byte < entries * 576 ? static_cast<uint8_t>(byte % 127) : static_cast<uint8_t>(128 + byte % 64);
    }
    const auto source = devices_[0]->convertIndexToBuffer(0, sources_[0][0]);
    ASSERT_EQ(source.size(), 1u);
    ASSERT_EQ(source[0].size_bytes, payload_);
    std::vector<uint8_t> reconstructed;
    for (size_t rank = 0; rank < cp_size; ++rank) {
        SCOPED_TRACE(::testing::Message() << "cp_rank=" << rank);
        const auto* slice = expected.data() + rank * payload_;
        ASSERT_EQ(cudaMemcpy(source[0].addr, slice, payload_, cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_TRUE(execute(Tier::DEVICE, Tier::HOST)->success());
        ASSERT_EQ(std::vector<uint8_t>(hostData(0), hostData(0) + payload_),
                  std::vector<uint8_t>(slice, slice + payload_));
        uint32_t footer = 0;
        std::memcpy(&footer, hostData(0) + encoded_ - sizeof(footer), sizeof(footer));
        EXPECT_EQ(footer, softwareCrc32c(slice, payload_));

        // Restore from host into a different block after invalidating both
        // device copies, so stale source data cannot conceal a truncated copy.
        ASSERT_EQ(cudaMemset(source[0].addr, 0xff, payload_), cudaSuccess);
        ASSERT_NO_FATAL_FAILURE(fillTargets(0xff));
        ASSERT_TRUE(execute(Tier::HOST, Tier::DEVICE)->success());
        const auto restored = readTarget(0);
        ASSERT_EQ(restored, std::vector<uint8_t>(slice, slice + payload_));
        reconstructed.insert(reconstructed.end(), restored.begin(), restored.end());
    }
    EXPECT_EQ(reconstructed, expected);
}

TEST_F(PerRankBlockTransferEngineCrcTest, PhysicalLayersWithReorderedTagsAndScalePresenceMatchIndependentCrc) {
    ASSERT_NO_FATAL_FAILURE(initialize(false, false, 3, 1, true));
    ASSERT_TRUE(group_->usesPhysicalPayloadGeometry());
    ASSERT_EQ(group_->groupTags(), (std::vector<std::string>{"group1", "group0"}));
    ASSERT_EQ(group_->topologyPtr()->layerIdsForGroup("group0"), (std::vector<int>{0, 2, 4}));
    ASSERT_EQ(group_->topologyPtr()->layerIdsForGroup("group1"), (std::vector<int>{1, 3, 5}));
    // 10 physical tiles rather than 9 logical tiles, with larger MTP layers.
    ASSERT_EQ(payload_, 312u);
    ASSERT_TRUE(execute(Tier::DEVICE, Tier::HOST)->success());
    for (size_t item = 0; item < hosts_.size(); ++item) {
        EXPECT_EQ(std::vector<uint8_t>(hostData(item), hostData(item) + payload_), expected_[item]);
        uint32_t footer = 0;
        std::memcpy(&footer, hostData(item) + encoded_ - sizeof(footer), sizeof(footer));
        EXPECT_EQ(footer, softwareCrc32c(expected_[item].data(), expected_[item].size()));
    }
    ASSERT_TRUE(execute(Tier::HOST, Tier::DISK)->success());
    ASSERT_TRUE(execute(Tier::DISK, Tier::DEVICE)->success());
    for (size_t item = 0; item < targets_.size(); ++item) {
        EXPECT_EQ(readTarget(item), expected_[item]);
    }
}

TEST_F(PerRankBlockTransferEngineCrcTest, LastPhysicalLayerCorruptionRejectsWholeBatchBeforeScatter) {
    ASSERT_NO_FATAL_FAILURE(initialize(false, false, 3, 1, true));
    ASSERT_TRUE(execute(Tier::DEVICE, Tier::HOST)->success());
    // The last byte belongs to a scale buffer absent from this tag's logical spec.
    hostData(hosts_.size() - 1)[payload_ - 1] ^= 1;
    expectIntegrityFailure(execute(Tier::HOST, Tier::DEVICE));
    expectUntouchedTargets();
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
    ASSERT_EQ(last_descriptors_.size(), 3u);
    EXPECT_FALSE(last_descriptors_[0].corrupted());
    EXPECT_FALSE(last_descriptors_[1].corrupted());
    EXPECT_TRUE(last_descriptors_[2].corrupted());
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

TEST_F(PerRankBlockTransferEngineCrcTest, HostDiskPreservesCorruptRecordUntilDeviceLoad) {
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
    const std::vector<uint8_t> corrupt(hostData(2), hostData(2) + encoded_);
    const auto                 dumps = captureDumps("host_to_device_dump");
    ASSERT_TRUE(execute(Tier::HOST, Tier::DISK)->success());
    for (size_t item = 0; item < hosts_.size(); ++item) {
        EXPECT_FALSE(last_descriptors_[item].corrupted());
        std::vector<uint8_t> raw(disk_->strideBytes());
        ASSERT_EQ(disk_->read(disks_[item], raw.data(), raw.size()), BlockIOStatus::OK);
        EXPECT_EQ(std::vector<uint8_t>(raw.begin(), raw.begin() + encoded_), item == 2 ? corrupt : sealed[item]);
        std::memset(hostData(item), 0xcc, host_->strideBytes());
    }
    ASSERT_TRUE(execute(Tier::DISK, Tier::HOST)->success());
    for (size_t item = 0; item < hosts_.size(); ++item) {
        EXPECT_FALSE(last_descriptors_[item].corrupted());
        EXPECT_EQ(std::vector<uint8_t>(hostData(item), hostData(item) + encoded_), item == 2 ? corrupt : sealed[item]);
    }
    EXPECT_FALSE(std::filesystem::exists(dumps));
    expectIntegrityFailure(execute(Tier::HOST, Tier::DEVICE));
    expectUntouchedTargets();
    ASSERT_NO_FATAL_FAILURE(
        expectDump(dumps,
                   corrupt,
                   {"\"cpu_backing\": \"memory_pool\"", "\"operation\": \"load\"", "\"direction\": \"HOST->DEVICE\""}));
}

TEST_F(PerRankBlockTransferEngineCrcTest, DiskCorruptionIsRejectedWithoutDeviceScatter) {
    ASSERT_NO_FATAL_FAILURE(initialize());
    ASSERT_TRUE(execute(Tier::DEVICE, Tier::DISK)->success());
    std::vector<uint8_t> raw(disk_->strideBytes());
    ASSERT_EQ(disk_->read(disks_[2], raw.data(), raw.size()), BlockIOStatus::OK);
    raw[payload_ / 2] ^= 0x40;
    ASSERT_EQ(disk_->write(disks_[2], raw.data(), raw.size()), BlockIOStatus::OK);
    const auto host_dumps = captureDumps("disk_to_host_dump");
    ASSERT_TRUE(execute(Tier::DISK, Tier::HOST)->success());
    EXPECT_EQ(std::vector<uint8_t>(hostData(2), hostData(2) + encoded_),
              std::vector<uint8_t>(raw.begin(), raw.begin() + encoded_));
    EXPECT_FALSE(std::filesystem::exists(host_dumps));
    expectIntegrityFailure(execute(Tier::HOST, Tier::DEVICE));
    expectUntouchedTargets();
    ASSERT_NO_FATAL_FAILURE(
        expectDump(host_dumps,
                   raw,
                   {"\"cpu_backing\": \"memory_pool\"", "\"operation\": \"load\"", "\"direction\": \"HOST->DEVICE\""}));
    const auto device_dumps      = captureDumps("disk_to_device_dump");
    const auto read_bytes_before = disk_->readBytes();
    expectIntegrityFailure(execute(Tier::DISK, Tier::DEVICE));
    EXPECT_EQ(disk_->readBytes() - read_bytes_before, disks_.size() * disk_->strideBytes());
    ASSERT_EQ(last_descriptors_.size(), 3u);
    EXPECT_FALSE(last_descriptors_[0].corrupted());
    EXPECT_FALSE(last_descriptors_[1].corrupted());
    EXPECT_TRUE(last_descriptors_[2].corrupted());
    expectUntouchedTargets();
    ASSERT_NO_FATAL_FAILURE(expectDump(device_dumps,
                                       raw,
                                       {"\"cpu_backing\": \"disk_staging_buffer\"",
                                        "\"operation\": \"load\"",
                                        "\"direction\": \"DISK->DEVICE\"",
                                        "\"layout_tile_count\": \"8\"",
                                        "\"tile_count\": \"8\"",
                                        "\"cpu_captured\": \"true\"",
                                        "\"staging_captured\": \"true\"",
                                        "\"output_written\": \"false\"",
                                        "\"disk_file\": \"" + disk_->filePath() + "\"",
                                        "\"disk_slot\": \"" + std::to_string(disks_[2]) + "\"",
                                        "\"disk_offset\": \"" + std::to_string(disk_->blockOffset(disks_[2])) + "\""}));
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

TEST_F(PerRankBlockTransferEngineCrcTest, SegmentedHostDiskPreservesCorruptionUntilDeviceLoad) {
    ASSERT_NO_FATAL_FAILURE(initialize(false, false, 32, 256));
    ASSERT_TRUE(execute(Tier::DEVICE, Tier::HOST)->success());
    std::vector<std::vector<uint8_t>> records;
    for (size_t item = 0; item < hosts_.size(); ++item) {
        records.emplace_back(hostData(item), hostData(item) + encoded_);
    }
    ASSERT_TRUE(execute(Tier::HOST, Tier::DISK)->success());
    hostData(31)[16384] ^= 0x80;
    const std::vector<uint8_t> corrupt(hostData(31), hostData(31) + encoded_);
    ASSERT_TRUE(execute(Tier::HOST, Tier::DISK)->success());
    EXPECT_EQ(std::vector<uint8_t>(hostData(31), hostData(31) + encoded_), corrupt);
    for (size_t item = 0; item < hosts_.size(); ++item) {
        std::memset(hostData(item), 0xcc, host_->strideBytes());
    }
    ASSERT_TRUE(execute(Tier::DISK, Tier::HOST)->success());
    for (size_t item = 0; item < hosts_.size(); ++item) {
        EXPECT_EQ(std::vector<uint8_t>(hostData(item), hostData(item) + encoded_),
                  item == 31 ? corrupt : records[item]);
    }
    fillTargets(0xa5);
    expectIntegrityFailure(execute(Tier::HOST, Tier::DEVICE));
    expectUntouchedTargets();
    expectIntegrityFailure(execute(Tier::DISK, Tier::DEVICE));
    expectUntouchedTargets();
}

}  // namespace
}  // namespace rtp_llm

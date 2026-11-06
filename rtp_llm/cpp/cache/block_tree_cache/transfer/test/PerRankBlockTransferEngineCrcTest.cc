#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/LegacyTransferEngine.h"
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <future>
#include <limits>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostStagingBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferRequestConverter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"
#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcTransferService.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyGeometry.h"

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

class PerRankBlockTransferEngineCrcTest: public ::testing::TestWithParam<bool> {
protected:
    void SetUp() override {
        if (!CrcCopyWorkspace::available()) {
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
        if (dump_root_restore_) {
            auto                        writer = CrcDumpWriter::forRank(0);
            std::lock_guard<std::mutex> lock(writer->mutex_);
            writer->rank_dir_ = *dump_root_restore_;
        }
        for (auto& [pool, block] : owned_) {
            if (async_caller_references_)
                pool->decTreeRef(block, BlockTreeRefType::CACHE);
            else
                releasePoolBlock(*pool, block);
        }
        owned_.clear();
        group_.reset();
        devices_.clear();
        host_.reset();
        disk_.reset();
        temp_dir_.reset();
    }

    void retainAsyncAllocations() {
        for (const auto& entry : owned_)
            entry.first->incTreeRef(entry.second, BlockTreeRefType::CACHE);
        async_caller_references_ = true;
    }
    bool async_caller_references_{false};

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
        // Keep every corruption case in one executor batch.
        // Disk staging reserves half its slots for FULL.
        engine_ = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group_},
                                                               true,
                                                               DeviceHostCopyOptions{},
                                                               std::max<size_t>(8, (GetParam() ? 4 : 2) * item_count),
                                                               std::max<size_t>(8, item_count),
                                                               1,
                                                               10000,
                                                               nullptr,
                                                               0,
                                                               4,
                                                               GetParam());
        ASSERT_TRUE(engine_->asyncEnabled());
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

    std::shared_ptr<TransferAsyncContext> execute(Tier from, Tier to) {
        std::vector<TransferDescriptor> descriptors;
        for (size_t item = 0; item < sources_.size(); ++item) {
            descriptors.push_back(makeDescriptor(
                from, to, from == Tier::DEVICE ? sources_[item] : targets_[item], hosts_[item], disks_[item]));
        }
        auto context = engine_->execute(makeTransferTask(std::move(descriptors)));
        context->waitDone();
        EXPECT_TRUE(context->done());
        return context;
    }

    void expectIntegrityFailure(const std::shared_ptr<TransferAsyncContext>& context) {
        ASSERT_FALSE(context->success());
        EXPECT_EQ(context->errorInfo().code(), ErrorCode::EXECUTION_EXCEPTION);
        EXPECT_EQ(context->result().copy_error, CacheCopyError::CRC_MISMATCH);
    }

    void expectUntouchedTargets() {
        for (size_t item = 0; item < targets_.size(); ++item) {
            EXPECT_EQ(readTarget(item), std::vector<uint8_t>(payload_, 0xa5)) << "item " << item;
        }
    }

    void expectScatteredTargets(size_t count) {
        for (size_t item = 0; item < targets_.size(); ++item) {
            const auto expected = item < count ? std::vector<uint8_t>(hostData(item), hostData(item) + payload_) :
                                                 std::vector<uint8_t>(payload_, 0xa5);
            EXPECT_EQ(readTarget(item), expected) << "item " << item;
        }
    }

    std::filesystem::path captureDumps(const char* name) {
        const auto root = std::filesystem::path(temp_dir_->path) / name;
        if (GetParam()) {
            // The shared writer serializes directory changes with diagnostics;
            // no legacy executor exists in WorkLoop mode.
            auto                        writer = CrcDumpWriter::forRank(0);
            std::lock_guard<std::mutex> lock(writer->mutex_);
            if (!dump_root_restore_)
                dump_root_restore_ = writer->rank_dir_;
            writer->rank_dir_ = (root / "rank_0").string();
            return root / "rank_0";
        }
        const auto writer          = std::make_shared<CrcDumpWriter>(0, root.string());
        auto&      executor        = *engine_->legacy_->device_host_executor_;
        executor.load_crc_service_ = std::make_shared<CrcTransferService>(
            std::vector<GroupSetPtr>{group_}, std::max<size_t>(8, hosts_.size()), 1, 0, writer, false);
        executor.offload_crc_service_ = std::make_shared<CrcTransferService>(
            std::vector<GroupSetPtr>{group_}, std::max<size_t>(8, hosts_.size()), 1, 0, writer, false);
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

    std::optional<std::string>                                        dump_root_restore_;
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

// Test-owned result storage is not a backend operation handle. Service poll
// delivers all callbacks, including those whose test result owner was dropped.
struct ServiceCompletion {
    std::optional<TransferResult> result;
    size_t                        callbacks{0};
};
using ServiceCompletionPtr = std::shared_ptr<ServiceCompletion>;

ServiceCompletionPtr submitService(CrcTransferService&                    service,
                                   const std::vector<HostBufferView>&     hosts,
                                   const std::vector<TransferDescriptor>& descriptors,
                                   const GroupSetPtr&                     group) {
    auto completion = std::make_shared<ServiceCompletion>();
    if (!service.trySubmit(hosts, descriptors, group, [completion](TransferResult result) {
            EXPECT_EQ(++completion->callbacks, 1u);
            completion->result = std::move(result);
        }))
        return nullptr;
    return completion;
}

std::optional<TransferResult> waitService(CrcTransferService& service, const ServiceCompletionPtr& completion) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!completion->result) {
        service.poll();
        if (std::chrono::steady_clock::now() >= deadline)
            throw std::runtime_error("CRC service completion deadline exceeded");
        std::this_thread::yield();
    }
    return completion->result;
}

void expectRejected(CrcTransferService& service, const ServiceCompletionPtr& completion) {
    ASSERT_TRUE(completion);
    ASSERT_TRUE(completion->result);  // Immediate rejection must precede return.
    EXPECT_EQ(completion->result->copy_error, CacheCopyError::INVALID_REQUEST);
    service.poll();
    EXPECT_EQ(completion->callbacks, 1u);
}

TEST_P(PerRankBlockTransferEngineCrcTest, BufferBoundsFailuresDoNotCopyOrConsumeSlots) {
    ASSERT_NO_FATAL_FAILURE(initialize());
    retainAsyncAllocations();
    CrcTransferService         service({group_}, 1, 1);
    CrcTransferService         sync_service({group_}, 1, 1, 0, nullptr, false);
    const auto                 buffer = host_->blockBuffer(hosts_[0]);
    const HostBufferView       view{buffer.addr, payload_, buffer.stride_bytes};
    const auto                 descriptor = TransferDescriptor::deviceToHost(0, sources_[0], hosts_[0]);
    const std::vector<uint8_t> guard(encoded_, 0x5a);
    std::memcpy(buffer.addr, guard.data(), guard.size());
    const auto reject = [&](const HostBufferView& host, const TransferDescriptor& desc) {
        SCOPED_TRACE(desc.debugString());
        EXPECT_EQ(sync_service.copy({host}, {desc}, {group_.get()}).copy_error, CacheCopyError::INVALID_REQUEST);
        const auto submitted = submitService(service, {host}, {desc}, group_);
        expectRejected(service, submitted);
    };
    auto outside_pool             = descriptor;
    outside_pool.source_blocks[0] = std::numeric_limits<BlockIdxType>::max();
    reject(view, outside_pool);
    reject({nullptr, payload_, buffer.stride_bytes}, descriptor);
    reject({buffer.addr, payload_, encoded_ - 1}, descriptor);
    EXPECT_EQ(sync_service.copy({view}, {}, {group_.get()}).copy_error, CacheCopyError::INVALID_REQUEST);
    expectRejected(service, submitService(service, {view}, {}, group_));
    EXPECT_EQ(sync_service.copy({view, view}, {descriptor, descriptor}, {group_.get(), group_.get()}).copy_error,
              CacheCopyError::INVALID_REQUEST);
    expectRejected(service, submitService(service, {view, view}, {descriptor, descriptor}, group_));
    EXPECT_EQ(std::memcmp(buffer.addr, guard.data(), guard.size()), 0);
    expectUntouchedTargets();
    ASSERT_TRUE(sync_service.copy({view}, {descriptor}, {group_.get()}).ok());
    EXPECT_EQ(service.copy({view}, {descriptor}, {group_.get()}).copy_error, CacheCopyError::INVALID_REQUEST);
    auto accepted = submitService(service, {view}, {descriptor}, group_);
    ASSERT_TRUE(accepted);
    EXPECT_TRUE(waitService(service, accepted)->ok());
    EXPECT_EQ(std::memcmp(buffer.addr, expected_[0].data(), payload_), 0);
}

TEST_P(PerRankBlockTransferEngineCrcTest, CopyPreservesSameDeviceMixedGroupBatches) {
    ASSERT_NO_FATAL_FAILURE(initialize());
    auto second_group = makeTestGroupSet(1, group_->topologyPtr(), group_->groupTags(), devices_, host_, disk_, true);
    CrcTransferService                 service({group_, second_group}, 3, 1, 0, nullptr, false);
    std::vector<HostBufferView>        views;
    std::vector<TransferDescriptor>    descriptors;
    const std::vector<const GroupSet*> groups{group_.get(), second_group.get(), group_.get()};
    for (size_t i = 0; i < groups.size(); ++i) {
        const auto buffer = host_->blockBuffer(hosts_[i]);
        views.push_back({buffer.addr, payload_, buffer.stride_bytes});
        descriptors.push_back(TransferDescriptor::deviceToHost(groups[i]->groupSetId(), sources_[i], hosts_[i]));
    }
    ASSERT_TRUE(service.copy(views, descriptors, groups).ok());
    for (size_t i = 0; i < groups.size(); ++i)
        EXPECT_EQ(std::memcmp(views[i].base, expected_[i].data(), payload_), 0);
}

TEST_P(PerRankBlockTransferEngineCrcTest, AsyncDirectionQuotasAndIndependentBytes) {
    initialize(false, false, 17);
    retainAsyncAllocations();
    for (size_t parallel : {1u, 4u, 8u}) {
        CrcTransferService store_service({group_}, 8, parallel);
        CrcTransferService load_service({group_}, 8, parallel);
        ASSERT_EQ(store_service.slots_.at(0).size(), parallel);
        ASSERT_EQ(load_service.slots_.at(0).size(), parallel);
        std::vector<ServiceCompletionPtr> stores, loads;
        for (size_t i = 0; i < parallel; ++i) {
            const auto buffer    = host_->blockBuffer(hosts_[i]);
            auto       submitted = submitService(store_service,
                                                 {{buffer.addr, payload_, buffer.stride_bytes}},
                                                 {TransferDescriptor::deviceToHost(0, sources_[i], hosts_[i])},
                                           group_);
            ASSERT_TRUE(submitted);
            stores.push_back(std::move(submitted));
        }
        for (size_t i = parallel; i < 2 * parallel; ++i) {
            const auto buffer = host_->blockBuffer(hosts_[i]);
            std::memcpy(buffer.addr, expected_[i].data(), payload_);
            const auto checksum = softwareCrc32c(expected_[i].data(), payload_);
            std::memcpy(static_cast<uint8_t*>(buffer.addr) + encoded_ - 4, &checksum, 4);
            auto submitted = submitService(load_service,
                                           {{buffer.addr, payload_, buffer.stride_bytes}},
                                           {TransferDescriptor::hostToDevice(0, hosts_[i], targets_[i])},
                                           group_);
            ASSERT_TRUE(submitted);
            loads.push_back(std::move(submitted));
        }
        const size_t extra  = 2 * parallel;
        const auto   buffer = host_->blockBuffer(hosts_[extra]);
        EXPECT_EQ(submitService(store_service,
                                {{buffer.addr, payload_, buffer.stride_bytes}},
                                {TransferDescriptor::deviceToHost(0, sources_[extra], hosts_[extra])},
                                group_),
                  nullptr);
        EXPECT_EQ(submitService(load_service,
                                {{buffer.addr, payload_, buffer.stride_bytes}},
                                {TransferDescriptor::hostToDevice(0, hosts_[extra], targets_[extra])},
                                group_),
                  nullptr);
        for (size_t i = 0; i < parallel; ++i) {
            auto result = waitService(store_service, stores[i]);
            ASSERT_TRUE(result->ok()) << result->status.ToString();
            EXPECT_EQ(std::memcmp(hostData(i), expected_[i].data(), payload_), 0);
            uint32_t checksum;
            std::memcpy(&checksum, hostData(i) + encoded_ - 4, 4);
            EXPECT_EQ(checksum, softwareCrc32c(expected_[i].data(), payload_));
            ASSERT_TRUE(waitService(load_service, loads[i])->ok());
            EXPECT_EQ(readTarget(i + parallel), expected_[i + parallel]);
            EXPECT_EQ(host_->treeRefCount(hosts_[i]), 1u);
            EXPECT_EQ(host_->treeRefCount(hosts_[i + parallel]), 1u);
            for (size_t member = 0; member < devices_.size(); ++member) {
                EXPECT_EQ(devices_[member]->treeRefCount(sources_[i][member]), 1u);
                EXPECT_EQ(devices_[member]->treeRefCount(targets_[i + parallel][member]), 1u);
            }
            EXPECT_EQ(stores[i]->callbacks, 1u);
        }
    }
}

TEST_P(PerRankBlockTransferEngineCrcTest, AsyncFixedWorkspaceSwitchesFullAndSwaFullAndTailBatches) {
    initialize();
    auto policy                         = defaultCacheGroupPolicy(CacheGroupType::SWA);
    policy.sliding_window_size          = 128;
    constexpr size_t     swa_payload    = 4096;
    auto                 swa_device     = makeTestDevicePool({{swa_payload, 0}}, 2, "async_swa");
    auto                 swa_host       = makeHostPool(swa_payload, 2, true);
    auto                 swa            = makeTestGroupSet(1,
                                makeTestTopology({makeTestGroupBase(policy, {0}, swa_payload, 0)}),
                                                           {"group0"},
                                                           {swa_device},
                                swa_host,
                                nullptr,
                                true);
    const auto           swa_block      = allocate(swa_device);
    const auto           swa_host_block = allocate(swa_host);
    std::vector<uint8_t> swa_bytes(swa_payload);
    for (size_t i = 0; i < swa_bytes.size(); ++i)
        swa_bytes[i] = static_cast<uint8_t>(i * 29 + 97);
    const auto swa_buffer = swa_device->convertIndexToBuffer(0, swa_block).front();
    ASSERT_EQ(cudaMemcpy(swa_buffer.addr, swa_bytes.data(), swa_payload, cudaMemcpyHostToDevice), cudaSuccess);
    retainAsyncAllocations();
    CrcTransferService                                store_service({group_, swa}, 3, 1);
    CrcTransferService                                load_service({group_, swa}, 3, 1);
    const std::vector<std::pair<GroupSetPtr, size_t>> turns{{group_, 3}, {swa, 1}, {group_, 1}, {swa, 1}, {group_, 3}};
    for (const auto& [group, count] : turns) {
        for (bool store : {true, false}) {
            std::vector<HostBufferView>     views;
            std::vector<TransferDescriptor> descriptors;
            for (size_t i = 0; i < count; ++i) {
                const bool is_swa     = group == swa;
                const auto host_block = is_swa ? swa_host_block : hosts_[i];
                const auto buffer     = group->hostPool()->blockBuffer(host_block);
                views.push_back({buffer.addr, group->payloadBytes(), buffer.stride_bytes});
                const auto blocks = is_swa ? std::vector<BlockIdxType>{swa_block} : (store ? sources_[i] : targets_[i]);
                descriptors.push_back(store ?
                                          TransferDescriptor::deviceToHost(group->groupSetId(), blocks, host_block) :
                                          TransferDescriptor::hostToDevice(group->groupSetId(), host_block, blocks));
            }
            auto& service   = store ? store_service : load_service;
            auto  submitted = submitService(service, views, descriptors, group);
            ASSERT_TRUE(submitted);
            EXPECT_TRUE(waitService(service, submitted)->ok());
            for (size_t i = 0; i < count; ++i) {
                const auto& expected = group == swa ? swa_bytes : expected_[i];
                EXPECT_EQ(std::memcmp(views[i].base, expected.data(), expected.size()), 0);
                uint32_t checksum;
                std::memcpy(&checksum, static_cast<uint8_t*>(views[i].base) + group->storageBytes() - 4, 4);
                EXPECT_EQ(checksum, softwareCrc32c(expected.data(), expected.size()));
                if (!store && group != swa) {
                    EXPECT_EQ(readTarget(i), expected);
                }
            }
        }
    }
}

TEST_P(PerRankBlockTransferEngineCrcTest, AsyncCompletedCallbackCannotAffectReusedSlot) {
    ASSERT_NO_FATAL_FAILURE(initialize());
    retainAsyncAllocations();
    CrcTransferService service({group_}, 3, 1);
    const auto         submit = [&](size_t index) {
        const auto buffer = host_->blockBuffer(hosts_[index]);
        return submitService(service,
                                     {{buffer.addr, payload_, buffer.stride_bytes}},
                                     {TransferDescriptor::deviceToHost(0, sources_[index], hosts_[index])},
                             group_);
    };
    auto first = submit(0);
    ASSERT_TRUE(first);
    ASSERT_TRUE(waitService(service, first)->ok());
    auto second = submit(1);
    ASSERT_TRUE(second);
    EXPECT_EQ(first->callbacks, 1u);
    EXPECT_EQ(submit(2), nullptr);
    first.reset();
    EXPECT_EQ(submit(2), nullptr);
    ASSERT_TRUE(waitService(service, second)->ok());
    EXPECT_EQ(std::memcmp(hostData(1), expected_[1].data(), payload_), 0);
    auto third = submit(2);
    ASSERT_TRUE(third);
    ASSERT_TRUE(waitService(service, third)->ok());
    EXPECT_EQ(std::memcmp(hostData(2), expected_[2].data(), payload_), 0);
}

TEST_P(PerRankBlockTransferEngineCrcTest, AsyncDroppedResultOwnerDrainsWithCallerOwnedReferences) {
    initialize();
    retainAsyncAllocations();
    CrcTransferService service({group_}, 3, 1);
    const auto         buffer    = host_->blockBuffer(hosts_[0]);
    auto               submitted = submitService(service,
                                                 {{buffer.addr, payload_, buffer.stride_bytes}},
                                                 {TransferDescriptor::deviceToHost(0, sources_[0], hosts_[0])},
                                   group_);
    ASSERT_TRUE(submitted);
    EXPECT_EQ(host_->treeRefCount(hosts_[0]), 1u);
    for (size_t member = 0; member < devices_.size(); ++member)
        EXPECT_EQ(devices_[member]->treeRefCount(sources_[0][member]), 1u);
    std::weak_ptr<ServiceCompletion> weak = submitted;
    submitted.reset();
    EXPECT_FALSE(weak.expired());
    // Dropping the handle does not retire CUDA work or release caller-owned
    // business references. Keep those references until drain proves completion.
    service.stopAndWaitForIo();
    EXPECT_FALSE(weak.expired());
    EXPECT_FALSE(service.poll());
    EXPECT_TRUE(weak.expired());
    EXPECT_EQ(std::memcmp(hostData(0), expected_[0].data(), payload_), 0);
    auto releaseCaller = [&](const std::shared_ptr<IBlockPool>& pool, BlockIdxType block) {
        const auto found = std::find(owned_.begin(), owned_.end(), std::make_pair(pool, block));
        ASSERT_NE(found, owned_.end());
        EXPECT_EQ(pool->treeRefCount(block), 1u);
        pool->decTreeRef(block, BlockTreeRefType::CACHE);
        owned_.erase(found);
        EXPECT_FALSE(pool->isAllocated(block));
    };
    for (size_t member = 0; member < devices_.size(); ++member)
        releaseCaller(devices_[member], sources_[0][member]);
    releaseCaller(host_, hosts_[0]);
    const auto rejected = submitService(service,
                                        {{buffer.addr, payload_, buffer.stride_bytes}},
                                        {TransferDescriptor::deviceToHost(0, sources_[1], hosts_[1])},
                                        group_);
    expectRejected(service, rejected);
}

TEST_P(PerRankBlockTransferEngineCrcTest, AsyncBatchFailurePreservesIndicesAndRejectsInvalidAdmission) {
    initialize();
    retainAsyncAllocations();
    CrcTransferService              service({group_}, 3, 1);
    std::vector<HostBufferView>     views;
    std::vector<TransferDescriptor> descriptors;
    for (size_t i = 0; i < hosts_.size(); ++i) {
        const auto buffer = host_->blockBuffer(hosts_[i]);
        std::memcpy(buffer.addr, expected_[i].data(), payload_);
        const auto checksum = softwareCrc32c(expected_[i].data(), payload_);
        std::memcpy(hostData(i) + encoded_ - 4, &checksum, 4);
        views.push_back({buffer.addr, payload_, buffer.stride_bytes});
        descriptors.push_back(TransferDescriptor::hostToDevice(0, hosts_[i], targets_[i]));
    }
    const auto capacity         = views.back().capacity_bytes;
    views.back().capacity_bytes = encoded_ - 1;
    expectRejected(service, submitService(service, views, descriptors, group_));
    views.back().capacity_bytes = capacity;
    hostData(2)[encoded_ - 1] ^= 1;
    auto submitted = submitService(service, views, descriptors, group_);
    ASSERT_TRUE(submitted);
    // All operation metadata must survive destruction of the submission inputs.
    views.clear();
    descriptors.clear();
    auto result = waitService(service, submitted);
    EXPECT_EQ(result->copy_error, CacheCopyError::CRC_MISMATCH);
    EXPECT_EQ(result->block_errors,
              (std::vector<CacheCopyError>{
                  CacheCopyError::NOT_EXECUTED, CacheCopyError::NOT_EXECUTED, CacheCopyError::CRC_MISMATCH}));
    EXPECT_THROW((CrcTransferService({group_}, 3, 0)), std::invalid_argument);
    EXPECT_THROW((CrcTransferService({group_}, 3, std::numeric_limits<size_t>::max())), std::overflow_error);
}

TEST_P(PerRankBlockTransferEngineCrcTest, AsyncDeviceErrorsPreserveCallerReferencesAndReuseSlots) {
    initialize();
    retainAsyncAllocations();
    const auto buffer     = host_->blockBuffer(hosts_[0]);
    const auto descriptor = TransferDescriptor::deviceToHost(0, sources_[0], hosts_[0]);
    // Cover the service's immediate submit error and accepted-operation error.
    // Per-CUDA-boundary injection and drain counts belong to CrcBlockCopyBatchTest.
    for (const char* phase : {"metadata", "query"}) {
        SCOPED_TRACE(phase);
        CrcTransferService service({group_}, 3, 1);
        for (auto& slot : service.slots_.begin()->second)
            slot.copy->setTestHook([phase](const char* point, void*, void*) {
                return int(std::strcmp(point, phase) == 0 ? cudaErrorInvalidValue : cudaSuccess);
            });
        auto submitted = submitService(service, {{buffer.addr, payload_, buffer.stride_bytes}}, {descriptor}, group_);
        ASSERT_TRUE(submitted);
        const bool immediate = std::strcmp(phase, "metadata") == 0;
        if (!immediate) {
            EXPECT_EQ(waitService(service, submitted)->copy_error, CacheCopyError::COPY_FAILED);
        }
        EXPECT_EQ(host_->treeRefCount(hosts_[0]), 1u);
        for (size_t member = 0; member < devices_.size(); ++member)
            EXPECT_EQ(devices_[member]->treeRefCount(sources_[0][member]), 1u);
        for (auto& slot : service.slots_.begin()->second)
            slot.copy->setTestHook({});
        auto healthy = submitService(service, {{buffer.addr, payload_, buffer.stride_bytes}}, {descriptor}, group_);
        ASSERT_TRUE(healthy);
        // An immediate error retains its result after the slot is reused.
        if (immediate) {
            EXPECT_EQ(waitService(service, submitted)->copy_error, CacheCopyError::COPY_FAILED);
        }
        EXPECT_EQ(submitted->callbacks, 1u);
        EXPECT_TRUE(waitService(service, healthy)->ok());
    }
}

TEST_P(PerRankBlockTransferEngineCrcTest, AsyncDrainPreservesUnconsumedResultAndCallerReferences) {
    initialize();
    retainAsyncAllocations();
    const auto         buffer     = host_->blockBuffer(hosts_[0]);
    const auto         descriptor = TransferDescriptor::deviceToHost(0, sources_[0], hosts_[0]);
    CrcTransferService service({group_}, 3, 1);
    auto submitted = submitService(service, {{buffer.addr, payload_, buffer.stride_bytes}}, {descriptor}, group_);
    ASSERT_TRUE(submitted);
    EXPECT_EQ(host_->treeRefCount(hosts_[0]), 1u);
    service.stopAndWaitForIo();
    EXPECT_EQ(host_->treeRefCount(hosts_[0]), 1u);
    EXPECT_FALSE(submitted->result);
    EXPECT_FALSE(service.poll());
    auto result = submitted->result;
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->ok());
    EXPECT_EQ(submitted->callbacks, 1u);
    for (size_t member = 0; member < devices_.size(); ++member)
        EXPECT_EQ(devices_[member]->treeRefCount(sources_[0][member]), 1u);
    // Business settlement can recycle a safe block while a consumed handle
    // still exists. Destroying that handle must not touch the recycled block.
    const auto owned =
        std::find(owned_.begin(), owned_.end(), std::make_pair(std::shared_ptr<IBlockPool>(host_), hosts_[0]));
    ASSERT_NE(owned, owned_.end());
    host_->decTreeRef(hosts_[0], BlockTreeRefType::CACHE);
    owned_.erase(owned);
    EXPECT_FALSE(host_->isAllocated(hosts_[0]));
    submitted.reset();
    EXPECT_FALSE(host_->isAllocated(hosts_[0]));
    for (size_t member = 0; member < devices_.size(); ++member)
        EXPECT_EQ(devices_[member]->treeRefCount(sources_[0][member]), 1u);
}

TEST_P(PerRankBlockTransferEngineCrcTest, AsyncUsesCallerOwnedStagingLeases) {
    initialize();
    retainAsyncAllocations();
    HostStagingBlockPool pool(1, group_->diskPool()->strideBytes());
    auto                 leases = pool.tryMallocBatch(1);
    ASSERT_TRUE(leases.has_value());
    // Drain the service before the caller-owned leases/pool on assertion failure.
    CrcTransferService service({group_}, 3, 1);
    const auto         view = leases->front().blockBuffer(payload_);
    auto               submitted =
        submitService(service, {view}, {TransferDescriptor::deviceToDisk(0, sources_[0], disks_[0])}, group_);
    ASSERT_TRUE(submitted);
    EXPECT_FALSE(pool.tryMallocBatch(1).has_value());
    const auto result = waitService(service, submitted);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->ok());
    EXPECT_EQ(std::memcmp(view.base, expected_[0].data(), payload_), 0);
    EXPECT_FALSE(pool.tryMallocBatch(1).has_value());
    leases.reset();
    EXPECT_TRUE(pool.tryMallocBatch(1).has_value());
}

TEST_P(PerRankBlockTransferEngineCrcTest, AsyncExecutionSelectsDeviceOnForeignThread) {
    int count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
    if (count < 2)
        GTEST_SKIP() << "requires two locked GPUs to check a different initial device";
    initialize();
    retainAsyncAllocations();
    CrcTransferService service({group_}, 3, 1);
    const auto         buffer = host_->blockBuffer(hosts_[0]);
    std::exception_ptr error;
    std::thread        owner([&] {
        try {
            if (cudaSetDevice(1) != cudaSuccess)
                throw std::runtime_error("cannot select foreign CUDA device");
            auto submitted = submitService(service,
                                                  {{buffer.addr, payload_, buffer.stride_bytes}},
                                                  {TransferDescriptor::deviceToHost(0, sources_[0], hosts_[0])},
                                           group_);
            if (!submitted || !waitService(service, submitted)->ok())
                throw std::runtime_error("CRC from foreign CUDA thread failed");
        } catch (...) {
            error = std::current_exception();
        }
    });
    owner.join();
    if (error)
        std::rethrow_exception(error);
    EXPECT_EQ(std::memcmp(hostData(0), expected_[0].data(), payload_), 0);
}

TEST_P(PerRankBlockTransferEngineCrcTest, AsyncInlineDiagnosticsAndTeardownSelectNonDefaultDevice) {
    int count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
    if (count < 2)
        GTEST_SKIP() << "requires two locked GPUs for inline diagnostics device binding";
    ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
    initialize();
    ASSERT_EQ(devices_.front()->deviceIndex(), 1);
    retainAsyncAllocations();
    const auto buffer = host_->blockBuffer(hosts_[0]);
    std::memcpy(buffer.addr, expected_[0].data(), payload_);
    const auto checksum = softwareCrc32c(expected_[0].data(), payload_);
    std::memcpy(hostData(0) + encoded_ - 4, &checksum, 4);
    hostData(0)[0] ^= 1;
    int snapshot_device{-1}, drain_device{-1};
    {
        auto               writer = std::make_shared<CrcDumpWriter>(0, temp_dir_->path + "/device_1");
        CrcTransferService load_service({group_}, 3, 1, 0, writer);
        CrcTransferService store_service({group_}, 3, 1, 0, writer);
        for (auto& slot : load_service.slots_.at(1))
            slot.copy->setTestHook([&](const char* point, void*, void*) {
                if (std::strcmp(point, "snapshot") == 0)
                    cudaGetDevice(&snapshot_device);
                return int(cudaSuccess);
            });
        for (auto& slot : store_service.slots_.at(1))
            slot.copy->setTestHook([&](const char* point, void*, void*) {
                if (std::strcmp(point, "drain") == 0)
                    cudaGetDevice(&drain_device);
                return int(std::strcmp(point, "metadata") == 0 ? cudaErrorInvalidValue : cudaSuccess);
            });
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        auto load = submitService(load_service,
                                  {{buffer.addr, payload_, buffer.stride_bytes}},
                                  {TransferDescriptor::hostToDevice(0, hosts_[0], targets_[0])},
                                  group_);
        ASSERT_TRUE(load);
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        EXPECT_EQ(waitService(load_service, load)->copy_error, CacheCopyError::CRC_MISMATCH);
        EXPECT_EQ(snapshot_device, 1);
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        auto store = submitService(store_service,
                                   {{buffer.addr, payload_, buffer.stride_bytes}},
                                   {TransferDescriptor::deviceToHost(0, sources_[0], hosts_[0])},
                                   group_);
        ASSERT_TRUE(store);
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        EXPECT_FALSE(waitService(store_service, store)->ok());
        EXPECT_EQ(drain_device, 1);
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    }
    int device = -1;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    EXPECT_EQ(device, 1);
}

TEST_P(PerRankBlockTransferEngineCrcTest, AsyncCallbackRunsOnOwnerAfterSlotRetirement) {
    initialize();
    retainAsyncAllocations();
    CrcTransferService service({group_}, 3, 1);
    const auto         buffer = host_->blockBuffer(hosts_[0]);
    bool               called = false, retired = false;
    const auto         owner = std::this_thread::get_id();
    ASSERT_TRUE(service.trySubmit(
        {{buffer.addr, payload_, buffer.stride_bytes}},
        {TransferDescriptor::deviceToHost(0, sources_[0], hosts_[0])},
        group_,
        [&](TransferResult result) {
            called  = true;
            retired = std::all_of(service.slots_.at(0).begin(), service.slots_.at(0).end(), [](const auto& slot) {
                return !slot.operation && !slot.updating;
            });
            EXPECT_TRUE(result.ok());
            EXPECT_EQ(std::this_thread::get_id(), owner);
        }));
    service.stopAndWaitForIo();
    EXPECT_FALSE(called);
    EXPECT_FALSE(service.poll());
    EXPECT_TRUE(called);
    EXPECT_TRUE(retired);
    service.poll();
}

TEST_P(PerRankBlockTransferEngineCrcTest, MultiBackingMemberLayerScaleRoundTripMatchesIndependentCrc) {
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

TEST_P(PerRankBlockTransferEngineCrcTest, MixedRpcBatchIsRejectedBeforeCopy) {
    ASSERT_NO_FATAL_FAILURE(initialize(true, false, 2));
    ASSERT_TRUE(execute(Tier::DEVICE, Tier::HOST)->success());
    engine_.reset();
    auto topology = makeTestTopology({makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, payload_),
                                      makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {1}, payload_)});
    group_->initialize(0, topology, {"group0"}, makeTestBackingLayout(payload_, true));
    auto                           ordinary = makeTestGroupSet(1, topology, {"group1"}, devices_, host_);
    const std::vector<GroupSetPtr> groups{group_, ordinary};
    engine_ = std::make_shared<PerRankBlockTransferEngine>(
        groups, false, DeviceHostCopyOptions{}, 8, 8, 1, 10000, nullptr, 0, 4, GetParam());
    ASSERT_FALSE(engine_->asyncEnabled());

    MemoryOperationRequestPB request;
    ASSERT_TRUE(BlockTransferRequestConverter::encodeTransfer(
        request,
        makeTransferTask({TransferDescriptor::hostToDevice(0, hosts_[0], targets_[0]),
                          TransferDescriptor::hostToDevice(1, hosts_[1], targets_[1])}),
        groups));
    EXPECT_EQ(request.copy_direction(), MemoryOperationRequestPB::H2D);
    // The ordinary GroupSet uses only its payload, even when the pool has room for a footer.
    hostData(1)[encoded_ - 1] ^= 1;
    std::vector<TransferDescriptor> decoded;
    ASSERT_TRUE(BlockTransferRequestConverter::decodeTransfer(request, decoded, groups));
    for (const bool reverse : {false, true}) {
        SCOPED_TRACE(reverse);
        if (reverse) {
            std::reverse(decoded.begin(), decoded.end());
        }
        auto failed = engine_->execute(makeTransferTask(decoded));
        failed->waitDone();
        EXPECT_FALSE(failed->success());
        EXPECT_EQ(failed->errorInfo().code(), ErrorCode::INVALID_PARAMS);

        expectUntouchedTargets();
    }

    // Each uniform batch still works on the same engine, which owns a CRC service.
    for (size_t index = 0; index < groups.size(); ++index) {
        auto recovered = engine_->execute(
            makeTransferTask({TransferDescriptor::hostToDevice(index, hosts_[index], targets_[index])}));
        recovered->waitDone();
        ASSERT_TRUE(recovered->success()) << recovered->errorInfo().ToString();
        EXPECT_EQ(readTarget(index), expected_[index]);
    }
}

TEST_P(PerRankBlockTransferEngineCrcTest, CorruptionDumpIncludesPhysicalLayoutAndKeepsFailureWhenIoIsUnavailable) {
    ASSERT_NO_FATAL_FAILURE(initialize(false, false, 3, 1, true));
    ASSERT_TRUE(execute(Tier::DEVICE, Tier::HOST)->success());
    hostData(0)[1] ^= 1;
    const auto           root = std::filesystem::path(temp_dir_->path) / "diagnostics";
    CrcTransferService   service({group_}, 3, 1, 17, std::make_shared<CrcDumpWriter>(17, root.string()), false);
    const HostBufferView host{hostData(0), payload_, host_->strideBytes()};
    auto                 descriptor = TransferDescriptor::hostToDevice(0, hosts_[0], targets_[0]);
    EXPECT_EQ(service.copy({host}, {descriptor}, {group_.get()}).copy_error, CacheCopyError::CRC_MISMATCH);

    expectScatteredTargets(1);
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
                              "\"output_written\": \"true\"",
                              "\"tile_copy_stage\": \"scatter_completed\"",
                              "\"gpu_snapshot_role\": \"checked_source\""})
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
    const auto         old_dump = dumps.front();
    CrcTransferService rotated_service(
        {group_}, 3, 1, 17, std::make_shared<CrcDumpWriter>(17, root.string(), 1), false);
    auto rotated = TransferDescriptor::hostToDevice(0, hosts_[0], targets_[0]);
    EXPECT_EQ(rotated_service.copy({host}, {rotated}, {group_.get()}).copy_error, CacheCopyError::CRC_MISMATCH);

    expectScatteredTargets(1);
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

    // An unavailable dump directory must preserve the CRC result after scatter
    // and release the slot for the next operation.
    const auto blocked = std::filesystem::path(temp_dir_->path) / "blocked_diagnostics";
    std::ofstream(blocked) << "not a directory";
    CrcTransferService blocked_service(
        {group_}, 3, 1, 17, std::make_shared<CrcDumpWriter>(17, blocked.string()), false);
    auto next = TransferDescriptor::hostToDevice(0, hosts_[0], targets_[0]);
    EXPECT_EQ(blocked_service.copy({host}, {next}, {group_.get()}).copy_error, CacheCopyError::CRC_MISMATCH);

    expectScatteredTargets(1);
    hostData(0)[1] ^= 1;
    auto healthy = TransferDescriptor::hostToDevice(0, hosts_[0], targets_[0]);
    EXPECT_TRUE(blocked_service.copy({host}, {healthy}, {group_.get()}).ok());

    EXPECT_EQ(readTarget(0), expected_[0]);
}

TEST_P(PerRankBlockTransferEngineCrcTest, SlotWriterRotatesCompleteDumpsWithinBatch) {
    ASSERT_NO_FATAL_FAILURE(initialize());
    ASSERT_TRUE(execute(Tier::DEVICE, Tier::HOST)->success());
    hostData(0)[1] ^= 1;
    hostData(1)[1] ^= 1;
    const auto         root = std::filesystem::path(temp_dir_->path) / "slot_diagnostics";
    CrcTransferService service({group_}, 3, 1, 17, std::make_shared<CrcDumpWriter>(17, root.string(), 1), false);
    std::vector<HostBufferView>     hosts;
    std::vector<TransferDescriptor> descriptors;
    for (size_t index = 0; index < hosts_.size(); ++index) {
        hosts.push_back({hostData(index), payload_, host_->strideBytes()});
        descriptors.push_back(TransferDescriptor::hostToDevice(0, hosts_[index], targets_[index]));
    }
    const std::vector<const GroupSet*> groups(hosts.size(), group_.get());
    const auto                         result = service.copy(hosts, descriptors, groups);
    EXPECT_EQ(result.copy_error, CacheCopyError::CRC_MISMATCH);
    ASSERT_EQ(result.block_errors.size(), 3u);
    EXPECT_EQ(result.block_errors[0], CacheCopyError::CRC_MISMATCH);
    EXPECT_EQ(result.block_errors[1], CacheCopyError::CRC_MISMATCH);
    EXPECT_EQ(result.block_errors[2], CacheCopyError::NOT_EXECUTED);
    expectScatteredTargets(targets_.size());
    // Each failure is dumped; the second completed dump replaces the first.
    expectDump(
        root / "rank_17", std::vector<uint8_t>(hostData(1), hostData(1) + encoded_), {"\"copy_item_index\": \"1\""});
    hostData(0)[1] ^= 1;
    hostData(1)[1] ^= 1;
    EXPECT_TRUE(service.copy(hosts, descriptors, groups).ok());
    for (size_t index = 0; index < targets_.size(); ++index)
        EXPECT_EQ(readTarget(index), expected_[index]);
}

TEST_P(PerRankBlockTransferEngineCrcTest, Dsv4Cp16ShardedSwaRoundtripPreservesOpaqueBytes) {
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
    engine_ = std::make_shared<PerRankBlockTransferEngine>(
        std::vector<GroupSetPtr>{group_}, false, DeviceHostCopyOptions{}, 128, 8, 4, 10000, nullptr, 0, 4, GetParam());
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

TEST_P(PerRankBlockTransferEngineCrcTest, PhysicalLayersWithReorderedTagsAndScalePresenceMatchIndependentCrc) {
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

TEST_P(PerRankBlockTransferEngineCrcTest, LastPhysicalLayerCorruptionRejectsScatteredBatch) {
    ASSERT_NO_FATAL_FAILURE(initialize(false, false, 3, 1, true));
    ASSERT_TRUE(execute(Tier::DEVICE, Tier::HOST)->success());
    // The last byte belongs to a scale buffer absent from this tag's logical spec.
    hostData(hosts_.size() - 1)[payload_ - 1] ^= 1;
    const auto context = execute(Tier::HOST, Tier::DEVICE);
    expectIntegrityFailure(context);
    EXPECT_EQ(context->result().block_errors,
              (std::vector<CacheCopyError>{
                  CacheCopyError::NOT_EXECUTED, CacheCopyError::NOT_EXECUTED, CacheCopyError::CRC_MISMATCH}));
    expectScatteredTargets(targets_.size());
}

TEST_P(PerRankBlockTransferEngineCrcTest, LocalRequestPreservesIndependentBatchResultsOnFailure) {
    ASSERT_NO_FATAL_FAILURE(initialize());
    engine_ = std::make_shared<PerRankBlockTransferEngine>(
        std::vector<GroupSetPtr>{group_}, false, DeviceHostCopyOptions{}, 8, 1, 1, 10000, nullptr, 0, 4, GetParam());
    BlockTransferDispatcher dispatcher(engine_, nullptr, 1);
    const auto              execute_request = [&](Tier from, Tier to) {
        std::vector<TransferDescriptor> descriptors;
        for (size_t item = 0; item < sources_.size(); ++item)
            descriptors.push_back(makeDescriptor(
                from, to, from == Tier::DEVICE ? sources_[item] : targets_[item], hosts_[item], disks_[item]));
        auto context = dispatcher.executePerRank(makeTransferTask(std::move(descriptors)));
        context->waitDone();
        return context;
    };
    ASSERT_TRUE(execute_request(Tier::DEVICE, Tier::HOST)->success());
    hostData(1)[payload_ - 1] ^= 1;
    const auto context = execute_request(Tier::HOST, Tier::DEVICE);
    expectIntegrityFailure(context);
    EXPECT_EQ(context->result().block_errors,
              (std::vector<CacheCopyError>{CacheCopyError::NONE, CacheCopyError::CRC_MISMATCH, CacheCopyError::NONE}));
    EXPECT_EQ(readTarget(0), expected_[0]);
    EXPECT_EQ(readTarget(1), std::vector<uint8_t>(hostData(1), hostData(1) + payload_));
    EXPECT_EQ(readTarget(2), expected_[2]);
}

TEST_P(PerRankBlockTransferEngineCrcTest, RpcWorkerUsesValidDeviceIndicesWithoutLocalAllocation) {
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
    expectScatteredTargets(targets_.size());
}

TEST_P(PerRankBlockTransferEngineCrcTest, InvalidDeviceIndicesAndMemberCountsNeverWriteTargets) {
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
        for (const auto& direction :
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

TEST_P(PerRankBlockTransferEngineCrcTest, LastBackingPayloadCorruptionRejectsScatteredBatch) {
    ASSERT_NO_FATAL_FAILURE(initialize());
    ASSERT_TRUE(execute(Tier::DEVICE, Tier::HOST)->success());
    hostData(2)[payload_ - 1] ^= 0x80;
    expectIntegrityFailure(execute(Tier::HOST, Tier::DEVICE));

    expectScatteredTargets(targets_.size());
}

TEST_P(PerRankBlockTransferEngineCrcTest, LastBackingFooterCorruptionRejectsScatteredBatch) {
    ASSERT_NO_FATAL_FAILURE(initialize());
    ASSERT_TRUE(execute(Tier::DEVICE, Tier::HOST)->success());
    hostData(2)[encoded_ - 1] ^= 1;
    expectIntegrityFailure(execute(Tier::HOST, Tier::DEVICE));
    expectScatteredTargets(targets_.size());
}

TEST_P(PerRankBlockTransferEngineCrcTest, EncodedAndDiskPaddingAreOutsideChecksum) {
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

TEST_P(PerRankBlockTransferEngineCrcTest, HostDiskPreservesCorruptRecordUntilDeviceLoad) {
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
        std::vector<uint8_t> raw(disk_->strideBytes());
        ASSERT_EQ(disk_->read(disks_[item], raw.data(), raw.size()), BlockIOStatus::OK);
        EXPECT_EQ(std::vector<uint8_t>(raw.begin(), raw.begin() + encoded_), item == 2 ? corrupt : sealed[item]);
        std::memset(hostData(item), 0xcc, host_->strideBytes());
    }
    ASSERT_TRUE(execute(Tier::DISK, Tier::HOST)->success());
    for (size_t item = 0; item < hosts_.size(); ++item) {
        EXPECT_EQ(std::vector<uint8_t>(hostData(item), hostData(item) + encoded_), item == 2 ? corrupt : sealed[item]);
    }
    EXPECT_FALSE(std::filesystem::exists(dumps));
    expectIntegrityFailure(execute(Tier::HOST, Tier::DEVICE));
    expectScatteredTargets(targets_.size());
    ASSERT_NO_FATAL_FAILURE(
        expectDump(dumps,
                   corrupt,
                   {"\"cpu_backing\": \"host_pool\"", "\"operation\": \"load\"", "\"direction\": \"HOST->DEVICE\""}));
}

TEST_P(PerRankBlockTransferEngineCrcTest, DiskCorruptionIsRejectedAfterDeviceScatter) {
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
    expectScatteredTargets(targets_.size());
    ASSERT_NO_FATAL_FAILURE(
        expectDump(host_dumps,
                   raw,
                   {"\"cpu_backing\": \"host_pool\"", "\"operation\": \"load\"", "\"direction\": \"HOST->DEVICE\""}));
    const auto device_dumps      = captureDumps("disk_to_device_dump");
    const auto read_bytes_before = disk_->readBytes();
    const auto failed            = execute(Tier::DISK, Tier::DEVICE);
    expectIntegrityFailure(failed);
    EXPECT_NE(failed->errorInfo().ToString().find(
                  "disk-to-device transfer failed, stage=staging-to-device, descriptor_range=[0,3): cache integrity "
                  "verification failed"),
              std::string::npos);
    EXPECT_NE(failed->errorInfo().ToString().find("descriptor_index=2"), std::string::npos);
    EXPECT_EQ(failed->result().block_errors,
              (std::vector<CacheCopyError>{
                  CacheCopyError::NOT_EXECUTED, CacheCopyError::NOT_EXECUTED, CacheCopyError::CRC_MISMATCH}));
    EXPECT_EQ(disk_->readBytes() - read_bytes_before, disks_.size() * disk_->strideBytes());

    expectScatteredTargets(targets_.size());
    ASSERT_NO_FATAL_FAILURE(expectDump(device_dumps,
                                       raw,
                                       {"\"cpu_backing\": \"disk_staging_buffer\"",
                                        "\"operation\": \"load\"",
                                        "\"direction\": \"DISK->DEVICE\"",
                                        "\"layout_tile_count\": \"8\"",
                                        "\"tile_count\": \"8\"",
                                        "\"cpu_captured\": \"true\"",
                                        "\"staging_captured\": \"true\"",
                                        "\"output_written\": \"true\"",
                                        "\"tile_copy_stage\": \"scatter_completed\"",
                                        "\"gpu_snapshot_role\": \"checked_source\"",
                                        "\"disk_file\": \"" + disk_->filePath() + "\"",
                                        "\"disk_slot\": \"" + std::to_string(disks_[2]) + "\"",
                                        "\"disk_offset\": \"" + std::to_string(disk_->blockOffset(disks_[2])) + "\""}));
}

TEST_P(PerRankBlockTransferEngineCrcTest, PageBoundaryPayloadUsesExtraPageAndDeviceDiskRoundTrips) {
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

TEST_P(PerRankBlockTransferEngineCrcTest, PayloadSizedHostCapacityCannotHoldFooter) {
    EXPECT_THROW(initialize(true, true), std::invalid_argument);
    ASSERT_NE(host_, nullptr);
    EXPECT_EQ(host_->strideBytes(), 4096u);
    EXPECT_EQ(engine_, nullptr);
}

TEST_P(PerRankBlockTransferEngineCrcTest, LargeBatchRoundTripRejectsScatteredCorruption) {
    ASSERT_NO_FATAL_FAILURE(initialize(false, false, 32, 256));
    ASSERT_EQ(hosts_.size(), 32u);
    ASSERT_GT(payload_, 16384u);
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
    // Corrupt payload positions and the stored footer. Every item is scattered,
    // but the failure result prevents the caller from adopting this batch.
    for (size_t offset : {size_t(0), payload_ / 2, payload_ - 1, encoded_ - 1}) {
        SCOPED_TRACE(offset);
        hostData(31)[offset] ^= 0x40;
        fillTargets(0xa5);
        expectIntegrityFailure(execute(Tier::HOST, Tier::DEVICE));
        expectScatteredTargets(targets_.size());
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

TEST_P(PerRankBlockTransferEngineCrcTest, LargeBatchHostDiskPreservesCorruptionUntilDeviceLoad) {
    ASSERT_NO_FATAL_FAILURE(initialize(false, false, 32, 256));
    ASSERT_TRUE(execute(Tier::DEVICE, Tier::HOST)->success());
    std::vector<std::vector<uint8_t>> records;
    for (size_t item = 0; item < hosts_.size(); ++item) {
        records.emplace_back(hostData(item), hostData(item) + encoded_);
    }
    ASSERT_TRUE(execute(Tier::HOST, Tier::DISK)->success());
    hostData(31)[payload_ / 2] ^= 0x80;
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
    expectScatteredTargets(targets_.size());
    expectIntegrityFailure(execute(Tier::DISK, Tier::DEVICE));
    expectScatteredTargets(targets_.size());
}

// DSV4 production shapes contending for 2/4/8 parallel service slots. Each
// slot owns private streams and staging, so concurrent store->load round-trips
// from eight threads must stay isolated: with fewer slots than threads the
// condition-variable wait path is exercised, and every host record, footer,
// and restored target is re-checked after the threads join. The full shape
// (62 tiles, 510,912 B) stays on the serial launch path; the prefill shape
// (85 tiles, 1,235,040 B) crosses the 4 MiB wave threshold at 4 items per
// batch, so both load paths run under slot contention.
TEST_P(PerRankBlockTransferEngineCrcTest, MultiSlotParallelDsv4ShapeRoundTripsStayIsolated) {
    constexpr size_t kItemsPerThread = 4;
    constexpr size_t kMaxWorkers     = 8;
    struct Shape {
        const char*                            name;
        std::vector<std::pair<size_t, size_t>> layers;
        size_t                                 tiles;
    };
    const auto dsv4Layers = [](bool full) {
        std::vector<std::pair<size_t, size_t>> layers;
        if (full) {
            // 21 scaled layers then 20 kv-only layers: 62 tiles, 510,912 B.
            layers.insert(layers.end(), 21, std::make_pair<size_t, size_t>(19008, 4224));
            layers.insert(layers.end(), 20, std::make_pair<size_t, size_t>(1152, 0));
        } else {
            // 21 scaled layers, 22 more 18,720 B kv layers, then 21 16,384 B kv
            // layers: 85 tiles, 1,235,040 B.
            layers.insert(layers.end(), 21, std::make_pair<size_t, size_t>(18720, 4096));
            layers.insert(layers.end(), 22, std::make_pair<size_t, size_t>(18720, 0));
            layers.insert(layers.end(), 21, std::make_pair<size_t, size_t>(16384, 0));
        }
        return layers;
    };
    const Shape shapes[] = {{"dsv4_full", dsv4Layers(true), 62}, {"dsv4_prefill_cp4", dsv4Layers(false), 85}};

    TempDirGuard slot_dumps("crc_parallel_slots");
    auto         dump_writer = std::make_shared<CrcDumpWriter>(0, slot_dumps.path, 1);
    for (const auto& shape : shapes) {
        payload_ = 0;
        for (const auto& [kv, scale] : shape.layers) {
            payload_ += kv + scale;
        }
        encoded_ = ((payload_ + 4 + 15) / 16) * 16;
        layers_  = shape.layers.size();
        devices_ = {
            makeTestDevicePool(shape.layers, 2 * kItemsPerThread * kMaxWorkers, "crc_slot_" + std::string(shape.name))};
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        host_ = makeHostPool(payload_, kItemsPerThread * kMaxWorkers, true);
        std::vector<int> layer_ids(layers_);
        std::iota(layer_ids.begin(), layer_ids.end(), 0);
        // The logical spec describes only the first layer; the pool's physical
        // per-layer geometry is authoritative, as in MTP-style production pools.
        group_ = makeTestGroupSet(0,
                                  makeTestTopology({makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL),
                                                                      layer_ids,
                                                                      shape.layers.front().first,
                                                                      shape.layers.front().second)}),
                                  {"group0"},
                                  devices_,
                                  host_,
                                  nullptr,
                                  /*enable_crc=*/true,
                                  /*physical_payload_bytes=*/payload_);
        ASSERT_TRUE(group_->crcEnabled());
        ASSERT_EQ(group_->storageBytes(), encoded_);
        ASSERT_EQ(getDeviceHostCopyTiles(*group_, std::vector<BlockIdxType>(group_->devicePools().size(), 0)).size(),
                  shape.tiles);

        sources_.clear();
        targets_.clear();
        hosts_.clear();
        expected_.clear();
        for (size_t item = 0; item < kItemsPerThread * kMaxWorkers; ++item) {
            sources_.push_back({allocate(devices_[0])});
            targets_.push_back({allocate(devices_[0])});
            hosts_.push_back(allocate(host_));
            std::vector<uint8_t> expected;
            for (size_t layer = 0; layer < layers_; ++layer) {
                const auto buffers = devices_[0]->convertIndexToBuffer(static_cast<int>(layer), sources_[item][0]);
                for (size_t kind = 0; kind < buffers.size(); ++kind) {
                    std::vector<uint8_t> bytes(buffers[kind].size_bytes);
                    for (size_t i = 0; i < bytes.size(); ++i) {
                        bytes[i] = static_cast<uint8_t>((i * 29 + item * 71 + layer * 17 + kind * 13) & 255);
                    }
                    ASSERT_EQ(cudaMemcpy(buffers[kind].addr, bytes.data(), bytes.size(), cudaMemcpyHostToDevice),
                              cudaSuccess);
                    expected.insert(expected.end(), bytes.begin(), bytes.end());
                }
            }
            expected_.push_back(std::move(expected));
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        for (size_t workers : {2, 4, 8}) {
            SCOPED_TRACE(::testing::Message() << shape.name << " slots=" << workers);
            CrcTransferService store_service({group_}, kItemsPerThread, workers, 0, dump_writer, false);
            CrcTransferService load_service({group_}, kItemsPerThread, workers, 0, dump_writer, false);
            fillTargets(0xa5);
            std::vector<std::exception_ptr> failures(kMaxWorkers);
            std::mutex                      start_mutex;
            std::condition_variable         start_condition;
            size_t                          ready  = 0;
            bool                            start  = false;
            auto                            worker = [&](size_t thread) {
                {
                    std::unique_lock<std::mutex> lock(start_mutex);
                    ++ready;
                    start_condition.notify_all();
                    start_condition.wait(lock, [&] { return start; });
                }
                try {
                    if (cudaSetDevice(0) != cudaSuccess) {
                        throw std::runtime_error("cudaSetDevice failed");
                    }
                    std::vector<HostBufferView>     thread_hosts;
                    std::vector<TransferDescriptor> thread_descriptors;
                    std::vector<const GroupSet*>    thread_groups;
                    for (size_t offset = 0; offset < kItemsPerThread; ++offset) {
                        const size_t item = thread * kItemsPerThread + offset;
                        thread_hosts.push_back({hostData(item), payload_, host_->strideBytes()});
                        thread_descriptors.push_back(TransferDescriptor::deviceToHost(0, sources_[item], hosts_[item]));
                        thread_groups.push_back(group_.get());
                    }
                    if (!store_service.copy(thread_hosts, thread_descriptors, thread_groups).ok()) {
                        throw std::runtime_error("concurrent store failed");
                    }
                    for (size_t offset = 0; offset < kItemsPerThread; ++offset) {
                        const size_t item = thread * kItemsPerThread + offset;
                        thread_descriptors[offset] = TransferDescriptor::hostToDevice(0, hosts_[item], targets_[item]);
                    }
                    if (!load_service.copy(thread_hosts, thread_descriptors, thread_groups).ok()) {
                        throw std::runtime_error("concurrent load failed");
                    }
                    // Each caller owns disjoint targets. A mismatch must release
                    // its slot and preserve per-item status under contention.
                    const size_t damaged = thread % kItemsPerThread;
                    auto*        damaged_host = static_cast<uint8_t*>(thread_hosts[damaged].base);
                    damaged_host[thread] ^= 1;
                    const auto rejected = load_service.copy(thread_hosts, thread_descriptors, thread_groups);
                    if (rejected.copy_error != CacheCopyError::CRC_MISMATCH
                        || rejected.block_errors.size() != kItemsPerThread) {
                        throw std::runtime_error("concurrent mismatch lost its result");
                    }
                    for (size_t offset = 0; offset < kItemsPerThread; ++offset) {
                        const auto expected_error =
                            offset == damaged ? CacheCopyError::CRC_MISMATCH : CacheCopyError::NOT_EXECUTED;
                        if (rejected.block_errors[offset] != expected_error) {
                            throw std::runtime_error("concurrent mismatch changed another item's result");
                        }
                    }
                    damaged_host[thread] ^= 1;
                    if (!load_service.copy(thread_hosts, thread_descriptors, thread_groups).ok()) {
                        throw std::runtime_error("concurrent recovery failed");
                    }
                } catch (...) {
                    failures[thread] = std::current_exception();
                }
            };
            std::vector<std::thread> threads;
            threads.reserve(kMaxWorkers);
            for (size_t thread = 0; thread < kMaxWorkers; ++thread) {
                threads.emplace_back(worker, thread);
            }
            {
                std::unique_lock<std::mutex> lock(start_mutex);
                start_condition.wait(lock, [&] { return ready == kMaxWorkers; });
                start = true;
            }
            start_condition.notify_all();
            for (auto& thread : threads) {
                thread.join();
            }
            for (const auto& failure : failures) {
                if (failure) {
                    std::rethrow_exception(failure);
                }
            }
            for (size_t item = 0; item < kItemsPerThread * kMaxWorkers; ++item) {
                EXPECT_EQ(std::vector<uint8_t>(hostData(item), hostData(item) + payload_), expected_[item]);
                uint32_t footer = 0;
                std::memcpy(&footer, hostData(item) + encoded_ - sizeof(footer), sizeof(footer));
                EXPECT_EQ(footer, softwareCrc32c(expected_[item].data(), expected_[item].size()));
                EXPECT_EQ(readTarget(item), expected_[item]);
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(ExecutionModes,
                         PerRankBlockTransferEngineCrcTest,
                         ::testing::Values(true),
                         [](const auto& info) { return info.param ? "WorkLoop" : "Legacy"; });

}  // namespace
}  // namespace rtp_llm

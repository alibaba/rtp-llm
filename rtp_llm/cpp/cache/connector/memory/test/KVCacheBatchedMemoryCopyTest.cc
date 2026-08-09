// Copyright (c) RTP-LLM

#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <torch/torch.h>
#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"

namespace rtp_llm {

void execBatchCopy(const BatchCopyParams&) {}

}  // namespace rtp_llm

namespace rtp_llm::test {
namespace {

BlockDependency rootDep(uint32_t ordinal = 0) {
    BlockDependency dep;
    dep.ordinal = ordinal;
    return dep;
}

void addTaggedGpuBlocks(MemoryOperationRequestPB::CopyItem&                        item,
                        const std::vector<KVCacheMemoryConnector::LayerGroupSlot>& slots,
                        const std::vector<BlockIdxType>&                           block_ids) {
    ASSERT_EQ(block_ids.size(), slots.size());
    for (size_t i = 0; i < slots.size(); ++i) {
        auto* tagged_block = item.add_tagged_gpu_blocks();
        tagged_block->set_layer_id(slots[i].layer_id);
        tagged_block->set_tag(slots[i].tag);
        tagged_block->set_block_id(block_ids[i]);
    }
}

TEST(KVCacheMemoryProtocolTest, TaggedBlocksAreReorderedByLocalLayerAndTag) {
    std::vector<KVCacheMemoryConnector::LayerGroupSlot> slots = {
        {0, "linear", 16, CacheGroupType::LINEAR, CacheBlockKind::STATE_SWA_KV},
        {0, "full", 32, CacheGroupType::FULL, CacheBlockKind::COMPRESSED_KV},
    };
    MemoryOperationRequestPB::CopyItem item;
    auto*                              full = item.add_tagged_gpu_blocks();
    full->set_layer_id(0);
    full->set_tag("full");
    full->set_block_id(7);
    auto* linear = item.add_tagged_gpu_blocks();
    linear->set_layer_id(0);
    linear->set_tag("linear");
    linear->set_block_id(3);

    const auto gpu_blocks = KVCacheMemoryConnector::normalizeCopyItemGpuBlocks(item, slots);
    ASSERT_EQ(gpu_blocks.size(), 2u);
    EXPECT_EQ(gpu_blocks[0], 3);
    EXPECT_EQ(gpu_blocks[1], 7);
}

TEST(KVCacheMemoryProtocolTest, TaglessBlocksAreAlwaysRejected) {
    std::vector<KVCacheMemoryConnector::LayerGroupSlot> slots = {
        {0, "linear", 16, CacheGroupType::LINEAR, CacheBlockKind::STATE_SWA_KV},
        {0, "full", 32, CacheGroupType::FULL, CacheBlockKind::COMPRESSED_KV},
    };
    MemoryOperationRequestPB::CopyItem item;

    EXPECT_ANY_THROW(KVCacheMemoryConnector::normalizeCopyItemGpuBlocks(item, slots));
}

CacheConfig makeCompactDsv4TypedMemoryCopyConfig(bool use_flash) {
    CacheConfig        config;
    constexpr auto     dtype                    = rtp_llm::DataType::TYPE_UINT8;
    constexpr uint32_t block_num                = 512;
    config.layer_num                            = use_flash ? 43 : 61;
    config.seq_size_per_block                   = 256;
    constexpr size_t               kDsv4PoolNum = 7;
    const std::vector<std::string> group_tags   = {
        "csa_kv", "hca_kv", "indexer_kv", "indexer_state", "csa_state", "hca_state", "swa_kv"};
    std::vector<CacheGroupType> group_types;
    group_types.reserve(kDsv4PoolNum);
    group_types.push_back(CacheGroupType::FULL);
    group_types.push_back(CacheGroupType::FULL);
    group_types.push_back(CacheGroupType::FULL);
    group_types.push_back(CacheGroupType::SWA);
    group_types.push_back(CacheGroupType::SWA);
    group_types.push_back(CacheGroupType::SWA);
    group_types.push_back(CacheGroupType::SWA);
    std::unordered_map<std::string, CacheGroupPolicy> group_policies;
    for (size_t i = 0; i < group_tags.size(); ++i) {
        group_policies.emplace(group_tags[i], defaultCacheGroupPolicy(group_types[i]));
    }
    auto& hca_state_policy                = group_policies.at("hca_state");
    hca_state_policy.enable_prefix_reuse  = false;
    hca_state_policy.active_tail_blocks   = 1;
    hca_state_policy.validate_tail_blocks = false;
    for (std::string_view tag : {"indexer_state", "csa_state", "hca_state", "swa_kv"}) {
        group_policies.at(std::string(tag)).evict_policy = CacheEvictPolicy::INDEPENDENT;
    }
    for (std::string_view tag : {"indexer_state", "csa_state", "swa_kv"}) {
        group_policies.at(std::string(tag)).enable_prefix_reuse = true;
    }
    const std::vector<size_t>     group_kv_block_stride_bytes = {64, 16, 32, 48, 80, 40, 96};
    const std::vector<size_t>     group_kv_scale_stride_bytes(kDsv4PoolNum, 0);
    const std::vector<uint32_t>   group_block_nums(kDsv4PoolNum, block_num);
    std::vector<std::vector<int>> layers_by_group(kDsv4PoolNum);

    auto make_spec = [&](size_t gid) -> KVCacheSpecPtr {
        return makeResolvedOpaqueSpec(group_types[gid] != CacheGroupType::FULL,
                                      group_tags[gid],
                                      dtype,
                                      group_kv_block_stride_bytes[gid],
                                      static_cast<uint32_t>(config.seq_size_per_block));
    };

    auto add_tag = [&](size_t layer, const std::string& tag, int gid) {
        (void)tag;
        layers_by_group[static_cast<size_t>(gid)].push_back(static_cast<int>(layer));
    };

    for (size_t layer = 0; layer < config.layer_num; ++layer) {
        const bool is_csa = layer >= 2 && layer % 2 == 0;
        const bool is_hca = use_flash ? (layer >= 2 && layer % 2 == 1) : (!is_csa);
        if (is_csa) {
            add_tag(layer, "csa_kv", 0);
            add_tag(layer, "indexer_kv", 2);
            add_tag(layer, "indexer_state", 3);
            add_tag(layer, "csa_state", 4);
        } else if (is_hca) {
            add_tag(layer, "hca_kv", 1);
            add_tag(layer, "hca_state", 5);
        }
        add_tag(layer, "swa_kv", 6);
    }

    std::vector<GroupBase> groups;
    groups.reserve(kDsv4PoolNum);
    for (size_t gid = 0; gid < kDsv4PoolNum; ++gid) {
        auto group                      = makeTestGroupForConfig(config,
                                            make_spec(gid),
                                            std::move(layers_by_group[gid]),
                                            group_types[gid],
                                            group_tags[gid],
                                            group_policies.at(group_tags[gid]));
        group.policy.explicit_block_num = group_block_nums[gid];
        group.kv_block_stride_bytes     = group_kv_block_stride_bytes[gid];
        group.kv_scale_stride_bytes     = group_kv_scale_stride_bytes[gid];
        groups.push_back(std::move(group));
    }
    setTestTopology(config, std::move(groups));
    return config;
}

CacheConfig makeCanonicalCompositeMemoryCopyConfig() {
    CacheConfig config;
    config.layer_num          = 3;
    config.seq_size_per_block = 1;

    auto make_group = [&](std::string tag, size_t stride, std::vector<int> layers, CacheGroupType group_type) {
        RTP_LLM_CHECK_WITH_INFO(stride % 2 == 0, "test MHA stride must contain equally-sized K/V payloads");
        KVCacheSpecPtr spec  = makeResolvedMhaSpec(TYPE_UINT8, 1, stride / 2, 1, tag);
        auto           group = makeTestGroupForConfig(config, std::move(spec), std::move(layers), group_type, tag);
        group.policy.explicit_block_num = 8;
        return group;
    };

    std::vector<GroupBase> groups;
    groups.push_back(make_group("zeta", 32, {0}, CacheGroupType::FULL));
    groups.push_back(make_group("beta", 24, {1, 2}, CacheGroupType::FULL));
    groups.push_back(make_group("alpha", 16, {0, 2}, CacheGroupType::FULL));
    setTestTopology(config, std::move(groups));
    return config;
}

void initResourceGroupsForConfig(KVCacheResource& resource, const CacheConfig& config) {
    resource.initGroups(config.topologyPtr());
}

void setGroupStridesForConfig(CacheConfig& config, size_t kv_block_stride_bytes, size_t kv_scale_stride_bytes) {
    const auto             topology_groups = config.topology().groups();
    std::vector<GroupBase> groups(topology_groups.begin(), topology_groups.end());
    for (auto& group : groups) {
        group.kv_block_stride_bytes = kv_block_stride_bytes;
        group.kv_scale_stride_bytes = kv_scale_stride_bytes;
    }
    config.setTopology(std::move(groups), config.topology().layers());
}

void setBlockBytes(const BlockInfo& b, size_t byte_offset, size_t byte_len, char c) {
    ASSERT_NE(b.addr, nullptr);
    ASSERT_LE(byte_offset + byte_len, b.size_bytes);
    auto* addr = static_cast<char*>(b.addr) + byte_offset;
    if (b.is_cuda) {
        const auto rc = cudaMemset(addr, c, byte_len);
        ASSERT_EQ(rc, cudaSuccess) << cudaGetErrorString(rc);
        const auto sync_rc = cudaDeviceSynchronize();
        ASSERT_EQ(sync_rc, cudaSuccess) << cudaGetErrorString(sync_rc);
    } else {
        memset(addr, c, byte_len);
    }
}

void verifyBlockBytesEq(const BlockInfo& b, size_t byte_offset, size_t byte_len, char expected) {
    ASSERT_NE(b.addr, nullptr);
    ASSERT_LE(byte_offset + byte_len, b.size_bytes);
    auto* addr = static_cast<const char*>(b.addr) + byte_offset;

    std::vector<unsigned char> data(byte_len, 0);
    if (b.is_cuda) {
        const auto rc = cudaMemcpy(data.data(), addr, byte_len, cudaMemcpyDeviceToHost);
        ASSERT_EQ(rc, cudaSuccess) << cudaGetErrorString(rc);
    } else {
        memcpy(data.data(), addr, byte_len);
    }
    size_t mismatch = 0;
    for (; mismatch < byte_len; ++mismatch) {
        if (data[mismatch] != static_cast<unsigned char>(expected)) {
            break;
        }
    }
    ASSERT_EQ(mismatch, byte_len) << "mismatch at byte offset " << mismatch << " expect '" << expected << "' got 0x"
                                  << std::hex << static_cast<int>(data[mismatch]) << std::dec;
}

void setBlockInfosContent(const std::vector<BlockInfo>& infos, char c) {
    for (const auto& b : infos) {
        if (b.addr && b.size_bytes > 0) {
            setBlockBytes(b, /*byte_offset=*/0, b.size_bytes, c);
        }
    }
}

void verifyBlockInfosContent(const std::vector<BlockInfo>& infos, char c) {
    for (const auto& b : infos) {
        if (b.addr && b.size_bytes > 0) {
            verifyBlockBytesEq(b, /*byte_offset=*/0, b.size_bytes, c);
        }
    }
}

class FakeTypedKVCacheAllocator: public KVCacheAllocator {
public:
    explicit FakeTypedKVCacheAllocator(const CacheConfig&    config,
                                       size_t                payload_gap_bytes = 0,
                                       std::set<std::string> host_groups       = {}):
        KVCacheAllocator(config, AllocationType::DEVICE),
        host_groups_(std::move(host_groups)),
        payload_gap_bytes_(payload_gap_bytes) {
        const auto cuda_options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
        const auto host_options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
        for (int layer = 0; layer < static_cast<int>(config.totalLayerNum()); ++layer) {
            for (const auto& group_ref : config.groupsForLayer(layer)) {
                const auto&  group  = group_ref.get();
                const size_t stride = group.kv_block_stride_bytes + group.kv_scale_stride_bytes;
                if (stride == 0) {
                    continue;
                }
                const bool host_group = host_groups_.count(group.tag) > 0;
                auto       tensor     = torch::empty(
                    {static_cast<int64_t>(config.blockNumForGroup(group.tag)), static_cast<int64_t>(stride)},
                    host_group ? host_options : cuda_options);
                if (host_group) {
                    tensor = tensor.pin_memory();
                }
                tensors_[key(layer, group.tag)] = std::move(tensor);
                strides_[key(layer, group.tag)] = stride;
            }
        }
    }

    void free(const FreeInfo&) override {}
    void insertIntoCache(const InsertInfo&) override {}

    BlockAddrInfo convertIndexToAddr(int layer_id, const std::string& tag, int block_id) const override {
        const auto buffers = convertIndexToBuffer(layer_id, tag, block_id);
        return buffers.empty() ? BlockAddrInfo{} : BlockAddrInfo{buffers[0].addr, nullptr};
    }

    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, const std::string& tag, int block_id) const override {
        const auto k         = key(layer_id, tag);
        const auto tensor_it = tensors_.find(k);
        const auto stride_it = strides_.find(k);
        if (tensor_it == tensors_.end() || stride_it == strides_.end() || block_id < 0
            || static_cast<uint32_t>(block_id) >= config_.blockNumForGroup(tag)) {
            return {};
        }
        const auto& tensor       = tensor_it->second;
        const auto  stride       = stride_it->second;
        auto*       addr         = static_cast<char*>(tensor.data_ptr()) + static_cast<size_t>(block_id) * stride;
        const auto  payload_size = payload_gap_bytes_ < stride ? stride - payload_gap_bytes_ : stride;
        return {BlockInfo{
            /*is_cuda=*/tensor.is_cuda(),
            /*device_index=*/tensor.is_cuda() ? static_cast<int32_t>(tensor.get_device()) : -1,
            /*scalar_type=*/static_cast<int32_t>(tensor.scalar_type()),
            /*addr=*/addr,
            /*size_bytes=*/payload_size,
        }};
    }

    std::vector<BlockInfo> convertIndexToBuffer(
        int layer_id, const std::string& tag, int block_id, int partition_count, int partition_id) const override {
        return convertIndexToBuffer(layer_id, tag, block_id);
    }

    std::shared_ptr<KVCacheResource> incrKVCacheRef(const KVCacheResource&, const CacheKeysType&, bool) override {
        return nullptr;
    }

    GroupedCacheLayerLayout allLayerCacheBase() const override {
        return {};
    }

    bool updateKVBlock(const BatchKVCacheResourcePtr&,
                       const std::vector<int>&,
                       bool,
                       std::vector<GroupBlockIdPair>&) override {
        return false;
    }

    int seqSizePerBlock() const override {
        return static_cast<int>(config_.seq_size_per_block);
    }

    BlockPoolPtr getBlockPool(std::string_view) const override {
        return nullptr;
    }

    int singleBatchNeedBlocks(const BatchKVCacheResourcePtr&, int, int) const override {
        return 0;
    }

    int estimatePeakNeedBlocks(const KVCacheResource&, int, int, int, bool) const override {
        return 0;
    }

    int estimateInitialBatchPeakNeedBlocks(int, int, int, int, bool, int) const override {
        return 0;
    }

private:
    static std::pair<int, std::string> key(int layer_id, std::string_view tag) {
        return {layer_id, std::string(tag)};
    }

    bool doInit() override {
        return true;
    }

    MallocResult incrMalloc(const MallocInfo&) override {
        return {false, 0};
    }

    MallocResult initMallocForCommonLen(const MallocInfo&) override {
        return {false, 0};
    }

    void decrKVCacheRef(const KVCacheResource&, bool) override {}

    std::map<std::pair<int, std::string>, torch::Tensor> tensors_;
    std::map<std::pair<int, std::string>, size_t>        strides_;
    std::set<std::string>                                host_groups_;
    size_t                                               payload_gap_bytes_ = 0;
};

class FakeDiskBlockIO: public IDiskBlockIO {
public:
    bool openAndPreallocate(const std::string&, size_t bytes, bool) override {
        data_.assign(bytes, 0);
        return true;
    }

    bool read(uint64_t offset, void* dst, size_t bytes) override {
        if (offset + bytes > data_.size()) {
            return false;
        }
        std::memcpy(dst, data_.data() + offset, bytes);
        return true;
    }

    bool write(uint64_t offset, const void* src, size_t bytes) override {
        if (offset + bytes > data_.size()) {
            return false;
        }
        std::memcpy(data_.data() + offset, src, bytes);
        return true;
    }

    void close() override {}

    std::string debugString() const override {
        return "fake-disk-block-io";
    }

private:
    std::vector<unsigned char> data_;
};

}  // namespace

TEST(KVCacheBatchedMemoryCopyTest, CanonicalCompositeSlotsAreSortedAndSizedBySlotSum) {
    auto config = makeCanonicalCompositeMemoryCopyConfig();

    KVCacheConfig kv_config;
    kv_config.memory_cache_size_mb         = 1;
    kv_config.memory_cache_sync_timeout_ms = 1000;

    auto connector = std::make_shared<KVCacheMemoryConnector>(
        config, kv_config, std::shared_ptr<KVCacheAllocator>(), std::vector<std::string>{"127.0.0.1:1"});
    const auto& slots = connector->layerGroupSlots();

    ASSERT_EQ(slots.size(), 5u);
    EXPECT_EQ(slots[0].layer_id, 0);
    EXPECT_EQ(slots[0].tag, "alpha");
    EXPECT_EQ(slots[0].stride_bytes, 16u);
    EXPECT_EQ(slots[1].layer_id, 0);
    EXPECT_EQ(slots[1].tag, "zeta");
    EXPECT_EQ(slots[1].stride_bytes, 32u);
    EXPECT_EQ(slots[2].layer_id, 1);
    EXPECT_EQ(slots[2].tag, "beta");
    EXPECT_EQ(slots[2].stride_bytes, 24u);
    EXPECT_EQ(slots[3].layer_id, 2);
    EXPECT_EQ(slots[3].tag, "alpha");
    EXPECT_EQ(slots[3].stride_bytes, 16u);
    EXPECT_EQ(slots[4].layer_id, 2);
    EXPECT_EQ(slots[4].tag, "beta");
    EXPECT_EQ(slots[4].stride_bytes, 24u);
    EXPECT_EQ(connector->memoryCacheBlockSizeBytes(), 112u);
}

TEST(KVCacheBatchedMemoryCopyTest, CanonicalCompositeMemoryRoundTripPreservesNullSlotOffset) {
    auto config = makeCanonicalCompositeMemoryCopyConfig();

    KVCacheConfig kv_config;
    kv_config.memory_cache_size_mb         = 1;
    kv_config.memory_cache_sync_timeout_ms = 1000;

    auto allocator = std::make_shared<FakeTypedKVCacheAllocator>(
        config, /*payload_gap_bytes=*/0, std::set<std::string>{"alpha", "beta", "zeta"});
    auto connector =
        std::make_shared<KVCacheMemoryConnector>(config, kv_config, allocator, std::vector<std::string>{"127.0.0.1:1"});
    ASSERT_TRUE(connector->init());
    const auto& slots = connector->layerGroupSlots();
    ASSERT_EQ(connector->memoryCacheBlockSizeBytes(), 112u);

    auto mem_blocks = connector->block_pool_->malloc(1);
    ASSERT_EQ(mem_blocks.size(), 1u);
    const auto mem_block = static_cast<BlockIdxType>(mem_blocks[0]);
    const auto mem_bufs  = connector->block_pool_->convertIndexToBuffer(0, mem_block);
    ASSERT_EQ(mem_bufs.size(), 1u);
    const auto& mem_buffer = mem_bufs[0];
    setBlockBytes(mem_buffer, 0, connector->memoryCacheBlockSizeBytes(), 'N');

    std::vector<BlockIdxType> gpu_blocks{1, NULL_BLOCK_IDX, 3, 4, 5};
    for (size_t i = 0; i < slots.size(); ++i) {
        if (isNullBlockIdx(gpu_blocks[i])) {
            continue;
        }
        setBlockInfosContent(allocator->convertIndexToBuffer(slots[i].layer_id, slots[i].tag, gpu_blocks[i]),
                             static_cast<char>('a' + i));
    }

    MemoryOperationRequestPB d2h;
    d2h.set_copy_direction(MemoryOperationRequestPB::D2H);
    auto* d2h_item = d2h.add_copy_items();
    d2h_item->set_mem_block(mem_block);
    d2h_item->set_is_complete(true);
    d2h_item->set_cache_block_kind(MemoryOperationRequestPB::COMPLETE_KV);
    addTaggedGpuBlocks(*d2h_item, slots, gpu_blocks);
    MemoryOperationResponsePB d2h_response;
    ASSERT_TRUE(connector->copyCache(d2h, d2h_response));
    ASSERT_TRUE(d2h_response.success());

    size_t byte_offset = 0;
    for (size_t i = 0; i < slots.size(); ++i) {
        verifyBlockBytesEq(mem_buffer,
                           byte_offset,
                           slots[i].stride_bytes,
                           isNullBlockIdx(gpu_blocks[i]) ? 'N' : static_cast<char>('a' + i));
        byte_offset += slots[i].stride_bytes;
    }

    for (size_t i = 0; i < slots.size(); ++i) {
        if (isNullBlockIdx(gpu_blocks[i])) {
            continue;
        }
        setBlockInfosContent(allocator->convertIndexToBuffer(slots[i].layer_id, slots[i].tag, gpu_blocks[i]), '0');
    }
    MemoryOperationRequestPB h2d;
    h2d.set_copy_direction(MemoryOperationRequestPB::H2D);
    auto* h2d_item = h2d.add_copy_items();
    h2d_item->set_mem_block(mem_block);
    h2d_item->set_is_complete(true);
    h2d_item->set_cache_block_kind(MemoryOperationRequestPB::COMPLETE_KV);
    addTaggedGpuBlocks(*h2d_item, slots, gpu_blocks);
    MemoryOperationResponsePB h2d_response;
    ASSERT_TRUE(connector->copyCache(h2d, h2d_response));
    ASSERT_TRUE(h2d_response.success());
    for (size_t i = 0; i < slots.size(); ++i) {
        if (isNullBlockIdx(gpu_blocks[i])) {
            continue;
        }
        verifyBlockInfosContent(allocator->convertIndexToBuffer(slots[i].layer_id, slots[i].tag, gpu_blocks[i]),
                                static_cast<char>('a' + i));
    }
}

TEST(KVCacheBatchedMemoryCopyTest, CanonicalCompositeSlotsRoundTripThroughInjectedDiskPool) {
    auto config = makeCanonicalCompositeMemoryCopyConfig();

    KVCacheConfig kv_config;
    kv_config.memory_cache_size_mb         = 1;
    kv_config.memory_cache_sync_timeout_ms = 1000;

    auto allocator = std::make_shared<FakeTypedKVCacheAllocator>(
        config, /*payload_gap_bytes=*/0, std::set<std::string>{"alpha", "beta", "zeta"});
    auto connector =
        std::make_shared<KVCacheMemoryConnector>(config, kv_config, allocator, std::vector<std::string>{"127.0.0.1:1"});
    ASSERT_TRUE(connector->init());
    const auto& slots = connector->layerGroupSlots();

    // This test isolates composite-slot copying with a fake IO-backed pool. Production disk-pool construction and
    // mount/Posix IO behavior are covered separately by DiskBlockPoolTest.
    DiskBlockPoolConfig disk_config;
    disk_config.work_dir           = "/unused/fake-disk";
    disk_config.disk_size_bytes    = 4096;
    disk_config.block_size_bytes   = connector->memoryCacheBlockSizeBytes();
    disk_config.buffered_io        = true;
    disk_config.pool_kind          = CacheBlockKind::COMPLETE;
    connector->complete_disk_pool_ = std::make_shared<DiskBlockPool>(disk_config, std::make_unique<FakeDiskBlockIO>());
    ASSERT_TRUE(connector->complete_disk_pool_->init());
    ASSERT_EQ(connector->complete_disk_pool_->blockSizeBytes(), 112u);
    const auto disk_slot = connector->complete_disk_pool_->malloc();
    ASSERT_TRUE(disk_slot.has_value());

    const std::vector<BlockIdxType> gpu_blocks{1, 2, 3, 4, 5};
    for (size_t i = 0; i < slots.size(); ++i) {
        setBlockInfosContent(allocator->convertIndexToBuffer(slots[i].layer_id, slots[i].tag, gpu_blocks[i]),
                             static_cast<char>('a' + i));
    }

    MemoryOperationRequestPB d2h;
    d2h.set_copy_direction(MemoryOperationRequestPB::D2H);
    auto* d2h_item = d2h.add_copy_items();
    d2h_item->set_mem_block(NULL_BLOCK_IDX);
    d2h_item->set_backing_type(MemoryOperationRequestPB::DISK);
    d2h_item->set_disk_slot(*disk_slot);
    d2h_item->set_is_complete(true);
    d2h_item->set_cache_block_kind(MemoryOperationRequestPB::COMPLETE_KV);
    addTaggedGpuBlocks(*d2h_item, slots, gpu_blocks);
    MemoryOperationResponsePB d2h_response;
    ASSERT_TRUE(connector->copyCache(d2h, d2h_response));
    ASSERT_TRUE(d2h_response.success());

    for (size_t i = 0; i < slots.size(); ++i) {
        setBlockInfosContent(allocator->convertIndexToBuffer(slots[i].layer_id, slots[i].tag, gpu_blocks[i]), '0');
    }

    MemoryOperationRequestPB h2d;
    h2d.set_copy_direction(MemoryOperationRequestPB::H2D);
    auto* h2d_item = h2d.add_copy_items();
    h2d_item->set_mem_block(NULL_BLOCK_IDX);
    h2d_item->set_backing_type(MemoryOperationRequestPB::DISK);
    h2d_item->set_disk_slot(*disk_slot);
    h2d_item->set_is_complete(true);
    h2d_item->set_cache_block_kind(MemoryOperationRequestPB::COMPLETE_KV);
    addTaggedGpuBlocks(*h2d_item, slots, gpu_blocks);
    MemoryOperationResponsePB h2d_response;
    ASSERT_TRUE(connector->copyCache(h2d, h2d_response));
    ASSERT_TRUE(h2d_response.success());
    for (size_t i = 0; i < slots.size(); ++i) {
        verifyBlockInfosContent(allocator->convertIndexToBuffer(slots[i].layer_id, slots[i].tag, gpu_blocks[i]),
                                static_cast<char>('a' + i));
    }
}

TEST(KVCacheBatchedMemoryCopyTest, PrefixTreeKindRequiredUsesRuntimeNullSlots) {
    auto config = makeCompactDsv4TypedMemoryCopyConfig(/*use_flash=*/true);

    KVCacheConfig kv_config;
    kv_config.memory_cache_size_mb         = 64;
    kv_config.memory_cache_sync_timeout_ms = 1000;

    std::vector<std::string> server_addrs = {"127.0.0.1:1"};
    auto                     connector =
        std::make_shared<KVCacheMemoryConnector>(config, kv_config, std::shared_ptr<KVCacheAllocator>(), server_addrs);
    const auto slots = connector->layerGroupSlots();
    ASSERT_TRUE(connector->supportsTypedPrefixCacheLayout(slots));

    KVCacheResource resource;
    initResourceGroupsForConfig(resource, config);
    resource.resizeBlocks(/*reserver_blocks=*/2, NULL_BLOCK_IDX);

    resource.mutableBlockIds("csa_kv").setAt(0, 10);
    resource.mutableBlockIds("hca_kv").setAt(0, 11);
    resource.mutableBlockIds("indexer_kv").setAt(0, 12);
    resource.mutableBlockIds("csa_kv").setAt(1, 0);
    resource.mutableBlockIds("swa_kv").setAt(1, 66);

    const auto layer_attn_blocks = connector->resourceLayerRegionBlocks(resource, slots);

    EXPECT_TRUE(connector->kindRequiredAt(layer_attn_blocks, slots, 0, CacheBlockKind::COMPRESSED_KV));
    EXPECT_FALSE(connector->kindRequiredAt(layer_attn_blocks, slots, 0, CacheBlockKind::STATE_SWA_KV));
    EXPECT_FALSE(connector->kindRequiredAt(layer_attn_blocks, slots, 1, CacheBlockKind::COMPRESSED_KV));
    EXPECT_TRUE(connector->kindRequiredAt(layer_attn_blocks, slots, 1, CacheBlockKind::STATE_SWA_KV));

    const auto compressed_mask =
        connector->prefixSlotValidMask(layer_attn_blocks, slots, 0, CacheBlockKind::COMPRESSED_KV);
    ASSERT_EQ(compressed_mask.size(), slots.size());
    for (size_t i = 0; i < slots.size(); ++i) {
        const bool expected = slots[i].tag == "csa_kv" || slots[i].tag == "hca_kv" || slots[i].tag == "indexer_kv";
        EXPECT_EQ(compressed_mask[i] != 0, expected) << i;
    }

    const auto state_mask = connector->prefixSlotValidMask(layer_attn_blocks, slots, 1, CacheBlockKind::STATE_SWA_KV);
    ASSERT_EQ(state_mask.size(), slots.size());
    for (size_t i = 0; i < slots.size(); ++i) {
        EXPECT_EQ(state_mask[i] != 0, slots[i].tag == "swa_kv") << i;
    }
}

TEST(KVCacheBatchedMemoryCopyTest, PrefixTreeWritePlanSkipsHCAStateAndKeepsRuntimeSlotMask) {
    auto config = makeCompactDsv4TypedMemoryCopyConfig(/*use_flash=*/true);

    KVCacheConfig kv_config;
    kv_config.memory_cache_size_mb                    = 64;
    kv_config.memory_cache_sync_timeout_ms            = 1000;
    kv_config.enable_prefix_tree_memory_cache         = true;
    kv_config.enable_legacy_memory_connector_fallback = false;

    std::vector<std::string> server_addrs = {"127.0.0.1:1"};
    auto                     connector =
        std::make_shared<KVCacheMemoryConnector>(config, kv_config, std::shared_ptr<KVCacheAllocator>(), server_addrs);
    ASSERT_TRUE(connector->init());
    ASSERT_TRUE(connector->usePrefixTreeMemoryCache());

    const auto slots = connector->layerGroupSlots();
    ASSERT_TRUE(connector->supportsTypedPrefixCacheLayout(slots));
    for (const auto& slot : slots) {
        ASSERT_NE(slot.tag, "hca_state");
    }

    const int hca_layer = 3;
    ASSERT_EQ(config.groupForLayer(hca_layer, "hca_kv").tag, "hca_kv");
    ASSERT_EQ(config.groupForLayer(hca_layer, "hca_state").tag, "hca_state");
    ASSERT_EQ(config.groupForLayer(hca_layer, "swa_kv").tag, "swa_kv");

    KVCacheResource resource;
    resource.setCacheKeys({901, 902});
    initResourceGroupsForConfig(resource, config);
    resource.resizeBlocks(/*reserver_blocks=*/2, NULL_BLOCK_IDX);

    resource.mutableBlockIds("hca_kv").assign({11, 12});
    resource.mutableBlockIds("hca_state").assign({51, 52});
    resource.mutableBlockIds("swa_kv").assign({61, NULL_BLOCK_IDX});

    const auto layer_attn_blocks = connector->resourceLayerRegionBlocks(resource, slots);
    bool       no_need_write     = true;
    auto       plan              = connector->buildPrefixCopyPlanForWrite(resource.cacheKeys(),
                                                       resource.blockDependencies(),
                                                       layer_attn_blocks,
                                                       slots,
                                                       /*start_index=*/0,
                                                       /*write_num=*/2,
                                                       no_need_write);
    ASSERT_NE(plan, nullptr);
    EXPECT_FALSE(no_need_write);
    ASSERT_EQ(plan->copy_infos.size(), 3u);

    EXPECT_EQ(plan->copy_infos[0].cache_key, 901);
    EXPECT_EQ(plan->copy_infos[0].kind, CacheBlockKind::COMPRESSED_KV);
    EXPECT_EQ(plan->copy_infos[1].cache_key, 901);
    EXPECT_EQ(plan->copy_infos[1].kind, CacheBlockKind::STATE_SWA_KV);
    EXPECT_EQ(plan->copy_infos[2].cache_key, 902);
    EXPECT_EQ(plan->copy_infos[2].kind, CacheBlockKind::COMPRESSED_KV);

    auto slot_index = [&](std::string_view tag) -> size_t {
        for (size_t i = 0; i < slots.size(); ++i) {
            if (slots[i].layer_id == hca_layer && slots[i].tag == tag) {
                return i;
            }
        }
        return slots.size();
    };
    const size_t hca_kv_slot = slot_index("hca_kv");
    const size_t swa_slot    = slot_index("swa_kv");
    ASSERT_LT(hca_kv_slot, slots.size());
    ASSERT_LT(swa_slot, slots.size());

    EXPECT_NE(plan->copy_infos[0].slot_valid_mask[hca_kv_slot], 0);
    EXPECT_EQ(plan->copy_infos[0].slot_valid_mask[swa_slot], 0);
    EXPECT_EQ(plan->copy_infos[1].slot_valid_mask[hca_kv_slot], 0);
    EXPECT_NE(plan->copy_infos[1].slot_valid_mask[swa_slot], 0);
    EXPECT_NE(plan->copy_infos[2].slot_valid_mask[hca_kv_slot], 0);
    EXPECT_EQ(plan->copy_infos[2].slot_valid_mask[swa_slot], 0);
}

TEST(KVCacheBatchedMemoryCopyTest, PrefixTreeReadRejectsCompressedOnlyWhenStateSwaRequired) {
    auto config = makeCompactDsv4TypedMemoryCopyConfig(/*use_flash=*/true);

    KVCacheConfig kv_config;
    kv_config.memory_cache_size_mb                    = 64;
    kv_config.memory_cache_sync_timeout_ms            = 1000;
    kv_config.enable_prefix_tree_memory_cache         = true;
    kv_config.enable_legacy_memory_connector_fallback = false;

    auto allocator = std::make_shared<FakeTypedKVCacheAllocator>(config);

    std::vector<std::string> server_addrs = {"127.0.0.1:1"};
    auto connector = std::make_shared<KVCacheMemoryConnector>(config, kv_config, allocator, server_addrs);
    ASSERT_TRUE(connector->init());
    ASSERT_TRUE(connector->usePrefixTreeMemoryCache());

    const auto slots = connector->layerGroupSlots();
    ASSERT_TRUE(connector->supportsTypedPrefixCacheLayout(slots));

    KVCacheResource resource;
    resource.setCacheKeys({901, 902});
    initResourceGroupsForConfig(resource, config);
    resource.resizeBlocks(/*reserver_blocks=*/2, NULL_BLOCK_IDX);
    resource.mutableBlockIds("hca_kv").assign({11, 12});
    resource.mutableBlockIds("swa_kv").assign({61, 62});

    const auto layer_attn_blocks = connector->resourceLayerRegionBlocks(resource, slots);
    const auto compressed_mask =
        connector->prefixSlotValidMask(layer_attn_blocks, slots, 0, CacheBlockKind::COMPRESSED_KV);
    ASSERT_TRUE(std::any_of(compressed_mask.begin(), compressed_mask.end(), [](uint8_t valid) { return valid != 0; }));
    const auto state_mask = connector->prefixSlotValidMask(layer_attn_blocks, slots, 0, CacheBlockKind::STATE_SWA_KV);
    ASSERT_TRUE(std::any_of(state_mask.begin(), state_mask.end(), [](uint8_t valid) { return valid != 0; }));

    auto mem_blocks = connector->compressed_pool_->malloc(1);
    ASSERT_EQ(mem_blocks.size(), 1u);

    KVCacheMemoryConnector::CopyInfoPerKey copy_info;
    copy_info.cache_key       = 901;
    copy_info.kind            = CacheBlockKind::COMPRESSED_KV;
    copy_info.backing_type    = CacheBackingType::MEMORY;
    copy_info.mem_block       = mem_blocks[0];
    copy_info.block_size      = connector->prefixKindBlockSize(CacheBlockKind::COMPRESSED_KV, slots);
    copy_info.slot_valid_mask = compressed_mask;
    connector->putPrefixToCache(copy_info, resource.blockDependencies()[0], slots);

    auto read_plan = connector->buildPrefixCopyPlanForRead(resource.cacheKeys(),
                                                           resource.blockDependencies(),
                                                           layer_attn_blocks,
                                                           slots,
                                                           /*start_index=*/0,
                                                           /*read_num=*/1);
    EXPECT_EQ(read_plan, nullptr);
    EXPECT_TRUE(connector->prefix_block_cache_->match(901, CacheBlockKind::COMPRESSED_KV, compressed_mask).found);
}

TEST(KVCacheBatchedMemoryCopyTest, PrefixTreeReadAllowsStateOnlyWhenCompressedNotRequired) {
    auto config = makeCompactDsv4TypedMemoryCopyConfig(/*use_flash=*/true);

    KVCacheConfig kv_config;
    kv_config.memory_cache_size_mb                    = 64;
    kv_config.memory_cache_sync_timeout_ms            = 1000;
    kv_config.enable_prefix_tree_memory_cache         = true;
    kv_config.enable_legacy_memory_connector_fallback = false;

    auto allocator = std::make_shared<FakeTypedKVCacheAllocator>(config);

    std::vector<std::string> server_addrs = {"127.0.0.1:1"};
    auto connector = std::make_shared<KVCacheMemoryConnector>(config, kv_config, allocator, server_addrs);
    ASSERT_TRUE(connector->init());

    const auto slots = connector->layerGroupSlots();
    ASSERT_TRUE(connector->supportsTypedPrefixCacheLayout(slots));

    KVCacheResource resource;
    resource.setCacheKeys({901, 902});
    initResourceGroupsForConfig(resource, config);
    resource.resizeBlocks(/*reserver_blocks=*/2, NULL_BLOCK_IDX);
    resource.mutableBlockIds("hca_kv").assign({0, NULL_BLOCK_IDX});
    resource.mutableBlockIds("swa_kv").assign({61, 62});

    const auto layer_attn_blocks = connector->resourceLayerRegionBlocks(resource, slots);
    const auto compressed_mask =
        connector->prefixSlotValidMask(layer_attn_blocks, slots, 0, CacheBlockKind::COMPRESSED_KV);
    EXPECT_FALSE(std::any_of(compressed_mask.begin(), compressed_mask.end(), [](uint8_t valid) { return valid != 0; }));
    const auto state_mask = connector->prefixSlotValidMask(layer_attn_blocks, slots, 0, CacheBlockKind::STATE_SWA_KV);
    ASSERT_TRUE(std::any_of(state_mask.begin(), state_mask.end(), [](uint8_t valid) { return valid != 0; }));

    auto mem_blocks = connector->state_swa_pool_->malloc(1);
    ASSERT_EQ(mem_blocks.size(), 1u);

    KVCacheMemoryConnector::CopyInfoPerKey copy_info;
    copy_info.cache_key       = 901;
    copy_info.kind            = CacheBlockKind::STATE_SWA_KV;
    copy_info.backing_type    = CacheBackingType::MEMORY;
    copy_info.mem_block       = mem_blocks[0];
    copy_info.block_size      = connector->prefixKindBlockSize(CacheBlockKind::STATE_SWA_KV, slots);
    copy_info.slot_valid_mask = state_mask;
    connector->putPrefixToCache(copy_info, resource.blockDependencies()[0], slots);

    auto read_plan = connector->buildPrefixCopyPlanForRead(resource.cacheKeys(),
                                                           resource.blockDependencies(),
                                                           layer_attn_blocks,
                                                           slots,
                                                           /*start_index=*/0,
                                                           /*read_num=*/1);
    ASSERT_NE(read_plan, nullptr);
    ASSERT_EQ(read_plan->copy_infos.size(), 1u);
    EXPECT_EQ(read_plan->copy_infos[0].cache_key, 901);
    EXPECT_EQ(read_plan->copy_infos[0].kind, CacheBlockKind::STATE_SWA_KV);
}

TEST(KVCacheBatchedMemoryCopyTest, PrefixTreeBlockZeroAndNullSlotsAreNotCopiedForD2HAndH2D) {
    const auto set_device_rc = cudaSetDevice(0);
    ASSERT_EQ(set_device_rc, cudaSuccess) << cudaGetErrorString(set_device_rc);

    auto config = makeCompactDsv4TypedMemoryCopyConfig(/*use_flash=*/true);

    KVCacheConfig kv_config;
    kv_config.memory_cache_size_mb                    = 64;
    kv_config.memory_cache_sync_timeout_ms            = 1000;
    kv_config.enable_prefix_tree_memory_cache         = true;
    kv_config.enable_legacy_memory_connector_fallback = false;

    auto allocator = std::make_shared<FakeTypedKVCacheAllocator>(config);

    std::vector<std::string> server_addrs = {"127.0.0.1:1"};
    auto connector = std::make_shared<KVCacheMemoryConnector>(config, kv_config, allocator, server_addrs);
    ASSERT_TRUE(connector->init());

    const auto          slots = connector->layerGroupSlots();
    std::vector<size_t> state_slots;
    for (size_t i = 0; i < slots.size(); ++i) {
        if (slots[i].block_kind == CacheBlockKind::STATE_SWA_KV) {
            state_slots.push_back(i);
        }
    }
    ASSERT_GE(state_slots.size(), 3u);

    auto blocks = connector->state_swa_pool_->malloc(1);
    ASSERT_EQ(blocks.size(), 1u);
    const auto mem_block = blocks[0];

    auto set_prefix_slot = [&](BlockIdxType block, size_t target_slot, char value) {
        auto buffers = connector->state_swa_pool_->convertIndexToBuffer(/*layer_id=*/0, block);
        ASSERT_EQ(buffers.size(), 1u);
        size_t byte_off = 0;
        for (size_t slot_idx = 0; slot_idx < slots.size(); ++slot_idx) {
            if (slots[slot_idx].block_kind != CacheBlockKind::STATE_SWA_KV) {
                continue;
            }
            if (slot_idx == target_slot) {
                setBlockBytes(buffers[0], byte_off, slots[slot_idx].stride_bytes, value);
                return;
            }
            byte_off += slots[slot_idx].stride_bytes;
        }
        FAIL() << "target slot not found";
    };
    auto verify_prefix_slot = [&](BlockIdxType block, size_t target_slot, char value) {
        auto buffers = connector->state_swa_pool_->convertIndexToBuffer(/*layer_id=*/0, block);
        ASSERT_EQ(buffers.size(), 1u);
        size_t byte_off = 0;
        for (size_t slot_idx = 0; slot_idx < slots.size(); ++slot_idx) {
            if (slots[slot_idx].block_kind != CacheBlockKind::STATE_SWA_KV) {
                continue;
            }
            if (slot_idx == target_slot) {
                verifyBlockBytesEq(buffers[0], byte_off, slots[slot_idx].stride_bytes, value);
                return;
            }
            byte_off += slots[slot_idx].stride_bytes;
        }
        FAIL() << "target slot not found";
    };

    const auto& valid_slot      = slots[state_slots[0]];
    const auto  valid_gpu_block = static_cast<BlockIdxType>(7);
    setBlockInfosContent(allocator->convertIndexToBuffer(valid_slot.layer_id, valid_slot.tag, valid_gpu_block), 'V');
    setBlockInfosContent(allocator->convertIndexToBuffer(slots[state_slots[1]].layer_id,
                                                         slots[state_slots[1]].tag,
                                                         /*block_id=*/0),
                         'Z');
    set_prefix_slot(mem_block, state_slots[0], 'M');
    set_prefix_slot(mem_block, state_slots[1], 'M');
    set_prefix_slot(mem_block, state_slots[2], 'M');

    MemoryOperationRequestPB request;
    request.set_copy_direction(MemoryOperationRequestPB::D2H);
    auto* item = request.add_copy_items();
    item->set_mem_block(mem_block);
    item->set_backing_type(MemoryOperationRequestPB::MEMORY);
    item->set_cache_block_kind(MemoryOperationRequestPB::STATE_SWA_KV);
    item->set_is_complete(true);
    std::vector<BlockIdxType> gpu_blocks(slots.size(), NULL_BLOCK_IDX);
    for (size_t i = 0; i < slots.size(); ++i) {
        if (i == state_slots[0]) {
            gpu_blocks[i] = valid_gpu_block;
        } else if (i == state_slots[1]) {
            gpu_blocks[i] = 0;
        }
        item->add_slot_valid_mask(i == state_slots[0] || i == state_slots[1] || i == state_slots[2] ? 1 : 0);
    }
    addTaggedGpuBlocks(*item, slots, gpu_blocks);

    MemoryOperationResponsePB response;
    ASSERT_TRUE(connector->copyCache(request, response));
    EXPECT_TRUE(response.success());
    verify_prefix_slot(mem_block, state_slots[0], 'V');
    verify_prefix_slot(mem_block, state_slots[1], 'M');
    verify_prefix_slot(mem_block, state_slots[2], 'M');

    set_prefix_slot(mem_block, state_slots[0], 'A');
    set_prefix_slot(mem_block, state_slots[1], 'B');
    set_prefix_slot(mem_block, state_slots[2], 'C');
    setBlockInfosContent(allocator->convertIndexToBuffer(valid_slot.layer_id, valid_slot.tag, valid_gpu_block), 'x');
    setBlockInfosContent(allocator->convertIndexToBuffer(slots[state_slots[1]].layer_id,
                                                         slots[state_slots[1]].tag,
                                                         /*block_id=*/0),
                         'z');

    request.set_copy_direction(MemoryOperationRequestPB::H2D);
    response.Clear();
    ASSERT_TRUE(connector->copyCache(request, response));
    EXPECT_TRUE(response.success());
    verifyBlockInfosContent(allocator->convertIndexToBuffer(valid_slot.layer_id, valid_slot.tag, valid_gpu_block), 'A');
    verifyBlockInfosContent(
        allocator->convertIndexToBuffer(slots[state_slots[1]].layer_id, slots[state_slots[1]].tag, 0), 'z');
}

TEST(KVCacheBatchedMemoryCopyTest, PrefixTreeD2HMergeSourceKeepsOldSlotsAndOverlaysNewSlots) {
    const auto set_device_rc = cudaSetDevice(0);
    ASSERT_EQ(set_device_rc, cudaSuccess) << cudaGetErrorString(set_device_rc);

    auto config = makeCompactDsv4TypedMemoryCopyConfig(/*use_flash=*/true);

    KVCacheConfig kv_config;
    kv_config.memory_cache_size_mb                    = 64;
    kv_config.memory_cache_sync_timeout_ms            = 1000;
    kv_config.enable_prefix_tree_memory_cache         = true;
    kv_config.enable_legacy_memory_connector_fallback = false;

    auto allocator = std::make_shared<FakeTypedKVCacheAllocator>(config);

    std::vector<std::string> server_addrs = {"127.0.0.1:1"};
    auto connector = std::make_shared<KVCacheMemoryConnector>(config, kv_config, allocator, server_addrs);
    ASSERT_TRUE(connector->init());
    ASSERT_TRUE(connector->usePrefixTreeMemoryCache());

    const auto          slots = connector->layerGroupSlots();
    std::vector<size_t> state_slots;
    for (size_t i = 0; i < slots.size(); ++i) {
        if (slots[i].block_kind == CacheBlockKind::STATE_SWA_KV) {
            state_slots.push_back(i);
        }
    }
    ASSERT_GE(state_slots.size(), 2u);

    auto blocks = connector->state_swa_pool_->malloc(2);
    ASSERT_EQ(blocks.size(), 2u);
    const auto old_block = blocks[0];
    const auto new_block = blocks[1];

    auto set_prefix_slot = [&](BlockIdxType block, size_t target_slot, char value) {
        auto buffers = connector->state_swa_pool_->convertIndexToBuffer(/*layer_id=*/0, block);
        ASSERT_EQ(buffers.size(), 1u);
        size_t byte_off = 0;
        for (size_t slot_idx = 0; slot_idx < slots.size(); ++slot_idx) {
            if (slots[slot_idx].block_kind != CacheBlockKind::STATE_SWA_KV) {
                continue;
            }
            if (slot_idx == target_slot) {
                setBlockBytes(buffers[0], byte_off, slots[slot_idx].stride_bytes, value);
                return;
            }
            byte_off += slots[slot_idx].stride_bytes;
        }
        FAIL() << "target slot not found";
    };
    auto verify_prefix_slot = [&](BlockIdxType block, size_t target_slot, char value) {
        auto buffers = connector->state_swa_pool_->convertIndexToBuffer(/*layer_id=*/0, block);
        ASSERT_EQ(buffers.size(), 1u);
        size_t byte_off = 0;
        for (size_t slot_idx = 0; slot_idx < slots.size(); ++slot_idx) {
            if (slots[slot_idx].block_kind != CacheBlockKind::STATE_SWA_KV) {
                continue;
            }
            if (slot_idx == target_slot) {
                verifyBlockBytesEq(buffers[0], byte_off, slots[slot_idx].stride_bytes, value);
                return;
            }
            byte_off += slots[slot_idx].stride_bytes;
        }
        FAIL() << "target slot not found";
    };

    setBlockInfosContent(connector->state_swa_pool_->convertIndexToBuffer(/*layer_id=*/0, old_block), 0);
    setBlockInfosContent(connector->state_swa_pool_->convertIndexToBuffer(/*layer_id=*/0, new_block), 0);
    set_prefix_slot(old_block, state_slots[0], 'O');

    const auto& new_slot      = slots[state_slots[1]];
    const auto  new_gpu_block = static_cast<BlockIdxType>(7);
    setBlockInfosContent(allocator->convertIndexToBuffer(new_slot.layer_id, new_slot.tag, new_gpu_block), 'N');

    MemoryOperationRequestPB request;
    request.set_copy_direction(MemoryOperationRequestPB::D2H);
    auto* item = request.add_copy_items();
    item->set_mem_block(new_block);
    item->set_src_mem_block(old_block);
    item->set_backing_type(MemoryOperationRequestPB::MEMORY);
    item->set_cache_block_kind(MemoryOperationRequestPB::STATE_SWA_KV);
    item->set_is_complete(true);
    std::vector<BlockIdxType> gpu_blocks(slots.size(), NULL_BLOCK_IDX);
    for (size_t i = 0; i < slots.size(); ++i) {
        gpu_blocks[i] = i == state_slots[1] ? new_gpu_block : NULL_BLOCK_IDX;
        item->add_slot_valid_mask(i == state_slots[0] || i == state_slots[1] ? 1 : 0);
    }
    addTaggedGpuBlocks(*item, slots, gpu_blocks);

    MemoryOperationResponsePB response;
    ASSERT_TRUE(connector->copyCache(request, response));
    EXPECT_TRUE(response.success());
    verify_prefix_slot(new_block, state_slots[0], 'O');
    verify_prefix_slot(new_block, state_slots[1], 'N');
}

TEST(KVCacheBatchedMemoryCopyTest, PrefixTreeCommitConflictMergesDisjointSlotMasks) {
    auto config = makeCompactDsv4TypedMemoryCopyConfig(/*use_flash=*/true);

    KVCacheConfig kv_config;
    kv_config.memory_cache_size_mb                    = 64;
    kv_config.memory_cache_sync_timeout_ms            = 1000;
    kv_config.enable_prefix_tree_memory_cache         = true;
    kv_config.enable_legacy_memory_connector_fallback = false;

    auto allocator = std::make_shared<FakeTypedKVCacheAllocator>(config);

    std::vector<std::string> server_addrs = {"127.0.0.1:1"};
    auto connector = std::make_shared<KVCacheMemoryConnector>(config, kv_config, allocator, server_addrs);
    ASSERT_TRUE(connector->init());
    ASSERT_TRUE(connector->usePrefixTreeMemoryCache());

    const auto          slots = connector->layerGroupSlots();
    std::vector<size_t> state_slots;
    for (size_t i = 0; i < slots.size(); ++i) {
        if (slots[i].block_kind == CacheBlockKind::STATE_SWA_KV) {
            state_slots.push_back(i);
        }
    }
    ASSERT_GE(state_slots.size(), 2u);

    auto blocks = connector->state_swa_pool_->malloc(2);
    ASSERT_EQ(blocks.size(), 2u);
    const auto old_block = blocks[0];
    const auto new_block = blocks[1];

    auto set_prefix_slot = [&](BlockIdxType block, size_t target_slot, char value) {
        auto buffers = connector->state_swa_pool_->convertIndexToBuffer(/*layer_id=*/0, block);
        ASSERT_EQ(buffers.size(), 1u);
        size_t byte_off = 0;
        for (size_t slot_idx = 0; slot_idx < slots.size(); ++slot_idx) {
            if (slots[slot_idx].block_kind != CacheBlockKind::STATE_SWA_KV) {
                continue;
            }
            if (slot_idx == target_slot) {
                setBlockBytes(buffers[0], byte_off, slots[slot_idx].stride_bytes, value);
                return;
            }
            byte_off += slots[slot_idx].stride_bytes;
        }
        FAIL() << "target slot not found";
    };
    auto verify_prefix_slot = [&](BlockIdxType block, size_t target_slot, char value) {
        auto buffers = connector->state_swa_pool_->convertIndexToBuffer(/*layer_id=*/0, block);
        ASSERT_EQ(buffers.size(), 1u);
        size_t byte_off = 0;
        for (size_t slot_idx = 0; slot_idx < slots.size(); ++slot_idx) {
            if (slots[slot_idx].block_kind != CacheBlockKind::STATE_SWA_KV) {
                continue;
            }
            if (slot_idx == target_slot) {
                verifyBlockBytesEq(buffers[0], byte_off, slots[slot_idx].stride_bytes, value);
                return;
            }
            byte_off += slots[slot_idx].stride_bytes;
        }
        FAIL() << "target slot not found";
    };

    setBlockInfosContent(connector->state_swa_pool_->convertIndexToBuffer(/*layer_id=*/0, old_block), 0);
    setBlockInfosContent(connector->state_swa_pool_->convertIndexToBuffer(/*layer_id=*/0, new_block), 0);
    set_prefix_slot(old_block, state_slots[0], 'O');
    set_prefix_slot(new_block, state_slots[1], 'N');

    auto make_mask = [&](size_t target_slot) {
        std::vector<uint8_t> mask(slots.size(), 0);
        mask[target_slot] = 1;
        return mask;
    };

    KVCacheMemoryConnector::CopyInfoPerKey old_info;
    old_info.cache_key       = 901;
    old_info.kind            = CacheBlockKind::STATE_SWA_KV;
    old_info.backing_type    = CacheBackingType::MEMORY;
    old_info.mem_block       = old_block;
    old_info.block_size      = connector->prefixKindBlockSize(CacheBlockKind::STATE_SWA_KV, slots);
    old_info.slot_valid_mask = make_mask(state_slots[0]);
    connector->putPrefixToCache(old_info, rootDep(0), slots);

    KVCacheMemoryConnector::CopyInfoPerKey new_info;
    new_info.cache_key       = 901;
    new_info.kind            = CacheBlockKind::STATE_SWA_KV;
    new_info.backing_type    = CacheBackingType::MEMORY;
    new_info.mem_block       = new_block;
    new_info.block_size      = connector->prefixKindBlockSize(CacheBlockKind::STATE_SWA_KV, slots);
    new_info.slot_valid_mask = make_mask(state_slots[1]);
    connector->putPrefixToCache(new_info, rootDep(0), slots);

    std::vector<uint8_t> required(slots.size(), 0);
    required[state_slots[0]] = 1;
    required[state_slots[1]] = 1;
    auto match               = connector->prefix_block_cache_->match(901, CacheBlockKind::STATE_SWA_KV, required);
    ASSERT_TRUE(match.found);
    EXPECT_EQ(match.block_index, new_block);
    verify_prefix_slot(new_block, state_slots[0], 'O');
    verify_prefix_slot(new_block, state_slots[1], 'N');
}

TEST(KVCacheBatchedMemoryCopyTest, PrefixTreeCommitConflictMergesOverlappingSlotMasksPreferNewSlots) {
    auto config = makeCompactDsv4TypedMemoryCopyConfig(/*use_flash=*/true);

    KVCacheConfig kv_config;
    kv_config.memory_cache_size_mb                    = 64;
    kv_config.memory_cache_sync_timeout_ms            = 1000;
    kv_config.enable_prefix_tree_memory_cache         = true;
    kv_config.enable_legacy_memory_connector_fallback = false;

    auto allocator = std::make_shared<FakeTypedKVCacheAllocator>(config);

    std::vector<std::string> server_addrs = {"127.0.0.1:1"};
    auto connector = std::make_shared<KVCacheMemoryConnector>(config, kv_config, allocator, server_addrs);
    ASSERT_TRUE(connector->init());

    const auto          slots = connector->layerGroupSlots();
    std::vector<size_t> state_slots;
    for (size_t i = 0; i < slots.size(); ++i) {
        if (slots[i].block_kind == CacheBlockKind::STATE_SWA_KV) {
            state_slots.push_back(i);
        }
    }
    ASSERT_GE(state_slots.size(), 3u);

    auto blocks = connector->state_swa_pool_->malloc(2);
    ASSERT_EQ(blocks.size(), 2u);
    const auto old_block = blocks[0];
    const auto new_block = blocks[1];

    auto set_prefix_slot = [&](BlockIdxType block, size_t target_slot, char value) {
        auto buffers = connector->state_swa_pool_->convertIndexToBuffer(/*layer_id=*/0, block);
        ASSERT_EQ(buffers.size(), 1u);
        size_t byte_off = 0;
        for (size_t slot_idx = 0; slot_idx < slots.size(); ++slot_idx) {
            if (slots[slot_idx].block_kind != CacheBlockKind::STATE_SWA_KV) {
                continue;
            }
            if (slot_idx == target_slot) {
                setBlockBytes(buffers[0], byte_off, slots[slot_idx].stride_bytes, value);
                return;
            }
            byte_off += slots[slot_idx].stride_bytes;
        }
        FAIL() << "target slot not found";
    };
    auto verify_prefix_slot = [&](BlockIdxType block, size_t target_slot, char value) {
        auto buffers = connector->state_swa_pool_->convertIndexToBuffer(/*layer_id=*/0, block);
        ASSERT_EQ(buffers.size(), 1u);
        size_t byte_off = 0;
        for (size_t slot_idx = 0; slot_idx < slots.size(); ++slot_idx) {
            if (slots[slot_idx].block_kind != CacheBlockKind::STATE_SWA_KV) {
                continue;
            }
            if (slot_idx == target_slot) {
                verifyBlockBytesEq(buffers[0], byte_off, slots[slot_idx].stride_bytes, value);
                return;
            }
            byte_off += slots[slot_idx].stride_bytes;
        }
        FAIL() << "target slot not found";
    };
    auto make_mask = [&](std::initializer_list<size_t> targets) {
        std::vector<uint8_t> mask(slots.size(), 0);
        for (auto target : targets) {
            mask[target] = 1;
        }
        return mask;
    };

    setBlockInfosContent(connector->state_swa_pool_->convertIndexToBuffer(/*layer_id=*/0, old_block), 0);
    setBlockInfosContent(connector->state_swa_pool_->convertIndexToBuffer(/*layer_id=*/0, new_block), 0);
    set_prefix_slot(old_block, state_slots[0], 'A');
    set_prefix_slot(old_block, state_slots[1], 'O');
    set_prefix_slot(new_block, state_slots[1], 'N');
    set_prefix_slot(new_block, state_slots[2], 'C');

    KVCacheMemoryConnector::CopyInfoPerKey old_info;
    old_info.cache_key       = 902;
    old_info.kind            = CacheBlockKind::STATE_SWA_KV;
    old_info.backing_type    = CacheBackingType::MEMORY;
    old_info.mem_block       = old_block;
    old_info.block_size      = connector->prefixKindBlockSize(CacheBlockKind::STATE_SWA_KV, slots);
    old_info.slot_valid_mask = make_mask({state_slots[0], state_slots[1]});
    connector->putPrefixToCache(old_info, rootDep(0), slots);

    KVCacheMemoryConnector::CopyInfoPerKey new_info;
    new_info.cache_key       = 902;
    new_info.kind            = CacheBlockKind::STATE_SWA_KV;
    new_info.backing_type    = CacheBackingType::MEMORY;
    new_info.mem_block       = new_block;
    new_info.block_size      = connector->prefixKindBlockSize(CacheBlockKind::STATE_SWA_KV, slots);
    new_info.slot_valid_mask = make_mask({state_slots[1], state_slots[2]});
    connector->putPrefixToCache(new_info, rootDep(0), slots);

    const auto required = make_mask({state_slots[0], state_slots[1], state_slots[2]});
    auto       match    = connector->prefix_block_cache_->match(902, CacheBlockKind::STATE_SWA_KV, required);
    ASSERT_TRUE(match.found);
    EXPECT_EQ(match.block_index, new_block);
    verify_prefix_slot(new_block, state_slots[0], 'A');
    verify_prefix_slot(new_block, state_slots[1], 'N');
    verify_prefix_slot(new_block, state_slots[2], 'C');
}

TEST(KVCacheBatchedMemoryCopyTest, PrefixTreeCommitCoveredMaskReleasesRejectedBacking) {
    auto config = makeCompactDsv4TypedMemoryCopyConfig(/*use_flash=*/true);

    KVCacheConfig kv_config;
    kv_config.memory_cache_size_mb                    = 64;
    kv_config.memory_cache_sync_timeout_ms            = 1000;
    kv_config.enable_prefix_tree_memory_cache         = true;
    kv_config.enable_legacy_memory_connector_fallback = false;

    auto allocator = std::make_shared<FakeTypedKVCacheAllocator>(config);

    std::vector<std::string> server_addrs = {"127.0.0.1:1"};
    auto connector = std::make_shared<KVCacheMemoryConnector>(config, kv_config, allocator, server_addrs);
    ASSERT_TRUE(connector->init());

    const auto          slots = connector->layerGroupSlots();
    std::vector<size_t> state_slots;
    for (size_t i = 0; i < slots.size(); ++i) {
        if (slots[i].block_kind == CacheBlockKind::STATE_SWA_KV) {
            state_slots.push_back(i);
        }
    }
    ASSERT_GE(state_slots.size(), 2u);

    auto blocks = connector->state_swa_pool_->malloc(2);
    ASSERT_EQ(blocks.size(), 2u);
    const auto old_block      = blocks[0];
    const auto rejected_block = blocks[1];
    auto       make_mask      = [&](std::initializer_list<size_t> targets) {
        std::vector<uint8_t> mask(slots.size(), 0);
        for (auto target : targets) {
            mask[target] = 1;
        }
        return mask;
    };

    KVCacheMemoryConnector::CopyInfoPerKey old_info;
    old_info.cache_key       = 903;
    old_info.kind            = CacheBlockKind::STATE_SWA_KV;
    old_info.backing_type    = CacheBackingType::MEMORY;
    old_info.mem_block       = old_block;
    old_info.block_size      = connector->prefixKindBlockSize(CacheBlockKind::STATE_SWA_KV, slots);
    old_info.slot_valid_mask = make_mask({state_slots[0], state_slots[1]});
    connector->putPrefixToCache(old_info, rootDep(0), slots);

    const auto free_after_old_commit = connector->state_swa_pool_->freeBlocksNum();

    KVCacheMemoryConnector::CopyInfoPerKey rejected_info;
    rejected_info.cache_key       = 903;
    rejected_info.kind            = CacheBlockKind::STATE_SWA_KV;
    rejected_info.backing_type    = CacheBackingType::MEMORY;
    rejected_info.mem_block       = rejected_block;
    rejected_info.block_size      = connector->prefixKindBlockSize(CacheBlockKind::STATE_SWA_KV, slots);
    rejected_info.slot_valid_mask = make_mask({state_slots[0]});
    connector->putPrefixToCache(rejected_info, rootDep(0), slots);

    EXPECT_EQ(connector->state_swa_pool_->freeBlocksNum(), free_after_old_commit + 1);
    auto match = connector->prefix_block_cache_->match(
        903, CacheBlockKind::STATE_SWA_KV, make_mask({state_slots[0], state_slots[1]}));
    ASSERT_TRUE(match.found);
    EXPECT_EQ(match.block_index, old_block);
}

TEST(KVCacheBatchedMemoryCopyTest, PrefixTreeWriteAllocationFailureDoesNotDoubleFreePartialBlocks) {
    auto config = makeCompactDsv4TypedMemoryCopyConfig(/*use_flash=*/true);
    setGroupStridesForConfig(config, 3072, 0);

    KVCacheConfig kv_config;
    kv_config.memory_cache_size_mb                    = 1;
    kv_config.memory_cache_sync_timeout_ms            = 1000;
    kv_config.enable_prefix_tree_memory_cache         = true;
    kv_config.enable_legacy_memory_connector_fallback = false;

    std::vector<std::string> server_addrs = {"127.0.0.1:1"};
    auto                     connector =
        std::make_shared<KVCacheMemoryConnector>(config, kv_config, std::shared_ptr<KVCacheAllocator>(), server_addrs);
    ASSERT_TRUE(connector->init());
    ASSERT_TRUE(connector->usePrefixTreeMemoryCache());
    ASSERT_EQ(connector->compressed_pool_->totalBlocksNum(), 1u);
    ASSERT_EQ(connector->state_swa_pool_->totalBlocksNum(), 1u);

    const CacheKeysType cache_keys{101, 102};
    KVCacheResource     resource;
    initResourceGroupsForConfig(resource, config);
    resource.resizeBlocks(static_cast<int>(cache_keys.size()), NULL_BLOCK_IDX);
    resource.setCacheKeys(cache_keys);

    for (const auto& group : config.topology().groups()) {
        auto& blocks = resource.mutableBlockIds(group.tag);
        blocks.setAt(0, 10);
        blocks.setAt(1, 20);
    }

    const auto slots = connector->layerGroupSlots();
    ASSERT_TRUE(connector->supportsTypedPrefixCacheLayout(slots));
    const auto layer_attn_blocks = connector->resourceLayerRegionBlocks(resource, slots);
    bool       no_need_write     = true;

    auto plan = connector->buildPrefixCopyPlanForWrite(cache_keys,
                                                       resource.blockDependencies(),
                                                       layer_attn_blocks,
                                                       slots,
                                                       /*start_index=*/0,
                                                       /*write_num=*/static_cast<int>(cache_keys.size()),
                                                       no_need_write);
    EXPECT_EQ(plan, nullptr);
    EXPECT_FALSE(no_need_write);
    EXPECT_EQ(connector->compressed_pool_->freeBlocksNum(), 1u);
    EXPECT_EQ(connector->state_swa_pool_->freeBlocksNum(), 1u);
}

}  // namespace rtp_llm::test

int main(int argc, char** argv) {
    rtp_llm::initLogger();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

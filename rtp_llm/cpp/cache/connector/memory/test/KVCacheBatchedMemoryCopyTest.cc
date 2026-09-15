// Copyright (c) RTP-LLM

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <filesystem>
#include <ftw.h>
#include <fcntl.h>
#include <unistd.h>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <torch/torch.h>
#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/HybridPoolConfigCreator.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.h"
#include "rtp_llm/cpp/cache/connector/memory/MemoryAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/memory/test/mock/TestRpcService.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

namespace rtp_llm {

bool KVCacheAllocator::init() {
    return doInit();
}

MallocResult KVCacheAllocator::malloc(const MallocInfo&) {
    return {false, 0};
}

MallocResult KVCacheAllocator::initMalloc(const MallocInfo&) {
    return {false, 0};
}

MallocStatus KVCacheAllocator::evaluateInitCapacity(const MallocInfo&, size_t, InitCapacityMode) const {
    return MallocStatus::NONE;
}

BlockAddrInfo KVCacheAllocator::convertIndexToAddr(int layer_id, KVCacheRegionName, int block_id) const {
    return convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo> KVCacheAllocator::convertIndexToBuffer(int layer_id, KVCacheRegionName, int block_id) const {
    return convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> KVCacheAllocator::convertIndexToBuffer(
    int layer_id, KVCacheRegionName, int block_id, int partition_count, int partition_id) const {
    return convertIndexToBuffer(layer_id, block_id, partition_count, partition_id);
}

void KVCacheAllocator::blockCopy(int, int) {}
void KVCacheAllocator::blockBatchCopy(const std::vector<BlockIdPair>&) {}
void KVCacheAllocator::blockBatchCopy(const BlockIdPair*, const BlockIdPair*) {}
void KVCacheAllocator::blockBatchCopy(const torch::Tensor&) {}
void KVCacheAllocator::regUserMr(size_t, std::shared_ptr<CacheStore>) {}

int64_t KVCacheAllocator::getMrCostTimeMs() const {
    return 0;
}

size_t KVCacheAllocator::freeBlocksNum() const {
    return 0;
}

size_t KVCacheAllocator::availableBlocksNum() const {
    return 0;
}

BatchKVCacheResourcePtr KVCacheAllocator::popBlocksFromCache(size_t) {
    return nullptr;
}

void KVCacheAllocator::blockCacheFree(const BatchKVCacheResourcePtr&) {}

size_t KVCacheAllocator::requestRefBlocksNum() const {
    return 0;
}

size_t KVCacheAllocator::connectorRefBlocksNum() const {
    return 0;
}

size_t KVCacheAllocator::blockCacheRefBlocksNum() const {
    return 0;
}

size_t KVCacheAllocator::notInUseBlocksNum() const {
    return 0;
}

size_t KVCacheAllocator::availableTokensNum() const {
    return 0;
}

size_t KVCacheAllocator::totalTokensNum() const {
    return 0;
}

size_t KVCacheAllocator::totalBlocksNum() const {
    return 0;
}

size_t KVCacheAllocator::maxAvailableTokensNum() const {
    return 0;
}

uint32_t KVCacheAllocator::convertToGlobalLayerId(size_t, int local_layer_id) const {
    return static_cast<uint32_t>(local_layer_id);
}

}  // namespace rtp_llm

namespace rtp_llm::test {
namespace {

BlockDependency rootDep(uint32_t ordinal = 0) {
    BlockDependency dep;
    dep.ordinal = ordinal;
    return dep;
}

ModelConfig makeDsv4ProModelConfig() {
    ModelConfig mc;
    mc.num_layers                   = 61;
    mc.hidden_size                  = 7168;
    mc.attn_config.head_num         = 128;
    mc.attn_config.kv_head_num      = 1;
    mc.attn_config.size_per_head    = 512;
    mc.attn_config.rope_head_dim    = 64;
    mc.attn_config.sliding_window   = 128;
    mc.attn_config.indexer_head_dim = 128;
    mc.attn_config.indexer_head_num = 64;
    mc.attn_config.indexer_topk     = 1024;
    mc.attn_config.o_groups         = 16;
    mc.attn_config.o_lora_rank      = 1024;
    mc.attn_config.kv_cache_dtype   = KvCacheDataType::FP8;

    std::vector<int> ratios;
    ratios.push_back(128);
    ratios.push_back(128);
    for (int i = 2; i < mc.num_layers; ++i) {
        ratios.push_back((i % 2 == 0) ? 4 : 128);
    }
    mc.attn_config.layer_compress_ratios = ratios;
    return mc;
}

ModelConfig makeDsv4FlashModelConfig() {
    ModelConfig mc;
    mc.num_layers                   = 43;
    mc.hidden_size                  = 4096;
    mc.attn_config.head_num         = 64;
    mc.attn_config.kv_head_num      = 1;
    mc.attn_config.size_per_head    = 512;
    mc.attn_config.rope_head_dim    = 64;
    mc.attn_config.sliding_window   = 128;
    mc.attn_config.indexer_head_dim = 128;
    mc.attn_config.indexer_head_num = 64;
    mc.attn_config.indexer_topk     = 512;
    mc.attn_config.o_groups         = 8;
    mc.attn_config.o_lora_rank      = 1024;
    mc.attn_config.kv_cache_dtype   = KvCacheDataType::FP8;

    std::vector<int> ratios = {0, 0};
    for (int i = 2; i < mc.num_layers; ++i) {
        ratios.push_back((i % 2 == 0) ? 4 : 128);
    }
    mc.attn_config.layer_compress_ratios = ratios;
    return mc;
}

CacheConfig makeRealDsv4TypedMemoryCopyConfig(bool use_flash, const ParallelismConfig& pc = {}) {
    auto              mc = use_flash ? makeDsv4FlashModelConfig() : makeDsv4ProModelConfig();
    KVCacheConfig     kv_config;
    kv_config.seq_size_per_block        = 128;
    kv_config.kernel_seq_size_per_block = 128;
    kv_config.dsv4_fixed_pool_blocks    = 512;
    auto config                         = HybridPoolConfigCreator::createConfig(mc, pc, kv_config, false, 0);
    config.block_num                    = 512;
    return config;
}

CacheConfig makeTinyTypedHybridPoolConfig() {
    CacheConfig config;
    config.dtype                       = rtp_llm::DataType::TYPE_FP16;
    config.layer_num                   = 2;
    config.layer_all_num               = 2;
    config.block_num                   = 16;
    config.seq_size_per_block          = 4;
    config.kernel_seq_size_per_block   = 4;
    config.use_independent_block_pools = true;

    auto make_spec = [&](uint32_t size_per_head) {
        auto spec                = std::make_shared<MHAKVCacheSpec>();
        spec->type               = KVCacheSpecType::MultiHeadAttention;
        spec->dtype              = config.dtype;
        spec->layer_num          = config.layer_num;
        spec->local_head_num_kv  = 1;
        spec->size_per_head      = size_per_head;
        spec->seq_size_per_block = static_cast<uint32_t>(config.seq_size_per_block);
        return spec;
    };
    auto csa_spec = make_spec(/*size_per_head=*/4);
    auto swa_spec = make_spec(/*size_per_head=*/8);

    config.layer_ids                = {{0, 1}, {0, 1}};
    config.global_layer_ids         = config.layer_ids;
    config.cache_specs              = {csa_spec, swa_spec};
    config.group_types              = {CacheGroupType::FULL, CacheGroupType::FULL};
    config.group_region_names       = {KVCacheRegionName::CSA_KV, KVCacheRegionName::SWA_KV};
    config.group_block_nums         = {config.block_num, config.block_num};
    config.group_seq_size_per_block = {config.seq_size_per_block, config.seq_size_per_block};

    config.layer_to_group_id.assign(config.layer_all_num, 0);
    config.layer_to_group_ids.assign(config.layer_all_num, std::vector<int>{0, 1});
    config.layer_region_to_group_id.assign(config.layer_all_num,
                                           std::vector<int>(static_cast<size_t>(KVCacheRegionName::REGION_COUNT), -1));
    config.layer_group_types.assign(config.layer_all_num, CacheGroupType::FULL);
    for (size_t layer = 0; layer < config.layer_all_num; ++layer) {
        config.layer_region_to_group_id[layer][static_cast<size_t>(KVCacheRegionName::CSA_KV)] = 0;
        config.layer_region_to_group_id[layer][static_cast<size_t>(KVCacheRegionName::SWA_KV)] = 1;
    }

    config.group_kv_block_stride_bytes = {csa_spec->block_size_bytes(), swa_spec->block_size_bytes()};
    config.group_kv_scale_stride_bytes = {csa_spec->scale_block_size_bytes(), swa_spec->scale_block_size_bytes()};
    config.kv_block_stride_bytes       = swa_spec->block_size_bytes();
    config.kv_scale_stride_bytes       = 0;
    config.kv_block_size_bytes         = static_cast<size_t>(config.layer_all_num) * config.kv_block_stride_bytes;
    config.kv_scale_size_bytes         = 0;
    config.block_size_bytes            = config.kv_block_size_bytes;

    const size_t csa_stride = csa_spec->block_size_bytes() + csa_spec->scale_block_size_bytes();
    const size_t swa_stride = swa_spec->block_size_bytes() + swa_spec->scale_block_size_bytes();
    config.layer_to_block_stride_bytes.assign(config.layer_all_num, static_cast<int>(csa_stride + swa_stride));
    return config;
}

CacheConfig makeCompactDsv4TypedMemoryCopyConfig(bool use_flash) {
    CacheConfig config;
    config.dtype                       = rtp_llm::DataType::TYPE_UINT8;
    config.layer_num                   = use_flash ? 43 : 61;
    config.layer_all_num               = config.layer_num;
    config.block_num                   = 512;
    config.seq_size_per_block          = 256;
    config.kernel_seq_size_per_block   = 256;
    config.use_independent_block_pools = true;
    config.use_typed_cache_regions     = true;
    config.use_opaque_kv_cache_store   = true;
    config.is_sparse                   = true;

    constexpr size_t kDsv4PoolNum      = 7;
    config.group_region_names          = {KVCacheRegionName::CSA_KV,
                                          KVCacheRegionName::HCA_KV,
                                          KVCacheRegionName::INDEXER_KV,
                                          KVCacheRegionName::INDEXER_STATE,
                                          KVCacheRegionName::CSA_STATE,
                                          KVCacheRegionName::HCA_STATE,
                                          KVCacheRegionName::SWA_KV};
    config.group_types                 = {CacheGroupType::FULL,
                                          CacheGroupType::FULL,
                                          CacheGroupType::FULL,
                                          CacheGroupType::SWA,
                                          CacheGroupType::SWA,
                                          CacheGroupType::SWA,
                                          CacheGroupType::SWA};
    config.group_kv_block_stride_bytes = {64, 16, 32, 48, 80, 40, 96};
    config.group_kv_scale_stride_bytes = std::vector<size_t>(kDsv4PoolNum, 0);
    config.group_seq_size_per_block    = std::vector<size_t>(kDsv4PoolNum, config.seq_size_per_block);
    config.group_block_nums            = std::vector<uint32_t>(kDsv4PoolNum, config.block_num);
    config.dsv4_fixed_pool_blocks      = config.block_num;
    config.layer_ids                   = std::vector<std::vector<int>>(kDsv4PoolNum);
    config.global_layer_ids            = std::vector<std::vector<int>>(kDsv4PoolNum);
    config.layer_to_group_id           = std::vector<int>(config.layer_all_num, 6);
    config.layer_to_group_ids          = std::vector<std::vector<int>>(config.layer_all_num);
    config.layer_group_types           = std::vector<CacheGroupType>(config.layer_all_num, CacheGroupType::SWA);
    config.layer_region_to_group_id    = std::vector<std::vector<int>>(
        config.layer_all_num, std::vector<int>(static_cast<size_t>(KVCacheRegionName::REGION_COUNT), -1));
    config.layer_to_block_stride_bytes = std::vector<int>(config.layer_all_num, 0);
    config.cache_specs.reserve(kDsv4PoolNum);

    auto make_spec = [&](uint32_t layer_num) {
        auto spec                = std::make_shared<MHAKVCacheSpec>();
        spec->type               = KVCacheSpecType::MultiHeadAttention;
        spec->dtype              = config.dtype;
        spec->layer_num          = layer_num;
        spec->local_head_num_kv  = 1;
        spec->size_per_head      = 16;
        spec->seq_size_per_block = static_cast<uint32_t>(config.seq_size_per_block);
        return spec;
    };

    auto add_region = [&](size_t layer, KVCacheRegionName region_name, int gid) {
        config.layer_region_to_group_id[layer][static_cast<size_t>(region_name)] = gid;
        config.layer_to_group_ids[layer].push_back(gid);
        config.layer_ids[static_cast<size_t>(gid)].push_back(static_cast<int>(layer));
    };

    for (size_t layer = 0; layer < config.layer_all_num; ++layer) {
        const bool is_csa = layer >= 2 && layer % 2 == 0;
        const bool is_hca = use_flash ? (layer >= 2 && layer % 2 == 1) : (!is_csa);
        if (is_csa) {
            add_region(layer, KVCacheRegionName::CSA_KV, 0);
            add_region(layer, KVCacheRegionName::INDEXER_KV, 2);
            add_region(layer, KVCacheRegionName::INDEXER_STATE, 3);
            add_region(layer, KVCacheRegionName::CSA_STATE, 4);
        } else if (is_hca) {
            add_region(layer, KVCacheRegionName::HCA_KV, 1);
            add_region(layer, KVCacheRegionName::HCA_STATE, 5);
        }
        add_region(layer, KVCacheRegionName::SWA_KV, 6);
    }

    config.global_layer_ids = config.layer_ids;
    for (size_t gid = 0; gid < kDsv4PoolNum; ++gid) {
        config.cache_specs.push_back(make_spec(static_cast<uint32_t>(config.layer_ids[gid].size())));
        config.group_block_size_bytes.push_back(config.group_kv_block_stride_bytes[gid] * config.layer_ids[gid].size());
    }
    return config;
}

char copyTag(size_t index) {
    return static_cast<char>(33 + (index % 90));
}

size_t sumBlockInfosBytes(const std::vector<BlockInfo>& infos) {
    size_t total = 0;
    for (const auto& b : infos) {
        if (b.addr && b.size_bytes > 0) {
            total += b.size_bytes;
        }
    }
    return total;
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
    explicit FakeTypedKVCacheAllocator(const CacheConfig&          config,
                                       size_t                      payload_gap_bytes = 0,
                                       std::set<KVCacheRegionName> host_regions            = {},
                                       bool                        coalesced_group_storage = false):
        KVCacheAllocator(config, AllocationType::DEVICE),
        host_regions_(std::move(host_regions)),
        payload_gap_bytes_(payload_gap_bytes) {
        const auto cuda_options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
        const auto host_options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
        if (coalesced_group_storage) {
            RTP_LLM_CHECK_WITH_INFO(config.mtp_sub_configs.empty(), "coalesced benchmark requires no MTP");
            for (size_t gid = 0; gid < config.global_layer_ids.size(); ++gid) {
                const auto&  layers = config.global_layer_ids[gid];
                const auto   region = config.group_region_names.at(gid);
                const size_t stride = config.group_kv_block_stride_bytes.at(gid);
                RTP_LLM_CHECK_WITH_INFO(gid >= config.group_kv_scale_stride_bytes.size()
                                            || config.group_kv_scale_stride_bytes[gid] == 0,
                                        "coalesced DSV4 benchmark requires scales packed in each KV tile");
                const auto blocks      = gid < config.group_block_nums.size() && config.group_block_nums[gid] > 0 ?
                                             config.group_block_nums[gid] :
                                             config.block_num;
                const bool host_region = host_regions_.count(region) > 0;
                // Match BlockPoolConfigHelper/MemoryLayoutStrategy: LayerMajor,
                // with local layers ordered by global_layer_ids[gid]. Views retain
                // the single allocation for the group, including its pinned range.
                auto storage = torch::empty(
                    {static_cast<int64_t>(layers.size()), static_cast<int64_t>(blocks), static_cast<int64_t>(stride)},
                    host_region ? host_options : cuda_options);
                if (host_region) {
                    storage = storage.pin_memory();
                }
                for (size_t local = 0; local < layers.size(); ++local) {
                    tensors_[key(layers[local], region)] = storage.select(0, static_cast<int64_t>(local));
                    strides_[key(layers[local], region)] = stride;
                }
            }
            return;
        }
        for (int layer = 0; layer < static_cast<int>(config.layer_all_num); ++layer) {
            if (static_cast<size_t>(layer) >= config.layer_region_to_group_id.size()) {
                continue;
            }
            const auto& region_to_group = config.layer_region_to_group_id[static_cast<size_t>(layer)];
            for (size_t region = 0; region < region_to_group.size(); ++region) {
                const int gid = region_to_group[region];
                if (gid < 0 || static_cast<size_t>(gid) >= config.group_kv_block_stride_bytes.size()) {
                    continue;
                }
                const size_t stride = config.group_kv_block_stride_bytes[static_cast<size_t>(gid)]
                                      + (static_cast<size_t>(gid) < config.group_kv_scale_stride_bytes.size() ?
                                             config.group_kv_scale_stride_bytes[static_cast<size_t>(gid)] :
                                             0);
                if (stride == 0) {
                    continue;
                }
                const auto region_name = static_cast<KVCacheRegionName>(region);
                const bool host_region = host_regions_.count(region_name) > 0;
                auto       tensor = torch::empty({static_cast<int64_t>(config.block_num), static_cast<int64_t>(stride)},
                                           host_region ? host_options : cuda_options);
                if (host_region) {
                    tensor = tensor.pin_memory();
                }
                tensors_[key(layer, static_cast<KVCacheRegionName>(region))] = std::move(tensor);
                strides_[key(layer, static_cast<KVCacheRegionName>(region))] = stride;
            }
        }
    }

    void free(const FreeInfo&) override {}
    void insertIntoCache(const InsertInfo&) override {}

    BlockAddrInfo convertIndexToAddr(int layer_id, int block_id) const override {
        return convertIndexToAddr(layer_id, KVCacheRegionName::CSA_KV, block_id);
    }

    BlockAddrInfo convertIndexToAddr(int layer_id, KVCacheRegionName region_name, int block_id) const override {
        const auto buffers = convertIndexToBuffer(layer_id, region_name, block_id);
        return buffers.empty() ? BlockAddrInfo{} : BlockAddrInfo{buffers[0].addr, nullptr};
    }

    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, int block_id) const override {
        return convertIndexToBuffer(layer_id, KVCacheRegionName::CSA_KV, block_id);
    }

    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, int block_id, int, int) const override {
        return convertIndexToBuffer(layer_id, block_id);
    }

    std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, KVCacheRegionName region_name, int block_id) const override {
        const auto k         = key(layer_id, region_name);
        const auto tensor_it = tensors_.find(k);
        const auto stride_it = strides_.find(k);
        if (tensor_it == tensors_.end() || stride_it == strides_.end() || block_id < 0
            || static_cast<int64_t>(block_id) >= tensor_it->second.size(0)) {
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

    std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, KVCacheRegionName region_name, int block_id, int, int) const override {
        return convertIndexToBuffer(layer_id, region_name, block_id);
    }

    std::shared_ptr<KVCacheResource> incrKVCacheRef(const KVCacheResource&, const CacheKeysType&, bool) override {
        return nullptr;
    }

    CacheLayerLayout allLayerCacheBase() const override {
        return {};
    }

    bool
    updateKVBlock(const BatchKVCacheResourcePtr&, const std::vector<int>&, bool, std::vector<BlockIdPair>&) override {
        return false;
    }

    int seqSizePerBlock() const override {
        return static_cast<int>(config_.seq_size_per_block);
    }

    int singleBatchNeedBlocks(const BatchKVCacheResourcePtr&, int, int) const override {
        return 0;
    }

private:
    static std::pair<int, KVCacheRegionName> key(int layer_id, KVCacheRegionName region_name) {
        return {layer_id, region_name};
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

    int getNeedBlocks(const MallocInfo&) const override {
        return 0;
    }

    void decrKVCacheRef(const KVCacheResource&, bool) override {}

    std::map<std::pair<int, KVCacheRegionName>, torch::Tensor> tensors_;
    std::map<std::pair<int, KVCacheRegionName>, size_t>        strides_;
    std::set<KVCacheRegionName>                                host_regions_;
    size_t                                                     payload_gap_bytes_ = 0;
};

}  // namespace

TEST(KVCacheBatchedMemoryCopyTest, StagedCopyEligibilityRequiresDsv4TypedLayout) {
    KVCacheConfig            kv_config;
    std::vector<std::string> server_addrs = {"127.0.0.1:1"};

    auto non_dsv4_config    = makeTinyTypedHybridPoolConfig();
    auto non_dsv4_connector = std::make_shared<KVCacheMemoryConnector>(
        non_dsv4_config, kv_config, std::shared_ptr<KVCacheAllocator>(), server_addrs);
    const auto non_dsv4_slots = non_dsv4_connector->layerRegionSlots();
    ASSERT_TRUE(non_dsv4_connector->hasTypedLayerRegionSlots(non_dsv4_slots));
    EXPECT_FALSE(non_dsv4_connector->isDsv4TypedCacheLayout(non_dsv4_slots));

    auto non_sparse_config      = makeCompactDsv4TypedMemoryCopyConfig(/*use_flash=*/true);
    non_sparse_config.is_sparse = false;
    auto non_sparse_connector   = std::make_shared<KVCacheMemoryConnector>(
        non_sparse_config, kv_config, std::shared_ptr<KVCacheAllocator>(), server_addrs);
    EXPECT_FALSE(non_sparse_connector->isDsv4TypedCacheLayout(non_sparse_connector->layerRegionSlots()));

    auto small_kernel_config                      = makeCompactDsv4TypedMemoryCopyConfig(/*use_flash=*/true);
    small_kernel_config.seq_size_per_block        = 256;
    small_kernel_config.kernel_seq_size_per_block = 64;
    auto small_kernel_connector                   = std::make_shared<KVCacheMemoryConnector>(
        small_kernel_config, kv_config, std::shared_ptr<KVCacheAllocator>(), server_addrs);
    EXPECT_FALSE(small_kernel_connector->isDsv4TypedCacheLayout(small_kernel_connector->layerRegionSlots()));

    auto non_divisible_config                      = makeCompactDsv4TypedMemoryCopyConfig(/*use_flash=*/true);
    non_divisible_config.seq_size_per_block        = 16384;
    non_divisible_config.kernel_seq_size_per_block = 384;
    auto non_divisible_connector                   = std::make_shared<KVCacheMemoryConnector>(
        non_divisible_config, kv_config, std::shared_ptr<KVCacheAllocator>(), server_addrs);
    EXPECT_FALSE(non_divisible_connector->isDsv4TypedCacheLayout(non_divisible_connector->layerRegionSlots()));

    auto decoupled_config                      = makeCompactDsv4TypedMemoryCopyConfig(/*use_flash=*/true);
    decoupled_config.seq_size_per_block        = 16384;
    decoupled_config.kernel_seq_size_per_block = 128;
    auto decoupled_connector                   = std::make_shared<KVCacheMemoryConnector>(
        decoupled_config, kv_config, std::shared_ptr<KVCacheAllocator>(), server_addrs);
    EXPECT_TRUE(decoupled_connector->isDsv4TypedCacheLayout(decoupled_connector->layerRegionSlots()));

    auto wrong_schema_config = makeCompactDsv4TypedMemoryCopyConfig(/*use_flash=*/true);
    ASSERT_GT(wrong_schema_config.group_region_names.size(), 6u);
    wrong_schema_config.group_region_names[6] = KVCacheRegionName::CSA_KV;
    auto wrong_schema_connector               = std::make_shared<KVCacheMemoryConnector>(
        wrong_schema_config, kv_config, std::shared_ptr<KVCacheAllocator>(), server_addrs);
    EXPECT_FALSE(wrong_schema_connector->isDsv4TypedCacheLayout(wrong_schema_connector->layerRegionSlots()));

    auto flash_config    = makeRealDsv4TypedMemoryCopyConfig(/*use_flash=*/true);
    auto flash_connector = std::make_shared<KVCacheMemoryConnector>(
        flash_config, kv_config, std::shared_ptr<KVCacheAllocator>(), server_addrs);
    EXPECT_EQ(flash_config.layer_num, 43u);
    EXPECT_TRUE(flash_connector->isDsv4TypedCacheLayout(flash_connector->layerRegionSlots()));

    auto pro_config    = makeRealDsv4TypedMemoryCopyConfig(/*use_flash=*/false);
    auto pro_connector = std::make_shared<KVCacheMemoryConnector>(
        pro_config, kv_config, std::shared_ptr<KVCacheAllocator>(), server_addrs);
    EXPECT_EQ(pro_config.layer_num, 61u);
    EXPECT_TRUE(pro_connector->isDsv4TypedCacheLayout(pro_connector->layerRegionSlots()));
}

void runDsv4TypedStagedCopyRoundTrip(const std::set<KVCacheRegionName>& host_regions) {
    const auto set_device_rc = cudaSetDevice(0);
    ASSERT_EQ(set_device_rc, cudaSuccess) << cudaGetErrorString(set_device_rc);

    auto config = makeCompactDsv4TypedMemoryCopyConfig(/*use_flash=*/true);

    KVCacheConfig kv_config;
    kv_config.memory_cache_size_mb            = 64;
    kv_config.memory_cache_sync_timeout_ms    = 1000;
    kv_config.enable_prefix_tree_memory_cache = false;

    auto allocator = std::make_shared<FakeTypedKVCacheAllocator>(config, /*payload_gap_bytes=*/8, host_regions);

    std::vector<std::string> server_addrs = {"127.0.0.1:1"};
    auto connector = std::make_shared<KVCacheMemoryConnector>(config, kv_config, allocator, server_addrs);
    connector->crc_enabled_ = false;  // Preserve coverage of the original CPU payload format.
    ASSERT_TRUE(connector->init());
    auto memory_pool = connector->isDualPool() ? connector->complete_pool_ : connector->block_pool_;
    ASSERT_NE(memory_pool, nullptr);

    const auto slots = connector->layerRegionSlots();
    ASSERT_TRUE(connector->hasTypedLayerRegionSlots(slots));
    ASSERT_TRUE(connector->isDsv4TypedCacheLayout(slots));
    ASSERT_GT(slots.size(), config.layer_all_num);

    auto mem_blocks = memory_pool->malloc(2);
    ASSERT_EQ(mem_blocks.size(), 2u);
    const std::vector<BlockIdxType> request_mem_blocks{static_cast<BlockIdxType>(mem_blocks[1]),
                                                       static_cast<BlockIdxType>(mem_blocks[0])};

    MemoryOperationRequestPB               req;
    std::vector<std::vector<BlockIdxType>> gpu_block_sets(request_mem_blocks.size(),
                                                          std::vector<BlockIdxType>(slots.size(), NULL_BLOCK_IDX));
    BlockIdxType                           next_gpu_block = 1;
    for (auto& gpu_blocks : gpu_block_sets) {
        for (auto& gpu_block : gpu_blocks) {
            gpu_block = next_gpu_block++;
        }
    }
    ASSERT_LT(next_gpu_block, static_cast<BlockIdxType>(config.block_num));
    ASSERT_EQ(gpu_block_sets.size(), request_mem_blocks.size());
    for (size_t block_idx = 0; block_idx < request_mem_blocks.size(); ++block_idx) {
        auto* item = req.add_copy_items();
        item->set_mem_block(request_mem_blocks[block_idx]);
        item->set_is_complete(true);
        ASSERT_EQ(gpu_block_sets[block_idx].size(), slots.size());
        for (const auto block : gpu_block_sets[block_idx]) {
            item->add_gpu_blocks(block);
        }
    }

    for (size_t block_idx = 0; block_idx < request_mem_blocks.size(); ++block_idx) {
        const auto mem_bufs = memory_pool->convertIndexToBuffer(0, request_mem_blocks[block_idx]);
        ASSERT_EQ(mem_bufs.size(), 1u);
        const auto& mem_buffer = mem_bufs[0];
        ASSERT_NE(mem_buffer.addr, nullptr);
        setBlockBytes(mem_buffer, /*byte_offset=*/0, mem_buffer.size_bytes, '#');

        size_t byte_off = 0;
        for (size_t i = 0; i < slots.size(); ++i) {
            const auto& slot = slots[i];
            const char  tag  = copyTag(block_idx * slots.size() + i);
            const auto  gpu_bufs =
                allocator->convertIndexToBuffer(slot.layer_id, slot.region_name, gpu_block_sets[block_idx][i]);
            ASSERT_GT(sumBlockInfosBytes(gpu_bufs), 0u);
            ASSERT_LE(sumBlockInfosBytes(gpu_bufs), slot.stride_bytes);
            setBlockInfosContent(gpu_bufs, tag);
            setBlockBytes(mem_buffer, byte_off, sumBlockInfosBytes(gpu_bufs), 0);
            byte_off += slot.stride_bytes;
        }
    }

    ASSERT_TRUE(connector->tryCopyCacheWithStagedMemoryCopy(req, KVCacheMemoryConnector::CopyDirection::D2H, slots));

    for (size_t block_idx = 0; block_idx < request_mem_blocks.size(); ++block_idx) {
        const auto mem_bufs = memory_pool->convertIndexToBuffer(0, request_mem_blocks[block_idx]);
        ASSERT_EQ(mem_bufs.size(), 1u);
        const auto& mem_buffer = mem_bufs[0];

        size_t byte_off = 0;
        for (size_t i = 0; i < slots.size(); ++i) {
            const auto& slot = slots[i];
            const auto  gpu_bufs =
                allocator->convertIndexToBuffer(slot.layer_id, slot.region_name, gpu_block_sets[block_idx][i]);
            verifyBlockBytesEq(
                mem_buffer, byte_off, sumBlockInfosBytes(gpu_bufs), copyTag(block_idx * slots.size() + i));
            if (slot.stride_bytes > sumBlockInfosBytes(gpu_bufs)) {
                verifyBlockBytesEq(mem_buffer,
                                   byte_off + sumBlockInfosBytes(gpu_bufs),
                                   slot.stride_bytes - sumBlockInfosBytes(gpu_bufs),
                                   '#');
            }
            byte_off += slot.stride_bytes;
        }
    }

    for (size_t block_idx = 0; block_idx < request_mem_blocks.size(); ++block_idx) {
        const auto mem_bufs = memory_pool->convertIndexToBuffer(0, request_mem_blocks[block_idx]);
        ASSERT_EQ(mem_bufs.size(), 1u);
        const auto& mem_buffer = mem_bufs[0];

        size_t byte_off = 0;
        for (size_t i = 0; i < slots.size(); ++i) {
            const auto& slot = slots[i];
            const char  tag  = copyTag(1000 + block_idx * slots.size() + i);
            const auto  gpu_bufs =
                allocator->convertIndexToBuffer(slot.layer_id, slot.region_name, gpu_block_sets[block_idx][i]);
            setBlockInfosContent(gpu_bufs, 0);
            setBlockBytes(mem_buffer, byte_off, sumBlockInfosBytes(gpu_bufs), tag);
            byte_off += slot.stride_bytes;
        }
    }

    ASSERT_TRUE(connector->tryCopyCacheWithStagedMemoryCopy(req, KVCacheMemoryConnector::CopyDirection::H2D, slots));

    for (size_t block_idx = 0; block_idx < request_mem_blocks.size(); ++block_idx) {
        for (size_t i = 0; i < slots.size(); ++i) {
            const auto& slot = slots[i];
            const auto  gpu_bufs =
                allocator->convertIndexToBuffer(slot.layer_id, slot.region_name, gpu_block_sets[block_idx][i]);
            verifyBlockInfosContent(gpu_bufs, copyTag(1000 + block_idx * slots.size() + i));
        }
    }
}

TEST(KVCacheBatchedMemoryCopyTest, Dsv4TypedLayoutUsesStagedCopyForD2HAndH2D) {
    runDsv4TypedStagedCopyRoundTrip({});
}

TEST(KVCacheBatchedMemoryCopyTest, Dsv4TypedStagedCopySupportsHostBackedStateRegions) {
    runDsv4TypedStagedCopyRoundTrip(
        {KVCacheRegionName::INDEXER_STATE, KVCacheRegionName::CSA_STATE, KVCacheRegionName::HCA_STATE});
}

class FailWriteDiskIO final: public IDiskBlockIO {
public:
    explicit FailWriteDiskIO(std::unique_ptr<IDiskBlockIO> io): io_(std::move(io)) {}
    bool openAndPreallocate(const std::string&, size_t, bool) override {
        return false;
    }
    bool read(uint64_t offset, void* dst, size_t bytes) override {
        return io_->read(offset, dst, bytes);
    }
    bool write(uint64_t, const void*, size_t) override {
        return false;
    }
    void close() override {
        io_->close();
    }
    std::string debugString() const override {
        return "injected write failure";
    }
    std::unique_ptr<IDiskBlockIO> io_;
};

TEST(KVCacheBatchedMemoryCopyTest, CrcRoundtripRejectsCorruptionAcrossBackingsAndLayouts) {
    if (!CrcBlockCopy::supported()) {
        GTEST_SKIP() << "CRC backend unavailable";
    }
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    struct EnvironmentGuard {
        const char* name    = "RTP_LLM_PIN_HOST_BLOCK_POOL";
        bool        present = std::getenv(name) != nullptr;
        std::string value   = present ? std::getenv(name) : "";
        ~EnvironmentGuard() {
            if (present)
                setenv(name, value.c_str(), 1);
            else
                unsetenv(name);
        }
    } guard;
    for (bool prefix : {false, true}) {
        for (bool pinned : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "prefix=" << prefix << " pinned=" << pinned);
            setenv(guard.name, pinned ? "1" : "0", 1);
            char directory[] = "/tmp/rtp-crc-integration-XXXXXX";
            ASSERT_NE(mkdtemp(directory), nullptr);
            const auto    dump_path = std::string(directory) + "/dump";
            const auto    config    = makeCompactDsv4TypedMemoryCopyConfig(/*use_flash=*/true);
            KVCacheConfig kv;
            kv.enable_memory_cache             = true;
            kv.memory_cache_size_mb            = 64;
            kv.enable_prefix_tree_memory_cache = prefix;
            kv.enable_memory_cache_sm_copy     = pinned;  // CRC dispatch must cover both settings.
            kv.enable_memory_cache_disk        = true;
            kv.memory_cache_disk_paths         = directory;
            kv.memory_cache_disk_size_mb       = 64;
            auto allocator                     = std::make_shared<FakeTypedKVCacheAllocator>(
                config,
                8,
                std::set<KVCacheRegionName>{
                    KVCacheRegionName::INDEXER_STATE, KVCacheRegionName::CSA_STATE, KVCacheRegionName::HCA_STATE});
            auto connector = std::make_shared<KVCacheMemoryConnector>(
                config, kv, allocator, std::vector<std::string>{"127.0.0.1:1"});
            ASSERT_TRUE(connector->crc_enabled_);  // The integration matrix must use automatic CRC selection.
            connector->crc_dump_path_ = dump_path;
            ASSERT_TRUE(connector->init());
            ASSERT_EQ(connector->crc_copy_slots_.size(), KVCacheMemoryConnector::kCopyThreadCount);
            std::vector<const void*> workspaces;
            for (const auto& slot : connector->crc_copy_slots_)
                workspaces.push_back(slot.copy->impl_.get());
            const auto expired = dump_path + "/crc_rank0_expired";
            std::filesystem::create_directories(expired);
            const int expired_fd = ::open((expired + "/cpu.bin").c_str(), O_CREAT | O_WRONLY | O_EXCL, 0600);
            ASSERT_GE(expired_fd, 0);
            const int truncate_result = ::ftruncate(expired_fd, 2LL * 1024 * 1024 * 1024);
            ::close(expired_fd);
            ASSERT_EQ(truncate_result, 0);
            ASSERT_TRUE(std::filesystem::exists(expired));
            const auto                        slots = connector->layerRegionSlots();
            const std::vector<CacheBlockKind> kinds =
                prefix ? std::vector<CacheBlockKind>{CacheBlockKind::COMPRESSED_KV, CacheBlockKind::STATE_SWA_KV} :
                         std::vector<CacheBlockKind>{CacheBlockKind::COMPLETE};
            for (auto kind : kinds) {
                for (bool disk : {false, true}) {
                    SCOPED_TRACE(::testing::Message() << "kind=" << int(kind) << " disk=" << disk);
                    auto       memory_pool = connector->memoryPoolFor(kind);
                    auto       disk_pool   = connector->diskPoolFor(kind);
                    const auto allocated   = memory_pool->malloc(2);
                    const auto disk_slot   = disk_pool->malloc();
                    ASSERT_EQ(allocated.size(), 2u);
                    ASSERT_TRUE(disk_slot.has_value());
                    const size_t payload_bytes =
                        prefix ? connector->prefixKindBlockSize(kind, slots) : connector->complete_block_size_;
                    auto verify_writer = [&](const void* data, int64_t expected) {
                        CrcBlockHostMetadata metadata;
                        std::memcpy(&metadata,
                                    static_cast<const uint8_t*>(data) + CrcBlockCopy::transferBytes(payload_bytes),
                                    sizeof(metadata));
                        EXPECT_EQ(metadata.writer_request_id, expected);
                    };
                    MemoryOperationRequestPB request;
                    request.set_request_id(1001);
                    request.set_copy_direction(MemoryOperationRequestPB::D2H);
                    auto* item = request.add_copy_items();
                    item->set_cache_key(101);
                    item->set_is_complete(true);
                    if (prefix) {
                        item->set_cache_block_kind(kind == CacheBlockKind::COMPRESSED_KV ?
                                                       MemoryOperationRequestPB::COMPRESSED_KV :
                                                       MemoryOperationRequestPB::STATE_SWA_KV);
                    }
                    item->set_backing_type(disk ? MemoryOperationRequestPB::DISK : MemoryOperationRequestPB::MEMORY);
                    item->set_mem_block(disk ? NULL_BLOCK_IDX : allocated[0]);
                    if (disk)
                        item->set_disk_slot(*disk_slot);
                    auto included = [&](size_t i) { return !prefix || connector->kindForSlot(slots[i]) == kind; };
                    for (size_t i = 0; i < slots.size(); ++i) {
                        item->add_gpu_blocks(included(i) ? 7 : NULL_BLOCK_IDX);
                        if (included(i)) {
                            setBlockInfosContent(
                                allocator->convertIndexToBuffer(slots[i].layer_id, slots[i].region_name, 7),
                                copyTag(i));
                        }
                    }
                    MemoryOperationResponsePB response;
                    ASSERT_TRUE(connector->copyCache(request, response));
                    ASSERT_TRUE(response.success());
                    for (size_t i = 0; i < slots.size(); ++i) {
                        if (included(i)) {
                            setBlockInfosContent(
                                allocator->convertIndexToBuffer(slots[i].layer_id, slots[i].region_name, 7), 'x');
                        }
                    }
                    request.set_copy_direction(MemoryOperationRequestPB::H2D);
                    request.set_request_id(2001);
                    response.Clear();
                    ASSERT_TRUE(connector->copyCache(request, response));
                    ASSERT_TRUE(response.success());
                    for (size_t i = 0; i < slots.size(); ++i) {
                        if (included(i)) {
                            const auto buffers =
                                allocator->convertIndexToBuffer(slots[i].layer_id, slots[i].region_name, 7);
                            verifyBlockInfosContent(buffers, copyTag(i));
                            setBlockInfosContent(buffers, 'q');
                        }
                    }
                    if (!disk)
                        verify_writer(memory_pool->convertIndexToBuffer(0, allocated[0])[0].addr, 1001);
                    MemoryOperationRequestPB merge = request;
                    merge.set_request_id(1002);
                    auto* merged = merge.mutable_copy_items(0);
                    merged->set_src_backing_type(item->backing_type());
                    if (disk)
                        merged->set_src_disk_slot(*disk_slot);
                    else
                        merged->set_src_mem_block(allocated[0]);
                    merged->set_backing_type(MemoryOperationRequestPB::MEMORY);
                    merged->clear_disk_slot();
                    merged->set_mem_block(allocated[1]);
                    size_t first_slot = 0;
                    while (!included(first_slot))
                        ++first_slot;
                    for (size_t i = 0; i < slots.size(); ++i) {
                        merged->set_gpu_blocks(i, i == first_slot ? 7 : NULL_BLOCK_IDX);
                    }
                    merge.set_copy_direction(MemoryOperationRequestPB::D2H);
                    response.Clear();
                    ASSERT_TRUE(connector->copyCache(merge, response));
                    ASSERT_TRUE(response.success());
                    *merged->mutable_gpu_blocks() = item->gpu_blocks();
                    merge.set_copy_direction(MemoryOperationRequestPB::H2D);
                    merge.set_request_id(2002);
                    response.Clear();
                    ASSERT_TRUE(connector->copyCache(merge, response));
                    ASSERT_TRUE(response.success());
                    for (size_t i = 0; i < slots.size(); ++i) {
                        if (included(i)) {
                            const auto buffers =
                                allocator->convertIndexToBuffer(slots[i].layer_id, slots[i].region_name, 7);
                            verifyBlockInfosContent(buffers, i == first_slot ? 'q' : copyTag(i));
                            setBlockInfosContent(buffers, 'q');
                        }
                    }
                    const auto candidate = memory_pool->convertIndexToBuffer(0, allocated[1])[0];
                    verify_writer(candidate.addr, 1002);  // H2D must preserve the last writer, not the reader ID.
                    const auto candidate_slot = disk_pool->malloc();
                    ASSERT_TRUE(candidate_slot.has_value());
                    auto failed_write = merge;
                    failed_write.set_copy_direction(MemoryOperationRequestPB::D2H);
                    auto* failed_item = failed_write.mutable_copy_items(0);
                    failed_item->set_backing_type(MemoryOperationRequestPB::DISK);
                    failed_item->set_mem_block(NULL_BLOCK_IDX);
                    failed_item->set_disk_slot(*candidate_slot);
                    auto  fail_io     = std::make_unique<FailWriteDiskIO>(std::move(disk_pool->io_));
                    auto* fail_io_ptr = fail_io.get();
                    disk_pool->io_    = std::move(fail_io);
                    response.Clear();
                    const bool handled    = connector->copyCache(failed_write, response);
                    auto       working_io = std::move(fail_io_ptr->io_);
                    disk_pool->io_        = std::move(working_io);
                    ASSERT_TRUE(handled);
                    EXPECT_FALSE(response.success());
                    disk_pool->requestFree(*candidate_slot);
                    // Flip one byte in the persisted CPU payload; no destination slot may be scattered.
                    if (disk) {
                        void* data = nullptr;
                        ASSERT_EQ(posix_memalign(&data, 4096, disk_pool->slotStrideBytes()), 0);
                        std::unique_ptr<void, decltype(&std::free)> owned(data, &std::free);
                        ASSERT_TRUE(disk_pool->read(*disk_slot, data, disk_pool->slotStrideBytes()));
                        verify_writer(data, 1001);
                        static_cast<unsigned char*>(data)[0] ^= 1;
                        ASSERT_TRUE(disk_pool->write(*disk_slot, data, disk_pool->slotStrideBytes()));
                    } else {
                        static_cast<unsigned char*>(memory_pool->convertIndexToBuffer(0, allocated[0])[0].addr)[0] ^= 1;
                    }
                    const auto*                candidate_bytes = static_cast<const uint8_t*>(candidate.addr);
                    const std::vector<uint8_t> candidate_before(candidate_bytes,
                                                                candidate_bytes + candidate.size_bytes);
                    merge.set_copy_direction(MemoryOperationRequestPB::D2H);
                    response.Clear();
                    ASSERT_TRUE(connector->copyCache(merge, response));
                    EXPECT_FALSE(response.success());
                    EXPECT_EQ(std::memcmp(candidate.addr, candidate_before.data(), candidate_before.size()), 0);
                    response.Clear();
                    ASSERT_TRUE(connector->copyCache(request, response));  // RPC handled, cache rejected.
                    EXPECT_FALSE(response.success());
                    for (size_t i = 0; i < slots.size(); ++i) {
                        if (included(i)) {
                            verifyBlockInfosContent(
                                allocator->convertIndexToBuffer(slots[i].layer_id, slots[i].region_name, 7), 'q');
                        }
                    }
                    memory_pool->requestFree(allocated);
                    disk_pool->requestFree(*disk_slot);
                    // The next backing case copies, merges and rejects corruption
                    // after sleep/wake. Fixed CRC workspaces survive host-pool reset.
                    ASSERT_TRUE(connector->releaseMemoryCacheBacking());
                    EXPECT_TRUE(connector->cacheKeys().empty());
                    ASSERT_TRUE(connector->restoreMemoryCacheBacking());
                    EXPECT_EQ(memory_pool->requestRefBlocksNum(), 0u);
                    EXPECT_EQ(memory_pool->blockCacheRefBlocksNum(), 0u);
                    ASSERT_EQ(connector->crc_copy_slots_.size(), workspaces.size());
                    for (size_t i = 0; i < workspaces.size(); ++i)
                        EXPECT_EQ(connector->crc_copy_slots_[i].copy->impl_.get(), workspaces[i]);
                }
            }
            bool dumps_drained = false;
            for (int poll = 0; poll < 500; ++poll) {
                {
                    std::lock_guard<std::mutex> lock(connector->crc_mutex_);
                    dumps_drained = connector->crc_dump_pending_ == 0;
                }
                if (dumps_drained)
                    break;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            ASSERT_TRUE(dumps_drained);
            connector.reset();
            EXPECT_FALSE(std::filesystem::exists(expired));
            size_t manifests = 0;
            for (const auto& entry : std::filesystem::recursive_directory_iterator(dump_path)) {
                if (entry.path().filename() == "manifest.json")
                    ++manifests;
            }
            EXPECT_EQ(manifests, 2u);
            ASSERT_EQ(::nftw(
                          directory,
                          [](const char* path, const struct stat*, int, struct FTW*) { return ::remove(path); },
                          16,
                          FTW_DEPTH | FTW_PHYS),
                      0);
        }
    }
}

TEST(KVCacheBatchedMemoryCopyTest, BenchmarkFullLogicalProBlockCopyCache) {
    if (!CrcBlockCopy::supported()) {
        GTEST_SKIP() << "CRC backend unavailable";
    }
    if (!std::getenv("RTP_LLM_CRC_LOGICAL_BENCHMARK")) {
        GTEST_SKIP() << "opt-in full logical block benchmark";
    }
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    const char* layout = std::getenv("RTP_LLM_CRC_LOGICAL_LAYOUT");
    ASSERT_TRUE(!layout || std::strcmp(layout, "prefix") == 0 || std::strcmp(layout, "legacy") == 0);
    const bool        prefix    = !layout || std::strcmp(layout, "prefix") == 0;
    const bool        fixed_cpu = std::getenv("RTP_LLM_CRC_LOGICAL_FIXED_CPU") != nullptr;
    ParallelismConfig pc;
    const char*       topology = std::getenv("RTP_LLM_CRC_LOGICAL_TOPOLOGY");
    if (topology) {
        ASSERT_TRUE(std::strcmp(topology, "cp1") == 0 || std::strcmp(topology, "cp8") == 0
                    || std::strcmp(topology, "cp8_sharded") == 0);
        if (std::strcmp(topology, "cp1") != 0) {
            pc.role_type = RoleType::PREFILL;
            pc.tp_size = pc.world_size            = 8;
            pc.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
            pc.prefill_cp_config.kv_cache_sharded = std::strcmp(topology, "cp8_sharded") == 0;
        }
    }
    struct Workload {
        int logical_blocks;
        int workers;
    };
    for (const auto workload : {Workload{1, 1}, Workload{8, 1}, Workload{1, 8}, Workload{8, 8}}) {
        if ((!prefix || fixed_cpu) && (workload.logical_blocks != 1 || workload.workers != 1)) {
            continue;
        }
        // CP8 is a single rank's full backing pair, not an eight-GPU/engine latency measurement.
        // Compute CP alone does not divide the strides: fixed/SWA slicing also requires KV sharding.
        auto      config        = makeRealDsv4TypedMemoryCopyConfig(/*use_flash=*/false, pc);
        const int logical_count = workload.logical_blocks * workload.workers;
        config.block_num        = logical_count + 1;  // Block zero is reserved.
        std::fill(config.group_block_nums.begin(), config.group_block_nums.end(), config.block_num);
        ASSERT_EQ(config.layer_all_num, 61u);
        ASSERT_EQ(config.seq_size_per_block, 128u);
        ASSERT_EQ(config.kernel_seq_size_per_block, 128u);
        config.fixed_pool_uses_pinned_cpu = fixed_cpu;
        std::set<KVCacheRegionName> host_regions;
        if (fixed_cpu) {
            for (auto region : config.group_region_names) {
                if (isDsv4FixedRegion(region)) {
                    host_regions.insert(region);
                }
            }
        }
        auto allocator =
            std::make_shared<FakeTypedKVCacheAllocator>(config, 0, host_regions, /*coalesced_group_storage=*/true);
        std::vector<bool> modes{false, true};
        if (std::getenv("RTP_LLM_CRC_BENCH_REVERSE")) {
            std::reverse(modes.begin(), modes.end());
        }
        int reference_pinned = -1;
        for (const bool crc : modes) {
            KVCacheConfig kv;
            kv.enable_memory_cache             = true;
            kv.enable_prefix_tree_memory_cache = prefix;
            // Reserve enough real CPU backings for the batch, block zero, and any legacy incomplete pool.
            const size_t     state_divisor          = pc.prefill_cp_config.kv_cache_sharded ? pc.tp_size : 1;
            const size_t     expected_state_bytes   = 7025280 / state_divisor;
            const size_t     expected_logical_bytes = 732672 + expected_state_bytes;
            constexpr size_t mib                    = 1024 * 1024;
            const size_t     incomplete_bytes =
                !prefix && config.linear_step > 1 ? CrcBlockCopy::storageBytes(732672) * (config.linear_step - 1) : 0;
            const size_t logical_storage_bytes =
                prefix ? CrcBlockCopy::storageBytes(732672) + CrcBlockCopy::storageBytes(expected_state_bytes) :
                         CrcBlockCopy::storageBytes(expected_logical_bytes);
            kv.memory_cache_size_mb =
                ((logical_count + 2) * (logical_storage_bytes + incomplete_bytes) + mib - 1) / mib;
            auto connector = std::make_shared<KVCacheMemoryConnector>(
                config, kv, pc, allocator, std::vector<std::string>{"127.0.0.1:1"}, nullptr);
            ASSERT_TRUE(connector->crc_enabled_);  // Exercise automatic production selection for new.
            if (!crc) {
                // Test-only baseline override, before pools/workspaces are allocated. No production switch.
                connector->crc_enabled_ = false;
            }
            ASSERT_TRUE(connector->init());
            ASSERT_EQ(connector->usePrefixTreeMemoryCache(), prefix);
            ASSERT_EQ(connector->crc_copy_slots_.size(), crc ? KVCacheMemoryConnector::kCopyThreadCount : 0u);
            const auto slots = connector->layerRegionSlots();
            ASSERT_EQ(slots.size(), 212u);
            const std::vector<CacheBlockKind> kinds =
                prefix ? std::vector<CacheBlockKind>{CacheBlockKind::COMPRESSED_KV, CacheBlockKind::STATE_SWA_KV} :
                         std::vector<CacheBlockKind>{CacheBlockKind::COMPLETE};
            const std::vector<size_t> payload_bytes =
                prefix ? std::vector<size_t>{connector->compressed_block_size_, connector->state_swa_block_size_} :
                         std::vector<size_t>{connector->complete_block_size_};
            auto included = [&](const auto& slot, size_t kind) {
                return !prefix || connector->kindForSlot(slot) == kinds[kind];
            };
            if (prefix) {
                ASSERT_EQ(payload_bytes[0], 732672u);
                ASSERT_EQ(payload_bytes[1], expected_state_bytes);
                ASSERT_EQ(payload_bytes[0] + payload_bytes[1], expected_logical_bytes);
            } else {
                ASSERT_EQ(payload_bytes[0], expected_logical_bytes);
            }
            ASSERT_EQ(config.group_kv_block_stride_bytes[3], 16384u / state_divisor);
            ASSERT_EQ(config.group_kv_block_stride_bytes[4], 65536u / state_divisor);
            ASSERT_EQ(config.group_kv_block_stride_bytes[6], 74880u / state_divisor);
            std::vector<std::shared_ptr<BlockPool>> pools;
            std::vector<BlockIndicesType>           host_blocks(kinds.size());
            for (auto kind : kinds) {
                pools.push_back(connector->memoryPoolFor(kind));
            }
            size_t gpu_payload_bytes = 0, host_payload_bytes = 0, host_tile_count = 0;
            size_t gpu_backing_count = 0, host_backing_count = 0;
            for (size_t kind = 0; kind < kinds.size(); ++kind) {
                ASSERT_NE(pools[kind], nullptr);
                host_blocks[kind] = pools[kind]->malloc(logical_count);
                ASSERT_EQ(host_blocks[kind].size(), static_cast<size_t>(logical_count));
                size_t bytes          = 0;
                size_t tiles          = 0;
                bool   has_host_tiles = false;
                bool   has_gpu_tiles  = false;
                for (const auto& slot : slots) {
                    if (!included(slot, kind)) {
                        continue;
                    }
                    const auto buffers = allocator->convertIndexToBuffer(slot.layer_id, slot.region_name, 1);
                    ASSERT_EQ(sumBlockInfosBytes(buffers), slot.stride_bytes);
                    bytes += sumBlockInfosBytes(buffers);
                    tiles += buffers.size();
                    for (const auto& buffer : buffers) {
                        ASSERT_EQ(buffer.is_cuda, !fixed_cpu || !isDsv4FixedRegion(slot.region_name));
                        if (buffer.is_cuda) {
                            gpu_payload_bytes += buffer.size_bytes;
                            has_gpu_tiles = true;
                        } else {
                            host_payload_bytes += buffer.size_bytes;
                            ++host_tile_count;
                            has_host_tiles = true;
                        }
                    }
                }
                host_backing_count += has_host_tiles;
                gpu_backing_count += has_gpu_tiles;
                ASSERT_EQ(bytes, payload_bytes[kind]);
                ASSERT_EQ(tiles, prefix ? (kind == 0 ? 91u : 121u) : 212u);
                ASSERT_EQ(pools[kind]->where(), pools[0]->where());
            }
            ASSERT_EQ(gpu_payload_bytes + host_payload_bytes, expected_logical_bytes);
            ASSERT_EQ(host_tile_count, fixed_cpu ? 121u : 0u);
            ASSERT_EQ(host_backing_count, fixed_cpu ? 1u : 0u);
            const bool pinned = pools[0]->where() == MemoryType::MEMORY_CPU_PINNED;
            if (reference_pinned < 0) {
                reference_pinned = pinned;
            }
            ASSERT_EQ(reference_pinned, static_cast<int>(pinned));
            std::vector<MemoryOperationRequestPB> requests(workload.workers);
            for (int w = 0; w < workload.workers; ++w) {
                auto& request = requests[w];
                request.set_trace_id("full-logical-pro-benchmark");
                request.set_request_id(1000 + w);
                for (int block = 0; block < workload.logical_blocks; ++block) {
                    const int logical = w * workload.logical_blocks + block;
                    for (size_t kind = 0; kind < kinds.size(); ++kind) {
                        auto* item = request.add_copy_items();
                        item->set_cache_key(100 + logical);
                        item->set_is_complete(true);
                        item->set_backing_type(MemoryOperationRequestPB::MEMORY);
                        item->set_mem_block(host_blocks[kind][logical]);
                        item->set_cache_block_kind(!prefix ? MemoryOperationRequestPB::LEGACY_COMPLETE :
                                                             (kind == 0 ? MemoryOperationRequestPB::COMPRESSED_KV :
                                                                          MemoryOperationRequestPB::STATE_SWA_KV));
                        for (const auto& slot : slots) {
                            item->add_gpu_blocks(included(slot, kind) ? logical + 1 : NULL_BLOCK_IDX);
                        }
                    }
                }
                ASSERT_EQ(static_cast<size_t>(request.copy_items_size()), kinds.size() * workload.logical_blocks);
            }
            auto tag = [](int logical, size_t slot) { return static_cast<char>(1 + (logical * 37 + slot * 17) % 251); };
            auto fill_pool = [&](bool original) {
                for (int logical = 0; logical < logical_count; ++logical) {
                    for (size_t i = 0; i < slots.size(); ++i) {
                        for (const auto& buffer :
                             allocator->convertIndexToBuffer(slots[i].layer_id, slots[i].region_name, logical + 1)) {
                            if (buffer.is_cuda) {
                                ASSERT_EQ(cudaMemsetAsync(
                                              buffer.addr, original ? tag(logical, i) : 0, buffer.size_bytes, nullptr),
                                          cudaSuccess);
                            } else {
                                std::memset(buffer.addr, original ? tag(logical, i) : 0, buffer.size_bytes);
                            }
                        }
                    }
                }
                ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
            };
            auto verify_cpu = [&] {
                for (int logical = 0; logical < logical_count; ++logical) {
                    for (size_t kind = 0; kind < kinds.size(); ++kind) {
                        const auto buffers = pools[kind]->convertIndexToBuffer(0, host_blocks[kind][logical]);
                        ASSERT_EQ(buffers.size(), 1u);
                        size_t offset = 0;
                        for (size_t i = 0; i < slots.size(); ++i) {
                            if (included(slots[i], kind)) {
                                verifyBlockBytesEq(buffers[0], offset, slots[i].stride_bytes, tag(logical, i));
                                offset += slots[i].stride_bytes;
                            }
                        }
                        ASSERT_EQ(offset, payload_bytes[kind]);
                        if (crc) {
                            CrcBlockHostMetadata metadata;
                            std::memcpy(&metadata,
                                        static_cast<const uint8_t*>(buffers[0].addr)
                                            + CrcBlockCopy::transferBytes(payload_bytes[kind]),
                                        sizeof(metadata));
                            EXPECT_EQ(metadata.writer_request_id, 1000 + logical / workload.logical_blocks);
                        }
                    }
                }
            };
            fill_pool(true);
            ASSERT_FALSE(::testing::Test::HasFailure());
            for (const bool h2d : {false, true}) {
                for (auto& request : requests) {
                    request.set_copy_direction(h2d ? MemoryOperationRequestPB::H2D : MemoryOperationRequestPB::D2H);
                }
                if (h2d) {
                    fill_pool(false);  // A skipped GPU scatter or fixed-CPU DMA must fail the readback check.
                }
                // Time the actual RPC handler boundary, including its slot/tile/Torch descriptor preparation.
                // Request construction, allocation, warmup, and correctness readbacks are outside the samples.
                const int               rounds = workload.logical_blocks == 1 ? 300 : 100;
                std::atomic<bool>       ok{true};
                std::mutex              barrier_mutex;
                std::condition_variable barrier_cv;
                int                     arrived = 0, generation = 0;
                auto                    barrier = [&] {
                    std::unique_lock<std::mutex> lock(barrier_mutex);
                    const int                    current = generation;
                    if (++arrived == workload.workers) {
                        arrived = 0;
                        ++generation;
                        barrier_cv.notify_all();
                    } else {
                        barrier_cv.wait(lock, [&] { return generation != current; });
                    }
                };
                std::vector<std::vector<double>> samples(workload.workers);
                std::vector<std::thread>         workers;
                for (int w = 0; w < workload.workers; ++w) {
                    samples[w].reserve(rounds);
                    workers.emplace_back([&, w] {
                        if (cudaSetDevice(0) != cudaSuccess) {
                            ok.store(false);
                        }
                        MemoryOperationResponsePB response;
                        for (int round = -20; round < rounds; ++round) {
                            barrier();
                            if (!ok.load()) {
                                continue;
                            }
                            response.Clear();
                            const auto before  = std::chrono::steady_clock::now();
                            const bool handled = connector->copyCache(requests[w], response);
                            const auto after   = std::chrono::steady_clock::now();
                            if (!handled || !response.success()) {
                                ADD_FAILURE() << "worker=" << w << " crc=" << crc;
                                ok.store(false);
                            }
                            if (round >= 0) {
                                samples[w].push_back(std::chrono::duration<double, std::micro>(after - before).count());
                            }
                        }
                    });
                }
                for (auto& worker : workers) {
                    worker.join();
                }
                ASSERT_TRUE(ok.load());
                verify_cpu();
                if (h2d) {
                    for (int logical = 0; logical < logical_count; ++logical) {
                        for (size_t i = 0; i < slots.size(); ++i) {
                            verifyBlockInfosContent(
                                allocator->convertIndexToBuffer(slots[i].layer_id, slots[i].region_name, logical + 1),
                                tag(logical, i));
                        }
                    }
                }
                ASSERT_FALSE(::testing::Test::HasFailure());
                ASSERT_EQ(connector->crc_copy_slots_.size(), crc ? KVCacheMemoryConnector::kCopyThreadCount : 0u);
                if (!crc && !prefix) {
                    // Old staged copy packs GPU tiles only; CPU fixed-pool tiles use its host-copy path.
                    ASSERT_EQ(connector->staged_copy_scratch_by_device_.size(), 1u);
                    ASSERT_NE(connector->staged_copy_scratch_by_device_.at(0), nullptr);
                    ASSERT_GE(connector->staged_copy_scratch_by_device_.at(0)->device_capacity,
                              gpu_payload_bytes * workload.logical_blocks);
                } else {
                    ASSERT_TRUE(connector->staged_copy_scratch_by_device_.empty());
                }
                std::vector<double> all;
                for (const auto& part : samples) {
                    ASSERT_EQ(part.size(), static_cast<size_t>(rounds));
                    all.insert(all.end(), part.begin(), part.end());
                }
                std::sort(all.begin(), all.end());
                // Successful per-backing explicit calls, counted from the executors, not profiler kernel totals:
                // Old prefix: one memcpy per tile + one sync per backing. Old legacy: one staged launch/sync per call.
                // New D2H: gather + CRC + D2H, one wait per backing. H2D waits for load/CRC validation,
                // then scatters and waits again, including GPU-only backings. Inherited D2H adds a load/CRC wait;
                // this benchmark uses fresh writes. CPU-only backings skip the fused GPU copy launch.
                // nvCOMP may launch more than one kernel internally. No engine collective is added.
                const int cpu_blocks     = kinds.size() * workload.logical_blocks;
                const int explicit_syncs = crc ? cpu_blocks * (h2d ? 2 : 1) : (prefix ? cpu_blocks : 1);
                std::printf(
                    "CRC_LOGICAL_COPY_BENCH boundary=copyCache model=Pro seq=128 cp=%ld kv_sharded=%d "
                    "rank=0 ranks_executed=1 mtp=0 "
                    "layout=%s mode=%s direction=%s pinned=%d fixed_cpu=%d workers=%d logical_blocks_per_call=%d "
                    "cpu_blocks_per_call=%d bytes_per_logical_block=%zu tiles_per_logical_block=%zu "
                    "gpu_pool_bytes_per_logical_block=%zu cpu_pool_bytes_per_logical_block=%zu "
                    "cpu_pool_tiles_per_logical_block=%zu "
                    "bytes_per_call=%zu workspaces=%zu explicit_stream_syncs=%d "
                    "fused_copy_launches=%d nvcomp_crc_calls=%d counts=source "
                    "n=%zu p50_us=%.3f p99_us=%.3f\n",
                    pc.tp_size,
                    pc.prefill_cp_config.kv_cache_sharded,
                    prefix ? "prefix" : "legacy",
                    crc ? "auto_crc" : (prefix ? "original_prefix" : "original_staged"),
                    h2d ? "H2D" : "D2H",
                    pinned,
                    fixed_cpu,
                    workload.workers,
                    workload.logical_blocks,
                    cpu_blocks,
                    expected_logical_bytes,
                    slots.size(),
                    gpu_payload_bytes,
                    host_payload_bytes,
                    host_tile_count,
                    expected_logical_bytes * workload.logical_blocks,
                    connector->crc_copy_slots_.size(),
                    explicit_syncs,
                    crc ? static_cast<int>(gpu_backing_count * workload.logical_blocks) : (!prefix ? 1 : 0),
                    crc ? cpu_blocks : 0,
                    all.size(),
                    all[all.size() / 2],
                    all[(all.size() - 1) * 99 / 100]);
                std::fflush(stdout);
            }
            for (size_t kind = 0; kind < kinds.size(); ++kind) {
                pools[kind]->requestFree(host_blocks[kind]);
            }
        }
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
    const auto slots = connector->layerRegionSlots();
    ASSERT_TRUE(connector->isDsv4TypedCacheLayout(slots));

    KVCacheResource resource;
    resource.initGroups(static_cast<int>(config.group_types.size()),
                        static_cast<int>(config.layer_all_num),
                        config.layer_to_group_id,
                        /*kernel_blocks_per_kv_block=*/1,
                        config.group_types,
                        config.layer_region_to_group_id);
    resource.resizeBlocks(/*reserver_blocks=*/2, NULL_BLOCK_IDX);

    for (int gid = 0; gid <= 2; ++gid) {
        resource.mutableBlockIds(gid).setAt(0, static_cast<BlockIdxType>(10 + gid));
    }
    resource.mutableBlockIds(0).setAt(1, 0);
    resource.mutableBlockIds(6).setAt(1, 66);

    const auto layer_attn_blocks = connector->resourceLayerRegionBlocks(resource, slots);

    EXPECT_TRUE(connector->kindRequiredAt(layer_attn_blocks, slots, 0, CacheBlockKind::COMPRESSED_KV));
    EXPECT_FALSE(connector->kindRequiredAt(layer_attn_blocks, slots, 0, CacheBlockKind::STATE_SWA_KV));
    EXPECT_FALSE(connector->kindRequiredAt(layer_attn_blocks, slots, 1, CacheBlockKind::COMPRESSED_KV));
    EXPECT_TRUE(connector->kindRequiredAt(layer_attn_blocks, slots, 1, CacheBlockKind::STATE_SWA_KV));

    const auto compressed_mask =
        connector->prefixSlotValidMask(layer_attn_blocks, slots, 0, CacheBlockKind::COMPRESSED_KV);
    ASSERT_EQ(compressed_mask.size(), slots.size());
    for (size_t i = 0; i < slots.size(); ++i) {
        const bool expected = slots[i].group_id >= 0 && slots[i].group_id <= 2;
        EXPECT_EQ(compressed_mask[i] != 0, expected) << i;
    }

    const auto state_mask = connector->prefixSlotValidMask(layer_attn_blocks, slots, 1, CacheBlockKind::STATE_SWA_KV);
    ASSERT_EQ(state_mask.size(), slots.size());
    for (size_t i = 0; i < slots.size(); ++i) {
        EXPECT_EQ(state_mask[i] != 0, slots[i].group_id == 6) << i;
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
    connector->crc_enabled_ = false;  // Preserve coverage of the original CPU payload format.
    ASSERT_TRUE(connector->init());
    ASSERT_TRUE(connector->usePrefixTreeMemoryCache());

    const auto slots = connector->layerRegionSlots();
    ASSERT_TRUE(connector->isDsv4TypedCacheLayout(slots));
    for (const auto& slot : slots) {
        ASSERT_NE(slot.region_name, KVCacheRegionName::HCA_STATE);
    }

    const int   hca_layer        = 3;
    const auto& hca_layer_groups = config.layer_region_to_group_id[static_cast<size_t>(hca_layer)];
    ASSERT_EQ(hca_layer_groups[static_cast<size_t>(KVCacheRegionName::HCA_KV)], 1);
    ASSERT_EQ(hca_layer_groups[static_cast<size_t>(KVCacheRegionName::HCA_STATE)], 5);
    ASSERT_EQ(hca_layer_groups[static_cast<size_t>(KVCacheRegionName::SWA_KV)], 6);

    KVCacheResource resource;
    resource.cacheKeys() = {901, 902};
    resource.initGroups(static_cast<int>(config.group_types.size()),
                        static_cast<int>(config.layer_all_num),
                        config.layer_to_group_id,
                        /*kernel_blocks_per_kv_block=*/1,
                        config.group_types,
                        config.layer_region_to_group_id);
    resource.resizeBlocks(/*reserver_blocks=*/2, NULL_BLOCK_IDX);

    resource.mutableBlockIds(hca_layer, KVCacheRegionName::HCA_KV).assign({11, 12});
    resource.mutableBlockIds(hca_layer, KVCacheRegionName::HCA_STATE).assign({51, 52});
    resource.mutableBlockIds(hca_layer, KVCacheRegionName::SWA_KV).assign({61, NULL_BLOCK_IDX});
    resource.ensureLinearBlockDependencies();

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

    auto slot_index = [&](KVCacheRegionName region_name) -> size_t {
        for (size_t i = 0; i < slots.size(); ++i) {
            if (slots[i].layer_id == hca_layer && slots[i].region_name == region_name) {
                return i;
            }
        }
        return slots.size();
    };
    const size_t hca_kv_slot = slot_index(KVCacheRegionName::HCA_KV);
    const size_t swa_slot    = slot_index(KVCacheRegionName::SWA_KV);
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
    connector->crc_enabled_ = false;  // Preserve coverage of the original CPU payload format.
    ASSERT_TRUE(connector->init());
    ASSERT_TRUE(connector->usePrefixTreeMemoryCache());

    const auto slots = connector->layerRegionSlots();
    ASSERT_TRUE(connector->isDsv4TypedCacheLayout(slots));

    const int       hca_layer = 3;
    KVCacheResource resource;
    resource.cacheKeys() = {901, 902};
    resource.initGroups(static_cast<int>(config.group_types.size()),
                        static_cast<int>(config.layer_all_num),
                        config.layer_to_group_id,
                        /*kernel_blocks_per_kv_block=*/1,
                        config.group_types,
                        config.layer_region_to_group_id);
    resource.resizeBlocks(/*reserver_blocks=*/2, NULL_BLOCK_IDX);
    resource.mutableBlockIds(hca_layer, KVCacheRegionName::HCA_KV).assign({11, 12});
    resource.mutableBlockIds(hca_layer, KVCacheRegionName::SWA_KV).assign({61, 62});
    resource.ensureLinearBlockDependencies();

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
    connector->crc_enabled_ = false;  // Preserve coverage of the original CPU payload format.
    ASSERT_TRUE(connector->init());

    const auto slots = connector->layerRegionSlots();
    ASSERT_TRUE(connector->isDsv4TypedCacheLayout(slots));

    const int       hca_layer = 3;
    KVCacheResource resource;
    resource.cacheKeys() = {901, 902};
    resource.initGroups(static_cast<int>(config.group_types.size()),
                        static_cast<int>(config.layer_all_num),
                        config.layer_to_group_id,
                        /*kernel_blocks_per_kv_block=*/1,
                        config.group_types,
                        config.layer_region_to_group_id);
    resource.resizeBlocks(/*reserver_blocks=*/2, NULL_BLOCK_IDX);
    resource.mutableBlockIds(hca_layer, KVCacheRegionName::HCA_KV).assign({0, NULL_BLOCK_IDX});
    resource.mutableBlockIds(hca_layer, KVCacheRegionName::SWA_KV).assign({61, 62});
    resource.ensureLinearBlockDependencies();

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
    connector->crc_enabled_ = false;  // Preserve coverage of the original CPU payload format.
    ASSERT_TRUE(connector->init());

    const auto          slots = connector->layerRegionSlots();
    std::vector<size_t> state_slots;
    for (size_t i = 0; i < slots.size(); ++i) {
        if (connector->kindForSlot(slots[i]) == CacheBlockKind::STATE_SWA_KV) {
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
            if (connector->kindForSlot(slots[slot_idx]) != CacheBlockKind::STATE_SWA_KV) {
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
            if (connector->kindForSlot(slots[slot_idx]) != CacheBlockKind::STATE_SWA_KV) {
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
    setBlockInfosContent(allocator->convertIndexToBuffer(valid_slot.layer_id, valid_slot.region_name, valid_gpu_block),
                         'V');
    setBlockInfosContent(allocator->convertIndexToBuffer(slots[state_slots[1]].layer_id,
                                                         slots[state_slots[1]].region_name,
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
    for (size_t i = 0; i < slots.size(); ++i) {
        if (i == state_slots[0]) {
            item->add_gpu_blocks(valid_gpu_block);
        } else if (i == state_slots[1]) {
            item->add_gpu_blocks(0);
        } else {
            item->add_gpu_blocks(NULL_BLOCK_IDX);
        }
        item->add_slot_valid_mask(i == state_slots[0] || i == state_slots[1] || i == state_slots[2] ? 1 : 0);
    }

    MemoryOperationResponsePB response;
    ASSERT_TRUE(connector->copyCache(request, response));
    EXPECT_TRUE(response.success());
    verify_prefix_slot(mem_block, state_slots[0], 'V');
    verify_prefix_slot(mem_block, state_slots[1], 'M');
    verify_prefix_slot(mem_block, state_slots[2], 'M');

    set_prefix_slot(mem_block, state_slots[0], 'A');
    set_prefix_slot(mem_block, state_slots[1], 'B');
    set_prefix_slot(mem_block, state_slots[2], 'C');
    setBlockInfosContent(allocator->convertIndexToBuffer(valid_slot.layer_id, valid_slot.region_name, valid_gpu_block),
                         'x');
    setBlockInfosContent(allocator->convertIndexToBuffer(slots[state_slots[1]].layer_id,
                                                         slots[state_slots[1]].region_name,
                                                         /*block_id=*/0),
                         'z');

    request.set_copy_direction(MemoryOperationRequestPB::H2D);
    response.Clear();
    ASSERT_TRUE(connector->copyCache(request, response));
    EXPECT_TRUE(response.success());
    verifyBlockInfosContent(
        allocator->convertIndexToBuffer(valid_slot.layer_id, valid_slot.region_name, valid_gpu_block), 'A');
    verifyBlockInfosContent(
        allocator->convertIndexToBuffer(slots[state_slots[1]].layer_id, slots[state_slots[1]].region_name, 0), 'z');
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
    connector->crc_enabled_ = false;  // Preserve coverage of the original CPU payload format.
    ASSERT_TRUE(connector->init());
    ASSERT_TRUE(connector->usePrefixTreeMemoryCache());

    const auto          slots = connector->layerRegionSlots();
    std::vector<size_t> state_slots;
    for (size_t i = 0; i < slots.size(); ++i) {
        if (connector->kindForSlot(slots[i]) == CacheBlockKind::STATE_SWA_KV) {
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
            if (connector->kindForSlot(slots[slot_idx]) != CacheBlockKind::STATE_SWA_KV) {
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
            if (connector->kindForSlot(slots[slot_idx]) != CacheBlockKind::STATE_SWA_KV) {
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
    setBlockInfosContent(allocator->convertIndexToBuffer(new_slot.layer_id, new_slot.region_name, new_gpu_block), 'N');

    MemoryOperationRequestPB request;
    request.set_copy_direction(MemoryOperationRequestPB::D2H);
    auto* item = request.add_copy_items();
    item->set_mem_block(new_block);
    item->set_src_mem_block(old_block);
    item->set_backing_type(MemoryOperationRequestPB::MEMORY);
    item->set_cache_block_kind(MemoryOperationRequestPB::STATE_SWA_KV);
    item->set_is_complete(true);
    for (size_t i = 0; i < slots.size(); ++i) {
        item->add_gpu_blocks(i == state_slots[1] ? new_gpu_block : NULL_BLOCK_IDX);
        item->add_slot_valid_mask(i == state_slots[0] || i == state_slots[1] ? 1 : 0);
    }

    MemoryOperationResponsePB response;
    ASSERT_TRUE(connector->copyCache(request, response));
    EXPECT_TRUE(response.success());
    verify_prefix_slot(new_block, state_slots[0], 'O');
    verify_prefix_slot(new_block, state_slots[1], 'N');
}

TEST(KVCacheBatchedMemoryCopyTest, CrcPartialCommitConflictsVerifyAllWorkersAndReleaseReferences) {
    if (!CrcBlockCopy::supported()) {
        GTEST_SKIP() << "CRC backend unavailable";
    }
    using Plan      = KVCacheMemoryConnector::CopyPlan;
    using Info      = KVCacheMemoryConnector::CopyInfoPerKey;
    const auto kind = CacheBlockKind::STATE_SWA_KV;
    for (const std::string scenario : {"disjoint",
                                       "overlap",
                                       "shared_source",
                                       "removed_disk_source",
                                       "dirty_source",
                                       "disk_write_failure",
                                       "business_copy_rejected"}) {
        SCOPED_TRACE(scenario);
        const bool shared_source = scenario == "shared_source" || scenario == "removed_disk_source";
        const bool disk_enabled  = scenario == "removed_disk_source" || scenario == "disk_write_failure";
        char       directory[]   = "/tmp/rtp-crc-commit-XXXXXX";
        ASSERT_NE(mkdtemp(directory), nullptr);
        // Connectors retain configuration references, including those owned by RPC handlers.
        const auto config = makeCompactDsv4TypedMemoryCopyConfig(true);
        KVCacheConfig worker_configs[2];
        std::vector<std::unique_ptr<TestRpcServer>> servers;
        std::vector<TestRpcService*>                services;
        std::vector<std::string>                    addrs;
        for (int worker = 0; worker < 2; ++worker) {
            auto service = std::make_unique<TestRpcService>();
            services.push_back(service.get());
            auto server = std::make_unique<TestRpcServer>(std::move(service));
            ASSERT_TRUE(server->start());
            addrs.push_back("127.0.0.1:" + std::to_string(server->listenPort()));
            servers.push_back(std::move(server));
        }
        std::vector<std::shared_ptr<FakeTypedKVCacheAllocator>> allocators;
        std::vector<std::shared_ptr<KVCacheMemoryConnector>>    workers;
        std::atomic<int>                                        rpc_calls[2]{};
        std::atomic<bool>                                       copy_failure[2]{};
        std::atomic<bool>                                       reject_copy{false};
        for (int worker = 0; worker < 2; ++worker) {
            auto& kv = worker_configs[worker];
            kv.enable_memory_cache             = true;
            kv.enable_prefix_tree_memory_cache = true;
            kv.memory_cache_size_mb            = 64;
            kv.memory_cache_sync_timeout_ms    = 10000;
            kv.enable_memory_cache_disk        = disk_enabled;
            kv.memory_cache_disk_size_mb       = 64;
            kv.memory_cache_disk_paths         = std::string(directory) + "/worker" + std::to_string(worker);
            std::filesystem::create_directories(kv.memory_cache_disk_paths);
            auto              allocator = std::make_shared<FakeTypedKVCacheAllocator>(config);
            ParallelismConfig parallel;
            parallel.world_rank = worker;
            parallel.tp_rank    = worker;
            parallel.tp_size    = 2;
            parallel.world_size = 2;
            auto connector      = std::make_shared<KVCacheMemoryConnector>(config, kv, parallel, allocator, addrs);
            connector->crc_dump_path_ = kv.memory_cache_disk_paths + "/dump";
            ASSERT_TRUE(connector->crc_enabled_);
            ASSERT_TRUE(connector->init());
            services[worker]->setMemoryHandler(
                [&, worker, connector](const MemoryOperationRequestPB& req, MemoryOperationResponsePB& resp) {
                    ++rpc_calls[worker];
                    if (worker == 1 && reject_copy) {
                        // A completed business failure uses the same response as a failed CRC or disk write.
                        resp.set_success(false);
                    } else {
                        connector->copyCache(req, resp);
                    }
                    if (!resp.success())
                        copy_failure[worker] = true;
                });
            allocators.push_back(allocator);
            workers.push_back(connector);
        }
        auto                connector = workers[0];
        const auto          slots     = connector->layerRegionSlots();
        std::vector<size_t> state_slots;
        for (size_t i = 0; i < slots.size(); ++i)
            if (connector->kindForSlot(slots[i]) == kind)
                state_slots.push_back(i);
        ASSERT_GE(state_slots.size(), 3u);
        auto               pool             = connector->memoryPoolFor(kind);
        const auto         free_before      = pool->freeBlocksNum();
        const auto         request_before   = pool->requestRefBlocksNum();
        const auto         disk_free_before = disk_enabled ? connector->diskPoolFor(kind)->freeSlots() : 0;
        const CacheKeyType key              = 903;
        auto prepare = [&](std::vector<size_t> selected, int gpu_block, bool disk) -> std::shared_ptr<Plan> {
            Info info;
            info.cache_key  = key;
            info.kind       = kind;
            info.block_size = connector->prefixKindBlockSize(kind, slots);
            info.slot_valid_mask.assign(slots.size(), 0);
            info.gpu_blocks.assign(slots.size(), NULL_BLOCK_IDX);
            for (auto slot : selected) {
                info.slot_valid_mask[slot] = 1;
                info.gpu_blocks[slot]      = gpu_block;
                for (int worker = 0; worker < 2; ++worker)
                    setBlockInfosContent(allocators[worker]->convertIndexToBuffer(
                                             slots[slot].layer_id, slots[slot].region_name, gpu_block),
                                         'A' + worker * 10 + gpu_block);
            }
            if (disk) {
                const auto slot = connector->diskPoolFor(kind)->malloc();
                EXPECT_TRUE(slot.has_value());
                if (!slot)
                    return nullptr;
                info.backing_type = CacheBackingType::DISK;
                info.disk_slot    = *slot;
            } else {
                const auto allocated = pool->malloc(1);
                EXPECT_EQ(allocated.size(), 1u);
                if (allocated.empty())
                    return nullptr;
                info.mem_block = allocated[0];
            }
            std::vector<Info> infos{info};
            if (!connector->preparePrefixMergeSources(infos)) {
                connector->releasePrefixRequestBacking(info);
                ADD_FAILURE() << "source preparation failed";
                return nullptr;
            }
            auto plan        = connector->createCopyPlan(infos, KVCacheMemoryConnector::CopyDirection::D2H);
            plan->trace_id   = "crc-commit-" + scenario;
            plan->request_id = 10000 + gpu_block;
            return plan;
        };
        auto copy = [&](const std::shared_ptr<Plan>& plan) {
            MemoryAsyncContext context({}, true);
            context.setBroadcastResult(connector->sendCopyPlan(plan));
            context.waitDone();
            return context.success();
        };
        if (shared_source) {
            auto seed = prepare({state_slots[0]}, 3, scenario == "removed_disk_source");
            ASSERT_NE(seed, nullptr);
            ASSERT_TRUE(copy(seed));
            connector->putPrefixToCache(seed->copy_infos[0], rootDep(), slots, seed->trace_id, seed->request_id);
        }
        std::vector<size_t> a_slots =
            shared_source ? std::vector<size_t>{state_slots[1]} : std::vector<size_t>{state_slots[0]};
        std::vector<size_t> b_slots = {shared_source ? state_slots[2] : state_slots[1]};
        if (scenario == "overlap") {
            a_slots.push_back(state_slots[1]);
            b_slots.push_back(state_slots[2]);
        }
        // Both writers prepare before either commits, reproducing the overlapping-write ordering.
        auto a = prepare(a_slots, 1, false);
        auto b = prepare(b_slots, 2, scenario == "disk_write_failure");
        ASSERT_NE(a, nullptr);
        ASSERT_NE(b, nullptr);
        ASSERT_TRUE(copy(a));
        ASSERT_TRUE(copy(b));
        connector->putPrefixToCache(a->copy_infos[0], rootDep(), slots, a->trace_id, a->request_id);
        const auto current = connector->prefix_block_cache_->match(key, kind);
        ASSERT_TRUE(current.found);
        if (scenario == "removed_disk_source") {
            const auto removed = connector->prefix_block_cache_->detachIfMatch(
                key, kind, current.backing_type, current.block_index, current.disk_slot, current.generation);
            if (removed)
                connector->releasePrefixCacheBacking(*removed);
        } else if (scenario == "dirty_source") {
            auto buffers = workers[1]->memoryPoolFor(kind)->convertIndexToBuffer(0, current.block_index);
            ASSERT_EQ(buffers.size(), 1u);
            static_cast<uint8_t*>(buffers[0].addr)[0] ^= 1;
        } else if (scenario == "disk_write_failure") {
            auto disk = workers[1]->diskPoolFor(kind);
            disk->io_ = std::make_unique<FailWriteDiskIO>(std::move(disk->io_));
        } else if (scenario == "business_copy_rejected") {
            reject_copy = true;
        }
        const int calls_before = rpc_calls[0];
        connector->putPrefixToCache(b->copy_infos[0], rootDep(), slots, b->trace_id, b->request_id);
        EXPECT_EQ(rpc_calls[0], calls_before + 1);
        EXPECT_EQ(rpc_calls[1], calls_before + 1);
        const auto match = connector->prefix_block_cache_->match(key, kind);
        if (scenario == "dirty_source" || scenario == "disk_write_failure" || scenario == "business_copy_rejected") {
            // Rank 0 receives only success: any completed copy failure invalidates the inherited source.
            // Neither the failed candidate nor the previous source may remain reusable.
            EXPECT_FALSE(match.found);
            EXPECT_FALSE(copy_failure[0]);
            EXPECT_TRUE(copy_failure[1]);
        } else {
            ASSERT_TRUE(match.found);
            EXPECT_GT(match.generation, current.generation);
            std::vector<uint8_t> expected_mask(slots.size(), 0);
            if (scenario != "removed_disk_source") {
                if (shared_source)
                    expected_mask[state_slots[0]] = 1;
                for (auto slot : a_slots)
                    expected_mask[slot] = 1;
            }
            for (auto slot : b_slots)
                expected_mask[slot] = 1;
            EXPECT_EQ(match.slot_valid_mask, expected_mask);
            // Validate the final CRC and restore through both actual RPC workers.
            auto read             = std::make_shared<Plan>();
            read->direction       = KVCacheMemoryConnector::CopyDirection::H2D;
            read->request_id      = 20000;
            Info info             = b->copy_infos[0];
            info.src_mem_block    = NULL_BLOCK_IDX;
            info.src_disk_slot    = -1;
            info.src_backing_type = CacheBackingType::MEMORY;
            for (size_t i = 0; i < slots.size(); ++i)
                info.gpu_blocks[i] = expected_mask[i] ? 4 : NULL_BLOCK_IDX;
            read->copy_infos = {info};
            ASSERT_TRUE(copy(read));
            for (int worker = 0; worker < 2; ++worker) {
                const auto buffers = workers[worker]->memoryPoolFor(kind)->convertIndexToBuffer(0, match.block_index);
                ASSERT_EQ(buffers.size(), 1u);
                CrcBlockHostMetadata metadata;
                std::memcpy(&metadata,
                            static_cast<const uint8_t*>(buffers[0].addr) + CrcBlockCopy::transferBytes(info.block_size),
                            sizeof(metadata));
                EXPECT_EQ(metadata.writer_request_id, b->request_id);  // Includes the conflict-retry RPC.
            }
            for (size_t i = 0; i < slots.size(); ++i) {
                if (!expected_mask[i])
                    continue;
                const bool from_b = std::find(b_slots.begin(), b_slots.end(), i) != b_slots.end();
                const bool from_a = std::find(a_slots.begin(), a_slots.end(), i) != a_slots.end();
                for (int worker = 0; worker < 2; ++worker)
                    verifyBlockInfosContent(
                        allocators[worker]->convertIndexToBuffer(slots[i].layer_id, slots[i].region_name, 4),
                        'A' + worker * 10
                            + (from_b ? 2 :
                               from_a ? 1 :
                                        3));
            }
        }
        a.reset();
        b.reset();
        EXPECT_EQ(pool->requestRefBlocksNum(), request_before);
        if (match.found) {
            EXPECT_EQ(pool->freeBlocksNum(), free_before - 1);
            auto removed = connector->prefix_block_cache_->detachIfMatch(
                key, kind, match.backing_type, match.block_index, match.disk_slot, match.generation);
            ASSERT_TRUE(removed.has_value());
            connector->releasePrefixCacheBacking(*removed);
        }
        EXPECT_EQ(pool->freeBlocksNum(), free_before);
        if (disk_enabled) {
            EXPECT_EQ(connector->diskPoolFor(kind)->freeSlots(), disk_free_before);
        }
        servers.clear();
        connector.reset();
        workers.clear();
        ASSERT_EQ(::nftw(
                      directory,
                      [](const char* path, const struct stat*, int, struct FTW*) { return ::remove(path); },
                      16,
                      FTW_DEPTH | FTW_PHYS),
                  0);
    }
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
    connector->crc_enabled_ = false;  // Preserve coverage of the original CPU payload format.
    ASSERT_TRUE(connector->init());
    ASSERT_TRUE(connector->usePrefixTreeMemoryCache());

    const auto          slots = connector->layerRegionSlots();
    std::vector<size_t> state_slots;
    for (size_t i = 0; i < slots.size(); ++i) {
        if (connector->kindForSlot(slots[i]) == CacheBlockKind::STATE_SWA_KV) {
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
            if (connector->kindForSlot(slots[slot_idx]) != CacheBlockKind::STATE_SWA_KV) {
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
            if (connector->kindForSlot(slots[slot_idx]) != CacheBlockKind::STATE_SWA_KV) {
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
    connector->crc_enabled_ = false;  // Preserve coverage of the original CPU payload format.
    ASSERT_TRUE(connector->init());

    const auto          slots = connector->layerRegionSlots();
    std::vector<size_t> state_slots;
    for (size_t i = 0; i < slots.size(); ++i) {
        if (connector->kindForSlot(slots[i]) == CacheBlockKind::STATE_SWA_KV) {
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
            if (connector->kindForSlot(slots[slot_idx]) != CacheBlockKind::STATE_SWA_KV) {
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
            if (connector->kindForSlot(slots[slot_idx]) != CacheBlockKind::STATE_SWA_KV) {
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
    connector->crc_enabled_ = false;  // Preserve coverage of the original CPU payload format.
    ASSERT_TRUE(connector->init());

    const auto          slots = connector->layerRegionSlots();
    std::vector<size_t> state_slots;
    for (size_t i = 0; i < slots.size(); ++i) {
        if (connector->kindForSlot(slots[i]) == CacheBlockKind::STATE_SWA_KV) {
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
    auto config                        = makeCompactDsv4TypedMemoryCopyConfig(/*use_flash=*/true);
    config.group_kv_block_stride_bytes = std::vector<size_t>(config.group_kv_block_stride_bytes.size(), 3072);
    config.group_kv_scale_stride_bytes = std::vector<size_t>(config.group_kv_scale_stride_bytes.size(), 0);

    KVCacheConfig kv_config;
    kv_config.memory_cache_size_mb                    = 1;
    kv_config.memory_cache_sync_timeout_ms            = 1000;
    kv_config.enable_prefix_tree_memory_cache         = true;
    kv_config.enable_legacy_memory_connector_fallback = false;

    std::vector<std::string> server_addrs = {"127.0.0.1:1"};
    auto                     connector =
        std::make_shared<KVCacheMemoryConnector>(config, kv_config, std::shared_ptr<KVCacheAllocator>(), server_addrs);
    connector->crc_enabled_ = false;  // Preserve coverage of the original CPU payload format.
    ASSERT_TRUE(connector->init());
    ASSERT_TRUE(connector->usePrefixTreeMemoryCache());
    ASSERT_EQ(connector->compressed_pool_->totalBlocksNum(), 1u);
    ASSERT_EQ(connector->state_swa_pool_->totalBlocksNum(), 1u);

    const CacheKeysType cache_keys{101, 102};
    KVCacheResource     resource;
    resource.initGroups(static_cast<int>(config.group_types.size()),
                        static_cast<int>(config.layer_all_num),
                        config.layer_to_group_id,
                        /*kernel_blocks_per_kv_block=*/1,
                        config.group_types,
                        config.layer_region_to_group_id);
    resource.resizeBlocks(static_cast<int>(cache_keys.size()), NULL_BLOCK_IDX);
    resource.setCacheKeys(cache_keys);
    resource.ensureLinearBlockDependencies();

    for (size_t layer = 0; layer < config.layer_region_to_group_id.size(); ++layer) {
        for (size_t region = 0; region < config.layer_region_to_group_id[layer].size(); ++region) {
            const int gid = config.layer_region_to_group_id[layer][region];
            if (gid < 0) {
                continue;
            }
            auto& blocks = resource.mutableBlockIds(static_cast<int>(layer), static_cast<KVCacheRegionName>(region));
            blocks.setAt(0, static_cast<BlockIdxType>(10 + gid));
            blocks.setAt(1, static_cast<BlockIdxType>(20 + gid));
        }
    }

    const auto slots = connector->layerRegionSlots();
    ASSERT_TRUE(connector->isDsv4TypedCacheLayout(slots));
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

// Copyright (c) RTP-LLM

#include <cstring>
#include <map>
#include <memory>
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
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/utils/Logger.h"

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

CacheConfig makeRealDsv4TypedMemoryCopyConfig(bool use_flash) {
    auto              mc = use_flash ? makeDsv4FlashModelConfig() : makeDsv4ProModelConfig();
    ParallelismConfig pc;
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
    ASSERT_EQ(mismatch, byte_len) << "mismatch at byte offset " << mismatch << " expect '" << expected << "'";
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
                                       std::set<KVCacheRegionName> host_regions      = {}):
        KVCacheAllocator(config, AllocationType::DEVICE),
        host_regions_(std::move(host_regions)),
        payload_gap_bytes_(payload_gap_bytes) {
        const auto cuda_options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
        const auto host_options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
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
            || static_cast<uint32_t>(block_id) >= config_.block_num) {
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
    kv_config.memory_cache_size_mb         = 64;
    kv_config.memory_cache_sync_timeout_ms = 1000;
    kv_config.enable_prefix_tree_memory_cache = false;

    auto allocator = std::make_shared<FakeTypedKVCacheAllocator>(config, /*payload_gap_bytes=*/8, host_regions);

    std::vector<std::string> server_addrs = {"127.0.0.1:1"};
    auto connector = std::make_shared<KVCacheMemoryConnector>(config, kv_config, allocator, server_addrs);
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

TEST(KVCacheBatchedMemoryCopyTest, PrefixTreeKindRequiredUsesRuntimeNullSlots) {
    auto config = makeCompactDsv4TypedMemoryCopyConfig(/*use_flash=*/true);

    KVCacheConfig kv_config;
    kv_config.memory_cache_size_mb         = 64;
    kv_config.memory_cache_sync_timeout_ms = 1000;

    std::vector<std::string> server_addrs = {"127.0.0.1:1"};
    auto connector = std::make_shared<KVCacheMemoryConnector>(
        config, kv_config, std::shared_ptr<KVCacheAllocator>(), server_addrs);
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
    auto connector = std::make_shared<KVCacheMemoryConnector>(
        config, kv_config, std::shared_ptr<KVCacheAllocator>(), server_addrs);
    ASSERT_TRUE(connector->init());
    ASSERT_TRUE(connector->usePrefixTreeMemoryCache());

    const auto slots = connector->layerRegionSlots();
    ASSERT_TRUE(connector->isDsv4TypedCacheLayout(slots));
    for (const auto& slot : slots) {
        ASSERT_NE(slot.region_name, KVCacheRegionName::HCA_STATE);
    }

    const int hca_layer = 3;
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

TEST(KVCacheBatchedMemoryCopyTest, PrefixTreeD2HMergeSourceKeepsOldSlotsAndOverlaysNewSlots) {
    const auto set_device_rc = cudaSetDevice(0);
    ASSERT_EQ(set_device_rc, cudaSuccess) << cudaGetErrorString(set_device_rc);

    auto config = makeCompactDsv4TypedMemoryCopyConfig(/*use_flash=*/true);

    KVCacheConfig kv_config;
    kv_config.memory_cache_size_mb             = 64;
    kv_config.memory_cache_sync_timeout_ms     = 1000;
    kv_config.enable_prefix_tree_memory_cache  = true;
    kv_config.enable_legacy_memory_connector_fallback = false;

    auto allocator = std::make_shared<FakeTypedKVCacheAllocator>(config);

    std::vector<std::string> server_addrs = {"127.0.0.1:1"};
    auto connector = std::make_shared<KVCacheMemoryConnector>(config, kv_config, allocator, server_addrs);
    ASSERT_TRUE(connector->init());
    ASSERT_TRUE(connector->usePrefixTreeMemoryCache());

    const auto slots = connector->layerRegionSlots();
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

TEST(KVCacheBatchedMemoryCopyTest, PrefixTreeWriteAllocationFailureDoesNotDoubleFreePartialBlocks) {
    auto config = makeCompactDsv4TypedMemoryCopyConfig(/*use_flash=*/true);
    config.group_kv_block_stride_bytes = std::vector<size_t>(config.group_kv_block_stride_bytes.size(), 3072);
    config.group_kv_scale_stride_bytes = std::vector<size_t>(config.group_kv_scale_stride_bytes.size(), 0);

    KVCacheConfig kv_config;
    kv_config.memory_cache_size_mb             = 1;
    kv_config.memory_cache_sync_timeout_ms     = 1000;
    kv_config.enable_prefix_tree_memory_cache  = true;
    kv_config.enable_legacy_memory_connector_fallback = false;

    std::vector<std::string> server_addrs = {"127.0.0.1:1"};
    auto connector = std::make_shared<KVCacheMemoryConnector>(
        config, kv_config, std::shared_ptr<KVCacheAllocator>(), server_addrs);
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
            auto& blocks = resource.mutableBlockIds(
                static_cast<int>(layer), static_cast<KVCacheRegionName>(region));
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

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
    auto              config = HybridPoolConfigCreator::createConfig(mc, pc, kv_config);
    config.block_num         = 512;
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

    constexpr size_t kDsv4PoolNum           = 7;
    config.group_region_names               = {KVCacheRegionName::CSA_KV,
                                               KVCacheRegionName::HCA_KV,
                                               KVCacheRegionName::INDEXER_KV,
                                               KVCacheRegionName::INDEXER_STATE,
                                               KVCacheRegionName::CSA_STATE,
                                               KVCacheRegionName::HCA_STATE,
                                               KVCacheRegionName::SWA_KV};
    config.group_types                      = {CacheGroupType::FULL,
                                               CacheGroupType::FULL,
                                               CacheGroupType::FULL,
                                               CacheGroupType::SWA,
                                               CacheGroupType::SWA,
                                               CacheGroupType::SWA,
                                               CacheGroupType::SWA};
    config.group_kv_block_stride_bytes      = {64, 16, 32, 48, 80, 40, 96};
    config.group_kv_scale_stride_bytes      = std::vector<size_t>(kDsv4PoolNum, 0);
    config.group_seq_size_per_block         = std::vector<size_t>(kDsv4PoolNum, config.seq_size_per_block);
    config.group_block_nums                 = std::vector<uint32_t>(kDsv4PoolNum, config.block_num);
    config.non_full_addition_kvcache_blocks = 0;
    config.layer_ids                        = std::vector<std::vector<int>>(kDsv4PoolNum);
    config.global_layer_ids                 = std::vector<std::vector<int>>(kDsv4PoolNum);
    config.layer_to_group_id                = std::vector<int>(config.layer_all_num, 6);
    config.layer_to_group_ids               = std::vector<std::vector<int>>(config.layer_all_num);
    config.layer_group_types                = std::vector<CacheGroupType>(config.layer_all_num, CacheGroupType::SWA);
    config.layer_region_to_group_id         = std::vector<std::vector<int>>(
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
    explicit FakeTypedKVCacheAllocator(const CacheConfig&               config,
                                       size_t                           payload_gap_bytes = 0,
                                       std::set<KVCacheRegionName>      host_regions = {}):
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
                auto       tensor      = torch::empty({static_cast<int64_t>(config.block_num),
                                                       static_cast<int64_t>(stride)},
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

    auto wrong_tokens_config                      = makeCompactDsv4TypedMemoryCopyConfig(/*use_flash=*/true);
    wrong_tokens_config.seq_size_per_block        = 128;
    wrong_tokens_config.kernel_seq_size_per_block = 128;
    auto wrong_tokens_connector                   = std::make_shared<KVCacheMemoryConnector>(
        wrong_tokens_config, kv_config, std::shared_ptr<KVCacheAllocator>(), server_addrs);
    EXPECT_FALSE(wrong_tokens_connector->isDsv4TypedCacheLayout(wrong_tokens_connector->layerRegionSlots()));

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

}  // namespace rtp_llm::test

int main(int argc, char** argv) {
    rtp_llm::initLogger();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

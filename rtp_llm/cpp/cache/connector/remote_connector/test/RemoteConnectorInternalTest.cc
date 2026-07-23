#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <numeric>
#include <tuple>

#include "rtp_llm/cpp/cache/connector/remote_connector/RemoteConnector.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "autil/EnvUtil.h"

using namespace rtp_llm;
using namespace rtp_llm::remote_connector;

namespace rtp_llm {

namespace remote_connector {
bool operator==(const GroupPolicy::SpecInfo& lhs, const GroupPolicy::SpecInfo& rhs) {
    return lhs.tp_rank == rhs.tp_rank && lhs.tag == rhs.tag;
}
}  // namespace remote_connector

namespace test {
namespace {

KVCacheSpecPtr makeTestMhaSpec(const std::string& tag, uint32_t seq_size_per_block) {
    AttentionConfigs attn_config;
    attn_config.kv_head_num      = 8;
    attn_config.size_per_head    = 128;
    attn_config.tokens_per_block = seq_size_per_block;

    ParallelismConfig parallelism_config;
    parallelism_config.tp_size = 1;

    KVCacheSpecDesc desc;
    desc.tag        = tag;
    desc.cache_type = KVCacheSpecType::MultiHeadAttention;
    desc.dtype      = rtp_llm::DataType::TYPE_FP16;

    SpecBuildContext ctx;
    ctx.dtype              = rtp_llm::DataType::TYPE_FP16;
    ctx.seq_size_per_block = seq_size_per_block;
    ctx.attn_config        = &attn_config;
    ctx.parallelism_config = &parallelism_config;
    return SpecBuilder::build(desc, ctx);
}

KVCacheSpecPtr makeTestLinearSpec(const std::string& tag, uint32_t seq_size_per_block) {
    LinearAttentionConfig linear_config;
    linear_config.linear_conv_kernel_dim = 2;
    linear_config.linear_key_head_dim    = 1;
    linear_config.linear_value_head_dim  = 1;
    linear_config.linear_num_key_heads   = 1;
    linear_config.linear_num_value_heads = 1;

    ParallelismConfig parallelism_config;
    parallelism_config.tp_size = 1;

    KVCacheSpecDesc desc;
    desc.tag        = tag;
    desc.cache_type = KVCacheSpecType::LinearAttention;
    desc.dtype      = rtp_llm::DataType::TYPE_FP16;

    SpecBuildContext ctx;
    ctx.dtype                   = rtp_llm::DataType::TYPE_FP16;
    ctx.seq_size_per_block      = seq_size_per_block;
    ctx.linear_attention_config = &linear_config;
    ctx.parallelism_config      = &parallelism_config;
    return SpecBuilder::build(desc, ctx);
}

}  // namespace

class FakeKVCacheAllocator: public KVCacheAllocator {
public:
    explicit FakeKVCacheAllocator(const CacheConfig& config): KVCacheAllocator(config) {}
    void free(const FreeInfo& free_info) override {
        return;
    }
    void insertIntoCache(const InsertInfo& insert_info) override {
        return;
    }
    BlockAddrInfo convertIndexToAddr(int layer_id, int block_id) const override {
        return {};
    }
    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, int block_id) const override {
        return {};
    }
    std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const override {
        return {};
    }
    std::vector<BlockInfo>
    convertIndexToBufferByTag(int layer_id, const std::string& tag, int block_id) const override {
        tagged_buffer_requests_.emplace_back(layer_id, tag, block_id);
        BlockInfo info;
        info.addr         = reinterpret_cast<void*>(static_cast<uintptr_t>(block_id + 1));
        const auto& group = config_.group(tag);
        info.size_bytes   = tagged_buffer_size_override_ == 0 ?
                                group.kv_block_stride_bytes + group.kv_scale_stride_bytes :
                                tagged_buffer_size_override_;
        return {info};
    }
    GroupedCacheLayerLayout allLayerCacheBase() const override {
        ++all_layer_cache_base_call_count_;
        const auto                            topology = config_.topologyPtr();
        GroupedCacheLayerLayout::GroupLayouts groups;
        for (const auto& group : topology->groups()) {
            groups.emplace(group.tag, CacheLayerLayout(std::vector<BlockBufferPtrInfo>(topology->layers().size())));
        }
        return GroupedCacheLayerLayout(topology, std::move(groups));
    }
    int singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                              int                            seq_len,
                              int                            reserve_step) const override {
        return 0;
    }
    int estimatePeakNeedBlocks(const KVCacheResource& kv_cache_resource,
                               int                    seq_len,
                               int                    remaining_tokens,
                               int                    reserve_step,
                               bool                   enable_reuse_cache) const override {
        return 0;
    }
    int getNeedBlocks(const MallocInfo& malloc_info) const override {
        return 0;
    }
    int estimateInitialBatchPeakNeedBlocks(int  seq_len,
                                           int  common_seq_len,
                                           int  remaining_tokens,
                                           int  reserve_step,
                                           bool enable_reuse_cache,
                                           int  target_batch_size) const override {
        return 0;
    }
    std::shared_ptr<KVCacheResource> incrKVCacheRef(const KVCacheResource& kvcache_resource,
                                                    const CacheKeysType&   cache_keys,
                                                    bool                   is_connector = false) override {
        return {};
    }
    void decrKVCacheRef(const KVCacheResource& kvcache_resource, bool is_connector = false) override {
        return;
    }
    bool updateKVBlock(const BatchKVCacheResourcePtr&  batch_kv_cache_resource,
                       const std::vector<int>&         block_src_batch,
                       bool                            copy_last_block,
                       std::vector<TaggedBlockIdPair>& block_update_mapping) override {
        return false;
    }
    int seqSizePerBlock() const override {
        return 0;
    }

    bool doInit() override {
        return true;
    }

    void regUserMr(size_t model_id) {
        return;
    }
    size_t freeBlocksNum() const {
        return 0;
    }
    size_t availableBlocksNum() const {
        return 0;
    }
    size_t availableTokensNum() const {
        return 0;
    }
    size_t totalBlocksNum() const {
        return 0;
    }

    KVCacheBuffer kvCacheBuffer() const {
        return {};
    }

    void clearCache() {
        return;
    }

    size_t allLayerCacheBaseCallCount() const {
        return all_layer_cache_base_call_count_;
    }

    const std::vector<std::tuple<int, std::string, int>>& taggedBufferRequests() const {
        return tagged_buffer_requests_;
    }

    void clearTaggedBufferRequests() const {
        tagged_buffer_requests_.clear();
    }

    void setTaggedBufferSizeOverride(size_t size_bytes) {
        tagged_buffer_size_override_ = size_bytes;
    }

protected:
    MallocResult incrMalloc(const MallocInfo& malloc_info) override {
        return {};
    }
    MallocResult initMallocForCommonLen(const MallocInfo& malloc_info) override {
        return {};
    }

private:
    mutable size_t                                         all_layer_cache_base_call_count_ = 0;
    mutable std::vector<std::tuple<int, std::string, int>> tagged_buffer_requests_;
    size_t                                                 tagged_buffer_size_override_ = 0;
};

class RemoteConnectorInternalTest: public ::testing::Test {
public:
    void SetUp() override {
        rtp_llm::initLogger();
        auto mha_spec                  = makeTestMhaSpec("0", /*seq_size_per_block=*/8);
        auto linear_spec_1             = makeTestLinearSpec("1", /*seq_size_per_block=*/8);
        auto linear_spec_2             = makeTestLinearSpec("2", /*seq_size_per_block=*/8);
        cache_config_.block_num        = 8;
        cache_config_.layer_num        = layer_num_;
        cache_config_.layer_all_num    = layer_num_;
        byte_size_per_block_           = static_cast<size_t>(mha_spec->block_size_bytes()) * layer_num_;
        cache_config_.block_size_bytes = byte_size_per_block_;
        cache_config_.dtype            = rtp_llm::DataType::TYPE_FP16;
        std::vector<int> layers(layer_num_);
        std::iota(layers.begin(), layers.end(), 0);
        cache_config_.fromGroupedSpecs({mha_spec, linear_spec_1, linear_spec_2},
                                       {layers, layers, layers},
                                       {CacheGroupType::FULL, CacheGroupType::LINEAR, CacheGroupType::LINEAR},
                                       {"0", "1", "2"});
        cache_config_.setGroupBlockLayout(
            {8, 8, 8},
            {mha_spec->block_size_bytes(), mha_spec->block_size_bytes(), mha_spec->block_size_bytes()},
            {0, 0, 0});
    }

    void TearDown() override {}

private:
    std::shared_ptr<RemoteConnector> getFullLinearPolicyConnector() const {
        auto allocator = std::make_shared<FakeKVCacheAllocator>(cache_config_);
        return std::shared_ptr<RemoteConnector>(new RemoteConnector(
            cache_config_, kv_cache_config_, runtime_config_, parallelism_config_, sp_config_, nullptr, 0, allocator));
    }

    CacheConfig                cache_config_;
    KVCacheConfig              kv_cache_config_;
    RuntimeConfig              runtime_config_;
    ParallelismConfig          parallelism_config_;
    SpeculativeExecutionConfig sp_config_;
    size_t                     byte_size_per_block_ = 0;
    constexpr static int       layer_num_           = 10;
};

TEST_F(RemoteConnectorInternalTest, test_genClientConfig) {
    {
        auto connector = getFullLinearPolicyConnector();
        ASSERT_TRUE(connector->group_policy_->init());
        auto config_map = connector->genClientConfig();
        ASSERT_EQ(1, config_map.size());
    }
    {
        auto connector = getFullLinearPolicyConnector();
        ASSERT_TRUE(connector->group_policy_->init());

        auto config_map_1 = connector->genClientConfig();
        autil::EnvUtil::setEnv("BIZ_NAME", "test_biz");
        auto config_map_2 = connector->genClientConfig();
        autil::EnvUtil::unsetEnv("BIZ_NAME");
    }
}

TEST_F(RemoteConnectorInternalTest, test_genLocationSpecInfoMapAndGroups) {
    auto connector = getFullLinearPolicyConnector();
    ASSERT_TRUE(connector->group_policy_->init());

    auto [spec_info_map, spec_groups] = connector->genLocationSpecInfoMapAndGroups(2);
    ASSERT_EQ((std::map<std::string, int64_t>({{"tp0_F0", byte_size_per_block_},
                                               {"tp0_L1", byte_size_per_block_},
                                               {"tp0_L2", byte_size_per_block_},
                                               {"tp1_F0", byte_size_per_block_},
                                               {"tp1_L1", byte_size_per_block_},
                                               {"tp1_L2", byte_size_per_block_}})),
              *spec_info_map);
    EXPECT_EQ((std::map<std::string, std::vector<std::string>>(
                  {{"F0", {"tp0_F0", "tp1_F0"}},
                   {"F0L1L2", {"tp0_F0", "tp1_F0", "tp0_L1", "tp1_L1", "tp0_L2", "tp1_L2"}},
                   {"L1", {"tp0_L1", "tp1_L1"}},
                   {"L2", {"tp0_L2", "tp1_L2"}}})),
              *spec_groups);
    EXPECT_EQ(
        (std::unordered_map<uint64_t, std::string>({{0b111, "F0L1L2"}, {0b100, "L2"}, {0b010, "L1"}, {0b001, "F0"}})),
        connector->group_policy_->location_spec_group_map_);
    EXPECT_EQ((GroupPolicy::SpecInfoMap({{"tp0_F0", GroupPolicy::SpecInfo({0, "0"})},
                                         {"tp0_L1", GroupPolicy::SpecInfo({0, "1"})},
                                         {"tp0_L2", GroupPolicy::SpecInfo({0, "2"})},
                                         {"tp1_F0", GroupPolicy::SpecInfo({1, "0"})},
                                         {"tp1_L1", GroupPolicy::SpecInfo({1, "1"})},
                                         {"tp1_L2", GroupPolicy::SpecInfo({1, "2"})}})),
              connector->group_policy_->spec_name_to_info_);
}

TEST_F(RemoteConnectorInternalTest, PublishesTagLocalHeterogeneousGroupBlockSizes) {
    auto       heterogeneous_config = cache_config_;
    const auto per_layer_bytes      = byte_size_per_block_ / layer_num_;
    heterogeneous_config.setGroupBlockLayout(
        {8, 8, 8}, {per_layer_bytes, per_layer_bytes / 2, per_layer_bytes}, {0, 0, 0});

    auto allocator = std::make_shared<FakeKVCacheAllocator>(heterogeneous_config);
    auto connector = std::shared_ptr<RemoteConnector>(new RemoteConnector(heterogeneous_config,
                                                                          kv_cache_config_,
                                                                          runtime_config_,
                                                                          parallelism_config_,
                                                                          sp_config_,
                                                                          nullptr,
                                                                          0,
                                                                          allocator));
    ASSERT_TRUE(connector->group_policy_->init());
    auto [spec_info_map, spec_groups] = connector->genLocationSpecInfoMapAndGroups(/*tp_size=*/1);
    EXPECT_EQ(spec_info_map->at("tp0_F0"), byte_size_per_block_);
    EXPECT_EQ(spec_info_map->at("tp0_L1"), byte_size_per_block_ / 2);
    EXPECT_EQ(spec_info_map->at("tp0_L2"), byte_size_per_block_);
    EXPECT_EQ(spec_groups->at("F0L1L2"), (std::vector<std::string>{"tp0_F0", "tp0_L1", "tp0_L2"}));
}

TEST_F(RemoteConnectorInternalTest, test_genLocationSpecGroupsScalesLinearly) {
    constexpr size_t linear_group_count = 19;
    constexpr size_t group_count        = linear_group_count + 1;

    CacheConfig config;
    config.block_num        = 8;
    config.layer_num        = group_count;
    config.layer_all_num    = group_count;
    config.dtype            = rtp_llm::DataType::TYPE_FP16;
    auto full_spec          = makeTestMhaSpec("full", /*seq_size_per_block=*/8);
    config.block_size_bytes = full_spec->block_size_bytes();

    std::vector<KVCacheSpecPtr>   specs{full_spec};
    std::vector<std::vector<int>> layer_ids{{0}};
    std::vector<CacheGroupType>   group_types{CacheGroupType::FULL};
    std::vector<std::string>      group_tags{"full"};
    for (size_t i = 0; i < linear_group_count; ++i) {
        const auto group_index = static_cast<int32_t>(i + 1);
        specs.push_back(makeTestLinearSpec("linear" + std::to_string(i), /*seq_size_per_block=*/8));
        layer_ids.push_back({group_index});
        group_types.push_back(CacheGroupType::LINEAR);
        group_tags.push_back("linear" + std::to_string(i));
    }
    config.fromGroupedSpecs(specs, layer_ids, group_types, group_tags);
    config.setGroupBlockLayout(std::vector<uint32_t>(group_count, 8),
                               std::vector<size_t>(group_count, full_spec->block_size_bytes()),
                               std::vector<size_t>(group_count, 0));

    auto allocator = std::make_shared<FakeKVCacheAllocator>(config);
    auto connector = std::shared_ptr<RemoteConnector>(new RemoteConnector(
        config, kv_cache_config_, runtime_config_, parallelism_config_, sp_config_, nullptr, 0, allocator));
    ASSERT_TRUE(connector->group_policy_->init());

    auto [spec_info_map, spec_groups] = connector->genLocationSpecInfoMapAndGroups(/*tp_size=*/1);
    EXPECT_EQ(spec_info_map->size(), group_count);
    EXPECT_EQ(spec_groups->size(), group_count + 1);
    EXPECT_EQ(connector->group_policy_->location_spec_group_map_.size(), group_count + 1);
}

TEST(RemoteConnectorTagIdentityTest, GroupNamesDoNotDependOnNumericGroupOrder) {
    CacheConfig first_config;
    first_config.layer_num     = 1;
    first_config.layer_all_num = 1;
    first_config.fromGroupedSpecs({makeTestMhaSpec("full", 8), makeTestLinearSpec("linear", 8)},
                                  {{0}, {0}},
                                  {CacheGroupType::FULL, CacheGroupType::LINEAR},
                                  {"full", "linear"});
    auto first_allocator = std::make_shared<FakeKVCacheAllocator>(first_config);
    auto first_policy    = std::make_shared<FullLinearLayerGroupPolicy>(
        first_allocator, std::vector<std::string>{"full"}, std::vector<std::string>{"linear"}, 1);
    ASSERT_TRUE(first_policy->init());

    CacheConfig reversed_config;
    reversed_config.layer_num     = 1;
    reversed_config.layer_all_num = 1;
    reversed_config.fromGroupedSpecs({makeTestLinearSpec("linear", 8), makeTestMhaSpec("full", 8)},
                                     {{0}, {0}},
                                     {CacheGroupType::LINEAR, CacheGroupType::FULL},
                                     {"linear", "full"});
    auto reversed_allocator = std::make_shared<FakeKVCacheAllocator>(reversed_config);
    auto reversed_policy    = std::make_shared<FullLinearLayerGroupPolicy>(
        reversed_allocator, std::vector<std::string>{"full"}, std::vector<std::string>{"linear"}, 1);
    ASSERT_TRUE(reversed_policy->init());

    auto names_by_tag = [](const GroupPolicy& policy) {
        std::map<std::string, std::string> result;
        for (const auto& [tag, group] : policy.groups()) {
            result.emplace(tag, group.group_name);
        }
        return result;
    };
    EXPECT_EQ(names_by_tag(*first_policy), names_by_tag(*reversed_policy));
    EXPECT_EQ(names_by_tag(*first_policy),
              (std::map<std::string, std::string>{{"full", "Ffull"}, {"linear", "Llinear"}}));
}

TEST(RemoteConnectorTagIdentityTest, FullOnlyPolicyRoutesSameLayerGroupsByTagWithoutHotPathLayoutLookup) {
    CacheConfig first_config;
    first_config.layer_num     = 1;
    first_config.layer_all_num = 1;
    first_config.fromGroupedSpecs({makeTestMhaSpec("full_a", 8), makeTestMhaSpec("full_b", 8)},
                                  {{0}, {0}},
                                  {CacheGroupType::FULL, CacheGroupType::FULL},
                                  {"full_a", "full_b"});
    auto first_allocator = std::make_shared<FakeKVCacheAllocator>(first_config);
    auto first_policy    = std::make_shared<FullLayerGroupPolicy>(
        first_allocator, std::vector<std::string>{"full_a", "full_b"}, std::vector<std::string>{});
    ASSERT_TRUE(first_policy->init());
    ASSERT_EQ(first_allocator->allLayerCacheBaseCallCount(), 1u);
    EXPECT_EQ(first_policy->groups().at("full_a").tag, "full_a");
    EXPECT_EQ(first_policy->groups().at("full_b").tag, "full_b");
    EXPECT_EQ(first_policy->reachableAggregateMasks(), (std::vector<uint64_t>{0b11}));

    ASSERT_TRUE(first_policy->addSpecInfo("tp0_Ffull_b", "full_b", /*tp_rank=*/0));
    EXPECT_EQ(first_policy->spec_info_map().at("tp0_Ffull_b").tag, "full_b");

    kv_cache_manager::BlockBuffers first_buffers;
    ASSERT_TRUE(first_policy->genBlockBuffersByTag({"full_b", "full_a"}, {7, 9}, first_buffers));
    EXPECT_EQ(first_allocator->taggedBufferRequests(),
              (std::vector<std::tuple<int, std::string, int>>{{0, "full_b", 7}, {0, "full_a", 9}}));
    EXPECT_EQ(first_allocator->allLayerCacheBaseCallCount(), 1u);

    CacheConfig reversed_config;
    reversed_config.layer_num     = 1;
    reversed_config.layer_all_num = 1;
    reversed_config.fromGroupedSpecs({makeTestMhaSpec("full_b", 8), makeTestMhaSpec("full_a", 8)},
                                     {{0}, {0}},
                                     {CacheGroupType::FULL, CacheGroupType::FULL},
                                     {"full_b", "full_a"});
    auto reversed_allocator = std::make_shared<FakeKVCacheAllocator>(reversed_config);
    auto reversed_policy    = std::make_shared<FullLayerGroupPolicy>(
        reversed_allocator, std::vector<std::string>{"full_a", "full_b"}, std::vector<std::string>{});
    ASSERT_TRUE(reversed_policy->init());
    ASSERT_EQ(reversed_allocator->allLayerCacheBaseCallCount(), 1u);
    EXPECT_EQ(reversed_policy->groups().at("full_b").tag, "full_b");
    EXPECT_EQ(reversed_policy->groups().at("full_a").tag, "full_a");

    kv_cache_manager::BlockBuffers reversed_buffers;
    ASSERT_TRUE(reversed_policy->genBlockBuffersByTag({"full_b", "full_a"}, {7, 9}, reversed_buffers));
    EXPECT_EQ(reversed_allocator->taggedBufferRequests(),
              (std::vector<std::tuple<int, std::string, int>>{{0, "full_b", 7}, {0, "full_a", 9}}));
    EXPECT_EQ(reversed_allocator->allLayerCacheBaseCallCount(), 1u);
}

TEST(RemoteConnectorBlockBufferValidationTest, RejectsAllocatorBufferSizeThatDoesNotMatchTopology) {
    CacheConfig config;
    config.layer_num     = 1;
    config.layer_all_num = 1;
    config.fromGroupedSpecs({makeTestMhaSpec("full", 8)}, {{0}}, {CacheGroupType::FULL}, {"full"});

    auto allocator = std::make_shared<FakeKVCacheAllocator>(config);
    auto policy =
        std::make_shared<FullLayerGroupPolicy>(allocator, std::vector<std::string>{"full"}, std::vector<std::string>{});
    ASSERT_TRUE(policy->init());
    allocator->setTaggedBufferSizeOverride(config.topology().groups().at(0).kv_block_stride_bytes + 1);

    kv_cache_manager::BlockBuffers buffers;
    EXPECT_FALSE(policy->genBlockBuffersByTag({"full"}, {7}, buffers));
    EXPECT_TRUE(buffers.empty());
}

TEST(RemoteConnectorTopologyInvariantTest, ConstructorRejectsMissingTopology) {
    CacheConfig                cache_config;
    KVCacheConfig              kv_cache_config;
    RuntimeConfig              runtime_config;
    ParallelismConfig          parallelism_config;
    SpeculativeExecutionConfig sp_config;
    auto                       allocator = std::make_shared<FakeKVCacheAllocator>(cache_config);

    EXPECT_ANY_THROW((void)new RemoteConnector(
        cache_config, kv_cache_config, runtime_config, parallelism_config, sp_config, nullptr, 0, allocator));
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

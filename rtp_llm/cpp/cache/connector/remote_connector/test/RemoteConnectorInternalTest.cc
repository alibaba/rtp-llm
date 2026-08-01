#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <numeric>
#include <tuple>

#include "rtp_llm/cpp/cache/connector/remote_connector/RemoteConnector.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
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
    FakeKVCacheAllocator(const CacheConfig&              config,
                         const std::vector<std::string>& full_group_tags,
                         const std::vector<std::string>& other_group_tags,
                         size_t                          per_group_layer_num):
        KVCacheAllocator(config) {
        (void)full_group_tags;
        (void)other_group_tags;
        (void)per_group_layer_num;
    }
    void free(const FreeInfo& free_info) override {
        return;
    }
    void insertIntoCache(const InsertInfo& insert_info) override {
        return;
    }
    BlockAddrInfo convertIndexToAddr(int layer_id, const std::string& tag, int block_id) const override {
        return {};
    }
    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, const std::string& tag, int block_id) const override {
        tagged_buffer_requests_.emplace_back(layer_id, tag, block_id);
        BlockInfo info;
        info.addr       = reinterpret_cast<void*>(static_cast<uintptr_t>(block_id + 1));
        info.size_bytes = tagged_buffer_size_override_ == 0 ?
                              config_.kvBlockStrideBytesForGroup(tag) + config_.kvScaleStrideBytesForGroup(tag) :
                              tagged_buffer_size_override_;
        return {info};
    }
    std::vector<BlockInfo> convertIndexToBuffer(
        int layer_id, const std::string& tag, int block_id, int partition_count, int partition_id) const override {
        return convertIndexToBuffer(layer_id, tag, block_id);
    }
    BlockPoolPtr blockPool(std::string_view) const override {
        return nullptr;
    }
    GroupedCacheLayerLayout allLayerCacheBase() const override {
        ++all_layer_cache_base_call_count_;
        const auto                            topology = config_.topologyPtr();
        GroupedCacheLayerLayout::GroupLayouts groups;
        for (const auto& group : topology->groups()) {
            std::vector<BlockBufferPtrInfo> layers(topology->layers().size());
            for (const int layer_id : group.layer_ids) {
                layers.at(static_cast<size_t>(layer_id)).kv_addr = torch::empty({1}, torch::kUInt8);
            }
            groups.emplace(group.tag, CacheLayerLayout(std::move(layers)));
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
    std::shared_ptr<KVCacheResource> incrKVCacheRef(const KVCacheResource&  kvcache_resource,
                                                    const CacheKeysByGroup& cache_keys,
                                                    bool                    is_connector = false) override {
        return {};
    }
    void decrKVCacheRef(const KVCacheResource& kvcache_resource, bool is_connector = false) override {
        return;
    }
    bool updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                       const std::vector<int>&        block_src_batch,
                       bool                           copy_last_block,
                       std::vector<GroupBlockIdPair>& block_update_mapping) override {
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
        auto mha_spec               = makeTestMhaSpec("0", /*seq_size_per_block=*/8);
        cache_config_.layer_num     = layer_num_;
        cache_config_.layer_all_num = layer_num_;
        byte_size_per_block_        = static_cast<size_t>(mha_spec->block_size_bytes()) * layer_num_;
        cache_config_.dtype         = rtp_llm::DataType::TYPE_FP16;
        std::vector<int> layers(layer_num_);
        std::iota(layers.begin(), layers.end(), 0);
        setTestTopology(cache_config_,
                        {makeTestGroupForConfig(cache_config_, mha_spec, layers, CacheGroupType::FULL, "0")});
        const auto             topology_groups = cache_config_.topology().groups();
        std::vector<GroupBase> groups(topology_groups.begin(), topology_groups.end());
        for (auto& group : groups) {
            group.block_num             = 8;
            group.kv_block_stride_bytes = mha_spec->block_size_bytes();
            group.kv_scale_stride_bytes = 0;
        }
        cache_config_.setTopology(std::move(groups), cache_config_.topology().layers());
    }

    void TearDown() override {}

private:
    std::shared_ptr<RemoteConnector> getFullLinearPolicyConnector() const {
        std::vector<std::string> full_group_tags({"0"});
        std::vector<std::string> linear_group_tags;
        auto                     allocator =
            std::make_shared<FakeKVCacheAllocator>(cache_config_, full_group_tags, linear_group_tags, layer_num_);
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
    ASSERT_EQ((std::map<std::string, int64_t>({{"tp0_F0", byte_size_per_block_}, {"tp1_F0", byte_size_per_block_}})),
              *spec_info_map);
    EXPECT_EQ((std::map<std::string, std::vector<std::string>>({{"F0", {"tp0_F0", "tp1_F0"}}})), *spec_groups);
    EXPECT_EQ((std::unordered_map<uint64_t, std::string>({{0b1, "F0"}})),
              connector->group_policy_->location_spec_group_map_);
    EXPECT_EQ((GroupPolicy::SpecInfoMap(
                  {{"tp0_F0", GroupPolicy::SpecInfo({0, "0"})}, {"tp1_F0", GroupPolicy::SpecInfo({1, "0"})}})),
              connector->group_policy_->spec_name_to_info_);
}

TEST(RemoteConnectorTagIdentityTest, GroupNamesDoNotDependOnNumericGroupOrder) {
    CacheConfig first_config;
    first_config.layer_num     = 1;
    first_config.layer_all_num = 1;
    setTestTopology(
        first_config,
        {makeTestGroupForConfig(first_config, makeTestMhaSpec("full", 8), {0}, CacheGroupType::FULL, "full"),
         makeTestGroupForConfig(first_config, makeTestLinearSpec("linear", 8), {0}, CacheGroupType::LINEAR, "linear")});
    auto first_allocator = std::make_shared<FakeKVCacheAllocator>(
        first_config, std::vector<std::string>{"full"}, std::vector<std::string>{"linear"}, 1);
    auto first_policy = std::make_shared<FullLinearLayerGroupPolicy>(
        first_allocator, std::vector<std::string>{"full"}, std::vector<std::string>{"linear"}, 1);
    ASSERT_TRUE(first_policy->init());

    CacheConfig reversed_config;
    reversed_config.layer_num     = 1;
    reversed_config.layer_all_num = 1;
    setTestTopology(
        reversed_config,
        {makeTestGroupForConfig(
             reversed_config, makeTestLinearSpec("linear", 8), {0}, CacheGroupType::LINEAR, "linear"),
         makeTestGroupForConfig(reversed_config, makeTestMhaSpec("full", 8), {0}, CacheGroupType::FULL, "full")});
    auto reversed_allocator = std::make_shared<FakeKVCacheAllocator>(
        reversed_config, std::vector<std::string>{"full"}, std::vector<std::string>{"linear"}, 1);
    auto reversed_policy = std::make_shared<FullLinearLayerGroupPolicy>(
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
    setTestTopology(
        first_config,
        {makeTestGroupForConfig(first_config, makeTestMhaSpec("full_a", 8), {0}, CacheGroupType::FULL, "full_a"),
         makeTestGroupForConfig(first_config, makeTestMhaSpec("full_b", 8), {0}, CacheGroupType::FULL, "full_b")});
    auto first_allocator = std::make_shared<FakeKVCacheAllocator>(
        first_config, std::vector<std::string>{"full_a", "full_b"}, std::vector<std::string>{}, 1);
    auto first_policy = std::make_shared<FullLayerGroupPolicy>(
        first_allocator, std::vector<std::string>{"full_a", "full_b"}, std::vector<std::string>{});
    ASSERT_TRUE(first_policy->init());
    ASSERT_EQ(first_allocator->allLayerCacheBaseCallCount(), 1u);
    EXPECT_EQ(first_policy->groups().at("full_a").tag, "full_a");
    EXPECT_EQ(first_policy->groups().at("full_b").tag, "full_b");
    EXPECT_EQ(first_policy->reachableAggregateMasks(), (std::vector<uint64_t>{0b11}));

    ASSERT_TRUE(first_policy->addSpecInfo("tp0_Ffull_b", "full_b", /*tp_rank=*/0));
    EXPECT_EQ(first_policy->spec_info_map().at("tp0_Ffull_b").tag, "full_b");

    kv_cache_manager::BlockBuffers first_buffers;
    ASSERT_TRUE(first_policy->genBlockBuffers({"full_b", "full_a"}, {7, 9}, first_buffers));
    EXPECT_EQ(first_allocator->taggedBufferRequests(),
              (std::vector<std::tuple<int, std::string, int>>{{0, "full_b", 7}, {0, "full_a", 9}}));
    EXPECT_EQ(first_allocator->allLayerCacheBaseCallCount(), 1u);

    CacheConfig reversed_config;
    reversed_config.layer_num     = 1;
    reversed_config.layer_all_num = 1;
    setTestTopology(
        reversed_config,
        {makeTestGroupForConfig(reversed_config, makeTestMhaSpec("full_b", 8), {0}, CacheGroupType::FULL, "full_b"),
         makeTestGroupForConfig(reversed_config, makeTestMhaSpec("full_a", 8), {0}, CacheGroupType::FULL, "full_a")});
    auto reversed_allocator = std::make_shared<FakeKVCacheAllocator>(
        reversed_config, std::vector<std::string>{"full_a", "full_b"}, std::vector<std::string>{}, 1);
    auto reversed_policy = std::make_shared<FullLayerGroupPolicy>(
        reversed_allocator, std::vector<std::string>{"full_a", "full_b"}, std::vector<std::string>{});
    ASSERT_TRUE(reversed_policy->init());
    ASSERT_EQ(reversed_allocator->allLayerCacheBaseCallCount(), 1u);
    EXPECT_EQ(reversed_policy->groups().at("full_a").tag, "full_a");
    EXPECT_EQ(reversed_policy->groups().at("full_b").tag, "full_b");

    kv_cache_manager::BlockBuffers reversed_buffers;
    ASSERT_TRUE(reversed_policy->genBlockBuffers({"full_b", "full_a"}, {7, 9}, reversed_buffers));
    EXPECT_EQ(reversed_allocator->taggedBufferRequests(),
              (std::vector<std::tuple<int, std::string, int>>{{0, "full_b", 7}, {0, "full_a", 9}}));
    EXPECT_EQ(reversed_allocator->allLayerCacheBaseCallCount(), 1u);
}

TEST(RemoteConnectorBlockBufferValidationTest, RejectsAllocatorBufferSizeThatDoesNotMatchTopology) {
    CacheConfig config;
    config.layer_num     = 1;
    config.layer_all_num = 1;
    setTestTopology(config,
                    {makeTestGroupForConfig(config, makeTestMhaSpec("full", 8), {0}, CacheGroupType::FULL, "full")});

    auto allocator =
        std::make_shared<FakeKVCacheAllocator>(config, std::vector<std::string>{"full"}, std::vector<std::string>{}, 1);
    auto policy =
        std::make_shared<FullLayerGroupPolicy>(allocator, std::vector<std::string>{"full"}, std::vector<std::string>{});
    ASSERT_TRUE(policy->init());
    allocator->setTaggedBufferSizeOverride(config.kvBlockStrideBytesForGroup("full") + 1);

    kv_cache_manager::BlockBuffers buffers;
    EXPECT_FALSE(policy->genBlockBuffers({"full"}, {7}, buffers));
    EXPECT_TRUE(buffers.empty());
}

TEST(RemoteConnectorBlockBufferValidationTest, RejectsMisalignedOrInvalidTagsWithoutPartialOutput) {
    CacheConfig config;
    config.layer_num     = 1;
    config.layer_all_num = 1;
    setTestTopology(config,
                    {makeTestGroupForConfig(config, makeTestMhaSpec("full", 8), {0}, CacheGroupType::FULL, "full")});

    auto allocator =
        std::make_shared<FakeKVCacheAllocator>(config, std::vector<std::string>{"full"}, std::vector<std::string>{}, 1);
    auto policy =
        std::make_shared<FullLayerGroupPolicy>(allocator, std::vector<std::string>{"full"}, std::vector<std::string>{});
    ASSERT_TRUE(policy->init());

    kv_cache_manager::BlockBuffers buffers;
    EXPECT_FALSE(policy->genBlockBuffers({"full"}, {}, buffers));
    EXPECT_TRUE(buffers.empty());
    EXPECT_FALSE(policy->genBlockBuffers({""}, {7}, buffers));
    EXPECT_TRUE(buffers.empty());
    EXPECT_FALSE(policy->genBlockBuffers({"unknown"}, {7}, buffers));
    EXPECT_TRUE(buffers.empty());
}

TEST_F(RemoteConnectorInternalTest, CopyCacheRejectsMisalignedOrInvalidRequestColumns) {
    auto connector = getFullLinearPolicyConnector();
    ASSERT_TRUE(connector->group_policy_->init());

    RemoteOperationResponsePB response;
    RemoteOperationRequestPB  misaligned;
    misaligned.set_op(RemoteOpType::REMOTE_OPERATION_READ);
    misaligned.set_trace_id("misaligned");
    misaligned.add_group_tags("0");
    misaligned.add_block_ids(7);
    EXPECT_FALSE(connector->copyCache(misaligned, response));

    RemoteOperationRequestPB empty_tag;
    empty_tag.set_op(RemoteOpType::REMOTE_OPERATION_READ);
    empty_tag.set_trace_id("empty");
    empty_tag.add_group_tags("");
    empty_tag.add_block_ids(7);
    empty_tag.add_uris("uri");
    EXPECT_FALSE(connector->copyCache(empty_tag, response));

    RemoteOperationRequestPB unknown_tag;
    unknown_tag.set_op(RemoteOpType::REMOTE_OPERATION_READ);
    unknown_tag.set_trace_id("unknown");
    unknown_tag.add_group_tags("unknown");
    unknown_tag.add_block_ids(7);
    unknown_tag.add_uris("uri");
    EXPECT_FALSE(connector->copyCache(unknown_tag, response));
}

TEST(RemoteConnectorTopologyInvariantTest, ConstructorRejectsMissingTopology) {
    CacheConfig                cache_config;
    KVCacheConfig              kv_cache_config;
    RuntimeConfig              runtime_config;
    ParallelismConfig          parallelism_config;
    SpeculativeExecutionConfig sp_config;
    auto                       allocator =
        std::make_shared<FakeKVCacheAllocator>(cache_config, std::vector<std::string>{}, std::vector<std::string>{}, 0);

    EXPECT_ANY_THROW((void)new RemoteConnector(
        cache_config, kv_cache_config, runtime_config, parallelism_config, sp_config, nullptr, 0, allocator));
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

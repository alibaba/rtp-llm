// Why the end-to-end hybrid full+linear remote coverage that used to live here is gone:
//
//   * The approved design authorizes remote cache support for a **single FULL cache group
//     only**. Remote-cache validation rejects any other configuration and `RemoteConnector`
//     is hard-wired to `FullLayerGroupPolicy` with an empty other-tags list, so the
//     1-FULL-plus-2-LINEAR remote topology those 19 tests drove is no longer a reachable
//     configuration — they exercised a premise the design now deliberately refuses.
//   * The supported remote path keeps its end-to-end coverage in the sibling target
//     `remote_connector_mock_only_full_test` (RemoteConnectorMockOnlyFullTest.cc).
//   * `FullLinearLayerGroupPolicy` and `FullOtherGroupPolicy` keep unit-level coverage in
//     GroupPolicyTest.cc (`group_policy_test`), so the policy classes themselves stay tested.
//
// This target survives to pin the narrowed accepted set itself: the rejection of a
// full+linear remote topology at `RemoteConnector` construction time.

#include <algorithm>

#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/connector/remote_connector/test/RemoteConnectorMockTestBase.h"
#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"
#include "rtp_llm/cpp/utils/Exception.h"
#include "rtp_llm/cpp/config/StaticConfig.h"

using namespace kv_cache_manager;
using namespace ::testing;
using namespace rtp_llm;
using namespace rtp_llm::remote_connector;

namespace rtp_llm {
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
    return SpecBuilder::build(desc, ctx).spec;
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
    return SpecBuilder::build(desc, ctx).spec;
}

}  // namespace

// Builds the 1 FULL + 2 LINEAR remote topology that the design rejects. No connector is
// constructed in SetUp() — constructing it is what the single test below asserts about.
class RemoteConnectorMockFullLinearTest: public RemoteConnectorMockTestBase {
public:
    void SetUp() override {
        full_group_tags_  = {"full0"};
        other_group_tags_ = {"linear1", "linear2"};
        RemoteConnectorMockTestBase::SetUp();
        initHybridLayerCacheConfig(kFakeLayerNum, /*block_num=*/40, /*seq_size_per_block=*/8);
    }

    void TearDown() override {
        RemoteConnectorMockTestBase::TearDown();
    }

private:
    void initHybridLayerCacheConfig(int layer_num = 4, int block_num = 10, int seq_size_per_block = 8) {
        const size_t all_group_num = full_group_tags_.size() + other_group_tags_.size();
        cache_config_.layer_num    = all_group_num * layer_num;

        auto full_spec   = makeTestMhaSpec("full", static_cast<uint32_t>(seq_size_per_block));
        auto linear_spec = makeTestLinearSpec("linear", static_cast<uint32_t>(seq_size_per_block));

        std::vector<KVCacheSpecPtr>   specs(all_group_num);
        std::vector<std::vector<int>> layers_by_group(all_group_num);
        std::vector<CacheGroupType>   group_types(all_group_num);
        std::vector<std::string>      tags(all_group_num);
        int                           unique_layer_id = 0;
        // Declaration slot is only a builder position; each group's identity is its tag.
        size_t declared = 0;
        for (const auto& tag : full_group_tags_) {
            specs[declared]       = full_spec;
            group_types[declared] = CacheGroupType::FULL;
            tags[declared]        = tag;
            for (int j = 0; j < layer_num; j++) {
                layers_by_group[declared].push_back(unique_layer_id++);
            }
            ++declared;
        }
        for (const auto& tag : other_group_tags_) {
            specs[declared]       = linear_spec;
            group_types[declared] = CacheGroupType::LINEAR;
            tags[declared]        = tag;
            for (int j = 0; j < layer_num; j++) {
                layers_by_group[declared].push_back(unique_layer_id++);
            }
            ++declared;
        }
        cache_config_.block_num          = block_num;
        cache_config_.seq_size_per_block = seq_size_per_block;
        cache_config_.dtype              = rtp_llm::DataType::TYPE_FP16;

        rtp_llm::test::assignCacheConfigFromGroupedSpecs(
            cache_config_, cache_config_.layer_num, specs, layers_by_group, group_types, tags);
        cache_config_.finalizeBlockNums(static_cast<uint32_t>(block_num), runtime_config_);

        ASSERT_GE(full_spec->block_size_bytes(), linear_spec->block_size_bytes());
    }
};

// The remote connector's accepted set is exactly one FULL cache group and nothing else.
// A 1-FULL-plus-2-LINEAR topology is therefore rejected, and it is rejected at construction
// time — before init(), before any meta client exists. Only the failure stage and category
// are asserted here; the message text is free to change in this phase.
TEST_F(RemoteConnectorMockFullLinearTest, test_construct_rejects_full_plus_linear_remote_topology) {
    const auto& groups = cache_config_.groups();
    ASSERT_EQ(groups.size(), 3u);
    ASSERT_EQ(std::count_if(groups.begin(),
                            groups.end(),
                            [](const CacheGroup& group) { return group.policy.group_type == CacheGroupType::FULL; }),
              1);

    // The topology is a legal, allocatable hybrid pool: the rejection below belongs to the
    // remote connector's narrowed accepted set, not to a malformed cache config.
    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(cache_config_);
    ASSERT_TRUE(allocator->init());

    // Report the rejection through the exception path instead of aborting the process.
    const bool saved_core_dump_on_exception               = rtp_llm::StaticConfig::user_ft_core_dump_on_exception;
    rtp_llm::StaticConfig::user_ft_core_dump_on_exception = false;

    // Failure stage: construction, ahead of any remote client creation.
    EXPECT_CALL(*mock_client_factory_, CreateMetaClient(_, _)).Times(0);
    // Failure category: invalid-configuration check failure (rtp_llm::RTPException).
    EXPECT_THROW(
        std::make_shared<RemoteConnector>(
            cache_config_, kv_cache_config_, runtime_config_, parallelism_config_, sp_config_, nullptr, 0, allocator),
        rtp_llm::RTPException);

    rtp_llm::StaticConfig::user_ft_core_dump_on_exception = saved_core_dump_on_exception;
}

}  // namespace test
}  // namespace rtp_llm

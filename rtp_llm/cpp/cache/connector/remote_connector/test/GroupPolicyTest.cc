#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string_view>
#include <tuple>

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/connector/remote_connector/GroupPolicy.h"

using namespace rtp_llm;
using namespace rtp_llm::remote_connector;
using namespace ::testing;
using namespace kv_cache_manager;

namespace rtp_llm {
namespace remote_connector {

bool operator==(const GroupPolicy::Group& lhs, const GroupPolicy::Group& rhs) {
    return lhs.is_full == rhs.is_full && lhs.group_name == rhs.group_name
           && lhs.group_name_bithash == rhs.group_name_bithash && lhs.tag == rhs.tag;
}

bool operator==(const GroupPolicy::SpecInfo& lhs, const GroupPolicy::SpecInfo& rhs) {
    return lhs.tp_rank == rhs.tp_rank && lhs.tag == rhs.tag;
}

namespace test {

namespace {

KVCacheSpecPtr makeFakeSpec(const std::string& tag) {
    AttentionConfigs attn_config;
    attn_config.kv_head_num      = 1;
    attn_config.size_per_head    = 1;
    attn_config.tokens_per_block = 1;
    ParallelismConfig parallelism_config;
    parallelism_config.tp_size = 1;
    KVCacheSpecDesc desc;
    desc.tag        = tag;
    desc.cache_type = KVCacheSpecType::MultiHeadAttention;
    desc.dtype      = DataType::TYPE_FP16;
    SpecBuildContext ctx;
    ctx.dtype                     = DataType::TYPE_FP16;
    ctx.seq_size_per_block        = 1;
    ctx.kernel_seq_size_per_block = 1;
    ctx.attn_config               = &attn_config;
    ctx.parallelism_config        = &parallelism_config;
    return SpecBuilder::build(desc, ctx).spec;
}

// Build a fake cache plan whose groups are identified only by their semantic
// tags. Tag order in the returned config is the declaration order, which the
// policy under test must never treat as identity.
std::shared_ptr<const CacheConfig> makeFakeCacheConfig(const std::vector<std::string>& full_tags,
                                                       const std::vector<std::string>& other_tags,
                                                       size_t                          per_group_layer_num) {
    std::vector<std::string> ordered_tags;
    for (const auto& tags : {full_tags, other_tags}) {
        for (const auto& tag : tags) {
            if (std::find(ordered_tags.begin(), ordered_tags.end(), tag) == ordered_tags.end()) {
                ordered_tags.push_back(tag);
            }
        }
    }
    if (ordered_tags.empty() || per_group_layer_num == 0) {
        return nullptr;
    }

    std::vector<CacheGroup> groups;
    std::vector<CacheLayer> layers;
    groups.reserve(ordered_tags.size());
    layers.reserve(ordered_tags.size() * per_group_layer_num);
    for (const auto& tag : ordered_tags) {
        CacheGroup group;
        group.tag                   = tag;
        group.spec                  = makeFakeSpec(tag);
        group.policy                = defaultCacheGroupPolicy(CacheGroupType::FULL);
        group.block_num             = 8;
        group.kv_block_stride_bytes = group.spec->block_size_bytes();
        group.kv_scale_stride_bytes = group.spec->scale_block_size_bytes();
        for (size_t i = 0; i < per_group_layer_num; ++i) {
            layers.push_back({tag});
        }
        groups.push_back(std::move(group));
    }
    const auto layer_num       = static_cast<uint32_t>(layers.size());
    auto       config          = std::make_shared<CacheConfig>(std::move(groups), std::move(layers), layer_num);
    config->seq_size_per_block = config->groups().front().seqSizePerBlock();
    return config;
}

CacheConfig makeFakeConfig(const std::shared_ptr<const CacheConfig>& topology) {
    if (topology == nullptr) {
        return CacheConfig();
    }
    CacheConfig config(topology->groups(), topology->layers(), static_cast<uint32_t>(topology->layers().size()));
    config.seq_size_per_block = topology->groups().front().seqSizePerBlock();
    return config;
}

}  // namespace

class FakeKVCacheAllocator: public KVCacheAllocator {
public:
    FakeKVCacheAllocator(const CacheConfig& config, std::shared_ptr<const CacheConfig> topology):
        KVCacheAllocator(config), topology_(std::move(topology)) {}
    using KVCacheAllocator::convertIndexToAddr;
    using KVCacheAllocator::convertIndexToBuffer;
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
    std::vector<BlockInfo> convertIndexToBufferByTag(int layer_id, std::string_view tag, int block_id) const override {
        tagged_buffer_requests_.emplace_back(layer_id, tag, block_id);
        BlockInfo info;
        info.addr       = reinterpret_cast<void*>(static_cast<uintptr_t>(block_id + 1));
        info.size_bytes = config_.group(tag).kv_block_stride_bytes + config_.group(tag).kv_scale_stride_bytes;
        return {info};
    }
    GroupedCacheLayerLayout allLayerCacheBase() const override {
        RTP_LLM_CHECK_WITH_INFO(topology_ != nullptr, "fake allocator has no cache topology");
        GroupedCacheLayerLayout::GroupLayouts groups;
        for (const auto& group : topology_->groups()) {
            groups.emplace(group.tag, CacheLayerLayout(std::vector<BlockBufferPtrInfo>(topology_->layers().size())));
        }
        return GroupedCacheLayerLayout(topology_, std::move(groups));
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
        return nullptr;
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

    bool doInit() override {
        return true;
    }

    const std::vector<std::tuple<int, std::string, int>>& taggedBufferRequests() const {
        return tagged_buffer_requests_;
    }

protected:
    MallocResult incrMalloc(const MallocInfo& malloc_info) override {
        return {};
    }
    MallocResult initMallocForCommonLen(const MallocInfo& malloc_info) override {
        return {};
    }

private:
    std::shared_ptr<const CacheConfig>                     topology_;
    mutable std::vector<std::tuple<int, std::string, int>> tagged_buffer_requests_;
};

MATCHER_P(LocationsEqLocationsView, locations_view, "") {
    const kv_cache_manager::Locations& locations = arg;
    if (locations.size() != locations_view.size()) {
        *result_listener << "locations size mismatch: expected " << locations.size() << ", actual "
                         << locations_view.size();
        return false;
    }
    for (size_t i = 0; i < locations.size(); i++) {
        const auto& location      = locations[i];
        const auto& location_view = locations_view[i];
        if (location.size() != location_view.size()) {
            *result_listener << "location[" << i << "] size mismatch: expected " << location.size() << ", actual "
                             << location_view.size();
            return false;
        }
        for (size_t j = 0; j < location.size(); j++) {
            const auto& spec_unit      = location[j];
            const auto& spec_unit_view = location_view[j];
            if (spec_unit.spec_name != spec_unit_view.spec_name) {
                *result_listener << "location[" << i << "][" << j << "] spec_name mismatch: expected "
                                 << spec_unit.spec_name << ", actual " << spec_unit_view.spec_name;
                return false;
            }
            if (spec_unit.uri != spec_unit_view.uri) {
                *result_listener << "location[" << i << "][" << j << "] uri mismatch: expected " << spec_unit.spec_name
                                 << ", actual " << spec_unit_view.spec_name;
                return false;
            }
        }
    }
    return true;
}

class GroupPolicyTest: public ::testing::Test {
public:
    enum class RemoteConnectorGroupMode {
        RCGM_LAYER_DEFAULT,
        RCGM_ONLY_FULL_LAYER,
        RCGM_FULL_LINEAR_LAYER
    };

    void SetUp() override {
        rtp_llm::initLogger();
    }

    void TearDown() override {}

    void initGroupPolicy(size_t                          tp_size,
                         RemoteConnectorGroupMode        group_mode,
                         size_t                          per_group_layer_num,
                         const std::vector<std::string>& full_tags,
                         const std::vector<std::string>& other_tags                      = {},
                         uint32_t                        linear_attention_write_interval = 0,
                         size_t                          sink_size                       = 0,
                         size_t                          sw_size                         = 0) {
        topology_  = makeFakeCacheConfig(full_tags, other_tags, per_group_layer_num);
        config_    = makeFakeConfig(topology_);
        allocator_ = std::make_shared<FakeKVCacheAllocator>(config_, topology_);
        switch (group_mode) {
            case RemoteConnectorGroupMode::RCGM_LAYER_DEFAULT: {
                group_policy_ =
                    std::make_shared<remote_connector::DefaultLayerGroupPolicy>(allocator_, full_tags, other_tags);
                break;
            }
            case RemoteConnectorGroupMode::RCGM_ONLY_FULL_LAYER: {
                group_policy_ =
                    std::make_shared<remote_connector::FullLayerGroupPolicy>(allocator_, full_tags, other_tags);
                break;
            }
            case RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER: {
                group_policy_ = std::make_shared<remote_connector::FullLinearLayerGroupPolicy>(
                    allocator_, full_tags, other_tags, linear_attention_write_interval);
                break;
            }
        }
        ASSERT_TRUE(group_policy_->init());
        size_t group_size = group_policy_->groups().size();
        ASSERT_GT(group_size, 0);
        std::vector<std::string> all_group_names;
        std::vector<uint64_t>    all_group_name_bithashs;
        all_group_names.reserve(group_size);
        for (const auto& entry : group_policy_->groups()) {
            const auto& group = entry.second;
            ASSERT_EQ(entry.first, group.tag);
            all_group_names.push_back(group.group_name);
            all_group_name_bithashs.push_back(group.group_name_bithash);
            group_policy_->addLocationSpecGroup(group.group_name_bithash, group.group_name);
            for (int r = 0; r < tp_size; ++r) {
                std::string location_spec_name = genLocationSpecName(r, group.group_name);
                ASSERT_TRUE(group_policy_->addSpecInfo(location_spec_name, group.tag, r));
            }
        }
        if (group_mode == RemoteConnectorGroupMode::RCGM_LAYER_DEFAULT) {
            // DefaultLayerGroupPolicy is only a generic test policy. Populate every
            // combination so its arbitrary-subset behavior remains covered.
            for (int sub_group = 2; sub_group <= group_size; ++sub_group) {
                std::string bitmask(sub_group, 1);
                bitmask.resize(group_size, 0);
                do {
                    std::stringstream ss_group_name;
                    uint64_t          groups_name_bithash = 0;
                    for (int i = 0; i < group_size; ++i) {
                        if (static_cast<bool>(bitmask[i])) {
                            ss_group_name << all_group_names[i];
                            groups_name_bithash |= all_group_name_bithashs[i];
                        }
                    }
                    group_policy_->addLocationSpecGroup(groups_name_bithash, ss_group_name.str());
                } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
            }
        } else {
            for (const auto aggregate_mask : group_policy_->reachableAggregateMasks()) {
                std::stringstream ss_group_name;
                for (size_t i = 0; i < group_size; ++i) {
                    if ((aggregate_mask & all_group_name_bithashs[i]) != 0) {
                        ss_group_name << all_group_names[i];
                    }
                }
                group_policy_->addLocationSpecGroup(aggregate_mask, ss_group_name.str());
            }
        }
        RTP_LLM_LOG_INFO("initGroupPolicy debug info\n [%s]", group_policy_->debugString().c_str());
    }

private:
    // Build a request resource whose per-group blocks are addressed by semantic
    // tag; the local record order in the resource is not part of the contract.
    std::shared_ptr<KVCacheResource>
    makeResourceForConfig(const CacheConfig&                             config,
                          const CacheKeysType&                           cache_keys,
                          const std::map<std::string, BlockIndicesType>& block_offsets_by_tag,
                          bool                                           last_block_aligned = true) const {
        auto resource = std::make_shared<KVCacheResource>();
        resource->initGroups(config);
        resource->setCacheKeys(cache_keys);
        for (const auto& [tag, block_offsets] : block_offsets_by_tag) {
            // These table-driven fixtures use compact zero-based offsets. Shift
            // materialized entries into the positive physical-ID domain.
            auto shifted_blocks = block_offsets;
            for (auto& block : shifted_blocks) {
                if (!isNullBlockIdx(block)) {
                    ++block;
                }
            }
            resource->mutableBlockIds(tag).assign(std::move(shifted_blocks));
        }
        resource->setLastBlockAligned(last_block_aligned);
        return resource;
    }

    std::shared_ptr<KVCacheResource> makeResource(const CacheKeysType&                           cache_keys,
                                                  const std::map<std::string, BlockIndicesType>& blocks_by_tag,
                                                  bool last_block_aligned = true) const {
        return makeResourceForConfig(config_, cache_keys, blocks_by_tag, last_block_aligned);
    }

    kv_cache_manager::Locations genFullLinearLocations(size_t                          tp_size,
                                                       const std::vector<std::string>& full_tags,
                                                       const std::vector<std::string>& linear_tags,
                                                       size_t                          cache_key_size,
                                                       const std::vector<size_t>&      linear_pos_vec) const {
        kv_cache_manager::Locations locations;
        locations.resize(cache_key_size, {});
        for (size_t i = 0; i < cache_key_size; i++) {
            for (const auto& tag : full_tags) {
                std::string full_group_name = "F" + tag;
                for (int r = 0; r < tp_size; r++) {
                    std::string uri = "uri_" + full_group_name + "_" + std::to_string(r) + "_" + std::to_string(i);
                    locations[i].push_back(
                        kv_cache_manager::LocationSpecUnit({genLocationSpecName(r, full_group_name), uri}));
                }
            }
        }
        for (auto pos : linear_pos_vec) {
            for (const auto& tag : linear_tags) {
                std::string linear_group_name = "L" + tag;
                for (int r = 0; r < tp_size; r++) {
                    std::string uri = "uri_" + linear_group_name + "_" + std::to_string(r) + "_" + std::to_string(pos);
                    locations[pos].push_back(
                        kv_cache_manager::LocationSpecUnit({genLocationSpecName(r, linear_group_name), uri}));
                }
            }
        }
        return locations;
    }

    inline std::string genLocationSpecName(int tp_rank, const std::string& group_name) const {
        static std::string location_spec_name("tp");
        return location_spec_name + std::to_string(tp_rank) + "_" + group_name;
    }

    void test_FullLinearLayerGroupPolicy_filterNeedLoadLocations(size_t                          tp_size,
                                                                 const std::vector<std::string>& full_tags,
                                                                 const std::vector<std::string>& linear_tags) {
        {
            Locations     locations = genFullLinearLocations(tp_size, full_tags, linear_tags, 4, {3});
            LocationsView locations_view;
            ASSERT_TRUE(group_policy_->filterNeedLoadLocations(locations, locations_view));
            auto expect_locations = genFullLinearLocations(tp_size, full_tags, linear_tags, 4, {3});
            ASSERT_THAT(expect_locations, LocationsEqLocationsView(locations_view));
        }
        {  // only load the last linear block
            Locations     locations = genFullLinearLocations(tp_size, full_tags, linear_tags, 4, {1, 3});
            LocationsView locations_view;
            ASSERT_TRUE(group_policy_->filterNeedLoadLocations(locations, locations_view));
            auto expect_locations = genFullLinearLocations(tp_size, full_tags, linear_tags, 4, {3});
            ASSERT_THAT(expect_locations, LocationsEqLocationsView(locations_view));
        }
        {  // only load the last full + linear block
            Locations     locations = genFullLinearLocations(tp_size, full_tags, linear_tags, 4, {1, 2});
            LocationsView locations_view;
            ASSERT_TRUE(group_policy_->filterNeedLoadLocations(locations, locations_view));
            auto expect_locations = genFullLinearLocations(tp_size, full_tags, linear_tags, 3, {2});
            ASSERT_THAT(expect_locations, LocationsEqLocationsView(locations_view));
        }
        {  // only load the last full + linear block
            Locations     locations = genFullLinearLocations(tp_size, full_tags, linear_tags, 5, {1, 2});
            LocationsView locations_view;
            ASSERT_TRUE(group_policy_->filterNeedLoadLocations(locations, locations_view));
            auto expect_locations = genFullLinearLocations(tp_size, full_tags, linear_tags, 3, {2});
            ASSERT_THAT(expect_locations, LocationsEqLocationsView(locations_view));
        }
        {  // empty linear block
            Locations     locations = genFullLinearLocations(tp_size, full_tags, linear_tags, 4, {});
            LocationsView locations_view;
            ASSERT_TRUE(group_policy_->filterNeedLoadLocations(locations, locations_view));
            auto expect_locations = genFullLinearLocations(tp_size, full_tags, linear_tags, 0, {});
            ASSERT_THAT(expect_locations, LocationsEqLocationsView(locations_view));
        }
        {  // edge case : empty locations
            Locations     locations = genFullLinearLocations(tp_size, full_tags, linear_tags, 0, {});
            LocationsView locations_view;
            ASSERT_TRUE(group_policy_->filterNeedLoadLocations(locations, locations_view));
            auto expect_locations = genFullLinearLocations(tp_size, full_tags, linear_tags, 0, {});
            ASSERT_THAT(expect_locations, LocationsEqLocationsView(locations_view));
        }
        {  // edge case : one full block + empty linear block
            Locations     locations = genFullLinearLocations(tp_size, full_tags, linear_tags, 1, {});
            LocationsView locations_view;
            ASSERT_TRUE(group_policy_->filterNeedLoadLocations(locations, locations_view));
            auto expect_locations = genFullLinearLocations(tp_size, full_tags, linear_tags, 0, {});
            ASSERT_THAT(expect_locations, LocationsEqLocationsView(locations_view));
        }
        {  // edge case : one full block + one linear block
            Locations     locations = genFullLinearLocations(tp_size, full_tags, linear_tags, 1, {0});
            LocationsView locations_view;
            ASSERT_TRUE(group_policy_->filterNeedLoadLocations(locations, locations_view));
            auto expect_locations = genFullLinearLocations(tp_size, full_tags, linear_tags, 1, {0});
            ASSERT_THAT(expect_locations, LocationsEqLocationsView(locations_view));
        }
    }

    void test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_interval_2() {
        {
            auto resource =
                makeResource({0, 1, 2, 3}, {{"0", {0, 1, 2, 3}}, {"1", {4, 5, 6, 7}}, {"2", {8, 9, 10, 11}}});
            std::vector<std::string> real;
            ASSERT_TRUE(group_policy_->getNeedWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0", "F0L1L2", "F0", "F0L1L2"};
            ASSERT_EQ(expected, real);
        }
        {
            auto                     resource = makeResource({0, 1, 2, 3},
                                                             {{"0", {0, 1, 2, 3}}, {"1", {4, 5, 6, 7}}, {"2", {8, 9, 10, 11}}},
                                         /*last_block_aligned=*/false);
            std::vector<std::string> real;
            ASSERT_TRUE(group_policy_->getNeedWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0L1L2", "F0", "F0L1L2"};
            ASSERT_EQ(expected, real);
        }
        {
            auto                     resource = makeResource({0, 1, 2, 3, 4},
                                                             {{"0", {0, 1, 2, 3, 20}}, {"1", {4, 5, 6, 7, 21}}, {"2", {8, 9, 10, 11, 22}}});
            std::vector<std::string> real;
            ASSERT_TRUE(group_policy_->getNeedWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0L1L2", "F0", "F0L1L2", "F0", "F0L1L2"};
            ASSERT_EQ(expected, real);
        }
        {  // exist empty block
            auto resource =
                makeResource({0, 1, 2, 3}, {{"0", {0, 1, 2, 3}}, {"1", {-1, -1, -1, 7}}, {"2", {-1, -1, -1, 11}}});
            std::vector<std::string> real;
            ASSERT_TRUE(group_policy_->getNeedWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0", "F0", "F0", "F0L1L2"};
            ASSERT_EQ(expected, real);
        }
        {  // exist empty block
            auto resource =
                makeResource({0, 1, 2, 3}, {{"0", {0, 1, 2, 3}}, {"1", {4, -1, 6, 7}}, {"2", {8, -1, 10, 11}}});
            std::vector<std::string> real;
            ASSERT_TRUE(group_policy_->getNeedWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0L1L2", "F0", "F0", "F0L1L2"};
            ASSERT_EQ(expected, real);
        }
        {
            auto resource =
                makeResource({0, 1, 2, 3}, {{"0", {0, 1, 2, 3}}, {"1", {4, 5, 6, -1}}, {"2", {8, 9, 10, -1}}});
            std::vector<std::string> real;
            ASSERT_TRUE(group_policy_->getNeedWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0L1L2", "F0", "F0L1L2", "F0"};
            ASSERT_EQ(expected, real);
        }
        {  // exist empty block
            auto resource = makeResource(
                {0, 1, 2, 3, 4}, {{"0", {0, 1, 2, 3, 20}}, {"1", {4, 5, -1, 7, 21}}, {"2", {8, 9, -1, 11, 22}}});
            std::vector<std::string> real;
            ASSERT_TRUE(group_policy_->getNeedWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0", "F0L1L2", "F0", "F0", "F0L1L2"};
            ASSERT_EQ(expected, real);
        }
        {  // exist empty block
            auto resource = makeResource(
                {0, 1, 2, 3, 4}, {{"0", {0, 1, 2, 3, 20}}, {"1", {4, -1, -1, 7, 21}}, {"2", {8, -1, -1, 11, 22}}});
            std::vector<std::string> real;
            ASSERT_TRUE(group_policy_->getNeedWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0L1L2", "F0", "F0", "F0", "F0L1L2"};
            ASSERT_EQ(expected, real);
        }
        {  // exist empty block
            auto resource =
                makeResource({0, 1, 2, 3}, {{"0", {0, 1, 2, 3}}, {"1", {-1, 5, -1, 7}}, {"2", {-1, 9, -1, 11}}});
            std::vector<std::string> real;
            ASSERT_TRUE(group_policy_->getNeedWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0", "F0L1L2", "F0", "F0L1L2"};
            ASSERT_EQ(expected, real);
        }
        {  // edge case
            auto                     resource = makeResource({0}, {{"0", {0}}, {"1", {1}}, {"2", {2}}});
            std::vector<std::string> real;
            ASSERT_TRUE(group_policy_->getNeedWriteGroups(resource, real));
            std::vector<std::string> expected = {};  // all full linear
            ASSERT_EQ(expected, real);
        }
        {  // edge case
            auto                     resource = makeResource({0}, {{"0", {0}}, {"1", {-1}}, {"2", {-1}}});
            std::vector<std::string> real;
            ASSERT_TRUE(group_policy_->getNeedWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0"};
            ASSERT_EQ(expected, real);
        }
    }

    void test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_interval_1() {
        {
            auto resource =
                makeResource({0, 1, 2, 3}, {{"0", {0, 1, 2, 3}}, {"1", {4, 5, 6, 7}}, {"2", {8, 9, 10, 11}}});
            std::vector<std::string> real;
            ASSERT_TRUE(group_policy_->getNeedWriteGroups(resource, real));
            std::vector<std::string> expected = {};
            ASSERT_EQ(expected, real);
        }
        {
            auto resource = makeResource(
                {0, 1, 2, 3, 4}, {{"0", {0, 1, 2, 3, 20}}, {"1", {4, 5, 6, -1, 21}}, {"2", {8, 9, 10, -1, 22}}});
            std::vector<std::string> real;
            ASSERT_TRUE(group_policy_->getNeedWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0L1L2", "F0L1L2", "F0L1L2", "F0", "F0L1L2"};
            ASSERT_EQ(expected, real);
        }
        {
            auto                     resource = makeResource({0}, {{"0", {0}}, {"1", {4}}, {"2", {8}}});
            std::vector<std::string> real;
            ASSERT_TRUE(group_policy_->getNeedWriteGroups(resource, real));
            std::vector<std::string> expected = {};
            ASSERT_EQ(expected, real);
        }
    }

    void test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_interval_0() {
        {
            auto resource =
                makeResource({0, 1, 2, 3}, {{"0", {0, 1, 2, 3}}, {"1", {4, 5, 6, 7}}, {"2", {8, 9, 10, 11}}});
            std::vector<std::string> real;
            ASSERT_TRUE(group_policy_->getNeedWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0", "F0", "F0", "F0L1L2"};
            ASSERT_EQ(expected, real);
        }
        {
            auto                     resource = makeResource({0, 1, 2, 3, 4},
                                                             {{"0", {0, 1, 2, 3, 20}}, {"1", {4, 5, 6, 7, 21}}, {"2", {8, 9, 10, 11, 22}}});
            std::vector<std::string> real;
            ASSERT_TRUE(group_policy_->getNeedWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0", "F0", "F0", "F0", "F0L1L2"};
            ASSERT_EQ(expected, real);
        }
        {
            auto resource =
                makeResource({0, 1, 2, 3}, {{"0", {0, 1, 2, 3}}, {"1", {4, 5, 6, -1}}, {"2", {8, 9, 10, -1}}});
            std::vector<std::string> real;
            ASSERT_TRUE(group_policy_->getNeedWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0", "F0", "F0L1L2", "F0"};
            ASSERT_EQ(expected, real);
        }
        {  // edge case
            auto                     resource = makeResource({0}, {{"0", {0}}, {"1", {4}}, {"2", {8}}});
            std::vector<std::string> real;
            ASSERT_TRUE(group_policy_->getNeedWriteGroups(resource, real));
            std::vector<std::string> expected = {};
            ASSERT_EQ(expected, real);
        }
        {  // edge case
            auto                     resource = makeResource({0}, {{"0", {0}}, {"1", {-1}}, {"2", {-1}}});
            std::vector<std::string> real;
            ASSERT_TRUE(group_policy_->getNeedWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0"};
            ASSERT_EQ(expected, real);
        }
    }

private:
    std::shared_ptr<const CacheConfig> topology_;
    std::shared_ptr<KVCacheAllocator>  allocator_;
    std::shared_ptr<GroupPolicy>       group_policy_;
    CacheConfig                        config_;
};

TEST_F(GroupPolicyTest, test_init_FullLinearLayerGroupPolicy_success_single_tp) {
    initGroupPolicy(1, RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER, 4, {"0"}, {"1", "2"}, 0);
    auto cast_group_policy = std::dynamic_pointer_cast<FullLinearLayerGroupPolicy>(group_policy_);
    ASSERT_EQ(
        GroupPolicy::GroupTagMap(
            {{"0", {true, 0b001, "F0", "0"}}, {"1", {false, 0b010, "L1", "1"}}, {"2", {false, 0b100, "L2", "2"}}}),
        cast_group_policy->groups_);
    ASSERT_EQ(
        (std::unordered_map<uint64_t, std::string>({{0b111, "F0L1L2"}, {0b001, "F0"}, {0b010, "L1"}, {0b100, "L2"}})),
        cast_group_policy->location_spec_group_map_);
    EXPECT_EQ(cast_group_policy->reachableAggregateMasks(), (std::vector<uint64_t>{0b001, 0b111}));
    ASSERT_EQ(GroupPolicy::SpecInfoMap({{"tp0_F0", {0, "0"}}, {"tp0_L1", {0, "1"}}, {"tp0_L2", {0, "2"}}}),
              cast_group_policy->spec_name_to_info_);
    ASSERT_EQ(
        (std::map<std::string, std::vector<int>>({{"0", {0, 1, 2, 3}}, {"1", {4, 5, 6, 7}}, {"2", {8, 9, 10, 11}}})),
        cast_group_policy->tag_to_layer_ids_);
    ASSERT_EQ(0b001, cast_group_policy->valid_full_bithash_);
    ASSERT_EQ(0b111, cast_group_policy->valid_full_other_bithash_);
    ASSERT_EQ((std::map<std::string, uint64_t>({{"tp0_F0", 0b001}})), cast_group_policy->full_spec_name_bithash_);
    ASSERT_EQ((std::map<std::string, uint64_t>({{"tp0_F0", 0b001}, {"tp0_L1", 0b010}, {"tp0_L2", 0b100}})),
              cast_group_policy->full_other_spec_name_bithash_);
}

TEST_F(GroupPolicyTest, test_init_FullLinearLayerGroupPolicy_success_two_tp) {
    initGroupPolicy(2, RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER, 4, {"0"}, {"1", "2"}, 0);
    auto cast_group_policy = std::dynamic_pointer_cast<FullLinearLayerGroupPolicy>(group_policy_);
    ASSERT_EQ(
        GroupPolicy::GroupTagMap(
            {{"0", {true, 0b001, "F0", "0"}}, {"1", {false, 0b010, "L1", "1"}}, {"2", {false, 0b100, "L2", "2"}}}),
        cast_group_policy->groups_);
    ASSERT_EQ(
        (std::unordered_map<uint64_t, std::string>({{0b111, "F0L1L2"}, {0b001, "F0"}, {0b010, "L1"}, {0b100, "L2"}})),
        cast_group_policy->location_spec_group_map_);
    ASSERT_EQ(GroupPolicy::SpecInfoMap({{"tp0_F0", {0, "0"}},
                                        {"tp0_L1", {0, "1"}},
                                        {"tp0_L2", {0, "2"}},
                                        {"tp1_F0", {1, "0"}},
                                        {"tp1_L1", {1, "1"}},
                                        {"tp1_L2", {1, "2"}}}),
              cast_group_policy->spec_name_to_info_);
    ASSERT_EQ(
        (std::map<std::string, std::vector<int>>({{"0", {0, 1, 2, 3}}, {"1", {4, 5, 6, 7}}, {"2", {8, 9, 10, 11}}})),
        cast_group_policy->tag_to_layer_ids_);
    ASSERT_EQ(0b001, cast_group_policy->valid_full_bithash_);
    ASSERT_EQ(0b111, cast_group_policy->valid_full_other_bithash_);
    ASSERT_EQ((std::map<std::string, uint64_t>({{"tp0_F0", 0b001}, {"tp1_F0", 0b001}})),
              cast_group_policy->full_spec_name_bithash_);
    ASSERT_EQ((std::map<std::string, uint64_t>({{"tp0_F0", 0b001},
                                                {"tp0_L1", 0b010},
                                                {"tp0_L2", 0b100},
                                                {"tp1_F0", 0b001},
                                                {"tp1_L1", 0b010},
                                                {"tp1_L2", 0b100}})),
              cast_group_policy->full_other_spec_name_bithash_);
}

TEST_F(GroupPolicyTest, test_init_FullLinearLayerGroupPolicy_success_two_full_groups) {
    initGroupPolicy(2, RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER, 4, {"0", "1"}, {"2", "3"}, 0);
    auto cast_group_policy = std::dynamic_pointer_cast<FullLinearLayerGroupPolicy>(group_policy_);
    ASSERT_EQ(GroupPolicy::GroupTagMap({{"0", {true, 0b0001, "F0", "0"}},
                                        {"1", {true, 0b0010, "F1", "1"}},
                                        {"2", {false, 0b0100, "L2", "2"}},
                                        {"3", {false, 0b1000, "L3", "3"}}}),
              cast_group_policy->groups_);
    EXPECT_EQ(
        (std::unordered_map<uint64_t, std::string>(
            {{0b1111, "F0F1L2L3"}, {0b0001, "F0"}, {0b0010, "F1"}, {0b0100, "L2"}, {0b1000, "L3"}, {0b0011, "F0F1"}})),
        cast_group_policy->location_spec_group_map_);
    EXPECT_EQ(cast_group_policy->reachableAggregateMasks(), (std::vector<uint64_t>{0b0011, 0b1111}));
    EXPECT_EQ(GroupPolicy::SpecInfoMap({
                  {"tp0_F0", {0, "0"}},
                  {"tp0_F1", {0, "1"}},
                  {"tp0_L2", {0, "2"}},
                  {"tp0_L3", {0, "3"}},
                  {"tp1_F0", {1, "0"}},
                  {"tp1_F1", {1, "1"}},
                  {"tp1_L2", {1, "2"}},
                  {"tp1_L3", {1, "3"}},
              }),
              cast_group_policy->spec_name_to_info_);
    EXPECT_EQ((std::map<std::string, std::vector<int>>(
                  {{"0", {0, 1, 2, 3}}, {"1", {4, 5, 6, 7}}, {"2", {8, 9, 10, 11}}, {"3", {12, 13, 14, 15}}})),
              cast_group_policy->tag_to_layer_ids_);
    EXPECT_EQ(0b0011, cast_group_policy->valid_full_bithash_);
    EXPECT_EQ(0b1111, cast_group_policy->valid_full_other_bithash_);
    EXPECT_EQ((std::map<std::string, uint64_t>(
                  {{"tp0_F0", 0b0001}, {"tp0_F1", 0b0010}, {"tp1_F0", 0b0001}, {"tp1_F1", 0b0010}})),
              cast_group_policy->full_spec_name_bithash_);
    EXPECT_EQ((std::map<std::string, uint64_t>({
                  {"tp0_F0", 0b0001},
                  {"tp0_F1", 0b0010},
                  {"tp0_L2", 0b0100},
                  {"tp0_L3", 0b1000},
                  {"tp1_F0", 0b0001},
                  {"tp1_F1", 0b0010},
                  {"tp1_L2", 0b0100},
                  {"tp1_L3", 0b1000},
              })),
              cast_group_policy->full_other_spec_name_bithash_);
}

TEST_F(GroupPolicyTest, test_init_DefaultLayerGroupPolicy_fail_for_duplicate_group) {
    std::vector<std::string> full_tags  = {"0", "1"};
    std::vector<std::string> other_tags = {"0", "1"};
    topology_                           = makeFakeCacheConfig(full_tags, other_tags, 10);
    config_                             = makeFakeConfig(topology_);
    allocator_                          = std::make_shared<FakeKVCacheAllocator>(config_, topology_);
    group_policy_ = std::make_shared<remote_connector::DefaultLayerGroupPolicy>(allocator_, full_tags, other_tags);
    ASSERT_FALSE(group_policy_->init());
}

TEST_F(GroupPolicyTest, test_init_FullLayerGroupPolicy_fail_for_empty_full_group) {
    std::vector<std::string> full_tags;
    std::vector<std::string> other_tags;
    topology_     = makeFakeCacheConfig({"placeholder"}, {}, 10);
    config_       = makeFakeConfig(topology_);
    allocator_    = std::make_shared<FakeKVCacheAllocator>(config_, topology_);
    group_policy_ = std::make_shared<remote_connector::FullLayerGroupPolicy>(allocator_, full_tags, other_tags);
    ASSERT_FALSE(group_policy_->init());
}

TEST_F(GroupPolicyTest, test_init_FullLayerGroupPolicy_success_for_multiple_full_groups) {
    initGroupPolicy(/*tp_size=*/1,
                    RemoteConnectorGroupMode::RCGM_ONLY_FULL_LAYER,
                    /*per_group_layer_num=*/1,
                    /*full_tags=*/{"0", "1"});

    EXPECT_EQ(group_policy_->reachableAggregateMasks(), (std::vector<uint64_t>{0b11}));
    EXPECT_EQ(group_policy_->location_spec_group_map_,
              (std::unordered_map<uint64_t, std::string>{{0b01, "F0"}, {0b10, "F1"}, {0b11, "F0F1"}}));

    auto resource = makeResource({0, 1}, {{"0", {10, 11}}, {"1", {20, NULL_BLOCK_IDX}}});

    std::vector<std::string> need_write_groups;
    ASSERT_TRUE(group_policy_->getNeedWriteGroups(resource, need_write_groups));
    EXPECT_EQ(need_write_groups, (std::vector<std::string>{"F0F1", "F0"}));
}

TEST_F(GroupPolicyTest, test_init_FullLayerGroupPolicy_fail_for_not_empty_other_group) {
    std::vector<std::string> full_tags  = {"0"};
    std::vector<std::string> other_tags = {"1"};
    topology_                           = makeFakeCacheConfig(full_tags, other_tags, 10);
    config_                             = makeFakeConfig(topology_);
    allocator_                          = std::make_shared<FakeKVCacheAllocator>(config_, topology_);
    group_policy_ = std::make_shared<remote_connector::FullLayerGroupPolicy>(allocator_, full_tags, other_tags);
    ASSERT_FALSE(group_policy_->init());
}

TEST_F(GroupPolicyTest, test_init_FullLinearLayerGroupPolicy_fail_for_not_empty_group) {
    {
        std::vector<std::string> full_tags;
        std::vector<std::string> other_tags = {"1"};
        topology_                           = makeFakeCacheConfig(full_tags, other_tags, 10);
        config_                             = makeFakeConfig(topology_);
        allocator_                          = std::make_shared<FakeKVCacheAllocator>(config_, topology_);
        group_policy_ =
            std::make_shared<remote_connector::FullLinearLayerGroupPolicy>(allocator_, full_tags, other_tags, 0);
        ASSERT_FALSE(group_policy_->init());
    }
    {
        std::vector<std::string> full_tags = {"0"};
        std::vector<std::string> other_tags;
        topology_  = makeFakeCacheConfig(full_tags, other_tags, 10);
        config_    = makeFakeConfig(topology_);
        allocator_ = std::make_shared<FakeKVCacheAllocator>(config_, topology_);
        group_policy_ =
            std::make_shared<remote_connector::FullLinearLayerGroupPolicy>(allocator_, full_tags, other_tags, 0);
        ASSERT_FALSE(group_policy_->init());
    }
}

TEST_F(GroupPolicyTest, test_FullLinearLayerGroupPolicy_filterNeedLoadLocations_success_one_tp) {
    size_t                   tp_size                         = 1;
    std::vector<std::string> full_tags                       = {"0"};
    std::vector<std::string> linear_tags                     = {"1", "2"};
    uint32_t                 linear_attention_write_interval = 0;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER,
                    4,
                    full_tags,
                    linear_tags,
                    linear_attention_write_interval);
    test_FullLinearLayerGroupPolicy_filterNeedLoadLocations(tp_size, full_tags, linear_tags);
}

TEST_F(GroupPolicyTest, test_FullLinearLayerGroupPolicy_filterNeedLoadLocations_success_two_tp) {
    size_t                   tp_size                         = 2;
    std::vector<std::string> full_tags                       = {"0"};
    std::vector<std::string> linear_tags                     = {"1", "2"};
    uint32_t                 linear_attention_write_interval = 0;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER,
                    4,
                    full_tags,
                    linear_tags,
                    linear_attention_write_interval);
    test_FullLinearLayerGroupPolicy_filterNeedLoadLocations(tp_size, full_tags, linear_tags);
}

TEST_F(GroupPolicyTest, test_FullLinearLayerGroupPolicy_filterNeedLoadLocations_success_two_tp_two_full_group) {
    size_t                   tp_size                         = 2;
    std::vector<std::string> full_tags                       = {"0", "1"};
    std::vector<std::string> linear_tags                     = {"2", "3"};
    uint32_t                 linear_attention_write_interval = 0;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER,
                    4,
                    full_tags,
                    linear_tags,
                    linear_attention_write_interval);
    test_FullLinearLayerGroupPolicy_filterNeedLoadLocations(tp_size, full_tags, linear_tags);
}

TEST_F(GroupPolicyTest, test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_success_one_tp_interval_2) {
    size_t                   tp_size                         = 1;
    std::vector<std::string> full_tags                       = {"0"};
    std::vector<std::string> linear_tags                     = {"1", "2"};
    uint32_t                 linear_attention_write_interval = 2;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER,
                    4,
                    full_tags,
                    linear_tags,
                    linear_attention_write_interval);
    test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_interval_2();
}

TEST_F(GroupPolicyTest, test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_success_two_tp_interval_2) {
    size_t                   tp_size                         = 2;
    std::vector<std::string> full_tags                       = {"0"};
    std::vector<std::string> linear_tags                     = {"1", "2"};
    uint32_t                 linear_attention_write_interval = 2;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER,
                    4,
                    full_tags,
                    linear_tags,
                    linear_attention_write_interval);
    test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_interval_2();
}

TEST_F(GroupPolicyTest, test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_success_one_tp_interval_1) {
    size_t                   tp_size                         = 1;
    std::vector<std::string> full_tags                       = {"0"};
    std::vector<std::string> linear_tags                     = {"1", "2"};
    uint32_t                 linear_attention_write_interval = 1;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER,
                    4,
                    full_tags,
                    linear_tags,
                    linear_attention_write_interval);
    test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_interval_1();
}

TEST_F(GroupPolicyTest, test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_success_two_tp_interval_1) {
    size_t                   tp_size                         = 2;
    std::vector<std::string> full_tags                       = {"0"};
    std::vector<std::string> linear_tags                     = {"1", "2"};
    uint32_t                 linear_attention_write_interval = 1;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER,
                    4,
                    full_tags,
                    linear_tags,
                    linear_attention_write_interval);
    test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_interval_1();
}

TEST_F(GroupPolicyTest, test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_success_one_tp_interval_0) {
    size_t                   tp_size                         = 1;
    std::vector<std::string> full_tags                       = {"0"};
    std::vector<std::string> linear_tags                     = {"1", "2"};
    uint32_t                 linear_attention_write_interval = 0;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER,
                    4,
                    full_tags,
                    linear_tags,
                    linear_attention_write_interval);
    test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_interval_0();
}

TEST_F(GroupPolicyTest, test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_success_two_tp_interval_0) {
    size_t                   tp_size                         = 2;
    std::vector<std::string> full_tags                       = {"0"};
    std::vector<std::string> linear_tags                     = {"1", "2"};
    uint32_t                 linear_attention_write_interval = 0;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER,
                    4,
                    full_tags,
                    linear_tags,
                    linear_attention_write_interval);
    test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_interval_0();
}

TEST_F(GroupPolicyTest, test_FullLinearLayerGroupPolicy_filterNeedLoadLocations_fail) {
    size_t                   tp_size                         = 1;
    std::vector<std::string> full_tags                       = {"0"};
    std::vector<std::string> linear_tags                     = {"1", "2"};
    uint32_t                 linear_attention_write_interval = 0;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER,
                    4,
                    full_tags,
                    linear_tags,
                    linear_attention_write_interval);
    {
        Locations     locations({{{"tp0_F0", "uri"}, {"tp0_L1", "uri"}, {"tp0_L2", "uri"}}});
        LocationsView locations_view;
        ASSERT_TRUE(group_policy_->filterNeedLoadLocations(locations, locations_view));
    }
    {  // location size error
        Locations     locations({{{"tp0_F0", "uri"}, {"tp0_L1", "uri"}}});
        LocationsView locations_view;
        ASSERT_FALSE(group_policy_->filterNeedLoadLocations(locations, locations_view));
    }
    {  // not exist full location
        Locations     locations({{{"tp0_L2", "uri"}}});
        LocationsView locations_view;
        ASSERT_FALSE(group_policy_->filterNeedLoadLocations(locations, locations_view));
    }
    {  // invalid spec name
        Locations     locations({{{"not_exist", "uri"}, {"tp0_L1", "uri"}, {"tp0_L2", "uri"}}});
        LocationsView locations_view;
        ASSERT_FALSE(group_policy_->filterNeedLoadLocations(locations, locations_view));
    }
}

TEST_F(GroupPolicyTest, test_FullLinearLayerGroupPolicy_filterNeedLoadLocations_fail_two_full_group) {
    size_t                   tp_size                         = 1;
    std::vector<std::string> full_tags                       = {"0", "1"};
    std::vector<std::string> linear_tags                     = {"2", "3"};
    uint32_t                 linear_attention_write_interval = 0;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER,
                    4,
                    full_tags,
                    linear_tags,
                    linear_attention_write_interval);
    {
        Locations     locations({{{"tp0_F0", "uri"}, {"tp0_F1", "uri"}, {"tp0_L2", "uri"}, {"tp0_L3", "uri"}}});
        LocationsView locations_view;
        ASSERT_TRUE(group_policy_->filterNeedLoadLocations(locations, locations_view));
    }
    {
        Locations     locations({{{"tp0_F0", "uri"}, {"tp0_L2", "uri"}}});
        LocationsView locations_view;
        ASSERT_FALSE(group_policy_->filterNeedLoadLocations(locations, locations_view));
    }
    {
        Locations     locations({{{"tp0_F0", "uri"}}});
        LocationsView locations_view;
        ASSERT_FALSE(group_policy_->filterNeedLoadLocations(locations, locations_view));
    }
}

TEST_F(GroupPolicyTest, test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_fail) {
    size_t                   tp_size                         = 1;
    std::vector<std::string> full_tags                       = {"0"};
    std::vector<std::string> linear_tags                     = {"1", "2"};
    uint32_t                 linear_attention_write_interval = 2;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER,
                    4,
                    full_tags,
                    linear_tags,
                    linear_attention_write_interval);
    {  // incomplete block
        auto                     resource = makeResource({0, 1, 2, 3, 4},
                                                         {{"0", {0, 1, 2, 3, 20}}, {"1", {4, -1, 6, 7, -1}}, {"2", {8, 9, 10, 11, -1}}});
        std::vector<std::string> real;
        ASSERT_FALSE(group_policy_->getNeedWriteGroups(resource, real));
    }
    {  // resource carries fewer cache groups than the policy
        const auto narrow_config = makeFakeConfig(makeFakeCacheConfig({"0"}, {"1"}, 4));
        auto       resource =
            makeResourceForConfig(narrow_config, {0, 1, 2, 3, 4}, {{"0", {0, 1, 2, 3, 20}}, {"1", {4, 5, 6, 7, 21}}});
        std::vector<std::string> real;
        ASSERT_FALSE(group_policy_->getNeedWriteGroups(resource, real));
    }
    {  // resource carries more cache groups than the policy
        const auto wide_config = makeFakeConfig(makeFakeCacheConfig({"0"}, {"1", "2", "3"}, 4));
        auto       resource    = makeResourceForConfig(
            wide_config,
            {0, 1, 2, 3, 4},
            {{"0", {0, 1, 2, 3, 20}}, {"1", {4, 5, 6, 7, 21}}, {"2", {8, 9, 10, 11, 22}}, {"3", {12, 13, 14, 15, 23}}});
        std::vector<std::string> real;
        ASSERT_FALSE(group_policy_->getNeedWriteGroups(resource, real));
    }
}

TEST_F(GroupPolicyTest, test_FullLayerGroupPolicy_filterNeedLoadLocations_success) {
    size_t                   tp_size   = 2;
    std::vector<std::string> full_tags = {"0"};
    std::vector<std::string> other_tags;
    uint32_t                 linear_attention_write_interval = 0;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_ONLY_FULL_LAYER,
                    4,
                    full_tags,
                    other_tags,
                    linear_attention_write_interval);
    EXPECT_EQ(group_policy_->reachableAggregateMasks(), (std::vector<uint64_t>{0b1}));
    EXPECT_EQ(group_policy_->location_spec_group_map_.size(), 1u);
    {
        Locations     locations = genFullLinearLocations(tp_size, full_tags, other_tags, 4, {});
        LocationsView locations_view;
        ASSERT_TRUE(group_policy_->filterNeedLoadLocations(locations, locations_view));
        auto expect_locations = genFullLinearLocations(tp_size, full_tags, other_tags, 4, {});
        ASSERT_THAT(expect_locations, LocationsEqLocationsView(locations_view));
    }
}

TEST_F(GroupPolicyTest, test_FullLayerGroupPolicy_filterNeedWriteGroups_success) {
    size_t                   tp_size   = 2;
    std::vector<std::string> full_tags = {"0"};
    std::vector<std::string> other_tags;
    uint32_t                 linear_attention_write_interval = 0;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_ONLY_FULL_LAYER,
                    4,
                    full_tags,
                    other_tags,
                    linear_attention_write_interval);
    {
        auto                     resource = makeResource({0, 1, 2, 3}, {{"0", {0, 1, 2, 3}}});
        std::vector<std::string> real;
        ASSERT_TRUE(group_policy_->getNeedWriteGroups(resource, real));
        std::vector<std::string> expected = {};
        ASSERT_EQ(expected, real);
    }
}

TEST_F(GroupPolicyTest, test_DefaultLayerGroupPolicy_filterNeedWriteGroups_success) {
    size_t                   tp_size                         = 2;
    std::vector<std::string> full_tags                       = {"0"};
    std::vector<std::string> other_tags                      = {"1", "2"};
    uint32_t                 linear_attention_write_interval = 0;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_LAYER_DEFAULT,
                    4,
                    full_tags,
                    other_tags,
                    linear_attention_write_interval);
    {
        auto resource = makeResource({0, 1, 2, 3}, {{"0", {0, 1, 2, 3}}, {"1", {4, 5, 6, 7}}, {"2", {8, 9, 10, 11}}});
        std::vector<std::string> real;
        ASSERT_TRUE(group_policy_->getNeedWriteGroups(resource, real));
        std::vector<std::string> expected = {"F0G1G2", "F0G1G2", "F0G1G2", "F0G1G2"};
        ASSERT_EQ(expected, real);
    }
    {
        auto                     resource = makeResource({0, 1, 2, 3},
                                                         {{"0", {0, 1, 2, 3}}, {"1", {4, 5, 6, 7}}, {"2", {8, 9, 10, 11}}},
                                     /*last_block_aligned=*/false);
        std::vector<std::string> real;
        ASSERT_TRUE(group_policy_->getNeedWriteGroups(resource, real));
        std::vector<std::string> expected = {"F0G1G2", "F0G1G2", "F0G1G2"};
        ASSERT_EQ(expected, real);
    }
    {
        auto resource =
            makeResource({0, 1, 2, 3, 4, 5},
                         {{"0", {0, 1, 2, 3, 20, -1}}, {"1", {4, -1, 6, 7, -1, 21}}, {"2", {8, 9, -1, 11, -1, 22}}});
        std::vector<std::string> real;
        ASSERT_TRUE(group_policy_->getNeedWriteGroups(resource, real));
        std::vector<std::string> expected = {"F0G1G2", "F0G2", "F0G1", "F0G1G2", "F0", "G1G2"};
        ASSERT_EQ(expected, real);
    }
}

TEST_F(GroupPolicyTest, test_GroupPolicy_classifies_and_names_groups_by_tag_under_topology_reorder) {
    const std::map<std::string, BlockIndicesType> blocks_by_tag = {
        {"full", {0, 1, 2, 3}}, {"linear_a", {4, 5, 6, -1}}, {"linear_b", {8, 9, 10, -1}}};

    std::map<std::string, GroupPolicy::Group>   groups_by_tag[2];
    std::map<std::string, std::vector<int>>     layers_by_tag[2];
    std::vector<std::vector<std::string>>       write_groups(2);
    const std::vector<std::vector<std::string>> declaration_orders = {{"linear_a", "linear_b"},
                                                                      {"linear_b", "linear_a"}};
    for (size_t run = 0; run < declaration_orders.size(); ++run) {
        initGroupPolicy(/*tp_size=*/1,
                        RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER,
                        /*per_group_layer_num=*/2,
                        /*full_tags=*/{"full"},
                        declaration_orders[run],
                        /*linear_attention_write_interval=*/0);
        auto typed_policy = std::dynamic_pointer_cast<FullLinearLayerGroupPolicy>(group_policy_);
        ASSERT_NE(typed_policy, nullptr);
        for (const auto& [tag, group] : typed_policy->groups()) {
            groups_by_tag[run].emplace(tag, group);
        }
        layers_by_tag[run] = typed_policy->tag_to_layer_ids_;
        auto resource      = makeResource({0, 1, 2, 3}, blocks_by_tag);
        ASSERT_TRUE(group_policy_->getNeedWriteGroups(resource, write_groups[run]));
    }

    // Classification, remote group naming and layer membership are tag-derived.
    ASSERT_EQ(groups_by_tag[0].size(), 3u);
    EXPECT_TRUE(groups_by_tag[0].at("full").is_full);
    EXPECT_FALSE(groups_by_tag[0].at("linear_a").is_full);
    EXPECT_FALSE(groups_by_tag[0].at("linear_b").is_full);
    EXPECT_EQ(groups_by_tag[0].at("full").group_name, "Ffull");
    EXPECT_EQ(groups_by_tag[0].at("linear_a").group_name, "Llinear_a");
    EXPECT_EQ(groups_by_tag[0].at("linear_b").group_name, "Llinear_b");
    for (const auto& [tag, group] : groups_by_tag[0]) {
        EXPECT_EQ(group.is_full, groups_by_tag[1].at(tag).is_full) << tag;
        EXPECT_EQ(group.group_name, groups_by_tag[1].at(tag).group_name) << tag;
        EXPECT_EQ(group.tag, tag);
    }
    EXPECT_EQ(layers_by_tag[0].at("full"), (std::vector<int>{0, 1}));
    EXPECT_EQ(layers_by_tag[1].at("full"), (std::vector<int>{0, 1}));
    EXPECT_EQ(layers_by_tag[0].at("linear_a").size(), 2u);
    EXPECT_EQ(layers_by_tag[1].at("linear_a").size(), 2u);
    EXPECT_EQ(write_groups[0], write_groups[1]);
    EXPECT_EQ(write_groups[0], (std::vector<std::string>{"Ffull", "Ffull", "FfullLlinear_aLlinear_b", "Ffull"}));
}

TEST_F(GroupPolicyTest, test_GroupPolicy_rejects_invalid_transfer_tag_payload) {
    initGroupPolicy(/*tp_size=*/1,
                    RemoteConnectorGroupMode::RCGM_ONLY_FULL_LAYER,
                    /*per_group_layer_num=*/2,
                    /*full_tags=*/{"full"});

    // A remote transfer payload carries one (tag, block) pair per cache key, so a
    // repeated tag is legal; an empty, unknown or unpaired tag is not.
    kv_cache_manager::BlockBuffers buffers;
    EXPECT_ANY_THROW(group_policy_->genBlockBuffers({}, {}, buffers));
    EXPECT_ANY_THROW(group_policy_->genBlockBuffers({""}, {1}, buffers));
    EXPECT_ANY_THROW(group_policy_->genBlockBuffers({"unknown"}, {1}, buffers));
    EXPECT_ANY_THROW(group_policy_->genBlockBuffers({"full", "full"}, {1}, buffers));
    EXPECT_TRUE(buffers.empty());
}

TEST_F(GroupPolicyTest, InvalidLaterTagIsRejectedBeforeAnyBufferLookupOrOutputMutation) {
    initGroupPolicy(/*tp_size=*/1,
                    RemoteConnectorGroupMode::RCGM_ONLY_FULL_LAYER,
                    /*per_group_layer_num=*/2,
                    /*full_tags=*/{"full"});

    kv_cache_manager::BlockBuffers buffers(1);
    EXPECT_ANY_THROW(group_policy_->genBlockBuffers({"full", "unknown"}, {7, 9}, buffers));
    EXPECT_EQ(buffers.size(), 1u);
    EXPECT_TRUE(std::static_pointer_cast<FakeKVCacheAllocator>(allocator_)->taggedBufferRequests().empty());
}

}  // namespace test
}  // namespace remote_connector
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

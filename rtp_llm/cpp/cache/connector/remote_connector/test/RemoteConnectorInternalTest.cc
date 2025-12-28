#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/connector/remote_connector/RemoteConnector.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "autil/EnvUtil.h"

using namespace rtp_llm;
using namespace rtp_llm::remote_connector;

namespace rtp_llm {

namespace remote_connector {
bool operator==(const GroupPolicy::SpecInfo& lhs, const GroupPolicy::SpecInfo& rhs) {
    return lhs.group_id == rhs.group_id && lhs.tp_rank == rhs.tp_rank;
}
}  // namespace remote_connector

namespace test {

class FakeKVCacheAllocator: public KVCacheAllocator {
public:
    FakeKVCacheAllocator(const CacheConfig&          config,
                         const std::vector<int32_t>& full_group_ids,
                         const std::vector<int32_t>& other_group_ids,
                         size_t                      per_group_layer_num):
        KVCacheAllocator(config, nullptr) {
        for (int32_t full_group_id : full_group_ids) {
            for (int i = 0; i < per_group_layer_num; i++) {
                fake_layout_.layer_to_groups.push_back(full_group_id);
            }
        }
        for (int32_t other_group_id : other_group_ids) {
            for (int i = 0; i < per_group_layer_num; i++) {
                fake_layout_.layer_to_groups.push_back(other_group_id);
            }
        }
    }
    bool init() override {
        return false;
    }
    void free(const FreeInfo& free_info) override {
        return;
    }
    void insertIntoCache(const InsertInfo& insert_info) override {
        return;
    }
    BlockAddrInfo convertIndexToAddr(int layer_id, int block_id) const override {
        return {};
    }
    BlockBufferPtrInfo convertIndexToBuffer(int layer_id, int block_id) const override {
        return {};
    }
    CacheLayerLayout layerCacheBase() const override {
        return fake_layout_;
    }
    std::vector<BufferPtr>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const override {
        return {};
    }
    std::shared_ptr<KVCacheResourceV1> incrKVCacheRef(const KVCacheResourceV1& kvcache_resource,
                                                      const CacheKeysType&     cache_keys) {
        return nullptr;
    }
    void decrKVCacheRef(const KVCacheResourceV1& kvcache_resource) {
        return;
    }
    bool updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                       const std::vector<int>&        block_src_batch,
                       bool                           copy_last_block,
                       std::vector<BlockIdPair>&      block_update_mapping) override {
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

protected:
    MallocResult incrMalloc(const MallocInfo& malloc_info) override {
        return {};
    }
    MallocResult initMallocForCommonLen(const MallocInfo& malloc_info) override {
        return {};
    }

private:
    CacheLayerLayout fake_layout_;
};

class RemoteConnectorInternalTest: public ::testing::Test {
public:
    void SetUp() override {
        rtp_llm::initLogger();
        auto mha_spec       = std::make_shared<MHAKVCacheSpec>();
        mha_spec->layer_num = layer_num_;
        // mha_spec->block_nums         = 8;
        mha_spec->local_head_num_kv  = 8;
        mha_spec->size_per_head      = 128;
        mha_spec->seq_size_per_block = 8;
        mha_spec->dtype              = rtp_llm::DataType::TYPE_FP16;
        mha_spec->type               = KVCacheType::MultiHeadAttention;
        cache_config_.block_num      = 8;
        cache_config_.cache_specs.push_back(mha_spec);
        byte_size_per_block_           = static_cast<size_t>(mha_spec->block_size_bytes() * mha_spec->layer_num);
        cache_config_.block_size_bytes = byte_size_per_block_;
    }

    void TearDown() override {}

private:
    std::shared_ptr<RemoteConnector> getFullLinearPolicyConnector() const {
        auto allocator =
            std::make_shared<FakeKVCacheAllocator>(cache_config_, full_group_ids_, linear_group_ids_, layer_num_);
        return std::shared_ptr<RemoteConnector>(new RemoteConnector(cache_config_,
                                                                    kv_cache_config_,
                                                                    runtime_config_,
                                                                    parallelism_config_,
                                                                    nullptr,
                                                                    nullptr,
                                                                    0,
                                                                    allocator,
                                                                    RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER,
                                                                    full_group_ids_,
                                                                    linear_group_ids_,
                                                                    nullptr,
                                                                    1));
    }

    CacheConfig                              cache_config_;
    KVCacheConfig                            kv_cache_config_;
    RuntimeConfig                            runtime_config_;
    ParallelismConfig                        parallelism_config_;
    size_t                                   byte_size_per_block_ = 0;
    constexpr static int                     layer_num_           = 10;
    inline static const std::vector<int32_t> full_group_ids_      = {0};
    inline static const std::vector<int32_t> linear_group_ids_    = {1, 2};
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

        connector->init_params_->lora_info_map["lora1"] = "ckpt1";
        connector->init_params_->lora_info_map["lora2"] = "ckpt2";

        auto config_map = connector->genClientConfig();
        ASSERT_EQ(2, config_map.size());
        ASSERT_GT(config_map.count("lora1"), 0);
        ASSERT_GT(config_map.count("lora2"), 0);
        ASSERT_NE(config_map.at("lora1")->instance_id_, config_map.at("lora2")->instance_id_);
    }
    {
        auto connector = getFullLinearPolicyConnector();
        ASSERT_TRUE(connector->group_policy_->init());

        auto config_map_1 = connector->genClientConfig();
        autil::EnvUtil::setEnv("BIZ_NAME", "test_biz");
        auto config_map_2 = connector->genClientConfig();
        autil::EnvUtil::unsetEnv("BIZ_NAME");
        ASSERT_NE(config_map_1.at("")->instance_id_, config_map_2.at("")->instance_id_);
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
                   {"F0L1", {"tp0_F0", "tp1_F0", "tp0_L1", "tp1_L1"}},
                   {"F0L1L2", {"tp0_F0", "tp1_F0", "tp0_L1", "tp1_L1", "tp0_L2", "tp1_L2"}},
                   {"F0L2", {"tp0_F0", "tp1_F0", "tp0_L2", "tp1_L2"}},
                   {"L1", {"tp0_L1", "tp1_L1"}},
                   {"L1L2", {"tp0_L1", "tp1_L1", "tp0_L2", "tp1_L2"}},
                   {"L2", {"tp0_L2", "tp1_L2"}}})),
              *spec_groups);
    EXPECT_EQ((std::unordered_map<uint64_t, std::string>({{0b111, "F0L1L2"},
                                                          {0b110, "L1L2"},
                                                          {0b101, "F0L2"},
                                                          {0b011, "F0L1"},
                                                          {0b100, "L2"},
                                                          {0b010, "L1"},
                                                          {0b001, "F0"}})),
              connector->group_policy_->location_spec_group_map_);
    EXPECT_EQ((GroupPolicy::SpecInfoMap({{"tp0_F0", GroupPolicy::SpecInfo({0, 0})},
                                         {"tp0_L1", GroupPolicy::SpecInfo({1, 0})},
                                         {"tp0_L2", GroupPolicy::SpecInfo({2, 0})},
                                         {"tp1_F0", GroupPolicy::SpecInfo({0, 1})},
                                         {"tp1_L1", GroupPolicy::SpecInfo({1, 1})},
                                         {"tp1_L2", GroupPolicy::SpecInfo({2, 1})}})),
              connector->group_policy_->spec_name_to_info_);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

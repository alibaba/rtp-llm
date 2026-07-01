#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/connector/IKVCacheConnectorCoordinator.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

namespace {

class RecordingCoordinator: public IKVCacheConnectorCoordinator {
public:
    bool hasActiveConnectors() const override {
        return true;
    }

    bool hasP2PConnector() const override {
        return true;
    }

    uint32_t convertToGlobalLayerId(int model_id, int layer_id) const override {
        (void)model_id;
        return static_cast<uint32_t>(layer_id);
    }

    std::shared_ptr<AsyncContext>
    asyncRead(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context) override {
        (void)connector_context;
        return nullptr;
    }

    std::shared_ptr<AsyncContext>
    asyncWrite(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context) override {
        (void)connector_context;
        return nullptr;
    }

    std::shared_ptr<AsyncContext>
    asyncWriteByLayer(int layer_id, const std::shared_ptr<KVCacheConnectorLayerContext>& layer_context) override {
        called_layer_id   = layer_id;
        captured_request  = layer_context->requestId();
        captured_resource = std::make_shared<KVCacheResource>(layer_context->kvCacheResource());
        return nullptr;
    }

    std::shared_ptr<KVCacheResource> holdKVCacheResourceForConnector(const KVCacheResource& resource,
                                                                     int                    layer_id) override {
        (void)layer_id;
        return std::make_shared<KVCacheResource>(resource);
    }

    int64_t                          captured_request = -1;
    int                              called_layer_id  = -1;
    std::shared_ptr<KVCacheResource> captured_resource;
};

CacheStoreInputs makeLinearInputs() {
    CacheStoreInputs inputs;
    inputs.context_batch_size    = 1;
    inputs.decoder_batch_size    = 0;
    inputs.tokens_per_block      = 8;
    inputs.pd_separation         = true;
    inputs.model_id              = 0;
    inputs.layer_id              = 0;
    inputs.warmup                = false;
    inputs.request_id            = torch::tensor({1234L}, torch::TensorOptions().dtype(torch::kInt64));
    inputs.request_pd_separation = torch::tensor({true}, torch::TensorOptions().dtype(torch::kBool));
    inputs.prefix_lengths_host   = torch::tensor({16}, torch::TensorOptions().dtype(torch::kInt32));
    inputs.input_lengths_host    = torch::tensor({8}, torch::TensorOptions().dtype(torch::kInt32));
    inputs.host_kv_cache_offset  = torch::tensor({{10, 11, 12}}, torch::TensorOptions().dtype(torch::kInt32));
    inputs.kv_cache_group_types_host =
        torch::tensor({static_cast<int32_t>(CacheGroupType::LINEAR)}, torch::TensorOptions().dtype(torch::kInt32));
    inputs.cache_keys = {"100", "101", "102"};
    return inputs;
}

}  // namespace

TEST(P2PLinearWriteTest, LinearGroupOnlyWritesLastBlockToConnector) {
    RecordingCoordinator coordinator;
    KvCacheInfo          kv_cache_info;

    auto inputs = makeLinearInputs();
    execWriteCacheStore(inputs, kv_cache_info, false, nullptr, &coordinator);

    ASSERT_NE(coordinator.captured_resource, nullptr);
    ASSERT_EQ(coordinator.called_layer_id, 0);
    ASSERT_EQ(coordinator.captured_request, 1234);
    ASSERT_EQ(coordinator.captured_resource->cacheKeys().size(), 1u);
    EXPECT_EQ(coordinator.captured_resource->cacheKeys()[0], 102);

    const auto& layer_blocks = coordinator.captured_resource->layerBlocks();
    ASSERT_EQ(layer_blocks.size(), 1u);
    ASSERT_EQ(layer_blocks[0]->blocks().size(), 1u);
    EXPECT_EQ(layer_blocks[0]->blocks()[0], 12);
}

}  // namespace rtp_llm

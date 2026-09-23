#include "gtest/gtest.h"
#include "rtp_llm/cpp/multimodal_processor/RemoteMultimodalProcessor.h"

namespace rtp_llm {
namespace {

class FakeMMRdmaTransport: public MMRdmaTransport {
public:
    bool exportEmbedding(const std::vector<torch::Tensor>&,
                         const std::vector<MMRdmaTensorPB::Role>&,
                         MMRdmaDescPB*) override {
        return false;
    }
    void             releaseEmbedding(const std::vector<std::string>&) override {}
    MMRdmaReadStatus readEmbedding(const MMRdmaDescPB&, std::vector<torch::Tensor>*) override {
        return MMRdmaReadStatus::RETRYABLE_ERROR;
    }
    MMRdmaReadStatus allocatePinnedBuffer(uint64_t bytes, torch::Tensor* output) override {
        *output = torch::empty({static_cast<int64_t>(bytes)}, torch::kUInt8);
        return MMRdmaReadStatus::SUCCESS;
    }
};

int     transport_creations = 0;
int64_t receive_pool_bytes  = 0;

std::shared_ptr<MMRdmaTransport> createTestTransport(const VitConfig& config, MMRdmaRole role) {
    EXPECT_EQ(role, MMRdmaRole::LLM_CLIENT);
    ++transport_creations;
    receive_pool_bytes = config.mm_rdma_max_inflight_bytes;
    return std::make_shared<FakeMMRdmaTransport>();
}

class RemoteMultimodalProcessorTest: public ::testing::Test {
protected:
    void SetUp() override {
        transport_creations = 0;
        receive_pool_bytes  = 0;
        registerMMRdmaTransportCreator(createTestTransport);
    }
    void TearDown() override {
        registerMMRdmaTransportCreator(nullptr);
    }
};

TEST_F(RemoteMultimodalProcessorTest, NonRootRanksDoNotCreateReceivePools) {
    const VitConfig config;
    for (int tp_rank = 1; tp_rank < 8; ++tp_rank) {
        RemoteMultimodalProcessor processor(MMModelConfig{}, 1024, config, tp_rank);
        EXPECT_EQ(processor.rdma_transport_, nullptr);
    }
    EXPECT_EQ(transport_creations, 0);
    EXPECT_EQ(receive_pool_bytes, 0);
}

TEST_F(RemoteMultimodalProcessorTest, EachDpReplicaKeepsItsTpRootReceivePool) {
    VitConfig config;
    config.mm_rdma_max_inflight_bytes = 4LL * 1024 * 1024 * 1024;
    RemoteMultimodalProcessor first_replica(MMModelConfig{}, 1024, config, 0);
    RemoteMultimodalProcessor second_replica(MMModelConfig{}, 1024, config, 0);
    ASSERT_NE(first_replica.rdma_transport_, nullptr);
    ASSERT_NE(second_replica.rdma_transport_, nullptr);
    EXPECT_NE(first_replica.rdma_transport_, second_replica.rdma_transport_);
    EXPECT_EQ(transport_creations, 2);
    EXPECT_EQ(receive_pool_bytes, config.mm_rdma_max_inflight_bytes);
}

TEST_F(RemoteMultimodalProcessorTest, GrpcModeDoesNotCreateReceivePoolOnRoot) {
    VitConfig config;
    config.mm_transport_mode = "grpc";
    RemoteMultimodalProcessor processor(MMModelConfig{}, 1024, config, 0);
    EXPECT_EQ(processor.rdma_transport_, nullptr);
    EXPECT_EQ(transport_creations, 0);
}

TEST_F(RemoteMultimodalProcessorTest, ChunkedOutputPreservesGlmVideoLayoutAndPositions) {
    RemoteMultimodalProcessor processor(MMModelConfig{}, 1024, VitConfig{}, 0);
    auto                      embedding = torch::arange(24, torch::kFloat32).reshape({6, 4});
    auto                      positions = torch::arange(6, torch::kInt32);
    // GLM video temporal groups carry an interleaved-layout marker in extra_input.
    auto               layout0 = torch::tensor({-53530053, 1, 0, 0}, torch::kInt32);
    auto               layout1 = torch::tensor({-53530053, 0, 0, 0}, torch::kInt32);
    MultimodalOutputPB output;
    output.add_split_size(2);
    output.add_split_size(4);
    auto result = processor.assembleRdmaOutput(
        {embedding.narrow(0, 0, 3), embedding.narrow(0, 3, 3), positions, layout0, layout1},
        {MMRdmaTensorPB::EMBEDDING,
         MMRdmaTensorPB::EMBEDDING,
         MMRdmaTensorPB::POS_ID,
         MMRdmaTensorPB::EXTRA_INPUT,
         MMRdmaTensorPB::EXTRA_INPUT},
        &output);
    ASSERT_TRUE(result.ok());
    const auto& value = result.value();
    ASSERT_EQ(value.mm_features.size(), 2);
    EXPECT_TRUE(torch::equal(value.mm_features[0], embedding.narrow(0, 0, 2)));
    EXPECT_TRUE(torch::equal(value.mm_features[1], embedding.narrow(0, 2, 4)));
    ASSERT_TRUE(value.mm_position_ids.has_value());
    EXPECT_TRUE(torch::equal(value.mm_position_ids.value()[1], positions.narrow(0, 2, 4)));
    ASSERT_TRUE(value.mm_extra_input.has_value());
    EXPECT_TRUE(torch::equal(value.mm_extra_input.value()[0], layout0));
    EXPECT_TRUE(torch::equal(value.mm_extra_input.value()[1], layout1));
}

}  // namespace
}  // namespace rtp_llm

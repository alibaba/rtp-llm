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
    MMRdmaReadStatus allocatePinnedBuffer(uint64_t, torch::Tensor*) override {
        return MMRdmaReadStatus::RETRYABLE_ERROR;
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

}  // namespace
}  // namespace rtp_llm

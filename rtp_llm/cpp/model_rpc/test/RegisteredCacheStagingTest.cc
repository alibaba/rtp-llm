#include "rtp_llm/cpp/model_rpc/RegisteredCacheStaging.h"
#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

class TestMemoryUtil: public MemoryUtil {
public:
    bool  rdma            = true;
    bool  succeeds        = true;
    int   registrations   = 0;
    int   deregistrations = 0;
    void* address         = nullptr;

    bool regUserMr(void* pointer, uint64_t bytes, bool gpu, uint64_t alignment) override {
        EXPECT_FALSE(gpu);
        EXPECT_EQ(bytes % alignment, 0u);
        address = pointer;
        ++registrations;
        return succeeds;
    }
    bool deregUserMr(void* pointer, bool gpu) override {
        EXPECT_FALSE(gpu);
        EXPECT_EQ(pointer, address);
        EXPECT_EQ(*static_cast<uint8_t*>(pointer), 42u);
        ++deregistrations;
        return true;
    }
    bool isMemoryMr(void*, uint64_t, bool, bool) override {
        return false;
    }
    bool findMemoryMr(void*, void*, uint64_t, bool, bool) override {
        return false;
    }
    bool isRdmaMode() override {
        return rdma;
    }
};

TEST(RegisteredCacheStagingTest, CallbackKeepsRegistrationAndStorageAliveAfterCancellation) {
    auto memory                                          = std::make_shared<TestMemoryUtil>();
    auto staging                                         = std::make_shared<RegisteredCacheStaging>(17, memory);
    *static_cast<uint8_t*>(staging->tensor().data_ptr()) = 42;
    std::weak_ptr<RegisteredCacheStaging> weak           = staging;
    std::shared_ptr<void>                 callback_owner(staging, staging->tensor().data_ptr());
    staging.reset();
    EXPECT_FALSE(weak.expired());
    EXPECT_EQ(memory->deregistrations, 0);
    callback_owner.reset();
    EXPECT_TRUE(weak.expired());
    EXPECT_EQ(memory->registrations, 1);
    EXPECT_EQ(memory->deregistrations, 1);
}

TEST(RegisteredCacheStagingTest, RegistrationFailureDoesNotDeregisterUnregisteredMemory) {
    auto memory      = std::make_shared<TestMemoryUtil>();
    memory->succeeds = false;
    EXPECT_THROW(RegisteredCacheStaging(17, memory), std::runtime_error);
    EXPECT_EQ(memory->registrations, 1);
    EXPECT_EQ(memory->deregistrations, 0);
}

TEST(RegisteredCacheStagingTest, NonRdmaTransportNeedsNoRegistration) {
    auto memory  = std::make_shared<TestMemoryUtil>();
    memory->rdma = false;
    { auto staging = std::make_shared<RegisteredCacheStaging>(17, memory); }
    EXPECT_EQ(memory->registrations, 0);
    EXPECT_EQ(memory->deregistrations, 0);
}

TEST(RegisteredCacheStagingTest, MissingShardFailsBeforeAnyDestinationUpload) {
    auto memory           = std::make_shared<TestMemoryUtil>();
    memory->rdma          = false;
    auto      destination = torch::full({32}, 127, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    auto      plan        = TpCacheTransferPlan::mha(16, 16, 2, 0);
    auto      source      = std::make_shared<RegisteredCacheStaging>(plan.source_bytes, memory);
    BlockInfo info{true, static_cast<int32_t>(destination.device().index()), 0, destination.data_ptr(), 32};
    EXPECT_THROW(reassembleCacheTransfers({{source, info, plan}}, info.device_index), std::invalid_argument);
    EXPECT_TRUE(torch::all(destination == 127).item<bool>());
}

TEST(RegisteredCacheStagingTest, CompleteKAndVShardsUploadInHeadOrder) {
    auto memory           = std::make_shared<TestMemoryUtil>();
    memory->rdma          = false;
    auto      destination = torch::zeros({32}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    BlockInfo info{true, static_cast<int32_t>(destination.device().index()), 0, destination.data_ptr(), 32};
    std::vector<PendingCacheCopy> copies;
    for (size_t head = 0; head < 2; ++head) {
        auto plan   = TpCacheTransferPlan::mha(16, 16, 2, head);
        auto source = std::make_shared<RegisteredCacheStaging>(plan.source_bytes, memory);
        source->tensor().narrow(0, 0, 8).fill_(10 + head);
        source->tensor().narrow(0, 8, 8).fill_(20 + head);
        copies.push_back({source, info, plan});
    }
    reassembleCacheTransfers(copies, info.device_index);
    auto host = destination.cpu();
    for (int head = 0; head < 2; ++head) {
        EXPECT_TRUE(torch::all(host.narrow(0, head * 8, 8) == 10 + head).item<bool>());
        EXPECT_TRUE(torch::all(host.narrow(0, 16 + head * 8, 8) == 20 + head).item<bool>());
    }
}

}  // namespace
}  // namespace rtp_llm

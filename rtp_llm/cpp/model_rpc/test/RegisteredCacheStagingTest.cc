#include "rtp_llm/cpp/model_rpc/RegisteredCacheStaging.h"
#include "gtest/gtest.h"
#include <limits>
#include <unordered_set>

namespace rtp_llm {
namespace {

class TestMemoryUtil: public MemoryUtil {
public:
    bool                      rdma                   = true;
    bool                      succeeds               = true;
    uint64_t                  max_registration_bytes = std::numeric_limits<uint64_t>::max();
    int                       registrations          = 0;
    int                       deregistrations        = 0;
    std::unordered_set<void*> addresses;

    bool regUserMr(void* pointer, uint64_t bytes, bool gpu, uint64_t alignment) override {
        EXPECT_FALSE(gpu);
        EXPECT_EQ(bytes % alignment, 0u);
        EXPECT_EQ(bytes, alignment);
        if (succeeds && bytes <= max_registration_bytes) {
            addresses.insert(pointer);
        }
        ++registrations;
        return succeeds && bytes <= max_registration_bytes;
    }
    bool deregUserMr(void* pointer, bool gpu) override {
        EXPECT_FALSE(gpu);
        EXPECT_EQ(addresses.erase(pointer), 1u);
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

TEST(RegisteredCacheStagingTest, ManyFragmentsShareRegistrationAndKeepSlabAlive) {
    auto                                                 memory = std::make_shared<TestMemoryUtil>();
    std::vector<std::shared_ptr<RegisteredCacheStaging>> fragments;
    {
        RegisteredCacheStagingPool pool(memory);
        for (size_t i = 0; i < 4096; ++i) {
            auto fragment = pool.allocate(17);
            fragment->tensor().fill_(42);
            EXPECT_EQ(fragment->source_bytes, 17u);
            if (!fragments.empty()) {
                EXPECT_EQ(static_cast<uint8_t*>(fragment->tensor().data_ptr())
                              - static_cast<uint8_t*>(fragments.back()->tensor().data_ptr()),
                          64);
            }
            fragments.push_back(std::move(fragment));
        }
        EXPECT_EQ(memory->registrations, 1);
        EXPECT_THROW(pool.allocate(0), std::invalid_argument);
    }
    EXPECT_EQ(memory->deregistrations, 0);
    EXPECT_EQ(fragments.back()->tensor()[0].item<int>(), 42);
    fragments.clear();
    EXPECT_EQ(memory->deregistrations, 1);
}

TEST(RegisteredCacheStagingTest, SlabsShrinkToFitSingleRegistration) {
    auto memory                    = std::make_shared<TestMemoryUtil>();
    memory->max_registration_bytes = 64 * 1024;
    RegisteredCacheStagingPool pool(memory);
    auto                       first = pool.allocate(60000);
    first->tensor().fill_(42);
    // Exhaust the slab; a second valid single MR is allocated, not a slice
    // crossing the split-registration boundary.
    first.reset();
    auto second = pool.allocate(60000);
    second->tensor().fill_(42);
    EXPECT_EQ(second->source_bytes, 60000u);
    EXPECT_EQ(memory->deregistrations, 1);
    EXPECT_THROW(pool.allocate(65537), std::runtime_error);
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

std::vector<PendingCacheCopy> makeTwoPartCopies(const std::shared_ptr<MemoryUtil>& memory,
                                                const BlockInfo&                   info,
                                                std::vector<LinearTpCacheSlice>    second_part_slices) {
    TpCacheTransferPlan first{16, 32, 2, 0, {{0, 0, 16}}};
    TpCacheTransferPlan second{16, 32, 2, 1, std::move(second_part_slices)};
    return {{std::make_shared<RegisteredCacheStaging>(16, memory), info, first},
            {std::make_shared<RegisteredCacheStaging>(16, memory), info, second}};
}

TEST(RegisteredCacheStagingTest, SliceGapFailsBeforeAnyDestinationUpload) {
    auto memory           = std::make_shared<TestMemoryUtil>();
    memory->rdma          = false;
    auto      destination = torch::full({32}, 127, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    BlockInfo info{true, static_cast<int32_t>(destination.device().index()), 0, destination.data_ptr(), 32};
    // First shard covers [0, 16), second shard covers [20, 32): bytes 16..19 stay untiled.
    auto copies = makeTwoPartCopies(memory, info, {{0, 20, 12}});
    EXPECT_THROW(reassembleCacheTransfers(copies, info.device_index), std::invalid_argument);
    EXPECT_TRUE(torch::all(destination == 127).item<bool>());
}

TEST(RegisteredCacheStagingTest, SliceOverlapFailsBeforeAnyDestinationUpload) {
    auto memory           = std::make_shared<TestMemoryUtil>();
    memory->rdma          = false;
    auto      destination = torch::full({32}, 127, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    BlockInfo info{true, static_cast<int32_t>(destination.device().index()), 0, destination.data_ptr(), 32};
    // First shard covers [0, 16), second shard covers [8, 24): bytes 8..15 are written by both.
    auto copies = makeTwoPartCopies(memory, info, {{0, 8, 16}});
    EXPECT_THROW(reassembleCacheTransfers(copies, info.device_index), std::invalid_argument);
    EXPECT_TRUE(torch::all(destination == 127).item<bool>());
}

TEST(RegisteredCacheStagingTest, ShortCoverageFailsBeforeAnyDestinationUpload) {
    auto memory           = std::make_shared<TestMemoryUtil>();
    memory->rdma          = false;
    auto      destination = torch::full({32}, 127, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    BlockInfo info{true, static_cast<int32_t>(destination.device().index()), 0, destination.data_ptr(), 32};
    // Slices are consistent and ordered but only tile [0, 24) of the 32-byte destination.
    auto copies = makeTwoPartCopies(memory, info, {{0, 16, 8}});
    EXPECT_THROW(reassembleCacheTransfers(copies, info.device_index), std::invalid_argument);
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

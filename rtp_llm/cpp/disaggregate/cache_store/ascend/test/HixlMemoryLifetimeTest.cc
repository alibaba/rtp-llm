// Copyright (c) 2026 Alibaba Cloud.
// SPDX-License-Identifier: Apache-2.0
#include "rtp_llm/cpp/disaggregate/cache_store/ascend/HixlMemoryUtil.h"

#include <functional>
#include <gtest/gtest.h>

namespace {
struct SdkState {
    int                   registers         = 0;
    int                   deregisters       = 0;
    int                   finalizes         = 0;
    hixl::Status          register_status   = hixl::SUCCESS;
    hixl::Status          deregister_status = hixl::SUCCESS;
    std::function<void()> on_deregister;
    std::function<void()> on_finalize;
};
SdkState sdk;
}  // namespace

// Replace the SDK boundary so ownership and failure paths run without an NPU.
namespace hixl {
class Hixl::HixlImpl {};
Hixl::Hixl()  = default;
Hixl::~Hixl() = default;
Status Hixl::Initialize(const AscendString&, const std::map<AscendString, AscendString>&) {
    return SUCCESS;
}
Status Hixl::RegisterMem(const MemDesc&, MemType, MemHandle& handle) {
    ++sdk.registers;
    handle = reinterpret_cast<MemHandle>(static_cast<uintptr_t>(sdk.registers));
    return sdk.register_status;
}
Status Hixl::DeregisterMem(MemHandle) {
    ++sdk.deregisters;
    if (sdk.on_deregister)
        sdk.on_deregister();
    return sdk.deregister_status;
}
void Hixl::Finalize() {
    ++sdk.finalizes;
    if (sdk.on_finalize)
        sdk.on_finalize();
}
}  // namespace hixl

extern "C" aclError aclrtSetCurrentContext(aclrtContext) {
    return ACL_SUCCESS;
}
extern "C" aclError aclrtGetCurrentContext(aclrtContext* context) {
    *context = reinterpret_cast<aclrtContext>(1);
    return ACL_SUCCESS;
}

namespace rtp_llm {
namespace {
class HixlMemoryLifetimeTest: public ::testing::Test {
protected:
    void SetUp() override {
        sdk = SdkState{};
    }
    std::shared_ptr<HixlMemoryUtil> memory() {
        return std::make_shared<HixlMemoryUtil>(std::make_shared<hixl::Hixl>(), reinterpret_cast<aclrtContext>(1));
    }
};

TEST_F(HixlMemoryLifetimeTest, ExplicitDeregisterRetainsStorageUntilSdkCompletes) {
    auto               util    = memory();
    auto               storage = std::make_shared<int>(7);
    std::weak_ptr<int> alive   = storage;
    void*              address = storage.get();
    ASSERT_TRUE(util->regUserMrWithOwner(address, sizeof(int), true, 0, storage));
    storage.reset();
    EXPECT_FALSE(alive.expired());
    sdk.on_deregister = [&] { EXPECT_FALSE(alive.expired()); };
    ASSERT_TRUE(util->deregUserMr(address, true));
    EXPECT_TRUE(alive.expired());
    EXPECT_EQ(sdk.deregisters, 1);
}

TEST_F(HixlMemoryLifetimeTest, ShutdownRetainsAllStorageThroughFinalize) {
    auto               util         = memory();
    auto               first        = std::make_shared<int>(1);
    auto               second       = std::make_shared<int>(2);
    std::weak_ptr<int> first_alive  = first;
    std::weak_ptr<int> second_alive = second;
    ASSERT_TRUE(util->regUserMrWithOwner(first.get(), sizeof(int), true, 0, first));
    ASSERT_TRUE(util->regUserMrWithOwner(second.get(), sizeof(int), true, 0, second));
    first.reset();
    second.reset();
    sdk.on_deregister = [&] {
        EXPECT_FALSE(first_alive.expired());
        EXPECT_FALSE(second_alive.expired());
    };
    sdk.on_finalize = sdk.on_deregister;
    util.reset();
    EXPECT_EQ(sdk.deregisters, 2);
    EXPECT_EQ(sdk.finalizes, 1);
    EXPECT_TRUE(first_alive.expired());
    EXPECT_TRUE(second_alive.expired());
}

TEST_F(HixlMemoryLifetimeTest, FrozenRegistrationRetainsStorageUntilDisconnected) {
    auto               util    = memory();
    auto               storage = std::make_shared<int>(7);
    std::weak_ptr<int> alive   = storage;
    void*              address = storage.get();
    ASSERT_TRUE(util->regUserMrWithOwner(address, sizeof(int), true, 0, storage));
    storage.reset();
    util->freezeRegistrations();
    EXPECT_FALSE(util->deregUserMr(address, true));
    EXPECT_FALSE(alive.expired());
    EXPECT_EQ(sdk.deregisters, 0);
    util->thawAfterDisconnect();
    ASSERT_TRUE(util->deregUserMr(address, true));
    EXPECT_TRUE(alive.expired());
}

TEST_F(HixlMemoryLifetimeTest, DuplicateBorrowedRegistrationKeepsExistingOwner) {
    auto               util    = memory();
    auto               storage = std::make_shared<int>(7);
    std::weak_ptr<int> alive   = storage;
    void*              address = storage.get();
    ASSERT_TRUE(util->regUserMrWithOwner(address, sizeof(int), true, 0, storage));
    storage.reset();
    ASSERT_TRUE(util->regUserMr(address, sizeof(int), true));
    EXPECT_EQ(sdk.registers, 1);
    EXPECT_FALSE(alive.expired());
    ASSERT_TRUE(util->deregUserMr(address, true));
    EXPECT_TRUE(alive.expired());
}

TEST_F(HixlMemoryLifetimeTest, DuplicateRegistrationCanAcquireStorageOwnership) {
    auto               util    = memory();
    auto               storage = std::make_shared<int>(7);
    std::weak_ptr<int> alive   = storage;
    void*              address = storage.get();
    ASSERT_TRUE(util->regUserMr(address, sizeof(int), true));
    ASSERT_TRUE(util->regUserMrWithOwner(address, sizeof(int), true, 0, storage));
    storage.reset();
    EXPECT_EQ(sdk.registers, 1);
    EXPECT_FALSE(alive.expired());
    ASSERT_TRUE(util->deregUserMr(address, true));
    EXPECT_TRUE(alive.expired());
}

TEST_F(HixlMemoryLifetimeTest, FailedRegistrationDoesNotRetainStorage) {
    auto               util    = memory();
    auto               storage = std::make_shared<int>(7);
    std::weak_ptr<int> alive   = storage;
    sdk.register_status        = hixl::FAILED;
    EXPECT_FALSE(util->regUserMrWithOwner(storage.get(), sizeof(int), true, 0, storage));
    storage.reset();
    EXPECT_TRUE(alive.expired());
    EXPECT_EQ(sdk.deregisters, 0);
}

TEST_F(HixlMemoryLifetimeTest, FailedDeregisterKeepsStorageForSubsequentCleanup) {
    auto               util    = memory();
    auto               storage = std::make_shared<int>(7);
    std::weak_ptr<int> alive   = storage;
    void*              address = storage.get();
    ASSERT_TRUE(util->regUserMrWithOwner(address, sizeof(int), true, 0, storage));
    storage.reset();
    sdk.deregister_status = hixl::FAILED;
    EXPECT_FALSE(util->deregUserMr(address, true));
    EXPECT_FALSE(alive.expired());
    sdk.deregister_status = hixl::SUCCESS;
    EXPECT_TRUE(util->deregUserMr(address, true));
    EXPECT_TRUE(alive.expired());
}
}  // namespace
}  // namespace rtp_llm

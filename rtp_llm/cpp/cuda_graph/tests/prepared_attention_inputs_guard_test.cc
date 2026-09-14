#include "rtp_llm/cpp/cuda_graph/prepared_attention_inputs_guard.h"

#include <atomic>
#include <stdexcept>

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

TEST(PreparedAttentionInputsGuardTest, FailedAttemptIsInvalidatedBeforeSuccessfulRetry) {
    std::atomic<bool> prepared{true};

    EXPECT_THROW(
        {
            PreparedAttentionInputsGuard guard(prepared);
            EXPECT_FALSE(prepared.load(std::memory_order_acquire));
            throw std::runtime_error("injected prepare failure");
        },
        std::runtime_error);
    EXPECT_FALSE(prepared.load(std::memory_order_acquire));

    {
        PreparedAttentionInputsGuard guard(prepared);
        EXPECT_FALSE(prepared.load(std::memory_order_acquire));
        guard.commit();
    }
    EXPECT_TRUE(prepared.load(std::memory_order_acquire));
}

}  // namespace
}  // namespace rtp_llm

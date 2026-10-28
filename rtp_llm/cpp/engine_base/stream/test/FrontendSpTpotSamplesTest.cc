#include "rtp_llm/cpp/engine_base/stream/FrontendSpTpotSamples.h"
#include <gtest/gtest.h>
#include <chrono>

namespace rtp_llm {
TEST(FrontendSpTpotSamplesTest, FinalFrameWaitsForPendingSampleAndDrainsOnce) {
    auto mailbox = std::make_shared<FrontendSpTpotSamples>();
    auto pending = mailbox->begin();
    // The response keeps the mailbox after its stream/owner is released.
    auto final_frame_mailbox = mailbox;
    mailbox.reset();
    std::promise<void> entered;
    auto               reader = std::async(std::launch::async, [&] {
        entered.set_value();
        return final_frame_mailbox->take();
    });
    entered.get_future().wait();
    EXPECT_EQ(reader.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
    pending->complete(12000.0);
    const auto samples = reader.get();
    ASSERT_EQ(samples.size(), 1u);
    EXPECT_EQ(samples[0].second, 12000.0);
    EXPECT_TRUE(final_frame_mailbox->take().empty());
}
TEST(FrontendSpTpotSamplesTest, CancellationReleasesReaderAndCoalescingPreservesSamples) {
    FrontendSpTpotSamples mailbox;
    auto                  cancelled = mailbox.begin();
    cancelled.reset();
    auto second = mailbox.begin();
    auto third  = mailbox.begin();
    second->complete(7500.0);
    third->complete(2000.0);
    const auto samples = mailbox.take();
    ASSERT_EQ(samples.size(), 2u);
    EXPECT_EQ(samples[0].first, 2u);
    EXPECT_EQ(samples[1].first, 3u);
    EXPECT_TRUE(mailbox.take().empty());
}
}  // namespace rtp_llm

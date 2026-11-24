#include "rtp_llm/cpp/normal_engine/PressureStepRecorder.h"
#include "gtest/gtest.h"

namespace rtp_llm {
TEST(PressureStepRecorderTest, ReportsLostRecordsWithoutBridgingSteps) {
    PressureStepRecorder recorder(2);
    recorder.enable("s");
    for (uint64_t id = 1; id <= 3; ++id) {
        PressureStepRecord record;
        record.step_id       = id;
        record.context_batch = 1;
        record.complete      = true;
        recorder.append(record);
    }
    auto page = recorder.read(0, 10);
    ASSERT_EQ(page.records.size(), 2u);
    EXPECT_TRUE(page.lost);
    EXPECT_EQ(page.records.front().step_id, 2u);
    EXPECT_EQ(page.next_sequence, 3u);
    EXPECT_FALSE(recorder.disable("another-session"));
    EXPECT_TRUE(recorder.disable("s"));
}
TEST(PressureStepRecorderTest, SessionsAreValidatedIdempotentAndIsolated) {
    EXPECT_THROW(PressureStepRecorder(0), std::invalid_argument);
    EXPECT_THROW(PressureStepRecorder(65537), std::invalid_argument);
    PressureStepRecorder recorder(2);
    EXPECT_THROW(recorder.enable(""), std::invalid_argument);
    recorder.enable("first");
    PressureStepRecord record;
    record.step_id       = 1;
    record.context_batch = 2;
    record.complete      = true;
    record.request_ids   = {"11", "12"};
    recorder.append(record);
    recorder.enable("first");
    EXPECT_THROW(recorder.enable("second"), std::invalid_argument);
    EXPECT_THROW(recorder.enable("first", 3), std::invalid_argument);
    EXPECT_EQ(recorder.read("first", 0, 10).records.size(), 1);
    EXPECT_FALSE(recorder.read("second", 0, 10).available);
    EXPECT_FALSE(recorder.disable("second"));
    EXPECT_TRUE(recorder.disable("first"));
    EXPECT_TRUE(recorder.disable("first"));
    recorder.append(record);
    EXPECT_EQ(recorder.read("first", 0, 10).records.size(), 1);
    const auto cursor = recorder.read("first", 0, 10).next_sequence;
    EXPECT_THROW(recorder.enable("first"), std::invalid_argument);
    EXPECT_FALSE(recorder.enabled());
    EXPECT_EQ(recorder.read("first", 0, 10).next_sequence, cursor);
    EXPECT_TRUE(recorder.read("first", cursor, 10).records.empty());
    EXPECT_FALSE(recorder.read("first", cursor, 10).lost);
    recorder.enable("second", 3);
    EXPECT_FALSE(recorder.read("first", 0, 10).available);
    EXPECT_TRUE(recorder.read("second", 0, 10).records.empty());
    EXPECT_EQ(recorder.candidateId(), 1);
}

TEST(PressureStepRecorderTest, ActiveExecutionIsIndependentOfCompletedCursorAndCopies) {
    PressureStepRecorder recorder(2);
    recorder.enable("s");
    recorder.markPhase("s", "measurement");
    PressureStepRecord record;
    record.step_id       = 1;
    record.tokens        = 8;
    record.context_batch = 2;
    record.request_ids   = {"11", "12"};
    recorder.begin(record);
    auto active = recorder.read("s", 999, 0);
    ASSERT_TRUE(active.active);
    EXPECT_EQ(active.active->step_id, 1);
    EXPECT_EQ(active.active->sequence, 0);
    EXPECT_EQ(active.active->phase, "measurement");
    EXPECT_EQ(active.next_sequence, 999);
    recorder.markPhase("s", "tail");
    record.complete = true;
    recorder.append(record);
    EXPECT_FALSE(recorder.read("s", 1, 10).active);
    EXPECT_TRUE(active.active);
    auto copy = recorder.read("s", 0, 10);
    EXPECT_EQ(copy.records.front().phase, "measurement");
    copy.records.front().request_ids.clear();
    EXPECT_EQ(recorder.read("s", 0, 10).records.front().request_ids.size(), 2);
    record.step_id       = 2;
    record.context_batch = 0;
    record.decode_batch  = 1;
    recorder.begin(record);
    EXPECT_EQ(recorder.read("s", 1, 1).active->decode_batch, 1);
    recorder.abort();
    EXPECT_FALSE(recorder.read("s", 1, 1).active);
}

TEST(PressureStepRecorderTest, DisabledEmptyAndIncompleteStepsDoNotAdvanceCursor) {
    PressureStepRecorder recorder(4096);
    PressureStepRecord   record;
    record.step_id       = 1;
    record.complete      = true;
    record.context_batch = 1;
    recorder.append(record);
    EXPECT_EQ(recorder.candidateId(), 0);
    recorder.enable("s");
    record.context_batch = 0;
    recorder.append(record);
    record.context_batch = 1;
    record.complete      = false;
    recorder.append(record);
    EXPECT_EQ(recorder.candidateId(), 1);
    EXPECT_EQ(recorder.read(0, 10).next_sequence, 0);
    record.complete = true;
    for (uint64_t i = 1; i <= 1100; ++i) {
        record.step_id = i;
        recorder.append(record);
    }
    auto page = recorder.read(0, 9999);
    EXPECT_EQ(page.records.size(), 1024);
    EXPECT_EQ(page.next_sequence, 1024);
    EXPECT_FALSE(page.lost);
    page = recorder.read(page.next_sequence, 3);
    EXPECT_EQ(page.records.front().step_id, 1025);
    EXPECT_EQ(page.next_sequence, 1027);
}
}  // namespace rtp_llm

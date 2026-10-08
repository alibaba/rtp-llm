#include "rtp_llm/models_py/bindings/LinearReplay.h"

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

TEST(LinearReplayInputsTest, SequenceParallelDummyRowsCannotPublishState) {
    auto replay = LinearReplayInputs::allocate(2, 3, torch::kCPU);
    for (auto* tensor : replay.tensors()) {
        tensor->fill_(7);
    }
    replay.slot_ids.copy_(torch::tensor({2, 5}, torch::kInt32));
    replay.active_block_ids.copy_(torch::tensor({{11, 12}, {21, 22}, {31, 32}}, torch::kInt32));
    replay.state_read_block_ids.copy_(torch::tensor({{13, 14}, {23, 24}, {33, 34}}, torch::kInt32));

    auto padded = replay.padToBatch(4);
    EXPECT_EQ(padded.slot_ids.size(0), 4);
    EXPECT_EQ(padded.active_block_ids.size(0), 3);
    EXPECT_EQ(padded.active_block_ids.size(1), 4);
    EXPECT_EQ(padded.slot_ids[0].item<int>(), 2);
    EXPECT_EQ(padded.slot_ids[1].item<int>(), 5);
    for (int row = 2; row < 4; ++row) {
        EXPECT_EQ(padded.slot_ids[row].item<int>(), -1);
        EXPECT_EQ(padded.slot_generations[row].item<int64_t>(), 0);
        EXPECT_EQ(padded.verify_epochs[row].item<int64_t>(), 0);
        EXPECT_EQ(padded.prev_accept_lengths[row].item<int>(), 0);
        for (int group = 0; group < 3; ++group) {
            EXPECT_EQ(padded.active_block_ids[group][row].item<int>(), -1);
            EXPECT_EQ(padded.state_read_block_ids[group][row].item<int>(), -1);
        }
    }
    // Padding creates new request rows without changing the original round.
    EXPECT_EQ(replay.slot_ids.size(0), 2);
    EXPECT_EQ(replay.active_block_ids[2][1].item<int>(), 32);
}

TEST(LinearReplayInputsTest, RejectsInconsistentRequestRows) {
    auto replay = LinearReplayInputs::allocate(2, 2, torch::kCPU);
    replay.verify_epochs = torch::zeros({3}, torch::kInt64);
    EXPECT_THROW(replay.padToBatch(4), c10::Error);
    EXPECT_THROW(replay.padToBatch(1), c10::Error);
}

}  // namespace
}  // namespace rtp_llm

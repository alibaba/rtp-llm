#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/core/Types.h"
#include "torch/all.h"
#include "gtest/gtest.h"
#include <memory>

#define private public
#include "rtp_llm/cpp/models/position_ids/PositionIdsGenerator.h"
#include "rtp_llm/cpp/testing/TestBase.h"

using namespace std;

namespace rtp_llm {

class PositionIdsGeneratorTest: public DeviceTestBase {};

TEST_F(PositionIdsGeneratorTest, testSimple) {
    auto res = PositionIdsGenerator::generatePositionIds(3, PositionIdsStyle::DEFAULT);
    EXPECT_EQ(3, res.numel());
    int32_t* pos_ids = res.data_ptr<int32_t>();
    EXPECT_EQ(0, pos_ids[0]);
    EXPECT_EQ(1, pos_ids[1]);
    EXPECT_EQ(2, pos_ids[2]);

    int32_t now_pos;
    PositionIdsGenerator::generateNextPositionId(&now_pos, 10);
    EXPECT_EQ(9, now_pos);
}

TEST_F(PositionIdsGeneratorTest, testMMWithType) {
    auto res = PositionIdsGenerator::generatePositionIds(3, PositionIdsStyle::MMWITHTAG);
    EXPECT_EQ(3, res.numel());
    int32_t* pos_ids = res.data_ptr<int32_t>();
    EXPECT_EQ(0, pos_ids[0]);
    EXPECT_EQ(1, pos_ids[1]);
    EXPECT_EQ(2, pos_ids[2]);

    auto mm_locs = torch::tensor({1, 5}, torch::kInt32);

    vector<torch::Tensor> mm_position_ids;

    mm_position_ids.emplace_back(torch::tensor({2, 3}, torch::kInt32));
    mm_position_ids.emplace_back(torch::tensor({2, 2, 2, 6}, torch::kInt32));

    res     = PositionIdsGenerator::generatePositionIds(10, PositionIdsStyle::MMWITHTAG, mm_locs, mm_position_ids);
    pos_ids = res.data_ptr<int32_t>();
    // 0, 3, 4, 5, 6, 9, 9, 9, 13, 14
    EXPECT_EQ(0, pos_ids[0]);
    EXPECT_EQ(3, pos_ids[1]);
    EXPECT_EQ(9, pos_ids[5]);
    EXPECT_EQ(13, pos_ids[8]);
    EXPECT_EQ(14, pos_ids[9]);

    int32_t now_pos;
    PositionIdsGenerator::generateNextPositionId(&now_pos, 20, PositionIdsStyle::MMWITHTAG, res);
    EXPECT_EQ(24, now_pos);
}

TEST_F(PositionIdsGeneratorTest, testMrope) {
    auto res = PositionIdsGenerator::generatePositionIds(3, PositionIdsStyle::MROPE);
    EXPECT_EQ(9, res.numel());
    int32_t* pos_ids = res.data_ptr<int32_t>();
    EXPECT_EQ(0, pos_ids[0]);
    EXPECT_EQ(0, pos_ids[1]);
    EXPECT_EQ(0, pos_ids[2]);
    EXPECT_EQ(9, res.numel());

    auto mm_locs = torch::tensor({1}, torch::kInt32);

    vector<torch::Tensor> mm_position_ids;

    {
        auto flat_tensor = torch::tensor({1, 3, 2, 1, 9, 4}, torch::kInt32).reshape({2, 3});
        mm_position_ids.emplace_back(flat_tensor);
    }

    res     = PositionIdsGenerator::generatePositionIds(10, PositionIdsStyle::MROPE, mm_locs, mm_position_ids);
    pos_ids = res.data_ptr<int32_t>();
    // 0, 0, 0
    // 2, 4, 3
    // 2, 10, 5
    // 11, 11, 11
    // 12, 12, 12
    // ...
    // 17, 17, 17
    EXPECT_EQ(0, pos_ids[0]);
    EXPECT_EQ(0, pos_ids[1]);
    EXPECT_EQ(0, pos_ids[2]);
    EXPECT_EQ(2, pos_ids[3]);
    EXPECT_EQ(4, pos_ids[4]);
    EXPECT_EQ(3, pos_ids[5]);
    EXPECT_EQ(2, pos_ids[6]);
    EXPECT_EQ(10, pos_ids[7]);
    EXPECT_EQ(5, pos_ids[8]);
    EXPECT_EQ(17, pos_ids[27]);
    EXPECT_EQ(17, pos_ids[28]);
    EXPECT_EQ(17, pos_ids[29]);
    EXPECT_EQ(30, res.numel());

    int32_t now_pos[3];
    PositionIdsGenerator::generateNextPositionId(now_pos, 20, PositionIdsStyle::MROPE, res);
    EXPECT_EQ(27, now_pos[0]);
}

}  // namespace rtp_llm

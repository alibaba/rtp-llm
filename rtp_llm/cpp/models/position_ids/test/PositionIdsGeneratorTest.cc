#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/core/Types.h"
#include "torch/all.h"
#include "gtest/gtest.h"
#include <memory>

#define private public
#include "rtp_llm/cpp/models/position_ids/PositionIdsGenerator.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/core/BufferHelper.h"

using namespace std;

namespace rtp_llm {

class PositionIdsGeneratorTest: public DeviceTestBase {};

TEST_F(PositionIdsGeneratorTest, testSimple) {
    BufferPtr res = PositionIdsGenerator::generatePositionIds(device_, 3, PositionIdsStyle::DEFAULT);
    EXPECT_EQ(3, res->size());
    int32_t* pos_ids = res->data<int32_t>();
    EXPECT_EQ(0, pos_ids[0]);
    EXPECT_EQ(1, pos_ids[1]);
    EXPECT_EQ(2, pos_ids[2]);

    int32_t now_pos;
    PositionIdsGenerator::generateNextPositionId(&now_pos, 10);
    EXPECT_EQ(9, now_pos);
}

TEST_F(PositionIdsGeneratorTest, testMMWithType) {
    BufferPtr res = PositionIdsGenerator::generatePositionIds(device_, 3, PositionIdsStyle::MMWITHTAG);
    EXPECT_EQ(3, res->size());
    int32_t* pos_ids = res->data<int32_t>();
    EXPECT_EQ(0, pos_ids[0]);
    EXPECT_EQ(1, pos_ids[1]);
    EXPECT_EQ(2, pos_ids[2]);

    BufferPtr mm_locs    = device_->allocateBuffer({DataType::TYPE_INT32, {(size_t)2}, AllocationType::HOST});
    int32_t*  mm_loc_vec = mm_locs->data<int32_t>();
    mm_loc_vec[0]        = 1;
    mm_loc_vec[1]        = 5;

    vector<BufferPtr> mm_position_ids;

    mm_position_ids.emplace_back(device_->allocateBuffer({DataType::TYPE_INT32, {(size_t)2}, AllocationType::HOST}));
    int32_t* mm_pos_id_vec = mm_position_ids[0]->data<int32_t>();
    mm_pos_id_vec[0]       = 2;
    mm_pos_id_vec[1]       = 3;

    mm_position_ids.emplace_back(device_->allocateBuffer({DataType::TYPE_INT32, {(size_t)4}, AllocationType::HOST}));
    mm_pos_id_vec    = mm_position_ids[1]->data<int32_t>();
    mm_pos_id_vec[0] = 2;
    mm_pos_id_vec[1] = 2;
    mm_pos_id_vec[2] = 2;
    mm_pos_id_vec[3] = 6;

    res = PositionIdsGenerator::generatePositionIds(device_, 10, PositionIdsStyle::MMWITHTAG, mm_locs, mm_position_ids);
    pos_ids = res->data<int32_t>();
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
    BufferPtr res = PositionIdsGenerator::generatePositionIds(device_, 3, PositionIdsStyle::MROPE);
    EXPECT_EQ(9, res->size());
    int32_t* pos_ids = res->data<int32_t>();
    EXPECT_EQ(0, pos_ids[0]);
    EXPECT_EQ(0, pos_ids[1]);
    EXPECT_EQ(0, pos_ids[2]);
    EXPECT_EQ(9, res->size());

    BufferPtr mm_locs    = device_->allocateBuffer({DataType::TYPE_INT32, {(size_t)1}, AllocationType::HOST});
    int32_t*  mm_loc_vec = mm_locs->data<int32_t>();
    mm_loc_vec[0]        = 1;

    vector<BufferPtr> mm_position_ids;

    mm_position_ids.emplace_back(
        device_->allocateBuffer({DataType::TYPE_INT32, {(size_t)2, (size_t)3}, AllocationType::HOST}));
    int32_t* mm_pos_id_vec = mm_position_ids[0]->data<int32_t>();
    mm_pos_id_vec[0]       = 1;
    mm_pos_id_vec[1]       = 3;
    mm_pos_id_vec[2]       = 2;
    mm_pos_id_vec[3]       = 1;
    mm_pos_id_vec[4]       = 9;
    mm_pos_id_vec[5]       = 4;

    res     = PositionIdsGenerator::generatePositionIds(device_, 10, PositionIdsStyle::MROPE, mm_locs, mm_position_ids);
    pos_ids = res->data<int32_t>();
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
    EXPECT_EQ(30, res->size());

    int32_t now_pos[3];
    PositionIdsGenerator::generateNextPositionId(now_pos, 20, PositionIdsStyle::MROPE, res);
    EXPECT_EQ(27, now_pos[0]);
}

}  // namespace rtp_llm

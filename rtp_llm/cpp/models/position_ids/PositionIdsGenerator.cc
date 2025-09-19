#include "rtp_llm/cpp/models/position_ids/PositionIdsGenerator.h"

using namespace std;

namespace rtp_llm {

BufferPtr PositionIdsGenerator::generatePositionIds(DeviceBase*                 device,
                                                    int32_t                     input_len,
                                                    PositionIdsStyle            style,
                                                    optional<BufferPtr>         mm_locs,
                                                    optional<vector<BufferPtr>> mm_position_ids) {
    rtp_llm::BufferPtr res;
    int32_t*           position_ids;
    switch (style) {
        case PositionIdsStyle::DEFAULT: {
            res = device->allocateBuffer(
                {rtp_llm::DataType::TYPE_INT32, {(size_t)input_len}, rtp_llm::AllocationType::HOST}, {});
            position_ids = (int32_t*)res->data();
            for (int32_t i = 0; i < input_len; ++i) {
                position_ids[i] = i;
            }
            break;
        }
        case PositionIdsStyle::MMWITHTAG: {
            res = device->allocateBuffer(
                {rtp_llm::DataType::TYPE_INT32, {(size_t)input_len}, rtp_llm::AllocationType::HOST}, {});
            position_ids    = (int32_t*)res->data();
            int32_t now_pos = 0, now_id = 0, mm_idx = 0;
            if (!mm_locs && !mm_position_ids) {
                for (int32_t i = 0; i < input_len; ++i) {
                    position_ids[i] = i;
                }
            } else {
                int32_t* mm_loc_data = (int32_t*)(mm_locs.value()->data());
                auto     mm_pos_ids  = mm_position_ids.value();
                while (now_pos < input_len) {
                    if (mm_idx >= mm_locs.value()->size() || now_pos < mm_loc_data[mm_idx]) {
                        position_ids[now_pos++] = now_id++;
                    } else {
                        for (int32_t i = 0; i < mm_pos_ids[mm_idx]->size(); ++i) {
                            position_ids[now_pos + i] = *(mm_pos_ids[mm_idx]->dataWithOffset<int32_t>(i)) + now_id;
                        }
                        now_pos += mm_pos_ids[mm_idx]->size();
                        now_id = position_ids[now_pos - 1] + 1;
                        mm_idx++;
                    }
                }
            }
            break;
        }
        case PositionIdsStyle::MROPE: {
            res = device->allocateBuffer(
                {rtp_llm::DataType::TYPE_INT32, {(size_t)input_len, (size_t)3}, rtp_llm::AllocationType::HOST}, {});
            position_ids    = (int32_t*)res->data();
            int32_t now_pos = 0, now_id = 0, mm_idx = 0;

            if (!mm_locs && !mm_position_ids) {
                for (int32_t i = 0; i < input_len; ++i) {
                    position_ids[3 * i] = position_ids[3 * i + 1] = position_ids[3 * i + 2] = i;
                }
            } else {
                int32_t* mm_loc_data = (int32_t*)(mm_locs.value()->data());
                auto     mm_pos_ids  = mm_position_ids.value();
                while (now_pos < input_len) {
                    if (mm_idx >= mm_locs.value()->size() || now_pos < mm_loc_data[mm_idx]) {
                        position_ids[now_pos * 3] = position_ids[now_pos * 3 + 1] = position_ids[now_pos * 3 + 2] =
                            now_id;
                        now_pos++;
                        now_id++;
                    } else {
                        int32_t feature_len = mm_pos_ids[mm_idx]->shape()[0];
                        for (int32_t i = 0; i < feature_len; ++i) {
                            int32_t  now_pos_tmp              = now_pos + i;
                            int32_t* mm_pos_id                = mm_pos_ids[mm_idx]->data<int32_t>();
                            position_ids[3 * now_pos_tmp]     = mm_pos_id[3 * i] + now_id;
                            position_ids[3 * now_pos_tmp + 1] = mm_pos_id[3 * i + 1] + now_id;
                            position_ids[3 * now_pos_tmp + 2] = mm_pos_id[3 * i + 2] + now_id;
                        }
                        now_pos = now_pos + feature_len;
                        now_id  = std::max(position_ids[3 * now_pos - 1],
                                          std::max(position_ids[3 * now_pos - 2], position_ids[3 * now_pos - 3]))
                                 + 1;
                        mm_idx++;
                    }
                }
            }
            break;
        }
    }
    return res;
}

void PositionIdsGenerator::generateNextPositionId(int32_t*         now_pos,
                                                  int32_t          now_len,
                                                  PositionIdsStyle style,
                                                  BufferPtr        context_pos_ids) {
    int32_t context_len = 0;
    switch (style) {
        case PositionIdsStyle::DEFAULT:
            now_pos[0] = now_len - 1;
            break;
        case PositionIdsStyle::MMWITHTAG:
            context_len = context_pos_ids->size();
            now_pos[0]  = *context_pos_ids->dataWithOffset<int32_t>(context_len - 1) + now_len - context_len;
            break;
        case PositionIdsStyle::MROPE:
            context_len              = context_pos_ids->size();
            int32_t* context_pos     = context_pos_ids->data<int32_t>();
            int32_t  last_pos_max_id = std::max(context_pos[context_len - 1],
                                               std::max(context_pos[context_len - 2], context_pos[context_len - 3]));
            now_pos[0] = now_pos[1] = now_pos[2] = last_pos_max_id + now_len - context_len / 3;
            break;
    }
}

}  // namespace rtp_llm

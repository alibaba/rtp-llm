#pragma once

#include <algorithm>
#include <cstring>
#include <random>
#include <optional>
#include <vector>
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/devices/DeviceBase.h"
#include "absl/status/statusor.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"

namespace ft = fastertransformer;

namespace rtp_llm {

typedef enum
{
    DEFAULT   = 0,
    MMWITHTAG = 1, // chatglm4v
    MROPE     = 2  // qwen2vl
} positionIdsStyle;

// assume position ids in position encoding (word embedding) and rotary embedding is same
// TODO: not same -> implement different interface here and BatchStreamProcessor
class PositionIdsGenerator {
public:
    static ft::BufferPtr generate_position_ids(
        ft::DeviceBase* device, 
        int32_t input_len, 
        positionIdsStyle style = positionIdsStyle::DEFAULT, 
        std::optional<ft::BufferPtr> mm_locs = std::nullopt,
        std::optional<std::vector<ft::BufferPtr>> mm_position_ids = std::nullopt) 
    {
        ft::BufferPtr res;
        if (!mm_locs && !mm_position_ids) {
            style = positionIdsStyle::DEFAULT;
        }
        int32_t* position_ids;
        switch (style) {
            case positionIdsStyle::DEFAULT:
            {
                res = device->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)input_len}, ft::AllocationType::HOST}, {});
                position_ids = (int32_t*)res->data();
                for (int32_t i = 0;i < input_len;++i) {
                    position_ids[i] = i;
                }
                break;
            }
            case positionIdsStyle::MMWITHTAG:
            {
                res = device->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)input_len}, ft::AllocationType::HOST}, {});
                position_ids = (int32_t*)res->data();
                int32_t now_pos = 0, now_id = 0, mm_idx = 0;
                int32_t *mm_loc_data = (int32_t*)(mm_locs.value()->data());
                auto mm_pos_ids = mm_position_ids.value();
                while (now_pos < input_len) {
                    if (mm_idx >= mm_locs.value()->size() || now_pos < mm_loc_data[mm_idx]) {
                        position_ids[now_pos++] = now_id++;
                    } else {
                        for (int32_t i = 0;i < mm_pos_ids[mm_idx]->size();++i) {
                            position_ids[now_pos + i] = *(mm_pos_ids[mm_idx]->dataWithOffset<int32_t>(i)) + now_id;
                        }
                        now_pos += mm_pos_ids[mm_idx]->size();
                        now_id = position_ids[now_pos - 1] + 1;
                        mm_idx++;
                    }
                }
                break;
            }
            case positionIdsStyle::MROPE:
            {
                res = device->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)input_len, (size_t)3}, ft::AllocationType::HOST}, {});
                position_ids = (int32_t*)res->data();
                int32_t now_pos = 0, now_id = 0, mm_idx = 0;
                int32_t *mm_loc_data = (int32_t*)(mm_locs.value()->data());
                auto mm_pos_ids = mm_position_ids.value();
                while (now_pos < input_len) {
                    if (mm_idx >= mm_locs.value()->size() || now_pos < mm_loc_data[mm_idx]) {
                        position_ids[now_pos * 3] = position_ids[now_pos * 3 + 1] = position_ids[now_pos * 3 + 2] = now_id;
                        now_pos++;
                        now_id++;
                    } else {
                        int32_t feature_len = mm_pos_ids[mm_idx]->shape()[0];
                        for (int32_t i = 0;i < feature_len;++i) {
                            int32_t now_pos_tmp = now_pos + i;
                            int32_t* mm_pos_id = mm_pos_ids[mm_idx]->data<int32_t>();
                            position_ids[3 * now_pos_tmp] = mm_pos_id[3 * i] + now_id;
                            position_ids[3 * now_pos_tmp + 1] = mm_pos_id[3 * i + 1] + now_id;
                            position_ids[3 * now_pos_tmp + 2] = mm_pos_id[3 * i + 2] + now_id;
                        }
                        now_pos = now_pos + feature_len;
                        now_id = std::max(position_ids[3 * now_pos - 1], std::max(position_ids[3 * now_pos - 2], position_ids[3 * now_pos - 3])) + 1;
                        mm_idx++;
                    }
                }
                break;
            }
        }
        return res;
    }

    static void generate_next_pos_id(int32_t* now_pos,
                                     int32_t  now_len,
                                     positionIdsStyle style = positionIdsStyle::DEFAULT,
                                     ft::BufferPtr context_pos_ids = nullptr) {
        int32_t context_len = 0;
        switch (style) {
            case positionIdsStyle::DEFAULT:
                now_pos[0] = now_len - 1;
                break;
            case positionIdsStyle::MMWITHTAG:
                context_len = context_pos_ids->size();
                now_pos[0] = *context_pos_ids->dataWithOffset<int32_t>(context_len - 1) + now_len - context_len;
                break;
            case positionIdsStyle::MROPE:
                context_len = context_pos_ids->size();
                int32_t* context_pos = context_pos_ids->data<int32_t>();
                int32_t last_pos_max_id = std::max(context_pos[context_len - 1], std::max(context_pos[context_len - 2], context_pos[context_len - 3]));
                now_pos[0] = now_pos[1] = now_pos[2] = last_pos_max_id + now_len - context_len / 3;
                break;
        }
    }
};

} // namespace rtp_llm
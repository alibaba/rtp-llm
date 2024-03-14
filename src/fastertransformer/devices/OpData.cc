#pragma once

#include "src/fastertransformer/devices/OpData.h"


#include <optional>
#include <functional>
#include <sstream>

namespace fastertransformer {

void GemmParams::Check() const {
    // check dim
    RUNTIME_ASSERT_OP_ARG((A.dim() >= 2) &&
                          (B.dim() >= 2) &&
                          (D.dim() >= 2),
        "gemm params dim is must greater than 2!");
    

}

void ContextAttentionParams::check() const {
    // check dim
    FT_CHECK_WITH_INFO(qkv_input.dim() == 3, "");
    FT_CHECK_WITH_INFO(q_output.dim() == 4, "");
    FT_CHECK_WITH_INFO(k_output.dim() == 4, "");
    FT_CHECK_WITH_INFO(v_output.dim() == 4, "");
    FT_CHECK_WITH_INFO(bias.dim() == 1, "");
    FT_CHECK_WITH_INFO(position_ids.dim() == 1, "");
    FT_CHECK_WITH_INFO(cu_seqlens.dim() == 1, "");

    // check size
    size_t token_num = qkv_input.shape()[0];
    FT_CHECK_WITH_INFO(token_num == position_ids.shape()[0], "");
    FT_CHECK_WITH_INFO(token_num == padding_offset.shape()[0], "");

    size_t head_num = q_output.shape()[1];
    size_t head_kv_num = k_output.shape()[1];
    size_t hidden_num = head_num + 2 * head_kv_num;
    size_t head_size = qkv_input.shape()[2];
    size_t hidden_size = hidden_num * head_size;
    FT_CHECK_WITH_INFO(hidden_num == qkv_input.shape()[1], "");
    FT_CHECK_WITH_INFO(head_kv_num == v_output.shape()[1], "");
    FT_CHECK_WITH_INFO(hidden_size == bias.shape()[0],
        "hidden_size is %d , bias shape [0] is {%d}", hidden_size, bias.shape()[0]);

    size_t batch_size = q_output.shape()[0];
    FT_CHECK_WITH_INFO(batch_size == k_output.shape()[0], "");
    FT_CHECK_WITH_INFO(batch_size == v_output.shape()[0], "");

}

}  // namespace fastertransformer

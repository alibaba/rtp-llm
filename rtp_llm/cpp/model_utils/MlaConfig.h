#pragma once

#include <cstdint>
#include <string>
#include <algorithm>
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

enum MlaOpsType : int8_t {
    AUTO        = 0,
    MHA         = 1,
    FLASH_INFER = 2,
    FLASH_MLA   = 3,
};

inline MlaOpsType getMlaOpsType(std::string mla_ops_type_str) {
    // Convert to uppercase for case-insensitive comparison
    std::string upper_str = mla_ops_type_str;
    std::transform(upper_str.begin(), upper_str.end(), upper_str.begin(), ::toupper);
    
    if (upper_str == "AUTO" || mla_ops_type_str == "auto") {
        return MlaOpsType::AUTO;
    } else if (upper_str == "MHA" || mla_ops_type_str == "mha") {
        return MlaOpsType::MHA;
    } else if (upper_str == "FLASH_INFER" || mla_ops_type_str == "flash_infer" || mla_ops_type_str == "flash-infer") {
        return MlaOpsType::FLASH_INFER;
    } else if (upper_str == "FLASH_MLA" || mla_ops_type_str == "flash_mla" || mla_ops_type_str == "flash-mla") {
        return MlaOpsType::FLASH_MLA;
    } else {
        RTP_LLM_FAIL("MlaOpsType: " + mla_ops_type_str + " not supported !");
    }
    return MlaOpsType::AUTO;
}

}  // namespace rtp_llm
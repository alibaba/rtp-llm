#pragma once

#include <cstdint>

namespace rtp_llm {

enum MlaOpsType : int8_t {
    AUTO        = 0,
    MHA         = 1,
    FLASH_INFER = 2,
    FLASH_MLA   = 3,
};

}  // namespace rtp_llm
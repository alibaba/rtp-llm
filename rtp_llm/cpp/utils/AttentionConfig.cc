#include "rtp_llm/cpp/utils/AttentionConfig.h"

namespace rtp_llm {

void registerFMHAType(py::module m) {
    py::enum_<FMHAType>(m, "FMHAType")
        .value("NONE", FMHAType::NONE)
        .value("PAGED_TRT_V2", FMHAType::PAGED_TRT_V2)
        .value("TRT_V2", FMHAType::TRT_V2)
        .value("PAGED_OPEN_SOURCE", FMHAType::PAGED_OPEN_SOURCE)
        .value("OPEN_SOURCE", FMHAType::OPEN_SOURCE)
        .value("TRT_V1", FMHAType::TRT_V1)
        .value("FLASH_INFER", FMHAType::FLASH_INFER)
        .value("XQA", FMHAType::XQA);
}

}  // namespace rtp_llm

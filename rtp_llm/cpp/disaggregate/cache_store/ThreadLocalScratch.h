#pragma once

#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

namespace rtp_llm {

StagedMemoryCopyScratch& threadLocalScratch(int device_index);

}  // namespace rtp_llm

#pragma once

#include <stdint.h>

#include "rtp_llm/cpp/runtime/DeviceTypes.h"

namespace rtp_llm {

template<typename T>
void invokeMaskLogits(
    T* logits_batch, const uint8_t* mask_batch, const int batch_size, const int vocab_size, DeviceStream stream);

#if USING_CUDA
template<typename T>
void invokePackedMaskLogits(T*             logits_batch,
                            const int32_t*  packed_allow_mask,
                            const int32_t*  row_indices,
                            int             mask_rows,
                            int             logits_rows,
                            int             logits_row_stride,
                            int             vocab_size,
                            int             bitmask_row_stride,
                            int             bitmask_words,
                            DeviceStream    stream);
#endif

}  // namespace rtp_llm

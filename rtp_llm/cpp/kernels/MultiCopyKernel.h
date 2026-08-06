#pragma once

#include <assert.h>
#include <vector>

#include "rtp_llm/cpp/runtime/DeviceTypes.h"

namespace rtp_llm {

void InvokeMultiMergeCopyKernel(const void*                h_dst_ptr,
                                const std::vector<void*>&  h_src_ptrs,
                                const std::vector<size_t>& h_copy_sizes,
                                const std::vector<size_t>& h_begin_offsets,
                                DeviceStream               stream);

}  // namespace rtp_llm

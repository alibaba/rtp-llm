

#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/group_gemm/group_gemm_template.h"

namespace rtp_llm {
template class CutlassGroupGemmRunner<half>;

}  // namespace rtp_llm

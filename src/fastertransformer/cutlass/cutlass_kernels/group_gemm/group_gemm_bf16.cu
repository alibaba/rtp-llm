

#include "src/fastertransformer/cutlass/cutlass_kernels/group_gemm/group_gemm_template.h"

namespace fastertransformer {
template class CutlassGroupGemmRunner<__nv_bfloat16>;

}  // namespace fastertransformer

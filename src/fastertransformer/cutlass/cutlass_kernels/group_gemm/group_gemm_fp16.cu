

#include "src/fastertransformer/cutlass/cutlass_kernels/group_gemm/group_gemm_template.h"

namespace fastertransformer {
template class CutlassGroupGemmRunner<half>;

}  // namespace fastertransformer

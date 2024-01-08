// weighted only gemm
#include "src/fastertransformer/cutlass/cutlass_kernels/cutlass_preprocessors.h"
#include "src/fastertransformer/cutlass/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "src/fastertransformer/cutlass/cutlass_kernels/group_gemm/group_gemm.h"
#if defined(USE_WEIGHT_ONLY) && USE_WEIGHT_ONLY==1
#include "src/fastertransformer/cutlass/cutlass_kernels/weightOnlyBatchedGemv/kernelLauncher.h"
#endif


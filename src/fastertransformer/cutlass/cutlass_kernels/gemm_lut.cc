#include "src/fastertransformer/cutlass/cutlass_kernels/gemm_lut.h"

namespace tensorrt_llm {
namespace kernels {
namespace cutlass_kernels {
    
GemmLut GemmConfigMap::fp16_int4_lut = {};
GemmLut GemmConfigMap::fp16_int8_lut = {};

void GemmConfigMap::registerEntryForFp16Int8Lut(int m, int n, int k, CutlassGemmConfig config) {
   fp16_int8_lut[{m,n,k}] = config;
}

void GemmConfigMap::registerEntryForFp16Int4Lut(int m, int n, int k, CutlassGemmConfig config) {
   fp16_int4_lut[{m,n,k}] = config;
}

}
}
}
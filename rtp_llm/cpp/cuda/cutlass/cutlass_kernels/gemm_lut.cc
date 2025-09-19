#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/gemm_lut.h"

namespace tensorrt_llm {
namespace kernels {
namespace cutlass_kernels {

GemmLut GemmConfigMap::fp16_int4_lut = {};
GemmLut GemmConfigMap::fp16_int8_lut = {};
GemmLut GemmConfigMap::w8a8_lut = {};
GemmLut GemmConfigMap::moe_fp16_int4_lut = {};
GemmLut GemmConfigMap::moe_fp16_int8_lut = {};

void GemmConfigMap::registerEntryForFp16Int8Lut(int m, int n, int k, CutlassGemmConfig config) {
   fp16_int8_lut[{m,n,k}] = config;
}

void GemmConfigMap::registerEntryForFp16Int4Lut(int m, int n, int k, CutlassGemmConfig config) {
   fp16_int4_lut[{m,n,k}] = config;
}

void GemmConfigMap::registerEntryForW8A8Lut(int m, int n, int k, CutlassGemmConfig config) {
   w8a8_lut[{m,n,k}] = config;
}

void GemmConfigMap::registerEntryForMoeFp16Int8Lut(int m, int n, int k, CutlassGemmConfig config) {
   moe_fp16_int8_lut[{m,n,k}] = config;
}

void GemmConfigMap::registerEntryForMoeFp16Int4Lut(int m, int n, int k, CutlassGemmConfig config) {
   moe_fp16_int4_lut[{m,n,k}] = config;
}

}
}
}
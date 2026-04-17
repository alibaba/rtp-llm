#include "rtp_llm/models_py/bindings/cuda/cutlass/cutlass_kernels/gemm_lut.h"

namespace tensorrt_llm {
namespace kernels {
namespace cutlass_kernels {

GemmLut GemmConfigMap::moe_fp16_int4_lut = {};
GemmLut GemmConfigMap::moe_fp16_int8_lut = {};

}  // namespace cutlass_kernels
}  // namespace kernels
}  // namespace tensorrt_llm

#include "int4_dequant_comm.h"

namespace rtp_llm {

void int4_dequant_gemm_256x128x128x128_32_32x32_2x2_16x16x1_4x64x1_32_1x32x1x8_8_intrawave_v3(
    const ckGemmParam& params) {
    using DeviceInt4GemmInstance = DeviceInt4GemmHelper<256,
                                                        128,
                                                        128,
                                                        128,
                                                        32,
                                                        32,
                                                        32,
                                                        2,
                                                        2,
                                                        S<16, 16, 1>,
                                                        S<4, 64, 1>,
                                                        32,
                                                        S<1, 32, 1, 8>,
                                                        8,
                                                        ck::BlockGemmPipelineScheduler::Intrawave,
                                                        ck::BlockGemmPipelineVersion::v3>;

    int4Gemm_impl<DeviceInt4GemmInstance>(params);
}

}  // namespace rtp_llm
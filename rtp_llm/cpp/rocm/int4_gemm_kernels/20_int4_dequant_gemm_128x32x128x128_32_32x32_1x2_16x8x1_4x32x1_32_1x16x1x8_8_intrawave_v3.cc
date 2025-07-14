#include "int4_dequant_comm.h"

namespace rtp_llm {

void int4_dequant_gemm_128x32x128x128_32_32x32_1x2_16x8x1_4x32x1_32_1x16x1x8_8_intrawave_v3(const ckGemmParam& params) {
    using DeviceInt4GemmInstance = DeviceInt4GemmHelper<128,
                                                        32,
                                                        128,
                                                        128,
                                                        32,
                                                        32,
                                                        32,
                                                        1,
                                                        2,
                                                        S<16, 8, 1>,
                                                        S<4, 32, 1>,
                                                        32,
                                                        S<1, 16, 1, 8>,
                                                        8,
                                                        ck::BlockGemmPipelineScheduler::Intrawave,
                                                        ck::BlockGemmPipelineVersion::v3>;

    int4Gemm_impl<DeviceInt4GemmInstance>(params);
}

}  // namespace rtp_llm
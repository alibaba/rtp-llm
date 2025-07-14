#include "int4_dequant_comm.h"

namespace rtp_llm {

void int4_dequant_gemm_256x16x256x128_32_16x16_1x4_16x8x1_4x32x1_32_1x16x1x16_4_intrawave_v3(
    const ckGemmParam& params) {
    using DeviceInt4GemmInstance = DeviceInt4GemmHelper<256,
                                                        16,
                                                        256,
                                                        128,
                                                        32,
                                                        16,
                                                        16,
                                                        1,
                                                        4,
                                                        S<16, 8, 1>,
                                                        S<4, 32, 1>,
                                                        32,
                                                        S<1, 16, 1, 16>,
                                                        4,
                                                        ck::BlockGemmPipelineScheduler::Intrawave,
                                                        ck::BlockGemmPipelineVersion::v3>;

    int4Gemm_impl<DeviceInt4GemmInstance>(params);
}

}  // namespace rtp_llm
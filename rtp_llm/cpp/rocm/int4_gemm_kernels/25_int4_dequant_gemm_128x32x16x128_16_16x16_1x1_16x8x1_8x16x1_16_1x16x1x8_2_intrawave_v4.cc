#include "int4_dequant_comm.h"

namespace rtp_llm {

void int4_dequant_gemm_128x32x16x128_16_16x16_1x1_16x8x1_8x16x1_16_1x16x1x8_2_intrawave_v4(const ckGemmParam& params) {
    using DeviceInt4GemmInstance = DeviceInt4GemmHelper<128,
                                                        32,
                                                        16,
                                                        128,
                                                        16,
                                                        16,
                                                        16,
                                                        1,
                                                        1,
                                                        S<16, 8, 1>,
                                                        S<8, 16, 1>,
                                                        16,
                                                        S<1, 16, 1, 8>,
                                                        2,
                                                        ck::BlockGemmPipelineScheduler::Intrawave,
                                                        ck::BlockGemmPipelineVersion::v4>;

    int4Gemm_impl<DeviceInt4GemmInstance>(params);
}

}  // namespace rtp_llm
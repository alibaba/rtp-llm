#include "int4_dequant_comm.h"

namespace rtp_llm {

void int4_dequant_gemm_256x16x64x256_32_16x16_1x1_32x8x1_8x32x1_32_1x16x1x8_8_intrawave_v3(const ckGemmParam& params) {
    using DeviceInt4GemmInstance = DeviceInt4GemmHelper<256,
                                                        16,
                                                        64,
                                                        256,
                                                        32,
                                                        16,
                                                        16,
                                                        1,
                                                        1,
                                                        S<32, 8, 1>,
                                                        S<8, 32, 1>,
                                                        32,
                                                        S<1, 16, 1, 8>,
                                                        8,
                                                        ck::BlockGemmPipelineScheduler::Intrawave,
                                                        ck::BlockGemmPipelineVersion::v3>;

    int4Gemm_impl<DeviceInt4GemmInstance>(params);
}

}  // namespace rtp_llm
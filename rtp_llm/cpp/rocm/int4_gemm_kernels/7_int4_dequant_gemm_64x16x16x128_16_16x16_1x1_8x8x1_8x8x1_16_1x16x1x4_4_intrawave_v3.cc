#include "int4_dequant_comm.h"

namespace rtp_llm {

void int4_dequant_gemm_64x16x16x128_16_16x16_1x1_8x8x1_8x8x1_16_1x16x1x4_4_intrawave_v3(const ckGemmParam& params) {
    using DeviceInt4GemmInstance = DeviceInt4GemmHelper<64,
                                                        16,
                                                        16,
                                                        128,
                                                        16,
                                                        16,
                                                        16,
                                                        1,
                                                        1,
                                                        S<8, 8, 1>,
                                                        S<8, 8, 1>,
                                                        16,
                                                        S<1, 16, 1, 4>,
                                                        4,
                                                        ck::BlockGemmPipelineScheduler::Intrawave,
                                                        ck::BlockGemmPipelineVersion::v3>;

    int4Gemm_impl<DeviceInt4GemmInstance>(params);
}

}  // namespace rtp_llm
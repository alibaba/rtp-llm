#include "flash.h"
#include <cutlass/numeric_types.h>
#include "static_switch.h"
#include <stdexcept>

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel) {
    FP16_SWITCH(!params.is_bf16, [&] {
        FWD_HEADDIM_SWITCH(params.d, [&] {
            if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
                run_mha_fwd_<elem_type, kHeadDim>(params, stream);
            } else {
                throw std::runtime_error("not support split head");
                //run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim>(params, stream);
            }
        });
    });
}

bool flash_attention_enabled() {
    return true;
}

bool is_sm8x() {
    static bool IS_SM8X = [](){
        int device;
        FA_CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp deviceProp;
        FA_CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device));
        return deviceProp.major == 8 && deviceProp.minor > 0;
    }();
    return IS_SM8X;
}

#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/cutlass/interface.h"
#include "src/fastertransformer/utils/compiler_config.h"


using namespace std;

namespace fastertransformer {


#define ARGS_DISPATCH(Atype,        \
                      Dtype,        \
                      out,          \
                      bias,         \
                      gate,         \
                      gate_bias,    \
                      m,            \
                      n,            \
                      stream) do {  \
    invokeGenericActivation<Atype>( \
        (Dtype*) out,               \
        (const Dtype*) bias,        \
        (const Dtype*) gate,        \
        (const Dtype*) gate_bias,   \
        (const int*)  nullptr,      \
        (const Dtype*) nullptr,     \
        (int)m,                     \
        (int)n,                     \
        0,                          \
        (const float*) nullptr,     \
        (const float*) nullptr,     \
        stream);                    \
} while (0)



#define ATYPE_DISPATCH(Dtype, Atype, KERNEL, ...) do {          \
    if (Atype == ActivationType::Silu) {                        \
        KERNEL(SiluActivation, Dtype, __VA_ARGS__);             \
    } else if (Atype == ActivationType::Gelu) {                 \
        KERNEL(GeluActivation, Dtype, __VA_ARGS__);             \
    } else {                                                    \
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);    \
    }                                                           \
} while(0)


#define DTYPE_DISPATCH(Dtype, ...) do {                         \
    if (Dtype == DataType::TYPE_FP16) {                         \
        ATYPE_DISPATCH(half, __VA_ARGS__);                      \
    } else {                                                    \
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);    \
    }                                                           \
} while(0)

#define DISPATCH(Dtype, Atype, ...) do {                        \
    DTYPE_DISPATCH(                                             \
        Dtype,                                                  \
        Atype,                                                  \
        ARGS_DISPATCH,                                          \
        __VA_ARGS__                                             \
    );                                                          \
} while(0)



/// @brief   act op
void CudaDevice::activation(const ActivationParams& params) {
    size_t m = params.output.shape()[0];
    size_t n = params.output.shape()[0];

    void* bias = nullptr;
    void* gate = nullptr;
    void* gate_bias = nullptr;

    if (params.bias) {
        bias = params.bias.value().get().data();
    }

    if (params.gate) {
        gate = params.gate.value().get().data();
    }

    if (params.gate_bias) {
        gate_bias = params.gate_bias.value().get().data();
    }

    DISPATCH(
        params.output.type(), params.atype,
        params.output.data(), bias,
        gate, gate_bias, m, n, stream_
    );

    
}


} // namespace fastertransformer
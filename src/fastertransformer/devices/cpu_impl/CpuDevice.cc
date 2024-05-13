#include "src/fastertransformer/devices/cpu_impl/CpuDevice.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/core/allocator.h"
#include "src/fastertransformer/core/cpu_allocator.h"
#include "3rdparty/xft/include/layers_mlp.h"
#include <cstring>
#include <cmath>
#include <immintrin.h>

#define BLOCKSIZE_512b_FP32 16  // 512 bit include 16 * 32bit

namespace fastertransformer {

CpuDevice::CpuDevice(const DeviceInitParams& params) : DeviceBase(params) {
    allocator_.reset(new Allocator<AllocatorType::CPU>());
}

CpuDevice::~CpuDevice() {
}

DeviceProperties CpuDevice::getDeviceProperties() {
    DeviceProperties props;
    props.type = DeviceType::Cpu;
    return props;
}

void CpuDevice::copy(const CopyParams& params) {
    auto& src = params.src;
    auto& dst = params.dst;
    auto size = params.src.sizeBytes();
    memcpy(dst.data(), src.data(), size);
}


LayernormOutput CpuDevice::layernorm(const LayernormParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

BufferPtr CpuDevice::gemm(const GemmParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

GroupedGemmOutput CpuDevice::groupedGemm(const GroupedGemmParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

BufferPtr CpuDevice::embeddingLookup(const EmbeddingLookupParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void CpuDevice::activation(const ActivationParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

BufferPtr CpuDevice::softmax(const SoftmaxParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

AttentionModuleOutput CpuDevice::contextAttention(const AttentionModuleParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

AttentionModuleOutput CpuDevice::decoderSelfAttention(const AttentionModuleParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

AttentionLayerOutput CpuDevice::attentionLayer(const AttentionLayerParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

FfnLayerOutput CpuDevice::ffnLayer(const FfnLayerParams& params) {
    const auto& input = params.input;
    const auto& gate_weight = *(params.weights.gate_weight->kernel);
    const auto& up_weight = *(params.weights.up_weight->kernel);
    const auto& down_weight = *(params.weights.down_weight->kernel);

    const auto act_t = params.activation_type;
    xft::ActivationType at;
    if (act_t == ActivationType::Swiglu) { // gated-silu
        at = xft::ActivationType::SILU;
    } else if (act_t == ActivationType::Geglu) { // gated-gelu
        at = xft::ActivationType::GELU;
    } else {
        throw std::runtime_error("Not supported");
    }

    const int token_num = input.shape()[0];
    const int inter_size = gate_weight.shape()[1];
    const int hidden_size = input.shape()[1];

    float *rms_output = static_cast<float*>(aligned_alloc(64, token_num * hidden_size * sizeof(float)));
    memset(rms_output, 0, token_num * hidden_size * sizeof(float));

    xft::invokeMLPLLaMA(xft::DataType::fp16, at,
                        token_num, hidden_size, inter_size, rms_output, hidden_size,
                        input.data(), hidden_size, gate_weight.data(), up_weight.data(),
                        down_weight.data());

    FfnLayerOutput ffnout;
    ffnout.hidden_states = allocateBuffer({DataType::TYPE_FP32, {token_num, hidden_size},
                                          AllocationType::HOST}, {});

    /* If not add rmsnorm then need following extra process */
    float* input_ptr = static_cast<float*>(input.data());
    const int total_elements = token_num * hidden_size;
    const int num_iterations = total_elements / BLOCKSIZE_512b_FP32; // AVX-512 could process 16 * float

#pragma omp parallel for
    for (int i = 0; i < num_iterations; ++i) {
        const __m512 rms_vec = _mm512_loadu_ps(&rms_output[i * BLOCKSIZE_512b_FP32]);
        const __m512 input_vec = _mm512_loadu_ps(&input_ptr[i * BLOCKSIZE_512b_FP32]);
        const __m512 result = _mm512_sub_ps(rms_vec, input_vec);
        float* ffnout_data = static_cast<float*>(ffnout.hidden_states->data());
        _mm512_storeu_ps(&ffnout_data[i * BLOCKSIZE_512b_FP32], result);
    }

    for (int i = num_iterations * BLOCKSIZE_512b_FP32; i < total_elements; ++i) {
        ffnout.hidden_states->data<float>()[i] = rms_output[i] - input_ptr[i];
    }

    /* If not add rmsnorm then need above extra process */

    free(rms_output);
    return ffnout;
}

void CpuDevice::sampleGreedy(const GreedyParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void CpuDevice::sampleBeamSearch(const BeamSearchParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}


void CpuDevice::broadcast(const BroadcastParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void CpuDevice::allReduce(const AllReduceParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

RTP_LLM_REGISTER_DEVICE(Cpu);


} // namespace fastertransformer

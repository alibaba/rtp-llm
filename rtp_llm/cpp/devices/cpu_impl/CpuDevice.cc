#include "rtp_llm/cpp/devices/cpu_impl/CpuDevice.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/core/allocator.h"
#include "rtp_llm/cpp/core/cpu_allocator.h"
#include "xfastertransformer/include/layers_mlp.h"
#include "xfastertransformer/include/layers_attention.h"
#include "xfastertransformer/include/layers_norm.h"
#include "xfastertransformer/include/layers_token_embedding.h"
#include "xfastertransformer/include/layers_gemm.h"
#include <cstring>
#include <cmath>
#include <immintrin.h>

#define BLOCKSIZE_512b_FP32 16  // 512 bit include 16 * 32bit
#define BLOCKSIZE_512b_BF16 32  // 512 bit include 32 * 16bit

namespace rtp_llm {

CpuDevice::CpuDevice(const DeviceInitParams& params): DeviceBase(params) {
    allocator_.reset(new Allocator<AllocatorType::CPU>());
}

CpuDevice::~CpuDevice() {}

DeviceProperties CpuDevice::getDeviceProperties() {
    static DeviceProperties* prop = nullptr;
    if (prop == nullptr) {
        prop                         = new DeviceProperties();
        prop->type                   = DeviceType::Cpu;
        prop->attn_fuse_add_residual = true;
        prop->ffn_fuse_add_residual  = true;
    }
    return *prop;
}

void CpuDevice::copy(const CopyParams& params) {
    auto& src  = params.src;
    auto& dst  = params.dst;
    auto  size = params.src.sizeBytes();
    memcpy(dst.data(), src.data(), size);
}

LayernormOutput CpuDevice::layernorm(const LayernormParams& params) {
    const auto& input       = params.input;
    const auto& norm_weight = params.norm_weight;
    const auto& gamma       = norm_weight ? norm_weight->get().gamma.get()->data() : nullptr;
    const auto& beta      = (norm_weight && norm_weight->get().beta) ? norm_weight->get().beta.get()->data() : nullptr;
    const auto  norm_type = params.norm_type;
    const auto  data_type = input->type();
    int         rows      = input->shape()[0];
    int         cols      = input->shape()[1];
    const auto  eps       = params.eps;

    LayernormOutput layernorm_out;
    layernorm_out.output = allocateBuffer({data_type, {size_t(rows), size_t(cols)}, AllocationType::HOST}, {});

    xft::DataType xft_dt;
    if (data_type == TYPE_FP32) {
        xft_dt = xft::DataType::fp32;
    } else if (data_type == TYPE_BF16) {
        xft_dt = xft::DataType::bf16;
    } else {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }

    if (norm_type == NormType::layernorm) {
        invokeLayerNorm(xft_dt,
                        layernorm_out.output->data(),
                        (const void*)(input->data()),
                        (const void*)(gamma),
                        (const void*)(beta),
                        rows,
                        cols,
                        -1,
                        -1,
                        eps);
    } else if (norm_type == NormType::rmsnorm) {
        invokeRmsNorm(xft_dt,
                      layernorm_out.output->data(),
                      (const void*)(input->data()),
                      (const void*)(gamma),
                      rows,
                      cols,
                      -1,
                      -1,
                      eps);
    }
    return layernorm_out;
}

BufferPtr CpuDevice::gemm(const GemmParams& params) {
    auto A = params.A.data();
    auto B = params.B.data();

    const auto data_type = params.A.type();
    int        dim       = params.A.dim();

    bool transA = (params.transA == TransposeOperation::TRANSPOSE);
    bool transB = (params.transB == TransposeOperation::TRANSPOSE);

    int m = transA ? params.A.shape()[dim - 1] : params.A.shape()[dim - 2];
    int k = transA ? params.A.shape()[dim - 2] : params.A.shape()[dim - 1];
    int n = transB ? params.B.shape()[dim - 2] : params.B.shape()[dim - 1];

    BufferPtr output;
    if (params.D) {
        output = params.D;
    } else {
        output = allocateBuffer({data_type, {size_t(m), size_t(n)}, AllocationType::HOST});
    }

    xft::DataType xft_dt;
    if (data_type == TYPE_FP16) {
        xft_dt = xft::DataType::fp16;
    } else if (data_type == TYPE_BF16) {
        xft_dt = xft::DataType::bf16;
    } else {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }

    xft::invokeGemm(xft_dt, transA, transB, m, n, k, 1.0, A, k, B, 0.0, output->data(), n);

    return output;
}

GroupedGemmOutput CpuDevice::groupedGemm(const GroupedGemmParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

BufferPtr CpuDevice::embeddingLookup(const EmbeddingLookupParams& params) {
    const auto& tokens          = params.combo_tokens;
    const auto& embedding_table = params.embedding_table;

    const auto token_num   = tokens.size();
    const auto hidden_size = embedding_table.shape()[1];
    const auto data_type   = embedding_table.type();

    xft::DataType xft_dt;
    if (data_type == TYPE_FP16) {
        xft_dt = xft::DataType::fp16;
    } else if (data_type == TYPE_BF16) {
        xft_dt = xft::DataType::bf16;
    } else {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }

    BufferPtr emb_out = allocateBuffer({DataType::TYPE_FP32, {token_num, hidden_size}, AllocationType::HOST}, {});

    invokeTokenEmbedding(xft_dt, emb_out->data(), tokens.data(), embedding_table.data(), token_num, hidden_size);
    return emb_out;
}

BufferPtr CpuDevice::activation(const ActivationParams& params) {
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
    const auto& input            = params.input;
    const auto& input_lengths    = params.common.input_lengths;
    const auto& sequence_lengths = params.common.sequence_lengths;
    const auto& qkv_weight       = params.weights.qkv_weight->kernel;
    const auto& output_weight    = params.weights.output_weight->kernel;
    const auto& attention_conf   = params.configs;

    int batch_size      = input_lengths->shape()[0];
    int past_seq_len    = 0;
    int current_seq_len = 0;
    int next_token_num  = 0;
    int step            = sequence_lengths->size();
    int input_seq_len   = input_lengths->data<int>()[0];

    if (step == 0) {
        current_seq_len = input_seq_len;
    } else {
        next_token_num = batch_size;
        past_seq_len += (step == 1) ? input_seq_len : next_token_num;
        current_seq_len = next_token_num;
    }

    int hidden_size   = attention_conf.head_num * attention_conf.size_per_head;
    int head_num      = attention_conf.head_num;
    int kv_head_num   = attention_conf.kv_head_num;
    int head_dim      = attention_conf.size_per_head;
    int max_pos_embed = attention_conf.rope_config.max_pos;
    int max_positions = max_pos_embed;
    int q_size        = head_dim * head_num;
    int kv_size       = head_dim * kv_head_num;

    auto qkv_data_ptr = static_cast<float*>(qkv_weight->data());

    float* rms_atten_output =
        static_cast<float*>(aligned_alloc(64, batch_size * input_seq_len * hidden_size * sizeof(float)));
    memset(rms_atten_output, 0, batch_size * input_seq_len * hidden_size * sizeof(float));

    invokeAttentionLLaMA(xft::DataType::fp16,
                         batch_size,
                         input_seq_len,
                         head_dim,
                         head_num,
                         kv_head_num,
                         max_positions,
                         max_pos_embed,
                         past_seq_len,
                         current_seq_len,
                         step,
                         hidden_size,
                         rms_atten_output,
                         hidden_size,
                         (const void*)input.data(),
                         hidden_size,
                         qkv_data_ptr,
                         qkv_data_ptr + q_size,
                         qkv_data_ptr + q_size + kv_size,
                         output_weight->data());

    AttentionLayerOutput atten_out;
    atten_out.hidden_states = allocateBuffer(
        {DataType::TYPE_FP32, {size_t(batch_size * input_seq_len), size_t(hidden_size)}, AllocationType::HOST}, {});

    /* If not add rmsnorm then need following extra process */
    float* input_ptr      = static_cast<float*>(input.data());
    int    total_elements = batch_size * input_seq_len * hidden_size;
    int    num_iterations = total_elements / BLOCKSIZE_512b_FP32;  // AVX-512 could process 16 * float

#pragma omp parallel for
    for (int i = 0; i < num_iterations; ++i) {
        const __m512 rms_vec        = _mm512_loadu_ps(&rms_atten_output[i * BLOCKSIZE_512b_FP32]);
        const __m512 input_vec      = _mm512_loadu_ps(&input_ptr[i * BLOCKSIZE_512b_FP32]);
        const __m512 result         = _mm512_sub_ps(rms_vec, input_vec);
        float*       atten_out_data = static_cast<float*>(atten_out.hidden_states->data());
        _mm512_storeu_ps(&atten_out_data[i * BLOCKSIZE_512b_FP32], result);
    }

    for (int i = num_iterations * BLOCKSIZE_512b_FP32; i < total_elements; ++i) {
        atten_out.hidden_states->data<float>()[i] = rms_atten_output[i] - input_ptr[i];
    }

    /* If not add rmsnorm then need above extra process */

    free(rms_atten_output);
    return atten_out;
}

FfnLayerOutput CpuDevice::ffnLayer(const FfnLayerParams& params) {
    const auto& input       = params.input;
    const auto& gate_weight = *(params.weights.gate_weight->kernel);
    const auto& up_weight   = *(params.weights.up_weight->kernel);
    const auto& down_weight = *(params.weights.down_weight->kernel);

    const auto          act_t = params.configs.activation_type;
    xft::ActivationType at;
    if (act_t == ActivationType::Swiglu) {  // gated-silu
        at = xft::ActivationType::SILU;
    } else if (act_t == ActivationType::Geglu) {  // gated-gelu
        at = xft::ActivationType::GELU;
    } else {
        throw std::runtime_error("Not supported");
    }

    const int token_num   = input.shape()[0];
    const int inter_size  = gate_weight.shape()[1];
    const int hidden_size = input.shape()[1];

    float* rms_output = static_cast<float*>(aligned_alloc(64, token_num * hidden_size * sizeof(float)));
    memset(rms_output, 0, token_num * hidden_size * sizeof(float));

    xft::invokeMLPLLaMA(xft::DataType::fp16,
                        at,
                        token_num,
                        hidden_size,
                        inter_size,
                        rms_output,
                        hidden_size,
                        input.data(),
                        hidden_size,
                        gate_weight.data(),
                        up_weight.data(),
                        down_weight.data());

    FfnLayerOutput ffnout;
    ffnout.hidden_states =
        allocateBuffer({DataType::TYPE_FP32, {size_t(token_num), size_t(hidden_size)}, AllocationType::HOST}, {});

    /* If not add rmsnorm then need following extra process */
    float*    input_ptr      = static_cast<float*>(input.data());
    const int total_elements = token_num * hidden_size;
    const int num_iterations = total_elements / BLOCKSIZE_512b_FP32;  // AVX-512 could process 16 * float

#pragma omp parallel for
    for (int i = 0; i < num_iterations; ++i) {
        const __m512 rms_vec     = _mm512_loadu_ps(&rms_output[i * BLOCKSIZE_512b_FP32]);
        const __m512 input_vec   = _mm512_loadu_ps(&input_ptr[i * BLOCKSIZE_512b_FP32]);
        const __m512 result      = _mm512_sub_ps(rms_vec, input_vec);
        float*       ffnout_data = static_cast<float*>(ffnout.hidden_states->data());
        _mm512_storeu_ps(&ffnout_data[i * BLOCKSIZE_512b_FP32], result);
    }

    for (int i = num_iterations * BLOCKSIZE_512b_FP32; i < total_elements; ++i) {
        ffnout.hidden_states->data<float>()[i] = rms_output[i] - input_ptr[i];
    }

    /* If not add rmsnorm then need above extra process */

    free(rms_output);
    return ffnout;
}

BeamSearchOutput CpuDevice::sampleBeamSearch(const BeamSearchParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void CpuDevice::broadcast(const BroadcastParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

AllReduceOutput CpuDevice::allReduce(const AllReduceParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

RTP_LLM_REGISTER_DEVICE(Cpu);

}  // namespace rtp_llm

#include "rtp_llm/cpp/devices/arm_impl/ArmDevice.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/core/allocator.h"
#include "rtp_llm/cpp/core/cpu_allocator.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include <openblas/cblas.h>

#include <cfloat>
#include "kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p_bf16p_interface.h"
#include "kai/ukernels/matmul/pack/kai_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_bf16p8x4_f32_neon.h"

namespace rtp_llm {

/* Input has shape [dim0, dim1, dim2, dim3] */
void transposeDim12(BufferPtr input, void* output) {
    auto dim     = input->shape();
    auto elem_sz = input->typeSize();

    for (int k = 0; k < dim[0]; k++) {
        for (int i = 0; i < dim[2]; i++) {
            for (int j = 0; j < dim[1]; j++) {
                memcpy((char*)output + elem_sz * (k * dim[1] * dim[2] * dim[3] + (i * dim[1] + j) * dim[3]),
                       input->dataWithOffset(k * dim[1] * dim[2] * dim[3] + (j * dim[2] + i) * dim[3]),
                       elem_sz * dim[3]);
            }
        }
    }
}

void getCacheAddrFromIndex(const KvCacheInfo& kv_cache, size_t batch, size_t block_idx, void** k_addr, void** v_addr) {
    const auto& kv_blocks_offset     = *(kv_cache.kv_cache_block_id);
    const auto& k_cache              = *(kv_cache.k_cache_buffer);
    const auto& v_cache              = *(kv_cache.k_cache_buffer);
    const auto  max_blocks_per_batch = kv_blocks_offset.shape()[1];
    size_t      block_size           = k_cache[0].sizeBytes();
    int*        index                = (int*)kv_blocks_offset.data();

    *k_addr = (char*)k_cache.data() + index[batch * max_blocks_per_batch + block_idx] * block_size;
    *v_addr = (char*)v_cache.data() + index[batch * max_blocks_per_batch + block_idx] * block_size + block_size / 2;
}

void assemCache(const AttentionModuleParams& params, int batch, BufferPtr k_out, BufferPtr v_out) {
    auto   elem_sz          = k_out->typeSize();
    auto   kv_seq_len       = k_out->shape()[1];
    auto   head_num         = k_out->shape()[2];
    auto   head_dim         = k_out->shape()[3];
    auto   tokens_per_block = params.configs.tokens_per_block;
    size_t blocks_per_batch = (kv_seq_len + tokens_per_block - 1) / tokens_per_block;
    size_t copied_len       = 0;

    void* k_block_addr;
    void* v_block_addr;
    for (int i = 0; i < blocks_per_batch; i++) {
        size_t len = std::min(tokens_per_block, kv_seq_len - copied_len);
        getCacheAddrFromIndex(params.common.kv_cache.value(), batch, i, &k_block_addr, &v_block_addr);

        memcpy(k_out->dataWithOffset(i * tokens_per_block * head_num * head_dim),
               k_block_addr,
               elem_sz * len * head_num * head_dim);
        memcpy(v_out->dataWithOffset(i * tokens_per_block * head_num * head_dim),
               v_block_addr,
               elem_sz * len * head_num * head_dim);
        copied_len += len;
    }
}

/* Input 'qkv' consists of q & k & v, and each with shape [batch, seq_len, num_heads, head_dim]
 * 'bias' has shape [num_heads * head_dim]
 */
template<typename Ti, typename Tb>
void addQKVBias(
    void* qkv, const void* bias, int batch_sz, int seq_len, int num_heads, int kv_num_heads, int head_size) {
    const int N = batch_sz * seq_len;
    parallel_for(N, [&](int tid) {
        Ti* qkv_input = (Ti*)qkv + tid * (num_heads + 2 * kv_num_heads) * head_size;

        for (int i = 0; i < (num_heads + 2 * kv_num_heads) * head_size; i++) {
            qkv_input[i] += ((Tb*)bias)[i];
        }
    });
}

void updateKVCache(const AttentionModuleParams& params, int batch, size_t step, BufferPtr k, BufferPtr v) {
    size_t seq_len       = k->shape()[1];
    auto   kv_head_num   = params.configs.kv_head_num;
    auto   size_per_head = params.configs.size_per_head;
    auto   block_tokens  = params.configs.tokens_per_block;

    size_t block_num    = (seq_len + block_tokens - 1) / block_tokens;
    size_t block_offset = step / block_tokens;
    auto   elem_sz      = params.input.typeSize();
    size_t copied_len   = 0;

    void* k_block_addr;
    void* v_block_addr;
    for (int i = 0; i < block_num; i++) {
        size_t len = std::min(block_tokens, seq_len - copied_len);
        getCacheAddrFromIndex(params.common.kv_cache.value(), batch, i + block_offset, &k_block_addr, &v_block_addr);

        memcpy((uint8_t*)k_block_addr + (step % block_tokens) * kv_head_num * size_per_head * elem_sz,
               k->dataWithOffset(i * block_tokens * kv_head_num * size_per_head),
               elem_sz * len * kv_head_num * size_per_head);
        memcpy((uint8_t*)k_block_addr + (step % block_tokens) * kv_head_num * size_per_head * elem_sz,
               v->dataWithOffset(i * block_tokens * kv_head_num * size_per_head),
               elem_sz * len * kv_head_num * size_per_head);

        copied_len += len;
    }
}

/* Input 'qkv' consists of q & k & v, and each with shape [batch, seq_len, num_heads, head_dim].
 * Half RoPE is applied to q & k.
 * Retrieve pre-calculated Cos/Sin if exists.
 */
template<typename T>
void ArmCpuDevice::halfRopeQK(void*  qkv,
                              int    batch,
                              int    seq_len,
                              int    num_heads,
                              int    kv_num_heads,
                              int    head_size,
                              size_t step,
                              size_t base,
                              size_t embed_dim) {
    size_t inv_freq_size = (embed_dim + 1) / 2;

    auto&  value      = ropeCosSin[base];
    int    calced_seq = std::get<0>(value);
    float* cur_cos    = std::get<1>(value);
    float* cur_sin    = std::get<2>(value);

    const int N = batch * seq_len;
    parallel_for(N, [&](int tid) {
        int j       = tid % seq_len;
        T*  q_input = (T*)qkv + tid * (num_heads + 2 * kv_num_heads) * head_size;
        T*  k_input = (T*)qkv + tid * (num_heads + 2 * kv_num_heads) * head_size + num_heads * head_size;

        size_t seq = (j == 0) ? step : j;
        for (int h = 0; h < num_heads; h++) {
            for (int d = 0; d < inv_freq_size; d++) {
                float fcr, fci;
                if (seq < calced_seq) {
                    fcr = cur_cos[seq * inv_freq_size + d];
                    fci = cur_sin[seq * inv_freq_size + d];
                } else {
                    float freq = 1.0f / powf(base, (float)(d * 2) / embed_dim);
                    float val  = freq * seq;
                    fcr        = cosf(val);
                    fci        = sinf(val);
                }

                auto v0 = q_input + h * head_size + d;
                auto v1 = q_input + h * head_size + d + inv_freq_size;
                auto d0 = *v0;
                auto d1 = *v1;
                *v0     = d0 * fcr - d1 * fci;
                *v1     = d0 * fci + d1 * fcr;

                if (h < kv_num_heads) {
                    auto v2 = k_input + h * head_size + d;
                    auto v3 = k_input + h * head_size + d + inv_freq_size;
                    auto d2 = *v2;
                    auto d3 = *v3;
                    *v2     = d2 * fcr - d3 * fci;
                    *v3     = d2 * fci + d3 * fcr;
                }
            }
        }
    });
}

void ArmCpuDevice::printStat() {
    for (int i = 0; i < sizeof(a_cnt_) / sizeof(uint64_t); i++) {
        std::cout << "$$$   [" << i << "] time (us) - min: " << a_tmin_[i] << " ; max: " << a_tmax_[i]
                  << " ; ave: " << a_tave_[i] / a_cnt_[i] << std::endl;
    }

    for (int i = 0; i < sizeof(a_cnt_) / sizeof(uint64_t); i++) {
        a_tmin_[i] = 999999999;
        a_tmax_[i] = 0;
        a_tave_[i] = 0;
        a_cnt_[i]  = 0;
    }
}

void ArmCpuDevice::logTime(std::chrono::microseconds diff, size_t index) {
    if (diff.count() < a_tmin_[index])
        a_tmin_[index] = diff.count();
    if (diff.count() > a_tmax_[index])
        a_tmax_[index] = diff.count();
    a_tave_[index] += diff.count();
    a_cnt_[index] += 1;
}

void ArmCpuDevice::runOneBatch(
    const AttentionModuleParams& params, size_t past_seq, int batch, size_t seq_len, size_t step) {
    auto datatype      = params.input.type();
    auto head_num      = params.configs.head_num;
    auto kv_head_num   = params.configs.kv_head_num;
    auto size_per_head = params.configs.size_per_head;

    // if (!params.common.kv_cache.has_value()) {
    //     throw std::runtime_error("kv cache block pointers can not be null");
    // }

    std::chrono::steady_clock::time_point tStart, tEnd;
    std::chrono::microseconds             diff;

    // qkv to q_output, k_output, v_output
    auto qkv = params.input.dataWithOffset(past_seq * (head_num + 2 * kv_head_num) * size_per_head);
    printBufferData(params.input, "qkv");

    tStart = std::chrono::steady_clock::now();
    // if (params.weights.qkv_weight->bias) {
    if (params.configs.fuse_qkv_add_bias && params.weights.qkv_weight->bias) {
        if (datatype == DataType::TYPE_FP32) {
            auto bias_data_type = params.weights.qkv_weight->bias->type();
            if (bias_data_type == DataType::TYPE_FP32) {
                addQKVBias<float, float>(
                    qkv, params.weights.qkv_weight->bias->data(), 1, seq_len, head_num, kv_head_num, size_per_head);
            } else if (bias_data_type == DataType::TYPE_FP16) {
                addQKVBias<float, __fp16>(
                    qkv, params.weights.qkv_weight->bias->data(), 1, seq_len, head_num, kv_head_num, size_per_head);
            } else {
                throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
            }
        } else if (datatype == DataType::TYPE_FP16) {
            addQKVBias<__fp16, __fp16>(
                qkv, params.weights.qkv_weight->bias->data(), 1, seq_len, head_num, kv_head_num, size_per_head);
        } else {
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
        }
    }
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 1);
    printBufferData(params.input, "biased qkv");

    tStart = std::chrono::steady_clock::now();
    if (params.configs.rope_config.style != RopeStyle::No) {
        if (params.configs.rope_config.style == RopeStyle::Base) {
            if (datatype == DataType::TYPE_FP32) {
                halfRopeQK<float>(qkv,
                                  1,
                                  seq_len,
                                  head_num,
                                  kv_head_num,
                                  size_per_head,
                                  step,
                                  params.configs.rope_config.base,
                                  params.configs.rope_config.dim);
            } else if (datatype == DataType::TYPE_FP16) {
                halfRopeQK<__fp16>(qkv,
                                   1,
                                   seq_len,
                                   head_num,
                                   kv_head_num,
                                   size_per_head,
                                   step,
                                   params.configs.rope_config.base,
                                   params.configs.rope_config.dim);
            } else {
                throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
            }
        } else {
            throw std::runtime_error("SelfAttention RoPE type is not supported");
        }
    }
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 2);
    printBufferData(params.input, "roped qkv");

    tStart                                = std::chrono::steady_clock::now();
    arm_compute::DataType   acl_data_type = getAclDataType(datatype);
    arm_compute::NESplit    split;
    arm_compute::Tensor     src;
    arm_compute::TensorInfo src_info = arm_compute::TensorInfo(
        arm_compute::TensorShape(size_per_head, head_num + 2 * kv_head_num, seq_len, 1), 1, acl_data_type);

    src.allocator()->init(src_info);
    src.allocator()->import_memory(qkv);
    std::vector<arm_compute::Tensor>   dsts{};
    std::vector<arm_compute::ITensor*> dsts_ptr;

    arm_compute::TensorInfo q_info, kv_info;
    arm_compute::Tensor     q, k, v;
    q_info = arm_compute::TensorInfo(arm_compute::TensorShape(size_per_head, head_num, seq_len, 1), 1, acl_data_type);
    kv_info =
        arm_compute::TensorInfo(arm_compute::TensorShape(size_per_head, kv_head_num, seq_len, 1), 1, acl_data_type);

    auto q_input = allocateBuffer({datatype, {1, seq_len, head_num, size_per_head}, AllocationType::HOST}, {});
    auto k_input = allocateBuffer({datatype, {1, seq_len, kv_head_num, size_per_head}, AllocationType::HOST}, {});
    auto v_input = allocateBuffer({datatype, {1, seq_len, kv_head_num, size_per_head}, AllocationType::HOST}, {});

    q.allocator()->init(q_info);
    k.allocator()->init(kv_info);
    v.allocator()->init(kv_info);
    q.allocator()->import_memory(q_input->data());
    dsts.push_back(std::move(q));

    k.allocator()->import_memory(k_input->data());
    dsts.push_back(std::move(k));

    v.allocator()->import_memory(v_input->data());
    dsts.push_back(std::move(v));

    for (auto& dst : dsts) {
        dsts_ptr.emplace_back(&dst);
    }
    split.configure(&src, dsts_ptr, 1);
    split.run();
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 0);

    printBufferData(*q_input, "q_input");
    printBufferData(*k_input, "k_input");
    printBufferData(*v_input, "v_input");

    tStart = std::chrono::steady_clock::now();
    if (params.common.kv_cache.has_value()) {
        updateKVCache(params, batch, step, k_input, v_input);
    }
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 8);

    BufferPtr k_in, k_out, v_in, v_out;
    if (step == 0) {
        /* Context. */
        k_in = k_input;
        v_in = v_input;

        k_out = allocateBuffer({datatype, {1, kv_head_num, seq_len, size_per_head}, AllocationType::HOST}, {});
        v_out = allocateBuffer({datatype, {1, kv_head_num, seq_len, size_per_head}, AllocationType::HOST}, {});
    } else {
        /* Decoder. Retrieve k/v cache data. */
        k_in = allocateBuffer({datatype, {1, step + 1, kv_head_num, size_per_head}, AllocationType::HOST}, {});
        v_in = allocateBuffer({datatype, {1, step + 1, kv_head_num, size_per_head}, AllocationType::HOST}, {});
        assemCache(params, batch, k_in, v_in);

        printBufferData(*k_in, "k_in");
        printBufferData(*v_in, "v_in");

        k_out = allocateBuffer({datatype, {1, kv_head_num, step + 1, size_per_head}, AllocationType::HOST}, {});
        v_out = allocateBuffer({datatype, {1, kv_head_num, step + 1, size_per_head}, AllocationType::HOST}, {});
    }

    tStart        = std::chrono::steady_clock::now();
    auto q_output = allocateBuffer({datatype, {1, head_num, seq_len, size_per_head}, AllocationType::HOST}, {});

    transposeDim12(std::move(q_input), q_output->data());
    transposeDim12(std::move(k_in), k_out->data());
    transposeDim12(std::move(v_in), v_out->data());
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 3);

    printBufferData(*q_output, "q_output");
    printBufferData(*k_out, "k_out");
    printBufferData(*v_out, "v_out");

    if (kv_head_num != head_num) {
        /* repeat K/V */
        size_t len;
        if (step == 0) {
            len = seq_len;
        } else {
            len = step + 1;
        }
        auto      k_repeat = allocateBuffer({datatype, {1, head_num, len, size_per_head}, AllocationType::HOST}, {});
        auto      v_repeat = allocateBuffer({datatype, {1, head_num, len, size_per_head}, AllocationType::HOST}, {});
        auto      n_rep    = head_num / kv_head_num;
        const int N        = kv_head_num;
        parallel_for(N, [&](int tid) {
            for (int i = 0; i < n_rep; i++) {
                memcpy(k_repeat->dataWithOffset((tid * n_rep + i) * len * size_per_head),
                       k_out->dataWithOffset(tid * len * size_per_head),
                       k_out->sizeBytes() / kv_head_num);
                memcpy(v_repeat->dataWithOffset((tid * n_rep + i) * len * size_per_head),
                       v_out->dataWithOffset(tid * len * size_per_head),
                       v_out->sizeBytes() / kv_head_num);
            }
        });
        k_out = std::move(k_repeat);
        v_out = std::move(v_repeat);
    }

    tStart         = std::chrono::steady_clock::now();
    auto qk_output = gemm_acl({*q_output,
                               *k_out,
                               std::nullopt,
                               nullptr,
                               DataType::TYPE_INVALID,
                               params.compute_type,
                               TransposeOperation::NONE,
                               TransposeOperation::TRANSPOSE});
    printBufferData(*qk_output, "qk_output: ");
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 4);

    printBufferData(*qk_output, "qk_output");

    tStart      = std::chrono::steady_clock::now();
    float scale = (1.0f / sqrtf(size_per_head * 1.0f));

    BufferPtr softmax_qk_output;
    if (seq_len == 1) {
        /* Decoder */
        softmax_qk_output = softmax({qk_output, std::nullopt, std::nullopt, scale});
    } else {
        /* Context */
        auto attention_mask = (*params.common.attention_mask).view(batch, 1);
        softmax_qk_output   = softmax({qk_output, attention_mask, std::nullopt, scale});
    }
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 5);
    printBufferData(*softmax_qk_output, "softmax_qk_output");

    tStart          = std::chrono::steady_clock::now();
    auto qkv_output = gemm_acl({*softmax_qk_output, *v_out});
    tEnd            = std::chrono::steady_clock::now();
    diff            = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 6);

    tStart = std::chrono::steady_clock::now();
    transposeDim12(qkv_output, params.output.dataWithOffset(past_seq * head_num * size_per_head));
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 7);
}

/* Inits array with per head base addresses */
template<typename T>
void getPerHeadArray(void*  qkv,
                     void*  k_seen,
                     void*  v_seen,
                     void** q_array,
                     void** k_array,
                     void** v_array,
                     int    num_heads,
                     int    kv_num_heads,
                     int    head_size) {
    const int N = 1 * num_heads;
    parallel_for(N, [&](int tid) {
        q_array[tid] = (T*)qkv + tid * head_size;
        k_array[tid] = (T*)k_seen + tid * kv_num_heads / num_heads * head_size;
        v_array[tid] = (T*)v_seen + tid * kv_num_heads / num_heads * head_size;
    });
}

template<typename T>
void updateKVCacheStride(const AttentionModuleParams& params, void* input, int batch, size_t seq_len, size_t step) {
    auto head_num      = params.configs.head_num;
    auto kv_head_num   = params.configs.kv_head_num;
    auto size_per_head = params.configs.size_per_head;
    auto block_tokens  = params.configs.tokens_per_block;

    size_t block_num    = (seq_len + block_tokens - 1) / block_tokens;
    size_t block_offset = step / block_tokens;
    auto   elem_sz      = params.input.typeSize();
    size_t copied_len   = 0;

    void* k_block_addr;
    void* v_block_addr;
    for (int i = 0; i < block_num; i++) {
        size_t len = std::min(block_tokens, seq_len - copied_len);
        getCacheAddrFromIndex(params.common.kv_cache.value(), batch, i + block_offset, &k_block_addr, &v_block_addr);

        T* k_input =
            (T*)input + (i * block_tokens) * (head_num + 2 * kv_head_num) * size_per_head + head_num * size_per_head;
        T* v_input = (T*)input + (i * block_tokens) * (head_num + 2 * kv_head_num) * size_per_head
                     + (head_num + kv_head_num) * size_per_head;
        parallel_for(len, [&](int tid) {
            memcpy((char*)k_block_addr + (step % block_tokens + tid) * elem_sz * kv_head_num * size_per_head,
                   k_input + tid * (head_num + 2 * kv_head_num) * size_per_head,
                   elem_sz * 1 * kv_head_num * size_per_head);
            memcpy((char*)v_block_addr + (step % block_tokens + tid) * elem_sz * kv_head_num * size_per_head,
                   v_input + tid * (head_num + 2 * kv_head_num) * size_per_head,
                   elem_sz * 1 * kv_head_num * size_per_head);
        });

        copied_len += len;
    }
}

void assemCacheArray(const AttentionModuleParams& params, BufferPtr k_out, BufferPtr v_out, size_t tokens_per_block) {
    auto   elem_sz          = k_out->typeSize();
    auto   batch_size       = k_out->shape()[0];
    auto   head_num         = k_out->shape()[2];
    auto   kv_seq_len       = k_out->shape()[1];
    auto   head_dim         = k_out->shape()[3];
    size_t blocks_per_batch = (kv_seq_len + tokens_per_block - 1) / tokens_per_block;
    size_t copied_len;

    void* k_block_addr;
    void* v_block_addr;
    for (int batch = 0; batch < batch_size; batch++) {
        copied_len = 0;
        for (int i = 0; i < blocks_per_batch; i++) {
            size_t len = std::min(tokens_per_block, kv_seq_len - copied_len);
            getCacheAddrFromIndex(params.common.kv_cache.value(), batch, i, &k_block_addr, &v_block_addr);

            memcpy(k_out->dataWithOffset((batch * kv_seq_len + i * tokens_per_block) * head_num * head_dim),
                   k_block_addr,
                   elem_sz * len * head_num * head_dim);
            memcpy(v_out->dataWithOffset((batch * kv_seq_len + i * tokens_per_block) * head_num * head_dim),
                   v_block_addr,
                   elem_sz * len * head_num * head_dim);
            copied_len += len;
        }
    }
}

void ArmCpuDevice::runOneBatchStride(
    const AttentionModuleParams& params, size_t past_seq, int batch, size_t seq_len, size_t step) {
    auto                                  datatype      = params.input.type();
    auto                                  head_num      = params.configs.head_num;
    auto                                  kv_head_num   = params.configs.kv_head_num;
    auto                                  size_per_head = params.configs.size_per_head;
    std::chrono::steady_clock::time_point tStart, tEnd;
    std::chrono::microseconds             diff;

    if (datatype != DataType::TYPE_FP32) {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
    // if (!params.common.kv_cache.has_value()) {
    //     throw std::runtime_error("kv cache block pointers can not be null");
    // }

    // Retrieve q/k/v by stride and not to split.
    auto qkv = params.input.dataWithOffset(past_seq * (head_num + 2 * kv_head_num) * size_per_head);
    printBufferData(params.input, "qkv");

    tStart = std::chrono::steady_clock::now();
    void *q_array[head_num], *k_array[head_num], *v_array[head_num];
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 0);

    tStart = std::chrono::steady_clock::now();
    // if (params.weights.qkv_weight->bias) {
    if (params.configs.fuse_qkv_add_bias && params.weights.qkv_weight->bias) {
        auto bias_data_type = params.weights.qkv_weight->bias->type();
        if (bias_data_type == DataType::TYPE_FP32) {
            addQKVBias<float, float>(
                qkv, params.weights.qkv_weight->bias->data(), 1, seq_len, head_num, kv_head_num, size_per_head);
        } else if (bias_data_type == DataType::TYPE_FP16) {
            addQKVBias<float, __fp16>(
                qkv, params.weights.qkv_weight->bias->data(), 1, seq_len, head_num, kv_head_num, size_per_head);
        } else {
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
        }
    }
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 1);
    printBufferData(params.input, "biased qkv");

    tStart = std::chrono::steady_clock::now();
    if (params.configs.rope_config.style != RopeStyle::No) {
        if (params.configs.rope_config.style == RopeStyle::Base) {
            halfRopeQK<float>(qkv,
                              1,
                              seq_len,
                              head_num,
                              kv_head_num,
                              size_per_head,
                              step,
                              params.configs.rope_config.base,
                              params.configs.rope_config.dim);
        } else {
            throw std::runtime_error("SelfAttention RoPE type is not supported");
        }
    }
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 2);
    printBufferData(params.input, "roped qkv");

    tStart = std::chrono::steady_clock::now();
    if (params.common.kv_cache.has_value()) {
        updateKVCacheStride<float>(params, qkv, batch, seq_len, step);
    }
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 8);

    BufferPtr k_buffer, v_buffer;
    void *    k_in, *v_in;
    int       stride_kv;
    if (step == 0) {
        /* Context */
        k_in      = (float*)qkv + head_num * size_per_head;
        v_in      = (float*)qkv + (head_num + kv_head_num) * size_per_head;
        stride_kv = (head_num + 2 * kv_head_num) * size_per_head;

        step = seq_len - 1;  // Trick to unify context and decoder processes.
    } else {
        /* Decoder */
        k_buffer = allocateBuffer({datatype, {1, step + 1, kv_head_num, size_per_head}, AllocationType::HOST}, {});
        v_buffer = allocateBuffer({datatype, {1, step + 1, kv_head_num, size_per_head}, AllocationType::HOST}, {});

        assemCache(params, batch, k_buffer, v_buffer);
        printBufferData(*k_buffer, "k_buffer");
        printBufferData(*v_buffer, "v_buffer");
        k_in      = k_buffer->data();
        v_in      = v_buffer->data();
        stride_kv = kv_head_num * size_per_head;
    }

    tStart = std::chrono::steady_clock::now();

    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 3);

    tStart = std::chrono::steady_clock::now();
    /* Re-init arrary as per [batch, num_heads]. */
    getPerHeadArray<float>(qkv, k_in, v_in, q_array, k_array, v_array, head_num, kv_head_num, size_per_head);

    auto qk_output = allocateBuffer({datatype, {1, head_num, seq_len, step + 1}, AllocationType::HOST}, {"qk_output"});
    const int stride_q = (head_num + 2 * kv_head_num) * size_per_head;
    // const int N = 1 * head_num;
    // parallel_for(N, [&](int tid) {
    const int MHA_HEADS = 1 * head_num;
    parallel_for(MHA_HEADS, [&](int tid) {
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasTrans,
                    seq_len,
                    step + 1,
                    size_per_head,
                    1.0,
                    (const float*)q_array[tid],
                    stride_q,
                    (const float*)k_array[tid],
                    stride_kv,
                    0.0,
                    (float*)qk_output->dataWithOffset(tid * seq_len * (step + 1)),
                    step + 1);
    });

    printBufferData(*qk_output, "qk_output");
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 4);

    tStart      = std::chrono::steady_clock::now();
    float scale = (1.0f / sqrtf(size_per_head * 1.0f));

    BufferPtr softmax_qk_output;
    if (seq_len == 1) {
        /* Decoder */
        softmax_qk_output = softmax({qk_output, std::nullopt, std::nullopt, scale});
    } else {
        /* Context */
        RUNTIME_ASSERT_OP_ARG(params.common.attention_mask,
                              "attention_mask must be provided for default context attention implementation");
        auto attention_mask = (*params.common.attention_mask).view(batch, 1);
        printBufferData(attention_mask, "attention_mask");
        softmax_qk_output = softmax({qk_output, attention_mask, std::nullopt, scale});
    }
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 5);

    printBufferData(*softmax_qk_output, "softmax_qk_output");

    tStart = std::chrono::steady_clock::now();
    // const int NN = 1 * head_num;
    // parallel_for(NN, [&](int tid) {
    //     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, seq_len, size_per_head, step + 1, 1.0,
    //         (const float*)softmax_qk_output->dataWithOffset(tid * seq_len * (step + 1)), step + 1,
    //         (const float*)v_array[tid], stride_kv, 0.0,
    //         (float*)params.output.dataWithOffset(past_seq * head_num * size_per_head + tid * size_per_head), head_num
    //         * size_per_head);
    // });
    if (!isKAIenabled) {
        parallel_for(MHA_HEADS, [&](int tid) {
            cblas_sgemm(CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        seq_len,
                        size_per_head,
                        step + 1,
                        1.0,
                        (const float*)softmax_qk_output->dataWithOffset(tid * seq_len * (step + 1)),
                        step + 1,
                        (const float*)v_array[tid],
                        stride_kv,
                        0.0,
                        (float*)params.output.dataWithOffset(past_seq * head_num * size_per_head + tid * size_per_head),
                        head_num * size_per_head);
        });
    } else {
        if (seq_len == 1) {
            /* Decoder has higher performance with cblas for gemm. */
            parallel_for(MHA_HEADS, [&](int tid) {
                cblas_sgemm(
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    seq_len,
                    size_per_head,
                    step + 1,
                    1.0,
                    (const float*)softmax_qk_output->dataWithOffset(tid * seq_len * (step + 1)),
                    step + 1,
                    (const float*)v_array[tid],
                    stride_kv,
                    0.0,
                    (float*)params.output.dataWithOffset(past_seq * head_num * size_per_head + tid * size_per_head),
                    head_num * size_per_head);
            });
        } else {
            /* Context has higher performance with KleidiAI for gemm. */
            const size_t bias_size = size_per_head;
            float*       bias      = new float[bias_size];
            memset(bias, 0, bias_size * sizeof(float));
            const size_t M               = seq_len;
            const size_t N               = size_per_head;
            const size_t K               = step + 1;
            const size_t mr              = kai_get_mr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla();
            const size_t nr              = kai_get_nr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla();
            const size_t kr              = kai_get_kr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla();
            const size_t sr              = kai_get_sr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla();
            const size_t lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_bf16p8x4_f32_neon(M, K, mr, kr, sr);
            const size_t rhs_packed_size =
                kai_get_rhs_packed_size_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon(N, K, nr, kr);
            const size_t lhs_stride     = K * sizeof(float);
            const size_t rhs_stride     = stride_kv * sizeof(float);
            const size_t dst_stride_row = (head_num * size_per_head) * sizeof(float);
            const size_t dst_stride_col = sizeof(float);

            parallel_for(MHA_HEADS, [&](int tid) {
                uint8_t* lhs_packed = new uint8_t[lhs_packed_size];
                uint8_t* rhs_packed = new uint8_t[rhs_packed_size];
                kai_run_lhs_quant_pack_bf16p8x4_f32_neon(M,
                                                         K,
                                                         mr,
                                                         kr,
                                                         sr,
                                                         0,
                                                         (const void*)softmax_qk_output->dataWithOffset(tid * M * K),
                                                         lhs_stride,
                                                         lhs_packed);
                kai_run_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon(
                    1, N, K, nr, kr, sr, rhs_stride, (const void*)v_array[tid], bias, NULL, rhs_packed, 0, NULL);
                kai_run_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla(
                    M,
                    N,
                    K,
                    lhs_packed,
                    rhs_packed,
                    (void*)params.output.dataWithOffset(past_seq * head_num * size_per_head + tid * size_per_head),
                    dst_stride_row,
                    dst_stride_col,
                    -FLT_MAX,
                    FLT_MAX);
                delete[] rhs_packed;
                delete[] lhs_packed;
            });

            delete[] bias;
        }
    }
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 6);

    tStart = std::chrono::steady_clock::now();
    tEnd   = std::chrono::steady_clock::now();
    diff   = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 7);

    /* Print profile data at the end of operator unit test. */
    // if (a_cnt_[0] == 24)
    //     printStat();
}

AttentionModuleOutput ArmCpuDevice::contextAttention(const AttentionModuleParams& params) {
    auto   batch_size    = params.common.context_batch_size;
    auto   decoder_batch = params.common.decoder_batch_size;
    size_t past_seq      = 0;

    if (params.input.type() == DataType::TYPE_FP32) {
        for (int batch = 0; batch < batch_size; batch++) {
            size_t context_len = *static_cast<int*>(params.common.input_lengths->dataWithOffset(decoder_batch + batch));
            runOneBatchStride(params, past_seq, batch, context_len, 0);
            past_seq += context_len;
        }
    } else if (params.input.type() == DataType::TYPE_FP16) {
        RTP_LLM_LOG_WARNING("Attention performance could be suboptimal with FP16 input. Try FP32 input.");
        for (int batch = 0; batch < batch_size; batch++) {
            size_t context_len = *static_cast<int*>(params.common.input_lengths->dataWithOffset(decoder_batch + batch));
            runOneBatch(params, past_seq, batch, context_len, 0);
            past_seq += context_len;
        }
    } else {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
}

AttentionModuleOutput ArmCpuDevice::decoderSelfAttention(const AttentionModuleParams& params) {
    auto batch_size = params.common.decoder_batch_size;

    if (params.input.type() == DataType::TYPE_FP32) {
        for (int batch = 0; batch < batch_size; batch++) {
            size_t step = *static_cast<int*>(params.common.sequence_lengths->dataWithOffset(batch));
            runOneBatchStride(params, batch, batch, 1, step);
        }
    } else if (params.input.type() == DataType::TYPE_FP16) {
        for (int batch = 0; batch < batch_size; batch++) {
            size_t step = *static_cast<int*>(params.common.sequence_lengths->dataWithOffset(batch));
            runOneBatch(params, batch, batch, 1, step);
        }
    } else {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
}
}  // namespace rtp_llm

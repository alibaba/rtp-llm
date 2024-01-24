#pragma once

#include <cuda.h>
#include <string>

#include "src/fastertransformer/kernels/decoder_masked_multihead_attention.h"

namespace fastertransformer {

// WARNING: These functions lead to memory leak, do not use in producion !

void fwrite_gpu(const void* ptr, size_t size, FILE* stream);

void* fread_gpu(size_t size, FILE* stream, size_t allocation_size = 0);

template<typename T>
void serialize_multihead_attention_params(Masked_multihead_attention_params<T>& params, const std::string& path) {
    // create file at path and serialize params to it
    FILE* fd = fopen(path.c_str(), "wb");
    if (fd == NULL) {
        printf("[ERROR] create file %s failed\n", path.c_str());
        exit(-1);
    }
    fwrite(&params, sizeof(params), 1, fd);
    printf("[INFO] serialize params to %s\n", path.c_str());

    const auto hidden_units = params.num_heads_kv * params.hidden_size_per_head;
    const auto qkv_size     = hidden_units * params.batch_size * sizeof(T);
    const auto cache_size   = params.batch_size * params.timestep * hidden_units * sizeof(T);

    printf("[INFO] hidden_units=%d, num_heads_kv=%d, size_per_head=%d, stride=%d, "
           "int8_mode=%d, ",
           hidden_units,
           params.num_heads_kv,
           params.hidden_size_per_head,
           params.stride,
           params.int8_mode);

    fwrite_gpu(params.q, qkv_size, fd);
    fwrite_gpu(params.k, qkv_size, fd);
    fwrite_gpu(params.v, qkv_size, fd);
    fwrite_gpu(params.q_bias, qkv_size, fd);
    fwrite_gpu(params.k_bias, qkv_size, fd);
    fwrite_gpu(params.v_bias, qkv_size, fd);
    fwrite_gpu(params.k_cache, cache_size, fd);
    fwrite_gpu(params.v_cache, cache_size, fd);
    fwrite_gpu(params.out, qkv_size, fd);
    fwrite_gpu(params.length_per_sample, params.batch_size * sizeof(int32_t), fd);

    // fields to process: finished, prefix_prompt_lengths

    fclose(fd);
}

template<typename T>
void deserialize_multihead_attention_params(Masked_multihead_attention_params<T>& params, const std::string& path) {
    // read file at path and deserialize params from it
    FILE* fd = fopen(path.c_str(), "rb");
    if (fd == NULL) {
        printf("[ERROR] open file %s failed\n", path.c_str());
        exit(-1);
    }
    fread(&params, sizeof(params), 1, fd);
    // printf("[INFO] deserialize params from %s\n", path.c_str());

    const auto hidden_units     = params.num_heads_kv * params.hidden_size_per_head;
    const auto qkv_size         = hidden_units * params.batch_size * sizeof(T);
    const auto cache_alloc_size = params.batch_size * (params.timestep + 8192) * hidden_units * sizeof(T);
    const auto cache_size       = params.batch_size * params.timestep * hidden_units * sizeof(T);

    // printf("[INFO] hidden_units=%d, num_heads_kv=%d, hidden_size_per_head=%d, stride=%d, "
    //        "int8_mode=%d \n",
    //        hidden_units, params.num_heads_kv, params.hidden_size_per_head,
    //        params.stride, params.int8_mode);

    params.q                 = (T*)fread_gpu(qkv_size, fd);
    params.k                 = (T*)fread_gpu(qkv_size, fd);
    params.v                 = (T*)fread_gpu(qkv_size, fd);
    params.q_bias            = (T*)fread_gpu(qkv_size, fd);
    params.k_bias            = (T*)fread_gpu(qkv_size, fd);
    params.v_bias            = (T*)fread_gpu(qkv_size, fd);
    params.k_cache           = (T*)fread_gpu(cache_size, fd, cache_alloc_size);
    params.v_cache           = (T*)fread_gpu(cache_size, fd, cache_alloc_size);
    params.out               = (T*)fread_gpu(qkv_size, fd);
    params.length_per_sample = (int32_t*)fread_gpu(params.batch_size * sizeof(int32_t), fd);

    params.finished              = nullptr;
    params.prefix_prompt_lengths = nullptr;
    params.input_lengths         = nullptr;
    params.cache_indir           = nullptr;

    fclose(fd);
}

};  // namespace fastertransformer

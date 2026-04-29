#pragma once

#include <torch/extension.h>
#include "rtp_llm/models_py/bindings/rocm/kernels/ds_v4_compress_kernel.h"
#include "rtp_llm/models_py/bindings/rocm/kernels/ds_v4_topk512_kernel.h"
#include "rtp_llm/models_py/bindings/rocm/kernels/ds_v4_store_cache_kernel.h"

namespace rtp_llm {

// --- FlashCompress4/128 ---

void flashCompress4Decode(
    torch::Tensor kv_score_buffer,
    torch::Tensor kv_score_input,
    torch::Tensor kv_compressed_output,
    std::optional<torch::Tensor> score_bias,
    torch::Tensor indices,
    torch::Tensor seq_lens,
    std::optional<torch::Tensor> extra,
    uintptr_t hip_stream = 0) {

    TORCH_CHECK(kv_score_input.is_cuda(), "kv_score_input must be on CUDA/ROCm device");
    TORCH_CHECK(kv_score_input.dim() == 2, "kv_score_input must be [batch_size, head_dim * 4]");
    const uint32_t batch_size = static_cast<uint32_t>(kv_score_input.size(0));
    const int64_t head_dim = kv_score_input.size(1) / 4;

    invokeFlashCompress4Decode(
        kv_score_buffer.data_ptr(),
        kv_score_input.data_ptr(),
        kv_compressed_output.data_ptr(),
        score_bias.has_value() ? score_bias->data_ptr() : nullptr,
        static_cast<int32_t*>(indices.data_ptr()),
        static_cast<int32_t*>(seq_lens.data_ptr()),
        extra.has_value() ? static_cast<int32_t*>(extra->data_ptr()) : nullptr,
        batch_size,
        head_dim,
        reinterpret_cast<hipStream_t>(hip_stream));
}

void flashCompress4Prefill(
    torch::Tensor kv_score_buffer,
    torch::Tensor kv_score_input,
    torch::Tensor kv_compressed_output,
    std::optional<torch::Tensor> score_bias,
    torch::Tensor indices,
    torch::Tensor compress_plan,
    torch::Tensor write_plan,
    std::optional<torch::Tensor> extra,
    uintptr_t hip_stream = 0) {

    TORCH_CHECK(kv_score_input.is_cuda(), "kv_score_input must be on CUDA/ROCm device");
    TORCH_CHECK(compress_plan.dim() == 2, "compress_plan must be [num_compress, 4]");
    TORCH_CHECK(write_plan.dim() == 2, "write_plan must be [num_write, 4]");

    const uint32_t num_compress = static_cast<uint32_t>(compress_plan.size(0));
    const uint32_t num_write = static_cast<uint32_t>(write_plan.size(0));
    const int64_t head_dim = kv_score_input.size(1) / 4;

    invokeFlashCompress4Prefill(
        kv_score_buffer.data_ptr(),
        kv_score_input.data_ptr(),
        kv_compressed_output.data_ptr(),
        score_bias.has_value() ? score_bias->data_ptr() : nullptr,
        static_cast<int32_t*>(indices.data_ptr()),
        static_cast<int32_t*>(compress_plan.data_ptr()),
        static_cast<int32_t*>(write_plan.data_ptr()),
        extra.has_value() ? static_cast<int32_t*>(extra->data_ptr()) : nullptr,
        num_compress,
        num_write,
        head_dim,
        reinterpret_cast<hipStream_t>(hip_stream));
}

void flashCompress128Decode(
    torch::Tensor kv_score_buffer,
    torch::Tensor kv_score_input,
    torch::Tensor kv_compressed_output,
    std::optional<torch::Tensor> score_bias,
    torch::Tensor indices,
    torch::Tensor seq_lens,
    uintptr_t hip_stream = 0) {

    TORCH_CHECK(kv_score_input.is_cuda(), "kv_score_input must be on CUDA/ROCm device");
    TORCH_CHECK(kv_score_input.dim() == 2, "kv_score_input must be [batch_size, head_dim * 2]");
    const uint32_t batch_size = static_cast<uint32_t>(kv_score_input.size(0));
    const int64_t head_dim = kv_score_input.size(1) / 2;

    invokeFlashCompress128Decode(
        kv_score_buffer.data_ptr(),
        kv_score_input.data_ptr(),
        kv_compressed_output.data_ptr(),
        score_bias.has_value() ? score_bias->data_ptr() : nullptr,
        static_cast<int32_t*>(indices.data_ptr()),
        static_cast<int32_t*>(seq_lens.data_ptr()),
        batch_size,
        head_dim,
        reinterpret_cast<hipStream_t>(hip_stream));
}

void flashCompress128Prefill(
    torch::Tensor kv_score_buffer,
    torch::Tensor kv_score_input,
    torch::Tensor kv_compressed_output,
    std::optional<torch::Tensor> score_bias,
    torch::Tensor indices,
    torch::Tensor compress_plan,
    torch::Tensor write_plan,
    uintptr_t hip_stream = 0) {

    TORCH_CHECK(kv_score_input.is_cuda(), "kv_score_input must be on CUDA/ROCm device");
    const uint32_t num_compress = static_cast<uint32_t>(compress_plan.size(0));
    const uint32_t num_write = static_cast<uint32_t>(write_plan.size(0));
    const int64_t head_dim = kv_score_input.size(1) / 2;

    invokeFlashCompress128Prefill(
        kv_score_buffer.data_ptr(),
        kv_score_input.data_ptr(),
        kv_compressed_output.data_ptr(),
        score_bias.has_value() ? score_bias->data_ptr() : nullptr,
        static_cast<int32_t*>(indices.data_ptr()),
        static_cast<int32_t*>(compress_plan.data_ptr()),
        static_cast<int32_t*>(write_plan.data_ptr()),
        num_compress,
        num_write,
        head_dim,
        reinterpret_cast<hipStream_t>(hip_stream));
}

// --- TopK512 ---

void topk512(
    torch::Tensor scores,
    torch::Tensor seq_lens,
    torch::Tensor page_table,
    torch::Tensor page_indices,
    std::optional<torch::Tensor> raw_indices,
    int64_t page_size,
    uintptr_t hip_stream = 0) {

    TORCH_CHECK(scores.is_cuda(), "scores must be on CUDA/ROCm device");
    TORCH_CHECK(scores.dim() == 2, "scores must be [batch_size, score_stride]");
    TORCH_CHECK(page_indices.dim() == 2 && page_indices.size(1) == 512,
                "page_indices must be [batch_size, 512]");

    const uint32_t batch_size = static_cast<uint32_t>(scores.size(0));
    const int64_t score_stride = scores.stride(0);
    const int64_t page_table_stride = page_table.stride(0);

    invokeTopK512(
        static_cast<float*>(scores.data_ptr()),
        static_cast<int32_t*>(seq_lens.data_ptr()),
        static_cast<int32_t*>(page_table.data_ptr()),
        static_cast<int32_t*>(page_indices.data_ptr()),
        raw_indices.has_value() ? static_cast<int32_t*>(raw_indices->data_ptr()) : nullptr,
        batch_size,
        score_stride,
        page_table_stride,
        static_cast<uint32_t>(page_size),
        reinterpret_cast<hipStream_t>(hip_stream));
}

// --- FusedStoreCache ---

void fusedStoreCacheFlashMLA(
    torch::Tensor input,
    torch::Tensor cache,
    torch::Tensor indices,
    int64_t page_size,
    uintptr_t hip_stream = 0) {

    TORCH_CHECK(input.is_cuda(), "input must be on CUDA/ROCm device");
    TORCH_CHECK(input.dim() == 2 && input.size(1) == 512, "input must be [num_tokens, 512]");

    const uint32_t num_tokens = static_cast<uint32_t>(input.size(0));

    invokeFusedStoreCacheFlashMLA(
        input.data_ptr(),
        cache.data_ptr(),
        indices.data_ptr(),
        num_tokens,
        static_cast<uint32_t>(page_size),
        reinterpret_cast<hipStream_t>(hip_stream));
}

void fusedStoreCacheIndexer(
    torch::Tensor input,
    torch::Tensor cache,
    torch::Tensor indices,
    int64_t page_size,
    uintptr_t hip_stream = 0) {

    TORCH_CHECK(input.is_cuda(), "input must be on CUDA/ROCm device");
    TORCH_CHECK(input.dim() == 2 && input.size(1) == 128, "input must be [num_tokens, 128]");

    const uint32_t num_tokens = static_cast<uint32_t>(input.size(0));

    invokeFusedStoreCacheIndexer(
        input.data_ptr(),
        cache.data_ptr(),
        indices.data_ptr(),
        num_tokens,
        static_cast<uint32_t>(page_size),
        reinterpret_cast<hipStream_t>(hip_stream));
}

// --- Registration ---

void registerDeepSeekV4Ops(py::module& rtp_ops_m) {
    rtp_ops_m.def("flash_compress4_decode",
                  &flashCompress4Decode,
                  "DeepSeek-V4 C4 compression decode kernel",
                  py::arg("kv_score_buffer"),
                  py::arg("kv_score_input"),
                  py::arg("kv_compressed_output"),
                  py::arg("score_bias") = std::nullopt,
                  py::arg("indices"),
                  py::arg("seq_lens"),
                  py::arg("extra") = std::nullopt,
                  py::arg("hip_stream") = 0);

    rtp_ops_m.def("flash_compress4_prefill",
                  &flashCompress4Prefill,
                  "DeepSeek-V4 C4 compression prefill kernel",
                  py::arg("kv_score_buffer"),
                  py::arg("kv_score_input"),
                  py::arg("kv_compressed_output"),
                  py::arg("score_bias") = std::nullopt,
                  py::arg("indices"),
                  py::arg("compress_plan"),
                  py::arg("write_plan"),
                  py::arg("extra") = std::nullopt,
                  py::arg("hip_stream") = 0);

    rtp_ops_m.def("flash_compress128_decode",
                  &flashCompress128Decode,
                  "DeepSeek-V4 C128 compression decode kernel",
                  py::arg("kv_score_buffer"),
                  py::arg("kv_score_input"),
                  py::arg("kv_compressed_output"),
                  py::arg("score_bias") = std::nullopt,
                  py::arg("indices"),
                  py::arg("seq_lens"),
                  py::arg("hip_stream") = 0);

    rtp_ops_m.def("flash_compress128_prefill",
                  &flashCompress128Prefill,
                  "DeepSeek-V4 C128 compression prefill kernel",
                  py::arg("kv_score_buffer"),
                  py::arg("kv_score_input"),
                  py::arg("kv_compressed_output"),
                  py::arg("score_bias") = std::nullopt,
                  py::arg("indices"),
                  py::arg("compress_plan"),
                  py::arg("write_plan"),
                  py::arg("hip_stream") = 0);

    rtp_ops_m.def("topk512",
                  &topk512,
                  "DeepSeek-V4 TopK512 radix histogram kernel",
                  py::arg("scores"),
                  py::arg("seq_lens"),
                  py::arg("page_table"),
                  py::arg("page_indices"),
                  py::arg("raw_indices") = std::nullopt,
                  py::arg("page_size") = 16,
                  py::arg("hip_stream") = 0);

    rtp_ops_m.def("fused_store_cache_flashmla",
                  &fusedStoreCacheFlashMLA,
                  "DeepSeek-V4 FusedStoreCache FlashMLA variant",
                  py::arg("input"),
                  py::arg("cache"),
                  py::arg("indices"),
                  py::arg("page_size") = 16,
                  py::arg("hip_stream") = 0);

    rtp_ops_m.def("fused_store_cache_indexer",
                  &fusedStoreCacheIndexer,
                  "DeepSeek-V4 FusedStoreCache Indexer variant",
                  py::arg("input"),
                  py::arg("cache"),
                  py::arg("indices"),
                  py::arg("page_size") = 16,
                  py::arg("hip_stream") = 0);
}

}  // namespace rtp_llm

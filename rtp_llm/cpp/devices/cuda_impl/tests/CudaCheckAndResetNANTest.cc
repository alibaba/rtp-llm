#include "rtp_llm/cpp/devices/cuda_impl/tests/CudaTestUtils.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/kernels/nan_check_kernels.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif
#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <iostream>
#include <iomanip>

using namespace rtp_llm;

// Inf constants for different types
#ifndef CUDART_INF_FP16
#define CUDART_INF_FP16 __ushort_as_half((unsigned short)0x7C00U)
#endif
#ifndef CUDART_INF_BF16
#define CUDART_INF_BF16 __ushort_as_bfloat16((unsigned short)0x7F80U)
#endif

class CudaCheckAndResetNANTest: public DeviceTestBase {
protected:
    void SetUp() override {
        cudaGetLastError();
        cudaStreamCreate(&stream_);
    }

    void TearDown() override {
        if (stream_) {
            cudaStreamSynchronize(stream_);
            cudaGetLastError();
            cudaStreamDestroy(stream_);
        }
        DeviceTestBase::TearDown();
    }

    // Helper function to generate NaN values
    template<typename T>
    T generateNaN() {
        if constexpr (std::is_same_v<T, float>) {
            return std::nanf("");
        } else if constexpr (std::is_same_v<T, half>) {
            return __float2half(std::nanf(""));
        }
#ifdef ENABLE_BF16
        else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            return __float2bfloat16(std::nanf(""));
        }
#endif
#ifdef ENABLE_FP8
        else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
            return __nv_fp8_e4m3(std::nanf(""));
        }
#endif
        else {
            return T(0);
        }
    }

    // Helper function to check if value is NaN
    template<typename T>
    bool isNaNValue(T val) {
        if constexpr (std::is_same_v<T, float>) {
            return std::isnan(val);
        } else if constexpr (std::is_same_v<T, half>) {
            return std::isnan(__half2float(val));
        }
#ifdef ENABLE_BF16
        else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            return std::isnan(__bfloat162float(val));
        }
#endif
#ifdef ENABLE_FP8
        else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
            float fval = static_cast<float>(val);
            return std::isnan(fval);
        }
#endif
        else {
            return false;
        }
    }

    // Helper function to generate Inf values
    template<typename T>
    T generateInf(bool positive = true) {
        if constexpr (std::is_same_v<T, float>) {
            return positive ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity();
        } else if constexpr (std::is_same_v<T, half>) {
            return positive ? CUDART_INF_FP16 : -CUDART_INF_FP16;
        }
#ifdef ENABLE_BF16
        else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            return positive ? CUDART_INF_BF16 : -CUDART_INF_BF16;
        }
#endif
#ifdef ENABLE_FP8
        else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
            // FP8 E4M3: Inf = exponent 0xF, mantissa 0
            uint8_t bits = positive ? 0x78 : 0xF8;  // 0x78 = 0111 1000, 0xF8 = 1111 1000
            return *reinterpret_cast<const __nv_fp8_e4m3*>(&bits);
        }
#endif
        else {
            return T(0);
        }
    }

    // Helper function to check if value is Inf
    template<typename T>
    bool isInfValue(T val) {
        if constexpr (std::is_same_v<T, float>) {
            return std::isinf(val);
        } else if constexpr (std::is_same_v<T, half>) {
            float fval = __half2float(val);
            return std::isinf(fval);
        }
#ifdef ENABLE_BF16
        else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            float fval = __bfloat162float(val);
            return std::isinf(fval);
        }
#endif
#ifdef ENABLE_FP8
        else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
            uint8_t bits     = *reinterpret_cast<const uint8_t*>(&val);
            uint8_t exponent = (bits >> 3) & 0xF;
            uint8_t mantissa = bits & 0x7;
            return (exponent == 0xF) && (mantissa == 0);
        }
#endif
        else {
            return false;
        }
    }

    // Helper function to check if value is NaN or Inf
    template<typename T>
    bool isInvalidValue(T val) {
        return isNaNValue(val) || isInfValue(val);
    }

    // Helper function to convert to float for comparison
    template<typename T>
    float toFloat(T val) {
        if constexpr (std::is_same_v<T, float>) {
            return val;
        } else if constexpr (std::is_same_v<T, half>) {
            return __half2float(val);
        }
#ifdef ENABLE_BF16
        else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            return __bfloat162float(val);
        }
#endif
#ifdef ENABLE_FP8
        else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
            return static_cast<float>(val);
        }
#endif
        else {
            return static_cast<float>(val);
        }
    }

    // Helper function to test KV cache NaN check for Prefill
    // positions: (batch_id, layer_id, token_idx, is_inf)
    //   token_idx is the global token index (not block_idx + token_offset)
    //   is_inf: true for Inf, false for NaN
    template<typename T>
    void
    testKVCachePrefillCheckAndResetNAN(size_t                      batch_size,
                                       size_t                      layer_num,
                                       size_t                      max_blocks_per_batch,
                                       size_t                      block_size_bytes,
                                       size_t                      seq_size_per_block,
                                       const std::vector<int32_t>& prefix_lengths,
                                       const std::vector<int32_t>& seq_len_cu,
                                       const std::vector<std::vector<int32_t>>&
                                           kv_cache_block_id,  // [batch_size][max_blocks_per_batch], physical block IDs
                                       const std::vector<std::tuple<size_t, size_t, int32_t, bool, bool>>&
                                           positions,  // (batch_id, layer_id, token_idx, is_inf, is_k_part)
                                                       // is_k_part: true for K cache, false for V cache
                                       bool   use_mla       = false,
                                       size_t kv_lora_rank  = 0,
                                       size_t rope_head_dim = 0) {

        size_t element_size = sizeof(T);
        size_t token_bytes  = block_size_bytes / seq_size_per_block;
        size_t token_stride = block_size_bytes / seq_size_per_block;

        ASSERT_EQ(token_bytes, token_stride) << "token_bytes must equal token_stride";

        // Find the maximum physical block ID
        int32_t max_physical_block_id = -1;
        for (const auto& batch_blocks : kv_cache_block_id) {
            for (int32_t block_id : batch_blocks) {
                if (block_id > max_physical_block_id) {
                    max_physical_block_id = block_id;
                }
            }
        }
        size_t num_physical_blocks = max_physical_block_id + 1;

        // Allocate independent memory space for each layer
        size_t blocks_per_layer     = num_physical_blocks;
        size_t layer_size           = blocks_per_layer * block_size_bytes;
        size_t kv_cache_buffer_size = layer_num * layer_size;

        // Clear any previous CUDA errors
        cudaGetLastError();

        T*          d_kv_cache;
        cudaError_t err = cudaMalloc(&d_kv_cache, kv_cache_buffer_size);
        ASSERT_EQ(err, cudaSuccess) << "Failed to allocate d_kv_cache: " << cudaGetErrorString(err);

        // Initialize with normal values (1.0)
        std::vector<T> h_kv_cache_init(kv_cache_buffer_size / element_size, T(1.0f));
        cudaMemcpy(d_kv_cache, h_kv_cache_init.data(), kv_cache_buffer_size, cudaMemcpyHostToDevice);

        // Set up layer_base_addr array (one base pointer per layer)
        std::vector<void*> h_layer_base_addr(layer_num);
        for (size_t layer_id = 0; layer_id < layer_num; ++layer_id) {
            // Each layer has its own memory region
            h_layer_base_addr[layer_id] = reinterpret_cast<char*>(d_kv_cache) + layer_id * layer_size;
        }

        // Copy layer_base_addr to device
        void** d_layer_base_addr;
        err = cudaMalloc(&d_layer_base_addr, layer_num * sizeof(void*));
        ASSERT_EQ(err, cudaSuccess) << "Failed to allocate d_layer_base_addr: " << cudaGetErrorString(err);
        cudaMemcpy(d_layer_base_addr, h_layer_base_addr.data(), layer_num * sizeof(void*), cudaMemcpyHostToDevice);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        // Set up kv_cache_block_id array
        std::vector<int32_t> h_kv_cache_block_id(batch_size * max_blocks_per_batch, -1);
        for (size_t batch_id = 0; batch_id < batch_size; ++batch_id) {
            for (size_t block_idx = 0;
                 block_idx < max_blocks_per_batch && block_idx < kv_cache_block_id[batch_id].size();
                 ++block_idx) {
                h_kv_cache_block_id[batch_id * max_blocks_per_batch + block_idx] =
                    kv_cache_block_id[batch_id][block_idx];
            }
        }
        int32_t* d_kv_cache_block_id;
        err = cudaMalloc(&d_kv_cache_block_id, batch_size * max_blocks_per_batch * sizeof(int32_t));
        ASSERT_EQ(err, cudaSuccess) << "Failed to allocate d_kv_cache_block_id: " << cudaGetErrorString(err);
        cudaMemcpy(d_kv_cache_block_id,
                   h_kv_cache_block_id.data(),
                   batch_size * max_blocks_per_batch * sizeof(int32_t),
                   cudaMemcpyHostToDevice);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        // Set up prefix_lengths and seq_len_cu
        int32_t* d_prefix_lengths;
        err = cudaMalloc(&d_prefix_lengths, batch_size * sizeof(int32_t));
        ASSERT_EQ(err, cudaSuccess) << "Failed to allocate d_prefix_lengths: " << cudaGetErrorString(err);
        cudaMemcpy(d_prefix_lengths, prefix_lengths.data(), batch_size * sizeof(int32_t), cudaMemcpyHostToDevice);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        int32_t* d_seq_len_cu;
        err = cudaMalloc(&d_seq_len_cu, batch_size * sizeof(int32_t));
        ASSERT_EQ(err, cudaSuccess) << "Failed to allocate d_seq_len_cu: " << cudaGetErrorString(err);
        cudaMemcpy(d_seq_len_cu, seq_len_cu.data(), batch_size * sizeof(int32_t), cudaMemcpyHostToDevice);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        // Allocate nan_flag
        int32_t* d_nan_flag;
        err = cudaMalloc(&d_nan_flag, batch_size * sizeof(int32_t));
        ASSERT_EQ(err, cudaSuccess) << "Failed to allocate d_nan_flag: " << cudaGetErrorString(err);
        cudaMemset(d_nan_flag, 0, batch_size * sizeof(int32_t));
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        // Calculate parameters for new kernel signature
        size_t local_head_num_kv;
        size_t k_token_size_per_head;
        size_t v_token_size_per_head;

        if (use_mla) {
            // MLA: K = kv_lora_rank, V = rope_head_dim, local_head_num_kv = 1
            local_head_num_kv     = 1;
            k_token_size_per_head = kv_lora_rank;
            v_token_size_per_head = rope_head_dim;
        } else {
            // MHA: k_token_size = v_token_size = size_per_head
            // block_size_bytes = (k_token_size + v_token_size) * seq_size_per_block * element_size
            // So we can derive: k_token_size_per_head = k_token_size, v_token_size_per_head = v_token_size
            local_head_num_kv     = 1;  // Assume 1 head for test (can be adjusted if needed)
            k_token_size_per_head = token_bytes / (2 * element_size);  // k_token_size = v_token_size, so divide by 2
            v_token_size_per_head = k_token_size_per_head;
        }

        size_t k_token_bytes_per_head = k_token_size_per_head * element_size;
        size_t v_token_bytes_per_head = v_token_size_per_head * element_size;
        size_t k_block_size_bytes     = local_head_num_kv * k_token_size_per_head * seq_size_per_block * element_size;
        size_t v_block_size_bytes     = local_head_num_kv * v_token_size_per_head * seq_size_per_block * element_size;

        // Collect all positions where NaN/Inf values need to be inserted
        // For MHA layout: [layer_num, block_num, 2, local_head_num_kv, seq_size_per_block, k_token_size]
        // We need to insert NaN/Inf for each head's K and V parts
        std::vector<bool>   expected_has_invalid(batch_size, false);
        std::vector<size_t> token_offsets;
        std::vector<bool>   is_nan_flags;

        // Process NaN and Inf positions
        for (const auto& pos : positions) {
            size_t  batch_id  = std::get<0>(pos);
            size_t  layer_id  = std::get<1>(pos);
            int32_t token_idx = std::get<2>(pos);
            bool    is_inf    = std::get<3>(pos);
            bool    is_k_part = std::get<4>(pos);

            if (batch_id >= batch_size || layer_id >= layer_num)
                continue;

            int32_t logical_block_idx = token_idx / seq_size_per_block;
            int32_t offset_in_block   = token_idx % seq_size_per_block;

            if (logical_block_idx < 0 || logical_block_idx >= static_cast<int32_t>(max_blocks_per_batch))
                continue;

            if (logical_block_idx >= static_cast<int32_t>(kv_cache_block_id[batch_id].size()))
                continue;

            int32_t physical_block_id = kv_cache_block_id[batch_id][logical_block_idx];
            if (physical_block_id == -1 || physical_block_id >= static_cast<int32_t>(num_physical_blocks))
                continue;

            if (token_idx >= prefix_lengths[batch_id] && token_idx < seq_len_cu[batch_id]) {
                expected_has_invalid[batch_id] = true;
            }

            // Calculate base offsets
            size_t layer_offset_bytes = layer_id * layer_size;
            size_t block_offset_bytes = physical_block_id * block_size_bytes;
            char*  layer_base         = reinterpret_cast<char*>(d_kv_cache) + layer_offset_bytes;
            char*  block_base         = layer_base + block_offset_bytes;

            // Insert NaN/Inf for each head's K or V part based on is_k_part
            // Layout: K parts first (all heads), then V parts (all heads)
            for (size_t head_id = 0; head_id < local_head_num_kv; head_id++) {
                if (is_k_part) {
                    // K part: block_base + head_id * (seq_size_per_block * k_token_bytes_per_head) + offset_in_block *
                    // k_token_bytes_per_head
                    char* k_data = block_base + head_id * (seq_size_per_block * k_token_bytes_per_head)
                                   + offset_in_block * k_token_bytes_per_head;
                    size_t k_offset = reinterpret_cast<size_t>(k_data) - reinterpret_cast<size_t>(d_kv_cache);

                    if (k_offset < kv_cache_buffer_size) {
                        token_offsets.push_back(k_offset);
                        is_nan_flags.push_back(!is_inf);
                    }
                } else {
                    // V part: block_base + k_block_size_bytes + head_id * (seq_size_per_block * v_token_bytes_per_head)
                    // + offset_in_block * v_token_bytes_per_head
                    char* v_data = block_base + k_block_size_bytes
                                   + head_id * (seq_size_per_block * v_token_bytes_per_head)
                                   + offset_in_block * v_token_bytes_per_head;
                    size_t v_offset = reinterpret_cast<size_t>(v_data) - reinterpret_cast<size_t>(d_kv_cache);

                    if (v_offset < kv_cache_buffer_size) {
                        token_offsets.push_back(v_offset);
                        is_nan_flags.push_back(!is_inf);
                    }
                }
            }
        }

        // Batch insert NaN/Inf values
        if (!token_offsets.empty()) {
            std::vector<T> h_values(token_offsets.size());
            for (size_t i = 0; i < token_offsets.size(); ++i) {
                h_values[i] = is_nan_flags[i] ? generateNaN<T>() : generateInf<T>();
            }

            const size_t bytes_to_write = 16;  // Always write 16 bytes to ensure uint4 alignment
            for (size_t i = 0; i < token_offsets.size(); ++i) {
                void* dst = reinterpret_cast<char*>(d_kv_cache) + token_offsets[i];

                size_t         num_elems = (bytes_to_write + sizeof(T) - 1) / sizeof(T);
                std::vector<T> token_data(num_elems, h_values[i]);
                cudaMemcpyAsync(dst, token_data.data(), bytes_to_write, cudaMemcpyHostToDevice, stream_);
            }
            cudaStreamSynchronize(stream_);
        }

        // Create events for timing
        cudaEvent_t start, stop;
        cudaError_t event_err = cudaEventCreate(&start);
        ASSERT_EQ(event_err, cudaSuccess) << "Failed to create start event: " << cudaGetErrorString(event_err);

        event_err = cudaEventCreate(&stop);
        ASSERT_EQ(event_err, cudaSuccess) << "Failed to create stop event: " << cudaGetErrorString(event_err);

        // Run kernel
        cudaEventRecord(start, stream_);
        invokeCheckAndResetNANKvCachePrefill<T>(reinterpret_cast<const void* const*>(d_layer_base_addr),
                                                d_kv_cache_block_id,
                                                d_prefix_lengths,
                                                d_seq_len_cu,
                                                batch_size,
                                                layer_num,
                                                max_blocks_per_batch,
                                                local_head_num_kv,
                                                k_token_size_per_head,
                                                v_token_size_per_head,
                                                k_block_size_bytes,
                                                v_block_size_bytes,
                                                k_token_bytes_per_head,
                                                v_token_bytes_per_head,
                                                block_size_bytes,
                                                seq_size_per_block,
                                                d_nan_flag,
                                                stream_);
        cudaEventRecord(stop, stream_);
        cudaStreamSynchronize(stream_);

        // Get timing
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf(
            "Prefill Kernel batch_size: [%zu], layer_num: [%zu], max_blocks_per_batch: [%zu], Execution time: %f ms\n",
            batch_size,
            layer_num,
            max_blocks_per_batch,
            milliseconds);

        // Cleanup events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // Check for CUDA errors
        cudaError_t kernel_err = cudaGetLastError();
        ASSERT_EQ(kernel_err, cudaSuccess) << "CUDA error after kernel launch: " << cudaGetErrorString(kernel_err);

        // Copy nan_flag back
        std::vector<int32_t> h_nan_flag(batch_size);
        cudaMemcpy(h_nan_flag.data(), d_nan_flag, batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost);

        // Collect all positions that need verification and their offsets
        // For MHA layout, we need to verify each head's K and V parts
        struct VerifyPosition {
            size_t  batch_id;
            size_t  layer_id;
            int32_t token_idx;
            size_t  offset_bytes;
            bool    in_checked_region;
        };
        std::vector<VerifyPosition> verify_positions;

        // Collect NaN and Inf positions
        for (const auto& pos : positions) {
            size_t  batch_id  = std::get<0>(pos);
            size_t  layer_id  = std::get<1>(pos);
            int32_t token_idx = std::get<2>(pos);
            bool    is_k_part = std::get<4>(pos);

            if (batch_id >= batch_size || layer_id >= layer_num)
                continue;

            int32_t logical_block_idx = token_idx / seq_size_per_block;
            int32_t offset_in_block   = token_idx % seq_size_per_block;

            if (logical_block_idx < 0 || logical_block_idx >= static_cast<int32_t>(max_blocks_per_batch))
                continue;

            int32_t physical_block_id = kv_cache_block_id[batch_id][logical_block_idx];
            if (physical_block_id == -1)
                continue;

            bool in_checked_region = (token_idx >= prefix_lengths[batch_id] && token_idx < seq_len_cu[batch_id]);

            // Calculate base offsets
            size_t layer_offset_bytes = layer_id * layer_size;
            size_t block_offset_bytes = physical_block_id * block_size_bytes;
            char*  layer_base         = reinterpret_cast<char*>(d_kv_cache) + layer_offset_bytes;
            char*  block_base         = layer_base + block_offset_bytes;

            // Verify each head's K or V part based on is_k_part
            for (size_t head_id = 0; head_id < local_head_num_kv; head_id++) {
                if (is_k_part) {
                    // K part
                    char* k_data = block_base + head_id * (seq_size_per_block * k_token_bytes_per_head)
                                   + offset_in_block * k_token_bytes_per_head;
                    size_t k_offset = reinterpret_cast<size_t>(k_data) - reinterpret_cast<size_t>(d_kv_cache);
                    if (k_offset < kv_cache_buffer_size) {
                        verify_positions.push_back({batch_id, layer_id, token_idx, k_offset, in_checked_region});
                    }
                } else {
                    // V part
                    char* v_data = block_base + k_block_size_bytes
                                   + head_id * (seq_size_per_block * v_token_bytes_per_head)
                                   + offset_in_block * v_token_bytes_per_head;
                    size_t v_offset = reinterpret_cast<size_t>(v_data) - reinterpret_cast<size_t>(d_kv_cache);
                    if (v_offset < kv_cache_buffer_size) {
                        verify_positions.push_back({batch_id, layer_id, token_idx, v_offset, in_checked_region});
                    }
                }
            }
        }

        // Only copy the specific positions we need to verify with boundary check
        std::vector<T> h_verify_values(verify_positions.size());
        for (size_t i = 0; i < verify_positions.size(); ++i) {
            if (verify_positions[i].offset_bytes >= kv_cache_buffer_size) {
                continue;
            }
            T value;
            cudaMemcpy(&value,
                       reinterpret_cast<char*>(d_kv_cache) + verify_positions[i].offset_bytes,
                       sizeof(T),
                       cudaMemcpyDeviceToHost);
            h_verify_values[i] = value;
        }
        // Verify results
        for (size_t i = 0; i < verify_positions.size(); ++i) {
            const auto& vpos  = verify_positions[i];
            T           value = h_verify_values[i];

            if (vpos.in_checked_region) {
                // Values within the checked region should be reset to 0
                ASSERT_FALSE(isInvalidValue(value)) << "NaN/Inf not reset at batch " << vpos.batch_id << ", layer "
                                                    << vpos.layer_id << ", token_idx " << vpos.token_idx;
                ASSERT_EQ(toFloat(value), 0.0f) << "NaN/Inf not reset to 0 at batch " << vpos.batch_id << ", layer "
                                                << vpos.layer_id << ", token_idx " << vpos.token_idx;
            } else {
                // Values outside the checked region should remain as NaN/Inf
                ASSERT_TRUE(isInvalidValue(value))
                    << "NaN/Inf outside checked region should remain at batch " << vpos.batch_id << ", layer_id "
                    << vpos.layer_id << ", token_id " << vpos.token_idx;
            }
        }

        // Verify nan_flag
        for (size_t i = 0; i < batch_size; ++i) {
            if (expected_has_invalid[i]) {
                ASSERT_EQ(h_nan_flag[i], 1) << "Batch " << i << " has NaN/Inf but nan_flag is " << h_nan_flag[i];
            }
        }

        // Cleanup
        cudaFree(d_kv_cache);
        cudaFree(d_layer_base_addr);
        cudaFree(d_kv_cache_block_id);
        cudaFree(d_prefix_lengths);
        cudaFree(d_seq_len_cu);
        cudaFree(d_nan_flag);
    }

    // Helper function to test KV cache NaN check for Decode
    // positions: (batch_id, layer_id, token_idx, is_inf)
    //   token_idx should be seq_len - 1 (the last token)
    //   is_inf: true for Inf, false for NaN
    template<typename T>
    void
    testKVCacheDecodeCheckAndResetNAN(size_t                      batch_size,
                                      size_t                      layer_num,
                                      size_t                      max_blocks_per_batch,
                                      size_t                      block_size_bytes,
                                      size_t                      seq_size_per_block,
                                      const std::vector<int32_t>& sequence_lengths,
                                      const std::vector<std::vector<int32_t>>&
                                          kv_cache_block_id,  // [batch_size][max_blocks_per_batch], physical block IDs
                                      const std::vector<std::tuple<size_t, size_t, int32_t, bool, bool>>&
                                          positions,  // (batch_id, layer_id, token_idx, is_inf, is_k_part)
                                                      // is_k_part: true for K cache, false for V cache
                                      bool   use_mla       = false,
                                      size_t kv_lora_rank  = 0,
                                      size_t rope_head_dim = 0) {

        size_t element_size = sizeof(T);
        size_t token_bytes  = block_size_bytes / seq_size_per_block;
        size_t token_stride = block_size_bytes / seq_size_per_block;  // Should match kernel's token_stride

        // Verify token_bytes equals token_stride (as used in kernel)
        ASSERT_EQ(token_bytes, token_stride) << "token_bytes must equal token_stride";

        // Calculate parameters for MLA vs MHA
        size_t local_head_num_kv;
        size_t k_token_size_per_head;
        size_t v_token_size_per_head;

        if (use_mla) {
            // MLA: K = kv_lora_rank, V = rope_head_dim, local_head_num_kv = 1
            local_head_num_kv     = 1;
            k_token_size_per_head = kv_lora_rank;
            v_token_size_per_head = rope_head_dim;
        } else {
            // MHA: k_token_size = v_token_size = size_per_head
            local_head_num_kv     = 1;  // Assume 1 head for test (can be adjusted if needed)
            k_token_size_per_head = token_bytes / (2 * element_size);  // k_token_size = v_token_size, so divide by 2
            v_token_size_per_head = k_token_size_per_head;
        }

        size_t k_token_bytes_per_head = k_token_size_per_head * element_size;
        size_t v_token_bytes_per_head = v_token_size_per_head * element_size;
        size_t k_block_size_bytes     = local_head_num_kv * k_token_size_per_head * seq_size_per_block * element_size;
        size_t v_block_size_bytes     = local_head_num_kv * v_token_size_per_head * seq_size_per_block * element_size;

        // Allocate a large KV cache buffer (enough for all physical blocks)
        int32_t max_physical_block_id = -1;
        for (const auto& batch_blocks : kv_cache_block_id) {
            for (int32_t block_id : batch_blocks) {
                if (block_id > max_physical_block_id) {
                    max_physical_block_id = block_id;
                }
            }
        }
        size_t num_physical_blocks  = max_physical_block_id + 1;
        size_t kv_cache_buffer_size = num_physical_blocks * block_size_bytes * layer_num;

        // Clear any previous CUDA errors
        cudaGetLastError();

        T*          d_kv_cache;
        cudaError_t err = cudaMalloc(&d_kv_cache, kv_cache_buffer_size);
        ASSERT_EQ(err, cudaSuccess) << "Failed to allocate d_kv_cache: " << cudaGetErrorString(err);

        // Initialize with normal values (1.0)
        std::vector<T> h_kv_cache_init(kv_cache_buffer_size / element_size, T(1.0f));
        cudaMemcpy(d_kv_cache, h_kv_cache_init.data(), kv_cache_buffer_size, cudaMemcpyHostToDevice);

        // Set up layer_base_addr array (one base pointer per layer)
        std::vector<void*> h_layer_base_addr(layer_num);
        for (size_t layer_id = 0; layer_id < layer_num; ++layer_id) {
            h_layer_base_addr[layer_id] =
                reinterpret_cast<char*>(d_kv_cache) + layer_id * num_physical_blocks * block_size_bytes;
        }

        // Copy layer_base_addr to device
        void** d_layer_base_addr;
        err = cudaMalloc(&d_layer_base_addr, layer_num * sizeof(void*));
        ASSERT_EQ(err, cudaSuccess) << "Failed to allocate d_layer_base_addr: " << cudaGetErrorString(err);
        cudaMemcpy(d_layer_base_addr, h_layer_base_addr.data(), layer_num * sizeof(void*), cudaMemcpyHostToDevice);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        // Set up kv_cache_block_id array
        std::vector<int32_t> h_kv_cache_block_id(batch_size * max_blocks_per_batch, -1);
        for (size_t batch_id = 0; batch_id < batch_size; ++batch_id) {
            for (size_t block_idx = 0;
                 block_idx < max_blocks_per_batch && block_idx < kv_cache_block_id[batch_id].size();
                 ++block_idx) {
                h_kv_cache_block_id[batch_id * max_blocks_per_batch + block_idx] =
                    kv_cache_block_id[batch_id][block_idx];
            }
        }
        int32_t* d_kv_cache_block_id;
        err = cudaMalloc(&d_kv_cache_block_id, batch_size * max_blocks_per_batch * sizeof(int32_t));
        ASSERT_EQ(err, cudaSuccess) << "Failed to allocate d_kv_cache_block_id: " << cudaGetErrorString(err);
        cudaMemcpy(d_kv_cache_block_id,
                   h_kv_cache_block_id.data(),
                   batch_size * max_blocks_per_batch * sizeof(int32_t),
                   cudaMemcpyHostToDevice);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        // Set up sequence_lengths
        int32_t* d_sequence_lengths;
        err = cudaMalloc(&d_sequence_lengths, batch_size * sizeof(int32_t));
        ASSERT_EQ(err, cudaSuccess) << "Failed to allocate d_sequence_lengths: " << cudaGetErrorString(err);
        cudaMemcpy(d_sequence_lengths, sequence_lengths.data(), batch_size * sizeof(int32_t), cudaMemcpyHostToDevice);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        // Allocate nan_flag
        int32_t* d_nan_flag;
        err = cudaMalloc(&d_nan_flag, batch_size * sizeof(int32_t));
        ASSERT_EQ(err, cudaSuccess) << "Failed to allocate d_nan_flag: " << cudaGetErrorString(err);
        cudaMemset(d_nan_flag, 0, batch_size * sizeof(int32_t));
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        // Collect all positions for batch insertion
        // For MHA layout: [layer_num, block_num, 2, local_head_num_kv, seq_size_per_block, k_token_size]
        // We need to insert NaN/Inf for each head's K and V parts
        std::vector<bool>   expected_has_invalid(batch_size, false);
        std::vector<size_t> token_offsets;
        std::vector<bool>   is_nan_flags;

        // Process NaN and Inf positions
        for (const auto& pos : positions) {
            size_t  batch_id  = std::get<0>(pos);
            size_t  layer_id  = std::get<1>(pos);
            int32_t token_idx = std::get<2>(pos);
            bool    is_inf    = std::get<3>(pos);
            bool    is_k_part = std::get<4>(pos);

            if (batch_id >= batch_size || layer_id >= layer_num)
                continue;

            int32_t logical_block_idx = token_idx / seq_size_per_block;
            int32_t offset_in_block   = token_idx % seq_size_per_block;

            if (logical_block_idx < 0 || logical_block_idx >= static_cast<int32_t>(max_blocks_per_batch))
                continue;

            int32_t physical_block_id = kv_cache_block_id[batch_id][logical_block_idx];
            if (physical_block_id == -1)
                continue;

            if (token_idx == sequence_lengths[batch_id] - 1) {
                expected_has_invalid[batch_id] = true;
            }

            // Calculate base offsets
            size_t layer_offset_bytes = layer_id * num_physical_blocks * block_size_bytes;
            size_t block_offset_bytes = physical_block_id * block_size_bytes;
            char*  layer_base         = reinterpret_cast<char*>(d_kv_cache) + layer_offset_bytes;
            char*  block_base         = layer_base + block_offset_bytes;

            // Insert NaN/Inf for each head's K or V part based on is_k_part
            // Layout: K parts first (all heads), then V parts (all heads)
            for (size_t head_id = 0; head_id < local_head_num_kv; head_id++) {
                if (is_k_part) {
                    // K part: block_base + head_id * (seq_size_per_block * k_token_bytes_per_head) + offset_in_block *
                    // k_token_bytes_per_head
                    char* k_data = block_base + head_id * (seq_size_per_block * k_token_bytes_per_head)
                                   + offset_in_block * k_token_bytes_per_head;
                    size_t k_offset = reinterpret_cast<size_t>(k_data) - reinterpret_cast<size_t>(d_kv_cache);

                    if (k_offset < kv_cache_buffer_size) {
                        token_offsets.push_back(k_offset);
                        is_nan_flags.push_back(!is_inf);
                    }
                } else {
                    // V part: block_base + k_block_size_bytes + head_id * (seq_size_per_block * v_token_bytes_per_head)
                    // + offset_in_block * v_token_bytes_per_head
                    char* v_data = block_base + k_block_size_bytes
                                   + head_id * (seq_size_per_block * v_token_bytes_per_head)
                                   + offset_in_block * v_token_bytes_per_head;
                    size_t v_offset = reinterpret_cast<size_t>(v_data) - reinterpret_cast<size_t>(d_kv_cache);

                    if (v_offset < kv_cache_buffer_size) {
                        token_offsets.push_back(v_offset);
                        is_nan_flags.push_back(!is_inf);
                    }
                }
            }
        }

        // Batch insert NaN/Inf values using cudaMemcpyAsync for better performance
        if (!token_offsets.empty()) {
            // Prepare host-side values
            std::vector<T> h_values(token_offsets.size());
            for (size_t i = 0; i < token_offsets.size(); ++i) {
                h_values[i] = is_nan_flags[i] ? generateNaN<T>() : generateInf<T>();
            }

            // Batch copy using cudaMemcpyAsync
            // Kernel processes entire token using uint4 vectors (16 bytes), so we need to ensure
            // at least the first uint4 vector (16 bytes) contains NaN/Inf values
            // Always write 16 bytes to ensure uint4 alignment and complete coverage
            const size_t bytes_to_write = 16;  // Always write 16 bytes for uint4
            for (size_t i = 0; i < token_offsets.size(); ++i) {
                if (token_offsets[i] >= kv_cache_buffer_size) {
                    continue;
                }
                void* dst = reinterpret_cast<char*>(d_kv_cache) + token_offsets[i];

                size_t         num_elems = (bytes_to_write + sizeof(T) - 1) / sizeof(T);
                std::vector<T> token_data(num_elems, h_values[i]);
                cudaMemcpyAsync(dst, token_data.data(), bytes_to_write, cudaMemcpyHostToDevice, stream_);
            }
            cudaStreamSynchronize(stream_);
        }

        // Create events for timing
        cudaEvent_t start, stop;
        cudaError_t event_err = cudaEventCreate(&start);
        ASSERT_EQ(event_err, cudaSuccess) << "Failed to create start event: " << cudaGetErrorString(event_err);

        event_err = cudaEventCreate(&stop);
        ASSERT_EQ(event_err, cudaSuccess) << "Failed to create stop event: " << cudaGetErrorString(event_err);

        // Run kernel
        cudaEventRecord(start, stream_);
        invokeCheckAndResetNANKvCacheDecode<T>(reinterpret_cast<const void* const*>(d_layer_base_addr),
                                               d_kv_cache_block_id,
                                               d_sequence_lengths,
                                               batch_size,
                                               layer_num,
                                               max_blocks_per_batch,
                                               local_head_num_kv,
                                               k_token_size_per_head,
                                               v_token_size_per_head,
                                               k_block_size_bytes,
                                               v_block_size_bytes,
                                               k_token_bytes_per_head,
                                               v_token_bytes_per_head,
                                               block_size_bytes,
                                               seq_size_per_block,
                                               d_nan_flag,
                                               stream_);
        cudaEventRecord(stop, stream_);
        cudaStreamSynchronize(stream_);

        // Get timing
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf(
            "Decode Kernel batch_size: [%zu], layer_num: [%zu], max_blocks_per_batch: [%zu], Execution time: %f ms\n",
            batch_size,
            layer_num,
            max_blocks_per_batch,
            milliseconds);
        // Cleanup events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // Copy nan_flag back
        std::vector<int32_t> h_nan_flag(batch_size);
        cudaMemcpy(h_nan_flag.data(), d_nan_flag, batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost);

        // Collect all positions that need verification and their offsets
        struct VerifyPosition {
            size_t  batch_id;
            size_t  layer_id;
            int32_t token_idx;
            size_t  offset_bytes;
            bool    is_last_token;
        };
        std::vector<VerifyPosition> verify_positions;

        // Collect NaN and Inf positions
        for (const auto& pos : positions) {
            size_t  batch_id  = std::get<0>(pos);
            size_t  layer_id  = std::get<1>(pos);
            int32_t token_idx = std::get<2>(pos);
            bool    is_k_part = std::get<4>(pos);

            if (batch_id >= batch_size || layer_id >= layer_num)
                continue;

            int32_t logical_block_idx = token_idx / seq_size_per_block;
            int32_t offset_in_block   = token_idx % seq_size_per_block;

            if (logical_block_idx < 0 || logical_block_idx >= static_cast<int32_t>(max_blocks_per_batch))
                continue;

            int32_t physical_block_id = kv_cache_block_id[batch_id][logical_block_idx];
            if (physical_block_id == -1)
                continue;

            bool is_last_token = (token_idx == sequence_lengths[batch_id] - 1);
            if (is_last_token) {
                expected_has_invalid[batch_id] = true;
            }

            // Calculate base offsets
            size_t layer_offset_bytes = layer_id * num_physical_blocks * block_size_bytes;
            size_t block_offset_bytes = physical_block_id * block_size_bytes;
            char*  layer_base         = reinterpret_cast<char*>(d_kv_cache) + layer_offset_bytes;
            char*  block_base         = layer_base + block_offset_bytes;

            // Verify each head's K or V part based on is_k_part
            for (size_t head_id = 0; head_id < local_head_num_kv; head_id++) {
                if (is_k_part) {
                    // K part
                    char* k_data = block_base + head_id * (seq_size_per_block * k_token_bytes_per_head)
                                   + offset_in_block * k_token_bytes_per_head;
                    size_t k_offset = reinterpret_cast<size_t>(k_data) - reinterpret_cast<size_t>(d_kv_cache);
                    if (k_offset < kv_cache_buffer_size) {
                        verify_positions.push_back({batch_id, layer_id, token_idx, k_offset, is_last_token});
                    }
                } else {
                    // V part
                    char* v_data = block_base + k_block_size_bytes
                                   + head_id * (seq_size_per_block * v_token_bytes_per_head)
                                   + offset_in_block * v_token_bytes_per_head;
                    size_t v_offset = reinterpret_cast<size_t>(v_data) - reinterpret_cast<size_t>(d_kv_cache);
                    if (v_offset < kv_cache_buffer_size) {
                        verify_positions.push_back({batch_id, layer_id, token_idx, v_offset, is_last_token});
                    }
                }
            }
        }

        // Only copy the specific positions we need to verify with boundary check
        std::vector<T> h_verify_values(verify_positions.size());
        for (size_t i = 0; i < verify_positions.size(); ++i) {
            if (verify_positions[i].offset_bytes >= kv_cache_buffer_size) {
                continue;
            }
            T value;
            cudaMemcpy(&value,
                       reinterpret_cast<char*>(d_kv_cache) + verify_positions[i].offset_bytes,
                       sizeof(T),
                       cudaMemcpyDeviceToHost);
            h_verify_values[i] = value;
        }

        // Verify results
        for (size_t i = 0; i < verify_positions.size(); ++i) {
            const auto& vpos  = verify_positions[i];
            T           value = h_verify_values[i];

            if (vpos.is_last_token) {
                ASSERT_FALSE(isInvalidValue(value)) << "NaN/Inf not reset at batch " << vpos.batch_id << ", layer "
                                                    << vpos.layer_id << ", token_idx " << vpos.token_idx;
                ASSERT_EQ(toFloat(value), 0.0f)
                    << "NaN/Inf not reset to 0 at batch " << vpos.batch_id << ", layer " << vpos.layer_id;
            } else {
                // NaN/Inf in non-last token should remain
                ASSERT_TRUE(isInvalidValue(value))
                    << "NaN/Inf in non-last token should remain at batch " << vpos.batch_id << ", layer "
                    << vpos.layer_id << ", token_idx " << vpos.token_idx;
            }
        }

        // Verify nan_flag
        for (size_t i = 0; i < batch_size; ++i) {
            if (expected_has_invalid[i]) {
                ASSERT_EQ(h_nan_flag[i], 1) << "Batch " << i << " has NaN/Inf but nan_flag is " << h_nan_flag[i];
            }
        }

        // Cleanup
        cudaFree(d_kv_cache);
        cudaFree(d_layer_base_addr);
        cudaFree(d_kv_cache_block_id);
        cudaFree(d_sequence_lengths);
        cudaFree(d_nan_flag);
    }

    cudaStream_t stream_ = nullptr;
};

// ========== Prefill Tests ==========

// Basic prefill test
TEST_F(CudaCheckAndResetNANTest, TestFloat_KVCachePrefill_Basic) {
    size_t batch_size           = 2;
    size_t layer_num            = 20;
    size_t seq_size_per_block   = 64;
    size_t max_blocks_per_batch = 1024 * 1024 / seq_size_per_block;
    size_t k_token_size         = 128 * 2;
    size_t v_token_size         = 128 * 2;
    size_t element_size         = sizeof(float);
    size_t block_size_bytes     = (k_token_size + v_token_size) * seq_size_per_block * element_size;

    std::vector<int32_t> prefix_lengths = {64, 64};
    std::vector<int32_t> seq_len_cu     = {(int32_t)max_blocks_per_batch, (int32_t)(max_blocks_per_batch / 2)};

    // Block IDs: batch 0 uses blocks [0, 1], batch 1 uses blocks [2, 3]
    std::vector<std::vector<int32_t>> kv_cache_block_id = {{}, {}};
    for (size_t i = 0; i < max_blocks_per_batch; i++) {
        kv_cache_block_id[0].push_back(i);
        if (i < max_blocks_per_batch / 2) {
            kv_cache_block_id[1].push_back(i + max_blocks_per_batch);
        }
    }

    // Insert NaN at token 10 (batch 0, in checked range [5, 20))
    // Insert NaN at token 5 (batch 1, before checked range [10, 30), should not be reset)
    std::vector<std::tuple<size_t, size_t, int32_t, bool, bool>> positions = {
        {0, 0, 100, false, true},  // batch 0, layer 0, token 100, NaN, K part
        {1, 0, 50, false, true},  // batch 1, layer 0, token 50 (before prefix_length, should not be reset), NaN, K part
        {1, 6, 100, false, true}  // batch 1, layer 6, token 100, NaN, K part
    };

    testKVCachePrefillCheckAndResetNAN<float>(batch_size,
                                              layer_num,
                                              max_blocks_per_batch,
                                              block_size_bytes,
                                              seq_size_per_block,
                                              prefix_lengths,
                                              seq_len_cu,
                                              kv_cache_block_id,
                                              positions);
}

// Prefill test with multiple layers
TEST_F(CudaCheckAndResetNANTest, TestFloat_KVCachePrefill_MultiLayer) {
    size_t batch_size           = 1;
    size_t layer_num            = 100;
    size_t seq_size_per_block   = 8;
    size_t max_blocks_per_batch = 1024 * 1024 / seq_size_per_block;
    size_t k_token_size         = 64;
    size_t v_token_size         = 64;
    size_t element_size         = sizeof(float);
    size_t block_size_bytes     = (k_token_size + v_token_size) * seq_size_per_block * element_size;

    std::vector<int32_t> prefix_lengths = {(int32_t)(1024 * 1020)};
    std::vector<int32_t> seq_len_cu     = {(int32_t)(max_blocks_per_batch * seq_size_per_block)};

    std::vector<std::vector<int32_t>> kv_cache_block_id = {std::vector<int32_t>()};
    for (size_t i = 0; i < max_blocks_per_batch; i++) {
        kv_cache_block_id[0].push_back((int32_t)i);
    }

    // Insert NaN at different layers, all in checked range
    std::vector<std::tuple<size_t, size_t, int32_t, bool, bool>> positions = {
        {0, 0, 6, false, true},                                      // layer 0, token 6, NaN, K part
        {0, 1, 8, false, true},                                      // layer 1, token 8, NaN, K part
        {0, 2, 10, false, true},                                     // layer 2, token 10, NaN, K part
        {0, 6, 1024 * 1023, false, true},                            // layer 6, token 1024*1023, NaN, K part
        {0, 6, 1024 * 1024 - 1, true, true},                         // layer 6, token 1024*1023, Inf, K part
        {0, 99, 1024 * 1024 - seq_size_per_block + 1, false, false}  // layer 100, last block, NaN, V part
    };

    testKVCachePrefillCheckAndResetNAN<float>(batch_size,
                                              layer_num,
                                              max_blocks_per_batch,
                                              block_size_bytes,
                                              seq_size_per_block,
                                              prefix_lengths,
                                              seq_len_cu,
                                              kv_cache_block_id,
                                              positions);
}

// Prefill test: no new tokens
TEST_F(CudaCheckAndResetNANTest, TestFloat_KVCachePrefill_NoNewTokens) {
    size_t batch_size           = 1;
    size_t layer_num            = 100;
    size_t max_blocks_per_batch = 2;
    size_t seq_size_per_block   = 16;
    size_t k_token_size         = 128;
    size_t v_token_size         = 128;
    size_t element_size         = sizeof(float);
    size_t block_size_bytes     = (k_token_size + v_token_size) * seq_size_per_block * element_size;

    std::vector<int32_t> prefix_lengths = {20};
    std::vector<int32_t> seq_len_cu     = {20};  // No new tokens

    std::vector<std::vector<int32_t>> kv_cache_block_id = {{0, 1}};

    // Insert NaN, but it should not be checked (no new tokens)
    std::vector<std::tuple<size_t, size_t, int32_t, bool, bool>> positions = {
        {0, 0, 15, false, true},  // token 15, in range but prefix_length == seq_len, NaN, K part
    };

    testKVCachePrefillCheckAndResetNAN<float>(batch_size,
                                              layer_num,
                                              max_blocks_per_batch,
                                              block_size_bytes,
                                              seq_size_per_block,
                                              prefix_lengths,
                                              seq_len_cu,
                                              kv_cache_block_id,
                                              positions);
}

// Prefill test: cross block boundary
TEST_F(CudaCheckAndResetNANTest, TestFloat_KVCachePrefill_CrossBlockBoundary) {
    size_t batch_size           = 1;
    size_t layer_num            = 100;
    size_t max_blocks_per_batch = 3;
    size_t seq_size_per_block   = 8;
    size_t k_token_size         = 64;
    size_t v_token_size         = 64;
    size_t element_size         = sizeof(float);
    size_t block_size_bytes     = (k_token_size + v_token_size) * seq_size_per_block * element_size;

    std::vector<int32_t> prefix_lengths = {6};
    std::vector<int32_t> seq_len_cu     = {18};  // Spans blocks 0, 1, 2

    std::vector<std::vector<int32_t>> kv_cache_block_id = {{0, 1, 2}};

    // Insert NaN at tokens in different blocks
    std::vector<std::tuple<size_t, size_t, int32_t, bool, bool>> positions = {
        {0, 0, 7, false, true},   // block 0, last token, NaN, K part
        {0, 0, 8, false, true},   // block 1, first token, NaN, K part
        {0, 0, 15, false, true},  // block 1, last token, NaN, K part
        {0, 0, 16, false, true},  // block 2, first token, NaN, K part
    };

    testKVCachePrefillCheckAndResetNAN<float>(batch_size,
                                              layer_num,
                                              max_blocks_per_batch,
                                              block_size_bytes,
                                              seq_size_per_block,
                                              prefix_lengths,
                                              seq_len_cu,
                                              kv_cache_block_id,
                                              positions);
}

// Prefill test: null block ID
TEST_F(CudaCheckAndResetNANTest, TestFloat_KVCachePrefill_NullBlockId) {
    size_t batch_size           = 3;
    size_t layer_num            = 100;
    size_t max_blocks_per_batch = 3;
    size_t seq_size_per_block   = 8;
    size_t k_token_size         = 64;
    size_t v_token_size         = 64;
    size_t element_size         = sizeof(float);
    size_t block_size_bytes     = (k_token_size + v_token_size) * seq_size_per_block * element_size;

    std::vector<int32_t> prefix_lengths = {0, 0, 0};
    std::vector<int32_t> seq_len_cu     = {16, 16, 16};

    std::vector<std::vector<int32_t>> kv_cache_block_id = {{0, -1, 2},  // block 1 is null
                                                           {1, 4, 5},
                                                           {2, 7, -1}};

    // Insert NaN at token in null block (should be skipped)
    std::vector<std::tuple<size_t, size_t, int32_t, bool, bool>> positions = {
        {0, 0, 8, false, true},   // token 8 is in block 1 (null), should be skipped, NaN, K part
        {0, 0, 15, false, true},  // token 15 is in block 2, should be checked, NaN, K part
        {1, 0, 8, false, true}    // token 8 is in block 4, should be checked, NaN, K part
    };

    testKVCachePrefillCheckAndResetNAN<float>(batch_size,
                                              layer_num,
                                              max_blocks_per_batch,
                                              block_size_bytes,
                                              seq_size_per_block,
                                              prefix_lengths,
                                              seq_len_cu,
                                              kv_cache_block_id,
                                              positions);
}

// Prefill test: half precision
TEST_F(CudaCheckAndResetNANTest, TestHalf_KVCachePrefill_Basic) {
    size_t batch_size           = 1;
    size_t layer_num            = 100;
    size_t max_blocks_per_batch = 2;
    size_t seq_size_per_block   = 16;
    size_t k_token_size         = 128;
    size_t v_token_size         = 128;
    size_t element_size         = sizeof(half);
    size_t block_size_bytes     = (k_token_size + v_token_size) * seq_size_per_block * element_size;

    std::vector<int32_t> prefix_lengths = {5};
    std::vector<int32_t> seq_len_cu     = {20};

    std::vector<std::vector<int32_t>> kv_cache_block_id = {{0, 1}};

    std::vector<std::tuple<size_t, size_t, int32_t, bool, bool>> positions = {
        {0, 0, 10, false, true},
        {0, 0, 15, false, true},
    };

    testKVCachePrefillCheckAndResetNAN<half>(batch_size,
                                             layer_num,
                                             max_blocks_per_batch,
                                             block_size_bytes,
                                             seq_size_per_block,
                                             prefix_lengths,
                                             seq_len_cu,
                                             kv_cache_block_id,
                                             positions);
}

#ifdef ENABLE_BF16
// Prefill test: bfloat16
TEST_F(CudaCheckAndResetNANTest, TestBFloat16_KVCachePrefill_Basic) {
    size_t batch_size           = 1;
    size_t layer_num            = 100;
    size_t max_blocks_per_batch = 2;
    size_t seq_size_per_block   = 16;
    size_t k_token_size         = 128;
    size_t v_token_size         = 128;
    size_t element_size         = sizeof(__nv_bfloat16);
    size_t block_size_bytes     = (k_token_size + v_token_size) * seq_size_per_block * element_size;

    std::vector<int32_t> prefix_lengths = {5};
    std::vector<int32_t> seq_len_cu     = {20};

    std::vector<std::vector<int32_t>> kv_cache_block_id = {{0, 1}};

    std::vector<std::tuple<size_t, size_t, int32_t, bool, bool>> positions = {
        {0, 0, 10, false, true},
    };

    testKVCachePrefillCheckAndResetNAN<__nv_bfloat16>(batch_size,
                                                      layer_num,
                                                      max_blocks_per_batch,
                                                      block_size_bytes,
                                                      seq_size_per_block,
                                                      prefix_lengths,
                                                      seq_len_cu,
                                                      kv_cache_block_id,
                                                      positions);
}
#endif

#ifdef ENABLE_FP8
// Prefill test: FP8
TEST_F(CudaCheckAndResetNANTest, TestFP8_KVCachePrefill_Basic) {
    size_t batch_size           = 2;
    size_t layer_num            = 50;
    size_t seq_size_per_block   = 64;
    size_t max_blocks_per_batch = 1024 * 1024 / seq_size_per_block;
    size_t k_token_size         = 128 * 2;
    size_t v_token_size         = 128 * 2;
    size_t element_size         = sizeof(__nv_fp8_e4m3);
    size_t block_size_bytes     = (k_token_size + v_token_size) * seq_size_per_block * element_size;

    std::vector<int32_t> prefix_lengths = {64, 64};
    std::vector<int32_t> seq_len_cu     = {(int32_t)max_blocks_per_batch, (int32_t)(max_blocks_per_batch / 2)};

    // Block IDs: batch 0 uses blocks [0, 1], batch 1 uses blocks [2, 3]
    std::vector<std::vector<int32_t>> kv_cache_block_id = {{}, {}};
    for (size_t i = 0; i < max_blocks_per_batch; i++) {
        kv_cache_block_id[0].push_back(i);
        if (i < max_blocks_per_batch / 2) {
            kv_cache_block_id[1].push_back(i + max_blocks_per_batch);
        }
    }

    // Insert NaN at token 10 (batch 0, in checked range [5, 20))
    // Insert NaN at token 5 (batch 1, before checked range [10, 30), should not be reset)
    std::vector<std::tuple<size_t, size_t, int32_t, bool, bool>> positions = {
        {0, 0, 100, false, true},  // batch 0, layer 0, token 100, NaN, K part
        {1, 0, 50, false, true},  // batch 1, layer 0, token 50 (before prefix_length, should not be reset), NaN, K part
        {1, 6, 100, false, true}  // batch 1, layer 6, token 100, NaN, K part
    };

    testKVCachePrefillCheckAndResetNAN<__nv_fp8_e4m3>(batch_size,
                                                      layer_num,
                                                      max_blocks_per_batch,
                                                      block_size_bytes,
                                                      seq_size_per_block,
                                                      prefix_lengths,
                                                      seq_len_cu,
                                                      kv_cache_block_id,
                                                      positions);
}
#endif

// ========== Decode Tests ==========

// Basic decode test
TEST_F(CudaCheckAndResetNANTest, TestFloat_KVCacheDecode_Basic) {
    size_t batch_size           = 2;
    size_t layer_num            = 100;
    size_t max_blocks_per_batch = 3;
    size_t seq_size_per_block   = 16;
    size_t k_token_size         = 128;
    size_t v_token_size         = 128;
    size_t element_size         = sizeof(float);
    size_t block_size_bytes     = (k_token_size + v_token_size) * seq_size_per_block * element_size;

    std::vector<int32_t> sequence_lengths = {25, 35};  // Last tokens: 24, 34

    std::vector<std::vector<int32_t>> kv_cache_block_id = {
        {0, 1, 2},  // batch 0
        {3, 4, 5}   // batch 1
    };

    // Insert NaN at last token (should be reset) and non-last token (should remain)
    std::vector<std::tuple<size_t, size_t, int32_t, bool, bool>> positions = {
        {0, 0, 24, false, true},  // batch 0, last token, should be reset, NaN, K part
        {0, 0, 20, false, true},  // batch 0, non-last token, should remain, NaN, K part
        {1, 0, 34, false, true},  // batch 1, last token, should be reset, NaN, K part
    };

    testKVCacheDecodeCheckAndResetNAN<float>(batch_size,
                                             layer_num,
                                             max_blocks_per_batch,
                                             block_size_bytes,
                                             seq_size_per_block,
                                             sequence_lengths,
                                             kv_cache_block_id,
                                             positions);
}

// Decode test: multiple layers
TEST_F(CudaCheckAndResetNANTest, TestFloat_KVCacheDecode_MultiLayer) {
    size_t batch_size           = 1;
    size_t layer_num            = 3;
    size_t max_blocks_per_batch = 2;
    size_t seq_size_per_block   = 8;
    size_t k_token_size         = 64;
    size_t v_token_size         = 64;
    size_t element_size         = sizeof(float);
    size_t block_size_bytes     = (k_token_size + v_token_size) * seq_size_per_block * element_size;

    std::vector<int32_t> sequence_lengths = {15};  // Last token: 14

    std::vector<std::vector<int32_t>> kv_cache_block_id = {{0, 1}};

    // Insert NaN at last token in all layers
    std::vector<std::tuple<size_t, size_t, int32_t, bool, bool>> positions = {
        {0, 0, 14, false, true},  // layer 0, last token, NaN, K part
        {0, 1, 14, false, true},  // layer 1, last token, NaN, K part
        {0, 2, 14, false, true},  // layer 2, last token, NaN, K part
    };

    testKVCacheDecodeCheckAndResetNAN<float>(batch_size,
                                             layer_num,
                                             max_blocks_per_batch,
                                             block_size_bytes,
                                             seq_size_per_block,
                                             sequence_lengths,
                                             kv_cache_block_id,
                                             positions);
}

// Decode test: seq_len == 0
TEST_F(CudaCheckAndResetNANTest, TestFloat_KVCacheDecode_SeqLenZero) {
    size_t batch_size           = 1;
    size_t layer_num            = 100;
    size_t max_blocks_per_batch = 2;
    size_t seq_size_per_block   = 16;
    size_t k_token_size         = 128;
    size_t v_token_size         = 128;
    size_t element_size         = sizeof(float);
    size_t block_size_bytes     = (k_token_size + v_token_size) * seq_size_per_block * element_size;

    std::vector<int32_t> sequence_lengths = {0};  // No tokens

    std::vector<std::vector<int32_t>> kv_cache_block_id = {{0, 1}};

    // Insert NaN, but seq_len == 0, so nothing should be checked
    std::vector<std::tuple<size_t, size_t, int32_t, bool, bool>> positions = {
        {0, 0, 0, false, true},  // Should be skipped, NaN, K part
    };

    testKVCacheDecodeCheckAndResetNAN<float>(batch_size,
                                             layer_num,
                                             max_blocks_per_batch,
                                             block_size_bytes,
                                             seq_size_per_block,
                                             sequence_lengths,
                                             kv_cache_block_id,
                                             positions);
}

// Decode test: seq_len == 1
TEST_F(CudaCheckAndResetNANTest, TestFloat_KVCacheDecode_SeqLenOne) {
    size_t batch_size           = 1;
    size_t layer_num            = 100;
    size_t max_blocks_per_batch = 1;
    size_t seq_size_per_block   = 16;
    size_t k_token_size         = 128;
    size_t v_token_size         = 128;
    size_t element_size         = sizeof(float);
    size_t block_size_bytes     = (k_token_size + v_token_size) * seq_size_per_block * element_size;

    std::vector<int32_t> sequence_lengths = {1};  // Last token: 0

    std::vector<std::vector<int32_t>> kv_cache_block_id = {{0}};

    // Insert NaN at token 0 (last token)
    std::vector<std::tuple<size_t, size_t, int32_t, bool, bool>> positions = {
        {0, 0, 0, false, true},  // Should be reset, NaN, K part
    };

    testKVCacheDecodeCheckAndResetNAN<float>(batch_size,
                                             layer_num,
                                             max_blocks_per_batch,
                                             block_size_bytes,
                                             seq_size_per_block,
                                             sequence_lengths,
                                             kv_cache_block_id,
                                             positions);
}

// Decode test: last token in new block
TEST_F(CudaCheckAndResetNANTest, TestFloat_KVCacheDecode_LastTokenInNewBlock) {
    size_t batch_size           = 1;
    size_t layer_num            = 50;
    size_t max_blocks_per_batch = 3;
    size_t seq_size_per_block   = 8;
    size_t k_token_size         = 64;
    size_t v_token_size         = 64;
    size_t element_size         = sizeof(float);
    size_t block_size_bytes     = (k_token_size + v_token_size) * seq_size_per_block * element_size;

    std::vector<int32_t> sequence_lengths = {16};  // Last token: 15 (start of block 2)

    std::vector<std::vector<int32_t>> kv_cache_block_id = {{0, 1, 2}};

    // Insert NaN at token 15 (first token of block 2, which is the last token)
    std::vector<std::tuple<size_t, size_t, int32_t, bool, bool>> positions = {
        {0, 0, 15, false, true},  // Should be reset, NaN, K part
    };

    testKVCacheDecodeCheckAndResetNAN<float>(batch_size,
                                             layer_num,
                                             max_blocks_per_batch,
                                             block_size_bytes,
                                             seq_size_per_block,
                                             sequence_lengths,
                                             kv_cache_block_id,
                                             positions);
}

// Decode test: null block ID
TEST_F(CudaCheckAndResetNANTest, TestFloat_KVCacheDecode_NullBlockId) {
    size_t batch_size           = 1;
    size_t layer_num            = 100;
    size_t max_blocks_per_batch = 3;
    size_t seq_size_per_block   = 8;
    size_t k_token_size         = 64;
    size_t v_token_size         = 64;
    size_t element_size         = sizeof(float);
    size_t block_size_bytes     = (k_token_size + v_token_size) * seq_size_per_block * element_size;

    std::vector<int32_t> sequence_lengths = {17};  // Last token: 16 (in block 2)

    std::vector<std::vector<int32_t>> kv_cache_block_id = {
        {0, -1, 2}  // block 1 is null
    };

    // Insert NaN at token 16 (in block 2, which is valid)
    std::vector<std::tuple<size_t, size_t, int32_t, bool, bool>> positions = {
        {0, 0, 16, false, true},  // Should be reset, NaN, K part
    };

    testKVCacheDecodeCheckAndResetNAN<float>(batch_size,
                                             layer_num,
                                             max_blocks_per_batch,
                                             block_size_bytes,
                                             seq_size_per_block,
                                             sequence_lengths,
                                             kv_cache_block_id,
                                             positions);
}

// Decode test: half precision
TEST_F(CudaCheckAndResetNANTest, TestHalf_KVCacheDecode_Basic) {
    size_t batch_size           = 1;
    size_t layer_num            = 100;
    size_t max_blocks_per_batch = 2;
    size_t seq_size_per_block   = 16;
    size_t k_token_size         = 128;
    size_t v_token_size         = 128;
    size_t element_size         = sizeof(half);
    size_t block_size_bytes     = (k_token_size + v_token_size) * seq_size_per_block * element_size;

    std::vector<int32_t> sequence_lengths = {20};

    std::vector<std::vector<int32_t>> kv_cache_block_id = {{0, 1}};

    std::vector<std::tuple<size_t, size_t, int32_t, bool, bool>> positions = {
        {0, 0, 13, true, true}, {0, 0, 19, false, true},  // Last token, Inf/NaN, K part
    };

    testKVCacheDecodeCheckAndResetNAN<half>(batch_size,
                                            layer_num,
                                            max_blocks_per_batch,
                                            block_size_bytes,
                                            seq_size_per_block,
                                            sequence_lengths,
                                            kv_cache_block_id,
                                            positions);
}

#ifdef ENABLE_BF16
// Decode test: bfloat16
TEST_F(CudaCheckAndResetNANTest, TestBFloat16_KVCacheDecode_Basic) {
    size_t batch_size           = 1024;
    size_t layer_num            = 100;
    size_t max_blocks_per_batch = 2;
    size_t seq_size_per_block   = 16;
    size_t k_token_size         = 128;
    size_t v_token_size         = 128;
    size_t element_size         = sizeof(__nv_bfloat16);
    size_t block_size_bytes     = (k_token_size + v_token_size) * seq_size_per_block * element_size;

    std::vector<int32_t> sequence_lengths(1024, 20);

    std::vector<std::vector<int32_t>> kv_cache_block_id(1024, std::vector<int32_t>{0, 1});

    std::vector<std::tuple<size_t, size_t, int32_t, bool, bool>> positions(
        1024, std::make_tuple(static_cast<size_t>(0), static_cast<size_t>(0), static_cast<int32_t>(19), false, true));

    testKVCacheDecodeCheckAndResetNAN<__nv_bfloat16>(batch_size,
                                                     layer_num,
                                                     max_blocks_per_batch,
                                                     block_size_bytes,
                                                     seq_size_per_block,
                                                     sequence_lengths,
                                                     kv_cache_block_id,
                                                     positions);
}
#endif

#ifdef ENABLE_FP8
// Decode test: FP8
TEST_F(CudaCheckAndResetNANTest, TestFP8_KVCacheDecode_Basic) {
    size_t batch_size           = 1;
    size_t layer_num            = 100;
    size_t max_blocks_per_batch = 2;
    size_t seq_size_per_block   = 64;
    size_t k_token_size         = 128 * 4;
    size_t v_token_size         = 128 * 4;
    size_t element_size         = sizeof(__nv_fp8_e4m3);
    size_t block_size_bytes     = (k_token_size + v_token_size) * seq_size_per_block * element_size;

    std::vector<int32_t> sequence_lengths = {20};

    std::vector<std::vector<int32_t>> kv_cache_block_id = {{0, 1}};

    std::vector<std::tuple<size_t, size_t, int32_t, bool, bool>> positions = {
        {0, 0, 19, false, true},
        {0, 0, 19, true, true},
    };

    testKVCacheDecodeCheckAndResetNAN<__nv_fp8_e4m3>(batch_size,
                                                     layer_num,
                                                     max_blocks_per_batch,
                                                     block_size_bytes,
                                                     seq_size_per_block,
                                                     sequence_lengths,
                                                     kv_cache_block_id,
                                                     positions);
}
#endif

// MLA (Multi-head Latent Attention) tests
TEST_F(CudaCheckAndResetNANTest, TestFloat_MLA_KVCachePrefill_Basic) {
    size_t batch_size           = 2;
    size_t layer_num            = 8;
    size_t max_blocks_per_batch = 4;
    size_t seq_size_per_block   = 16;

    // MLA parameters
    size_t kv_lora_rank     = 64;   // K token size
    size_t rope_head_dim    = 128;  // V token size
    size_t element_size     = sizeof(float);
    size_t block_size_bytes = (kv_lora_rank + rope_head_dim) * seq_size_per_block * element_size;

    std::vector<int32_t> prefix_lengths = {5, 10};
    std::vector<int32_t> seq_len_cu     = {20, 30};

    std::vector<std::vector<int32_t>> kv_cache_block_id(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < max_blocks_per_batch; j++) {
            kv_cache_block_id[i].push_back(i * max_blocks_per_batch + j);
        }
    }

    // Insert NaN in K part and V part separately
    std::vector<std::tuple<size_t, size_t, int32_t, bool, bool>> positions = {
        {0, 0, 10, false, true},   // batch 0, layer 0, token 10, NaN, K part
        {0, 0, 15, false, false},  // batch 0, layer 0, token 15, NaN, V part
        {1, 0, 20, false, true},   // batch 1, layer 0, token 20, NaN, K part
        {1, 3, 25, false, false},  // batch 1, layer 3, token 25, NaN, V part
    };

    testKVCachePrefillCheckAndResetNAN<float>(batch_size,
                                              layer_num,
                                              max_blocks_per_batch,
                                              block_size_bytes,
                                              seq_size_per_block,
                                              prefix_lengths,
                                              seq_len_cu,
                                              kv_cache_block_id,
                                              positions,
                                              true,  // use_mla
                                              kv_lora_rank,
                                              rope_head_dim);
}

TEST_F(CudaCheckAndResetNANTest, TestFloat_MLA_KVCacheDecode_Basic) {
    size_t batch_size           = 2;
    size_t layer_num            = 8;
    size_t max_blocks_per_batch = 4;
    size_t seq_size_per_block   = 16;

    // MLA parameters
    size_t kv_lora_rank     = 64;   // K token size
    size_t rope_head_dim    = 128;  // V token size
    size_t element_size     = sizeof(float);
    size_t block_size_bytes = (kv_lora_rank + rope_head_dim) * seq_size_per_block * element_size;

    std::vector<int32_t> sequence_lengths = {25, 35};

    std::vector<std::vector<int32_t>> kv_cache_block_id(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < max_blocks_per_batch; j++) {
            kv_cache_block_id[i].push_back(i * max_blocks_per_batch + j);
        }
    }

    // Insert NaN in K part and V part separately (last tokens)
    std::vector<std::tuple<size_t, size_t, int32_t, bool, bool>> positions = {
        {0, 0, 24, false, true},   // batch 0, layer 0, last token, NaN, K part
        {0, 1, 24, false, false},  // batch 0, layer 1, last token, NaN, V part
        {1, 0, 34, false, true},   // batch 1, layer 0, last token, NaN, K part
        {1, 2, 34, false, false},  // batch 1, layer 2, last token, NaN, V part
    };

    testKVCacheDecodeCheckAndResetNAN<float>(batch_size,
                                             layer_num,
                                             max_blocks_per_batch,
                                             block_size_bytes,
                                             seq_size_per_block,
                                             sequence_lengths,
                                             kv_cache_block_id,
                                             positions,
                                             true,  // use_mla
                                             kv_lora_rank,
                                             rope_head_dim);
}

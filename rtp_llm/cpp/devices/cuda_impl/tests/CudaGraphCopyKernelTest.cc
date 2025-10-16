#include "ATen/core/TensorBody.h"
#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <random>
#include <typeinfo>
#include <iomanip>
#include <cstring>
#include <string>
#include <torch/torch.h>
#include "rtp_llm/cpp/kernels/cuda_graph_copy_kernel.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/devices/utils/DevicePerfWrapper.h"
namespace rtp_llm {

// Helper function to check CUDA errors and throw exceptions
void checkCudaError(cudaError_t error, const std::string& operation) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in " << operation << ": " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("CUDA operation failed: " + operation);
    }
}

// Helper function to convert different types to float for comparison
template<typename T>
float toFloat(T value) {
    return static_cast<float>(value);
}

template<>
float toFloat<half>(half value) {
    return __half2float(value);
}

#ifdef ENABLE_BF16
template<>
float toFloat<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}
#endif

// Helper function to compare tensors with tolerance
template<typename T>
void assertTensorEqual(const std::vector<T>& actual,
                       const std::vector<T>& expected,
                       const std::string&    test_name,
                       float                 tolerance = 1e-5f) {
    ASSERT_EQ(actual.size(), expected.size())
        << test_name << ": Size mismatch - actual: " << actual.size() << ", expected: " << expected.size();

    for (size_t i = 0; i < actual.size(); ++i) {
        float actual_val   = toFloat(actual[i]);
        float expected_val = toFloat(expected[i]);
        ASSERT_NEAR(actual_val, expected_val, tolerance)
            << test_name << ": Value mismatch at index " << i << " - actual: " << actual_val
            << ", expected: " << expected_val;
    }
}

// Helper function to create torch tensor from memory (in-place)
template<typename T>
torch::Tensor createTensorFromMemory(T* data_ptr, const std::vector<int64_t>& shape, bool is_cuda = true) {
    auto options = torch::TensorOptions()
                       .dtype(std::is_same_v<T, float> ? torch::kFloat32 :
                              std::is_same_v<T, half>  ? torch::kFloat16 :
                                                         torch::kBFloat16)
                       .device(is_cuda ? torch::kCUDA : torch::kCPU)
                       .requires_grad(false);

    return torch::from_blob(data_ptr, shape, options);
}

class CudaGraphCopyKernelTest: public DeviceTestBase {
protected:
    void SetUp() override {
        DeviceTestBase::SetUp();
        // Initialize CUDA stream
        cudaStreamCreate(&stream_);
    }

    void TearDown() override {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
        DeviceTestBase::TearDown();
    }

    template<typename T>
    void testCudaGraphCopySmall2Large();

    template<typename T>
    void testCudaGraphCopyLarge2Small();

    template<typename T>
    void testCudaGraphCopyRoundTrip();

    cudaStream_t stream_ = nullptr;
};

template<typename T>
void CudaGraphCopyKernelTest::testCudaGraphCopySmall2Large() {
    // int       batch_size     = 3;
    // const int max_seq_len    = 5;
    // const int hidden_size    = 4;
    // const int max_batch_size = 4;
    int       batch_size     = 64;
    const int max_seq_len    = 512;
    const int hidden_size    = 768;
    const int max_batch_size = 64;

    // Input lengths for each batch
    // std::vector<int> input_lengths = {3, 2, 4};
    std::vector<int> input_lengths = {114, 181, 148, 117, 132, 127, 134, 84,  121, 107, 151, 191, 107, 175, 172, 107,
                                      103, 82,  123, 109, 128, 115, 153, 128, 122, 164, 165, 158, 100, 142, 144, 155,
                                      97,  100, 191, 183, 136, 89,  136, 149, 104, 130, 162, 102, 191, 98,  111, 115,
                                      96,  151, 100, 95,  96,  108, 97,  134, 159, 86,  108, 99,  102, 75,  99,  125};
    for (int i = 0; i < input_lengths.size(); i++) {
        input_lengths[i] += 13;
    }
    // Calculate total compact size
    int total_compact_size = 0;
    int total_token_sum    = 0;
    for (int i = 0; i < batch_size; ++i) {
        total_compact_size += input_lengths[i] * hidden_size;
        total_token_sum += input_lengths[i];
    }

    // Allocate host memory
    std::vector<T> h_input_compact(total_compact_size);
    std::vector<T> h_output_aligned(max_batch_size * max_seq_len * hidden_size, T(0));
    std::vector<T> h_expected(max_batch_size * max_seq_len * hidden_size, T(0));

    // Transform host tensor
    torch::Tensor h_input_compact_tensor =
        createTensorFromMemory(h_input_compact.data(), {total_token_sum, hidden_size}, false);
    torch::Tensor h_output_aligned_tensor =
        createTensorFromMemory(h_output_aligned.data(), {max_batch_size * max_seq_len, hidden_size}, false);
    torch::Tensor h_expected_tensor =
        createTensorFromMemory(h_expected.data(), {max_batch_size * max_seq_len, hidden_size}, false);
    // Initialize input data
    std::random_device                    rd;
    std::mt19937                          gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < total_compact_size; ++i) {
        h_input_compact[i] = T(dis(gen));
    }
    std::cout << "================== testCudaGraphCopySmall2Large<" << typeid(T).name()
              << "> ==================" << std::endl;
    // std::cout << "h_expected_tensor before: " << std::endl << h_expected_tensor << std::endl;
    // Calculate expected output manually
    int compact_idx = 0;
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < input_lengths[b]; ++s) {
            for (int h = 0; h < hidden_size; ++h) {
                int aligned_idx         = b * max_seq_len * hidden_size + s * hidden_size + h;
                h_expected[aligned_idx] = h_input_compact[compact_idx];
                compact_idx++;
            }
        }
    }
    // std::cout << "h_input_compact_tensor: " << std::endl << h_input_compact_tensor << std::endl;
    // std::cout << "h_expected_tensor: " << std::endl << h_expected_tensor << std::endl;
    // Calculate cu_seq_len (cumulative lengths)
    std::vector<int> cu_seq_len(batch_size + 1);
    cu_seq_len[0] = 0;
    for (int i = 0; i < batch_size; ++i) {
        cu_seq_len[i + 1] = cu_seq_len[i] + input_lengths[i];
    }

    // Allocate device memory
    T*   d_input_compact;
    T*   d_output_aligned;
    int* d_input_lengths;
    int* d_cu_seq_len;  // Pinned memory for kernel access
    int* d_batch_size;  // Pinned memory for batch_size

    cudaMalloc(&d_input_compact, total_compact_size * sizeof(T));
    cudaMalloc(&d_output_aligned, batch_size * max_seq_len * hidden_size * sizeof(T));
    cudaMalloc(&d_input_lengths, batch_size * sizeof(int));

    // Use device allocateBuffer to create pinned memory (like CudaGraphRunner)
    auto cu_seq_len_buffer = device_->allocateBuffer(
        {rtp_llm::DataType::TYPE_INT32, {size_t(batch_size + 1)}, rtp_llm::AllocationType::HOST}, {});
    d_cu_seq_len = static_cast<int*>(cu_seq_len_buffer->data());

    auto batch_size_buffer =
        device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {1}, rtp_llm::AllocationType::HOST}, {});
    d_batch_size = static_cast<int*>(batch_size_buffer->data());

    // Copy data to device
    cudaMemcpy(d_input_compact, h_input_compact.data(), total_compact_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_lengths, input_lengths.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    // Copy data to pinned memory (host accessible)
    memcpy(d_cu_seq_len, cu_seq_len.data(), (batch_size + 1) * sizeof(int));
    memcpy(d_batch_size, &batch_size, sizeof(int));
    memcpy(d_batch_size, &batch_size, sizeof(int));

    // Initialize output to zero
    cudaMemset(d_output_aligned, 0, batch_size * max_seq_len * hidden_size * sizeof(T));
    // warm up once
    invokeCudaGraphCopySmall2Large(d_input_compact,
                                   d_output_aligned,
                                   d_batch_size,
                                   max_batch_size,
                                   max_seq_len,
                                   d_input_lengths,
                                   hidden_size,
                                   d_cu_seq_len,
                                   stream_);
    // Create events for timing
    cudaEvent_t launch_start, launch_stop, exec_start, exec_stop;
    cudaEventCreate(&launch_start);
    cudaEventCreate(&launch_stop);
    cudaEventCreate(&exec_start);
    cudaEventCreate(&exec_stop);
    // Debug: Print parameters before kernel call
    std::cout << "Before kernel call:" << std::endl;
    std::cout << "  d_input_compact: " << d_input_compact << std::endl;
    std::cout << "  d_output_aligned: " << d_output_aligned << std::endl;
    std::cout << "  batch_size: " << batch_size << std::endl;
    std::cout << "  max_batch_size: " << max_batch_size << std::endl;
    std::cout << "  max_seq_len: " << max_seq_len << std::endl;
    std::cout << "  d_input_lengths: " << d_input_lengths << std::endl;
    std::cout << "  hidden_size: " << hidden_size << std::endl;
    std::cout << "  d_cu_seq_len: " << d_cu_seq_len << std::endl;
    std::cout << "  stream: " << stream_ << std::endl;
    // Measure launch time
    cudaEventRecord(launch_start, stream_);
    invokeCudaGraphCopySmall2Large(d_input_compact,
                                   d_output_aligned,
                                   d_batch_size,
                                   max_batch_size,
                                   max_seq_len,
                                   d_input_lengths,
                                   hidden_size,
                                   d_cu_seq_len,
                                   stream_);
    cudaEventRecord(launch_stop, stream_);
    // Check for CUDA errors after kernel call
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error after kernel call: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "Kernel call completed without CUDA errors" << std::endl;
    }

    // Measure execution time (after launch completes)
    cudaEventRecord(exec_start, stream_);
    cudaStreamSynchronize(stream_);
    cudaEventRecord(exec_stop, stream_);

    // Check for CUDA errors after sync
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error after sync: " << cudaGetErrorString(err) << std::endl;
    }

    // Get timing
    float launch_time = 0, exec_time = 0;
    cudaEventElapsedTime(&launch_time, launch_start, launch_stop);
    cudaEventElapsedTime(&exec_time, exec_start, exec_stop);
    float total_time = exec_time + launch_time;

    // Copy result back
    cudaMemcpy(h_output_aligned.data(),
               d_output_aligned,
               batch_size * max_seq_len * hidden_size * sizeof(T),
               cudaMemcpyDeviceToHost);
    // std::cout << "h_output_aligned_tensor: " << std::endl << h_output_aligned_tensor << std::endl;
    // Verify results
    assertTensorEqual(h_output_aligned, h_expected, "Small2Large copy", 1e-5f);

    std::cout << "Small2Large copy - Launch: " << std::fixed << std::setprecision(6) << launch_time
              << " ms, Execution: " << exec_time << " ms, Total: " << total_time << " ms" << std::endl;

    // Cleanup
    cudaEventDestroy(launch_start);
    cudaEventDestroy(launch_stop);
    cudaEventDestroy(exec_start);
    cudaEventDestroy(exec_stop);
    cudaFree(d_input_compact);
    cudaFree(d_output_aligned);
    cudaFree(d_input_lengths);
    // cu_seq_len_buffer will be automatically freed by RAII
}

template<typename T>
void CudaGraphCopyKernelTest::testCudaGraphCopyLarge2Small() {
    std::cout << "================== testCudaGraphCopyLarge2Small<" << typeid(T).name()
              << "> ==================" << std::endl;
    // int       batch_size     = 3;
    // const int max_seq_len    = 5;
    // const int hidden_size    = 4;
    // const int max_batch_size = 4;

    // // Input lengths for each batch
    // std::vector<int> input_lengths = {3, 2, 4};

    int       batch_size     = 64;
    const int max_seq_len    = 512;
    const int hidden_size    = 768;
    const int max_batch_size = 64;

    // Input lengths for each batch
    // std::vector<int> input_lengths = {3, 2, 4};
    std::vector<int> input_lengths = {114, 181, 148, 117, 132, 127, 134, 84,  121, 107, 151, 191, 107, 175, 172, 107,
                                      103, 82,  123, 109, 128, 115, 153, 128, 122, 164, 165, 158, 100, 142, 144, 155,
                                      97,  100, 191, 183, 136, 89,  136, 149, 104, 130, 162, 102, 191, 98,  111, 115,
                                      96,  151, 100, 95,  96,  108, 97,  134, 159, 86,  108, 99,  102, 75,  99,  125};
    for (int i = 0; i < input_lengths.size(); i++) {
        input_lengths[i] += 13;
    }

    // Calculate total compact size
    int total_compact_size = 0;
    int total_token_sum    = 0;
    for (int i = 0; i < batch_size; ++i) {
        total_compact_size += input_lengths[i] * hidden_size;
        total_token_sum += input_lengths[i];
    }

    // Allocate host memory
    std::vector<T> h_input_aligned(max_batch_size * max_seq_len * hidden_size);
    std::vector<T> h_output_compact(total_compact_size, T(0));
    std::vector<T> h_expected(total_compact_size, T(0));

    // Transform host tensor
    torch::Tensor h_input_aligned_tensor =
        createTensorFromMemory(h_input_aligned.data(), {max_batch_size * max_seq_len, hidden_size}, false);
    torch::Tensor h_output_compact_tensor =
        createTensorFromMemory(h_output_compact.data(), {total_token_sum, hidden_size}, false);
    torch::Tensor h_expected_tensor = createTensorFromMemory(h_expected.data(), {total_token_sum, hidden_size}, false);

    // Initialize input data - only fill valid sequence lengths for each batch
    std::random_device                    rd;
    std::mt19937                          gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    // Initialize all to zero first
    std::fill(h_input_aligned.begin(), h_input_aligned.end(), T(0));

    // Fill only the valid sequence lengths for each batch
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < input_lengths[b]; ++s) {
            for (int h = 0; h < hidden_size; ++h) {
                int aligned_idx              = b * max_seq_len * hidden_size + s * hidden_size + h;
                h_input_aligned[aligned_idx] = T(dis(gen));
            }
        }
    }
    // std::cout << "h_expected_tensor before: " << std::endl << h_expected_tensor << std::endl;
    // Calculate expected output manually
    int compact_idx = 0;
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < input_lengths[b]; ++s) {
            for (int h = 0; h < hidden_size; ++h) {
                int aligned_idx         = b * max_seq_len * hidden_size + s * hidden_size + h;
                h_expected[compact_idx] = h_input_aligned[aligned_idx];
                compact_idx++;
            }
        }
    }

    // std::cout << "h_input_aligned_tensor: " << std::endl << h_input_aligned_tensor << std::endl;
    // std::cout << "h_expected_tensor: " << std::endl << h_expected_tensor << std::endl;

    // Calculate cu_seq_len (cumulative lengths)
    std::vector<int> cu_seq_len(batch_size + 1);
    cu_seq_len[0] = 0;
    for (int i = 0; i < batch_size; ++i) {
        cu_seq_len[i + 1] = cu_seq_len[i] + input_lengths[i];
    }

    // Allocate device memory
    T*   d_input_aligned;
    T*   d_output_compact;
    int* d_input_lengths;
    int* d_cu_seq_len;  // Device memory for kernel access
    int* d_batch_size;  // Pinned memory for batch_size

    cudaMalloc(&d_input_aligned, batch_size * max_seq_len * hidden_size * sizeof(T));
    cudaMalloc(&d_output_compact, total_compact_size * sizeof(T));
    cudaMalloc(&d_input_lengths, batch_size * sizeof(int));
    // Use device allocateBuffer to create pinned memory (like CudaGraphRunner)
    auto cu_seq_len_buffer = device_->allocateBuffer(
        {rtp_llm::DataType::TYPE_INT32, {size_t(batch_size + 1)}, rtp_llm::AllocationType::HOST}, {});
    d_cu_seq_len = static_cast<int*>(cu_seq_len_buffer->data());

    auto batch_size_buffer =
        device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {1}, rtp_llm::AllocationType::HOST}, {});
    d_batch_size = static_cast<int*>(batch_size_buffer->data());

    // Copy data to device
    cudaMemcpy(d_input_aligned,
               h_input_aligned.data(),
               batch_size * max_seq_len * hidden_size * sizeof(T),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_lengths, input_lengths.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    memcpy(d_cu_seq_len, cu_seq_len.data(), (batch_size + 1) * sizeof(int));
    memcpy(d_batch_size, &batch_size, sizeof(int));

    // Initialize output to zero
    cudaMemset(d_output_compact, 0, total_compact_size * sizeof(T));

    // Create events for timing
    cudaEvent_t launch_start, launch_stop, exec_start, exec_stop;
    cudaEventCreate(&launch_start);
    cudaEventCreate(&launch_stop);
    cudaEventCreate(&exec_start);
    cudaEventCreate(&exec_stop);
    // Measure launch time
    cudaEventRecord(launch_start, stream_);
    invokeCudaGraphCopyLarge2Small(d_input_aligned,
                                   d_output_compact,
                                   d_batch_size,
                                   max_batch_size,
                                   max_seq_len,
                                   d_input_lengths,
                                   hidden_size,
                                   d_cu_seq_len,
                                   stream_);
    cudaEventRecord(launch_stop, stream_);

    // Measure execution time (after launch completes)
    cudaEventRecord(exec_start, stream_);
    cudaStreamSynchronize(stream_);
    cudaEventRecord(exec_stop, stream_);

    // Get timing
    float launch_time = 0, exec_time = 0;
    cudaEventElapsedTime(&launch_time, launch_start, launch_stop);
    cudaEventElapsedTime(&exec_time, exec_start, exec_stop);

    float total_time = launch_time + exec_time;

    // Copy result back
    cudaMemcpy(h_output_compact.data(), d_output_compact, total_compact_size * sizeof(T), cudaMemcpyDeviceToHost);
    // std::cout << "h_output_compact_tensor: \n" << h_output_compact_tensor << std::endl;
    // Verify results
    assertTensorEqual(h_output_compact, h_expected, "Large2Small copy", 1e-5f);

    std::cout << "Large2Small copy - Launch: " << std::fixed << std::setprecision(6) << launch_time
              << " ms, Execution: " << exec_time << " ms, Total: " << total_time << " ms" << std::endl;

    // Cleanup
    cudaEventDestroy(launch_start);
    cudaEventDestroy(launch_stop);
    cudaEventDestroy(exec_start);
    cudaEventDestroy(exec_stop);
    cudaFree(d_input_aligned);
    cudaFree(d_output_compact);
    cudaFree(d_input_lengths);
    // cu_seq_len_buffer will be automatically freed by RAII
}

template<typename T>
void CudaGraphCopyKernelTest::testCudaGraphCopyRoundTrip() {
    std::cout << "================== testCudaGraphCopyRoundTrip<" << typeid(T).name()
              << "> ==================" << std::endl;
    // int       batch_size     = 3;
    // const int max_seq_len    = 5;
    // const int hidden_size    = 4;
    // const int max_batch_size = 4;

    // // Input lengths for each batch
    // std::vector<int> input_lengths = {3, 2, 4};

    int       batch_size     = 64;
    const int max_seq_len    = 512;
    const int hidden_size    = 768;
    const int max_batch_size = 64;

    // Input lengths for each batch
    // std::vector<int> input_lengths = {3, 2, 4};
    std::vector<int> input_lengths = {114, 181, 148, 117, 132, 127, 134, 84,  121, 107, 151, 191, 107, 175, 172, 107,
                                      103, 82,  123, 109, 128, 115, 153, 128, 122, 164, 165, 158, 100, 142, 144, 155,
                                      97,  100, 191, 183, 136, 89,  136, 149, 104, 130, 162, 102, 191, 98,  111, 115,
                                      96,  151, 100, 95,  96,  108, 97,  134, 159, 86,  108, 99,  102, 75,  99,  125};
    for (int i = 0; i < input_lengths.size(); i++) {
        input_lengths[i] += 13;
    }

    // Calculate total compact size
    int total_compact_size = 0;
    for (int i = 0; i < batch_size; ++i) {
        total_compact_size += input_lengths[i] * hidden_size;
    }

    // Allocate host memory
    std::vector<T> h_input_compact(total_compact_size);
    std::vector<T> h_intermediate_aligned(max_batch_size * max_seq_len * hidden_size, T(0));
    std::vector<T> h_output_compact(total_compact_size, T(0));
    std::vector<T> h_expected(total_compact_size);

    // Initialize input data
    std::random_device                    rd;
    std::mt19937                          gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < total_compact_size; ++i) {
        h_input_compact[i] = T(dis(gen));
        h_expected[i]      = h_input_compact[i];  // Expected output should be same as input
    }

    // Calculate cu_seq_len (cumulative lengths)
    std::vector<int> cu_seq_len(batch_size + 1);
    cu_seq_len[0] = 0;
    for (int i = 0; i < batch_size; ++i) {
        cu_seq_len[i + 1] = cu_seq_len[i] + input_lengths[i];
    }

    // Allocate device memory
    T*   d_input_compact;
    T*   d_intermediate_aligned;
    T*   d_output_compact;
    int* d_input_lengths;
    int* d_cu_seq_len;  // Device memory for kernel access
    int* d_batch_size;  // Pinned memory for batch_size

    cudaMalloc(&d_input_compact, total_compact_size * sizeof(T));
    cudaMalloc(&d_intermediate_aligned, batch_size * max_seq_len * hidden_size * sizeof(T));
    cudaMalloc(&d_output_compact, total_compact_size * sizeof(T));
    cudaMalloc(&d_input_lengths, batch_size * sizeof(int));
    // Use device allocateBuffer to create pinned memory (like CudaGraphRunner)
    auto cu_seq_len_buffer = device_->allocateBuffer(
        {rtp_llm::DataType::TYPE_INT32, {size_t(batch_size + 1)}, rtp_llm::AllocationType::HOST}, {});
    d_cu_seq_len = static_cast<int*>(cu_seq_len_buffer->data());

    auto batch_size_buffer =
        device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {1}, rtp_llm::AllocationType::HOST}, {});
    d_batch_size = static_cast<int*>(batch_size_buffer->data());

    // Copy data to device
    cudaMemcpy(d_input_compact, h_input_compact.data(), total_compact_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_lengths, input_lengths.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    memcpy(d_cu_seq_len, cu_seq_len.data(), (batch_size + 1) * sizeof(int));
    memcpy(d_batch_size, &batch_size, sizeof(int));

    // Initialize intermediate and output to zero
    cudaMemset(d_intermediate_aligned, 0, batch_size * max_seq_len * hidden_size * sizeof(T));
    cudaMemset(d_output_compact, 0, total_compact_size * sizeof(T));

    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernels: Small2Large -> Large2Small
    cudaEventRecord(start, stream_);
    invokeCudaGraphCopySmall2Large(d_input_compact,
                                   d_intermediate_aligned,
                                   d_batch_size,
                                   max_batch_size,
                                   max_seq_len,
                                   d_input_lengths,
                                   hidden_size,
                                   d_cu_seq_len,
                                   stream_);
    invokeCudaGraphCopyLarge2Small(d_intermediate_aligned,
                                   d_output_compact,
                                   d_batch_size,
                                   max_batch_size,
                                   max_seq_len,
                                   d_input_lengths,
                                   hidden_size,
                                   d_cu_seq_len,
                                   stream_);
    cudaEventRecord(stop, stream_);

    // Wait for completion
    cudaStreamSynchronize(stream_);

    // Get timing
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back
    cudaMemcpy(h_output_compact.data(), d_output_compact, total_compact_size * sizeof(T), cudaMemcpyDeviceToHost);

    // Verify results
    assertTensorEqual(h_output_compact, h_expected, "Round trip copy", 1e-5f);

    std::cout << "Round trip copy completed in " << std::fixed << std::setprecision(3) << milliseconds << " ms"
              << std::endl;

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input_compact);
    cudaFree(d_intermediate_aligned);
    cudaFree(d_output_compact);
    cudaFree(d_input_lengths);
    // cu_seq_len_buffer will be automatically freed by RAII
}

// Test cases for different data types
TEST_F(CudaGraphCopyKernelTest, Small2LargeFloat) {
    testCudaGraphCopySmall2Large<float>();
}

TEST_F(CudaGraphCopyKernelTest, Small2LargeHalf) {
    testCudaGraphCopySmall2Large<half>();
}

#ifdef ENABLE_BF16
TEST_F(CudaGraphCopyKernelTest, Small2LargeBFloat16) {
    testCudaGraphCopySmall2Large<__nv_bfloat16>();
}
#endif

TEST_F(CudaGraphCopyKernelTest, Large2SmallFloat) {
    testCudaGraphCopyLarge2Small<float>();
}

TEST_F(CudaGraphCopyKernelTest, Large2SmallHalf) {
    testCudaGraphCopyLarge2Small<half>();
}

#ifdef ENABLE_BF16
TEST_F(CudaGraphCopyKernelTest, Large2SmallBFloat16) {
    testCudaGraphCopyLarge2Small<__nv_bfloat16>();
}
#endif

TEST_F(CudaGraphCopyKernelTest, RoundTripFloat) {
    testCudaGraphCopyRoundTrip<float>();
}

TEST_F(CudaGraphCopyKernelTest, RoundTripHalf) {
    testCudaGraphCopyRoundTrip<half>();
}

#ifdef ENABLE_BF16
TEST_F(CudaGraphCopyKernelTest, RoundTripBFloat16) {
    testCudaGraphCopyRoundTrip<__nv_bfloat16>();
}
#endif

}  // namespace rtp_llm
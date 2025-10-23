#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include "rtp_llm/cpp/kernels/cuda_graph_copy_kernel.h"

namespace rtp_llm {

template<typename T>
void benchmarkCudaGraphCopyKernel() {
    std::cout << "\n=== Benchmarking CUDA Graph Copy Kernel ===" << std::endl;
    std::cout << "Data type: " << typeid(T).name() << std::endl;

    // Test different scenarios
    struct TestCase {
        int         batch_size;
        int         max_seq_len;
        int         hidden_size;
        std::string description;
    };

    std::vector<TestCase> test_cases = {{32, 512, 768, "Small batch (32)"},
                                        {128, 512, 768, "Medium batch (128)"},
                                        {512, 512, 768, "Large batch (512)"},
                                        {128, 1024, 768, "Long sequence (1024)"},
                                        {128, 512, 2048, "Large hidden (2048)"},
                                        {256, 2048, 1024, "Very large (256x2048x1024)"}};

    for (const auto& test_case : test_cases) {
        std::cout << "\n--- " << test_case.description << " ---" << std::endl;
        std::cout << "Batch: " << test_case.batch_size << ", SeqLen: " << test_case.max_seq_len
                  << ", Hidden: " << test_case.hidden_size << std::endl;

        const int batch_size     = test_case.batch_size;
        const int max_seq_len    = test_case.max_seq_len;
        const int hidden_size    = test_case.hidden_size;
        const int max_batch_size = batch_size;

        // Generate random input lengths (50-90% of max_seq_len)
        std::random_device                 rd;
        std::mt19937                       gen(rd());
        std::uniform_int_distribution<int> len_dis(max_seq_len * 0.5, max_seq_len * 0.9);

        std::vector<int> input_lengths(batch_size);
        int              total_compact_size = 0;
        for (int i = 0; i < batch_size; ++i) {
            input_lengths[i] = len_dis(gen);
            total_compact_size += input_lengths[i] * hidden_size;
        }

        // Calculate memory sizes
        const size_t compact_size_bytes = total_compact_size * sizeof(T);
        const size_t aligned_size_bytes = batch_size * max_seq_len * hidden_size * sizeof(T);
        const size_t total_memory_bytes = compact_size_bytes + aligned_size_bytes;

        std::cout << "Compact size: " << std::fixed << std::setprecision(2) << compact_size_bytes / (1024.0 * 1024.0)
                  << " MB" << std::endl;
        std::cout << "Aligned size: " << std::fixed << std::setprecision(2) << aligned_size_bytes / (1024.0 * 1024.0)
                  << " MB" << std::endl;
        std::cout << "Total memory: " << std::fixed << std::setprecision(2) << total_memory_bytes / (1024.0 * 1024.0)
                  << " MB" << std::endl;

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
        int* d_cu_seq_len;

        cudaMalloc(&d_input_compact, compact_size_bytes);
        cudaMalloc(&d_output_aligned, aligned_size_bytes);
        cudaMalloc(&d_input_lengths, batch_size * sizeof(int));
        cudaMalloc(&d_cu_seq_len, (batch_size + 1) * sizeof(int));

        // Initialize data
        std::uniform_real_distribution<float> data_dis(0.0f, 1.0f);
        std::vector<T>                        h_input_compact(total_compact_size);
        for (int i = 0; i < total_compact_size; ++i) {
            h_input_compact[i] = T(data_dis(gen));
        }

        cudaMemcpy(d_input_compact, h_input_compact.data(), compact_size_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_input_lengths, input_lengths.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cu_seq_len, cu_seq_len.data(), (batch_size + 1) * sizeof(int), cudaMemcpyHostToDevice);

        // Create stream
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        // Warm up
        for (int i = 0; i < 10; ++i) {
            invokeCudaGraphCopySmall2Large<T>(d_input_compact,
                                              d_output_aligned,
                                              const_cast<int*>(&batch_size),
                                              max_batch_size,
                                              max_seq_len,
                                              d_input_lengths,
                                              hidden_size,
                                              d_cu_seq_len,
                                              stream);
        }
        cudaStreamSynchronize(stream);

        // Benchmark Small2Large
        const int num_iterations = 100;
        auto      start          = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_iterations; ++i) {
            invokeCudaGraphCopySmall2Large<T>(d_input_compact,
                                              d_output_aligned,
                                              const_cast<int*>(&batch_size),
                                              max_batch_size,
                                              max_seq_len,
                                              d_input_lengths,
                                              hidden_size,
                                              d_cu_seq_len,
                                              stream);
        }
        cudaStreamSynchronize(stream);

        auto end      = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        double avg_time_us = duration.count() / double(num_iterations);
        double avg_time_ms = avg_time_us / 1000.0;

        // Calculate bandwidth
        double bandwidth_gbps = (compact_size_bytes / (1024.0 * 1024.0 * 1024.0)) / (avg_time_us / 1e6);

        std::cout << "Small2Large Performance:" << std::endl;
        std::cout << "  Average time: " << std::fixed << std::setprecision(3) << avg_time_ms << " ms" << std::endl;
        std::cout << "  Bandwidth: " << std::fixed << std::setprecision(2) << bandwidth_gbps << " GB/s" << std::endl;

        // Benchmark Large2Small
        start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_iterations; ++i) {
            invokeCudaGraphCopyLarge2Small<T>(d_output_aligned,
                                              d_input_compact,
                                              const_cast<int*>(&batch_size),
                                              max_batch_size,
                                              max_seq_len,
                                              d_input_lengths,
                                              hidden_size,
                                              d_cu_seq_len,
                                              stream);
        }
        cudaStreamSynchronize(stream);

        end      = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        avg_time_us    = duration.count() / double(num_iterations);
        avg_time_ms    = avg_time_us / 1000.0;
        bandwidth_gbps = (compact_size_bytes / (1024.0 * 1024.0 * 1024.0)) / (avg_time_us / 1e6);

        std::cout << "Large2Small Performance:" << std::endl;
        std::cout << "  Average time: " << std::fixed << std::setprecision(3) << avg_time_ms << " ms" << std::endl;
        std::cout << "  Bandwidth: " << std::fixed << std::setprecision(2) << bandwidth_gbps << " GB/s" << std::endl;

        // Performance analysis
        std::cout << "Performance Analysis:" << std::endl;

        // Calculate theoretical peak bandwidth (assuming H100 ~900 GB/s)
        double theoretical_peak_gbps = 900.0;
        double efficiency            = (bandwidth_gbps / theoretical_peak_gbps) * 100.0;

        std::cout << "  Memory efficiency: " << std::fixed << std::setprecision(1) << efficiency << "%" << std::endl;

        // Calculate overhead per element
        double overhead_per_element_ns = (avg_time_us * 1000.0) / total_compact_size;
        std::cout << "  Overhead per element: " << std::fixed << std::setprecision(2) << overhead_per_element_ns
                  << " ns" << std::endl;

        // Estimate relative cost
        double elements_per_second = total_compact_size / (avg_time_us / 1e6);
        std::cout << "  Elements per second: " << std::fixed << std::setprecision(0) << elements_per_second / 1e6 << "M"
                  << std::endl;

        // Cleanup
        cudaFree(d_input_compact);
        cudaFree(d_output_aligned);
        cudaFree(d_input_lengths);
        cudaFree(d_cu_seq_len);
        cudaStreamDestroy(stream);
    }
}

void runBenchmark() {
    std::cout << "=========================================" << std::endl;
    std::cout << "CUDA Graph Copy Kernel Performance Analysis" << std::endl;
    std::cout << "=========================================" << std::endl;

    // Get GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Peak Memory Bandwidth: ~" << (prop.memoryClockRate * prop.memoryBusWidth / 8 / 1e6) << " GB/s"
              << std::endl;

    benchmarkCudaGraphCopyKernel<float>();
    benchmarkCudaGraphCopyKernel<half>();

#ifdef ENABLE_BF16
    benchmarkCudaGraphCopyKernel<__nv_bfloat16>();
#endif

    std::cout << "\n=========================================" << std::endl;
    std::cout << "Performance Analysis Summary:" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "1. Memory Bandwidth Bound: This kernel is primarily limited by memory bandwidth" << std::endl;
    std::cout << "2. Index Calculation Overhead: Each thread performs complex index calculations" << std::endl;
    std::cout << "3. Irregular Memory Access: Non-contiguous access patterns reduce efficiency" << std::endl;
    std::cout << "4. Branch Divergence: Conditional checks cause warp divergence" << std::endl;
    std::cout << "5. Optimization Opportunities:" << std::endl;
    std::cout << "   - Use shared memory for input_lengths" << std::endl;
    std::cout << "   - Precompute offsets on CPU" << std::endl;
    std::cout << "   - Use vectorized memory operations" << std::endl;
    std::cout << "   - Consider using CUDA Graphs for repeated operations" << std::endl;
}

}  // namespace rtp_llm

int main() {
    rtp_llm::runBenchmark();
    return 0;
}

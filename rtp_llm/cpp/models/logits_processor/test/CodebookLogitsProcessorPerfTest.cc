#include "gtest/gtest.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <cuda_runtime.h>
#include <ATen/Parallel.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "rtp_llm/cpp/models/logits_processor/CodebookLogitsProcessor.h"

namespace rtp_llm {
namespace {
using Clock = std::chrono::steady_clock;

double micros(Clock::time_point start) {
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

TEST(CodebookLogitsProcessorPerfTest, RequestSetupAndMaskReuse) {
    ASSERT_TRUE(torch::cuda::is_available());
    at::set_num_threads(1);
    int device = 0;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    for (const int levels : {2, 4, 8}) {
        constexpr int                     vocab_size = 65537;  // 65536 code IDs plus EOS.
        std::vector<std::vector<int64_t>> groups(levels);
        for (int id = 1; id < vocab_size; ++id) {
            groups[(id - 1) / (65536 / levels)].push_back(id);
        }
        SamplerInputs inputs;
        inputs.logits        = torch::zeros({1, vocab_size}, torch::TensorOptions().device(torch::kCUDA));
        inputs.finished_mask = torch::zeros({1}, torch::kBool);
        for (const int requests : {1, 32, 128}) {
            std::vector<double> init_times, setup_times, first_times, warm_times;
            int64_t             mask_bytes = 0;
            // First iteration warms allocation and kernels; remaining five are measured.
            for (int repeat = 0; repeat < 6; ++repeat) {
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                const auto before_bytes =
                    c10::cuda::CUDACachingAllocator::getDeviceStats(device).allocated_bytes[0].current;
                auto       start   = Clock::now();
                auto       masks   = CodebookLogitsProcessor::createMasks(groups, vocab_size).to(torch::kCUDA);
                const auto init_us = micros(start);
                std::vector<std::unique_ptr<CodebookLogitsProcessor>> processors;
                processors.reserve(requests);
                start = Clock::now();
                for (int i = 0; i < requests; ++i) {
                    processors.push_back(std::make_unique<CodebookLogitsProcessor>(masks, 1));
                }
                const auto setup_us = micros(start);
                start               = Clock::now();
                for (const auto& processor : processors) {
                    ASSERT_FALSE(processor->process(inputs, 0, 1).has_value());
                }
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                const auto first_us = micros(start);
                mask_bytes =
                    c10::cuda::CUDACachingAllocator::getDeviceStats(device).allocated_bytes[0].current - before_bytes;
                start = Clock::now();
                for (const auto& processor : processors) {
                    ASSERT_FALSE(processor->process(inputs, 0, 1).has_value());
                }
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                const auto warm_us = micros(start);
                if (repeat > 0) {
                    init_times.push_back(init_us);
                    setup_times.push_back(setup_us);
                    first_times.push_back(first_us);
                    warm_times.push_back(warm_us);
                }
            }
            std::cout << "CODEBOOK_BENCH {\"levels\":" << levels << ",\"vocab_size\":" << vocab_size
                      << ",\"requests\":" << requests << ",\"processor_setup_us\":" << median(setup_times)
                      << ",\"model_init_us\":" << median(init_times) << ",\"first_process_us\":" << median(first_times)
                      << ",\"warm_process_us\":" << median(warm_times) << ",\"mask_gpu_bytes\":" << mask_bytes << "}"
                      << std::endl;
        }
    }
}
}  // namespace
}  // namespace rtp_llm

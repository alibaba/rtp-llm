#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "rtp_llm/models_py/bindings/cuda/kernels/cuda_graph_prepare.h"

namespace rtp_llm {
namespace {

// Paired comparisons inside one GPU-lock allocation call the production kernel
// to isolate scalar selection and hidden-tail clearing.
class PrepareBenchmark: public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream_));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&memory_, 32 * 1024 * 1024));
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(memory_, 0, 32 * 1024 * 1024, stream_));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
    }

    void TearDown() override {
        EXPECT_EQ(cudaSuccess, cudaFree(memory_));
        EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream_));
    }

    cudaGraphExec_t capture(CudaGraphPrepareFillParams params, size_t separate_zero_bytes = 0) {
        cudaGraph_t graph = nullptr;
        EXPECT_EQ(cudaSuccess, cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < kUnroll; ++i) {
            if (separate_zero_bytes) {
                EXPECT_EQ(cudaSuccess, cudaMemsetAsync(memory_ + 65536, 0, separate_zero_bytes, stream_));
            }
            invokeCudaGraphPrepareFill(params, stream_);
        }
        EXPECT_EQ(cudaSuccess, cudaStreamEndCapture(stream_, &graph));
        cudaGraphExec_t executable = nullptr;
        EXPECT_EQ(cudaSuccess, cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        EXPECT_EQ(cudaSuccess, cudaGraphDestroy(graph));
        return executable;
    }

    double measure(cudaGraphExec_t graph) {
        cudaEvent_t begin = nullptr, end = nullptr;
        EXPECT_EQ(cudaSuccess, cudaEventCreate(&begin));
        EXPECT_EQ(cudaSuccess, cudaEventCreate(&end));
        EXPECT_EQ(cudaSuccess, cudaEventRecord(begin, stream_));
        for (int i = 0; i < kReplays; ++i) {
            EXPECT_EQ(cudaSuccess, cudaGraphLaunch(graph, stream_));
        }
        EXPECT_EQ(cudaSuccess, cudaEventRecord(end, stream_));
        EXPECT_EQ(cudaSuccess, cudaEventSynchronize(end));
        float milliseconds = 0;
        EXPECT_EQ(cudaSuccess, cudaEventElapsedTime(&milliseconds, begin, end));
        EXPECT_EQ(cudaSuccess, cudaEventDestroy(end));
        EXPECT_EQ(cudaSuccess, cudaEventDestroy(begin));
        return milliseconds * 1000.0 / (kUnroll * kReplays);
    }

    void compare(const std::string&         name,
                 CudaGraphPrepareFillParams host,
                 CudaGraphPrepareFillParams device,
                 size_t                     separate_zero_bytes = 0) {
        auto first  = capture(host, separate_zero_bytes);
        auto second = capture(device);
        for (int i = 0; i < 10; ++i) {
            ASSERT_EQ(cudaSuccess, cudaGraphLaunch(first, stream_));
            ASSERT_EQ(cudaSuccess, cudaGraphLaunch(second, stream_));
        }
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
        const char* directory = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR");
        ASSERT_NE(directory, nullptr);
        std::ofstream output(std::string(directory) + "/prepare_samples.jsonl", std::ios::app);
        ASSERT_TRUE(output.good());
        cudaDeviceProp properties;
        int            ordinal = 0;
        ASSERT_EQ(cudaSuccess, cudaGetDevice(&ordinal));
        ASSERT_EQ(cudaSuccess, cudaGetDeviceProperties(&properties, ordinal));
        for (int sample = 0; sample < 15; ++sample) {
            // Alternate the order so clock drift cannot always favor one side.
            double first_us, second_us;
            if (sample % 2 == 0) {
                first_us  = measure(first);
                second_us = measure(second);
            } else {
                second_us = measure(second);
                first_us  = measure(first);
            }
            output << std::setprecision(9) << "{\"case\":\"" << name << "\",\"gpu\":\"" << properties.name
                   << "\",\"sample\":" << sample << ",\"iterations\":" << kUnroll * kReplays
                   << ",\"baseline_us\":" << first_us << ",\"candidate_us\":" << second_us << "}\n";
        }
        ASSERT_EQ(cudaSuccess, cudaGraphExecDestroy(second));
        ASSERT_EQ(cudaSuccess, cudaGraphExecDestroy(first));
    }

    static constexpr int kUnroll  = 64;
    static constexpr int kReplays = 16;
    cudaStream_t         stream_  = nullptr;
    uint8_t*             memory_  = nullptr;
};

TEST_F(PrepareBenchmark, HostScalarVersusDeviceScalar) {
    for (int query_width : {1, 4, 8}) {
        for (int live_batch : {1, 31, 127}) {
            constexpr int graph_batch = 128;
            const int32_t base        = live_batch * query_width;
            auto*         source      = reinterpret_cast<int32_t*>(memory_);
            auto*         destination = source + 256;
            ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(source, &base, sizeof(base), cudaMemcpyHostToDevice, stream_));
            ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
            CudaGraphPrepareFillParams host;
            host.region_count              = 1;
            host.regions[0].ptr            = destination;
            host.regions[0].count          = graph_batch - live_batch;
            host.regions[0].value          = base + query_width;
            host.regions[0].step           = query_width;
            auto device                    = host;
            device.regions[0].value_ptr    = source;
            device.regions[0].value_offset = query_width;
            device.regions[0].value        = 0;
            for (auto params : {host, device}) {
                invokeCudaGraphPrepareFill(params, stream_);
                std::vector<int32_t> actual(graph_batch - live_batch);
                ASSERT_EQ(
                    cudaSuccess,
                    cudaMemcpyAsync(
                        actual.data(), destination, actual.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_));
                ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
                for (size_t i = 0; i < actual.size(); ++i) {
                    ASSERT_EQ(actual[i], base + (static_cast<int>(i) + 1) * query_width);
                }
            }
            compare("scalar_q" + std::to_string(query_width) + "_b" + std::to_string(live_batch), host, device);
        }
    }
}

TEST_F(PrepareBenchmark, SeparateVersusFusedHiddenTailZero) {
    for (size_t bytes : {size_t(4096), size_t(1048576), size_t(16777216)}) {
        CudaGraphPrepareFillParams separate;
        separate.region_count     = 1;
        separate.regions[0].ptr   = reinterpret_cast<int32_t*>(memory_);
        separate.regions[0].count = 128;
        separate.regions[0].value = 4;
        auto fused                = separate;
        fused.zero_bytes          = memory_ + 65536;
        fused.zero_byte_count     = bytes;
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(fused.zero_bytes, 0xff, bytes, stream_));
        invokeCudaGraphPrepareFill(fused, stream_);
        std::vector<uint8_t> actual(bytes);
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpyAsync(actual.data(), fused.zero_bytes, bytes, cudaMemcpyDeviceToHost, stream_));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
        ASSERT_TRUE(std::all_of(actual.begin(), actual.end(), [](uint8_t value) { return value == 0; }));
        compare("hidden_zero_bytes_" + std::to_string(bytes), separate, fused, bytes);
    }
}

}  // namespace
}  // namespace rtp_llm

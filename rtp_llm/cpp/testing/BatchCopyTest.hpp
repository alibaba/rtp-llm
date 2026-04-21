#pragma once

#include "rtp_llm/cpp/testing/TestBase.h"
#include <numeric>
#include <random>
#include <vector>

using namespace std;
using namespace rtp_llm;

class BatchCopyTest: public DeviceTestBase {
public:
    void SetUp() override {
        DeviceTestBase::SetUp();
    }

protected:
    torch::Tensor createDeviceTensorWithData(size_t num_floats, float start_val) {
        vector<float> data(num_floats);
        iota(data.begin(), data.end(), start_val);
        return createDeviceTensor<float>({static_cast<int64_t>(num_floats)}, data);
    }

    torch::Tensor createEmptyDeviceTensor(size_t num_floats) {
        return createDeviceTensor({static_cast<int64_t>(num_floats)}, torch::kFloat32);
    }

    torch::Tensor createHostTensorWithData(size_t num_floats, float start_val) {
        vector<float> data(num_floats);
        iota(data.begin(), data.end(), start_val);
        return createHostTensor<float>({static_cast<int64_t>(num_floats)}, data);
    }

    torch::Tensor createEmptyHostTensor(size_t num_floats) {
        return torch::empty({static_cast<int64_t>(num_floats)}, torch::kFloat32);
    }

    void addTensorCopy(BatchCopyParams& params, const torch::Tensor& dst, const torch::Tensor& src) {
        params.add(dst.data_ptr(),
                   src.data_ptr(),
                   dst.nbytes(),
                   BatchCopyParams::get_copy_type(torchDeviceToMemoryType(dst.device()),
                                                  torchDeviceToMemoryType(src.device())));
    }

    void runBatchCopy(BatchCopyParams& params) {
        execBatchCopy(params);
        runtimeSyncAndCheck();
    }

    vector<float> readBack(const torch::Tensor& tensor) {
        return getTensorValues<float>(tensor);
    }

    // ---- D2D tests ----

    void testD2DSingleCopy() {
        const size_t n   = 64;
        auto         src = createDeviceTensorWithData(n, 1.0f);
        auto         dst = createEmptyDeviceTensor(n);

        BatchCopyParams params;
        addTensorCopy(params, dst, src);
        runBatchCopy(params);

        auto result = readBack(dst);
        for (size_t i = 0; i < n; i++) {
            EXPECT_FLOAT_EQ(result[i], static_cast<float>(i + 1));
        }
    }

    void testD2DMultipleCopies() {
        const size_t n1 = 32, n2 = 128, n3 = 17;
        auto         src1 = createDeviceTensorWithData(n1, 0.0f);
        auto         src2 = createDeviceTensorWithData(n2, 100.0f);
        auto         src3 = createDeviceTensorWithData(n3, 500.0f);

        auto dst1 = createEmptyDeviceTensor(n1);
        auto dst2 = createEmptyDeviceTensor(n2);
        auto dst3 = createEmptyDeviceTensor(n3);

        BatchCopyParams params;
        addTensorCopy(params, dst1, src1);
        addTensorCopy(params, dst2, src2);
        addTensorCopy(params, dst3, src3);
        runBatchCopy(params);

        auto r1 = readBack(dst1);
        auto r2 = readBack(dst2);
        auto r3 = readBack(dst3);

        for (size_t i = 0; i < n1; i++)
            EXPECT_FLOAT_EQ(r1[i], static_cast<float>(i));
        for (size_t i = 0; i < n2; i++)
            EXPECT_FLOAT_EQ(r2[i], static_cast<float>(i + 100));
        for (size_t i = 0; i < n3; i++)
            EXPECT_FLOAT_EQ(r3[i], static_cast<float>(i + 500));
    }

    void testD2DUniformSizes() {
        // All copies have the same size; exercises the row-aligned kernel path.
        const size_t copy_size  = 256;
        const int    num_copies = 16;

        vector<torch::Tensor> srcs, dsts;
        for (int i = 0; i < num_copies; i++) {
            srcs.push_back(createDeviceTensorWithData(copy_size, static_cast<float>(i * 1000)));
            dsts.push_back(createEmptyDeviceTensor(copy_size));
        }

        BatchCopyParams params;
        for (int i = 0; i < num_copies; i++) {
            addTensorCopy(params, dsts[i], srcs[i]);
        }
        runBatchCopy(params);

        for (int i = 0; i < num_copies; i++) {
            auto result = readBack(dsts[i]);
            for (size_t j = 0; j < copy_size; j++) {
                EXPECT_FLOAT_EQ(result[j], static_cast<float>(i * 1000 + j));
            }
        }
    }

    void testD2DVariableSizes() {
        // Different sizes per copy; exercises the variable-size kernel path.
        vector<size_t> sizes = {1, 7, 33, 64, 127, 256, 513, 1024};

        vector<torch::Tensor> srcs, dsts;
        for (size_t i = 0; i < sizes.size(); i++) {
            srcs.push_back(createDeviceTensorWithData(sizes[i], static_cast<float>(i * 10000)));
            dsts.push_back(createEmptyDeviceTensor(sizes[i]));
        }

        BatchCopyParams params;
        for (size_t i = 0; i < sizes.size(); i++) {
            addTensorCopy(params, dsts[i], srcs[i]);
        }
        runBatchCopy(params);

        for (size_t i = 0; i < sizes.size(); i++) {
            auto result = readBack(dsts[i]);
            ASSERT_EQ(result.size(), sizes[i]);
            for (size_t j = 0; j < sizes[i]; j++) {
                EXPECT_FLOAT_EQ(result[j], static_cast<float>(i * 10000 + j));
            }
        }
    }

    void testD2DLargeBatch() {
        // Stress test with many small copies
        const int    num_copies = 256;
        const size_t elem_count = 16;

        vector<torch::Tensor> srcs, dsts;
        for (int i = 0; i < num_copies; i++) {
            srcs.push_back(createDeviceTensorWithData(elem_count, static_cast<float>(i)));
            dsts.push_back(createEmptyDeviceTensor(elem_count));
        }

        BatchCopyParams params;
        for (int i = 0; i < num_copies; i++) {
            addTensorCopy(params, dsts[i], srcs[i]);
        }
        runBatchCopy(params);

        for (int i = 0; i < num_copies; i++) {
            auto result = readBack(dsts[i]);
            for (size_t j = 0; j < elem_count; j++) {
                EXPECT_FLOAT_EQ(result[j], static_cast<float>(i + j));
            }
        }
    }

    void testD2DWithH2DAndD2H() {
        // Mixed copy types in a single batchCopy call
        const size_t n          = 64;
        auto         device_src = createDeviceTensorWithData(n, 0.0f);
        auto         device_dst = createEmptyDeviceTensor(n);

        // Also add a D2H copy
        auto host_dst = createEmptyHostTensor(n);

        // And an H2D copy
        auto host_src    = createHostTensorWithData(n, 2000.0f);
        auto device_dst2 = createEmptyDeviceTensor(n);

        BatchCopyParams params;
        addTensorCopy(params, device_dst, device_src);   // D2D
        addTensorCopy(params, host_dst, device_src);     // D2H
        addTensorCopy(params, device_dst2, host_src);    // H2D
        runBatchCopy(params);

        // Verify D2D
        auto r1 = readBack(device_dst);
        for (size_t i = 0; i < n; i++)
            EXPECT_FLOAT_EQ(r1[i], static_cast<float>(i));

        // Verify D2H
        auto* h = host_dst.data_ptr<float>();
        for (size_t i = 0; i < n; i++)
            EXPECT_FLOAT_EQ(h[i], static_cast<float>(i));

        // Verify H2D
        auto r2 = readBack(device_dst2);
        for (size_t i = 0; i < n; i++)
            EXPECT_FLOAT_EQ(r2[i], static_cast<float>(i + 2000));
    }
};

#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/cuda_impl/tests/CudaTestUtils.h"
#include "rtp_llm/cpp/core/QBuffer.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include <cmath>
#include <limits>
#include <filesystem>
#include <vector>
#include <string>
#include <thread>
#include <chrono>

using namespace std;
using namespace rtp_llm;

class CudaCheckNANTest: public DeviceTestBase {
public:
    void TearDown() override {
        // Clean up .pt files generated during tests
        std::filesystem::path current_dir = std::filesystem::current_path();
        for (const auto& entry : std::filesystem::directory_iterator(current_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".pt") {
                std::string filename = entry.path().filename().string();
                if (filename.find("nan_tensor_") == 0) {
                    std::filesystem::remove(entry.path());
                }
            }
        }
        DeviceTestBase::TearDown();
    }

protected:
};

TEST_F(CudaCheckNANTest, testCheckNAN_NoNaN) {
    device_ = new CudaDevice(DeviceInitParams());
    device_->init();

    // Test normal tensor without NaN/INF
    std::vector<float> normal_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    BufferPtr          buffer      = createBuffer<float>({5}, normal_data, AllocationType::DEVICE);

    bool has_nan = device_->checkNAN(*buffer, "normal_tensor");
    EXPECT_FALSE(has_nan) << "Normal tensor should not have NaN/INF";
}

TEST_F(CudaCheckNANTest, testCheckNAN_WithNaN) {
    device_ = new CudaDevice(DeviceInitParams());
    device_->init();

    // Test tensor with NaN
    std::vector<float> nan_data = {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f, 4.0f, 5.0f};
    BufferPtr          buffer   = createBuffer<float>({5}, nan_data, AllocationType::DEVICE);

    bool has_nan = device_->checkNAN(*buffer, "nan_tensor");
    EXPECT_TRUE(has_nan) << "Tensor with NaN should be detected";
}

TEST_F(CudaCheckNANTest, testCheckNAN_WithInf) {
    device_ = new CudaDevice(DeviceInitParams());
    device_->init();

    // Test tensor with INF
    std::vector<float> inf_data = {1.0f, std::numeric_limits<float>::infinity(), 3.0f, 4.0f, 5.0f};
    BufferPtr          buffer   = createBuffer<float>({5}, inf_data, AllocationType::DEVICE);

    bool has_nan = device_->checkNAN(*buffer, "inf_tensor");
    EXPECT_TRUE(has_nan) << "Tensor with INF should be detected";
}

TEST_F(CudaCheckNANTest, testCheckNAN_FP16) {
    device_ = new CudaDevice(DeviceInitParams());
    device_->init();

    // Test FP16 tensor with NaN
    std::vector<half> fp16_data = {
        __float2half(1.0f), __float2half(std::numeric_limits<float>::quiet_NaN()), __float2half(3.0f)};
    BufferPtr buffer = createBuffer<half>({3}, fp16_data, AllocationType::DEVICE);

    bool has_nan = device_->checkNAN(*buffer, "fp16_nan_tensor");
    EXPECT_TRUE(has_nan) << "FP16 tensor with NaN should be detected";
}

TEST_F(CudaCheckNANTest, testCheckNAN_QBuffer) {
    device_ = new CudaDevice(DeviceInitParams());
    device_->init();

    // Create QBuffer with kernel, scales, and zeros
    // Kernel: int8 data
    std::vector<int8_t> kernel_data = {1, 2, 3, 4, 5, 6, 7, 8};
    BufferPtr           kernel      = createBuffer<int8_t>({8}, kernel_data, AllocationType::DEVICE);

    // Scales: FP16 data with NaN
    std::vector<half> scales_data = {
        __float2half(1.0f), __float2half(std::numeric_limits<float>::quiet_NaN()), __float2half(2.0f)};
    BufferPtr scales = createBuffer<half>({3}, scales_data, AllocationType::DEVICE);

    // Zeros: FP16 data
    std::vector<half> zeros_data = {__float2half(0.0f), __float2half(0.0f), __float2half(0.0f)};
    BufferPtr         zeros      = createBuffer<half>({3}, zeros_data, AllocationType::DEVICE);

    // Create QBuffer - need to ensure use_count is 1
    // Create new BufferPtrs and use std::move to ensure use_count is 1
    BufferPtr kernel_new =
        std::make_shared<Buffer>(kernel->where(), kernel->type(), kernel->shape(), kernel->data(), nullptr);
    BufferPtr scales_new =
        std::make_shared<Buffer>(scales->where(), scales->type(), scales->shape(), scales->data(), nullptr);
    BufferPtr zeros_new =
        std::make_shared<Buffer>(zeros->where(), zeros->type(), zeros->shape(), zeros->data(), nullptr);

    BufferPtr qbuffer = std::make_shared<QBuffer>(std::move(kernel_new), std::move(scales_new), std::move(zeros_new));

    // Test QBuffer checkNAN - should detect NaN in scales
    bool has_nan = device_->checkNAN(*qbuffer, "qbuffer_with_nan_scales");
    EXPECT_TRUE(has_nan) << "QBuffer with NaN in scales should be detected";
}

TEST_F(CudaCheckNANTest, testCheckNAN_QBuffer_AllParts) {
    device_ = new CudaDevice(DeviceInitParams());
    device_->init();

    // Create QBuffer where zeros have INF
    std::vector<int8_t> kernel_data = {1, 2, 3, 4};
    BufferPtr           kernel      = createBuffer<int8_t>({4}, kernel_data, AllocationType::DEVICE);

    std::vector<half> scales_data = {__float2half(1.0f), __float2half(2.0f)};
    BufferPtr         scales      = createBuffer<half>({2}, scales_data, AllocationType::DEVICE);

    std::vector<half> zeros_data = {__float2half(0.0f), __float2half(std::numeric_limits<float>::infinity())};
    BufferPtr         zeros      = createBuffer<half>({2}, zeros_data, AllocationType::DEVICE);

    // Create QBuffer - need to ensure use_count is 1
    // Create new BufferPtrs and use std::move to ensure use_count is 1
    BufferPtr kernel_new =
        std::make_shared<Buffer>(kernel->where(), kernel->type(), kernel->shape(), kernel->data(), nullptr);
    BufferPtr scales_new =
        std::make_shared<Buffer>(scales->where(), scales->type(), scales->shape(), scales->data(), nullptr);
    BufferPtr zeros_new =
        std::make_shared<Buffer>(zeros->where(), zeros->type(), zeros->shape(), zeros->data(), nullptr);

    BufferPtr qbuffer = std::make_shared<QBuffer>(std::move(kernel_new), std::move(scales_new), std::move(zeros_new));

    // Test QBuffer checkNAN - should detect INF in zeros
    bool has_nan = device_->checkNAN(*qbuffer, "qbuffer_with_inf_zeros");
    EXPECT_TRUE(has_nan) << "QBuffer with INF in zeros should be detected";
}

TEST_F(CudaCheckNANTest, testCheckNAN_ForcePrint) {
    device_ = new CudaDevice(DeviceInitParams());
    device_->init();

    // Test dumpTensor - should print even without NaN/INF
    std::vector<float> normal_data = {1.0f, 2.0f, 3.0f};
    BufferPtr          buffer      = createBuffer<float>({3}, normal_data, AllocationType::DEVICE);

    bool has_nan = device_->checkNAN(*buffer, "force_print_tensor");
    EXPECT_FALSE(has_nan) << "Normal tensor should not have NaN/INF";

    // Explicitly call dumpTensor to test force print functionality
    dumpTensor(*buffer, "force_print_tensor", 0);
}

TEST_F(CudaCheckNANTest, testCheckNAN_DumpFile) {
    device_ = new CudaDevice(DeviceInitParams());
    device_->init();

    // Get current working directory to check for dump files
    std::filesystem::path current_dir = std::filesystem::current_path();

    // Get initial file list (files matching the pattern before test)
    std::vector<std::string> initial_files;
    for (const auto& entry : std::filesystem::directory_iterator(current_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".pt") {
            std::string filename = entry.path().filename().string();
            if (filename.find("nan_tensor_") == 0) {
                initial_files.push_back(filename);
            }
        }
    }

    // Test tensor with NaN - should trigger file dump
    std::vector<float> nan_data = {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f};
    BufferPtr          buffer   = createBuffer<float>({3}, nan_data, AllocationType::DEVICE);

    bool has_nan = device_->checkNAN(*buffer, "dump_test_tensor");
    EXPECT_TRUE(has_nan) << "Tensor with NaN should be detected";

    // Wait a bit for file I/O to complete
    device_->syncAndCheck();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Check if dump file was created
    bool                  file_found = false;
    std::filesystem::path dump_file_path;
    for (const auto& entry : std::filesystem::directory_iterator(current_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".pt") {
            std::string filename = entry.path().filename().string();
            // Check if this is a new file matching the pattern
            if (filename.find("nan_tensor_dump_test_tensor") == 0) {
                // Check if it's not in initial_files
                bool is_new_file = true;
                for (const auto& initial_file : initial_files) {
                    if (initial_file == filename) {
                        is_new_file = false;
                        break;
                    }
                }
                if (is_new_file) {
                    file_found     = true;
                    dump_file_path = entry.path();
                    // Verify file is not empty
                    EXPECT_GT(std::filesystem::file_size(dump_file_path), 0) << "Dump file should not be empty";
                    break;
                }
            }
        }
    }

    EXPECT_TRUE(file_found) << "Dump file should be created when NaN is detected";

    // Clean up test file if found
    if (file_found && std::filesystem::exists(dump_file_path)) {
        std::filesystem::remove(dump_file_path);
    }
}

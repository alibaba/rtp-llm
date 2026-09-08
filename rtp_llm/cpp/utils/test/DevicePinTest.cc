#include "gtest/gtest.h"

#include "rtp_llm/cpp/utils/DevicePin.h"

#include <stdexcept>
#include <vector>

#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#elif USING_ROCM
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#endif

namespace rtp_llm {

#if USING_CUDA || USING_ROCM
namespace {

int gpuDeviceCountForTest() {
#if USING_CUDA
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess ? device_count : 0;
#elif USING_ROCM
    int device_count = 0;
    return hipGetDeviceCount(&device_count) == hipSuccess ? device_count : 0;
#else
    return 0;
#endif
}

}  // namespace
#endif

TEST(DevicePinTest, NegativeDeviceIdSkipsSetterAndKeepsCache) {
    int cached_device = 3;
    int device_calls  = 0;
    int stream_calls  = 0;

    detail::setCurrentThreadDeviceIfNeededImpl(
        -1, cached_device, [&device_calls](int) { ++device_calls; }, [&stream_calls](int) { ++stream_calls; });

    ASSERT_EQ(0, device_calls);
    ASSERT_EQ(0, stream_calls);
    ASSERT_EQ(3, cached_device);
}

TEST(DevicePinTest, CachedDeviceIdSkipsRepeatedSetDevice) {
    int              cached_device = -1;
    std::vector<int> set_devices;
    std::vector<int> stream_devices;

    auto set_device_context = [&set_devices](int device) { set_devices.push_back(device); };
    auto set_default_stream = [&stream_devices](int device) { stream_devices.push_back(device); };

    detail::setCurrentThreadDeviceIfNeededImpl(0, cached_device, set_device_context, set_default_stream);
    detail::setCurrentThreadDeviceIfNeededImpl(0, cached_device, set_device_context, set_default_stream);

    ASSERT_EQ(1u, set_devices.size());
    ASSERT_EQ(0, set_devices[0]);
    ASSERT_EQ((std::vector<int>{0, 0}), stream_devices);
    ASSERT_EQ(0, cached_device);
}

TEST(DevicePinTest, CachedDeviceMismatchRepinsThread) {
    int              cached_device = 0;
    std::vector<int> set_devices;
    std::vector<int> stream_devices;

    detail::setCurrentThreadDeviceIfNeededImpl(
        0,
        cached_device,
        [&set_devices](int device) { set_devices.push_back(device); },
        [&stream_devices](int device) { stream_devices.push_back(device); },
        []() { return 1; });

    ASSERT_EQ((std::vector<int>{0}), set_devices);
    ASSERT_EQ((std::vector<int>{0}), stream_devices);
    ASSERT_EQ(0, cached_device);
}

TEST(DevicePinTest, DifferentDeviceIdRetargetsThread) {
    int              cached_device = 0;
    std::vector<int> set_devices;

    detail::setCurrentThreadDeviceIfNeededImpl(
        1, cached_device, [&set_devices](int device) { set_devices.push_back(device); }, [](int) {});

    ASSERT_EQ(1u, set_devices.size());
    ASSERT_EQ(1, set_devices[0]);
    ASSERT_EQ(1, cached_device);
}

TEST(DevicePinTest, SetterExceptionPropagatesAndDoesNotUpdateCache) {
    int cached_device = 0;

    ASSERT_THROW(detail::setCurrentThreadDeviceIfNeededImpl(
                     1, cached_device, [](int) { throw std::runtime_error("set device failed"); }, [](int) {}),
                 std::runtime_error);
    ASSERT_EQ(0, cached_device);

    int set_calls = 0;
    detail::setCurrentThreadDeviceIfNeededImpl(1, cached_device, [&set_calls](int) { ++set_calls; }, [](int) {});

    ASSERT_EQ(1, set_calls);
    ASSERT_EQ(1, cached_device);
}

TEST(DevicePinTest, StreamSetterExceptionPropagatesAndDoesNotUpdateCache) {
    int cached_device = 0;

    ASSERT_THROW(detail::setCurrentThreadDeviceIfNeededImpl(
                     1, cached_device, [](int) {}, [](int) { throw std::runtime_error("set stream failed"); }),
                 std::runtime_error);
    ASSERT_EQ(0, cached_device);
}

#if USING_CUDA
TEST(DevicePinTest, SetCurrentThreadDeviceResetsCUDAStreamToDefault) {
    if (gpuDeviceCountForTest() < 1) {
        GTEST_SKIP() << "No GPU device available";
    }

    constexpr int kDevice = 0;
    at::cuda::setCurrentCUDAStream(at::cuda::getStreamFromPool(/*isHighPriority=*/false, kDevice));
    ASSERT_NE(at::cuda::getDefaultCUDAStream(kDevice), at::cuda::getCurrentCUDAStream(kDevice));

    setCurrentThreadDevice(kDevice);

    ASSERT_EQ(at::cuda::getDefaultCUDAStream(kDevice), at::cuda::getCurrentCUDAStream(kDevice));
}
#elif USING_ROCM
TEST(DevicePinTest, SetCurrentThreadDeviceResetsHIPStreamToDefault) {
    if (gpuDeviceCountForTest() < 1) {
        GTEST_SKIP() << "No GPU device available";
    }

    constexpr int kDevice = 0;
    c10::hip::setCurrentHIPStream(c10::hip::getStreamFromPool(/*isHighPriority=*/false, kDevice));
    ASSERT_NE(c10::hip::getDefaultHIPStream(kDevice), c10::hip::getCurrentHIPStream(kDevice));

    setCurrentThreadDevice(kDevice);

    ASSERT_EQ(c10::hip::getDefaultHIPStream(kDevice), c10::hip::getCurrentHIPStream(kDevice));
}
#endif

}  // namespace rtp_llm

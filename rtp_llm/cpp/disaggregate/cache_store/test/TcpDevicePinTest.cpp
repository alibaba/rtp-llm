#include "gtest/gtest.h"

#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpBlockReadClosure.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreServiceImpl.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/CacheStoreTestBase.h"

#include <memory>
#include <utility>
#include <vector>

#if USING_CUDA
#include <cuda_runtime.h>
#elif USING_ROCM
#include <hip/hip_runtime.h>
#endif

namespace rtp_llm {

#if USING_CUDA || USING_ROCM
namespace {

int gpuDeviceCountForTcpTest() {
#if USING_CUDA
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess ? device_count : 0;
#elif USING_ROCM
    int device_count = 0;
    return hipGetDeviceCount(&device_count) == hipSuccess ? device_count : 0;
#endif
}

int currentDeviceForTcpTest() {
#if USING_CUDA
    int device = -1;
    return cudaGetDevice(&device) == cudaSuccess ? device : -1;
#elif USING_ROCM
    int device = -1;
    return hipGetDevice(&device) == hipSuccess ? device : -1;
#endif
}

bool setDeviceForTcpTest(int device) {
#if USING_CUDA
    return cudaSetDevice(device) == cudaSuccess;
#elif USING_ROCM
    return hipSetDevice(device) == hipSuccess;
#endif
}

void clearGpuErrorForTcpTest() {
#if USING_CUDA
    cudaGetLastError();
#elif USING_ROCM
    hipGetLastError();
#endif
}

class ScopedDeviceResetForTcpTest {
public:
    ScopedDeviceResetForTcpTest(): original_device_(currentDeviceForTcpTest()) {}
    ~ScopedDeviceResetForTcpTest() {
        if (original_device_ >= 0) {
            setDeviceForTcpTest(original_device_);
        }
    }

    ScopedDeviceResetForTcpTest(const ScopedDeviceResetForTcpTest&) = delete;
    ScopedDeviceResetForTcpTest& operator=(const ScopedDeviceResetForTcpTest&) = delete;

private:
    int original_device_;
};

class RecordingDoneClosure: public ::google::protobuf::Closure {
public:
    void Run() override {
        ++run_count;
        observed_device = currentDeviceForTcpTest();
    }

    int run_count{0};
    int observed_device{-1};
};

TcpBlockReadClosure* makeBlockReadClosure(int device_id, TransferConnection::ReadDoneCallback callback) {
    auto response = new BlockReadResponse;
    response->set_error_code(KvCacheStoreServiceErrorCode::EC_SUCCESS);
    return new TcpBlockReadClosure(
        {}, {}, std::move(callback), new BlockReadRequest, response, new arpc::ANetRPCController, device_id);
}

}  // namespace
#endif

class TcpDevicePinTest: public CacheStoreTestBase {
protected:
    std::unique_ptr<TcpCacheStoreServiceImpl> makeService(int device_id) {
        auto buffer_store = std::make_shared<RequestBlockBufferStore>(memory_util_);
        return std::make_unique<TcpCacheStoreServiceImpl>(
            memory_util_, buffer_store, nullptr, nullptr, nullptr, nullptr, device_id);
    }
};

TEST_F(TcpDevicePinTest, BlockReadClosureReportsInvalidDevice) {
#if USING_CUDA || USING_ROCM
    const int device_count = gpuDeviceCountForTcpTest();
    if (device_count < 1) {
        GTEST_SKIP() << "No GPU device available";
    }

    int                 callback_count = 0;
    bool                callback_ok    = true;
    CacheStoreErrorCode callback_error = CacheStoreErrorCode::None;
    auto                closure        = makeBlockReadClosure(
        device_count, [&](bool ok, CacheStoreErrorCode error, const std::vector<std::shared_ptr<BlockBuffer>>&) {
            ++callback_count;
            callback_ok    = ok;
            callback_error = error;
        });

    closure->Run();
    clearGpuErrorForTcpTest();

    EXPECT_EQ(1, callback_count);
    EXPECT_FALSE(callback_ok);
    EXPECT_EQ(CacheStoreErrorCode::LoadErrorUnknown, callback_error);
#else
    GTEST_SKIP() << "GPU device pinning is unavailable in CPU-only builds";
#endif
}

TEST_F(TcpDevicePinTest, BlockReadClosurePinsNonDefaultDevice) {
#if USING_CUDA || USING_ROCM
    const int device_count = gpuDeviceCountForTcpTest();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least two GPU devices to prove non-default device pinning";
    }

    ScopedDeviceResetForTcpTest reset_device;
    constexpr int               kStartingDevice = 0;
    constexpr int               kTargetDevice   = 1;
    ASSERT_TRUE(setDeviceForTcpTest(kStartingDevice));
    ASSERT_EQ(kStartingDevice, currentDeviceForTcpTest());

    int  callback_count  = 0;
    int  observed_device = -1;
    auto closure         = makeBlockReadClosure(
        kTargetDevice, [&](bool ok, CacheStoreErrorCode error, const std::vector<std::shared_ptr<BlockBuffer>>&) {
            ++callback_count;
            observed_device = currentDeviceForTcpTest();
            EXPECT_TRUE(ok);
            EXPECT_EQ(CacheStoreErrorCode::None, error);
        });

    closure->Run();

    EXPECT_EQ(1, callback_count);
    EXPECT_EQ(kTargetDevice, observed_device);
    EXPECT_EQ(kTargetDevice, currentDeviceForTcpTest());
#else
    GTEST_SKIP() << "GPU device pinning is unavailable in CPU-only builds";
#endif
}

TEST_F(TcpDevicePinTest, CacheStoreServiceReportsInvalidDevice) {
#if USING_CUDA || USING_ROCM
    const int device_count = gpuDeviceCountForTcpTest();
    if (device_count < 1) {
        GTEST_SKIP() << "No GPU device available";
    }

    auto                 service = makeService(device_count);
    BlockReadRequest     request;
    BlockReadResponse    response;
    RecordingDoneClosure done;

    service->blockRead(nullptr, &request, &response, &done);
    clearGpuErrorForTcpTest();

    EXPECT_EQ(1, done.run_count);
    EXPECT_EQ(KvCacheStoreServiceErrorCode::EC_FAILED_INTERNAL, response.error_code());
    EXPECT_EQ(0, response.blocks_size());
#else
    GTEST_SKIP() << "GPU device pinning is unavailable in CPU-only builds";
#endif
}

TEST_F(TcpDevicePinTest, CacheStoreServicePinsNonDefaultDevice) {
#if USING_CUDA || USING_ROCM
    const int device_count = gpuDeviceCountForTcpTest();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least two GPU devices to prove non-default device pinning";
    }

    ScopedDeviceResetForTcpTest reset_device;
    constexpr int               kStartingDevice = 0;
    constexpr int               kTargetDevice   = 1;
    ASSERT_TRUE(setDeviceForTcpTest(kStartingDevice));
    ASSERT_EQ(kStartingDevice, currentDeviceForTcpTest());

    auto                 service = makeService(kTargetDevice);
    BlockReadRequest     request;
    BlockReadResponse    response;
    RecordingDoneClosure done;

    service->blockRead(nullptr, &request, &response, &done);

    EXPECT_EQ(1, done.run_count);
    EXPECT_EQ(kTargetDevice, done.observed_device);
    EXPECT_EQ(kTargetDevice, currentDeviceForTcpTest());
    EXPECT_EQ(KvCacheStoreServiceErrorCode::EC_SUCCESS, response.error_code());
#else
    GTEST_SKIP() << "GPU device pinning is unavailable in CPU-only builds";
#endif
}

}  // namespace rtp_llm

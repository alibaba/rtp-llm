#include "unittest/unittest.h"

// force unitetst header first, change private class member
#include "aios/network/rdma/accl/SimpleMessageConverter.h"
#include "aios/network/rdma/accl/unittest/mock/MockCudaUtil.h"
#include "aios/network/rdma/accl/unittest/mock/MockXSimpleMempool.h"

#ifdef GOOGLE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

using namespace ::testing;

ARPC_BEGIN_NAMESPACE(arpc);

class SimpleMessageConverterTestOnMock: public TESTBASE {
public:
    void setUp();
    void tearDown();

protected:
    void CheckIov(::accl::barex::xiovec      iov,
                  void*                      buf,
                  int                        len,
                  ::accl::barex::device_type device,
                  ::accl::barex::device_type remoteDevice);
    void CheckMessageIov(const std::shared_ptr<RdmaMessage>& message,
                         int                                 i,
                         void*                               buf,
                         int                                 len,
                         RdmaMessage::BufferMRStatus         mrStatus,
                         RdmaMempool::DeviceType             device,
                         RdmaMempool::DeviceType             remoteDevice);

protected:
    std::shared_ptr<RdmaMempool>                       _mempool;
    std::shared_ptr<::accl::barex::MockXSimpleMempool> _mockXMempool;
};

void SimpleMessageConverterTestOnMock::setUp() {
    _mockXMempool = std::make_shared<::accl::barex::MockXSimpleMempool>();
    EXPECT_CALL(*_mockXMempool, Shutdown()).WillOnce(Return(::accl::barex::BAREX_SUCCESS));
    EXPECT_CALL(*_mockXMempool, WaitStop()).WillOnce(Return(::accl::barex::BAREX_SUCCESS));

    _mempool = std::make_shared<RdmaMempool>();
    _mempool->_curState.store(RdmaMempool::INIT_SUCCESS);
    _mempool->_xmempool = std::dynamic_pointer_cast<::accl::barex::XSimpleMempool>(_mockXMempool);
}

void SimpleMessageConverterTestOnMock::tearDown() {}

void SimpleMessageConverterTestOnMock::CheckIov(::accl::barex::xiovec      iov,
                                                void*                      buf,
                                                int                        len,
                                                ::accl::barex::device_type device,
                                                ::accl::barex::device_type remoteDevice) {
    if (buf) {
        EXPECT_EQ(buf, iov.iov_base);
    }
    EXPECT_EQ(len, iov.iov_len);
    EXPECT_EQ(device, iov.device);
    EXPECT_EQ(remoteDevice, iov.remote_device);
}

void SimpleMessageConverterTestOnMock::CheckMessageIov(const std::shared_ptr<RdmaMessage>& message,
                                                       int                                 i,
                                                       void*                               buf,
                                                       int                                 len,
                                                       RdmaMessage::BufferMRStatus         mrStatus,
                                                       RdmaMempool::DeviceType             device,
                                                       RdmaMempool::DeviceType             remoteDevice) {
    auto iov = message->GetIov(i);
    ASSERT_TRUE(iov != nullptr);
    if (buf) {
        EXPECT_EQ(buf, iov->buf.get());
    }
    EXPECT_EQ(len, iov->len);
    EXPECT_EQ(mrStatus, iov->mrStatus);
    EXPECT_EQ(device, iov->device);
    EXPECT_EQ(remoteDevice, iov->remoteDevice);
}

// 验证RdmaMessage无法识别iov状态返回失败的处理. 现在没有返回失败的情况
TEST_F(SimpleMessageConverterTestOnMock, testFromRdmaMessage_verifyIovStatusFailed) {}

// 验证对于Mr过的CPUMemory的处理, 预期buf直接加入xmessage
TEST_F(SimpleMessageConverterTestOnMock, testFromRdmaMessage_MrCPUMemory_Success) {
    auto message = std::make_shared<RdmaMessage>();
    message->SetMempool(_mempool);

    auto cudaUtil = new MockCudaUtil();
    message->SetCudaUtill(cudaUtil);

    // 构造message
    std::string strIov;
    strIov.assign(100, 'a');
    std::shared_ptr<void> sharedStrIov((void*)(strIov.c_str()), [](void* p) {});
    message->AddIov(sharedStrIov, 100, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::CPU);

    ::accl::barex::XMessage xmessage;
    SimpleMessageConverter  converter;
    ASSERT_TRUE(converter.ConvertFromRdmaMessage(message.get(), xmessage));

    EXPECT_EQ(1, xmessage.IovCount());
    CheckIov(xmessage.Get(0), sharedStrIov.get(), 100, accl::barex::CPU, accl::barex::CPU);
}

// 验证对于没有Mr过的CPUMemory的处理, 预期buf memcpy到mr的内存后加入xmessage
TEST_F(SimpleMessageConverterTestOnMock, testFromRdmaMessage_UnMrCPUMemory_Success) {
    auto message = std::make_shared<RdmaMessage>();
    message->SetMempool(_mempool);

    auto cudaUtil = new MockCudaUtil();
    message->SetCudaUtill(cudaUtil);

    // 构造message
    std::string strIov;
    strIov.assign(100, 'a');
    std::shared_ptr<void> sharedStrIov((void*)(strIov.c_str()), [](void* p) {});
    message->AddIov(sharedStrIov, 100, RdmaMessage::BMS_NOT_MR, RdmaMempool::CPU, RdmaMempool::CPU);

    // UnMr的Message会申请新的对象,并将原有的数据Copy出来.
    void* buffer = malloc(100);
    ASSERT_TRUE(buffer != nullptr);

    EXPECT_CALL(*_mockXMempool, AllocBuffer(100, ::accl::barex::CPU, 0, nullptr)).WillOnce(Return(buffer));
    EXPECT_CALL(*_mockXMempool, ReleaseBuffer(buffer, ::accl::barex::CPU))
        .WillOnce(Return(::accl::barex::BAREX_SUCCESS));

    ::accl::barex::XMessage xmessage;
    SimpleMessageConverter  converter;
    ASSERT_TRUE(converter.ConvertFromRdmaMessage(message.get(), xmessage));

    EXPECT_EQ(1, xmessage.IovCount());
    CheckIov(xmessage.Get(0), buffer, 100, accl::barex::CPU, accl::barex::CPU);

    ASSERT_EQ('a', ((char*)buffer)[0]);

    free(buffer);
}

// 验证对于没有Mr过的CPUMemory的处理, 预期buf memcpy到mr的内存后加入xmessage, 验证申请mr内存失败的处理
TEST_F(SimpleMessageConverterTestOnMock, testFromRdmaMessage_UnMrCPUMemory_AllocateBufferFailed) {
    auto message = std::make_shared<RdmaMessage>();
    message->SetMempool(_mempool);

    auto cudaUtil = new MockCudaUtil();
    message->SetCudaUtill(cudaUtil);

    std::string strIov;
    strIov.assign(100, 'a');
    std::shared_ptr<void> sharedStrIov((void*)(strIov.c_str()), [](void* p) {});
    message->AddIov(sharedStrIov, 100, RdmaMessage::BMS_NOT_MR, RdmaMempool::CPU, RdmaMempool::CPU);

    EXPECT_CALL(*_mockXMempool, AllocBuffer(100, ::accl::barex::CPU, 0, nullptr)).WillOnce(Return(nullptr));

    SimpleMessageConverter converter;

    ::accl::barex::XMessage xmessage;
    EXPECT_FALSE(converter.ConvertFromRdmaMessage(message.get(), xmessage));
}

// 验证对于未知状态的CPUMemory的处理, 需要先到mempool检查所属的device和mrStatus
TEST_F(SimpleMessageConverterTestOnMock, testFromRdmaMessage_UnknownMrCPUMemory_Success) {
    auto message = std::make_shared<RdmaMessage>();
    message->SetMempool(_mempool);

    auto cudaUtil = new MockCudaUtil();
    message->SetCudaUtill(cudaUtil);

    // 构造message
    std::string strIov;
    strIov.assign(100, 'a');
    std::shared_ptr<void> sharedStrIov((void*)(strIov.c_str()), [](void* p) {});
    message->AddIov(sharedStrIov, 100, RdmaMessage::BMS_UNKNOWN, RdmaMempool::CPU, RdmaMempool::CPU);

    // UnMr的Message会申请新的对象,并将原有的数据Copy出来.
    void* buffer = malloc(100);
    ASSERT_TRUE(buffer != nullptr);

    EXPECT_CALL(*_mockXMempool, AllocBuffer(100, ::accl::barex::CPU, 0, nullptr)).WillOnce(Return(buffer));
    EXPECT_CALL(*_mockXMempool, ReleaseBuffer(buffer, ::accl::barex::CPU))
        .WillOnce(Return(::accl::barex::BAREX_SUCCESS));

    // mock as not mr
    EXPECT_CALL(*_mockXMempool, IsValid(sharedStrIov.get(), ::accl::barex::CPU, false)).WillOnce(Return(false));
    EXPECT_CALL(*_mockXMempool, IsValid(sharedStrIov.get(), ::accl::barex::GPU, false)).WillOnce(Return(false));
    EXPECT_CALL(*cudaUtil, IsGPUMemory(sharedStrIov.get())).WillOnce(Return(false));

    ::accl::barex::XMessage xmessage;
    SimpleMessageConverter  converter;
    ASSERT_TRUE(converter.ConvertFromRdmaMessage(message.get(), xmessage));

    EXPECT_EQ(1, xmessage.IovCount());
    CheckIov(xmessage.Get(0), buffer, 100, accl::barex::CPU, accl::barex::CPU);

    ASSERT_EQ('a', ((char*)buffer)[0]);

    free(buffer);
}

// 验证对于Mr过的GPUMemory的处理, 预期buf直接加入xmessage
TEST_F(SimpleMessageConverterTestOnMock, testFromRdmaMessage_MrGPUMemory_Success) {
    auto message = std::make_shared<RdmaMessage>();
    message->SetMempool(_mempool);

    auto cudaUtil = new MockCudaUtil();
    message->SetCudaUtill(cudaUtil);

    // 构造message
    std::string strIov;
    strIov.assign(100, 'a');
    std::shared_ptr<void> sharedGPUIov((void*)(strIov.c_str()), [](void* p) {});
    message->AddIov(sharedGPUIov, 100, RdmaMessage::BMS_MR, RdmaMempool::GPU, RdmaMempool::CPU);

    ::accl::barex::XMessage xmessage;
    SimpleMessageConverter  converter;
    ASSERT_TRUE(converter.ConvertFromRdmaMessage(message.get(), xmessage));

    EXPECT_EQ(1, xmessage.IovCount());
    CheckIov(xmessage.Get(0), sharedGPUIov.get(), 100, accl::barex::GPU, accl::barex::CPU);
}

// 验证对于没有Mr过的GPUMemory的处理, 申请Mr的显存失败, 返回失败
TEST_F(SimpleMessageConverterTestOnMock, testFromRdmaMessage_UnMrGPUMemory_AllocSharedBufferFailed) {
    auto message = std::make_shared<RdmaMessage>();
    message->SetMempool(_mempool);

    auto cudaUtil = new MockCudaUtil();
    message->SetCudaUtill(cudaUtil);

    // 构造message
    std::string strIov;
    strIov.assign(100, 'a');
    std::shared_ptr<void> sharedGPUIov((void*)(strIov.c_str()), [](void* p) {});
    message->AddIov(sharedGPUIov, 100, RdmaMessage::BMS_NOT_MR, RdmaMempool::GPU, RdmaMempool::CPU);

    EXPECT_CALL(*_mockXMempool, AllocBuffer(100, ::accl::barex::GPU, 0, nullptr)).WillOnce(Return(nullptr));

    ::accl::barex::XMessage xmessage;
    SimpleMessageConverter  converter;
    ASSERT_FALSE(converter.ConvertFromRdmaMessage(message.get(), xmessage));
    EXPECT_EQ(0, xmessage.IovCount());
}

// 验证对于没有Mr过的GPUMemory的处理, 申请Mr过的显存, cudaMemcpy失败
TEST_F(SimpleMessageConverterTestOnMock, testFromRdmaMessage_UnMrGPUMemory_CudaMemcpyFailed) {
    auto message = std::make_shared<RdmaMessage>();
    message->SetMempool(_mempool);

    auto cudaUtil = new MockCudaUtil();
    message->SetCudaUtill(cudaUtil);

    // 构造message
    std::string strIov;
    strIov.assign(100, 'a');
    std::shared_ptr<void> sharedGPUIov((void*)(strIov.c_str()), [](void* p) {});
    message->AddIov(sharedGPUIov, 100, RdmaMessage::BMS_NOT_MR, RdmaMempool::GPU, RdmaMempool::CPU);

    // UnMr的Message会申请新的对象,并将原有的数据Copy出来.
    void* buffer = malloc(100);
    ASSERT_TRUE(buffer != nullptr);

    EXPECT_CALL(*_mockXMempool, AllocBuffer(100, ::accl::barex::GPU, 0, nullptr)).WillOnce(Return(buffer));
    EXPECT_CALL(*_mockXMempool, ReleaseBuffer(buffer, ::accl::barex::GPU))
        .WillOnce(Return(::accl::barex::BAREX_SUCCESS));

    EXPECT_CALL(*cudaUtil, CudaMemcpyGPU2GPU(buffer, sharedGPUIov.get(), 100)).WillOnce(Return(false));

    ::accl::barex::XMessage xmessage;
    SimpleMessageConverter  converter;
    ASSERT_FALSE(converter.ConvertFromRdmaMessage(message.get(), xmessage));
    EXPECT_EQ(0, xmessage.IovCount());

    free(buffer);
}

// 验证对于没有Mr过的GPUMemory的处理, 申请Mr过的显存, cudaMemcpy成功, 返回成功
TEST_F(SimpleMessageConverterTestOnMock, testFromRdmaMessage_UnMrGPUMemory_Success) {
    auto message = std::make_shared<RdmaMessage>();
    message->SetMempool(_mempool);

    auto cudaUtil = new MockCudaUtil();
    message->SetCudaUtill(cudaUtil);

    // 构造message
    std::string strIov;
    strIov.assign(100, 'a');
    std::shared_ptr<void> sharedGPUIov((void*)(strIov.c_str()), [](void* p) {});
    message->AddIov(sharedGPUIov, 100, RdmaMessage::BMS_NOT_MR, RdmaMempool::GPU, RdmaMempool::CPU);

    // UnMr的Message会申请新的对象,并将原有的数据Copy出来.
    void* buffer = malloc(100);
    ASSERT_TRUE(buffer != nullptr);

    EXPECT_CALL(*_mockXMempool, AllocBuffer(100, ::accl::barex::GPU, 0, nullptr)).WillOnce(Return(buffer));
    EXPECT_CALL(*_mockXMempool, ReleaseBuffer(buffer, ::accl::barex::GPU))
        .WillOnce(Return(::accl::barex::BAREX_SUCCESS));

    EXPECT_CALL(*cudaUtil, CudaMemcpyGPU2GPU(buffer, sharedGPUIov.get(), 100)).WillOnce(Return(true));

    ::accl::barex::XMessage xmessage;
    SimpleMessageConverter  converter;
    ASSERT_TRUE(converter.ConvertFromRdmaMessage(message.get(), xmessage));

    EXPECT_EQ(1, xmessage.IovCount());
    CheckIov(xmessage.Get(0), buffer, 100, accl::barex::GPU, accl::barex::CPU);

    free(buffer);
}

// 验证对于未知状态的GPUMemory的处理, 先检查状态, 当做没有mr的GPU处理
TEST_F(SimpleMessageConverterTestOnMock, testFromRdmaMessage_UnknownMrGPUMemory_Success) {
    auto message = std::make_shared<RdmaMessage>();
    message->SetMempool(_mempool);

    auto cudaUtil = new MockCudaUtil();
    message->SetCudaUtill(cudaUtil);

    // 构造message
    std::string strIov;
    strIov.assign(100, 'a');
    std::shared_ptr<void> sharedGPUIov((void*)(strIov.c_str()), [](void* p) {});
    message->AddIov(sharedGPUIov, 100, RdmaMessage::BMS_NOT_MR, RdmaMempool::GPU, RdmaMempool::CPU);

    // UnMr的Message会申请新的对象,并将原有的数据Copy出来.
    void* buffer = malloc(100);
    ASSERT_TRUE(buffer != nullptr);

    EXPECT_CALL(*_mockXMempool, AllocBuffer(100, ::accl::barex::GPU, 0, nullptr)).WillOnce(Return(buffer));
    EXPECT_CALL(*_mockXMempool, ReleaseBuffer(buffer, ::accl::barex::GPU))
        .WillOnce(Return(::accl::barex::BAREX_SUCCESS));

    EXPECT_CALL(*cudaUtil, CudaMemcpyGPU2GPU(buffer, sharedGPUIov.get(), 100)).WillOnce(Return(true));

    ::accl::barex::XMessage xmessage;
    SimpleMessageConverter  converter;
    ASSERT_TRUE(converter.ConvertFromRdmaMessage(message.get(), xmessage));

    EXPECT_EQ(1, xmessage.IovCount());
    CheckIov(xmessage.Get(0), buffer, 100, accl::barex::GPU, accl::barex::CPU);

    free(buffer);
}

#ifdef GOOGLE_CUDA
// 验证对于未知状态的GPUMemory的处理, 先检查状态, 当做没有mr的GPU处理, 不MockCudaUtil, 直接用GPU
TEST_F(SimpleMessageConverterTestOnMock, testFromRdmaMessage_UnknownMrGPUMemory_Success_OnGPU) {
    auto message = std::make_shared<RdmaMessage>();
    message->SetMempool(_mempool);

    // 构造message
    void* gpuBuffer = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&gpuBuffer, 100));
    ASSERT_EQ(cudaSuccess, cudaMemset(gpuBuffer, 'a', 100));
    std::shared_ptr<void> sharedGPUIov(gpuBuffer, [](void* p) {});
    message->AddIov(sharedGPUIov, 100, RdmaMessage::BMS_NOT_MR, RdmaMempool::GPU, RdmaMempool::CPU);

    // UnMr的Message会申请新的对象,并将原有的数据Copy出来.
    void* buffer = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&buffer, 100));

    EXPECT_CALL(*_mockXMempool, AllocBuffer(100, ::accl::barex::GPU, 0, nullptr)).WillOnce(Return(buffer));
    EXPECT_CALL(*_mockXMempool, ReleaseBuffer(buffer, ::accl::barex::GPU))
        .WillOnce(Return(::accl::barex::BAREX_SUCCESS));

    ::accl::barex::XMessage xmessage;
    SimpleMessageConverter  converter;
    ASSERT_TRUE(converter.ConvertFromRdmaMessage(message.get(), xmessage));

    EXPECT_EQ(1, xmessage.IovCount());
    CheckIov(xmessage.Get(0), buffer, 100, accl::barex::GPU, accl::barex::CPU);

    CudaUtil cudaUtil;
    // verify content correct
    void* dst = malloc(100);
    ASSERT_TRUE(dst != nullptr);
    ASSERT_TRUE(cudaUtil.CudaMemcpyGPU2CPU(dst, buffer, 100));
    ASSERT_EQ('a', ((char*)dst)[0]);
    free(dst);
    ASSERT_EQ(cudaSuccess, cudaFree(gpuBuffer));
    ASSERT_EQ(cudaSuccess, cudaFree(buffer));
}
#endif

// 整体验证, Pack和Unpack
TEST_F(SimpleMessageConverterTestOnMock, testFromRdmaMessageAndTo_Success) {
    _mockXMempool->MockOnAnyCallAllocAndReleaseBuffer();

    auto message = std::make_shared<RdmaMessage>();
    message->SetMempool(_mempool);

    std::string strIov;
    strIov.assign(100, 'a');
    std::shared_ptr<void> sharedStrIov((void*)(strIov.c_str()), [](void* p) {});
    message->AddIov(sharedStrIov, 100, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::CPU);

    auto cpu2CPUBuffer = message->AllocateSharedSendBuffer(101, RdmaMempool::CPU);
    message->AddIov(cpu2CPUBuffer, 101, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::CPU);

    auto cpu2GPUBuffer = message->AllocateSharedSendBuffer(102, RdmaMempool::CPU);
    message->AddIov(cpu2GPUBuffer, 102, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::GPU);

    auto gpu2CPUBuffer = message->AllocateSharedSendBuffer(103, RdmaMempool::GPU);
    message->AddIov(gpu2CPUBuffer, 103, RdmaMessage::BMS_MR, RdmaMempool::GPU, RdmaMempool::CPU);

    auto gpu2GPUBuffer = message->AllocateSharedSendBuffer(104, RdmaMempool::GPU);
    message->AddIov(gpu2GPUBuffer, 104, RdmaMessage::BMS_MR, RdmaMempool::GPU, RdmaMempool::GPU);

    SimpleMessageConverter converter;

    ::accl::barex::XMessage xmessage;
    ASSERT_TRUE(converter.ConvertFromRdmaMessage(message.get(), xmessage));

    EXPECT_EQ(5, xmessage.IovCount());
    CheckIov(xmessage.Get(0), nullptr, 100, ::accl::barex::CPU, ::accl::barex::CPU);
    CheckIov(xmessage.Get(1), cpu2CPUBuffer.get(), 101, ::accl::barex::CPU, ::accl::barex::CPU);
    CheckIov(xmessage.Get(2), cpu2GPUBuffer.get(), 102, ::accl::barex::CPU, ::accl::barex::GPU);
    CheckIov(xmessage.Get(3), gpu2CPUBuffer.get(), 103, ::accl::barex::GPU, ::accl::barex::CPU);
    CheckIov(xmessage.Get(4), gpu2GPUBuffer.get(), 104, ::accl::barex::GPU, ::accl::barex::GPU);

    auto message2 = std::make_shared<RdmaMessage>();
    ASSERT_TRUE(converter.ConvertFromXMessage(message2.get(), xmessage));

    EXPECT_EQ(5, message2->GetIovSize());
    CheckMessageIov(message, 0, nullptr, 100, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::CPU);
    CheckMessageIov(message, 1, nullptr, 101, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::CPU);
    CheckMessageIov(message, 2, nullptr, 102, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::GPU);
    CheckMessageIov(message, 3, nullptr, 103, RdmaMessage::BMS_MR, RdmaMempool::GPU, RdmaMempool::CPU);
    CheckMessageIov(message, 4, nullptr, 104, RdmaMessage::BMS_MR, RdmaMempool::GPU, RdmaMempool::GPU);
}

ARPC_END_NAMESPACE(arpc);

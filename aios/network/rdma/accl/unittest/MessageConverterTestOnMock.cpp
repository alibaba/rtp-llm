#include "unittest/unittest.h"

// force unitetst header first, change private class member
#include "aios/network/rdma/accl/MessageConverter.h"
#include "aios/network/rdma/accl/unittest/mock/MockCudaUtil.h"
#include "aios/network/rdma/accl/unittest/mock/MockXSimpleMempool.h"

#ifdef GOOGLE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

using namespace ::testing;
ARPC_BEGIN_NAMESPACE(arpc);

class MessageConverterTestOnMock: public TESTBASE {
public:
    void setUp();
    void tearDown();

protected:
    bool AppendIov(RdmaMessage* message, RdmaMempool::DeviceType device);
    void CheckIovs(::accl::barex::XMessage&                                                                 xmessage,
                   std::vector<std::tuple<int64_t, ::accl::barex::device_type, ::accl::barex::device_type>> iovs);
    void
    CheckDesc(::accl::barex::XMessage& xmessage, int descCnt, std::vector<std::tuple<int64_t, int64_t, int64_t>> descs);
    ::accl::barex::XMessage CopyXMessage(::accl::barex::XMessage& message);
    void                    TestMessageEqual(RdmaMessage* message1, RdmaMessage* message2);

protected:
    std::shared_ptr<RdmaMempool>                       _mempool;
    std::shared_ptr<::accl::barex::MockXSimpleMempool> _mockXMempool;
};
void MessageConverterTestOnMock::setUp() {
    _mockXMempool = std::make_shared<::accl::barex::MockXSimpleMempool>();
    EXPECT_CALL(*_mockXMempool, Shutdown()).WillOnce(Return(::accl::barex::BAREX_SUCCESS));
    EXPECT_CALL(*_mockXMempool, WaitStop()).WillOnce(Return(::accl::barex::BAREX_SUCCESS));

    _mempool = std::make_shared<RdmaMempool>();
    _mempool->_curState.store(RdmaMempool::INIT_SUCCESS);
    _mempool->_xmempool = std::dynamic_pointer_cast<::accl::barex::XSimpleMempool>(_mockXMempool);
}

void MessageConverterTestOnMock::tearDown() {}

bool MessageConverterTestOnMock::AppendIov(RdmaMessage* message, RdmaMempool::DeviceType device) {
    auto buf = message->AllocateSharedSendBuffer(10);
    if (!buf) {
        return false;
    }
    message->AddIov(buf, 10, RdmaMessage::BMS_MR, RdmaMempool::CPU, device);

    buf = message->AllocateSharedSendBuffer(102);
    if (!buf) {
        return false;
    }
    message->AddIov(buf, 102, RdmaMessage::BMS_MR, RdmaMempool::CPU, device);

    buf = message->AllocateSharedSendBuffer(1024);
    if (!buf) {
        return false;
    }
    message->AddIov(buf, 1024, RdmaMessage::BMS_MR, RdmaMempool::CPU, device);

    buf = message->AllocateSharedSendBuffer(10240);
    if (!buf) {
        return false;
    }
    message->AddIov(buf, 10240, RdmaMessage::BMS_MR, RdmaMempool::CPU, device);
    return true;
}

void MessageConverterTestOnMock::CheckIovs(
    ::accl::barex::XMessage&                                                                 xmessage,
    std::vector<std::tuple<int64_t, ::accl::barex::device_type, ::accl::barex::device_type>> iovs) {
    ASSERT_EQ(xmessage.IovCount(), iovs.size());
    for (int i = 0; i < xmessage.IovCount(); i++) {
        auto iov          = xmessage.Get(i);
        auto len          = std::get<0>(iovs[i]);
        auto device       = std::get<1>(iovs[i]);
        auto remoteDevice = std::get<2>(iovs[i]);
        EXPECT_EQ(iov.iov_len, len);
        EXPECT_EQ(iov.device, device);
        EXPECT_EQ(iov.remote_device, remoteDevice);
    }
}

void MessageConverterTestOnMock::CheckDesc(::accl::barex::XMessage&                           xmessage,
                                           int                                                descCnt,
                                           std::vector<std::tuple<int64_t, int64_t, int64_t>> descs) {
    ASSERT_TRUE(xmessage.IovCount() > 0);
    auto iov0 = xmessage.Get(0);
    ASSERT_GE(iov0.iov_len, sizeof(struct SerializeDescriptor));

    auto desc = reinterpret_cast<struct SerializeDescriptor*>(iov0.iov_base);
    ASSERT_EQ(0x101010, desc->magic);
    ASSERT_EQ(descCnt, desc->iovCount);

    auto descIovs = reinterpret_cast<SerializeIovDescriptor*>((char*)(iov0.iov_base) + sizeof(SerializeDescriptor));

    int cpuAlign = 64;
    int gpuAlign = 64;
    for (int i = 0; i < descCnt; i++) {
        ASSERT_GE(iov0.iov_len, sizeof(SerializeDescriptor) + i * sizeof(SerializeIovDescriptor));
        auto descIov = descIovs[i];
        EXPECT_EQ(std::get<0>(descs[i]), descIov.iovIndex);
        EXPECT_EQ(std::get<1>(descs[i]), descIov.len);
        EXPECT_EQ(std::get<2>(descs[i]), descIov.offset);

        auto iov   = xmessage.Get(descIov.iovIndex);
        auto align = iov.device == ::accl::barex::CPU ? cpuAlign : gpuAlign;
        ASSERT_TRUE(((int64_t)iov.iov_base + descIov.offset) % align == 0);
    }
}

::accl::barex::XMessage MessageConverterTestOnMock::CopyXMessage(::accl::barex::XMessage& message) {
    ::accl::barex::XMessage xmessage;
    for (int i = 0; i < message.IovCount(); i++) {
        auto iov = message.Get(i);
        auto buf = _mempool->AllocBuffer(iov.iov_len);
        memcpy(buf, iov.iov_base, iov.iov_len);
        xmessage.Add(buf, iov.iov_len, iov.device, iov.remote_device);
    }
    return xmessage;
}

void MessageConverterTestOnMock::TestMessageEqual(RdmaMessage* message1, RdmaMessage* message2) {
    int cpuAlign = 64;
    int gpuAlign = 64;
    ASSERT_GE(message1->GetIovSize(), message2->GetIovSize());  // message1 contains converter alloc memory
    for (int i = 0; i < message2->GetIovSize(); i++) {
        auto iov1 = message1->GetIov(i);
        auto iov2 = message2->GetIov(i);
        EXPECT_EQ(iov1->len, iov2->len);
        EXPECT_EQ(iov1->mrStatus, iov2->mrStatus);
        EXPECT_EQ(iov1->device, iov2->device);
        EXPECT_EQ(iov1->remoteDevice, iov2->remoteDevice);

        auto align = iov2->device == RdmaMempool::CPU ? cpuAlign : gpuAlign;
        ASSERT_TRUE(((int64_t)(iov2->buf.get())) % align == 0);
    }
}

// 验证发送请求时要发到远端的未知Memory, 直接失败
TEST_F(MessageConverterTestOnMock, testConvertFromRdmaMessage_GenerateDescriptor_ToRemoteUnknownDevice) {
    _mockXMempool->MockOnCallAllocAndReleaseBuffer(1024, ::accl::barex::CPU, 0);

    RdmaMessage* message = new RdmaMessage();
    message->SetMempool(_mempool);

    auto cudaUtil = new MockCudaUtil();
    message->SetCudaUtill(cudaUtil);

    auto buf = message->AllocateSharedSendBuffer(1024);
    message->AddIov(buf, 1024, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::CPU);
    message->AddIov(buf, 1024, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::UNKNOWN);

    MessageConverter        converter;
    ::accl::barex::XMessage xmessage;
    EXPECT_FALSE(converter.ConvertFromRdmaMessage(message, xmessage));
    EXPECT_EQ(0, xmessage.IovCount());

    delete message;
}

// 验证分配desc需要的内存失败
TEST_F(MessageConverterTestOnMock, testConvertFromRdmaMessage_PackMessage_AllocatePackedCPUFailed) {
    _mockXMempool->MockOnCallAllocAndReleaseBuffer(1024, ::accl::barex::CPU, 0);
    EXPECT_CALL(*_mockXMempool, AllocBuffer(1088, ::accl::barex::CPU, 0, nullptr)).WillOnce(Return(nullptr));

    // allocate cpu buffer failed
    RdmaMessage* message = new RdmaMessage();
    message->SetMempool(_mempool);

    auto cudaUtil = new MockCudaUtil();
    message->SetCudaUtill(cudaUtil);

    auto buf = message->AllocateSharedSendBuffer(1024);
    message->AddIov(buf, 1024, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::CPU);

    MessageConverter        converter;
    ::accl::barex::XMessage xmessage;
    EXPECT_FALSE(converter.ConvertFromRdmaMessage(message, xmessage));

    delete message;
}

// 验证分配发到远端GPU的Packed CPUMemory失败
TEST_F(MessageConverterTestOnMock, testConvertFromRdmaMessage_PackMessage_AllocatePackedGPUFailed) {
    _mockXMempool->MockOnCallAllocAndReleaseBuffer(1024, ::accl::barex::CPU, 0);
    _mockXMempool->MockOnCallAllocAndReleaseBuffer(4096, ::accl::barex::CPU, 0);
    _mockXMempool->MockOnCallAllocAndReleaseBuffer(1088, ::accl::barex::CPU, 0);
    EXPECT_CALL(*_mockXMempool, AllocBuffer(2048, ::accl::barex::CPU, 0, nullptr)).WillOnce(Return(nullptr));

    // allocate cpu buffer failed
    RdmaMessage* message = new RdmaMessage();
    message->SetMempool(_mempool);

    auto cudaUtil = new MockCudaUtil();
    message->SetCudaUtill(cudaUtil);

    auto buf = message->AllocateSharedSendBuffer(1024);
    message->AddIov(buf, 1024, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::CPU);

    // add iov len 2048 solve mock conflict
    auto gpuBuf = message->AllocateSharedSendBuffer(4096);
    message->AddIov(gpuBuf, 2048, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::GPU);

    MessageConverter        converter;
    ::accl::barex::XMessage xmessage;
    EXPECT_FALSE(converter.ConvertFromRdmaMessage(message, xmessage));

    delete message;
}

// 验证对于没有Mr过的GPUMemory的处理, 申请Mr的显存失败, 返回失败
TEST_F(MessageConverterTestOnMock, testConvertFromRdmaMessage_PackMessage_UnMrGPUMemory_AllocSharedBufferFailed) {
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
    MessageConverter        converter;
    ASSERT_FALSE(converter.ConvertFromRdmaMessage(message.get(), xmessage));
    EXPECT_EQ(0, xmessage.IovCount());
}

// 验证对于没有Mr过的GPUMemory的处理, 申请Mr过的显存, cudaMemcpy失败
TEST_F(MessageConverterTestOnMock, testConvertFromRdmaMessage_PackMessage_UnMrGPUMemory_cudaMemcpyFailed) {
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
    _mockXMempool->MockOnCallAllocAndReleaseBuffer(100, ::accl::barex::GPU, 0);
    EXPECT_CALL(*cudaUtil, CudaMemcpyGPU2GPU(_, sharedGPUIov.get(), 100)).WillOnce(Return(false));

    ::accl::barex::XMessage xmessage;
    MessageConverter        converter;
    ASSERT_FALSE(converter.ConvertFromRdmaMessage(message.get(), xmessage));
    EXPECT_EQ(0, xmessage.IovCount());
}

// 验证对于没有Mr过的GPUMemory的处理, 申请Mr过的显存, 成功
TEST_F(MessageConverterTestOnMock, testConvertFromRdmaMessage_PackMessage_UnMrGPUMemory_Success) {
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
    _mockXMempool->MockOnCallAllocAndReleaseBuffer(64, ::accl::barex::CPU, 0);
    _mockXMempool->MockOnCallAllocAndReleaseBuffer(100, ::accl::barex::GPU, 0);

    EXPECT_CALL(*cudaUtil, CudaMemcpyGPU2GPU(_, sharedGPUIov.get(), 100)).WillOnce(Return(true));

    ::accl::barex::XMessage xmessage;
    MessageConverter        converter;
    ASSERT_TRUE(converter.ConvertFromRdmaMessage(message.get(), xmessage));

    EXPECT_EQ(2, xmessage.IovCount());
    CheckIovs(xmessage, {{64, ::accl::barex::CPU, ::accl::barex::CPU}, {100, ::accl::barex::GPU, ::accl::barex::CPU}});
    CheckDesc(xmessage, 1, {{1, 100, 0}});
}

#ifdef GOOGLE_CUDA
// 验证对于没有Mr过的GPUMemory的处理, 申请Mr过的显存, 成功, runOnGPU
TEST_F(MessageConverterTestOnMock, testConvertFromRdmaMessage_PackMessage_UnMrGPUMemory_Success_OnGPU) {
    auto message = std::make_shared<RdmaMessage>();
    message->SetMempool(_mempool);

    // 构造message
    void* gpuBuffer = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&gpuBuffer, 100));
    ASSERT_EQ(cudaSuccess, cudaMemset(gpuBuffer, 'a', 100));
    std::shared_ptr<void> sharedGPUIov(gpuBuffer, [](void* p) {});
    message->AddIov(sharedGPUIov, 100, RdmaMessage::BMS_NOT_MR, RdmaMempool::GPU, RdmaMempool::CPU);

    // UnMr的Message会申请新的对象,并将原有的数据Copy出来.
    _mockXMempool->MockOnCallAllocAndReleaseBuffer(64, ::accl::barex::CPU, 0);
    void* buffer = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&buffer, 100));

    EXPECT_CALL(*_mockXMempool, AllocBuffer(100, ::accl::barex::GPU, 0, nullptr)).WillOnce(Return(buffer));
    EXPECT_CALL(*_mockXMempool, ReleaseBuffer(buffer, ::accl::barex::GPU))
        .WillOnce(Return(::accl::barex::BAREX_SUCCESS));

    ::accl::barex::XMessage xmessage;
    MessageConverter        converter;
    ASSERT_TRUE(converter.ConvertFromRdmaMessage(message.get(), xmessage));

    EXPECT_EQ(2, xmessage.IovCount());
    CheckIovs(xmessage, {{64, ::accl::barex::CPU, ::accl::barex::CPU}, {100, ::accl::barex::GPU, ::accl::barex::CPU}});
    CheckDesc(xmessage, 1, {{1, 100, 0}});

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

TEST_F(MessageConverterTestOnMock, testConvertFromXMessage_GetMempoolFailed) {
    RdmaMessage*            message = new RdmaMessage();
    MessageConverter        converter;
    ::accl::barex::XMessage xmessage;

    EXPECT_FALSE(converter.ConvertFromXMessage(message, xmessage));
    delete message;
}

TEST_F(MessageConverterTestOnMock, testConvertFromXMessage_EmptyXMessage) {
    RdmaMessage* message = new RdmaMessage();
    message->SetMempool(_mempool);
    ::accl::barex::XMessage xmessage;
    MessageConverter        converter;
    EXPECT_TRUE(converter.ConvertFromXMessage(message, xmessage));

    delete message;
}

// 解析Desc时, 拿到的buf长度不够, 直接返回失败
TEST_F(MessageConverterTestOnMock, testConvertFromXMessage_ValidMessageFailed_SDescLenFailed) {
    _mockXMempool->MockOnAnyCallAllocAndReleaseBuffer();

    RdmaMessage* message = new RdmaMessage();
    message->SetMempool(_mempool);

    auto                    buf = message->AllocateSendBuffer(10);
    ::accl::barex::XMessage xmessage;
    xmessage.Add(buf, 10, ::accl::barex::CPU, ::accl::barex::CPU);
    MessageConverter converter;
    EXPECT_FALSE(converter.ConvertFromXMessage(message, xmessage));

    delete message;
}

// 解析到的Desc magic不对, 校验失败返回false
TEST_F(MessageConverterTestOnMock, testConvertFromXMessage_ValidMessageFailed_ErrorMagic) {

    _mockXMempool->MockOnAnyCallAllocAndReleaseBuffer();
    RdmaMessage* message = new RdmaMessage();
    message->SetMempool(_mempool);

    auto                    len = sizeof(SerializeDescriptor);
    auto                    buf = message->AllocateSendBuffer(len);
    ::accl::barex::XMessage xmessage;
    xmessage.Add(buf, len, ::accl::barex::CPU, ::accl::barex::CPU);
    MessageConverter converter;
    EXPECT_FALSE(converter.ConvertFromXMessage(message, xmessage));

    delete message;
}

// 解析到的desc iov描述和长度对不上, 直接返回false
TEST_F(MessageConverterTestOnMock, testConvertFromXMessage_ValidMessageFailed_DescLenFailed) {
    _mockXMempool->MockOnAnyCallAllocAndReleaseBuffer();
    RdmaMessage* message = new RdmaMessage();
    message->SetMempool(_mempool);

    auto len       = sizeof(SerializeDescriptor);
    auto buf       = message->AllocateSendBuffer(len);
    auto desc      = reinterpret_cast<SerializeDescriptor*>(buf);
    desc->iovCount = 1000;
    desc->magic    = 0x101010;
    ::accl::barex::XMessage xmessage;
    xmessage.Add(buf, len, ::accl::barex::CPU, ::accl::barex::CPU);
    MessageConverter converter;
    EXPECT_FALSE(converter.ConvertFromXMessage(message, xmessage));

    delete message;
}

// 解析拿到的desc iov, index记录有问题, 直接返回false
TEST_F(MessageConverterTestOnMock, testConvertFromXMessage_ValidMessageFailed_IovIndexFailed) {
    _mockXMempool->MockOnAnyCallAllocAndReleaseBuffer();
    RdmaMessage* message = new RdmaMessage();
    message->SetMempool(_mempool);

    auto len       = sizeof(SerializeDescriptor) + sizeof(SerializeIovDescriptor);
    auto buf       = message->AllocateSendBuffer(len);
    auto desc      = reinterpret_cast<SerializeDescriptor*>(buf);
    desc->iovCount = 1;
    desc->magic    = 0x101010;

    auto iovDesc      = reinterpret_cast<SerializeIovDescriptor*>((char*)buf + sizeof(SerializeDescriptor));
    iovDesc->iovIndex = 102;
    iovDesc->offset   = 0;
    iovDesc->len      = 0;

    ::accl::barex::XMessage xmessage;
    xmessage.Add(buf, len, ::accl::barex::CPU, ::accl::barex::CPU);
    MessageConverter converter;
    EXPECT_FALSE(converter.ConvertFromXMessage(message, xmessage));

    delete message;
}

// 解析拿到的desc iov, len记录有问题, 直接返回false
TEST_F(MessageConverterTestOnMock, testConvertFromXMessage_ValidMessageFailed_IovLenFailed) {
    _mockXMempool->MockOnAnyCallAllocAndReleaseBuffer();
    RdmaMessage* message = new RdmaMessage();
    message->SetMempool(_mempool);

    auto len       = sizeof(SerializeDescriptor) + sizeof(SerializeIovDescriptor);
    auto buf       = message->AllocateSendBuffer(len);
    auto desc      = reinterpret_cast<SerializeDescriptor*>(buf);
    desc->iovCount = 1;
    desc->magic    = 0x101010;

    auto iovDesc      = reinterpret_cast<SerializeIovDescriptor*>((char*)buf + sizeof(SerializeDescriptor));
    iovDesc->iovIndex = 0;
    iovDesc->offset   = 10000;
    iovDesc->len      = 0;

    ::accl::barex::XMessage xmessage;
    xmessage.Add(buf, len, ::accl::barex::CPU, ::accl::barex::CPU);
    MessageConverter converter;
    EXPECT_FALSE(converter.ConvertFromXMessage(message, xmessage));
    delete message;
}

// 设置较小的pack size, 所有的iov都不需要pack
TEST_F(MessageConverterTestOnMock, testConvert_Success_NoPacked_CPU) {
    _mockXMempool->MockOnAnyCallAllocAndReleaseBuffer();

    RdmaMessage* message = new RdmaMessage();

    auto cudaUtil = new MockCudaUtil();
    message->SetCudaUtill(cudaUtil);

    ::accl::barex::XMessage xmessage;
    {
        message->SetMempool(_mempool);
        ASSERT_TRUE(AppendIov(message, RdmaMempool::CPU));

        MessageConverter converter;
        converter.SetSmallIovThreshold(1);
        ASSERT_TRUE(converter.ConvertFromRdmaMessage(message, xmessage));
        ASSERT_EQ(5, xmessage.IovCount());
        CheckIovs(xmessage,
                  {{128, ::accl::barex::CPU, ::accl::barex::CPU},
                   {10, ::accl::barex::CPU, ::accl::barex::CPU},
                   {102, ::accl::barex::CPU, ::accl::barex::CPU},
                   {1024, ::accl::barex::CPU, ::accl::barex::CPU},
                   {10240, ::accl::barex::CPU, ::accl::barex::CPU}});

        CheckDesc(xmessage, 4, {{1, 10, 0}, {2, 102, 0}, {3, 1024, 0}, {4, 10240, 0}});
        EXPECT_EQ(5, message->GetIovSize());
    }
    auto         xmessage2 = CopyXMessage(xmessage);
    RdmaMessage* message2  = new RdmaMessage();
    {
        message2->SetMempool(_mempool);
        MessageConverter converter2;
        ASSERT_TRUE(converter2.ConvertFromXMessage(message2, xmessage2));
        ASSERT_EQ(4, message2->GetIovSize());
        ASSERT_EQ(11376, message2->GetTotalIovLength());
    }
    TestMessageEqual(message, message2);

    delete message;
    delete message2;
}

// 设置较小的pack size, 所有的iov都需要pack
TEST_F(MessageConverterTestOnMock, testConvert_Success_AllPacked_CPU) {
    _mockXMempool->MockOnAnyCallAllocAndReleaseBuffer();

    RdmaMessage* message = new RdmaMessage();

    auto cudaUtil = new MockCudaUtil();
    message->SetCudaUtill(cudaUtil);

    ::accl::barex::XMessage xmessage;
    {
        message->SetMempool(_mempool);
        ASSERT_TRUE(AppendIov(message, RdmaMempool::CPU));

        MessageConverter converter;
        converter.SetSmallIovThreshold(20480);
        ASSERT_TRUE(converter.ConvertFromRdmaMessage(message, xmessage));
        ASSERT_EQ(1, xmessage.IovCount());
        CheckIovs(xmessage,
                  {
                      {11584, ::accl::barex::CPU, ::accl::barex::CPU},
                  });

        CheckDesc(xmessage, 4, {{0, 10, 128}, {0, 102, 192}, {0, 1024, 320}, {0, 10240, 1344}});
        EXPECT_EQ(5, message->GetIovSize());
    }
    auto         xmessage2 = CopyXMessage(xmessage);
    RdmaMessage* message2  = new RdmaMessage();
    {
        message2->SetMempool(_mempool);
        MessageConverter converter2;
        ASSERT_TRUE(converter2.ConvertFromXMessage(message2, xmessage2));
        ASSERT_EQ(4, message2->GetIovSize());
        ASSERT_EQ(11376, message2->GetTotalIovLength());
    }

    TestMessageEqual(message, message2);

    delete message;
    delete message2;
}

// 设置正常的的pack size, 部分iov pack 部分不pack
TEST_F(MessageConverterTestOnMock, testConvert_Success_MixedPacked_CPU) {
    _mockXMempool->MockOnAnyCallAllocAndReleaseBuffer();
    RdmaMessage* message = new RdmaMessage();

    auto cudaUtil = new MockCudaUtil();
    message->SetCudaUtill(cudaUtil);

    ::accl::barex::XMessage xmessage;
    {
        message->SetMempool(_mempool);
        ASSERT_TRUE(AppendIov(message, RdmaMempool::CPU));

        MessageConverter converter;
        converter.SetSmallIovThreshold(1000);
        ASSERT_TRUE(converter.ConvertFromRdmaMessage(message, xmessage));
        ASSERT_EQ(3, xmessage.IovCount());
        CheckIovs(xmessage,
                  {
                      {320, ::accl::barex::CPU, ::accl::barex::CPU},
                      {1024, ::accl::barex::CPU, ::accl::barex::CPU},
                      {10240, ::accl::barex::CPU, ::accl::barex::CPU},
                  });

        CheckDesc(xmessage, 4, {{0, 10, 128}, {0, 102, 192}, {1, 1024, 0}, {2, 10240, 0}});
        EXPECT_EQ(5, message->GetIovSize());
    }
    auto         xmessage2 = CopyXMessage(xmessage);
    RdmaMessage* message2  = new RdmaMessage();
    {
        message2->SetMempool(_mempool);
        MessageConverter converter2;
        ASSERT_TRUE(converter2.ConvertFromXMessage(message2, xmessage2));
        ASSERT_EQ(4, message2->GetIovSize());
        ASSERT_EQ(11376, message2->GetTotalIovLength());
    }
    TestMessageEqual(message, message2);
    delete message;
    delete message2;
}

TEST_F(MessageConverterTestOnMock, testConvert_Success_NoPacked_GPU) {
    _mockXMempool->MockOnAnyCallAllocAndReleaseBuffer();
    RdmaMessage* message = new RdmaMessage();

    auto cudaUtil = new MockCudaUtil();
    message->SetCudaUtill(cudaUtil);

    ::accl::barex::XMessage xmessage;
    {
        message->SetMempool(_mempool);
        ASSERT_TRUE(AppendIov(message, RdmaMempool::GPU));

        MessageConverter converter;
        converter.SetSmallIovThreshold(1);
        ASSERT_TRUE(converter.ConvertFromRdmaMessage(message, xmessage));
        ASSERT_EQ(5, xmessage.IovCount());
        CheckIovs(xmessage,
                  {{128, ::accl::barex::CPU, ::accl::barex::CPU},
                   {10, ::accl::barex::CPU, ::accl::barex::GPU},
                   {102, ::accl::barex::CPU, ::accl::barex::GPU},
                   {1024, ::accl::barex::CPU, ::accl::barex::GPU},
                   {10240, ::accl::barex::CPU, ::accl::barex::GPU}});

        CheckDesc(xmessage, 4, {{1, 10, 0}, {2, 102, 0}, {3, 1024, 0}, {4, 10240, 0}});
        EXPECT_EQ(5, message->GetIovSize());
    }

    auto         xmessage2 = CopyXMessage(xmessage);
    RdmaMessage* message2  = new RdmaMessage();
    {
        message2->SetMempool(_mempool);
        MessageConverter converter2;
        ASSERT_TRUE(converter2.ConvertFromXMessage(message2, xmessage2));
        ASSERT_EQ(4, message2->GetIovSize());
        ASSERT_EQ(11376, message2->GetTotalIovLength());
    }

    TestMessageEqual(message, message2);
    delete message;
    delete message2;
}

TEST_F(MessageConverterTestOnMock, testConvert_Success_AllPacked_GPU) {
    _mockXMempool->MockOnAnyCallAllocAndReleaseBuffer();

    RdmaMessage* message = new RdmaMessage();

    auto cudaUtil = new MockCudaUtil();
    message->SetCudaUtill(cudaUtil);

    ::accl::barex::XMessage xmessage;
    {
        message->SetMempool(_mempool);
        ASSERT_TRUE(AppendIov(message, RdmaMempool::GPU));

        MessageConverter converter;
        converter.SetSmallIovThreshold(20480);
        ASSERT_TRUE(converter.ConvertFromRdmaMessage(message, xmessage));
        ASSERT_EQ(2, xmessage.IovCount());
        CheckIovs(xmessage,
                  {
                      {128, ::accl::barex::CPU, ::accl::barex::CPU},
                      {11456, ::accl::barex::CPU, ::accl::barex::GPU},
                  });

        CheckDesc(xmessage, 4, {{1, 10, 0}, {1, 102, 64}, {1, 1024, 192}, {1, 10240, 1216}});
        EXPECT_EQ(6, message->GetIovSize());
    }
    auto         xmessage2 = CopyXMessage(xmessage);
    RdmaMessage* message2  = new RdmaMessage();
    {
        message2->SetMempool(_mempool);
        MessageConverter converter2;
        ASSERT_TRUE(converter2.ConvertFromXMessage(message2, xmessage2));
        ASSERT_EQ(4, message2->GetIovSize());
        ASSERT_EQ(11376, message2->GetTotalIovLength());
    }

    TestMessageEqual(message, message2);
    delete message;
    delete message2;
}

TEST_F(MessageConverterTestOnMock, testConvert_Success_MixedPacked_CPU_GPU) {
    _mockXMempool->MockOnAnyCallAllocAndReleaseBuffer();
    RdmaMessage* message = new RdmaMessage();

    auto cudaUtil = new MockCudaUtil();
    message->SetCudaUtill(cudaUtil);

    ::accl::barex::XMessage xmessage;
    {
        message->SetMempool(_mempool);
        ASSERT_TRUE(AppendIov(message, RdmaMempool::CPU));
        ASSERT_TRUE(AppendIov(message, RdmaMempool::GPU));

        MessageConverter converter;
        converter.SetSmallIovThreshold(1000);
        ASSERT_TRUE(converter.ConvertFromRdmaMessage(message, xmessage));
        ASSERT_EQ(6, xmessage.IovCount());
        CheckIovs(xmessage,
                  {
                      {448, ::accl::barex::CPU, ::accl::barex::CPU},
                      {192, ::accl::barex::CPU, ::accl::barex::GPU},
                      {1024, ::accl::barex::CPU, ::accl::barex::CPU},
                      {10240, ::accl::barex::CPU, ::accl::barex::CPU},
                      {1024, ::accl::barex::CPU, ::accl::barex::GPU},
                      {10240, ::accl::barex::CPU, ::accl::barex::GPU},
                  });

        CheckDesc(xmessage,
                  8,
                  {{0, 10, 256},
                   {0, 102, 320},
                   {2, 1024, 0},
                   {3, 10240, 0},
                   {1, 10, 0},
                   {1, 102, 64},
                   {4, 1024, 0},
                   {5, 10240, 0}});
        EXPECT_EQ(10, message->GetIovSize());
    }
    auto         xmessage2 = CopyXMessage(xmessage);
    RdmaMessage* message2  = new RdmaMessage();
    {
        message2->SetMempool(_mempool);
        MessageConverter converter2;
        ASSERT_TRUE(converter2.ConvertFromXMessage(message2, xmessage2));
        ASSERT_EQ(8, message2->GetIovSize());
        ASSERT_EQ(22752, message2->GetTotalIovLength());
    }
    TestMessageEqual(message, message2);
    delete message;
    delete message2;
}

ARPC_END_NAMESPACE(arpc);

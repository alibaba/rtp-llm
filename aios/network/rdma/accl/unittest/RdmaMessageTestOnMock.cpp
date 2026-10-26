#include "unittest/unittest.h"

// force unitetst header first, change private class member
#include "aios/network/arpc/arpc/metric/ClientRPCStats.h"
#include "aios/network/arpc/arpc/metric/ServerRPCStats.h"
#include "aios/network/rdma/accl/RdmaMessage.h"
#include "aios/network/rdma/accl/unittest/mock/MockCudaUtil.h"
#include "aios/network/rdma/accl/unittest/mock/MockXSimpleMempool.h"

using namespace ::testing;

ARPC_BEGIN_NAMESPACE(arpc);

class RdmaMessageTestOnMock: public TESTBASE {
public:
    void setUp();
    void tearDown();

protected:
    void                    CheckMessageIov(RdmaMessage&                message,
                                            int                         index,
                                            std::shared_ptr<void>       buf,
                                            uint32_t                    len,
                                            RdmaMessage::BufferMRStatus mrStatus,
                                            RdmaMempool::DeviceType     device,
                                            RdmaMempool::DeviceType     remoteDevice);
    ::accl::barex::XMessage CopyXMessage(::accl::barex::XMessage& message);

protected:
    std::shared_ptr<RdmaMempool>                       _mempool;
    std::shared_ptr<::accl::barex::MockXSimpleMempool> _mockXMempool;
};

void RdmaMessageTestOnMock::setUp() {
    _mockXMempool = std::make_shared<::accl::barex::MockXSimpleMempool>();
    EXPECT_CALL(*_mockXMempool, Shutdown()).WillOnce(Return(::accl::barex::BAREX_SUCCESS));
    EXPECT_CALL(*_mockXMempool, WaitStop()).WillOnce(Return(::accl::barex::BAREX_SUCCESS));

    _mempool = std::make_shared<RdmaMempool>();
    _mempool->_curState.store(RdmaMempool::INIT_SUCCESS);
    _mempool->_xmempool = std::dynamic_pointer_cast<::accl::barex::XSimpleMempool>(_mockXMempool);
}

void RdmaMessageTestOnMock::tearDown() {}

void RdmaMessageTestOnMock::CheckMessageIov(RdmaMessage&                message,
                                            int                         i,
                                            std::shared_ptr<void>       buf,
                                            uint32_t                    len,
                                            RdmaMessage::BufferMRStatus mrStatus,
                                            RdmaMempool::DeviceType     device,
                                            RdmaMempool::DeviceType     remoteDevice) {
    auto iov = message.GetIov(i);
    ASSERT_TRUE(iov != nullptr);
    EXPECT_EQ(buf, iov->buf);
    EXPECT_EQ(len, iov->len);
    EXPECT_EQ(mrStatus, iov->mrStatus);
    EXPECT_EQ(device, iov->device);
    EXPECT_EQ(remoteDevice, iov->remoteDevice);
}

::accl::barex::XMessage RdmaMessageTestOnMock::CopyXMessage(::accl::barex::XMessage& xmessage) {
    ::accl::barex::XMessage recvXMessage;
    for (int i = 0; i < xmessage.IovCount(); i++) {
        auto iov = xmessage.Get(i);
        auto buf = _mempool->AllocBuffer(iov.iov_len, RdmaMempool::ConvertFromAcclDeviceType(iov.device));
        memcpy(buf, iov.iov_base, iov.iov_len);
        recvXMessage.Add(buf, iov.iov_len, iov.device, iov.remote_device);
    }
    recvXMessage.SetUserFlag0(xmessage.GetUserFlag0());
    recvXMessage.SetUserFlag1(xmessage.GetUserFlag1());
    return recvXMessage;
}

TEST_F(RdmaMessageTestOnMock, testAllocateAndReleaseBuffer) {
    _mockXMempool->MockOnAnyCallAllocAndReleaseBuffer();

    RdmaMessage message;
    message.SetMempool(_mempool);
    auto bufLen = 1024;
    auto buf    = message.AllocateSendBuffer(bufLen);
    ASSERT_TRUE(buf != nullptr);
    ASSERT_TRUE(message.ReleaseSendBuffer(buf));
}

TEST_F(RdmaMessageTestOnMock, testSetRecvMessage_IovNull) {
    _mockXMempool->MockOnAnyCallAllocAndReleaseBuffer();

    RdmaMessage message;
    message.SetMempool(_mempool);
    auto buf = message.AllocateSharedSendBuffer(1025);
    ASSERT_TRUE(buf != nullptr);
    message.AddIov(buf, 1025, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::CPU);

    ::accl::barex::XMessage xmessage;
    EXPECT_FALSE(message.SetRecvMessage(xmessage));
    EXPECT_FALSE(message.IsFailedMessage());
}

TEST_F(RdmaMessageTestOnMock, testSetRecvMessage_FailedMessage) {
    _mockXMempool->MockOnAnyCallAllocAndReleaseBuffer();

    RdmaMessage rdmaMessage;
    rdmaMessage.SetMempool(_mempool);
    auto buf = rdmaMessage.AllocateSharedSendBuffer(1024);
    rdmaMessage.AddIov(buf, 1024, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::CPU);
    auto xmessage0 = rdmaMessage.GetMessage();
    auto xmessage  = CopyXMessage(xmessage0);
    xmessage.SetUserFlag0(101);
    xmessage.SetUserFlag1(1);

    RdmaMessage message;
    message.SetMempool(_mempool);

    EXPECT_TRUE(message.SetRecvMessage(xmessage));
    EXPECT_TRUE(message.IsFailedMessage());
    EXPECT_EQ(101, message.GetRequestId());
}

TEST_F(RdmaMessageTestOnMock, testSetRecvMessage_FailedMessage_Empty) {
    ::accl::barex::XMessage xmessage;
    xmessage.SetUserFlag0(102);
    xmessage.SetUserFlag1(1);

    RdmaMessage message;
    message.SetMempool(_mempool);

    EXPECT_TRUE(message.SetRecvMessage(xmessage));
    EXPECT_TRUE(message.IsFailedMessage());
    EXPECT_EQ(102, message.GetRequestId());
}

TEST_F(RdmaMessageTestOnMock, testSetRecvMessage_SuccessMessage) {
    _mockXMempool->MockOnAnyCallAllocAndReleaseBuffer();

    RdmaMessage rdmaMessage;
    rdmaMessage.SetMempool(_mempool);
    auto buf = rdmaMessage.AllocateSharedSendBuffer(1024);
    rdmaMessage.AddIov(buf, 1024, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::CPU);
    auto xmessage0 = rdmaMessage.GetMessage();
    auto xmessage  = CopyXMessage(xmessage0);
    xmessage.SetUserFlag0(101);
    xmessage.SetUserFlag1(0);

    RdmaMessage message;
    message.SetMempool(_mempool);

    EXPECT_TRUE(message.SetRecvMessage(xmessage));
    EXPECT_FALSE(message.IsFailedMessage());
    EXPECT_EQ(101, message.GetRequestId());
}

TEST_F(RdmaMessageTestOnMock, testSetRecvMessage_Success_Empty) {
    ::accl::barex::XMessage xmessage;
    xmessage.SetUserFlag0(102);
    xmessage.SetUserFlag1(0);

    RdmaMessage message;
    message.SetMempool(_mempool);

    EXPECT_TRUE(message.SetRecvMessage(xmessage));
    EXPECT_FALSE(message.IsFailedMessage());
    EXPECT_EQ(102, message.GetRequestId());
}

TEST_F(RdmaMessageTestOnMock, testAllocAndReleaseSharedBuffer) {
    _mockXMempool->MockOnAnyCallAllocAndReleaseBuffer();

    RdmaMessage message;
    message.SetMempool(_mempool);
    auto bufLen = 1024;

    auto sharedBuf = message.AllocateSharedSendBuffer(bufLen);
    auto charBuf   = std::static_pointer_cast<char>(sharedBuf);
    auto voidBuf   = std::static_pointer_cast<void>(charBuf);
    EXPECT_EQ(3, sharedBuf.use_count());
    EXPECT_EQ(3, charBuf.use_count());
    EXPECT_EQ(3, voidBuf.use_count());
}

TEST_F(RdmaMessageTestOnMock, testIovOps) {
    _mockXMempool->MockOnAnyCallAllocAndReleaseBuffer();

    RdmaMessage message;
    message.SetMempool(_mempool);
    EXPECT_EQ(0, message.GetIovSize());
    EXPECT_EQ(0, message.GetTotalIovLength());

    EXPECT_EQ(0, message.GetIovSize());
    EXPECT_EQ(0, message.GetTotalIovLength());

    // Allocate and add get steal iov
    auto buf1 = message.AllocateSharedSendBuffer(1024);
    ASSERT_TRUE(buf1 != nullptr);
    message.AddIov(buf1, 1024, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::CPU);
    EXPECT_EQ(1, message.GetIovSize());
    EXPECT_EQ(1024, message.GetTotalIovLength());

    auto buf2 = message.AllocateSharedSendBuffer(10);
    ASSERT_TRUE(buf2 != nullptr);
    message.AddIov(buf2, 10, RdmaMessage::BMS_NOT_MR, RdmaMempool::GPU, RdmaMempool::GPU);
    EXPECT_EQ(2, message.GetIovSize());
    EXPECT_EQ(1034, message.GetTotalIovLength());

    CheckMessageIov(message, 0, buf1, 1024, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::CPU);
    CheckMessageIov(message, 1, buf2, 10, RdmaMessage::BMS_NOT_MR, RdmaMempool::GPU, RdmaMempool::GPU);
    ASSERT_TRUE(message.GetIov(100) == nullptr);
}

TEST_F(RdmaMessageTestOnMock, testSetGetXMessage_FailedMessage) {
    _mockXMempool->MockOnAnyCallAllocAndReleaseBuffer();

    RdmaMessage message;
    message.SetMempool(_mempool);
    message.SetRequestId(101);
    message.SetFailedMessage(true);

    auto buf = message.AllocateSharedSendBuffer(1024);
    ASSERT_TRUE(buf != nullptr);
    message.AddIov(buf, 1024, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::CPU);
    memset(buf.get(), '#', 1024);

    auto xmessage     = message.GetMessage();
    auto recvXMessage = CopyXMessage(xmessage);

    RdmaMessage recvMessage;
    recvMessage.SetMempool(_mempool);
    ASSERT_TRUE(recvMessage.SetRecvMessage(recvXMessage));
    EXPECT_EQ(101, recvMessage.GetRequestId());
    EXPECT_TRUE(recvMessage.IsFailedMessage());
    EXPECT_EQ(1, recvMessage.GetIovSize());

    auto iov = message.GetIov(0);
    ASSERT_TRUE(iov != nullptr);
    auto recvIov = recvMessage.GetIov(0);
    ASSERT_TRUE(recvIov);

    EXPECT_TRUE(memcmp(iov->buf.get(), recvIov->buf.get(), iov->len) == 0);
    EXPECT_EQ(iov->len, recvIov->len);
    EXPECT_EQ(iov->mrStatus, recvIov->mrStatus);
    EXPECT_EQ(iov->device, recvIov->device);
    EXPECT_EQ(RdmaMempool::CPU, recvIov->remoteDevice);
}

TEST_F(RdmaMessageTestOnMock, testSetGetXMessage_SuccessMessage) {
    _mockXMempool->MockOnAnyCallAllocAndReleaseBuffer();

    RdmaMessage message;
    message.SetMempool(_mempool);
    message.SetRequestId(101);
    message.SetFailedMessage(false);

    auto buf = message.AllocateSharedSendBuffer(1024);
    ASSERT_TRUE(buf != nullptr);
    message.AddIov(buf, 1024, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::CPU);
    memset(buf.get(), '#', 1024);

    buf = message.AllocateSharedSendBuffer(2048);
    ASSERT_TRUE(buf != nullptr);
    message.AddIov(buf, 2048, RdmaMessage::BMS_MR, RdmaMempool::GPU, RdmaMempool::GPU);
    memset(buf.get(), '$', 1024);

    auto xmessage     = message.GetMessage();
    auto recvXMessage = CopyXMessage(xmessage);

    RdmaMessage recvMessage;
    recvMessage.SetMempool(_mempool);
    ASSERT_TRUE(recvMessage.SetRecvMessage(recvXMessage));
    EXPECT_EQ(101, recvMessage.GetRequestId());
    EXPECT_FALSE(recvMessage.IsFailedMessage());
    EXPECT_EQ(2, recvMessage.GetIovSize());

    auto iov     = message.GetIov(0);
    auto recvIov = recvMessage.GetIov(0);
    EXPECT_TRUE(memcmp(iov->buf.get(), recvIov->buf.get(), iov->len) == 0);
    EXPECT_EQ(iov->len, recvIov->len);
    EXPECT_EQ(iov->mrStatus, recvIov->mrStatus);
    EXPECT_EQ(iov->device, recvIov->device);
    EXPECT_EQ(RdmaMempool::CPU, recvIov->remoteDevice);

    iov     = message.GetIov(1);
    recvIov = message.GetIov(1);
    EXPECT_TRUE(memcmp(iov->buf.get(), recvIov->buf.get(), iov->len) == 0);
    EXPECT_EQ(iov->len, recvIov->len);
    EXPECT_EQ(iov->mrStatus, recvIov->mrStatus);
    EXPECT_EQ(iov->device, recvIov->device);
    EXPECT_EQ(RdmaMempool::GPU, recvIov->remoteDevice);
}

TEST_F(RdmaMessageTestOnMock, testReportClientStats) {
    RdmaMessage             message;
    ::accl::barex::XMessage xmessage;

    // report metric on stats with empty message
    std::shared_ptr<ClientRPCStats> emptyStats;
    message.reportClientRPCStats(emptyStats, xmessage);

    auto stats = std::make_shared<ClientRPCStats>(nullptr);
    message.reportClientRPCStats(stats, xmessage);
    ASSERT_TRUE(stats->_iovSizes.empty());

    // 构造rdma message
    message.AddIov(std::shared_ptr<void>(), 102, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::CPU);
    message.AddIov(std::shared_ptr<void>(), 203, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::GPU);
    message.AddIov(std::shared_ptr<void>(), 304, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::UNKNOWN);
    message.AddIov(std::shared_ptr<void>(), 405, RdmaMessage::BMS_MR, RdmaMempool::GPU, RdmaMempool::CPU);
    message.AddIov(std::shared_ptr<void>(), 506, RdmaMessage::BMS_MR, RdmaMempool::GPU, RdmaMempool::GPU);
    message.AddIov(std::shared_ptr<void>(), 607, RdmaMessage::BMS_MR, RdmaMempool::GPU, RdmaMempool::UNKNOWN);
    message.AddIov(std::shared_ptr<void>(), 708, RdmaMessage::BMS_MR, RdmaMempool::UNKNOWN, RdmaMempool::CPU);
    message.AddIov(std::shared_ptr<void>(), 809, RdmaMessage::BMS_MR, RdmaMempool::UNKNOWN, RdmaMempool::GPU);
    message.AddIov(std::shared_ptr<void>(), 910, RdmaMessage::BMS_MR, RdmaMempool::UNKNOWN, RdmaMempool::UNKNOWN);

    // 构造xmessage
    xmessage.Add((void*)0x1, 102, ::accl::barex::CPU, ::accl::barex::CPU);
    xmessage.Add((void*)0x2, 203, ::accl::barex::CPU, ::accl::barex::GPU);
    xmessage.Add((void*)0x3, 405, ::accl::barex::GPU, ::accl::barex::CPU);
    xmessage.Add((void*)0x4, 506, ::accl::barex::GPU, ::accl::barex::GPU);

    // report metric on null stats
    message.reportClientRPCStats(emptyStats, xmessage);

    // report metric on stats
    message.reportClientRPCStats(stats, xmessage);

    ASSERT_EQ(4, stats->_iovSizes.size());

    auto checkIovFunc = [stats](int index, std::string device, std::string remoteDevice, int size) {
        ASSERT_EQ(device, stats->_iovSizes[index].device);
        ASSERT_EQ(remoteDevice, stats->_iovSizes[index].remoteDevice);
        ASSERT_EQ(size, stats->_iovSizes[index].size);
    };
    checkIovFunc(0, "CPU", "CPU", 102);
    checkIovFunc(1, "CPU", "GPU", 203);
    checkIovFunc(2, "GPU", "CPU", 405);
    checkIovFunc(3, "GPU", "GPU", 506);
}

TEST_F(RdmaMessageTestOnMock, testReportServerStats) {
    RdmaMessage             message;
    ::accl::barex::XMessage xmessage;

    // report metric on stats with empty message
    std::shared_ptr<ServerRPCStats> emptyStats;
    message.reportServerRPCStats(emptyStats, xmessage);

    auto stats = std::make_shared<ServerRPCStats>(nullptr);
    message.reportServerRPCStats(stats, xmessage);
    ASSERT_TRUE(stats->_iovSizes.empty());

    // 构造rdma message
    message.AddIov(std::shared_ptr<void>(), 102, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::CPU);
    message.AddIov(std::shared_ptr<void>(), 203, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::GPU);
    message.AddIov(std::shared_ptr<void>(), 304, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::UNKNOWN);
    message.AddIov(std::shared_ptr<void>(), 405, RdmaMessage::BMS_MR, RdmaMempool::GPU, RdmaMempool::CPU);
    message.AddIov(std::shared_ptr<void>(), 506, RdmaMessage::BMS_MR, RdmaMempool::GPU, RdmaMempool::GPU);
    message.AddIov(std::shared_ptr<void>(), 607, RdmaMessage::BMS_MR, RdmaMempool::GPU, RdmaMempool::UNKNOWN);
    message.AddIov(std::shared_ptr<void>(), 708, RdmaMessage::BMS_MR, RdmaMempool::UNKNOWN, RdmaMempool::CPU);
    message.AddIov(std::shared_ptr<void>(), 809, RdmaMessage::BMS_MR, RdmaMempool::UNKNOWN, RdmaMempool::GPU);
    message.AddIov(std::shared_ptr<void>(), 910, RdmaMessage::BMS_MR, RdmaMempool::UNKNOWN, RdmaMempool::UNKNOWN);

    // 构造xmessage
    xmessage.Add((void*)0x1, 102, ::accl::barex::CPU, ::accl::barex::CPU);
    xmessage.Add((void*)0x2, 203, ::accl::barex::CPU, ::accl::barex::GPU);
    xmessage.Add((void*)0x3, 405, ::accl::barex::GPU, ::accl::barex::CPU);
    xmessage.Add((void*)0x4, 506, ::accl::barex::GPU, ::accl::barex::GPU);

    // report metric on null stats
    message.reportServerRPCStats(emptyStats, xmessage);

    // report metric on stats
    message.reportServerRPCStats(stats, xmessage);

    ASSERT_EQ(4, stats->_iovSizes.size());

    auto checkIovFunc = [stats](int index, std::string device, std::string remoteDevice, int size) {
        ASSERT_EQ(device, stats->_iovSizes[index].device);
        ASSERT_EQ(remoteDevice, stats->_iovSizes[index].remoteDevice);
        ASSERT_EQ(size, stats->_iovSizes[index].size);
    };
    checkIovFunc(0, "CPU", "CPU", 102);
    checkIovFunc(1, "CPU", "GPU", 203);
    checkIovFunc(2, "GPU", "CPU", 405);
    checkIovFunc(3, "GPU", "GPU", 506);
}

TEST_F(RdmaMessageTestOnMock, testVerifyIovStatus_knownDeviceAndKnownMrStatus) {
    _mockXMempool->MockOnAnyCallAllocAndReleaseBuffer();

    RdmaMessage message;
    message.SetMempool(_mempool);

    auto mrCPUBuf   = message.AllocateSharedSendBuffer(2, RdmaMempool::CPU);
    auto unMrCPUBuf = message.AllocateSharedSendBuffer(2, RdmaMempool::CPU);

    auto mrGPUBuf   = message.AllocateSharedSendBuffer(2, RdmaMempool::GPU);
    auto unMrGPUBuf = message.AllocateSharedSendBuffer(2, RdmaMempool::GPU);

    message.AddIov(mrCPUBuf, 2, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::CPU);
    message.AddIov(unMrCPUBuf, 2, RdmaMessage::BMS_NOT_MR, RdmaMempool::CPU, RdmaMempool::CPU);
    message.AddIov(mrGPUBuf, 2, RdmaMessage::BMS_MR, RdmaMempool::GPU, RdmaMempool::CPU);
    message.AddIov(unMrGPUBuf, 2, RdmaMessage::BMS_NOT_MR, RdmaMempool::GPU, RdmaMempool::CPU);

    ASSERT_TRUE(message.VerifyIovStatus());

    CheckMessageIov(message, 0, mrCPUBuf, 2, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::CPU);
    CheckMessageIov(message, 1, unMrCPUBuf, 2, RdmaMessage::BMS_NOT_MR, RdmaMempool::CPU, RdmaMempool::CPU);
    CheckMessageIov(message, 2, mrGPUBuf, 2, RdmaMessage::BMS_MR, RdmaMempool::GPU, RdmaMempool::CPU);
    CheckMessageIov(message, 3, unMrGPUBuf, 2, RdmaMessage::BMS_NOT_MR, RdmaMempool::GPU, RdmaMempool::CPU);
}

TEST_F(RdmaMessageTestOnMock, testVerifyIovStatus_unknowMrStatus) {
    _mockXMempool->MockOnAnyCallAllocAndReleaseBuffer();
    _mockXMempool->MockOnAnyCallIsValid();

    RdmaMessage message;
    message.SetMempool(_mempool);

    auto cudaUtil = new MockCudaUtil();
    message.SetCudaUtill(cudaUtil);

    auto                  unknownMrCPUBuf = message.AllocateSharedSendBuffer(2, RdmaMempool::CPU);
    auto                  unknownMrGPUBuf = message.AllocateSharedSendBuffer(2, RdmaMempool::GPU);
    std::shared_ptr<void> unknownMrUnknownDeviceBuf(malloc(2), [](void* p) { free(p); });

    message.AddIov(unknownMrCPUBuf, 2, RdmaMessage::BMS_UNKNOWN, RdmaMempool::CPU, RdmaMempool::CPU);
    message.AddIov(unknownMrGPUBuf, 2, RdmaMessage::BMS_UNKNOWN, RdmaMempool::GPU, RdmaMempool::CPU);
    message.AddIov(unknownMrUnknownDeviceBuf, 2, RdmaMessage::BMS_UNKNOWN, RdmaMempool::UNKNOWN, RdmaMempool::CPU);

    EXPECT_CALL(*cudaUtil, IsGPUMemory(unknownMrUnknownDeviceBuf.get())).WillOnce(Return(false));

    ASSERT_TRUE(message.VerifyIovStatus());

    CheckMessageIov(message, 0, unknownMrCPUBuf, 2, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::CPU);
    CheckMessageIov(message, 1, unknownMrGPUBuf, 2, RdmaMessage::BMS_MR, RdmaMempool::GPU, RdmaMempool::CPU);
    CheckMessageIov(
        message, 2, unknownMrUnknownDeviceBuf, 2, RdmaMessage::BMS_NOT_MR, RdmaMempool::CPU, RdmaMempool::CPU);
}

TEST_F(RdmaMessageTestOnMock, testVerifyIovStatus_unknownDevice) {
    _mockXMempool->MockOnAnyCallIsValid();

    RdmaMessage message;
    message.SetMempool(_mempool);

    auto cudaUtil = new MockCudaUtil();
    message.SetCudaUtill(cudaUtil);

    std::shared_ptr<void> mrUnknownDeviceBuf(malloc(2), [](void* p) { free(p); });
    std::shared_ptr<void> unMrUnknownDeviceBuf(malloc(2), [](void* p) { free(p); });
    std::shared_ptr<void> unknownMrUnknownDeviceBuf(malloc(2), [](void* p) { free(p); });

    message.AddIov(mrUnknownDeviceBuf, 2, RdmaMessage::BMS_MR, RdmaMempool::UNKNOWN, RdmaMempool::CPU);
    message.AddIov(unMrUnknownDeviceBuf, 2, RdmaMessage::BMS_NOT_MR, RdmaMempool::UNKNOWN, RdmaMempool::CPU);
    message.AddIov(unknownMrUnknownDeviceBuf, 2, RdmaMessage::BMS_UNKNOWN, RdmaMempool::UNKNOWN, RdmaMempool::CPU);

    EXPECT_CALL(*cudaUtil, IsGPUMemory(mrUnknownDeviceBuf.get())).WillOnce(Return(true));
    EXPECT_CALL(*cudaUtil, IsGPUMemory(unMrUnknownDeviceBuf.get())).WillOnce(Return(false));
    EXPECT_CALL(*cudaUtil, IsGPUMemory(unknownMrUnknownDeviceBuf.get())).WillOnce(Return(true));

    ASSERT_TRUE(message.VerifyIovStatus());

    CheckMessageIov(message, 0, mrUnknownDeviceBuf, 2, RdmaMessage::BMS_NOT_MR, RdmaMempool::GPU, RdmaMempool::CPU);
    CheckMessageIov(message, 1, unMrUnknownDeviceBuf, 2, RdmaMessage::BMS_NOT_MR, RdmaMempool::CPU, RdmaMempool::CPU);
    CheckMessageIov(
        message, 2, unknownMrUnknownDeviceBuf, 2, RdmaMessage::BMS_NOT_MR, RdmaMempool::GPU, RdmaMempool::CPU);
}

ARPC_END_NAMESPACE(arpc);
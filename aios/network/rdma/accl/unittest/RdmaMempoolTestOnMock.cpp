#include "unittest/unittest.h"

// force unitetst header first, change private class member
#include "aios/network/rdma/accl/RdmaMempool.h"
#include "aios/network/rdma/accl/unittest/mock/MockXSimpleMempool.h"
#include "autil/EnvUtil.h"

using namespace ::testing;

ARPC_BEGIN_NAMESPACE(arpc);

class RdmaMempoolTestOnMock: public TESTBASE {
public:
    void setUp();
    void tearDown();

protected:
    std::shared_ptr<::accl::barex::MockXSimpleMempool> _mockXMempool;
    std::shared_ptr<::accl::barex::XSimpleMempool>     _xMempool;
    std::shared_ptr<RdmaMempool>                       _mempool;
};

void RdmaMempoolTestOnMock::setUp() {
    _mockXMempool = std::make_shared<::accl::barex::MockXSimpleMempool>();
    _xMempool     = std::dynamic_pointer_cast<::accl::barex::XSimpleMempool>(_mockXMempool);
    EXPECT_CALL(*_mockXMempool, Shutdown()).WillOnce(Return(::accl::barex::BAREX_SUCCESS));
    EXPECT_CALL(*_mockXMempool, WaitStop()).WillOnce(Return(::accl::barex::BAREX_SUCCESS));

    _mempool            = std::make_shared<RdmaMempool>();
    _mempool->_xmempool = _xMempool;
}

void RdmaMempoolTestOnMock::tearDown() {}

TEST_F(RdmaMempoolTestOnMock, testConstruct) {
    EXPECT_EQ(RdmaMempool::CREATED, _mempool->_curState.load());
    EXPECT_TRUE(_mempool->_shrinkThread == nullptr);
}

TEST_F(RdmaMempoolTestOnMock, testAllocBuffer_allocNullptr) {
    EXPECT_CALL(*_mockXMempool, AllocBuffer(_, _, _, _)).WillOnce(Return(nullptr));

    _mempool->_curState.store(RdmaMempool::INIT_SUCCESS);
    EXPECT_EQ(nullptr, _mempool->AllocBuffer(10, RdmaMempool::CPU, 0, nullptr));
}

TEST_F(RdmaMempoolTestOnMock, testAllocBuffer_Failed) {
    EXPECT_EQ(nullptr, _mempool->AllocBuffer(10, RdmaMempool::CPU, 0, nullptr));
}

TEST_F(RdmaMempoolTestOnMock, testAllocBuffer_Success) {
    EXPECT_CALL(*_mockXMempool, AllocBuffer(1024, ::accl::barex::CPU, 0, nullptr)).WillOnce(Return((void*)0x123));

    _mempool->_curState.store(RdmaMempool::INIT_SUCCESS);
    EXPECT_EQ((void*)0x123, _mempool->AllocBuffer(1024, RdmaMempool::CPU, 0, nullptr));
}

TEST_F(RdmaMempoolTestOnMock, testReleaseBuffer_Failed) {
    EXPECT_FALSE(_mempool->ReleaseBuffer((void*)0x123, RdmaMempool::CPU));
}

TEST_F(RdmaMempoolTestOnMock, testReleaseBuffer_Success) {
    EXPECT_CALL(*_mockXMempool, ReleaseBuffer((void*)0x123, ::accl::barex::CPU))
        .WillOnce(Return(::accl::barex::BAREX_SUCCESS));

    _mempool->_curState.store(RdmaMempool::INIT_SUCCESS);
    EXPECT_TRUE(_mempool->ReleaseBuffer((void*)0x123, RdmaMempool::CPU));
}

TEST_F(RdmaMempoolTestOnMock, testRegUserMr_Success) {
    uint64_t size = 1024;
    void*    buf  = malloc(size);

    EXPECT_CALL(*_mockXMempool, RegUserMr(buf, size, ::accl::barex::CPU, 0))
        .WillOnce(Return(::accl::barex::BAREX_SUCCESS));

    _mempool->_curState.store(RdmaMempool::INIT_SUCCESS);
    EXPECT_TRUE(_mempool->RegUserMr(buf, size, RdmaMempool::CPU));
    free(buf);
}

TEST_F(RdmaMempoolTestOnMock, testRegUserMr_Failed) {
    uint64_t size = 1024;
    void*    buf  = malloc(size);

    EXPECT_FALSE(_mempool->RegUserMr(buf, size, RdmaMempool::CPU));
    free(buf);
}

TEST_F(RdmaMempoolTestOnMock, testRegUserMr2_Success) {
    ::accl::barex::memp_t mem;
    uint64_t              size = 1024;
    void*                 buf  = malloc(size);
    EXPECT_CALL(*_mockXMempool, RegUserMr(_, buf, size, ::accl::barex::CPU, 0))
        .WillOnce(Return(::accl::barex::BAREX_SUCCESS));
    _mempool->_curState.store(RdmaMempool::INIT_SUCCESS);
    EXPECT_TRUE(_mempool->RegUserMr(mem, buf, size, RdmaMempool::CPU));
    free(buf);
}
TEST_F(RdmaMempoolTestOnMock, testRegUserMr2_Failed) {
    ::accl::barex::memp_t mem;
    uint64_t              size = 1024;
    void*                 buf  = malloc(size);
    EXPECT_FALSE(_mempool->RegUserMr(mem, buf, size, RdmaMempool::CPU));
    free(buf);
}

TEST_F(RdmaMempoolTestOnMock, testDeregUserMr_Success) {
    uint64_t size = 1024;
    void*    buf  = malloc(size);

    EXPECT_CALL(*_mockXMempool, RegUserMr(buf, size, ::accl::barex::CPU, 0))
        .WillOnce(Return(::accl::barex::BAREX_SUCCESS));
    EXPECT_CALL(*_mockXMempool, DeregUserMr(buf, ::accl::barex::CPU)).WillOnce(Return(::accl::barex::BAREX_SUCCESS));

    _mempool->_curState.store(RdmaMempool::INIT_SUCCESS);
    EXPECT_TRUE(_mempool->RegUserMr(buf, size, RdmaMempool::CPU));
    EXPECT_TRUE(_mempool->DeregUserMr(buf, RdmaMempool::CPU));
    free(buf);
}

TEST_F(RdmaMempoolTestOnMock, testDeregUserMr_Failed) {
    uint64_t size = 1024;
    void*    buf  = malloc(size);

    _mempool->_curState.store(RdmaMempool::INIT_SUCCESS);
    EXPECT_TRUE(_mempool->RegUserMr(buf, size, RdmaMempool::CPU));

    _mempool->_curState.store(RdmaMempool::INIT_FAIL);
    EXPECT_FALSE(_mempool->DeregUserMr(buf, RdmaMempool::CPU));
    free(buf);
}

TEST_F(RdmaMempoolTestOnMock, testFindBuffer_Failed) {
    EXPECT_FALSE(_mempool->FindBuffer((void*)0x123, RdmaMempool::CPU));
}

TEST_F(RdmaMempoolTestOnMock, testFindBuffer_Success_CPU) {
    EXPECT_CALL(*_mockXMempool, IsValid((void*)0x123, ::accl::barex::CPU, false)).WillOnce(Return(true));
    _mempool->_curState.store(RdmaMempool::INIT_SUCCESS);
    EXPECT_TRUE(_mempool->FindBuffer((void*)0x123, RdmaMempool::CPU));
}

TEST_F(RdmaMempoolTestOnMock, testFindBuffer_Success_GPU) {
    EXPECT_CALL(*_mockXMempool, IsValid((void*)0x123, ::accl::barex::GPU, false)).WillOnce(Return(true));
    _mempool->_curState.store(RdmaMempool::INIT_SUCCESS);
    EXPECT_TRUE(_mempool->FindBuffer((void*)0x123, RdmaMempool::GPU));
}

TEST_F(RdmaMempoolTestOnMock, testFindBufferDevice_CPU_preferCPU) {
    EXPECT_CALL(*_mockXMempool, IsValid((void*)0x123, ::accl::barex::CPU, false)).WillOnce(Return(true));
    _mempool->_curState.store(RdmaMempool::INIT_SUCCESS);
    EXPECT_EQ(RdmaMempool::CPU, _mempool->FindBufferDevice((void*)0x123, true));
}

TEST_F(RdmaMempoolTestOnMock, testFindBufferDevice_CPU_preferGPU) {
    EXPECT_CALL(*_mockXMempool, IsValid((void*)0x123, ::accl::barex::CPU, false)).WillOnce(Return(true));
    EXPECT_CALL(*_mockXMempool, IsValid((void*)0x123, ::accl::barex::GPU, false)).WillOnce(Return(false));
    _mempool->_curState.store(RdmaMempool::INIT_SUCCESS);
    EXPECT_EQ(RdmaMempool::CPU, _mempool->FindBufferDevice((void*)0x123, false));
}

TEST_F(RdmaMempoolTestOnMock, testFindBufferDevice_GPU_preferCPU) {
    EXPECT_CALL(*_mockXMempool, IsValid((void*)0x123, ::accl::barex::CPU, false)).WillOnce(Return(false));
    EXPECT_CALL(*_mockXMempool, IsValid((void*)0x123, ::accl::barex::GPU, false)).WillOnce(Return(true));
    _mempool->_curState.store(RdmaMempool::INIT_SUCCESS);
    EXPECT_EQ(RdmaMempool::GPU, _mempool->FindBufferDevice((void*)0x123, true));
}

TEST_F(RdmaMempoolTestOnMock, testFindBufferDevice_GPU_preferGPU) {
    EXPECT_CALL(*_mockXMempool, IsValid((void*)0x123, ::accl::barex::GPU, false)).WillOnce(Return(true));
    _mempool->_curState.store(RdmaMempool::INIT_SUCCESS);
    EXPECT_EQ(RdmaMempool::GPU, _mempool->FindBufferDevice((void*)0x123, false));
}

TEST_F(RdmaMempoolTestOnMock, testFindBufferDevice_UNKNOWN) {
    EXPECT_CALL(*_mockXMempool, IsValid((void*)0x123, ::accl::barex::CPU, false))
        .Times(2)
        .WillRepeatedly(Return(false));
    EXPECT_CALL(*_mockXMempool, IsValid((void*)0x123, ::accl::barex::GPU, false))
        .Times(2)
        .WillRepeatedly(Return(false));
    _mempool->_curState.store(RdmaMempool::INIT_SUCCESS);
    EXPECT_EQ(RdmaMempool::UNKNOWN, _mempool->FindBufferDevice((void*)0x123, true));
    EXPECT_EQ(RdmaMempool::UNKNOWN, _mempool->FindBufferDevice((void*)0x123, false));
}

TEST_F(RdmaMempoolTestOnMock, testConvert) {
    EXPECT_EQ(RdmaMempool::CPU, RdmaMempool::ConvertFromAcclDeviceType(::accl::barex::CPU));
    EXPECT_EQ(RdmaMempool::GPU, RdmaMempool::ConvertFromAcclDeviceType(::accl::barex::GPU));

    EXPECT_EQ(::accl::barex::CPU, RdmaMempool::ConvertToAcclDeviceType(RdmaMempool::CPU));
    EXPECT_EQ(::accl::barex::GPU, RdmaMempool::ConvertToAcclDeviceType(RdmaMempool::GPU));
    EXPECT_EQ(::accl::barex::CPU, RdmaMempool::ConvertToAcclDeviceType(RdmaMempool::UNKNOWN));
}

TEST_F(RdmaMempoolTestOnMock, testInit_autoShrink_disable) {
    RdmaMempoolConfig config;
    config.enableAutoShrink = false;
    EXPECT_TRUE(_mempool->Init(config));
    EXPECT_FALSE(_mempool->NotActive());
    EXPECT_TRUE(_mempool->_shrinkThread == nullptr);
}

TEST_F(RdmaMempoolTestOnMock, testInit_autoShrink_enable_defaultValue) {
    RdmaMempoolConfig config;
    EXPECT_TRUE(_mempool->Init(config));
    EXPECT_FALSE(_mempool->NotActive());
    EXPECT_TRUE(_mempool->_shrinkThread != nullptr);

    _mempool->Stop();
    EXPECT_TRUE(_mempool->_shrinkThread == nullptr);
}

TEST_F(RdmaMempoolTestOnMock, testInit_autoShrink_enable_shrinkSuccess) {
    RdmaMempoolConfig config;
    config.shrinkFreeRatio        = 0.1;
    config.shrinkFreeToBusy       = 0.5;
    config.shrinkIntervalSec      = 1;
    config.shrinkStartWaitTimeSec = 1;

    EXPECT_CALL(*_mockXMempool, Shrink(::accl::barex::CPU, float(0.1), float(0.5)))
        .Times(1)
        .WillRepeatedly(Return(::accl::barex::BAREX_SUCCESS));
#ifdef GOOGLE_CUDA
    EXPECT_CALL(*_mockXMempool, Shrink(::accl::barex::GPU, float(0.1), float(0.5)))
        .Times(1)
        .WillRepeatedly(Return(::accl::barex::BAREX_SUCCESS));
#endif

    EXPECT_TRUE(_mempool->Init(config));
    EXPECT_FALSE(_mempool->NotActive());
    EXPECT_TRUE(_mempool->_shrinkThread != nullptr);

    usleep(1500 * 1000);
}

TEST_F(RdmaMempoolTestOnMock, testInit_autoShrink_enable_shrinkFailed) {
    RdmaMempoolConfig config;
    config.shrinkFreeRatio        = 0.1;
    config.shrinkFreeToBusy       = 0.5;
    config.shrinkIntervalSec      = 1;
    config.shrinkStartWaitTimeSec = 0;
    EXPECT_CALL(*_mockXMempool, Shrink(::accl::barex::CPU, float(0.1), float(0.5)))
        .Times(2)
        .WillRepeatedly(Return(::accl::barex::BAREX_ERR_ARG));
#ifdef GOOGLE_CUDA
    EXPECT_CALL(*_mockXMempool, Shrink(::accl::barex::GPU, float(0.1), float(0.5)))
        .Times(2)
        .WillRepeatedly(Return(::accl::barex::BAREX_ERR_ARG));
#endif

    EXPECT_TRUE(_mempool->Init(config));
    EXPECT_FALSE(_mempool->NotActive());
    EXPECT_TRUE(_mempool->_shrinkThread != nullptr);

    usleep(1500 * 1000);
}

TEST_F(RdmaMempoolTestOnMock, testInit_reserve_envValue) {
    RdmaMempoolConfig config;
    config.cpuReserves = std::map<uint64_t, int>{{1, 2}, {3, 4}, {5, 6}};
    config.gpuReserves = std::map<uint64_t, int>{{2, 3}, {4, 5}, {6, 7}};

    EXPECT_CALL(*_mockXMempool, Reserve(::accl::barex::CPU, config.cpuReserves))
        .WillOnce(Return(::accl::barex::BAREX_SUCCESS));
    EXPECT_CALL(*_mockXMempool, Reserve(::accl::barex::GPU, config.gpuReserves))
        .WillOnce(Return(::accl::barex::BAREX_SUCCESS));

    EXPECT_TRUE(_mempool->Init(config));
    EXPECT_FALSE(_mempool->NotActive());
}

TEST_F(RdmaMempoolTestOnMock, testInit_reserve_reserveCPUFailed) {
    RdmaMempoolConfig config;
    config.cpuReserves = std::map<uint64_t, int>{{1, 2}, {3, 4}, {5, 6}};

    EXPECT_CALL(*_mockXMempool, Reserve(::accl::barex::CPU, config.cpuReserves))
        .WillOnce(Return(::accl::barex::BAREX_ERR_ARG));

    EXPECT_FALSE(_mempool->Init(config));
    EXPECT_TRUE(_mempool->NotActive());
}

TEST_F(RdmaMempoolTestOnMock, testInit_reserve_reserveGPUFailed) {
    RdmaMempoolConfig config;
    config.gpuReserves = std::map<uint64_t, int>{{2, 3}, {4, 5}, {6, 7}};

    EXPECT_CALL(*_mockXMempool, Reserve(::accl::barex::GPU, config.gpuReserves))
        .WillOnce(Return(::accl::barex::BAREX_ERR_ARG));

    EXPECT_FALSE(_mempool->Init(config));
    EXPECT_TRUE(_mempool->NotActive());
}

ARPC_END_NAMESPACE(arpc);
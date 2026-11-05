#include "unittest/unittest.h"

// force unitetst header first, change private class member
#include "aios/network/rdma/accl/RdmaChannelContext.h"
#include "aios/network/rdma/accl/RdmaMempool.h"
#include "aios/network/rdma/accl/RdmaMessage.h"
#include "aios/network/rdma/accl/TimerManager.h"
#include "aios/network/rdma/accl/unittest/mock/MockRdmaChannelClosure.h"
#include "aios/network/rdma/accl/unittest/mock/MockRdmaRequest.h"
#include "aios/network/rdma/accl/unittest/mock/MockXConnection.h"
#include "aios/network/rdma/accl/unittest/mock/MockXSimpleMempool.h"

using namespace ::testing;
ARPC_BEGIN_NAMESPACE(arpc);

class RdmaChannelContextTestOnMock: public TESTBASE {
public:
    void setUp();
    void tearDown();

protected:
    std::shared_ptr<std::string> addTimerOnContext(uint64_t timeoutMs);
    std::shared_ptr<RdmaMessage> addMessageOnContext();

protected:
    std::shared_ptr<MockRdmaRequest>        _mockRequest;
    std::shared_ptr<RdmaRequest>            _request;
    std::shared_ptr<MockRdmaChannelClosure> _mockClosure;
    std::shared_ptr<RdmaChannelClosure>     _closure;
    std::shared_ptr<TimerManager>           _timerManager;

    std::shared_ptr<RdmaChannelContext> _context;

    std::shared_ptr<::accl::barex::MockXSimpleMempool> _mockXMempool;
    std::shared_ptr<RdmaMempool>                       _mempool;
};

void RdmaChannelContextTestOnMock::setUp() {
    _mockRequest = std::make_shared<MockRdmaRequest>();
    _request     = std::dynamic_pointer_cast<RdmaRequest>(_mockRequest);

    _mockClosure = std::make_shared<MockRdmaChannelClosure>();
    _closure     = std::dynamic_pointer_cast<RdmaChannelClosure>(_mockClosure);

    _timerManager = std::make_shared<TimerManager>();

    _context =
        std::make_shared<RdmaChannelContext>(_request, _closure, _timerManager, std::shared_ptr<ClientRPCStats>());

    _mockXMempool = std::make_shared<::accl::barex::MockXSimpleMempool>();
    EXPECT_CALL(*_mockXMempool, Shutdown()).WillOnce(Return(::accl::barex::BAREX_SUCCESS));
    EXPECT_CALL(*_mockXMempool, WaitStop()).WillOnce(Return(::accl::barex::BAREX_SUCCESS));

    _mempool = std::make_shared<RdmaMempool>();
    _mempool->_curState.store(RdmaMempool::INIT_SUCCESS);
    _mempool->_xmempool = std::dynamic_pointer_cast<::accl::barex::XSimpleMempool>(_mockXMempool);
}

void RdmaChannelContextTestOnMock::tearDown() {}

std::shared_ptr<std::string> RdmaChannelContextTestOnMock::addTimerOnContext(uint64_t timeoutMs) {
    auto mark  = std::make_shared<std::string>();
    auto timer = _timerManager->addTimer(timeoutMs, [context = _context, mark]() {
        mark->assign("timeout");
        context->RunFailed(RdmaChannelClosure::ERROR_TIMEOUT);
    });
    EXPECT_TRUE(timer != nullptr);
    _context->SetTimer(timer);
    return mark;
}

std::shared_ptr<RdmaMessage> RdmaChannelContextTestOnMock::addMessageOnContext() {
    _mockXMempool->MockOnAnyCallAllocAndReleaseBuffer();

    auto message = std::make_shared<RdmaMessage>();
    message->SetMempool(_mempool);
    auto iov = message->AllocateSharedSendBuffer(1024);
    message->AddIov(iov, 1024, RdmaMessage::BMS_MR, RdmaMempool::CPU, RdmaMempool::CPU);

    _context->SetMessage(message);
    return message;
}

// 100ms sendDone -> 200ms response -> ...
TEST_F(RdmaChannelContextTestOnMock, testRunSuccess_messageReleased) {
    auto mark    = addTimerOnContext(300);
    auto message = addMessageOnContext();
    _request->SetTimeoutMs(100);

    EXPECT_CALL(*_mockClosure, Run(RdmaChannelClosure::ERROR_NONE)).WillOnce(Return());
    EXPECT_NE(UINT64_MAX, _context->GetMessageDeadlineMs());

    usleep(100 * 1000);
    _context->ReleaseMessageMemory();

    usleep(100 * 1000);
    _context->RunSuccess();
    _context.reset();

    usleep(200 * 1000);
    EXPECT_TRUE(mark->empty());
    EXPECT_EQ(0, message->GetIovSize());
}

// 100ms response -> 200ms timeout -> 1000ms message timeout
TEST_F(RdmaChannelContextTestOnMock, testRunSuccess_messageNoRelease) {
    auto mark    = addTimerOnContext(200);
    auto message = addMessageOnContext();
    _request->SetTimeoutMs(100);
    RdmaChannelContext::sDelayReleaseMessageMs = 1000;  // release delay 1s

    EXPECT_CALL(*_mockClosure, Run(RdmaChannelClosure::ERROR_NONE)).WillOnce(Return());
    EXPECT_NE(UINT64_MAX, _context->GetMessageDeadlineMs());

    usleep(100 * 1000);
    _context->RunSuccess();
    _context.reset();

    EXPECT_TRUE(mark->empty());
    EXPECT_NE(0, message->GetIovSize());

    usleep(1000 * 1000);
    EXPECT_NE(0, message->GetIovSize());

    usleep(1000 * 1000);
    EXPECT_EQ(0, message->GetIovSize());
}

// 100ms sendDone -> 200ms timeout
TEST_F(RdmaChannelContextTestOnMock, testRunTimout_messageReleased) {
    auto mark    = addTimerOnContext(200);
    auto message = addMessageOnContext();
    _request->SetTimeoutMs(100);

    EXPECT_CALL(*_mockClosure, Run(RdmaChannelClosure::ERROR_TIMEOUT)).WillOnce(Return());
    EXPECT_NE(UINT64_MAX, _context->GetMessageDeadlineMs());

    usleep(100 * 1000);
    _context->ReleaseMessageMemory();

    usleep(200 * 1000);

    EXPECT_EQ("timeout", *mark);
    EXPECT_EQ(0, message->GetIovSize());
}

// 200ms timeout -> 1000ms message timeout
TEST_F(RdmaChannelContextTestOnMock, testRunTimeout_messageNoRelease) {
    auto mark    = addTimerOnContext(200);
    auto message = addMessageOnContext();
    _request->SetTimeoutMs(100);
    RdmaChannelContext::sDelayReleaseMessageMs = 1000;  // release delay 1s

    EXPECT_CALL(*_mockClosure, Run(RdmaChannelClosure::ERROR_TIMEOUT)).WillOnce(Return());
    EXPECT_NE(UINT64_MAX, _context->GetMessageDeadlineMs());

    usleep(300 * 1000);

    EXPECT_EQ("timeout", *mark);
    EXPECT_NE(0, message->GetIovSize());

    usleep(1000 * 1000);
    EXPECT_NE(0, message->GetIovSize());

    usleep(1000 * 1000);
    EXPECT_EQ(0, message->GetIovSize());
}

// 100ms sendDone -> 200ms respnse failed -> ...
TEST_F(RdmaChannelContextTestOnMock, testRunFailed_messageReleased) {
    auto mark    = addTimerOnContext(300);
    auto message = addMessageOnContext();
    _request->SetTimeoutMs(100);

    EXPECT_CALL(*_mockClosure, Run(RdmaChannelClosure::ERROR_BAD_PACKET)).WillOnce(Return());
    EXPECT_NE(UINT64_MAX, _context->GetMessageDeadlineMs());

    usleep(100 * 1000);
    _context->ReleaseMessageMemory();

    usleep(100 * 1000);
    _context->RunFailed(RdmaChannelClosure::ERROR_BAD_PACKET);
    _context.reset();

    usleep(200 * 1000);
    EXPECT_TRUE(mark->empty());
    EXPECT_EQ(0, message->GetIovSize());
}

// 100ms response failed -> 200ms timeout -> 1000ms message timeout
TEST_F(RdmaChannelContextTestOnMock, testRunFailed_messageNoRelease) {
    auto mark    = addTimerOnContext(200);
    auto message = addMessageOnContext();
    _request->SetTimeoutMs(100);
    RdmaChannelContext::sDelayReleaseMessageMs = 1000;  // release delay 1s

    EXPECT_CALL(*_mockClosure, Run(RdmaChannelClosure::ERROR_BAD_PACKET)).WillOnce(Return());
    EXPECT_NE(UINT64_MAX, _context->GetMessageDeadlineMs());

    usleep(100 * 1000);
    _context->RunFailed(RdmaChannelClosure::ERROR_BAD_PACKET);
    _context.reset();

    EXPECT_TRUE(mark->empty());
    EXPECT_NE(0, message->GetIovSize());

    usleep(1000 * 1000);
    EXPECT_NE(0, message->GetIovSize());

    usleep(1000 * 1000);
    EXPECT_EQ(0, message->GetIovSize());
}

TEST_F(RdmaChannelContextTestOnMock, testGetMessageDeadlineMs_nullRequest) {
    std::shared_ptr<RdmaRequest> request;
    RdmaChannelContext           context(
        request, std::shared_ptr<RdmaChannelClosure>(), _timerManager, std::shared_ptr<ClientRPCStats>());

    EXPECT_EQ(UINT64_MAX, context.GetMessageDeadlineMs());
    usleep(10 * 1000);  // wait 10ms
    EXPECT_EQ(UINT64_MAX, context.GetMessageDeadlineMs());
}

TEST_F(RdmaChannelContextTestOnMock, testGetMessageDeadlineMs_normal) {
    auto currentTimeMs = autil::TimeUtility::currentTimeInMilliSeconds();

    _request->SetTimeoutMs(100);

    auto deadlineMs = _context->GetMessageDeadlineMs();
    EXPECT_TRUE(deadlineMs - currentTimeMs >= 1000);

    usleep(10 * 1000);  // wait 10ms
    EXPECT_EQ(deadlineMs, _context->GetMessageDeadlineMs());
}

ARPC_END_NAMESPACE(arpc);
#include "aios/network/rdma/accl/unittest/mock/MockXConnection.h"

#include "aios/network/accl-barex/include/accl/barex/highlevel/xmessage.h"

namespace accl {
namespace barex {

MATCHER_P(XMessageEq, message1, "") {
    if (message1.IovCount() != arg.IovCount()) {
        return false;
    }
    for (int i = 0; i < message1.IovCount(); i++) {
        auto iov1 = message1.Get(i);
        auto iov2 = arg.Get(i);
        if (iov1.iov_base != iov2.iov_base) {
            return false;
        }
        if (iov1.iov_len != iov2.iov_len) {
            return false;
        }
        if (iov1.device != iov2.device) {
            return false;
        }
        if (iov1.remote_device != iov2.remote_device) {
            return false;
        }
    }
    return true;
}

void MockXConnection::MockOnCallSend(::accl::barex::XMessage& message, autil::ThreadPtr& thread, int waitTimeMs) {
    ON_CALL(*this, Send(XMessageEq(message), _, _))
        .WillByDefault(
            Invoke([&thread, waitTimeMs](const XMessage& message, DoneCallback callback, XConnection::SendMode) {
                thread = autil::Thread::createThread([callback, waitTimeMs]() {
                    usleep(waitTimeMs * 1000);
                    callback(Status::OK());
                });
            }));
    EXPECT_CALL(*this, Send(XMessageEq(message), _, _)).Times(1);
}

void MockXConnection::MockOnFailedCallSend(::accl::barex::XMessage& message, autil::ThreadPtr& thread, int waitTimeMs) {
    ON_CALL(*this, Send(XMessageEq(message), _, _))
        .WillByDefault(
            Invoke([&thread, waitTimeMs](const XMessage& message, DoneCallback callback, XConnection::SendMode) {
                thread = autil::Thread::createThread([callback, waitTimeMs]() {
                    usleep(waitTimeMs * 1000);
                    Status status(BAREX_ERR_ADDRESS, "");
                    callback(status);
                });
            }));
    EXPECT_CALL(*this, Send(XMessageEq(message), _, _)).Times(1);
}

}  // namespace barex
}  // namespace accl
#pragma once

#include "aios/network/accl-barex/include/accl/barex/highlevel/xconnection.h"
#include "autil/Thread.h"
#include "unittest/unittest.h"

namespace accl {
namespace barex {

class MockXConnection: public XConnection {
public:
    virtual ~MockXConnection() {}

public:
    MOCK_METHOD3(Send, void(const XMessage&, DoneCallback, XConnection::SendMode));
    MOCK_METHOD0(Close, void());
    MOCK_METHOD0(DestroyChannel, void());
    MOCK_METHOD0(GetMempool, XSimpleMempool*());
    MOCK_METHOD0(GetChannel, XChannel*());
    MOCK_METHOD0(GetPeerAddrAndPort, std::pair<std::string, int>());
    MOCK_METHOD0(GetLocalAddrAndPort, std::pair<std::string, int>());

public:
    void MockOnCallSend(::accl::barex::XMessage& xmessage, autil::ThreadPtr& thread, int waitTimeMs);
    void MockOnFailedCallSend(::accl::barex::XMessage& xmessage, autil::ThreadPtr& thread, int waitTimeMs);
};

}  // namespace barex
}  // namespace accl
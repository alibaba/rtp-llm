#pragma once
#include "aios/network/accl-barex/include/accl/barex/highlevel/xclient_transport.h"
#include "autil/Thread.h"
#include "unittest/unittest.h"

namespace accl {
namespace barex {

class MockXClientTransport: public XClientTransport {
public:
    virtual ~MockXClientTransport() {}

public:
    MOCK_METHOD4(
        Connect,
        void(::accl::barex::TcpProtoType, const std::string&, int, const std::shared_ptr<XConnectionCallback>&));
    MOCK_METHOD1(Disconnect, bool(const std::shared_ptr<XConnection>&));
};

}  // namespace barex
}  // namespace accl
#pragma once

#include "aios/network/accl-barex/include/accl/barex/highlevel/xserver_transport.h"
#include "unittest/unittest.h"

namespace accl {
namespace barex {

class MockXServerTransport: public XServerTransport {
public:
    virtual ~MockXServerTransport() {}

public:
    MOCK_METHOD3(Listen,
                 BarexResult(::accl::barex::TcpProtoType, uint32_t, const std::shared_ptr<XConnectionCallback>&));
    MOCK_METHOD1(SetRateLimiter, void(const std::shared_ptr<XRateLimiter>&));
};

}  // namespace barex
}  // namespace accl
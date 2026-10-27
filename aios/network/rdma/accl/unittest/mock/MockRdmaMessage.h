#pragma once

#include "aios/network/rdma/accl/RdmaMessage.h"
#include "unittest/unittest.h"

ARPC_BEGIN_NAMESPACE(arpc);

class MockRdmaMessage: public RdmaMessage {
public:
    MOCK_METHOD0(GetMessage, ::accl::barex::XMessage());
    MOCK_METHOD0(ReleaseMessage, bool());
};

ARPC_END_NAMESPACE(arpc);
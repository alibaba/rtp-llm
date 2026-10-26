#pragma once

#include "aios/network/rdma/accl/RdmaChannelClosure.h"
#include "unittest/unittest.h"

ARPC_BEGIN_NAMESPACE(arpc);

class MockRdmaChannelClosure: public RdmaChannelClosure {
public:
    virtual ~MockRdmaChannelClosure() {}

public:
    MOCK_METHOD1(Run, void(ErrorState));
};

ARPC_END_NAMESPACE(arpc);
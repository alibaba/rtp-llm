#pragma once

#include "aios/network/rdma/accl/RdmaRequest.h"
#include "unittest/unittest.h"

ARPC_BEGIN_NAMESPACE(arpc);

class MockRdmaRequest: public RdmaRequest {
public:
    virtual ~MockRdmaRequest() {}

public:
    MOCK_METHOD0(Encode, std::shared_ptr<RdmaMessage>());
    MOCK_METHOD1(Decode, bool(const std::shared_ptr<RdmaMessage>&));
};

ARPC_END_NAMESPACE(arpc);
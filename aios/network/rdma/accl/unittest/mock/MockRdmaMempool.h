#pragma once

#include "aios/network/rdma/accl/RdmaMempool.h"
#include "unittest/unittest.h"

namespace arpc {

class MockRdmaMempool: public RdmaMempool {
public:
    virtual ~MockRdmaMempool() = default;

public:
    MOCK_CONST_METHOD4(AllocBuffer, void*(uint64_t, DeviceType, int, void*));
    MOCK_CONST_METHOD2(ReleaseBuffer, bool(void*, DeviceType));
    MOCK_CONST_METHOD4(RegUserMr, bool(::accl::barex::memp_t&, void*, uint64_t, DeviceType));
};

}  // namespace arpc

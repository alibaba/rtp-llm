#pragma once

#include "aios/network/accl-barex/include/accl/barex/xsimple_mempool.h"
#include "unittest/unittest.h"

namespace accl {
namespace barex {

class MockXSimpleMempool: public XSimpleMempool {
public:
    virtual ~MockXSimpleMempool();

public:
    MOCK_METHOD0(Init, BarexResult());
    MOCK_METHOD4(AllocBuffer, void*(uint64_t, device_type, int, void*));
    MOCK_METHOD6(AllocBuffer, BarexResult(memp_t&, uint64_t, device_type, int, int, void*));
    MOCK_METHOD6(AllocLimitBuffer, BarexResult(memp_t&, uint64_t, device_type, int, int, void*));
    MOCK_METHOD6(AllocBuffersAsync,
                 BarexResult(std::vector<mem_config>&,
                             std::function<void(std::vector<memp_t>, BarexResult)>,
                             int,
                             int,
                             bool,
                             void*));
    MOCK_METHOD2(ReleaseBuffer, BarexResult(void*, device_type));
    MOCK_METHOD2(ReleaseAndDeregBuffer, BarexResult(void*, device_type));
    MOCK_METHOD3(Shrink, BarexResult(device_type, float, float));
    MOCK_METHOD2(Reserve, BarexResult(device_type, std::map<uint64_t, int>&));
    MOCK_METHOD1(Stats, XMempoolStats(device_type));
    MOCK_METHOD5(FindBufferMr, BarexResult(memp_t&, void*, device_type, int, bool));
    MOCK_METHOD3(IsValid, bool(void*, device_type, bool));
    MOCK_METHOD0(Shutdown, BarexResult());
    MOCK_METHOD0(IsStopped, bool());
    MOCK_METHOD0(WaitStop, BarexResult());
    MOCK_METHOD2(GetMetric, uint64_t(device_type, std::string));
    MOCK_METHOD0(ToString, std::string());
    MOCK_METHOD4(RegUserMr, BarexResult(void*, uint64_t, device_type, int));
    MOCK_METHOD5(RegUserMr, BarexResult(memp_t&, void*, uint64_t, device_type, int));
    MOCK_METHOD2(DeregUserMr, BarexResult(void*, device_type));
    // XSimpleMempool gained this pure virtual after the closed repository last
    // touched this mock; without the override the mock stays abstract and no
    // test can instantiate it.
    MOCK_METHOD2(GetXAllocator, BarexResult(XAllocator*&, device_type));

public:
    void MockOnCallAllocAndReleaseBuffer(int len, device_type device, int deviceId = 0);
    void MockOnAnyCallIsValid();
    void MockOnAnyCallAllocAndReleaseBuffer();

private:
    std::map<void*, device_type> _buffers;
};

}  // namespace barex
}  // namespace accl

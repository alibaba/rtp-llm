#pragma once

#include "aios/network/accl-barex/include/accl/barex/barex_types.h"
#include "aios/network/arpc/arpc/CommonMacros.h"
#include "aios/network/rdma/accl/RdmaMempoolConfig.h"
#include "autil/Log.h"
#include "autil/LoopThread.h"

namespace accl {
namespace barex {
class XSimpleMempool;
}
}  // namespace accl

ARPC_BEGIN_NAMESPACE(arpc);

class RdmaMempool {
public:
    enum DeviceType {
        CPU = 0,
        GPU,
        UNKNOWN,
    };

public:
    virtual void* AllocBuffer(uint64_t size, DeviceType deviceType = CPU, int deviceId = 0, void* attr = nullptr) const;

    virtual bool ReleaseBuffer(void* buf, DeviceType deviceType = CPU) const;

    bool FindBuffer(void* buf, DeviceType deviceType = CPU) const;

    DeviceType FindBufferDevice(void* buf, bool preferCPU) const;

    bool         RegUserMr(void* buf, uint64_t size, DeviceType deviceType = CPU) const;
    virtual bool RegUserMr(::accl::barex::memp_t& out, void* buf, uint64_t size, DeviceType deviceType = CPU) const;

    bool DeregUserMr(void* buf, DeviceType deviceType = CPU) const;

    const std::shared_ptr<::accl::barex::XSimpleMempool>& GetXMempool() const {
        return _xmempool;
    }

    static ::accl::barex::device_type ConvertToAcclDeviceType(DeviceType deviceType);
    static DeviceType                 ConvertFromAcclDeviceType(::accl::barex::device_type deviceType);
    static std::string                DeviceTypeName(DeviceType deviceType);

public:
    RdmaMempool();
    virtual ~RdmaMempool();

    bool Init(const RdmaMempoolConfig& config);
    bool NeedInit() {
        return _curState.load() == CREATED;
    }
    bool NotActive() {
        return _curState.load() != INIT_SUCCESS;
    }
    void Stop() {
        _shrinkThread.reset();
    }

private:
    RdmaMempool(const RdmaMempool&);
    RdmaMempool& operator==(const RdmaMempool&);

    void TriggerShrink(uint64_t shrinkStartTimeSec, float shrinkFreeRatio, float shrinkFreeToBusy);
    bool ReserveMem(std::map<uint64_t, int> reserves, ::accl::barex::device_type device);

private:
    enum State {
        CREATED = 0,
        INIT_SUCCESS,
        INIT_FAIL,
    };

private:
    std::atomic<State>                             _curState;
    std::shared_ptr<::accl::barex::XSimpleMempool> _xmempool;

    autil::LoopThreadPtr _shrinkThread;

private:
    AUTIL_LOG_DECLARE();
};

ARPC_END_NAMESPACE(arpc);
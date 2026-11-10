#pragma once

#ifdef USE_RDMA
#include "aios/network/rdma/accl/RdmaMempoolImpl.h"
#else

#include "aios/network/arpc/arpc/CommonMacros.h"
#include "aios/network/rdma/accl/RdmaMempoolConfig.h"
#include "autil/Log.h"

ARPC_BEGIN_NAMESPACE(arpc);

class RdmaMempool {
public:
    enum DeviceType {
        CPU = 0,
        GPU,
        UNKNOWN,
    };

public:
    RdmaMempool()          = default;
    virtual ~RdmaMempool() = default;

    bool Init(const RdmaMempoolConfig& /*unused*/) {
        return false;
    }
    bool NeedInit() {
        return false;
    }

    void Stop() {}

    virtual void*
    AllocBuffer(uint64_t size, DeviceType deviceType = CPU, int deviceId = 0, void* attr = nullptr) const {
        return nullptr;
    }

    virtual bool ReleaseBuffer(void* buf, DeviceType deviceType = CPU) const {
        return false;
    }
};

ARPC_END_NAMESPACE(arpc);

#endif  // USE_RDMA
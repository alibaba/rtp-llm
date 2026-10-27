#pragma once

#include "aios/network/arpc/arpc/CommonMacros.h"
#include "autil/ThreadPool.h"

ARPC_BEGIN_NAMESPACE(arpc);

class RdmaChannelClosure {
public:
    RdmaChannelClosure() {}
    virtual ~RdmaChannelClosure() {}

public:
    enum ErrorState {
        ERROR_NONE,
        ERROR_CONNECTION_FAILED,
        ERROR_TIMEOUT,
        ERROR_BAD_PACKET,
        ERROR_UNKNOWN,
        ERROR_ACCL_SERVER_RATE_LIMIT,
        ERROR_SERVER_ERROR,
    };

public:
    virtual void Run(ErrorState errorState) = 0;
};

ARPC_END_NAMESPACE(arpc);
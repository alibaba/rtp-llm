#pragma once

#include "aios/network/rdma/accl/RdmaChannelClosure.h"

ARPC_BEGIN_NAMESPACE(arpc)

class FakeRdmaChannelClosure: public RdmaChannelClosure {
public:
    typedef std::function<void(ErrorState)> VoidFuncCall;

public:
    FakeRdmaChannelClosure(VoidFuncCall call = [](ErrorState) {}):
        _runDone(false), _state(ERROR_UNKNOWN), _call(call) {}

public:
    void Run(ErrorState errorState) override {
        _runDone = true;
        _state   = errorState;
        _call(errorState);
    }

    bool Done() {
        return _runDone;
    }
    ErrorState State() {
        return _state;
    }

private:
    bool         _runDone;
    ErrorState   _state;
    VoidFuncCall _call;
};

ARPC_END_NAMESPACE(arpc)
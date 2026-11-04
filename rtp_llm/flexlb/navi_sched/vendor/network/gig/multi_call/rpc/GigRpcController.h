#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "google/protobuf/service.h"

namespace multi_call {

class GigRpcController : public google::protobuf::RpcController
{
public:
    GigRpcController() : _ec(MULTI_CALL_ERROR_NO_RESPONSE), _rpcController(nullptr) {
    }
    ~GigRpcController() {
    }

private:
    GigRpcController(const GigRpcController &);
    GigRpcController &operator=(const GigRpcController &);

public:
    void Reset() override {
        if (_rpcController) {
            _rpcController->Reset();
        }
        _ec = MULTI_CALL_ERROR_NO_RESPONSE;
        _reason.clear();
    }
    bool Failed() const override {
        return _ec > MULTI_CALL_ERROR_DEC_WEIGHT;
    }
    std::string ErrorText() const override {
        return _reason;
    }
    void StartCancel() override {
        if (_rpcController) {
            _rpcController->StartCancel();
        }
    }
    void SetFailed(const std::string &reason) override {
        if (_rpcController) {
            _rpcController->SetFailed(reason);
        }
        _reason = reason;
        _ec = MULTI_CALL_ERROR_RPC_FAILED;
    }
    bool IsCanceled() const override {
        if (_rpcController) {
            return _rpcController->IsCanceled();
        } else {
            return false;
        }
    }
    void NotifyOnCancel(google::protobuf::Closure *callback) override {
        if (_rpcController) {
            _rpcController->NotifyOnCancel(callback);
        }
    }
    google::protobuf::RpcController *getProtocolController() const {
        return _rpcController;
    }

public:
    MultiCallErrorCode getErrorCode() const {
        return _ec;
    }
    void setErrorCode(MultiCallErrorCode ec) {
        _ec = ec;
    }
    void setRpcController(google::protobuf::RpcController *rpcController) {
        _rpcController = rpcController;
    }
    google::protobuf::RpcController *getRpcController() {
        return _rpcController;
    }
    std::string getClientIp() const;
private:
    MultiCallErrorCode _ec;
    std::string _reason;
    google::protobuf::RpcController *_rpcController;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(GigRpcController);

} // namespace multi_call


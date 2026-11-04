#include "navi/rpc_server/ArpcGraphClosure.h"
#include "navi/log/NaviLogger.h"
#include "navi/rpc_server/ArpcRegistryParam.h"
#include "navi/rpc_server/NaviArpcRequestData.h"
#include "navi/rpc_server/NaviArpcResponseData.h"
#include "navi/rpc_server/NaviDebugProtoJsonizer.h"
#include "navi/rpc_server/NaviStreamProtoJsonizer.h"
#include "navi/engine/NaviHostInfo.h"
#include "navi/engine/ComputeScope.h"
#include <multi_call/rpc/GigClosure.h>
#include <google/protobuf/message.h>

namespace navi {

ArpcGraphClosure::ArpcGraphClosure(const ArpcRegistryParamPtr &param,
                                   google::protobuf::RpcController *controller,
                                   const std::shared_ptr<NaviArpcRequestData> &requestData,
                                   google::protobuf::Message *response,
                                   google::protobuf::Closure *done,
                                   http_arpc::HTTPRPCServerClosure *httpClosure)
    : _param(param)
    , _controller(controller)
    , _requestData(requestData)
    , _response(response)
    , _done(done)
    , _httpClosure(httpClosure)
{
    assert(_param);
}

ArpcGraphClosure::~ArpcGraphClosure() {
}

void ArpcGraphClosure::run(NaviUserResultPtr result, const std::shared_ptr<ComputeScope> &computeScope) {
    std::string errorMsg;
    if (!doRun(std::move(result), computeScope, errorMsg)) {
        if (!needReplace()) {
            _controller->SetFailed(errorMsg);
        }
    }
    _requestData.reset();
    _done->Run();
    delete this;
}

bool ArpcGraphClosure::doRun(NaviUserResultPtr result,
                             const std::shared_ptr<ComputeScope> &computeScope,
                             std::string &errorMsg) const
{
    if (!result) {
        errorMsg = "run method " + _param->arpcParam.method + " failed: empty result";
        NAVI_KERNEL_LOG(ERROR,
                        "run arpc graph failed, empty result, method [%s]",
                        _param->arpcParam.method.c_str());
        return false;
    }
    std::string queryKey;
    bool success = false;
    {
        ThreadMemLeakDeduperScope deduperScope(
            computeScope ? computeScope->getThreadMemLeakCollector() : nullptr);
        success = processResult(result, queryKey);
        if (!success) {
            result->updateError(EC_ARPC_RESULT);
        }
    }
    auto naviResult = result->getNaviResult();
    {
        ThreadMemLeakDeduperScope deduperScope(
            computeScope ? computeScope->getThreadMemLeakCollector() : nullptr);
        logQuery(result->_hostInfo, naviResult, queryKey);
    }
    if (!naviResult) {
        errorMsg = "run method " + _param->arpcParam.method + " failed: get navi result failed";
        return false;
    }
    result.reset();
    if (needReplace()) {
        replaceProtoJsonizer(naviResult, computeScope);
    }
    if (!success) {
        errorMsg = "run method " + _param->arpcParam.method + " failed, navi error msg: ";
        const auto &errorEvent = naviResult->getErrorEvent();
        if (errorEvent)  {
            errorMsg += errorEvent->message;
        }
        return false;
    }
    auto gigClosure = dynamic_cast<multi_call::GigClosure *>(_done);
    if (gigClosure) {
        auto ec = multi_call::MULTI_CALL_ERROR_NONE;
        if (EC_NONE != naviResult->getErrorCode()) {
            ec = multi_call::MULTI_CALL_ERROR_DEC_WEIGHT;
        }
        gigClosure->setErrorCode(ec);
    }
    return true;
}

bool ArpcGraphClosure::processResult(const NaviUserResultPtr &result, std::string &queryKey) const {
    bool streamMode = false;
    while (true) {
        NaviUserData userData;
        bool eof = false;
        if (!result->nextData(userData, eof)) {
            NAVI_KERNEL_LOG(
                ERROR, "run arpc graph failed, result has no data, method [%s]",
                _param->arpcParam.method.c_str());
            return false;
        }
        if (!eof) {
            streamMode = true;
        }
        if (!processData(streamMode, userData, eof, queryKey)) {
            return false;
        }
        if (eof) {
            break;
        }
    }
    return true;
}

bool ArpcGraphClosure::processData(bool streamMode,
                                   const NaviUserData &userData,
                                   bool eof,
                                   std::string &queryKey) const {
    if (streamMode) {
        return processStreamData(userData, eof, queryKey);
    } else {
        std::unique_ptr<google::protobuf::Message> pbMsg(getDataMessage(userData.data, queryKey));
        if (pbMsg) {
            auto responseRefl = _response->GetReflection();
            auto dataRefl = pbMsg->GetReflection();
            if (responseRefl != dataRefl) {
                const auto &responseTypeName =
                    _response->GetDescriptor()->full_name();
                const auto &dataTypeName = pbMsg->GetDescriptor()->full_name();
                NAVI_KERNEL_LOG(
                    ERROR,
                    "run arpc graph failed, invalid protobuf msg format in "
                    "response data, expect [%s], got [%s], method [%s]",
                    responseTypeName.c_str(), dataTypeName.c_str(),
                    _param->arpcParam.method.c_str());
                return false;
            }
            responseRefl->Swap(_response, pbMsg.get());
            return true;
        } else {
            return false;
        }
    }
}

bool ArpcGraphClosure::processStreamData(const NaviUserData &userData, bool eof, std::string &queryKey) const {
    if (!_httpClosure) {
        NAVI_KERNEL_LOG(ERROR, "not http protocol, can not process stream data");
        return false;
    }
    auto protoJsonizer = _httpClosure->getProtoJsonizer();
    auto streamProtoJsonizer =
        dynamic_cast<NaviStreamProtoJsonizer *>(protoJsonizer.get());
    if (!streamProtoJsonizer) {
        NAVI_KERNEL_LOG(ERROR,
                        "result is data stream, which is not supported by this "
                        "proto jsonizer, method [%s]",
                        _param->arpcParam.method.c_str());
        return false;
    }
    std::unique_ptr<google::protobuf::Message> pbMsg(getDataMessage(userData.data, queryKey));
    if (pbMsg) {
        if (_response->GetReflection() != pbMsg->GetReflection()) {
            const auto &responseTypeName =
                _response->GetDescriptor()->full_name();
            const auto &dataTypeName = pbMsg->GetDescriptor()->full_name();
            NAVI_KERNEL_LOG(
                ERROR,
                "run arpc graph failed, invalid protobuf msg format in "
                "response data, expect [%s], got [%s], method [%s]",
                responseTypeName.c_str(), dataTypeName.c_str(),
                _param->arpcParam.method.c_str());
            return false;
        }
        if (!streamProtoJsonizer->collect(_response, userData, eof,
                                          pbMsg.release()))
        {
            NAVI_KERNEL_LOG(
                ERROR,
                "StreamProtoJsonizer collect message failed, method [%s]",
                _param->arpcParam.method.c_str());
            return false;
        }
        return true;
    } else {
        return false;
    }
}

google::protobuf::Message *
ArpcGraphClosure::getDataMessage(const DataPtr &data, std::string &queryKey) const {
    if (!data) {
        NAVI_KERNEL_LOG(
            ERROR, "run arpc graph failed: result data is null, method [%s]",
            _param->arpcParam.method.c_str());
        return nullptr;
    }
    auto responseData = dynamic_cast<NaviArpcResponseData *>(data.get());
    if (!responseData) {
        NAVI_KERNEL_LOG(ERROR,
                        "run arpc graph failed, invalid result data type [%s], "
                        "only support [%s], method [%s]",
                        data->getTypeId().c_str(),
                        NaviArpcResponseData::TYPE_ID.c_str(), _param->arpcParam.method.c_str());
        return nullptr;
    }
    queryKey = responseData->getQueryKey();
    auto pbMsg = responseData->stealResponse();
    if (!pbMsg) {
        NAVI_KERNEL_LOG(ERROR,
                        "run arpc graph failed, null protobuf msg in response "
                        "data, method [%s]",
                        _param->arpcParam.method.c_str());
        return nullptr;
    }
    if (pbMsg && NAVI_COLLECT_MALLOC()) {
        auto str = pbMsg->SerializeAsString();
        pbMsg->Clear();
        pbMsg->ParseFromString(str);
    }
    return pbMsg;
}

bool ArpcGraphClosure::needReplace() const {
    return (_needDebug && _httpClosure);
}

std::string ArpcGraphClosure::getClientIp() const {
    if (_requestData) {
        return _requestData->getClientIp();
    } else {
        return std::string();
    }
}

void ArpcGraphClosure::replaceProtoJsonizer(
    const NaviResultPtr &naviResult,
    const std::shared_ptr<ComputeScope> &computeScope) const {
    auto visDef = naviResult->getVisProto(computeScope);
    auto oldProtoJsonizer = _httpClosure->getProtoJsonizer();
    if (visDef && oldProtoJsonizer) {
        auto protoJsonizer =
            std::make_shared<NaviDebugProtoJsonizer>(visDef, oldProtoJsonizer);
        _httpClosure->setProtoJsonizer(protoJsonizer);
    }
}

void ArpcGraphClosure::logQuery(const std::shared_ptr<NaviHostInfo> &hostInfo,
                                const NaviResultPtr &naviResult,
                                const std::string &queryKey) const {
    if (!_param->arpcParam.logQuery) {
        return;
    }
    if (!hostInfo) {
        return;
    }
    hostInfo->getHistory()->addQuery(
        _requestData, _param->arpcParam, _httpClosure, naviResult, queryKey, _response, _needDebug);
}
}

#pragma once

#include "navi/engine/NaviUserResult.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/HTTPRPCServerClosure.h"
#include <google/protobuf/service.h>

namespace navi {

class NaviArpcRequestData;
struct ArpcRegistryParam;

class ArpcGraphClosure : public NaviUserResultClosure
{
public:
    ArpcGraphClosure(const std::shared_ptr<ArpcRegistryParam> &param,
                     google::protobuf::RpcController *controller,
                     const std::shared_ptr<NaviArpcRequestData> &requestData,
                     google::protobuf::Message *response,
                     google::protobuf::Closure *done,
                     http_arpc::HTTPRPCServerClosure *httpClosure);
    ~ArpcGraphClosure();
    ArpcGraphClosure(const ArpcGraphClosure &) = delete;
    ArpcGraphClosure &operator=(const ArpcGraphClosure &) = delete;
public:
    void run(NaviUserResultPtr result, const std::shared_ptr<ComputeScope> &computeScope) override;
public:
    void setNeedDebug(bool needDebug) {
        _needDebug = needDebug;
    }
    std::string getClientIp() const;
private:
    bool doRun(NaviUserResultPtr result,
               const std::shared_ptr<ComputeScope> &computeScope,
               std::string &errorMsg) const;
    bool processResult(const NaviUserResultPtr &result, std::string &queryKey) const;
    bool needReplace() const;
    void replaceProtoJsonizer(
        const NaviResultPtr &naviResult,
        const std::shared_ptr<ComputeScope> &computeScope) const;
    bool processData(bool streamMode, const NaviUserData &userData,
                     bool eof, std::string &queryKey) const;
    bool processStreamData(const NaviUserData &userData, bool eof, std::string &queryKey) const;
    google::protobuf::Message *getDataMessage(const DataPtr &data, std::string &queryKey) const;
    void logQuery(const std::shared_ptr<NaviHostInfo> &hostInfo,
                  const NaviResultPtr &naviResult,
                  const std::string &queryKey) const;

private:
    std::shared_ptr<ArpcRegistryParam> _param;
    google::protobuf::RpcController *_controller;
    std::shared_ptr<NaviArpcRequestData> _requestData;
    google::protobuf::Message *_response;
    google::protobuf::Closure *_done;
    http_arpc::HTTPRPCServerClosure *_httpClosure = nullptr;
    bool _needDebug = false;
};

}


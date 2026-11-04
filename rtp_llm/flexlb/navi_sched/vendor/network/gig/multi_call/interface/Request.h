#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/ControllerParam.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/Response.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/proto/GigCommonProto.pb.h"
#include "google/protobuf/arena.h"

namespace multi_call {

using GigMetaEmbedType = GigMetaByTcp;

class GigQueryInfo;
class Request;
MULTI_CALL_TYPEDEF_PTR(Request);
typedef std::map<PartIdTy, RequestPtr> PartRequestMap;
typedef std::vector<RequestPtr> RequestVec;
class SearchServiceResource;
typedef std::map<std::string, std::string> NodeAttributeMap;
class PropagationStats;

class Request
{
public:
    Request(ProtocolType type, const std::shared_ptr<google::protobuf::Arena> &arena);
    virtual ~Request() {
        _queryInfo.reset();
    }

private:
    Request(const Request &);
    Request &operator=(const Request &);

public:
    virtual ResponsePtr newResponse() = 0;
    virtual bool serialize() = 0;
    virtual size_t size() const = 0;
    ;

public:
    ProtocolType getProtocolType() const {
        return _type;
    }
    // ms
    void setTimeout(uint64_t timeout) {
        _timeout = timeout;
    }
    uint64_t getTimeout() const {
        return _timeout;
    }
    std::shared_ptr<GigQueryInfo> &getQueryInfo() {
        return _queryInfo;
    }
    void setBizName(const std::string &bizName);
    const std::string &getBizName() const;
    void setPartId(PartIdTy partId);
    PartIdTy getPartId() const;
    void setRequestType(RequestType type);
    RequestType getRequestType() const;
    void setUserRequestType(RequestType type);
    RequestType getUserRequestType() const;
    void setReturnMetaEnv(bool return_meta_env);
    bool getReturnMetaEnv() const;
    bool isProbeRequest() const {
        return RT_PROBE == getRequestType();
    }
    bool isCopyRequest() const {
        return RT_COPY == getRequestType();
    }
    bool isNormalRequest() const {
        return RT_NORMAL == getRequestType();
    }
    void setMethodName(const std::string &methodName) {
        _methodName = methodName;
    }
    const std::string &getMethodName() const {
        return _methodName;
    }
    std::string getAgentQueryInfo() const;
    std::string getProviderAttribute(const std::string &key) const;
    const std::shared_ptr<google::protobuf::Arena> &getProtobufArena() const {
        return _arena;
    }

    


private:
    void setErrorRatioLimit(float limit);
    void setLatencyLimit(float limit);
    void setLoadBalanceLatencyLimit(float limit);
    void setBeginServerDegradeLatency(float latency);
    void setBeginDegradeLatency(float latency);
    void setProviderWeight(WeightTy weight);
    void setProviderAttributes(const NodeAttributeMap *attrs);
    void setBeginServerDegradeErrorRatio(float ratio);
    void setBeginDegradeErrorRatio(float ratio);
    PropagationStats *getPropagationStats();
    // for test
    float getErrorRatioLimit() const;
    float getLatencyLimit() const;
    float getLoadBalanceLatencyLimit() const;
    float getBeginServerDegradeLatency() const;
    float getBeginDegradeLatency() const;
    int32_t getProviderWeight() const;
    float getBeginServerDegradeErrorRatio() const;
    float getBeginDegradeErrorRatio() const;
    friend class SearchServiceResource;

protected:
    void embedGigData(std::string &body, GigMetaEmbedType metaType) const;

protected:
    std::shared_ptr<google::protobuf::Arena> _arena;
    ProtocolType _type;
    uint64_t _timeout;
    std::string _methodName;
    // queryInfo
    std::shared_ptr<GigQueryInfo> _queryInfo;
    const NodeAttributeMap *_providerAttributeMap;

private:
    AUTIL_LOG_DECLARE();
};

} // namespace multi_call


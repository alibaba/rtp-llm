#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/QuerySessionContext.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/SearchService.h"

namespace multi_call {

struct CallParam {
public:
    CallParam() : closure(NULL) {
    }

public:
    RequestGeneratorVec generatorVec;
    Closure *closure;
};

class QuerySession;
class LoadBalancerContext;
MULTI_CALL_TYPEDEF_PTR(QuerySession);
MULTI_CALL_TYPEDEF_PTR(LoadBalancerContext);

class QuerySession : public std::enable_shared_from_this<QuerySession>
{
public:
    QuerySession(const SearchServicePtr &searchService,
                 const QueryInfoPtr &queryInfo = QueryInfoPtr());
    QuerySession(const SearchServicePtr &searchService,
                 const std::shared_ptr<QuerySessionContext> &context);

    virtual ~QuerySession();

public:
    static QuerySessionPtr get(google::protobuf::Closure *doneClosure);
    static QuerySessionPtr createWithOrigin(const SearchServicePtr &searchService,
                                            const QuerySessionPtr &origin);

public:
    void selectBiz(const RequestGeneratorPtr &generator, BizInfo &bizInfo);
    void call(CallParam &callParam, ReplyPtr &reply);
    bool bind(const GigClientStreamPtr &stream);

    RequestType getRequestType() const {
        return _sessionContext->getRequestType();
    }
    // user_request_type用以传递整个链路是normal, probe or copy类型
    // 比如：服务A->B->C, 从A发起的probe query到B后，触发B->C, 这里user_probe_type=probe,
    // 类型不会丢失
    RequestType getUserRequestType() const {
        return _sessionContext->getUserRequestType();
    }
    RequestType sessionRequestType() const {
        return _sessionContext->sessionRequestType();
    }
    virtual std::string getBiz() const {
        return _sessionContext->getBiz();
    }
    void setBiz(const std::string &biz) {
        _sessionContext->setBiz(biz);
    }
    SearchServicePtr getSearchService() const {
        return _searchService;
    }
    std::shared_ptr<QuerySessionContext> getSessionContext() const {
        return _sessionContext;
    }
    LoadBalancerContextPtr getLoadBalancerContext() const {
        return _loadBalancerContext;
    }
    const QueryInfoPtr &getQueryInfo() const {
        return _sessionContext->getQueryInfo();
    }
    const std::string &getGigData() const {
        return _sessionContext->getGigData();
    }

    void putUserData(const std::string &k, const std::string &v) {
        _sessionContext->putUserData(k, v);
    }
    void putUserData(const std::map<std::string, std::string> &dataMap) {
        _sessionContext->putUserData(dataMap);
    }
    virtual std::string getUserData(const std::string &k) const {
        return _sessionContext->getUserData(k);
    }
    std::map<std::string, std::string> getUserDataMap() const {
        return _sessionContext->getUserDataMap();
    }

public:
    // 用户设置新的鹰眼信息, 创建新的Trace
    
    
    
    

private:
    SearchServicePtr _searchService;
    std::shared_ptr<QuerySessionContext> _sessionContext;
    LoadBalancerContextPtr _loadBalancerContext;

private:
    AUTIL_LOG_DECLARE();
};

} // namespace multi_call


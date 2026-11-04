#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/proto/GigCallProto.pb.h"

namespace multi_call {

class SearchService;
class QueryInfo;
MULTI_CALL_TYPEDEF_PTR(SearchService);
MULTI_CALL_TYPEDEF_PTR(QueryInfo);

class QuerySessionContext
{
public:
    QuerySessionContext(const QueryInfoPtr &queryInfo = QueryInfoPtr());
    ~QuerySessionContext();

public:
    RequestType getRequestType() const {
        return _requestType;
    }
    RequestType getUserRequestType() const {
        return _userRequestType;
    }
    RequestType sessionRequestType() const;
    std::string getBiz() const {
        return _biz;
    }
    void setBiz(const std::string &biz) {
        _biz = biz;
    }

    const QueryInfoPtr &getQueryInfo() const {
        return _queryInfo;
    }
    const std::string &getGigData() const;

    void putUserData(const std::string &k, const std::string &v);
    void putUserData(const std::map<std::string, std::string> &dataMap);
    std::string getUserData(const std::string &k) const;
    std::map<std::string, std::string> getUserDataMap() const;

private:
    RequestType _requestType;
    RequestType _userRequestType;
    std::string _biz;
    QueryInfoPtr _queryInfo;
    std::map<std::string, std::string> _userData;
}; // end class

} // namespace multi_call

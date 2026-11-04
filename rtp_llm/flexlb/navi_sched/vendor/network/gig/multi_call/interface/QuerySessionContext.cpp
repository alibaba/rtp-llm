#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/QuerySessionContext.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/agent/QueryInfo.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/SearchService.h"

using namespace std;
using namespace autil;

namespace multi_call {

QuerySessionContext::QuerySessionContext(const QueryInfoPtr &queryInfo)
    : _requestType(RT_NORMAL)
    , _userRequestType(RT_NORMAL)
    , _queryInfo(queryInfo) {
    if (queryInfo) {
        const GigQueryInfo *info = queryInfo->getQueryInfo();
        if (info->has_request_type()) {
            _requestType = (RequestType)info->request_type();
        }
        if (info->has_user_request_type()) {
            _userRequestType = (RequestType)info->user_request_type();
        }
        if (info->has_biz_name()) {
            _biz = info->biz_name();
        }
    }
}

QuerySessionContext::~QuerySessionContext() = default;

RequestType QuerySessionContext::sessionRequestType() const {
    return _userRequestType != RT_NORMAL ? _userRequestType : _requestType;
}

void QuerySessionContext::putUserData(const std::string &key, const std::string &value) {
    _userData[key] = value;
}

void QuerySessionContext::putUserData(const std::map<std::string, std::string> &values) {
    for (const auto &[key, value] : values) {
        _userData[key] = value;
    }
}

std::string QuerySessionContext::getUserData(const std::string &key) const {
    auto found = _userData.find(key);
    return found == _userData.end() ? EMPTY_STRING : found->second;
}

std::map<std::string, std::string> QuerySessionContext::getUserDataMap() const {
    return _userData;
}

const std::string &QuerySessionContext::getGigData() const {
    if (_queryInfo) {
        return _queryInfo->getQueryInfoStr();
    }
    return EMPTY_STRING;
}

} // namespace multi_call

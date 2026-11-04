#include "navi/resource/QuerySessionR.h"
#include "navi/engine/NaviRegisterMacro.h"

namespace navi {

const std::string QuerySessionR::RESOURCE_ID = "navi.query_session_r";

QuerySessionR::QuerySessionR(const std::shared_ptr<multi_call::QuerySession> &querySession)
    : ExternalResource(RESOURCE_ID, RS_RUN_GRAPH_EXTERNAL)
    , _querySession(querySession)
{
    initTags();
}

QuerySessionR::QuerySessionR()
    : ExternalResource(RESOURCE_ID, RS_RUN_GRAPH_EXTERNAL)
{
}

QuerySessionR::~QuerySessionR() {
}

std::map<std::string, std::string> QuerySessionR::getTagsMap() const {
    autil::ScopedReadLock lock(_lock);
    return _tagsMap;
}

std::string QuerySessionR::getQueryStrategy(const std::string &bizName) const {
    if (_isStressQuery) {
        return multi_call::getSubFlowControlStrategy(bizName, multi_call::RPC_DATA_STRESS_TEST);
    } else {
        return bizName;
    }
}

void QuerySessionR::addTag(const std::string &tag, const std::string &value) {
    autil::ScopedWriteLock lock(_lock);
    _tagsMap[tag] = value;
    if ("src" == tag) {
        if (_querySession) {
            if (unlikely(NAVI_COLLECT_MALLOC())) {

                _querySession->putUserData(
                    std::string(tag.c_str(), tag.size()),
                    std::string(value.c_str(), value.size()));
            } else {
                _querySession->putUserData(tag, value);
            }
        }
    }
}

void QuerySessionR::initTags() {

    addGigTag("src");
    addGigTag(multi_call::RPC_DATA_STRESS_TEST);
    if (_querySession) {
        auto requestType = _querySession->getRequestType();
        auto userRequestType = _querySession->getUserRequestType();
        if (multi_call::RT_NORMAL != requestType) {
            _tagsMap[multi_call::RPC_DATA_STRESS_TEST] = multi_call::convertRequestTypeToStr(requestType);
        } else if (multi_call::RT_NORMAL != userRequestType) {
            _tagsMap[multi_call::RPC_DATA_STRESS_TEST] = "USER_" + multi_call::convertRequestTypeToStr(userRequestType);
        }
    }
}

void QuerySessionR::addGigTag(const std::string &tagk) {
    if (!_querySession) {
        _tagsMap[tagk] = "unknown";
        return;
    }
    const auto &value = _querySession->getUserData(tagk);
    if (value.empty()) {
        _tagsMap[tagk] = "unknown";
    } else {
        _tagsMap[tagk] = value;
        if (multi_call::RPC_DATA_STRESS_TEST == tagk && ("1" == value || "2" == value)) {
            _isStressQuery = true;
        }
    }
}

REGISTER_BUILDIN_RESOURCE(QuerySessionR);

}

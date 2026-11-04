#pragma once

#include "navi/engine/Resource.h"
#include <multi_call/interface/QuerySession.h>

namespace navi {

class QuerySessionR : public ExternalResource
{
public:
    QuerySessionR(const std::shared_ptr<multi_call::QuerySession> &querySession);
    QuerySessionR();
    ~QuerySessionR();
    QuerySessionR(const QuerySessionR &) = delete;
    QuerySessionR &operator=(const QuerySessionR &) = delete;
public:
    const std::shared_ptr<multi_call::QuerySession> &getQuerySession() const {
        return _querySession;
    }
    std::map<std::string, std::string> getTagsMap() const;
    std::string getQueryStrategy(const std::string &bizName) const;
    void addTag(const std::string &tag, const std::string &value);
private:
    void initTags();
    void addGigTag(const std::string &tagk);
public:
    static const std::string RESOURCE_ID;
private:
    mutable autil::ReadWriteLock _lock;
    std::shared_ptr<multi_call::QuerySession> _querySession;
    std::map<std::string, std::string> _tagsMap;
    bool _isStressQuery = false;
};

NAVI_TYPEDEF_PTR(QuerySessionR);

}


#pragma once

#include "navi/proto/GraphVis.pb.h"
#include "navi/common.h"
#include "navi/log/LoggingEvent.h"

namespace navi {

class NaviError;
NAVI_TYPEDEF_PTR(NaviError);

class NaviLogger;

class NaviError
{
public:
    NaviError() {
    }
    ~NaviError() {
    }
    NaviError(const NaviError &) = delete;
    NaviError &operator=(const NaviError &) = delete;
public:
    void toProto(NaviErrorDef &errorDef) const;
    void fromProto(const NaviErrorDef &errorDef);
public:
    static NaviErrorPtr makeError(const std::shared_ptr<NaviLogger> &logger, ErrorCode ec);
private:
    static void parseSessionId(const SessionIdDef &session, SessionId &id);
    static void fillSessionId(SessionIdDef *session, const SessionId &id);
private:
    friend class QueryLRUEntry;
    friend class GraphResult;
    friend struct LocalSubGraphInternal;
public:
    SessionId id;
    ErrorCode ec = EC_NONE;
    LoggingEventPtr errorEvent;
    ResultLocationDef location;
    const void *result = nullptr;
};

}


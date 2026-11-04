#include "navi/engine/NaviError.h"
#include "navi/util/CommonUtil.h"

namespace navi {

NaviErrorPtr NaviError::makeError(const NaviLoggerPtr &logger, ErrorCode ec) {
    if (EC_NONE == ec || !logger) {
        return nullptr;
    }
    auto errorPtr = std::make_shared<NaviError>();
    auto &error = *errorPtr;
    error.id = logger->getLoggerId();
    error.ec = ec;
    error.errorEvent = logger->firstErrorEvent();
    return errorPtr;
}

void NaviError::toProto(NaviErrorDef &errorDef) const {
    fillSessionId(errorDef.mutable_session(), id);
    errorDef.set_result_ptr((uint64_t)result);
    errorDef.set_ec(ec);
    if (EC_NONE != ec) {
        errorDef.set_ec_str(CommonUtil::getErrorString(ec));
        if (errorEvent) {
            errorEvent->toProto(errorDef.mutable_error_event(), nullptr);
        }
        if (location.locations_size() > 0) {
            errorDef.mutable_location()->CopyFrom(location);
        }
    }
}

void NaviError::fromProto(const NaviErrorDef &errorDef) {
    parseSessionId(errorDef.session(), id);
    result = (void *)errorDef.result_ptr();
    ec = (ErrorCode)errorDef.ec();
    if (EC_NONE != ec) {
        if (errorDef.has_error_event()) {
            errorEvent = std::make_shared<LoggingEvent>();
            errorEvent->fromProto(errorDef.error_event());
        }
        if (errorDef.has_location()) {
            location.CopyFrom(errorDef.location());
        }
    }
}

void NaviError::parseSessionId(const SessionIdDef &session, SessionId &id) {
    id.instance = session.instance();
    id.currentInstance = session.current_instance();
    id.queryId = session.query_id();
}

void NaviError::fillSessionId(SessionIdDef *sessionDef, const SessionId &id) {
    sessionDef->set_instance(id.instance);
    sessionDef->set_current_instance(id.currentInstance);
    sessionDef->set_query_id(id.queryId);
}

}


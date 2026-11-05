#pragma once

#include "autil/legacy/jsonizable.h"

#include <exception>
#include <string>

namespace navi_sched {

inline bool executionGeometryEnabledByFingerprint(const std::string &fingerprint)
{
    try {
        autil::legacy::json::JsonMap fields;
        autil::legacy::FromJsonString(fields, fingerprint);
        return autil::legacy::AnyCast<bool>(fields.at("execution_geometry_enabled"));
    } catch (const std::exception &) {
        return false;
    }
}

inline bool projectedExecutionGeometryEnabledByFingerprint(const std::string &fingerprint)
{
    try {
        autil::legacy::json::JsonMap fields;
        autil::legacy::FromJsonString(fields, fingerprint);
        return autil::legacy::AnyCast<bool>(
            fields.at("projected_execution_geometry_enabled"));
    } catch (const std::exception &) {
        return false;
    }
}

} // namespace navi_sched

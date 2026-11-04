#pragma once

#include <string>

#include "autil/legacy/exception.h"

namespace autil {
namespace legacy {

class NotJsonizableException : public ExceptionBase {
public:
    AUTIL_LEGACY_DEFINE_EXCEPTION(NotJsonizableException, ExceptionBase);
};

class TypeNotMatchException : public ExceptionBase {
public:
    AUTIL_LEGACY_DEFINE_EXCEPTION(TypeNotMatchException, ExceptionBase);
};

class WrongFormatJsonException : public ExceptionBase {
public:
    AUTIL_LEGACY_DEFINE_EXCEPTION(WrongFormatJsonException, ExceptionBase);
};

class NotSupportException : public ExceptionBase {
public:
    AUTIL_LEGACY_DEFINE_EXCEPTION(NotSupportException, ExceptionBase);
};

} // namespace legacy
} // namespace autil

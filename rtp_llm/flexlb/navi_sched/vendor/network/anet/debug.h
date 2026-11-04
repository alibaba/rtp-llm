#pragma once

#include <assert.h>

namespace anet {

#define DBGASSERT(exp) assert(exp)

#define ERRSTR(errcodevalue)                                                                                           \
    char *errStr;                                                                                                      \
    char errStrBuf[100];                                                                                               \
    errStr = strerror_r(errcodevalue, &errStrBuf[0], 100);

} // namespace anet

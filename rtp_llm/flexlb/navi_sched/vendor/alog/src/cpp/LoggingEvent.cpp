#include "LoggingEvent.h"

namespace alog {

const char *s_gLevelName[] = {
    "DISABLE", "FATAL", "ERROR", "WARN", "INFO", "DEBUG", "TRACE1", "TRACE2", "TRACE3", "NOTSET"};

int32_t sysPid = getpid();
thread_local int64_t sysTid = (long)syscall(SYS_gettid);

} // namespace alog

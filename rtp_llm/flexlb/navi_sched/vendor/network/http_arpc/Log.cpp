#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/Log.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/alogadapter.h"

namespace http_arpc {

AlogAdapter alogadapter(std::string("http_arpc"));
ILogger *httparpclogger = &alogadapter;

} // namespace http_arpc

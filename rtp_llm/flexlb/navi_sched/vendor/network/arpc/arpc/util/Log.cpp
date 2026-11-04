#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/util/Log.h"

#include <string>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/alogadapter.h"

ARPC_BEGIN_NAMESPACE(arpc);

AlogAdapter alogadapter(std::string("arpc"));
ILogger *arpclogger = &alogadapter;

ARPC_END_NAMESPACE(arpc);

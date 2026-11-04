#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCChannelManagerBase.h"

#include <cstddef>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/connection.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ilogger.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/runnable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/util/Log.h"

using namespace std;
using namespace anet;
ARPC_BEGIN_NAMESPACE(arpc);
ARPC_DECLARE_AND_SETUP_LOGGER(RPCChannelManagerBase);

RPCChannelManagerBase::RPCChannelManagerBase() {}

RPCChannelManagerBase::~RPCChannelManagerBase() {}

void RPCChannelManagerBase::Close() {}

ARPC_END_NAMESPACE(arpc);

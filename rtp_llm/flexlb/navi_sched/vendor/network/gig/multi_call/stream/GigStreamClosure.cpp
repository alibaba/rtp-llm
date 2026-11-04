#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigStreamClosure.h"

using namespace std;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, GigStreamSendClosure);
AUTIL_LOG_SETUP(multi_call, GigStreamReceiveClosure);
AUTIL_LOG_SETUP(multi_call, GigStreamInitClosure);
AUTIL_LOG_SETUP(multi_call, GigStreamFinishClosure);
AUTIL_LOG_SETUP(multi_call, GigStreamCancelClosure);

void GrpcStreamBackWorkItem::process() {
    _closure->finishClosure();
}

void GrpcStreamBackWorkItem::destroy() {
    delete _closure;
}

void GrpcStreamBackWorkItem::drop() {
    _closure->finishClosure();
    destroy();
}

} // namespace multi_call

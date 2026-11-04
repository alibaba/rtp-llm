#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/arpc/CommonClosure.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/agent/QueryInfo.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/HTTPRPCServerClosure.h"
#include "autil/legacy/base64.h"

using namespace std;
using namespace autil;
using namespace autil::legacy;

namespace multi_call {

void CommonClosure::Run() {
    if (_closure) {
        auto httpClosure = dynamic_cast<http_arpc::HTTPRPCServerClosure *>(_closure);
        if (httpClosure) {
            const QueryInfoPtr &queryInfo = getQueryInfo();
            if (queryInfo) {
                string responseInfo = queryInfo->finish(
                    (TimeUtility::currentTime() - getStartTime()) / FACTOR_US_TO_MS, getErrorCode(),
                    getTargetWeight());
                httpClosure->addResponseHeader(GIG_DATA, Base64EncodeFast(responseInfo));
            }
        }

        _closure->Run();
    }
    delete this;
}

} // namespace multi_call

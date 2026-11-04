#include "navi/util/EnvParam.h"

#include "autil/EnvUtil.h"

namespace navi {

static const std::string NAVI_ENV_FORCE_WAIT_NO_QUERY_TIME = "forceWaitNoQueryTime";
static constexpr int64_t DEFAULT_FORCE_WAIT_NO_QUERY_TIME = 10; // 10s

EnvParam::EnvParam() {
    forceWaitNoQueryTime = autil::EnvUtil::getEnv(NAVI_ENV_FORCE_WAIT_NO_QUERY_TIME, DEFAULT_FORCE_WAIT_NO_QUERY_TIME);
}

EnvParam::~EnvParam() {
}

const EnvParam &EnvParam::inst() {
    static EnvParam inst;
    return inst;
}

} // namespace navi

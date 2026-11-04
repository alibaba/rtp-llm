#include "autil/DailyRunMode.h"

#include <iosfwd>
#include <stdlib.h>

#include "autil/EnvUtil.h"
#include "autil/Log.h"

using namespace std;

namespace autil {
AUTIL_DECLARE_AND_SETUP_LOGGER(autil, DailyRunMode);

const std::string DailyRunMode::DAILY_RUN_MODE("DAILY_RUN_MODE");
bool DailyRunMode::_isEnable = autil::EnvUtil::hasEnv(DAILY_RUN_MODE);
bool DailyRunMode::enable() { return _isEnable; }

} // namespace autil

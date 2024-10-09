#include "src/fastertransformer/th_op/common/LogLevelOps.h"

using namespace fastertransformer;

namespace torch_ext {

void setDebugLogLevel(bool debug) {
    auto& logger = Logger::getEngineLogger();
    if (debug) {
        logger.setBaseLevel(alog::LOG_LEVEL_DEBUG);
    } else {
        logger.setBaseLevel(alog::LOG_LEVEL_INFO);
    }
}

} // namespace torch_ext

// maybe faster than torch copy
static auto debug_log_level_func =
    torch::RegisterOperators("fastertransformer::set_log_level", &torch_ext::setDebugLogLevel);



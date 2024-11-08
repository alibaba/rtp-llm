#include "src/fastertransformer/th_op/common/LogLevelOps.h"

namespace torch_ext {

bool setLogLevel(const std::string& log_level_str) {
    uint32_t log_level = alog::LOG_LEVEL_INFO;
    if (log_level_str == "INFO") {
        log_level = alog::LOG_LEVEL_INFO;
    } else if (log_level_str == "DEBUG") {
        log_level = alog::LOG_LEVEL_DEBUG;
    } else if (log_level_str == "TRACE") {
        log_level = alog::LOG_LEVEL_TRACE1;
    } else {
        FT_LOG_WARNING("set log level failed, unknown log level: %s", log_level_str.c_str());
        return false;
    }
    auto& logger = rtp_llm::Logger::getEngineLogger();
    logger.setBaseLevel(log_level);
    return true;
}

// maybe faster than torch copy
static auto log_level_func =
    torch::RegisterOperators("fastertransformer::set_log_level", &torch_ext::setLogLevel);

} // namespace torch_ext


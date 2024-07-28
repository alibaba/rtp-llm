#include "src/fastertransformer/th_op/common/LogLevelOps.h"

using namespace fastertransformer;

namespace torch_ext {

void setDebugLogLevel(bool debug) {
    auto& logger = Logger::getLogger();
    if (debug) {
        logger.setLevel(Logger::Level::DEBUG);
    } else {
        logger.setLevel(Logger::Level::INFO);
    }
}

void setDebugPrintLevel(bool debug) {
    auto& logger = Logger::getLogger();
    if (debug) {
        logger.setPrintLevel(Logger::Level::DEBUG);
    } else {
        logger.setPrintLevel(Logger::Level::INFO);
    }
}

} // namespace torch_ext

// maybe faster than torch copy
static auto debug_log_level_func =
    torch::RegisterOperators("fastertransformer::set_debug_log_level", &torch_ext::setDebugLogLevel);

// maybe faster than torch copy
static auto debug_print_level_func =
    torch::RegisterOperators("fastertransformer::set_debug_print_level", &torch_ext::setDebugPrintLevel);



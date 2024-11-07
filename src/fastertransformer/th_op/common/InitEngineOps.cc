#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "autil/EnvUtil.h"

#include "maga_transformer/cpp/utils/SignalUtils.h"
#include "src/fastertransformer/th_op/common/InitEngineOps.h"

namespace torch_ext {

bool initLogger() {
    std::string log_conf_file = autil::EnvUtil::getEnv("FT_ALOG_CONF_PATH", "");
    if ("" == log_conf_file) {
        bool exist = std::filesystem::exists("alog.conf");
        if (!exist) {
            AUTIL_ROOT_LOG_CONFIG();
            AUTIL_ROOT_LOG_SETLEVEL(INFO);
            return true;
        }
        log_conf_file = "alog.conf";
    }

    bool exist = std::filesystem::exists(log_conf_file);

    if (exist) {
        try {
            alog::Configurator::configureLogger(log_conf_file.c_str());
        } catch (std::exception &e) {
            std::cerr << "Failed to configure logger. Logger config file [" << log_conf_file << "], errorMsg ["
                      << e.what() << "]." << std::endl;
            return false;
        }
    } else {
        std::cerr << "log config file [" << log_conf_file << "] doesn't exist. errorCode: [" << std::endl;
        return false;
    }

    return true;
}

void initEngine() {
        if (!initLogger()) {
        std::runtime_error("init logger failed");
    }

    FT_LOG_INFO("install sighandler begin");
    if (!rtp_llm::installSighandler()) {
        std::cerr << "install sighandler failed" << std::endl;
        std::runtime_error("install sighandler failed");
    }
    FT_LOG_INFO("install sighandler success");
    return;
}

static auto init_engine_func =
    torch::RegisterOperators("fastertransformer::init_engine", &torch_ext::initEngine);

} // namespace torch_ext


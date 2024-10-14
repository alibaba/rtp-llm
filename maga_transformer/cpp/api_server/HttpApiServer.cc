#include "maga_transformer/cpp/api_server/HttpApiServer.h"
#include "src/fastertransformer/utils/logger.h"

namespace rtp_llm {

void HttpApiServer::init_controller(const ft::GptInitParameter& params) {
    bool block = autil::EnvUtil::getEnv("CONCURRENCY_WITH_BLOCK", false);
    if (params.tp_rank_ == 0) {
        int limit = autil::EnvUtil::getEnv("CONCURRENCY_LIMIT", 32);
        FT_LOG_INFO("CONCURRENCY_LIMIT to %d", limit);
        controller_ = std::make_shared<ConcurrencyController>(limit, block);
    } else /* if (params.tp_size_ != 1) */ {
        FT_LOG_INFO("use gang cluster and is worker, set CONCURRENCY_LIMIT to 99");
        controller_ = std::make_shared<ConcurrencyController>(99, block);
    }
}

bool HttpApiServer::start(const std::string& address) {
    http_server_.reset(new http_server::HttpServer());

    if (!registerServices()) {
        FT_LOG_ERROR("HttpApiServer start failed, register services failed, address is %s.", address.c_str());
        return false;
    }

    if (!http_server_->Start(address)) {
        FT_LOG_ERROR("HttpApiServer start failed, start http server failed, address is %s.", address.c_str());
        return false;
    }

    is_stoped_.store(false);
    FT_LOG_INFO("HttpApiServer start success, listen address is %s.", address.c_str());
    return true;
}

bool HttpApiServer::start() {
    return start(addr_);
}

bool HttpApiServer::registerServices() {
    FT_LOG_INFO("HttpApiServer register services success.");
    return true;
}

void HttpApiServer::stop() {
    is_stoped_.store(true);

    // stop http server
    // TODO: maybe wait all request finished

    if (http_server_) {
        http_server_->Stop();
    }
}

bool HttpApiServer::isStoped() const {
    // TODO:
    return is_stoped_.load();
}

}  // namespace rtp_llm

#pragma once

#include <atomic>
#include <string>

#include "maga_transformer/cpp/engine_base/EngineBase.h"
#include "maga_transformer/cpp/embedding_engine/EmbeddingEngine.h"
#include "maga_transformer/cpp/multimodal_processor/MultimodalProcessor.h"
#include "maga_transformer/cpp/utils/ConcurrencyControllerUtil.h"

#include "maga_transformer/cpp/http_server/http_server/HttpServer.h"
#include "maga_transformer/cpp/api_server/Pipeline.h"
#include "maga_transformer/cpp/api_server/EmbeddingEndpoint.h"
#include "autil/AtomicCounter.h"

namespace rtp_llm {

class HttpApiServer {
public:
    // normal engine
    HttpApiServer(std::shared_ptr<EngineBase> engine,
                  std::string                 address,
                  const ft::GptInitParameter& params,
                  py::object                  token_processor):
        engine_(engine), addr_(address), params_(params), pipeline_(new Pipeline(token_processor)) {

        request_counter_.reset(new autil::AtomicCounter());
        init_controller(params);
    }

    // embedding engine
    HttpApiServer(std::shared_ptr<EmbeddingEngine>     embedding_engine,
                  std::shared_ptr<MultimodalProcessor> mm_processor,
                  const ft::GptInitParameter&          params,
                  py::object                           py_render):
        params_(params), embedding_endpoint_(EmbeddingEndpoint(embedding_engine, mm_processor, py_render)) {

        request_counter_.reset(new autil::AtomicCounter());
        init_controller(params);
    }

    ~HttpApiServer() = default;

public:
    bool        start();
    bool        start(const std::string& address);
    void        stop();
    bool        isStoped() const;
    std::string getListenAddr() const {
        return addr_;
    }

private:
    void init_controller(const ft::GptInitParameter& params);

private:
    bool registerServices();

private:
    std::atomic_bool                      is_stoped_{true};
    std::shared_ptr<autil::AtomicCounter> request_counter_;

    std::shared_ptr<EngineBase>            engine_;
    std::string                            addr_;
    ft::GptInitParameter                   params_;
    std::shared_ptr<ConcurrencyController> controller_;
    std::shared_ptr<Pipeline>              pipeline_;

    std::optional<EmbeddingEndpoint> embedding_endpoint_;

    std::unique_ptr<http_server::HttpServer> http_server_;
};

}  // namespace rtp_llm

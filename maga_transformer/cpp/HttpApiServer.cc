#include <cstdlib>
#include "maga_transformer/cpp/HttpApiServer.h"
#include "autil/AtomicCounter.h"
#include "autil/legacy/json.h"
#include "autil/legacy/jsonizable.h"
#include "autil/EnvUtil.h"
#include "src/fastertransformer/utils/py_utils/pybind_utils.h"
#include "maga_transformer/cpp/dataclass/LoadBalance.h"
#include "maga_transformer/cpp/dataclass/Query.h"

namespace torch_ext {
extern bool setLogLevel(const std::string& log_level);
}  // namespace torch_ext

namespace rtp_llm {

using namespace std::placeholders;
using namespace autil::legacy;
using namespace autil::legacy::json;

class AuxInfoAdapter: public Jsonizable, public AuxInfo {
public:
    void Jsonize(Jsonizable::JsonWrapper& json) override {
        json.Jsonize("cost_time_ms", cost_time_ms, cost_time_ms);
        json.Jsonize("iter_count", iter_count, iter_count);
        json.Jsonize("input_len", input_len, input_len);
        json.Jsonize("prefix_len", prefix_len, prefix_len);
        json.Jsonize("reuse_len", reuse_len, reuse_len);
        json.Jsonize("output_len", output_len, output_len);
        json.Jsonize("fallback_tokens", fallback_tokens, fallback_tokens);
        json.Jsonize("fallback_times", fallback_times, fallback_times);
    }
    AuxInfoAdapter() {
        AuxInfo();
    }
    AuxInfoAdapter(const AuxInfo& base) {
        cost_time_us    = base.cost_time_us;
        iter_count      = base.iter_count;
        input_len       = base.input_len;
        prefix_len      = base.prefix_len;
        reuse_len       = base.reuse_len;
        output_len      = base.output_len;
        fallback_tokens = base.fallback_tokens;
        fallback_times  = base.fallback_times;

        cost_time_ms = cost_time_us / 1000.0;
    }
    float cost_time_ms;
};

struct PipelineResponse: public Jsonizable {
    void Jsonize(Jsonizable::JsonWrapper& json) override {
        json.Jsonize("response", response, response);
        json.Jsonize("finished", finished, finished);
        json.Jsonize("aux_info", aux_info, aux_info);
    }
    std::string    response;
    bool           finished;
    AuxInfoAdapter aux_info;
};

class TokenizerEncodeResponse: public Jsonizable {
public:
    TokenizerEncodeResponse()           = default;
    ~TokenizerEncodeResponse() override = default;

public:
    void Jsonize(Jsonizable::JsonWrapper& json) override {
        if (!offset_mapping.empty()) {
            json.Jsonize("offset_mapping", offset_mapping);
        }
        json.Jsonize("token_ids", token_ids, token_ids);
        json.Jsonize("tokens", tokens, tokens);
        json.Jsonize("error", error, error);
    }

public:
    std::vector<std::vector<int>> offset_mapping;
    std::vector<std::string>      tokens;
    std::vector<int>              token_ids;
    std::string                   error;
};

class WorkerStatusResponse: public Jsonizable {
public:
    void Jsonize(Jsonizable::JsonWrapper& json) override {
        json.Jsonize("available_concurrency", available_concurrency);
        json.Jsonize("available_kv_cache", load_balance_info.available_kv_cache);
        json.Jsonize("total_kv_cache", load_balance_info.total_kv_cache);
        json.Jsonize("step_latency_ms", load_balance_info.step_latency_us / 1000.0);
        json.Jsonize("step_per_minute", load_balance_info.step_per_minute);
        json.Jsonize("iterate_count", load_balance_info.iterate_count);
        json.Jsonize("alive", alive);
    }

public:
    int             available_concurrency;
    LoadBalanceInfo load_balance_info;
    bool            alive;
};

class ErrorResponse: public Jsonizable {
public:
    void Jsonize(Jsonizable::JsonWrapper& json) override {
        json.Jsonize("error_code", error_code, error_code);
        json.Jsonize("message", error_msg, error_msg);
    }

public:
    int         error_code;
    std::string error_msg;
};

inline std::string CreateErrorResponseJsonString(int error_code, const std::string& error_msg) {
    ErrorResponse response;
    response.error_code = error_code;
    response.error_msg  = error_msg;
    return ToJsonString(response, /*isCompact=*/true);
}

class ParallelInfo {
public:
    static bool isMaster() {
        int world_rank = autil::EnvUtil::getEnv("WORLD_RANK", 0);
        return world_rank == 0;
    }
    static bool isWorker() {
        return !isMaster();
    }
};

autil::AtomicCounter requestCounter;

void inferResponse(std::unique_ptr<http_server::HttpResponseWriter> writer,
                   const http_server::HttpRequest&                  request,
                   std::shared_ptr<EngineBase>                      engine,
                   ft::GptInitParameter                             params,
                   Pipeline                                         pipeline_,
                   std::shared_ptr<ConcurrencyController>           controller_) {

    if (controller_->increment() == false) {
        writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
        writer->SetStatus(429, "Too Many Requests");
        writer->Write("");
        return;
    }

    std::shared_ptr<GenerateInput> input = std::make_shared<GenerateInput>();
    input->request_id                    = requestCounter.incAndReturn();
    input->begin_time_ms                 = autil::TimeUtility::currentTimeInMicroSeconds();
    input->generate_config               = std::make_shared<GenerateConfig>();

    auto body    = ParseJson(request.GetBody());
    auto bodyMap = AnyCast<JsonMap>(body);

    // generate_config
    auto it = bodyMap.find("generate_config");
    if (it == bodyMap.end()) {
        FT_LOG_INFO("no generate_config in http request.");
    } else {
        FromJson(input->generate_config, it->second);
    }

    // merge stop_words_list
    for (const auto& innerVec : params.special_tokens_.stop_words_list_) {
        std::vector<int> tmpVec;
        for (int64_t val : innerVec) {
            tmpVec.push_back(static_cast<int>(val));
        }
        input->generate_config->stop_words_list.push_back(tmpVec);
    }

    // merge stop_words_str
    std::vector<std::string> stop_words_str;
    auto                     generate_config_map = AnyCast<JsonMap>(it->second);
    it                                           = generate_config_map.find("stop_words_str");
    if (it == generate_config_map.end()) {
        FT_LOG_INFO("no stop_words_str in http request.");
    } else {
        auto words = AnyCast<JsonArray>(it->second);
        for (auto word : words) {
            stop_words_str.push_back(AnyCast<std::string>(word));
        }
    }
    stop_words_str.insert(stop_words_str.begin(),
                          params.special_tokens_.stop_words_str_.begin(),
                          params.special_tokens_.stop_words_str_.end());

    // urls/images: list[list[str]]
    it = bodyMap.find("urls");
    if (it != bodyMap.end()) {
        auto                         listListUrls = AnyCast<JsonArray>(it->second);
        auto                         listUrls     = AnyCast<JsonArray>(listListUrls[0]);
        std::vector<MultimodalInput> mm_inputs;
        for (auto url : listUrls) {
            mm_inputs.emplace_back(AnyCast<std::string>(url));
        }
        input->multimodal_inputs = std::move(mm_inputs);
    } else {
        FT_LOG_INFO("no urls in http request.");
        it = bodyMap.find("images");
        if (it != bodyMap.end()) {
            auto                         listListUrls = AnyCast<JsonArray>(it->second);
            auto                         listUrls     = AnyCast<JsonArray>(listListUrls[0]);
            std::vector<MultimodalInput> mm_inputs;
            for (auto url : listUrls) {
                mm_inputs.emplace_back(AnyCast<std::string>(url));
            }
            input->multimodal_inputs = std::move(mm_inputs);
        } else {
            FT_LOG_INFO("no images in http request.");
        }
    }

    it                 = bodyMap.find("prompt");
    std::string prompt = "hello";
    if (it == bodyMap.end()) {
        FT_LOG_INFO("no prompt in http request.");
    } else {
        prompt = AnyCast<std::string>(it->second);
    }

    auto vec = pipeline_.encode(prompt);

    auto device      = ft::DeviceFactory::getDefaultDevice();
    input->input_ids = device->allocateBuffer({ft::DataType::TYPE_INT32, {vec.size()}, ft::AllocationType::HOST}, {});
    memcpy(input->input_ids->data(), vec.data(), input->input_ids->sizeBytes());

    writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Stream);
    writer->AddHeader("Content-Type", "application/json");
    auto stream = engine->enqueue(input);
    while (!stream->finished()) {
        const auto output_status = stream->nextOutput();
        if (!output_status.ok()) {
            break;
        }
        const GenerateOutputs* responses = &(output_status.value());

        for (size_t i = 0; i < responses->generate_outputs.size(); i++) {
            const auto&          response   = responses->generate_outputs[i];
            const ft::Buffer*    buffer     = response.output_ids.get();
            int32_t*             output_ids = reinterpret_cast<int32_t*>(buffer->data());
            std::vector<int32_t> tokens;
            for (size_t i = 0; i < buffer->size(); i++) {
                tokens.emplace_back(output_ids[i]);
            }

            // TODO
            // auto genenate_texts = Pipeline::decode_tokens(tokens, stop_words_str, finished, token_buffer,
            // incremental, print_stop_words);
            auto generate_texts = pipeline_.decode(tokens);
            auto json_response  = Pipeline::format_response(generate_texts, responses);
            if (input->generate_config->is_streaming) {
                auto sse_response = HttpApiServer::SseResponse(json_response);
                writer->AddHeader("Content-Type", "text/event-stream");
                writer->Write(sse_response);
            } else {
                writer->Write(json_response);
            }
        }
    }
    writer->WriteDone();
    controller_->decrement();
}

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

void HttpApiServer::registerResponses() {
    // TODO: register other routes
    registerRoot();
    registerHealth();
    registerV1Model();
    registerSetLogLevel();
    registerTokenizerEncode();
    registerInference();
    registerInferenceInternal();
    registerWorkerStatus();
}

bool HttpApiServer::registerRoot() {
    auto shared_this = shared_from_this();
    auto callback    = [shared_this](std::unique_ptr<http_server::HttpResponseWriter> writer,
                                  const http_server::HttpRequest&                  request) -> void {
        writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
        writer->AddHeader("Content-Type", "application/json");
        if (shared_this->isStopped()) {
            FT_LOG_WARNING("called root route, but server has been shutdown");
            writer->SetStatus(503, "Service Unavailable");
            writer->Write(R"({"detail":"this server has been shutdown"})");
            return;
        }
        writer->Write(R"({"status":"home"})");
    };
    return http_server_.RegisterRoute("GET", "/", callback);
}

bool HttpApiServer::registerEmbedding() {
    auto shared_this = shared_from_this();
    auto callback    = [shared_this](std::unique_ptr<http_server::HttpResponseWriter> writer,
                                     const http_server::HttpRequest&                  request) -> void {
        if (!shared_this->embedding_endpoint_.has_value()) {
            FT_LOG_WARNING("non-embedding model can't handle embedding request!");
                writer->SetStatus(503, "Service Unavailable");
                writer->Write(R"({"detail":"this server has been shutdown"})");
            return;
        }
        auto& embedding_endpoint = shared_this->embedding_endpoint_.value();

        writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
        writer->AddHeader("Content-Type", "application/json");
        if (shared_this->isStopped()) {
            FT_LOG_WARNING("http api server has been shutdown!");
            writer->SetStatus(503, "Service Unavailable");
            writer->Write(R"({"detail":"this server has been shutdown"})");
            return;
        }

        try {
            auto [response, logable_response] = embedding_endpoint.handle(request.GetBody());
            if (logable_response.has_value()) {
                // TODO: access log response
                FT_LOG_WARNING("TODO: access log embedding model response");
            }
            writer->Write(response);
        } catch (const py::error_already_set& e) {
            FT_LOG_WARNING("embedding endpoint handle request failed, found python exception: %s", e.what());
            writer->SetStatus(503, "Service Unavailable");
            writer->Write(R"({"detail":"embedding endpoint handle request failed"})");
        } catch (const std::exception& e) {
            FT_LOG_WARNING("embedding endpoint handle request failed, found exception: %s", e.what());
            writer->SetStatus(503, "Service Unavailable");
            writer->Write(R"({"detail":"embedding endpoint handle request failed"})");
        }
    };

    return http_server_.RegisterRoute("POST", "/v1/embeddings", callback)
           && http_server_.RegisterRoute("POST", "/v1/embeddings/similarity", callback)
           && http_server_.RegisterRoute("POST", "/v1/classifier", callback)
           && http_server_.RegisterRoute("POST", "/v1/reranker", callback);
}

bool HttpApiServer::registerHealth() {
    auto shared_this = shared_from_this();
    auto callback    = [shared_this](std::unique_ptr<http_server::HttpResponseWriter> writer,
                                  const http_server::HttpRequest&                  request) -> void {
        writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
        writer->AddHeader("Content-Type", "application/json");
        if (shared_this->isStopped()) {
            FT_LOG_WARNING("called health route, but server has been shutdown");
            writer->SetStatus(503, "Service Unavailable");
            writer->Write(R"({"detail":"this server has been shutdown"})");
            return;
        }
        writer->Write("ok");
    };

    return http_server_.RegisterRoute("GET", "/health", callback)
           && http_server_.RegisterRoute("POST", "/health", callback)
           && http_server_.RegisterRoute("GET", "/GraphService/cm2_status", callback)
           && http_server_.RegisterRoute("POST", "/GraphService/cm2_status", callback)
           && http_server_.RegisterRoute("GET", "/SearchService/cm2_status", callback)
           && http_server_.RegisterRoute("POST", "/SearchService/cm2_status", callback)
           && http_server_.RegisterRoute("GET", "/status", callback)
           && http_server_.RegisterRoute("POST", "/status", callback)
           && http_server_.RegisterRoute("POST", "/health_check", callback);
}

bool HttpApiServer::registerV1Model() {
    auto callback = [](std::unique_ptr<http_server::HttpResponseWriter> writer,
                       const http_server::HttpRequest&                  request) -> void {
        std::string model_content = R"del({
    "object": "list",
    "data": [
        {
            "id": "AsyncModel",
            "object": "model",
            "created": 1725874765,
            "owned_by": "owner",
            "root": null,
            "parent": null,
            "permission": null
        }
    ]
})del";
        writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
        writer->AddHeader("Content-Type", "application/json");
        writer->Write(model_content);
    };
    return http_server_.RegisterRoute("GET", "/v1/models", callback);
}

bool HttpApiServer::registerSetLogLevel() {
    auto callback = [](std::unique_ptr<http_server::HttpResponseWriter> writer,
                       const http_server::HttpRequest&                  request) -> void {
        writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
        writer->AddHeader("Content-Type", "application/json");
        const auto body = request.GetBody();
        try {
            auto body_map = AnyCast<JsonMap>(ParseJson(body));
            auto it       = body_map.find("log_level");
            if (it == body_map.end()) {
                FT_LOG_WARNING("set log level failed, request has no log level info, request body: %s", body.c_str());
                writer->Write(R"({"error":"set log level failed, request has no log level info"})");
                return;
            }
            auto value = AnyCast<std::string>(it->second);
            if (torch_ext::setLogLevel(value)) {
                writer->Write(R"({"status":"ok"})");
            } else {
                FT_LOG_WARNING("set log level failed, invalid log level: %s", value);
                writer->Write(R"({"error":"set debug log level failed, invalid log level"})");
            }
            return;
        } catch (const std::exception& e) {
            FT_LOG_WARNING("set debug log level failed, found exception. request body: %s, exception: [%s]",
                           body.c_str(),
                           e.what());
            writer->Write(R"({"error":"set debug log level failed, exception occurred when parse request"})");
            return;
        }
    };
    return http_server_.RegisterRoute("POST", "/set_log_level", callback);
}

bool HttpApiServer::registerTokenizerEncode() {
    auto callback = [pipeline = pipeline_](std::unique_ptr<http_server::HttpResponseWriter> writer,
                                           const http_server::HttpRequest&                  request) mutable -> void {
        writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
        writer->AddHeader("Content-Type", "application/json");
        if (!ParallelInfo::isMaster()) {
            FT_LOG_WARNING("gang worker should not access /tokenizer/encode api directly");
            auto msg =
                CreateErrorResponseJsonString(515, "gang worker should not access /tokenizer/encode api directly");
            writer->Write(msg);
            return;
        }
        const auto body = request.GetBody();
        try {
            auto body_map  = AnyCast<JsonMap>(ParseJson(body));
            auto prompt_it = body_map.find("prompt");
            if (prompt_it == body_map.end()) {
                FT_LOG_WARNING("tokenizer encode failed, request has no prompt, request body: %s", body.c_str());
                writer->SetStatus(500, "Internal Server Error");
                auto msg = CreateErrorResponseJsonString(500, "tokenizer encode failed, request has no prompt");
                writer->Write(msg);
                return;
            }
            auto prompt         = AnyCast<std::string>(prompt_it->second);
            bool offset_mapping = false;
            if (auto offset_mapping_it = body_map.find("return_offsets_mapping"); offset_mapping_it != body_map.end()) {
                offset_mapping = AnyCast<bool>(offset_mapping_it->second);
            }
            std::shared_ptr<TokenizerEncodeResponse> tokenizer_response;
            if (offset_mapping) {
                tokenizer_response = pipeline.tokenizer(prompt);
            } else {
                auto                     token_ids = pipeline.encode(prompt);
                std::vector<std::string> tokens;
                for (auto id : token_ids) {
                    tokens.push_back(pipeline.decode(std::vector<int>{id}));
                }
                tokenizer_response            = std::make_shared<TokenizerEncodeResponse>();
                tokenizer_response->token_ids = token_ids;
                tokenizer_response->tokens    = tokens;
            }
            if (!tokenizer_response) {
                FT_LOG_WARNING("tokenizer encode failed, response is null, request body: %s", body.c_str());
                writer->SetStatus(500, "Internal Server Error");
                auto msg = CreateErrorResponseJsonString(500, "tokenizer encode failed, maybe tokenizer failed");
                writer->Write(msg);
                return;
            }
            auto response_json_str = ToJsonString(*tokenizer_response, /*isCompact=*/true);
            writer->Write(response_json_str);
            return;
        } catch (const std::exception& e) {
            FT_LOG_WARNING(
                "tokenizer encode failed, found exception. request body: %s, exception: [%s]", body.c_str(), e.what());
            writer->SetStatus(500, "Internal Server Error");
            auto msg = CreateErrorResponseJsonString(500, "tokenizer encode failed, exception occurred");
            writer->Write(msg);
            return;
        }
    };
    return http_server_.RegisterRoute("POST", "/tokenizer/encode", callback);
}

bool HttpApiServer::registerInference() {
    auto shared_this = shared_from_this();
    auto callback    = [shared_this](std::unique_ptr<http_server::HttpResponseWriter> writer,
                                  const http_server::HttpRequest&                  request) {
        shared_this->active_request_count_.fetch_add(1);
        try {
            inferResponse(std::move(writer),
                          request,
                          shared_this->engine_,
                          shared_this->params_,
                          shared_this->pipeline_,
                          shared_this->controller_);
        } catch (const std::exception& e) {
            FT_LOG_WARNING("called inference route but found exception: [%s]", e.what());
        }
        shared_this->active_request_count_.fetch_sub(1);
    };
    return http_server_.RegisterRoute("POST", "/inference", callback);
}

bool HttpApiServer::registerInferenceInternal() {
    auto callback = [engine = engine_, params = params_, pipeline = pipeline_, controller = controller_](
                        std::unique_ptr<http_server::HttpResponseWriter> writer,
                        const http_server::HttpRequest&                  request) -> void {
        if (!ParallelInfo::isWorker()) {
            FT_LOG_WARNING("gang master should not access /inference_internal api directly");
            writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
            writer->AddHeader("Content-Type", "application/json");
            auto msg =
                CreateErrorResponseJsonString(515, "gang master should not access /inference_internal api directly");
            writer->Write(msg);
            return;
        }
        inferResponse(std::move(writer), request, engine, params, pipeline, controller);
    };
    auto shared_this      = shared_from_this();
    auto callback_wrapper = [shared_this, callback](std::unique_ptr<http_server::HttpResponseWriter> writer,
                                                    const http_server::HttpRequest&                  request) {
        shared_this->active_request_count_.fetch_add(1);
        try {
            callback(std::move(writer), request);
        } catch (const std::exception& e) {
            FT_LOG_WARNING("called inference internal but found exception: [%s]", e.what());
        }
        shared_this->active_request_count_.fetch_sub(1);
    };
    return http_server_.RegisterRoute("POST", "/inference_internal", callback_wrapper);
}

bool HttpApiServer::registerWorkerStatus() {
    auto shared_this = shared_from_this();
    auto callback    = [shared_this](std::unique_ptr<http_server::HttpResponseWriter> writer,
                                  const http_server::HttpRequest&                  request) -> void {
        writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
        writer->AddHeader("Content-Type", "application/json");
        if (shared_this->isStopped()) {
            FT_LOG_WARNING("called worker status route, but server has been shutdown");
            writer->SetStatus(503, "Service Unavailable");
            writer->Write(R"({"detail":"this server has been shutdown"})");
            return;
        }
        // load balance info
        LoadBalanceInfo load_balance_info;
        if (shared_this->engine_) {
            load_balance_info = shared_this->engine_->getLoadBalanceInfo();
        } else {
            FT_LOG_WARNING("called register worker status route, engine is null");
        }
        // concurrency
        int        available_concurrency = 0;
        const auto load_balance_env      = autil::EnvUtil::getEnv("LOAD_BALANCE", 0);
        if (load_balance_env && load_balance_info.step_per_minute > 0 && load_balance_info.step_latency_us > 0) {
            available_concurrency = load_balance_info.step_per_minute;
        } else {
            if (shared_this->controller_) {  // controller should not be null
                available_concurrency = shared_this->controller_->get_available_concurrency();
            } else {
                FT_LOG_WARNING("called register worker status route, concurrency controller is null");
            }
        }
        WorkerStatusResponse worker_status_response;
        worker_status_response.available_concurrency = available_concurrency;
        worker_status_response.load_balance_info     = load_balance_info;
        worker_status_response.alive                 = true;
        auto response_json_str                       = ToJsonString(worker_status_response, /*isCompact=*/true);
        writer->Write(response_json_str);
        return;
    };
    return http_server_.RegisterRoute("GET", "/worker_status", callback);
}

void HttpApiServer::stop() {
    FT_LOG_WARNING("http api server stopped");
    is_stopped_.store(true);
    while (active_request_count_.load() > 0) {
        FT_LOG_DEBUG("http api server stop called, wait active request processed. active request count: %d",
                     active_request_count_.load());
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    http_server_.Stop();
}

HttpApiServer::~HttpApiServer() {
    stop();
}

// ------------------------------- EmbeddingEndpoint -------------------------------

std::pair<std::string, std::optional<std::string>> EmbeddingEndpoint::handle(const std::string& body) {

    // if isinstance(request, str):
    //     request = json.loads(request)
    // formated_request = await self.custom_model_.renderer.render_request(request)
    // batch_input = self.custom_model_.renderer.create_input(formated_request)
    // batch_output = await self.decoder_engine_.decode(batch_input)
    // response = await self.custom_model_.renderer.render_response(formated_request, batch_input, batch_output)
    // logable_response = await self.custom_model_.renderer.render_log_response(response)
    // return response, logable_response

    py::gil_scoped_acquire gil;
    py::module json = py::module::import("json");
    py::object request = json.attr("loads")(body);

    auto formated_request = py_render_.attr("render_request")(request);
    auto batch_input = py_render_.attr("create_input")(formated_request);

    std::vector<MultimodalInput> mm_inputs;
    auto py_mm_inputs = batch_input.attr("multimodal_inputs");
    if (!py::isinstance<py::list>(py_mm_inputs)) {
        throw std::runtime_error("Expected a list, but get " + py::cast<std::string>(py::str(py_mm_inputs)));
    }
    py::list py_list = py::reinterpret_borrow<py::list>(py_mm_inputs);
    for (const auto& item : py_list) {
        mm_inputs.emplace_back(py::cast<std::string>(item.attr("url")),
                               py::cast<int>(item.attr("mm_type")));
    }
    std::optional<MultimodalFeature> multimodal_features = std::nullopt;
    auto token_ids = py::cast<th::Tensor>(batch_input.attr("token_ids"));
    if (mm_processor_ != nullptr && !mm_inputs.empty()) {
        auto mm_res = mm_processor_->get_mm_features(ft::torchTensor2Buffer(token_ids), mm_inputs);
        if (!mm_res.ok()) {
            throw std::runtime_error(mm_res.status().ToString());
        }
        token_ids = ft::Buffer2torchTensor(mm_res.value().expanded_ids, true);
        multimodal_features.emplace(mm_res.value());
    }
    auto results = embedding_engine_->decode(token_ids,
                                             py::cast<th::Tensor>(batch_input.attr("token_type_ids")),
                                             py::cast<th::Tensor>(batch_input.attr("input_lengths")),
                                             /*request_id=*/0,
                                             multimodal_features);

    py::module embedding_interface = py::module::import("maga_transformer.async_decoder_engine.embedding.interface");
    py::object batch_output = embedding_interface.attr("EngineOutputs")();
    batch_output.attr("outputs") = py::cast(results);
    batch_output.attr("input_length") = batch_input.attr("input_length");

    auto response = py_render_.attr("render_response")(formated_request, batch_input, batch_output);
    auto logable_response = py_render_.attr("render_log_response")(response);
    if (logable_response.is_none()) {
        auto json_response = json.attr("dumps")(response);
        return std::make_pair(py::cast<std::string>(json_response), std::nullopt);
    } else {
        auto json_response = json.attr("dumps")(response);
        auto json_logable_response = json.attr("dumps")(logable_response);
        return std::make_pair(py::cast<std::string>(json_response), py::cast<std::string>(logable_response));
    }
}

// ------------------------------- Pipeline -------------------------------

std::string Pipeline::decode(std::vector<int> token_ids) {
    py::gil_scoped_acquire acquire;
    std::string            res = py::cast<std::string>(token_processor_.attr("decode")(token_ids));
    return res;
}

std::vector<int> Pipeline::encode(std::string prompt) {
    py::gil_scoped_acquire acquire;
    auto                   res = token_processor_.attr("encode")(prompt);
    std::vector<int>       vecInt;
    if (!py::isinstance<py::list>(res)) {
        throw std::runtime_error("Expected a list, but get " + py::cast<std::string>(py::str(res)));
    }
    py::list py_list = py::reinterpret_borrow<py::list>(res);
    for (auto item : py_list) {
        vecInt.push_back(py::cast<int>(item));
    }
    return vecInt;
}

std::string Pipeline::format_response(std::string generate_texts, const GenerateOutputs* generate_outputs) {
    PipelineResponse res;
    res.response = generate_texts;
    res.finished = generate_outputs->generate_outputs[0].finished;
    res.aux_info = AuxInfoAdapter(generate_outputs->generate_outputs[0].aux_info);
    return ToJsonString(res, /*isCompact=*/true);
}

std::shared_ptr<TokenizerEncodeResponse> Pipeline::tokenizer(const std::string& prompt) {
    auto                   response = std::make_shared<TokenizerEncodeResponse>();
    py::gil_scoped_acquire acquire;
    auto                   res    = token_processor_(prompt);
    auto                   py_res = py::cast<py::dict>(res);

    // offset_mapping
    if (py_res.contains("offset_mapping")) {
        auto py_offset_mapping = py_res["offset_mapping"];
        if (!py::isinstance<py::list>(py_offset_mapping)) {
            FT_LOG_WARNING("tokenizer failed, offset mapping expected list but type is %s, offset mapping: %s",
                           py::cast<std::string>(py::str(py::type::of(py_offset_mapping))).c_str(),
                           py::cast<std::string>(py::str(py_offset_mapping)).c_str());
            return nullptr;
        }
        std::vector<std::vector<int>> offset_mapping;
        auto                          py_offset_mapping_list = py::cast<py::list>(py_offset_mapping);
        for (auto& py_offset : py_offset_mapping_list) {
            offset_mapping.push_back({});
            auto py_offset_list = py::cast<py::list>(py_offset);
            for (auto py_num : py_offset_list) {
                offset_mapping.back().push_back(py::cast<int>(py_num));
            }
        }
        response->offset_mapping = offset_mapping;
    } else {
        FT_LOG_WARNING("tokenizer result has no offset_mapping");
    }

    // input_ids
    if (py_res.contains("input_ids")) {
        auto py_input_ids = py_res["input_ids"];
        if (!py::isinstance<py::list>(py_input_ids)) {
            FT_LOG_WARNING("tokenizer failed, input ids expected list but type is: %s, input ids: %s",
                           py::cast<std::string>(py::str(py::type::of(py_input_ids))).c_str(),
                           py::cast<std::string>(py::str(py_input_ids)).c_str());
            return nullptr;
        }
        std::vector<int> input_ids;
        auto             py_input_ids_list = py::cast<py::list>(py_input_ids);
        for (auto& py_id : py_input_ids_list) {
            input_ids.push_back(py::cast<int>(py_id));
        }
        response->token_ids = input_ids;
    } else {
        FT_LOG_WARNING("tokenizer result has no input_ids");
    }

    return response;
}

}  // namespace rtp_llm

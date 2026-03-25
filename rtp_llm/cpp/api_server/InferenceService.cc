#include "rtp_llm/cpp/api_server/InferenceService.h"

#include <algorithm>
#include <iterator>
#include <torch/python.h>
#include <functional>
#include <fstream>
#include <openssl/sha.h>

#include "rtp_llm/cpp/pybind/PyUtils.h"

#include "rtp_llm/cpp/api_server/Exception.h"
#include "rtp_llm/cpp/api_server/ErrorResponse.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/api_server/AccessLogWrapper.h"

using namespace autil::legacy;
using namespace autil::legacy::json;

namespace rtp_llm {

void InferenceParsedRequest::extractRequestTexts(const RawRequest& req, InferenceParsedRequest& pr) {
    if (req.prompt_batch.has_value()) {
        pr.input_texts = req.prompt_batch.value();
    } else {
        if (req.prompt.has_value()) {
            pr.input_texts.push_back(req.prompt.value());
        } else {
            throw HttpApiServerException(HttpApiServerException::NO_PROMPT_ERROR, "no prompt in request!");
        }
    }
}

void InferenceParsedRequest::extractRequestUrls(const RawRequest& req, InferenceParsedRequest& pr) {
    if (req.images.has_value()) {
        pr.input_urls.push_back(req.images.value());
        if (pr.input_urls.size() != pr.input_texts.size()) {
            throw HttpApiServerException(HttpApiServerException::ERROR_INPUT_FORMAT_ERROR,
                                         "urls and texts are not matched!");
        }
    } else if (req.images_batch.has_value()) {
        pr.input_urls = req.images_batch.value();
        if (pr.input_urls.size() != pr.input_texts.size()) {
            throw HttpApiServerException(HttpApiServerException::ERROR_INPUT_FORMAT_ERROR,
                                         "urls and texts are not matched!");
        }
    } else {
        for (int i = 0; i < pr.input_texts.size(); i++) {
            pr.input_urls.push_back(std::vector<std::string>());
        }
    }
}

void InferenceParsedRequest::extractRequestGenerateConfigs(RawRequest&                            req,
                                                           InferenceParsedRequest&                pr,
                                                           const ModelConfig&                     model_config,
                                                           const std::shared_ptr<TokenProcessor>& token_processor) {
    if (req.generate_config.has_value()) {
        auto& config = req.generate_config.value();
        if (req.yield_generator == false && config.return_incremental == true) {
            throw HttpApiServerException(HttpApiServerException::ERROR_INPUT_FORMAT_ERROR,
                                         "request is non_stream but use incremental decoder");
        }
        config.addSpecialTokens(model_config.special_tokens);
        if (config.sp_advice_prompt.empty() == false) {
            config.sp_advice_prompt_token_ids = token_processor->encode(config.sp_advice_prompt);
        }
    }
    // TODO: adapter_name
    for (int i = 0; i < pr.input_texts.size(); i++) {
        if (req.generate_config.has_value()) {
            const auto& config = req.generate_config.value();
            if (config.adapter_name.size() > 0 && pr.input_texts.size() != 1) {
                throw HttpApiServerException(HttpApiServerException::ERROR_INPUT_FORMAT_ERROR,
                                             "adapter_name is not alignment");
            }
            if (config.adapter_names.size() > 0 && pr.input_texts.size() != config.adapter_names.size()) {
                throw HttpApiServerException(HttpApiServerException::ERROR_INPUT_FORMAT_ERROR,
                                             "adapter_name is not alignment");
            }
            // copy constructor
            pr.generate_configs.push_back(std::make_shared<GenerateConfig>(req.generate_config.value()));
            if (config.adapter_names.size() > 0) {
                pr.generate_configs[i]->adapter_name = config.adapter_names[i];
            }
        } else {
            pr.generate_configs.push_back(std::make_shared<GenerateConfig>());
        }
    }
}

InferenceParsedRequest InferenceParsedRequest::extractRequest(const std::string&                     body,
                                                              const ModelConfig&                     model_config,
                                                              const std::shared_ptr<TokenProcessor>& token_processor) {
    RawRequest req;
    FromJsonString(req, body);

    InferenceParsedRequest pr;
    pr.private_request = req.private_request;
    pr.source          = req.source;
    pr.batch_infer     = req.prompt_batch.has_value();
    pr.is_streaming    = req.generate_config.value_or(GenerateConfig()).is_streaming;

    InferenceParsedRequest::extractRequestTexts(req, pr);
    InferenceParsedRequest::extractRequestUrls(req, pr);
    InferenceParsedRequest::extractRequestGenerateConfigs(req, pr, model_config, token_processor);

    return pr;
}

InferenceService::InferenceService(const std::shared_ptr<EngineBase>&              engine,
                                   const std::shared_ptr<MultimodalProcessor>&     mm_processor,
                                   const std::shared_ptr<autil::AtomicCounter>&    request_counter,
                                   const std::shared_ptr<TokenProcessor>&          token_processor,
                                   const std::shared_ptr<ConcurrencyController>&   controller,
                                   const ModelConfig&                              model_config,
                                   const std::shared_ptr<ApiServerMetricReporter>& metric_reporter,
                                   py::object                                      py_model):
    engine_(engine),
    mm_processor_(mm_processor),
    token_processor_(token_processor),
    request_counter_(request_counter),
    controller_(controller),
    model_config_(model_config),
    metric_reporter_(metric_reporter),
    py_model_(py_model) {
    // Load extra input config once at construction time (no per-request overhead)
    if (!py_model_.is_none()) {
        loadExtraInputConfig();
    }
}

void checkMasterWorker(bool isInternal) {
    if (isInternal) {
        if (!ParallelInfo::globalParallelInfo().isWorker()) {
            RTP_LLM_LOG_WARNING("gang master should not access /inference_internal api directly");
            throw HttpApiServerException(HttpApiServerException::UNSUPPORTED_OPERATION,
                                         "gang master should not access /inference_internal api directly");
        }
    } else {
        if (!ParallelInfo::globalParallelInfo().isMaster()) {
            RTP_LLM_LOG_WARNING("gang worker should not access /inference api directly");
            throw HttpApiServerException(HttpApiServerException::UNSUPPORTED_OPERATION,
                                         "gang worker should not access /inference api directly");
        }
    }
}

void InferenceService::inference(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                                 const http_server::HttpRequest&                         request,
                                 bool                                                    isInternal) {
    int64_t request_id = -1;
    if (request_counter_ == nullptr) {
        RTP_LLM_LOG_WARNING("inference failed, request_counter is null");
        return;
    }
    request_id = request_counter_->incAndReturn();
    try {
        checkMasterWorker(isInternal);
        inferResponse(request_id, writer, request);
    } catch (const std::exception& e) {
        HttpApiServerException::handleException(e, request_id, metric_reporter_, request, writer);
    }
}

void InferenceService::inferResponse(int64_t                                                 request_id,
                                     const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                                     const http_server::HttpRequest&                         request) {
    if (!controller_) {
        throw HttpApiServerException(HttpApiServerException::UNKNOWN_ERROR,
                                     "infer response failed, concurrency controller is null");
    }
    autil::StageTime iterate_stage_timer;
    auto             start_time_ms = autil::TimeUtility::currentTimeInMilliSeconds();
    const auto       body          = request.GetBody();
    auto             req           = InferenceParsedRequest::extractRequest(body, model_config_, token_processor_);
    if (metric_reporter_) {
        metric_reporter_->reportQpsMetric(req.source);
    }
    AccessLogWrapper::logQueryAccess(body, request_id, req.private_request);

    ConcurrencyControllerGuard controller_guard(controller_);
    if (controller_guard.isPassed() == false) {
        if (metric_reporter_) {
            metric_reporter_->reportConflictQpsMetric();
        }
        throw HttpApiServerException(HttpApiServerException::CONCURRENCY_LIMIT_ERROR, "Too Many Requests");
    }
    if (writer->isConnected() == false) {
        throw HttpApiServerException(HttpApiServerException::CANCELLED_ERROR, "client disconnects");
    }
    auto inputs      = batchFillGenerateInputs(request_id, req.input_texts, req.input_urls, req.generate_configs);
    auto ori_streams = engine_->batchEnqueue(inputs);
    std::vector<std::shared_ptr<GenerateStreamWrapper>> streams;
    streams.reserve(ori_streams.size());
    for (size_t idx = 0; idx < ori_streams.size(); ++idx) {
        auto stream_wrapper = std::make_shared<GenerateStreamWrapper>(metric_reporter_, token_processor_);
        stream_wrapper->init(ori_streams[idx], engine_);
        streams.push_back(stream_wrapper);
    }
    auto [iterate_count, complete_response] = iterateStreams(streams, writer, req, iterate_stage_timer);

    iterate_stage_timer.end_stage();
    writer->WriteDone();
    AccessLogWrapper::logSuccessAccess(body, request_id, complete_response, req.private_request);
    if (metric_reporter_) {
        metric_reporter_->reportSuccessQpsMetric(req.source);
        metric_reporter_->reportResponseIterateCountMetric(iterate_count);
        metric_reporter_->reportResponseLatencyMs(autil::TimeUtility::currentTimeInMilliSeconds() - start_time_ms);
    }
}

std::shared_ptr<GenerateInput>
InferenceService::fillGenerateInput(int64_t                                request_id,
                                    const std::string&                     text,
                                    const std::vector<std::string>&        urls,
                                    const std::shared_ptr<GenerateConfig>& generate_config) {

    std::shared_ptr<GenerateInput> input = std::make_shared<GenerateInput>();
    input->request_id                    = request_id;
    input->begin_time_us                 = autil::TimeUtility::currentTimeInMicroSeconds();
    input->generate_config               = generate_config;

    if (urls.size() > 0) {
        std::vector<MultimodalInput> mm_inputs;
        for (auto url : urls) {
            mm_inputs.emplace_back(url);
        }
        input->multimodal_inputs = std::move(mm_inputs);
    }
    if (mm_processor_ != nullptr && input->multimodal_inputs) {
        auto mm_res = mm_processor_->updateMultimodalFeatures(input);
        if (!mm_res.ok()) {
            throw HttpApiServerException(HttpApiServerException::MULTIMODAL_ERROR,
                                         "mm_processor updateMultimodalFeatures failed: " + mm_res.ToString());
        }
    }
    input->lora_id = engine_->getLoraManager()->getLoraId(input->generate_config->adapter_name);
    if (metric_reporter_) {
        metric_reporter_->reportFTInputTokenLengthMetric(input->generate_config->select_tokens_id.size());
        metric_reporter_->reportFTNumBeansMetric(input->generate_config->maxNumBeams());
    }
    autil::ScopedTime2 timer;
    auto               vec = token_processor_->encode(text);

    // Process extra input if model supports it
    auto result = processExtraInput(vec);
    vec         = result.processed_input_ids;

    // Store extra_input_ids and location if present
    if (result.extra_input_ids.has_value()) {
        auto device = rtp_llm::DeviceFactory::getDefaultDevice();
        // extra_input_ids buffer is stored as optional<BufferPtr> in GenerateInput
        input->extra_input_ids = device->allocateBuffer({rtp_llm::DataType::TYPE_INT32,
                                                         {static_cast<size_t>(result.extra_input_ids->size())},
                                                         rtp_llm::AllocationType::HOST},
                                                        {});
        // optional<BufferPtr> → BufferPtr → Buffer
        std::memcpy(
            (*input->extra_input_ids)->data(), result.extra_input_ids->data(), (*input->extra_input_ids)->sizeBytes());
        // Store location information (relative to single sequence; later converted to global index)
        input->extra_input_ids_loc = result.extra_input_ids_loc;
    }

    if (metric_reporter_) {
        metric_reporter_->reportFTPreTokenProcessorRtMetric(timer.done_ms());
    }
    auto device = rtp_llm::DeviceFactory::getDefaultDevice();
    input->input_ids =
        device->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {vec.size()}, rtp_llm::AllocationType::HOST}, {});
    memcpy(input->input_ids->data(), vec.data(), input->input_ids->sizeBytes());

    return input;
}

InferenceService::ProcessExtraInputResult InferenceService::processExtraInput(const std::vector<int>& input_ids) {
    if (py_model_.is_none()) {
        return {input_ids, std::nullopt, -1};
    }

    // Try C++ implementation first (no GIL needed)
    return processExtraInputCpp(input_ids);
}

void InferenceService::loadExtraInputConfig() {
    if (extra_input_config_loaded_)
        return;
    extra_input_config_loaded_ = true;

    // Read config from Python model's tse_config (single GIL acquisition, only once)
    try {
        py::gil_scoped_acquire acquire;
        if (py::hasattr(py_model_, "tse_config")) {
            auto tse_config = py_model_.attr("tse_config");
            if (py::hasattr(tse_config, "mask_token_id")) {
                mask_token_id_ = py::cast<int>(tse_config.attr("mask_token_id"));
            }
            if (py::hasattr(tse_config, "num_last_tokens")) {
                num_last_tokens_ = py::cast<int>(tse_config.attr("num_last_tokens"));
            }
        }
        RTP_LLM_LOG_INFO(
            "loadExtraInputConfig: mask_token_id=%d, num_last_tokens=%d", mask_token_id_, num_last_tokens_);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING(
            "Failed to load extra input config from Python model: %s, using defaults (mask_token_id=%d, num_last_tokens=%d)",
            e.what(),
            mask_token_id_,
            num_last_tokens_);
    }
}

InferenceService::ProcessExtraInputResult InferenceService::processExtraInputCpp(const std::vector<int>& input_ids) {
    // Config already loaded in constructor

    // Find mask token positions
    std::vector<int> mask_indices;
    for (int i = 0; i < (int)input_ids.size(); i++) {
        if (input_ids[i] == mask_token_id_) {
            mask_indices.push_back(i);
        }
    }

    if (mask_indices.size() < 2) {
        return {input_ids, std::nullopt, -1};
    }

    int first_mask_idx  = mask_indices[0];
    int second_mask_idx = mask_indices[1];

    // Extract item_input (between two masks)
    std::vector<int> item_input(input_ids.begin() + first_mask_idx + 1, input_ids.begin() + second_mask_idx);

    // Build decoder_input_ids: before_item + [mask_token_id] * num_last_tokens + after_item
    std::vector<int> decoder_input_ids;
    decoder_input_ids.reserve(first_mask_idx + num_last_tokens_ + (int)input_ids.size() - second_mask_idx - 1);
    decoder_input_ids.insert(decoder_input_ids.end(), input_ids.begin(), input_ids.begin() + first_mask_idx);
    int placeholder_start = (int)decoder_input_ids.size();

    // Compute item_hash_token_id: SHA256 hash matching Python's hashlib.sha256(json.dumps(item_input, sort_keys=True))
    std::string item_json = "[";
    for (size_t i = 0; i < item_input.size(); i++) {
        if (i > 0)
            item_json += ", ";
        item_json += std::to_string(item_input[i]);
    }
    item_json += "]";

    unsigned char sha256_hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(item_json.data()), item_json.size(), sha256_hash);
    // Take first 8 hex chars (4 bytes) matching Python's int(hexdigest[:8], 16)
    char hex_buf[9];
    snprintf(
        hex_buf, sizeof(hex_buf), "%02x%02x%02x%02x", sha256_hash[0], sha256_hash[1], sha256_hash[2], sha256_hash[3]);
    unsigned long item_hash          = strtoul(hex_buf, nullptr, 16);
    int           item_hash_token_id = 100000 + (int)(item_hash % 900000);

    for (int i = 0; i < num_last_tokens_; i++) {
        decoder_input_ids.push_back(item_hash_token_id);
    }
    decoder_input_ids.insert(decoder_input_ids.end(), input_ids.begin() + second_mask_idx + 1, input_ids.end());

    return {decoder_input_ids, item_input, placeholder_start};
}

std::vector<std::shared_ptr<GenerateInput>>
InferenceService::batchFillGenerateInputs(int64_t                                             request_id,
                                          const std::vector<std::string>&                     texts,
                                          const std::vector<std::vector<std::string>>&        urls,
                                          const std::vector<std::shared_ptr<GenerateConfig>>& generate_configs) {
    // Step 1: Batch tokenize all texts (single GIL acquisition)
    auto all_token_ids = token_processor_->batchEncode(texts);

    // Step 2: Process each input (no GIL needed - C++ processExtraInput)
    auto                                        device = rtp_llm::DeviceFactory::getDefaultDevice();
    std::vector<std::shared_ptr<GenerateInput>> inputs;
    inputs.reserve(texts.size());

    for (int i = 0; i < (int)texts.size(); i++) {
        auto input             = std::make_shared<GenerateInput>();
        input->request_id      = request_id;
        input->begin_time_us   = autil::TimeUtility::currentTimeInMicroSeconds();
        input->generate_config = generate_configs[i];

        if (urls[i].size() > 0) {
            std::vector<MultimodalInput> mm_inputs;
            for (auto url : urls[i]) {
                mm_inputs.emplace_back(url);
            }
            input->multimodal_inputs = std::move(mm_inputs);
        }
        if (mm_processor_ != nullptr && input->multimodal_inputs) {
            auto mm_res = mm_processor_->updateMultimodalFeatures(input);
            if (!mm_res.ok()) {
                throw HttpApiServerException(HttpApiServerException::MULTIMODAL_ERROR,
                                             "mm_processor updateMultimodalFeatures failed: " + mm_res.ToString());
            }
        }
        input->lora_id = engine_->getLoraManager()->getLoraId(input->generate_config->adapter_name);

        // Process extra input (pure C++, no GIL)
        auto& vec    = all_token_ids[i];
        auto  result = processExtraInput(vec);
        vec          = result.processed_input_ids;

        if (result.extra_input_ids.has_value()) {
            input->extra_input_ids = device->allocateBuffer({rtp_llm::DataType::TYPE_INT32,
                                                             {static_cast<size_t>(result.extra_input_ids->size())},
                                                             rtp_llm::AllocationType::HOST},
                                                            {});
            std::memcpy((*input->extra_input_ids)->data(),
                        result.extra_input_ids->data(),
                        (*input->extra_input_ids)->sizeBytes());
            input->extra_input_ids_loc = result.extra_input_ids_loc;
        }

        input->input_ids =
            device->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {vec.size()}, rtp_llm::AllocationType::HOST}, {});
        memcpy(input->input_ids->data(), vec.data(), input->input_ids->sizeBytes());

        inputs.push_back(input);
    }

    return inputs;
}

std::pair<int, std::vector<std::string>>
InferenceService::iterateStreams(std::vector<std::shared_ptr<GenerateStreamWrapper>>&    streams,
                                 const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                                 const InferenceParsedRequest&                           req,
                                 autil::StageTime&                                       iterate_stage_timer) {

    std::vector<MultiSeqsResponse> batch_state(streams.size());
    std::vector<std::string>       complete_response;
    std::set<int>                  done_idxs;
    int                            iterate_counter = 0;

    while (true) {
        for (int i = 0; i < streams.size(); i++) {
            auto [response, finished] = streams[i]->generateResponse();
            batch_state[i]            = response;
            if (finished) {
                done_idxs.insert(i);
            }
        }
        if (done_idxs.size() == streams.size()) {
            writeDoneResponse(writer, req);
            break;
        }
        auto res             = formatResponse(batch_state, req.batch_infer);
        auto [err, response] = writeResponse(writer, req, res);
        if (err) {
            throw HttpApiServerException(HttpApiServerException::CANCELLED_ERROR, "client disconnects");
        }
        complete_response.push_back(response);

        if (metric_reporter_) {
            if (iterate_counter == 0) {
                metric_reporter_->reportResponseFirstTokenLatencyMs(iterate_stage_timer.last_ms());
            } else {
                metric_reporter_->reportResponseIterateLatencyMs(iterate_stage_timer.last_ms());
            }
            metric_reporter_->reportResponseIterateQpsMetric();
        }
        iterate_counter++;
    }
    return std::make_pair(iterate_counter, complete_response);
}

std::pair<bool, std::string>
InferenceService::writeResponse(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                                const InferenceParsedRequest&                           req,
                                const std::any&                                         res) {
    bool        connection_broken = false;
    std::string response_str;
    if (req.is_streaming) {
        response_str = streamingResponse(res);
        writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Stream);
        writer->AddHeader("Content-Type", "text/event-stream");
    } else {
        response_str = completeResponse(res);
        writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
        writer->AddHeader("Content-Type", "application/json");
    }
    if (writer->Write(response_str) == false) {
        if (metric_reporter_) {
            metric_reporter_->reportCancelQpsMetric(req.source);
        }
        connection_broken = true;
    }
    return std::make_pair(connection_broken, response_str);
}

void InferenceService::writeDoneResponse(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                                         const InferenceParsedRequest&                           req) {
    if (req.is_streaming) {
        std::string response_str = doneResponse();
        writer->AddHeader("Content-Type", "text/event-stream");
        writer->Write(response_str);
    }
}

std::string InferenceService::doneResponse() {
    return "data:[done]\n\n";
}

std::string InferenceService::completeResponse(const std::any& response) {
    if (response.type() == typeid(MultiSeqsResponse)) {
        auto res = std::any_cast<MultiSeqsResponse>(response);
        return ToJsonString(res, /*isCompact=*/true);
    } else if (response.type() == typeid(BatchResponse)) {
        return ToJsonString(std::any_cast<BatchResponse>(response), /*isCompact=*/true);
    } else {
        RTP_LLM_LOG_WARNING("unknown complete response type: %s", response.type().name());
        return "";
    }
}

std::string InferenceService::streamingResponse(const std::any& response) {
    if (response.type() == typeid(MultiSeqsResponse)) {
        auto res = std::any_cast<MultiSeqsResponse>(response);
        return "data:" + ToJsonString(res, /*isCompact=*/true) + "\n\n";
    } else if (response.type() == typeid(BatchResponse)) {
        return "data:" + ToJsonString(std::any_cast<BatchResponse>(response), /*isCompact=*/true) + "\n\n";
    } else {
        RTP_LLM_LOG_WARNING("unknown streaming response type: %s", response.type().name());
        return "data:\n\n";
    }
}

std::any InferenceService::formatResponse(std::vector<MultiSeqsResponse> batch_state, bool batch_infer) {
    if (batch_infer) {
        return BatchResponse(batch_state);
    } else {
        return batch_state[0];
    }
}

}  // namespace rtp_llm

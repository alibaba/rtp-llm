#include "rtp_llm/cpp/api_server/ChatService.h"

#include "rtp_llm/cpp/api_server/http_server/http_server/HttpResponseWriter.h"
#include "rtp_llm/cpp/api_server/http_server/http_server/HttpRequest.h"
#include "rtp_llm/cpp/api_server/AccessLogWrapper.h"
#include "rtp_llm/cpp/api_server/Exception.h"

using namespace autil::legacy;
using namespace autil::legacy::json;

namespace rtp_llm {

// ChatService::ChatService() {}

std::shared_ptr<GenerateInput> ChatService::fillGenerateInput(int64_t                      request_id,
                                                              const ChatCompletionRequest& chat_request,
                                                              const RenderedInputs&        rendered_input) {
    std::shared_ptr<GenerateInput> input = std::make_shared<GenerateInput>();
    input->request_id                    = request_id;
    input->begin_time_us                 = autil::TimeUtility::currentTimeInMicroSeconds();
    input->generate_config               = openai_endpoint_->extract_generation_config(chat_request);
    metric_reporter_->reportFTInputTokenLengthMetric(input->generate_config->select_tokens_id.size());
    metric_reporter_->reportFTNumBeansMetric(input->generate_config->maxNumBeams());

    const auto& vec    = rendered_input.input_ids;
    auto        device = rtp_llm::DeviceFactory::getDefaultDevice();
    input->input_ids =
        device->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {vec.size()}, rtp_llm::AllocationType::HOST}, {});
    memcpy(input->input_ids->data(), vec.data(), input->input_ids->sizeBytes());

    input->multimodal_inputs = std::move(rendered_input.multimodal_inputs);
    if (mm_processor_ != nullptr && input->multimodal_inputs) {
        auto mm_res = mm_processor_->updateMultimodalFeatures(input);
        if (!mm_res.ok()) {
            throw HttpApiServerException(HttpApiServerException::MULTIMODAL_ERROR,
                                         "mm_processor updateMultimodalFeatures failed: " + mm_res.ToString());
        }
    }

    return input;
}

void ChatService::generateResponse(const std::shared_ptr<GenerateConfig>&                  config,
                                   const GenerateStreamPtr&                                stream,
                                   const RenderedInputs&                                   rendered_input,
                                   autil::StageTime&                                       iterate_stage_timer,
                                   const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                                   const ChatCompletionRequest&                            chat_request,
                                   const std::string&                                      body,
                                   int64_t                                                 request_id,
                                   int64_t                                                 start_time_us) {
    int                      index           = 0;
    int                      iterate_counter = 0;
    int                      token_counter   = 0;
    std::vector<std::string> complete_response;
    int                      num_return_sequences = config->num_return_sequences;

    auto                           chat_render = openai_endpoint_->getChatRender();
    std::shared_ptr<RenderContext> ctx         = chat_render->getRenderContext();
    ctx->init(num_return_sequences, body, chat_render);

    GenerateOutputs outputs;
    while (!stream->finished() || stream->hasOutput()) {
        const auto result = stream->nextOutput();
        if (!result.ok()) {
            RTP_LLM_LOG_INFO("stream nextOutput failed");
            break;
        }
        outputs = result.value();
        RTP_LLM_CHECK_WITH_INFO(outputs.generate_outputs.size() == num_return_sequences,
                                "generate_outputs.size() != num_return_sequences");

        iterate_stage_timer.end_stage();
        if (iterate_counter++ == 0) {
            metric_reporter_->reportResponseFirstTokenLatencyMs(iterate_stage_timer.last_ms());
        }
        metric_reporter_->reportResponseIterateLatencyMs(iterate_stage_timer.last_ms());

        if (index == 0) {
            ctx->render_stream_response_first_blocking(num_return_sequences);
            index += 1;
        }
        ctx->render_stream_response_blocking(outputs, config, chat_request.stream.value_or(false));
        index += 1;
    }
    if (index != 0) {
        ctx->render_stream_response_flush_blocking(outputs, config, chat_request.stream.value_or(false));
        index += 1;
        ctx->render_stream_response_final_blocking(outputs);
        index += 1;
    }
    auto json_response = ctx->collect_complete_response();

    writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
    writer->AddHeader("Content-Type", "application/json");
    writer->Write(json_response);

    metric_reporter_->reportSuccessQpsMetric(chat_request.source.value_or("unknown"));
    metric_reporter_->reportResponseIterateCountMetric(iterate_counter);
    metric_reporter_->reportFTIterateCountMetric(iterate_counter);
    metric_reporter_->reportFTOutputTokenLengthMetric(token_counter);
    metric_reporter_->reportResponseLatencyMs(autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us);

    AccessLogWrapper::logSuccessAccess(body, request_id, complete_response, chat_request.private_request);
}

void ChatService::generateStreamingResponse(const std::shared_ptr<GenerateConfig>&                  config,
                                            const GenerateStreamPtr&                                stream,
                                            const RenderedInputs&                                   rendered_input,
                                            autil::StageTime&                                       iterate_stage_timer,
                                            const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                                            const ChatCompletionRequest&                            chat_request,
                                            const std::string&                                      body,
                                            int64_t                                                 request_id,
                                            int64_t                                                 start_time_us) {
    int                      index           = 0;
    int                      iterate_counter = 0;
    int                      token_counter   = 0;
    std::vector<std::string> complete_response;
    int                      num_return_sequences = config->num_return_sequences;

    auto                           chat_render = openai_endpoint_->getChatRender();
    std::shared_ptr<RenderContext> ctx         = chat_render->getRenderContext();
    ctx->init(num_return_sequences, body, chat_render);

    auto write_sse_response = [&](std::string json_response) {
        std::string sse_response = sseResponse(json_response);
        writer->AddHeader("Content-Type", "text/event-stream");
        writer->Write(sse_response);
        index += 1;
    };

    writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Stream);
    GenerateOutputs outputs;
    while (!stream->finished()) {
        const auto output_status = stream->nextOutput();
        if (!output_status.ok()) {
            RTP_LLM_LOG_INFO("stream nextOutput failed");
            break;
        }
        outputs = output_status.value();
        RTP_LLM_CHECK_WITH_INFO(outputs.generate_outputs.size() == num_return_sequences,
                                "generate_outputs.size() != num_return_sequences");

        iterate_stage_timer.end_stage();
        if (iterate_counter++ == 0) {
            metric_reporter_->reportResponseFirstTokenLatencyMs(iterate_stage_timer.last_ms());
        }
        metric_reporter_->reportResponseIterateLatencyMs(iterate_stage_timer.last_ms());

        if (index == 0) {
            std::string debug_info    = openai_endpoint_->getDebugInfo(chat_request, rendered_input);
            std::string json_response = ctx->render_stream_response_first(num_return_sequences, debug_info);
            write_sse_response(json_response);
        }
        std::string json_response =
            ctx->render_stream_response(output_status.value(), config, chat_request.stream.value_or(false));
        write_sse_response(json_response);
    }
    if (index != 0) {
        std::string json_response =
            ctx->render_stream_response_flush(outputs, config, chat_request.stream.value_or(false));
        write_sse_response(json_response);
        json_response = ctx->render_stream_response_final(outputs);
        write_sse_response(json_response);
    }
    writer->WriteDone();
    metric_reporter_->reportSuccessQpsMetric(chat_request.source.value_or("unknown"));
    metric_reporter_->reportResponseIterateCountMetric(iterate_counter);
    metric_reporter_->reportFTIterateCountMetric(iterate_counter);
    metric_reporter_->reportFTOutputTokenLengthMetric(token_counter);
    metric_reporter_->reportResponseLatencyMs(autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us);
    AccessLogWrapper::logSuccessAccess(body, request_id, complete_response, chat_request.private_request);
}

void ChatService::chatCompletions(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                                  const http_server::HttpRequest&                         request,
                                  int64_t                                                 request_id) {
    auto             start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    autil::StageTime iterate_stage_timer;

    const auto            body = request.GetBody();
    ChatCompletionRequest chat_request;
    FromJsonString(chat_request, body);

    AccessLogWrapper::logQueryAccess(body, request_id, chat_request.private_request);

    auto       chat_render    = openai_endpoint_->getChatRender();
    const auto rendered_input = chat_render->render_chat_request(body);

    auto input  = fillGenerateInput(request_id, chat_request, rendered_input);
    auto stream = engine_->enqueue(input);

    if (chat_request.stream.value_or(false) == false) {
        generateResponse(input->generate_config,
                         stream,
                         rendered_input,
                         iterate_stage_timer,
                         writer,
                         chat_request,
                         body,
                         request_id,
                         start_time_us);
    } else {
        generateStreamingResponse(input->generate_config,
                                  stream,
                                  rendered_input,
                                  iterate_stage_timer,
                                  writer,
                                  chat_request,
                                  body,
                                  request_id,
                                  start_time_us);
    }
}

void ChatService::chatRender(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                             const http_server::HttpRequest&                         request) {
    const auto            body = request.GetBody();
    ChatCompletionRequest chat_request;
    FromJsonString(chat_request, body);

    auto        chat_render    = openai_endpoint_->getChatRender();
    const auto  rendered_input = chat_render->render_chat_request(body);
    std::string debug_info     = openai_endpoint_->getDebugInfo(chat_request, rendered_input);

    writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
    writer->Write(debug_info);
}

}  // namespace rtp_llm

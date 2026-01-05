#include "rtp_llm/cpp/api_server/EmbeddingEndpoint.h"

#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingStream.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingScheduler.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingExecutor.h"

using namespace py::literals;

namespace rtp_llm {

std::string EmbeddingEndpoint::embeddingTypeToString(EmbeddingType type) {
    switch (type) {
        case DENSE:
            return "dense";
        case SPARSE:
            return "sparse";
        case COLBERT:
            return "colbert";
        default:
            return "unknown_embedding_type";
    }
}

std::pair<std::string, std::optional<std::string>>
EmbeddingEndpoint::handle(const std::string&                              body,
                          std::optional<EmbeddingEndpoint::EmbeddingType> type,
                          const kmonitor::MetricsReporterPtr&             metrics_reporter,
                          int64_t                                         start_time_us) {
    py::gil_scoped_acquire gil_before_deocde;
    py::module             embedding_endpoint = py::module::import("rtp_llm.embedding.embedding_endpoint");
    py::object             EmbeddingHandler   = embedding_endpoint.attr("EmbeddingHandler");
    py::object             embedding_handler = EmbeddingHandler("request"_a = body, "custom_module"_a = custom_module_);
    if (type.has_value()) {
        auto type_str = embeddingTypeToString(type.value());
        embedding_handler.attr("set_embedding_type")(type_str);
    }
    auto batch_input    = embedding_handler.attr("create_batch_input")();
    auto token_ids      = py::cast<th::Tensor>(batch_input.attr("token_ids"));
    auto token_type_ids = py::cast<th::Tensor>(batch_input.attr("token_type_ids"));
    auto input_lengths  = py::cast<th::Tensor>(batch_input.attr("input_lengths"));
    auto mm_features    = getMultimodalFeature(batch_input.attr("multimodal_inputs"), token_ids);

    py::gil_scoped_release gil_release;
    metrics_reporter->report((autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us) / 1000.0,
                             "ft_pre_pipeline_rt",
                             kmonitor::MetricType::GAUGE,
                             nullptr,
                             true);
    auto results  = embedding_engine_->decode(token_ids, token_type_ids, input_lengths, 0, mm_features);
    start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();

    py::gil_scoped_acquire gil_after_deocde;
    py::object             batch_output;
    if (results->output.isTensor) {
        RTP_LLM_CHECK_WITH_INFO(results->output.t.has_value(), "embedding output has null tensor value");
        batch_output = rtp_llm::convertTensorToObject(results->output.t.value());
    } else {
        RTP_LLM_CHECK_WITH_INFO(results->output.map.has_value(), "embedding output has null map value");
        batch_output = rtp_llm::convertTensorMapVectorToObject(results->output.map.value());
    }

    py::module::import("nest_asyncio").attr("apply")();
    auto loop             = py::module::import("asyncio").attr("get_event_loop")();
    auto coro             = embedding_handler.attr("render_response")(batch_output);
    auto response         = getAsyncResult(loop, coro);
    coro                  = embedding_handler.attr("render_log_response")();
    auto logable_response = getAsyncResult(loop, coro);

    metrics_reporter->report((autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us) / 1000.0,
                             "ft_post_pipeline_rt",
                             kmonitor::MetricType::GAUGE,
                             nullptr,
                             true);
    if (logable_response == "null") {
        return std::make_pair(response, std::nullopt);
    } else {
        return std::make_pair(response, logable_response);
    }
}

std::string EmbeddingEndpoint::getAsyncResult(py::object loop, py::object coro) {
    auto future = loop.attr("create_task")(coro);
    loop.attr("run_until_complete")(future);
    auto result = future.attr("result")();
    return py::cast<std::string>(result);
}

std::optional<MultimodalFeature> EmbeddingEndpoint::getMultimodalFeature(py::object  py_mm_inputs,
                                                                         th::Tensor& token_ids) {
    if (!py::isinstance<py::list>(py_mm_inputs)) {
        throw std::runtime_error("Expected a list, but get " + py::cast<std::string>(py::str(py_mm_inputs)));
    }
    std::vector<MultimodalInput> mm_inputs;
    auto                         py_list = py::reinterpret_borrow<py::list>(py_mm_inputs);
    for (const auto& item : py_list) {
        mm_inputs.emplace_back(py::cast<std::string>(item.attr("url")),
                               std::vector<th::Tensor>{py::cast<th::Tensor>(item.attr("tensor"))},
                               py::cast<int32_t>(item.attr("mm_type")));
    }
    std::optional<MultimodalFeature> multimodal_features = std::nullopt;
    if (mm_processor_ != nullptr && !mm_inputs.empty()) {
        auto mm_res = mm_processor_->getMultimodalFeatures(rtp_llm::torchTensor2Buffer(token_ids), mm_inputs);
        if (!mm_res.ok()) {
            throw std::runtime_error(mm_res.status().ToString());
        }
        token_ids = rtp_llm::Buffer2torchTensor(mm_res.value().expanded_ids, true);
        multimodal_features.emplace(mm_res.value());
    }
    return multimodal_features;
}

}  // namespace rtp_llm

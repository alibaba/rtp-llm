#include "src/fastertransformer/th_op/multi_gpu_gpt/RtpEmbeddingOp.h"

#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/embedding_engine/EmbeddingQueryConverter.h"
#include "src/fastertransformer/devices/utils/BufferTorchUtils.h"

using namespace std;

namespace ft = fastertransformer;
namespace th = torch;

namespace torch_ext {

std::unordered_map<std::string, torch::Tensor> convert_pyobject_to_dict(py::handle obj) {

    if (!py::isinstance<py::dict>(obj)) {
        throw std::runtime_error("Expected a dicts");
    }

    // 从列表项中提取 dict
    py::dict py_dict = py::reinterpret_borrow<py::dict>(obj);
    // 创建一个 unordered_map
    std::unordered_map<std::string, torch::Tensor> map;

    // 遍历 dict 键值对
    for (auto kv : py_dict) {
        std::string   key   = py::cast<std::string>(kv.first);
        torch::Tensor value = py::cast<torch::Tensor>(kv.second);
        // 添加到 map 中
        map[key] = value;
    }
    return map;
}

std::vector<std::unordered_map<std::string, torch::Tensor>> convert_pyobject_to_vector_dict(py::object obj) {
    // 确认 obj 是一个 Python 列表
    if (!py::isinstance<py::list>(obj)) {
        throw std::runtime_error("Expected a list");
    }

    // 从 py::object 提取 py::list
    py::list py_list = py::reinterpret_borrow<py::list>(obj);

    // 创建一个 std::vector<std::unordered_map<std::string, torch::Tensor>>
    std::vector<std::unordered_map<std::string, torch::Tensor>> vec;

    // 遍历 Python 列表，逐项转换
    for (auto item : py_list) {
        vec.push_back(std::move(convert_pyobject_to_dict(item)));
    }
    return vec;
}

RtpEmbeddingOp::RtpEmbeddingOp(const ft::GptInitParameter& gpt_init_params, py::object handler_impl):
    gpt_init_params_(gpt_init_params), handler_(handler_impl) {}

void RtpEmbeddingOp::init(py::object layer_weights, py::object weights) {
    AUTIL_ROOT_LOG_CONFIG();
    AUTIL_ROOT_LOG_SETLEVEL(INFO);
    (void)rtp_llm::initKmonitorFactory();
    auto kmon_tags = rtp_llm::getHippoTags();
    metrics_reporter_.reset(new kmonitor::MetricsReporter("", "", kmon_tags));    

    ft::DeviceFactory::initDevices(ft::DeviceFactory::getDefaultGlobalDeviceParams());

    auto layer_weights_cc = convert_pyobject_to_vector_dict(layer_weights);
    auto weights_cc = convert_pyobject_to_dict(weights);
    for (auto& it : weights_cc) {
        global_weights_.emplace(it.first, ft::torchTensor2Buffer(it.second));
    }
    for (auto& weights : layer_weights_cc) {
        std::unordered_map<std::string, ft::ConstBufferPtr> __weights;
        for (auto& it : weights) {
            __weights.emplace(it.first, ft::torchTensor2Buffer(it.second));
        }
        layer_weights_.emplace_back(std::move(__weights));
    }
    embedding_engine_.reset(new rtp_llm::EmbeddingEngine(gpt_init_params_, layer_weights_, global_weights_, handler_, metrics_reporter_));
}

void RtpEmbeddingOp::stop() {
    if (!is_server_shutdown_) {
        (void)embedding_engine_->stop();
        is_server_shutdown_ = true;
    }
}

th::Tensor RtpEmbeddingOp::decode(th::Tensor token_ids, th::Tensor token_type_ids, th::Tensor input_lengths, int64_t request_id) {
    if (is_server_shutdown_) {
        throw std::runtime_error("server is shut down, can't handle request");
    }
    auto embedding_stream = rtp_llm::EmbeddingQueryConverter::convertEmbeddingInputs(token_ids, token_type_ids, input_lengths, request_id);
    embedding_stream->setMetricReporter(metrics_reporter_);
    THROW_IF_STATUS_ERROR(embedding_engine_->enqueue(embedding_stream));
    embedding_stream->waitFinish();
    return rtp_llm::EmbeddingQueryConverter::convertEmbeddingOutputs(embedding_stream);
}

RtpEmbeddingOp::~RtpEmbeddingOp() {
    stop();
}

void registerRtpEmbeddingOp(const py::module_& m) {
    pybind11::class_<torch_ext::RtpEmbeddingOp>(m, "RtpEmbeddingOp")
        .def(pybind11::init<const ft::GptInitParameter&, py::object>())  // quant_pre_scales
        .def("init", &torch_ext::RtpEmbeddingOp::init)
        .def("stop", &torch_ext::RtpEmbeddingOp::stop)
        .def("decode", &torch_ext::RtpEmbeddingOp::decode, py::call_guard<py::gil_scoped_release>());        
}

}  // namespace torch_ext

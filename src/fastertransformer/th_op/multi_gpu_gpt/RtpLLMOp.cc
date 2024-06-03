#include "src/fastertransformer/th_op/multi_gpu_gpt/RtpLLMOp.h"
#include "autil/Log.h"
#include "c10/util/intrusive_ptr.h"
#include "maga_transformer/cpp/common/torch_bind.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/utils/pybind_utils.h"
#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"

using namespace std;

namespace ft = fastertransformer;
namespace th = torch;

namespace torch_ext {

RtpLLMOp::RtpLLMOp() {}

void RtpLLMOp::init(const ft::GptInitParameter& gpt_init_params,
                    py::object                  py_layers_weights,
                    py::object                  py_global_weights) {
    AUTIL_ROOT_LOG_CONFIG();
    AUTIL_ROOT_LOG_SETLEVEL(INFO);

    auto                      global_weights = rtp_llm::WeightsConverter::convertPyWeightsMap(py_global_weights);
    auto                      layers_weights = rtp_llm::WeightsConverter::convertPyWeightsMapVec(py_layers_weights);
    rtp_llm::EngineInitParams params(gpt_init_params, layers_weights, global_weights);
    // kmon metric init
    (void)rtp_llm::initKmonitorFactory();
    auto kmon_tags = rtp_llm::getHippoTags();
    params.metrics_reporter.reset(new kmonitor::MetricsReporter("", "", kmon_tags));

    grpc_server_thread_ = std::thread(&RtpLLMOp::_init, this, gpt_init_params.model_rpc_port_, std::move(params));
    grpc_server_thread_.detach();
    while (!is_server_ready_) {
        sleep(1);  // wait 1s for server ready
    }
}

void RtpLLMOp::addLoRA(const int64_t lora_id, py::object py_lora_a_weights, py::object py_lora_b_weights) {
    auto lora_a_weights = rtp_llm::WeightsConverter::convertPyWeightsMapVec(py_lora_a_weights);
    auto lora_b_weights = rtp_llm::WeightsConverter::convertPyWeightsMapVec(py_lora_b_weights);
    model_rpc_server_->addLoRA(lora_id, lora_a_weights, lora_b_weights);
}

void RtpLLMOp::removeLoRA(const int64_t lora_id) {
    model_rpc_server_->removeLoRA(lora_id);
}

void RtpLLMOp::_init(const int64_t model_rpc_port, const rtp_llm::EngineInitParams params) {
    std::string server_address("0.0.0.0:" + std::to_string(model_rpc_port));
    model_rpc_server_.reset(new rtp_llm::ModelRpcServiceImpl(params));
    if (model_rpc_port < 0) {
        is_server_ready_ = true;
        return;
    }
    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(model_rpc_server_.get());
    grpc_server_ = builder.BuildAndStart();

    FT_LOG_INFO("Server listening on %s", server_address.c_str());
    is_server_ready_ = true;
    grpc_server_->Wait();
    is_server_shutdown_ = true;
}

void RtpLLMOp::stop() {
    if (!is_server_shutdown_) {
        if (grpc_server_) {
            grpc_server_->Shutdown();
        }
        model_rpc_server_.reset();
    }
}

RtpLLMOp::~RtpLLMOp() {
    stop();
}

void registerRtpLLMOp(const py::module& m) {
    pybind11::class_<torch_ext::RtpLLMOp>(m, "RtpLLMOp")
        .def(pybind11::init<>())
        .def("init", &torch_ext::RtpLLMOp::init)
        .def("add_lora", &torch_ext::RtpLLMOp::addLoRA)
        .def("remove_lora", &torch_ext::RtpLLMOp::removeLoRA)
        .def("stop", &torch_ext::RtpLLMOp::stop);
}

}  // namespace torch_ext

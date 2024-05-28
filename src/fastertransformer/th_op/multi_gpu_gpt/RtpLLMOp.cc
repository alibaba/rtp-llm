#include "src/fastertransformer/th_op/multi_gpu_gpt/RtpLLMOp.h"
#include "c10/util/intrusive_ptr.h"
#include "maga_transformer/cpp/common/torch_bind.h"
#include "maga_transformer/cpp/dataclass/MagaInitParameter.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/devices/utils/BufferUtils.h"
#include "src/fastertransformer/devices/utils/BufferTorchUtils.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/utils/pybind_utils.h"
#include "autil/Log.h"

using namespace std;

namespace ft = fastertransformer;
namespace th = torch;

namespace torch_ext {

RtpLLMOp::RtpLLMOp() {}

void RtpLLMOp::init(const ft::GptInitParameter& gpt_init_params, py::object layer_weights, py::object weights) {
    AUTIL_ROOT_LOG_CONFIG();
    AUTIL_ROOT_LOG_SETLEVEL(INFO);
    rtp_llm::MagaInitParams params(gpt_init_params);    

    std::unordered_map<std::string, ft::ConstBufferPtr>              global_weights;
    std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>> layer_weights_;
    auto layer_weights_cc = fastertransformer::convertPyobjectToVectorDict(layer_weights);
    auto weights_cc = fastertransformer::convertPyObjectToDict(weights);

    for (auto& it : weights_cc) {
        global_weights.emplace(it.first, ft::torchTensor2Buffer(it.second));
    }
    for (auto& weights : layer_weights_cc) {
        std::unordered_map<std::string, ft::ConstBufferPtr> __weights;
        for (auto& it : weights) {
            __weights.emplace(it.first, ft::torchTensor2Buffer(it.second));
        }
        layer_weights_.emplace_back(std::move(__weights));
    }
    grpc_server_thread_ = std::thread(&RtpLLMOp::_init, this, gpt_init_params.model_rpc_port_, params, std::move(layer_weights_), std::move(global_weights));
    grpc_server_thread_.detach();
    while(!is_server_ready_) {
        sleep(1); // wait 1s for server ready
    }
    // _init(params, layer_weights_, global_weights);
}

void RtpLLMOp::addLoRA(const int64_t lora_id, py::object lora_a_weights, py::object lora_b_weights) {
    std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>> lora_a_weights_, lora_b_weights_;
    auto lora_a_weights_cc = ft::convertPyobjectToVectorDict(lora_a_weights);
    auto lora_b_weights_cc = ft::convertPyobjectToVectorDict(lora_b_weights);
    for (auto& weights : lora_a_weights_cc) {
        std::unordered_map<std::string, ft::ConstBufferPtr> weights_;
        for (auto& it : weights) {
            weights_.emplace(it.first, ft::torchTensor2Buffer(it.second));
        }
        lora_a_weights_.emplace_back(std::move(weights_));
    }    
    for (auto& weights : lora_b_weights_cc) {
        std::unordered_map<std::string, ft::ConstBufferPtr> weights_;
        for (auto& it : weights) {
            weights_.emplace(it.first, ft::torchTensor2Buffer(it.second));
        }
        lora_b_weights_.emplace_back(std::move(weights_));
    }
    model_rpc_server_->addLoRA(lora_id, lora_a_weights_, lora_b_weights_);
}

void RtpLLMOp::removeLoRA(const int64_t lora_id) {
    model_rpc_server_->removeLoRA(lora_id);
}

void RtpLLMOp::_init(const int64_t                                                          model_rpc_port,
                     const rtp_llm::MagaInitParams                                          params,
                     const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>> layer_weights,
                     const std::unordered_map<std::string, ft::ConstBufferPtr>              weights) {
    std::string                  server_address("0.0.0.0:" + std::to_string(model_rpc_port));
    model_rpc_server_.reset(new rtp_llm::ModelRpcServiceImpl(params, layer_weights, weights));
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
        .def(pybind11::init<>())  // quant_pre_scales
        .def("init", &torch_ext::RtpLLMOp::init)
        .def("add_lora", &torch_ext::RtpLLMOp::addLoRA)
        .def("remove_lora", &torch_ext::RtpLLMOp::removeLoRA)
        .def("stop", &torch_ext::RtpLLMOp::stop);
}

}  // namespace torch_ext

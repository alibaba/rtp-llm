#include "src/fastertransformer/th_op/multi_gpu_gpt/RtpLLMOp.h"
#include "autil/Log.h"
#include "c10/util/intrusive_ptr.h"
#include "maga_transformer/cpp/common/torch_bind.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"
#include "maga_transformer/cpp/dataclass/LoadBalance.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"
#include "src/fastertransformer/utils/py_utils/pybind_utils.h"
#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"
#include <cstddef>
#include <memory>
#include <pybind11/pytypes.h>
#include <tuple>

using namespace std;

namespace ft = fastertransformer;
namespace th = torch;

namespace torch_ext {

RtpLLMOp::RtpLLMOp() {}

void RtpLLMOp::init(py::object model,
                    py::object mm_process_engine,
                    py::object propose_model,
                    py::object token_processor) {
    AUTIL_ROOT_LOG_CONFIG();
    AUTIL_ROOT_LOG_SETLEVEL(INFO);
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    rtp_llm::EngineInitParams params = initModel(model);
    std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params = initProposeModel(propose_model);
    grpc_server_thread_ = std::thread(&RtpLLMOp::_init, this, params.gpt_init_parameter.model_rpc_port_,
                                                              params.gpt_init_parameter.http_port_,
                                                              std::move(params),
                                                              std::move(mm_process_engine),
                                                              std::move(propose_params),
                                                              std::move(token_processor));
    grpc_server_thread_.detach();
    while (!is_server_ready_) {
        sleep(1);  // wait 1s for server ready
    }
}

rtp_llm::EngineInitParams RtpLLMOp::initModel(py::object model) {
    try {

        auto [gpt_init_params, gpt_weight] = rtp_llm::prepareEngineInitParams(model);
        rtp_llm::EngineInitParams params(gpt_init_params, std::move(*gpt_weight));
        if (gpt_init_params.tp_rank_ == 0) {
            // kmon metric init
            (void)rtp_llm::initKmonitorFactory();
            auto kmon_tags = rtp_llm::getHippoTags();
            params.metrics_reporter.reset(new kmonitor::MetricsReporter("", "", kmon_tags));
        }
        return params;
     } catch (const std::exception& e){
        FT_FAIL("init engine params failed, error msg: %s", e.what());
        return rtp_llm::EngineInitParams();
    }
}

std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> RtpLLMOp::initProposeModel(py::object propose_model) {
    try {
        if (propose_model.is_none()) {
            return nullptr;
        }
        std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> params = nullptr;
        std::string sp_type = propose_model.attr("sp_type").cast<std::string>();
        if (sp_type == "vanilla") {
            auto [gpt_init_params, gpt_weight] = rtp_llm::prepareEngineInitParams(propose_model, true);
            params = std::make_unique<rtp_llm::ProposeModelEngineInitParams>(sp_type,
                                                                             gpt_init_params,
                                                                             std::move(*gpt_weight));
        } else if (sp_type == "prompt_lookup") {
            params = std::make_unique<rtp_llm::ProposeModelEngineInitParams>(sp_type);
        } else if (sp_type == "eagle") {
            FT_FAIL("sp_type %s not support", sp_type.c_str());
        } else if (sp_type == "medusa") {
            FT_FAIL("sp_type %s not support", sp_type.c_str());
        } else {
            FT_FAIL("sp_type %s not support", sp_type.c_str());
        }
        return params;
     } catch (const std::exception& e ){
        FT_FAIL("init propose engine params failed, error msg: %s", e.what());
        return nullptr;
    }
}


void RtpLLMOp::addLora(const std::string& adapter_name, py::object py_lora_a_weights, py::object py_lora_b_weights) {
    auto convert = rtp_llm::WeightsConverter(true);
    auto lora_a_weights = convert.convertLayerWeights_(py_lora_a_weights);
    auto lora_b_weights = convert.convertLayerWeights_(py_lora_b_weights);
    model_rpc_server_->addLora(adapter_name, *lora_a_weights, *lora_b_weights);
}

void RtpLLMOp::removeLora(const std::string& adapter_name) {
    model_rpc_server_->removeLora(adapter_name);
}

rtp_llm::LoadBalanceInfo RtpLLMOp::getLoadBalanceInfo() {
    return model_rpc_server_->getLoadBalanceInfo();
}

void RtpLLMOp::_init(const int64_t model_rpc_port,
                     const int64_t http_port,
                     const rtp_llm::EngineInitParams params,
                     py::object mm_process_engine,
                     std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                     py::object token_processor) {
    std::string server_address("0.0.0.0:" + std::to_string(model_rpc_port));
    std::unique_ptr<rtp_llm::ModelRpcServiceImpl> rpc_server = std::make_unique<rtp_llm::ModelRpcServiceImpl>();
    grpc::Status grpc_status = rpc_server->init(params, std::move(mm_process_engine), std::move(propose_params));
    if (!grpc_status.ok()) {
        FT_FAIL("init rpc server failed, error msg: %s", grpc_status.error_message().c_str());
    }
    model_rpc_server_ = std::move(rpc_server);
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
    {
        http_server_.reset(new rtp_llm::HttpApiServer(model_rpc_server_->getEngine(),
                                                      params.gpt_init_parameter,
                                                      token_processor));
        http_server_->registerResponses();
        std::string http_server_address("tcp:0.0.0.0:" + std::to_string(http_port));
        if (http_server_->start(http_server_address)) {
            FT_LOG_INFO("normal HTTP Server listening on %s", http_server_address.c_str());
        } else {
            FT_LOG_ERROR("normal HTTP Server start fail.");
        }
    }
    grpc_server_->Wait();
    is_server_shutdown_ = true;
}

void RtpLLMOp::stop() {
    if (!is_server_shutdown_) {
        if (grpc_server_) {
            grpc_server_->Shutdown();
        }
        model_rpc_server_.reset();
        if (http_server_) {
            http_server_->stop();
        }
    }
}

RtpLLMOp::~RtpLLMOp() {
    stop();
}

void registerRtpLLMOp(const py::module& m) {
    rtp_llm::registerLoadBalanceInfo(m);
    pybind11::class_<torch_ext::RtpLLMOp>(m, "RtpLLMOp")
        .def(pybind11::init<>())
        .def("init", &torch_ext::RtpLLMOp::init)
        .def("add_lora", &torch_ext::RtpLLMOp::addLora)
        .def("remove_lora", &torch_ext::RtpLLMOp::removeLora)
        .def("get_load_balance_info", &torch_ext::RtpLLMOp::getLoadBalanceInfo)
        .def("stop", &torch_ext::RtpLLMOp::stop);
}

}  // namespace torch_ext

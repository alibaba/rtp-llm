#include <cstddef>
#include <memory>
#include <tuple>
#include "autil/Log.h"
#include "c10/util/intrusive_ptr.h"
#include <grpcpp/grpcpp.h>
#include <grpcpp/resource_quota.h>
#include "rtp_llm/cpp/dataclass/EngineInitParameter.h"
#include "rtp_llm/cpp/dataclass/LoadBalance.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/th_op/GptInitParameter.h"
#include "rtp_llm/cpp/th_op/multi_gpu_gpt/RtpLLMOp.h"

using namespace std;
namespace th = torch;


namespace torch_ext {

RtpLLMOp::RtpLLMOp() {}

void RtpLLMOp::init(py::object model,
                    py::object mm_process_engine,
                    py::object propose_model,
                    py::object token_processor) {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    rtp_llm::EngineInitParams params = initModel(model);
    RTP_LLM_LOG_INFO("init engine params success");
    params.showGptInitParameter();
    std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params = initProposeModel(propose_model);
    pybind11::gil_scoped_release release;
    grpc_server_thread_ = std::thread(&RtpLLMOp::initRPCServer, this,
        std::move(params), std::move(mm_process_engine), std::move(propose_params), std::move(token_processor));
    grpc_server_thread_.detach();
    while (!is_server_ready_) {
        sleep(1);  // wait 1s for server ready
    }
}

rtp_llm::EngineInitParams RtpLLMOp::initModel(py::object model) {
    try {
        auto [gpt_init_params, gpt_weight] = rtp_llm::prepareEngineInitParams(model);
        rtp_llm::EngineInitParams params(model_id_, gpt_init_params, std::move(*gpt_weight));
        model_id_++;
        if (gpt_init_params.tp_rank_ == 0) {
            // kmon metric init
            (void)rtp_llm::initKmonitorFactory();
            auto kmon_tags = kmonitor::MetricsTags();
            kmon_tags.AddTag("dp_rank", std::to_string(gpt_init_params.dp_rank_));
            params.metrics_reporter.reset(new kmonitor::MetricsReporter("", "", kmon_tags));
        }
        return params;
     } catch (const std::exception& e){
        RTP_LLM_FAIL("init engine params failed, error msg: %s", e.what());
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
        size_t gen_num_per_circle = propose_model.attr("gen_num_per_circle").cast<size_t>();
        if (sp_type == "vanilla") {
            auto [gpt_init_params, gpt_weight] = rtp_llm::prepareEngineInitParams(propose_model, true);
            params = std::make_unique<rtp_llm::ProposeModelEngineInitParams>(model_id_, 
                                                                             sp_type,
                                                                             gen_num_per_circle,
                                                                             gpt_init_params,
                                                                             std::move(*gpt_weight));
            model_id_++;
        } else if (sp_type == "mtp" || sp_type == "eagle3") {
            params = rtp_llm::prepareMTPEngineInitParams(model_id_, propose_model);
            model_id_ += gen_num_per_circle;
        } else if (sp_type == "deterministic") {
            params = std::make_unique<rtp_llm::ProposeModelEngineInitParams>(sp_type, gen_num_per_circle);
        } else {
            RTP_LLM_FAIL("sp_type %s not support", sp_type.c_str());
        }
        return params;
     } catch (const std::exception& e ){
        RTP_LLM_FAIL("init propose engine params failed, error msg: %s", e.what());
        return nullptr;
    }
}

void RtpLLMOp::addLora(const std::string& adapter_name, py::object py_lora_a_weights, py::object py_lora_b_weights) {
    auto convert = rtp_llm::WeightsConverter(true);
    auto lora_a_weights = convert.convertLayerWeights_(py_lora_a_weights);
    auto lora_b_weights = convert.convertLayerWeights_(py_lora_b_weights);
    model_rpc_service_->addLora(adapter_name, *lora_a_weights, *lora_b_weights);
}

void RtpLLMOp::removeLora(const std::string& adapter_name) {
    model_rpc_service_->removeLora(adapter_name);
}

rtp_llm::LoadBalanceInfo RtpLLMOp::getLoadBalanceInfo() {
    return model_rpc_service_->getLoadBalanceInfo();
}

rtp_llm::EngineScheduleInfo RtpLLMOp::getEngineScheduleInfo() {
    return model_rpc_service_->getEngineScheduleInfo();
}

void RtpLLMOp::initRPCServer(
                     const rtp_llm::EngineInitParams maga_init_params,
                     py::object mm_process_engine,
                     std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                     py::object token_processor) {
    auto http_port = maga_init_params.gpt_init_parameter.http_port_;
    auto model_rpc_port = maga_init_params.gpt_init_parameter.model_rpc_port_;
    auto use_cache_store = maga_init_params.gpt_init_parameter.use_cache_store_;
    auto py_inference_log_response = maga_init_params.gpt_init_parameter.profiling_debug_logging_config.py_inference_log_response;
    std::string server_address("0.0.0.0:" + std::to_string(model_rpc_port));
    {
        pybind11::gil_scoped_acquire acquire;
        if (use_cache_store) {
            model_rpc_service_.reset(new rtp_llm::RemoteRpcServiceImpl());
        } else {
            model_rpc_service_.reset(new rtp_llm::LocalRpcServiceImpl());
        }
        grpc::Status grpc_status = model_rpc_service_->init(maga_init_params, std::move(mm_process_engine), std::move(propose_params));
        if (!grpc_status.ok()) {
            RTP_LLM_FAIL("init rpc server failed, error msg: %s", grpc_status.error_message().c_str());
        }

        std::string http_server_address("tcp:0.0.0.0:" + std::to_string(http_port));
        http_server_.reset(new rtp_llm::HttpApiServer(model_rpc_service_->getEngine(),
                                                    model_rpc_service_->getMultimodalProcessor(),
                                                    http_server_address,
                                                    maga_init_params,
                                                    token_processor,
                                                    py_inference_log_response));
        if (model_rpc_port < 0) {
            is_server_ready_ = true;
            return;
        }
    }
    grpc::ServerBuilder builder;
    builder.AddChannelArgument(GRPC_ARG_MAX_CONCURRENT_STREAMS, 100000);
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(model_rpc_service_.get());
    grpc_server_ = builder.BuildAndStart();
    RTP_LLM_CHECK_WITH_INFO(grpc_server_ != nullptr, "grpc server start failed at address " + server_address);

    RTP_LLM_LOG_INFO("Server listening on %s", server_address.c_str());
    is_server_ready_ = true;
    grpc_server_->Wait();
    RTP_LLM_LOG_INFO("Server exit on %s", server_address.c_str());
}

bool RtpLLMOp::ready() {
    return model_rpc_service_->ready();
}

void RtpLLMOp::startHttpServer(py::object model_weights_loader,
                               py::object lora_infos,
                               py::object gang_info,
                               py::object tokenizer,
                               py::object render) {
    if (http_server_ == nullptr) {
        RTP_LLM_FAIL("normal HTTP Server nullptr error.");
        return;
    }
    if (http_server_->start(model_weights_loader, lora_infos, gang_info, tokenizer, render)) {
        RTP_LLM_LOG_INFO("normal HTTP Server listening on %s", http_server_->getListenAddr().c_str());
    } else {
        RTP_LLM_FAIL("normal HTTP Server start fail.");
    }
}

void RtpLLMOp::updateSchedulerInfo(const std::string& scheduler_info) {
    model_rpc_service_->getEngine()->getScheduler().updateSchedulerInfo(scheduler_info);
}

bool RtpLLMOp::updateEplbConfig(const rtp_llm::EplbConfig& config) {
    if (model_rpc_service_) {
        return model_rpc_service_->getEngine()->updateEplbConfig(config);
    }
    return false;
}

void RtpLLMOp::stop() {
    int64_t STOP_TIMEOUT_MS = 60 * 1000;
    if (!is_server_shutdown_) {
        if (grpc_server_) {
            auto begin_wait_us = autil::TimeUtility::currentTimeInMicroSeconds();
            while (auto onflight_request = model_rpc_service_->onflightRequestNum()) {
                RTP_LLM_LOG_INFO("rpc service has [%lu] onflight request, waitting 1s", onflight_request);
                sleep(1);
                if (autil::TimeUtility::currentTimeInMicroSeconds() - begin_wait_us > STOP_TIMEOUT_MS * 1000) {
                    RTP_LLM_LOG_INFO("rpc service wait timeout, no more waiting");
                    break;
                }
            }
            RTP_LLM_LOG_INFO("Server shutdowning");
            grpc_server_->Shutdown();
            grpc_server_.reset();
        }
        {
            pybind11::gil_scoped_release release;
            model_rpc_service_->stop();
            pybind11::gil_scoped_acquire acquire;
        }
        model_rpc_service_.reset();
        if (http_server_) {
            http_server_->stop();
            http_server_.reset();
        }
        is_server_shutdown_ = true;
        rtp_llm::stopKmonitorFactory();
    }
}

RtpLLMOp::~RtpLLMOp() {
    stop();
}

void registerRtpLLMOp(const py::module& m) {
    pybind11::class_<torch_ext::RtpLLMOp>(m, "RtpLLMOp")
        .def(pybind11::init<>())
        .def("init",
             &torch_ext::RtpLLMOp::init,
             py::arg("model"),
             py::arg("mm_process_engine"),
             py::arg("propose_model"),
             py::arg("token_processor"))
        .def("start_http_server",
             &torch_ext::RtpLLMOp::startHttpServer,
             py::arg("model_weights_loader"),
             py::arg("lora_infos"),
             py::arg("gang_info"),
             py::arg("tokenizer"),
             py::arg("render"))
        .def("add_lora",
             &torch_ext::RtpLLMOp::addLora,
             py::arg("adapter_name"),
             py::arg("lora_a_weights"),
             py::arg("lora_b_weights"))
        .def("remove_lora", &torch_ext::RtpLLMOp::removeLora, py::arg("adapter_name"))
        .def("get_load_balance_info", &torch_ext::RtpLLMOp::getLoadBalanceInfo)
        .def("get_engine_schedule_info", &torch_ext::RtpLLMOp::getEngineScheduleInfo)
        .def("update_scheduler_info", &torch_ext::RtpLLMOp::updateSchedulerInfo)
        .def("stop", &torch_ext::RtpLLMOp::stop)
        .def("ready", &torch_ext::RtpLLMOp::ready)
        .def("update_eplb_config", &torch_ext::RtpLLMOp::updateEplbConfig, py::arg("config"));
}

}  // namespace torch_ext

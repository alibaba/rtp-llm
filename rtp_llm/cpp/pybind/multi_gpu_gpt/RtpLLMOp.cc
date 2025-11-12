#include <cstddef>
#include <memory>
#include <tuple>
#include "autil/Log.h"
#include "c10/util/intrusive_ptr.h"
#include <grpcpp/grpcpp.h>
#include <grpcpp/resource_quota.h>
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/pybind/multi_gpu_gpt/RtpLLMOp.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/engine_base/WeightsConverter.h"
#include "rtp_llm/cpp/engine_base/WorkerStatusInfo.h"
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/models/models_weight/W.h"

using namespace std;
namespace th = torch;

namespace rtp_llm {

std::unique_ptr<ProposeModelEngineInitParams> prepareMTPEngineInitParams(size_t model_id, py::object model) {
    auto        sp_model           = model.attr("model");
    std::string sp_type            = model.attr("sp_type").cast<std::string>();
    RTP_LLM_CHECK(sp_type == "mtp" || sp_type == "eagle3" || sp_type == "eagle");

    std::unique_ptr<std::vector<std::unique_ptr<EngineInitParams>>> mtp_params =
        std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    py::object config_obj = sp_model.attr("config");
    // Extract individual config members from Python config object
    auto model_config = config_obj.attr("py_model_config").cast<ModelConfig>();
    auto parallelism_config = config_obj.attr("parallelism_config").cast<ParallelismConfig>();
    auto runtime_config = config_obj.attr("runtime_config").cast<RuntimeConfig>();
    auto pd_sep_config = config_obj.attr("pd_sep_config").cast<PDSepConfig>();
    auto concurrency_config = config_obj.attr("concurrency_config").cast<ConcurrencyConfig>();
    auto fmha_config = config_obj.attr("fmha_config").cast<FMHAConfig>();
    auto kv_cache_config = config_obj.attr("kv_cache_config").cast<KVCacheConfig>();
    auto profiling_debug_logging_config = config_obj.attr("profiling_debug_logging_config").cast<ProfilingDebugLoggingConfig>();
    auto hw_kernel_config = config_obj.attr("hw_kernel_config").cast<HWKernelConfig>();
    auto device_resource_config = config_obj.attr("device_resource_config").cast<DeviceResourceConfig>();
    auto moe_config = config_obj.attr("moe_config").cast<MoeConfig>();
    auto model_specific_config = config_obj.attr("model_specific_config").cast<ModelSpecificConfig>();
    auto sp_config = config_obj.attr("sp_config").cast<SpeculativeExecutionConfig>();
    auto cache_store_config = config_obj.attr("cache_store_config").cast<CacheStoreConfig>();
    auto misc_config = config_obj.attr("misc_config").cast<MiscellaneousConfig>();
    auto arpc_config = config_obj.attr("arpc_config").cast<ArpcConfig>();
    auto ffn_disaggregate_config = config_obj.attr("ffn_disaggregate_config").cast<FfnDisAggregateConfig>();
    auto vit_config = config_obj.attr("vit_config").cast<VitConfig>();

    py::object py_layers_weights     = sp_model.attr("weight").attr("weights");
    py::object py_global_weights     = sp_model.attr("weight").attr("global_weights");
    auto       convert               = WeightsConverter(false, model_config.quant_algo);
    auto       py_layers_weights_vec = convertPyObjectToVec(py_layers_weights);
    size_t     model_num             = py_layers_weights_vec.size();
    size_t     gen_num_per_cycle     = sp_config.gen_num_per_cycle;
    if (gen_num_per_cycle > 1 && py_layers_weights_vec.size() == 1) {
        RTP_LLM_LOG_WARNING("duplicate py_layers_weights_vec from 1 to sp_config.gen_num_per_cycle: %ld",
                            gen_num_per_cycle);
        for (size_t i = 1; i < gen_num_per_cycle; i++) {
            py_layers_weights_vec.push_back(py_layers_weights_vec[0]);
        }
        model_num = gen_num_per_cycle;
    }
    if (gen_num_per_cycle != py_layers_weights_vec.size()) {
        RTP_LLM_LOG_WARNING("sp_config.gen_num_per_cycle: %ld  != py_layers_weights_vec.size(): %ld",
                            gen_num_per_cycle,
                            py_layers_weights_vec.size());
        model_num = std::min(model_num, size_t(gen_num_per_cycle));
    }
    if (sp_type == "eagle" || sp_type == "eagle3") {
        model_num = 1;
    }

    // Get py_eplb if available
    py::object py_eplb = py::none();
    if (py::hasattr(config_obj, "py_eplb")) {
        py_eplb = config_obj.attr("py_eplb");
    }
    
    // Create a temporary ModelConfig with num_layers = 1 for MTP
    ModelConfig temp_model_config = model_config;
    temp_model_config.num_layers = 1;

    for (int i = 0; i < model_num; i++) {
        auto     layer_weigths = py_layers_weights_vec[i];
        py::list tmp;
        tmp.append(layer_weigths);
        auto gpt_weight = convert.createGptWeights(tmp, py_global_weights);
        mtp_params->push_back(
            std::move(std::make_unique<EngineInitParams>(
                model_id,
                temp_model_config,
                parallelism_config,
                runtime_config,
                pd_sep_config,
                concurrency_config,
                fmha_config,
                kv_cache_config,
                profiling_debug_logging_config,
                hw_kernel_config,
                device_resource_config,
                moe_config,
                model_specific_config,
                sp_config,
                cache_store_config,
                misc_config,
                arpc_config,
                ffn_disaggregate_config,
                vit_config,
                std::move(*gpt_weight),
                py::none(),
                py_eplb)));
        model_id++;
    }

    return std::move(
        std::make_unique<ProposeModelEngineInitParams>(sp_type, gen_num_per_cycle, std::move(mtp_params)));
};

RtpLLMOp::RtpLLMOp() {}

void RtpLLMOp::init(py::object model,
                    py::object mm_process_engine,
                    py::object propose_model,
                    py::object token_processor) {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    EngineInitParams params = initModel(model);
    RTP_LLM_LOG_INFO("init engine params success");
    params.showDebugInfo();
    std::unique_ptr<ProposeModelEngineInitParams> propose_params = initProposeModel(propose_model);
    pybind11::gil_scoped_release                           release;
    grpc_server_thread_ = std::thread(&RtpLLMOp::initRPCServer,
                                      this,
                                      std::move(params),
                                      std::move(mm_process_engine),
                                      std::move(propose_params),
                                      std::move(token_processor));
    grpc_server_thread_.detach();
    while (!is_server_ready_) {
        sleep(1);  // wait 1s for server ready
    }
}

EngineInitParams RtpLLMOp::initModel(py::object model) {
    try {
        py::object config_obj = model.attr("config");
        // Extract individual config members from Python config object
        auto model_config = config_obj.attr("py_model_config").cast<ModelConfig>();
        // Assign mm_model_config to model_config.mm_model_config
        model_config.mm_model_config = config_obj.attr("mm_model_config").cast<MMModelConfig>();
        auto parallelism_config = config_obj.attr("parallelism_config").cast<ParallelismConfig>();
        auto runtime_config = config_obj.attr("runtime_config").cast<RuntimeConfig>();
        auto pd_sep_config = config_obj.attr("pd_sep_config").cast<PDSepConfig>();
        auto concurrency_config = config_obj.attr("concurrency_config").cast<ConcurrencyConfig>();
        auto fmha_config = config_obj.attr("fmha_config").cast<FMHAConfig>();
        auto kv_cache_config = config_obj.attr("kv_cache_config").cast<KVCacheConfig>();
        auto profiling_debug_logging_config = config_obj.attr("profiling_debug_logging_config").cast<ProfilingDebugLoggingConfig>();
        auto hw_kernel_config = config_obj.attr("hw_kernel_config").cast<HWKernelConfig>();
        auto device_resource_config = config_obj.attr("device_resource_config").cast<DeviceResourceConfig>();
        auto moe_config = config_obj.attr("moe_config").cast<MoeConfig>();
        auto model_specific_config = config_obj.attr("model_specific_config").cast<ModelSpecificConfig>();
        auto sp_config = config_obj.attr("sp_config").cast<SpeculativeExecutionConfig>();
        auto cache_store_config = config_obj.attr("cache_store_config").cast<CacheStoreConfig>();
        auto misc_config = config_obj.attr("misc_config").cast<MiscellaneousConfig>();
        auto arpc_config = config_obj.attr("arpc_config").cast<ArpcConfig>();
        auto ffn_disaggregate_config = config_obj.attr("ffn_disaggregate_config").cast<FfnDisAggregateConfig>();
        VitConfig vit_config;
        if (py::hasattr(config_obj, "vit_config")) {
            py::object py_vit_config = config_obj.attr("vit_config");
            if (!py_vit_config.is_none()) {
                vit_config.vit_separation = py_vit_config.attr("vit_separation").cast<VitSeparation>();
            }
        }
        
        py::object py_layers_weights = model.attr("weight").attr("weights");
        py::object py_global_weights = model.attr("weight").attr("global_weights");
        
        auto convert    = WeightsConverter(false, model_config.quant_algo);
        auto gpt_weight = convert.createGptWeights(py_layers_weights, py_global_weights);
        
        auto py_model = model.attr("py_model");
        // TODO(wangyin.yx): Only one of `py_model` and `gpt_weight` is actually needed.

        // Get py_eplb if available
        py::object py_eplb = py::none();
        if (py::hasattr(config_obj, "py_eplb")) {
            py_eplb = config_obj.attr("py_eplb");
        }

        EngineInitParams params(model_id_,
                                model_config,
                                parallelism_config,
                                runtime_config,
                                pd_sep_config,
                                concurrency_config,
                                fmha_config,
                                kv_cache_config,
                                profiling_debug_logging_config,
                                hw_kernel_config,
                                device_resource_config,
                                moe_config,
                                model_specific_config,
                                sp_config,
                                cache_store_config,
                                misc_config,
                                arpc_config,
                                ffn_disaggregate_config,
                                vit_config,
                                std::move(*gpt_weight),
                                py_model,
                                py_eplb);
        model_id_++;
        if (parallelism_config.tp_rank == 0) {
            // kmon metric init
            (void)initKmonitorFactory();
            auto kmon_tags = kmonitor::MetricsTags();
            kmon_tags.AddTag("dp_rank", std::to_string(parallelism_config.dp_rank));
            params.metrics_reporter.reset(new kmonitor::MetricsReporter("", "", kmon_tags));
        }
        return params;
    } catch (const std::exception& e) {
        RTP_LLM_FAIL("init engine params failed, error msg: %s", e.what());
        return EngineInitParams();
    }
}

std::unique_ptr<ProposeModelEngineInitParams> RtpLLMOp::initProposeModel(py::object propose_model) {
    try {
        if (propose_model.is_none()) {
            return nullptr;
        }
        std::unique_ptr<ProposeModelEngineInitParams> params = nullptr;
        std::string sp_type            = propose_model.attr("sp_type").cast<std::string>();
        if (sp_type == "vanilla") {
            py::object sp_model = propose_model.attr("model");
            py::object config_obj = sp_model.attr("config");
            // Extract individual config members from Python config object
            auto model_config = config_obj.attr("py_model_config").cast<ModelConfig>();
            // Assign mm_model_config to model_config.mm_model_config
            model_config.mm_model_config = config_obj.attr("mm_model_config").cast<MMModelConfig>();
            auto parallelism_config = config_obj.attr("parallelism_config").cast<ParallelismConfig>();
            auto runtime_config = config_obj.attr("runtime_config").cast<RuntimeConfig>();
            auto pd_sep_config = config_obj.attr("pd_sep_config").cast<PDSepConfig>();
            auto concurrency_config = config_obj.attr("concurrency_config").cast<ConcurrencyConfig>();
            auto fmha_config = config_obj.attr("fmha_config").cast<FMHAConfig>();
            auto kv_cache_config = config_obj.attr("kv_cache_config").cast<KVCacheConfig>();
            auto profiling_debug_logging_config = config_obj.attr("profiling_debug_logging_config").cast<ProfilingDebugLoggingConfig>();
            auto hw_kernel_config = config_obj.attr("hw_kernel_config").cast<HWKernelConfig>();
            auto device_resource_config = config_obj.attr("device_resource_config").cast<DeviceResourceConfig>();
            auto moe_config = config_obj.attr("moe_config").cast<MoeConfig>();
            auto model_specific_config = config_obj.attr("model_specific_config").cast<ModelSpecificConfig>();
            auto sp_config = config_obj.attr("sp_config").cast<SpeculativeExecutionConfig>();
            auto cache_store_config = config_obj.attr("cache_store_config").cast<CacheStoreConfig>();
            auto misc_config = config_obj.attr("misc_config").cast<MiscellaneousConfig>();
            auto arpc_config = config_obj.attr("arpc_config").cast<ArpcConfig>();
            auto ffn_disaggregate_config = config_obj.attr("ffn_disaggregate_config").cast<FfnDisAggregateConfig>();
            VitConfig vit_config;
            if (py::hasattr(config_obj, "vit_config")) {
                py::object py_vit_config = config_obj.attr("vit_config");
                if (!py_vit_config.is_none()) {
                    vit_config.vit_separation = py_vit_config.attr("vit_separation").cast<VitSeparation>();
                }
            }
            
            py::object py_layers_weights = sp_model.attr("weight").attr("weights");
            py::object py_global_weights = sp_model.attr("weight").attr("global_weights");
            
            auto convert    = WeightsConverter(false, model_config.quant_algo);
            auto gpt_weight = convert.createGptWeights(py_layers_weights, py_global_weights);
            
            // Get py_eplb if available
            py::object py_eplb = py::none();
            if (py::hasattr(config_obj, "py_eplb")) {
                py_eplb = config_obj.attr("py_eplb");
            }
            
            size_t gen_num_per_cycle = sp_config.gen_num_per_cycle;
            params                             = std::make_unique<ProposeModelEngineInitParams>(
                model_id_, sp_type, gen_num_per_cycle,
                model_config,
                parallelism_config,
                runtime_config,
                pd_sep_config,
                concurrency_config,
                fmha_config,
                kv_cache_config,
                profiling_debug_logging_config,
                hw_kernel_config,
                device_resource_config,
                moe_config,
                model_specific_config,
                sp_config,
                cache_store_config,
                misc_config,
                arpc_config,
                ffn_disaggregate_config,
                vit_config,
                std::move(*gpt_weight));
            model_id_++;
        } else if (sp_type == "mtp") {
            params = prepareMTPEngineInitParams(model_id_, propose_model);
            // Get gen_num_per_cycle from sp_config
            py::object config_obj = propose_model.attr("model").attr("config");
            auto sp_config = config_obj.attr("sp_config").cast<SpeculativeExecutionConfig>();
            size_t gen_num_per_cycle = sp_config.gen_num_per_cycle;
            model_id_ += gen_num_per_cycle;
        } else if (sp_type == "eagle" || sp_type == "eagle3") {
            params = prepareMTPEngineInitParams(model_id_, propose_model);
            model_id_++;
        } else if (sp_type == "deterministic") {
            // Get gen_num_per_cycle from sp_config
            py::object config_obj = propose_model.attr("config");
            auto sp_config = config_obj.attr("sp_config").cast<SpeculativeExecutionConfig>();
            size_t gen_num_per_cycle = sp_config.gen_num_per_cycle;
            params = std::make_unique<ProposeModelEngineInitParams>(sp_type, gen_num_per_cycle);
        } else {
            RTP_LLM_FAIL("sp_type %s not support", sp_type.c_str());
        }
        return params;
    } catch (const std::exception& e) {
        RTP_LLM_FAIL("init propose engine params failed, error msg: %s", e.what());
        return nullptr;
    }
}

void RtpLLMOp::addLora(const std::string& adapter_name, py::object py_lora_a_weights, py::object py_lora_b_weights) {
    auto                         convert        = WeightsConverter(true);
    auto                         lora_a_weights = convert.convertLayerWeights_(py_lora_a_weights);
    auto                         lora_b_weights = convert.convertLayerWeights_(py_lora_b_weights);
    pybind11::gil_scoped_release release;
    model_rpc_service_->addLora(adapter_name, *lora_a_weights, *lora_b_weights);
}

void RtpLLMOp::removeLora(const std::string& adapter_name) {
    pybind11::gil_scoped_release release;
    model_rpc_service_->removeLora(adapter_name);
}

EngineScheduleInfo RtpLLMOp::getEngineScheduleInfo(int64_t latest_finised_version) {
    pybind11::gil_scoped_release release;
    return model_rpc_service_->getEngineScheduleInfo(latest_finised_version);
}

WorkerStatusInfo RtpLLMOp::getWorkerStatusInfo(int64_t latest_finished_version) {
    pybind11::gil_scoped_release release;
    return model_rpc_service_->getWorkerStatusInfo(latest_finished_version);
}

KVCacheInfo RtpLLMOp::getCacheStatusInfo(int64_t latest_cache_version) {
    pybind11::gil_scoped_release release;
    return model_rpc_service_->getCacheStatusInfo(latest_cache_version, true);
}

void RtpLLMOp::initRPCServer(const EngineInitParams                        maga_init_params,
                             py::object                                             mm_process_engine,
                             std::unique_ptr<ProposeModelEngineInitParams> propose_params,
                             py::object                                             token_processor) {
    auto http_port      = maga_init_params.parallelism_config.http_port;
    auto model_rpc_port = maga_init_params.parallelism_config.model_rpc_port;
    auto role_type      = maga_init_params.pd_sep_config.role_type;
    // NOTE: ip/ip段可自定义为所需范围。
    std::string server_address("0.0.0.0:" + std::to_string(model_rpc_port));
    {
        pybind11::gil_scoped_acquire acquire;
        if (role_type == RoleType::PREFILL || role_type == RoleType::DECODE) {
            model_rpc_service_.reset(new RemoteRpcServiceImpl());
        } else {
            model_rpc_service_.reset(new LocalRpcServiceImpl());
        }
        grpc::Status grpc_status =
            model_rpc_service_->init(maga_init_params, std::move(mm_process_engine), std::move(propose_params));
        if (!grpc_status.ok()) {
            RTP_LLM_FAIL("init rpc server failed, error msg: %s", grpc_status.error_message().c_str());
        }

        // NOTE: ip/ip段可自定义为所需范围。
        std::string http_server_address("tcp:0.0.0.0:" + std::to_string(http_port));
        http_server_.reset(new HttpApiServer(model_rpc_service_->getEngine(),
                                                      model_rpc_service_->getMultimodalProcessor(),
                                                      http_server_address,
                                                      maga_init_params,
                                                      token_processor));
        if (model_rpc_port < 0) {
            is_server_ready_ = true;
            return;
        }
    }
    grpc::ServerBuilder builder;
    builder.AddChannelArgument(GRPC_ARG_MAX_CONCURRENT_STREAMS, 100000);
    builder.AddChannelArgument(GRPC_ARG_MAX_METADATA_SIZE, 1024 * 1024 * 1024);
    builder.AddChannelArgument(GRPC_ARG_MAX_CONNECTION_IDLE_MS, 600000);
    builder.AddChannelArgument(GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS, 1000);
    builder.AddChannelArgument(GRPC_ARG_HTTP2_MAX_PING_STRIKES, 1000);
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(model_rpc_service_.get());

    grpc_server_ = builder.BuildAndStart();
    RTP_LLM_CHECK_WITH_INFO(grpc_server_ != nullptr, "grpc server start failed at address " + server_address);

    RTP_LLM_LOG_INFO("Server listening on %s", server_address.c_str());
    is_server_ready_ = true;
    grpc_server_->Wait();
    RTP_LLM_LOG_INFO("Server exit on %s", server_address.c_str());
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
    pybind11::gil_scoped_release release;
    model_rpc_service_->getEngine()->getScheduler().updateSchedulerInfo(scheduler_info);
}

bool RtpLLMOp::updateEplbConfig(const EPLBConfig& config) {
    if (model_rpc_service_) {
        pybind11::gil_scoped_release release;
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
                RTP_LLM_LOG_INFO("rpc service has [%lu] onflight request, waiting 1s", onflight_request);
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
        stopKmonitorFactory();
    }
}

RtpLLMOp::~RtpLLMOp() {
    stop();
}

void RtpLLMOp::pause() {
    auto engine = model_rpc_service_->getEngine();
    engine->pause();
}

void RtpLLMOp::restart() {
    auto engine = model_rpc_service_->getEngine();
    engine->restart();
}

void registerRtpLLMOp(const py::module& m) {
    pybind11::class_<RtpLLMOp>(m, "RtpLLMOp")
        .def(pybind11::init<>())
        .def("init",
             &RtpLLMOp::init,
             py::arg("model"),
             py::arg("mm_process_engine"),
             py::arg("propose_model"),
             py::arg("token_processor"))
        .def("start_http_server",
             &RtpLLMOp::startHttpServer,
             py::arg("model_weights_loader"),
             py::arg("lora_infos"),
             py::arg("gang_info"),
             py::arg("tokenizer"),
             py::arg("render"))
        .def("add_lora",
             &RtpLLMOp::addLora,
             py::arg("adapter_name"),
             py::arg("lora_a_weights"),
             py::arg("lora_b_weights"))
        .def("remove_lora", &RtpLLMOp::removeLora, py::arg("adapter_name"))
        .def("get_engine_schedule_info", &RtpLLMOp::getEngineScheduleInfo)
        .def("get_worker_status_info", &RtpLLMOp::getWorkerStatusInfo, py::arg("latest_finished_version"))
        .def("get_cache_status_info", &RtpLLMOp::getCacheStatusInfo, py::arg("latest_cache_version"))
        .def("update_scheduler_info", &RtpLLMOp::updateSchedulerInfo, py::arg("scheduler_info"))
        .def("stop", &RtpLLMOp::stop)
        .def("update_eplb_config", &RtpLLMOp::updateEplbConfig, py::arg("config"))
        .def("pause", &RtpLLMOp::pause)
        .def("restart", &RtpLLMOp::restart);
}

}

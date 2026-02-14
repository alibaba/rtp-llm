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
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/models/models_weight/W.h"

using namespace std;
namespace th = torch;

namespace rtp_llm {

std::unique_ptr<ProposeModelEngineInitParams>
prepareMTPEngineInitParams(size_t model_id, py::object propose_model, const EngineInitParams& base_params) {
    auto            sp_model = propose_model.attr("model");
    SpeculativeType sp_type  = propose_model.attr("sp_type").cast<SpeculativeType>();
    RTP_LLM_CHECK(sp_type == SP_TYPE_MTP || sp_type == SP_TYPE_EAGLE3 || sp_type == SP_TYPE_EAGLE);

    std::unique_ptr<std::vector<std::unique_ptr<EngineInitParams>>> mtp_params =
        std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();

    // Get model_config from model (only difference between propose and score models)
    auto model_config = sp_model.attr("model_config").cast<ModelConfig>();

    py::object py_layers_weights     = sp_model.attr("weight").attr("weights");
    py::object py_global_weights     = sp_model.attr("weight").attr("global_weights");
    auto       convert               = WeightsConverter(false, model_config.quant_algo);
    auto       py_layers_weights_vec = convertPyObjectToVec(py_layers_weights);
    size_t     model_num             = py_layers_weights_vec.size();
    size_t     gen_num_per_cycle     = base_params.sp_config.gen_num_per_cycle;
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
    if (sp_type == SP_TYPE_EAGLE || sp_type == SP_TYPE_EAGLE3) {
        model_num = 1;
    }

    // Get py_eplb if available (from model)
    py::object py_eplb = py::none();
    if (py::hasattr(sp_model, "py_eplb")) {
        py_eplb = sp_model.attr("py_eplb");
    }

    // Create a temporary ModelConfig with num_layers = 1 for MTP
    ModelConfig temp_model_config = model_config;
    temp_model_config.num_layers  = 1;

    for (int i = 0; i < model_num; i++) {
        auto     layer_weigths = py_layers_weights_vec[i];
        py::list tmp;
        tmp.append(layer_weigths);
        auto gpt_weight = convert.createGptWeights(tmp, py_global_weights);
        mtp_params->push_back(std::move(std::make_unique<EngineInitParams>(model_id,
                                                                           temp_model_config,
                                                                           base_params.parallelism_config,
                                                                           base_params.runtime_config,
                                                                           base_params.pd_sep_config,
                                                                           base_params.concurrency_config,
                                                                           base_params.fmha_config,
                                                                           base_params.kv_cache_config,
                                                                           base_params.profiling_debug_logging_config,
                                                                           base_params.hw_kernel_config,
                                                                           base_params.device_resource_config,
                                                                           base_params.moe_config,
                                                                           base_params.model_specific_config,
                                                                           base_params.sp_config,
                                                                           base_params.cache_store_config,
                                                                           base_params.misc_config,
                                                                           base_params.arpc_config,
                                                                           base_params.grpc_config,
                                                                           base_params.ffn_disaggregate_config,
                                                                           base_params.vit_config,
                                                                           std::move(*gpt_weight),
                                                                           py::none(),
                                                                           py_eplb)));
        model_id++;
    }

    return std::move(std::make_unique<ProposeModelEngineInitParams>(sp_type, gen_num_per_cycle, std::move(mtp_params)));
};

RtpLLMOp::RtpLLMOp() {}

void RtpLLMOp::init(py::object model,
                    py::object engine_config,
                    py::object vit_config,
                    py::object mm_process_engine,
                    py::object propose_model,
                    py::object token_processor) {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    EngineInitParams params = initModel(model, engine_config, vit_config);

    if (!propose_model.is_none()) {
        if (!propose_model.attr("model").is_none()) {
            params.py_sp_model = propose_model.attr("model").attr("py_model");
        }
    }

    RTP_LLM_LOG_INFO("init engine params success");

    params.showDebugInfo();
    std::unique_ptr<ProposeModelEngineInitParams> propose_params = initProposeModel(propose_model, params);
    pybind11::gil_scoped_release                  release;
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

EngineInitParams RtpLLMOp::initModel(py::object model, py::object engine_config, py::object vit_config) {
    try {
        // Get model_config from model
        auto model_config = model.attr("model_config").cast<ModelConfig>();

        // Extract individual config members from engine_config
        auto parallelism_config = engine_config.attr("parallelism_config").cast<ParallelismConfig>();
        auto runtime_config     = engine_config.attr("runtime_config").cast<RuntimeConfig>();
        auto pd_sep_config      = engine_config.attr("pd_sep_config").cast<PDSepConfig>();
        auto concurrency_config = engine_config.attr("concurrency_config").cast<ConcurrencyConfig>();
        auto fmha_config        = engine_config.attr("fmha_config").cast<FMHAConfig>();
        auto kv_cache_config    = engine_config.attr("kv_cache_config").cast<KVCacheConfig>();
        auto profiling_debug_logging_config =
            engine_config.attr("profiling_debug_logging_config").cast<ProfilingDebugLoggingConfig>();
        auto hw_kernel_config       = engine_config.attr("hw_kernel_config").cast<HWKernelConfig>();
        auto device_resource_config = engine_config.attr("device_resource_config").cast<DeviceResourceConfig>();
        auto moe_config             = engine_config.attr("moe_config").cast<MoeConfig>();
        auto model_specific_config  = engine_config.attr("model_specific_config").cast<ModelSpecificConfig>();
        auto sp_config              = engine_config.attr("sp_config").cast<SpeculativeExecutionConfig>();
        auto cache_store_config     = engine_config.attr("cache_store_config").cast<CacheStoreConfig>();
        auto misc_config            = engine_config.attr("misc_config").cast<MiscellaneousConfig>();
        auto arpc_config            = engine_config.attr("arpc_config").cast<ArpcConfig>();
        auto grpc_config            = engine_config.attr("grpc_config").cast<GrpcConfig>();

        // Extract vit_config
        VitConfig vit_config_cpp;
        if (!vit_config.is_none()) {
            vit_config_cpp.vit_separation = vit_config.attr("vit_separation").cast<VitSeparation>();
        }

        py::object py_layers_weights = model.attr("weight").attr("weights");
        py::object py_global_weights = model.attr("weight").attr("global_weights");

        auto convert    = WeightsConverter(false, model_config.quant_algo);
        auto gpt_weight = convert.createGptWeights(py_layers_weights, py_global_weights);

        auto py_model       = model.attr("py_model");
        auto weight_manager = model.attr("weight_manager");
        // TODO(wangyin.yx): Only one of `py_model` and `gpt_weight` is actually needed.

        // Get py_eplb if available (from model)
        py::object py_eplb = py::none();
        if (py::hasattr(model, "py_eplb")) {
            py_eplb = model.attr("py_eplb");
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
                                grpc_config,
                                parallelism_config.ffn_disaggregate_config,
                                vit_config_cpp,
                                std::move(*gpt_weight),
                                py_model,
                                weight_manager,
                                py_eplb);
        params.nccl_comm_config = engine_config.attr("nccl_comm_config").cast<NcclCommConfig>();
        params.server_config    = engine_config.attr("server_config");
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

std::unique_ptr<ProposeModelEngineInitParams> RtpLLMOp::initProposeModel(py::object              propose_model,
                                                                         const EngineInitParams& base_params) {
    try {
        if (propose_model.is_none()) {
            return nullptr;
        }
        std::unique_ptr<ProposeModelEngineInitParams> params  = nullptr;
        SpeculativeType                               sp_type = propose_model.attr("sp_type").cast<SpeculativeType>();
        if (sp_type == SP_TYPE_VANILLA) {
            py::object sp_model = propose_model.attr("model");
            // Get model_config from model (only difference between propose and score models)
            auto model_config = sp_model.attr("model_config").cast<ModelConfig>();

            py::object py_layers_weights = sp_model.attr("weight").attr("weights");
            py::object py_global_weights = sp_model.attr("weight").attr("global_weights");

            auto convert    = WeightsConverter(false, model_config.quant_algo);
            auto gpt_weight = convert.createGptWeights(py_layers_weights, py_global_weights);

            // Get py_eplb if available (from model)
            py::object py_eplb = py::none();
            if (py::hasattr(sp_model, "py_eplb")) {
                py_eplb = sp_model.attr("py_eplb");
            }

            size_t gen_num_per_cycle = base_params.sp_config.gen_num_per_cycle;
            params                   = std::make_unique<ProposeModelEngineInitParams>(model_id_,
                                                                    sp_type,
                                                                    gen_num_per_cycle,
                                                                    model_config,
                                                                    base_params,
                                                                    std::move(*gpt_weight),
                                                                    py::none(),
                                                                    py_eplb);
            model_id_++;
        } else if (sp_type == SP_TYPE_MTP || sp_type == SP_TYPE_EAGLE || sp_type == SP_TYPE_EAGLE3) {
            params = prepareMTPEngineInitParams(model_id_, propose_model, base_params);
            if (sp_type == SP_TYPE_MTP) {
                size_t gen_num_per_cycle = base_params.sp_config.gen_num_per_cycle;
                model_id_ += gen_num_per_cycle;
            } else {
                model_id_++;
            }
        } else if (sp_type == SP_TYPE_DETERMINISTIC) {
            // Get gen_num_per_cycle directly from propose_model.gen_num_per_circle
            size_t gen_num_per_cycle = propose_model.attr("gen_num_per_circle").cast<size_t>();
            params                   = std::make_unique<ProposeModelEngineInitParams>(sp_type, gen_num_per_cycle);
        } else {
            RTP_LLM_FAIL("sp_type %s not support", SpeculativeExecutionConfig::to_string(sp_type).c_str());
        }
        return params;
    } catch (const std::exception& e) {
        RTP_LLM_FAIL("init propose engine params failed, error msg: %s", e.what());
        return nullptr;
    }
}

void RtpLLMOp::initRPCServer(const EngineInitParams                        maga_init_params,
                             py::object                                    mm_process_engine,
                             std::unique_ptr<ProposeModelEngineInitParams> propose_params,
                             py::object                                    token_processor) {
    std::string server_address;
    {
        pybind11::gil_scoped_acquire acquire;
        int64_t                      http_port = maga_init_params.server_config.attr("http_port").cast<int64_t>();
        int64_t model_rpc_port                 = maga_init_params.server_config.attr("rpc_server_port").cast<int64_t>();
        auto    role_type                      = maga_init_params.pd_sep_config.role_type;
        // NOTE: ip/ip段可自定义为所需范围。
        server_address = "0.0.0.0:" + std::to_string(model_rpc_port);
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
    const GrpcConfig&   grpc_config   = maga_init_params.grpc_config;
    auto                server_config = grpc_config.get_server_config();
    for (auto it = server_config.begin(); it != server_config.end(); ++it) {
        RTP_LLM_LOG_INFO("grpc server add channel argument %s: %d", it->first.c_str(), it->second);
        builder.AddChannelArgument(it->first, it->second);
    }
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
                               py::object world_info,
                               py::object tokenizer,
                               py::object render) {
    if (http_server_ == nullptr) {
        RTP_LLM_FAIL("normal HTTP Server nullptr error.");
        return;
    }
    if (http_server_->start(model_weights_loader, lora_infos, world_info, tokenizer, render)) {
        RTP_LLM_LOG_INFO("normal HTTP Server listening on %s", http_server_->getListenAddr().c_str());
    } else {
        RTP_LLM_FAIL("normal HTTP Server start fail.");
    }
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
        if (model_rpc_service_) {
            pybind11::gil_scoped_release release;
            model_rpc_service_->stop();
            pybind11::gil_scoped_acquire acquire;
            model_rpc_service_.reset();
        }
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
             py::arg("engine_config"),
             py::arg("vit_config"),
             py::arg("mm_process_engine"),
             py::arg("propose_model"),
             py::arg("token_processor"))
        .def("start_http_server",
             &RtpLLMOp::startHttpServer,
             py::arg("model_weights_loader"),
             py::arg("lora_infos"),
             py::arg("world_info"),
             py::arg("tokenizer"),
             py::arg("render"))
        .def("stop", &RtpLLMOp::stop);
}

}  // namespace rtp_llm

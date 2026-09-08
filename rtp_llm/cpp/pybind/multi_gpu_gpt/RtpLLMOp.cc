#include <cstddef>
#include <memory>
#include <tuple>
#include "autil/EnvUtil.h"
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
#include "rtp_llm/cpp/models/models_weight/W.h"

using namespace std;
namespace th = torch;

namespace rtp_llm {

namespace {

int64_t getGrpcStopTimeoutMs() {
    constexpr int64_t kDefaultStopTimeoutMs = 600 * 1000;
    const auto        explicit_timeout_ms   = autil::EnvUtil::getEnv("RTP_LLM_STOP_TIMEOUT_MS", int64_t(0));
    if (explicit_timeout_ms > 0) {
        return explicit_timeout_ms;
    }
    const auto shutdown_timeout_s = autil::EnvUtil::getEnv("SHUTDOWN_TIMEOUT", int64_t(600));
    if (shutdown_timeout_s <= 0) {
        return kDefaultStopTimeoutMs;
    }
    return shutdown_timeout_s * 1000;
}

}  // namespace

std::unique_ptr<ProposeModelEngineInitParams>
prepareMTPEngineInitParams(size_t model_id, py::object propose_model, const EngineInitParams& base_params) {
    auto            sp_model = propose_model.attr("model");
    SpeculativeType sp_type  = propose_model.attr("sp_type").cast<SpeculativeType>();
    RTP_LLM_CHECK(sp_type == SP_TYPE_MTP || sp_type == SP_TYPE_EAGLE3 || sp_type == SP_TYPE_EAGLE
                  || sp_type == SP_TYPE_DSPARK);

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

    // Get py_eplb if available (from model)
        py::object py_eplb    = py::none();
        if (py::hasattr(sp_model, "py_eplb")) {
            py_eplb = sp_model.attr("py_eplb");
        }

    auto make_engine_params = [&](size_t id, const ModelConfig& cfg, auto gpt_weight) {
        return std::make_unique<EngineInitParams>(id,
                                                  cfg,
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
                                                  py_eplb);
    };

    if (sp_type == SP_TYPE_DSPARK) {
        // DSpARK is one multi-layer draft model. gamma controls its output
        // width, not the number of one-layer MTP model instances.
        mtp_params->push_back(
            make_engine_params(model_id, model_config, convert.createGptWeights(py_layers_weights, py_global_weights)));
        return std::make_unique<ProposeModelEngineInitParams>(sp_type, gen_num_per_cycle, std::move(mtp_params));
    }
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

    // Create a temporary ModelConfig with num_layers = 1 for MTP
    ModelConfig temp_model_config = model_config;
    temp_model_config.num_layers  = 1;

    for (int i = 0; i < model_num; i++) {
        auto     layer_weigths = py_layers_weights_vec[i];
        py::list tmp;
        tmp.append(layer_weigths);
        mtp_params->push_back(
            make_engine_params(model_id, temp_model_config, convert.createGptWeights(tmp, py_global_weights)));
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
                    py::object token_processor,
                    bool      defer_service_start) {
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
    if (defer_service_start) {
        // Build the model/KV/executor state before the SCR arrival.  Only the
        // network listeners are deferred, so the template contains the static
        // runtime state needed by a restored worker without accepting traffic.
        deferred_init_params_       = std::make_unique<EngineInitParams>(std::move(params));
        deferred_mm_process_engine_ = std::move(mm_process_engine);
        deferred_propose_params_    = std::move(propose_params);
        deferred_token_processor_   = std::move(token_processor);
        prepareRPCService(*deferred_init_params_,
                          std::move(deferred_mm_process_engine_),
                          std::move(deferred_propose_params_),
                          std::move(deferred_token_processor_),
                          true);
        rpc_server_deferred_        = true;
        RTP_LLM_LOG_INFO("RPC/HTTP listener start deferred until pre-service checkpoint barrier; engine state prepared");
        return;
    }
    pybind11::gil_scoped_release release;
    server_start_failed_ = false;
    stop_requested_      = false;
    deferred_init_params_ = std::make_unique<EngineInitParams>(std::move(params));
    grpc_server_thread_   = std::thread([this,
                                       mm_process_engine = std::move(mm_process_engine),
                                       propose_params = std::move(propose_params),
                                       token_processor = std::move(token_processor)]() mutable {
        try {
            initRPCServer(*deferred_init_params_,
                          std::move(mm_process_engine),
                          std::move(propose_params),
                          std::move(token_processor));
        } catch (const std::exception& e) {
            setServerStartError(e.what());
        } catch (...) {
            setServerStartError("unknown exception while starting RPC server");
        }
        // The listener thread does not otherwise hold the Python GIL.  Make
        // moved-from pybind objects explicitly empty while holding it so their
        // final decref cannot run on a native thread without Python state.
        pybind11::gil_scoped_acquire acquire;
        mm_process_engine = py::object();
        propose_params.reset();
        token_processor = py::object();
    });
    while (!is_server_ready_) {
        if (server_start_failed_) {
            std::lock_guard<std::mutex> lock(server_state_mutex_);
            RTP_LLM_FAIL("RPC server start failed: %s", server_start_error_.c_str());
        }
        sleep(1);  // wait 1s for server ready
    }
}

void RtpLLMOp::startRPCServer() {
    if (!rpc_server_deferred_) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(deferred_init_params_ != nullptr, "deferred RPC init params are missing");
    rpc_server_deferred_ = false;

    {
        pybind11::gil_scoped_release release;
        server_start_failed_ = false;
        stop_requested_      = false;
        grpc_server_thread_ = std::thread([this]() {
            try {
                startRPCServerInternal(*deferred_init_params_);
            } catch (const std::exception& e) {
                setServerStartError(e.what());
            } catch (...) {
                setServerStartError("unknown exception while starting RPC server");
            }
        });
        while (!is_server_ready_) {
            if (server_start_failed_) {
                std::lock_guard<std::mutex> lock(server_state_mutex_);
                RTP_LLM_FAIL("deferred RPC server start failed: %s", server_start_error_.c_str());
            }
            sleep(1);
        }
    }
}

void RtpLLMOp::updateRuntimeEndpoints(py::object runtime_config) {
    RTP_LLM_CHECK_WITH_INFO(rpc_server_deferred_ && deferred_init_params_ != nullptr,
                            "runtime endpoints can only be updated before deferred RPC start");
    auto config = runtime_config.cast<RuntimeConfig>();
    deferred_init_params_->runtime_config.worker_addrs = config.worker_addrs;
    deferred_init_params_->runtime_config.worker_grpc_addrs = config.worker_grpc_addrs;
    if (model_rpc_service_) {
        model_rpc_service_->updateRuntimeEndpoints(config);
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
        auto grammar_config         = engine_config.attr("grammar_config").cast<GrammarConfig>();

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
        params.grammar_config   = grammar_config;
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
        } else if (sp_type == SP_TYPE_MTP || sp_type == SP_TYPE_EAGLE || sp_type == SP_TYPE_EAGLE3
                   || sp_type == SP_TYPE_DSPARK) {
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

void RtpLLMOp::initRPCServer(const EngineInitParams&                       maga_init_params,
                             py::object                                    mm_process_engine,
                             std::unique_ptr<ProposeModelEngineInitParams> propose_params,
                             py::object                                    token_processor) {
    {
        pybind11::gil_scoped_acquire acquire;
        prepareRPCService(maga_init_params,
                          std::move(mm_process_engine),
                          std::move(propose_params),
                          std::move(token_processor),
                          false);
        // Keep all Python-owned temporaries empty before releasing the GIL;
        // the HttpApiServer/MultimodalProcessor retain their own references.
        mm_process_engine = py::object();
        propose_params.reset();
        token_processor = py::object();
    }
    startRPCServerInternal(maga_init_params);
}

void RtpLLMOp::prepareRPCService(const EngineInitParams&                       maga_init_params,
                                 py::object                                    mm_process_engine,
                                 std::unique_ptr<ProposeModelEngineInitParams> propose_params,
                                 py::object                                    token_processor,
                                 bool                                           defer_network_services) {
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
        model_rpc_service_->setDeferServiceStart(defer_network_services);
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
        deferred_server_address_ = server_address;
    }
}

void RtpLLMOp::startRPCServerInternal(const EngineInitParams& maga_init_params) {
    int64_t model_rpc_port = -1;
    {
        pybind11::gil_scoped_acquire acquire;
        model_rpc_port = maga_init_params.server_config.attr("rpc_server_port").cast<int64_t>();
    }
    // Remote PD cache-store listeners are also deferred in template mode,
    // independently of whether this process exposes the model RPC port.
    model_rpc_service_->startDeferredServices();
    if (model_rpc_port < 0) {
        is_server_ready_ = true;
        return;
    }
    grpc::ServerBuilder builder;
    // Set large message limits as C++-level defaults (overridable via server_config from grpc_group_args.py)
    builder.AddChannelArgument(GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH, 1024 * 1024 * 1024);
    builder.AddChannelArgument(GRPC_ARG_MAX_SEND_MESSAGE_LENGTH, 1024 * 1024 * 1024);
    const GrpcConfig& grpc_config   = maga_init_params.grpc_config;
    auto              server_config = grpc_config.get_server_config();
    for (auto it = server_config.begin(); it != server_config.end(); ++it) {
        RTP_LLM_LOG_INFO("grpc server add channel argument %s: %d", it->first.c_str(), it->second);
        builder.AddChannelArgument(it->first, it->second);
    }
    if (grpc_config.max_server_pollers > 0) {
        builder.SetSyncServerOption(grpc::ServerBuilder::MAX_POLLERS, grpc_config.max_server_pollers);
        RTP_LLM_LOG_INFO("grpc sync server MAX_POLLERS: %d", grpc_config.max_server_pollers);
    }
    builder.AddListeningPort(deferred_server_address_, grpc::InsecureServerCredentials());
    builder.RegisterService(model_rpc_service_.get());

    {
        std::lock_guard<std::mutex> lock(server_state_mutex_);
        grpc_server_ = builder.BuildAndStart();
    }
    grpc::Server* grpc_server = nullptr;
    {
        std::lock_guard<std::mutex> lock(server_state_mutex_);
        grpc_server = grpc_server_.get();
    }
    RTP_LLM_CHECK_WITH_INFO(grpc_server != nullptr,
                            "grpc server start failed at address " + deferred_server_address_);
    RTP_LLM_LOG_INFO("Server listening on %s", deferred_server_address_.c_str());
    is_server_ready_ = true;
    // stop() may race with BuildAndStart.  Check the shutdown intent after
    // publishing the server pointer so the joining thread cannot wait forever
    // on a listener that the stopper observed as not-yet-created.
    if (stop_requested_) {
        grpc_server->Shutdown();
    }
    grpc_server->Wait();
    RTP_LLM_LOG_INFO("Server exit on %s", deferred_server_address_.c_str());
}

void RtpLLMOp::setServerStartError(const std::string& error) {
    {
        std::lock_guard<std::mutex> lock(server_state_mutex_);
        server_start_error_ = error;
    }
    server_start_failed_ = true;
}

void RtpLLMOp::startHttpServer(py::object model_weights_loader,
                               py::object world_info,
                               py::object tokenizer,
                               py::object render) {
    if (http_server_ == nullptr) {
        RTP_LLM_FAIL("normal HTTP Server nullptr error.");
        return;
    }
    if (http_server_->start(model_weights_loader, world_info, tokenizer, render)) {
        RTP_LLM_LOG_INFO("normal HTTP Server listening on %s", http_server_->getListenAddr().c_str());
    } else {
        RTP_LLM_FAIL("normal HTTP Server start fail.");
    }
}

void RtpLLMOp::stop() {
    const int64_t stop_timeout_ms = getGrpcStopTimeoutMs();
    if (!is_server_shutdown_) {
        stop_requested_             = true;
        rpc_server_deferred_        = false;
        if (model_rpc_service_) {
            model_rpc_service_->beginShutdown();
        }
        grpc::Server* grpc_server = nullptr;
        {
            std::lock_guard<std::mutex> lock(server_state_mutex_);
            grpc_server = grpc_server_.get();
        }
        if (grpc_server) {
            auto begin_wait_us = autil::TimeUtility::currentTimeInMicroSeconds();
            while (auto onflight_request = model_rpc_service_->onflightRequestNum()) {
                RTP_LLM_LOG_INFO("rpc service has [%lu] onflight request, waiting 1s, stop_timeout_ms=%ld",
                                 onflight_request,
                                 stop_timeout_ms);
                sleep(1);
                if (autil::TimeUtility::currentTimeInMicroSeconds() - begin_wait_us > stop_timeout_ms * 1000) {
                    RTP_LLM_LOG_INFO("rpc service wait timeout, no more waiting");
                    break;
                }
            }
            RTP_LLM_LOG_INFO("Server shutdowning");
            grpc_server->Shutdown();
        }
        // The listener thread may still be constructing the server or waiting
        // in Server::Wait.  Join before releasing the service/params it uses;
        // this avoids the detached-thread lifetime race during CRIU restore
        // and shutdown.
        if (grpc_server_thread_.joinable()) {
            grpc_server_thread_.join();
        }
        {
            std::lock_guard<std::mutex> lock(server_state_mutex_);
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
        deferred_init_params_.reset();
        deferred_propose_params_.reset();
        deferred_mm_process_engine_ = py::object();
        deferred_token_processor_   = py::object();
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
             py::arg("token_processor"),
             py::arg("defer_service_start") = false)
        .def("start_rpc_server", &RtpLLMOp::startRPCServer)
        .def("update_runtime_endpoints", &RtpLLMOp::updateRuntimeEndpoints)
        .def("start_http_server",
             &RtpLLMOp::startHttpServer,
             py::arg("model_weights_loader"),
             py::arg("world_info"),
             py::arg("tokenizer"),
             py::arg("render"))
        .def("stop", &RtpLLMOp::stop);
}

}  // namespace rtp_llm

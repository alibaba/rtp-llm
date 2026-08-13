#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
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
#include "rtp_llm/cpp/models/models_weight/W.h"
#include "rtp_llm/cpp/utils/GilScopedThreadArgs.h"

using namespace std;
namespace th = torch;

namespace rtp_llm {

namespace {

std::chrono::milliseconds remainingShutdownGrace(std::chrono::steady_clock::time_point deadline) {
    if (deadline == std::chrono::steady_clock::time_point::max()) {
        return std::chrono::milliseconds::max();
    }
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
        return std::chrono::milliseconds::zero();
    }
    return std::max(std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now),
                    std::chrono::milliseconds(1));
}

class PromiseCompletion {
public:
    explicit PromiseCompletion(std::shared_ptr<std::promise<void>> signal): signal_(std::move(signal)) {}

    ~PromiseCompletion() noexcept {
        try {
            signal_->set_value();
        } catch (...) {
            RTP_LLM_LOG_ERROR("failed to publish RPC server thread completion");
        }
    }

    PromiseCompletion(const PromiseCompletion&)            = delete;
    PromiseCompletion& operator=(const PromiseCompletion&) = delete;

private:
    std::shared_ptr<std::promise<void>> signal_;
};

bool waitForFuture(const std::shared_future<void>& result, std::chrono::steady_clock::time_point deadline) {
    if (!result.valid()) {
        return true;
    }
    if (deadline == std::chrono::steady_clock::time_point::max()) {
        result.wait();
        return true;
    }
    return result.wait_until(deadline) == std::future_status::ready;
}

}  // namespace

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
    RTP_LLM_CHECK_WITH_INFO(base_params.sp_config.gen_num_per_cycle > 0,
                            "speculative proposal steps must be positive, got %ld",
                            base_params.sp_config.gen_num_per_cycle);
    const size_t gen_num_per_cycle = static_cast<size_t>(base_params.sp_config.gen_num_per_cycle);
    RTP_LLM_CHECK_WITH_INFO(!py_layers_weights_vec.empty(), "draft model weights must contain at least one layer");

    // Get py_eplb if available (from model)
    py::object py_eplb = py::none();
    if (py::hasattr(sp_model, "py_eplb")) {
        py_eplb = sp_model.attr("py_eplb");
    }

    ModelConfig draft_model_config  = model_config;
    py::object  draft_layer_weights = py_layers_weights;
    if (sp_type == SP_TYPE_MTP) {
        RTP_LLM_CHECK_WITH_INFO(model_config.num_layers > 0,
                                "MTP model must contain at least one layer, got %ld",
                                model_config.num_layers);
        RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(model_config.num_layers) == py_layers_weights_vec.size(),
                                "MTP model layer count mismatch: config=%ld, weights=%zu",
                                model_config.num_layers,
                                py_layers_weights_vec.size());
    } else {
        draft_model_config.num_layers = 1;
        py::list first_layer_weights;
        first_layer_weights.append(py_layers_weights_vec.front());
        draft_layer_weights = std::move(first_layer_weights);
    }

    auto gpt_weight = convert.createGptWeights(draft_layer_weights, py_global_weights);
    mtp_params->push_back(std::make_unique<EngineInitParams>(model_id,
                                                             draft_model_config,
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
                                                             py_eplb));

    return std::make_unique<ProposeModelEngineInitParams>(sp_type, gen_num_per_cycle, std::move(mtp_params));
};

RtpLLMOp::RtpLLMOp() {}

struct RtpLLMOp::RpcServerThreadArgs {
    RpcServerThreadArgs(EngineInitParams                              maga_init_params,
                        std::unique_ptr<ProposeModelEngineInitParams> propose_params,
                        py::object                                    token_processor,
                        py::object                                    mm_process_engine):
        maga_init_params(std::move(maga_init_params)),
        propose_params(std::move(propose_params)),
        token_processor(std::move(token_processor)),
        mm_process_engine(std::move(mm_process_engine)) {}

    EngineInitParams                              maga_init_params;
    std::unique_ptr<ProposeModelEngineInitParams> propose_params;
    py::object                                    token_processor;
    py::object                                    mm_process_engine;
};

void RtpLLMOp::init(py::object model,
                    py::object engine_config,
                    py::object vit_config,
                    py::object propose_model,
                    py::object token_processor,
                    py::object mm_process_engine) {
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
    auto server_args = std::make_shared<RpcServerThreadArgs>(std::move(params),
                                                              std::move(propose_params),
                                                              std::move(token_processor),
                                                              std::move(mm_process_engine));
    auto startup_signal = std::make_shared<std::promise<void>>();
    auto startup_result = startup_signal->get_future();
    auto exit_signal    = std::make_shared<std::promise<void>>();
    grpc_server_exit_result_ = exit_signal->get_future().share();

    initializeGilThreadStateTracking();

    // Construct the thread while this Python entry point still holds the GIL. The worker may block acquiring it,
    // but std::thread construction itself does not wait for the worker. If construction throws, every argument
    // owner therefore unwinds on this GIL-holding thread.
    try {
        grpc_server_thread_ =
            std::thread(&RtpLLMOp::initRPCServer, this, server_args, startup_signal, exit_signal);
    } catch (...) {
        grpc_server_exit_result_ = {};
        throw;
    }

    std::exception_ptr startup_error;
    {
        GilScopedRelease release;
        try {
            startup_result.get();
        } catch (...) {
            startup_error = std::current_exception();
        }
        if (startup_error && grpc_server_thread_.joinable()) {
            grpc_server_exit_result_.wait();
            grpc_server_thread_.join();
            grpc_server_exit_result_ = {};
        }
        if (startup_error) {
            forceStopNoThrow();
        }
    }
    if (startup_error) {
        std::rethrow_exception(startup_error);
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
            vit_config_cpp.vit_separation = static_cast<VitSeparation>(vit_config.attr("vit_separation").cast<int>());
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
            model_id_++;
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

void RtpLLMOp::initRPCServer(std::shared_ptr<RpcServerThreadArgs> args,
                             std::shared_ptr<std::promise<void>>  startup_signal,
                             std::shared_ptr<std::promise<void>>  exit_signal) {
    PromiseCompletion exit_completion(std::move(exit_signal));
    std::string server_address;
    GrpcConfig  grpc_config;
    bool        grpc_disabled = false;

    std::optional<std::string> python_setup_error;
    try {
        GilScopedThreadArgs<RpcServerThreadArgs> scoped_thread_args(std::move(args));
        auto& thread_args = scoped_thread_args.get();
        int64_t http_port = thread_args.maga_init_params.server_config.attr("http_port").cast<int64_t>();
        int64_t model_rpc_port =
            thread_args.maga_init_params.server_config.attr("rpc_server_port").cast<int64_t>();
        auto role_type = thread_args.maga_init_params.pd_sep_config.role_type;
        // NOTE: ip/ip段可自定义为所需范围。
        server_address = "0.0.0.0:" + std::to_string(model_rpc_port);
        grpc_config    = thread_args.maga_init_params.grpc_config;
        grpc_disabled  = model_rpc_port < 0;

        std::unique_ptr<RpcServiceImpl> model_rpc_service;
        if (role_type == RoleType::PREFILL || role_type == RoleType::DECODE) {
            model_rpc_service.reset(new RemoteRpcServiceImpl());
        } else {
            model_rpc_service.reset(new LocalRpcServiceImpl());
        }
        grpc::Status grpc_status = model_rpc_service->init(thread_args.maga_init_params,
                                                           std::move(thread_args.propose_params),
                                                           thread_args.mm_process_engine);
        if (!grpc_status.ok()) {
            RTP_LLM_FAIL("init rpc server failed, error msg: %s", grpc_status.error_message().c_str());
        }

        // NOTE: ip/ip段可自定义为所需范围。
        std::string http_server_address("tcp:0.0.0.0:" + std::to_string(http_port));
        auto http_server = std::make_unique<HttpApiServer>(model_rpc_service->getEngine(),
                                                           model_rpc_service->getMultimodalProcessor(),
                                                           http_server_address,
                                                           thread_args.maga_init_params,
                                                           thread_args.token_processor);
        model_rpc_service_ = std::move(model_rpc_service);
        http_server_       = std::move(http_server);
    } catch (const std::exception& e) {
        python_setup_error = e.what();
    } catch (...) {
        python_setup_error = "unknown RPC server initialization failure";
    }

    if (python_setup_error) {
        startup_signal->set_exception(std::make_exception_ptr(std::runtime_error(*python_setup_error)));
        return;
    }
    if (grpc_disabled) {
        is_server_ready_.store(true, std::memory_order_release);
        startup_signal->set_value();
        return;
    }

    try {
        grpc::ServerBuilder builder;
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
        is_server_ready_.store(true, std::memory_order_release);
        startup_signal->set_value();
    } catch (const std::exception& e) {
        startup_signal->set_exception(std::make_exception_ptr(std::runtime_error(e.what())));
        return;
    } catch (...) {
        startup_signal->set_exception(
            std::make_exception_ptr(std::runtime_error("unknown RPC transport initialization failure")));
        return;
    }

    try {
        grpc_server_->Wait();
        RTP_LLM_LOG_INFO("Server exit on %s", server_address.c_str());
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("RPC server wait failed: %s", e.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("RPC server wait failed with an unknown error");
    }
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

void RtpLLMOp::startHttpTransportStop() {
    if (http_server_ == nullptr || http_stop_result_.valid()) {
        return;
    }

    auto stop_signal = std::make_shared<std::promise<void>>();
    http_stop_result_ = stop_signal->get_future().share();
    auto* http_server = http_server_.get();
    try {
        http_stop_thread_ = std::thread([http_server, stop_signal]() {
            try {
                http_server->forceStopTransport();
                stop_signal->set_value();
            } catch (...) {
                try {
                    stop_signal->set_exception(std::current_exception());
                } catch (...) {
                    RTP_LLM_LOG_ERROR("failed to publish HTTP transport stop failure");
                }
            }
        });
    } catch (...) {
        http_stop_result_ = {};
        throw;
    }
}

bool RtpLLMOp::waitForHttpTransportStop(std::chrono::steady_clock::time_point deadline) {
    if (!http_stop_result_.valid()) {
        return true;
    }
    if (!waitForFuture(http_stop_result_, deadline)) {
        return false;
    }
    RTP_LLM_CHECK_WITH_INFO(!http_stop_thread_.joinable()
                                || http_stop_thread_.get_id() != std::this_thread::get_id(),
                            "HTTP stop thread cannot join itself");
    if (http_stop_thread_.joinable()) {
        http_stop_thread_.join();
    }
    http_stop_result_.get();
    return true;
}

void RtpLLMOp::startGrpcShutdown(std::chrono::steady_clock::time_point deadline) {
    if (grpc_server_ == nullptr || grpc_shutdown_result_.valid()) {
        return;
    }

    auto shutdown_signal = std::make_shared<std::promise<void>>();
    grpc_shutdown_result_ = shutdown_signal->get_future().share();
    auto* grpc_server     = grpc_server_.get();
    try {
        grpc_shutdown_thread_ = std::thread([grpc_server, deadline, shutdown_signal]() {
            try {
                RTP_LLM_LOG_INFO("Server shutting down");
                if (deadline == std::chrono::steady_clock::time_point::max()) {
                    grpc_server->Shutdown();
                } else {
                    const auto remaining = std::max(deadline - std::chrono::steady_clock::now(),
                                                    std::chrono::steady_clock::duration::zero());
                    grpc_server->Shutdown(std::chrono::system_clock::now() + remaining);
                }
                shutdown_signal->set_value();
            } catch (...) {
                try {
                    shutdown_signal->set_exception(std::current_exception());
                } catch (...) {
                    RTP_LLM_LOG_ERROR("failed to publish RPC transport shutdown failure");
                }
            }
        });
    } catch (...) {
        grpc_shutdown_result_ = {};
        throw;
    }
}

bool RtpLLMOp::waitForGrpcShutdown(std::chrono::steady_clock::time_point deadline) {
    if (!grpc_shutdown_result_.valid()) {
        return true;
    }
    if (!waitForFuture(grpc_shutdown_result_, deadline)) {
        return false;
    }

    RTP_LLM_CHECK_WITH_INFO(!grpc_shutdown_thread_.joinable()
                                || grpc_shutdown_thread_.get_id() != std::this_thread::get_id(),
                            "RPC shutdown thread cannot join itself");
    if (grpc_shutdown_thread_.joinable()) {
        grpc_shutdown_thread_.join();
    }
    try {
        grpc_shutdown_result_.get();
    } catch (...) {
        // No transport ownership was released. Clear only the failed attempt so
        // a later explicit stop can issue a fresh Shutdown call against the same server.
        grpc_shutdown_result_ = {};
        throw;
    }
    return true;
}

bool RtpLLMOp::waitForGrpcServerExit(std::chrono::steady_clock::time_point deadline) {
    if (!grpc_server_thread_.joinable()) {
        return true;
    }
    RTP_LLM_CHECK_WITH_INFO(grpc_server_thread_.get_id() != std::this_thread::get_id(),
                            "RPC server thread cannot join itself");
    RTP_LLM_CHECK_WITH_INFO(grpc_server_exit_result_.valid(), "RPC server exit signal is unavailable");
    if (!waitForFuture(grpc_server_exit_result_, deadline)) {
        return false;
    }
    grpc_server_exit_result_.get();
    grpc_server_thread_.join();
    return true;
}

void RtpLLMOp::startServiceStop() {
    if (model_rpc_service_ == nullptr || service_stop_result_.valid()) {
        return;
    }

    auto stop_signal = std::make_shared<std::promise<void>>();
    service_stop_result_ = stop_signal->get_future().share();
    auto* model_rpc_service = model_rpc_service_.get();
    try {
        service_stop_thread_ = std::thread([model_rpc_service, stop_signal]() {
            try {
                model_rpc_service->stop();
                stop_signal->set_value();
            } catch (...) {
                try {
                    stop_signal->set_exception(std::current_exception());
                } catch (...) {
                    RTP_LLM_LOG_ERROR("failed to publish RPC service stop failure");
                }
            }
        });
    } catch (...) {
        service_stop_result_ = {};
        throw;
    }
}

bool RtpLLMOp::waitForServiceStop(std::chrono::steady_clock::time_point deadline) {
    if (!service_stop_result_.valid()) {
        return true;
    }
    if (!waitForFuture(service_stop_result_, deadline)) {
        return false;
    }
    RTP_LLM_CHECK_WITH_INFO(!service_stop_thread_.joinable()
                                || service_stop_thread_.get_id() != std::this_thread::get_id(),
                            "RPC service stop thread cannot join itself");
    if (service_stop_thread_.joinable()) {
        service_stop_thread_.join();
    }
    // A failed service stop is terminal for this instance. Keep the shared result
    // so later calls cannot start a second stop concurrently with partial teardown.
    service_stop_result_.get();
    return true;
}

void RtpLLMOp::stopWithDeadline(std::chrono::steady_clock::time_point deadline) {
    std::unique_lock<std::mutex> stop_lock(stop_mutex_);
    if (is_server_shutdown_) {
        return;
    }
    if (service_stop_result_.valid()) {
        RTP_LLM_CHECK_WITH_INFO(waitForServiceStop(deadline), "RPC service failed to stop before deadline");
        service_stop_result_ = {};
        if (model_rpc_service_ || http_server_) {
            if (currentThreadHoldsGil()) {
                model_rpc_service_.reset();
                http_server_.reset();
            } else {
                RTP_LLM_CHECK_WITH_INFO(pythonRuntimeCanAcquireGil(),
                                        "Python runtime cannot acquire the GIL while releasing server resources");
                pybind11::gil_scoped_acquire acquire;
                model_rpc_service_.reset();
                http_server_.reset();
            }
        }
        stopKmonitorFactory();
        is_server_shutdown_ = true;
        return;
    }

    auto start_transport_stops = [this](std::chrono::steady_clock::time_point grpc_deadline) {
        std::exception_ptr start_error;
        try {
            startGrpcShutdown(grpc_deadline);
        } catch (...) {
            start_error = std::current_exception();
        }
        try {
            startHttpTransportStop();
        } catch (...) {
            if (!start_error) {
                start_error = std::current_exception();
            }
        }
        if (start_error) {
            std::rethrow_exception(start_error);
        }
    };

    if (http_server_) {
        http_server_->beginDrain();
    }
    if (model_rpc_service_) {
        model_rpc_service_->beginDrain();
    }

    if (http_server_ && !http_server_->waitForDrain(deadline)) {
        start_transport_stops(std::chrono::steady_clock::now());
        RTP_LLM_FAIL("HTTP requests failed to drain before shutdown");
    }
    if (model_rpc_service_ && !model_rpc_service_->waitForRequestDrain(deadline)) {
        start_transport_stops(std::chrono::steady_clock::now());
        RTP_LLM_FAIL("RPC requests failed to drain before shutdown");
    }

    startHttpTransportStop();
    if (model_rpc_service_ && !model_rpc_service_->prepareStop(remainingShutdownGrace(deadline))) {
        startGrpcShutdown(std::chrono::steady_clock::now());
        RTP_LLM_FAIL("remote load leases failed to quiesce before RPC shutdown");
    }

    startGrpcShutdown(deadline);
    RTP_LLM_CHECK_WITH_INFO(waitForHttpTransportStop(deadline),
                            "HTTP transport failed to stop before deadline");
    RTP_LLM_CHECK_WITH_INFO(waitForGrpcShutdown(deadline), "RPC transport failed to shut down before deadline");
    RTP_LLM_CHECK_WITH_INFO(waitForGrpcServerExit(deadline), "RPC server thread failed to exit before deadline");
    grpc_server_.reset();
    grpc_shutdown_result_    = {};
    grpc_server_exit_result_ = {};
    http_stop_result_        = {};

    startServiceStop();
    RTP_LLM_CHECK_WITH_INFO(waitForServiceStop(deadline), "RPC service failed to stop before deadline");
    service_stop_result_ = {};

    if (model_rpc_service_ || http_server_) {
        if (currentThreadHoldsGil()) {
            model_rpc_service_.reset();
            http_server_.reset();
        } else {
            RTP_LLM_CHECK_WITH_INFO(pythonRuntimeCanAcquireGil(),
                                    "Python runtime cannot acquire the GIL while releasing server resources");
            pybind11::gil_scoped_acquire acquire;
            model_rpc_service_.reset();
            http_server_.reset();
        }
    }
    stopKmonitorFactory();
    is_server_shutdown_ = true;
}

void RtpLLMOp::stop() {
    constexpr auto stop_timeout = std::chrono::seconds(60);
    const auto     stop_deadline = std::chrono::steady_clock::now() + stop_timeout;
    if (shouldReleaseGilForBlockingOperation(currentThreadHoldsGil(), pythonRuntimeIsFinalizing())) {
        GilScopedRelease release;
        stopWithDeadline(stop_deadline);
    } else {
        stopWithDeadline(stop_deadline);
    }
}

void RtpLLMOp::forceStopNoThrow() noexcept {
    constexpr auto force_stop_timeout = std::chrono::seconds(5);
    forceStopNoThrow(std::chrono::steady_clock::now() + force_stop_timeout);
}

void RtpLLMOp::forceStopNoThrow(std::chrono::steady_clock::time_point force_deadline) noexcept {
    std::unique_lock<std::mutex> stop_lock(stop_mutex_);
    if (is_server_shutdown_) {
        return;
    }

    const bool     service_stop_started = service_stop_result_.valid();

    if (!service_stop_started) {
        try {
            if (model_rpc_service_) {
                model_rpc_service_->beginDrain();
                (void)model_rpc_service_->prepareStop(std::chrono::milliseconds::zero());
            }
        } catch (const std::exception& e) {
            RTP_LLM_LOG_ERROR("failed to close RPC admission during destruction: %s", e.what());
            std::abort();
        } catch (...) {
            RTP_LLM_LOG_ERROR("failed to close RPC admission during destruction");
            std::abort();
        }

        try {
            std::exception_ptr start_error;
            try {
                startGrpcShutdown(std::chrono::steady_clock::now());
            } catch (...) {
                start_error = std::current_exception();
            }
            try {
                startHttpTransportStop();
            } catch (...) {
                if (!start_error) {
                    start_error = std::current_exception();
                }
            }
            if (start_error) {
                std::rethrow_exception(start_error);
            }
            if (http_server_ && !http_server_->waitForDrain(force_deadline)) {
                RTP_LLM_LOG_ERROR("HTTP requests remained active during final destruction");
                std::abort();
            }
            if (model_rpc_service_ && !model_rpc_service_->waitForRequestDrain(force_deadline)) {
                RTP_LLM_LOG_ERROR("RPC requests remained active during final destruction");
                std::abort();
            }
            if (model_rpc_service_
                && !model_rpc_service_->prepareStop(remainingShutdownGrace(force_deadline))) {
                RTP_LLM_LOG_ERROR("remote load leases remained active during final destruction");
                std::abort();
            }
            if (!waitForHttpTransportStop(force_deadline)) {
                RTP_LLM_LOG_ERROR("HTTP transport remained blocked during final destruction");
                std::abort();
            }
            if (!waitForGrpcShutdown(force_deadline)) {
                RTP_LLM_LOG_ERROR("RPC transport remained blocked during final destruction");
                std::abort();
            }
            if (!waitForGrpcServerExit(force_deadline)) {
                RTP_LLM_LOG_ERROR("RPC server thread remained blocked during final destruction");
                std::abort();
            }
        } catch (const std::exception& e) {
            RTP_LLM_LOG_ERROR("failed to shut down server transports during destruction: %s", e.what());
            std::abort();
        } catch (...) {
            RTP_LLM_LOG_ERROR("failed to shut down server transports during destruction");
            std::abort();
        }
        grpc_server_.reset();
        grpc_shutdown_result_    = {};
        grpc_server_exit_result_ = {};
        http_stop_result_        = {};
    }

    try {
        startServiceStop();
        if (!waitForServiceStop(force_deadline)) {
            RTP_LLM_LOG_ERROR("RPC service remained blocked during final destruction");
            std::abort();
        }
        service_stop_result_ = {};
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("failed to stop RPC service during destruction: %s", e.what());
        std::abort();
    } catch (...) {
        RTP_LLM_LOG_ERROR("failed to stop RPC service during destruction");
        std::abort();
    }

    auto* model_rpc_service = model_rpc_service_.release();
    auto* http_server       = http_server_.release();
    if (currentThreadHoldsGil()) {
        std::unique_ptr<RpcServiceImpl> model_rpc_service_owner(model_rpc_service);
        std::unique_ptr<HttpApiServer>  http_server_owner(http_server);
    } else if (!pythonRuntimeCanAcquireGil()) {
        RTP_LLM_LOG_ERROR(
            "Python runtime cannot acquire the GIL while releasing server resources; stopped resources are retained");
    } else {
        try {
            pybind11::gil_scoped_acquire acquire;
            std::unique_ptr<RpcServiceImpl> model_rpc_service_owner(model_rpc_service);
            std::unique_ptr<HttpApiServer>  http_server_owner(http_server);
        } catch (const std::exception& e) {
            RTP_LLM_LOG_ERROR(
                "failed to acquire the GIL while releasing server resources; stopped resources are retained: %s",
                e.what());
        } catch (...) {
            RTP_LLM_LOG_ERROR(
                "failed to acquire the GIL while releasing server resources; stopped resources are retained");
        }
    }

    try {
        stopKmonitorFactory();
    } catch (...) {
        RTP_LLM_LOG_ERROR("failed to stop metrics during destruction");
    }
    is_server_shutdown_ = true;
}

RtpLLMOp::~RtpLLMOp() noexcept {
    auto stop_without_gil = [this]() noexcept {
        try {
            stopWithDeadline(std::chrono::steady_clock::now() + std::chrono::seconds(60));
        } catch (const std::exception& e) {
            RTP_LLM_LOG_ERROR("graceful server destruction failed: %s", e.what());
            forceStopNoThrow();
        } catch (...) {
            RTP_LLM_LOG_ERROR("graceful server destruction failed");
            forceStopNoThrow();
        }
    };

    if (shouldReleaseGilForBlockingOperation(currentThreadHoldsGil(), pythonRuntimeIsFinalizing())) {
        GilScopedRelease release;
        stop_without_gil();
    } else {
        stop_without_gil();
    }
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
             py::arg("propose_model"),
             py::arg("token_processor"),
             py::arg("mm_process_engine"))
        .def("start_http_server",
             &RtpLLMOp::startHttpServer,
             py::arg("model_weights_loader"),
             py::arg("world_info"),
             py::arg("tokenizer"),
             py::arg("render"))
        .def("stop", &RtpLLMOp::stop);
}

}  // namespace rtp_llm

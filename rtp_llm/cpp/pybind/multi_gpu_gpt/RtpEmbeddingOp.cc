#include <optional>
#include <pybind11/pytypes.h>

#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/pybind/multi_gpu_gpt/RtpEmbeddingOp.h"

#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/engine_base/WeightsConverter.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/RoleTypes.h"

using namespace std;

namespace th = torch;

namespace rtp_llm {

RtpEmbeddingOp::RtpEmbeddingOp() {}

void RtpEmbeddingOp::init(py::object model,
                          py::object engine_config,
                          py::object vit_config,
                          py::object mm_process_engine) {
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

        auto             py_model = model.attr("py_model");
        EngineInitParams params(0,
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
                                py_model);
        py::object       custom_module     = model.attr("custom_module");
        py::object       py_render         = model.attr("custom_module").attr("renderer");
        py::object       py_tokenizer      = model.attr("tokenizer");
        py::object       py_handler        = model.attr("custom_module").attr("handler");
        bool             need_post_process = py_handler.attr("need_post_process").cast<bool>();

        if (parallelism_config.tp_rank == 0) {
            // kmon metric init
            (void)initKmonitorFactory();
            auto kmon_tags = kmonitor::MetricsTags();
            kmon_tags.AddTag("dp_rank", std::to_string(parallelism_config.dp_rank));
            params.metrics_reporter.reset(new kmonitor::MetricsReporter("", "", kmon_tags));
        }
        embedding_engine_.reset(new EmbeddingEngine(params, py_handler));
        if (!mm_process_engine.is_none()) {
            mm_processor_.reset(new LocalMultimodalProcessor(
                mm_process_engine, params.model_config_.mm_model_config, params.model_config_.max_seq_len));
        } else {
            mm_processor_.reset(
                new RemoteMultimodalProcessor(params.model_config_.mm_model_config, params.model_config_.max_seq_len));
        }

        startRpcServer(parallelism_config.model_rpc_port,
                       arpc_config.threadNum,
                       arpc_config.queueNum,
                       arpc_config.ioThreadNum,
                       py_render,
                       py_tokenizer,
                       params.metrics_reporter,
                       mm_processor_);
        startHttpServer(embedding_engine_, mm_processor_, params, custom_module);
        grpc_server_thread_ = std::thread(&RtpEmbeddingOp::initGrpcServer,
                                          this,
                                          parallelism_config.embedding_rpc_server_port,
                                          grpc_config,
                                          embedding_engine_,
                                          py_render,
                                          py_handler,
                                          mm_processor_,
                                          need_post_process);
        grpc_server_thread_.detach();
        while (!is_server_ready_) {
            sleep(1);  // wait 1s for server ready
        }
    } catch (const std::exception& e) {
        RTP_LLM_FAIL("init embedding engine failed, error msg: %s", e.what());
    }
}

void RtpEmbeddingOp::initGrpcServer(int64_t                              embedding_rpc_port,
                                    const GrpcConfig&                    grpc_config,
                                    std::shared_ptr<EmbeddingEngine>     embedding_engine,
                                    py::object                           py_render,
                                    py::object                           py_handler,
                                    std::shared_ptr<MultimodalProcessor> mm_processor,
                                    bool                                 need_post_process) {
    RTP_LLM_LOG_INFO("init embedding Grpc Server, port %ld", embedding_rpc_port);
    // NOTE: ip/ip段可自定义为所需范围。
    std::string server_address("0.0.0.0:" + std::to_string(embedding_rpc_port));
    {
        embedding_grpc_service_.reset(
            new EmbeddingRpcServiceImpl(embedding_engine, py_render, py_handler, mm_processor, need_post_process));
    }
    grpc::ServerBuilder builder;
    auto                server_config = grpc_config.get_server_config();
    for (auto it = server_config.begin(); it != server_config.end(); ++it) {
        RTP_LLM_LOG_INFO("grpc server add channel argument %s: %d", it->first.c_str(), it->second);
        builder.AddChannelArgument(it->first, it->second);
    }
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(embedding_grpc_service_.get());
    grpc_server_ = builder.BuildAndStart();
    RTP_LLM_CHECK_WITH_INFO(grpc_server_ != nullptr, "grpc server start failed at address " + server_address);

    RTP_LLM_LOG_DEBUG("Server listening on %s", server_address.c_str());
    is_server_ready_ = true;
    grpc_server_->Wait();
}

void RtpEmbeddingOp::stop() {
    if (embedding_rpc_service_) {
        embedding_rpc_service_->stop();
        embedding_rpc_service_.reset();
    }
    if (http_server_) {
        http_server_->stop();
        http_server_.reset();
    }
    if (!is_server_shutdown_ && embedding_engine_) {
        (void)embedding_engine_->stop();
        embedding_engine_.reset();
        is_server_shutdown_ = true;
    }
    if (grpc_server_) {
        grpc_server_->Shutdown();
        grpc_server_.reset();
    }
    embedding_grpc_service_.reset();
    stopKmonitorFactory();
}

void RtpEmbeddingOp::startHttpServer(std::shared_ptr<EmbeddingEngine>     embedding_engine,
                                     std::shared_ptr<MultimodalProcessor> mm_processor,
                                     const EngineInitParams&              params,
                                     py::object                           custom_module) {
    http_server_.reset(new HttpApiServer(embedding_engine, mm_processor, params, custom_module));
    std::string http_server_address("tcp:0.0.0.0:" + std::to_string(params.parallelism_config.http_port));
    if (http_server_->start(http_server_address)) {
        RTP_LLM_LOG_INFO("embedding HTTP Server listening on %s", http_server_address.c_str());
    } else {
        throw std::runtime_error("embedding HTTP Server start fail.");
    }
}

void RtpEmbeddingOp::startRpcServer(int64_t                              model_rpc_port,
                                    int64_t                              arpc_thread_num,
                                    int64_t                              arpc_queue_num,
                                    int64_t                              arpc_io_thread_num,
                                    py::object                           py_render,
                                    py::object                           py_tokenizer,
                                    kmonitor::MetricsReporterPtr         reporter,
                                    std::shared_ptr<MultimodalProcessor> mm_processor) {
    auto arpc_service = std::move(createEmbeddingArpcService(model_rpc_port,
                                                             arpc_thread_num,
                                                             arpc_queue_num,
                                                             arpc_io_thread_num,
                                                             py_render,
                                                             py_tokenizer,
                                                             mm_processor,
                                                             embedding_engine_,
                                                             reporter));
    if (arpc_service) {
        RTP_LLM_LOG_INFO("creating arpc service");
        embedding_rpc_service_.reset(new ArpcServerWrapper(
            std::move(arpc_service), arpc_thread_num, arpc_queue_num, arpc_io_thread_num, model_rpc_port));
        embedding_rpc_service_->start();
    } else {
        RTP_LLM_LOG_INFO("Embedding RPC not supported, skip");
    }
}

py::object RtpEmbeddingOp::decode(th::Tensor                   token_ids,
                                  th::Tensor                   token_type_ids,
                                  th::Tensor                   input_lengths,
                                  int64_t                      request_id,
                                  std::vector<MultimodalInput> multimodal_inputs) {
    if (is_server_shutdown_) {
        throw std::runtime_error("server is shut down, can't handle request");
    }
    std::optional<MultimodalFeature> multimodal_features = std::nullopt;
    auto                             embedding_input =
        std::make_shared<EmbeddingInput>(token_ids, token_type_ids, input_lengths, request_id, multimodal_features);
    if (mm_processor_ != nullptr && !multimodal_inputs.empty()) {
        auto mm_res = mm_processor_->updateMultimodalFeatures(embedding_input, multimodal_inputs);
        if (!mm_res.ok()) {
            throw std::runtime_error(mm_res.ToString());
        }
    }
    auto                   embedding_output = embedding_engine_->decode(embedding_input);
    py::gil_scoped_acquire acquire;
    if (embedding_output->output.isTensor) {
        RTP_LLM_CHECK_WITH_INFO(embedding_output->output.t.has_value(), "embedding output has null tensor value");
        return convertTensorToObject(embedding_output->output.t.value());
    } else {
        RTP_LLM_CHECK_WITH_INFO(embedding_output->output.map.has_value(), "embedding output has null map value");
        return convertTensorMapVectorToObject(embedding_output->output.map.value());
    }
}

RtpEmbeddingOp::~RtpEmbeddingOp() {
    stop();
}

void registerRtpEmbeddingOp(const py::module& m) {
    pybind11::class_<RtpEmbeddingOp>(m, "RtpEmbeddingOp")
        .def(pybind11::init<>())
        .def("init",
             &RtpEmbeddingOp::init,
             py::arg("model"),
             py::arg("engine_config"),
             py::arg("vit_config"),
             py::arg("mm_process_engine"))
        .def("stop", &RtpEmbeddingOp::stop)
        .def("decode",
             &RtpEmbeddingOp::decode,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("token_ids"),
             py::arg("token_type_ids"),
             py::arg("input_lengths"),
             py::arg("request_id"),
             py::arg("multimodal_inputs"));
}

}  // namespace rtp_llm

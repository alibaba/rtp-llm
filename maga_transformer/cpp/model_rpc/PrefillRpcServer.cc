#include "autil/TimeUtility.h"
#include "maga_transformer/cpp/utils/Cm2Config.h"
#include "maga_transformer/cpp/model_rpc/QueryConverter.h"
#include "maga_transformer/cpp/model_rpc/PrefillRpcServer.h"

#include <cstring>
#include <memory>
#include <unistd.h>
#include <limits.h>

using namespace std;
using namespace autil::legacy;
using namespace fastertransformer;

using grpc::ClientContext;
using grpc::Status;

namespace rtp_llm {

#define CLIENT_GRPC_RET_IF_ERROR(prefill_context, state, error_code_value)                                  \
    if (!(state)) {                                                                                         \
        auto new_error_code = error_code_value;                                                             \
        prefill_context.error_msg = "decode addr is " + prefill_context.decode_addr + ", ";                 \
        prefill_context.client_stream->WritesDone();                                                        \
        auto status = prefill_context.client_stream->Finish();                                              \
        if (!status.ok()) {                                                                                 \
            const auto& error_msg = status.error_message();                                                 \
            if (error_msg.find("Connect Failed") != std::string::npos) {                                    \
                new_error_code = ErrorCode::CONNECT_FAILED;                                                 \
            } else if(error_msg.find("Connection reset by peer") != std::string::npos) {                    \
                new_error_code = ErrorCode::CONNECTION_RESET_BY_PEER;                                       \
            }                                                                                               \
            prefill_context.error_msg += status.error_message();                                            \
        } else {                                                                                            \
            prefill_context.error_msg += "server disconnected with status::ok";                             \
        }                                                                                                   \
        prefill_context.error_code = new_error_code;                                                        \
        prefill_context.error_status = serializeErrorMsg(                                                   \
                prefill_context.request_id, prefill_context.error_code, prefill_context.error_msg);         \
        return;                                                                                             \
    }

#define EXECUTE_STAGE_FUNC(func, prefill_context)                               \
    func(prefill_context);                                                      \
    CHECK_ERROR_STATUS(prefill_context)

#define CHECK_ERROR_STATUS(prefill_context)                                     \
    if (prefill_context.finished || !prefill_context.error_status.ok()) {       \
        return prefill_context.error_status;                                    \
    }

#define EXECUTE_WITH_RETRY(func, prefill_context, max_retries, timeout_ms)      \
    int64_t begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();    \
    for (int attempt = 0; attempt <= max_retries; ++attempt) {                  \
        prefill_context.error_status = grpc::Status::OK;                        \
        func(prefill_context);                                                  \
        if (prefill_context.error_status.ok()) {                                \
            break;                                                              \
        }                                                                       \
        auto cost_time_us =                                                     \
            autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us;    \
        if (cost_time_us >= timeout_ms * 1000) {                                \
            break;                                                              \
        }                                                                       \
        usleep(1000 * 5);                                                       \
    }

PrefillGenerateContext::~PrefillGenerateContext() {
    if (stream) {
        // if is waiting, cancel it
        stream->cancelIfNotRunning();
        // if is running, waiting util done
        while (stream->running()) {
            FT_LOG_DEBUG("waiting prefill stream [%d] running done to cancel", stream->generateInput()->request_id);
            usleep(1000);
        }
        markRequestEnd();
    }
}

void PrefillGenerateContext::markRequestEnd() {
    server->cacheStore()->markRequestEnd(std::to_string(request_id));
    const auto& prefill_workers = server->workers();
    if (prefill_workers.empty()) {
        return;
    }
    RemoteFinishRequestPB finish_request;
    finish_request.set_request_id(request_id);
    for (int i = 0; i < prefill_workers.size(); i++) {
        auto& prefill_worker = prefill_workers[i];
        auto connect_status = server->rpcPool().getConnection(prefill_worker);
        if (!connect_status.ok()) {
            FT_LOG_WARNING("request [%d], get grpc connection for ip %s failed, ignore markRequestEnd for it",
                            request_id, prefill_worker.c_str());
            continue;
        }
        auto          stub = connect_status.value().stub.get();
        ClientContext client_context;
        EmptyPB       response;
        auto          grpc_status = stub->remote_finish(&client_context, finish_request, &response);
        if (!grpc_status.ok()) {
            FT_LOG_WARNING("request [%d], remote finish for ip %s failed, ignore markRequestEnd for it",
                            request_id, prefill_worker.c_str());
            continue;
        }
    }
}

grpc::Status PrefillRpcServer::init(const EngineInitParams&                                maga_init_params,
                                    py::object                                             mm_process_engine,
                                    std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) {
    FT_CHECK_WITH_INFO(maga_init_params.gpt_init_parameter.pd_separation_, "prefill's pd_separation must be true");
    auto ret = RemoteRpcServer::init(maga_init_params, mm_process_engine, std::move(propose_params));
    if (!ret.ok()) {
        return ret;
    }
    initLoadBalancer();
    return grpc::Status::OK;
}

void PrefillRpcServer::initLoadBalancer() {
    auto config        = makeConfig();
    load_balancer_ = std::make_shared<RRLoadBalancer>();
    FT_CHECK_WITH_INFO(load_balancer_->init(config), "load_balancer init failed");
    FT_LOG_INFO("load balancer init success");
}

LoadBalancerInitParams PrefillRpcServer::makeConfig() {
    char* use_local_env = std::getenv("USE_LOCAL");
    SubscribeServiceConfig subscribe_config;
    if (use_local_env) {
        // fake test
        char* remote_rpc_server_ip_env = std::getenv("REMOTE_RPC_SERVER_IP");
        FT_CHECK_WITH_INFO(remote_rpc_server_ip_env, "rpc server ip must be not empty");
        string remote_ip = string(remote_rpc_server_ip_env);
        uint32_t remote_port = maga_init_params_.gpt_init_parameter.remote_rpc_server_port_;
        FT_LOG_INFO("remote rpc server addr: %s:%d", remote_ip.c_str(), remote_port);

        docode_cluster_name_ = "LOCAL";
        LocalNodeJsonize node1(docode_cluster_name_, remote_ip, remote_port);
        LocalSubscribeServiceConfig local_config;
        local_config.nodes.push_back(node1);
        subscribe_config.local_configs.push_back(local_config);
    } else {
        char* decode_cm2_config_env = std::getenv("RTP_LLM_DECODE_CM2_CONFIG");
        FT_CHECK_WITH_INFO(decode_cm2_config_env, "decode_cm2_config_env must be not empty");
        string decode_cm2_config_str = string(decode_cm2_config_env);

        Cm2ClusterConfig decode_cm2_config;
        FromJsonString(decode_cm2_config, decode_cm2_config_str);
        docode_cluster_name_ = decode_cm2_config.cluster_name;
        CM2SubscribeServiceConfig cm2_service_config;
        cm2_service_config.zk_host = decode_cm2_config.zk_host;
        cm2_service_config.zk_path = decode_cm2_config.zk_path;
        cm2_service_config.zk_timeout_ms = 10 * 1000;
        cm2_service_config.clusters = {decode_cm2_config.cluster_name};
        subscribe_config.cm2_configs.push_back(cm2_service_config);
    }
    LoadBalancerInitParams params;
    params.subscribe_config = subscribe_config;
    params.update_interval_ms = 100;
    return params;
}

absl::Status PrefillRpcServer::waitStreamBeforeRun(std::shared_ptr<GenerateStream> stream) {
    while (stream->waiting()) {
        usleep(100);
    }
    return absl::OkStatus();
}

void PrefillRpcServer::getRpcConnection(PrefillGenerateContext& prefill_context) {
    auto host = load_balancer_->chooseHost(docode_cluster_name_);
    if (!host || host->ip.empty()) {
        prefill_context.error_code = ErrorCode::GET_HOST_FAILED;
        prefill_context.error_msg = "get host for decode cluster " + docode_cluster_name_ + " failed";
        prefill_context.error_status = serializeErrorMsg(prefill_context.request_id,
                                        prefill_context.error_code, prefill_context.error_msg);
        return;
    }
    auto decode_addr = host->ip + ":" + std::to_string(host->port);
    auto connect_status = rpc_pool_.getConnection(decode_addr);
    if (!connect_status.ok()) {
        prefill_context.error_code = ErrorCode::GET_CONNECTION_FAILED;
        prefill_context.error_msg = "get grpc connection for decode addr " + decode_addr + " failed";
        prefill_context.error_status = serializeErrorMsg(prefill_context.request_id,
                                        prefill_context.error_code, prefill_context.error_msg);
        return;
    }
    prefill_context.decode_addr = decode_addr;
    prefill_context.stub = connect_status.value().stub;
}

void PrefillRpcServer::remoteAllocateResource(PrefillGenerateContext& prefill_context) {
    prefill_context.client_context.reset(new ClientContext());
    prefill_context.client_stream = std::move(prefill_context.stub->remote_generate(prefill_context.client_context.get()));
    auto& client_stream = prefill_context.client_stream;
    GenerateRequestPB alloc_request;
    alloc_request.set_stage(RemoteStage::ALLOCATE);
    alloc_request.set_client_id(process_id_);
    alloc_request.set_request_id(prefill_context.request_id);
    GenerateInputPB* new_request = new GenerateInputPB(*prefill_context.rpc_context.request);
    alloc_request.set_allocated_input(new_request);

    CLIENT_GRPC_RET_IF_ERROR(prefill_context, client_stream->Write(alloc_request),
                            ErrorCode::REMOTE_ALLOCATE_RESOURCE_FAILED);
    GenerateOutputsPB allocate_response;
    CLIENT_GRPC_RET_IF_ERROR(prefill_context, client_stream->Read(&allocate_response),
                            ErrorCode::REMOTE_ALLOCATE_RESOURCE_FAILED);
}

void PrefillRpcServer::enqueueRequest(PrefillGenerateContext& prefill_context) {
    auto input                               = QueryConverter::transQuery(prefill_context.rpc_context.request);
    input->generate_config->pd_separation    = true;
    if (mm_processor_ != nullptr && input->multimodal_inputs) {
        auto mm_res = mm_processor_->updateMultimodalFeatures(input);
        if (!mm_res.ok()) {
            prefill_context.error_status = grpc::Status(grpc::StatusCode::CANCELLED, mm_res.ToString());
            return;
        }
    }
    auto lora_guard = lora::LoraResourceGuard(engine_->getLoraManager(), input->generate_config->adapter_name);
    FT_LOG_DEBUG("request:[%ld] trans to stream success", prefill_context.request_id);
    auto stream = engine_->enqueue(input);
    prefill_context.stream = stream;
    FT_LOG_DEBUG("request:[%ld] enqueue success", prefill_context.request_id);
}

void PrefillRpcServer::remoteLoadCache(PrefillGenerateContext& prefill_context) {
    auto wait_stat = waitStreamBeforeRun(prefill_context.stream);
    if (!wait_stat.ok()) {
        prefill_context.error_status = grpc::Status(grpc::StatusCode::INTERNAL, "failed to wait stream from waiting");
        return;
    }
    GenerateRequestPB load_request;
    load_request.set_client_id(process_id_);
    load_request.set_request_id(prefill_context.request_id);
    load_request.set_stage(RemoteStage::LOAD);
    FT_LOG_DEBUG("request:[%ld] before send load request", prefill_context.request_id);
    CLIENT_GRPC_RET_IF_ERROR(prefill_context, prefill_context.client_stream->Write(load_request),
            ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED);
    GenerateOutputsPB load_response;
    CLIENT_GRPC_RET_IF_ERROR(prefill_context, prefill_context.client_stream->Read(&load_response),
            ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED);
}

void PrefillRpcServer::pollLocalOutput(PrefillGenerateContext& prefill_context) {
    // TODO(xinfei.sxf) first token write to client affected by decode's load kv cache
    auto first_status = pollStreamOutput(prefill_context.rpc_context.context, prefill_context.request_id,
                                         prefill_context.rpc_context.writer, prefill_context.stream);
    if (!first_status.ok()) {
        prefill_context.client_stream->WritesDone();
        prefill_context.error_status = first_status;
        return;
    }
    FT_LOG_DEBUG("request:[%ld] poll local output end", prefill_context.request_id);

    if (prefill_context.stream->finished()) {
        prefill_context.client_stream->WritesDone();
        prefill_context.finished = true;
        prefill_context.error_status = grpc::Status::OK;
    }
}

void PrefillRpcServer::remoteGenerate(PrefillGenerateContext& prefill_context) {
    auto first_token = prefill_context.stream->currentExecuteTokens()[0];
    GenerateRequestPB generate_request;
    generate_request.set_client_id(process_id_);
    generate_request.set_request_id(prefill_context.request_id);
    generate_request.set_first_generate_token_id(first_token);
    generate_request.set_stage(RemoteStage::GENERATE);
    CLIENT_GRPC_RET_IF_ERROR(prefill_context, prefill_context.client_stream->Write(generate_request),
                            ErrorCode::REMOTE_GENERATE_FAILED);
}

void PrefillRpcServer::pollRemoteOutput(PrefillGenerateContext& prefill_context) {
    auto& request_id = prefill_context.request_id;
    GenerateOutputsPB response;
    prefill_context.response = &response;
    while (prefill_context.client_stream->Read(&response)) {
        // for last response, only record time info
        if (response.generate_outputs_size() == 0) {
            break;
        }
        for (size_t i = 0; i < response.generate_outputs_size(); i++) {
            response.mutable_generate_outputs(i)->mutable_aux_info()->set_pd_sep(true);
        }
        prefill_context.remote_cost_time_us = response.generate_outputs(0).aux_info().cost_time_us();
        if (!prefill_context.rpc_context.writer->Write(response)) {
            FT_LOG_WARNING("request:[%ld] write outputs pb failed", request_id);
            prefill_context.error_status = grpc::Status(grpc::StatusCode::INTERNAL, "request write outputs pb failed");
            return;
        }
        if (prefill_context.rpc_context.context->IsCancelled()) {
            FT_LOG_WARNING("request:[%ld] cancel by user", request_id);
            prefill_context.error_status = grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled");
            return;
        }
    }
    Status status = prefill_context.client_stream->Finish();
    if (!status.ok()) {
        prefill_context.stream->setStop(status.error_message(), absl::StatusCode::kInternal);
        prefill_context.error_msg = "decode addr is " + prefill_context.decode_addr + ", " + status.error_message();
        prefill_context.error_status = serializeErrorMsg(request_id,
                                        ErrorCode::REMOTE_GENERATE_FAILED, prefill_context.error_msg);
        return;
    }
    prefill_context.stream->setFinishedWithoutLock();
    reportTime(prefill_context);
}

void PrefillRpcServer::reportTime(PrefillGenerateContext& prefill_context) {
    auto request_id = prefill_context.request_id;
    const auto& response = *prefill_context.response;
    auto time_info = prefill_context.stream->getTimeInfo();
    auto query_start_time = time_info.begin_time_us;
    auto first_token_latency_us = time_info.first_token_latency_us;
    auto receive_load_cost_time = response.receive_load_time() - query_start_time;
    auto start_load_cost_time = response.start_load_time() - response.receive_load_time();
    auto load_cost_time = response.load_done_time() - response.start_load_time();
    auto receive_generate_cost_time = response.receive_generate_time() - response.receive_load_time();
    auto begin_compute_cost_time = response.begin_compute_time() - response.receive_generate_time();
    auto compute_cost_time = response.compute_done_time() - response.begin_compute_time();

    FT_LOG_DEBUG("request_id = [%d], first_token_latency_us = %ld", request_id, first_token_latency_us);
    FT_LOG_DEBUG("request_id = [%d], receive_load_cost_time = %ld, start_load_cost_time = %ld, load_cost_time = %ld",
                request_id, receive_load_cost_time, start_load_cost_time, load_cost_time);
    FT_LOG_DEBUG("request_id = [%d], receive_generate_cost_time = %ld, begin_compute_cost_time = %ld, "
                "compute_cost_time = %ld, remote_cost_time = %ld",
                request_id, receive_generate_cost_time, begin_compute_cost_time,
                compute_cost_time, prefill_context.remote_cost_time_us);

    RPCMetricsCollector collector;
    collector.load_latency_us  = load_cost_time;
    collector.remote_compute_latency_us = compute_cost_time;
    collector.total_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - query_start_time;
    reportMetrics(&collector);
}

grpc::Status PrefillRpcServer::prepareAllocateResource(PrefillGenerateContext& prefill_context) {
    EXECUTE_STAGE_FUNC(getRpcConnection, prefill_context);
    EXECUTE_STAGE_FUNC(remoteAllocateResource, prefill_context);
    return grpc::Status::OK;
}

grpc::Status PrefillRpcServer::generate_stream(grpc::ServerContext*                   context,
                                               const GenerateInputPB*                 request,
                                               grpc::ServerWriter<GenerateOutputsPB>* writer) {
    auto pd_separation = request->generate_config().max_new_tokens() > 1
                         && request->generate_config().num_beams() <= 1
                         && request->generate_config().num_return_sequences() <= 1;
    if (!pd_separation) {
        return LocalRpcServer::generate_stream(context, request, writer);
    }

    AtomicGuard request_guard(onflight_requests_);
    RPCContext rpc_context{context, request, writer};
    auto prefill_context = PrefillGenerateContext(this, rpc_context);
    EXECUTE_WITH_RETRY(prepareAllocateResource, prefill_context,
                        maga_init_params_.gpt_init_parameter.prefill_retry_times_,
                        maga_init_params_.gpt_init_parameter.prefill_retry_timeout_ms_);
    if (!prefill_context.error_status.ok()) {
        FT_LOG_WARNING("request [%ld] prepare allocate resource failed after retry %d times",
            prefill_context.request_id,
            maga_init_params_.gpt_init_parameter.prefill_retry_times_ + 1);
        if (maga_init_params_.gpt_init_parameter.pd_sep_enable_fallback_) {
            FT_LOG_WARNING("request [%ld] fallback to local server");
            return LocalRpcServer::generate_stream(context, request, writer);
        }
    }
    EXECUTE_STAGE_FUNC(enqueueRequest, prefill_context);
    EXECUTE_STAGE_FUNC(remoteLoadCache, prefill_context);
    EXECUTE_STAGE_FUNC(pollLocalOutput, prefill_context);
    EXECUTE_STAGE_FUNC(remoteGenerate, prefill_context);
    EXECUTE_STAGE_FUNC(pollRemoteOutput, prefill_context);
    return grpc::Status::OK;
}

bool PrefillRpcServer::ready() {
    if (maga_init_params_.gpt_init_parameter.pd_sep_enable_fallback_) {
        return true;
    }
    if (!load_balancer_) {
        FT_LOG_INFO("load balance is nullptr, server is not ready");
        return false;
    }
    auto ret = load_balancer_->isReady(docode_cluster_name_);
    if (!ret) {
        FT_LOG_INFO("load balancer is not ready now");
    }
    return ret;
}

grpc::Status PrefillRpcServer::remote_finish(grpc::ServerContext*           ontext,
                                             const RemoteFinishRequestPB*   request,
                                             EmptyPB*                       response) {
    auto request_id = request->request_id();
    cache_store_->markRequestEnd(std::to_string(request_id));
    return grpc::Status::OK;
}

}  // namespace rtp_llm

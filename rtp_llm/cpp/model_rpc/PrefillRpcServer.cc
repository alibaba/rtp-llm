#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/Host.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/utils/HashUtil.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include <cstring>
#include <cstdlib>
#include <future>
#include <strings.h>
#include <memory>
#include <unistd.h>
#include <limits.h>

using namespace std;
using namespace autil::legacy;

using grpc::Status;
using grpc::ClientContext;

namespace rtp_llm {

namespace {

bool envValueIsFalse(const char* value) {
    return value != nullptr
           && (strcmp(value, "0") == 0 || strcasecmp(value, "false") == 0 || strcasecmp(value, "off") == 0
               || strcasecmp(value, "no") == 0);
}

bool prefillCacheHitMetricEnabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("PREFILL_CACHE_HIT_METRIC_ENABLE");
        if (value == nullptr || value[0] == 0) {
            return true;
        }
        return !envValueIsFalse(value);
    }();
    return enabled;
}

std::vector<CacheKeyType> buildFullBlockCacheKeys(torch::Tensor input_ids, int seq_size_per_block) {
    std::vector<CacheKeyType> cache_keys;
    if (seq_size_per_block <= 0 || !input_ids.defined() || input_ids.numel() <= 0) {
        return cache_keys;
    }

    if (!input_ids.device().is_cpu()) {
        input_ids = input_ids.cpu();
    }
    if (!input_ids.is_contiguous()) {
        input_ids = input_ids.contiguous();
    }
    if (input_ids.scalar_type() != torch::kInt32) {
        input_ids = input_ids.to(torch::kInt32);
    }

    const int64_t token_num   = input_ids.numel();
    const int64_t block_count = token_num / seq_size_per_block;
    if (block_count <= 0) {
        return cache_keys;
    }
    cache_keys.reserve(static_cast<size_t>(block_count));

    auto*   token_ids    = input_ids.data_ptr<int32_t>();
    int64_t rolling_hash = 0;
    for (int64_t block_idx = 0; block_idx < block_count; ++block_idx) {
        const int64_t pos = block_idx * seq_size_per_block;
        rolling_hash      = rtp_llm::hashInt64Array(
            rolling_hash, token_ids + pos, token_ids + pos + static_cast<int64_t>(seq_size_per_block));
        cache_keys.push_back(static_cast<CacheKeyType>(rolling_hash));
    }
    return cache_keys;
}

void fillPrefillRecentCacheKeyMetricsCollector(PrefillRecentCacheKeyMetricsCollector& collector,
                                               const RecentCacheKeyWindow::Snapshot&  snapshot) {
    collector.has_value                  = true;
    collector.request_count              = true;
    collector.empty_request_count        = snapshot.request_occurrences == 0;
    collector.hit_count                  = snapshot.request_hit_occurrences;
    collector.total_count                = snapshot.request_occurrences;
    collector.hit_ratio                  = snapshot.request_hit_ratio;
    collector.retained_occurrences       = snapshot.retained_occurrences;
    collector.retained_unique_cache_keys = static_cast<int64_t>(snapshot.retained_unique_cache_keys);
    collector.time_window_ms             = snapshot.time_window_ms;
}

}  // namespace

#define CLIENT_GRPC_RET_IF_ERROR(prefill_context, state, error_code_value)                                             \
    if (!(state)) {                                                                                                    \
        auto   new_error_code = error_code_value;                                                                      \
        string new_error_msg  = "decode addr is " + prefill_context.decode_addr + ", ";                                \
        new_error_msg += "execute time is " + std::to_string(prefill_context.executeTimeMs()) + "ms, ";                \
        new_error_msg += "request timeout is " + std::to_string(prefill_context.request_timeout_ms) + "ms, ";          \
        new_error_msg += "rpc connection pointer is "                                                                  \
                         + std::to_string((int64_t)prefill_context.grpc_connection.channel.get()) + ", ";              \
        if (prefill_context.getStream()) {                                                                             \
            auto first_token_rt_ms = prefill_context.getStream()->getTimeInfo().first_token_rt_us / 1000;              \
            if (first_token_rt_ms) {                                                                                   \
                new_error_msg += "stream first token rt is " + std::to_string(first_token_rt_ms) + "ms, ";             \
            }                                                                                                          \
            auto wait_time_ms = prefill_context.getStream()->getTimeInfo().wait_time_us / 1000;                        \
            if (wait_time_ms) {                                                                                        \
                new_error_msg += "stream wait time is " + std::to_string(wait_time_ms) + "ms, ";                       \
            }                                                                                                          \
        }                                                                                                              \
        auto status = prefill_context.closeGrpcStream();                                                               \
        if (!status.ok()) {                                                                                            \
            const auto& error_msg = status.error_message();                                                            \
            if (error_msg.find("Connect Failed") != std::string::npos) {                                               \
                new_error_code = ErrorCode::CONNECT_FAILED;                                                            \
                prefill_context.closeGrpcConnection();                                                                 \
            } else if (error_msg.find("No route to host") != std::string::npos) {                                      \
                new_error_code = ErrorCode::CONNECT_FAILED;                                                            \
                prefill_context.closeGrpcConnection();                                                                 \
            } else if (error_msg.find("Connection reset by peer") != std::string::npos) {                              \
                new_error_code = ErrorCode::CONNECTION_RESET_BY_PEER;                                                  \
                prefill_context.closeGrpcConnection();                                                                 \
            } else if (error_msg.find("Connection timed out") != std::string::npos) {                                  \
                new_error_code = ErrorCode::CONNECT_TIMEOUT;                                                           \
                prefill_context.closeGrpcConnection();                                                                 \
            } else if (error_msg.find("Deadline Exceeded") != std::string::npos) {                                     \
                new_error_code = ErrorCode::DEADLINE_EXCEEDED;                                                         \
                prefill_context.closeGrpcConnection();                                                                 \
            } else if (error_msg.find("keepalive watchdog timeout") != std::string::npos) {                            \
                new_error_code = ErrorCode::KEEP_ALIVE_TIMEOUT;                                                        \
                prefill_context.closeGrpcConnection();                                                                 \
            }                                                                                                          \
            new_error_msg += error_msg;                                                                                \
            if (status.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED) {                                         \
                new_error_code = ErrorCode::DECODE_MALLOC_FAILED;                                                      \
            }                                                                                                          \
        } else {                                                                                                       \
            if (prefill_context.client_stream) {                                                                       \
                new_error_msg += "server disconnected with status::ok";                                                \
            }                                                                                                          \
        }                                                                                                              \
        if (prefill_context.getStream()) {                                                                             \
            prefill_context.getStream()->reportEvent(StreamEvents::Error, new_error_code, new_error_msg);              \
        }                                                                                                              \
        prefill_context.error_info   = ErrorInfo(new_error_code, new_error_msg);                                       \
        prefill_context.error_status = serializeErrorMsg(prefill_context.request_key, prefill_context.error_info);     \
        return;                                                                                                        \
    }

grpc::Status PrefillRpcServer::init(const EngineInitParams&                                maga_init_params,
                                    py::object                                             mm_process_engine,
                                    std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) {
    RTP_LLM_CHECK_WITH_INFO(maga_init_params.pd_sep_config.role_type == RoleType::PREFILL,
                            "prefill's role_type must be PREFILL");
    auto ret = RemoteRpcServer::init(maga_init_params, mm_process_engine, std::move(propose_params));
    if (!ret.ok()) {
        return ret;
    }
    if (prefillCacheHitMetricEnabled()) {
        prefill_recent_cache_key_window_ = std::make_unique<RecentCacheKeyWindow>();
    } else {
        RTP_LLM_LOG_INFO("prefill recent-cache-key metrics disabled by PREFILL_CACHE_HIT_METRIC_ENABLE");
    }
    return grpc::Status::OK;
}

ErrorInfo PrefillRpcServer::waitStreamBeforeRun(std::shared_ptr<GenerateStream> stream) {
    static int max_wait_timeout_us = maga_init_params_.pd_sep_config.prefill_max_wait_timeout_ms * 1000;
    auto       begin_time_us       = currentTimeUs();
    while (!stream->hasError() && stream->getStatus() == StreamState::WAITING) {
        usleep(100);
        auto current_time_us = currentTimeUs();
        auto cost_time_us    = current_time_us - begin_time_us;
        if (cost_time_us > max_wait_timeout_us) {
            string new_error_msg = "wait to run timeout, timeout is " + std::to_string(max_wait_timeout_us) + " us";
            stream->reportEvent(StreamEvents::Error, ErrorCode::WAIT_TO_RUN_TIMEOUT, new_error_msg);
            return ErrorInfo(ErrorCode::WAIT_TO_RUN_TIMEOUT, new_error_msg);
        }
    }
    if (stream->hasError()) {
        return stream->statusInfo();
    }
    return ErrorInfo::OkStatus();
}

void PrefillRpcServer::getRpcConnection(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%ld] trans query", prefill_context.request_id);
    auto input = QueryConverter::transQuery(prefill_context.rpc_context.request);
    if (applyTimelineGate(prefill_context.request_key,
                          input->generate_config->gen_timeline,
                          input->generate_config->profile_step,
                          input->generate_config->profile_trace_name)) {
        input->generate_config->gen_timeline = true;
    }
    input->generate_config->pd_separation = true;
    if (engine_->isMTPEagle()) {
        input->generate_config->force_disable_sp_run = false;
    } else {
        input->generate_config->force_disable_sp_run = true;
    }
    prefill_context.generate_input = input;

    RTP_LLM_LOG_DEBUG("request [%ld] get rpc connection", prefill_context.request_id);

    auto&                       role_addrs = prefill_context.generate_input->generate_config->role_addrs;
    std::shared_ptr<const Host> host;

    // Check if request specifies host for DECODE role
    for (auto& role_addr : role_addrs) {
        if (role_addr.role == RoleType::DECODE) {
            host = std::make_shared<const Host>(role_addr.ip, role_addr.grpc_port, role_addr.http_port);
            break;
        }
    }

    // If no host specified in request, check if there's a master role
    char* remote_rpc_server_ip_env = std::getenv("REMOTE_RPC_SERVER_IP");
    bool  has_master_role          = (remote_rpc_server_ip_env != nullptr && strlen(remote_rpc_server_ip_env) > 0);

    // If no host specified in request and no master role, this is a direct prefill request
    // In this case, we still need to select decode machines as specified in the requirements
    if (!host && !has_master_role) {
        // For direct prefill requests without master role, we still need to select decode machines
        // The current logic will fail as expected since no host is available
        RTP_LLM_LOG_DEBUG(
            "request [%ld] no host specified in request and no master role, need to select decode machines",
            prefill_context.request_id);
    }

    if (!host || host->ip.empty()) {
        prefill_context.error_info =
            ErrorInfo(ErrorCode::GET_HOST_FAILED, "get host for decode cluster " + decode_cluster_name_ + " failed");
        prefill_context.error_status = serializeErrorMsg(prefill_context.request_key, prefill_context.error_info);
        return;
    }
    auto decode_addr    = host->ip + ":" + std::to_string(host->rpc_port);
    auto connect_status = resource_.rpc_pool.getConnection(decode_addr);
    if (!connect_status.ok()) {
        prefill_context.error_info   = ErrorInfo(ErrorCode::GET_CONNECTION_FAILED,
                                               "get grpc connection for decode addr " + decode_addr + " failed");
        prefill_context.error_status = serializeErrorMsg(prefill_context.request_key, prefill_context.error_info);
        return;
    }
    prefill_context.decode_addr     = decode_addr;
    prefill_context.grpc_connection = connect_status.value();

    RTP_LLM_LOG_DEBUG("request [%ld] get rpc connection done", prefill_context.request_id);
}

void PrefillRpcServer::multimodalProcess(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    auto& input = prefill_context.generate_input;
    if (mm_processor_ != nullptr && input->multimodal_inputs) {
        auto result = mm_processor_->updateMultimodalFeatures(input);
        CLIENT_GRPC_RET_IF_ERROR(prefill_context, result.ok(), result.code());

        auto mutable_request = const_cast<GenerateInputPB*>(prefill_context.rpc_context.request);
        mutable_request->clear_token_ids();
        // TODO(xinfei.sxf) optimize copy
        auto* ids_ptr = input->input_ids.data_ptr<int32_t>();
        for (size_t i = 0; i < input->input_ids.numel(); i++) {
            mutable_request->add_token_ids(ids_ptr[i]);
        }
    }
}

void PrefillRpcServer::remoteAllocateResource(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%ld] start to remote allocate resource", prefill_context.request_id);
    prefill_context.client_context.reset(new ClientContext());
    auto request_timeout_ms = prefill_context.request_timeout_ms;
    auto max_rpc_timeout_ms = maga_init_params_.pd_sep_config.max_rpc_timeout_ms;
    auto final_timeout_ms   = max_rpc_timeout_ms > 0 ? max_rpc_timeout_ms : MAX_GRPC_TIMEOUT_MS;
    final_timeout_ms        = request_timeout_ms > 0 ? request_timeout_ms : final_timeout_ms;

    auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(final_timeout_ms);
    prefill_context.client_context->set_deadline(deadline);
    prefill_context.client_stream =
        std::move(prefill_context.grpc_connection.stub->RemoteGenerate(prefill_context.client_context.get()));
    auto&             client_stream = prefill_context.client_stream;
    GenerateRequestPB alloc_request;
    alloc_request.set_stage(RemoteStage::ALLOCATE);
    alloc_request.set_client_id(process_id_);
    alloc_request.set_request_id(prefill_context.request_id);
    // TODO(xinfei.sxf) reduce copy
    GenerateInputPB* new_request = new GenerateInputPB(*prefill_context.rpc_context.request);
    alloc_request.set_allocated_input(new_request);
    for (auto& addrs : prefill_context.prefill_worker_cache_store_addrs) {
        alloc_request.add_peer_addrs(addrs);
    }

    // Propagate CP size so decode knows prefill used context-parallel page-RR.
    const auto& cp_cfg = maga_init_params_.parallelism_config.prefill_cp_config;
    if (cp_cfg.kv_cache_sharded && maga_init_params_.parallelism_config.tp_size > 1) {
        alloc_request.set_prefill_cp_size(static_cast<int32_t>(maga_init_params_.parallelism_config.tp_size));
    }

    CLIENT_GRPC_RET_IF_ERROR(
        prefill_context, client_stream->Write(alloc_request), ErrorCode::REMOTE_ALLOCATE_RESOURCE_WRITE_FAILED);
    GenerateOutputsPB allocate_response;
    CLIENT_GRPC_RET_IF_ERROR(
        prefill_context, client_stream->Read(&allocate_response), ErrorCode::REMOTE_ALLOCATE_RESOURCE_READ_FAILED);
    RTP_LLM_LOG_DEBUG("request [%ld] remote allocate resource done", prefill_context.request_id);
}

void PrefillRpcServer::enqueueRequest(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%ld] trans query", prefill_context.request_id);
    RTP_LLM_LOG_DEBUG("request [%ld] trans to stream success", prefill_context.request_id);
    auto stream = engine_->enqueue(prefill_context.generate_input);
    prefill_context.setStream(stream);
    RTP_LLM_LOG_DEBUG("request [%ld] enqueue success", prefill_context.request_id);
}

void PrefillRpcServer::remoteLoadCacheStart(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%ld] remote load cache", prefill_context.request_id);
    prefill_context.error_info = waitStreamBeforeRun(prefill_context.getStream());
    if (prefill_context.error_info.hasError()) {
        prefill_context.error_status = serializeErrorMsg(prefill_context.request_key, prefill_context.error_info);
        return;
    }
    AtomicGuard       request_guard(loading_cache_requests_);
    GenerateRequestPB load_request;
    load_request.set_client_id(process_id_);
    load_request.set_request_id(prefill_context.request_id);
    load_request.set_start_time(currentTimeUs());
    CLIENT_GRPC_RET_IF_ERROR(
        prefill_context, prefill_context.client_stream->Write(load_request), ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED);
}

void PrefillRpcServer::pollLocalOutput(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%ld] start to poll local output", prefill_context.request_id);
    auto first_status = pollStreamOutput(prefill_context.server_context,
                                         prefill_context.request_key,
                                         prefill_context.rpc_context.writer,
                                         prefill_context.getStream());
    if (!first_status.ok()) {
        prefill_context.error_status = first_status;
        return;
    }
    RTP_LLM_LOG_DEBUG("request [%ld] poll local output end", prefill_context.request_id);

    auto stream = prefill_context.getStream();
    if (stream->hasError()) {
        prefill_context.finished     = true;
        prefill_context.error_status = grpc::Status(grpc::StatusCode::INTERNAL, stream->statusInfo().ToString());
    }
}

void PrefillRpcServer::remoteLoadCacheEnd(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    GenerateOutputsPB load_response;
    CLIENT_GRPC_RET_IF_ERROR(
        prefill_context, prefill_context.client_stream->Read(&load_response), ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED);
    auto error_code = transRPCErrorCode(load_response.error_info().error_code());

    // Decode has finished loading cache, now safe to release KV cache blocks.
    // This is called after cache store transfer is complete.
    if (prefill_context.generate_input->generate_config->pd_separation) {
        prefill_context.getStream()->releaseKVCacheForPDSep();
    }

    CLIENT_GRPC_RET_IF_ERROR(prefill_context, error_code == ErrorCode::NONE_ERROR, error_code);
    RTP_LLM_LOG_DEBUG("request [%ld] remote load cache done", prefill_context.request_id);

    meta_->dequeue(prefill_context.request_id, prefill_context.getStream());
    if (!prefill_context.getStream()->hasEvent(StreamEvents::NeedRemoteGenerate)) {
        RTP_LLM_LOG_DEBUG("request [%ld] pd-sep prefill finished locally without remote generate, "
                          "skipping remote generate stages",
                          prefill_context.request_id);
        // Exit here to keep the remote load-cache completion and release ordering intact.
        prefill_context.finished = true;
    }
}

void PrefillRpcServer::remoteGenerate(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%ld] start to remote generate", prefill_context.request_id);
    std::shared_ptr<GenerateStream> stream = prefill_context.getStream();
    RTP_LLM_LOG_DEBUG("remote generate stream[%s]: %s", stream->streamLogTag().c_str(), stream->debugString().c_str());
    vector<int> all_token   = stream->currentExecuteTokens();
    int         first_token = all_token[all_token.size() - 1];
    RTP_LLM_LOG_DEBUG("first token token id %d", first_token);
    GenerateRequestPB generate_request;
    generate_request.set_client_id(process_id_);
    generate_request.set_request_id(prefill_context.request_id);
    generate_request.set_first_generate_token_id(first_token);
    auto context_position_ids = stream->getContextPositionIds();
    if (context_position_ids.defined()) {
        generate_request.mutable_position_ids()->CopyFrom(
            {context_position_ids.data_ptr<int32_t>(),
             context_position_ids.data_ptr<int32_t>() + context_position_ids.numel()});
    }
    if (engine_->isMTPEagle()) {
        RTP_LLM_CHECK_WITH_INFO(stream->getProposeToken().size() > 0,
                                "mtp remote generate propose token should not be empty");
    }
    generate_request.mutable_propose_token_ids()->CopyFrom(
        {stream->getProposeToken().begin(), stream->getProposeToken().end()});

    auto sp_output_buffer = stream->getSPOutputBuffer();

    if (sp_output_buffer) {
        auto all_probs_cpu =
            sp_output_buffer->all_probs.is_cuda() ? sp_output_buffer->all_probs.cpu() : sp_output_buffer->all_probs;
        torch::Tensor hidden_states_cpu;
        if (!sp_output_buffer->hidden_states.defined()) {
            // dummy hidden states, so datatype is not important
            hidden_states_cpu = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat16));
        } else {
            hidden_states_cpu = sp_output_buffer->hidden_states.is_cuda() ? sp_output_buffer->hidden_states.cpu() :
                                                                            sp_output_buffer->hidden_states;
        }
        QueryConverter::transTensorPB(generate_request.mutable_propose_probs(), all_probs_cpu);
        QueryConverter::transTensorPB(generate_request.mutable_propose_hidden(), hidden_states_cpu);
    }

    generate_request.set_stage(RemoteStage::GENERATE);

    CLIENT_GRPC_RET_IF_ERROR(
        prefill_context, prefill_context.client_stream->Write(generate_request), ErrorCode::REMOTE_GENERATE_FAILED);
}

void PrefillRpcServer::pollRemoteOutput(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%ld] start to poll remote output", prefill_context.request_id);
    auto&             request_id = prefill_context.request_id;
    GenerateOutputsPB response;
    auto              prefill_total_reuse_len  = prefill_context.getStream()->initialReuseLength();
    auto              prefill_local_reuse_len  = prefill_context.getStream()->localReuseLength();
    auto              prefill_remote_reuse_len = prefill_context.getStream()->remoteReuseLength();
    auto              prefill_memory_reuse_len = prefill_context.getStream()->memoryReuseLength();

    auto first_token_rt_us = prefill_context.getStream()->getTimeInfo().first_token_rt_us;
    while (prefill_context.client_stream->Read(&response)) {
        if (prefill_context.server_context && prefill_context.server_context->IsCancelled()) {
            RTP_LLM_LOG_WARNING("request [%ld] cancel by user", request_id);
            prefill_context.error_status = grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled");
            return;
        }
        if (response.flatten_output().aux_info_size() == 0) {
            RTP_LLM_LOG_ERROR("request [%ld] generate output size is 0", request_id);
            break;
        }
        for (size_t i = 0; i < response.flatten_output().aux_info_size(); i++) {
            response.mutable_flatten_output()->mutable_aux_info(i)->set_pd_sep(true);
        }
        int64_t cost_time_us = currentTimeUs() - prefill_context.request_begin_time_us;
        for (size_t i = 0; i < response.flatten_output().aux_info_size(); i++) {
            auto decode_total_reuse_len  = response.flatten_output().aux_info(i).total_reuse_len();
            auto decode_local_reuse_len  = response.flatten_output().aux_info(i).local_reuse_len();
            auto decode_remote_reuse_len = response.flatten_output().aux_info(i).remote_reuse_len();
            auto decode_memory_reuse_len = response.flatten_output().aux_info(i).memory_reuse_len();

            response.mutable_flatten_output()->mutable_aux_info(i)->set_first_token_cost_time_us(first_token_rt_us);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_cost_time_us(cost_time_us);

            response.mutable_flatten_output()->mutable_aux_info(i)->set_total_reuse_len(prefill_total_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_local_reuse_len(prefill_local_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_remote_reuse_len(prefill_remote_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_memory_reuse_len(prefill_memory_reuse_len);

            response.mutable_flatten_output()->mutable_aux_info(i)->set_prefill_total_reuse_len(
                prefill_total_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_prefill_local_reuse_len(
                prefill_local_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_prefill_remote_reuse_len(
                prefill_remote_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_prefill_memory_reuse_len(
                prefill_memory_reuse_len);

            response.mutable_flatten_output()->mutable_aux_info(i)->set_decode_total_reuse_len(decode_total_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_decode_local_reuse_len(decode_local_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_decode_remote_reuse_len(
                decode_remote_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_decode_memory_reuse_len(
                decode_memory_reuse_len);
        }
        if (!prefill_context.rpc_context.writer->Write(response)) {
            RTP_LLM_LOG_WARNING("request [%ld] write outputs pb failed", request_id);
            prefill_context.error_status = grpc::Status(grpc::StatusCode::INTERNAL, "request write outputs pb failed");
            return;
        }
    }
    CLIENT_GRPC_RET_IF_ERROR(
        prefill_context, prefill_context.closeGrpcStream().ok(), ErrorCode::REMOTE_GENERATE_FAILED);
}

grpc::Status PrefillRpcServer::prepareAllocateResource(PrefillGenerateContext& prefill_context) {
    EXECUTE_STAGE_FUNC(getRpcConnection, prefill_context);
    EXECUTE_STAGE_FUNC(multimodalProcess, prefill_context);
    EXECUTE_STAGE_FUNC(remoteAllocateResource, prefill_context);
    return grpc::Status::OK;
}

void PrefillRpcServer::reportPrefillRecentCacheKeyMetrics(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    if (!prefillCacheHitMetricEnabled()) {
        return;
    }
    if (!prefill_recent_cache_key_window_) {
        return;
    }
    if (!prefill_context.generate_input) {
        return;
    }

    const int seq_size_per_block = maga_init_params_.kv_cache_config.seq_size_per_block;
    auto      cache_keys = buildFullBlockCacheKeys(prefill_context.generate_input->input_ids, seq_size_per_block);
    auto      snapshot   = prefill_recent_cache_key_window_->record(cache_keys);

    if (metrics_reporter_) {
        PrefillRecentCacheKeyMetricsCollector collector;
        fillPrefillRecentCacheKeyMetricsCollector(collector, snapshot);
        metrics_reporter_->report<PrefillRecentCacheKeyMetrics, PrefillRecentCacheKeyMetricsCollector>(nullptr,
                                                                                                       &collector);
    }
}

grpc::Status PrefillRpcServer::syncPrefix(PrefillGenerateContext& prefill_context) {
    auto max_retry_times      = maga_init_params_.pd_sep_config.prefill_retry_times;
    auto max_retry_timeout_ms = maga_init_params_.pd_sep_config.prefill_retry_timeout_ms;
    int  retry_interval_ms    = 1;
    EXECUTE_WITH_RETRY(
        prepareAllocateResource, prefill_context, max_retry_times, max_retry_timeout_ms, retry_interval_ms);
    if (prefill_context.hasError()) {
        RTP_LLM_LOG_WARNING(
            "request [%ld] prepare allocate resource failed after retry [%d] times, cost time ms [%ld], "
            "max retry time [%ld], max retry timeout ms [%ld]",
            prefill_context.request_id,
            prefill_context.retry_times,
            prefill_context.retry_cost_time_ms,
            max_retry_times + 1,
            max_retry_timeout_ms);
        return prefill_context.error_status;
    }
    EXECUTE_STAGE_FUNC(enqueueRequest, prefill_context);
    EXECUTE_STAGE_FUNC(remoteLoadCacheStart, prefill_context);
    return grpc::Status::OK;
}

grpc::Status PrefillRpcServer::finishStream(PrefillGenerateContext& prefill_context) {
    EXECUTE_STAGE_FUNC(pollLocalOutput, prefill_context);
    EXECUTE_STAGE_FUNC(remoteLoadCacheEnd, prefill_context);
    EXECUTE_STAGE_FUNC(remoteGenerate, prefill_context);
    EXECUTE_STAGE_FUNC(pollRemoteOutput, prefill_context);
    prefill_context.stat_info.nextStage();
    return grpc::Status::OK;
}

grpc::Status PrefillRpcServer::GenerateStreamCall(grpc::ServerContext*                   server_context,
                                                  const GenerateInputPB*                 request,
                                                  grpc::ServerWriter<GenerateOutputsPB>* writer) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%ld] start generate stream call", request->request_id());
    auto pd_separation = request->generate_config().max_new_tokens() > 1 && request->generate_config().num_beams() <= 1
                         && request->generate_config().variable_num_beams().size() == 0
                         && request->generate_config().num_return_sequences() <= 1
                         && request->generate_config().can_use_pd_separation();
    if (!pd_separation) {
        return LocalRpcServer::GenerateStreamCall(server_context, request, writer);
    }

    AtomicGuardPtr request_guard = make_shared<AtomicGuard>(onflight_requests_);
    RPCContext     rpc_context{request, writer};
    auto           prefill_context         = PrefillGenerateContext(&this->resource(),
                                                  rpc_context,
                                                  request->generate_config().timeout_ms(),
                                                  server_context,
                                                  metrics_reporter_,
                                                  meta_);
    prefill_context.onflight_requests      = onflight_requests_;
    prefill_context.loading_cache_requests = loading_cache_requests_;

    try {
        auto status = syncPrefix(prefill_context);
        if (!status.ok()) {
            return status;
        }
        reportPrefillRecentCacheKeyMetrics(prefill_context);
        status = finishStream(prefill_context);
        if (!status.ok()) {
            return status;
        }
    } catch (const std::exception& e) {
        auto error_msg = "request [" + prefill_context.request_key + "] catch exception [" + e.what() + "]";
        prefill_context.error_status = grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
        return prefill_context.error_status;
    } catch (...) {
        auto error_msg               = "request [" + prefill_context.request_key + "] catch unknown exception";
        prefill_context.error_status = grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
        return prefill_context.error_status;
    }

    RTP_LLM_LOG_DEBUG("request [%ld] all done", prefill_context.request_id);

    return grpc::Status::OK;
}

grpc::Status
PrefillRpcServer::RemoteFinish(grpc::ServerContext* context, const RemoteFinishRequestPB* request, EmptyPB* response) {
    RTP_LLM_PROFILE_FUNCTION();
    auto request_id = request->request_id();
    resource_.cache_store->markRequestEnd(std::to_string(request_id));
    return grpc::Status::OK;
}

grpc::Status
PrefillRpcServer::Enqueue(grpc::ServerContext* context, const EnqueueRequestPB* request, EnqueueResponsePB* response) {
    RTP_LLM_PROFILE_FUNCTION();
    const auto& input      = request->input();
    auto        request_id = input.request_id();
    response->set_request_id(request_id);

    if (input.is_fake_query()) {
        auto fake_stream = engine_->createMinFakeStream(/*max_new_tokens=*/1);
        RTP_LLM_LOG_INFO("Enqueue fake_query req=%ld dp_rank=%d stream=%p",
                         request_id,
                         input.has_dp_rank() ? input.dp_rank().value() : -1,
                         fake_stream.get());
        if (!fake_stream) {
            return grpc::Status(grpc::StatusCode::INTERNAL,
                                "createMinFakeStream returned null for fake_query req "
                                + std::to_string(request_id));
        }
        engine_->enqueue(fake_stream);
        return grpc::Status::OK;
    }

    if (response_registry_.get(request_id) != nullptr) {
        auto msg = "request [" + std::to_string(request_id) + "] already enqueued";
        RTP_LLM_LOG_WARNING("%s", msg.c_str());
        return grpc::Status(grpc::StatusCode::ALREADY_EXISTS, msg);
    }

    auto rpc_context                        = std::make_shared<RPCContext>(RPCContext{&input, nullptr});
    auto prefill_context                    = std::make_shared<PrefillGenerateContext>(&this->resource(),
                                                                    *rpc_context,
                                                                    input.generate_config().timeout_ms(),
                                                                    /*server_context=*/nullptr,
                                                                    metrics_reporter_,
                                                                    meta_);
    prefill_context->onflight_requests      = onflight_requests_;
    prefill_context->loading_cache_requests = loading_cache_requests_;
    AtomicGuardPtr request_guard            = std::make_shared<AtomicGuard>(onflight_requests_);

    grpc::Status status;
    try {
        status = syncPrefix(*prefill_context);
    } catch (const std::exception& e) {
        auto error_msg = "request [" + prefill_context->request_key + "] enqueue exception [" + e.what() + "]";
        return grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
    }
    if (!status.ok()) {
        return status;
    }

    auto entry                          = response_registry_.create(request_id);
    auto writer                         = std::make_shared<ResponseBufferWriter>(entry);
    prefill_context->rpc_context.writer = writer.get();

    std::thread worker([this, prefill_context, rpc_context, writer, entry, request_guard, request_id]() mutable {
        grpc::Status finish_status;
        try {
            finish_status = finishStream(*prefill_context);
        } catch (const std::exception& e) {
            auto error_msg = "request [" + prefill_context->request_key + "] finishStream exception [" + e.what() + "]";
            finish_status  = grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
        } catch (...) {
            finish_status = grpc::Status(grpc::StatusCode::INTERNAL, "finishStream unknown exception");
        }
        if (!finish_status.ok()) {
            std::lock_guard<std::mutex> lock(entry->mu);
            entry->error_status = finish_status;
        }
        entry->done.store(true);
        entry->cv.notify_all();
        RTP_LLM_LOG_DEBUG("request [%ld] detached finishStream done, ok=%d", request_id, finish_status.ok());
    });
    worker.detach();

    return grpc::Status::OK;
}

grpc::Status PrefillRpcServer::BatchEnqueue(grpc::ServerContext*         context,
                                            const BatchEnqueueRequestPB* request,
                                            BatchEnqueueResponsePB*      response) {
    RTP_LLM_PROFILE_FUNCTION();
    const auto& cfg          = maga_init_params_.parallelism_config;
    const auto  self_dp_rank = static_cast<int32_t>(cfg.dp_rank);
    const auto  dp_size      = cfg.dp_size;
    const auto& peer_addrs   = cfg.dp_peer_addrs;
    response->set_batch_id(request->batch_id());

    {
        std::string addrs_dump;
        for (size_t i = 0; i < peer_addrs.size(); ++i) {
            if (i) addrs_dump += ",";
            addrs_dump += peer_addrs[i];
        }
        RTP_LLM_LOG_INFO(
            "BatchEnqueue batch=%ld self_dp=%d dp_size=%d peer_addrs.size=%zu addrs=[%s] slots=%d dp_controller=%d",
            request->batch_id(), self_dp_rank, dp_size, peer_addrs.size(), addrs_dump.c_str(),
            request->inputs_size(), cfg.dp_controller_managed ? 1 : 0);
    }

    const int slot_num = request->inputs_size();
    response->mutable_acks()->Reserve(slot_num);
    for (int i = 0; i < slot_num; ++i) {
        response->add_acks();
    }

    auto fill_error = [](EnqueueResponsePB* ack, int64_t request_id, int64_t code, const std::string& msg) {
        ack->set_request_id(request_id);
        auto* err = ack->mutable_error_info();
        err->set_error_code(code);
        err->set_error_message(msg);
    };

    // Part C — reorder fan-out so DP0 (the BatchEnqueue receiver) does not
    // enter forward before peer DPs have queued their slots. Phase 1 launches
    // all peer Enqueues async, phase 2 waits for them, phase 3 finally does
    // local self-Enqueue. This guarantees the peer's FIFOScheduler has the
    // stream queued before DP0's engine loop can pick up its own — closing the
    // request-level timing skew that would otherwise cause DP0's first DeepEP
    // dispatch to find no peer and CPU-recv-timeout.
    std::vector<std::future<void>> peer_tasks;
    peer_tasks.reserve(slot_num);

    // Deadline ceiling: peer Enqueue must not hang forever — if a peer pod is
    // unreachable or stuck, task.get() would block phase 2 indefinitely and
    // freeze DP0's engine loop. Mirror remoteAllocateResource (line 190-196):
    // request.timeout_ms > 0 > max_rpc_timeout_ms > MAX_GRPC_TIMEOUT_MS.
    const auto max_rpc_timeout_ms = maga_init_params_.pd_sep_config.max_rpc_timeout_ms;
    const auto default_timeout_ms = max_rpc_timeout_ms > 0 ? max_rpc_timeout_ms : MAX_GRPC_TIMEOUT_MS;

    // Phase 1: async fan-out to every non-self slot.
    for (int i = 0; i < slot_num; ++i) {
        const auto&   input     = request->inputs(i);
        const int32_t slot_rank = input.has_dp_rank() ? input.dp_rank().value() : self_dp_rank;
        auto*         ack       = response->mutable_acks(i);

        const bool is_self = (slot_rank == self_dp_rank || dp_size <= 1 || !cfg.dp_controller_managed);
        if (is_self) {
            continue;  // deferred to phase 3
        }

        if (slot_rank < 0 || static_cast<size_t>(slot_rank) >= peer_addrs.size()) {
            fill_error(ack,
                       input.request_id(),
                       grpc::StatusCode::INVALID_ARGUMENT,
                       "dp_rank " + std::to_string(slot_rank) + " out of range [0, " + std::to_string(peer_addrs.size())
                           + ")");
            continue;
        }

        int64_t slot_timeout_ms = input.generate_config().timeout_ms();
        if (slot_timeout_ms <= 0) {
            slot_timeout_ms = default_timeout_ms;
        }
        auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(slot_timeout_ms);

        const std::string peer = peer_addrs[slot_rank];
        peer_tasks.emplace_back(
            std::async(std::launch::async, [this, &input, ack, peer, deadline, slot_timeout_ms, fill_error]() {
                auto connect_status = resource_.rpc_pool.getConnection(peer);
                if (!connect_status.ok()) {
                    RTP_LLM_LOG_WARNING("BatchEnqueue peer connect failed: request [%ld] peer [%s] msg [%s]",
                                        input.request_id(),
                                        peer.c_str(),
                                        std::string(connect_status.status().message()).c_str());
                    fill_error(ack,
                               input.request_id(),
                               grpc::StatusCode::UNAVAILABLE,
                               "get grpc connection for peer " + peer
                                   + " failed: " + std::string(connect_status.status().message()));
                    return;
                }
                EnqueueRequestPB peer_req;
                *peer_req.mutable_input() = input;
                grpc::ClientContext ctx;
                ctx.set_deadline(deadline);
                auto peer_status = connect_status.value().stub->Enqueue(&ctx, peer_req, ack);
                if (!peer_status.ok()) {
                    RTP_LLM_LOG_WARNING(
                        "BatchEnqueue peer Enqueue failed: request [%ld] peer [%s] timeout_ms [%ld] code [%d] msg [%s]",
                        input.request_id(),
                        peer.c_str(),
                        slot_timeout_ms,
                        static_cast<int>(peer_status.error_code()),
                        peer_status.error_message().c_str());
                    fill_error(ack, input.request_id(), peer_status.error_code(), peer_status.error_message());
                }
            }));
    }

    // Phase 2: wait for every peer Enqueue to return. The remote handler only
    // returns after it has placed the slot into its FIFOScheduler (real path
    // also runs syncPrefix; fake path is constant-time createMinFakeStream +
    // engine_->enqueue). When this loop exits, every peer DP has its slot
    // visible to its engine loop.
    for (auto& task : peer_tasks) {
        task.get();
    }

    // Phase 3: local self-Enqueue with two-pass structure.
    // For force_batch groups (dpSize=1), the old sequential Enqueue() deadlocks:
    // each Enqueue blocks at waitStreamBeforeRun until the scheduler admits the
    // stream, but the scheduler waits for the full group. Two passes fix this:
    //   Pass A: prepareAllocateResource + enqueueRequest for all self-slots
    //   Pass B: remoteLoadCacheStart + finishStream (group now complete, instant admit)
    struct LocalSlot {
        int                                     index;
        std::shared_ptr<RPCContext>             rpc_context;
        std::shared_ptr<PrefillGenerateContext> prefill_context;
        AtomicGuardPtr                          request_guard;
        int64_t                                 request_id;
    };
    std::vector<LocalSlot> local_slots;
    local_slots.reserve(slot_num);

    auto max_retry_times      = maga_init_params_.pd_sep_config.prefill_retry_times;
    auto max_retry_timeout_ms = maga_init_params_.pd_sep_config.prefill_retry_timeout_ms;

    // Phase 3a: prepare + enqueue all self-slots into scheduler
    for (int i = 0; i < slot_num; ++i) {
        const auto&   input     = request->inputs(i);
        const int32_t slot_rank = input.has_dp_rank() ? input.dp_rank().value() : self_dp_rank;
        auto*         ack       = response->mutable_acks(i);

        const bool is_self = (slot_rank == self_dp_rank || dp_size <= 1 || !cfg.dp_controller_managed);
        if (!is_self) {
            continue;
        }

        auto req_id = input.request_id();
        ack->set_request_id(req_id);

        if (input.is_fake_query()) {
            auto fake_stream = engine_->createMinFakeStream(/*max_new_tokens=*/1);
            RTP_LLM_LOG_INFO("BatchEnqueue fake_query req=%ld dp_rank=%d", req_id, slot_rank);
            if (fake_stream) {
                engine_->enqueue(fake_stream);
            }
            continue;
        }

        if (response_registry_.get(req_id) != nullptr) {
            fill_error(ack, req_id, grpc::StatusCode::ALREADY_EXISTS, "already enqueued");
            continue;
        }

        auto rpc_ctx = std::make_shared<RPCContext>(RPCContext{&input, nullptr});
        auto pfx_ctx = std::make_shared<PrefillGenerateContext>(
            &this->resource(), *rpc_ctx, input.generate_config().timeout_ms(),
            /*server_context=*/nullptr, metrics_reporter_, meta_);
        pfx_ctx->onflight_requests      = onflight_requests_;
        pfx_ctx->loading_cache_requests = loading_cache_requests_;
        auto guard = std::make_shared<AtomicGuard>(onflight_requests_);

        // prepareAllocateResource with retry (mirrors syncPrefix logic)
        bool alloc_ok = false;
        {
            int64_t begin_time_us = currentTimeUs();
            auto    stage         = pfx_ctx->stat_info.saveStage();
            for (int attempt = 0; attempt <= max_retry_times; ++attempt) {
                pfx_ctx->reset();
                pfx_ctx->stat_info.restoreStage(stage);
                pfx_ctx->retry_times++;
                prepareAllocateResource(*pfx_ctx);
                if (pfx_ctx->ok()) {
                    alloc_ok = true;
                    break;
                }
                auto cost_time_us             = currentTimeUs() - begin_time_us;
                pfx_ctx->retry_cost_time_ms   = cost_time_us / 1000;
                if (max_retry_timeout_ms > 0 && cost_time_us >= max_retry_timeout_ms * 1000) {
                    break;
                }
                usleep(1000);  // 1ms retry interval
            }
        }
        if (!alloc_ok) {
            RTP_LLM_LOG_WARNING("BatchEnqueue req=%ld prepareAllocateResource failed after %ld retries",
                                req_id, pfx_ctx->retry_times);
            fill_error(ack, req_id, grpc::StatusCode::INTERNAL,
                       pfx_ctx->error_info.ToString());
            continue;
        }

        // enqueueRequest: adds stream to FIFOScheduler's waiting_streams_ (instant)
        pfx_ctx->stat_info.nextStage();
        enqueueRequest(*pfx_ctx);
        if (pfx_ctx->hasError()) {
            fill_error(ack, req_id, grpc::StatusCode::INTERNAL,
                       pfx_ctx->error_info.ToString());
            continue;
        }

        local_slots.push_back({i, rpc_ctx, pfx_ctx, guard, req_id});
    }

    // Phase 3b: remoteLoadCacheStart + spawn finishStream for each successful slot.
    // All streams are now in the scheduler — force_batch group is complete, CanRun fires.
    for (auto& slot : local_slots) {
        auto* ack     = response->mutable_acks(slot.index);
        auto& pfx_ctx = slot.prefill_context;

        pfx_ctx->stat_info.nextStage();
        remoteLoadCacheStart(*pfx_ctx);
        if (pfx_ctx->hasError()) {
            fill_error(ack, slot.request_id, grpc::StatusCode::INTERNAL,
                       pfx_ctx->error_info.ToString());
            continue;
        }

        auto entry  = response_registry_.create(slot.request_id);
        auto writer = std::make_shared<ResponseBufferWriter>(entry);
        pfx_ctx->rpc_context.writer = writer.get();

        std::thread worker(
            [this, pfx_ctx = slot.prefill_context, rpc_ctx = slot.rpc_context,
             writer, entry, guard = slot.request_guard, rid = slot.request_id]() mutable {
                grpc::Status finish_status;
                try {
                    finish_status = finishStream(*pfx_ctx);
                } catch (const std::exception& e) {
                    finish_status = grpc::Status(grpc::StatusCode::INTERNAL, e.what());
                } catch (...) {
                    finish_status = grpc::Status(grpc::StatusCode::INTERNAL, "finishStream unknown exception");
                }
                if (!finish_status.ok()) {
                    std::lock_guard<std::mutex> lock(entry->mu);
                    entry->error_status = finish_status;
                }
                entry->done.store(true);
                entry->cv.notify_all();
                RTP_LLM_LOG_DEBUG("BatchEnqueue req=%ld finishStream done ok=%d", rid, finish_status.ok());
            });
        worker.detach();
    }

    RTP_LLM_LOG_INFO("BatchEnqueue batch=%ld self_dp=%d phase3_local_done", request->batch_id(), self_dp_rank);
    return grpc::Status::OK;
}

grpc::Status PrefillRpcServer::FetchResponse(grpc::ServerContext*                   context,
                                             const FetchRequestPB*                  request,
                                             grpc::ServerWriter<GenerateOutputsPB>* writer) {
    RTP_LLM_PROFILE_FUNCTION();
    auto request_id = request->request_id();
    auto entry      = response_registry_.get(request_id);
    if (!entry) {
        auto msg = "request [" + std::to_string(request_id) + "] not found in response registry";
        return grpc::Status(grpc::StatusCode::NOT_FOUND, msg);
    }

    while (true) {
        if (context && context->IsCancelled()) {
            entry->cancelled.store(true);
            entry->cv.notify_all();
            response_registry_.erase(request_id);
            return grpc::Status(grpc::StatusCode::CANCELLED, "fetch response cancelled by client");
        }

        std::deque<GenerateOutputsPB> drained;
        grpc::Status                  terminal_status = grpc::Status::OK;
        bool                          terminal        = false;
        {
            std::unique_lock<std::mutex> lock(entry->mu);
            entry->cv.wait_for(lock, std::chrono::milliseconds(100), [&] {
                return !entry->queue.empty() || entry->done.load() || entry->cancelled.load()
                       || entry->error_status.has_value();
            });
            drained.swap(entry->queue);
            if (entry->cancelled.load()) {
                terminal        = true;
                terminal_status = grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled");
            } else if (entry->error_status.has_value()) {
                terminal        = true;
                terminal_status = *entry->error_status;
            } else if (entry->done.load()) {
                terminal = true;
            }
        }

        for (auto& out : drained) {
            if (!writer->Write(out)) {
                entry->cancelled.store(true);
                entry->cv.notify_all();
                response_registry_.erase(request_id);
                return grpc::Status(grpc::StatusCode::CANCELLED, "client writer closed");
            }
        }

        if (terminal) {
            response_registry_.erase(request_id);
            return terminal_status;
        }
    }
}

grpc::Status PrefillRpcServer::Cancel(grpc::ServerContext* context, const CancelRequestPB* request, EmptyPB* response) {
    RTP_LLM_PROFILE_FUNCTION();
    auto request_id = request->request_id();
    auto entry      = response_registry_.get(request_id);
    if (!entry) {
        // Idempotent: unknown or already-completed request_ids return OK
        // (model_rpc_service.proto Cancel contract).
        return grpc::Status::OK;
    }
    entry->cancelled.store(true);
    entry->cv.notify_all();
    // FetchResponse holds its own shared_ptr to the entry; erasing from the
    // registry only drops the map's reference. Any in-flight FetchResponse
    // wakes up via cv.notify_all, takes the entry->cancelled branch, and
    // calls erase a second time — which is a no-op.
    response_registry_.erase(request_id);
    return grpc::Status::OK;
}

}  // namespace rtp_llm

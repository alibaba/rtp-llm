#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/Host.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include <cstring>
#include <memory>
#include <unistd.h>
#include <limits.h>
#include <c10/core/InferenceMode.h>

using namespace std;
using namespace autil::legacy;

using grpc::Status;
using grpc::ClientContext;

namespace rtp_llm {

namespace {

class BatchOutputCollector: public grpc::internal::WriterInterface<GenerateOutputsPB> {
public:
    bool Write(const GenerateOutputsPB& output, grpc::WriteOptions) override {
        if (!local_output_complete_) {
            local_output_.CopyFrom(output);
            has_local_output_ = true;
        } else {
            remote_output_.CopyFrom(output);
            has_remote_output_ = true;
        }
        return true;
    }

    void markLocalOutputComplete() {
        local_output_complete_ = true;
    }

    bool hasOutput() const {
        return has_local_output_ || has_remote_output_;
    }

    GenerateOutputsPB finalOutput() const {
        if (!has_remote_output_) {
            return local_output_;
        }
        GenerateOutputsPB output = remote_output_;
        if (!has_local_output_) {
            return output;
        }

        const auto& local_ids  = local_output_.flatten_output().output_ids();
        const auto& remote_ids = remote_output_.flatten_output().output_ids();
        if (local_ids.shape_size() == 0) {
            return output;
        }
        if (remote_ids.shape_size() == 0) {
            output.mutable_flatten_output()->mutable_output_ids()->CopyFrom(local_ids);
            return output;
        }

        auto local_tensor  = QueryConverter::transTensor(local_ids);
        auto remote_tensor = QueryConverter::transTensor(remote_ids);
        RTP_LLM_CHECK_WITH_INFO(local_tensor.dim() > 0 && local_tensor.dim() == remote_tensor.dim(),
                                "PD batch output ids rank mismatch, got local=%ld remote=%ld",
                                local_tensor.dim(),
                                remote_tensor.dim());
        const auto token_dim = local_tensor.dim() - 1;
        for (int64_t dim = 0; dim < token_dim; ++dim) {
            RTP_LLM_CHECK_WITH_INFO(local_tensor.size(dim) == remote_tensor.size(dim),
                                    "PD batch output ids shape mismatch at dim=%ld, local=%ld remote=%ld",
                                    dim,
                                    local_tensor.size(dim),
                                    remote_tensor.size(dim));
        }
        auto  merged     = torch::cat({local_tensor, remote_tensor}, token_dim).contiguous();
        auto* merged_ids = output.mutable_flatten_output()->mutable_output_ids();
        merged_ids->Clear();
        QueryConverter::transTensorPB(merged_ids, merged);
        return output;
    }

private:
    GenerateOutputsPB local_output_;
    GenerateOutputsPB remote_output_;
    bool              local_output_complete_{false};
    bool              has_local_output_{false};
    bool              has_remote_output_{false};
};

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
                                    std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                                    py::object                                             mm_process_engine) {
    RTP_LLM_CHECK_WITH_INFO(maga_init_params.pd_sep_config.role_type == RoleType::PREFILL,
                            "prefill's role_type must be PREFILL");
    auto ret = RemoteRpcServer::init(maga_init_params, std::move(propose_params), mm_process_engine);
    if (!ret.ok()) {
        return ret;
    }
    return grpc::Status::OK;
}

bool PrefillRpcServer::canUsePDSep(const GenerateInputPB& request) const {
    const auto& config = request.generate_config();
    const bool  has_prefill_only_output =
        config.calculate_loss() != 0 || config.return_hidden_states() || config.return_all_hidden_states()
        || config.return_logits() || config.return_all_probs() || config.return_all_probs_mode() > 1
        || config.return_softmax_probs() || config.return_cum_log_probs() || config.return_prompt_logits();
    return config.max_new_tokens() > 1 && config.num_beams() <= 1 && config.variable_num_beams().size() == 0
           && config.num_return_sequences() <= 1 && config.can_use_pd_separation() && !has_prefill_only_output;
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
    auto input                            = QueryConverter::transQuery(prefill_context.rpc_context.request);
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
    CLIENT_GRPC_RET_IF_ERROR(prefill_context, error_code == ErrorCode::NONE_ERROR, error_code);
    RTP_LLM_LOG_DEBUG("request [%ld] remote load cache done", prefill_context.request_id);

    // Decode has finished loading cache, now safe to release KV cache blocks.
    // This is called after cache store transfer is complete.
    if (prefill_context.generate_input->generate_config->pd_separation) {
        prefill_context.getStream()->releaseKVCacheForPDSep();
    }
}

void PrefillRpcServer::remoteGenerate(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%ld] start to remote generate", prefill_context.request_id);
    std::shared_ptr<GenerateStream> stream = prefill_context.getStream();
    RTP_LLM_LOG_DEBUG("remote generate stream[%ld]: %s", stream->streamId(), stream->debugString().c_str());
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
        if (prefill_context.server_context->IsCancelled()) {
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

grpc::Status PrefillRpcServer::GenerateStreamCall(grpc::ServerContext*                   server_context,
                                                  const GenerateInputPB*                 request,
                                                  grpc::ServerWriter<GenerateOutputsPB>* writer) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%ld] start generate stream call", request->request_id());
    c10::InferenceMode inference_guard(true);
    if (!canUsePDSep(*request)) {
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

    auto max_retry_times      = maga_init_params_.pd_sep_config.prefill_retry_times;
    auto max_retry_timeout_ms = maga_init_params_.pd_sep_config.prefill_retry_timeout_ms;
    int  retry_interval_ms    = 1;

    try {
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
        EXECUTE_STAGE_FUNC(pollLocalOutput, prefill_context);
        EXECUTE_STAGE_FUNC(remoteLoadCacheEnd, prefill_context);
        meta_->dequeue(prefill_context.request_id, prefill_context.getStream());
        EXECUTE_STAGE_FUNC(remoteGenerate, prefill_context);
        EXECUTE_STAGE_FUNC(pollRemoteOutput, prefill_context);
        prefill_context.stat_info.nextStage();
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

grpc::Status PrefillRpcServer::BatchGenerateCall(grpc::ServerContext*        server_context,
                                                 const BatchGenerateInputPB* request,
                                                 BatchGenerateOutputsPB*     response) {
    RTP_LLM_PROFILE_SCOPE("rpc.prefill_batch_generate_call");
    c10::InferenceMode inference_guard(true);
    const int          batch_size = request->inputs_size();
    if (batch_size == 0) {
        return grpc::Status::OK;
    }

    bool has_pd_request     = false;
    bool has_non_pd_request = false;
    for (int i = 0; i < batch_size; ++i) {
        if (canUsePDSep(request->inputs(i))) {
            has_pd_request = true;
        } else {
            has_non_pd_request = true;
        }
    }
    if (!has_pd_request) {
        return LocalRpcServer::BatchGenerateCall(server_context, request, response);
    }
    if (has_non_pd_request) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                            "mixing PD and non-PD requests in one atomic batch is not supported");
    }

    RTP_LLM_LOG_INFO("receive PD batch generate request, batch_size=%d", batch_size);
    AtomicGuardPtr request_guard = make_shared<AtomicGuard>(onflight_requests_);

    std::vector<std::unique_ptr<BatchOutputCollector>>   collectors;
    std::vector<std::unique_ptr<PrefillGenerateContext>> contexts;
    collectors.reserve(batch_size);
    contexts.reserve(batch_size);

    const auto    max_retry_times      = maga_init_params_.pd_sep_config.prefill_retry_times;
    const auto    max_retry_timeout_ms = maga_init_params_.pd_sep_config.prefill_retry_timeout_ms;
    constexpr int retry_interval_ms    = 1;

    for (int i = 0; i < batch_size; ++i) {
        auto       collector = std::make_unique<BatchOutputCollector>();
        RPCContext rpc_context{&request->inputs(i), collector.get()};
        auto       context              = std::make_unique<PrefillGenerateContext>(&this->resource(),
                                                                rpc_context,
                                                                request->inputs(i).generate_config().timeout_ms(),
                                                                server_context,
                                                                metrics_reporter_,
                                                                meta_);
        context->onflight_requests      = onflight_requests_;
        context->loading_cache_requests = loading_cache_requests_;

        auto& prefill_context = *context;
        EXECUTE_WITH_RETRY(
            prepareAllocateResource, prefill_context, max_retry_times, max_retry_timeout_ms, retry_interval_ms);
        CHECK_ERROR_STATUS(prefill_context);

        collectors.emplace_back(std::move(collector));
        contexts.emplace_back(std::move(context));
    }

    std::vector<std::shared_ptr<GenerateInput>> inputs;
    inputs.reserve(batch_size);
    for (const auto& context : contexts) {
        inputs.emplace_back(context->generate_input);
    }
    auto streams = engine_->batchEnqueue(inputs);
    if (streams.size() != contexts.size()) {
        return grpc::Status(grpc::StatusCode::INTERNAL,
                            "batchEnqueue returned " + std::to_string(streams.size()) + " streams for "
                                + std::to_string(contexts.size()) + " inputs");
    }
    for (size_t i = 0; i < contexts.size(); ++i) {
        contexts[i]->setStream(streams[i]);
        contexts[i]->stat_info.nextStage();  // enqueueRequest
    }

    for (auto& context : contexts) {
        auto& prefill_context = *context;
        EXECUTE_STAGE_FUNC(remoteLoadCacheStart, prefill_context);
    }
    for (auto& context : contexts) {
        auto& prefill_context = *context;
        EXECUTE_STAGE_FUNC(pollLocalOutput, prefill_context);
    }
    for (auto& collector : collectors) {
        collector->markLocalOutputComplete();
    }
    for (auto& context : contexts) {
        auto& prefill_context = *context;
        EXECUTE_STAGE_FUNC(remoteLoadCacheEnd, prefill_context);
        meta_->dequeue(context->request_id, context->getStream());
        EXECUTE_STAGE_FUNC(remoteGenerate, prefill_context);
    }
    for (auto& context : contexts) {
        auto& prefill_context = *context;
        EXECUTE_STAGE_FUNC(pollRemoteOutput, prefill_context);
        context->stat_info.nextStage();
    }

    for (size_t i = 0; i < collectors.size(); ++i) {
        auto* result = response->add_results();
        if (!collectors[i]->hasOutput()) {
            auto* error = result->mutable_error_info();
            error->set_error_code(ErrorCodePB::UNKNOWN_ERROR);
            error->set_error_message("PD batch item completed without an output");
            continue;
        }
        result->mutable_final_output()->CopyFrom(collectors[i]->finalOutput());
    }

    RTP_LLM_LOG_INFO("PD batch generate done, batch_size=%d", batch_size);
    return grpc::Status::OK;
}

grpc::Status
PrefillRpcServer::RemoteFinish(grpc::ServerContext* context, const RemoteFinishRequestPB* request, EmptyPB* response) {
    RTP_LLM_PROFILE_FUNCTION();
    auto request_id = request->request_id();
    resource_.cache_store->markRequestEnd(std::to_string(request_id));
    return grpc::Status::OK;
}

}  // namespace rtp_llm

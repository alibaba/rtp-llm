#include <mutex>
#include <memory>
#include <unistd.h>
#include <limits.h>
#include <condition_variable>
#include <future>
#include <c10/core/InferenceMode.h>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/CacheTransferBlockSelector.h"
#include "rtp_llm/cpp/model_rpc/CacheTransferLease.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PrefillPeerSelector.h"
#include "rtp_llm/cpp/model_rpc/PropagatedClientContext.h"
#include "rtp_llm/cpp/model_rpc/RequestDeadlineBudget.h"
#include "rtp_llm/cpp/model_rpc/RemoteLoadBudget.h"
#include "rtp_llm/cpp/model_rpc/RpcFanoutUtils.h"
#include "rtp_llm/cpp/model_rpc/SamplerGeneratorState.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "autil/LockFreeThreadPool.h"

using namespace std;
using namespace autil::legacy;

using grpc::Status;
using grpc::ClientContext;
using grpc::CompletionQueue;
using grpc::ClientAsyncResponseReader;

const int LOAD_TIMEOUT_MS         = 5 * 1000;
const int RDMA_CONNECT_RETRY_TIME = 3;

#define GRPC_RET_IF_ERROR(decode_context, stat, code, msg)                                                             \
    if (!(stat)) {                                                                                                     \
        const auto stop_error = serverContextStopError(decode_context.server_context, msg);                           \
        if (stop_error.hasError()) {                                                                                   \
            decode_context.error_info   = stop_error;                                                                  \
            decode_context.error_status = serializeErrorMsg(decode_context.request_key, stop_error);                  \
        } else {                                                                                                       \
            decode_context.error_status = grpc::Status(code, msg);                                                     \
        }                                                                                                              \
        return;                                                                                                        \
    }

string makeRequestKey(const string& client_id, size_t request_id) {
    return client_id + "_request_id_" + std::to_string(request_id);
}

namespace rtp_llm {

namespace {

void setRemoteLoadResponseError(BroadcastLoadResponsePB* response, const ErrorInfo& error_info) {
    response->mutable_error_info()->set_error_code(transErrorCodeToRPC(error_info.code()));
    response->mutable_error_info()->set_error_message(error_info.ToString());
}

void setRemoteLoadQuiesceResponseError(RemoteLoadQuiesceResponsePB* response, const ErrorInfo& error_info) {
    response->mutable_error_info()->set_error_code(transErrorCodeToRPC(error_info.code()));
    response->mutable_error_info()->set_error_message(error_info.ToString());
}

std::chrono::milliseconds retentionTimeoutUntil(int64_t deadline_unix_ms) {
    const auto now_unix_ms = remoteLoadUnixMillis(RemoteLoadSystemClock::now());
    const auto remaining_ms = static_cast<__int128>(deadline_unix_ms) - now_unix_ms;
    if (remaining_ms <= 1) {
        return std::chrono::milliseconds(1);
    }
    return std::chrono::milliseconds(clampRequestDeadlineToInt64(remaining_ms));
}

}  // namespace

grpc::Status DecodeRpcServer::init(const EngineInitParams&                                maga_init_params,
                                   std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                                   py::object                                             mm_process_engine) {
    auto ret = RemoteRpcServer::init(maga_init_params, std::move(propose_params), mm_process_engine);
    if (!ret.ok()) {
        return ret;
    }
    return grpc::Status::OK;
}

void DecodeRpcServer::initThreadPool() {
    if (resource_.workers.size() > 0) {
        return;
    }
    thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(
        resource_.workers.size() * 8, resource_.workers.size() * 8, nullptr, "RemoteCacheLoadPool");
    RTP_LLM_CHECK_WITH_INFO(thread_pool_->start(), "DecodeRpcServer init ThreadPool failed");
    RTP_LLM_LOG_INFO("normal cache store init done");
}

DecodeRpcServer::~DecodeRpcServer() {
    if (thread_pool_) {
        thread_pool_->stop();
        thread_pool_.reset();
    }
}

bool DecodeRpcServer::drainRemoteLoads() {
    return drainRemoteLoads(std::chrono::milliseconds(LOAD_TIMEOUT_MS));
}

bool DecodeRpcServer::drainRemoteLoads(std::chrono::milliseconds grace) {
    return remote_load_leases_.stop(grace);
}

void DecodeRpcServer::stop() {
    RTP_LLM_CHECK_WITH_INFO(drainRemoteLoads(), "remote load leases failed to quiesce before decode stop");
    if (thread_pool_) {
        thread_pool_->stop();
        thread_pool_.reset();
    }
    LocalRpcServer::stop();
}

void DecodeRpcServer::prepareGenerateContext(DecodeGenerateContext& decode_context) {
    RTP_LLM_PROFILE_FUNCTION();
    decode_context.time_info.updateRequestBegineTime();
    auto& allocate_request = decode_context.allocate_request;
    GRPC_RET_IF_ERROR(decode_context,
                      decode_context.rpc_context.grpc_stream->Read(&allocate_request),
                      grpc::StatusCode::INTERNAL,
                      "failed to get message");
    GRPC_RET_IF_ERROR(decode_context,
                      allocate_request.stage() == RemoteStage::ALLOCATE,
                      grpc::StatusCode::INTERNAL,
                      "message first status != RemoteStage::ALLOCATE");
    decode_context.request_id  = allocate_request.request_id();
    decode_context.request_key = makeRequestKey(allocate_request.client_id(), allocate_request.request_id());

    for (auto& addr : allocate_request.peer_addrs()) {
        decode_context.peer_addrs.push_back(addr);
    }
    RTP_LLM_LOG_DEBUG("request [%s] prepare generate context done", decode_context.request_key.c_str());
}

void DecodeRpcServer::allocateResource(DecodeGenerateContext& decode_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%s] start to allocate resource", decode_context.request_key.c_str());
    const auto initial_stop_error =
        serverContextStopError(decode_context.server_context, "before decode allocation");
    if (initial_stop_error.hasError()) {
        decode_context.error_info   = initial_stop_error;
        decode_context.error_status = serializeErrorMsg(decode_context.request_key, initial_stop_error);
        return;
    }
    const auto& input_pb = decode_context.allocate_request.input();
    const auto authoritative_deadline_us =
        decode_context.server_context == nullptr ? 0 : requestDeadlineUnixUs(decode_context.server_context->deadline());
    const auto deadline_budget = makeRequestDeadlineBudget(input_pb.request_deadline_unix_ms(),
                                                           input_pb.generate_config().timeout_ms(),
                                                           currentTimeUs(),
                                                           authoritative_deadline_us);
    if (deadline_budget.expired) {
        decode_context.error_info =
            ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "request deadline expired before decode allocation");
        decode_context.error_status = serializeErrorMsg(decode_context.request_key, decode_context.error_info);
        return;
    }
    auto input                        = QueryConverter::transQuery(&input_pb, &deadline_budget);
    auto generate_stream              = engine_->makeStream(input);
    decode_context.request_timeout_ms = generate_stream->getTimeoutMs();
    decode_context.setStream(generate_stream);

    auto finish_stream = [&generate_stream](ErrorCode error_code, const string& error_msg) {
        generate_stream->reportError(error_code, error_msg);
        generate_stream->moveToNext();
    };
    auto reject_stopped_request = [&](const std::string& operation) {
        const auto now_unix_us = currentTimeUs();
        auto       stop_error  = requestDeadlineReached(
                                    input->begin_time_us, input->generate_config->timeout_ms, now_unix_us) ?
                                     ErrorInfo(ErrorCode::GENERATE_TIMEOUT, operation + ": request deadline expired") :
                                     serverContextStopError(decode_context.server_context, operation, now_unix_us);
        if (!stop_error.hasError()) {
            return false;
        }
        finish_stream(stop_error.code(), stop_error.ToString());
        decode_context.error_info   = stop_error;
        decode_context.error_status = serializeErrorMsg(decode_context.request_key, stop_error);
        return true;
    };

    if (reject_stopped_request("before decode cache allocation")) {
        return;
    }

    // Decode owns an explicit P->D cache transfer, so its destination blocks must exist before the
    // transfer starts. Drive the state machine to do that: handleWaiting() runs initKVBlock() and
    // asyncLoadCache(), and LoadInitiated stays the state machine's own marker for "both attempted".
    // Admission still belongs to the scheduler, after the transfer and first-token handoff.
    // The busy-wait is safe because the stream is not enqueued yet, so this gRPC thread is the only
    // one driving moveToNext().
    generate_stream->reportEvent(StreamEvents::CanRun);
    while (!generate_stream->hasError() && generate_stream->moveToNext() == StreamState::LOADING_CACHE) {
        this_thread::sleep_for(chrono::milliseconds(1));
    }
    if (generate_stream->hasError()) {
        auto   stream_error = generate_stream->statusInfo();
        string error_msg    = stream_error.ToString();
        if (error_msg.empty()) {
            error_msg = "malloc kv cache block failed at decode node";
        }
        error_msg = "request: [" + decode_context.request_key + "] " + error_msg;
        RTP_LLM_LOG_ERROR(error_msg);
        generate_stream->moveToNext();
        decode_context.error_info   = ErrorInfo(ErrorCode::MALLOC_FAILED, error_msg);
        decode_context.error_status = grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, error_msg);
        return;
    }

    if (reject_stopped_request("after decode allocation")) {
        return;
    }

    auto cache_manager = engine_->resourceContext().cache_manager;
    auto kv_cache      = generate_stream->kvCachePtr();
    if (kv_cache == nullptr || kv_cache->batchSize() <= 0) {
        const string error_msg = "request: [" + decode_context.request_key
                                 + "] decode allocation produced no cache resource";
        finish_stream(ErrorCode::MALLOC_FAILED, error_msg);
        decode_context.error_info   = ErrorInfo(ErrorCode::MALLOC_FAILED, error_msg);
        decode_context.error_status = grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, error_msg);
        return;
    }

    const auto& allocated_resource = kv_cache->cacheResource(0);
    auto lease_resource = makeCacheTransferLeaseResource(cache_manager->cacheConfig().groupNums(),
                                                         allocated_resource.groupBlocks(),
                                                         allocated_resource.cacheKeys().size(),
                                                         cache_manager->totalBlocksNum());
    if (!lease_resource.ok()) {
        const string error_msg = "request: [" + decode_context.request_key
                                 + "] invalid decode allocation lease: " + lease_resource.status().ToString();
        finish_stream(ErrorCode::MALLOC_FAILED, error_msg);
        decode_context.error_info   = ErrorInfo(ErrorCode::MALLOC_FAILED, error_msg);
        decode_context.error_status = grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, error_msg);
        return;
    }
    std::shared_ptr<KVCacheResource> master_cache_lease;
    if (!allocated_resource.cacheKeys().empty()) {
        master_cache_lease =
            cache_manager->incrKVCacheRef(*lease_resource, lease_resource->cacheKeys(), /*is_connector=*/true);
        if (master_cache_lease == nullptr) {
            const string error_msg = "request: [" + decode_context.request_key
                                     + "] failed to retain decode allocation";
            finish_stream(ErrorCode::MALLOC_FAILED, error_msg);
            decode_context.error_info   = ErrorInfo(ErrorCode::MALLOC_FAILED, error_msg);
            decode_context.error_status = grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, error_msg);
            return;
        }
    }

    if (reject_stopped_request("before retaining decode allocation")) {
        return;
    }

    const auto now = std::chrono::system_clock::now();
    auto       max_rpc_timeout_ms = maga_init_params_.pd_sep_config.max_rpc_timeout_ms;
    auto       allocation_timeout_ms =
        decode_context.request_timeout_ms > 0 ? decode_context.request_timeout_ms :
                                               (max_rpc_timeout_ms > 0 ? max_rpc_timeout_ms : MAX_GRPC_TIMEOUT_MS);
    const auto configured_load_timeout_ms = maga_init_params_.pd_sep_config.load_cache_timeout_ms;
    const auto load_timeout_ms = configured_load_timeout_ms > 0 ? configured_load_timeout_ms : LOAD_TIMEOUT_MS;
    allocation_timeout_ms      = std::max<int64_t>(1, std::min(allocation_timeout_ms, load_timeout_ms));
    auto load_deadline = now + std::chrono::milliseconds(allocation_timeout_ms);
    if (decode_context.server_context != nullptr) {
        load_deadline = std::min(load_deadline, decode_context.server_context->deadline());
    }
    decode_context.load_deadline_unix_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(load_deadline.time_since_epoch()).count();
    const auto token_identity = decode_context.request_key + ":"
                                + std::to_string(allocation_token_counter_.fetch_add(1));
    auto allocation_token =
        makeRemoteLoadAllocationToken(process_id_, token_identity, decode_context.load_deadline_unix_ms);
    if (!allocation_token.ok()) {
        const string error_msg = "request: [" + decode_context.request_key
                                 + "] failed to create allocation token: " + allocation_token.status().ToString();
        finish_stream(ErrorCode::MALLOC_FAILED, error_msg);
        decode_context.error_info   = ErrorInfo(ErrorCode::MALLOC_FAILED, error_msg);
        decode_context.error_status = grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
        return;
    }
    decode_context.allocation_token = std::move(*allocation_token);
    decode_context.remote_load_targets =
        std::make_shared<RemoteLoadTargetTracker>(resource_.grpc_workers);

    if (master_cache_lease != nullptr) {
        const auto remote_load_targets = decode_context.remote_load_targets;
        auto       ticket = remote_load_leases_.reserve(
            decode_context.allocation_token,
            std::move(master_cache_lease),
            [this,
             allocation_token = decode_context.allocation_token,
             load_deadline_unix_ms = decode_context.load_deadline_unix_ms,
             remote_load_targets]() {
                return quiesceRemoteLoadTargets(allocation_token,
                                                load_deadline_unix_ms,
                                                remote_load_targets->startedTargets(),
                                                std::chrono::milliseconds(LOAD_TIMEOUT_MS),
                                                retentionTimeoutUntil(load_deadline_unix_ms),
                                                RemoteLoadFenceRegistry::UnseenTokenPolicy::Seal);
            });
        if (!ticket.ok()) {
            const string error_msg = "request: [" + decode_context.request_key
                                     + "] failed to retain decode allocation: " + ticket.status().ToString();
            finish_stream(ErrorCode::MALLOC_FAILED, error_msg);
            decode_context.error_info   = ErrorInfo(ErrorCode::MALLOC_FAILED, error_msg);
            decode_context.error_status = grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, error_msg);
            return;
        }
        decode_context.cache_lease_ticket = std::move(*ticket);
    }

    if (reject_stopped_request("before writing decode allocation")) {
        return;
    }

    GenerateOutputsPB allocate_response;
    allocate_response.set_allocation_token(decode_context.allocation_token);
    if (!decode_context.rpc_context.grpc_stream->Write(allocate_response)) {
        const string error_msg = "request: [" + decode_context.request_key + "] failed to write allocate output";
        const auto   stop_error = serverContextStopError(decode_context.server_context, error_msg);
        const auto   error = stop_error.hasError() ? stop_error : ErrorInfo(ErrorCode::RPC_FINISH_FAILED, error_msg);
        finish_stream(error.code(), error.ToString());
        decode_context.error_info   = error;
        decode_context.error_status = serializeErrorMsg(decode_context.request_key, error);
        return;
    }

    RTP_LLM_LOG_DEBUG("request [%s] allocate resource done", decode_context.request_key.c_str());
}

void DecodeRpcServer::loadCacheFromPrefill(DecodeGenerateContext& decode_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%s] load cache from prefill", decode_context.request_key.c_str());
    AtomicGuard       request_guard(loading_cache_requests_);
    auto&             grpc_stream = decode_context.rpc_context.grpc_stream;
    GenerateRequestPB load_request;
    GRPC_RET_IF_ERROR(
        decode_context, grpc_stream->Read(&load_request), grpc::StatusCode::INTERNAL, "failed to get loadReqeust");
    GRPC_RET_IF_ERROR(decode_context,
                      load_request.stage() == RemoteStage::LOAD,
                      grpc::StatusCode::INVALID_ARGUMENT,
                      "message second status != RemoteStage::LOAD");
    GRPC_RET_IF_ERROR(decode_context,
                      load_request.request_id() == decode_context.request_id,
                      grpc::StatusCode::INVALID_ARGUMENT,
                      "load request id does not match allocation");
    GRPC_RET_IF_ERROR(decode_context,
                      load_request.allocation_token() == decode_context.allocation_token,
                      grpc::StatusCode::INVALID_ARGUMENT,
                      "load allocation token does not match allocation");
    GRPC_RET_IF_ERROR(decode_context,
                      load_request.load_deadline_unix_ms() == decode_context.load_deadline_unix_ms,
                      grpc::StatusCode::INVALID_ARGUMENT,
                      "load deadline does not match allocation");
    decode_context.time_info.updateLoadBeginTime();
    auto error_info = loadCacheForAllRank(decode_context);
    decode_context.time_info.updateLoadEndTime();
    if (decode_context.remote_load_quiesced && decode_context.cache_lease_ticket != nullptr
        && !decode_context.cache_lease_ticket->complete()) {
        error_info = ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, "failed to complete decode cache lease");
    }
    if (!error_info.ok()) {
        RTP_LLM_LOG_WARNING("request [%s] load kv cache failed, error code [%s], cost time [%ld] ms",
                            decode_context.request_key.c_str(),
                            error_info.ToString().c_str(),
                            decode_context.time_info.loadCacheTimeMs());
    }

    GenerateOutputsPB load_response;
    load_response.mutable_error_info()->set_error_code(transErrorCodeToRPC(error_info.code()));
    load_response.set_remote_load_quiesced(decode_context.remote_load_quiesced);
    GRPC_RET_IF_ERROR(
        decode_context, grpc_stream->Write(load_response), grpc::StatusCode::INTERNAL, "send load response failed");
    GRPC_RET_IF_ERROR(decode_context, error_info.ok(), grpc::StatusCode::INTERNAL, error_info.ToString().c_str());
    RTP_LLM_LOG_DEBUG("request [%s] load cache from prefill done", decode_context.request_key.c_str());
}

void DecodeRpcServer::localGenerate(DecodeGenerateContext& decode_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%s] start to local generate", decode_context.request_key.c_str());
    auto&             grpc_stream     = decode_context.rpc_context.grpc_stream;
    auto&             generate_stream = decode_context.getStream();
    auto reject_stopped_request = [&]() {
        ErrorInfo error = ErrorInfo::OkStatus();
        if (generate_stream->getTimeoutMs() > 0 && currentTimeUs() / 1000 >= generate_stream->deadlineMs()) {
            error = ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "request deadline expired before decode admission");
        } else {
            error = serverContextStopError(decode_context.server_context, "decode admission");
        }
        if (!error.hasError()) {
            return false;
        }
        generate_stream->reportError(error.code(), error.ToString());
        generate_stream->moveToNext();
        decode_context.error_info   = error;
        decode_context.error_status = serializeErrorMsg(decode_context.request_key, error);
        return true;
    };
    if (reject_stopped_request()) {
        return;
    }
    GenerateRequestPB generate_request;
    GRPC_RET_IF_ERROR(decode_context,
                      grpc_stream->Read(&generate_request),
                      grpc::StatusCode::INTERNAL,
                      "poll generate request failed");
    GRPC_RET_IF_ERROR(decode_context,
                      generate_request.stage() == RemoteStage::GENERATE,
                      grpc::StatusCode::INTERNAL,
                      "message first status != RemoteStage::GENERATE");
    const auto generator_state_status = restoreSamplerGeneratorState(
        generate_request.sampler_generator_state_version(),
        generate_stream->generateConfig()->random_seed.has_value(),
        generate_stream->getGenerator(),
        generate_request.sampler_generator_state());
    GRPC_RET_IF_ERROR(decode_context,
                      generator_state_status.ok(),
                      grpc::StatusCode::INVALID_ARGUMENT,
                      generator_state_status.ToString());
    decode_context.time_info.updateGenerateBeginTime();
    generate_stream->setIsContextStream(false);
    generate_stream->step();

    auto new_tokens = torch::zeros({(int64_t)generate_stream->nextBatchSize(), 1}, torch::kInt32);

    new_tokens.data_ptr<int32_t>()[0] = generate_request.first_generate_token_id();
    generate_stream->incLastOutputPos();
    generate_stream->update({new_tokens,
                             1,
                             torch::Tensor(),
                             torch::Tensor(),
                             torch::Tensor(),
                             torch::Tensor(),
                             torch::Tensor(),
                             torch::Tensor(),
                             torch::Tensor(),
                             torch::Tensor()});
    if (generate_request.position_ids_size() > 0) {
        auto context_position_ids = torch::from_blob(const_cast<int32_t*>(generate_request.position_ids().data()),
                                                     {(int64_t)generate_request.position_ids_size()},
                                                     torch::kInt32)
                                        .clone();
        generate_stream->setContextPositionIds(context_position_ids);
    }
    if (propose_maga_init_params_) {
        generate_stream->setReuseLength(generate_stream->seqLength() - 1);
        generate_stream->setSpEditRun(false);
        generate_stream->setMtpTokenIndex(generate_stream->seqLength() - 1);
        generate_stream->setContainProposeToken(true);
        std::vector<int> propose_tokens;
        propose_tokens.assign(generate_request.propose_token_ids().begin(), generate_request.propose_token_ids().end());
        generate_stream->setProposeToken(propose_tokens);

        auto sp_output_buffer    = std::make_shared<SpeculativeExecutorStreamOutput>();
        sp_output_buffer->tokens = torch::zeros({1, (int64_t)propose_tokens.size()}, torch::kInt32);
        memcpy(sp_output_buffer->tokens.data_ptr<int>(), propose_tokens.data(), propose_tokens.size() * sizeof(int));

        auto propose_probs_t  = QueryConverter::transTensor(generate_request.propose_probs());
        auto propose_hidden_t = QueryConverter::transTensor(generate_request.propose_hidden());

        auto& tensors_holder = sp_output_buffer->tensors_holder;
        tensors_holder.emplace_back(std::move(propose_probs_t));
        tensors_holder.emplace_back(std::move(propose_hidden_t));

        generate_stream->setSPOutputBuffer(sp_output_buffer);
    }

    if (reject_stopped_request()) {
        return;
    }
    generate_stream->resetBeginTime(currentTimeUs());
    RTP_LLM_LOG_DEBUG(
        "decode init stream[%d]: %s", generate_stream->streamId(), generate_stream->debugString().c_str());
    engine_->enqueue(generate_stream);
    RTP_LLM_LOG_DEBUG("request [%s] enqueue success", decode_context.request_key.c_str());
    decode_context.error_status =
        pollStreamOutput(decode_context.server_context,
                         decode_context.request_key,
                         dynamic_cast<grpc::internal::WriterInterface<GenerateOutputsPB>*>(grpc_stream),
                         generate_stream);
    decode_context.time_info.updateGenerateEndTime();
    meta_->dequeue(decode_context.request_id, decode_context.getStream());

    RTP_LLM_LOG_DEBUG("request [%s] local generate done", decode_context.request_key.c_str());
}

BroadcastLoadRequestPB DecodeRpcServer::constructRemoteLoadRequestForMla(
    const LoadKVCacheContext& load_context, int index, const std::vector<std::string>& peer_addrs) const {
    BroadcastLoadRequestPB request;
    request.set_request_id(load_context.request_id);
    request.set_request_key(load_context.request_key);
    request.set_dp_rank(maga_init_params_.parallelism_config.dp_rank);
    request.set_partition_count(1);
    request.set_partition_id(0);
    request.set_allocation_token(load_context.allocation_token);
    request.set_load_deadline_unix_ms(load_context.load_deadline_unix_ms);

    const auto peer_index = selectMlaPrefillPeerIndex(
        load_context.request_id,
        maga_init_params_.parallelism_config.dp_rank,
        static_cast<size_t>(index),
        resource_.workers.size(),
        peer_addrs.size());
    request.add_peer_addrs(peer_addrs[peer_index]);
    for (auto& cache_key : load_context.cache_keys) {
        request.add_cache_keys(cache_key);
    }
    if (!load_context.block_ids_by_group.empty()) {
        for (const auto& group_block : load_context.block_ids_by_group) {
            auto* row = request.add_group_block_ids();
            RTP_LLM_CHECK_WITH_INFO(group_block != nullptr, "null group_block in block_ids_by_group");
            for (const auto& block_id : group_block->blocks()) {
                row->add_values(block_id);
            }
        }
    }
    request.set_timeout_ms(load_context.timeout_ms);
    return request;
}

BroadcastLoadRequestPB DecodeRpcServer::constructRemoteLoadRequest(const LoadKVCacheContext&       load_context,
                                                                   int                             index,
                                                                   const std::vector<std::string>& peer_addrs) const {
    BroadcastLoadRequestPB request;
    request.set_request_id(load_context.request_id);
    request.set_request_key(load_context.request_key);
    request.set_dp_rank(maga_init_params_.parallelism_config.dp_rank);
    request.set_allocation_token(load_context.allocation_token);
    request.set_load_deadline_unix_ms(load_context.load_deadline_unix_ms);
    // prefill worker has full kv cache each rank
    if (maga_init_params_.parallelism_config.prefill_cp_config.is_prefill_enabled()) {
        int part_cnt = resource_.workers.size();
        int peer_cnt = peer_addrs.size();
        request.set_partition_count(part_cnt);
        request.set_partition_id(index % part_cnt);
        request.add_peer_addrs(peer_addrs[index % peer_cnt]);
    } else {
        if (resource_.workers.size() % peer_addrs.size() == 0) {
            // D >= P, load part block of prefill
            int part_cnt = resource_.workers.size() / peer_addrs.size();
            request.set_partition_count(part_cnt);
            request.set_partition_id(index % part_cnt);
            request.add_peer_addrs(peer_addrs[index / part_cnt]);
        } else {
            // P >= D, load multi block of prefill
            request.set_partition_count(1);
            request.set_partition_id(0);
            int group_num = peer_addrs.size() / resource_.workers.size();
            for (int i = 0; i < group_num; i++) {
                request.add_peer_addrs(peer_addrs[index * group_num + i]);
            }
        }
    }

    for (auto& cache_key : load_context.cache_keys) {
        request.add_cache_keys(cache_key);
    }
    // Prefer per-group block ids if available (hybrid KV cache).
    if (!load_context.block_ids_by_group.empty()) {
        for (const auto& group_block : load_context.block_ids_by_group) {
            auto* row = request.add_group_block_ids();
            RTP_LLM_CHECK_WITH_INFO(group_block != nullptr, "null group_block in block_ids_by_group");
            for (const auto& block_id : group_block->blocks()) {
                row->add_values(block_id);
            }
        }
    }
    request.set_timeout_ms(load_context.timeout_ms);
    return request;
}

ErrorInfo DecodeRpcServer::loadCacheForAllRank(DecodeGenerateContext& decode_context) {
    RTP_LLM_PROFILE_FUNCTION();
    auto*       generate_stream    = decode_context.getStream().get();
    auto&       cache_keys         = generate_stream->cacheKeys(0);
    const auto& block_ids_by_group = generate_stream->kvCachePtr()->groupBlocks(0);

    if (resource_.workers.empty() || decode_context.peer_addrs.empty()) {
        RTP_LLM_LOG_WARNING("request:[%s] cache worker or peer address list is empty",
                            decode_context.request_key.c_str());
        return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, "cache worker or peer address list is empty");
    }
    if (resource_.workers.size() % decode_context.peer_addrs.size() != 0
        && decode_context.peer_addrs.size() % resource_.workers.size() != 0) {
        RTP_LLM_LOG_WARNING("request:[%s] peer ips size %d not equal to worker size %d",
                            decode_context.request_key.c_str(),
                            decode_context.peer_addrs.size(),
                            resource_.workers.size());
        return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, "peer ips size not equal to worker size");
    }

    auto load_cache_timeout_ms = maga_init_params_.pd_sep_config.load_cache_timeout_ms;
    load_cache_timeout_ms      = load_cache_timeout_ms > 0 ? load_cache_timeout_ms : LOAD_TIMEOUT_MS;
    auto max_rpc_timeout_ms    = maga_init_params_.pd_sep_config.max_rpc_timeout_ms;
    auto rpc_timeout           = max_rpc_timeout_ms > 0 ? max_rpc_timeout_ms : MAX_GRPC_TIMEOUT_MS;
    auto min_timeout_ms        = std::min(load_cache_timeout_ms, rpc_timeout);
    auto request_timeout_ms    = decode_context.request_timeout_ms;
    min_timeout_ms             = request_timeout_ms > 0 ? std::min(request_timeout_ms, min_timeout_ms) : min_timeout_ms;

    const auto parent_deadline = decode_context.server_context == nullptr ?
                                     RemoteLoadSystemClock::time_point::max() :
                                     decode_context.server_context->deadline();
    const auto remote_load_budget = makeRemoteLoadBudget(decode_context.load_deadline_unix_ms,
                                                         parent_deadline,
                                                         RemoteLoadSystemClock::now(),
                                                         RemoteLoadSteadyClock::now(),
                                                         min_timeout_ms);
    const bool request_cancelled =
        decode_context.server_context != nullptr && decode_context.server_context->IsCancelled();
    if (!canAdmitRemoteLoad(remote_load_budget, request_cancelled)) {
        return remote_load_budget.expired() ?
                   ErrorInfo(ErrorCode::LOAD_CACHE_TIMEOUT, "remote load deadline expired before load admission") :
                   ErrorInfo(ErrorCode::CANCELLED, "request is cancelled before load admission");
    }
    LoadKVCacheContext load_context{decode_context.request_id,
                                    decode_context.request_key,
                                    decode_context.peer_addrs,
                                    cache_keys,
                                    block_ids_by_group,
                                    generate_stream->reuseBlockSize(),
                                    remote_load_budget.remaining_ms,
                                    1,
                                    0,
                                    remote_load_budget.steady_deadline,
                                    decode_context.server_context};
    load_context.allocation_token      = decode_context.allocation_token;
    load_context.load_deadline_unix_ms = decode_context.load_deadline_unix_ms;

    // Prefill: TP = 1 && Decode: TP = 1
    if (resource_.workers.size() == 1 && decode_context.peer_addrs.size() == 1) {
        if (decode_context.cache_lease_ticket != nullptr
            && !decode_context.cache_lease_ticket->markStarted()) {
            return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED,
                             "decode cache lease is no longer accepting remote load");
        }
        auto operation_status = remote_load_fences_.begin(load_context.allocation_token,
                                                          load_context.load_deadline_unix_ms,
                                                          remote_load_budget.steady_deadline);
        if (!operation_status.ok()) {
            return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, operation_status.status().ToString());
        }
        auto operation         = std::move(*operation_status);
        load_context.operation = operation;

        ErrorInfo error_info(ErrorCode::LOAD_KV_CACHE_FAILED, "remote load connection retries exhausted");
        auto remaining_connect_retries =
            std::max<int64_t>(0, maga_init_params_.pd_sep_config.rdma_connect_retry_times);
        while (true) {
            error_info = loadCache(load_context);
            if (error_info.code() != ErrorCode::CACHE_STORE_LOAD_CONNECT_FAILED
                && error_info.code() != ErrorCode::CACHE_STORE_LOAD_RDMA_CONNECT_FAILED) {
                break;
            }
            if (remaining_connect_retries == 0) {
                break;
            }
            --remaining_connect_retries;
        }

        load_context.operation.reset();
        operation.reset();
        const auto fence_status = remote_load_fences_.sealAndWait(
            load_context.allocation_token,
            load_context.load_deadline_unix_ms,
            remote_load_budget.steady_deadline,
            RemoteLoadFenceRegistry::UnseenTokenPolicy::Seal,
            remote_load_budget.steady_deadline);
        decode_context.remote_load_quiesced = fence_status.ok();
        if (!fence_status.ok() && error_info.ok()) {
            return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, fence_status.ToString());
        }
        return error_info;
    }

    if (resource_.grpc_workers.size() != resource_.workers.size()) {
        RTP_LLM_LOG_WARNING("request:[%s] cache worker count %zu does not match grpc worker count %zu",
                            decode_context.request_key.c_str(),
                            resource_.workers.size(),
                            resource_.grpc_workers.size());
        return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, "cache worker and grpc worker counts do not match");
    }
    return loadCacheAsyncForTp(decode_context, load_context, remote_load_budget);
}

ErrorInfo DecodeRpcServer::loadCacheAsyncForTp(DecodeGenerateContext&  decode_context,
                                               LoadKVCacheContext&     load_context,
                                               const RemoteLoadBudget& remote_load_budget) {
    RTP_LLM_PROFILE_FUNCTION();
    auto fanout_stop_error = [&]() {
        if (RemoteLoadSteadyClock::now() >= remote_load_budget.steady_deadline) {
            return ErrorInfo(ErrorCode::LOAD_CACHE_TIMEOUT, "remote load deadline expired during worker fanout");
        }
        if (load_context.server_context != nullptr && load_context.server_context->IsCancelled()) {
            return ErrorInfo(ErrorCode::CANCELLED, "request is cancelled during worker fanout");
        }
        return ErrorInfo::OkStatus();
    };
    if (auto stop_error = fanout_stop_error(); stop_error.hasError()) {
        return stop_error;
    }
    int64_t load_cache_begin_time_us = currentTimeUs();

    struct WorkerRpcContext {
        BroadcastLoadResponsePB           response;
        Status                            status;
        std::shared_ptr<RpcService::Stub> stub;
        std::shared_ptr<ClientContext>    client_context;
        std::unique_ptr<ClientAsyncResponseReader<BroadcastLoadResponsePB>> reader;
        bool                                                              completed = false;
    };

    const uint32_t           worker_size = resource_.grpc_workers.size();
    const auto               cq_plan     = makeCompletionQueuePlan(worker_size);
    vector<WorkerRpcContext> all_context(worker_size);
    vector<CompletionQueue>  completion_queues(cq_plan.queue_count);
    vector<size_t>           each_finished_count(cq_plan.queue_count, 0);
    if (worker_size == 0 || cq_plan.queue_count == 0) {
        RTP_LLM_LOG_WARNING("request:[%s] cq_size or worker_size is 0, worker size = %u, cq size = %zu",
                            decode_context.request_key.c_str(),
                            worker_size,
                            cq_plan.queue_count);
        return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, "worker size or cq size is 0");
    }
    if (decode_context.cache_lease_ticket != nullptr
        && !decode_context.cache_lease_ticket->markStarted()) {
        return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED,
                         "decode cache lease is no longer accepting remote load");
    }

    struct AsyncRpcCleanup {
        vector<WorkerRpcContext>& contexts;
        vector<CompletionQueue>&  completion_queues;

        ~AsyncRpcCleanup() {
            for (auto& context : contexts) {
                if (context.reader && !context.completed) {
                    context.client_context->TryCancel();
                }
            }
            for (auto& completion_queue : completion_queues) {
                completion_queue.Shutdown();
            }
            for (auto& completion_queue : completion_queues) {
                void* ignored_tag = nullptr;
                bool  ignored_ok  = false;
                while (completion_queue.Next(&ignored_tag, &ignored_ok)) {
                }
            }
        }
    } cleanup{all_context, completion_queues};

    RTP_LLM_LOG_DEBUG("request:[%s] start to async remote load for all rank", decode_context.request_key.c_str());
    for (uint32_t i = 0; i < worker_size; i++) {
        if (auto stop_error = fanout_stop_error(); stop_error.hasError()) {
            return stop_error;
        }
        auto& worker         = resource_.grpc_workers[i];
        auto  connect_status = resource_.rpc_pool.getConnection(worker);
        if (!connect_status.ok()) {
            string error_msg = "get grpc connection for rank:" + std::to_string(i) + ", addr:" + worker + " failed";
            return ErrorInfo(ErrorCode::GET_CONNECTION_FAILED, error_msg);
        }
        auto& rpc_context = all_context[i];
        rpc_context.stub  = connect_status.value().stub;
        rpc_context.client_context =
            makePropagatedClientContext(load_context.server_context, remote_load_budget.system_deadline);
        BroadcastLoadRequestPB load_request;

        if (engine_->resourceContext().cache_manager->cacheConfig().use_mla) {
            load_request = constructRemoteLoadRequestForMla(load_context, i, decode_context.peer_addrs);
        } else {
            load_request = constructRemoteLoadRequest(load_context, i, decode_context.peer_addrs);
        }
        const auto queue_index = cq_plan.queueIndexForWorker(i);
        if (auto stop_error = fanout_stop_error(); stop_error.hasError()) {
            return stop_error;
        }
        RTP_LLM_CHECK_WITH_INFO(decode_context.remote_load_targets != nullptr
                                    && decode_context.remote_load_targets->markStarted(i),
                                "failed to record a remote load target before starting its RPC");
        rpc_context.reader     = rpc_context.stub->AsyncRemoteLoad(
            rpc_context.client_context.get(), load_request, &completion_queues[queue_index]);
        if (rpc_context.reader == nullptr) {
            return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED,
                             "failed to start async remote load for rank:" + std::to_string(i));
        }
        rpc_context.reader->Finish(
            &rpc_context.response, &rpc_context.status, reinterpret_cast<void*>(static_cast<uintptr_t>(i) + 1));
    }

    bool        all_success               = true;
    bool        all_quiesced              = true;
    size_t      finished_count            = 0;
    ErrorCode   error_code                = ErrorCode::NONE_ERROR;
    std::string error_msg                 = "failed to load kv cache in rank: ";
    int64_t     min_response_done_time_us = 1lu << 60;
    int64_t     max_response_done_time_us = 0;
    while (true) {
        RTP_LLM_LOG_DEBUG("request [%s] load cache loop step", decode_context.request_key.c_str());
        if (RemoteLoadSteadyClock::now() >= remote_load_budget.steady_deadline) {
            error_msg = "remote load deadline expired while waiting for worker fanout";
            return ErrorInfo(ErrorCode::LOAD_CACHE_TIMEOUT, error_msg);
        }
        if (load_context.server_context != nullptr && load_context.server_context->IsCancelled()) {
            string error_msg = "request is cancelled";
            return ErrorInfo(ErrorCode::CANCELLED, error_msg);
        }
        auto once_deadline = std::min(
            RemoteLoadSystemClock::now()
                + std::chrono::milliseconds(maga_init_params_.pd_sep_config.decode_polling_kv_cache_step_ms),
            remote_load_budget.system_deadline);
        RTP_LLM_LOG_DEBUG("request [%s] start to execute async next", decode_context.request_key.c_str());
        // TODO(xinfei.sxf) There is a problem with complete queue next call delay here, the reason is yet to be
        // investigated
        void* got_tag;
        bool  ok = false;
        for (uint32_t i = 0; i < completion_queues.size(); i++) {
            if (each_finished_count[i] == cq_plan.expected_completions[i]) {
                continue;
            }
            const auto next_status = completion_queues[i].AsyncNext(&got_tag, &ok, once_deadline);
            if (next_status == grpc::CompletionQueue::NextStatus::TIMEOUT) {
                RTP_LLM_LOG_DEBUG("request [%s] async next timeout", decode_context.request_key.c_str());
                continue;
            }
            if (next_status == grpc::CompletionQueue::NextStatus::SHUTDOWN) {
                return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED,
                                 "async remote load completion queue shut down unexpectedly");
            }
            each_finished_count[i]++;
            if (!ok) {
                string error_msg = "async get next event from grpc completion queue failed";
                return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, error_msg);
            }
            const auto raw_tag = reinterpret_cast<uintptr_t>(got_tag);
            if (raw_tag == 0 || raw_tag > worker_size) {
                return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, "async remote load returned an invalid tag");
            }
            const auto  rank             = raw_tag - 1;
            if (all_context[rank].completed) {
                return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, "async remote load returned a duplicate tag");
            }
            all_context[rank].completed  = true;
            const auto& status           = all_context[rank].status;
            const auto& response         = all_context[rank].response;
            const auto& pb_error_code    = response.error_info().error_code();
            const auto& pb_error_message = response.error_info().error_message();
            min_response_done_time_us    = std::min(min_response_done_time_us, response.done_time_us());
            max_response_done_time_us    = std::max(max_response_done_time_us, response.done_time_us());
            RTP_LLM_LOG_DEBUG("request [%s] load cache for rank [%zu] done",
                              decode_context.request_key.c_str(),
                              static_cast<size_t>(rank));
            if (!status.ok()) {
                all_success = false;
                all_quiesced = false;
                error_code = mergeRemoteLoadErrorCode(
                    error_code,
                    transRemoteLoadGrpcStatus(
                        status.error_code(), RemoteLoadSteadyClock::now() >= remote_load_budget.steady_deadline));
                error_msg += std::to_string(rank) + ": " + status.error_message() + ", ";
            } else if (pb_error_code != ErrorCodePB::NONE_ERROR) {
                all_success = false;
                error_code  = mergeRemoteLoadErrorCode(error_code, transRPCErrorCode(pb_error_code));
                error_msg += std::to_string(rank) + ": " + pb_error_message + ", ";
            }
            if (!response.quiesced()) {
                all_quiesced = false;
                all_success  = false;
                error_code = mergeRemoteLoadErrorCode(error_code, ErrorCode::LOAD_KV_CACHE_FAILED);
                error_msg += std::to_string(rank) + ": remote load did not quiesce, ";
            }
            finished_count++;
            if (finished_count == worker_size) {
                break;
            }
        }
        if (finished_count == worker_size) {
            break;
        }
    }

    if (finished_count != worker_size) {
        all_success = false;
        all_quiesced = false;
    }
    decode_context.remote_load_quiesced = all_quiesced;
    if (!all_success) {
        return ErrorInfo(error_code, error_msg);
    }

    decode_context.stat_info.load_cache_min_rt_us       = min_response_done_time_us - load_cache_begin_time_us;
    decode_context.stat_info.load_cache_max_rt_us       = max_response_done_time_us - load_cache_begin_time_us;
    decode_context.stat_info.load_cache_polling_cost_us = currentTimeUs() - max_response_done_time_us;

    RTP_LLM_LOG_DEBUG("load_cache_min_rt_us = %ld, load_cache_max_rt_us = %ld, load_cache_polling_cost_us = %ld",
                      decode_context.stat_info.load_cache_min_rt_us,
                      decode_context.stat_info.load_cache_max_rt_us,
                      decode_context.stat_info.load_cache_polling_cost_us);

    return ErrorInfo::OkStatus();
}

ErrorInfo DecodeRpcServer::loadCacheSyncForTp(DecodeGenerateContext& decode_context, LoadKVCacheContext& load_context) {
    RTP_LLM_PROFILE_FUNCTION();
    int64_t                                               load_cache_begin_time_us  = currentTimeUs();
    int64_t                                               min_response_done_time_us = 1lu << 60;
    int64_t                                               max_response_done_time_us = 0;
    std::vector<autil::ThreadPoolBase::Future<ErrorInfo>> futures;
    auto                                                  local_task = [&] { return this->loadCache(load_context); };
    futures.emplace_back(thread_pool_->async(local_task));

    for (int i = 0; i < resource_.grpc_workers.size(); i++) {
        auto& worker      = resource_.grpc_workers[i];
        auto  remote_task = [&]() {
            auto connect_status = resource_.rpc_pool.getConnection(worker);
            if (!connect_status.ok()) {
                string error_msg = "get grpc connection for ip " + worker + " failed";
                return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, error_msg);
            }
            auto                   stub = connect_status.value().stub.get();
            ClientContext          client_context;
            BroadcastLoadRequestPB load_request;

            if (engine_->resourceContext().cache_manager->cacheConfig().use_mla) {
                load_request = constructRemoteLoadRequestForMla(load_context, i, decode_context.peer_addrs);
            } else {
                load_request = constructRemoteLoadRequest(load_context, i, decode_context.peer_addrs);
            }
            BroadcastLoadResponsePB response;
            auto                    grpc_status      = stub->RemoteLoad(&client_context, load_request, &response);
            const auto&             pb_error_code    = response.error_info().error_code();
            const auto&             pb_error_message = response.error_info().error_message();
            if (!grpc_status.ok()) {
                return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, grpc_status.error_message());
            } else if (pb_error_code != ErrorCodePB::NONE_ERROR) {
                auto error_code = transRPCErrorCode(pb_error_code);
                return ErrorInfo(error_code, pb_error_message);
            }
            min_response_done_time_us = std::min(min_response_done_time_us, response.done_time_us());
            max_response_done_time_us = std::max(max_response_done_time_us, response.done_time_us());
            return ErrorInfo::OkStatus();
        };
        futures.emplace_back(thread_pool_->async(remote_task));
    }

    std::string err_msg = "failed to load kv cache in rank: ";
    bool        success = true;
    for (int i = 0; i < futures.size(); i++) {
        auto status = futures[i].get();
        if (!status.ok()) {
            // TODO(xinfei.sxf) 可以不等待其他rank的结果吗
            success = false;
            err_msg += std::to_string(i) + ": " + status.ToString() + ", ";
        }
    }
    if (!success) {
        RTP_LLM_LOG_WARNING(err_msg);
        return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, err_msg);
    }

    decode_context.stat_info.load_cache_min_rt_us       = min_response_done_time_us - load_cache_begin_time_us;
    decode_context.stat_info.load_cache_max_rt_us       = max_response_done_time_us - load_cache_begin_time_us;
    decode_context.stat_info.load_cache_polling_cost_us = currentTimeUs() - max_response_done_time_us;

    RTP_LLM_LOG_DEBUG("load_cache_min_rt_us = %ld, load_cache_max_rt_us = %ld, load_cache_polling_cost_us = %ld",
                      decode_context.stat_info.load_cache_min_rt_us,
                      decode_context.stat_info.load_cache_max_rt_us,
                      decode_context.stat_info.load_cache_polling_cost_us);

    return ErrorInfo::OkStatus();
}

ErrorInfo DecodeRpcServer::loadCache(const LoadKVCacheContext& load_context) {
    RTP_LLM_PROFILE_FUNCTION();
    AtomicGuard request_guard(onflight_load_cache_requests_);
    uint32_t    remaining_timeout_ms = 0;
    if (!getCacheStoreLoadRemainingTimeoutMs(
            load_context.steady_deadline, CacheStoreLoadClock::now(), remaining_timeout_ms)) {
        return ErrorInfo(ErrorCode::LOAD_CACHE_TIMEOUT, "remote load deadline expired before cache preparation");
    }
    if (load_context.server_context != nullptr && load_context.server_context->IsCancelled()) {
        return ErrorInfo(ErrorCode::CANCELLED, "request is cancelled before cache preparation");
    }
    const auto& request_key   = load_context.request_key;
    auto        cache_manager = engine_->resourceContext().cache_manager;
    const auto& cache_config  = cache_manager->cacheConfig();
    auto        layer_num     = maga_init_params_.model_config_.num_layers;

    const int peer_cnt = static_cast<int>(load_context.peer_addrs.size());
    RTP_LLM_CHECK_WITH_INFO(peer_cnt > 0, "peer_addrs is empty");

    auto lease_resource = makeCacheTransferLeaseResource(
        cache_config.groupNums(),
        load_context.block_ids_by_group,
        load_context.cache_keys.size(),
        cache_manager->totalBlocksNum());
    if (!lease_resource.ok()) {
        const auto error_msg = "invalid cache transfer lease layout: " + lease_resource.status().ToString();
        RTP_LLM_LOG_WARNING("request [%s] %s", request_key.c_str(), error_msg.c_str());
        return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, error_msg);
    }

    std::shared_ptr<KVCacheResource> cache_transfer_lease;
    std::shared_ptr<CacheTransferLifetime> cache_transfer_lifetime;
    if (!load_context.cache_keys.empty()) {
        cache_transfer_lease = cache_manager->incrKVCacheRef(
            *lease_resource, lease_resource->cacheKeys(), /*is_connector=*/true);
        if (cache_transfer_lease == nullptr) {
            const auto error_msg = "failed to acquire cache transfer block lease";
            RTP_LLM_LOG_WARNING("request [%s] %s", request_key.c_str(), error_msg);
            return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, error_msg);
        }
        if (load_context.operation != nullptr) {
            cache_transfer_lifetime =
                makeCacheTransferLifetime(cache_transfer_lease, load_context.operation);
            if (cache_transfer_lifetime == nullptr) {
                const auto error_msg = "failed to bind cache transfer lifetime";
                RTP_LLM_LOG_WARNING("request [%s] %s", request_key.c_str(), error_msg);
                return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, error_msg);
            }
        }
    }

    const bool   use_mla       = cache_config.use_mla;
    const bool   use_hybrid    = cache_config.groupNums() > 1;
    const auto&  spec          = cache_config.cache_specs[0];
    const size_t k_total_bytes = spec->k_block_size_bytes();
    const size_t v_total_bytes = spec->v_block_size_bytes();

    if (!use_mla && peer_cnt > 1) {
        RTP_LLM_CHECK_WITH_INFO(k_total_bytes % static_cast<size_t>(peer_cnt) == 0,
                                "k_block bytes[%zu] not divisible by peer_cnt[%d]",
                                k_total_bytes,
                                peer_cnt);
        RTP_LLM_CHECK_WITH_INFO(v_total_bytes % static_cast<size_t>(peer_cnt) == 0,
                                "v_block bytes[%zu] not divisible by peer_cnt[%d]",
                                v_total_bytes,
                                peer_cnt);
    }

    auto cancel_check_func = [&load_context]() -> bool {
        return load_context.server_context != nullptr && load_context.server_context->IsCancelled();
    };
    auto start_load_time_us = currentTimeUs();
    std::vector<std::shared_ptr<LoadContext>> load_contexts;
    for (int i = 0; i < load_context.peer_addrs.size(); i++) {
        auto&                                            peer_addr = load_context.peer_addrs[i];
        std::vector<std::shared_ptr<RequestBlockBuffer>> layer_caches;
        RTP_LLM_LOG_DEBUG("load context request id is %d", load_context.request_id);

        for (size_t layer_id = 0; layer_id < layer_num; layer_id++) {
            auto request_key = std::to_string(load_context.request_id) + "-" + std::to_string(layer_id);
            auto load_layer_cache =
                std::make_shared<RequestBlockBuffer>(std::to_string(load_context.request_id), request_key);
            size_t gid = 0;
            if (use_hybrid && layer_id < cache_config.layer_to_group_id.size()) {
                const int mapped_gid = cache_config.layer_to_group_id[layer_id];
                if (mapped_gid >= 0) {
                    gid = static_cast<size_t>(mapped_gid);
                }
            }
            RTP_LLM_CHECK_WITH_INFO(gid < load_context.block_ids_by_group.size(),
                                    "group id out of range: gid=%zu group_num=%zu",
                                    gid,
                                    load_context.block_ids_by_group.size());
            RTP_LLM_CHECK_WITH_INFO(load_context.block_ids_by_group[gid] != nullptr, "null group_block: gid=%zu", gid);
            const auto& block_ids = load_context.block_ids_by_group[gid]->blocks();
            size_t      model_id  = maga_init_params_.model_id;

            CacheGroupType group_type = CacheGroupType::FULL;
            if (use_hybrid) {
                if (gid >= cache_config.group_types.size()) {
                    const auto error_msg = "missing cache group type for group " + std::to_string(gid);
                    return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, error_msg);
                }
                group_type = cache_config.group_types[gid];
            }
            auto block_pos_list = selectCacheTransferBlockPositions(
                group_type, block_ids, load_context.cache_keys.size(), load_context.reuse_block_size);
            if (!block_pos_list.ok()) {
                const auto error_msg = "invalid cache transfer layout for model layer " + std::to_string(layer_id)
                                       + ": " + block_pos_list.status().ToString();
                RTP_LLM_LOG_WARNING("%s", error_msg.c_str());
                return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, error_msg);
            }

            for (size_t block_pos : *block_pos_list) {
                auto cache_key = makeCacheKey(model_id, std::to_string(load_context.cache_keys[block_pos]), layer_id);
                // FT_LOG_DEBUG("large model load cache_key %s", cache_key.c_str());
                auto block_id = block_ids[block_pos];

                const int local_part_cnt = peer_cnt;
                const int local_part_id  = i;
                auto parts = cache_manager->convertIndexToBuffer(block_id, layer_id, local_part_cnt, local_part_id);

                auto addBufBlock = [&](const std::string& key, const BlockInfo& block) {
                    RTP_LLM_CHECK_WITH_INFO(block.addr != nullptr, "null block addr for key=%s", key.c_str());
                    RTP_LLM_CHECK_WITH_INFO(block.size_bytes > 0, "zero block size for key=%s", key.c_str());
                    auto addr = cache_transfer_lifetime != nullptr ?
                                    makeCacheTransferAddress(cache_transfer_lifetime, block.addr) :
                                    makeCacheTransferAddress(cache_transfer_lease, block.addr);
                    RTP_LLM_CHECK_WITH_INFO(addr != nullptr, "failed to retain block lease for key=%s", key.c_str());
                    load_layer_cache->addBlock(key, addr, static_cast<uint32_t>(block.size_bytes), true, true);
                };

                // Hybrid Attention not support asymmetric TP, thus transfer the whole kvache blocks
                if (use_mla || use_hybrid) {
                    RTP_LLM_CHECK_WITH_INFO(parts.size() == 1 || parts.size() == 2,
                                            "unexpected mla convertIndexToBuffer parts size=%zu",
                                            parts.size());
                    addBufBlock("kv_" + cache_key, parts[0]);
                    if (parts.size() == 2) {
                        addBufBlock("kv_scale_" + cache_key, parts[1]);
                    }
                } else {
                    RTP_LLM_CHECK_WITH_INFO(parts.size() == 2 || parts.size() == 4,
                                            "unexpected convertIndexToBuffer parts size=%zu",
                                            parts.size());
                    addBufBlock("k_" + cache_key, parts[0]);
                    addBufBlock("v_" + cache_key, parts[1]);
                    if (parts.size() == 4) {
                        addBufBlock("k_scale_" + cache_key, parts[2]);
                        addBufBlock("v_scale_" + cache_key, parts[3]);
                    }
                }
            }
            layer_caches.push_back(load_layer_cache);
        }

        if (engine_->isMTPEagle()) {
            if (propose_maga_init_params_ && propose_maga_init_params_->mtp_model_params_
                && !propose_maga_init_params_->mtp_model_params_->empty()) {
                const size_t mtp_base_model_id = propose_maga_init_params_->mtp_model_params_->at(0)->model_id;
                for (size_t mtp_model_id = 0; mtp_model_id < propose_maga_init_params_->mtp_model_params_->size();
                     mtp_model_id++) {
                    EngineInitParams* mtp_engine_init_params =
                        propose_maga_init_params_->mtp_model_params_->at(mtp_model_id).get();
                    if (mtp_engine_init_params == nullptr) {
                        return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED,
                                         "mtp_model_params_[" + std::to_string(mtp_model_id) + "] is nullptr");
                    }

                    const auto&  mtp_cache_cfg = cache_manager->getMTPModuleCacheConfig(static_cast<int>(mtp_model_id));
                    const size_t layer_num     = mtp_engine_init_params->model_config_.num_layers;

                    RTP_LLM_CHECK_WITH_INFO(layer_num == mtp_cache_cfg.layer_num,
                                            "mtp layer_num mismatch: engine=" + std::to_string(layer_num)
                                                + " cache_cfg=" + std::to_string(mtp_cache_cfg.layer_num)
                                                + " (mtp_model_id=" + std::to_string(mtp_model_id) + ")");
                    RTP_LLM_CHECK_WITH_INFO(
                        !mtp_cache_cfg.global_layer_ids.empty(),
                        "mtp_cache_cfg.global_layer_ids is empty (mtp_model_id=" + std::to_string(mtp_model_id) + ")");

                    for (size_t layer_id = 0; layer_id < layer_num; layer_id++) {
                        auto request_key = std::to_string(load_context.request_id) + "-" + std::to_string(layer_id);
                        auto load_layer_cache =
                            std::make_shared<RequestBlockBuffer>(std::to_string(load_context.request_id), request_key);
                        size_t     gid            = 0;
                        const bool mtp_use_hybrid = mtp_cache_cfg.groupNums() > 1;
                        if (mtp_use_hybrid && layer_id < mtp_cache_cfg.layer_to_group_id.size()) {
                            const int mapped_gid = mtp_cache_cfg.layer_to_group_id[layer_id];
                            if (mapped_gid >= 0) {
                                gid = static_cast<size_t>(mapped_gid);
                            }
                        }
                        RTP_LLM_CHECK_WITH_INFO(gid < load_context.block_ids_by_group.size(),
                                                "mtp group id out of range: gid=%zu group_num=%zu",
                                                gid,
                                                load_context.block_ids_by_group.size());
                        RTP_LLM_CHECK_WITH_INFO(
                            load_context.block_ids_by_group[gid] != nullptr, "null mtp group_block: gid=%zu", gid);
                        const auto& block_ids = load_context.block_ids_by_group[gid]->blocks();
                        size_t      model_id  = mtp_base_model_id;

                        // Use per-module global_layer_ids for address lookup.
                        const int global_layer_id = mtp_cache_cfg.global_layer_ids[0][layer_id];

                        CacheGroupType group_type = CacheGroupType::FULL;
                        if (mtp_use_hybrid) {
                            if (gid >= mtp_cache_cfg.group_types.size()) {
                                const auto error_msg = "missing draft cache group type for group " + std::to_string(gid);
                                return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, error_msg);
                            }
                            group_type = mtp_cache_cfg.group_types[gid];
                        }
                        auto block_pos_list = selectCacheTransferBlockPositions(
                            group_type, block_ids, load_context.cache_keys.size(), load_context.reuse_block_size);
                        if (!block_pos_list.ok()) {
                            const auto error_msg = "invalid cache transfer layout for draft layer "
                                                   + std::to_string(layer_id) + ": "
                                                   + block_pos_list.status().ToString();
                            RTP_LLM_LOG_WARNING("%s", error_msg.c_str());
                            return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, error_msg);
                        }

                        for (size_t block_pos : *block_pos_list) {
                            auto cache_key =
                                makeCacheKey(model_id, std::to_string(load_context.cache_keys[block_pos]), layer_id);
                            auto       block_id       = block_ids[block_pos];
                            const bool mtp_use_mla    = mtp_cache_cfg.use_mla;
                            const int  local_part_cnt = peer_cnt;
                            const int  local_part_id  = i;
                            auto       parts          = cache_manager->convertIndexToBuffer(
                                block_id, global_layer_id, local_part_cnt, local_part_id);

                            auto addBufBlock = [&](const std::string& key, const BlockInfo& block) {
                                RTP_LLM_CHECK_WITH_INFO(
                                    block.addr != nullptr, "null block addr for key=%s", key.c_str());
                                RTP_LLM_CHECK_WITH_INFO(
                                    block.size_bytes > 0, "zero block size for key=%s", key.c_str());
                                auto addr = cache_transfer_lifetime != nullptr ?
                                                makeCacheTransferAddress(cache_transfer_lifetime, block.addr) :
                                                makeCacheTransferAddress(cache_transfer_lease, block.addr);
                                RTP_LLM_CHECK_WITH_INFO(
                                    addr != nullptr, "failed to retain draft block lease for key=%s", key.c_str());
                                load_layer_cache->addBlock(
                                    key, addr, static_cast<uint32_t>(block.size_bytes), true, true);
                            };

                            if (mtp_use_mla || mtp_use_hybrid) {
                                RTP_LLM_CHECK_WITH_INFO(parts.size() == 1 || parts.size() == 2,
                                                        "unexpected mtp mla convertIndexToBuffer parts size=%zu",
                                                        parts.size());
                                addBufBlock("kv_" + cache_key, parts[0]);
                                if (parts.size() == 2) {
                                    addBufBlock("kv_scale_" + cache_key, parts[1]);
                                }
                            } else {
                                RTP_LLM_CHECK_WITH_INFO(parts.size() == 2 || parts.size() == 4,
                                                        "unexpected mtp convertIndexToBuffer parts size=%zu",
                                                        parts.size());
                                addBufBlock("k_" + cache_key, parts[0]);
                                addBufBlock("v_" + cache_key, parts[1]);
                                if (parts.size() == 4) {
                                    addBufBlock("k_scale_" + cache_key, parts[2]);
                                    addBufBlock("v_scale_" + cache_key, parts[3]);
                                }
                            }
                        }
                        layer_caches.push_back(load_layer_cache);
                    }
                }
            }
        }

        auto ip_parts = autil::StringUtil::split(peer_addr, ":");
        if (ip_parts.size() != 3) {
            RTP_LLM_LOG_WARNING("invalid peer ip to load [%s]", peer_addr.c_str());
            return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, "invalid peer ip");
        }

        if (!getCacheStoreLoadRemainingTimeoutMs(
                load_context.steady_deadline, CacheStoreLoadClock::now(), remaining_timeout_ms)) {
            return ErrorInfo(ErrorCode::LOAD_CACHE_TIMEOUT,
                             "remote load deadline expired before cache-store submission");
        }
        if (cancel_check_func()) {
            return ErrorInfo(ErrorCode::CANCELLED, "request is cancelled before cache-store submission");
        }
        auto layer_cache_load_context =
            resource_.cache_store->loadBuffersUntil(
                layer_caches,
                ip_parts[0],
                autil::StringUtil::strToInt32WithDefault(ip_parts[1].c_str(), 0),
                autil::StringUtil::strToInt32WithDefault(ip_parts[2].c_str(), 0),
                load_context.steady_deadline,
                cancel_check_func,
                load_context.partition_count,
                load_context.partition_id);
        if (!layer_cache_load_context) {
            RTP_LLM_LOG_WARNING("request [%s] load cache failed, layer cache load context is nullptr",
                                request_key.c_str());
            return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, "load kv cache failed");
        }
        load_contexts.push_back(layer_cache_load_context);
    }

    for (auto& layer_cache_load_context : load_contexts) {
        layer_cache_load_context->waitDone();
        if (layer_cache_load_context->success()) {
            RTP_LLM_LOG_DEBUG("request [%s] load kv cache success", request_key.c_str());
        } else {
            // TODO(xinfei.sxf) add retry for part failed blocks.
            auto load_done_time_us = currentTimeUs();
            RTP_LLM_LOG_WARNING("request [%s] load cache failed, status [%s], cost time [%ld] ms",
                                request_key.c_str(),
                                layer_cache_load_context->getErrorInfoString().c_str(),
                                (load_done_time_us - start_load_time_us) / 1000);
            return layer_cache_load_context->getErrorInfo();
        }
    }

    return ErrorInfo::OkStatus();
}

grpc::Status DecodeRpcServer::RemoteLoad(grpc::ServerContext*          server_context,
                                         const BroadcastLoadRequestPB* request,
                                         BroadcastLoadResponsePB*      response) {
    RTP_LLM_PROFILE_FUNCTION();
    response->set_quiesced(false);

    if (request->allocation_token().empty() || request->load_deadline_unix_ms() <= 0) {
        setRemoteLoadResponseError(
            response,
            ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, "remote load request has no valid allocation token"));
        return grpc::Status::OK;
    }

    const auto parent_deadline = server_context == nullptr ? RemoteLoadSystemClock::time_point::max() :
                                                             server_context->deadline();
    const auto remote_load_budget = makeRemoteLoadBudget(request->load_deadline_unix_ms(),
                                                         parent_deadline,
                                                         RemoteLoadSystemClock::now(),
                                                         RemoteLoadSteadyClock::now(),
                                                         request->timeout_ms(),
                                                         /*parent_deadline_authoritative=*/true);
    auto remote_load_stop_error = [&]() {
        if (RemoteLoadSteadyClock::now() >= remote_load_budget.steady_deadline) {
            return ErrorInfo(ErrorCode::LOAD_CACHE_TIMEOUT, "remote load request deadline has expired");
        }
        if (server_context != nullptr && server_context->IsCancelled()) {
            return ErrorInfo(ErrorCode::CANCELLED, "remote load request is cancelled");
        }
        return ErrorInfo::OkStatus();
    };
    if (auto error_info = remote_load_stop_error(); error_info.hasError()) {
        setRemoteLoadResponseError(response, error_info);
        return grpc::Status::OK;
    }

    if (request->dp_rank() != maga_init_params_.parallelism_config.dp_rank) {
        RTP_LLM_LOG_WARNING("only load when in dp group, skip load for dp rank %d", request->dp_rank());
        const auto fence_status = remote_load_fences_.sealAndWait(request->allocation_token(),
                                                                 request->load_deadline_unix_ms(),
                                                                 remote_load_budget.steady_deadline,
                                                                 RemoteLoadFenceRegistry::UnseenTokenPolicy::Seal,
                                                                 remote_load_budget.steady_deadline);
        response->set_quiesced(fence_status.ok());
        const auto error_message = fence_status.ok() ? "remote load was routed to the wrong data-parallel rank" :
                                                       fence_status.ToString();
        setRemoteLoadResponseError(
            response, ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, error_message));
        return grpc::Status::OK;
    }

    if (auto error_info = remote_load_stop_error(); error_info.hasError()) {
        setRemoteLoadResponseError(response, error_info);
        return grpc::Status::OK;
    }
    auto operation_status = remote_load_fences_.begin(request->allocation_token(),
                                                      request->load_deadline_unix_ms(),
                                                      remote_load_budget.steady_deadline);
    if (!operation_status.ok()) {
        const auto error_code = operation_status.status().code() == absl::StatusCode::kDeadlineExceeded ?
                                    ErrorCode::LOAD_CACHE_TIMEOUT :
                                    ErrorCode::LOAD_KV_CACHE_FAILED;
        setRemoteLoadResponseError(response, ErrorInfo(error_code, operation_status.status().ToString()));
        return grpc::Status::OK;
    }
    auto operation = std::move(operation_status).value();

    std::vector<CacheKeyType> cache_keys(request->cache_keys().begin(), request->cache_keys().end());
    GroupBlockIds             block_ids_by_group;
    block_ids_by_group.reserve(static_cast<size_t>(request->group_block_ids_size()));
    for (int i = 0; i < request->group_block_ids_size(); ++i) {
        const auto& row              = request->group_block_ids(i);
        auto        block_ids_holder = std::make_shared<BlockIds>();
        block_ids_holder->assign(BlockIndicesType(row.values().begin(), row.values().end()));
        block_ids_by_group.push_back(std::move(block_ids_holder));
    }

    std::vector<std::string> peer_addrs(request->peer_addrs().begin(), request->peer_addrs().end());

    // TODO(xinfei.sxf) add retry
    LoadKVCacheContext load_context{request->request_id(),
                                    request->request_key(),
                                    peer_addrs,
                                    cache_keys,
                                    block_ids_by_group,
                                    request->reuse_block_size(),
                                    remote_load_budget.remaining_ms,
                                    request->partition_count(),
                                    request->partition_id(),
                                    remote_load_budget.steady_deadline,
                                    server_context};
    load_context.allocation_token      = request->allocation_token();
    load_context.load_deadline_unix_ms = request->load_deadline_unix_ms();
    load_context.operation             = operation;

    auto error_info = loadCache(load_context);
    load_context.operation.reset();
    operation.reset();

    const auto fence_status = remote_load_fences_.sealAndWait(request->allocation_token(),
                                                             request->load_deadline_unix_ms(),
                                                             remote_load_budget.steady_deadline,
                                                             RemoteLoadFenceRegistry::UnseenTokenPolicy::Seal,
                                                             remote_load_budget.steady_deadline);
    response->set_quiesced(fence_status.ok());
    if (!fence_status.ok() && error_info.ok()) {
        error_info = ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, fence_status.ToString());
    }
    setRemoteLoadResponseError(response, error_info);
    response->set_done_time_us(currentTimeUs());
    RTP_LLM_LOG_DEBUG("request: %s, remote load cache grpc done", request->request_key().c_str());
    return grpc::Status::OK;
}

bool DecodeRpcServer::quiesceRemoteLoadTargets(const std::string&              allocation_token,
                                               int64_t                         load_deadline_unix_ms,
                                               const std::vector<std::string>& worker_addrs,
                                               std::chrono::milliseconds       attempt_timeout,
                                               std::chrono::milliseconds       retention_timeout,
                                               RemoteLoadFenceRegistry::UnseenTokenPolicy local_unseen_token_policy) {
    attempt_timeout = std::max(attempt_timeout, std::chrono::milliseconds(1));
    retention_timeout = std::max(retention_timeout, std::chrono::milliseconds(1));
    const auto steady_deadline = std::chrono::steady_clock::now() + attempt_timeout;
    const auto rpc_deadline    = std::chrono::system_clock::now() + attempt_timeout;
    const auto local_expiry = saturatingSteadyDeadline(
        std::chrono::steady_clock::now(), retention_timeout.count());

    struct WorkerQuiesceContext {
        RemoteLoadQuiesceResponsePB response;
        grpc::Status                status;
        std::shared_ptr<RpcService::Stub> stub;
        std::unique_ptr<grpc::ClientContext> client_context;
        std::unique_ptr<grpc::ClientAsyncResponseReader<RemoteLoadQuiesceResponsePB>> reader;
        bool completed = false;
    };

    grpc::CompletionQueue             completion_queue;
    std::vector<WorkerQuiesceContext> contexts(worker_addrs.size());
    size_t                            started_count = 0;
    bool                              all_quiesced  = true;

    RemoteLoadQuiesceRequestPB worker_request;
    worker_request.set_allocation_token(allocation_token);
    worker_request.set_load_deadline_unix_ms(load_deadline_unix_ms);
    worker_request.set_local_only(true);
    worker_request.set_retention_timeout_ms(retention_timeout.count());

    for (size_t index = 0; index < worker_addrs.size(); ++index) {
        auto connection = resource_.rpc_pool.getConnection(worker_addrs[index]);
        if (!connection.ok()) {
            all_quiesced = false;
            continue;
        }
        auto& context          = contexts[index];
        context.stub           = connection->stub;
        context.client_context = std::make_unique<grpc::ClientContext>();
        context.client_context->set_deadline(rpc_deadline);
        context.reader = context.stub->AsyncQuiesceRemoteLoad(
            context.client_context.get(), worker_request, &completion_queue);
        if (context.reader == nullptr) {
            all_quiesced = false;
            continue;
        }
        context.reader->Finish(&context.response,
                               &context.status,
                               reinterpret_cast<void*>(static_cast<uintptr_t>(index) + 1));
        ++started_count;
    }

    const auto local_status = remote_load_fences_.sealAndWait(
        allocation_token, load_deadline_unix_ms, steady_deadline, local_unseen_token_policy, local_expiry);
    all_quiesced = local_status.ok() && all_quiesced;

    size_t finished_count = 0;
    while (finished_count < started_count && std::chrono::steady_clock::now() < steady_deadline) {
        void* got_tag = nullptr;
        bool  ok      = false;
        const auto next_status = completion_queue.AsyncNext(&got_tag, &ok, rpc_deadline);
        if (next_status != grpc::CompletionQueue::NextStatus::GOT_EVENT) {
            all_quiesced = false;
            break;
        }
        const auto raw_tag = reinterpret_cast<uintptr_t>(got_tag);
        if (raw_tag == 0 || raw_tag > contexts.size()) {
            all_quiesced = false;
            continue;
        }
        auto& context = contexts[raw_tag - 1];
        if (context.completed) {
            all_quiesced = false;
            continue;
        }
        context.completed = true;
        ++finished_count;
        all_quiesced = ok && context.status.ok() && context.response.quiesced()
                        && context.response.error_info().error_code() == ErrorCodePB::NONE_ERROR && all_quiesced;
    }

    if (finished_count != started_count) {
        all_quiesced = false;
    }
    for (auto& context : contexts) {
        if (context.reader != nullptr && !context.completed) {
            context.client_context->TryCancel();
        }
    }
    completion_queue.Shutdown();
    void* ignored_tag = nullptr;
    bool  ignored_ok  = false;
    while (completion_queue.Next(&ignored_tag, &ignored_ok)) {
    }
    return all_quiesced;
}

grpc::Status DecodeRpcServer::QuiesceRemoteLoad(grpc::ServerContext*                 server_context,
                                                const RemoteLoadQuiesceRequestPB* request,
                                                RemoteLoadQuiesceResponsePB*       response) {
    RTP_LLM_PROFILE_FUNCTION();
    response->set_quiesced(false);
    if (request->allocation_token().empty() || request->load_deadline_unix_ms() <= 0) {
        setRemoteLoadQuiesceResponseError(
            response,
            ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, "quiesce request has no valid allocation token"));
        return grpc::Status::OK;
    }
    if (!request->local_only()) {
        const auto owner_status = validateRemoteLoadAllocationOwner(request->allocation_token(), process_id_);
        if (!owner_status.ok()) {
            setRemoteLoadQuiesceResponseError(
                response, ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, owner_status.ToString()));
            return grpc::Status::OK;
        }
    }

    auto attempt_timeout = std::chrono::milliseconds(LOAD_TIMEOUT_MS);
    if (server_context != nullptr) {
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            server_context->deadline() - std::chrono::system_clock::now());
        attempt_timeout = std::min(attempt_timeout, std::max(remaining, std::chrono::milliseconds(1)));
    }
    const std::vector<std::string> worker_addrs = request->local_only() ? std::vector<std::string>{} :
                                                                          resource_.grpc_workers;
    const auto configured_retention_ms = maga_init_params_.pd_sep_config.load_cache_timeout_ms > 0 ?
                                             maga_init_params_.pd_sep_config.load_cache_timeout_ms :
                                             LOAD_TIMEOUT_MS;
    const auto retention_timeout = request->local_only() ?
                                       std::chrono::milliseconds(std::max<int64_t>(
                                           1,
                                           request->retention_timeout_ms() > 0 ?
                                               request->retention_timeout_ms() :
                                               configured_retention_ms)) :
                                       retentionTimeoutUntil(request->load_deadline_unix_ms());
    const bool all_quiesced = quiesceRemoteLoadTargets(request->allocation_token(),
                                                       request->load_deadline_unix_ms(),
                                                       worker_addrs,
                                                       attempt_timeout,
                                                       retention_timeout,
                                                       RemoteLoadFenceRegistry::UnseenTokenPolicy::Seal);
    response->set_quiesced(all_quiesced);
    if (!all_quiesced) {
        setRemoteLoadQuiesceResponseError(
            response, ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, "one or more remote load targets did not quiesce"));
    } else {
        setRemoteLoadQuiesceResponseError(response, ErrorInfo::OkStatus());
    }
    return grpc::Status::OK;
}

grpc::Status DecodeRpcServer::allocateResourceFunc(DecodeGenerateContext& decode_context) {
    EXECUTE_STAGE_FUNC(allocateResource, decode_context);
    return grpc::Status::OK;
}

grpc::Status DecodeRpcServer::RemoteGenerate(grpc::ServerContext* server_context, ServerStream* grpc_stream) {
    RTP_LLM_PROFILE_FUNCTION();
    c10::InferenceMode inference_guard(true);
    AtomicGuard        request_guard(onflight_requests_);
    DecodeRpcContext   rpc_context{grpc_stream};
    // TODO(xinfei.sxf) request id is 0 here
    auto decode_context              = DecodeGenerateContext(rpc_context, 0, server_context, metrics_reporter_, meta_);
    decode_context.onflight_requests = onflight_requests_;
    decode_context.loading_cache_requests = loading_cache_requests_;

    auto max_retry_times      = maga_init_params_.pd_sep_config.decode_retry_times;
    auto max_retry_timeout_ms = maga_init_params_.pd_sep_config.decode_retry_timeout_ms;
    int  retry_interval_ms    = maga_init_params_.pd_sep_config.decode_retry_interval_ms;

    try {
        EXECUTE_STAGE_FUNC(prepareGenerateContext, decode_context);
        EXECUTE_WITH_RETRY(
            allocateResourceFunc, decode_context, max_retry_times, max_retry_timeout_ms, retry_interval_ms);
        if (decode_context.hasError()) {
            RTP_LLM_LOG_WARNING("request [%s] allocate resource failed after retry %d times, cost time ms [%ld], "
                                "max retry time [%ld], max retry timeout ms [%ld]",
                                decode_context.request_key.c_str(),
                                decode_context.retry_times,
                                decode_context.retry_cost_time_ms,
                                max_retry_times + 1,
                                max_retry_timeout_ms);
            return decode_context.error_status;
        }
        EXECUTE_STAGE_FUNC(loadCacheFromPrefill, decode_context);
        EXECUTE_STAGE_FUNC(localGenerate, decode_context);
        decode_context.stat_info.nextStage();
    } catch (const std::exception& e) {
        auto error_msg              = "request [" + decode_context.request_key + "] catch exception [" + e.what() + "]";
        decode_context.error_status = grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
        return decode_context.error_status;
    } catch (...) {
        auto error_msg              = "request [" + decode_context.request_key + "] catch unknown exception";
        decode_context.error_status = grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
        return decode_context.error_status;
    }

    return grpc::Status::OK;
}

}  // namespace rtp_llm

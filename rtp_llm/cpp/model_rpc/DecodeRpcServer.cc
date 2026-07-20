#include <algorithm>
#include <mutex>
#include <memory>
#include <unistd.h>
#include <limits.h>
#include <condition_variable>
#include <unordered_set>
#include <c10/core/InferenceMode.h>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/KVCacheTransferPlanner.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"

using namespace std;
using namespace autil::legacy;

using grpc::Status;
using grpc::ClientContext;
using grpc::CompletionQueue;
using grpc::ClientAsyncResponseReader;

const int LOAD_TIMEOUT_MS         = 5 * 1000;
const int EXTRA_TIMEOUT_MS        = 100;
const int RDMA_CONNECT_RETRY_TIME = 3;

#define GRPC_RET_IF_ERROR(decode_context, stat, code, msg)                                                             \
    if (!(stat)) {                                                                                                     \
        decode_context.error_status = grpc::Status(code, msg);                                                         \
        return;                                                                                                        \
    }

string makeRequestKey(const string& client_id, size_t request_id) {
    return client_id + "_request_id_" + std::to_string(request_id);
}

namespace rtp_llm {

grpc::Status DecodeRpcServer::init(const EngineInitParams&                                maga_init_params,
                                   std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                                   py::object                                             mm_process_engine) {
    auto ret = RemoteRpcServer::init(maga_init_params, std::move(propose_params), mm_process_engine);
    if (!ret.ok()) {
        return ret;
    }
    return grpc::Status::OK;
}

std::string
DecodeRpcServer::makeMTPModuleCacheKey(size_t mtp_base_model_id, const std::string& token_id_str, size_t layer_id) {
    return makeCacheKey(mtp_base_model_id, token_id_str, layer_id);
}

std::string DecodeRpcServer::makeTaggedRequestKey(int64_t request_id, size_t layer_id, const std::string& tag) {
    return std::to_string(request_id) + "-" + std::to_string(layer_id) + "-tag-" + tag;
}

std::vector<DecodeRpcServer::MTPModuleLoadPlan>
DecodeRpcServer::makeMTPModuleLoadPlan(const ProposeModelEngineInitParams* propose_params) {
    if (propose_params == nullptr || propose_params->mtp_model_params_ == nullptr
        || propose_params->mtp_model_params_->empty() || propose_params->mtp_model_params_->front() == nullptr) {
        return {};
    }

    const auto* active_module = propose_params->mtp_model_params_->front().get();
    return {{/*module_index=*/0, active_module, active_module->model_id}};
}

void DecodeRpcServer::logReadFailures(int64_t                         request_id,
                                      const std::string&              peer_addr,
                                      ErrorCode                       error_code,
                                      const std::string&              error_message,
                                      const std::vector<std::string>& buffer_debug_infos) {
    if (error_code == ErrorCode::CANCELLED) {
        return;
    }
    if (buffer_debug_infos.empty()) {
        RTP_LLM_LOG_WARNING("PD_CACHE_KEY_READ_FAILED request_id=%ld peer=%s error_code=%d error=%s buffer={}",
                            static_cast<long>(request_id),
                            peer_addr.c_str(),
                            static_cast<int>(error_code),
                            error_message.c_str());
        return;
    }
    for (const auto& debug_info : buffer_debug_infos) {
        RTP_LLM_LOG_WARNING("PD_CACHE_KEY_READ_FAILED request_id=%ld peer=%s error_code=%d error=%s buffer={%s}",
                            static_cast<long>(request_id),
                            peer_addr.c_str(),
                            static_cast<int>(error_code),
                            error_message.c_str(),
                            debug_info.c_str());
    }
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
    if (maga_init_params_.parallelism_config.prefill_cp_config.kv_cache_sharded
        && maga_init_params_.parallelism_config.prefill_cp_config.is_prefill_enabled()) {
        const auto configured_prefill_cp_size = maga_init_params_.parallelism_config.prefill_cp_config.prefill_cp_size;
        RTP_LLM_CHECK_WITH_INFO(configured_prefill_cp_size > 1,
                                "decode PREFILL_CP sharded mode requires explicit PREFILL_CP_SIZE");
        decode_context.prefill_cp_size = static_cast<int32_t>(configured_prefill_cp_size);
    } else {
        decode_context.prefill_cp_size = 1;
    }
    RTP_LLM_LOG_DEBUG("request [%s] prepare generate context done, prefill_cp_size=%d",
                      decode_context.request_key.c_str(),
                      decode_context.prefill_cp_size);
}

void DecodeRpcServer::allocateResource(DecodeGenerateContext& decode_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%s] start to allocate resource", decode_context.request_key.c_str());
    auto input                        = QueryConverter::transQuery(&decode_context.allocate_request.input());
    auto generate_stream              = engine_->makeStream(input);
    decode_context.request_timeout_ms = generate_stream->getTimeoutMs();

    // Set CanRun event so that handleWaiting() will execute initKVBlock()
    generate_stream->reportEvent(StreamEvents::CanRun);
    decode_context.setStream(generate_stream);

    // WAITING -> LOADING_CACHE -> WAITING, 直到load cache完成并移动到 WAITING 状态
    // NOTE: 此处的 busy-wait 是安全的，因为 stream 尚未 enqueue 到 scheduler，
    // 不会与其他线程并发调用 moveToNext()。gRPC 线程独占驱动状态机直到 WAITING。
    while (!generate_stream->hasError() && generate_stream->moveToNext() == StreamState::LOADING_CACHE) {
        this_thread::sleep_for(chrono::milliseconds(1));
    }
    if (generate_stream->hasError()) {
        string error_msg = "request: [" + decode_context.request_key + "] malloc kv cache block failed at decode node";
        RTP_LLM_LOG_ERROR(error_msg);
        decode_context.error_status = grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, error_msg);
        return;
    }

    GRPC_RET_IF_ERROR(decode_context,
                      decode_context.rpc_context.grpc_stream->Write(GenerateOutputsPB()),
                      grpc::StatusCode::INTERNAL,
                      "failed to write allocate output");

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
    decode_context.time_info.updateLoadBeginTime();
    auto error_info = loadCacheForAllRank(decode_context);
    decode_context.time_info.updateLoadEndTime();
    if (!error_info.ok()) {
        RTP_LLM_LOG_WARNING("request [%s] load kv cache failed, error code [%s], cost time [%ld] ms",
                            decode_context.request_key.c_str(),
                            error_info.ToString().c_str(),
                            decode_context.time_info.loadCacheTimeMs());
    }

    GenerateOutputsPB load_response;
    load_response.mutable_error_info()->set_error_code(transErrorCodeToRPC(error_info.code()));
    GRPC_RET_IF_ERROR(
        decode_context, grpc_stream->Write(load_response), grpc::StatusCode::INTERNAL, "send load response failed");
    GRPC_RET_IF_ERROR(decode_context, error_info.ok(), grpc::StatusCode::INTERNAL, error_info.ToString().c_str());
    RTP_LLM_LOG_DEBUG("request [%s] load cache from prefill done", decode_context.request_key.c_str());
}

void DecodeRpcServer::localGenerate(DecodeGenerateContext& decode_context) {
    RTP_LLM_PROFILE_FUNCTION();
    cuda_graph::setDevice(static_cast<int>(maga_init_params_.parallelism_config.local_rank));
    RTP_LLM_LOG_DEBUG("request [%s] start to local generate", decode_context.request_key.c_str());
    auto&             grpc_stream     = decode_context.rpc_context.grpc_stream;
    auto&             generate_stream = decode_context.getStream();
    GenerateRequestPB generate_request;
    GRPC_RET_IF_ERROR(decode_context,
                      grpc_stream->Read(&generate_request),
                      grpc::StatusCode::INTERNAL,
                      "poll generate request failed");
    GRPC_RET_IF_ERROR(decode_context,
                      generate_request.stage() == RemoteStage::GENERATE,
                      grpc::StatusCode::INTERNAL,
                      "message first status != RemoteStage::GENERATE");
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
    {
        const auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        generate_stream->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{
            .epoch                 = 0,
            .last_sample_token_gpu = new_tokens.reshape({1}).to(cuda_i32),
            .next_seq_len_gpu      = torch::full({1}, static_cast<int64_t>(generate_stream->seqLength()), cuda_i32),
        });
    }
    if (generate_request.position_ids_size() > 0) {
        auto context_position_ids = torch::from_blob(const_cast<int32_t*>(generate_request.position_ids().data()),
                                                     {(int64_t)generate_request.position_ids_size()},
                                                     torch::kInt32)
                                        .clone();
        generate_stream->setContextPositionIds(context_position_ids);
    }
    if (propose_maga_init_params_) {
        const size_t propose_step = propose_maga_init_params_->gen_num_per_circle;
        RTP_LLM_CHECK_WITH_INFO(propose_step > 0, "decode rpc propose_step should be positive");
        if (maga_init_params_.sp_config.gen_num_per_cycle > 0) {
            RTP_LLM_CHECK_WITH_INFO(propose_step == static_cast<size_t>(maga_init_params_.sp_config.gen_num_per_cycle),
                                    "decode rpc propose_step mismatch, propose_params=%zu, sp_config=%ld",
                                    propose_step,
                                    maga_init_params_.sp_config.gen_num_per_cycle);
        }

        generate_stream->setReuseLength(generate_stream->seqLength() - 1);
        generate_stream->setSpEditRun(false);
        generate_stream->setMtpTokenIndex(generate_stream->seqLength() - 1);
        generate_stream->setContainProposeToken(true);
        std::vector<int> propose_tokens;
        propose_tokens.assign(generate_request.propose_token_ids().begin(), generate_request.propose_token_ids().end());
        RTP_LLM_CHECK_WITH_INFO(propose_tokens.size() >= 2,
                                "decode rpc propose_tokens should contain target and draft token, count=%zu",
                                propose_tokens.size());
        generate_stream->setProposeToken(propose_tokens);

        auto sp_output_buffer          = std::make_shared<SpeculativeExecutorStreamOutput>();
        sp_output_buffer->propose_step = propose_step;
        sp_output_buffer->tokens       = torch::zeros({1, (int64_t)propose_tokens.size()},
                                                torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
        memcpy(sp_output_buffer->tokens.data_ptr<int>(), propose_tokens.data(), propose_tokens.size() * sizeof(int));

        auto propose_probs_t  = QueryConverter::transTensor(generate_request.propose_probs());
        auto propose_hidden_t = QueryConverter::transTensor(generate_request.propose_hidden());

        const auto cuda_i32             = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        sp_output_buffer->all_probs     = propose_probs_t.to(torch::kCUDA);
        sp_output_buffer->hidden_states = propose_hidden_t.to(torch::kCUDA);

        auto propose_tokens_gpu = torch::empty({1}, cuda_i32);
        auto accept_len         = torch::ones({1}, cuda_i32);
        auto accept_tokens      = torch::zeros({1, static_cast<int64_t>(propose_step + 1)}, cuda_i32);
        accept_tokens[0][0]     = sp_output_buffer->tokens[0][0];
        propose_tokens_gpu[0]   = sp_output_buffer->tokens[0][1];

        auto next_seq_len = torch::ones({1}, cuda_i32);
        next_seq_len[0]   = generate_stream->seqLength();

        generate_stream->setSPOutputBuffer(sp_output_buffer);
        generate_stream->setMtpAsyncDeviceState(GenerateStream::MtpAsyncDeviceState{
            .epoch                  = 0,
            .accept_len_gpu         = std::move(accept_len),
            .accept_tokens_gpu      = std::move(accept_tokens),
            .next_seq_len_gpu       = std::move(next_seq_len),
            .propose_tokens_gpu     = std::move(propose_tokens_gpu),
            .last_hidden_states_gpu = sp_output_buffer->hidden_states,
            .draft_all_probs_gpu    = sp_output_buffer->all_probs,
            .last_real_seq_len      = generate_stream->seqLength(),
            .next_real_seq_len      = generate_stream->seqLength(),
        });
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
    request.set_prefill_cp_size(load_context.prefill_cp_size);

    if (load_context.prefill_cp_size > 1) {
        // CP-sharded prefill: every decode rank must pull the shard owned by
        // every prefill CP peer.
        for (const auto& addr : peer_addrs) {
            request.add_peer_addrs(addr);
        }
    } else if (resource_.workers.size() % peer_addrs.size() == 0) {
        // D >= P
        int part_cnt = resource_.workers.size() / peer_addrs.size();
        request.add_peer_addrs(peer_addrs[index / part_cnt]);
    } else {
        // P >= D, load multi block of prefill
        int group_num = peer_addrs.size() / resource_.workers.size();
        request.add_peer_addrs(peer_addrs[index * group_num]);
    }
    for (auto& cache_key : load_context.cache_keys) {
        request.add_cache_keys(cache_key);
    }
    if (!load_context.block_ids_by_group.empty()) {
        const auto& topology = engine_->resourceContext().cache_manager->cacheConfig().topology();
        for (size_t group_id = 0; group_id < load_context.block_ids_by_group.size(); ++group_id) {
            const auto& group_block = load_context.block_ids_by_group[group_id];
            RTP_LLM_CHECK_WITH_INFO(group_block != nullptr, "null group_block in block_ids_by_group");
            auto* tagged_row = request.add_tagged_group_block_ids();
            tagged_row->set_tag(topology.groupById(group_id).tag);
            for (const auto& block_id : group_block->blocks()) {
                tagged_row->add_block_ids(block_id);
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
    request.set_prefill_cp_size(load_context.prefill_cp_size);
    if (load_context.prefill_cp_size > 1) {
        // CP-sharded prefill: each peer owns a page-level or in-page shard.
        // Keep one logical partition and let loadCache route groups/blocks to
        // the owning peer.
        request.set_partition_count(1);
        request.set_partition_id(0);
        for (const auto& addr : peer_addrs) {
            request.add_peer_addrs(addr);
        }
    } else if (maga_init_params_.parallelism_config.prefill_cp_config.is_prefill_enabled()) {
        // Prefill worker has full KV cache on each rank.
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
        const auto& topology = engine_->resourceContext().cache_manager->cacheConfig().topology();
        for (size_t group_id = 0; group_id < load_context.block_ids_by_group.size(); ++group_id) {
            const auto& group_block = load_context.block_ids_by_group[group_id];
            RTP_LLM_CHECK_WITH_INFO(group_block != nullptr, "null group_block in block_ids_by_group");
            auto* tagged_row = request.add_tagged_group_block_ids();
            tagged_row->set_tag(topology.groupById(group_id).tag);
            for (const auto& block_id : group_block->blocks()) {
                tagged_row->add_block_ids(block_id);
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

    LoadKVCacheContext load_context{decode_context.request_id,
                                    decode_context.request_key,
                                    decode_context.peer_addrs,
                                    cache_keys,
                                    block_ids_by_group,
                                    generate_stream->reuseBlockSize(),
                                    min_timeout_ms,
                                    1,
                                    0,
                                    decode_context.server_context,
                                    decode_context.prefill_cp_size};

    // Prefill: TP = 1 && Decode: TP = 1
    if (resource_.workers.size() == 1 && decode_context.peer_addrs.size() == 1) {
        for (size_t i = 0; i < maga_init_params_.pd_sep_config.rdma_connect_retry_times + 1; i++) {
            auto error_info = loadCache(load_context);
            if (error_info.code() != ErrorCode::CACHE_STORE_LOAD_CONNECT_FAILED
                && error_info.code() != ErrorCode::CACHE_STORE_LOAD_RDMA_CONNECT_FAILED) {
                return error_info;
            }
        }
    }

    return loadCacheAsyncForTp(decode_context, load_context);
}

ErrorInfo DecodeRpcServer::loadCacheAsyncForTp(DecodeGenerateContext& decode_context,
                                               LoadKVCacheContext&    load_context) {
    RTP_LLM_PROFILE_FUNCTION();
    int64_t load_cache_begin_time_us = currentTimeUs();

    struct WorkerRpcContext {
        WorkerRpcContext() {
            client_context = make_shared<ClientContext>();
        }
        BroadcastLoadResponsePB           response;
        Status                            status;
        std::shared_ptr<RpcService::Stub> stub;
        std::shared_ptr<ClientContext>    client_context;
    };

    uint32_t                 worker_size = resource_.grpc_workers.size();
    vector<WorkerRpcContext> all_context(worker_size);
    uint32_t                 cq_size = worker_size % 2 == 0 ? worker_size / 2 : worker_size / 2 + 1;
    vector<CompletionQueue>  completion_queues(cq_size);
    vector<int>              each_finished_count(cq_size, 0);
    if (worker_size == 0 || cq_size == 0) {
        RTP_LLM_LOG_WARNING("request:[%s] cq_size or worker_size is 0, worker size = %d, cq size = %d",
                            decode_context.request_key.c_str(),
                            worker_size,
                            cq_size);
        return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, "worker size or cq size is 0");
    }
    auto worker_size_per_queue = worker_size / completion_queues.size();
    RTP_LLM_LOG_DEBUG("request:[%s] start to async remote load for all rank", decode_context.request_key.c_str());
    for (int i = 0; i < worker_size; i++) {
        auto& worker         = resource_.grpc_workers[i];
        auto  connect_status = resource_.rpc_pool.getConnection(worker);
        if (!connect_status.ok()) {
            string error_msg = "get grpc connection for rank:" + std::to_string(i) + ", addr:" + worker + " failed";
            return ErrorInfo(ErrorCode::GET_CONNECTION_FAILED, error_msg);
        }
        all_context.push_back(WorkerRpcContext());
        auto& rpc_context = all_context[i];
        rpc_context.stub  = connect_status.value().stub;
        BroadcastLoadRequestPB load_request;

        if (engine_->resourceContext().cache_manager->cacheConfig().use_mla) {
            load_request = constructRemoteLoadRequestForMla(load_context, i, decode_context.peer_addrs);
        } else {
            load_request = constructRemoteLoadRequest(load_context, i, decode_context.peer_addrs);
        }
        std::unique_ptr<ClientAsyncResponseReader<BroadcastLoadResponsePB>> reader(rpc_context.stub->AsyncRemoteLoad(
            rpc_context.client_context.get(), load_request, &completion_queues[i % completion_queues.size()]));
        reader->Finish(&rpc_context.response, &rpc_context.status, reinterpret_cast<void*>(i));
    }

    bool        all_success               = true;
    size_t      finished_count            = 0;
    auto        total_timeout_ms          = load_context.timeout_ms + EXTRA_TIMEOUT_MS;
    ErrorCode   error_code                = ErrorCode::NONE_ERROR;
    std::string error_msg                 = "failed to load kv cache in rank: ";
    int64_t     min_response_done_time_us = 1lu << 60;
    int64_t     max_response_done_time_us = 0;
    while (true) {
        RTP_LLM_LOG_DEBUG("request [%s] load cache loop step", decode_context.request_key.c_str());
        auto cost_time_ms = (currentTimeUs() - load_cache_begin_time_us) / 1000;
        if (cost_time_ms > total_timeout_ms) {
            error_msg = "load cache timeout : cost time is " + std::to_string(cost_time_ms)
                        + "ms, "
                          "total timeout for load cache is "
                        + std::to_string(total_timeout_ms) + "ms";
            return ErrorInfo(ErrorCode::LOAD_CACHE_TIMEOUT, error_msg);
        }
        if (load_context.server_context->IsCancelled()) {
            string error_msg = "request is cancelled";
            return ErrorInfo(ErrorCode::CANCELLED, error_msg);
        }
        auto once_deadline =
            std::chrono::system_clock::now()
            + std::chrono::milliseconds(maga_init_params_.pd_sep_config.decode_polling_kv_cache_step_ms);
        RTP_LLM_LOG_DEBUG("request [%s] start to execute async next", decode_context.request_key.c_str());
        // TODO(xinfei.sxf) There is a problem with complete queue next call delay here, the reason is yet to be
        // investigated
        void* got_tag;
        bool  ok = false;
        for (uint32_t i = 0; i < completion_queues.size(); i++) {
            if (each_finished_count[i] == worker_size_per_queue) {
                continue;
            }
            if (completion_queues[i].AsyncNext(&got_tag, &ok, once_deadline)
                == grpc::CompletionQueue::NextStatus::TIMEOUT) {
                RTP_LLM_LOG_DEBUG("request [%s] async next timeout", decode_context.request_key.c_str());
                continue;
            }
            each_finished_count[i]++;
            if (!ok) {
                string error_msg = "async get next event from grpc completion queue failed";
                return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, error_msg);
            }
            auto        rank             = reinterpret_cast<uintptr_t>(got_tag);
            const auto& status           = all_context[rank].status;
            const auto& response         = all_context[rank].response;
            const auto& pb_error_code    = response.error_info().error_code();
            const auto& pb_error_message = response.error_info().error_message();
            min_response_done_time_us    = std::min(min_response_done_time_us, response.done_time_us());
            max_response_done_time_us    = std::max(max_response_done_time_us, response.done_time_us());
            RTP_LLM_LOG_DEBUG("request [%s] load cache for rank [%d] done", decode_context.request_key.c_str(), rank);
            if (!status.ok()) {
                all_success = false;
                error_code  = ErrorCode::LOAD_KV_CACHE_FAILED;
                error_msg += std::to_string(rank) + ": " + status.error_message() + ", ";
            } else if (pb_error_code != ErrorCodePB::NONE_ERROR) {
                all_success = false;
                error_code  = transRPCErrorCode(pb_error_code);
                error_msg += std::to_string(rank) + ": " + pb_error_message + ", ";
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

    for (auto& completion_queue : completion_queues) {
        completion_queue.Shutdown();
    }

    if (finished_count != worker_size) {
        all_success = false;
    }
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
    const auto& request_key   = load_context.request_key;
    auto        cache_manager = engine_->resourceContext().cache_manager;
    const auto& cache_config  = cache_manager->cacheConfig();
    auto        layer_num     = maga_init_params_.model_config_.num_layers;

    const int peer_cnt = static_cast<int>(load_context.peer_addrs.size());
    RTP_LLM_CHECK_WITH_INFO(peer_cnt > 0, "peer_addrs is empty");

    const bool   use_mla             = cache_config.use_mla;
    const bool   use_hybrid          = cache_config.groupNums() > 1;
    const bool   use_opaque_kv_store = cache_config.use_opaque_kv_cache_store;
    const auto&  spec                = cache_config.specForGroup(0);
    const size_t k_total_bytes       = spec->k_block_size_bytes();
    const size_t v_total_bytes       = spec->v_block_size_bytes();

    if (!use_mla && !use_opaque_kv_store && peer_cnt > 1) {
        RTP_LLM_CHECK_WITH_INFO(k_total_bytes % static_cast<size_t>(peer_cnt) == 0,
                                "k_block bytes[%zu] not divisible by peer_cnt[%d]",
                                k_total_bytes,
                                peer_cnt);
        RTP_LLM_CHECK_WITH_INFO(v_total_bytes % static_cast<size_t>(peer_cnt) == 0,
                                "v_block bytes[%zu] not divisible by peer_cnt[%d]",
                                v_total_bytes,
                                peer_cnt);
    }

    auto cancel_check_func  = [&load_context]() -> bool { return load_context.server_context->IsCancelled(); };
    auto start_load_time_us = currentTimeUs();
    std::vector<std::pair<std::string, std::shared_ptr<LoadContext>>> load_contexts;
    auto buffersDebugInfos = [](const std::vector<std::shared_ptr<RequestBlockBuffer>>& buffers) {
        std::vector<std::string> debug_infos;
        debug_infos.reserve(buffers.size());
        for (const auto& buffer : buffers) {
            debug_infos.push_back(buffer == nullptr ? "null" : buffer->debugInfo());
        }
        return debug_infos;
    };
    const bool is_page_level_rr = load_context.prefill_cp_size > 1
                                  && static_cast<int>(load_context.peer_addrs.size()) == load_context.prefill_cp_size;
    auto layerGroupIds = [](const CacheConfig& cfg, bool use_hybrid, size_t layer_id) {
        std::vector<int> layer_gids;
        if (use_hybrid) {
            const auto layer_group_ids = cfg.layerGroupIdsSnapshot();
            RTP_LLM_CHECK_WITH_INFO(layer_id < layer_group_ids.size(),
                                    "hybrid cache layer %zu missing layer_to_group_ids, size=%zu",
                                    layer_id,
                                    layer_group_ids.size());
            RTP_LLM_CHECK_WITH_INFO(
                !layer_group_ids[layer_id].empty(), "hybrid cache layer %zu has empty layer_to_group_ids", layer_id);
            layer_gids = layer_group_ids[layer_id];
        } else {
            layer_gids.push_back(0);
        }
        return layer_gids;
    };
    auto groupType = [](const CacheConfig& cfg, bool use_hybrid, size_t gid) {
        if (use_hybrid && static_cast<int>(gid) < cfg.groupNums()) {
            return cfg.typeForGroup(gid);
        }
        return CacheGroupType::FULL;
    };
    auto groupTag = [](const CacheConfig& cfg, size_t gid) -> std::string {
        RTP_LLM_CHECK_WITH_INFO(static_cast<int>(gid) < cfg.groupNums(),
                                "cache group id out of range: gid=%zu group_num=%d",
                                gid,
                                cfg.groupNums());
        return cfg.tagForGroup(gid);
    };
    auto cpMapperForGroup = [&](const CacheConfig& cfg, size_t gid) {
        return CPSlotMapper(load_context.prefill_cp_size - 1,
                            load_context.prefill_cp_size,
                            static_cast<int>(cfg.seqSizePerBlockForGroup(gid)));
    };
    auto groupUsesCpSlice = [&](const CacheConfig& cfg, size_t gid) {
        if (load_context.prefill_cp_size <= 1 || static_cast<int>(gid) >= cfg.groupNums()) {
            return false;
        }
        return cpMapperForGroup(cfg, gid).layoutForGroup(cfg, gid).slice != CpBlockSliceMode::NONE;
    };
    auto shouldLoadGroupFromPeer = [&](const CacheConfig& cfg, CacheGroupType group_type, size_t gid, int peer_idx) {
        if (!is_page_level_rr) {
            return true;
        }
        if (group_type == CacheGroupType::FULL) {
            return true;
        }
        // Some specs are CP-sliced inside one logical block on prefill, while
        // decode still owns the full block. Pull every peer slice and place it
        // into the destination offset declared by the spec.
        return groupUsesCpSlice(cfg, gid) || peer_idx == 0;
    };
    auto shouldLoadBlockFromPeer = [&](CacheGroupType group_type, size_t block_pos, int peer_idx) {
        if (!is_page_level_rr || group_type != CacheGroupType::FULL) {
            return true;
        }
        return (static_cast<int>(block_pos) % load_context.prefill_cp_size) == peer_idx;
    };
    auto sliceCpDestinationForPeer = [&](std::vector<BlockInfo> parts,
                                         const CacheConfig&     cfg,
                                         size_t                 gid,
                                         int                    peer_idx) {
        if (!is_page_level_rr || !groupUsesCpSlice(cfg, gid) || load_context.prefill_cp_size <= 1) {
            return parts;
        }
        return cpMapperForGroup(cfg, gid).sliceBlockForPeer(cfg, gid, std::move(parts), static_cast<size_t>(peer_idx));
    };
    auto isCompactFixedBlockTable = [&](const CacheConfig& cfg, size_t gid) {
        if (!is_page_level_rr || !groupUsesCpSlice(cfg, gid) || load_context.prefill_cp_size <= 1) {
            return false;
        }
        const auto group_tokens = cfg.seqSizePerBlockForGroup(gid);
        return group_tokens > 0
               && group_tokens == cfg.seq_size_per_block * static_cast<size_t>(load_context.prefill_cp_size);
    };
    auto blockPositionsForLoad =
        [&](size_t block_num, const CacheConfig& cfg, bool cfg_use_hybrid, CacheGroupType group_type, size_t gid) {
            const auto   policy = cfg.policyForGroup(gid);
            const size_t tail_block_count =
                policy.active_tail_blocks > 0 ? static_cast<size_t>(policy.active_tail_blocks) : 0;
            const bool transfer_tail_blocks = tail_block_count > 0;
            if (!is_page_level_rr || !groupUsesCpSlice(cfg, gid) || load_context.prefill_cp_size <= 1) {
                return blockPositionsForCacheTransfer(block_num,
                                                      load_context.reuse_block_size,
                                                      cfg_use_hybrid,
                                                      transfer_tail_blocks,
                                                      tail_block_count,
                                                      /*hybrid_full_from_begin=*/true);
            }
            if (isCompactFixedBlockTable(cfg, gid)) {
                return blockPositionsForCacheTransfer(block_num,
                                                      load_context.reuse_block_size,
                                                      cfg_use_hybrid,
                                                      transfer_tail_blocks,
                                                      tail_block_count,
                                                      /*hybrid_full_from_begin=*/true);
            }

            std::vector<size_t> block_pos_list;
            if (block_num == 0) {
                return block_pos_list;
            }
            const size_t cp_size        = static_cast<size_t>(load_context.prefill_cp_size);
            const size_t compact_blocks = (block_num + cp_size - 1) / cp_size;
            const size_t reuse_blocks   = static_cast<size_t>(std::max<int64_t>(load_context.reuse_block_size, 0));
            const size_t tail_count     = std::max<size_t>(1, tail_block_count);
            const size_t start = cfg_use_hybrid ? (compact_blocks > tail_count ? compact_blocks - tail_count : 0) :
                                                  std::min(reuse_blocks, compact_blocks);
            block_pos_list.reserve(compact_blocks - start);
            for (size_t compact_pos = start; compact_pos < compact_blocks; ++compact_pos) {
                block_pos_list.push_back(std::min((compact_pos + 1) * cp_size - 1, block_num - 1));
            }
            return block_pos_list;
        };
    auto cacheKeyIndexForBlock =
        [&](const CacheConfig& cfg, size_t gid, size_t block_pos, size_t cache_key_count, size_t& cache_key_index) {
            if (cache_key_count == 0) {
                return false;
            }
            cache_key_index = block_pos;
            if (isCompactFixedBlockTable(cfg, gid)) {
                cache_key_index = std::min((block_pos + 1) * static_cast<size_t>(load_context.prefill_cp_size) - 1,
                                           cache_key_count - 1);
            }
            return cache_key_index < cache_key_count;
        };
    for (int i = 0; i < load_context.peer_addrs.size(); i++) {
        auto&                                            peer_addr = load_context.peer_addrs[i];
        std::vector<std::shared_ptr<RequestBlockBuffer>> layer_caches;
        RTP_LLM_LOG_DEBUG("load context request id is %d", load_context.request_id);

        for (size_t layer_id = 0; layer_id < layer_num; layer_id++) {
            // Some typed-region cache layouts let one logical layer own
            // multiple groups. Iterate every group the layer owns; other
            // layouts reduce to the legacy one-gid-per-layer behaviour.
            std::vector<int> layer_gids = layerGroupIds(cache_config, use_hybrid, layer_id);

            for (int gid_int : layer_gids) {
                const size_t gid         = static_cast<size_t>(gid_int);
                const auto   tag         = groupTag(cache_config, gid);
                auto         request_key = makeTaggedRequestKey(load_context.request_id, layer_id, tag);
                auto         load_layer_cache =
                    std::make_shared<RequestBlockBuffer>(std::to_string(load_context.request_id), request_key);

                RTP_LLM_CHECK_WITH_INFO(gid < load_context.block_ids_by_group.size(),
                                        "group id out of range: gid=%zu group_num=%zu",
                                        gid,
                                        load_context.block_ids_by_group.size());
                RTP_LLM_CHECK_WITH_INFO(
                    load_context.block_ids_by_group[gid] != nullptr, "null group_block: gid=%zu", gid);
                const auto& block_ids = load_context.block_ids_by_group[gid]->blocks();
                auto        block_num = block_ids.size();
                size_t      model_id  = maga_init_params_.model_id;

                CacheGroupType group_type = groupType(cache_config, use_hybrid, gid);
                auto block_pos_list       = blockPositionsForLoad(block_num, cache_config, use_hybrid, group_type, gid);

                if (!shouldLoadGroupFromPeer(cache_config, group_type, gid, i)) {
                    continue;
                }
                for (size_t block_pos : block_pos_list) {
                    if (!shouldLoadBlockFromPeer(group_type, block_pos, i)) {
                        continue;
                    }
                    auto block_id = block_ids[block_pos];
                    if (isNullBlockIdx(block_id)) {
                        continue;
                    }
                    size_t cache_key_index = 0;
                    if (!cacheKeyIndexForBlock(
                            cache_config, gid, block_pos, load_context.cache_keys.size(), cache_key_index)) {
                        continue;
                    }
                    auto cache_key =
                        makeCacheKey(model_id, std::to_string(load_context.cache_keys[cache_key_index]), layer_id, tag);

                    const bool             use_kv_key_prefix  = use_mla || use_opaque_kv_store || use_hybrid;
                    const bool             use_whole_kv_block = is_page_level_rr || use_kv_key_prefix;
                    std::vector<BlockInfo> parts;
                    if (use_whole_kv_block) {
                        parts = cache_manager->convertIndexToBufferByTag(block_id, layer_id, tag);
                    } else {
                        parts = cache_manager->convertIndexToBufferByTag(block_id, layer_id, tag, peer_cnt, i);
                    }

                    parts            = sliceCpDestinationForPeer(std::move(parts), cache_config, gid, i);
                    auto addBufBlock = [&](const std::string& key, const BlockInfo& block) {
                        RTP_LLM_CHECK_WITH_INFO(block.addr != nullptr, "null block addr for key=%s", key.c_str());
                        RTP_LLM_CHECK_WITH_INFO(block.size_bytes > 0, "zero block size for key=%s", key.c_str());
                        RTP_LLM_LOG_DEBUG("PD_CACHE_KEY_READ_BLOCK key=%s request_id=%ld tag=%s layer=%zu "
                                          "peer_idx=%d peer=%s cp_size=%d block_pos=%zu block_id=%d addr=%p len=%zu",
                                          key.c_str(),
                                          static_cast<long>(load_context.request_id),
                                          tag.c_str(),
                                          layer_id,
                                          i,
                                          peer_addr.c_str(),
                                          load_context.prefill_cp_size,
                                          block_pos,
                                          block_id,
                                          block.addr,
                                          block.size_bytes);
                        std::shared_ptr<void> addr(block.addr, [](void*) {});
                        load_layer_cache->addBlock(
                            key, addr, static_cast<uint32_t>(block.size_bytes), block.is_cuda, true);
                    };

                    if (use_kv_key_prefix) {
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
        }

        if (engine_->isMTPEagle()) {
            if (propose_maga_init_params_ && propose_maga_init_params_->mtp_model_params_
                && !propose_maga_init_params_->mtp_model_params_->empty()) {
                const auto mtp_load_plan = makeMTPModuleLoadPlan(propose_maga_init_params_);
                if (mtp_load_plan.empty()) {
                    return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, "active MTP module0 is missing");
                }

                for (const auto& module_plan : mtp_load_plan) {
                    const size_t            mtp_model_id           = module_plan.module_index;
                    const EngineInitParams* mtp_engine_init_params = module_plan.engine_init_params;

                    const auto&  mtp_cache_cfg = cache_manager->getMTPModuleCacheConfig(static_cast<int>(mtp_model_id));
                    const size_t layer_num     = mtp_engine_init_params->model_config_.num_layers;

                    RTP_LLM_CHECK_WITH_INFO(layer_num == mtp_cache_cfg.layer_num,
                                            "mtp layer_num mismatch: engine=" + std::to_string(layer_num)
                                                + " cache_cfg=" + std::to_string(mtp_cache_cfg.layer_num)
                                                + " (mtp_model_id=" + std::to_string(mtp_model_id) + ")");

                    for (size_t layer_id = 0; layer_id < layer_num; layer_id++) {
                        const bool mtp_use_hybrid          = mtp_cache_cfg.groupNums() > 1;
                        const bool mtp_use_opaque_kv_store = mtp_cache_cfg.use_opaque_kv_cache_store;

                        // Same multi-group iteration as the main path.
                        std::vector<int> mtp_layer_gids = layerGroupIds(mtp_cache_cfg, mtp_use_hybrid, layer_id);

                        const auto global_layer_id = CacheConfig::mtpGlobalLayerId(
                            static_cast<uint32_t>(maga_init_params_.model_config_.num_layers),
                            static_cast<int>(mtp_model_id),
                            static_cast<uint32_t>(layer_num),
                            static_cast<int>(layer_id));
                        RTP_LLM_CHECK_WITH_INFO(global_layer_id != std::numeric_limits<uint32_t>::max(),
                                                "invalid decode MTP global layer: main=%ld module=%zu "
                                                "module_layers=%zu local=%zu",
                                                maga_init_params_.model_config_.num_layers,
                                                mtp_model_id,
                                                layer_num,
                                                layer_id);

                        for (int gid_int : mtp_layer_gids) {
                            const size_t gid         = static_cast<size_t>(gid_int);
                            const auto   tag         = groupTag(mtp_cache_cfg, gid);
                            auto         request_key = makeTaggedRequestKey(load_context.request_id, layer_id, tag);
                            auto         load_layer_cache = std::make_shared<RequestBlockBuffer>(
                                std::to_string(load_context.request_id), request_key);

                            RTP_LLM_CHECK_WITH_INFO(gid < load_context.block_ids_by_group.size(),
                                                    "mtp group id out of range: gid=%zu group_num=%zu",
                                                    gid,
                                                    load_context.block_ids_by_group.size());
                            RTP_LLM_CHECK_WITH_INFO(
                                load_context.block_ids_by_group[gid] != nullptr, "null mtp group_block: gid=%zu", gid);
                            const auto& block_ids = load_context.block_ids_by_group[gid]->blocks();
                            auto        block_num = block_ids.size();
                            size_t      model_id  = module_plan.cache_model_id;

                            CacheGroupType group_type = groupType(mtp_cache_cfg, mtp_use_hybrid, gid);
                            auto           block_pos_list =
                                blockPositionsForLoad(block_num, mtp_cache_cfg, mtp_use_hybrid, group_type, gid);

                            if (!shouldLoadGroupFromPeer(mtp_cache_cfg, group_type, gid, i)) {
                                continue;
                            }
                            for (size_t block_pos : block_pos_list) {
                                if (!shouldLoadBlockFromPeer(group_type, block_pos, i)) {
                                    continue;
                                }
                                auto block_id = block_ids[block_pos];
                                if (isNullBlockIdx(block_id)) {
                                    continue;
                                }
                                size_t cache_key_index = 0;
                                if (!cacheKeyIndexForBlock(mtp_cache_cfg,
                                                           gid,
                                                           block_pos,
                                                           load_context.cache_keys.size(),
                                                           cache_key_index)) {
                                    continue;
                                }
                                auto cache_key = makeCacheKey(
                                    model_id, std::to_string(load_context.cache_keys[cache_key_index]), layer_id, tag);
                                const bool mtp_use_mla = mtp_cache_cfg.use_mla;
                                const bool mtp_use_kv_key_prefix =
                                    mtp_use_mla || mtp_use_opaque_kv_store || mtp_use_hybrid;
                                const bool mtp_use_whole_kv_block = is_page_level_rr || mtp_use_kv_key_prefix;
                                std::vector<BlockInfo> parts;
                                if (mtp_use_whole_kv_block) {
                                    parts = cache_manager->convertIndexToBufferByTag(block_id, global_layer_id, tag);
                                } else {
                                    parts = cache_manager->convertIndexToBufferByTag(
                                        block_id, global_layer_id, tag, peer_cnt, i);
                                }

                                parts            = sliceCpDestinationForPeer(std::move(parts), mtp_cache_cfg, gid, i);
                                auto addBufBlock = [&](const std::string& key, const BlockInfo& block) {
                                    RTP_LLM_CHECK_WITH_INFO(
                                        block.addr != nullptr, "null block addr for key=%s", key.c_str());
                                    RTP_LLM_CHECK_WITH_INFO(
                                        block.size_bytes > 0, "zero block size for key=%s", key.c_str());
                                    RTP_LLM_LOG_DEBUG("PD_CACHE_KEY_READ_BLOCK key=%s request_id=%ld tag=%s layer=%zu "
                                                      "model_id=%zu mtp_module=%zu peer_idx=%d peer=%s cp_size=%d "
                                                      "block_pos=%zu key_index=%zu block_id=%d addr=%p len=%zu",
                                                      key.c_str(),
                                                      static_cast<long>(load_context.request_id),
                                                      tag.c_str(),
                                                      layer_id,
                                                      model_id,
                                                      mtp_model_id,
                                                      i,
                                                      peer_addr.c_str(),
                                                      load_context.prefill_cp_size,
                                                      block_pos,
                                                      cache_key_index,
                                                      block_id,
                                                      block.addr,
                                                      block.size_bytes);
                                    std::shared_ptr<void> addr(block.addr, [](void*) {});
                                    load_layer_cache->addBlock(
                                        key, addr, static_cast<uint32_t>(block.size_bytes), block.is_cuda, true);
                                };

                                if (mtp_use_kv_key_prefix) {
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
            } else {
                return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, "active MTP module0 is missing");
            }
        }

        auto ip_parts = autil::StringUtil::split(peer_addr, ":");
        if (ip_parts.size() != 3) {
            logReadFailures(load_context.request_id,
                            peer_addr,
                            ErrorCode::LOAD_KV_CACHE_FAILED,
                            "invalid_peer",
                            buffersDebugInfos(layer_caches));
            return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, "invalid peer ip");
        }

        auto layer_cache_load_context =
            resource_.cache_store->loadBuffers(layer_caches,
                                               ip_parts[0],
                                               autil::StringUtil::strToInt32WithDefault(ip_parts[1].c_str(), 0),
                                               autil::StringUtil::strToInt32WithDefault(ip_parts[2].c_str(), 0),
                                               load_context.timeout_ms,
                                               cancel_check_func,
                                               load_context.partition_count,
                                               load_context.partition_id);
        if (!layer_cache_load_context) {
            logReadFailures(load_context.request_id,
                            peer_addr,
                            ErrorCode::LOAD_KV_CACHE_FAILED,
                            "null_load_context",
                            buffersDebugInfos(layer_caches));
            return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, "load kv cache failed");
        }
        load_contexts.emplace_back(peer_addr, layer_cache_load_context);
    }

    for (auto& [peer_addr, layer_cache_load_context] : load_contexts) {
        layer_cache_load_context->waitDone();
        if (layer_cache_load_context->success()) {
            RTP_LLM_LOG_DEBUG("request [%s] load kv cache success", request_key.c_str());
        } else {
            // TODO(xinfei.sxf) add retry for part failed blocks.
            auto       load_done_time_us = currentTimeUs();
            const auto error_info        = layer_cache_load_context->getErrorInfo();
            logReadFailures(load_context.request_id,
                            peer_addr,
                            error_info.code(),
                            error_info.ToString(),
                            layer_cache_load_context->failedBlockDebugInfos());
            RTP_LLM_LOG_WARNING("request [%s] load cache failed, status [%s], cost time [%ld] ms",
                                request_key.c_str(),
                                layer_cache_load_context->getErrorInfoString().c_str(),
                                (load_done_time_us - start_load_time_us) / 1000);
            return error_info;
        }
    }

    return ErrorInfo::OkStatus();
}

grpc::Status DecodeRpcServer::RemoteLoad(grpc::ServerContext*          server_context,
                                         const BroadcastLoadRequestPB* request,
                                         BroadcastLoadResponsePB*      response) {
    RTP_LLM_PROFILE_FUNCTION();
    if (request->dp_rank() != maga_init_params_.parallelism_config.dp_rank) {
        RTP_LLM_LOG_WARNING("only load when in dp group, skip load for dp rank %d", request->dp_rank());
        return grpc::Status::OK;
    }

    std::vector<CacheKeyType> cache_keys(request->cache_keys().begin(), request->cache_keys().end());
    const auto&               cache_config       = engine_->resourceContext().cache_manager->cacheConfig();
    const auto&               topology           = cache_config.topology();
    GroupBlockIds             block_ids_by_group = decodeGroupBlockIds(*request, topology);

    std::vector<std::string> peer_addrs(request->peer_addrs().begin(), request->peer_addrs().end());

    // TODO(xinfei.sxf) add retry
    auto error_info = loadCache({request->request_id(),
                                 request->request_key(),
                                 peer_addrs,
                                 cache_keys,
                                 block_ids_by_group,
                                 request->reuse_block_size(),
                                 request->timeout_ms(),
                                 request->partition_count(),
                                 request->partition_id(),
                                 server_context,
                                 request->prefill_cp_size() > 0 ? request->prefill_cp_size() : 1});
    response->mutable_error_info()->set_error_code(transErrorCodeToRPC(error_info.code()));
    response->mutable_error_info()->set_error_message(error_info.ToString());
    response->set_done_time_us(currentTimeUs());
    RTP_LLM_LOG_DEBUG("request: %s, remote load cache grpc done", request->request_key().c_str());
    return grpc::Status::OK;
}

GroupBlockIds DecodeRpcServer::decodeGroupBlockIds(const BroadcastLoadRequestPB& request,
                                                   const CacheTopology&          topology) {
    GroupBlockIds block_ids_by_group(topology.groups().size());
    for (const auto& tagged_row : request.tagged_group_block_ids()) {
        const auto group_id = topology.groupIdForTag(tagged_row.tag());
        RTP_LLM_CHECK_WITH_INFO(
            block_ids_by_group[group_id] == nullptr, "duplicate RPC cache tag=%s", tagged_row.tag().c_str());
        auto holder = std::make_shared<BlockIds>();
        holder->assign(BlockIndicesType(tagged_row.block_ids().begin(), tagged_row.block_ids().end()));
        block_ids_by_group[group_id] = std::move(holder);
    }
    RTP_LLM_CHECK_WITH_INFO(
        std::all_of(block_ids_by_group.begin(), block_ids_by_group.end(), [](const auto& value) { return value; }),
        "RPC cache tag set does not match local topology");
    return block_ids_by_group;
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

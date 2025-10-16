#include <mutex>
#include <memory>
#include <unistd.h>
#include <limits.h>
#include <condition_variable>

#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "autil/LockFreeThreadPool.h"

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
                                   py::object                                             mm_process_engine,
                                   std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) {
    auto ret = RemoteRpcServer::init(maga_init_params, mm_process_engine, std::move(propose_params));
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

void DecodeRpcServer::prepareGenerateContext(DecodeGenerateContext& decode_context) {
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
    RTP_LLM_LOG_DEBUG("request [%s] start to allocate resource", decode_context.request_key.c_str());
    auto input                        = QueryConverter::transQuery(&decode_context.allocate_request.input());
    auto generate_stream              = engine_->makeStream(input);
    decode_context.request_timeout_ms = generate_stream->getTimeoutMs();
    auto status                       = generate_stream->initKVBlock(0);
    decode_context.setStream(generate_stream);
    if (!status.ok()) {
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

    auto new_tokens = engine_->getDevice()->allocateBuffer({rtp_llm::DataType::TYPE_INT32,
                                                            {(size_t)generate_stream->nextBatchSize(), (size_t)1},
                                                            rtp_llm::AllocationType::HOST},
                                                           {});

    auto data           = new_tokens->data<int32_t>();
    auto first_token_id = generate_request.first_generate_token_id();
    *data               = first_token_id;
    generate_stream->incLastOutputPos();
    generate_stream->update({new_tokens, 1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr});
    if (generate_request.position_ids_size() > 0) {
        auto context_position_ids =
            engine_->getDevice()->allocateBuffer({rtp_llm::DataType::TYPE_INT32,
                                                  {(size_t)generate_request.position_ids_size()},
                                                  rtp_llm::AllocationType::HOST},
                                                 {});
        memcpy(context_position_ids->data<int32_t>(),
               generate_request.position_ids().data(),
               generate_request.position_ids_size() * sizeof(int32_t));
        generate_stream->setContextPositionIds(context_position_ids);
    }
    if (propose_maga_init_params_) {
        generate_stream->setReuseLength(generate_stream->seqLength() - 1);
        generate_stream->setFallbackPrefixLength(generate_stream->reuseLength());
        generate_stream->setSpEditRun(false);
        generate_stream->setMtpTokenIndex(generate_stream->seqLength() - 1);
        generate_stream->setContainProposeToken(true);
        std::vector<int> propose_tokens;
        propose_tokens.assign(generate_request.propose_token_ids().begin(), generate_request.propose_token_ids().end());
        generate_stream->setProposeToken(propose_tokens);
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
    request.set_dp_rank(maga_init_params_.gpt_init_parameter.dp_rank_);
    request.set_partition_count(1);
    request.set_partition_id(0);

    // D >= P
    if (resource_.workers.size() % peer_addrs.size() == 0) {
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
    for (auto& block_id : load_context.block_ids) {
        request.add_block_ids(block_id);
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
    request.set_dp_rank(maga_init_params_.gpt_init_parameter.dp_rank_);

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

    for (auto& cache_key : load_context.cache_keys) {
        request.add_cache_keys(cache_key);
    }
    for (auto& block_id : load_context.block_ids) {
        request.add_block_ids(block_id);
    }
    request.set_timeout_ms(load_context.timeout_ms);
    return request;
}

ErrorInfo DecodeRpcServer::loadCacheForAllRank(DecodeGenerateContext& decode_context) {
    auto* generate_stream = decode_context.getStream().get();
    auto& cache_keys      = generate_stream->cacheKeys(0);
    auto& block_ids       = generate_stream->kvCache().blocks(0);

    if (cache_keys.size() != block_ids.size()) {
        return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED,
                         "cache keys size " + std::to_string(cache_keys.size()) + " not equal to block size "
                             + std::to_string(block_ids.size()));
    }

    if (resource_.workers.size() % decode_context.peer_addrs.size() != 0
        && decode_context.peer_addrs.size() % resource_.workers.size() != 0) {
        RTP_LLM_LOG_WARNING("request:[%s] peer ips size %d not equal to worker size %d",
                            decode_context.request_key.c_str(),
                            decode_context.peer_addrs.size(),
                            resource_.workers.size());
        return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, "peer ips size not equal to worker size");
    }

    auto load_cache_timeout_ms = maga_init_params_.gpt_init_parameter.load_cache_timeout_ms_;
    load_cache_timeout_ms      = load_cache_timeout_ms > 0 ? load_cache_timeout_ms : LOAD_TIMEOUT_MS;
    auto max_rpc_timeout_ms    = maga_init_params_.gpt_init_parameter.max_rpc_timeout_ms_;
    auto rpc_timeout           = max_rpc_timeout_ms > 0 ? max_rpc_timeout_ms : MAX_GRPC_TIMEOUT_MS;
    auto min_timeout_ms        = std::min(load_cache_timeout_ms, rpc_timeout);
    auto request_timeout_ms    = decode_context.request_timeout_ms;
    min_timeout_ms             = request_timeout_ms > 0 ? std::min(request_timeout_ms, min_timeout_ms) : min_timeout_ms;

    LoadKVCacheContext load_context{decode_context.request_id,
                                    decode_context.request_key,
                                    decode_context.peer_addrs,
                                    cache_keys,
                                    block_ids,
                                    generate_stream->reuseBlockSize(),
                                    min_timeout_ms,
                                    1,
                                    0,
                                    decode_context.server_context};

    // Prefill: TP = 1 && Decode: TP = 1
    if (resource_.workers.size() == 1 && decode_context.peer_addrs.size() == 1) {
        for (size_t i = 0; i < maga_init_params_.gpt_init_parameter.rdma_connect_retry_times_ + 1; i++) {
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
            + std::chrono::milliseconds(maga_init_params_.gpt_init_parameter.decode_polling_kv_cache_step_ms_);
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
    AtomicGuard request_guard(onflight_load_cache_requests_);
    const auto& request_key      = load_context.request_key;
    auto        cache_manager    = engine_->resourceContext().cache_manager;
    const auto& cache_config     = cache_manager->cacheConfig();
    auto        k_block_size     = cache_config.k_block_stride;
    auto        scale_block_size = cache_config.kv_scale_block_stride;
    auto        layer_num        = maga_init_params_.gpt_init_parameter.num_layers_;

    if (k_block_size % load_context.peer_addrs.size() != 0 || scale_block_size % load_context.peer_addrs.size() != 0) {
        RTP_LLM_LOG_WARNING("k block size [%d] or scale block size [%d] is not divisible by peer ips size [%d]",
                            k_block_size,
                            scale_block_size,
                            load_context.peer_addrs.size());
        return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, "block size is not divisible by peer ips size");
    }
    k_block_size     = k_block_size / load_context.peer_addrs.size();
    scale_block_size = scale_block_size / load_context.peer_addrs.size();

    auto cancel_check_func  = [&load_context]() -> bool { return load_context.server_context->IsCancelled(); };
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
            auto   block_num = load_context.block_ids.size();
            size_t model_id  = maga_init_params_.model_id;

            for (size_t block_pos = load_context.reuse_block_size; block_pos < block_num; block_pos++) {
                auto cache_key = makeCacheKey(model_id, std::to_string(load_context.cache_keys[block_pos]), layer_id);
                // FT_LOG_DEBUG("large model load cache_key %s", cache_key.c_str());
                auto                  block_id  = load_context.block_ids[block_pos];
                auto                  addr_info = cache_manager->convertIndexToAddr(block_id, layer_id);
                void*                 k_addr    = (void*)((int64_t)addr_info.k_addr + i * k_block_size);
                std::shared_ptr<void> k_block_addr(k_addr, [](void* p) {});
                load_layer_cache->addBlock("k_" + cache_key, k_block_addr, k_block_size, true, true);
            }
            layer_caches.push_back(load_layer_cache);
        }

        if (engine_->isMTPEagle()) {
            for (size_t mtp_model_id = 0; mtp_model_id < propose_maga_init_params_->mtp_model_params_->size();
                 mtp_model_id++) {
                EngineInitParams* mtp_engine_init_params =
                    propose_maga_init_params_->mtp_model_params_->at(mtp_model_id).get();
                const auto& sp_cache_manager = engine_->resourceContext().mtp_cache_managers[mtp_model_id];
                const auto& cache_config     = sp_cache_manager->cacheConfig();
                const auto  sp_k_block_size  = cache_config.k_block_stride / load_context.peer_addrs.size();
                size_t      layer_num        = mtp_engine_init_params->gpt_init_parameter.num_layers_;
                for (size_t layer_id = 0; layer_id < layer_num; layer_id++) {
                    auto request_key = std::to_string(load_context.request_id) + "-" + std::to_string(layer_id);
                    auto load_layer_cache =
                        std::make_shared<RequestBlockBuffer>(std::to_string(load_context.request_id), request_key);
                    auto   block_num = load_context.block_ids.size();
                    size_t model_id  = mtp_engine_init_params->model_id;

                    for (size_t block_pos = load_context.reuse_block_size; block_pos < block_num; block_pos++) {
                        auto cache_key =
                            makeCacheKey(model_id, std::to_string(load_context.cache_keys[block_pos]), layer_id);
                        // FT_LOG_DEBUG("small model load cache_key %s", cache_key.c_str());
                        auto                  block_id  = load_context.block_ids[block_pos];
                        auto                  addr_info = sp_cache_manager->convertIndexToAddr(block_id, layer_id);
                        void*                 k_addr    = (void*)((int64_t)addr_info.k_addr + i * sp_k_block_size);
                        std::shared_ptr<void> k_block_addr(k_addr, [](void* p) {});
                        load_layer_cache->addBlock("k_" + cache_key, k_block_addr, sp_k_block_size, true, true);
                    }
                    layer_caches.push_back(load_layer_cache);
                }
            }
        }

        auto ip_parts = autil::StringUtil::split(peer_addr, ":");
        if (ip_parts.size() != 3) {
            RTP_LLM_LOG_WARNING("invalid peer ip to load [%s]", peer_addr.c_str());
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
    if (request->dp_rank() != maga_init_params_.gpt_init_parameter.dp_rank_) {
        RTP_LLM_LOG_WARNING("only load when in dp group, skip load for dp rank %d", request->dp_rank());
        return grpc::Status::OK;
    }

    std::vector<int64_t>     cache_keys(request->cache_keys().begin(), request->cache_keys().end());
    std::vector<int32_t>     block_ids(request->block_ids().begin(), request->block_ids().end());
    std::vector<std::string> peer_addrs(request->peer_addrs().begin(), request->peer_addrs().end());

    // TODO(xinfei.sxf) add retry
    auto error_info = loadCache({request->request_id(),
                                 request->request_key(),
                                 peer_addrs,
                                 cache_keys,
                                 block_ids,
                                 request->reuse_block_size(),
                                 request->timeout_ms(),
                                 request->partition_count(),
                                 request->partition_id(),
                                 server_context});
    response->mutable_error_info()->set_error_code(transErrorCodeToRPC(error_info.code()));
    response->mutable_error_info()->set_error_message(error_info.ToString());
    response->set_done_time_us(currentTimeUs());
    RTP_LLM_LOG_DEBUG("request: %s, remote load cache grpc done", request->request_key().c_str());
    return grpc::Status::OK;
}

grpc::Status DecodeRpcServer::allocateResourceFunc(DecodeGenerateContext& decode_context) {
    EXECUTE_STAGE_FUNC(allocateResource, decode_context);
    return grpc::Status::OK;
}

grpc::Status DecodeRpcServer::RemoteGenerate(grpc::ServerContext* server_context, ServerStream* grpc_stream) {
    AtomicGuard      request_guard(onflight_requests_);
    DecodeRpcContext rpc_context{grpc_stream};
    // TODO(xinfei.sxf) request id is 0 here
    auto decode_context              = DecodeGenerateContext(rpc_context, 0, server_context, metrics_reporter_, meta_);
    decode_context.onflight_requests = onflight_requests_;
    decode_context.loading_cache_requests = loading_cache_requests_;

    auto max_retry_times      = maga_init_params_.gpt_init_parameter.decode_retry_times_;
    auto max_retry_timeout_ms = maga_init_params_.gpt_init_parameter.decode_retry_timeout_ms_;

    try {
        EXECUTE_STAGE_FUNC(prepareGenerateContext, decode_context);
        EXECUTE_WITH_RETRY(allocateResourceFunc, decode_context, max_retry_times, max_retry_timeout_ms);
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

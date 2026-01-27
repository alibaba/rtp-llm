#include "rtp_llm/cpp/model_rpc/PrefillRpcServerNew.h"
#include "autil/StringUtil.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"

namespace rtp_llm {

grpc::Status PrefillRpcServerNew::init(const EngineInitParams&                                maga_init_params,
                                       py::object                                             mm_process_engine,
                                       std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) {
    RTP_LLM_LOG_INFO("prefill rpc server new init");
    return RemoteRpcServer::init(maga_init_params, mm_process_engine, std::move(propose_params));
}

grpc::Status PrefillRpcServerNew::RemoteGenerateNew(grpc::ServerContext*              context,
                                                    const RemoteGenerateRequestPBNew* request,
                                                    RemoteGenerateResponsePBNew*      response) {
    auto             modified_request = const_cast<RemoteGenerateRequestPBNew*>(request);
    GenerateInputPB* mutable_input    = modified_request->mutable_input();

    // reset request_id in prefill
    auto request_id = loading_cache_requests_.fetch_add(1, std::memory_order_relaxed);
    mutable_input->set_request_id(request_id);

    // ignore inter_request_id in prefill
    auto modified_config = mutable_input->mutable_generate_config();
    modified_config->set_inter_request_id(-1);

    PrefillGenerateContextNew prefill_context(&resource_, context, request, response, metrics_reporter_, meta_);
    RTP_LLM_LOG_INFO("request [%s] RemoteGenerateNew", prefill_context.request_key.c_str());

    prefill_context.error_info = prefill_context.init(engine_);
    if (!prefill_context.error_info.ok()) {
        RTP_LLM_LOG_WARNING("request [%s] prepare generate context failed, err: %s",
                            prefill_context.request_key.c_str(),
                            prefill_context.error_info.ToString().c_str());
        return serializeErrorMsg(prefill_context.request_key, prefill_context.error_info);
    }

    if (!validRequest(prefill_context)) {
        RTP_LLM_LOG_WARNING("request [%s] is invalid", prefill_context.request_key.c_str());
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "invalid request");
    }

    prefill_context.error_info = notifyStoreCacheForAllRank(prefill_context);
    if (!prefill_context.error_info.ok()) {
        RTP_LLM_LOG_WARNING("request [%s] prepare generate context failed, err: %s",
                            prefill_context.request_key.c_str(),
                            prefill_context.error_info.ToString().c_str());
        return serializeErrorMsg(prefill_context.request_key, prefill_context.error_info);
    }
    prefill_context.notify_store_cache_done_time_us = currentTimeUs();

    prefill_context.error_info = generateFirstToken(prefill_context);
    if (!prefill_context.error_info.ok()) {
        RTP_LLM_LOG_WARNING("request [%s] prepare generate context failed, err: %s",
                            prefill_context.request_key.c_str(),
                            prefill_context.error_info.ToString().c_str());
        return serializeErrorMsg(prefill_context.request_key, prefill_context.error_info);
    }
    prefill_context.generate_first_token_done_time_us = currentTimeUs();

    prefill_context.error_info = waitStoreCacheForAllRankDone(prefill_context);
    if (!prefill_context.error_info.ok()) {
        RTP_LLM_LOG_WARNING("request [%s] prepare generate context failed, err: %s",
                            prefill_context.request_key.c_str(),
                            prefill_context.error_info.ToString().c_str());
        return serializeErrorMsg(prefill_context.request_key, prefill_context.error_info);
    }
    prefill_context.wait_store_cache_done_time_us = currentTimeUs();

    // TODO: notify remote store for hidden state
    // if (engine_->isMTPEagle() &&
    //        engine_->getDevice()->getDeviceProperties().tp_rank == 0 &&
    //        !request->mtp_hidden_states_key.empty()) {
    //}

    RTP_LLM_LOG_DEBUG("request [%s] RemoteGenerateNew success, response is %s",
                      prefill_context.request_key.c_str(),
                      response->ShortDebugString().c_str());
    return grpc::Status::OK;
}

bool PrefillRpcServerNew::validRequest(PrefillGenerateContextNew& prefill_context) {
    auto& request = prefill_context.request;
    if (request->deadline_us() <= currentTimeUs()) {
        RTP_LLM_LOG_WARNING(
            "request [%s] deadline exceed [%ld]", prefill_context.request_key.c_str(), request->deadline_us());
        return false;
    }

    auto decode_worker_size  = request->addrs_size();
    auto prefill_worker_size = resource_.workers.size();
    if (decode_worker_size % prefill_worker_size != 0 && prefill_worker_size % decode_worker_size != 0) {
        RTP_LLM_LOG_WARNING("request [%s] decode_worker_size [%d] not devisible to prefill_worker_size [%d]",
                            prefill_context.request_key.c_str(),
                            decode_worker_size,
                            prefill_worker_size);
        return false;
    }

    auto  generate_stream = prefill_context.getStream();
    auto& block_ids       = generate_stream->kvCachePtr()->blocks(0);
    if (block_ids.size() != request->block_ids_size()) {
        RTP_LLM_LOG_WARNING("request [%s] block_ids size [%d] not match request block_ids size [%d]",
                            prefill_context.request_key.c_str(),
                            block_ids.size(),
                            request->block_ids_size());
        return false;
    }

    if (engine_->isMTPEagle() && request->mtp_hidden_states_key().empty()) {
        RTP_LLM_LOG_WARNING("request [%s] mtp_hidden_states_key is empty", prefill_context.request_key.c_str());
        return false;
    }

    if (request->use_mla() != maga_init_params_.model_config_.attn_config.use_mla) {
        RTP_LLM_LOG_WARNING("request [%s] request is invalid, mla config not match",
                            prefill_context.request_key.c_str());
        return false;
    }

    if (request->layer_num() != maga_init_params_.model_config_.num_layers) {
        RTP_LLM_LOG_WARNING("request [%s] request is invalid, layer_num %d vs %d not match",
                            prefill_context.request_key.c_str(),
                            request->layer_num(),
                            maga_init_params_.model_config_.num_layers);
        return false;
    }
    return true;
}

ErrorInfo PrefillRpcServerNew::notifyStoreCacheForAllRank(PrefillGenerateContextNew& prefill_context) {
    for (int i = 0; i < resource_.workers.size(); ++i) {
        auto error_info = notifyStoreCache(prefill_context, i);
        if (!error_info.ok()) {
            RTP_LLM_LOG_ERROR("request [%s] notify store cache for rank [%d] failed, err: %s",
                              prefill_context.request_key.c_str(),
                              i,
                              error_info.ToString().c_str());
            return error_info;
        }
    }
    return ErrorInfo::OkStatus();
}

ErrorInfo PrefillRpcServerNew::notifyStoreCache(PrefillGenerateContextNew& prefill_context, int index) {
    auto& worker         = resource_.grpc_workers[index];
    auto  connect_status = resource_.rpc_pool.getConnection(worker);
    if (!connect_status.ok()) {
        std::string error_msg =
            "get grpc connection for rank:" + std::to_string(index) + ", addr:" + worker + " failed";
        return ErrorInfo(ErrorCode::GET_CONNECTION_FAILED, error_msg);
    }

    auto rpc_context = std::make_shared<PrefillRpcContext>();
    prefill_context.rpc_contexts.push_back(rpc_context);
    rpc_context->stub = connect_status.value().stub;

    if (index >= prefill_context.rpc_contexts.size()) {
        RTP_LLM_LOG_WARNING(
            "request [%s] rpc_contexts size [%d] not enough", prefill_context.request_key.c_str(), index);
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "rpc_contexts size not enough");
    }

    constructRemoteLoadRequest(prefill_context, index);
    RTP_LLM_LOG_DEBUG("remote load request is %s", rpc_context->request.ShortDebugString().c_str());

    std::unique_ptr<grpc::ClientAsyncResponseReader<RemoteStoreResponsePB>> reader(rpc_context->stub->AsyncRemoteStore(
        rpc_context->client_context.get(), rpc_context->request, rpc_context->completion_queue.get()));
    reader->Finish(&rpc_context->response, &rpc_context->status, reinterpret_cast<void*>(index));
    rpc_context->reader = std::move(reader);
    return ErrorInfo::OkStatus();
}

void PrefillRpcServerNew::constructRemoteLoadRequest(PrefillGenerateContextNew& prefill_context, int index) {
    auto& request = prefill_context.rpc_contexts[index]->request;
    request.set_dp_rank(maga_init_params_.parallelism_config.dp_rank);
    request.set_request_id(prefill_context.request_id);
    request.set_request_key(prefill_context.request_key);
    request.set_deadline_us(prefill_context.request->deadline_us());
    request.set_client_id(prefill_context.request->client_id());

    for (int i = prefill_context.request->reuse_block_size(); i < prefill_context.request->block_ids_size(); i++) {
        request.add_decode_block_ids(prefill_context.request->block_ids(i));
    }
    auto& block_ids = prefill_context.getStream()->kvCachePtr()->blocks(0);
    for (int i = prefill_context.request->reuse_block_size(); i < block_ids.size(); i++) {
        request.add_prefill_block_ids(block_ids[i]);
    }
    request.set_reuse_block_size(prefill_context.request->reuse_block_size());

    auto decode_worker_size  = prefill_context.decode_workers.size();
    auto prefill_worker_size = resource_.workers.size();
    if (engine_->resourceContext().cache_manager->cacheConfig().use_mla) {
        // mla 下, d 和 p不对称不需要拼接
        if (decode_worker_size > prefill_worker_size) {
            auto group_num = decode_worker_size / prefill_worker_size;
            for (int i = 0; i < group_num; i++) {
                auto partition_info = request.add_partition_infos();
                partition_info->set_remote_addr(prefill_context.decode_workers[group_num * index + i]);
                partition_info->set_remote_partition_id(0);
                partition_info->set_remote_partition_count(1);
                partition_info->set_local_partition_id(0);
                partition_info->set_local_partition_count(1);
            }
        } else {
            auto group_num = prefill_worker_size / decode_worker_size;
            if (index % group_num == 0) {
                auto partition_info = request.add_partition_infos();
                partition_info->set_remote_addr(prefill_context.decode_workers[index / group_num]);
                partition_info->set_remote_partition_id(0);
                partition_info->set_remote_partition_count(1);
                partition_info->set_local_partition_id(0);
                partition_info->set_local_partition_count(1);
            }
        }
    } else {
        if (decode_worker_size > prefill_worker_size) {
            // 每个d 只读取 p的一部分
            auto group_num = decode_worker_size / prefill_worker_size;
            for (int i = 0; i < group_num; i++) {
                auto partition_info = request.add_partition_infos();
                partition_info->set_remote_addr(prefill_context.decode_workers[group_num * index + i]);
                partition_info->set_remote_partition_count(1);
                partition_info->set_remote_partition_id(0);
                partition_info->set_local_partition_count(group_num);
                partition_info->set_local_partition_id(i);
            }
        } else {
            // d 对应多个 p
            auto group_num      = prefill_worker_size / decode_worker_size;
            auto partition_info = request.add_partition_infos();
            partition_info->set_remote_addr(prefill_context.decode_workers[index / group_num]);
            partition_info->set_remote_partition_count(group_num);
            partition_info->set_remote_partition_id(index % group_num);
            partition_info->set_local_partition_count(1);
            partition_info->set_local_partition_id(0);
        }
    }
}

ErrorInfo PrefillRpcServerNew::generateFirstToken(PrefillGenerateContextNew& prefill_context) {
    auto stream = prefill_context.getStream();
    engine_->enqueue(stream);
    while (!stream->finished() || stream->hasOutput()) {
        const auto result = stream->nextOutput();
        if (!result.ok()) {
            if (result.status().code() != ErrorCode::FINISHED) {
                return result.status();
            }
        }
        RTP_LLM_LOG_DEBUG("request [%s] generate next output success", prefill_context.request_key.c_str());
        auto response_output = prefill_context.response->mutable_output();

        QueryConverter::transResponse(response_output,
                                      &(result.value()),
                                      stream->generateConfig()->aux_info,
                                      maga_init_params_.misc_config.aux_string,
                                      stream->specialTokens().eos_token_id);
        // should only generate one token
        break;
    }
    if (prefill_context.getStream()->finished()) {
        RTP_LLM_LOG_INFO("request [%s] generate first token success and finished", prefill_context.request_key.c_str());
    }
    auto first_token = prefill_context.getStream()->currentExecuteTokens()[0];
    prefill_context.response->set_finished(prefill_context.getStream()->finished());
    prefill_context.response->set_first_generate_token_id(first_token);
    return ErrorInfo::OkStatus();
}

ErrorInfo PrefillRpcServerNew::waitStoreCacheForAllRankDone(PrefillGenerateContextNew& prefill_context) {
    int  finished_count = 0;
    bool all_success    = true;

    ErrorCode   error_code = ErrorCode::NONE_ERROR;
    std::string error_msg  = "failed to load kv cache in rank: ";

    while (finished_count < prefill_context.rpc_contexts.size()) {
        if (currentTimeUs() > prefill_context.request->deadline_us()) {
            RTP_LLM_LOG_WARNING("request [%s] deadline exceed [%ld]",
                                prefill_context.request_key.c_str(),
                                prefill_context.request->deadline_us());
            return ErrorInfo(ErrorCode::LOAD_CACHE_TIMEOUT, "remote store cache timeout");
        }

        if (prefill_context.server_context->IsCancelled()) {
            RTP_LLM_LOG_WARNING("request [%s] cancel by user", prefill_context.request_key.c_str());
            return ErrorInfo(ErrorCode::CANCELLED, "request cancelled");
        }

        auto once_deadline =
            std::chrono::system_clock::now()
            + std::chrono::milliseconds(maga_init_params_.pd_sep_config.decode_polling_kv_cache_step_ms);
        void* got_tag;
        bool  ok = false;

        for (int rank = 0; rank < prefill_context.rpc_contexts.size(); rank++) {
            auto& rpc_context = prefill_context.rpc_contexts[rank];
            if (rpc_context->finished) {
                continue;
            }

            if (rpc_context->completion_queue->AsyncNext(&got_tag, &ok, once_deadline)
                == grpc::CompletionQueue::NextStatus::TIMEOUT) {
                RTP_LLM_LOG_DEBUG("request [%s] async next timeout", prefill_context.request_key.c_str());
                continue;
            }
            if (!ok) {
                return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED,
                                 "async get next event from grpc completion queue failed");
            }
            const auto& status           = rpc_context->status;
            const auto& response         = rpc_context->response;
            const auto& pb_error_code    = response.error_info().error_code();
            const auto& pb_error_message = response.error_info().error_message();
            prefill_context.min_response_done_time_us =
                std::min(prefill_context.min_response_done_time_us, response.done_time_us());
            prefill_context.max_response_done_time_us =
                std::max(prefill_context.max_response_done_time_us, response.done_time_us());

            if (!status.ok()) {
                all_success = false;
                error_code  = ErrorCode::LOAD_KV_CACHE_FAILED;
                error_msg += std::to_string(rank) + ": " + status.error_message() + ", ";
                RTP_LLM_LOG_WARNING("request [%s] load kv cache status failed, err: %d %s",
                                    prefill_context.request_key.c_str(),
                                    status.error_code(),
                                    status.error_message().c_str());
            } else if (pb_error_code != ErrorCodePB::NONE_ERROR) {
                all_success = false;
                error_code  = transRPCErrorCode(pb_error_code);
                error_msg += std::to_string(rank) + ": " + pb_error_message + ", ";
                RTP_LLM_LOG_WARNING("request [%s] load kv cache code failed, err: %s",
                                    prefill_context.request_key.c_str(),
                                    pb_error_message.c_str());
            }

            rpc_context->finished = true;
            finished_count++;
        }
    }
    if (!all_success) {
        RTP_LLM_LOG_WARNING(
            "request [%s] load kv cache failed, err: %s", prefill_context.request_key.c_str(), error_msg.c_str());
        return ErrorInfo(error_code, error_msg);
    }

    // release kvcache in scheduler not in rpc, otherwise may cause kvcache not saved
    prefill_context.getStream()->setNeedReleaseKVCache(true);
    return ErrorInfo::OkStatus();
}

grpc::Status PrefillRpcServerNew::RemoteStore(grpc::ServerContext*        server_context,
                                              const RemoteStoreRequestPB* request,
                                              RemoteStoreResponsePB*      response) {
    RTP_LLM_LOG_DEBUG("request [%s] remote store", request->request_key().c_str());
    if (request->dp_rank() != maga_init_params_.parallelism_config.dp_rank) {
        RTP_LLM_LOG_WARNING("only load when in dp group, skip load for dp rank %d", request->dp_rank());
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "error dp rank");
    }
    auto start_time_us = currentTimeUs();
    if (start_time_us >= request->deadline_us()) {
        RTP_LLM_LOG_WARNING("deadline exceed [%ld]", request->deadline_us());
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "deadline exceed");
    }

    auto        cache_manager = engine_->resourceContext().cache_manager;
    const auto& cache_config  = cache_manager->cacheConfig();
    auto        k_block_size  = cache_config.kv_block_stride_bytes;
    auto        v_block_size  = cache_config.kv_block_stride_bytes;
    auto        layer_num     = maga_init_params_.model_config_.num_layers;

    auto remote_addr_size = request->partition_infos_size();
    if (remote_addr_size == 0) {
        RTP_LLM_LOG_WARNING("remote addr size is 0");
        return grpc::Status::OK;
    }

    if (v_block_size % remote_addr_size != 0 || k_block_size % remote_addr_size != 0) {
        RTP_LLM_LOG_WARNING(
            "k block size [%d] or v block size [%d] or scale block size [%d] is not divisible by peer ips size [%d]",
            k_block_size,
            v_block_size,
            remote_addr_size);
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "block size is not divisible by peer ips size");
    }

    auto cancel_check_func = [server_context, deadline_us = request->deadline_us()]() -> bool {
        RTP_LLM_LOG_DEBUG("cancel check currentTimeUs() [%ld] deadline_us [%ld]", currentTimeUs(), deadline_us);
        return server_context->IsCancelled() || currentTimeUs() >= deadline_us;
    };

    auto request_id = request->request_id();

    std::vector<std::shared_ptr<RemoteStoreTask>> tasks;
    for (int i = 0; i < remote_addr_size; i++) {
        auto partition_info = request->partition_infos(i);
        auto ip_parts       = autil::StringUtil::split(partition_info.remote_addr(), ":");
        if (ip_parts.size() != 3) {
            RTP_LLM_LOG_WARNING("invalid peer ip to load [%s]", partition_info.remote_addr().c_str());
            // TODO: detail error
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "invalid peer ip to load");
        }

        auto store_request =
            std::make_shared<RemoteStoreRequest>(request->client_id(),
                                                 std::to_string(request->request_id()),
                                                 ip_parts[0],
                                                 autil::StringUtil::strToInt32WithDefault(ip_parts[1].c_str(), 0),
                                                 autil::StringUtil::strToInt32WithDefault(ip_parts[2].c_str(), 0),
                                                 request->deadline_us(),
                                                 partition_info.local_partition_count(),
                                                 partition_info.local_partition_id(),
                                                 partition_info.remote_partition_count(),
                                                 partition_info.remote_partition_id());
        size_t model_id = maga_init_params_.model_id;
        for (size_t layer_id = 0; layer_id < layer_num; layer_id++) {
            auto request_key = std::to_string(request_id) + "-" + std::to_string(layer_id);
            auto block_num   = request->decode_block_ids_size();
            for (int i = 0; i < block_num; i++) {
                auto decode_block_key = makeCacheKey(model_id, std::to_string(request->decode_block_ids(i)), layer_id);
                auto prefill_block_key =
                    makeCacheKey(model_id, std::to_string(request->prefill_block_ids(i)), layer_id);
                store_request->buffer_pairs["k_" + prefill_block_key] = "k_" + decode_block_key;
                if (engine_->resourceContext().cache_manager->cacheConfig().use_mla) {
                    continue;
                }
                store_request->buffer_pairs["v_" + prefill_block_key] = "v_" + decode_block_key;
            }
        }

        auto collector = std::make_shared<CacheStoreRemoteStoreMetricsCollector>(metrics_reporter_,
                                                                                 store_request->buffer_pairs.size());

        auto task = resource_.cache_store->submitRemoteStoreTask(store_request, collector, cancel_check_func);
        if (!task) {
            RTP_LLM_LOG_WARNING("submit remote store task failed");
            // TODO: detail error
            response->mutable_error_info()->set_error_code(ErrorCodePB::UNKNOWN_ERROR);
            return grpc::Status::OK;
        }
        tasks.push_back(task);
    }

    for (auto& task : tasks) {
        task->waitDone();
    }

    std::string error_msg;
    for (auto& task : tasks) {
        if (!task->success()) {
            error_msg += task->getErrorInfo().ToString();
        }
        resource_.cache_store->releaseRemoteStoreTask(task);
    }
    if (error_msg.empty()) {
        RTP_LLM_LOG_DEBUG("request [%s] remote store success", request->request_key().c_str());
        return grpc::Status::OK;
    }

    RTP_LLM_LOG_WARNING("request [%s] remote store failed, err: %s", request->request_key().c_str(), error_msg.c_str());
    // TODO: detail error
    response->mutable_error_info()->set_error_code(ErrorCodePB::UNKNOWN_ERROR);
    response->mutable_error_info()->set_error_message(error_msg);
    return grpc::Status::OK;
}

grpc::Status PrefillRpcServerNew::RemoteFinish(grpc::ServerContext*         context,
                                               const RemoteFinishRequestPB* request,
                                               EmptyPB*                     response) {
    auto request_id = request->request_id();
    resource_.cache_store->markRequestEnd(std::to_string(request_id));
    return grpc::Status::OK;
}

}  // namespace rtp_llm

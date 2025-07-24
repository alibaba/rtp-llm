#include "rtp_llm/cpp/model_rpc/DecodeRpcServerNew.h"
#include "rtp_llm/cpp/disaggregate/load_balancer/RRLoadBalancer.h"
#include "rtp_llm/cpp/disaggregate/load_balancer/WRRLoadBalancer.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"

namespace rtp_llm {

grpc::Status DecodeRpcServerNew::init(const EngineInitParams&                                maga_init_params,
                                      py::object                                             mm_process_engine,
                                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) {
    auto ret = RemoteRpcServer::init(maga_init_params, mm_process_engine, std::move(propose_params));
    if (!ret.ok()) {
        RTP_LLM_LOG_ERROR("decode rpc server new init failed, err: %s", ret.error_message().c_str());
        return ret;
    }

    if (!initLoadBalancer()) {
        RTP_LLM_LOG_ERROR("decode rpc server new init load balancer failed");
        return grpc::Status(grpc::StatusCode::INTERNAL, "init load balancer failed");
    }

    RTP_LLM_LOG_INFO("decode rpc server new init");
    return grpc::Status::OK;
}

bool DecodeRpcServerNew::ready() {
    char* decode_cm2_config_env = std::getenv("RTP_LLM_DECODE_CM2_CONFIG");
    if(!maga_init_params_.gpt_init_parameter.service_discovery_config.use_local && decode_cm2_config_env == nullptr) {
        RTP_LLM_LOG_INFO("service discovery by master, skip load balancer check");
        return true;
    }

    if (!load_balancer_) {
        RTP_LLM_LOG_INFO("load balance is nullptr, server is not ready");
        return false;
    }

    if (!load_balancer_->isReady(prefill_cluster_name_)) {
        RTP_LLM_LOG_INFO("load balancer is not ready now, cluster not exist", prefill_cluster_name_.c_str());
        return false;
    }
    return true;
}

bool DecodeRpcServerNew::initLoadBalancer() {
    LoadBalancerInitParams init_params;
    init_params.update_interval_ms      = 100;
    init_params.sync_status_interval_ms = maga_init_params_.gpt_init_parameter.sync_status_interval_ms_;

    if (maga_init_params_.gpt_init_parameter.service_discovery_config.use_local) {
        char* remote_rpc_server_ip_env = std::getenv("REMOTE_RPC_SERVER_IP");
        RTP_LLM_CHECK_WITH_INFO(remote_rpc_server_ip_env, "rpc server ip must be not empty");

        prefill_cluster_name_ = "LOCAL";
        if (!BaseLoadBalancer::makeLocalSubscribeConfig(init_params.subscribe_config,
                                                        prefill_cluster_name_,
                                                        remote_rpc_server_ip_env,
                                                        maga_init_params_.gpt_init_parameter.remote_rpc_server_port_)) {
            RTP_LLM_LOG_ERROR("make local subscribe config from config %s failed", remote_rpc_server_ip_env);
            return false;
        }
        RTP_LLM_LOG_INFO("init load balancer with local mode, config is %s", remote_rpc_server_ip_env);
    } else {
        char* decode_cm2_config_env = std::getenv("RTP_LLM_DECODE_CM2_CONFIG");
        if (decode_cm2_config_env == nullptr) {
            RTP_LLM_LOG_INFO("RTP_LLM_DECODE_CM2_CONFIG is not set, use default subscribe config");
            load_balancer_.reset(nullptr);
            return true;
        }
        if (!BaseLoadBalancer::makeCm2SubscribeConfig(
                init_params.subscribe_config, prefill_cluster_name_, decode_cm2_config_env)) {
            RTP_LLM_LOG_ERROR("make cm2 subscribe config from config %s failed", decode_cm2_config_env);
            return false;
        }
        RTP_LLM_LOG_INFO("init load balancer with cm2 mode, config is %s", decode_cm2_config_env);
    }

    if (maga_init_params_.gpt_init_parameter.load_balance_policy_name_ == "RR") {
        load_balancer_.reset(new RRLoadBalancer);
    } else {
        load_balancer_.reset(new WRRLoadBalancer(maga_init_params_.gpt_init_parameter.cache_store_config));
    }

    RTP_LLM_CHECK_WITH_INFO(load_balancer_->init(init_params), "load_balancer init failed");
    RTP_LLM_LOG_INFO("load balancer init success, policy is %s",
                     maga_init_params_.gpt_init_parameter.load_balance_policy_name_.c_str());
    return true;
}

grpc::Status DecodeRpcServerNew::GenerateStreamCall(grpc::ServerContext*                   server_context,
                                                    const GenerateInputPB*                 request,
                                                    grpc::ServerWriter<GenerateOutputsPB>* response_writer) {
    DecodeGenerateContextNew decode_context(server_context, request, response_writer, metrics_reporter_, meta_);

    RTP_LLM_LOG_DEBUG("request [%s] start generate", decode_context.request_key.c_str());

    decode_context.error_info = decode_context.init(engine_);
    if (!decode_context.error_info.ok()) {
        RTP_LLM_LOG_WARNING("request [%s] prepare generate context failed, err: %s",
                            decode_context.request_key.c_str(),
                            decode_context.error_info.ToString().c_str());
        return serializeErrorMsg(decode_context.request_key, decode_context.error_info);
    }

    decode_context.error_info = loadCacheFromPrefill(decode_context);
    if (!decode_context.error_info.ok()) {
        RTP_LLM_LOG_WARNING("request [%s] load cache from prefill failed, err: %s",
                            decode_context.request_key.c_str(),
                            decode_context.error_info.ToString().c_str());
        return serializeErrorMsg(decode_context.request_key, decode_context.error_info);
    }

    auto ret                                   = localGenerate(decode_context);
    decode_context.local_generate_done_time_us = currentTimeUs();
    RTP_LLM_LOG_DEBUG("request [%s] generate done", decode_context.request_key.c_str());
    return ret;
}

ErrorInfo DecodeRpcServerNew::loadCacheFromPrefill(DecodeGenerateContextNew& decode_context) {
    RTP_LLM_LOG_DEBUG("request [%s] start to load cache from prefill", decode_context.request_key.c_str());

    makeRemoteGenerateRequest(decode_context);
    RTP_LLM_LOG_DEBUG("request [%s] make remote generate request done, request is %s",
                      decode_context.request_key.c_str(),
                      decode_context.remote_generate_request.ShortDebugString().c_str());

    auto ret = callPrefill(decode_context);
    if (!ret.ok()) {
        RTP_LLM_LOG_WARNING(
            "request [%s] call prefill failed, err: %s", decode_context.request_key.c_str(), ret.ToString().c_str());
        return ret;
    }
    RTP_LLM_LOG_DEBUG("request [%s] call prefill done, response is %s",
                      decode_context.request_key.c_str(),
                      decode_context.remote_generate_response.ShortDebugString().c_str());

    decode_context.load_cache_from_prefill_done_time_us = currentTimeUs();
    RTP_LLM_LOG_DEBUG("request [%s] load cache from prefill done", decode_context.request_key.c_str());
    return ErrorInfo::OkStatus();
}

void DecodeRpcServerNew::makeRemoteGenerateRequest(DecodeGenerateContextNew& decode_context) {
    auto& request = decode_context.remote_generate_request;

    GenerateInputPB* new_request = new GenerateInputPB(*decode_context.request);
    request.set_allocated_input(const_cast<GenerateInputPB*>(new_request));
    request.set_client_id(process_id_);
    request.set_start_time_us(currentTimeUs());

    for (auto& addr : resource_.workers) {
        request.add_addrs(addr);
    }

    auto  generate_stream = decode_context.getStream();
    auto& block_ids       = generate_stream->kvCache().blocks(0);
    for (auto& block_id : block_ids) {
        request.add_block_ids(block_id);
    }

    // reuse block no need sent back from prefill
    request.set_reuse_block_size(generate_stream->reuseBlockSize());

    request.set_use_mla(engine_->resourceContext().cache_manager->cacheConfig().use_mla);
    request.set_layer_num(maga_init_params_.gpt_init_parameter.num_layers_);
    request.set_deadline_us(currentTimeUs() + decode_context.request_timeout_ms * 1000);
}

ErrorInfo DecodeRpcServerNew::callPrefill(DecodeGenerateContextNew& decode_context) {
    RTP_LLM_LOG_DEBUG("request [%s] start to call prefill", decode_context.request_key.c_str());

    auto role_addrs = QueryConverter::getRoleAddrs(&decode_context.request->generate_config());
    std::shared_ptr<const Host> host;
    for (auto& role_addr: role_addrs) {
        if (role_addr.role == RoleType::PREFILL) {
            host = std::make_shared<const Host>(role_addr.ip, role_addr.grpc_port, role_addr.http_port);
            break;
        }
    }
    if (!host && load_balancer_) {
        host = load_balancer_->chooseHost(prefill_cluster_name_,
                                            decode_context.request->generate_config().global_request_id());
    }

    if (!host || host->ip.empty()) {
        return ErrorInfo(ErrorCode::GET_HOST_FAILED,
                         "get host for decode cluster " + prefill_cluster_name_ + " failed");
    }

    auto prefill_addr   = host->ip + ":" + std::to_string(host->rpc_port);
    auto connect_status = resource_.rpc_pool.getConnection(prefill_addr);
    if (!connect_status.ok()) {
        return ErrorInfo(ErrorCode::GET_CONNECTION_FAILED,
                         "get grpc connection for decode addr " + prefill_addr + " failed");
    }

    auto rpc_context     = std::make_shared<DecodeRpcContextNew>();
    auto grpc_connection = connect_status.value();
    auto stub            = grpc_connection.stub;
    rpc_context->request = decode_context.remote_generate_request;

    std::unique_ptr<grpc::ClientAsyncResponseReader<RemoteGenerateResponsePBNew>> reader(stub->AsyncRemoteGenerateNew(
        rpc_context->client_context.get(), rpc_context->request, rpc_context->completion_queue.get()));

    reader->Finish(&decode_context.remote_generate_response, &rpc_context->status, reinterpret_cast<void*>(0));
    rpc_context->reader = std::move(reader);

    void* got_tag;
    bool  ok = false;

    auto deadline_us = rpc_context->request.deadline_us();

    while (!rpc_context->finished) {
        if (rpc_context->completion_queue->AsyncNext(
                &got_tag,
                &ok,
                std::chrono::system_clock::now()
                    + std::chrono::milliseconds(maga_init_params_.gpt_init_parameter.decode_polling_call_prefill_ms_))
            == grpc::CompletionQueue::NextStatus::TIMEOUT) {
            if (decode_context.server_context->IsCancelled()) {
                RTP_LLM_LOG_WARNING("request [%s] is cancelled", decode_context.request_key.c_str());
                rpc_context->client_context->TryCancel();
                return ErrorInfo(ErrorCode::CANCELLED, "request is cancelled");
            } else if (currentTimeUs() > deadline_us) {
                RTP_LLM_LOG_WARNING(
                    "request [%s] deadline exceed [%ld]", decode_context.request_key.c_str(), deadline_us);
                rpc_context->client_context->TryCancel();
                return ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "request deadline exceeded");
            }
            continue;
        }

        if (!ok) {
            RTP_LLM_LOG_WARNING("request [%s] async get next event from grpc completion queue failed",
                                decode_context.request_key.c_str());
            return ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, "async get next event from grpc completion queue failed");
        }

        if (!rpc_context->status.ok()) {
            const auto& error_msg      = rpc_context->status.error_message();
            ErrorCode   new_error_code = ErrorCode::LOAD_KV_CACHE_FAILED;
            if (error_msg.find("Connect Failed") != std::string::npos) {
                new_error_code = ErrorCode::CONNECT_FAILED;
                resource_.rpc_pool.removeConnection(prefill_addr);
            } else if (error_msg.find("No route to host") != std::string::npos) {
                new_error_code = ErrorCode::CONNECT_FAILED;
                resource_.rpc_pool.removeConnection(prefill_addr);
            } else if (error_msg.find("Connection reset by peer") != std::string::npos) {
                new_error_code = ErrorCode::CONNECTION_RESET_BY_PEER;
                resource_.rpc_pool.removeConnection(prefill_addr);
            } else if (error_msg.find("Connection timed out") != std::string::npos) {
                new_error_code = ErrorCode::CONNECT_TIMEOUT;
                resource_.rpc_pool.removeConnection(prefill_addr);
            } else if (error_msg.find("Deadline Exceeded") != std::string::npos) {
                new_error_code = ErrorCode::DEADLINE_EXCEEDED;
                resource_.rpc_pool.removeConnection(prefill_addr);
            }
            return ErrorInfo(new_error_code, error_msg);
        } else {
            rpc_context->finished = true;
        }
    }

    auto block_ids = decode_context.getStream()->kvCache().blocks(0);
    for (auto block_id : block_ids) {
        auto [k_buffer, v_buffer] = engine_->resourceContext().cache_manager->getKVBlockValue(block_id);
    }

    decode_context.load_cache_from_prefill_done_time_us = currentTimeUs();
    RTP_LLM_LOG_DEBUG("request [%s] call prefill done", decode_context.request_key.c_str());
    return ErrorInfo::OkStatus();
}

grpc::Status DecodeRpcServerNew::localGenerate(DecodeGenerateContextNew& decode_context) {
    auto generate_stream = decode_context.getStream();
    auto error_info      = writeAppendFirstToken(decode_context);
    if (!error_info.ok()) {
        return serializeErrorMsg(decode_context.request_key, error_info);
    }

    if (decode_context.remote_generate_response.finished()) {
        return grpc::Status::OK;
    }

    engine_->enqueue(decode_context.getStream());
    decode_context.error_status = pollStreamOutput(decode_context.server_context,
                                                   decode_context.request_key,
                                                   decode_context.response_writer,
                                                   decode_context.getStream());

    RTP_LLM_LOG_DEBUG("request [%s] local generate done", decode_context.request_key.c_str());
    return decode_context.error_status;
}

ErrorInfo DecodeRpcServerNew::writeAppendFirstToken(DecodeGenerateContextNew& decode_context) {
    if (decode_context.server_context->IsCancelled()) {
        RTP_LLM_LOG_WARNING("request [%s] is cancelled", decode_context.request_key.c_str());
        return ErrorInfo(ErrorCode::CANCELLED, "request is cancelled");
    }

    auto&   response          = decode_context.remote_generate_response;
    auto    initial_reuse_len = decode_context.getStream()->initialReuseLength();
    auto    first_token_rt_us = response.first_token_rt_us();
    int64_t cost_time_us      = currentTimeUs() - decode_context.request_begin_time_us;

    auto response_output = response.mutable_output();
    for (size_t i = 0; i < response_output->generate_outputs_size(); i++) {
        response_output->mutable_generate_outputs(i)->mutable_aux_info()->set_first_token_cost_time_us(
            first_token_rt_us);
        response_output->mutable_generate_outputs(i)->mutable_aux_info()->set_cost_time_us(cost_time_us);
        response_output->mutable_generate_outputs(i)->mutable_aux_info()->set_reuse_len(initial_reuse_len);
    }

    if (!decode_context.response_writer->Write(*response_output)) {
        RTP_LLM_LOG_WARNING("request [%ld] write outputs pb failed", decode_context.request_id);
        return ErrorInfo(ErrorCode::UNKNOWN_ERROR, "write outputs pb failed");
    }

    auto generate_stream = decode_context.getStream();
    generate_stream->setIsContextStream(false);
    generate_stream->step();

    // append first token to generate stream
    auto new_tokens = engine_->getDevice()->allocateBuffer(
        {rtp_llm::DataType::TYPE_INT32, {(size_t)generate_stream->tileNum(), (size_t)1}, rtp_llm::AllocationType::HOST},
        {});
    auto data           = new_tokens->data<int32_t>();
    auto first_token_id = response.first_generate_token_id();
    *data               = first_token_id;
    generate_stream->incLastOutputPos();
    generate_stream->update({new_tokens, 1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr});
    if (propose_maga_init_params_) {
        generate_stream->setReuseLength(generate_stream->seqLength() - 1);
        generate_stream->setFallbackPrefixLength(generate_stream->reuseLength());
        generate_stream->setSpEditRun(false);
    }
    generate_stream->resetBeginTime(currentTimeUs());

    return ErrorInfo::OkStatus();
}

}  // namespace rtp_llm
#include "maga_transformer/cpp/model_rpc/PrefillGenerateContext.h"

using grpc::Status;
using grpc::ClientContext;

namespace rtp_llm {

PrefillStatInfo::ExecuteStage PrefillStatInfo::saveStage() const {
    return stage;
}

void PrefillStatInfo::restoreStage(PrefillStatInfo::ExecuteStage stage_) {
    stage = stage_;
}

void PrefillStatInfo::nextStage() {
    stage = static_cast<PrefillStatInfo::ExecuteStage>(static_cast<int>(stage) + 1);
    auto cost_time_us = currentTimeUs() - begin_time;
    begin_time = currentTimeUs();
    switch (stage) {
        case getRpcConnection: {
            break;
        }
        case remoteAllocateResource: {
            get_rpc_connection_rt_us += cost_time_us;
            break;
        }
        case enqueueRequest: {
            remote_allocate_resource_rt_us += cost_time_us;
            break;
        }
        case remoteLoadCache: {
            enqueue_request_rt_us += cost_time_us;
            break;
        }
        case pollLocalOutput: {
            remote_load_cache_rt_us += cost_time_us;
            break;
        }
        case RemoteGenerate: {
            poll_local_output_rt_us += cost_time_us;
            break;
        }
        case pollRemoteOutput: {
            remote_generate_rt_us += cost_time_us;
            break;
        }
        case finish: {
            poll_remote_output_rt_us += cost_time_us;
            break;
        }
        default: {
            FT_CHECK_WITH_INFO(false, "error stage");
        }
    }
}

PrefillGenerateContext::~PrefillGenerateContext() {
    printTime();
    reportTime();
    closeGrpcStream();
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

grpc::Status PrefillGenerateContext::closeGrpcStream() {
    if (grpc_stream_closed) {
        return grpc::Status::OK;
    }
    grpc_stream_closed = true;
    if (cancelled()) {
        client_context->TryCancel();
    }
    client_stream->WritesDone();
    return client_stream->Finish();
}

void PrefillGenerateContext::reset() {
    GenerateContext::reset();
    grpc_stream_closed = false;
}

void PrefillGenerateContext::nextStage() {
    stat_info.nextStage();
}

void PrefillGenerateContext::markRequestEnd() {
    if (resource->tpSize() == 1) { 
        resource->cache_store->markRequestEnd(std::to_string(request_id));
        return;
    }
    const auto& prefill_workers = resource->workers;
    RemoteFinishRequestPB finish_request;
    finish_request.set_request_id(request_id);
    for (int i = 0; i < prefill_workers.size(); i++) {
        auto& prefill_worker = prefill_workers[i];
        auto connect_status = resource->rpc_pool.getConnection(prefill_worker);
        if (!connect_status.ok()) {
            FT_LOG_WARNING("request [%d], get grpc connection for ip %s failed, ignore markRequestEnd for it",
                            request_id, prefill_worker.c_str());
            continue;
        }
        auto          stub = connect_status.value().stub.get();
        ClientContext client_context;
        EmptyPB       response;
        auto          grpc_status = stub->RemoteFinish(&client_context, finish_request, &response);
        if (!grpc_status.ok()) {
            FT_LOG_WARNING("request [%d], remote finish for ip %s failed, ignore markRequestEnd for it",
                            request_id, prefill_worker.c_str());
            continue;
        }
    }
}

// for debug, will delete in future
void PrefillGenerateContext::printTime() {
    if (!stream) return;
    auto first_token_rt_us = stream->getTimeInfo().first_token_rt_us;
    auto receive_load_cost_time = response.receive_load_time() - request_begin_time_us;
    auto start_load_cost_time = response.start_load_time() - response.receive_load_time();
    auto load_cost_time = response.load_done_time() - response.start_load_time();
    auto receive_generate_cost_time = response.receive_generate_time() - response.receive_load_time();
    auto begin_compute_cost_time = response.begin_compute_time() - response.receive_generate_time();
    auto compute_cost_time = response.compute_done_time() - response.begin_compute_time();

    FT_LOG_DEBUG("request_id = [%d], first_token_rt_us = %ld", request_id, first_token_rt_us);
    FT_LOG_DEBUG("request_id = [%d], receive_load_cost_time = %ld, start_load_cost_time = %ld, load_cost_time = %ld",
                request_id, receive_load_cost_time, start_load_cost_time, load_cost_time);
    FT_LOG_DEBUG("request_id = [%d], receive_generate_cost_time = %ld, begin_compute_cost_time = %ld, "
                "compute_cost_time = %ld, remote_cost_time = %ld",
                request_id, receive_generate_cost_time, begin_compute_cost_time,
                compute_cost_time, remote_cost_time_us);
}

void PrefillGenerateContext::reportTime() {
    RpcMetricsCollector collector;
    collectBasicMetrics(collector);
    collector.retry_times                       = retry_times;
    collector.loading_cache_request             = loading_cache_requests;
    collector.get_rpc_connection_rt_us          = stat_info.get_rpc_connection_rt_us;
    collector.remote_allocate_resource_rt_us    = stat_info.remote_allocate_resource_rt_us;
    collector.enqueue_request_rt_us             = stat_info.enqueue_request_rt_us;
    collector.remote_load_cache_rt_us           = stat_info.remote_load_cache_rt_us;
    collector.poll_local_output_rt_us           = stat_info.poll_local_output_rt_us;
    collector.remote_generate_rt_us             = stat_info.remote_generate_rt_us;
    collector.poll_remote_output_rt_us          = stat_info.poll_remote_output_rt_us;
    reportMetrics(collector);
    metrics_reporter.reset();  // avoid to report metrics in base class
}

}  // namespace rtp_llm

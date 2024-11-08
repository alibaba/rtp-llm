#include "maga_transformer/cpp/model_rpc/RemoteRpcServer.h"

using namespace std;
using namespace fastertransformer;

namespace rtp_llm {

grpc::Status RemoteRpcServer::init(const EngineInitParams&                                maga_init_params,
                                   py::object                                             mm_process_engine,
                                   std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) {
    auto ret = LocalRpcServer::init(maga_init_params, mm_process_engine, std::move(propose_params));
    if (!ret.ok()) {
        return ret;
    }
    initLocalHostInfo();
    initLocalPeerInfo();
    initCacheStore();
    return grpc::Status::OK;
}

void RemoteRpcServer::initLocalHostInfo() {
    char hostname[HOST_NAME_MAX];
    FT_CHECK_WITH_INFO(gethostname(hostname, HOST_NAME_MAX) == 0, "failed to get local hostname");

    auto pid           = getpid();
    auto process_start_time = autil::TimeUtility::currentTimeInMicroSeconds();
    process_id_ = "hostname_" + string(hostname) + "_pid_" + std::to_string(pid)
                        + "_timestamp_" + std::to_string(process_start_time);
    FT_LOG_INFO("local process id is %s", process_id_.c_str());
}

void RemoteRpcServer::initLocalPeerInfo() {
    // not init when tp size = 1
    if (maga_init_params_.gpt_init_parameter.tp_size_ == 1) {
        return;
    }
    // not init when tp rank != 0
    if (maga_init_params_.gpt_init_parameter.tp_rank_ > 0) {
        return;
    }
    for (int i = 1; i < maga_init_params_.gpt_init_parameter.tp_size_; i++) {
        int port = maga_init_params_.gpt_init_parameter.model_rpc_port_;
        workers_.push_back("localhost:" + std::to_string((port + i * maga_init_params_.gpt_init_parameter.worker_port_offset_)));
    }
    for (auto& worker : workers_) {
        FT_LOG_INFO("worker address = %s, ", worker.c_str());
    }
    FT_LOG_INFO("\n");
}

void RemoteRpcServer::initCacheStore() {
    cache_store_ = engine_->getDevice()->cacheStore();
    if (maga_init_params_.gpt_init_parameter.use_cache_store_ && !cache_store_) {
        FT_FAIL("cache store is nullptr");
    }
}

}  // namespace rtp_llm

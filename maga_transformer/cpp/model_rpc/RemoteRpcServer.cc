#include "autil/NetUtil.h"
#include "maga_transformer/cpp/utils/NetUtil.h"
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
    string local_id, local_ip, hostname;
    if (!autil::NetUtil::GetDefaultIp(local_ip) || local_ip.empty()) {
        FT_LOG_WARNING("failed to get local ip, use hostname instead");
        FT_CHECK_WITH_INFO(autil::NetUtil::GetHostName(hostname), "get hostname failed");
        local_id = "hostname_" + hostname;
    } else {
        local_id = "ip_" + local_ip;
    }
    auto    pid        = getpid();
    auto    start_time = currentTimeUs();
    process_id_ = local_id + "_pid_" + std::to_string(pid)
                        + "_timestamp_" + std::to_string(start_time);
    FT_LOG_INFO("local process id is %s", process_id_.c_str());
}

void RemoteRpcServer::initLocalPeerInfo() {
    // not init when tp rank != 0
    if (maga_init_params_.gpt_init_parameter.tp_rank_ > 0) {
        return;
    }
    // worker 0 is master (rank 0)
    for (int i = 0; i < maga_init_params_.gpt_init_parameter.tp_size_; i++) {
        int port = maga_init_params_.gpt_init_parameter.model_rpc_port_;
        resource_.workers.push_back("localhost:" + std::to_string((port + i * maga_init_params_.gpt_init_parameter.worker_port_offset_)));
    }
    string worker_info = "worker address is ";
    for (auto& worker : resource_.workers) {
        worker_info += worker + ", ";
    }
    FT_LOG_INFO(worker_info);
}

void RemoteRpcServer::initCacheStore() {
    resource_.cache_store = engine_->getDevice()->cacheStore();
    if (maga_init_params_.gpt_init_parameter.use_cache_store_ && !resource_.cache_store) {
        FT_FAIL("cache store is nullptr");
    }
}

}  // namespace rtp_llm

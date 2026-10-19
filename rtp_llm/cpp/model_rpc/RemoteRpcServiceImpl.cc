#include <memory>
#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServiceImpl.h"
#include "rtp_llm/cpp/model_rpc/PrefillBatchRpcServer.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"

namespace rtp_llm {

void RemoteRpcServiceImpl::setDeferServiceStart(bool defer) {
    LocalRpcServiceImpl::setDeferServiceStart(defer);
    defer_cache_store_ = defer;
}

void RemoteRpcServiceImpl::startDeferredServices() {
    if (prefill_server_) {
        prefill_server_->startDeferredServices();
    }
    if (decode_server_) {
        decode_server_->startDeferredServices();
    }
    defer_cache_store_ = false;
}

grpc::Status RemoteRpcServiceImpl::init(const EngineInitParams&                                maga_init_params,
                                        py::object                                             mm_process_engine,
                                        std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) {
    // Standalone perf uses the batch scheduler to synthesize the selected phase.
    // Keep the real role/cache layout, but bypass remote PD request orchestration.
    perf_local_requests_ = maga_init_params.runtime_config.use_batch_decode_scheduler
                           && autil::EnvUtil::getEnv("RTP_LLM_PERF_LOCAL_REQUESTS", false);
    if (perf_local_requests_) {
        RTP_LLM_LOG_WARNING("Standalone perf RPC enabled: requests execute locally without PD transfer");
    }
    if (maga_init_params.pd_sep_config.role_type == RoleType::PREFILL) {
        prefill_server_ = std::make_shared<PrefillBatchRpcServer>();
        local_server_   = prefill_server_;
        local_server_->setDeferServiceStart(defer_service_start_);
        prefill_server_->setDeferCacheStore(defer_cache_store_);
        return prefill_server_->init(maga_init_params, mm_process_engine, std::move(propose_params));
    } else {
        decode_server_ = std::make_shared<DecodeRpcServer>();
        local_server_  = decode_server_;
        local_server_->setDeferServiceStart(defer_service_start_);
        decode_server_->setDeferCacheStore(defer_cache_store_);
        return decode_server_->init(maga_init_params, mm_process_engine, std::move(propose_params));
    }
}

}  // namespace rtp_llm

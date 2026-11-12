#include <memory>
#include "rtp_llm/cpp/model_rpc/RemoteRpcServiceImpl.h"
#include "rtp_llm/cpp/model_rpc/PrefillBatchRpcServer.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"

namespace rtp_llm {

void RemoteRpcServiceImpl::deferServiceStart() {
    LocalRpcServiceImpl::deferServiceStart();
    defer_cache_store_ = true;
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
                                        std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                                        py::object                                             mm_process_engine) {
    if (maga_init_params.pd_sep_config.role_type == RoleType::PREFILL) {
        prefill_server_ = std::make_shared<PrefillBatchRpcServer>();
        local_server_   = prefill_server_;
        local_server_->deferServiceStart();
        prefill_server_->setDeferCacheStore(defer_cache_store_);
        return prefill_server_->init(maga_init_params, std::move(propose_params), mm_process_engine);
    } else {
        decode_server_ = std::make_shared<DecodeRpcServer>();
        local_server_  = decode_server_;
        local_server_->deferServiceStart();
        decode_server_->setDeferCacheStore(defer_cache_store_);
        return decode_server_->init(maga_init_params, std::move(propose_params), mm_process_engine);
    }
}

}  // namespace rtp_llm

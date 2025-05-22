#include <memory>
#include "rtp_llm/cpp/model_rpc/RemoteRpcServiceImpl.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"

namespace rtp_llm {

grpc::Status RemoteRpcServiceImpl::init(const EngineInitParams&                           maga_init_params,
                                        py::object                                        mm_process_engine,
                                        std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) {
    if (maga_init_params.gpt_init_parameter.pd_separation_ == true) {
        prefill_server_ = std::make_shared<PrefillRpcServer>();
        local_server_ = prefill_server_;
        return prefill_server_->init(maga_init_params, mm_process_engine, std::move(propose_params));
    } else {
        decode_server_ = std::make_shared<DecodeRpcServer>();
        local_server_ = decode_server_;
        return decode_server_->init(maga_init_params, mm_process_engine, std::move(propose_params));
    }
}

}  // namespace rtp_llm

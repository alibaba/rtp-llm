#pragma once

#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/api_server/HttpApiServer.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServiceImpl.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServiceImpl.h"
#include <mutex>

namespace th = torch;

namespace rtp_llm {

class RtpLLMOp: public th::jit::CustomClassHolder {
public:
    RtpLLMOp();
    ~RtpLLMOp();

    void init(py::object model,
              py::object engine_config,
              py::object vit_config,
              py::object mm_process_engine,
              py::object propose_model,
              py::object token_processor,
              bool      defer_service_start = false);
    void startRPCServer();
    void updateRuntimeEndpoints(py::object runtime_config);
    void stop();
    void
    startHttpServer(py::object model_weights_loader, py::object world_info, py::object tokenizer, py::object render);
    void pause();
    void restart();

private:
    void             _init(int64_t                                       model_rpc_port,
                           int64_t                                       http_port,
                           const EngineInitParams                        maga_init_params,
                           py::object                                    mm_process_engine,
                           std::unique_ptr<ProposeModelEngineInitParams> propose_params,
                           py::object                                    token_processor);
    EngineInitParams initModel(py::object model, py::object engine_config, py::object vit_config);
    std::unique_ptr<ProposeModelEngineInitParams> initProposeModel(py::object              propose_model,
                                                                   const EngineInitParams& base_params);
    void                                          initRPCServer(const EngineInitParams&                       maga_init_params,
                                                                py::object                                    mm_process_engine,
                                                                std::unique_ptr<ProposeModelEngineInitParams> propose_params,
                                                                py::object                                    token_processor);
    void                                          prepareRPCService(const EngineInitParams&                       maga_init_params,
                                                                     py::object                                    mm_process_engine,
                                                                     std::unique_ptr<ProposeModelEngineInitParams> propose_params,
                                                                     py::object                                    token_processor,
                                                                     bool                                          defer_network_services);
    void                                          startRPCServerInternal(const EngineInitParams& maga_init_params);
    void                                          setServerStartError(const std::string& error);

private:
    std::unique_ptr<RpcServiceImpl> model_rpc_service_;
    std::shared_ptr<HttpApiServer>  http_server_;
    std::unique_ptr<grpc::Server>   grpc_server_;
    std::thread                     grpc_server_thread_;
    std::atomic<bool>               is_server_ready_{false};
    std::atomic<bool>               is_server_shutdown_{false};
    bool                            rpc_server_deferred_{false};
    std::unique_ptr<EngineInitParams> deferred_init_params_;
    py::object                      deferred_mm_process_engine_ = py::none();
    std::unique_ptr<ProposeModelEngineInitParams> deferred_propose_params_;
    py::object                      deferred_token_processor_ = py::none();
    std::string                     deferred_server_address_;
    std::atomic<bool>               server_start_failed_{false};
    std::atomic<bool>               stop_requested_{false};
    std::mutex                      server_state_mutex_;
    std::string                     server_start_error_;
    size_t                          model_id_ = 0;
};

void registerRtpLLMOp(const py::module& m);

}  // namespace rtp_llm

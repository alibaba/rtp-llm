#pragma once

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <thread>

#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/api_server/HttpApiServer.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServiceImpl.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServiceImpl.h"

namespace th = torch;

namespace rtp_llm {

class RtpLLMOp: public th::jit::CustomClassHolder {
public:
    RtpLLMOp();
    ~RtpLLMOp() noexcept;

    void init(py::object model,
              py::object engine_config,
              py::object vit_config,
              py::object propose_model,
              py::object token_processor,
              py::object mm_process_engine);
    void stop();
    void
    startHttpServer(py::object model_weights_loader, py::object world_info, py::object tokenizer, py::object render);
    void pause();
    void restart();

private:
    struct RpcServerThreadArgs;

    void             _init(int64_t                                       model_rpc_port,
                           int64_t                                       http_port,
                           const EngineInitParams                        maga_init_params,
                           std::unique_ptr<ProposeModelEngineInitParams> propose_params,
                           py::object                                    token_processor,
                           py::object                                    mm_process_engine);
    EngineInitParams initModel(py::object model, py::object engine_config, py::object vit_config);
    std::unique_ptr<ProposeModelEngineInitParams> initProposeModel(py::object              propose_model,
                                                                   const EngineInitParams& base_params);
    void initRPCServer(std::shared_ptr<RpcServerThreadArgs> args,
                       std::shared_ptr<std::promise<void>>  startup_signal,
                       std::shared_ptr<std::promise<void>>  exit_signal);
    void startHttpTransportStop();
    bool waitForHttpTransportStop(std::chrono::steady_clock::time_point deadline);
    void startGrpcShutdown(std::chrono::steady_clock::time_point deadline);
    bool waitForGrpcShutdown(std::chrono::steady_clock::time_point deadline);
    bool waitForGrpcServerExit(std::chrono::steady_clock::time_point deadline);
    void startServiceStop();
    bool waitForServiceStop(std::chrono::steady_clock::time_point deadline);
    void stopWithDeadline(std::chrono::steady_clock::time_point deadline);
    void forceStopNoThrow() noexcept;
    void forceStopNoThrow(std::chrono::steady_clock::time_point force_deadline) noexcept;

private:
    std::unique_ptr<RpcServiceImpl> model_rpc_service_;
    std::unique_ptr<HttpApiServer>  http_server_;
    std::unique_ptr<grpc::Server>   grpc_server_;
    std::thread                     grpc_server_thread_;
    std::shared_future<void>        grpc_server_exit_result_;
    std::thread                     grpc_shutdown_thread_;
    std::shared_future<void>        grpc_shutdown_result_;
    std::thread                     http_stop_thread_;
    std::shared_future<void>        http_stop_result_;
    std::thread                     service_stop_thread_;
    std::shared_future<void>        service_stop_result_;
    std::atomic<bool>               is_server_ready_{false};
    std::atomic<bool>               is_server_shutdown_{false};
    std::mutex                      stop_mutex_;
    size_t                          model_id_ = 0;
};

void registerRtpLLMOp(const py::module& m);

}  // namespace rtp_llm

#pragma once

#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <vector>
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"

namespace rtp_llm {

// This barrier uses RPCs, not collectives: the engine must keep participating in
// its normal forward collectives until every rank has finished its requests.
class ShutdownCoordinator {
public:
    using Deadline = std::chrono::system_clock::time_point;
    struct State {
        bool   draining;
        size_t onflight;
    };
    ShutdownCoordinator(
        std::vector<std::string> addresses,
        size_t                   rank,
        std::function<State()>   state,
        std::function<void()>    begin_shutdown = [] {});

    grpc::Status handle(const ShutdownRequestPB& request, ShutdownResponsePB* response);
    bool         wait(Deadline deadline);

private:
    Deadline effectiveDeadline(Deadline deadline);
    bool     acceptRelease(const std::vector<std::string>& instances,
                           size_t                          sender_rank        = 0,
                           const std::string&              sender_instance_id = "");
    bool
    recordPeerCertificate(size_t rank, const std::vector<std::string>& instances, const ShutdownResponsePB& response);
    std::vector<bool> deliveredRanks(const std::vector<std::string>& instances);
    grpc::Status call(size_t rank, const ShutdownRequestPB& request, ShutdownResponsePB* response, Deadline deadline);
    std::vector<std::string> releasedInstances();

    const std::vector<std::string>                 addresses_;
    const size_t                                   rank_;
    const std::function<State()>                   state_;
    const std::string                              instance_id_;
    const std::function<void()>                    begin_shutdown_;
    std::string                                    group_id_;
    std::vector<std::unique_ptr<RpcService::Stub>> peers_;
    std::mutex                                     mutex_;
    Deadline                                       deadline_ = Deadline::max();
    std::vector<std::string>                       released_instances_;
    // Protected by mutex_ and valid only for the immutable released_instances_.
    // True means this incarnation is known to hold that exact certificate.
    std::vector<bool> release_delivered_;
};

}  // namespace rtp_llm

#include "rtp_llm/cpp/model_rpc/ShutdownCoordinator.h"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <thread>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

namespace rtp_llm {
namespace {
std::string instanceId() {
    std::random_device random;
    return std::to_string(random()) + "-" + std::to_string(random()) + "-" + std::to_string(random()) + "-"
           + std::to_string(random());
}
}  // namespace

ShutdownCoordinator::ShutdownCoordinator(std::vector<std::string> addresses,
                                         size_t                   rank,
                                         std::function<State()>   state,
                                         std::function<void()>    begin_shutdown):
    addresses_(std::move(addresses)),
    rank_(rank),
    state_(std::move(state)),
    instance_id_(instanceId()),
    begin_shutdown_(std::move(begin_shutdown)) {
    if (addresses_.empty() || rank_ >= addresses_.size()) {
        throw std::invalid_argument("invalid shutdown group membership");
    }
    for (const auto& address : addresses_) {
        group_id_ += std::to_string(address.size()) + ":" + address;
        peers_.push_back(RpcService::NewStub(grpc::CreateChannel(address, grpc::InsecureChannelCredentials())));
    }
}

ShutdownCoordinator::Deadline ShutdownCoordinator::effectiveDeadline(Deadline deadline) {
    std::lock_guard<std::mutex> lock(mutex_);
    deadline_ = std::min(deadline_, deadline);
    return deadline_;
}

std::vector<std::string> ShutdownCoordinator::releasedInstances() {
    std::lock_guard<std::mutex> lock(mutex_);
    return released_instances_;
}

bool ShutdownCoordinator::acceptRelease(const std::vector<std::string>& instances) {
    const auto state = state_();
    if (instances.size() != addresses_.size() || instances[rank_] != instance_id_ || !state.draining || state.onflight
        || std::any_of(instances.begin(), instances.end(), [](const auto& id) { return id.empty(); })) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    released_instances_ = instances;
    return true;
}

grpc::Status ShutdownCoordinator::handle(const ShutdownRequestPB& request, ShutdownResponsePB* response) {
    if (request.group_id() != group_id_) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "shutdown group membership mismatch"};
    }
    if (request.begin_shutdown()) {
        if (request.deadline_unix_ms() <= 0) {
            return {grpc::StatusCode::INVALID_ARGUMENT, "shutdown requires an absolute deadline"};
        }
        effectiveDeadline(Deadline(std::chrono::milliseconds(request.deadline_unix_ms())));
        begin_shutdown_();
    }
    response->set_deadline_unix_ms(
        std::chrono::duration_cast<std::chrono::milliseconds>(effectiveDeadline(Deadline::max()).time_since_epoch())
            .count());
    if (request.drained_instances_size()
        && !acceptRelease({request.drained_instances().begin(), request.drained_instances().end()})) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "stale or premature shutdown release"};
    }
    const auto state = state_();
    response->set_draining(state.draining);
    response->set_onflight(state.onflight);
    response->set_instance_id(instance_id_);
    for (const auto& instance : releasedInstances()) {
        response->add_drained_instances(instance);
    }
    return grpc::Status::OK;
}

grpc::Status ShutdownCoordinator::call(size_t                   rank,
                                       const ShutdownRequestPB& request,
                                       ShutdownResponsePB*      response,
                                       Deadline                 deadline) {
    if (rank == rank_) {
        return handle(request, response);
    }
    grpc::ClientContext context;
    context.set_deadline(std::min(deadline, std::chrono::system_clock::now() + std::chrono::milliseconds(200)));
    FunctionRequestPB wire_request;
    *wire_request.mutable_shutdown_request() = request;
    FunctionResponsePB wire_response;
    auto               status = peers_[rank]->ExecuteFunction(&context, wire_request, &wire_response);
    if (status.ok() && !wire_response.has_shutdown_response()) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "peer does not support shutdown coordination"};
    }
    if (status.ok()) {
        *response = wire_response.shutdown_response();
    }
    return status;
}

bool ShutdownCoordinator::wait(Deadline deadline) {
    ShutdownRequestPB request;
    request.set_group_id(group_id_);
    request.set_begin_shutdown(true);
    deadline = effectiveDeadline(deadline);
    while (std::chrono::system_clock::now() < deadline) {
        deadline = effectiveDeadline(deadline);
        request.set_deadline_unix_ms(
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline.time_since_epoch()).count());
        auto instances = releasedInstances();
        if (instances.empty()) {
            std::vector<std::string> observed(addresses_.size());
            bool                     all_drained = true;
            for (size_t i = 0; i < addresses_.size() && std::chrono::system_clock::now() < deadline; ++i) {
                ShutdownResponsePB response;
                if (!call(i, request, &response, deadline).ok()) {
                    // A disappeared peer is not evidence that it drained.
                    all_drained = false;
                    continue;
                }
                if (response.deadline_unix_ms() > 0) {
                    deadline = effectiveDeadline(Deadline(std::chrono::milliseconds(response.deadline_unix_ms())));
                }
                if (response.drained_instances_size()
                    && acceptRelease({response.drained_instances().begin(), response.drained_instances().end()})) {
                    instances = releasedInstances();
                    break;
                }
                observed[i] = response.instance_id();
                all_drained = all_drained && response.draining() && response.onflight() == 0;
            }
            if (instances.empty() && all_drained && acceptRelease(observed)) {
                instances = std::move(observed);
            }
        }
        if (!instances.empty()) {
            // Preserve the all-drained certificate at each peer before exiting.
            // Peers can also relay it to a lagging rank after the observer exits.
            for (const auto& instance : instances) {
                request.add_drained_instances(instance);
            }
            std::vector<bool> delivered(addresses_.size(), false);
            do {
                deadline = effectiveDeadline(deadline);
                request.set_deadline_unix_ms(
                    std::chrono::duration_cast<std::chrono::milliseconds>(deadline.time_since_epoch()).count());
                for (size_t i = 0; i < addresses_.size() && std::chrono::system_clock::now() < deadline; ++i) {
                    if (!delivered[i]) {
                        ShutdownResponsePB response;
                        delivered[i] = call(i, request, &response, deadline).ok();
                        if (delivered[i] && response.deadline_unix_ms() > 0) {
                            deadline =
                                effectiveDeadline(Deadline(std::chrono::milliseconds(response.deadline_unix_ms())));
                        }
                    }
                }
                if (std::all_of(delivered.begin(), delivered.end(), [](bool done) { return done; })) {
                    break;
                }
                std::this_thread::sleep_until(
                    std::min(deadline, std::chrono::system_clock::now() + std::chrono::milliseconds(20)));
            } while (std::chrono::system_clock::now() < deadline);
            // Even if a peer exits before acknowledging delivery, the
            // certificate proves there are no user requests left to interrupt.
            return true;
        }
        std::this_thread::sleep_until(
            std::min(deadline, std::chrono::system_clock::now() + std::chrono::milliseconds(20)));
    }
    return false;
}

}  // namespace rtp_llm

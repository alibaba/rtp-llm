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

bool ShutdownCoordinator::acceptRelease(const std::vector<std::string>& instances,
                                        size_t                          sender_rank,
                                        const std::string&              sender_instance_id) {
    const auto state = state_();
    if (instances.size() != addresses_.size() || instances[rank_] != instance_id_ || !state.draining || state.onflight
        || std::any_of(instances.begin(), instances.end(), [](const auto& id) { return id.empty(); })
        || (!sender_instance_id.empty()
            && (sender_rank >= instances.size() || instances[sender_rank] != sender_instance_id))) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!released_instances_.empty() && released_instances_ != instances) {
        // Never mix delivery evidence from different incarnation certificates.
        return false;
    }
    if (released_instances_.empty()) {
        released_instances_ = instances;
        release_delivered_.assign(addresses_.size(), false);
    }
    release_delivered_[rank_] = true;
    if (!sender_instance_id.empty()) {
        release_delivered_[sender_rank] = true;
    }
    return true;
}

bool ShutdownCoordinator::recordPeerCertificate(size_t                          rank,
                                                const std::vector<std::string>& instances,
                                                const ShutdownResponsePB&       response) {
    if (rank >= instances.size() || response.instance_id() != instances[rank]
        || std::vector<std::string>(response.drained_instances().begin(), response.drained_instances().end())
               != instances) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (released_instances_ != instances) {
        return false;
    }
    release_delivered_[rank] = true;
    return true;
}

std::vector<bool> ShutdownCoordinator::deliveredRanks(const std::vector<std::string>& instances) {
    std::lock_guard<std::mutex> lock(mutex_);
    return released_instances_ == instances ? release_delivered_ : std::vector<bool>(addresses_.size(), false);
}

grpc::Status ShutdownCoordinator::handle(const ShutdownRequestPB& request, ShutdownResponsePB* response) {
    if (request.group_id() != group_id_) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "shutdown group membership mismatch"};
    }
    if (!request.sender_instance_id().empty() && request.drained_instances_size() == 0) {
        return {grpc::StatusCode::INVALID_ARGUMENT, "shutdown sender proof requires a release certificate"};
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
        && !acceptRelease({request.drained_instances().begin(), request.drained_instances().end()},
                          request.sender_rank(),
                          request.sender_instance_id())) {
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
                if (response.drained_instances_size() && !response.instance_id().empty()
                    && acceptRelease({response.drained_instances().begin(), response.drained_instances().end()},
                                     i,
                                     response.instance_id())) {
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
            request.set_sender_rank(rank_);
            request.set_sender_instance_id(instance_id_);
            auto delivered = deliveredRanks(instances);
            do {
                deadline = effectiveDeadline(deadline);
                request.set_deadline_unix_ms(
                    std::chrono::duration_cast<std::chrono::milliseconds>(deadline.time_since_epoch()).count());
                for (size_t i = 0; i < addresses_.size() && std::chrono::system_clock::now() < deadline; ++i) {
                    // Incoming release RPCs can prove that an issuer holds
                    // this certificate even if it exits before our next call.
                    delivered = deliveredRanks(instances);
                    if (!delivered[i]) {
                        ShutdownResponsePB response;
                        const bool         acknowledged =
                            call(i, request, &response, deadline).ok() && recordPeerCertificate(i, instances, response);
                        if (acknowledged && response.deadline_unix_ms() > 0) {
                            deadline =
                                effectiveDeadline(Deadline(std::chrono::milliseconds(response.deadline_unix_ms())));
                        }
                    }
                }
                delivered = deliveredRanks(instances);
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

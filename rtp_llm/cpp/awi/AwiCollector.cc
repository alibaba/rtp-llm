#include "rtp_llm/cpp/awi/AwiCollector.h"

namespace rtp_llm {

AwiCollector& AwiCollector::instance() {
    static AwiCollector collector;
    return collector;
}

void AwiCollector::setRank(int rank) {
    std::lock_guard<std::mutex> lock(mu_);
    rank_ = rank;
}

int AwiCollector::rank() const {
    std::lock_guard<std::mutex> lock(mu_);
    return rank_;
}

void AwiCollector::recordEvent(AwiEvent event) {
    std::lock_guard<std::mutex> lock(mu_);
    if (event.rank < 0) {
        event.rank = rank_;
    }
    if (event.operation_id.empty()) {
        if (has_armed_) {
            event.operation_id = armed_.operation_id;
        } else {
            // No armed barrier: attribute to the container-level operation
            // (AWI_OPERATION_ID) so operation-scoped fetches observe
            // smoke-phase events; empty default keeps legacy behavior.
            event.operation_id = default_operation_id_;
        }
    }
    events_.push_back(std::move(event));
    while (events_.size() > kMaxEvents) {
        events_.pop_front();
    }
}

std::vector<AwiEvent> AwiCollector::events(const std::string& operation_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<AwiEvent>       result;
    for (const auto& event : events_) {
        if (operation_id.empty() || event.operation_id == operation_id) {
            result.push_back(event);
        }
    }
    return result;
}

void AwiCollector::clearEvents() {
    std::lock_guard<std::mutex> lock(mu_);
    events_.clear();
}

void AwiCollector::armOperation(AwiOperation operation) {
    std::lock_guard<std::mutex> lock(mu_);
    if (operation.operation_id.empty()) {
        has_armed_ = false;
        armed_     = AwiOperation{};
        return;
    }
    armed_     = std::move(operation);
    has_armed_ = true;
}

bool AwiCollector::hasArmedOperation() const {
    std::lock_guard<std::mutex> lock(mu_);
    return has_armed_;
}

AwiOperation AwiCollector::armedOperation() const {
    std::lock_guard<std::mutex> lock(mu_);
    return armed_;
}

void AwiCollector::bindInternalId(const std::string& awi_request_id, int64_t internal_id) {
    std::lock_guard<std::mutex> lock(mu_);
    if (has_armed_ && armed_.awi_request_ids.count(awi_request_id)) {
        armed_.request_ids.insert(internal_id);
    }
}

void AwiCollector::markReleased() {
    std::lock_guard<std::mutex> lock(mu_);
    if (has_armed_) {
        armed_.released = true;
    }
}

bool AwiCollector::markResidentEmitted() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!has_armed_ || armed_.resident_emitted) {
        return false;
    }
    armed_.resident_emitted = true;
    return true;
}

void AwiCollector::clearOperation() {
    std::lock_guard<std::mutex> lock(mu_);
    has_armed_ = false;
    armed_     = AwiOperation{};
}

void AwiCollector::setDefaultOperationId(std::string operation_id) {
    std::lock_guard<std::mutex> lock(mu_);
    default_operation_id_ = std::move(operation_id);
}

void AwiCollector::setBuildInfo(std::string internal_sha, std::string public_sha) {
    std::lock_guard<std::mutex> lock(mu_);
    internal_sha_ = std::move(internal_sha);
    public_sha_   = std::move(public_sha);
}

void AwiCollector::setWeightsJson(std::string json) {
    std::lock_guard<std::mutex> lock(mu_);
    weights_json_ = std::move(json);
}

void AwiCollector::setStartupNetworkFetches(int64_t count) {
    std::lock_guard<std::mutex> lock(mu_);
    startup_network_fetches_ = count;
}

void AwiCollector::setConfigurationJson(std::string json) {
    std::lock_guard<std::mutex> lock(mu_);
    configuration_json_ = std::move(json);
}

std::string AwiCollector::buildInfoJson() const {
    std::lock_guard<std::mutex> lock(mu_);
    return std::string("{\"source\":{\"internal\":\"") + internal_sha_ + "\",\"public\":\"" + public_sha_ + "\"}}";
}

std::string AwiCollector::weightsJson() const {
    std::lock_guard<std::mutex> lock(mu_);
    return weights_json_;
}

int64_t AwiCollector::startupNetworkFetches() const {
    std::lock_guard<std::mutex> lock(mu_);
    return startup_network_fetches_;
}

std::string AwiCollector::configurationJson() const {
    std::lock_guard<std::mutex> lock(mu_);
    return configuration_json_;
}

bool AwiCollector::markPrefillEmitted(int64_t request_id) {
    std::lock_guard<std::mutex> lock(mu_);
    return prefill_emitted_.insert(request_id).second;
}

}  // namespace rtp_llm

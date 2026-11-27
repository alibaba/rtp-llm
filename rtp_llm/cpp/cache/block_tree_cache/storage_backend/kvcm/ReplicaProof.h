#pragma once

#include <algorithm>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "WorkerPayloadProof.h"

namespace rtp_llm::kvcm {

// A location can carry several members packed into one transfer. The member
// identity belongs to the storage layout, independent of its wire encoding.
struct ReplicaMember {
    std::string name;
    int32_t     rank;

    bool operator<(const ReplicaMember& other) const noexcept {
        return std::tie(name, rank) < std::tie(other.name, other.rank);
    }
};

struct ReplicaLocation {
    std::string                name;
    std::string                uri;
    std::vector<ReplicaMember> members;
};

template<typename SpecSizes, typename SpecGroups>
void validateReplicaRegistration(const SpecSizes& sizes, const SpecGroups& groups) {
    for (const auto& [name, bytes] : sizes) {
        if (name.empty() || bytes <= 0) {
            throw std::invalid_argument("invalid KVCM location spec: " + name);
        }
    }
    for (const auto& [group, specs] : groups) {
        if (group.empty() || specs.empty()) {
            throw std::invalid_argument("empty KVCM location spec group: " + group);
        }
        std::set<std::string> seen;
        for (const auto& name : specs) {
            if (!sizes.count(name) || !seen.insert(name).second) {
                throw std::invalid_argument("missing or duplicate KVCM spec " + name + " in " + group);
            }
        }
    }
}

// Select complete locations for one key row. A packed spec that also writes
// an unbound member cannot be used for this request, even if its URI is valid.
inline std::vector<const ReplicaLocation*> coverReplicaMembers(const std::vector<ReplicaLocation>& locations,
                                                               const std::vector<ReplicaMember>&   required) {
    std::set<ReplicaMember>             missing(required.begin(), required.end());
    const auto                          wanted = missing;
    std::vector<const ReplicaLocation*> selected;
    for (const auto& location : locations) {
        if (location.uri.empty() || location.members.empty()
            || !std::all_of(location.members.begin(), location.members.end(), [&](const auto& member) {
                   return wanted.count(member);
               })) {
            continue;
        }
        for (const auto& member : location.members) {
            missing.erase(member);
        }
        selected.push_back(&location);
    }
    if (!missing.empty()) {
        selected.clear();
    }
    return selected;
}

// Closing a session does not establish replica visibility. Unknown payload
// safety cannot authorize publication; protocol-specific cleanup is separate.
template<typename Finish, typename Confirm>
bool finishReplicaWrite(WorkerPayloadProof::Result proof, Finish&& finish, Confirm&& confirm) {
    if (proof == WorkerPayloadProof::Result::UNKNOWN) {
        return false;
    }
    const bool complete = proof == WorkerPayloadProof::Result::SUCCESS;
    finish(complete);
    return complete && confirm();
}

}  // namespace rtp_llm::kvcm

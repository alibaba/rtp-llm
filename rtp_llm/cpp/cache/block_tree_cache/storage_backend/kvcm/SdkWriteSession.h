#pragma once

#include <stdexcept>
#include <utility>
#include <vector>

#include "kvcm_client/common.h"
#include "ReplicaProof.h"

namespace rtp_llm::kvcm {

// The metadata allocation and terminal decision are independent of how ranks
// move payload bytes or how an SDK version carries checksum metadata.
class SdkWriteSession {
public:
    SdkWriteSession() = default;
    explicit SdkWriteSession(kv_cache_manager::WriteLocation location): location_(std::move(location)) {}
    SdkWriteSession(const SdkWriteSession&) = delete;
    SdkWriteSession& operator=(const SdkWriteSession&) = delete;
    SdkWriteSession(SdkWriteSession&& other) noexcept:
        location_(std::move(other.location_)),
        finish_attempted_(std::exchange(other.finish_attempted_, true)) {}
    SdkWriteSession& operator=(SdkWriteSession&& other) noexcept {
        if (this != &other) {
            location_ = std::move(other.location_);
            finish_attempted_ = std::exchange(other.finish_attempted_, true);
        }
        return *this;
    }

    kv_cache_manager::WriteLocation& location() { return location_; }

    std::vector<size_t> allocatedKeys(size_t key_count) const {
        std::vector<size_t> keys;
        if (const auto* offset = std::get_if<kv_cache_manager::BlockMaskOffset>(&location_.block_mask)) {
            if (*offset > key_count) {
                throw std::invalid_argument("KVCM write offset exceeds key count");
            }
            for (auto key = *offset; key < key_count; ++key) {
                keys.push_back(key);
            }
        } else {
            const auto& mask = std::get<kv_cache_manager::BlockMaskVector>(location_.block_mask);
            if (mask.size() != key_count) {
                throw std::invalid_argument("KVCM write mask size mismatch");
            }
            for (size_t key = 0; key < mask.size(); ++key) {
                if (!mask[key]) {
                    keys.push_back(key);
                }
            }
        }
        if (keys.size() != location_.locations.size() || (!keys.empty() && location_.write_session_id.empty())) {
            throw std::invalid_argument("KVCM write allocation does not match its session and key mask");
        }
        return keys;
    }

    template<class Before, class Finish, class Confirm>
    bool finish(WorkerPayloadProof::Result proof, Before&& before, Finish&& close, Confirm&& confirm) {
        if (finish_attempted_ || location_.write_session_id.empty()) {
            return false;
        }
        return finishReplicaWrite(proof,
                                  [&](bool complete) {
                                      before();
                                      finish_attempted_ = true;
                                      close(location_.write_session_id, complete);
                                  },
                                  std::forward<Confirm>(confirm));
    }

    template<class Abort>
    void abort(WorkerPayloadProof::Result proof, bool allow_unknown_cleanup, Abort&& close) {
        const bool unknown = proof == WorkerPayloadProof::Result::UNKNOWN;
        if (finish_attempted_ || location_.write_session_id.empty() || (unknown && !allow_unknown_cleanup)) {
            return;
        }
        finish_attempted_ = true;
        close(location_.write_session_id, unknown);
    }

private:
    kv_cache_manager::WriteLocation location_;
    bool finish_attempted_{false};
};

}  // namespace rtp_llm::kvcm

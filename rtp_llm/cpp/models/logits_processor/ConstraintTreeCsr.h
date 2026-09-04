#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rtp_llm/cpp/core/Buffer.h"

namespace rtp_llm {

class DeviceBase;

// Immutable CSR snapshot pinned by a generation request for its entire lifetime.
class ConstraintTreeCsrSnapshot {
public:
    static constexpr int32_t INVALID_TRANSITION = -2;

    uint64_t version() const {
        return version_;
    }
    int32_t startTokenId() const {
        return start_token_id_;
    }
    int32_t endTokenId() const {
        return end_token_id_;
    }
    uint64_t sidCount() const {
        return sid_count_;
    }
    size_t stateCount() const {
        return row_ptr_.size() - 1;
    }
    size_t edgeCount() const {
        return col_idx_.size();
    }
    size_t rootCandidateCount() const {
        return static_cast<size_t>(row_ptr_[1] - row_ptr_[0]);
    }
    bool deviceReady() const {
        return device_row_ptr_ != nullptr && device_col_idx_ != nullptr;
    }

    const std::vector<int32_t>& rowPtr() const {
        return row_ptr_;
    }
    const std::vector<int32_t>& colIdx() const {
        return col_idx_;
    }
    const std::vector<int32_t>& nextState() const {
        return next_state_;
    }
    const BufferPtr& deviceRowPtr() const {
        return device_row_ptr_;
    }
    const BufferPtr& deviceColIdx() const {
        return device_col_idx_;
    }
    // Returns -1 for the terminal edge and INVALID_TRANSITION when the edge
    // does not exist. Keeping those outcomes distinct makes strict decoding
    // fail closed if a sampler ever returns a masked token.
    int32_t transition(int32_t state, int32_t token) const;

private:
    friend class ConstraintTreeCsrManager;

    uint64_t             version_        = 0;
    int32_t              start_token_id_ = -1;
    int32_t              end_token_id_   = -1;
    uint64_t             sid_count_      = 0;
    std::vector<int32_t> row_ptr_;
    std::vector<int32_t> col_idx_;
    std::vector<int32_t> next_state_;
    BufferPtr            device_row_ptr_;
    BufferPtr            device_col_idx_;
};

using ConstraintTreeCsrSnapshotPtr = std::shared_ptr<const ConstraintTreeCsrSnapshot>;

enum class ConstraintTreeCsrUpdateCode {
    UPDATED,
    ALREADY_CURRENT,
    STALE_VERSION,
    INVALID_ARTIFACT,
    DEVICE_ERROR,
    RESOURCE_ERROR,
};

struct ConstraintTreeCsrUpdateResult {
    ConstraintTreeCsrUpdateCode code;
    uint64_t                    current_version;
    std::string                 message;

    bool ok() const {
        return code == ConstraintTreeCsrUpdateCode::UPDATED || code == ConstraintTreeCsrUpdateCode::ALREADY_CURRENT;
    }
};

const char* constraintTreeCsrUpdateCodeName(ConstraintTreeCsrUpdateCode code);

// Process-wide owner of the current runtime CSR snapshot.
class ConstraintTreeCsrManager {
public:
    static std::shared_ptr<ConstraintTreeCsrManager> instance();

    ConstraintTreeCsrSnapshotPtr snapshot() const;
    uint64_t                     currentVersion() const;

    static ConstraintTreeCsrUpdateResult peekVersion(const std::string& artifact, uint64_t& version);
    ConstraintTreeCsrUpdateResult        updateFromBinary(const std::string& artifact, DeviceBase* device);

private:
    ConstraintTreeCsrManager()                                = default;
    ConstraintTreeCsrManager(const ConstraintTreeCsrManager&) = delete;
    ConstraintTreeCsrManager(ConstraintTreeCsrManager&&)      = delete;
    ConstraintTreeCsrManager& operator=(const ConstraintTreeCsrManager&) = delete;

private:
    mutable std::mutex           update_mutex_;
    ConstraintTreeCsrSnapshotPtr snapshot_;
};

}  // namespace rtp_llm

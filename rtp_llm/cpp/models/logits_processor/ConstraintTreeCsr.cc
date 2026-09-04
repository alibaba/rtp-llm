#include "rtp_llm/cpp/models/logits_processor/ConstraintTreeCsr.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string_view>

#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

constexpr char     kMagic[]       = {'R', 'T', 'P', 'C', 'S', 'R', '0', '1'};
constexpr uint32_t kFormatVersion = 1;
constexpr uint32_t kHeaderSize    = 48;
constexpr int32_t  kTerminalState = -1;

uint32_t readU32(const char* data) {
    return static_cast<uint32_t>(static_cast<unsigned char>(data[0]))
           | (static_cast<uint32_t>(static_cast<unsigned char>(data[1])) << 8)
           | (static_cast<uint32_t>(static_cast<unsigned char>(data[2])) << 16)
           | (static_cast<uint32_t>(static_cast<unsigned char>(data[3])) << 24);
}

uint64_t readU64(const char* data) {
    return static_cast<uint64_t>(readU32(data)) | (static_cast<uint64_t>(readU32(data + 4)) << 32);
}

int32_t readI32(const char* data) {
    return static_cast<int32_t>(readU32(data));
}

struct WireHeader {
    uint64_t version;
    int32_t  start_token_id;
    int32_t  end_token_id;
    uint32_t state_count;
    uint32_t edge_count;
    uint64_t sid_count;
};

bool parseHeader(std::string_view artifact, WireHeader& header, std::string& error, bool validate_length) {
    if (artifact.size() < kHeaderSize) {
        error = "CSR artifact is shorter than its header";
        return false;
    }
    if (std::memcmp(artifact.data(), kMagic, sizeof(kMagic)) != 0) {
        error = "CSR artifact has invalid magic";
        return false;
    }
    if (readU32(artifact.data() + 8) != kFormatVersion || readU32(artifact.data() + 12) != kHeaderSize) {
        error = "unsupported CSR artifact format";
        return false;
    }

    header.version        = readU64(artifact.data() + 16);
    header.start_token_id = readI32(artifact.data() + 24);
    header.end_token_id   = readI32(artifact.data() + 28);
    header.state_count    = readU32(artifact.data() + 32);
    header.edge_count     = readU32(artifact.data() + 36);
    header.sid_count      = readU64(artifact.data() + 40);
    if (header.version == 0 || header.start_token_id < 0 || header.end_token_id < 0
        || header.start_token_id == header.end_token_id || header.state_count == 0 || header.edge_count == 0
        || header.sid_count == 0 || header.state_count > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())
        || header.edge_count > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())
        || header.sid_count > header.edge_count) {
        error = "CSR artifact header contains invalid tokens, counts, or version";
        return false;
    }

    const uint64_t element_count =
        static_cast<uint64_t>(header.state_count) + 1 + 2 * static_cast<uint64_t>(header.edge_count);
    if (element_count > (std::numeric_limits<uint64_t>::max() - kHeaderSize) / sizeof(int32_t)) {
        error = "CSR artifact size overflows uint64";
        return false;
    }
    const uint64_t expected_size = kHeaderSize + element_count * sizeof(int32_t);
    if (validate_length && expected_size != artifact.size()) {
        error = "CSR artifact length does not match its header";
        return false;
    }
    return true;
}

void readVector(std::string_view artifact, size_t& offset, std::vector<int32_t>& values) {
    for (auto& value : values) {
        value = readI32(artifact.data() + offset);
        offset += sizeof(int32_t);
    }
}

std::string validateSnapshot(const ConstraintTreeCsrSnapshot& snapshot) {
    const auto& row_ptr    = snapshot.rowPtr();
    const auto& col_idx    = snapshot.colIdx();
    const auto& next_state = snapshot.nextState();
    if (row_ptr.size() < 2 || col_idx.empty() || col_idx.size() != next_state.size()) {
        return "CSR arrays must not be empty and edge arrays must have equal lengths";
    }
    if (row_ptr.front() != 0 || row_ptr.back() != static_cast<int32_t>(col_idx.size())) {
        return "CSR row_ptr boundaries are invalid";
    }

    uint64_t      terminal_edges = 0;
    const int32_t state_count    = static_cast<int32_t>(snapshot.stateCount());
    for (int32_t state = 0; state < state_count; ++state) {
        const int32_t begin = row_ptr[state];
        const int32_t end   = row_ptr[state + 1];
        if (begin < 0 || end <= begin || end > static_cast<int32_t>(col_idx.size())) {
            return "CSR row_ptr must be increasing and every state must have an outgoing edge";
        }
        int32_t previous_token = -1;
        for (int32_t edge = begin; edge < end; ++edge) {
            const int32_t token = col_idx[edge];
            const int32_t next  = next_state[edge];
            if (token < 0 || token <= previous_token || token == snapshot.startTokenId()) {
                return "CSR candidate rows must contain sorted unique non-negative token ids";
            }
            if (token == snapshot.endTokenId()) {
                if (next != kTerminalState) {
                    return "CSR end-token edge must point to the terminal state";
                }
                terminal_edges++;
            } else if (next < 0 || next >= state_count) {
                return "CSR non-terminal edge points to an invalid state";
            }
            previous_token = token;
        }
    }
    if (terminal_edges != snapshot.sidCount()) {
        return "CSR terminal edge count does not match sid_count";
    }
    return {};
}

BufferPtr upload(const std::vector<int32_t>& values, DeviceBase* device) {
    auto host = vector2Buffer(values);
    return device->clone({*host, AllocationType::DEVICE});
}

}  // namespace

int32_t ConstraintTreeCsrSnapshot::transition(int32_t state, int32_t token) const {
    if (state < 0 || static_cast<size_t>(state + 1) >= row_ptr_.size()) {
        return INVALID_TRANSITION;
    }
    const auto begin = col_idx_.begin() + row_ptr_[state];
    const auto end   = col_idx_.begin() + row_ptr_[state + 1];
    const auto found = std::lower_bound(begin, end, token);
    if (found == end || *found != token) {
        return INVALID_TRANSITION;
    }
    return next_state_[static_cast<size_t>(found - col_idx_.begin())];
}

const char* constraintTreeCsrUpdateCodeName(ConstraintTreeCsrUpdateCode code) {
    switch (code) {
        case ConstraintTreeCsrUpdateCode::UPDATED:
            return "updated";
        case ConstraintTreeCsrUpdateCode::ALREADY_CURRENT:
            return "already_current";
        case ConstraintTreeCsrUpdateCode::STALE_VERSION:
            return "stale_version";
        case ConstraintTreeCsrUpdateCode::INVALID_ARTIFACT:
            return "invalid_artifact";
        case ConstraintTreeCsrUpdateCode::DEVICE_ERROR:
            return "device_error";
        case ConstraintTreeCsrUpdateCode::RESOURCE_ERROR:
            return "resource_error";
    }
    return "unknown";
}

std::shared_ptr<ConstraintTreeCsrManager> ConstraintTreeCsrManager::instance() {
    static auto singleton = std::shared_ptr<ConstraintTreeCsrManager>(new ConstraintTreeCsrManager());
    return singleton;
}

ConstraintTreeCsrSnapshotPtr ConstraintTreeCsrManager::snapshot() const {
    return std::atomic_load_explicit(&snapshot_, std::memory_order_acquire);
}

uint64_t ConstraintTreeCsrManager::currentVersion() const {
    const auto current = snapshot();
    return current ? current->version() : 0;
}

ConstraintTreeCsrUpdateResult ConstraintTreeCsrManager::peekVersion(const std::string& artifact, uint64_t& version) {
    WireHeader  header{};
    std::string error;
    if (!parseHeader(artifact, header, error, true)) {
        return {ConstraintTreeCsrUpdateCode::INVALID_ARTIFACT, 0, std::move(error)};
    }
    version = header.version;
    return {ConstraintTreeCsrUpdateCode::UPDATED, version, "CSR header is valid"};
}

ConstraintTreeCsrUpdateResult ConstraintTreeCsrManager::updateFromBinary(const std::string& artifact,
                                                                         DeviceBase*        device) {
    WireHeader  header{};
    std::string error;
    if (!parseHeader(artifact, header, error, true)) {
        return {ConstraintTreeCsrUpdateCode::INVALID_ARTIFACT, currentVersion(), std::move(error)};
    }

    const auto active = snapshot();
    if (active && header.version < active->version()) {
        return {ConstraintTreeCsrUpdateCode::STALE_VERSION, active->version(), "a newer CSR tree is already active"};
    }
    if (active && header.version == active->version()) {
        return {ConstraintTreeCsrUpdateCode::ALREADY_CURRENT, active->version(), "CSR tree version is already active"};
    }

    std::shared_ptr<ConstraintTreeCsrSnapshot> next;
    try {
        next                  = std::make_shared<ConstraintTreeCsrSnapshot>();
        next->version_        = header.version;
        next->start_token_id_ = header.start_token_id;
        next->end_token_id_   = header.end_token_id;
        next->sid_count_      = header.sid_count;
        next->row_ptr_.resize(static_cast<size_t>(header.state_count) + 1);
        next->col_idx_.resize(header.edge_count);
        next->next_state_.resize(header.edge_count);
        size_t offset = kHeaderSize;
        readVector(artifact, offset, next->row_ptr_);
        readVector(artifact, offset, next->col_idx_);
        readVector(artifact, offset, next->next_state_);
    } catch (const std::bad_alloc&) {
        return {
            ConstraintTreeCsrUpdateCode::RESOURCE_ERROR, currentVersion(), "insufficient host memory for CSR snapshot"};
    }

    error = validateSnapshot(*next);
    if (!error.empty()) {
        return {ConstraintTreeCsrUpdateCode::INVALID_ARTIFACT, currentVersion(), std::move(error)};
    }

    if (device != nullptr) {
        try {
            next->device_row_ptr_ = upload(next->row_ptr_, device);
            next->device_col_idx_ = upload(next->col_idx_, device);
            // State transitions run on the host, so next_state intentionally
            // stays in host memory. Do not spend one extra int32 per edge in VRAM.
            // clone() uses the device stream asynchronously. Do not publish the
            // snapshot until both immutable GPU mask buffers are ready.
            device->syncDeviceStream(DeviceStream::DEFAULT);
        } catch (const std::exception& e) {
            return {ConstraintTreeCsrUpdateCode::DEVICE_ERROR,
                    currentVersion(),
                    std::string("failed to upload CSR buffers: ") + e.what()};
        }
    }

    std::lock_guard<std::mutex> lock(update_mutex_);
    const auto                  current = snapshot();
    if (current && header.version < current->version()) {
        return {ConstraintTreeCsrUpdateCode::STALE_VERSION, current->version(), "a newer CSR tree is already active"};
    }
    if (current && header.version == current->version()) {
        return {ConstraintTreeCsrUpdateCode::ALREADY_CURRENT, current->version(), "CSR tree version is already active"};
    }
    const auto activated = ConstraintTreeCsrSnapshotPtr(std::move(next));
    std::atomic_store_explicit(&snapshot_, activated, std::memory_order_release);
    RTP_LLM_LOG_INFO(
        "activated CSR constraint tree version=[%llu], states=[%zu], edges=[%zu], sids=[%llu], device=[%d]",
        static_cast<unsigned long long>(header.version),
        activated->stateCount(),
        activated->edgeCount(),
        static_cast<unsigned long long>(header.sid_count),
        device != nullptr);
    return {ConstraintTreeCsrUpdateCode::UPDATED, header.version, "CSR tree activated"};
}

}  // namespace rtp_llm

#include "rtp_llm/cpp/utils/DecodeProbeTrigger.h"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace rtp_llm {
namespace {

constexpr const char* kDebugEnv       = "RTPLLM_RETROSPECTIVE_PROBE_DEBUG";
constexpr const char* kShmNameEnv     = "RTPLLM_RETROSPECTIVE_PROBE_SHM_NAME";
constexpr const char* kRankEnv        = "WORLD_RANK";
constexpr const char* kWorldSizeEnv   = "WORLD_SIZE";
constexpr const char* kDefaultShmName = "/rtpllm_retrospective_probe";

uint64_t currentTimeUs() noexcept {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

bool envEnabled() noexcept {
    const char* value = std::getenv(kDebugEnv);
    return value != nullptr
           && (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 || std::strcmp(value, "TRUE") == 0
               || std::strcmp(value, "yes") == 0 || std::strcmp(value, "on") == 0);
}

uint64_t parseUnsignedEnv(const char* name, uint64_t fallback) noexcept {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    errno              = 0;
    char*    end        = nullptr;
    uint64_t parsed     = std::strtoull(value, &end, 10);
    return errno == 0 && end != value && *end == '\0' ? parsed : fallback;
}

uint64_t requiredRankMask() noexcept {
    const uint64_t world_size = parseUnsignedEnv(kWorldSizeEnv, 1);
    if (world_size == 0) {
        return 1;
    }
    if (world_size >= 64) {
        return ~uint64_t{0};
    }
    return (uint64_t{1} << world_size) - 1;
}

bool worldRank(uint32_t& rank) noexcept {
    const uint64_t value = parseUnsignedEnv(kRankEnv, 0);
    if (value >= 64) {
        return false;
    }
    rank = static_cast<uint32_t>(value);
    return true;
}

std::string normalizedShmName(const char* name) {
    std::string normalized = name == nullptr || *name == '\0' ? kDefaultShmName : name;
    if (normalized.front() != '/') {
        normalized.insert(normalized.begin(), '/');
    }
    return normalized;
}

void logFailure(const char* operation) noexcept {
    std::fprintf(stderr, "DecodeProbeTrigger disabled after %s: %s\n", operation, std::strerror(errno));
}

bool recordExpired(const detail::DecodeProbeTriggerSharedRecord& record, uint64_t expiry_us) noexcept {
    const uint64_t timestamp = record.timestamp_us;
    const uint64_t now       = currentTimeUs();
    return timestamp != 0 && now >= timestamp && now - timestamp > expiry_us;
}

bool allRanksAcknowledged(const detail::DecodeProbeTriggerSharedRecord& record) noexcept {
    const uint64_t required = record.required_rank_mask;
    return (record.ack_rank_mask.load(std::memory_order_acquire) & required) == required;
}

bool recordIsZero(const detail::DecodeProbeTriggerSharedRecord& record) noexcept {
    const auto* bytes = reinterpret_cast<const unsigned char*>(&record);
    for (size_t index = 0; index < sizeof(record); ++index) {
        if (bytes[index] != 0) {
            return false;
        }
    }
    return true;
}

void copyString(char* destination, size_t capacity, const std::string& source) noexcept {
    if (capacity == 0) {
        return;
    }
    const size_t length = source.size() < capacity - 1 ? source.size() : capacity - 1;
    std::memcpy(destination, source.data(), length);
    destination[length] = '\0';
    if (length + 1 < capacity) {
        std::memset(destination + length + 1, 0, capacity - length - 1);
    }
}

size_t boundedLength(const char* value, size_t capacity) noexcept {
    size_t length = 0;
    while (length < capacity && value[length] != '\0') {
        ++length;
    }
    return length;
}

DecodeProbeTriggerRegistry& productionRegistry() noexcept {
    static DecodeProbeTriggerRegistry registry(std::getenv(kShmNameEnv), true);
    return registry;
}

}  // namespace

DecodeProbeTriggerRegistry::DecodeProbeTriggerRegistry(const char* shm_name, bool enabled, uint64_t expiry_us) noexcept:
    expiry_us_(expiry_us), enabled_(enabled) {
    if (!enabled_) {
        return;
    }
    if (!initialize(shm_name)) {
        disableUnlocked();
    }
}

DecodeProbeTriggerRegistry::~DecodeProbeTriggerRegistry() noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        disableUnlocked();
    } catch (...) {
        std::fprintf(stderr, "DecodeProbeTrigger teardown synchronization failed\n");
        disableUnlocked();
    }
}

bool DecodeProbeTriggerRegistry::initialize(const char* shm_name) noexcept {
    try {
        const std::string normalized_name = normalizedShmName(shm_name);
        fd_                               = shm_open(normalized_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        if (fd_ == -1 && errno == EEXIST) {
            fd_ = shm_open(normalized_name.c_str(), O_RDWR, 0600);
        }
        if (fd_ == -1) {
            logFailure("shm_open");
            return false;
        }
        if (flock(fd_, LOCK_EX) != 0) {
            logFailure("flock");
            return false;
        }

        bool initialize_record = false;
        bool valid              = false;
        struct stat status {};
        if (fstat(fd_, &status) != 0) {
            logFailure("fstat");
        } else if (status.st_size == 0) {
            if (ftruncate(fd_, sizeof(detail::DecodeProbeTriggerSharedRecord)) != 0) {
                logFailure("ftruncate");
            } else {
                initialize_record = true;
            }
        } else if (status.st_size != static_cast<off_t>(sizeof(detail::DecodeProbeTriggerSharedRecord))) {
            errno = EPROTO;
            logFailure("shared-memory size validation");
        }

        if (status.st_size == static_cast<off_t>(sizeof(detail::DecodeProbeTriggerSharedRecord)) || initialize_record) {
            void* mapping = mmap(nullptr,
                                 sizeof(detail::DecodeProbeTriggerSharedRecord),
                                 PROT_READ | PROT_WRITE,
                                 MAP_SHARED,
                                 fd_,
                                 0);
            if (mapping == MAP_FAILED) {
                logFailure("mmap");
            } else {
                record_ = static_cast<detail::DecodeProbeTriggerSharedRecord*>(mapping);
                if (!initialize_record && record_->magic == detail::kDecodeProbeTriggerRecordMagic
                    && record_->version == detail::kDecodeProbeTriggerRecordVersion) {
                    valid = true;
                } else if (initialize_record || recordIsZero(*record_)) {
                    std::memset(record_, 0, sizeof(*record_));
                    new (&record_->generation) std::atomic<uint64_t>(0);
                    new (&record_->ack_rank_mask) std::atomic<uint64_t>(0);
                    new (&record_->failure_rank_mask) std::atomic<uint64_t>(0);
                    record_->magic    = detail::kDecodeProbeTriggerRecordMagic;
                    record_->version  = detail::kDecodeProbeTriggerRecordVersion;
                    record_->observed_sequence_length = -1;
                    valid = true;
                } else {
                    errno = EPROTO;
                    logFailure("shared-memory record validation");
                }
                if (valid) {
                    const bool atomics_lock_free = record_->generation.is_lock_free()
                                                   && record_->ack_rank_mask.is_lock_free()
                                                   && record_->failure_rank_mask.is_lock_free();
                    if (!atomics_lock_free) {
                        std::fprintf(stderr, "DecodeProbeTrigger disabled: shared atomics are not lock-free\n");
                        valid = false;
                    }
                }
            }
        }
        flock(fd_, LOCK_UN);
        return valid;
    } catch (...) {
        std::fprintf(stderr, "DecodeProbeTrigger disabled after shared-memory initialization exception\n");
        return false;
    }
}

void DecodeProbeTriggerRegistry::disableUnlocked() noexcept {
    enabled_ = false;
    const bool locked = fd_ != -1 && flock(fd_, LOCK_EX) == 0;
    if (record_ != nullptr) {
        munmap(record_, sizeof(*record_));
        record_ = nullptr;
    }
    if (fd_ != -1) {
        if (locked) {
            flock(fd_, LOCK_UN);
        }
        close(fd_);
        fd_ = -1;
    }
}

bool DecodeProbeTriggerRegistry::publish(const DecodeProbeTriggerEvent& event) noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!enabled_ || record_ == nullptr) {
            return false;
        }
        if (flock(fd_, LOCK_EX) != 0) {
            logFailure("publish flock");
            disableUnlocked();
            return false;
        }
        bool published       = false;
        bool disable_mapping = false;
        try {
            if (record_->magic != detail::kDecodeProbeTriggerRecordMagic
                || record_->version != detail::kDecodeProbeTriggerRecordVersion) {
                errno = EPROTO;
                logFailure("publish record validation");
                disable_mapping = true;
            } else {
                const uint64_t existing_generation = record_->generation.load(std::memory_order_acquire);
                if (existing_generation != 0 && !allRanksAcknowledged(*record_)
                    && !recordExpired(*record_, expiry_us_)) {
                    published = false;
                } else {
                    const uint64_t generation = event.generation == 0 ? existing_generation + 1 : event.generation;
                    if (generation == 0 || (existing_generation != 0 && generation <= existing_generation)) {
                        published = false;
                    } else {
                        record_->generation.store(0, std::memory_order_release);
                        record_->timestamp_us = event.timestamp_us == 0 ? currentTimeUs() : event.timestamp_us;
                        record_->observed_sequence_length = event.observed_sequence_length;
                        copyString(record_->trace_id, sizeof(record_->trace_id), event.trace_id);
                        copyString(record_->reason, sizeof(record_->reason), event.reason);
                        record_->required_rank_mask = event.required_rank_mask;
                        record_->ack_rank_mask.store(0, std::memory_order_relaxed);
                        record_->failure_rank_mask.store(0, std::memory_order_relaxed);
                        record_->generation.store(generation, std::memory_order_release);
                        published = true;
                    }
                }
            }
        } catch (...) {
            std::fprintf(stderr, "DecodeProbeTrigger publish failed\n");
            disable_mapping = true;
        }
        flock(fd_, LOCK_UN);
        if (disable_mapping) {
            disableUnlocked();
        }
        return published;
    } catch (...) {
        std::fprintf(stderr, "DecodeProbeTrigger publish synchronization failed\n");
        return false;
    }
}

bool DecodeProbeTriggerRegistry::peek(DecodeProbeTriggerEvent& event) const noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        auto* self = const_cast<DecodeProbeTriggerRegistry*>(this);
        if (!enabled_ || record_ == nullptr) {
            return false;
        }
        if (flock(fd_, LOCK_SH) != 0) {
            logFailure("peek flock");
            self->disableUnlocked();
            return false;
        }
        bool observed        = false;
        bool disable_mapping = false;
        try {
            if (record_->magic != detail::kDecodeProbeTriggerRecordMagic
                || record_->version != detail::kDecodeProbeTriggerRecordVersion) {
                std::fprintf(stderr, "DecodeProbeTrigger disabled after peek record validation\n");
                disable_mapping = true;
            } else {
                const uint64_t generation = record_->generation.load(std::memory_order_acquire);
                if (generation != 0) {
                    DecodeProbeTriggerEvent candidate;
                    candidate.generation               = generation;
                    candidate.timestamp_us             = record_->timestamp_us;
                    candidate.observed_sequence_length = record_->observed_sequence_length;
                    candidate.trace_id.assign(record_->trace_id,
                                              boundedLength(record_->trace_id, sizeof(record_->trace_id)));
                    candidate.reason.assign(record_->reason, boundedLength(record_->reason, sizeof(record_->reason)));
                    candidate.required_rank_mask  = record_->required_rank_mask;
                    candidate.ack_rank_mask        = record_->ack_rank_mask.load(std::memory_order_acquire);
                    candidate.failure_rank_mask    = record_->failure_rank_mask.load(std::memory_order_acquire);
                    event                          = std::move(candidate);
                    observed                       = true;
                }
            }
        } catch (...) {
            std::fprintf(stderr, "DecodeProbeTrigger peek failed\n");
            disable_mapping = true;
        }
        flock(fd_, LOCK_UN);
        if (disable_mapping) {
            self->disableUnlocked();
        }
        return observed;
    } catch (...) {
        std::fprintf(stderr, "DecodeProbeTrigger peek synchronization failed\n");
        return false;
    }
}

bool DecodeProbeTriggerRegistry::acknowledge(uint64_t generation, uint32_t rank, bool failed) noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!enabled_ || record_ == nullptr || rank >= 64) {
            return false;
        }
        if (flock(fd_, LOCK_EX) != 0) {
            logFailure("acknowledge flock");
            disableUnlocked();
            return false;
        }
        bool acknowledged   = false;
        bool invalid_record = false;
        if (record_->magic != detail::kDecodeProbeTriggerRecordMagic
            || record_->version != detail::kDecodeProbeTriggerRecordVersion) {
            std::fprintf(stderr, "DecodeProbeTrigger disabled after acknowledge record validation\n");
            invalid_record = true;
        } else if (record_->generation.load(std::memory_order_acquire) == generation) {
            const uint64_t mask = uint64_t{1} << rank;
            if ((record_->required_rank_mask & mask) != 0) {
                record_->ack_rank_mask.fetch_or(mask, std::memory_order_acq_rel);
                if (failed) {
                    record_->failure_rank_mask.fetch_or(mask, std::memory_order_acq_rel);
                }
                acknowledged = true;
            }
        }
        flock(fd_, LOCK_UN);
        if (invalid_record) {
            disableUnlocked();
            return false;
        }
        return acknowledged;
    } catch (...) {
        std::fprintf(stderr, "DecodeProbeTrigger acknowledge synchronization failed\n");
        return false;
    }
}

bool DecodeProbeTriggerRegistry::enabled() const noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        return enabled_;
    } catch (...) {
        std::fprintf(stderr, "DecodeProbeTrigger enabled synchronization failed\n");
        return false;
    }
}

bool DecodeProbeTrigger::publish(const DecodeProbeTriggerEvent& event) noexcept {
    if (!envEnabled()) {
        return false;
    }
    try {
        DecodeProbeTriggerEvent configured = event;
        if (configured.required_rank_mask == 0) {
            configured.required_rank_mask = requiredRankMask();
        }
        return productionRegistry().publish(configured);
    } catch (...) {
        std::fprintf(stderr, "DecodeProbeTrigger production publish failed\n");
        return false;
    }
}

bool DecodeProbeTrigger::peek(DecodeProbeTriggerEvent& event) noexcept {
    return envEnabled() && productionRegistry().peek(event);
}

bool DecodeProbeTrigger::acknowledge(uint64_t generation, bool failed) noexcept {
    if (!envEnabled()) {
        return false;
    }
    uint32_t rank = 0;
    return worldRank(rank) && productionRegistry().acknowledge(generation, rank, failed);
}

bool DecodeProbeTrigger::enabled() noexcept {
    return envEnabled() && productionRegistry().enabled();
}

}  // namespace rtp_llm

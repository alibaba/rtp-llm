#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"

namespace rtp_llm {

// One writer per rank shares admission and disk rotation across transfer workers
// and GroupSets. Admission happens before allocating any large failure snapshot.
class CrcDumpWriter {
public:
    using Clock                              = std::chrono::steady_clock;
    using Metadata                           = std::map<std::string, std::string>;
    static constexpr uint64_t kQuotaBytes    = 2ULL * 1024 * 1024 * 1024;
    static constexpr uint64_t kMetadataBytes = 1024 * 1024;

    class Reservation {
    public:
        ~Reservation();
        Reservation(const Reservation&)            = delete;
        Reservation& operator=(const Reservation&) = delete;

    private:
        friend class CrcDumpWriter;
        Reservation(CrcDumpWriter& owner, uint64_t bytes): owner_(owner), bytes_(bytes) {}
        CrcDumpWriter& owner_;
        uint64_t       bytes_;
    };

    static std::shared_ptr<CrcDumpWriter> forRank(int64_t world_rank);
    explicit CrcDumpWriter(int64_t     world_rank,
                           std::string root        = "logs/kv_cache_crc",
                           uint64_t    quota_bytes = kQuotaBytes);
    std::unique_ptr<Reservation> reserve(size_t encoded_bytes, Clock::time_point now = Clock::now()) noexcept;
    // Returns the directory containing the evidence, or empty on failure. The
    // reservation is consumed on every exit; diagnostic I/O never escapes.
    std::string
    write(Reservation& reservation, const CrcCopyItem& item, const CrcCopyFailure& failure, Metadata metadata) noexcept;

private:
    void release(Reservation& reservation);
    void rotate(uint64_t needed_bytes);

    int64_t           world_rank_;
    std::string       rank_dir_;
    uint64_t          quota_bytes_;
    uint64_t          reserved_bytes_{0};
    std::mutex        mutex_;
    Clock::time_point window_{};
    unsigned          count_{0};
};

}  // namespace rtp_llm

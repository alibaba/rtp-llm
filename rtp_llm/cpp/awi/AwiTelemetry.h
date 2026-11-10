#pragma once

// Runtime observation journal for the AWI adaptation evaluator.
//
// Events are emitted by the engine at the point where the corresponding work
// actually happens (executor forward, graph capture, KV allocation).  The
// frontend only aggregates these lines; it never invents an event.  One JSON
// object per line is appended with a single O_APPEND write, which keeps
// concurrent ranks/processes from interleaving a record.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>

namespace rtp_llm {

struct AwiResidentSequence {
    int64_t request_id{0};
    int64_t input_tokens{0};
    int64_t kv_tokens{0};
    int64_t reserved_tokens{0};
};

class AwiTelemetry {
public:
    static AwiTelemetry& instance() {
        static AwiTelemetry telemetry;
        return telemetry;
    }

    bool enabled() const {
        return enabled_;
    }

    // A prefill chunk.  The aggregator merges chunks per request by taking the
    // earliest start and latest end, and preserves the authoritative total.
    void recordPrefill(int64_t request_id, int64_t input_tokens, int64_t start_ns, int64_t end_ns) {
        if (!enabled_) {
            return;
        }
        std::ostringstream out;
        out << "{\"kind\":\"prefill\",\"request_id\":" << request_id << ",\"input_tokens\":" << input_tokens
            << ",\"start_ns\":" << start_ns << ",\"end_ns\":" << end_ns << ",\"backend\":\"" << backend_ << "\""
            << commonFields() << "}";
        appendLine(out.str());
    }

    // One executed decode step.  The step counter is owned here so graph
    // replay (which bypasses Python) still produces exactly one event per
    // request per generated token.
    void recordDecode(int64_t request_id, bool graph_replay, double elapsed_ms) {
        if (!enabled_) {
            return;
        }
        int step = 0;
        {
            std::lock_guard<std::mutex> lock(mu_);
            step = ++decode_steps_[request_id];
        }
        std::ostringstream out;
        out << "{\"kind\":\"decode\",\"request_id\":" << request_id << ",\"step\":" << step
            << ",\"backend\":\"" << backend_ << "\",\"graph_replay\":" << (graph_replay ? "true" : "false")
            << ",\"elapsed_ms\":" << elapsed_ms << commonFields() << "}";
        appendLine(out.str());
    }

    // A single scheduler snapshot: every sequence in the batch is resident and
    // has its full reservation.  Repeated identical snapshots are coalesced.
    void recordResident(const std::vector<AwiResidentSequence>& sequences, int64_t context_tokens) {
        if (!enabled_ || sequences.empty()) {
            return;
        }
        std::ostringstream signature;
        for (const auto& seq : sequences) {
            signature << seq.request_id << ':' << seq.input_tokens << ':' << seq.reserved_tokens << ';';
        }
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (signature.str() == last_resident_signature_) {
                return;
            }
            last_resident_signature_ = signature.str();
        }
        std::ostringstream out;
        out << "{\"kind\":\"resident\",\"context_tokens\":" << context_tokens << ",\"sequences\":[";
        for (size_t i = 0; i < sequences.size(); ++i) {
            if (i) {
                out << ',';
            }
            const auto& seq = sequences[i];
            out << "{\"request_id\":" << seq.request_id << ",\"input_tokens\":" << seq.input_tokens
                << ",\"kv_tokens\":" << seq.kv_tokens << ",\"reserved_tokens\":" << seq.reserved_tokens << '}';
        }
        out << ']' << commonFields() << '}';
        appendLine(out.str());
    }

    void recordReuse(int64_t request_id, int64_t hit_tokens) {
        if (!enabled_ || hit_tokens <= 0) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (!reuse_seen_.insert(request_id).second) {
                return;
            }
        }
        std::ostringstream out;
        out << "{\"kind\":\"reuse\",\"request_id\":" << request_id << ",\"hit_tokens\":" << hit_tokens
            << commonFields() << '}';
        appendLine(out.str());
    }

    void recordCapture(int graph_key) {
        if (!enabled_) {
            return;
        }
        std::ostringstream out;
        out << "{\"kind\":\"capture\",\"backend\":\"" << backend_ << "\",\"graph_key\":" << graph_key
            << commonFields() << '}';
        appendLine(out.str());
    }

    // Emitted only from the real KV/admission failure path.  The barrier file
    // carries the operation-scoped requested concurrency/context.
    void recordCapacityFailure(const char* failure_kind) {
        if (!enabled_) {
            return;
        }
        std::string operation_id;
        int64_t     concurrency    = 0;
        int64_t     context_tokens = 0;
        readBarrier(operation_id, concurrency, context_tokens);
        std::ostringstream out;
        out << "{\"kind\":\"capacity_failure\",\"operation_id\":\"" << escape(operation_id)
            << "\",\"concurrency\":" << concurrency << ",\"context_tokens\":" << context_tokens
            << ",\"failure_kind\":\"" << escape(failure_kind) << "\"" << commonFields() << '}';
        appendLine(out.str());
    }

    static void setThreadGraphReplay(bool value) {
        thread_graph_replay() = value;
    }

    static bool threadGraphReplay() {
        return thread_graph_replay();
    }

private:
    AwiTelemetry() {
        const char* dir = std::getenv("AWI_TELEMETRY_DIR");
        dir_           = dir && *dir ? dir : "/tmp/awi-telemetry";
        const char* op = std::getenv("AWI_OPERATION_ID");
        operation_id_  = op && *op ? op : "";
        const char* rank = std::getenv("AWI_RANK");
        if (!rank) {
            rank = std::getenv("WORLD_RANK");
        }
        rank_ = rank && *rank ? std::atoi(rank) : 0;
        const char* backend = std::getenv("AWI_GRAPH_BACKEND");
        if (backend && *backend) {
            backend_ = backend;
        } else {
#if defined(USING_ROCM) && USING_ROCM
            backend_ = "hip";
#else
            backend_ = "cuda";
#endif
        }
        enabled_ = (op && *op) || (dir && *dir);
        if (!enabled_) {
            return;
        }
        ::mkdir(dir_.c_str(), 0700);
        char path[1024];
        std::snprintf(path, sizeof(path), "%s/rank-%d-%d.jsonl", dir_.c_str(), rank_, static_cast<int>(::getpid()));
        path_ = path;
    }

    static bool& thread_graph_replay() {
        static thread_local bool value = false;
        return value;
    }

    std::string commonFields() const {
        std::ostringstream out;
        out << ",\"rank\":" << rank_ << ",\"pid\":" << static_cast<int>(::getpid())
            << ",\"recorded_ns\":" << nowNs();
        return out.str();
    }

    static int64_t nowNs() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
            .count();
    }

    static std::string escape(const std::string& raw) {
        std::string out;
        out.reserve(raw.size() + 8);
        for (char ch : raw) {
            switch (ch) {
                case '"':
                    out += "\\\"";
                    break;
                case '\\':
                    out += "\\\\";
                    break;
                case '\n':
                    out += "\\n";
                    break;
                case '\r':
                    out += "\\r";
                    break;
                case '\t':
                    out += "\\t";
                    break;
                default:
                    out += ch;
            }
        }
        return out;
    }

    void appendLine(const std::string& line) {
        int fd = ::open(path_.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0600);
        if (fd < 0) {
            return;
        }
        std::string payload = line;
        payload.push_back('\n');
        size_t written = 0;
        while (written < payload.size()) {
            ssize_t n = ::write(fd, payload.data() + written, payload.size() - written);
            if (n <= 0) {
                break;
            }
            written += static_cast<size_t>(n);
        }
        ::close(fd);
    }

    void readBarrier(std::string& operation_id, int64_t& concurrency, int64_t& context_tokens) const {
        std::ifstream stream(dir_ + "/barrier.json");
        if (!stream) {
            return;
        }
        std::string line;
        std::string last;
        while (std::getline(stream, line)) {
            if (!line.empty()) {
                last = line;
            }
        }
        auto number_after = [&](const char* key) -> int64_t {
            const std::string needle = std::string("\"") + key + "\":";
            auto              pos    = last.find(needle);
            if (pos == std::string::npos) {
                return 0;
            }
            pos += needle.size();
            return std::strtoll(last.c_str() + pos, nullptr, 10);
        };
        auto string_after = [&](const char* key) -> std::string {
            const std::string needle = std::string("\"") + key + "\":\"";
            auto              pos    = last.find(needle);
            if (pos == std::string::npos) {
                return "";
            }
            pos += needle.size();
            auto end = last.find('"', pos);
            if (end == std::string::npos) {
                return "";
            }
            return last.substr(pos, end - pos);
        };
        operation_id   = string_after("operation_id");
        concurrency    = number_after("concurrency");
        context_tokens = number_after("context_tokens");
    }

    bool                                      enabled_{false};
    std::string                               dir_;
    std::string                               path_;
    std::string                               operation_id_;
    std::string                               backend_{"cuda"};
    int                                       rank_{0};
    std::mutex                                mu_;
    std::unordered_map<int64_t, int>          decode_steps_;
    std::unordered_set<int64_t>               reuse_seen_;
    std::string                               last_resident_signature_;
};

}  // namespace rtp_llm

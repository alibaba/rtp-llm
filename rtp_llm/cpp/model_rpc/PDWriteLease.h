#pragma once

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace rtp_llm {

// Tokens are issued once per worker incarnation. Absence of an issued token
// proves that it was never claimed, or that its last physical writer stopped.
class PDWriteLeaseRegistry: public std::enable_shared_from_this<PDWriteLeaseRegistry> {
public:
    using Clock = std::chrono::steady_clock;

    PDWriteLeaseRegistry();
    PDWriteLeasePB        prepare(const PreparePDWriteLeaseRequestPB& request);
    std::shared_ptr<void> claim(const BroadcastLoadRequestPB& request);
    void                  registerRecovery(const PDWriteLeasePB& lease, std::function<void()> recover);
    bool                  seal(const PDWriteLeasePB& lease, bool& stopped);
    size_t                activeWriters();
    void                  closeAdmission();

private:
    using Expiry     = std::multimap<Clock::time_point, uint64_t>;
    using Recoveries = std::list<std::function<void()>>;
    struct Entry {
        int64_t          request_id;
        std::string      request_key;
        int64_t          dp_rank;
        Expiry::iterator expiry;  // end() after the single permitted claim.
        bool             abandoned = false;
        Recoveries       recoveries;
    };
    struct Writer {
        std::shared_ptr<PDWriteLeaseRegistry> owner;
        uint64_t                              token = 0;
        ~Writer();
    };
    void                                startRecovery(uint64_t token, Recoveries& recoveries);
    const std::string                   epoch_;
    std::mutex                          mutex_;
    uint64_t                            issued_ = 0;
    bool                                closed_ = false;
    Expiry                              expiry_;
    std::unordered_map<uint64_t, Entry> entries_;
};

// Preallocate settlement ownership before any address can cross the wire.
// A lost reply moves these existing list nodes to one process-wide retry worker.
struct PDWriteLeasePeer {
    std::shared_ptr<RpcService::Stub> stub;
    PDWriteLeasePB                    lease;
};
struct PDWriteLeaseDrain {
    PDWriteLeasePeer                      peer;
    std::shared_ptr<void>                 destination;
    std::chrono::steady_clock::time_point next_poll{};
};
class PDWriteLeaseRecovery {
public:
    static PDWriteLeaseRecovery& instance();
    void                         adopt(std::list<PDWriteLeaseDrain>& pending) noexcept;

private:
    PDWriteLeaseRecovery();
    void                         run();
    std::mutex                   mutex_;
    std::condition_variable      changed_;
    std::list<PDWriteLeaseDrain> pending_;
};

bool         samePDWriteLease(const PDWriteLeasePB& expected, const PDWriteLeasePB& actual);
grpc::Status preparePDWriteLeases(std::vector<PDWriteLeasePeer>&        peers,
                                  const PreparePDWriteLeaseRequestPB&   request,
                                  std::chrono::system_clock::time_point deadline,
                                  const std::function<bool()>&          cancelled);

}  // namespace rtp_llm

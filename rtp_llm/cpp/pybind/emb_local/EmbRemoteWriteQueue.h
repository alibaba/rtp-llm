#pragma once

#include "rtp_llm/cpp/pybind/EmbRegisteredWrite.h"
#include "rtp_llm/cpp/pybind/emb_local/EmbLocalTransfer.h"

#include <condition_variable>
#include <deque>
#include <list>
#include <shared_mutex>
#include <thread>
#include <unordered_set>

namespace rtp_llm::emb_local {

// Uses the existing SDK client and MR. The transport stops and joins this
// queue before closing the SDK or releasing its capsule and allocator.
class RemoteWriteQueue {
public:
    struct Stats {
        int64_t pending          = 0;
        int64_t bytes            = 0;
        int64_t success          = 0;
        int64_t error            = 0;
        int64_t skipped          = 0;
        int64_t inflight_failure = 0;
    };

    RemoteWriteQueue(std::shared_ptr<EmbRegisteredArena> arena,
                     at::Tensor                          root,
                     KvcmObjectNativeApiV2               api,
                     int64_t                             max_object_bytes,
                     int64_t                             max_pending,
                     int64_t                             max_bytes,
                     int64_t                             timeout_ms,
                     int                                 workers,
                     bool                                request_scoped);
    ~RemoteWriteQueue();

    // Called after the exact local generation has left the index, with a
    // reference to its block until copy-out completes. No SDK I/O here.
    void  submit(const std::string&                     key,
                 const Audit&                           audit,
                 const std::shared_ptr<const Resident>& record,
                 std::optional<c10::cuda::CUDAStream>   stream);
    void  submitEncoded(const std::string&                                             key,
                        const std::string&                                             trace,
                        const Audit&                                                   audit,
                        const Tree&                                                    tree,
                        const std::vector<at::Tensor>&                                 inputs,
                        const std::vector<std::pair<c10::DeviceIndex, c10::StreamId>>& producers);
    bool  reserveExternal(int64_t bytes, bool check_task_limit);
    void  releaseExternal(int64_t bytes);
    Stats stats() const;
    void  close();

    static std::shared_timed_mutex& graphGate();

private:
    struct Job {
        std::string                                        key;
        std::string                                        trace;
        WriteClock::time_point                             deadline;
        int64_t                                            bytes          = 0;
        int64_t                                            charge         = 0;
        int64_t                                            payload_offset = 0;
        std::string                                        prefix;
        std::vector<EmbTensorInput>                        tensors;
        std::vector<std::unique_ptr<EmbCudaSnapshotReady>> events;
    };

    template<typename Snapshot>
    void enqueue(const std::string&        key,
                 const std::string&        trace,
                 const Audit&              audit,
                 const Tree&               tree,
                 const std::vector<Slice>& slices,
                 Snapshot                  snapshot);

    void           run();
    EmbWriteResult writeStaged(Job& job, const std::string& trace);
    void           finish(const std::string& key, int64_t bytes, const EmbWriteResult& result);

    const std::shared_ptr<EmbRegisteredArena> arena_;
    at::Tensor                                root_;
    const KvcmObjectNativeApiV2               api_;
    const int64_t                             max_object_bytes_;
    const int64_t                             max_pending_;
    const int64_t                             max_bytes_;
    const std::chrono::milliseconds           timeout_;
    const bool                                request_scoped_;
    std::mutex                                close_mutex_;
    mutable std::mutex                        mutex_;
    std::condition_variable                   ready_;
    std::deque<Job>                           jobs_;
    std::unordered_set<std::string>           keys_;
    std::vector<std::thread>                  workers_;
    // Buffers that the SDK might still access survive until process exit.
    // Track the candidate before the SDK call so failure handling cannot allocate.
    std::list<at::Tensor> staging_buffers_;
    Stats                 stats_;
    int64_t               external_bytes_ = 0;
    int64_t               external_tasks_ = 0;
    bool                  closed_         = false;
};

}  // namespace rtp_llm::emb_local

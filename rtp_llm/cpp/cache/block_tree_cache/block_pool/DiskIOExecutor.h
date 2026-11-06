#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockIO.h"

namespace rtp_llm {

// Engine-wide budgets. Queue sizes count worker tasks across all backing files.
struct DiskIOExecutorConfig {
    size_t read_thread_count{16};
    size_t write_thread_count{16};
    size_t read_queue_size{20000};
    size_t write_queue_size{20000};
    size_t max_batch_blocks{4096};
    size_t max_batch_bytes{256 * 1024 * 1024};

    // Reads fan out up to read_thread_count tasks per batch; writes use one task per batch.
    // Converts queued batch budgets to worker-task budgets; rejects invalid/overflowing capacities.
    bool setQueueSizesFromBatchCapacity(size_t read_batches, size_t write_batches, size_t max_blocks_per_batch);
    bool valid() const;
};

// One read pool and one write pool shared by all files of a transfer engine.
// Workers start lazily; synchronous file I/O does not require an executor.
class DiskIOExecutor {
public:
    using Callback = std::function<void(DiskBlockIOStatus)>;

    explicit DiskIOExecutor(DiskIOExecutorConfig config = {});
    ~DiskIOExecutor();
    DiskIOExecutor(const DiskIOExecutor&)            = delete;
    DiskIOExecutor& operator=(const DiskIOExecutor&) = delete;

    const DiskIOExecutorConfig& config() const;

    // OK: exactly one worker callback, possibly before return. IO_ERROR: no I/O or callback.
    // Owns the request vector; the caller keeps io open and buffers alive until all
    // accepted I/O and callbacks have returned, and orders conflicting accesses.
    // Empty batches complete asynchronously; failed batches may be partly applied.
    // Reads share blocks across workers; each write batch is sorted by file offset
    // and executed by one worker.
    DiskBlockIOStatus asyncRead(DiskBlockIO& io, std::vector<DiskRead> reads, Callback callback);
    DiskBlockIOStatus asyncWrite(DiskBlockIO& io, std::vector<DiskWrite> writes, Callback callback);

    // Close both queues and join all accepted I/O and callbacks. The external
    // lifecycle owner drains its WorkLoops first; never call from a worker/callback.
    // Lifecycle calls are serialized by the owner; sequential repeats are safe.
    void stop();

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace rtp_llm

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskIOExecutor.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <mutex>
#include <type_traits>
#include <utility>

#include "autil/ThreadPool.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool DiskIOExecutorConfig::setQueueSizesFromBatchCapacity(size_t read_batches,
                                                          size_t write_batches,
                                                          size_t max_blocks_per_batch) {
    const size_t read_tasks = std::min(max_blocks_per_batch, read_thread_count);
    // autil::CircularQueue requires capacity < INT32_MAX.
    constexpr size_t max_queue_size = std::numeric_limits<int32_t>::max() - 1;
    if (read_batches == 0 || write_batches == 0 || read_tasks == 0 || write_thread_count == 0
        || read_batches > max_queue_size / read_tasks || write_batches > max_queue_size) {
        return false;
    }
    read_queue_size  = read_batches * read_tasks;
    write_queue_size = write_batches;
    return true;
}

bool DiskIOExecutorConfig::valid() const {
    constexpr size_t max_int = std::numeric_limits<int32_t>::max();
    return read_thread_count > 0 && read_thread_count <= max_int && write_thread_count > 0
           && write_thread_count <= max_int && read_queue_size > 0 && read_queue_size < max_int && write_queue_size > 0
           && write_queue_size < max_int && max_batch_blocks > 0 && max_batch_bytes > 0;
}

struct DiskIOExecutor::State {
    explicit State(DiskIOExecutorConfig config): config(config) {}

    template<typename Request>
    struct Batch {
        Batch(std::vector<Request> requests, Callback callback):
            requests(std::move(requests)), callback(std::move(callback)) {}

        void run(DiskBlockIO& io) {
            try {
                process(io);
            } catch (const std::exception& e) {
                fail(DiskBlockIOStatus::IO_ERROR);
                RTP_LLM_LOG_ERROR("async disk worker threw: %s", e.what());
            } catch (...) {
                fail(DiskBlockIOStatus::IO_ERROR);
                RTP_LLM_LOG_ERROR("async disk worker threw an unknown exception");
            }
            finish();
        }

        void process(DiskBlockIO& io) {
            if constexpr (std::is_same_v<Request, DiskWrite>) {
                // One worker owns the whole write batch. Keep buffers paired with their file offsets.
                std::sort(requests.begin(), requests.end(), [](const DiskWrite& lhs, const DiskWrite& rhs) {
                    return lhs.offset < rhs.offset;
                });
                const auto result = io.write(requests);
                if (result != DiskBlockIOStatus::OK) {
                    fail(result);
                }
            } else {
                while (true) {
                    size_t index;
                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        if (status != DiskBlockIOStatus::OK || next == requests.size()) {
                            break;
                        }
                        index = next++;
                    }
                    const auto& request = requests[index];
                    const auto  result  = io.read(request.offset, request.buffer, request.bytes);
                    if (result != DiskBlockIOStatus::OK) {
                        fail(result);
                        break;
                    }
                }
            }
        }

        void fail(DiskBlockIOStatus error) {
            std::lock_guard<std::mutex> lock(mutex);
            if (status == DiskBlockIOStatus::OK) {
                status = error;
            }
        }

        void finish() {
            Callback          done;
            DiskBlockIOStatus result;
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (--workers != 0) {
                    return;
                }
                result = status;
                done   = std::move(callback);
            }
            try {
                done(result);
            } catch (const std::exception& e) {
                RTP_LLM_LOG_ERROR("async disk callback threw: %s", e.what());
            } catch (...) {
                RTP_LLM_LOG_ERROR("async disk callback threw an unknown exception");
            }
        }

        std::vector<Request> requests;
        Callback             callback;
        std::mutex           mutex;
        size_t               next{0};
        size_t               workers{0};
        DiskBlockIOStatus    status{DiskBlockIOStatus::OK};
    };

    template<typename Request>
    DiskBlockIOStatus submit(DiskBlockIO& io, std::vector<Request> requests, Callback callback) {
        if (!callback || requests.size() > config.max_batch_blocks) {
            RTP_LLM_LOG_WARNING("async disk submission rejected: empty callback or block limit");
            return DiskBlockIOStatus::IO_ERROR;
        }
        size_t total_bytes = 0;
        for (const auto& request : requests) {
            // Subtraction also prevents size_t overflow.
            if (request.bytes > config.max_batch_bytes - total_bytes) {
                RTP_LLM_LOG_WARNING("async disk submission rejected: byte limit");
                return DiskBlockIOStatus::IO_ERROR;
            }
            total_bytes += request.bytes;
        }

        auto                        batch = std::make_shared<Batch<Request>>(std::move(requests), std::move(callback));
        size_t                      accepted = 0;
        std::lock_guard<std::mutex> admission_lock(admission_mutex);
        auto*                       pool = std::is_same_v<Request, DiskRead> ? read_pool.get() : write_pool.get();
        if (stopped) {
            RTP_LLM_LOG_WARNING("async disk submission rejected: executor is closed");
            return DiskBlockIOStatus::IO_ERROR;
        }
        if (pool == nullptr) {
            try {
                if (!config.valid())
                    return DiskBlockIOStatus::IO_ERROR;
                auto reads  = std::make_unique<autil::ThreadPool>(config.read_thread_count, config.read_queue_size);
                auto writes = std::make_unique<autil::ThreadPool>(config.write_thread_count, config.write_queue_size);
                if (!reads->start("btc_disk_read") || !writes->start("btc_disk_write")) {
                    RTP_LLM_LOG_ERROR("failed to start async disk executor");
                    return DiskBlockIOStatus::IO_ERROR;
                }
                read_pool  = std::move(reads);
                write_pool = std::move(writes);
                pool       = std::is_same_v<Request, DiskRead> ? read_pool.get() : write_pool.get();
            } catch (const std::exception& e) {
                RTP_LLM_LOG_ERROR("starting async disk executor threw: %s", e.what());
                return DiskBlockIOStatus::IO_ERROR;
            }
        }
        // Register all participants before any worker can finish.
        std::lock_guard<std::mutex> batch_lock(batch->mutex);

        const size_t task_count = std::is_same_v<Request, DiskRead> ?
                                      std::max(size_t{1}, std::min(batch->requests.size(), pool->getThreadNum())) :
                                      1;
        for (size_t i = 0; i < task_count; ++i) {
            try {
                const auto error = pool->pushTask([batch, &io] { batch->run(io); }, false, false);
                if (error != autil::ThreadPool::ERROR_NONE) {
                    RTP_LLM_LOG_DEBUG("async disk enqueue rejected: error=%d, accepted=%zu", int(error), accepted);
                    break;
                }
            } catch (...) {
                // Earlier participants already own completion, including after allocation failure.
                RTP_LLM_LOG_ERROR("async disk enqueue threw, accepted=%zu", accepted);
                break;
            }
            ++batch->workers;
            ++accepted;
        }
        // Any accepted participant owns the whole batch, even after later enqueue failures.
        return accepted == 0 ? DiskBlockIOStatus::IO_ERROR : DiskBlockIOStatus::OK;
    }

    const DiskIOExecutorConfig         config;
    std::mutex                         admission_mutex;
    std::unique_ptr<autil::ThreadPool> read_pool;
    std::unique_ptr<autil::ThreadPool> write_pool;
    bool                               stopped{false};
};

DiskIOExecutor::DiskIOExecutor(DiskIOExecutorConfig config): state_(std::make_unique<State>(config)) {}

DiskIOExecutor::~DiskIOExecutor() {
    stop();
}

const DiskIOExecutorConfig& DiskIOExecutor::config() const {
    return state_->config;
}

DiskBlockIOStatus DiskIOExecutor::asyncRead(DiskBlockIO& io, std::vector<DiskRead> reads, Callback callback) {
    return state_->submit(io, std::move(reads), std::move(callback));
}

DiskBlockIOStatus DiskIOExecutor::asyncWrite(DiskBlockIO& io, std::vector<DiskWrite> writes, Callback callback) {
    return state_->submit(io, std::move(writes), std::move(callback));
}

void DiskIOExecutor::stop() {
    auto&                              state = *state_;
    std::unique_ptr<autil::ThreadPool> reads;
    std::unique_ptr<autil::ThreadPool> writes;
    {
        std::lock_guard<std::mutex> admission_lock(state.admission_mutex);
        state.stopped = true;
        reads         = std::move(state.read_pool);
        writes        = std::move(state.write_pool);
    }
    // Close admission in both directions before draining.
    if (reads)
        reads->stop(autil::ThreadPool::STOP_AFTER_QUEUE_EMPTY);
    if (writes)
        writes->stop(autil::ThreadPool::STOP_AFTER_QUEUE_EMPTY);
}

}  // namespace rtp_llm

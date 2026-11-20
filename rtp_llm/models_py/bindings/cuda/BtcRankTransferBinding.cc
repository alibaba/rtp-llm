#include <cuda_runtime_api.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskIOExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferWorkLoop.h"

namespace rtp_llm {
void writeBtcTransferTrace(const char* format, ...) noexcept;
}

extern "C" {
const char* btc_transfer_last_error() noexcept;
int btc_transfer_submit(int,
                        int,
                        int,
                        uint64_t,
                        uintptr_t,
                        const uintptr_t*,
                        const uintptr_t*,
                        const size_t*,
                        size_t,
                        void**) noexcept;
int btc_transfer_query(void*) noexcept;
void btc_transfer_destroy(void*) noexcept;
}

namespace {

using namespace rtp_llm;

constexpr size_t kMaxBatchPages = 256;
constexpr size_t kQueueBatches = 1024;
constexpr auto kQueueTimeout = std::chrono::minutes(5);
constexpr int64_t kHostPage = -1;
thread_local char last_error[512];

int fail(const char* message) noexcept {
    std::snprintf(last_error, sizeof(last_error), "%s", message);
    return -1;
}

template<typename Call>
int guard(Call&& call) noexcept {
    try {
        return call();
    } catch (const std::exception& error) {
        return fail(error.what());
    } catch (...) {
        return fail("unknown BTC rank transfer error");
    }
}

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

class CudaDevice {
public:
    explicit CudaDevice(int device) {
        checkCuda(cudaGetDevice(&previous_), "get CUDA device");
        changed_ = previous_ != device;
        if (changed_)
            checkCuda(cudaSetDevice(device), "select CUDA device");
    }

    ~CudaDevice() {
        if (changed_)
            (void)cudaSetDevice(previous_);
    }

private:
    int previous_{-1};
    bool changed_{false};
};

class CudaStream {
public:
    explicit CudaStream(int device): device_(device) {
        CudaDevice scope(device_);
        checkCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create BTC WorkLoop stream");
    }

    ~CudaStream() {
        try {
            CudaDevice scope(device_);
            (void)cudaStreamDestroy(stream_);
        } catch (...) {
            std::abort();
        }
    }

    cudaStream_t get() const {
        return stream_;
    }

private:
    int device_;
    cudaStream_t stream_{};
};

class CudaDependency {
public:
    CudaDependency(int device, uintptr_t caller_stream): device_(device) {
        CudaDevice scope(device_);
        checkCuda(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming), "create BTC producer event");
        const auto status = cudaEventRecord(event_, reinterpret_cast<cudaStream_t>(caller_stream));
        if (status != cudaSuccess) {
            (void)cudaEventDestroy(event_);
            checkCuda(status, "record BTC producer event");
        }
    }

    ~CudaDependency() {
        try {
            CudaDevice scope(device_);
            (void)cudaEventDestroy(event_);
        } catch (...) {
            std::abort();
        }
    }

    cudaEvent_t get() const {
        return event_;
    }

private:
    int device_;
    cudaEvent_t event_{};
};

struct Span {
    size_t offset;
    uintptr_t gpu;
    size_t bytes;
};

struct Row {
    uintptr_t host;
    int64_t disk;
    std::vector<Span> spans;
};

struct CudaBatch {
    int device;
    int rank;
    bool load;
    uint64_t job;
    cudaStream_t stream;
    std::shared_ptr<CudaDependency> dependency;
    std::vector<std::vector<Span>> spans;
    std::unique_ptr<void, decltype(&btc_transfer_destroy)> copy{nullptr, btc_transfer_destroy};

    void submit(const std::vector<HostBufferView>& host_rows) {
        std::vector<uintptr_t> source;
        std::vector<uintptr_t> destination;
        std::vector<size_t> lengths;
        for (size_t row = 0; row < spans.size(); ++row) {
            for (const auto& span : spans[row]) {
                const auto host = reinterpret_cast<uintptr_t>(host_rows[row].base) + span.offset;
                source.push_back(load ? host : span.gpu);
                destination.push_back(load ? span.gpu : host);
                lengths.push_back(span.bytes);
            }
        }
        CudaDevice scope(device);
        checkCuda(cudaStreamWaitEvent(stream, dependency->get(), 0), "wait for BTC producer event");
        void* native_ticket = nullptr;
        if (btc_transfer_submit(device,
                                rank,
                                load ? 0 : 1,
                                job,
                                reinterpret_cast<uintptr_t>(stream),
                                source.data(),
                                destination.data(),
                                lengths.data(),
                                lengths.size(),
                                &native_ticket) < 0) {
            throw std::runtime_error(btc_transfer_last_error());
        }
        copy.reset(native_ticket);
    }

    std::optional<TransferResult> poll() {
        const int status = btc_transfer_query(copy.get());
        if (status == 0)
            return std::nullopt;
        if (status < 0)
            return TransferResult::executionException(true, btc_transfer_last_error());
        return TransferResult::emptyAggregate(spans.size());
    }
};

class RankRuntime;

class DiskPins {
public:
    DiskPins(RankRuntime& runtime, bool writing, uint64_t job): runtime_(runtime), writing_(writing), job_(job) {}
    ~DiskPins();
    void settle(bool success) noexcept;

    const std::vector<std::pair<int64_t, BlockIdxType>>& slots() const {
        return slots_;
    }

private:
    friend class RankRuntime;
    RankRuntime& runtime_;
    std::vector<std::pair<int64_t, BlockIdxType>> slots_;
    bool writing_;
    bool settled_{false};
    uint64_t job_;
};

class RankRuntime {
public:
    RankRuntime(int device,
                int rank,
                const char* disk_dir,
                size_t disk_slots,
                size_t row_bytes,
                size_t staging_slots):
        device(device),
        rank(rank),
        row_bytes(row_bytes),
        batch_disk_pages(std::min(staging_slots, kMaxBatchPages)),
        load_stream(device),
        evict_stream(device) {
        const auto* trace_setting = std::getenv("RTP_BTC_TRANSFER_TRACE");
        trace_ = trace_setting && std::strcmp(trace_setting, "1") == 0;
        std::shared_ptr<HostStagingBlockPool> staging;
        if (disk_slots) {
            CudaDevice scope(device);
            auto config = std::make_shared<DiskBlockPoolConfig>();
            config->pool_type = BlockPoolType::DISK;
            config->pool_name = "vllm";
            config->work_dir = disk_dir;
            config->world_rank = rank;
            config->local_rank = device;
            config->stride_bytes = row_bytes;
            config->payload_bytes = row_bytes;
            config->disk_size_bytes = (disk_slots + 1) * row_bytes;
            config->buffered_io = false;
            disk_ = std::make_unique<DiskBlockPool>(std::move(config));
            if (!disk_->init())
                throw std::runtime_error("initialize BTC rank Disk pool");
            DiskIOExecutorConfig io;
            io.read_thread_count = 4;
            io.write_thread_count = 1;
            if (!io.setQueueSizesFromBatchCapacity(kQueueBatches, kQueueBatches, kMaxBatchPages))
                throw std::runtime_error("invalid BTC WorkLoop I/O budget");
            io_ = std::make_unique<DiskIOExecutor>(io);
            io_->start();
            staging = std::make_shared<HostStagingBlockPool>(staging_slots, row_bytes);
        }
        load_loop = createLoop(TransferOperation::LOAD, std::move(staging));
        evict_loop = createLoop(TransferOperation::EVICT, nullptr);
        load_loop->start();
        evict_loop->start();
    }

    DiskBlockPool* disk() const {
        return disk_.get();
    }

    std::shared_ptr<DiskPins> pinDisk(const std::vector<Row>& pages, bool writing, uint64_t job) {
        if (!disk_)
            throw std::invalid_argument("BTC rank has no Disk pool");
        auto pins = std::make_shared<DiskPins>(*this, writing, job);
        pins->slots_.reserve(pages.size());
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& page : pages) {
            auto mapped = disk_slots_.find(page.disk);
            if (writing) {
                if (mapped != disk_slots_.end())
                    throw std::invalid_argument("BTC logical Disk slot already exists on this rank");
                auto inserted = disk_slots_.emplace(page.disk, DiskSlot{}).first;
                const auto physical = disk_->malloc();
                if (!physical) {
                    disk_slots_.erase(inserted);
                    throw std::runtime_error("BTC rank Disk pool exhausted");
                }
                disk_->incTreeRef(*physical, BlockTreeRefType::CACHE);
                inserted->second = {*physical, 0, true};
                pins->slots_.emplace_back(page.disk, *physical);
            } else {
                if (mapped == disk_slots_.end() || mapped->second.writing)
                    throw std::invalid_argument("BTC logical Disk slot is not readable on this rank");
                ++mapped->second.readers;
                pins->slots_.emplace_back(page.disk, mapped->second.physical);
            }
        }
        return pins;
    }

    void settleDisk(const DiskPins& pins, bool success) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& [id, physical] : pins.slots()) {
            auto slot = disk_slots_.find(id);
            if (pins.writing_) {
                if (success) {
                    slot->second.writing = false;
                } else {
                    releaseLocked(slot, pins.job_);
                }
            } else {
                --slot->second.readers;
            }
        }
    }

    void releaseDisk(size_t count, const int64_t* ids) {
        if (!disk_ || (count && !ids))
            throw std::invalid_argument("invalid BTC rank Disk release");
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t index = 0; index < count; ++index) {
            if (ids[index] < 0)
                throw std::invalid_argument("invalid BTC logical Disk id");
            auto slot = disk_slots_.find(ids[index]);
            if (slot != disk_slots_.end() && (slot->second.readers || slot->second.writing))
                throw std::runtime_error("BTC logical Disk slot still has rank-local I/O in flight");
        }
        for (size_t index = 0; index < count; ++index) {
            auto slot = disk_slots_.find(ids[index]);
            if (slot != disk_slots_.end())
                releaseLocked(slot, 0);
        }
    }

    void observe(uint64_t job,
                 size_t segment,
                 const DiskPins* pins,
                 size_t pages,
                 size_t spans,
                 size_t dma_bytes,
                 Tier src,
                 Tier dst,
                 const char* phase,
                 bool ok) const noexcept {
        if (!trace_)
            return;
        const bool read = src == Tier::DISK;
        const bool write = dst == Tier::DISK;
        const char* stage = read ? "DISK_TO_HOST" :
                            write ? "HOST_TO_DISK" :
                            src == Tier::DEVICE ? "DEVICE_TO_HOST" : "HOST_TO_DEVICE";
        const size_t bytes = (read || write) ? row_bytes * pages : dma_bytes;
        writeBtcTransferTrace(
                     "BTC_RANK_TRANSFER rank=%d job=%llu stage=%s phase=%s status=%d pages=%zu spans=%zu bytes=%zu "
                     "backend=workloop pid=%d segment=%zu\n",
                     rank,
                     static_cast<unsigned long long>(job),
                     stage,
                     phase,
                     ok ? 0 : -1,
                     pages,
                     (read || write) ? pages : spans,
                     bytes,
                     static_cast<int>(getpid()),
                     segment);
        if ((read || write) && std::strcmp(phase, "begin") == 0) {
            for (const auto& [id, physical] : pins->slots())
                logSlot(job, id, physical, read ? "read" : "write");
        }
    }

    const int device;
    const int rank;
    const size_t row_bytes;
    const size_t batch_disk_pages;
    const CudaStream load_stream;
    const CudaStream evict_stream;

private:
    std::unique_ptr<TransferWorkLoop> createLoop(TransferOperation operation,
                                                 std::shared_ptr<HostStagingBlockPool> staging) {
#ifdef RTP_BTC_RANK_STANDALONE
        return std::make_unique<TransferWorkLoop>(
            io_.get(), operation, kMaxBatchPages, kQueueBatches, std::move(staging), rank);
#else
        return std::make_unique<TransferWorkLoop>(std::vector<GroupSetPtr>{},
                                                  nullptr,
                                                  io_.get(),
                                                  operation,
                                                  kMaxBatchPages,
                                                  kQueueBatches,
                                                  std::move(staging),
                                                  nullptr,
                                                  rank,
                                                  0);
#endif
    }

    struct DiskSlot {
        BlockIdxType physical{0};
        size_t readers{0};
        bool writing{true};
    };
    using Slots = std::map<int64_t, DiskSlot>;

    void releaseLocked(Slots::iterator slot, uint64_t job) noexcept {
        logSlot(job, slot->first, slot->second.physical, "release");
        disk_->decTreeRef(slot->second.physical, BlockTreeRefType::CACHE);
        disk_slots_.erase(slot);
    }

    void logSlot(uint64_t job, int64_t logical, BlockIdxType physical, const char* action) const noexcept {
        if (!trace_)
            return;
        writeBtcTransferTrace(
                     "BTC_RANK_DISK_SLOT rank=%d job=%llu logical=%lld physical=%d action=%s pid=%d\n",
                     rank,
                     static_cast<unsigned long long>(job),
                     static_cast<long long>(logical),
                     static_cast<int>(physical),
                     action,
                     static_cast<int>(getpid()));
    }

    bool trace_{false};
    std::mutex mutex_;
    Slots disk_slots_;
    std::unique_ptr<DiskBlockPool> disk_;
    std::unique_ptr<DiskIOExecutor> io_;

public:
    // Declared last so both owners drain before any backing resources are destroyed.
    std::unique_ptr<TransferWorkLoop> load_loop;
    std::unique_ptr<TransferWorkLoop> evict_loop;
};

DiskPins::~DiskPins() {
    settle(false);
}

void DiskPins::settle(bool success) noexcept {
    if (!settled_) {
        runtime_.settleDisk(*this, success);
        settled_ = true;
    }
}

using RankHandle = std::shared_ptr<RankRuntime>;

class Ticket {
public:
    explicit Ticket(RankHandle rank): rank(std::move(rank)) {}

    ~Ticket() {
        try {
            (void)complete(true);
        } catch (...) {
            std::abort();
        }
    }

    struct Batch {
        std::shared_ptr<DiskPins> pins;
        std::shared_ptr<TransferAsyncContext> context;
    };

    int complete(bool wait) {
        for (auto& batch : batches) {
            if (!batch.context)
                continue;
            if (wait)
                batch.context->waitDone();
            else if (!batch.context->done())
                return 0;
        }
        for (auto& batch : batches) {
            if (!batch.context)
                continue;
            const auto result = batch.context->result();
            if (!result.buffer_safe)
                std::abort();
            if (batch.pins)
                batch.pins->settle(result.ok());
            if (!result.ok() && error.empty())
                error = result.status.ToString();
        }
        batches.clear();
        return error.empty() ? 1 : fail(error.c_str());
    }

    RankHandle rank;
    std::vector<Batch> batches;
    std::string error;
};

RankHandle checkedRank(void* handle) {
    if (!handle)
        throw std::invalid_argument("missing BTC rank runtime");
    return *static_cast<RankHandle*>(handle);
}

void prepareTicket(void** result) {
    if (!result)
        throw std::invalid_argument("missing BTC ticket output");
    *result = nullptr;
}

void submitBatch(Ticket& ticket,
                 std::vector<Row> rows,
                 Tier source,
                 Tier target,
                 const std::shared_ptr<CudaDependency>& dependency,
                 uint64_t job) {
    auto& rank = *ticket.rank;
    const bool disk_read = source == Tier::DISK;
    const bool disk_write = target == Tier::DISK;
    const bool load = target == Tier::DEVICE;
    const auto pins = disk_read || disk_write ? rank.pinDisk(rows, disk_write, job) : nullptr;
    auto& loop = load ? rank.load_loop : rank.evict_loop;
    TransferWorkLoop::BorrowedBuffers borrowed;
    borrowed.disk_pool = rank.disk();
    borrowed.row_bytes = rank.row_bytes;
    borrowed.hosts.reserve(rows.size());
    std::vector<TransferDescriptor> descriptors;
    descriptors.reserve(rows.size());
    size_t spans = 0;
    size_t bytes = 0;
    for (size_t index = 0; index < rows.size(); ++index) {
        const auto& row = rows[index];
        if (disk_read) {
            descriptors.push_back(TransferDescriptor::diskToDevice(0, pins->slots()[index].second, {}));
        } else {
            borrowed.hosts.push_back({reinterpret_cast<void*>(row.host), rank.row_bytes, rank.row_bytes});
            if (disk_write)
                descriptors.push_back(TransferDescriptor::hostToDisk(0, 0, pins->slots()[index].second));
            else if (load)
                descriptors.push_back(TransferDescriptor::hostToDevice(0, 0, {}));
            else
                descriptors.push_back(TransferDescriptor::deviceToHost(0, {}, 0));
        }
        spans += row.spans.size();
        for (const auto& span : row.spans) {
            if (span.bytes > std::numeric_limits<size_t>::max() - bytes)
                throw std::invalid_argument("BTC rank DMA byte count overflow");
            bytes += span.bytes;
        }
    }
    if (!disk_write) {
        auto cuda = std::make_shared<CudaBatch>();
        cuda->device = rank.device;
        cuda->rank = rank.rank;
        cuda->load = load;
        cuda->job = job;
        cuda->stream = (load ? rank.load_stream : rank.evict_stream).get();
        cuda->dependency = dependency;
        for (auto& row : rows)
            cuda->spans.push_back(std::move(row.spans));
        borrowed.submit_cuda = [cuda](const auto& hosts) { cuda->submit(hosts); };
        borrowed.poll_cuda = [cuda] { return cuda->poll(); };
    }
    borrowed.observe_stage = [&rank, pins, job, segment = ticket.batches.size(), pages = rows.size(), spans, bytes](Tier from,
                                                                                   Tier to,
                                                                                   const char* phase,
                                                                                   bool ok) {
        rank.observe(job, segment, pins.get(), pages, spans, bytes, from, to, phase, ok);
    };
    TransferTask task(std::move(descriptors), kQueueTimeout);
    ticket.batches.push_back({pins, nullptr});
    ticket.batches.back().context = loop->execute(std::move(task), {}, nullptr, std::move(borrowed));
}

void submitRows(Ticket& ticket,
                std::vector<Row> rows,
                Tier source,
                Tier target,
                const std::shared_ptr<CudaDependency>& dependency,
                uint64_t job) {
    if (rows.empty())
        return;
    const auto batch_limit = source == Tier::DISK ? ticket.rank->batch_disk_pages : kMaxBatchPages;
    if (!batch_limit)
        throw std::invalid_argument("BTC rank Disk staging is disabled");
    for (size_t first = 0; first < rows.size();) {
        const auto end = first + std::min(batch_limit, rows.size() - first);
        std::vector<Row> batch(std::make_move_iterator(rows.begin() + first), std::make_move_iterator(rows.begin() + end));
        submitBatch(ticket, std::move(batch), source, target, dependency, job);
        first = end;
    }
}

int submitDma(void* runtime,
              bool load,
              uintptr_t stream,
              size_t page_count,
              const uintptr_t* host_rows,
              const int64_t* disk_ids,
              size_t span_count,
              const size_t* span_page,
              const size_t* span_offsets,
              const uintptr_t* gpu_ptrs,
              const size_t* sizes,
              uint64_t job,
              void** output) {
    prepareTicket(output);
    auto ticket = std::make_unique<Ticket>(checkedRank(runtime));
    auto& rank = *ticket->rank;
    if ((page_count && (!host_rows || !disk_ids))
        || (span_count && (!span_page || !span_offsets || !gpu_ptrs || !sizes))) {
        throw std::invalid_argument("missing BTC rank descriptor arrays");
    }
    std::vector<Row> rows;
    rows.reserve(page_count);
    for (size_t index = 0; index < page_count; ++index) {
        const auto disk = disk_ids[index];
        if (disk < kHostPage || (disk != kHostPage && (!load || !rank.disk()))
            || (disk == kHostPage && !host_rows[index]))
            throw std::invalid_argument("invalid BTC rank source page");
        rows.push_back({host_rows[index], disk, {}});
    }
    for (size_t index = 0; index < span_count; ++index) {
        const auto page = span_page[index];
        const auto offset = span_offsets[index];
        const auto gpu = gpu_ptrs[index];
        const auto bytes = sizes[index];
        const auto max_address = std::numeric_limits<uintptr_t>::max();
        if (page >= rows.size() || !gpu || !bytes || bytes > max_address - gpu
            || offset > max_address - bytes
            || (rank.row_bytes && (offset > rank.row_bytes || bytes > rank.row_bytes - offset))
            || offset + bytes > max_address - rows[page].host) {
            throw std::invalid_argument("invalid BTC rank DMA span");
        }
        rows[page].spans.push_back({offset, gpu, bytes});
    }
    if (!rows.empty()) {
        auto dependency = std::make_shared<CudaDependency>(rank.device, stream);
        std::vector<Row> host;
        std::vector<Row> disk;
        for (auto& row : rows)
            (row.disk == kHostPage ? host : disk).push_back(std::move(row));
        if (load) {
            submitRows(*ticket, std::move(disk), Tier::DISK, Tier::DEVICE, dependency, job);
            submitRows(*ticket, std::move(host), Tier::HOST, Tier::DEVICE, dependency, job);
        } else {
            submitRows(*ticket, std::move(host), Tier::DEVICE, Tier::HOST, dependency, job);
        }
    }
    *output = ticket.release();
    return 0;
}

}  // namespace

extern "C" {

const char* btc_rank_last_error() noexcept {
    return last_error;
}

int btc_rank_create(int device,
                    int rank,
                    const char* disk_dir,
                    size_t disk_slots,
                    size_t row_bytes,
                    size_t staging_slots,
                    void** output) noexcept {
    return guard([&] {
        prepareTicket(output);
        const auto alignment = HostStagingBlockPool::kAlignment;
        if (device < 0 || rank < 0
            || (disk_slots && (!disk_dir || !*disk_dir || !staging_slots || !row_bytes || row_bytes % alignment
                               || disk_slots >= static_cast<size_t>(std::numeric_limits<int32_t>::max())
                               || disk_slots + 1 > static_cast<size_t>(std::numeric_limits<int64_t>::max()) / row_bytes))) {
            throw std::invalid_argument("invalid BTC rank WorkLoop or Disk configuration");
        }
        *output = new RankHandle(std::make_shared<RankRuntime>(device, rank, disk_dir, disk_slots, row_bytes, staging_slots));
        return 0;
    });
}

void btc_rank_destroy(void* runtime) noexcept {
    delete static_cast<RankHandle*>(runtime);
}

int btc_rank_submit_load(void* runtime,
                         uintptr_t stream,
                         size_t page_count,
                         const uintptr_t* host_rows,
                         const int64_t* disk_ids,
                         size_t span_count,
                         const size_t* span_page,
                         const size_t* span_offsets,
                         const uintptr_t* gpu_ptrs,
                         const size_t* sizes,
                         uint64_t job,
                         void** output) noexcept {
    return guard([&] {
        return submitDma(runtime, true, stream, page_count, host_rows, disk_ids, span_count, span_page, span_offsets,
                         gpu_ptrs, sizes, job, output);
    });
}

int btc_rank_submit_store(void* runtime,
                          uintptr_t stream,
                          size_t page_count,
                          const uintptr_t* host_rows,
                          const int64_t* disk_ids,
                          size_t span_count,
                          const size_t* span_page,
                          const size_t* span_offsets,
                          const uintptr_t* gpu_ptrs,
                          const size_t* sizes,
                          uint64_t job,
                          void** output) noexcept {
    return guard([&] {
        return submitDma(runtime, false, stream, page_count, host_rows, disk_ids, span_count, span_page, span_offsets,
                         gpu_ptrs, sizes, job, output);
    });
}

int btc_rank_submit_disk_store(void* runtime,
                               size_t count,
                               const uintptr_t* host_rows,
                               const int64_t* disk_ids,
                               uint64_t job,
                               void** output) noexcept {
    return guard([&] {
        prepareTicket(output);
        auto ticket = std::make_unique<Ticket>(checkedRank(runtime));
        if (!ticket->rank->disk() || (count && (!host_rows || !disk_ids)))
            throw std::invalid_argument("invalid BTC rank Disk write descriptor arrays");
        std::vector<Row> rows;
        rows.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            if (disk_ids[index] < 0 || !host_rows[index] || host_rows[index] % HostStagingBlockPool::kAlignment)
                throw std::invalid_argument("invalid BTC rank Disk write row");
            rows.push_back({host_rows[index], disk_ids[index], {}});
        }
        submitRows(*ticket, std::move(rows), Tier::HOST, Tier::DISK, nullptr, job);
        *output = ticket.release();
        return 0;
    });
}

int btc_rank_release_disk(void* runtime, size_t count, const int64_t* ids) noexcept {
    return guard([&] {
        checkedRank(runtime)->releaseDisk(count, ids);
        return 0;
    });
}

int btc_rank_query(void* ticket) noexcept {
    return guard([&] {
        if (!ticket)
            return fail("missing BTC rank ticket");
        return static_cast<Ticket*>(ticket)->complete(false);
    });
}

int btc_rank_wait(void* ticket) noexcept {
    return guard([&] {
        if (!ticket)
            return fail("missing BTC rank ticket");
        return static_cast<Ticket*>(ticket)->complete(true);
    });
}

void btc_rank_destroy_ticket(void* ticket) noexcept {
    delete static_cast<Ticket*>(ticket);
}

}  // extern "C"

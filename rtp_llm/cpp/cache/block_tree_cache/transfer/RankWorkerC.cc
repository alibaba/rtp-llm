#include "block_tree_cache_worker.h"

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheCStatus.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"

using namespace rtp_llm;

namespace {

using Payload = std::unique_ptr<void, btc_rank_release_fn>;

struct RankJob {
    RankJob(Payload data, btc_rank_submit_fn callback): payload(std::move(data)), submit(callback) {}

    void markSafe() noexcept {
        {
            std::lock_guard lock(mutex);
            safe = true;
        }
        ready.notify_all();
    }

    bool safety(bool wait) {
        std::unique_lock lock(mutex);
        if (wait)
            ready.wait(lock, [this] { return safe; });
        return safe;
    }

    Payload                               payload;
    btc_rank_submit_fn                     submit;
    std::shared_ptr<TransferAsyncContext>    context;
    std::mutex                             mutex;
    std::condition_variable                ready;
    bool                                   safe{false};
};

struct RankBatch {
    explicit RankBatch(std::atomic<uint64_t>& failures, TransferAsyncContext::DoneCallback callback):
        failed_batches(failures), done(std::move(callback)) {}

    static void terminal(void* context, uint32_t success) noexcept {
        auto* batch = static_cast<RankBatch*>(context);
        batch->ok = success != 0;
        batch->arrive();
    }

    void arrive() noexcept {
        if (remaining.fetch_sub(1, std::memory_order_acq_rel) != 1)
            return;
        std::unique_ptr<RankBatch> retired(this);
        if (!ok)
            ++failed_batches;
        auto result = ok ? TransferResult{} : TransferResult::executionException(true, "Rank hardware copy failed");
        result.buffer_safe = true;
        done(std::move(result));
    }

    std::atomic<uint64_t>&                 failed_batches;
    TransferAsyncContext::DoneCallback     done;
    // Both the submit stack and the terminal event must release the batch.
    std::atomic<unsigned>                 remaining{2};
    bool                                  ok{false};
};

}  // namespace

struct btc_rank_worker {
    explicit btc_rank_worker(btc_rank_worker_config value):
        config(value),
        engine({}, PerRankWorkerOptions{32, 2, 10000, value.world_rank},
               PerRankWorkerIO{
                   [](const TransferTask&) { return true; },  // C admission checks all pages.
                   [this](const TransferTask& task, TransferAsyncContext::DoneCallback done) {
                       run(task, std::move(done));
                   },
               }) {}

    std::shared_ptr<RankJob> find(uint64_t id) {
        std::lock_guard lock(mutex);
        return jobs.at(id);
    }

    void erase(uint64_t id) {
        std::lock_guard lock(mutex);
        jobs.erase(id);
    }

    void run(const TransferTask& task, TransferAsyncContext::DoneCallback done) {
        const auto pages = task.descriptors();
        assert(!pages.empty() && pages.back().path_index - pages.front().path_index + 1 == pages.size());
        auto job = find(task.traceId());
        auto* batch = new RankBatch(failed_batches, std::move(done));
        ++executed_batches;
        job->submit(job->payload.get(), pages.front().path_index, pages.size(), &RankBatch::terminal, batch);
        job.reset();
        batch->arrive();
    }

    btc_rank_worker_config                                 config;
    std::mutex                                             mutex;
    std::unordered_map<uint64_t, std::shared_ptr<RankJob>>    jobs;
    std::atomic<uint64_t>                                  submitted_jobs{0};
    std::atomic<uint64_t>                                  executed_batches{0};
    std::atomic<uint64_t>                                  failed_batches{0};
    PerRankBlockTransferEngine                              engine;
};

namespace {

btc_status observe(btc_rank_worker* worker, uint64_t id, btc_rank_completion* completion, bool wait) {
    return c_api::status([&]() -> btc_status {
        auto& rank = c_api::require(worker);
        auto& output = c_api::require(completion);
        const auto job = rank.find(id);
        if (wait)
            job->context->waitDone();
        else if (!job->context->done())
            return BTC_PENDING;
        output.result = job->context->success() ? BTC_SUCCESS : BTC_FAILED;
        output.safe = job->safety(wait);
        if (output.safe) {
            rank.erase(id);
            job->payload.reset();
        }
        return BTC_OK;
    });
}

}  // namespace

extern "C" {

btc_status btc_rank_worker_open(const btc_rank_worker_config* config, btc_rank_worker** worker) {
    return c_api::status([&] {
        const auto& value = c_api::require(config);
        auto& output = c_api::require(worker);
        output = nullptr;
        if (value.abi_version != BTC_RANK_WORKER_ABI_VERSION || value.group_count == 0
            || value.device_slots <= 1 || value.host_slots <= 0) {
            throw std::invalid_argument("Invalid BlockTreeCache rank worker configuration");
        }
        output = new btc_rank_worker(value);
        return BTC_OK;
    });
}

btc_status btc_rank_worker_close(btc_rank_worker* worker) {
    return c_api::status([&] {
        // Engine member destruction joins the owners before jobs free their
        // payload. Every accepted batch must report actual terminal safety.
        delete worker;
        return BTC_OK;
    });
}

btc_status btc_rank_worker_submit(btc_rank_worker* worker,
                                  uint64_t id,
                                  uint32_t load,
                                  const btc_rank_page* pages,
                                  size_t count,
                                  void* payload,
                                  btc_rank_submit_fn submit,
                                  btc_rank_release_fn release) {
    if (!release)
        return c_api::rememberError(BTC_INVALID, "Rank job requires a payload release callback");
    Payload owner(payload, release);
    return c_api::status([&] {
        auto& rank = c_api::require(worker);
        if (id == 0 || load > 1 || !pages || count == 0 || !submit)
            throw std::invalid_argument("Invalid BlockTreeCache rank job");
        std::vector<TransferDescriptor> descriptors;
        descriptors.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            const auto& page = pages[index];
            if (page.group >= rank.config.group_count || page.device_slot <= 0
                || page.device_slot >= rank.config.device_slots || page.host_slot < 0
                || page.host_slot >= rank.config.host_slots) {
                throw std::invalid_argument("Rank job page is outside the configured group or slots");
            }
            auto descriptor = load ? TransferDescriptor::hostToDevice(page.group, page.host_slot, {page.device_slot})
                                   : TransferDescriptor::deviceToHost(page.group, {page.device_slot}, page.host_slot);
            descriptor.path_index = index;
            descriptors.push_back(std::move(descriptor));
        }
        TransferTask task(std::move(descriptors), std::nullopt);
        if (!task.sanityCheck())
            throw std::invalid_argument("Rank job pages must be grouped contiguously");
        task.setTraceId(id);
        auto job = std::make_shared<RankJob>(std::move(owner), submit);
        task.setCompletionProof(std::make_shared<TransferBufferSafety>([weak = std::weak_ptr<RankJob>(job)] {
            if (const auto live = weak.lock())
                live->markSafe();
        }));
        {
            std::lock_guard lock(rank.mutex);
            if (!rank.jobs.emplace(id, job).second)
                throw std::invalid_argument("Duplicate BlockTreeCache rank job");
        }
        try {
            job->context = rank.engine.execute(std::move(task));
        } catch (...) {
            rank.engine.waitForIdle();
            rank.erase(id);
            throw;
        }
        ++rank.submitted_jobs;
        return BTC_OK;
    });
}

btc_status btc_rank_worker_poll(btc_rank_worker* worker, uint64_t id, btc_rank_completion* completion) {
    return observe(worker, id, completion, false);
}

btc_status btc_rank_worker_wait(btc_rank_worker* worker, uint64_t id, btc_rank_completion* completion) {
    return observe(worker, id, completion, true);
}

btc_status btc_rank_worker_get_stats(btc_rank_worker* worker, btc_rank_worker_stats* stats) {
    return c_api::status([&] {
        const auto& rank = c_api::require(worker);
        c_api::require(stats) = {
            rank.submitted_jobs.load(), rank.executed_batches.load(), rank.failed_batches.load(),
        };
        return BTC_OK;
    });
}

}  // extern "C"

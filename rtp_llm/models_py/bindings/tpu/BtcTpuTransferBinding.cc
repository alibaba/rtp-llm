#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferWorkLoop.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/tpu/TpuPageDma.h"

namespace py = pybind11;

namespace rtp_llm {
namespace {

using SpanArgs = std::tuple<size_t, int64_t, int64_t, int64_t>;

// As in the CUDA rank worker, the queue deadline applies before submission.
// Once the first device copy is issued, its events determine completion.
constexpr auto   kQueueTimeout = std::chrono::minutes(5);
constexpr size_t kQueueBatches = 1024;

std::unique_ptr<TransferWorkLoop> makeLoop(TransferOperation direction) {
    // The control plane consumes one receipt per job. The DMA batch owns all
    // of that job's physical spans, including those issued before an error.
#ifdef RTP_BTC_RANK_STANDALONE
    return std::make_unique<TransferWorkLoop>(nullptr, direction, 1, kQueueBatches, nullptr, 0);
#else
    return std::make_unique<TransferWorkLoop>(
        std::vector<GroupSetPtr>{}, nullptr, nullptr, direction, 1, kQueueBatches, nullptr);
#endif
}

class RankTransfer {
public:
    RankTransfer(const std::vector<at::Tensor>& device_tensors, at::Tensor host_region, int64_t num_blocks):
        dma_(std::make_unique<TpuPageDma>(device_tensors, std::move(host_region), num_blocks)),
        load_(makeLoop(TransferOperation::LOAD)),
        evict_(makeLoop(TransferOperation::EVICT)) {
        load_->start();
        evict_->start();
    }

    std::vector<size_t> bufferSizes() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!load_)
            throw std::logic_error("TPU transfer worker is closed");
        return dma_->bufferSizes();
    }

    void submit(uint64_t job_id, bool load, const std::vector<SpanArgs>& spans) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!load_)
            throw std::logic_error("TPU transfer worker is closed");
        if (jobs_.count(job_id))
            throw std::invalid_argument("duplicate TPU transfer job " + std::to_string(job_id));

        std::vector<TpuCopySpan> physical_spans;
        physical_spans.reserve(spans.size());
        for (const auto& [buffer, device_offset, host_offset, bytes] : spans)
            physical_spans.push_back({buffer, device_offset, host_offset, bytes});
        const auto direction = load ? TpuCopyDirection::HostToDevice : TpuCopyDirection::DeviceToHost;
        auto       batch = dma_->prepare(direction, std::move(physical_spans));

        TransferWorkLoop::BorrowedBuffers borrowed;
        borrowed.submit_device = [batch](const std::vector<HostBufferView>&) { batch->submit(); };
        borrowed.poll_device = [batch]() -> std::optional<TransferResult> {
            auto status = batch->poll();
            if (!status)
                return std::nullopt;
            if (!status->empty())
                return TransferResult::executionException(true, status->c_str());
            return TransferResult::emptyAggregate();
        };

        // The WorkLoop routes this borrowed batch by tier. Only the prepared
        // batch resolves addresses, so no RTP block pool or synthetic ID is used.
        TransferDescriptor route;
        route.source_tier = load ? Tier::HOST : Tier::DEVICE;
        route.target_tier = load ? Tier::DEVICE : Tier::HOST;
        TransferTask task({std::move(route)}, kQueueTimeout);
        // Publish the receipt slot before admission can make any DMA visible.
        auto job = jobs_.try_emplace(job_id).first;
        try {
            auto& loop = load ? load_ : evict_;
            job->second = loop->execute(std::move(task), {}, nullptr, std::move(borrowed));
        } catch (...) {
            jobs_.erase(job);
            throw;
        }
    }

    std::optional<bool> poll(uint64_t job_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto job = findJob(job_id);
        if (!job->second->done())
            return std::nullopt;
        return consume(job);
    }

    bool wait(uint64_t job_id) {
        std::shared_ptr<TransferAsyncContext> context;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            context = findJob(job_id)->second;
        }
        context->waitDone();
        std::lock_guard<std::mutex> lock(mutex_);
        auto job = findJob(job_id);
        if (context != job->second)
            throw std::logic_error("TPU transfer receipt was already consumed");
        return consume(job);
    }

    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!dma_)
            return;
        // Destruction stops admission and joins each WorkLoop. If unmap
        // fails, the stopped worker can retry close without admitting jobs.
        load_.reset();
        evict_.reset();
        jobs_.clear();
        dma_->close();
        dma_.reset();
    }

private:
    using Jobs = std::unordered_map<uint64_t, std::shared_ptr<TransferAsyncContext>>;

    Jobs::iterator findJob(uint64_t job_id) {
        auto job = jobs_.find(job_id);
        if (job == jobs_.end())
            throw std::out_of_range("unknown TPU transfer job " + std::to_string(job_id));
        return job;
    }

    bool consume(Jobs::iterator job) {
        const bool success = job->second->success();
        jobs_.erase(job);
        return success;
    }

    // Both WorkLoops also drain on destruction. Keep the registered device
    // and host storage alive until after both owners have joined.
    std::unique_ptr<TpuPageDma>       dma_;
    std::unique_ptr<TransferWorkLoop> load_;
    std::unique_ptr<TransferWorkLoop> evict_;
    std::mutex                       mutex_;
    Jobs                             jobs_;
};

}  // namespace
}  // namespace rtp_llm

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    using rtp_llm::RankTransfer;
    using ReleaseGil = py::call_guard<py::gil_scoped_release>;
    py::register_exception<rtp_llm::TpuDmaAdmissionError>(module, "TpuDmaAdmissionError", PyExc_RuntimeError);
    py::class_<RankTransfer>(module, "RankTransfer")
        .def(py::init<const std::vector<at::Tensor>&, at::Tensor, int64_t>(),
             py::arg("device_tensors"), py::arg("host_region"), py::arg("num_blocks"), ReleaseGil())
        .def("buffer_sizes", &RankTransfer::bufferSizes, ReleaseGil())
        .def("submit", &RankTransfer::submit, py::arg("job_id"), py::arg("load"), py::arg("spans"), ReleaseGil())
        .def("poll", &RankTransfer::poll, py::arg("job_id"), ReleaseGil())
        .def("wait", &RankTransfer::wait, py::arg("job_id"), ReleaseGil())
        .def("close", &RankTransfer::close, ReleaseGil());
}

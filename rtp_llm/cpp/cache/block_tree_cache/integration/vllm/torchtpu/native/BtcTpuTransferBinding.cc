// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Adapted from the TPU rank binding in RTP-LLM commit eee89f6f89de.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <dlfcn.h>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "block_tree_cache_worker.h"
#include "TpuPageDma.h"

namespace py = pybind11;

namespace vllm_torchtpu::btc {
namespace {

using SpanArgs =
    std::tuple<std::size_t, std::int64_t, std::int64_t, std::int64_t>;
using PageArgs = std::tuple<std::uint32_t, std::int32_t, std::int32_t,
                            std::vector<SpanArgs>>;
using Completion = std::pair<bool, bool>;

void checkWorker(btc_status status) {
  if (status == BTC_INVALID)
    throw std::invalid_argument(btc_last_error());
  if (status != BTC_OK)
    throw std::runtime_error(btc_last_error());
}

std::string workerLibrary() {
  Dl_info info{};
  if (!dladdr(reinterpret_cast<const void *>(&btc_rank_worker_open), &info) ||
      !info.dli_fname)
    throw std::runtime_error("Could not identify the loaded BTC rank worker");
  return info.dli_fname;
}

class RankTransfer {
public:
  RankTransfer(const std::vector<at::Tensor> &device_tensors,
               at::Tensor host_region, std::int32_t num_blocks,
               std::int32_t num_slots, std::uint32_t groups,
               std::uint32_t rank)
      : dma_(std::make_unique<TpuPageDma>(device_tensors, std::move(host_region),
                                          num_blocks)) {
    const btc_rank_worker_config config{BTC_RANK_WORKER_ABI_VERSION, rank,
                                         groups, num_blocks, num_slots};
    btc_rank_worker *worker = nullptr;
    checkWorker(btc_rank_worker_open(&config, &worker));
    worker_.reset(worker);
  }

  std::vector<std::size_t> bufferSizes() {
    std::lock_guard lock(mutex_);
    requireOpen();
    return dma_->bufferSizes();
  }

  void submit(std::uint64_t job_id, bool load, std::vector<PageArgs> rows) {
    std::lock_guard lock(mutex_);
    requireOpen();
    if (rows.empty())
      throw std::invalid_argument("BTC rank job requires a page");
    std::stable_sort(rows.begin(), rows.end(), [](const auto &lhs, const auto &rhs) {
      return std::get<0>(lhs) < std::get<0>(rhs);
    });
    std::vector<btc_rank_page> pages;
    std::vector<TpuCopySpan> spans;
    std::vector<std::size_t> starts{0};
    pages.reserve(rows.size());
    starts.reserve(rows.size() + 1);
    for (const auto &[group, device_slot, host_slot, segments] : rows) {
      pages.push_back({group, device_slot, host_slot});
      for (const auto &[buffer, device_offset, host_offset, bytes] : segments)
        spans.push_back({buffer, device_offset, host_offset, bytes});
      starts.push_back(spans.size());
    }
    // Resolve storage on the runner thread before a forwarding graph can
    // replace the carriers. Only bounded physical copy runs on a WorkLoop.
    const auto direction =
        load ? TpuCopyDirection::HostToDevice : TpuCopyDirection::DeviceToHost;
    auto plan = dma_->prepare(direction, std::move(spans), std::move(starts));
    checkWorker(btc_rank_worker_submit(
        worker_.get(), job_id, load, pages.data(), pages.size(), plan.release(),
        [](void *payload, std::size_t first, std::size_t count,
           btc_rank_copy_done_fn done, void *context) {
          static_cast<const TpuDmaJob *>(payload)->submit(first, count, done,
                                                         context);
        },
        [](void *payload) { delete static_cast<TpuDmaJob *>(payload); }));
  }

  std::optional<Completion> poll(std::uint64_t job_id) {
    std::lock_guard lock(mutex_);
    requireOpen();
    btc_rank_completion proof{};
    const auto status = btc_rank_worker_poll(worker_.get(), job_id, &proof);
    if (status == BTC_PENDING)
      return std::nullopt;
    checkWorker(status);
    return Completion{proof.result == BTC_SUCCESS, proof.safe != 0};
  }

  Completion wait(std::uint64_t job_id) {
    std::lock_guard lock(mutex_);
    requireOpen();
    btc_rank_completion proof{};
    checkWorker(btc_rank_worker_wait(worker_.get(), job_id, &proof));
    return {proof.result == BTC_SUCCESS, proof.safe != 0};
  }

  std::tuple<std::uint64_t, std::uint64_t, std::uint64_t> stats() {
    std::lock_guard lock(mutex_);
    requireOpen();
    btc_rank_worker_stats values{};
    checkWorker(btc_rank_worker_get_stats(worker_.get(), &values));
    return {values.submitted_jobs, values.executed_batches, values.failed_batches};
  }

  void close() {
    std::lock_guard lock(mutex_);
    // Native engines drain their copied aliases prior to DmaUnmap. Retain
    // the DMA owner if explicit unmap fails so close() can retry.
    worker_.reset();
    if (dma_) {
      dma_->close();
      dma_.reset();
    }
  }

private:
  struct CloseWorker {
    void operator()(btc_rank_worker *worker) const noexcept {
      btc_rank_worker_close(worker);
    }
  };

  void requireOpen() const {
    if (!worker_)
      throw std::logic_error("TPU transfer worker is closed");
  }

  std::mutex mutex_;
  std::unique_ptr<TpuPageDma> dma_;
  std::unique_ptr<btc_rank_worker, CloseWorker> worker_;
};

} // namespace
} // namespace vllm_torchtpu::btc

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  using vllm_torchtpu::btc::RankTransfer;
  using vllm_torchtpu::btc::TpuDmaAdmissionError;
  using ReleaseGil = py::call_guard<py::gil_scoped_release>;
  py::register_exception<TpuDmaAdmissionError>(module, "TpuDmaAdmissionError",
                                               PyExc_RuntimeError);
  module.def("worker_library", &vllm_torchtpu::btc::workerLibrary, ReleaseGil());
  py::class_<RankTransfer>(module, "RankTransfer")
      .def(py::init<const std::vector<at::Tensor> &, at::Tensor, std::int32_t,
                    std::int32_t, std::uint32_t, std::uint32_t>(),
           py::arg("device_tensors"), py::arg("host_region"),
           py::arg("num_blocks"), py::arg("num_slots"), py::arg("groups"),
           py::arg("rank"), ReleaseGil())
      .def("buffer_sizes", &RankTransfer::bufferSizes, ReleaseGil())
      .def("submit", &RankTransfer::submit, py::arg("job_id"), py::arg("load"),
           py::arg("pages"), ReleaseGil())
      .def("poll", &RankTransfer::poll, py::arg("job_id"), ReleaseGil())
      .def("wait", &RankTransfer::wait, py::arg("job_id"), ReleaseGil())
      .def("stats", &RankTransfer::stats, ReleaseGil())
      .def("close", &RankTransfer::close, ReleaseGil());
}

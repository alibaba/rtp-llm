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

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <ATen/core/TensorBody.h>

namespace vllm_torchtpu::btc {

enum class TpuCopyDirection { DeviceToHost, HostToDevice };

// All offsets are bytes within the registered base storage or host region.
struct TpuCopySpan {
  std::size_t buffer_index;
  std::int64_t device_offset;
  std::int64_t host_offset;
  std::int64_t bytes;
};

struct TpuDmaRegion;

// No DMA was enqueued. A previously registered carrier could not be
// reacquired after the model ran, so the old contents remain unpublished.
class TpuDmaAdmissionError final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// The runner snapshots the pre-forward carriers and validates all destinations.
// The native WorkLoop then submits disjoint page intervals. Completion is
// delivered from PJRT after all their events are terminal.
class TpuDmaJob final {
public:
  using Completion = void (*)(void *, std::uint32_t);
  ~TpuDmaJob();
  TpuDmaJob(const TpuDmaJob &) = delete;
  TpuDmaJob &operator=(const TpuDmaJob &) = delete;

  void submit(std::size_t first_page, std::size_t page_count, Completion done,
              void *context) const noexcept;

private:
  friend class TpuPageDma;
  struct State;
  explicit TpuDmaJob(std::unique_ptr<State> state);
  std::unique_ptr<State> state_;
};

class TpuPageDma final {
public:
  // The host tensor must own (or retain the owner of) a page-aligned,
  // page-sized multiple of contiguous CPU byte storage, exclusively granted
  // to this instance for mlock/DmaMap until close().
  TpuPageDma(const std::vector<at::Tensor> &device_buffers,
             at::Tensor host_region, std::int64_t num_blocks);
  TpuPageDma(const TpuPageDma &) = delete;
  TpuPageDma &operator=(const TpuPageDma &) = delete;
  std::unique_ptr<TpuDmaJob>
  prepare(TpuCopyDirection direction, std::vector<TpuCopySpan> spans,
          std::vector<std::size_t> span_starts) const;
  std::vector<std::size_t> bufferSizes() const;
  // Release the prepared jobs before closing. A failed unmap keeps the
  // registration available for retry.
  void close();

private:
  mutable std::mutex mutex_;
  std::shared_ptr<TpuDmaRegion> region_;
  bool closing_ = false;
};

} // namespace vllm_torchtpu::btc

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

// Adapted from RTP-LLM commit eee89f6f89de, TpuPageDma.cc.
#include "TpuPageDma.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cstdio>
#include <exception>
#include <optional>
#include <span>
#include <stdexcept>
#include <sys/mman.h>
#include <system_error>
#include <tuple>
#include <unistd.h>
#include <utility>

#include "csrc/api/tensor_buffer.h"
#include "xla/layout.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_layouts_extension.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_layout.h"

namespace vllm_torchtpu::btc {
namespace {

void destroyError(const PJRT_Api *api, PJRT_Error *error) noexcept {
  if (error) {
    PJRT_Error_Destroy_Args args{PJRT_Error_Destroy_Args_STRUCT_SIZE, nullptr,
                                 error};
    api->PJRT_Error_Destroy(&args);
  }
}

std::string errorMessage(const PJRT_Api *api, PJRT_Error *error) {
  PJRT_Error_Message_Args args{PJRT_Error_Message_Args_STRUCT_SIZE, nullptr,
                               error, nullptr, 0};
  api->PJRT_Error_Message(&args);
  return args.message_size ? std::string(args.message, args.message_size)
                           : "PJRT reported an error";
}

void check(const PJRT_Api *api, PJRT_Error *error, const char *operation) {
  if (error) {
    const auto deleter = [api](PJRT_Error *e) { destroyError(api, e); };
    std::unique_ptr<PJRT_Error, decltype(deleter)> owner(error, deleter);
    throw std::runtime_error(std::string(operation) + ": " +
                             errorMessage(api, error));
  }
}

class PjrtBatchEvents {
public:
  PjrtBatchEvents(const PJRT_Api *api, std::size_t count,
                  TpuDmaJob::Completion done, void *context)
      : api_(api), done_(done), context_(context) {
    events_.reserve(count);
  }

  bool record(PJRT_Error *error) noexcept {
    if (!error)
      return true;
    ok_.store(false, std::memory_order_relaxed);
    destroyError(api_, error);
    return false;
  }

  bool track(PJRT_Event *event) noexcept {
    events_.push_back(event);  // All capacity is reserved before the first DMA.
    remaining_.fetch_add(1, std::memory_order_relaxed);
    PJRT_Event_OnReady_Args args{
        PJRT_Event_OnReady_Args_STRUCT_SIZE, nullptr, event,
        [](PJRT_Error *error, void *context) {
          auto *batch = static_cast<PjrtBatchEvents *>(context);
          batch->record(error);
          batch->release();
        },
        this};
    if (record(api_->PJRT_Event_OnReady(&args)))
      return true;
    // Registration failed. This thread must obtain terminal proof for the
    // still-live event before reporting failure.
    PJRT_Event_Await_Args wait{PJRT_Event_Await_Args_STRUCT_SIZE, nullptr, event};
    record(api_->PJRT_Event_Await(&wait));
    release();
    return false;
  }

  void release() noexcept {
    if (remaining_.fetch_sub(1, std::memory_order_acq_rel) != 1)
      return;
    std::unique_ptr<PjrtBatchEvents> retired(this);
    for (auto *event : events_) {
      PJRT_Event_Destroy_Args destroy{
          PJRT_Event_Destroy_Args_STRUCT_SIZE, nullptr, event};
      record(api_->PJRT_Event_Destroy(&destroy));
    }
    done_(context_, ok_.load(std::memory_order_relaxed));
  }

private:
  const PJRT_Api *api_;
  TpuDmaJob::Completion done_;
  void *context_;
  std::vector<PJRT_Event *> events_;
  // The submission stack holds a reference while callbacks can run inline.
  std::atomic<std::size_t> remaining_{1};
  std::atomic<bool> ok_{true};
};

struct RawDeleter {
  const PJRT_Api *api;
  const PJRT_RawBuffer_Extension *extension;
  void operator()(PJRT_RawBuffer *buffer) const noexcept {
    PJRT_RawBuffer_Destroy_Args args{PJRT_RawBuffer_Destroy_Args_STRUCT_SIZE,
                                     nullptr, buffer};
    destroyError(api, extension->PJRT_RawBuffer_Destroy(&args));
  }
};

struct CarrierFormat {
  std::size_t bytes;
  PJRT_Buffer_Type element_type;
  std::vector<std::int64_t> dimensions;
  std::shared_ptr<const xla::PjRtLayout> layout;

  bool matches(const CarrierFormat &other) const {
    return bytes == other.bytes && element_type == other.element_type &&
           dimensions == other.dimensions && *layout == *other.layout;
  }
};

struct Storage {
  torch_tpu::TensorBufferHandle handle;
  PJRT_Buffer *identity;
  std::unique_ptr<PJRT_RawBuffer, RawDeleter> raw;
  CarrierFormat format;
};

bool fits(std::int64_t offset, std::int64_t length, std::size_t capacity) {
  return offset >= 0 && length > 0 &&
         static_cast<std::uint64_t>(offset) <= capacity &&
         static_cast<std::uint64_t>(length) <=
             capacity - static_cast<std::uint64_t>(offset);
}

// A manager row is a consecutive slice of the dimension-0 traversal of
// the live storage. A raw flat allocation can also be exposed as a full
// block-major view. Reject layouts we cannot prove preserve this ordering.
void checkManagerRows(const at::Tensor &tensor,
                      std::span<const std::int64_t> dims,
                      const xla::Layout &layout, std::size_t bytes,
                      std::int64_t blocks) {
  if (tensor.dim() < 1 || dims.empty() || tensor.size(0) <= 0 ||
      dims.front() <= 0 || tensor.size(0) % blocks != 0 ||
      dims.front() % blocks != 0 || bytes != tensor.nbytes() || bytes == 0 ||
      bytes % blocks != 0)
    throw std::invalid_argument("BTC storage cannot be split into whole "
                                "contiguous physical manager rows");
  if (dims.size() != 1 &&
      (dims.size() != static_cast<std::size_t>(tensor.dim()) ||
       !std::equal(dims.begin(), dims.end(), tensor.sizes().begin())))
    throw std::invalid_argument("BTC device tensor must expose the original "
                                "dimensions or full flat storage");
  if (layout.minor_to_major().size() != dims.size() ||
      layout.minor_to_major().back() != 0)
    throw std::invalid_argument("BTC manager rows require the leading "
                                "dimension to be physically outermost");
  for (const auto &tile : layout.tiles()) {
    const auto tile_dims = tile.dimensions();
    if (tile_dims.empty() || tile_dims.size() > dims.size() ||
        std::any_of(tile_dims.begin(), tile_dims.end(),
                    [](auto d) { return d <= 0; }))
      throw std::invalid_argument(
          "unsupported TPU tile shape for BTC manager rows");
    if (tile_dims.size() == dims.size() && tile_dims.front() > 1 &&
        (dims.size() != 1 || (dims.front() / blocks) % tile_dims.front() != 0))
      throw std::invalid_argument(
          "BTC direct transfer cannot split a tile across manager rows");
  }
}

// PjRtCApiBuffer::layout() synthesizes a descending layout when the plugin
// omits the Layouts extension. Only the plugin can attest to physical order.
std::shared_ptr<const xla::PjRtLayout>
readLayout(const PJRT_Api *api, const PJRT_Layouts_Extension *ext,
           PJRT_Buffer *buffer) {
  PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args query{
      PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args_STRUCT_SIZE, nullptr, buffer,
      nullptr};
  check(api, ext->PJRT_Layouts_PJRT_Buffer_MemoryLayout(&query),
        "get BTC physical layout");
  const auto release = [api, ext](PJRT_Layouts_MemoryLayout *layout) {
    PJRT_Layouts_MemoryLayout_Destroy_Args args{
        PJRT_Layouts_MemoryLayout_Destroy_Args_STRUCT_SIZE, nullptr, layout};
    destroyError(api, ext->PJRT_Layouts_MemoryLayout_Destroy(&args));
  };
  std::unique_ptr<PJRT_Layouts_MemoryLayout, decltype(release)> owner(
      query.layout, release);
  PJRT_Layouts_MemoryLayout_Serialize_Args encoded{};
  encoded.struct_size = PJRT_Layouts_MemoryLayout_Serialize_Args_STRUCT_SIZE;
  encoded.layout = owner.get();
  check(api, ext->PJRT_Layouts_MemoryLayout_Serialize(&encoded),
        "serialize BTC physical layout");
  std::unique_ptr<PJRT_Layouts_SerializedLayout,
                  decltype(encoded.serialized_layout_deleter)>
      encoded_owner(encoded.serialized_layout,
                    encoded.serialized_layout_deleter);
  auto parsed = xla::PjRtLayout::Deserialize(
      {encoded.serialized_bytes, encoded.serialized_bytes_size});
  if (!parsed.ok())
    throw std::runtime_error(parsed.status().ToString());
  return *std::move(parsed);
}

} // namespace

struct TpuDmaRegion {
  TpuDmaRegion(const std::vector<at::Tensor> &tensors, at::Tensor host_tensor,
               std::int64_t block_count)
      : host(std::make_unique<at::Tensor>(std::move(host_tensor))),
        carriers(tensors), blocks(block_count) {
    if (tensors.empty())
      throw std::invalid_argument(
          "BTC TPU DMA requires at least one device storage");
    if (blocks <= 0)
      throw std::invalid_argument(
          "BTC TPU DMA requires a positive manager block count");
    if (!host->defined() || !host->device().is_cpu() ||
        !host->is_contiguous() ||
        (host->scalar_type() != at::kByte && host->scalar_type() != at::kChar))
      throw std::invalid_argument(
          "BTC Host DMA region must be contiguous CPU byte storage");
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0)
      throw std::runtime_error(
          "BTC could not determine the Host system page size");
    base = static_cast<std::uint8_t *>(host->data_ptr());
    host_bytes = host->nbytes();
    if (!base || host_bytes == 0 ||
        reinterpret_cast<std::uintptr_t>(base) % page != 0 ||
        host_bytes % page != 0)
      throw std::invalid_argument(
          "BTC Host DMA base and region size must be system-page aligned");

    formats.reserve(tensors.size());
    for (const auto &tensor : tensors)
      formats.push_back(resolveStorage(tensor).format);
    // Map only once all potentially throwing descriptor construction is done.
    if (::mlock(base, host_bytes) != 0)
      throw std::system_error(errno, std::generic_category(),
                              "BTC mlock Host region (check RLIMIT_MEMLOCK)");
    PJRT_Client_DmaMap_Args args{PJRT_Client_DmaMap_Args_STRUCT_SIZE, nullptr,
                                 client, base, host_bytes};
    if (auto *error = api->PJRT_Client_DmaMap(&args)) {
      ::munlock(base, host_bytes);
      check(api, error, "BTC Host DmaMap");
    }
    mapped = true;
  }

  Storage resolveStorage(const at::Tensor &tensor) {
    if (!tensor.defined() ||
        tensor.device().type() != at::DeviceType::PrivateUse1 ||
        !tensor.is_contiguous() || tensor.storage_offset() != 0)
      throw std::invalid_argument(
          "BTC requires a contiguous TPU base tensor at storage offset zero");
    // A materialized view can be a different allocation from the one the
    // model writes. Pin and address the original storage, in physical bytes.
    auto ref = torch_tpu::GetBaseTensorBuffer(tensor);
    if (!ref.ok())
      throw std::runtime_error(ref.status().ToString());
    auto handle = std::move(ref).value();
    if (handle.size_bytes() != tensor.nbytes())
      throw std::invalid_argument(
          "BTC device tensor must span the complete backing storage");
    if (auto status = handle.Materialize(); !status.ok())
      throw std::runtime_error(status.ToString());
    auto buffer = handle.AwaitBuffer();
    if (!buffer.ok())
      throw std::runtime_error(buffer.status().ToString());
    auto *capi = dynamic_cast<xla::PjRtCApiBuffer *>(*buffer);
    auto *owner =
        capi ? dynamic_cast<xla::PjRtCApiClient *>(capi->client()) : nullptr;
    if (!owner)
      throw std::invalid_argument(
          "BTC TPU storage requires a PJRT C API buffer and client");

    if (!api) {
      api = owner->pjrt_c_api();
      client = owner->pjrt_c_client();
      extension = owner->FindExtension<PJRT_RawBuffer_Extension>(
          PJRT_Extension_Type_RawBuffer);
      layouts = owner->FindExtension<PJRT_Layouts_Extension>(
          PJRT_Extension_Type_Layouts);
      if (!layouts ||
          layouts->base.struct_size <
              PJRT_STRUCT_SIZE(PJRT_Layouts_Extension,
                               PJRT_Layouts_PJRT_Buffer_MemoryLayout) ||
          !layouts->PJRT_Layouts_PJRT_Buffer_MemoryLayout ||
          !layouts->PJRT_Layouts_MemoryLayout_Serialize ||
          !layouts->PJRT_Layouts_MemoryLayout_Destroy)
        throw std::runtime_error(
            "BTC requires the PJRT physical buffer layout extension");
      if (!client || !api ||
          api->struct_size < PJRT_STRUCT_SIZE(PJRT_Api, PJRT_Client_DmaUnmap) ||
          !api->PJRT_Client_DmaMap || !api->PJRT_Client_DmaUnmap ||
          !api->PJRT_Event_OnReady ||
          !extension ||
          extension->base.struct_size <
              PJRT_STRUCT_SIZE(PJRT_RawBuffer_Extension,
                               PJRT_RawBuffer_CopyRawDeviceToHost) ||
          !extension->PJRT_RawBuffer_CreateRawAliasOfBuffer ||
          !extension->PJRT_RawBuffer_Destroy ||
          !extension->PJRT_RawBuffer_GetOnDeviceSizeInBytes ||
          !extension->PJRT_RawBuffer_CopyRawDeviceToHost ||
          !extension->PJRT_RawBuffer_CopyRawHostToDevice)
        throw std::runtime_error("BTC requires PJRT completion callbacks, Host DMA "
                                 "mapping and the bidirectional RawBuffer extension");
    } else if (owner->pjrt_c_client() != client) {
      throw std::invalid_argument(
          "BTC device storages must belong to one rank-local PJRT client");
    }
    PJRT_Buffer *identity = capi->c_buffer();
    PJRT_Buffer_Device_Args where{PJRT_Buffer_Device_Args_STRUCT_SIZE, nullptr,
                                  identity, nullptr};
    check(api, api->PJRT_Buffer_Device(&where), "PJRT_Buffer_Device");
    if (device && device != where.device)
      throw std::invalid_argument(
          "BTC device storages must belong to one TPU device");
    device = where.device;
    PJRT_RawBuffer_CreateRawAliasOfBuffer_Args alias{
        PJRT_RawBuffer_CreateRawAliasOfBuffer_Args_STRUCT_SIZE, nullptr,
        identity, nullptr};
    check(api, extension->PJRT_RawBuffer_CreateRawAliasOfBuffer(&alias),
          "create BTC raw buffer alias");
    std::unique_ptr<PJRT_RawBuffer, RawDeleter> raw(alias.raw_buffer,
                                                    RawDeleter{api, extension});
    PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args size{
        PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args_STRUCT_SIZE, nullptr,
        raw.get(), 0};
    check(api, extension->PJRT_RawBuffer_GetOnDeviceSizeInBytes(&size),
          "get BTC physical buffer size");
    PJRT_Buffer_Dimensions_Args dims{PJRT_Buffer_Dimensions_Args_STRUCT_SIZE,
                                     nullptr, identity, nullptr, 0};
    check(api, api->PJRT_Buffer_Dimensions(&dims),
          "get BTC storage dimensions");
    PJRT_Buffer_ElementType_Args kind{};
    kind.struct_size = PJRT_Buffer_ElementType_Args_STRUCT_SIZE;
    kind.buffer = identity;
    check(api, api->PJRT_Buffer_ElementType(&kind),
          "get BTC storage element type");
    const auto layout = readLayout(api, layouts, identity);
    checkManagerRows(tensor, {dims.dims, dims.num_dims}, layout->xla_layout(),
                     size.on_device_size_in_bytes, blocks);
    return {std::move(handle),
            identity,
            std::move(raw),
            {size.on_device_size_in_bytes,
             kind.type,
             {dims.dims, dims.dims + dims.num_dims},
             layout}};
  }

  PJRT_Error *unmap() noexcept {
    if (!mapped)
      return nullptr;
    PJRT_Client_DmaUnmap_Args args{PJRT_Client_DmaUnmap_Args_STRUCT_SIZE,
                                   nullptr, client, base};
    PJRT_Error *error = api->PJRT_Client_DmaUnmap(&args);
    mapped = error != nullptr;
    if (!mapped && ::munlock(base, host_bytes) != 0)
      std::perror("BTC munlock after DmaUnmap");
    return error;
  }

  ~TpuDmaRegion() {
    if (PJRT_Error *error = unmap()) {
      // Explicit close() supports retry. In destructor fallback the
      // mapping can still refer to these pages: quarantine the owner.
      std::fputs("BTC: PJRT DmaUnmap failed during teardown; retaining "
                 "registered Host storage\n",
                 stderr);
      host.release();
      destroyError(api, error);
    }
  }

  std::unique_ptr<at::Tensor> host;
  std::uint8_t *base = nullptr;
  std::size_t host_bytes = 0;
  const PJRT_Api *api = nullptr;
  PJRT_Client *client = nullptr;
  PJRT_Device *device = nullptr;
  const PJRT_RawBuffer_Extension *extension = nullptr;
  const PJRT_Layouts_Extension *layouts = nullptr;
  std::vector<at::Tensor> carriers;
  std::int64_t blocks;
  std::vector<CarrierFormat> formats;
  bool mapped = false;
};

struct TpuDmaJob::State {
  State(std::shared_ptr<TpuDmaRegion> registered, TpuCopyDirection dir,
        std::vector<TpuCopySpan> copies, std::vector<std::size_t> starts)
      : region(std::move(registered)), direction(dir), spans(std::move(copies)),
        span_starts(std::move(starts)), storage(region->carriers.size()) {
    assert(span_starts.size() > 1 && span_starts.front() == 0 &&
           span_starts.back() == spans.size() &&
           std::is_sorted(span_starts.begin(), span_starts.end()));
    using Range = std::tuple<std::uintptr_t, std::uint64_t, std::uint64_t>;
    std::vector<Range> writes;
    writes.reserve(spans.size());
    for (const auto &s : spans) {
      if (s.buffer_index >= storage.size())
        throw std::out_of_range(
            "BTC TPU DMA buffer index outside registered storages");
      const auto &format = region->formats[s.buffer_index];
      if (!fits(s.device_offset, s.bytes, format.bytes) ||
          !fits(s.host_offset, s.bytes, region->host_bytes))
        throw std::out_of_range(
            "BTC TPU DMA byte interval outside registered storage");
      // Donating model programs can replace a carrier's PJRT allocation.
      // Resolve it on the runner thread. All native batches use this exact
      // alias, even when the queue starts their copies after submission.
      auto &live = storage[s.buffer_index];
      if (!live) {
        try {
          live.emplace(
              region->resolveStorage(region->carriers[s.buffer_index]));
        } catch (const std::exception &e) {
          throw TpuDmaAdmissionError(e.what());
        }
        if (!live->format.matches(format))
          throw TpuDmaAdmissionError(
              "BTC TPU carrier changed physical layout after registration");
      }
      const bool h2d = direction == TpuCopyDirection::HostToDevice;
      const auto start =
          static_cast<std::uint64_t>(h2d ? s.device_offset : s.host_offset);
      writes.emplace_back(h2d ? reinterpret_cast<std::uintptr_t>(live->identity)
                              : 0,
                          start, start + s.bytes);
    }
    std::sort(writes.begin(), writes.end());
    for (std::size_t i = 1; i < writes.size(); ++i) {
      const auto &[prior, begin, end] = writes[i - 1];
      const auto &[current, next, next_end] = writes[i];
      if (prior == current && end > next)
        throw std::invalid_argument(
            "BTC TPU DMA batch has overlapping destination ranges");
    }
  }

  std::shared_ptr<TpuDmaRegion> region;
  TpuCopyDirection direction;
  std::vector<TpuCopySpan> spans;
  std::vector<std::size_t> span_starts;
  std::vector<std::optional<Storage>> storage;
};

TpuPageDma::TpuPageDma(const std::vector<at::Tensor> &device_buffers,
                       at::Tensor host_region, std::int64_t num_blocks)
    : region_(std::make_shared<TpuDmaRegion>(
          device_buffers, std::move(host_region), num_blocks)) {}

std::unique_ptr<TpuDmaJob>
TpuPageDma::prepare(TpuCopyDirection direction,
                    std::vector<TpuCopySpan> spans,
                    std::vector<std::size_t> span_starts) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (closing_)
    throw std::logic_error("BTC TPU DMA registration is closed");
  return std::unique_ptr<TpuDmaJob>(new TpuDmaJob(
      std::make_unique<TpuDmaJob::State>(region_, direction, std::move(spans),
                                         std::move(span_starts))));
}

std::vector<std::size_t> TpuPageDma::bufferSizes() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (closing_)
    throw std::logic_error("BTC TPU DMA registration is closed");
  std::vector<std::size_t> sizes;
  sizes.reserve(region_->formats.size());
  for (const auto &format : region_->formats)
    sizes.push_back(format.bytes);
  return sizes;
}

void TpuPageDma::close() {
  std::lock_guard<std::mutex> lock(mutex_);
  closing_ = true;
  if (!region_)
    return;
  if (region_.use_count() != 1)
    throw std::logic_error(
        "release all BTC TPU DMA jobs before closing the registration");
  check(region_->api, region_->unmap(), "BTC Host DmaUnmap");
  region_.reset();
}

TpuDmaJob::TpuDmaJob(std::unique_ptr<State> state)
    : state_(std::move(state)) {}

TpuDmaJob::~TpuDmaJob() = default;

void TpuDmaJob::submit(std::size_t first_page, std::size_t page_count,
                       Completion done, void *context) const noexcept {
  const auto &s = *state_;
  assert(first_page < s.span_starts.size() - 1 &&
         page_count && page_count < s.span_starts.size() - first_page);
  const auto begin = s.span_starts[first_page];
  const auto end = s.span_starts[first_page + page_count];
  if (begin == end) {
    done(context, true);
    return;
  }
  PjrtBatchEvents *events;
  try {
    events = new PjrtBatchEvents(s.region->api, end - begin, done, context);
  } catch (...) {
    done(context, false);  // No PJRT call was made.
    return;
  }
  for (std::size_t i = begin; i < end; ++i) {
    const auto &span = s.spans[i];
    auto *raw = s.storage[span.buffer_index]->raw.get();
    auto *host = s.region->base + span.host_offset;
    PJRT_Error *error = nullptr;
    PJRT_Event *event = nullptr;
    if (s.direction == TpuCopyDirection::HostToDevice) {
      PJRT_RawBuffer_CopyRawHostToDevice_Args args{
          PJRT_RawBuffer_CopyRawHostToDevice_Args_STRUCT_SIZE,
          nullptr,
          raw,
          host,
          span.device_offset,
          span.bytes,
          nullptr};
      error = s.region->extension->PJRT_RawBuffer_CopyRawHostToDevice(&args);
      event = args.event;
    } else {
      PJRT_RawBuffer_CopyRawDeviceToHost_Args args{
          PJRT_RawBuffer_CopyRawDeviceToHost_Args_STRUCT_SIZE,
          nullptr,
          raw,
          host,
          span.device_offset,
          span.bytes,
          nullptr};
      error = s.region->extension->PJRT_RawBuffer_CopyRawDeviceToHost(&args);
      event = args.event;
    }
    const bool launched = events->record(error);
    // A launch error leaves earlier segments and any returned event to drain.
    if (launched && !event) {
      // Successful async submission without the required completion
      // event violates the PJRT ABI. Returning a receipt would authorize
      // reuse of a buffer that may still have a writer.
      std::fputs("BTC: PJRT accepted a DMA without a completion event\n",
                 stderr);
      std::terminate();
    }
    const bool tracked = event && events->track(event);
    if (!launched || !tracked)
      break;
  }
  events->release();
}

} // namespace vllm_torchtpu::btc

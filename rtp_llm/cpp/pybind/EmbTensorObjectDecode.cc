#include "rtp_llm/cpp/pybind/EmbTensorCodec.h"

#if USING_CUDA

#include "rtp_llm/cpp/pybind/EmbNativeObjectApi.h"
#include "rtp_llm/cpp/pybind/EmbRegisteredArena.h"
#include "rtp_llm/cpp/pybind/EmbRegisteredWrite.h"

#include <c10/util/ScopeExit.h>
#include <rapidjson/document.h>

#include <array>
#include <chrono>
#include <optional>
#include <string_view>
#include <unordered_set>

namespace rtp_llm {
namespace {

struct UnsupportedLayout {};

std::string_view stringValue(const rapidjson::Value& value) {
    if (!value.IsString()) {
        throw UnsupportedLayout{};
    }
    return std::string_view(value.GetString(), value.GetStringLength());
}

const rapidjson::Value& field(const rapidjson::Value& obj, const char* key) {
    if (!obj.IsObject()) {
        throw UnsupportedLayout{};
    }
    const auto it = obj.FindMember(key);
    if (it == obj.MemberEnd()) {
        throw UnsupportedLayout{};
    }
    return it->value;
}

int64_t integral(const rapidjson::Value& value) {
    if (!value.IsInt64() || value.GetInt64() < 0) {
        throw UnsupportedLayout{};
    }
    return value.GetInt64();
}

at::ScalarType dtype(const rapidjson::Value& entry) {
    static constexpr std::pair<std::string_view, at::ScalarType> supported[] = {
        {"torch.bool", at::kBool},
        {"torch.uint8", at::kByte},
        {"torch.int8", at::kChar},
        {"torch.int16", at::kShort},
        {"torch.int32", at::kInt},
        {"torch.int64", at::kLong},
        {"torch.float16", at::kHalf},
        {"torch.bfloat16", at::kBFloat16},
        {"torch.float32", at::kFloat},
        {"torch.float64", at::kDouble},
    };
    const auto name = stringValue(entry);
    for (const auto& [label, type] : supported) {
        if (name == label) {
            return type;
        }
    }
    throw UnsupportedLayout{};
}

void validateTree(const rapidjson::Value& node, size_t depth, size_t& nodes, std::vector<bool>& referenced) {
    if (!node.IsArray() || node.Size() != 2 || depth > 32 || ++nodes > 4096) {
        throw UnsupportedLayout{};
    }
    const auto  kind    = stringValue(node[0]);
    const auto& content = node[1];
    if (kind == "tensor") {
        const auto index = integral(content);
        if (index >= static_cast<int64_t>(referenced.size())) {
            throw UnsupportedLayout{};
        }
        referenced[index] = true;
    } else if (kind == "tuple" || kind == "list") {
        if (!content.IsArray()) {
            throw UnsupportedLayout{};
        }
        for (const auto& child : content.GetArray()) {
            validateTree(child, depth + 1, nodes, referenced);
        }
    } else if (kind == "dict") {
        if (!content.IsArray()) {
            throw UnsupportedLayout{};
        }
        std::unordered_set<std::string_view> keys;
        for (const auto& item : content.GetArray()) {
            if (!item.IsArray() || item.Size() != 2 || !keys.insert(stringValue(item[0])).second) {
                throw UnsupportedLayout{};
            }
            validateTree(item[1], depth + 1, nodes, referenced);
        }
    } else if (kind != "scalar"
               || !(content.IsNull() || content.IsString() || content.IsBool() || content.IsInt64()
                    || content.IsUint64())) {
        // Python can represent integers beyond uint64. RapidJSON can parse
        // them as doubles, so defer those and floating scalars to Python.
        throw UnsupportedLayout{};
    }
}

pybind11::object toPython(const rapidjson::Value& node, const std::vector<pybind11::object>& tensors) {
    const auto  kind    = stringValue(node[0]);
    const auto& content = node[1];
    if (kind == "tensor") {
        return tensors[static_cast<size_t>(content.GetInt64())];
    }
    if (kind == "tuple" || kind == "list") {
        pybind11::list children;
        for (const auto& child : content.GetArray()) {
            children.append(toPython(child, tensors));
        }
        if (kind == "tuple") {
            return pybind11::tuple(children);
        }
        return children;
    }
    if (kind == "dict") {
        pybind11::dict result;
        for (const auto& item : content.GetArray()) {
            result[pybind11::str(item[0].GetString(), item[0].GetStringLength())] = toPython(item[1], tensors);
        }
        return result;
    }
    if (content.IsNull()) {
        return pybind11::none();
    }
    if (content.IsString()) {
        return pybind11::str(content.GetString(), content.GetStringLength());
    }
    if (content.IsBool()) {
        return pybind11::bool_(content.GetBool());
    }
    if (content.IsInt64()) {
        return pybind11::int_(content.GetInt64());
    }
    if (content.IsUint64()) {
        return pybind11::int_(content.GetUint64());
    }
    throw UnsupportedLayout{};
}

std::vector<EmbTensorSlice> validateLayout(rapidjson::Document&              document,
                                           const std::string&                header,
                                           const at::Tensor&                 input,
                                           const std::optional<std::string>& expected,
                                           bool                              restoreDevices) {
    constexpr auto parseFlags =
        rapidjson::kParseIterativeFlag | rapidjson::kParseValidateEncodingFlag | rapidjson::kParseFullPrecisionFlag;
    if (header.find(char{0}) != std::string::npos
        || document.Parse<parseFlags>(header.data(), header.size()).HasParseError() || !document.IsObject()
        || document.MemberCount() != 2) {
        throw UnsupportedLayout{};
    }
    const auto& tree  = field(document, "tree");
    const auto& table = field(document, "tensors");
    if (!table.IsArray() || table.Size() > 4096) {
        throw UnsupportedLayout{};
    }
    size_t            nodes = 0;
    std::vector<bool> referenced(table.Size());
    validateTree(tree, 0, nodes, referenced);
    if (std::find(referenced.begin(), referenced.end(), false) != referenced.end()) {
        throw UnsupportedLayout{};
    }
    if (expected) {
        if (stringValue(tree[0]) != "dict") {
            throw UnsupportedLayout{};
        }
        bool keyMatched = false;
        for (const auto& item : tree[1].GetArray()) {
            if (stringValue(item[0]) == "cache_key" && stringValue(item[1][0]) == "scalar" && item[1][1].IsString()) {
                keyMatched = stringValue(item[1][1]) == *expected;
            }
        }
        if (!keyMatched) {
            throw UnsupportedLayout{};
        }
    }
    const int64_t               payloadOffset = (16 + static_cast<int64_t>(header.size()) + 63) / 64 * 64;
    const int64_t               payloadSize   = input.numel() - payloadOffset;
    int64_t                     end           = 0;
    std::vector<EmbTensorSlice> slices;
    slices.reserve(table.Size());
    for (const auto& desc : table.GetArray()) {
        const auto& dims = field(desc, "shape");
        if (desc.MemberCount() != 5 || !dims.IsArray() || dims.Size() > 32) {
            throw UnsupportedLayout{};
        }
        const auto type   = dtype(field(desc, "dtype"));
        const auto offset = integral(field(desc, "offset"));
        const auto nbytes = integral(field(desc, "nbytes"));
        const auto device = stringValue(field(desc, "device"));
        if ((device != "cpu" && device != "cuda") || offset != (end + 63) / 64 * 64 || offset > payloadSize
            || nbytes > payloadSize - offset) {
            throw UnsupportedLayout{};
        }
        std::vector<int64_t> shape;
        shape.reserve(dims.Size());
        for (const auto& dim : dims.GetArray()) {
            const auto size = integral(dim);
            if (size > (INT64_C(1) << 31)) {
                throw UnsupportedLayout{};
            }
            shape.push_back(size);
        }
        auto expectedBytes = static_cast<int64_t>(c10::elementSize(type));
        if (std::find(shape.begin(), shape.end(), 0) != shape.end()) {
            expectedBytes = 0;
        } else {
            for (const auto dim : shape) {
                if (dim > nbytes / expectedBytes) {
                    throw UnsupportedLayout{};
                }
                expectedBytes *= dim;
            }
        }
        if (expectedBytes != nbytes) {
            throw UnsupportedLayout{};
        }
        end = offset + nbytes;
        slices.emplace_back(payloadOffset + offset,
                            nbytes,
                            type,
                            std::move(shape),
                            restoreDevices && device == "cuda" ? input.device() : at::Device(at::kCPU));
    }
    if (end != payloadSize) {
        throw UnsupportedLayout{};
    }
    return slices;
}

template<typename Stamp>
std::optional<std::vector<at::Tensor>> restoreObjectPayload(const at::Tensor&                 input,
                                                            const std::optional<std::string>& expected,
                                                            bool                              restoreDevices,
                                                            const std::optional<int64_t>&     consumerStreamId,
                                                            std::optional<int32_t>            headerHint,
                                                            rapidjson::Document&              document,
                                                            std::string&                      header,
                                                            EmbHeaderReadStats*               headerStats,
                                                            Stamp                             stamp,
                                                            bool* payloadStarted = nullptr) {
    header = readEmbCudaObjectHeader(input, headerHint, headerStats);
    stamp(1);
    std::optional<std::vector<EmbTensorSlice>> slices;
    try {
        slices = validateLayout(document, header, input, expected, restoreDevices);
    } catch (const UnsupportedLayout&) {
        // No payload copy has been submitted.
    }
    stamp(2);
    if (!slices) {
        return std::nullopt;
    }
    if (payloadStarted) {
        *payloadStarted = true;
    }
    auto decoded = restoreEmbTensorSlices(input, *slices);
    if (consumerStreamId) {
        const auto consumer = c10::Stream::unpack3(*consumerStreamId, input.get_device(), c10::kCUDA);
        for (const auto& tensor : decoded) {
            if (tensor.is_cuda()) {
                tensor.record_stream(consumer);
            }
        }
    }
    waitEmbCudaStream(input);
    return decoded;
}

pybind11::object buildObjectTree(const rapidjson::Document& document, const std::vector<at::Tensor>& decoded) {
    std::vector<pybind11::object> tensors;
    tensors.reserve(decoded.size());
    for (const auto& tensor : decoded) {
        tensors.push_back(pybind11::cast(tensor));
    }
    return toPython(field(document, "tree"), tensors);
}

template<bool observe, bool profileHeader = false>
pybind11::tuple restoreObject(const at::Tensor&                 input,
                              const std::optional<std::string>& expected,
                              bool                              restoreDevices,
                              const std::optional<int64_t>&     consumerStreamId,
                              std::optional<int32_t>            headerHint) {
    using Clock = std::chrono::steady_clock;
    std::array<Clock::time_point, 6> ticks;
    const auto                       stamp = [&](size_t index) {
        if constexpr (observe) {
            ticks[index] = Clock::now();
        }
    };
    stamp(0);
    rapidjson::Document                    document;
    std::string                            header;
    EmbHeaderReadStats                     headerStats;
    std::optional<std::vector<at::Tensor>> decoded;
    {
        pybind11::gil_scoped_release release;
        const c10::cuda::CUDAGuard   deviceGuard(input.device());
        decoded = restoreObjectPayload(input,
                                       expected,
                                       restoreDevices,
                                       consumerStreamId,
                                       headerHint,
                                       document,
                                       header,
                                       profileHeader ? &headerStats : nullptr,
                                       stamp);
        stamp(3);
    }
    stamp(4);
    const auto finish = [&](pybind11::object value, pybind11::object fallback) -> pybind11::tuple {
        if constexpr (observe) {
            stamp(5);
            pybind11::object timing = pybind11::none();
            // Capture only slow successful native restores. The host-resume
            // component includes GIL reacquisition after the CUDA fence.
            if (decoded && ticks[5] - ticks[0] >= std::chrono::milliseconds(25)) {
                std::array<int64_t, 5> nanoseconds;
                for (size_t i = 0; i < nanoseconds.size(); ++i) {
                    nanoseconds[i] =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(ticks[i + 1] - ticks[i]).count();
                }
                timing = pybind11::cast(nanoseconds);
            }
            if constexpr (profileHeader) {
                return pybind11::make_tuple(
                    std::move(value),
                    std::move(fallback),
                    std::move(timing),
                    static_cast<int64_t>(header.size()),
                    headerStats.bytes,
                    headerStats.copies);
            }
            return pybind11::make_tuple(std::move(value), std::move(fallback), std::move(timing));
        } else {
            return pybind11::make_tuple(std::move(value), std::move(fallback));
        }
    };
    if (!decoded) {
        return finish(pybind11::none(), pybind11::bytes(header));
    }
    return finish(buildObjectTree(document, *decoded), pybind11::none());
}

enum class EmbReadState {
    Success,
    Expired,
    LeaseUnavailable,
    LayoutFallback,
    InvalidObject,
    CudaFailure,
    SdkFailure
};

using ReadClock = std::chrono::steady_clock;

int64_t readElapsed(ReadClock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(ReadClock::now() - start).count();
}

struct EmbReadResult {
    EmbReadState                           state         = EmbReadState::LeaseUnavailable;
    int32_t                                code          = 0;
    bool                                   inflight      = false;
    int64_t                                acquireNs     = 0;
    int64_t                                sdkNs         = 0;
    int64_t                                restoreNs     = 0;
    int64_t                                headerNs      = 0;
    int64_t                                heldNs        = 0;
    int64_t                                acquireLockNs = 0;
    int64_t                                releaseLockNs = 0;
    EmbHeaderReadStats                     headerStats;
    std::string                            header;
    rapidjson::Document                    document;
    std::optional<std::vector<at::Tensor>> decoded;
    at::Tensor                             fallback;
};

const KvcmObjectNativeLoadApiV1* loadApi(pybind11::handle capsule) {
    const auto* api = static_cast<const KvcmObjectNativeLoadApiV1*>(
        PyCapsule_GetPointer(capsule.ptr(), KVCM_OBJECT_NATIVE_LOAD_CAPSULE_V1));
    if (!api) {
        throw pybind11::error_already_set();
    }
    if (api->version != KVCM_OBJECT_NATIVE_LOAD_VERSION_1 || api->struct_size < sizeof(*api) || !api->load
        || !api->context || !api->client_identity) {
        throw std::invalid_argument("invalid exact-object native load API");
    }
    return api;
}

EmbReadResult readRegisteredObject(EmbRegisteredArena&               arena,
                                   const at::Tensor&                 root,
                                   int64_t                           length,
                                   const KvcmObjectNativeLoadApiV1&  api,
                                   const std::string&                trace,
                                   const std::optional<std::string>& expected,
                                   const std::optional<int64_t>&     consumerStreamId,
                                   bool                              useHeaderHint,
                                   bool                              profileHeader,
                                   int64_t                           budgetNs,
                                   bool                              sampleLocks) {
    TORCH_CHECK(root.is_cuda() && root.is_contiguous() && root.scalar_type() == at::kByte && root.dim() == 1,
                "expected a registered CUDA byte arena");
    TORCH_CHECK(length > 0 && length <= root.numel() && static_cast<uint64_t>(length) == api.object_size
                    && budgetNs <= 600'000'000'000,
                "invalid native read size or deadline");
    EmbReadResult result;
    const auto    start = ReadClock::now();
    if (budgetNs <= 0) {
        result.state = EmbReadState::Expired;
        return result;
    }
    const auto deadline = start + std::chrono::nanoseconds(budgetNs);
    // Under the shared CUDA graph gate, only try an immediate acquisition.
    // The original bounded wait runs outside this gate on contention.
    const auto [acquired, ticket, offset, capacity, lockNs] =
        arena.tryAcquireForNativeRead(length, sampleLocks);
    if (acquired != EmbRegisteredArena::Result::Acquired) {
      return result;
    }
    const auto leased = ReadClock::now();
    bool reusable = true;
    auto cleanup = c10::make_scope_exit(
        [&] { arena.release(ticket, offset, reusable, false); });
    result.acquireNs =
        std::chrono::duration_cast<std::chrono::nanoseconds>(leased - start)
            .count();
    result.acquireLockNs = lockNs;
    if (leased >= deadline) {
      result.state = EmbReadState::Expired;
    } else {
      const auto input = root.narrow(0, offset, length);
      const KvcmObjectNativeLoadRequestV1 request{
          trace.data(), trace.size(), input.data_ptr(),
          static_cast<uint64_t>(length), 1};
      const auto sdkStart = ReadClock::now();
      const auto transfer = api.load(api.context, &request);
      result.sdkNs = readElapsed(sdkStart);
      reusable = !transfer.data_transfer_may_be_inflight;
      if (transfer.code || !reusable) {
        result.state = EmbReadState::SdkFailure;
        result.code = transfer.code;
      } else if (ReadClock::now() >= deadline) {
        result.state = EmbReadState::Expired;
      } else {
        const auto restoreStart = ReadClock::now();
        bool cudaPayloadStarted = false;
        std::optional<EmbTransportCudaStream> transport;
        const auto failCuda = [&] {
          reusable = false;
          result.state = EmbReadState::CudaFailure;
          if (transport) {
            transport->quarantine();
          }
          arena.failDevice();
        };
        try {
          const c10::cuda::CUDAGuard device(root.device());
          transport.emplace(root.get_device());
          const c10::cuda::CUDAStreamGuard stream(
              c10::cuda::getStreamFromExternal(
                  reinterpret_cast<cudaStream_t>(transport->address()),
                  root.get_device()));
          const int32_t hint =
              useHeaderHint && api.has_userdata1 ? api.userdata1 : 0;
          result.decoded = restoreObjectPayload(
              input, expected, true, consumerStreamId, hint, result.document,
              result.header, profileHeader ? &result.headerStats : nullptr,
              [&](size_t step) {
                if (step == 1 && profileHeader) {
                  result.headerNs = readElapsed(restoreStart);
                }
              },
              &cudaPayloadStarted);
          if (result.decoded) {
            result.state = EmbReadState::Success;
          } else {
            // The compatible Python parser may accept this layout.
            // Hand it owned bytes after fencing; the original MR is
            // already releasable when Python starts parsing.
            cudaPayloadStarted = true;
            result.fallback = readEmbCudaHeader(input, 0, length);
            result.state = EmbReadState::LayoutFallback;
          }
        } catch (const std::invalid_argument &) {
          if (cudaPayloadStarted) {
            failCuda();
          } else {
            result.state = EmbReadState::InvalidObject;
          }
        } catch (...) {
          failCuda();
        }
        result.restoreNs = readElapsed(restoreStart);
      }
    }
    result.inflight      = !reusable;
    result.heldNs        = readElapsed(leased);
    result.releaseLockNs = arena.release(ticket, offset, reusable, sampleLocks);
    cleanup.release();
    return result;
}

}  // namespace

void registerEmbTensorObjectDecode(pybind11::module& module) {
    namespace py    = pybind11;
    const auto bind = [&](const char* name, auto restore) {
        module.def(name,
                   restore,
                   py::arg("buffer"),
                   py::arg("expected_key"),
                   py::arg("restore_devices"),
                   py::arg("consumer_stream_id"),
                   py::arg("header_hint") = py::none());
    };
    bind("restore_emb_cuda_object", &restoreObject<false>);
    bind("restore_emb_cuda_object_observed", &restoreObject<true>);
    bind("restore_emb_cuda_object_profiled", &restoreObject<true, true>);
}

void registerEmbRegisteredRead(pybind11::module& module) {
    namespace py = pybind11;
    py::enum_<EmbReadState>(module, "EmbRegisteredReadState")
        .value("Success", EmbReadState::Success)
        .value("Expired", EmbReadState::Expired)
        .value("LeaseUnavailable", EmbReadState::LeaseUnavailable)
        .value("LayoutFallback", EmbReadState::LayoutFallback)
        .value("InvalidObject", EmbReadState::InvalidObject)
        .value("CudaFailure", EmbReadState::CudaFailure)
        .value("SdkFailure", EmbReadState::SdkFailure);
    module.def("read_emb_registered_object",
               [](EmbRegisteredArena&               arena,
                  const at::Tensor&                 root,
                  int64_t                           length,
                  py::capsule                       metadata,
                  py::capsule                       registeredClient,
                  const std::string&                trace,
                  const std::optional<std::string>& expected,
                  const std::optional<int64_t>&     consumerStreamId,
                  bool                              useHeaderHint,
                  bool                              profileHeader,
                  int64_t                           budgetNs,
                  bool                              sampleLocks) {
                   const auto* meta   = loadApi(metadata);
                   const auto* client = static_cast<const KvcmObjectNativeApiV2*>(
                       PyCapsule_GetPointer(registeredClient.ptr(), KVCM_OBJECT_NATIVE_API_CAPSULE_V2));
                   if (!client) {
                       throw py::error_already_set();
                   }
                   if (!validEmbNativeWriteApi(client) || client->context != meta->client_identity) {
                       throw std::invalid_argument("native EMB metadata belongs to a different client");
                   }
                   EmbReadResult result;
                   {
                       py::gil_scoped_release release;
                       result = readRegisteredObject(arena,
                                                     root,
                                                     length,
                                                     *meta,
                                                     trace,
                                                     expected,
                                                     consumerStreamId,
                                                     useHeaderHint,
                                                     profileHeader,
                                                     budgetNs,
                                                     sampleLocks);
                   }
                   py::object value = py::none();
                   if (result.decoded) {
                       value = buildObjectTree(result.document, *result.decoded);
                   } else if (result.fallback.defined()) {
                       value = py::cast(std::move(result.fallback));
                   }
                   return py::make_tuple(result.state,
                                         std::move(value),
                                         result.code,
                                         result.inflight,
                                         result.acquireNs,
                                         result.sdkNs,
                                         result.restoreNs,
                                         result.heldNs,
                                         result.acquireLockNs,
                                         result.releaseLockNs,
                                         result.header.size(),
                                         result.headerStats.bytes,
                                         result.headerStats.copies,
                                         result.headerNs);
               });
}

}  // namespace rtp_llm

#endif

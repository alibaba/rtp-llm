#include "rtp_llm/cpp/pybind/EmbTensorCodec.h"

#if USING_CUDA

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

template<bool observe>
pybind11::tuple restoreObject(const at::Tensor&                 input,
                              const std::optional<std::string>& expected,
                              bool                              restoreDevices,
                              const std::optional<int64_t>&     consumerStreamId) {
    using Clock = std::chrono::steady_clock;
    std::array<Clock::time_point, 6> ticks;
    const auto stamp = [&](size_t index) {
        if constexpr (observe) {
            ticks[index] = Clock::now();
        }
    };
    stamp(0);
    rapidjson::Document                    document;
    std::string                            header;
    std::optional<std::vector<at::Tensor>> decoded;
    {
        pybind11::gil_scoped_release release;
        const c10::cuda::CUDAGuard   deviceGuard(input.device());
        header = readEmbCudaObjectHeader(input);
        stamp(1);
        std::optional<std::vector<EmbTensorSlice>> slices;
        try {
            slices = validateLayout(document, header, input, expected, restoreDevices);
        } catch (const UnsupportedLayout&) {
            // Return the bytes already fetched. Python keeps its validation and
            // full JSON compatibility; nothing has used the payload yet.
        }
        stamp(2);
        if (slices) {
            decoded = restoreEmbTensorSlices(input, *slices);
            if (consumerStreamId) {
                const auto consumer = c10::Stream::unpack3(*consumerStreamId, input.get_device(), c10::kCUDA);
                for (const auto& tensor : *decoded) {
                    if (tensor.is_cuda()) {
                        tensor.record_stream(consumer);
                    }
                }
            }
            waitEmbCudaStream(input);
        }
        stamp(3);
    }
    stamp(4);
    const auto finish = [&](pybind11::object value, pybind11::object fallback) -> pybind11::tuple {
        if constexpr (observe) {
            stamp(5);
            // Capture only slow successful native restores. The host-resume
            // component includes GIL reacquisition after the CUDA fence.
            if (decoded && ticks[5] - ticks[0] >= std::chrono::milliseconds(25)) {
                std::array<int64_t, 5> nanoseconds;
                for (size_t i = 0; i < nanoseconds.size(); ++i) {
                    nanoseconds[i] =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(ticks[i + 1] - ticks[i]).count();
                }
                return pybind11::make_tuple(std::move(value), std::move(fallback), nanoseconds);
            }
            return pybind11::make_tuple(std::move(value), std::move(fallback), pybind11::none());
        } else {
            return pybind11::make_tuple(std::move(value), std::move(fallback));
        }
    };
    if (!decoded) {
        return finish(pybind11::none(), pybind11::bytes(header));
    }
    std::vector<pybind11::object> tensors;
    tensors.reserve(decoded->size());
    for (const auto& tensor : *decoded) {
        tensors.push_back(pybind11::cast(tensor));
    }
    return finish(toPython(field(document, "tree"), tensors), pybind11::none());
}

}  // namespace

void registerEmbTensorObjectDecode(pybind11::module& module) {
    module.def("restore_emb_cuda_object", &restoreObject<false>);
    module.def("restore_emb_cuda_object_observed", &restoreObject<true>);
}

}  // namespace rtp_llm

#endif

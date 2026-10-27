#include <ATen/ops/as_strided.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/cat.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/autograd/python_variable.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <dlfcn.h>
#include <tuple>
#include <numeric>
#include <vector>

namespace py = pybind11;

py::tuple specializationKey(py::tuple args) {
    py::tuple result(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
        auto object = args[i];
        if (THPVariable_Check(object.ptr())) {
            const auto& tensor = THPVariable_Unpack(object.ptr());
            result[i]          = py::make_tuple(0,
                                       static_cast<int>(tensor.scalar_type()),
                                       tensor.device().index(),
                                       reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16);
        } else {
            result[i] = py::make_tuple(py::type::of(object), object);
        }
    }
    return result;
}

union LaunchValue {
    void*    pointer;
    int32_t  i32;
    uint32_t u32;
    int64_t  i64;
    uint64_t u64;
    float    f32;
    double   f64;
};

class NativeLauncher {
public:
    NativeLauncher(
        uint64_t function, int shared, int warps, std::vector<int> positions, std::vector<int> types, py::object owner):
        positions_(std::move(positions)),
        types_(std::move(types)),
        function_(reinterpret_cast<void*>(function)),
        shared_(shared),
        warps_(warps),
        owner_(std::move(owner)) {
        TORCH_CHECK(positions_.size() == types_.size() && positions_.size() <= 64);
    }

    void execute(unsigned x, unsigned y, unsigned z, uint64_t stream, LaunchValue* values) const {
        if (!x || !y || !z)
            return;
        using Launch =
            int (*)(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void*, void**, void**);
        static Launch launch = [] {
            auto library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
            TORCH_CHECK(library, "Cannot load CUDA driver: ", dlerror());
            auto function = reinterpret_cast<Launch>(dlsym(library, "cuLaunchKernel"));
            TORCH_CHECK(function, "Cannot resolve cuLaunchKernel");
            return function;
        }();
        std::array<void*, 66> params;
        for (size_t i = 0; i < positions_.size(); ++i)
            params[i] = &values[i];
        // Triton 3.6 appends these two ABI parameters even when scratch sizes
        // are zero. The Python support gate pins this executor to that ABI.
        void* scratch                 = nullptr;
        params[positions_.size()]     = &scratch;
        params[positions_.size() + 1] = &scratch;
        int status                    = launch(
            function_, x, y, z, warps_ * 32, 1, 1, shared_, reinterpret_cast<void*>(stream), params.data(), nullptr);
        TORCH_CHECK(status == 0, "CUDA driver launch failed, status=", status);
    }

    void launch(py::tuple args, py::tuple grid, uint64_t stream) const {
        TORCH_CHECK(grid.size() == 3);
        std::array<LaunchValue, 64> values;
        for (size_t i = 0; i < positions_.size(); ++i) {
            auto object = args[positions_[i]];
            switch (types_[i]) {
                case 0:
                    if (THPVariable_Check(object.ptr()))
                        values[i].pointer = THPVariable_Unpack(object.ptr()).data_ptr();
                    else if (object.is_none())
                        values[i].pointer = nullptr;
                    else
                        values[i].pointer = reinterpret_cast<void*>(py::cast<uint64_t>(object));
                    break;
                case 1:
                    values[i].i32 = py::cast<int32_t>(object);
                    break;
                case 2:
                    values[i].u32 = py::cast<uint32_t>(object);
                    break;
                case 3:
                    values[i].i64 = py::cast<int64_t>(object);
                    break;
                case 4:
                    values[i].u64 = py::cast<uint64_t>(object);
                    break;
                case 5:
                    values[i].f32 = py::cast<float>(object);
                    break;
                case 6:
                    values[i].f64 = py::cast<double>(object);
                    break;
                default:
                    TORCH_CHECK(false, "Unsupported launch scalar type");
            }
        }
        execute(py::cast<unsigned>(grid[0]),
                py::cast<unsigned>(grid[1]),
                py::cast<unsigned>(grid[2]),
                stream,
                values.data());
    }
    std::vector<int> positions_, types_;
    at::Tensor       readKv(const at::Tensor& cache, const at::Tensor& slots, int64_t rows, uint64_t stream) const {
        if (!(cache.is_cuda() && cache.scalar_type() == at::kByte && cache.dim() == 3 && cache.size(1) == 1
              && cache.size(2) == 656 && cache.is_contiguous() && slots.device() == cache.device()
              && slots.is_contiguous() && (slots.scalar_type() == at::kInt || slots.scalar_type() == at::kLong)
              && slots.numel() <= rows && rows >= 0))
            return {};
        if (reinterpret_cast<uintptr_t>(cache.data_ptr()) % 16 || reinterpret_cast<uintptr_t>(slots.data_ptr()) % 16)
            return {};
        TORCH_CHECK(positions_ == std::vector<int>({0, 1, 2, 3}) && types_.size() == 4);
        auto n = slots.numel();
        if (types_[3] == 1 && n > INT32_MAX)
            return {};
        TORCH_CHECK(types_[3] == 1 || types_[3] == 3);
        auto                       out = at::empty({rows, 576}, cache.options().dtype(at::kBFloat16));
        std::array<LaunchValue, 4> args;
        args[0].pointer = cache.data_ptr();
        args[1].pointer = slots.data_ptr();
        args[2].pointer = out.data_ptr();
        if (types_[3] == 1)
            args[3].i32 = n;
        else
            args[3].i64 = n;
        execute(rows, 1, 1, stream, args.data());
        return out;
    }
    bool indexedCopy(const at::Tensor& src,
                     const at::Tensor& si,
                     const at::Tensor& di,
                     const at::Tensor& out,
                     uint64_t          stream) const {
        if (!(src.is_cuda() && src.dim() == 2 && out.dim() == 2 && src.device() == out.device()
              && src.device() == si.device() && src.device() == di.device() && src.scalar_type() == out.scalar_type()
              && (src.scalar_type() == at::kBFloat16 || src.scalar_type() == at::kHalf
                  || src.scalar_type() == at::kFloat)
              && src.stride(1) == 1 && out.stride(1) == 1 && src.size(1) == out.size(1) && src.size(1) > 0
              && src.stride(0) >= src.size(1) && out.stride(0) >= out.size(1) && si.dim() == 1 && di.dim() == 1
              && si.is_contiguous() && di.is_contiguous()
              && (si.scalar_type() == at::kInt || si.scalar_type() == at::kLong)
              && (di.scalar_type() == at::kInt || di.scalar_type() == at::kLong) && si.numel() == di.numel()
              && !src.requires_grad() && !out.requires_grad()))
            return false;
        auto storage = out.storage().data_ptr().get();
        if (storage == src.storage().data_ptr().get() || storage == si.storage().data_ptr().get()
            || storage == di.storage().data_ptr().get())
            return false;
        for (auto* tensor : {&src, &si, &di, &out})
            if (reinterpret_cast<uintptr_t>(tensor->data_ptr()) % 16)
                return false;
        TORCH_CHECK(positions_ == std::vector<int>({0, 1, 2, 3, 4}) && types_.size() == 5);
        auto n = si.numel();
        if (types_[4] == 1 && n > INT32_MAX)
            return false;
        TORCH_CHECK(types_[4] == 1 || types_[4] == 3);
        std::array<LaunchValue, 5> args;
        args[0].pointer = src.data_ptr();
        args[1].pointer = si.data_ptr();
        args[2].pointer = di.data_ptr();
        args[3].pointer = out.data_ptr();
        if (types_[4] == 1)
            args[4].i32 = n;
        else
            args[4].i64 = n;
        execute((n * src.size(1) + 1023) / 1024, 1, 1, stream, args.data());
        return true;
    }

private:
    void* function_;
    int   shared_, warps_;
    // Prepared executors can outlive the bounded Python dispatch cache.
    // Retain the compiled module, never tensor arguments or stream objects.
    py::object owner_;
};

std::tuple<at::Tensor, at::Tensor> routerOutputs(int64_t, int64_t, int64_t, bool);

class NativeRouter {
public:
    NativeRouter(std::shared_ptr<NativeLauncher> route, std::shared_ptr<NativeLauncher> pad):
        route_(std::move(route)), pad_(std::move(pad)) {
        TORCH_CHECK(route_->positions_ == std::vector<int>({0, 1, 2, 3})
                    && route_->types_ == std::vector<int>({0, 0, 0, 0}));
        if (pad_)
            TORCH_CHECK(pad_->positions_ == std::vector<int>({0, 1, 2, 3})
                        && pad_->types_ == std::vector<int>({0, 0, 1, 1}));
    }

    std::tuple<at::Tensor, at::Tensor> run(const at::Tensor& input,
                                           const at::Tensor& weight,
                                           const at::Tensor& bias,
                                           int64_t           chunk_rows,
                                           bool              ids64,
                                           uint64_t          stream) const {
        TORCH_CHECK(input.is_cuda() && input.dim() == 2 && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16);
        TORCH_CHECK(weight.sizes() == at::IntArrayRef({256, input.size(1)}));
        TORCH_CHECK(bias.scalar_type() == at::kFloat && bias.numel() == 256 && bias.is_contiguous());
        TORCH_CHECK(weight.device() == input.device() && bias.device() == input.device());
        const auto    rows = input.size(0), width = input.size(1);
        const int64_t step = chunk_rows > 0 ? chunk_rows : std::max<int64_t>(1, rows);
        TORCH_CHECK(step * width < INT32_MAX && rows > 0);
        std::vector<at::Tensor> chunks;
        for (int64_t begin = 0; begin < rows; begin += step) {
            auto count = std::min(step, rows - begin);
            auto chunk = input.narrow(0, begin, count);
            if (count != step) {
                TORCH_CHECK(pad_);
                auto                       padded = at::empty({step, width}, input.options());
                std::array<LaunchValue, 4> args;
                args[0].pointer = chunk.data_ptr();
                args[1].pointer = padded.data_ptr();
                args[2].i32     = count * width;
                args[3].i32     = step * width;
                pad_->execute((step * width + 1023) / 1024, 1, 1, stream, args.data());
                chunk = padded;
            }
            chunks.push_back(at::linear(chunk, weight).narrow(0, 0, count));
        }
        auto                       logits  = chunks.size() == 1 ? chunks[0] : at::cat(chunks, 0);
        auto                       outputs = routerOutputs(rows, 8, input.device().index(), ids64);
        std::array<LaunchValue, 4> args;
        args[0].pointer = logits.data_ptr();
        args[1].pointer = bias.data_ptr();
        args[2].pointer = std::get<0>(outputs).data_ptr();
        args[3].pointer = std::get<1>(outputs).data_ptr();
        route_->execute(rows, 1, 1, stream, args.data());
        return outputs;
    }

private:
    std::shared_ptr<NativeLauncher> route_, pad_;
};

// Only allocates fresh tensors. No retained tensors, stream changes, or GPU work.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
normOutputs(int64_t rows, int64_t width, int64_t kv_width, int64_t device, bool ue8, bool dual) {
    TORCH_CHECK(rows >= 0 && width > 0 && width % 128 == 0 && kv_width >= 0);
    TORCH_CHECK(!ue8 || width % 512 == 0);
    auto    options = at::TensorOptions().device(c10::Device(c10::kCUDA, device));
    auto    bf16    = dual ? at::empty({rows, width}, options.dtype(at::kBFloat16)) : at::Tensor();
    auto    fp8     = at::empty({rows, width}, options.dtype(at::kFloat8_e4m3fn));
    int64_t padded  = (rows + 3) / 4 * 4;
    int64_t cols    = width / (ue8 ? 512 : 128);
    auto    base    = at::empty({cols, padded}, options.dtype(ue8 ? at::kInt : at::kFloat));
    auto    scale   = base.as_strided({rows, cols}, {1, std::max<int64_t>(1, padded)});
    auto    kv      = kv_width ? at::empty({rows, kv_width}, options.dtype(at::kBFloat16)) : at::Tensor();
    return {bf16, fp8, scale, kv};
}

std::tuple<at::Tensor, at::Tensor> routerOutputs(int64_t rows, int64_t topk, int64_t device, bool ids64) {
    TORCH_CHECK(rows >= 0 && topk > 0);
    auto options = at::TensorOptions().device(c10::Device(c10::kCUDA, device));
    return {at::empty({rows, topk}, options.dtype(at::kFloat)),
            at::empty({rows, topk}, options.dtype(ids64 ? at::kLong : at::kInt))};
}

py::dict prefixPlan(const std::vector<int64_t>& prefixes,
                    const std::vector<int64_t>& lengths,
                    int64_t                     size,
                    int64_t                     rank,
                    int64_t                     owner) {
    TORCH_CHECK(!prefixes.empty() && prefixes.size() == lengths.size() && size > 0 && rank >= 0 && rank < size
                && owner > 0);
    auto                 i64     = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
    auto                 i32     = i64.dtype(at::kInt);
    int64_t              history = 0, total = 0, padded = 0, actual = 0;
    std::vector<int64_t> pads, counts;
    for (size_t i = 0; i < prefixes.size(); ++i) {
        auto p = prefixes[i], q = lengths[i];
        TORCH_CHECK(p >= 0 && q >= 0);
        auto pad = (p + size * owner - 1) / (size * owner) * owner;
        auto count =
            p / (size * owner) * owner + std::min(owner, std::max<int64_t>(0, p % (size * owner) - rank * owner));
        TORCH_CHECK(count <= INT32_MAX && actual <= INT32_MAX);
        pads.push_back(pad);
        counts.push_back(count);
        history += p;
        total += q;
        padded += pad;
        actual += count;
    }
    auto restore = at::empty({history + total}, i64);
    auto scatter = at::empty({actual}, i64), dest = at::empty({total}, i64);
    auto new_restore     = at::empty({total}, i64);
    auto history_lengths = at::empty({static_cast<int64_t>(prefixes.size())}, i32);
    auto history_starts  = at::empty_like(history_lengths);
    auto rp = restore.data_ptr<int64_t>(), sp = scatter.data_ptr<int64_t>(), dp = dest.data_ptr<int64_t>();
    std::fill(rp, rp + history + total, int64_t(0));
    auto                              nr = new_restore.data_ptr<int64_t>();
    std::vector<std::vector<int64_t>> rows(size);
    int64_t                           pad_start = 0, full_start = 0, act_start = 0, new_start = 0;
    for (size_t i = 0; i < prefixes.size(); ++i) {
        auto p = prefixes[i], q = lengths[i];
        history_lengths.data_ptr<int32_t>()[i] = counts[i];
        history_starts.data_ptr<int32_t>()[i]  = act_start;
        for (int64_t block = 0; block * owner < p; ++block) {
            auto count = std::min(owner, p - block * owner);
            auto start = (block % size) * padded + pad_start + block / size * owner;
            std::iota(rp + full_start + block * owner, rp + full_start + block * owner + count, start);
        }
        for (int64_t t = 0; t < counts[i]; ++t)
            sp[act_start + t] = pad_start + t;
        for (int64_t t = 0; t < q; ++t) {
            dp[new_start + t] = p + t + full_start;
            rows[(p + t) / owner % size].push_back(new_start + t);
        }
        pad_start += pads[i];
        full_start += p + q;
        act_start += counts[i];
        new_start += q;
    }
    int64_t new_max = 0;
    for (auto& r : rows)
        new_max = std::max<int64_t>(new_max, r.size());
    for (int64_t r = 0; r < size; ++r)
        for (size_t j = 0; j < rows[r].size(); ++j)
            nr[rows[r][j]] = r * new_max + j;
    auto local_rows = at::empty({static_cast<int64_t>(rows[rank].size())}, i64);
    std::copy(rows[rank].begin(), rows[rank].end(), local_rows.data_ptr<int64_t>());
    py::dict result;
    result["history_lengths"] = history_lengths;
    result["history_starts"]  = history_starts;
    result["history_scatter"] = scatter;
    result["history_restore"] = restore;
    result["new_local_rows"]  = local_rows;
    result["new_restore"]     = new_restore;
    result["new_dest"]        = dest;
    result["history_padded"]  = padded;
    result["history_actual"]  = actual;
    result["new_max"]         = new_max;
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prefix_plan", &prefixPlan);
    m.def("norm_outputs", &normOutputs);
    m.def("router_outputs", &routerOutputs);
    m.def("specialization_key", &specializationKey);
    py::class_<NativeLauncher, std::shared_ptr<NativeLauncher>>(m, "NativeLauncher")
        .def(py::init<uint64_t, int, int, std::vector<int>, std::vector<int>, py::object>())
        .def("launch", &NativeLauncher::launch)
        .def("read_kv", &NativeLauncher::readKv)
        .def("indexed_copy", &NativeLauncher::indexedCopy);
    py::class_<NativeRouter>(m, "NativeRouter")
        .def(py::init<std::shared_ptr<NativeLauncher>, std::shared_ptr<NativeLauncher>>())
        .def("run", &NativeRouter::run);
}

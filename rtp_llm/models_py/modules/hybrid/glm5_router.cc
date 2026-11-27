#include <ATen/ops/empty.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/cat.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/autograd/python_variable.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <dlfcn.h>
#include <tuple>
#include <vector>

namespace py = pybind11;

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

    std::vector<int> positions_, types_;

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

std::tuple<at::Tensor, at::Tensor> routerOutputs(int64_t rows, int64_t topk, int64_t device, bool ids64) {
    TORCH_CHECK(rows >= 0 && topk > 0);
    auto options = at::TensorOptions().device(c10::Device(c10::kCUDA, device));
    return {at::empty({rows, topk}, options.dtype(at::kFloat)),
            at::empty({rows, topk}, options.dtype(ids64 ? at::kLong : at::kInt))};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<NativeLauncher, std::shared_ptr<NativeLauncher>>(m, "NativeLauncher")
        .def(py::init<uint64_t, int, int, std::vector<int>, std::vector<int>, py::object>());
    py::class_<NativeRouter>(m, "NativeRouter")
        .def(py::init<std::shared_ptr<NativeLauncher>, std::shared_ptr<NativeLauncher>>())
        .def("run", &NativeRouter::run);
}

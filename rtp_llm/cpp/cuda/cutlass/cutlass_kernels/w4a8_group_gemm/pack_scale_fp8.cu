#include "cute/numeric/integral_constant.hpp"
#include "cute/util/type_traits.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/device_memory.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/w4a8_group_gemm/w4a8_group_gemm.h"

template<typename T>
class packed_scale_t {
public:
    static_assert(cute::is_same_v<T, cutlass::float_e4m3_t> || cute::is_same_v<T, cutlass::float_e5m2_t>,
                  "only 8 bit arithmetic types are supported.");
    CUTLASS_HOST_DEVICE
    explicit packed_scale_t(T val) {
        if constexpr (!cute::is_unsigned_v<T>) {
            // Only pack negative values. The positive values are generated in flight in the mainloop.
            storage[0] = pack4(T(float(val) * -8.f), T(float(val) * -7.f), T(float(val) * -6.f), T(float(val) * -5.f));
            storage[1] = pack4(T(float(val) * -4.f), T(float(val) * -3.f), T(float(val) * -2.f), -val);
        } else {
            storage[0] = pack4(T(float(val) * 8.f), T(float(val) * 7.f), T(float(val) * 6.f), T(float(val) * 5.f));
            storage[1] = pack4(T(float(val) * 4.f), T(float(val) * 3.f), T(float(val) * 2.f), val);
        }
    }
    packed_scale_t() = default;
    CUTLASS_HOST_DEVICE
    explicit operator float() const {
        return float(get());
    }
    CUTLASS_HOST_DEVICE
    bool operator==(packed_scale_t const& rhs) const {
        return storage[0] == rhs.storage[0] && storage[1] == rhs.storage[1];
    }
    CUTLASS_HOST_DEVICE
    bool operator!=(packed_scale_t const& rhs) const {
        return !(*this == rhs);
    }
    CUTLASS_HOST_DEVICE
    friend packed_scale_t operator+(packed_scale_t const& lhs, packed_scale_t const& rhs) {
        return packed_scale_t(lhs.get() + rhs.get());
    }
    CUTLASS_HOST_DEVICE
    friend packed_scale_t operator-(packed_scale_t const& lhs, packed_scale_t const& rhs) {
        return packed_scale_t(lhs.get() - rhs.get());
    }
    CUTLASS_HOST_DEVICE
    friend packed_scale_t operator*(packed_scale_t const& lhs, packed_scale_t const& rhs) {
        return packed_scale_t(lhs.get() * rhs.get());
    }
    CUTLASS_HOST_DEVICE
    friend packed_scale_t operator/(packed_scale_t const& lhs, packed_scale_t const& rhs) {
        return packed_scale_t(lhs.get() / rhs.get());
    }

private:
    using Storage = uint32_t;
    using Stage   = uint8_t;

    Storage storage[2]{};

    CUTLASS_HOST_DEVICE
    static Storage pack4(T c1, T c2, T c3, T c4) {
        Storage result = 0;
        result |= (static_cast<Storage>(reinterpret_cast<Stage const&>(c4)) << 24);
        result |= (static_cast<Storage>(reinterpret_cast<Stage const&>(c3)) << 16);
        result |= (static_cast<Storage>(reinterpret_cast<Stage const&>(c2)) << 8);
        result |= static_cast<Storage>(reinterpret_cast<Stage const&>(c1));
        return result;
    }
    CUTLASS_HOST_DEVICE
    T get() const {
        auto stage = static_cast<Stage>(storage[0] >> 8);
#if defined(__CUDA_ARCH__)
        return reinterpret_cast<T const&>(stage);
#else
        T tmp;
        std::memcpy(&tmp, &stage, sizeof(Stage));
        return tmp;
#endif
    }
    CUTLASS_HOST_DEVICE
    T get(int idx) const {
        Stage stage;
        if (idx < 4)
            stage = static_cast<Stage>(storage[0] >> (8 * idx));
        else
            stage = static_cast<Stage>(storage[1] >> (8 * idx - 32));
#if defined(__CUDA_ARCH__)
        return reinterpret_cast<T const&>(stage);
#else
        T tmp;
        std::memcpy(&tmp, &stage, sizeof(Stage));
        return tmp;
#endif
    }
};

static bool pack_scale_fp8(cutlass::float_e4m3_t const*              block_in,
                           cutlass::Array<cutlass::float_e4m3_t, 8>* block_out,
                           const size_t                              block_size,
                           const bool                                is_cpu) {
    std::vector<cutlass::float_e4m3_t>                    data_in(block_size);
    std::vector<cutlass::Array<cutlass::float_e4m3_t, 8>> data_out(block_size);

    try {
        if (is_cpu) {
            cutlass::device_memory::copy_host_to_host(data_in.data(), block_in, block_size);
        } else {
            cutlass::device_memory::copy_to_host(data_in.data(), block_in, block_size);
        }
    } catch (cutlass::cuda_exception const& e) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(e.cudaError()) << std::endl;
        return false;
    }

    for (size_t i = 0; i < block_size; i++) {
        packed_scale_t<cutlass::float_e4m3_t> tmp(data_in[i]);
        data_out[i] = reinterpret_cast<cutlass::Array<cutlass::float_e4m3_t, 8> const&>(tmp);
    }

    try {
        if (is_cpu) {
            cutlass::device_memory::copy_host_to_host(block_out, data_out.data(), block_size);
        } else {
            cutlass::device_memory::copy_to_device(block_out, data_out.data(), block_size);
        }
    } catch (cutlass::cuda_exception const& e) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(e.cudaError()) << std::endl;
        return false;
    }
    return true;
}

template<int ElementsPerThread>
__global__ void pack_scale_fp8_kernel(cutlass::float_e4m3_t const*              block_in,
                                      cutlass::Array<cutlass::float_e4m3_t, 8>* block_out,
                                      const size_t                              block_size) {
    auto idx = blockIdx.x * blockDim.x * ElementsPerThread + threadIdx.x;
    for (int k = 0; k < ElementsPerThread; k++) {
        if (idx >= block_size)
            return;

        packed_scale_t<cutlass::float_e4m3_t> tmp(block_in[idx]);
        block_out[idx] = reinterpret_cast<cutlass::Array<cutlass::float_e4m3_t, 8> const&>(tmp);
        idx += blockDim.x;
    }
}

torch::Tensor rtp_llm::run_pack_scale_fp8(const torch::Tensor& input) {
    TORCH_CHECK(input.dtype() == torch::kFloat8_e4m3fn, "Input must be of type float8_e4m3fn.");

    auto output_sizes = input.sizes().vec();
    output_sizes.push_back(8);
    auto output        = torch::empty(output_sizes, input.options());
    auto input_buffer  = static_cast<const cutlass::float_e4m3_t*>(input.data_ptr());
    auto output_buffer = static_cast<cutlass::Array<cutlass::float_e4m3_t, 8>*>(output.data_ptr());
    auto size          = input.numel();

    if (input.device().type() == torch::kCUDA) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();

        constexpr int threads_per_block   = 128;
        constexpr int elements_per_thread = 32;

        auto blocks = cutlass::ceil_div(input.numel(), threads_per_block * elements_per_thread);
        pack_scale_fp8_kernel<elements_per_thread>
            <<<blocks, threads_per_block, 0, stream>>>(input_buffer, output_buffer, size);
    } else {
        pack_scale_fp8(input_buffer, output_buffer, size, input.device().type() == torch::kCPU);
    }

    return output;
}

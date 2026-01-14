#include "cute/numeric/integral_constant.hpp"
#include "cute/util/type_traits.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/device_memory.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/w4a8_group_gemm/w4a8_group_gemm.h"

// In the mainloop, PRMT selects 1 byte from only 8 bytes so the sign bit is handled in an extra PRMT.
// Here the encodings of positive values and negative values are unified (except for the sign bit).
// For instance, 1 becomes 0b0111, which is the same encoding as -1 (0b1111).
static bool unified_encode_int4b(cutlass::int4b_t const* block_in,
                                 cutlass::int4b_t*       block_out,
                                 const size_t            block_size,
                                 const bool              is_cpu) {
    using StorageType                      = cutlass::int4b_t::Storage;
    constexpr int            pack          = cute::sizeof_bits_v<StorageType> / 4;
    const size_t             host_buf_size = block_size / pack;
    std::vector<StorageType> host_buf(host_buf_size);
    if (is_cpu) {
        cutlass::device_memory::copy_host_to_host(host_buf.data(), (StorageType*)block_in, host_buf_size);
    } else {
        cutlass::device_memory::copy_to_host(host_buf.data(), (StorageType*)block_in, host_buf_size);
    }

    for (auto&& d : host_buf) {
        StorageType out  = 0;
        StorageType mask = 0x0f;
        for (int i = 0; i < pack; i++) {
            cutlass::int4b_t curr;
            curr.storage = (d >> (i * 4)) & 0x0f;
            switch (curr) {
                case 1:
                    curr.storage = StorageType(0b0111);
                    break;  // 2's complement
                case 2:
                    curr.storage = StorageType(0b0110);
                    break;  // 2's complement
                case 3:
                    curr.storage = StorageType(0b0101);
                    break;  // 2's complement
                case 4:
                    curr.storage = StorageType(0b0100);
                    break;  // 2's complement
                case 5:
                    curr.storage = StorageType(0b0011);
                    break;  // 2's complement
                case 6:
                    curr.storage = StorageType(0b0010);
                    break;  // 2's complement
                case 7:
                    curr.storage = StorageType(0b0001);
                    break;  // 2's complement
                default:
                    break;
            }
            out |= (curr.storage << (4 * i)) & mask;
            mask <<= 4;
        }
        d = out;
    }

    if (is_cpu) {
        cutlass::device_memory::copy_host_to_host((StorageType*)block_out, host_buf.data(), host_buf_size);
    } else {
        cutlass::device_memory::copy_to_device((StorageType*)block_out, host_buf.data(), host_buf_size);
    }

    return true;
}

torch::Tensor rtp_llm::run_unified_encode_int4b(const torch::Tensor& input) {
    TORCH_CHECK(input.dtype() == torch::kInt8, "Input must be of type int8.");

    auto output = torch::empty(input.sizes().vec(), input.options());

    unified_encode_int4b(static_cast<const cutlass::int4b_t*>(input.data_ptr()),
                         static_cast<cutlass::int4b_t*>(output.data_ptr()),
                         input.numel() * 8 / cutlass::sizeof_bits<cutlass::int4b_t>::value,
                         input.device().type() == torch::kCPU);

    return output;
}

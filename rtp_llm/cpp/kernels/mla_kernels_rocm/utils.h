#define cudaLaunchKernel hipLaunchKernel

#define FLASHINFER_CUDA_CALL(func, ...)                                                                                \
    {                                                                                                                  \
        hipError_t e = (func);                                                                                         \
        if (e != hipSuccess) {                                                                                         \
            return e;                                                                                                  \
        }                                                                                                              \
    }

// convert head_dim to compile-time constant
#define DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...)                                                                     \
    switch (head_dim) {                                                                                                \
        case 64: {                                                                                                     \
            constexpr size_t HEAD_DIM = 64;                                                                            \
            __VA_ARGS__                                                                                                \
            break;                                                                                                     \
        }                                                                                                              \
        case 128: {                                                                                                    \
            constexpr size_t HEAD_DIM = 128;                                                                           \
            __VA_ARGS__                                                                                                \
            break;                                                                                                     \
        }                                                                                                              \
        case 256: {                                                                                                    \
            constexpr size_t HEAD_DIM = 256;                                                                           \
            __VA_ARGS__                                                                                                \
            break;                                                                                                     \
        }                                                                                                              \
        case 512: {                                                                                                    \
            constexpr size_t HEAD_DIM = 512;                                                                           \
            __VA_ARGS__                                                                                                \
            break;                                                                                                     \
        }                                                                                                              \
        default: {                                                                                                     \
            std::ostringstream err_msg;                                                                                \
            err_msg << "Unsupported head_dim: " << head_dim;                                                           \
            FLASHINFER_ERROR(err_msg.str());                                                                           \
        }                                                                                                              \
    }

#include <arm_sve.h>
#include <cstring>
// #define PACK_DEBUG
#ifdef PACK_DEBUG
#include <iomanip>
#endif

#include "ArmGemmKernel.h"
#include "gemm_microkernel_macro_m8_bf16.h"
#include "activation_const.hpp"
#include "arm_common.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/devices/arm_impl/ArmDevice.h"
#include "src/fastertransformer/models/W.h"

namespace fastertransformer {

ConstBufferPtr prepareGemmWeight(const std::string& key, ConstBufferPtr input) {
    // Transpose and reorder
    if (key == W::lm_head) {
        return prepareGemmOptWeight(transposeWeight(input), true);
    }

    // Reorder RHS weight matrics for better GEMM performance
    if (key == W::attn_qkv_w ||
        key == W::attn_o_w ||
        key == W::ffn_w1 ||
        key == W::ffn_w2 ||
        key == W::ffn_w3) {
        return prepareGemmOptWeight(input);
    }

    return input;
}

BufferPtr transposeWeight(ConstBufferPtr input) {

    std::vector<size_t> Bshape = input->shape();
    auto dim = input->dim();
    size_t k;
    size_t n;

    k = Bshape[dim - 2];
    n = Bshape[dim - 1];

    arm_compute::NETranspose transB;
    arm_compute::Tensor      wei_tran_tensor;
    arm_compute::TensorInfo  wei_data_info;
    arm_compute::TensorInfo  wei_tran_info;
    arm_compute::Tensor wei_tensor;

    BufferPtr output;

    auto data_type = input->type();
    arm_compute::DataType acl_data_type;

    if (data_type == DataType::TYPE_FP16)
        acl_data_type = arm_compute::DataType::F16;
    else if (data_type == DataType::TYPE_FP32)
        acl_data_type = arm_compute::DataType::F32;
    else
        printf("Not supported data type %d\n", data_type);

    wei_data_info = arm_compute::TensorInfo(arm_compute::TensorShape(n, k), 1, acl_data_type);
    wei_tran_info = arm_compute::TensorInfo(arm_compute::TensorShape(k, n), 1, acl_data_type);

    std::vector<size_t> weight_workspace_shape = std::vector<size_t>(Bshape.begin(), Bshape.end() - 2);
    weight_workspace_shape.insert(weight_workspace_shape.end(), {n, k});

    size_t element_num = k * n;
    size_t data_size = data_type == DataType::TYPE_FP32 ? sizeof(float) : sizeof(float16_t);
    const void *data = malloc(element_num * data_size);
    output = BufferPtr(new Buffer(MemoryType::MEMORY_CPU,
                                                    data_type,
                                                    weight_workspace_shape,
                                                    data)),

    wei_tensor.allocator()->init(wei_data_info);
    wei_tran_tensor.allocator()->init(wei_tran_info);
    wei_tensor.allocator()->import_memory(input->data());
    wei_tran_tensor.allocator()->import_memory(output->data());

    transB.configure(&wei_tensor, &wei_tran_tensor);
    transB.run();

    return output;
}

BufferPtr prepareGemmOptWeight(ConstBufferPtr input, bool isTranspose) {
    BufferPtr weight_workspace;

    GemmKernel gemm_kernel;
    std::vector<size_t> Bshape = input->shape();
    auto dim = input->dim();

    size_t k;
    size_t n;

    k = Bshape[dim - 2];
    n = Bshape[dim - 1];

    size_t batch_size = std::accumulate(Bshape.begin(), Bshape.end() - 2, (size_t)1, std::multiplies<size_t>());

    size_t weight_k_pack = std::ceil(k / 8.0) * 8;
    size_t width = weight_k_pack * 2;
    size_t height = n / 2 + n % 2;
    if (input->type() == DataType::TYPE_FP32 || input->type() == DataType::TYPE_FP16) {
        // allocate a temp workspace to pack weight fp32->bf16
        std::vector<size_t> weight_workspace_shape = std::vector<size_t>(Bshape.begin(), Bshape.end() - 2);
        if (isTranspose)
            weight_workspace_shape.insert(weight_workspace_shape.end(), {n, k});
        else
            weight_workspace_shape.insert(weight_workspace_shape.end(), {k, n});

        // weight_workspace = device->allocateBuffer({DataType::TYPE_BF16, weight_workspace_shape, AllocationType::DEVICE}, {"gemm_weight_workspace"});
        size_t element_num = std::accumulate(Bshape.begin(), Bshape.end(), (size_t)1, std::multiplies<size_t>());
        const void *data = malloc(element_num * sizeof(hie::bfloat16));
        weight_workspace = BufferPtr(new Buffer(MemoryType::MEMORY_CPU,
                                                     DataType::TYPE_BF16,
                                                     weight_workspace_shape,
                                                     data)),
        memset(weight_workspace->data(), 0, weight_workspace->sizeBytes());
        // pack weight
        for (size_t batch = 0; batch < batch_size; ++batch) {
            hie::bfloat16* weight_workspace_cur_ptr = reinterpret_cast<hie::bfloat16*>(weight_workspace->dataWithOffset(batch * height * width));
            if (input->type() == DataType::TYPE_FP32) {
                float* B_fp32_ptr = reinterpret_cast<float*>(input->dataWithOffset(batch * k * n));
                gemm_kernel.gemm_pack_weight_FP32toBF16_arm(n, k, weight_k_pack, B_fp32_ptr, weight_workspace_cur_ptr);
            } else { // if(params.B.type() == DataType::TYPE_FP16)
                float16_t* B_fp16_ptr = reinterpret_cast<float16_t*>(input->dataWithOffset(batch * k * n));
                gemm_kernel.gemm_pack_weight_FP16toBF16_arm(n, k, weight_k_pack, B_fp16_ptr, weight_workspace_cur_ptr);
            }
        }
    }
    return weight_workspace;
}

void GemmKernel::pack_input_arm(int M, int N, int K, int lda, int K_pack, float* a_fp32, hie::bfloat16* a_bf16) {
    pack_input_impl_parallel_simd(M, N, K, lda, K_pack, a_fp32, a_bf16);
    return;
}

void GemmKernel::gemm_pack_weight_FP32toBF16_arm(int N, int K, int K_pack, const float* b_fp32, hie::bfloat16* b_bf16) {
    int k_tile   = 1024;  // empirical var: 1024, 5120
    int k_thread = std::ceil(K_pack * 1.0 / k_tile);

    parallel_for(k_thread, [&](int k) {
        for (int n = 0; n < N; n += 2) {
            float*         b_fp32_ptr1 = (float*)b_fp32 + k * k_tile * N + n + 0;
            float*         b_fp32_ptr2 = (float*)b_fp32 + k * k_tile * N + n + 1;
            hie::bfloat16* b_bf16_ptr  = b_bf16 + n * K_pack + k * k_tile * 2; // [n, k*k_tile*2]
            int            kk_max      = (k + 1) * k_tile < K ? (k + 1) * k_tile : K;
            for (int kk = k * k_tile; kk < kk_max; kk += 4) {
                for (int i = 0; i < 4 && (kk + i < kk_max); i++) {
                    b_bf16_ptr[i] = b_fp32_ptr1[i * N];
                    if (n != (N - 1)) {
                        b_bf16_ptr[i + 4] = b_fp32_ptr2[i * N];
                    }
                }
                b_bf16_ptr += 8;
                b_fp32_ptr1 += 4 * N;
                b_fp32_ptr2 += 4 * N;
            }
        }
    });

#ifdef PACK_DEBUG
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            if (j % 8 == 0) {
                printf("\n");
            }
            printf("%f ", b_fp32[j * N + i]);
        }
        printf("\n");
        printf("\n");
    }
    printf("\n");

    auto N_aligned = N / 2 + (N % 2);
    for (int i = 0; i < N_aligned; i++) {
        for (int j = 0; j < K_pack * 2; j++) {
            if (j % 8 == 0) {
                printf("\n");
            }
            std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(6) << b_bf16[i * K_pack * 2 + j] << " ";
        }
        printf("\n");
        printf("\n");
    }
    printf("\n");
#endif

    return;
}


void GemmKernel::gemm_pack_weight_FP16toBF16_arm(int N, int K, int K_pack, const float16_t* b_fp16, hie::bfloat16* b_bf16) {
    int k_tile   = 1024;  // empirical var: 1024, 5120
    int k_thread = std::ceil(K_pack * 1.0 / k_tile);

    parallel_for(k_thread, [&](int k) {
        for (int n = 0; n < N; n += 2) {
            float16_t*     b_fp16_ptr1 = (float16_t*)b_fp16 + k * k_tile * N + n + 0;
            float16_t*     b_fp16_ptr2 = (float16_t*)b_fp16 + k * k_tile * N + n + 1;
            hie::bfloat16* b_bf16_ptr  = b_bf16 + n * K_pack + k * k_tile * 2; // [n, k*k_tile*2]
            int            kk_max      = (k + 1) * k_tile < K ? (k + 1) * k_tile : K;
            for (int kk = k * k_tile; kk < kk_max; kk += 4) {
                for (int i = 0; i < 4 && (kk + i < kk_max); i++) {
                    b_bf16_ptr[i] = b_fp16_ptr1[i * N];
                    if (n != (N - 1)) {
                        b_bf16_ptr[i + 4] = b_fp16_ptr2[i * N];
                    }
                }
                b_bf16_ptr += 8;
                b_fp16_ptr1 += 4 * N;
                b_fp16_ptr2 += 4 * N;
            }
        }
    });


#ifdef PACK_DEBUG
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            if (j % 8 == 0) {
                printf("\n");
            }
            std::cout << b_fp16[j * N + i] << " ";
        }
        printf("\n");
        printf("\n");
    }
    printf("\n");

    auto N_aligned = N / 2 + (N % 2);
    for (int i = 0; i < N_aligned; i++) {
        for (int j = 0; j < K_pack * 2; j++) {
            if (j % 8 == 0) {
                printf("\n");
            }
            std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(6) << b_bf16[i * K_pack * 2 + j] << " ";
        }
        printf("\n");
        printf("\n");
    }
    printf("\n");
#endif

    return;
}


void GemmKernel::pack_input_fp16tobf16_impl_parallel_simd(
    int M, int N, int K, int lda, int K_pack, float16_t* a_fp16, hie::bfloat16* a_bf16) {
#define LABEL_FOR_LOOP_M "0"
#define LABEL_FOR_LOOP_K "1"
#define LABEL_m_EQ_M_1 "2"
    int k_tile   = 1024;  // empirical var: 1024, 5120
    int k_thread = std::ceil(K * 1.0 / k_tile);

    // printf("k_tile: %d, k_thread: %d\n", k_tile, k_thread);

    // fp32 [ a[i,  j+0], a[i,  j+1], a[i,  j+2], a[i,  j+3] ]
    // fp32 [ a[i+1,j+0], a[i+1,j+1], a[i+1,j+2], a[i+1,j+3] ]
    // bf16 [ a[i,  j+0], a[i,  j+1], a[i,  j+2], a[i,  j+3],
    //        a[i+1,j+0], a[i+1,j+1], a[i+1,j+2], a[i+1,j+3]]

    parallel_for(k_thread, [&](int k) {
        float16_t*     a_fp16_ptr1   = a_fp16 + 0 * lda + k * k_tile;
        float16_t*     a_fp16_ptr2   = a_fp16 + 1 * lda + k * k_tile;
        hie::bfloat16* a_bf16_ptr    = a_bf16 + k * k_tile * 2;
        int            a_fp16_offset = 2 * lda * sizeof(float16_t);
        int            a_bf16_offset = 2 * K_pack * sizeof(hie::bfloat16); // if K_pack % 16 == 8, for the remain 8 zero elements, use next line to cover it
        int            kk            = k * k_tile;
        int            kk_max        = (k + 1) * k_tile < K ? (k + 1) * k_tile : K;

        // clang-format off
        asm volatile(
            "ptrue   p0.b                                    \n"
            "sub     x1,    %[M], #1                         \n"  // M - 1
            "mov     x2,    #0                               \n"  // m

            "" LABEL_FOR_LOOP_M
            ":\n"
            "mov     x3,    %[a_fp16_ptr1]                   \n"
            "mov     x4,    %[a_fp16_ptr2]                   \n"
            "mov     x5,    %[a_bf16_ptr]                    \n"

            "prfw    pldl1strm, p0, [x3,    #0, MUL VL]      \n"  // prefetch
            "prfw    pldl1strm, p0, [x4,    #0, MUL VL]      \n"

            "mov     x0,    %[kk]                            \n"
            "whilelt p1.h,  x0,   %[kk_max]                  \n"  // compare kk
                                                                  // and kk_max

            "" LABEL_FOR_LOOP_K
            ":\n"
            "ld1h   z4.h, p1/z, [x3,    #0, MUL VL]          \n" // load 8 fp16
            "dup    z6.h, #0                                 \n"
            "zip1   z0.h, z4.h, z6.h                         \n"  // zip 4(or less) fp16 values with 0
            "zip2   z1.h, z4.h, z6.h                         \n"  // zip 4(or less) fp16 values with 0
            "fcvt   z0.s, p0/m, z0.h                         \n"  // fp16 -> fp32
            "dup    z2.h, #0                                 \n"
            "fcvt   z1.s, p0/m, z1.h                         \n"  // fp16 -> fp32
            "dup    z3.h, #0                                 \n"
            "cmp    x2, x1                                   \n"  // compare m,
                                                                  // M - 1
            "b.none  " LABEL_m_EQ_M_1
            "f                     \n"
            "ld1h   z5.h, p1/z, [x4,    #0, MUL VL]          \n"  // load, when
                                                                  // m != M - 1
            "zip1   z2.h, z5.h, z6.h                         \n"  // zip 4(or less) fp16 values with 0
            "zip2   z3.h, z5.h, z6.h                         \n"  // zip 4(or less) fp16 values with 0
            "fcvt   z2.s, p0/m, z2.h                         \n"  // fp16 -> fp32
            "fcvt   z3.s, p0/m, z3.h                         \n"  // fp16 -> fp32

            "" LABEL_m_EQ_M_1
            ":\n"
            "add     x3, x3, #16                             \n"  // a_fp16_ptr1 += 8
            "add     x4, x4, #16                             \n"  // a_fp16_ptr2 += 8
            // "add     x3, x3, #8                              \n"  // a_fp16_ptr1 += 4
            // "add     x4, x4, #8                              \n"  // a_fp16_ptr2 += 4

            "prfw    pldl1strm, p0, [x3,    #0, MUL VL]      \n"
            "prfw    pldl1strm, p0, [x4,    #0, MUL VL]      \n"

            "bfcvt   z0.h, p0/m, z0.s                        \n"  // fp32 ->
                                                                  // bf16
            "bfcvt   z1.h, p0/m, z1.s                        \n"
            "bfcvt   z2.h, p0/m, z2.s                        \n"
            "bfcvt   z3.h, p0/m, z3.s                        \n"

            "uzp1    z4.h, z0.h, z2.h                        \n"  // combine
                                                                  // bf16
            "uzp1    z5.h, z1.h, z3.h                        \n"  // combine bf16
            "zip1    p3.d, p1.d, p1.d                        \n"  // cp 4 least significant half to 4 most significant half
            ""
            "st1h    z4.h, p3,   [x5, #0, MUL VL]            \n"  // store bf16 data

            "zip2    p3.d, p1.d, p1.d                        \n"  // cp 4 most significant half to 4 least significant half
            "st1h    z5.h, p3,   [x5, #1, MUL VL]            \n"  // store bf16
            "add     x5, x5, #32                             \n"  // a_bf16_ptr += 16
            // "add     x5, x5, #16                             \n"  // a_bf16_ptr += 8

            //   "prfw    pstl1keep, p0, [x5,    #0, MUL VL]      \n"

            "add     x0,    x0,   #8                         \n"  // kk += 8
            // "add     x0,    x0,   #4                         \n"  // kk += 4
            "whilelt p1.h,  x0,   %[kk_max]                  \n"  // compare kk
                                                                  // and kk_max
            "b.tstop " LABEL_FOR_LOOP_K
            "b                   \n"  // if k < K_MAX, go to label

            "add     %[a_fp16_ptr1], %[a_fp16_ptr1], %[a_fp16_offset] \n"
            "add     %[a_fp16_ptr2], %[a_fp16_ptr2], %[a_fp16_offset] \n"
            "add     %[a_bf16_ptr],  %[a_bf16_ptr],  %[a_bf16_offset] \n"
            "add     x2,    x2,   #2                         \n"  // m += 2
            "cmp     x2, %[M]                                \n"  // compare m,
                                                                  // M
            "b.tstop " LABEL_FOR_LOOP_M
            "b                   \n"  // if m < M, go to label

            : /* empty OutputOperands */
            : [a_fp16_ptr1] "r"(a_fp16_ptr1), [a_fp16_ptr2] "r"(a_fp16_ptr2),
              [a_bf16_ptr] "r"(a_bf16_ptr), [kk] "r"(kk), [kk_max] "r"(kk_max),
              [M] "r"(M), [a_fp16_offset] "r"(a_fp16_offset),
              [a_bf16_offset] "r"(a_bf16_offset)
            : "x0", "x1", "x2", "x3", "x4", "x5",
              "p0", "p1", "p2", "p3",
              "z0", "z1", "z2", "z3", "z4", "z5", "z6",
              "cc", "memory");
        // clang-format on
    });

#ifdef PACK_DEBUG
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            if (j % 8 == 0) {
                printf("\n");
            }
            printf("%f ", a_fp16[i * lda + j]);
            // std::cout << a_fp16[i * lda + j] << " ";
        }
        printf("\n");
        printf("\n");
    }
    printf("\n");
    
    
    // int k_pack_compute = std::ceil(K / 16.0) * 16;
    auto M_aligned = M + (M % 2);
    for (int i = 0; i < M_aligned / 2; i++) {
        for (int j = 0; j < K_pack * 2; j++) {
            if (j % 8 == 0) {
                printf("\n");
            }
            std::cout << a_bf16[i * K_pack * 2 + j] << " ";
        }
        printf("\n");
        printf("\n");
    }
    printf("\n");
#endif

    return;
}

void GemmKernel::pack_input_impl_parallel_simd(
    int M, int N, int K, int lda, int K_pack, float* a_fp32, hie::bfloat16* a_bf16) {
#define LABEL_FOR_LOOP_M "0"
#define LABEL_FOR_LOOP_K "1"
#define LABEL_m_EQ_M_1 "2"
    int k_tile   = 1024;  // empirical var: 1024, 5120
    int k_thread = std::ceil(K * 1.0 / k_tile);

    // printf("k_tile: %d, k_thread: %d\n", k_tile, k_thread);

    // fp32 [ a[i,  j+0], a[i,  j+1], a[i,  j+2], a[i,  j+3] ]
    // fp32 [ a[i+1,j+0], a[i+1,j+1], a[i+1,j+2], a[i+1,j+3] ]
    // bf16 [ a[i+1,j+0], a[i+1,j+1], a[i+1,j+2], a[i+1,j+3],
    //        a[i,  j+0], a[i,  j+1], a[i,  j+2], a[i,  j+3]] ???

    parallel_for(k_thread, [&](int k) {
        float*         a_fp32_ptr1   = a_fp32 + 0 * lda + k * k_tile;
        float*         a_fp32_ptr2   = a_fp32 + 1 * lda + k * k_tile;
        hie::bfloat16* a_bf16_ptr    = a_bf16 + k * k_tile * 2;
        int            a_fp32_offset = 2 * lda * sizeof(float);
        int            a_bf16_offset = 2 * K_pack * sizeof(hie::bfloat16);
        int            kk            = k * k_tile;
        int            kk_max        = (k + 1) * k_tile < K ? (k + 1) * k_tile : K;

        // clang-format off
        asm volatile(
            "ptrue   p0.b                                    \n"
            "sub     x1,    %[M], #1                         \n"  // M - 1
            "mov     x2,    #0                               \n"  // m

            "" LABEL_FOR_LOOP_M
            ":\n"
            "mov     x3,    %[a_fp32_ptr1]                   \n"
            "mov     x4,    %[a_fp32_ptr2]                   \n"
            "mov     x5,    %[a_bf16_ptr]                    \n"

            "prfw    pldl1strm, p0, [x3,    #0, MUL VL]      \n"  // prefetch
            "prfw    pldl1strm, p0, [x4,    #0, MUL VL]      \n"

            "mov     x0,    %[kk]                            \n"
            "whilelt p1.s,  x0,   %[kk_max]                  \n"  // compare kk
                                                                  // and kk_max

            "" LABEL_FOR_LOOP_K
            ":\n"
            "ld1w   z0.s, p1/z, [x3,    #0, MUL VL]          \n"
            "dup    z1.h, #0                                 \n"
            "cmp    x2, x1                                   \n"  // compare m,
                                                                  // M - 1
            "b.none  " LABEL_m_EQ_M_1
            "f                     \n"
            "ld1w   z1.s, p1/z, [x4,    #0, MUL VL]          \n"  // load, when
                                                                  // m != M - 1

            "" LABEL_m_EQ_M_1
            ":\n"
            "add     x3, x3, #16                             \n"
            "add     x4, x4, #16                             \n"

            "prfw    pldl1strm, p0, [x3,    #0, MUL VL]      \n"
            "prfw    pldl1strm, p0, [x4,    #0, MUL VL]      \n"

            "bfcvt   z0.h, p0/m, z0.s                        \n"  // fp32 ->
                                                                  // bf16
            "bfcvt   z1.h, p0/m, z1.s                        \n"
            "uzp1    z2.h, z0.h, z1.h                        \n"  // combine
                                                                  // bf16

            "uzp1    p3.h, p1.h, p1.h                        \n"
            "st1h    z2.h, p3,   [x5, #0, MUL VL]            \n"  // store bf16
                                                                  // data
            "add     x5, x5, #16                             \n"

            //   "prfw    pstl1keep, p0, [x5,    #0, MUL VL]      \n"

            "add     x0,    x0,   #4                         \n"  // kk += 4
            "whilelt p1.s,  x0,   %[kk_max]                  \n"  // compare kk
                                                                  // and kk_max
            "b.tstop " LABEL_FOR_LOOP_K
            "b                   \n"  // if k < K_MAX, go to label

            "add     %[a_fp32_ptr1], %[a_fp32_ptr1], %[a_fp32_offset] \n"
            "add     %[a_fp32_ptr2], %[a_fp32_ptr2], %[a_fp32_offset] \n"
            "add     %[a_bf16_ptr],  %[a_bf16_ptr],  %[a_bf16_offset] \n"
            "add     x2,    x2,   #2                         \n"  // m += 2
            "cmp     x2, %[M]                                \n"  // compare m,
                                                                  // M
            "b.tstop " LABEL_FOR_LOOP_M
            "b                   \n"  // if m < M, go to label

            : /* empty OutputOperands */
            : [a_fp32_ptr1] "r"(a_fp32_ptr1), [a_fp32_ptr2] "r"(a_fp32_ptr2),
              [a_bf16_ptr] "r"(a_bf16_ptr), [kk] "r"(kk), [kk_max] "r"(kk_max),
              [M] "r"(M), [a_fp32_offset] "r"(a_fp32_offset),
              [a_bf16_offset] "r"(a_bf16_offset)
            : "x0", "x1", "x2", "x3", "x4", "x5", "p0", "p1", "p2", "p3", "z0",
              "z1", "z2", "cc", "memory");
        // clang-format on
    });

#ifdef PACK_DEBUG
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            if (j % 8 == 0) {
                printf("\n");
            }
            printf("%f ", a_fp32[i * lda + j]);
        }
        printf("\n");
        printf("\n");
    }
    printf("\n");


    auto M_aligned = M + (M % 2);
    for (int i = 0; i < M_aligned / 2; i++) {
        for (int j = 0; j < K_pack * 2; j++) {
            if (j % 8 == 0) {
                printf("\n");
            }
            std::cout << a_bf16[i * K_pack * 2 + j] << " ";
        }
        printf("\n");
        printf("\n");
    }
    printf("\n");
#endif

    return;
}

}  // namespace fastertransformer
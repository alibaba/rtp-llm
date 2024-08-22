#include <arm_sve.h>
#include <cstring>
// #define DEBUG
#ifdef DEBUG
#include <iomanip>
#endif

#include "ArmGemmKernel.h"
#include "gemm_microkernel_macro_m8_bf16.h"
#include "activation_const.hpp"
#include "arm_common.h"

namespace fastertransformer {

void GemmKernel::thread_block_bf16_m8(
    GemmPartParam<hie::bfloat16, hie::bfloat16, float, float>& p, int m, int n, int k, int k_tile) {
#define LABEL_FOR_LOOP_K "1"
#define LABEL_SKIP_PRF "2"

    int M = p.M;
    int N = p.N;

    hie::bfloat16* a_bf16_ptr1 = p.a_ptr + (m + 0) * p.K_pack + k * 2; // [m, k*2], *2 is for processing 2*k_tile per kernel
    hie::bfloat16* a_bf16_ptr2 = p.a_ptr + (m + 2) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr3 = p.a_ptr + (m + 4) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr4 = p.a_ptr + (m + 6) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr1 = p.b_ptr + (n + 0) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr2 = p.b_ptr + (n + 2) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr3 = p.b_ptr + (n + 4) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr4 = p.b_ptr + (n + 6) * p.K_pack + k * 2;

    uint64_t c_fp32_ptr = reinterpret_cast<uint64_t>(p.c_ptr + (m + 0) * N + n);

    int next_line_offset = N * sizeof(float);

    float* bias_ptr = p.bias_ptr + n;

    int k_init = k * 2;
    int K_MAX  = (k + k_tile) * 2;
    K_MAX      = K_MAX < p.K_pack * 2 ? K_MAX : p.K_pack * 2;
    int K_MAIN = K_MAX / 16 * 16; // floor

    activation_const_t constant;

    // clang-format off

    asm volatile(
        "ptrue   p0.b                               \n"
        "ptrue   p4.b                               \n"
        "ptrue   p5.b                               \n"

        // ASM_BLOCK_PREFETCH_PART_0
        
        "mov     x0, %[k_init]                      \n" // k
        "mov     x2, %[m]                           \n"
        "mov     x3, %[n]                           \n"

        // "mov     x7, #0                             \n"

        /* clear bfmmla result regs */
        ASM_BLOCK_CLEAR_BFMMLA_REG

        LABEL_FOR_LOOP_K ":\n"
        /* load bf16 input & weight */
        ASM_BLOCK_LOAD_A
        ASM_BLOCK_LOAD_B

        // ASM_BLOCK_PREFETCH_PART_1

        /* matmul */
        ASM_BLOCK_BFMMLA

        "add     x0,    x0,   #16                \n" // k += 16
        "cmp     x0,    %[K_MAIN]                \n" // compare k and K_MAIN
        "b.tstop " LABEL_FOR_LOOP_K "b           \n" // if k < K_MAIN, go to label


        /* calculate the remaining A and B */
        /* load bf16 input & weight */
        "mov     x4,    x0                       \n"
        "whilelt p5.h,  x4,   %[K_MAX]           \n" // compare k and K_MAX
        "add     x4,    x4,   #8                 \n"
        "whilelt p4.h,  x4,   %[K_MAX]           \n"

        ASM_BLOCK_LOAD_A
        ASM_BLOCK_LOAD_B

        // ASM_BLOCK_PREFETCH_PART_1

        /* matmul */
        ASM_BLOCK_BFMMLA

        /* reorder mmla output */
        ASM_BLOCK_REORDER_BFMMLA_OUTPUT

        "whilelt p1.s, x3, %[N]                  \n" // compare n, N
        "add     x6,   x3, #4                    \n" // n + 2
        "whilelt p2.s, x6, %[N]                  \n" // compare n, N
        : /* empty OutputOperands */
        : [a_bf16_ptr1] "r"(a_bf16_ptr1), [a_bf16_ptr2] "r"(a_bf16_ptr2),
        [a_bf16_ptr3] "r"(a_bf16_ptr3), [a_bf16_ptr4] "r"(a_bf16_ptr4),
        [b_bf16_ptr1] "r"(b_bf16_ptr1), [b_bf16_ptr2] "r"(b_bf16_ptr2),
        [b_bf16_ptr3] "r"(b_bf16_ptr3), [b_bf16_ptr4] "r"(b_bf16_ptr4),
        [next_line_offset] "r"(next_line_offset),
        [m] "r"(m), [n] "r"(n), [k_init] "r"(k_init),
        [M] "r"(M), [N] "r"(N), [K_MAIN] "r"(K_MAIN), [K_MAX] "r"(K_MAX)
        : "p0", "p1", "p2", "p4", "p5",
        "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
        "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", 
        "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19",
        "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29",
        "z30", "z31", 
        "cc", "memory");

    if (p.with_bias && k == 0) {
        ASM_BLOCK_ADD_BIAS
    }

    if (LIKELY(k != 0)) {
        ASM_BLOCK_C_ACCUMULATE
    }

    if (p.do_act == 1) {
        switch (p.actType) {
            case UnaryType::UNARYTYPE_UNDEFINED: {
                break;
            }
            case UnaryType::RELU: {
                ASM_BLOCK_ACTIVE_RELU
                break;
            }
            case UnaryType::SILU: {
                ASM_BLOCK_ACTIVE_SILU
                break;
            }
            case UnaryType::TANH: {
                ASM_BLOCK_ACTIVE_TANH
                break;
            }
            case UnaryType::GELU_ERF: {
                ASM_BLOCK_ACTIVE_GELU_ERF
                break;
            }
            case UnaryType::GELU_TANH: {
                ASM_BLOCK_ACTIVE_GELU_TANH
                break;
            }
            default:
                break;
        }
    }

    ASM_BLOCK_C_STORE

    // clang-format on

#undef LABEL_FOR_LOOP_K
#undef LABEL_SKIP_PRF
    return;
}

/*********************************************************/
void GemmKernel::thread_block_bf16_m8(
    GemmPartParam<hie::bfloat16, hie::bfloat16, float16_t, float>& p, int m, int n, int k, int k_tile) {
#define LABEL_FOR_LOOP_K "1"
#define LABEL_SKIP_PRF "2"

    int M = p.M;
    int N = p.N;

    hie::bfloat16* a_bf16_ptr1 = p.a_ptr + (m + 0) * p.K_pack + k * 2; // [m, k*2], *2 is for processing 2*k_tile per kernel
    hie::bfloat16* a_bf16_ptr2 = p.a_ptr + (m + 2) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr3 = p.a_ptr + (m + 4) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr4 = p.a_ptr + (m + 6) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr1 = p.b_ptr + (n + 0) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr2 = p.b_ptr + (n + 2) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr3 = p.b_ptr + (n + 4) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr4 = p.b_ptr + (n + 6) * p.K_pack + k * 2;

    uint64_t c_fp16_ptr = reinterpret_cast<uint64_t>(p.c_ptr + (m + 0) * N + n);

    int next_line_offset = N * sizeof(float16_t);

    float* bias_ptr = p.bias_ptr + n; // TODO: handle float16_t bias

    int k_init = k * 2;
    int K_MAX  = (k + k_tile) * 2;
    K_MAX      = K_MAX < p.K_pack * 2 ? K_MAX : p.K_pack * 2;
    int K_MAIN = K_MAX / 16 * 16; // floor

    activation_const_t constant;

    // clang-format off

    asm volatile(
        "ptrue   p0.b                               \n"
        "ptrue   p4.b                               \n"
        "ptrue   p5.b                               \n"

        // ASM_BLOCK_PREFETCH_PART_0
        
        "mov     x0, %[k_init]                      \n" // k
        "mov     x2, %[m]                           \n"
        "mov     x3, %[n]                           \n"

        // "mov     x7, #0                             \n"

        /* clear bfmmla result regs */
        ASM_BLOCK_CLEAR_BFMMLA_REG

        LABEL_FOR_LOOP_K ":\n"
        /* load bf16 input & weight */
        ASM_BLOCK_LOAD_A
        ASM_BLOCK_LOAD_B

        // ASM_BLOCK_PREFETCH_PART_1

        /* matmul */
        ASM_BLOCK_BFMMLA

        "add     x0,    x0,   #16                \n" // k += 16
        "cmp     x0,    %[K_MAIN]                \n" // compare k and K_MAIN
        "b.tstop " LABEL_FOR_LOOP_K "b           \n" // if k < K_MAIN, go to label


        /* calculate the remaining A and B */
        /* load bf16 input & weight */
        "mov     x4,    x0                       \n"
        "whilelt p5.h,  x4,   %[K_MAX]           \n" // compare k and K_MAX
        "add     x4,    x4,   #8                 \n"
        "whilelt p4.h,  x4,   %[K_MAX]           \n"

        ASM_BLOCK_LOAD_A
        ASM_BLOCK_LOAD_B

        // ASM_BLOCK_PREFETCH_PART_1

        /* matmul */
        ASM_BLOCK_BFMMLA

        /* reorder mmla output */
        ASM_BLOCK_REORDER_BFMMLA_OUTPUT_FP16

        "whilelt p1.h, x3, %[N]                  \n" // compare n, N
        // "add     x6,   x3, #4                    \n" // n + 4
        // "whilelt p2.s, x6, %[N]                  \n" // compare n, N
        : /* empty OutputOperands */
        : [a_bf16_ptr1] "r"(a_bf16_ptr1), [a_bf16_ptr2] "r"(a_bf16_ptr2),
        [a_bf16_ptr3] "r"(a_bf16_ptr3), [a_bf16_ptr4] "r"(a_bf16_ptr4),
        [b_bf16_ptr1] "r"(b_bf16_ptr1), [b_bf16_ptr2] "r"(b_bf16_ptr2),
        [b_bf16_ptr3] "r"(b_bf16_ptr3), [b_bf16_ptr4] "r"(b_bf16_ptr4),
        [next_line_offset] "r"(next_line_offset),
        [m] "r"(m), [n] "r"(n), [k_init] "r"(k_init),
        [M] "r"(M), [N] "r"(N), [K_MAIN] "r"(K_MAIN), [K_MAX] "r"(K_MAX)
        : "p0", "p1", "p2", "p4", "p5", "p3",
        "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
        "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", 
        "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19",
        "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29",
        "z30", "z31", 
        "cc", "memory");

    if (p.with_bias && k == 0) {
        ASM_BLOCK_ADD_BIAS
    }

    if (LIKELY(k != 0)) {
        ASM_BLOCK_C_ACCUMULATE_FP16
    }

    if (p.do_act == 1) {
        switch (p.actType) {
            case UnaryType::UNARYTYPE_UNDEFINED: {
                break;
            }
            case UnaryType::RELU: {
                ASM_BLOCK_ACTIVE_RELU
                break;
            }
            case UnaryType::SILU: {
                ASM_BLOCK_ACTIVE_SILU
                break;
            }
            case UnaryType::TANH: {
                ASM_BLOCK_ACTIVE_TANH
                break;
            }
            case UnaryType::GELU_ERF: {
                ASM_BLOCK_ACTIVE_GELU_ERF
                break;
            }
            case UnaryType::GELU_TANH: {
                ASM_BLOCK_ACTIVE_GELU_TANH
                break;
            }
            default:
                break;
        }
    }

    ASM_BLOCK_C_STORE_FP16

    // clang-format on

#undef LABEL_FOR_LOOP_K
#undef LABEL_SKIP_PRF
    return;
}


/*********************************************************/

void GemmKernel::thread_block_bf16_m8_mres(
    GemmPartParam<hie::bfloat16, hie::bfloat16, float, float>& p, int m, int n, int k, int k_tile) {
#define LABEL_FOR_LOOP_K "1"
#define LABEL_SKIP_PRF "2"
#define LABEL_SKIP_STORE "3"
#define LABEL_SKIP_LD_A1 "4"
#define LABEL_SKIP_LD_W1 "5"
#define LABEL_SKIP_ACCUMULATE "6"

    int M = p.M;
    int N = p.N;

    hie::bfloat16* a_bf16_ptr1 = p.a_ptr + (m + 0) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr2 = p.a_ptr + (m + 2) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr3 = p.a_ptr + (m + 4) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr4 = p.a_ptr + (m + 6) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr1 = p.b_ptr + (n + 0) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr2 = p.b_ptr + (n + 2) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr3 = p.b_ptr + (n + 4) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr4 = p.b_ptr + (n + 6) * p.K_pack + k * 2;

    uint64_t c_fp32_ptr = reinterpret_cast<uint64_t>(p.c_ptr + (m + 0) * N + n);

    int next_line_offset = N * sizeof(float);

    float* bias_ptr = p.bias_ptr + n;

    int k_init = k * 2;
    int K_MAX  = (k + k_tile) * 2;
    K_MAX      = K_MAX < p.K_pack * 2 ? K_MAX : p.K_pack * 2;
    int K_MAIN = K_MAX / 16 * 16;

    activation_const_t constant;

    // clang-format off

    asm volatile(
        "ptrue   p0.b                               \n"
        "ptrue   p4.b                               \n"
        "ptrue   p5.b                               \n"

        // ASM_BLOCK_PREFETCH_PART_0

        "mov     x0, %[k_init]                      \n" // k
        "mov     x2, %[m]                           \n"
        "mov     x3, %[n]                           \n"

        /* clear bfmmla result regs */
        ASM_BLOCK_CLEAR_BFMMLA_REG

        " " LABEL_FOR_LOOP_K ":\n"

        /* load bf16 input & weight */
        ASM_BLOCK_LOAD_A_RES
        ASM_BLOCK_LOAD_B

        // ASM_BLOCK_PREFETCH_PART_1

        /* matmul */
        ASM_BLOCK_BFMMLA

        "add     x0,    x0,   #16                \n" // k += 16
        "cmp     x0,    %[K_MAIN]                \n" // compare k and K_MAIN
        "b.tstop " LABEL_FOR_LOOP_K "b           \n" // if k < K_MAIN, go to label

        /* load bf16 input & weight */
        "mov     x4,    x0                           \n"
        "whilelt p5.h,  x4,   %[K_MAX]               \n" // compare k and K_MAX
        "add     x4,    x0,   #8                     \n"
        "whilelt p4.h,  x4,   %[K_MAX]               \n"

        ASM_BLOCK_LOAD_A_RES
        ASM_BLOCK_LOAD_B

        // ASM_BLOCK_PREFETCH_PART_1

        /* matmul */
        ASM_BLOCK_BFMMLA

        /* reorder mmla output */
        ASM_BLOCK_REORDER_BFMMLA_OUTPUT

        "whilelt p1.s, x3, %[N]                  \n" // compare n, N
        "add     x6,   x3, #4                    \n" // n + 2
        "whilelt p2.s, x6, %[N]                  \n" // compare n, N

        : /* empty OutputOperands */
        : [a_bf16_ptr1] "r"(a_bf16_ptr1), [a_bf16_ptr2] "r"(a_bf16_ptr2),
        [a_bf16_ptr3] "r"(a_bf16_ptr3), [a_bf16_ptr4] "r"(a_bf16_ptr4),
        [b_bf16_ptr1] "r"(b_bf16_ptr1), [b_bf16_ptr2] "r"(b_bf16_ptr2),
        [b_bf16_ptr3] "r"(b_bf16_ptr3), [b_bf16_ptr4] "r"(b_bf16_ptr4),
        [next_line_offset] "r"(next_line_offset),
        [m] "r"(m), [n] "r"(n), [k_init] "r"(k_init),
        [M] "r"(M), [N] "r"(N), [K_MAIN] "r"(K_MAIN), [K_MAX] "r"(K_MAX)
        : "p0", "p1", "p2", "p4", "p5", 
        "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
        "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9",
        "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19",
        "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29",
        "z30", "z31", 
        "cc", "memory");

    if (p.with_bias && k == 0) {
        ASM_BLOCK_ADD_BIAS
    }

    if (LIKELY(k != 0)) {
        ASM_BLOCK_C_RES_ACCUMULATE
    }

    if (p.do_act == 1) {
        switch (p.actType) {
            case UnaryType::UNARYTYPE_UNDEFINED: {
                break;
            }
            case UnaryType::RELU: {
                ASM_BLOCK_ACTIVE_RELU
                break;
            }
            case UnaryType::SILU: {
                ASM_BLOCK_ACTIVE_SILU
                break;
            }
            case UnaryType::TANH: {
                ASM_BLOCK_ACTIVE_TANH
                break;
            }
            case UnaryType::GELU_ERF: {
                ASM_BLOCK_ACTIVE_GELU_ERF
                break;
            }
            case UnaryType::GELU_TANH: {
                ASM_BLOCK_ACTIVE_GELU_TANH
                break;
            }
            default:
                break;
        }
    }

    ASM_BLOCK_C_RES_STORE

    // clang-format on

#undef LABEL_FOR_LOOP_K
#undef LABEL_SKIP_PRF
#undef LABEL_SKIP_STORE
#undef LABEL_SKIP_LD_A1
#undef LABEL_SKIP_LD_W1
#undef LABEL_SKIP_ACCUMULATE
    return;
}

void GemmKernel::thread_block_bf16_m8_mres(
    GemmPartParam<hie::bfloat16, hie::bfloat16, float16_t, float>& p, int m, int n, int k, int k_tile) {
#define LABEL_FOR_LOOP_K "1"
#define LABEL_SKIP_PRF "2"
#define LABEL_SKIP_STORE "3"
#define LABEL_SKIP_LD_A1 "4"
#define LABEL_SKIP_LD_W1 "5"
#define LABEL_SKIP_ACCUMULATE "6"

    int M = p.M;
    int N = p.N;

    hie::bfloat16* a_bf16_ptr1 = p.a_ptr + (m + 0) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr2 = p.a_ptr + (m + 2) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr3 = p.a_ptr + (m + 4) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr4 = p.a_ptr + (m + 6) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr1 = p.b_ptr + (n + 0) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr2 = p.b_ptr + (n + 2) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr3 = p.b_ptr + (n + 4) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr4 = p.b_ptr + (n + 6) * p.K_pack + k * 2;

    uint64_t c_fp16_ptr = reinterpret_cast<uint64_t>(p.c_ptr + (m + 0) * N + n);

    int next_line_offset = N * sizeof(float16_t);

    float* bias_ptr = p.bias_ptr + n;

    int k_init = k * 2;
    int K_MAX  = (k + k_tile) * 2;
    K_MAX      = K_MAX < p.K_pack * 2 ? K_MAX : p.K_pack * 2;
    int K_MAIN = K_MAX / 16 * 16;

    activation_const_t constant;

    // clang-format off

    asm volatile(
        "ptrue   p0.b                               \n"
        "ptrue   p4.b                               \n"
        "ptrue   p5.b                               \n"

        // ASM_BLOCK_PREFETCH_PART_0

        "mov     x0, %[k_init]                      \n" // k
        "mov     x2, %[m]                           \n"
        "mov     x3, %[n]                           \n"

        /* clear bfmmla result regs */
        ASM_BLOCK_CLEAR_BFMMLA_REG

        " " LABEL_FOR_LOOP_K ":\n"

        /* load bf16 input & weight */
        ASM_BLOCK_LOAD_A_RES
        ASM_BLOCK_LOAD_B

        // ASM_BLOCK_PREFETCH_PART_1

        /* matmul */
        ASM_BLOCK_BFMMLA

        "add     x0,    x0,   #16                \n" // k += 16
        "cmp     x0,    %[K_MAIN]                \n" // compare k and K_MAIN
        "b.tstop " LABEL_FOR_LOOP_K "b           \n" // if k < K_MAIN, go to label

        /* load bf16 input & weight */
        "mov     x4,    x0                           \n"
        "whilelt p5.h,  x4,   %[K_MAX]               \n" // compare k and K_MAX
        "add     x4,    x0,   #8                     \n"
        "whilelt p4.h,  x4,   %[K_MAX]               \n"

        ASM_BLOCK_LOAD_A_RES
        ASM_BLOCK_LOAD_B

        // ASM_BLOCK_PREFETCH_PART_1

        /* matmul */
        ASM_BLOCK_BFMMLA

        /* reorder mmla output */
        ASM_BLOCK_REORDER_BFMMLA_OUTPUT_FP16

        "whilelt p1.h, x3, %[N]                  \n" // compare n, N
        // "add     x6,   x3, #4                    \n" // n + 4
        // "whilelt p2.s, x6, %[N]                  \n" // compare n, N

        : /* empty OutputOperands */
        : [a_bf16_ptr1] "r"(a_bf16_ptr1), [a_bf16_ptr2] "r"(a_bf16_ptr2),
        [a_bf16_ptr3] "r"(a_bf16_ptr3), [a_bf16_ptr4] "r"(a_bf16_ptr4),
        [b_bf16_ptr1] "r"(b_bf16_ptr1), [b_bf16_ptr2] "r"(b_bf16_ptr2),
        [b_bf16_ptr3] "r"(b_bf16_ptr3), [b_bf16_ptr4] "r"(b_bf16_ptr4),
        [next_line_offset] "r"(next_line_offset),
        [m] "r"(m), [n] "r"(n), [k_init] "r"(k_init),
        [M] "r"(M), [N] "r"(N), [K_MAIN] "r"(K_MAIN), [K_MAX] "r"(K_MAX)
        : "p0", "p1", "p2", "p4", "p5", "p3",
        "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
        "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9",
        "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19",
        "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29",
        "z30", "z31", 
        "cc", "memory");

    if (p.with_bias && k == 0) {
        ASM_BLOCK_ADD_BIAS
    }

    if (LIKELY(k != 0)) {
        ASM_BLOCK_C_RES_ACCUMULATE_FP16
    }

    if (p.do_act == 1) {
        switch (p.actType) {
            case UnaryType::UNARYTYPE_UNDEFINED: {
                break;
            }
            case UnaryType::RELU: {
                ASM_BLOCK_ACTIVE_RELU
                break;
            }
            case UnaryType::SILU: {
                ASM_BLOCK_ACTIVE_SILU
                break;
            }
            case UnaryType::TANH: {
                ASM_BLOCK_ACTIVE_TANH
                break;
            }
            case UnaryType::GELU_ERF: {
                ASM_BLOCK_ACTIVE_GELU_ERF
                break;
            }
            case UnaryType::GELU_TANH: {
                ASM_BLOCK_ACTIVE_GELU_TANH
                break;
            }
            default:
                break;
        }
    }

    ASM_BLOCK_C_RES_STORE_FP16

    // clang-format on

#undef LABEL_FOR_LOOP_K
#undef LABEL_SKIP_PRF
#undef LABEL_SKIP_STORE
#undef LABEL_SKIP_LD_A1
#undef LABEL_SKIP_LD_W1
#undef LABEL_SKIP_ACCUMULATE
    return;
}

/*********************************************************/

void GemmKernel::thread_block_bf16_m8_nres(
    GemmPartParam<hie::bfloat16, hie::bfloat16, float, float>& p, int m, int n, int k, int k_tile) {
#define LABEL_FOR_LOOP_K "1"
#define LABEL_SKIP_PRF "2"
#define LABEL_SKIP_STORE "3"
#define LABEL_SKIP_LD_A1 "4"
#define LABEL_SKIP_LD_W1 "5"
#define LABEL_SKIP_ACCUMULATE "6"

    int M = p.M;
    int N = p.N;

    hie::bfloat16* a_bf16_ptr1 = p.a_ptr + (m + 0) * p.K_pack + k * 2;  // 2 --> sizeof(bfloat16)
    hie::bfloat16* a_bf16_ptr2 = p.a_ptr + (m + 2) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr3 = p.a_ptr + (m + 4) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr4 = p.a_ptr + (m + 6) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr1 = p.b_ptr + (n + 0) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr2 = p.b_ptr + (n + 2) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr3 = p.b_ptr + (n + 4) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr4 = p.b_ptr + (n + 6) * p.K_pack + k * 2;

    uint64_t c_fp32_ptr = reinterpret_cast<uint64_t>(p.c_ptr + (m + 0) * N + n);

    int next_line_offset = N * sizeof(float);

    float* bias_ptr = p.bias_ptr + n;

    int k_init = k * 2;
    int K_MAX  = (k + k_tile) * 2;
    K_MAX      = K_MAX < p.K_pack * 2 ? K_MAX : p.K_pack * 2;
    int K_MAIN = K_MAX / 16 * 16;

    activation_const_t constant;

    // clang-format off

    asm volatile(
        "ptrue   p0.b                               \n"
        "ptrue   p4.b                               \n"
        "ptrue   p5.b                               \n"

        // ASM_BLOCK_PREFETCH_PART_0

        "mov     x0, %[k_init]                      \n" // k
        "mov     x2, %[m]                           \n"
        "mov     x3, %[n]                           \n"

        // "mov     x7, #0                             \n"

        /* clear bfmmla result regs */
        ASM_BLOCK_CLEAR_BFMMLA_REG

        " " LABEL_FOR_LOOP_K ":\n"
        /* load bf16 input & weight */
        ASM_BLOCK_LOAD_A
        ASM_BLOCK_LOAD_B_RES

        // ASM_BLOCK_PREFETCH_PART_1

        /* matmul */
        ASM_BLOCK_BFMMLA

        "add     x0,    x0,   #16                \n" // k += 16
        "cmp     x0,    %[K_MAIN]                \n" // compare k and K_MAIN
        "b.tstop " LABEL_FOR_LOOP_K "b           \n" // if k < K_MAIN, go to label

        /* load bf16 input & weight */
        "mov     x4,    x0                           \n"
        "whilelt p5.h,  x4,   %[K_MAX]               \n" // compare k and K_MAX
        "add     x4,    x4,   #8                     \n"
        "whilelt p4.h,  x4,   %[K_MAX]               \n"

        ASM_BLOCK_LOAD_A
        ASM_BLOCK_LOAD_B_RES

        // ASM_BLOCK_PREFETCH_PART_1

        /* matmul */
        ASM_BLOCK_BFMMLA

        /* reorder mmla output */
        ASM_BLOCK_REORDER_BFMMLA_OUTPUT

        "whilelt p1.s, x3, %[N]                  \n" // compare n, N
        "add     x6,   x3, #4                    \n" // n + 2
        "whilelt p2.s, x6, %[N]                  \n" // compare n, N
        : /* empty OutputOperands */
        : [a_bf16_ptr1] "r"(a_bf16_ptr1), [a_bf16_ptr2] "r"(a_bf16_ptr2),
        [a_bf16_ptr3] "r"(a_bf16_ptr3), [a_bf16_ptr4] "r"(a_bf16_ptr4),
        [b_bf16_ptr1] "r"(b_bf16_ptr1), [b_bf16_ptr2] "r"(b_bf16_ptr2),
        [b_bf16_ptr3] "r"(b_bf16_ptr3), [b_bf16_ptr4] "r"(b_bf16_ptr4),
        [next_line_offset] "r"(next_line_offset),
        [m] "r"(m), [n] "r"(n), [k_init] "r"(k_init),
        [M] "r"(M), [N] "r"(N), [K_MAIN] "r"(K_MAIN), [K_MAX] "r"(K_MAX)
        : "p0", "p1", "p2", "p4", "p5", 
        "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
        "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9",
        "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19",
        "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29",
        "z30", "z31", 
        "cc", "memory");

    if (p.with_bias && k == 0) {
        ASM_BLOCK_ADD_BIAS
    }

    if (LIKELY(k != 0)) {
        ASM_BLOCK_C_ACCUMULATE
    }

    if (p.do_act == 1) {
        switch (p.actType) {
            case UnaryType::UNARYTYPE_UNDEFINED: {
                break;
            }
            case UnaryType::RELU: {
                ASM_BLOCK_ACTIVE_RELU
                break;
            }
            case UnaryType::SILU: {
                ASM_BLOCK_ACTIVE_SILU
                break;
            }
            case UnaryType::TANH: {
                ASM_BLOCK_ACTIVE_TANH
                break;
            }
            case UnaryType::GELU_ERF: {
                ASM_BLOCK_ACTIVE_GELU_ERF
                break;
            }
            case UnaryType::GELU_TANH: {
                ASM_BLOCK_ACTIVE_GELU_TANH
                break;
            }
            default:
                break;
        }
    }

    ASM_BLOCK_C_STORE

    // clang-format on

#undef LABEL_FOR_LOOP_K
#undef LABEL_SKIP_PRF
#undef LABEL_SKIP_STORE
#undef LABEL_SKIP_LD_A1
#undef LABEL_SKIP_LD_W1
#undef LABEL_SKIP_ACCUMULATE
    return;
}

void GemmKernel::thread_block_bf16_m8_nres(
    GemmPartParam<hie::bfloat16, hie::bfloat16, float16_t, float>& p, int m, int n, int k, int k_tile) {
#define LABEL_FOR_LOOP_K "1"
#define LABEL_SKIP_PRF "2"
#define LABEL_SKIP_STORE "3"
#define LABEL_SKIP_LD_A1 "4"
#define LABEL_SKIP_LD_W1 "5"
#define LABEL_SKIP_ACCUMULATE "6"

    int M = p.M;
    int N = p.N;

    hie::bfloat16* a_bf16_ptr1 = p.a_ptr + (m + 0) * p.K_pack + k * 2;  // 2 --> sizeof(bfloat16)
    hie::bfloat16* a_bf16_ptr2 = p.a_ptr + (m + 2) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr3 = p.a_ptr + (m + 4) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr4 = p.a_ptr + (m + 6) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr1 = p.b_ptr + (n + 0) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr2 = p.b_ptr + (n + 2) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr3 = p.b_ptr + (n + 4) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr4 = p.b_ptr + (n + 6) * p.K_pack + k * 2;

    uint64_t c_fp16_ptr = reinterpret_cast<uint64_t>(p.c_ptr + (m + 0) * N + n);

    int next_line_offset = N * sizeof(float16_t);

    float* bias_ptr = p.bias_ptr + n;

    int k_init = k * 2;
    int K_MAX  = (k + k_tile) * 2;
    K_MAX      = K_MAX < p.K_pack * 2 ? K_MAX : p.K_pack * 2;
    int K_MAIN = K_MAX / 16 * 16;

    activation_const_t constant;

    // clang-format off

    asm volatile(
        "ptrue   p0.b                               \n"
        "ptrue   p4.b                               \n"
        "ptrue   p5.b                               \n"

        // ASM_BLOCK_PREFETCH_PART_0

        "mov     x0, %[k_init]                      \n" // k
        "mov     x2, %[m]                           \n"
        "mov     x3, %[n]                           \n"

        // "mov     x7, #0                             \n"

        /* clear bfmmla result regs */
        ASM_BLOCK_CLEAR_BFMMLA_REG

        " " LABEL_FOR_LOOP_K ":\n"
        /* load bf16 input & weight */
        ASM_BLOCK_LOAD_A
        ASM_BLOCK_LOAD_B_RES

        // ASM_BLOCK_PREFETCH_PART_1

        /* matmul */
        ASM_BLOCK_BFMMLA

        "add     x0,    x0,   #16                \n" // k += 16
        "cmp     x0,    %[K_MAIN]                \n" // compare k and K_MAIN
        "b.tstop " LABEL_FOR_LOOP_K "b           \n" // if k < K_MAIN, go to label

        /* load bf16 input & weight */
        "mov     x4,    x0                           \n"
        "whilelt p5.h,  x4,   %[K_MAX]               \n" // compare k and K_MAX
        "add     x4,    x4,   #8                     \n"
        "whilelt p4.h,  x4,   %[K_MAX]               \n"

        ASM_BLOCK_LOAD_A
        ASM_BLOCK_LOAD_B_RES

        // ASM_BLOCK_PREFETCH_PART_1

        /* matmul */
        ASM_BLOCK_BFMMLA

        /* reorder mmla output */
        ASM_BLOCK_REORDER_BFMMLA_OUTPUT_FP16

        "whilelt p1.h, x3, %[N]                  \n" // compare n, N
        // "add     x6,   x3, #4                    \n" // n + 4
        // "whilelt p2.s, x6, %[N]                  \n" // compare n, N
        : /* empty OutputOperands */
        : [a_bf16_ptr1] "r"(a_bf16_ptr1), [a_bf16_ptr2] "r"(a_bf16_ptr2),
        [a_bf16_ptr3] "r"(a_bf16_ptr3), [a_bf16_ptr4] "r"(a_bf16_ptr4),
        [b_bf16_ptr1] "r"(b_bf16_ptr1), [b_bf16_ptr2] "r"(b_bf16_ptr2),
        [b_bf16_ptr3] "r"(b_bf16_ptr3), [b_bf16_ptr4] "r"(b_bf16_ptr4),
        [next_line_offset] "r"(next_line_offset),
        [m] "r"(m), [n] "r"(n), [k_init] "r"(k_init),
        [M] "r"(M), [N] "r"(N), [K_MAIN] "r"(K_MAIN), [K_MAX] "r"(K_MAX)
        : "p0", "p1", "p2", "p4", "p5", "p3",
        "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
        "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9",
        "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19",
        "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29",
        "z30", "z31", 
        "cc", "memory");

    if (p.with_bias && k == 0) {
        ASM_BLOCK_ADD_BIAS
    }

    if (LIKELY(k != 0)) {
        ASM_BLOCK_C_ACCUMULATE_FP16
    }

    if (p.do_act == 1) {
        switch (p.actType) {
            case UnaryType::UNARYTYPE_UNDEFINED: {
                break;
            }
            case UnaryType::RELU: {
                ASM_BLOCK_ACTIVE_RELU
                break;
            }
            case UnaryType::SILU: {
                ASM_BLOCK_ACTIVE_SILU
                break;
            }
            case UnaryType::TANH: {
                ASM_BLOCK_ACTIVE_TANH
                break;
            }
            case UnaryType::GELU_ERF: {
                ASM_BLOCK_ACTIVE_GELU_ERF
                break;
            }
            case UnaryType::GELU_TANH: {
                ASM_BLOCK_ACTIVE_GELU_TANH
                break;
            }
            default:
                break;
        }
    }

    ASM_BLOCK_C_STORE_FP16

    // clang-format on

#undef LABEL_FOR_LOOP_K
#undef LABEL_SKIP_PRF
#undef LABEL_SKIP_STORE
#undef LABEL_SKIP_LD_A1
#undef LABEL_SKIP_LD_W1
#undef LABEL_SKIP_ACCUMULATE
    return;
}

/*********************************************************/

void GemmKernel::thread_block_bf16_m8_res(
    GemmPartParam<hie::bfloat16, hie::bfloat16, float, float>& p, int m, int n, int k, int k_tile) {
#define LABEL_FOR_LOOP_K "1"
#define LABEL_SKIP_PRF "2"
#define LABEL_SKIP_STORE "3"
#define LABEL_SKIP_LD_A1 "4"
#define LABEL_SKIP_LD_W1 "5"
#define LABEL_SKIP_ACCUMULATE "6"

    int M = p.M;
    int N = p.N;

    hie::bfloat16* a_bf16_ptr1 = p.a_ptr + (m + 0) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr2 = p.a_ptr + (m + 2) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr3 = p.a_ptr + (m + 4) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr4 = p.a_ptr + (m + 6) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr1 = p.b_ptr + (n + 0) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr2 = p.b_ptr + (n + 2) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr3 = p.b_ptr + (n + 4) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr4 = p.b_ptr + (n + 6) * p.K_pack + k * 2;

    uint64_t c_fp32_ptr = reinterpret_cast<uint64_t>(p.c_ptr + (m + 0) * N + n);

    int next_line_offset = N * sizeof(float);

    float* bias_ptr = p.bias_ptr + n;

    int k_init = k * 2;
    int K_MAX  = (k + k_tile) * 2;
    K_MAX      = K_MAX < p.K_pack * 2 ? K_MAX : p.K_pack * 2;
    int K_MAIN = K_MAX / 16 * 16;

    activation_const_t constant;

    // clang-format off

    asm volatile(
        "ptrue   p0.b                               \n"
        "ptrue   p4.b                               \n"
        "ptrue   p5.b                               \n"

        // ASM_BLOCK_PREFETCH_PART_0

        "mov     x0, %[k_init]                      \n" // k
        "mov     x2, %[m]                           \n"
        "mov     x3, %[n]                           \n"

        /* clear bfmmla result regs */
        ASM_BLOCK_CLEAR_BFMMLA_REG

        " " LABEL_FOR_LOOP_K ":\n"
        /* load bf16 input & weight */
        ASM_BLOCK_LOAD_A_RES
        ASM_BLOCK_LOAD_B_RES

        // ASM_BLOCK_PREFETCH_PART_1

        /* matmul */
        ASM_BLOCK_BFMMLA

        "add     x0,    x0,   #16                \n" // k += 16
        "cmp     x0,    %[K_MAIN]                \n" // compare k and K_MAIN
        "b.tstop " LABEL_FOR_LOOP_K "b           \n" // if k < K_MAIN, go to label

        /* load bf16 input & weight */
        "mov     x4,    x0                           \n"
        "whilelt p5.h,  x4,   %[K_MAX]               \n" // compare k and K_MAX
        "add     x4,    x0,   #8                     \n"
        "whilelt p4.h,  x4,   %[K_MAX]               \n"

        ASM_BLOCK_LOAD_A_RES
        ASM_BLOCK_LOAD_B_RES

        // ASM_BLOCK_PREFETCH_PART_1

        /* matmul */
        ASM_BLOCK_BFMMLA

        /* reorder mmla output */
        ASM_BLOCK_REORDER_BFMMLA_OUTPUT

        "whilelt p1.s, x3, %[N]                  \n" // compare n, N
        "add     x6,   x3, #4                    \n" // n + 2
        "whilelt p2.s, x6, %[N]                  \n" // compare n, N

        : /* empty OutputOperands */
        : [a_bf16_ptr1] "r"(a_bf16_ptr1), [a_bf16_ptr2] "r"(a_bf16_ptr2),
        [a_bf16_ptr3] "r"(a_bf16_ptr3), [a_bf16_ptr4] "r"(a_bf16_ptr4),
        [b_bf16_ptr1] "r"(b_bf16_ptr1), [b_bf16_ptr2] "r"(b_bf16_ptr2),
        [b_bf16_ptr3] "r"(b_bf16_ptr3), [b_bf16_ptr4] "r"(b_bf16_ptr4),
        [next_line_offset] "r"(next_line_offset),
        [m] "r"(m), [n] "r"(n), [k_init] "r"(k_init),
        [M] "r"(M), [N] "r"(N), [K_MAIN] "r"(K_MAIN), [K_MAX] "r"(K_MAX)
        : "p0", "p1", "p2", "p4", "p5", 
        "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
        "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9",
        "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19",
        "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29",
        "z30", "z31", 
        "cc", "memory");

    if (p.with_bias && k == 0) {
        ASM_BLOCK_ADD_BIAS
    }

    if (LIKELY(k != 0)) {
        ASM_BLOCK_C_RES_ACCUMULATE
    }

    if (p.do_act == 1) {
        switch (p.actType) {
            case UnaryType::UNARYTYPE_UNDEFINED: {
                break;
            }
            case UnaryType::RELU: {
                ASM_BLOCK_ACTIVE_RELU
                break;
            }
            case UnaryType::SILU: {
                ASM_BLOCK_ACTIVE_SILU
                break;
            }
            case UnaryType::TANH: {
                ASM_BLOCK_ACTIVE_TANH
                break;
            }
            case UnaryType::GELU_ERF: {
                ASM_BLOCK_ACTIVE_GELU_ERF
                break;
            }
            case UnaryType::GELU_TANH: {
                ASM_BLOCK_ACTIVE_GELU_TANH
                break;
            }
            default:
                break;
        }
    }

  ASM_BLOCK_C_RES_STORE

    // clang-format on

#undef LABEL_FOR_LOOP_K
#undef LABEL_SKIP_PRF
#undef LABEL_SKIP_STORE
#undef LABEL_SKIP_LD_A1
#undef LABEL_SKIP_LD_W1
#undef LABEL_SKIP_ACCUMULATE
    return;
}

void GemmKernel::thread_block_bf16_m8_res(
    GemmPartParam<hie::bfloat16, hie::bfloat16, float16_t, float>& p, int m, int n, int k, int k_tile) {
#define LABEL_FOR_LOOP_K "1"
#define LABEL_SKIP_PRF "2"
#define LABEL_SKIP_STORE "3"
#define LABEL_SKIP_LD_A1 "4"
#define LABEL_SKIP_LD_W1 "5"
#define LABEL_SKIP_ACCUMULATE "6"

    int M = p.M;
    int N = p.N;

    hie::bfloat16* a_bf16_ptr1 = p.a_ptr + (m + 0) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr2 = p.a_ptr + (m + 2) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr3 = p.a_ptr + (m + 4) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr4 = p.a_ptr + (m + 6) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr1 = p.b_ptr + (n + 0) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr2 = p.b_ptr + (n + 2) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr3 = p.b_ptr + (n + 4) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr4 = p.b_ptr + (n + 6) * p.K_pack + k * 2;

    uint64_t c_fp16_ptr = reinterpret_cast<uint64_t>(p.c_ptr + (m + 0) * N + n);

    int next_line_offset = N * sizeof(float16_t);

    float* bias_ptr = p.bias_ptr + n;

    int k_init = k * 2;
    int K_MAX  = (k + k_tile) * 2;
    K_MAX      = K_MAX < p.K_pack * 2 ? K_MAX : p.K_pack * 2;
    int K_MAIN = K_MAX / 16 * 16;

    activation_const_t constant;

    // clang-format off

    asm volatile(
        "ptrue   p0.b                               \n"
        "ptrue   p4.b                               \n"
        "ptrue   p5.b                               \n"

        // ASM_BLOCK_PREFETCH_PART_0

        "mov     x0, %[k_init]                      \n" // k
        "mov     x2, %[m]                           \n"
        "mov     x3, %[n]                           \n"

        /* clear bfmmla result regs */
        ASM_BLOCK_CLEAR_BFMMLA_REG

        " " LABEL_FOR_LOOP_K ":\n"
        /* load bf16 input & weight */
        ASM_BLOCK_LOAD_A_RES
        ASM_BLOCK_LOAD_B_RES

        // ASM_BLOCK_PREFETCH_PART_1

        /* matmul */
        ASM_BLOCK_BFMMLA

        "add     x0,    x0,   #16                \n" // k += 16
        "cmp     x0,    %[K_MAIN]                \n" // compare k and K_MAIN
        "b.tstop " LABEL_FOR_LOOP_K "b           \n" // if k < K_MAIN, go to label

        /* load bf16 input & weight */
        "mov     x4,    x0                           \n"
        "whilelt p5.h,  x4,   %[K_MAX]               \n" // compare k and K_MAX
        "add     x4,    x0,   #8                     \n"
        "whilelt p4.h,  x4,   %[K_MAX]               \n"

        ASM_BLOCK_LOAD_A_RES
        ASM_BLOCK_LOAD_B_RES

        // ASM_BLOCK_PREFETCH_PART_1

        /* matmul */
        ASM_BLOCK_BFMMLA

        /* reorder mmla output */
        ASM_BLOCK_REORDER_BFMMLA_OUTPUT_FP16

        "whilelt p1.h, x3, %[N]                  \n" // compare n, N
        // "add     x6,   x3, #4                    \n" // n + 4
        // "whilelt p2.s, x6, %[N]                  \n" // compare n, N

        : /* empty OutputOperands */
        : [a_bf16_ptr1] "r"(a_bf16_ptr1), [a_bf16_ptr2] "r"(a_bf16_ptr2),
        [a_bf16_ptr3] "r"(a_bf16_ptr3), [a_bf16_ptr4] "r"(a_bf16_ptr4),
        [b_bf16_ptr1] "r"(b_bf16_ptr1), [b_bf16_ptr2] "r"(b_bf16_ptr2),
        [b_bf16_ptr3] "r"(b_bf16_ptr3), [b_bf16_ptr4] "r"(b_bf16_ptr4),
        [next_line_offset] "r"(next_line_offset),
        [m] "r"(m), [n] "r"(n), [k_init] "r"(k_init),
        [M] "r"(M), [N] "r"(N), [K_MAIN] "r"(K_MAIN), [K_MAX] "r"(K_MAX)
        : "p0", "p1", "p2", "p4", "p5", "p3",
        "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
        "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9",
        "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19",
        "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29",
        "z30", "z31", 
        "cc", "memory");

    if (p.with_bias && k == 0) {
        ASM_BLOCK_ADD_BIAS
    }

    if (LIKELY(k != 0)) {
        ASM_BLOCK_C_RES_ACCUMULATE_FP16
    }

    if (p.do_act == 1) {
        switch (p.actType) {
            case UnaryType::UNARYTYPE_UNDEFINED: {
                break;
            }
            case UnaryType::RELU: {
                ASM_BLOCK_ACTIVE_RELU
                break;
            }
            case UnaryType::SILU: {
                ASM_BLOCK_ACTIVE_SILU
                break;
            }
            case UnaryType::TANH: {
                ASM_BLOCK_ACTIVE_TANH
                break;
            }
            case UnaryType::GELU_ERF: {
                ASM_BLOCK_ACTIVE_GELU_ERF
                break;
            }
            case UnaryType::GELU_TANH: {
                ASM_BLOCK_ACTIVE_GELU_TANH
                break;
            }
            default:
                break;
        }
    }

  ASM_BLOCK_C_RES_STORE_FP16

    // clang-format on

#undef LABEL_FOR_LOOP_K
#undef LABEL_SKIP_PRF
#undef LABEL_SKIP_STORE
#undef LABEL_SKIP_LD_A1
#undef LABEL_SKIP_LD_W1
#undef LABEL_SKIP_ACCUMULATE
    return;
}


}  // namespace fastertransformer
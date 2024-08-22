/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_microkernel_macro_m8_bf16.h
 */

#pragma once

#include <arm_sve.h>

#include "activation_macro.h"

// clang-format off
/***********************/

#define ASM_BLOCK_CLEAR_BFMMLA_REG                   \
        "dup     z10.s, #0  \n"                      \
        "dup     z11.s, #0  \n"                      \
        "dup     z12.s, #0  \n"                      \
        "dup     z13.s, #0  \n"                      \
        "dup     z14.s, #0  \n"                      \
        "dup     z15.s, #0  \n"                      \
        "dup     z16.s, #0  \n"                      \
        "dup     z17.s, #0  \n"                      \
                                                     \
        "dup     z18.s, #0  \n"                      \
        "dup     z19.s, #0  \n"                      \
        "dup     z20.s, #0  \n"                      \
        "dup     z21.s, #0  \n"                      \
        "dup     z22.s, #0  \n"                      \
        "dup     z23.s, #0  \n"                      \
        "dup     z24.s, #0  \n"                      \
        "dup     z25.s, #0  \n"

/***********************/

#define ASM_BLOCK_LOAD_A                                       \
        "ld1h    z0.h,  p5/z, [%[a_bf16_ptr1], #0, MUL VL] \n" \
        "ld1h    z1.h,  p4/z, [%[a_bf16_ptr1], #1, MUL VL] \n" \
        "ld1h    z2.h,  p5/z, [%[a_bf16_ptr2], #0, MUL VL] \n" \
        "ld1h    z3.h,  p4/z, [%[a_bf16_ptr2], #1, MUL VL] \n" \
        "ld1h    z4.h,  p5/z, [%[a_bf16_ptr3], #0, MUL VL] \n" \
        "ld1h    z5.h,  p4/z, [%[a_bf16_ptr3], #1, MUL VL] \n" \
        "ld1h    z6.h,  p5/z, [%[a_bf16_ptr4], #0, MUL VL] \n" \
        "ld1h    z7.h,  p4/z, [%[a_bf16_ptr4], #1, MUL VL] \n" \
                                                               \
        "add     %[a_bf16_ptr1], %[a_bf16_ptr1], #32 \n"       \
        "add     %[a_bf16_ptr2], %[a_bf16_ptr2], #32 \n"       \
        "add     %[a_bf16_ptr3], %[a_bf16_ptr3], #32 \n"       \
        "add     %[a_bf16_ptr4], %[a_bf16_ptr4], #32 \n"

#define ASM_BLOCK_LOAD_A_RES                                   \
        "ld1h    z0.h,  p5/z, [%[a_bf16_ptr1], #0, MUL VL] \n" \
        "ld1h    z1.h,  p4/z, [%[a_bf16_ptr1], #1, MUL VL] \n" \
        "dup     z2.h,  #0                           \n"       \
        "dup     z3.h,  #0                           \n"       \
        "dup     z4.h,  #0                           \n"       \
        "dup     z5.h,  #0                           \n"       \
        "dup     z6.h,  #0                           \n"       \
        "dup     z7.h,  #0                           \n"       \
                                                               \
        /* if (m + 2) > M, go to label (skip load) */          \
        "add     x5,    x2, #2                       \n"       \
        "cmp     x5,    %[M]                         \n"       \
        "b.tcont " LABEL_SKIP_LD_A1 "f               \n"       \
        "ld1h    z2.h,  p5/z, [%[a_bf16_ptr2], #0, MUL VL] \n" \
        "ld1h    z3.h,  p4/z, [%[a_bf16_ptr2], #1, MUL VL] \n" \
                                                               \
        /* if (m + 4) > M, go to label (skip load) */          \
        "add     x5,    x2, #4                       \n"       \
        "cmp     x5,    %[M]                         \n"       \
        "b.tcont " LABEL_SKIP_LD_A1 "f               \n"       \
        "ld1h    z4.h,  p5/z, [%[a_bf16_ptr3], #0, MUL VL] \n" \
        "ld1h    z5.h,  p4/z, [%[a_bf16_ptr3], #1, MUL VL] \n" \
                                                               \
        /* if (m + 6) > M, go to label (skip load) */          \
        "add     x5,    x2, #6                       \n"       \
        "cmp     x5,    %[M]                         \n"       \
        "b.tcont " LABEL_SKIP_LD_A1 "f               \n"       \
        "ld1h    z6.h,  p5/z, [%[a_bf16_ptr4], #0, MUL VL] \n" \
        "ld1h    z7.h,  p4/z, [%[a_bf16_ptr4], #1, MUL VL] \n" \
                                                               \
        " " LABEL_SKIP_LD_A1 ":\n"                             \
                                                               \
        "add     %[a_bf16_ptr1], %[a_bf16_ptr1], #32 \n"       \
        "add     %[a_bf16_ptr2], %[a_bf16_ptr2], #32 \n"       \
        "add     %[a_bf16_ptr3], %[a_bf16_ptr3], #32 \n"       \
        "add     %[a_bf16_ptr4], %[a_bf16_ptr4], #32 \n"

/***********************/

#define ASM_BLOCK_LOAD_B                                       \
        "ld1h    z8.h,  p5/z, [%[b_bf16_ptr1], #0, MUL VL] \n" \
        "ld1h    z9.h,  p4/z, [%[b_bf16_ptr1], #1, MUL VL] \n" \
        "ld1h    z26.h, p5/z, [%[b_bf16_ptr2], #0, MUL VL] \n" \
        "ld1h    z27.h, p4/z, [%[b_bf16_ptr2], #1, MUL VL] \n" \
        "ld1h    z28.h, p5/z, [%[b_bf16_ptr3], #0, MUL VL] \n" \
        "ld1h    z29.h, p4/z, [%[b_bf16_ptr3], #1, MUL VL] \n" \
        "ld1h    z30.h, p5/z, [%[b_bf16_ptr4], #0, MUL VL] \n" \
        "ld1h    z31.h, p4/z, [%[b_bf16_ptr4], #1, MUL VL] \n" \
                                                               \
        "add     %[b_bf16_ptr1], %[b_bf16_ptr1], #32 \n"       \
        "add     %[b_bf16_ptr2], %[b_bf16_ptr2], #32 \n"       \
        "add     %[b_bf16_ptr3], %[b_bf16_ptr3], #32 \n"       \
        "add     %[b_bf16_ptr4], %[b_bf16_ptr4], #32 \n"

#define ASM_BLOCK_LOAD_B_RES                                   \
        "ld1h    z8.h,  p5/z, [%[b_bf16_ptr1], #0, MUL VL] \n" \
        "ld1h    z9.h,  p4/z, [%[b_bf16_ptr1], #1, MUL VL] \n" \
        "dup     z26.h, #0                           \n"       \
        "dup     z27.h, #0                           \n"       \
        "dup     z28.h, #0                           \n"       \
        "dup     z29.h, #0                           \n"       \
        "dup     z30.h, #0                           \n"       \
        "dup     z31.h, #0                           \n"       \
                                                               \
        /* if (n + 2) > N, go to label (skip load) */          \
        "add     x6,    x3, #2                       \n"       \
        "cmp     x6,    %[N]                         \n"       \
        "b.tcont " LABEL_SKIP_LD_W1 "f               \n"       \
        "ld1h    z26.h, p5/z, [%[b_bf16_ptr2], #0, MUL VL] \n" \
        "ld1h    z27.h, p4/z, [%[b_bf16_ptr2], #1, MUL VL] \n" \
                                                               \
        /* if (n + 4) > N, go to label (skip load) */          \
        "add     x6,    x3, #4                       \n"       \
        "cmp     x6,    %[N]                         \n"       \
        "b.tcont " LABEL_SKIP_LD_W1 "f               \n"       \
        "ld1h    z28.h, p5/z, [%[b_bf16_ptr3], #0, MUL VL] \n" \
        "ld1h    z29.h, p4/z, [%[b_bf16_ptr3], #1, MUL VL] \n" \
                                                               \
        /* if (n + 6) > N, go to label (skip load) */          \
        "add     x6,    x3, #6                       \n"       \
        "cmp     x6,    %[N]                         \n"       \
        "b.tcont " LABEL_SKIP_LD_W1 "f               \n"       \
        "ld1h    z30.h, p5/z, [%[b_bf16_ptr4], #0, MUL VL] \n" \
        "ld1h    z31.h, p4/z, [%[b_bf16_ptr4], #1, MUL VL] \n" \
                                                               \
        " " LABEL_SKIP_LD_W1 ":\n"                             \
                                                               \
        "add     %[b_bf16_ptr1], %[b_bf16_ptr1], #32 \n"       \
        "add     %[b_bf16_ptr2], %[b_bf16_ptr2], #32 \n"       \
        "add     %[b_bf16_ptr3], %[b_bf16_ptr3], #32 \n"       \
        "add     %[b_bf16_ptr4], %[b_bf16_ptr4], #32 \n"

/***********************/

#define ASM_BLOCK_BFMMLA                             \
        "bfmmla  z10.s,  z0.h, z8.h              \n" \
        "bfmmla  z11.s,  z0.h, z26.h             \n" \
        "bfmmla  z12.s,  z0.h, z28.h             \n" \
        "bfmmla  z13.s,  z0.h, z30.h             \n" \
        "bfmmla  z14.s,  z2.h, z8.h              \n" \
        "bfmmla  z15.s,  z2.h, z26.h             \n" \
        "bfmmla  z16.s,  z2.h, z28.h             \n" \
        "bfmmla  z17.s,  z2.h, z30.h             \n" \
                                                     \
        "bfmmla  z18.s,  z4.h, z8.h              \n" \
        "bfmmla  z19.s,  z4.h, z26.h             \n" \
        "bfmmla  z20.s,  z4.h, z28.h             \n" \
        "bfmmla  z21.s,  z4.h, z30.h             \n" \
        "bfmmla  z22.s,  z6.h, z8.h              \n" \
        "bfmmla  z23.s,  z6.h, z26.h             \n" \
        "bfmmla  z24.s,  z6.h, z28.h             \n" \
        "bfmmla  z25.s,  z6.h, z30.h             \n" \
                                                     \
        "bfmmla  z10.s,  z1.h, z9.h              \n" \
        "bfmmla  z11.s,  z1.h, z27.h             \n" \
        "bfmmla  z12.s,  z1.h, z29.h             \n" \
        "bfmmla  z13.s,  z1.h, z31.h             \n" \
        "bfmmla  z14.s,  z3.h, z9.h              \n" \
        "bfmmla  z15.s,  z3.h, z27.h             \n" \
        "bfmmla  z16.s,  z3.h, z29.h             \n" \
        "bfmmla  z17.s,  z3.h, z31.h             \n" \
                                                     \
        "bfmmla  z18.s,  z5.h, z9.h              \n" \
        "bfmmla  z19.s,  z5.h, z27.h             \n" \
        "bfmmla  z20.s,  z5.h, z29.h             \n" \
        "bfmmla  z21.s,  z5.h, z31.h             \n" \
        "bfmmla  z22.s,  z7.h, z9.h              \n" \
        "bfmmla  z23.s,  z7.h, z27.h             \n" \
        "bfmmla  z24.s,  z7.h, z29.h             \n" \
        "bfmmla  z25.s,  z7.h, z31.h             \n"

/***********************/

#define ASM_BLOCK_REORDER_BFMMLA_OUTPUT              \
        "trn1    z8.s,  z10.s, z11.s             \n" \
        "trn2    z9.s,  z10.s, z11.s             \n" \
        "zip1    z10.s, z8.s,  z9.s              \n" \
        "zip2    z11.s, z8.s,  z9.s              \n" \
                                                     \
        "trn1    z8.s,  z12.s, z13.s             \n" \
        "trn2    z9.s,  z12.s, z13.s             \n" \
        "zip1    z12.s, z8.s,  z9.s              \n" \
        "zip2    z13.s, z8.s,  z9.s              \n" \
                                                     \
        "trn1    z8.s,  z14.s, z15.s             \n" \
        "trn2    z9.s,  z14.s, z15.s             \n" \
        "zip1    z14.s, z8.s,  z9.s              \n" \
        "zip2    z15.s, z8.s,  z9.s              \n" \
                                                     \
        "trn1    z8.s,  z16.s, z17.s             \n" \
        "trn2    z9.s,  z16.s, z17.s             \n" \
        "zip1    z16.s, z8.s,  z9.s              \n" \
        "zip2    z17.s, z8.s,  z9.s              \n" \
                                                     \
        "trn1    z8.s,  z18.s, z19.s             \n" \
        "trn2    z9.s,  z18.s, z19.s             \n" \
        "zip1    z18.s, z8.s,  z9.s              \n" \
        "zip2    z19.s, z8.s,  z9.s              \n" \
                                                     \
        "trn1    z8.s,  z20.s, z21.s             \n" \
        "trn2    z9.s,  z20.s, z21.s             \n" \
        "zip1    z20.s, z8.s,  z9.s              \n" \
        "zip2    z21.s, z8.s,  z9.s              \n" \
                                                     \
        "trn1    z8.s,  z22.s, z23.s             \n" \
        "trn2    z9.s,  z22.s, z23.s             \n" \
        "zip1    z22.s, z8.s,  z9.s              \n" \
        "zip2    z23.s, z8.s,  z9.s              \n" \
                                                     \
        "trn1    z8.s,  z24.s, z25.s             \n" \
        "trn2    z9.s,  z24.s, z25.s             \n" \
        "zip1    z24.s, z8.s,  z9.s              \n" \
        "zip2    z25.s, z8.s,  z9.s              \n"

// fp16: z10, z11, z14, z15, z18, z19, z22, z23
#define ASM_BLOCK_REORDER_BFMMLA_OUTPUT_FP16         \
        "ptrue   p3.b                            \n" \
                                                     \
        "trn1    z8.s,  z10.s, z11.s             \n" \
        "trn2    z9.s,  z10.s, z11.s             \n" \
        "zip1    z10.s, z8.s,  z9.s              \n" \
        "zip2    z11.s, z8.s,  z9.s              \n" \
                                                     \
        "fcvt    z10.h, p3/m,  z10.s             \n" \
        "fcvt    z11.h, p3/m,  z11.s             \n" \
                                                     \
        "trn1    z8.s,  z12.s, z13.s             \n" \
        "trn2    z9.s,  z12.s, z13.s             \n" \
        "zip1    z12.s, z8.s,  z9.s              \n" \
        "zip2    z13.s, z8.s,  z9.s              \n" \
                                                     \
        "fcvt    z12.h, p3/m,  z12.s             \n" \
        "fcvt    z13.h, p3/m,  z13.s             \n" \
                                                     \
        "uzp1    z10.h, z10.h, z12.h             \n" \
        "uzp1    z11.h, z11.h, z13.h             \n" \
                                                     \
        "trn1    z8.s,  z14.s, z15.s             \n" \
        "trn2    z9.s,  z14.s, z15.s             \n" \
        "zip1    z14.s, z8.s,  z9.s              \n" \
        "zip2    z15.s, z8.s,  z9.s              \n" \
                                                     \
        "fcvt    z14.h, p3/m,  z14.s             \n" \
        "fcvt    z15.h, p3/m,  z15.s             \n" \
                                                     \
        "trn1    z8.s,  z16.s, z17.s             \n" \
        "trn2    z9.s,  z16.s, z17.s             \n" \
        "zip1    z16.s, z8.s,  z9.s              \n" \
        "zip2    z17.s, z8.s,  z9.s              \n" \
                                                     \
        "fcvt    z16.h, p3/m,  z16.s             \n" \
        "fcvt    z17.h, p3/m,  z17.s             \n" \
                                                     \
        "uzp1    z14.h, z14.h, z16.h             \n" \
        "uzp1    z15.h, z15.h, z17.h             \n" \
                                                     \
        "trn1    z8.s,  z18.s, z19.s             \n" \
        "trn2    z9.s,  z18.s, z19.s             \n" \
        "zip1    z18.s, z8.s,  z9.s              \n" \
        "zip2    z19.s, z8.s,  z9.s              \n" \
                                                     \
        "fcvt    z18.h, p3/m,  z18.s             \n" \
        "fcvt    z19.h, p3/m,  z19.s             \n" \
                                                     \
        "trn1    z8.s,  z20.s, z21.s             \n" \
        "trn2    z9.s,  z20.s, z21.s             \n" \
        "zip1    z20.s, z8.s,  z9.s              \n" \
        "zip2    z21.s, z8.s,  z9.s              \n" \
                                                     \
        "fcvt    z20.h, p3/m,  z20.s             \n" \
        "fcvt    z21.h, p3/m,  z21.s             \n" \
                                                     \
        "uzp1    z18.h, z18.h, z20.h             \n" \
        "uzp1    z19.h, z19.h, z21.h             \n" \
                                                     \
        "trn1    z8.s,  z22.s, z23.s             \n" \
        "trn2    z9.s,  z22.s, z23.s             \n" \
        "zip1    z22.s, z8.s,  z9.s              \n" \
        "zip2    z23.s, z8.s,  z9.s              \n" \
                                                     \
        "fcvt    z22.h, p3/m,  z22.s             \n" \
        "fcvt    z23.h, p3/m,  z23.s             \n" \
                                                     \
        "trn1    z8.s,  z24.s, z25.s             \n" \
        "trn2    z9.s,  z24.s, z25.s             \n" \
        "zip1    z24.s, z8.s,  z9.s              \n" \
        "zip2    z25.s, z8.s,  z9.s              \n" \
                                                     \
        "fcvt    z24.h, p3/m,  z24.s             \n" \
        "fcvt    z25.h, p3/m,  z25.s             \n" \
                                                     \
        "uzp1    z22.h, z22.h, z24.h             \n" \
        "uzp1    z23.h, z23.h, z25.h             \n"

/***********************/

#define ASM_BLOCK_PREFETCH_PART_0                                \
        "prfw    pldl2keep, p0, [%[a_bf16_ptr1], #0, MUL VL] \n" \
        "prfw    pldl2keep, p0, [%[a_bf16_ptr2], #0, MUL VL] \n" \
        "prfw    pldl2keep, p0, [%[a_bf16_ptr3], #0, MUL VL] \n" \
        "prfw    pldl2keep, p0, [%[a_bf16_ptr4], #0, MUL VL] \n"

#define ASM_BLOCK_PREFETCH_PART_1                    \
        "add     x7,    x7,    #1                \n" \
        "cmp     x7,    #2                       \n" \
        "b.any   " LABEL_SKIP_PRF "f             \n" \
        "add     x8,    %[a_bf16_ptr1],  #64     \n" \
        "prfw    pldl2keep, p0, [x8, #0, MUL VL] \n" \
        "add     x8,    %[a_bf16_ptr2],  #64     \n" \
        "prfw    pldl2keep, p0, [x8, #0, MUL VL] \n" \
        "add     x8,    %[a_bf16_ptr3],  #64     \n" \
        "prfw    pldl2keep, p0, [x8, #0, MUL VL] \n" \
        "add     x8,    %[a_bf16_ptr4],  #64     \n" \
        "prfw    pldl2keep, p0, [x8, #0, MUL VL] \n" \
                                                     \
        "add     x8,    %[b_bf16_ptr1],  #64     \n" \
        "prfw    pldl2keep, p0, [x8, #0, MUL VL] \n" \
        "add     x8,    %[b_bf16_ptr2],  #64     \n" \
        "prfw    pldl2keep, p0, [x8, #0, MUL VL] \n" \
        "add     x8,    %[b_bf16_ptr3],  #64     \n" \
        "prfw    pldl2keep, p0, [x8, #0, MUL VL] \n" \
        "add     x8,    %[b_bf16_ptr4],  #64     \n" \
        "prfw    pldl2keep, p0, [x8, #0, MUL VL] \n" \
        "mov     x7,    #0                       \n" \
        " " LABEL_SKIP_PRF ":                    \n"

/***********************/

#define ASM_BLOCK_ADD_BIAS                                           \
    asm volatile(                                                    \
        "ld1w    z0.s,  p1/z, [%[bias_p], #0, MUL VL] \n"            \
        "fadd    z10.s, z10.s, z0.s             \n"                  \
        "fadd    z11.s, z11.s, z0.s             \n"                  \
        "fadd    z14.s, z14.s, z0.s             \n"                  \
        "fadd    z15.s, z15.s, z0.s             \n"                  \
        "fadd    z18.s, z18.s, z0.s             \n"                  \
        "fadd    z19.s, z19.s, z0.s             \n"                  \
        "fadd    z22.s, z22.s, z0.s             \n"                  \
        "fadd    z23.s, z23.s, z0.s             \n"                  \
                                                                     \
        "ld1w    z1.s,  p2/z, [%[bias_p], #1, MUL VL] \n"            \
        "fadd    z12.s, z12.s, z1.s             \n"                  \
        "fadd    z13.s, z13.s, z1.s             \n"                  \
        "fadd    z16.s, z16.s, z1.s             \n"                  \
        "fadd    z17.s, z17.s, z1.s             \n"                  \
        "fadd    z20.s, z20.s, z1.s             \n"                  \
        "fadd    z21.s, z21.s, z1.s             \n"                  \
        "fadd    z24.s, z24.s, z1.s             \n"                  \
        "fadd    z25.s, z25.s, z1.s             \n"                  \
        : /* empty OutputOperands */                                 \
        : [bias_p] "r"(bias_ptr)                                     \
        : "p1", "p2",                                                \
          "z0", "z1", "z10", "z11", "z12", "z13", "z14", "z15",      \
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",    \
          "z24", "z25",                                              \
          "cc", "memory");

/***********************/

#define ASM_BLOCK_C_STORE                                                \
    asm volatile(                                                        \
        "mov     x9,    %[c_fp32_ptr]                \n"                 \
        "st1w    z10.s, p1,   [x9, #0, MUL VL]       \n"                 \
        "st1w    z12.s, p2,   [x9, #1, MUL VL]       \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1w    z11.s, p1,   [x9, #0, MUL VL]       \n"                 \
        "st1w    z13.s, p2,   [x9, #1, MUL VL]       \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1w    z14.s, p1,   [x9, #0, MUL VL]       \n"                 \
        "st1w    z16.s, p2,   [x9, #1, MUL VL]       \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1w    z15.s, p1,   [x9, #0, MUL VL]       \n"                 \
        "st1w    z17.s, p2,   [x9, #1, MUL VL]       \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1w    z18.s, p1,   [x9, #0, MUL VL]       \n"                 \
        "st1w    z20.s, p2,   [x9, #1, MUL VL]       \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1w    z19.s, p1,   [x9, #0, MUL VL]       \n"                 \
        "st1w    z21.s, p2,   [x9, #1, MUL VL]       \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1w    z22.s, p1,   [x9, #0, MUL VL]       \n"                 \
        "st1w    z24.s, p2,   [x9, #1, MUL VL]       \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1w    z23.s, p1,   [x9, #0, MUL VL]       \n"                 \
        "st1w    z25.s, p2,   [x9, #1, MUL VL]       \n"                 \
        : /* empty OutputOperands */                                     \
        : [c_fp32_ptr] "r"(c_fp32_ptr),                                  \
          [next_line_offset] "r"(next_line_offset),                      \
          [M] "r"(M)                                                     \
        : "p1", "p2", "x9",                                              \
          "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", \
          "z20", "z21", "z22", "z23", "z24", "z25",                             \
          "cc", "memory");

#define ASM_BLOCK_C_STORE_FP16                                           \
    asm volatile(                                                        \
        "mov     x9,    %[c_fp16_ptr]                \n"                 \
        "st1h    z10.h, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1h    z11.h, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1h    z14.h, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1h    z15.h, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1h    z18.h, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1h    z19.h, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1h    z22.h, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1h    z23.h, p1,   [x9, #0, MUL VL]       \n"                 \
        : /* empty OutputOperands */                                     \
        : [c_fp16_ptr] "r"(c_fp16_ptr),                                  \
          [next_line_offset] "r"(next_line_offset),                      \
          [M] "r"(M)                                                     \
        : "p1", "p2", "x9",                                              \
          "z10", "z11", "z14", "z15", "z18", "z19",                      \
          "z22", "z23",                                                  \
          "cc", "memory");

#define ASM_BLOCK_C_ACCUMULATE                                           \
    asm volatile(                                                        \
        "mov     x9,    %[c_fp32_ptr]                \n"                 \
        "ld1w    z2.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "ld1w    z3.s,  p2/z, [x9, #1, MUL VL]       \n"                 \
        "fadd    z10.s, z10.s,  z2.s                 \n"                 \
        "fadd    z12.s, z12.s,  z3.s                 \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1w    z2.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "ld1w    z3.s,  p2/z, [x9, #1, MUL VL]       \n"                 \
        "fadd    z11.s, z11.s,  z2.s                 \n"                 \
        "fadd    z13.s, z13.s,  z3.s                 \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1w    z2.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "ld1w    z3.s,  p2/z, [x9, #1, MUL VL]       \n"                 \
        "fadd    z14.s, z14.s,  z2.s                 \n"                 \
        "fadd    z16.s, z16.s,  z3.s                 \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1w    z2.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "ld1w    z3.s,  p2/z, [x9, #1, MUL VL]       \n"                 \
        "fadd    z15.s, z15.s,  z2.s                 \n"                 \
        "fadd    z17.s, z17.s,  z3.s                 \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1w    z2.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "ld1w    z3.s,  p2/z, [x9, #1, MUL VL]       \n"                 \
        "fadd    z18.s, z18.s,  z2.s                 \n"                 \
        "fadd    z20.s, z20.s,  z3.s                 \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1w    z2.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "ld1w    z3.s,  p2/z, [x9, #1, MUL VL]       \n"                 \
        "fadd    z19.s, z19.s,  z2.s                 \n"                 \
        "fadd    z21.s, z21.s,  z3.s                 \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1w    z2.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "ld1w    z3.s,  p2/z, [x9, #1, MUL VL]       \n"                 \
        "fadd    z22.s, z22.s,  z2.s                 \n"                 \
        "fadd    z24.s, z24.s,  z3.s                 \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1w    z2.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "ld1w    z3.s,  p2/z, [x9, #1, MUL VL]       \n"                 \
        "fadd    z23.s, z23.s,  z2.s                 \n"                 \
        "fadd    z25.s, z25.s,  z3.s                 \n"                 \
        : /* empty OutputOperands */                                     \
        : [c_fp32_ptr] "r"(c_fp32_ptr),                                  \
          [next_line_offset] "r"(next_line_offset),                      \
          [M] "r"(M)                                                     \
        : "p1", "p2", "x9",                                              \
          "z2", "z3",                                                    \
          "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", \
          "z20", "z21", "z22", "z23", "z24", "z25",                             \
          "cc", "memory");

#define ASM_BLOCK_C_ACCUMULATE_FP16                                      \
    asm volatile(                                                        \
        "mov     x9,    %[c_fp16_ptr]                \n"                 \
        "ld1h    z2.h,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z10.h, z10.h,  z2.h                 \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1h    z2.h,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z11.h, z11.h,  z2.h                 \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1h    z2.h,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z14.h, z14.h,  z2.h                 \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1h    z2.h,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z15.h, z15.h,  z2.h                 \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1h    z2.h,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z18.h, z18.h,  z2.h                 \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1h    z2.h,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z19.h, z19.h,  z2.h                 \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1h    z2.h,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z22.h, z22.h,  z2.h                 \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1h    z2.h,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z23.h, z23.h,  z2.h                 \n"                 \
        : /* empty OutputOperands */                                     \
        : [c_fp16_ptr] "r"(c_fp16_ptr),                                  \
          [next_line_offset] "r"(next_line_offset),                      \
          [M] "r"(M)                                                     \
        : "p1", "x9",                                                    \
          "z2",                                                          \
          "z10", "z11", "z14", "z15", "z18", "z19",                      \
          "z22", "z23",                                                  \
          "cc", "memory");


#define ASM_BLOCK_C_RES_STORE                                            \
    asm volatile(                                                        \
        "mov     x9,    %[c_fp32_ptr]                \n"                 \
        "st1w    z10.s, p1,   [x9, #0, MUL VL]       \n"                 \
        "st1w    z12.s, p2,   [x9, #1, MUL VL]       \n"                 \
                                                                         \
        /* if (m + 1) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #1                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_STORE "f               \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1w    z11.s, p1,   [x9, #0, MUL VL]       \n"                 \
        "st1w    z13.s, p2,   [x9, #1, MUL VL]       \n"                 \
                                                                         \
        /* if (m + 2) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #2                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_STORE "f               \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1w    z14.s, p1,   [x9, #0, MUL VL]       \n"                 \
        "st1w    z16.s, p2,   [x9, #1, MUL VL]       \n"                 \
                                                                         \
        /* if (m + 3) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #3                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_STORE "f               \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1w    z15.s, p1,   [x9, #0, MUL VL]       \n"                 \
        "st1w    z17.s, p2,   [x9, #1, MUL VL]       \n"                 \
                                                                         \
        /* if (m + 4) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #4                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_STORE "f               \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1w    z18.s, p1,   [x9, #0, MUL VL]       \n"                 \
        "st1w    z20.s, p2,   [x9, #1, MUL VL]       \n"                 \
                                                                         \
        /* if (m + 5) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #5                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_STORE "f               \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1w    z19.s, p1,   [x9, #0, MUL VL]       \n"                 \
        "st1w    z21.s, p2,   [x9, #1, MUL VL]       \n"                 \
                                                                         \
        /* if (m + 6) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #6                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_STORE "f               \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1w    z22.s, p1,   [x9, #0, MUL VL]       \n"                 \
        "st1w    z24.s, p2,   [x9, #1, MUL VL]       \n"                 \
                                                                         \
        /* if (m + 7) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #7                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_STORE "f               \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1w    z23.s, p1,   [x9, #0, MUL VL]       \n"                 \
        "st1w    z25.s, p2,   [x9, #1, MUL VL]       \n"                 \
                                                                         \
        " " LABEL_SKIP_STORE ":\n"                                       \
        : /* empty OutputOperands */                                     \
        : [c_fp32_ptr] "r"(c_fp32_ptr),                                  \
          [next_line_offset] "r"(next_line_offset),                      \
          [M] "r"(M)                                                     \
        : "p1", "p2", "x2", "x5", "x9",                                  \
          "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", \
          "z20", "z21", "z22", "z23", "z24", "z25",                             \
          "cc", "memory");

#define ASM_BLOCK_C_RES_STORE_FP16                                       \
    asm volatile(                                                        \
        "mov     x9,    %[c_fp16_ptr]                \n"                 \
        "st1h    z10.h, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        /* if (m + 1) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #1                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_STORE "f               \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1h    z11.h, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        /* if (m + 2) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #2                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_STORE "f               \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1h    z14.h, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        /* if (m + 3) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #3                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_STORE "f               \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1h    z15.h, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        /* if (m + 4) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #4                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_STORE "f               \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1h    z18.h, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        /* if (m + 5) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #5                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_STORE "f               \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1h    z19.h, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        /* if (m + 6) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #6                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_STORE "f               \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1h    z22.h, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        /* if (m + 7) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #7                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_STORE "f               \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1h    z23.h, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        " " LABEL_SKIP_STORE ":\n"                                       \
        : /* empty OutputOperands */                                     \
        : [c_fp16_ptr] "r"(c_fp16_ptr),                                  \
          [next_line_offset] "r"(next_line_offset),                      \
          [M] "r"(M)                                                     \
        : "p1", "p2", "x2", "x5", "x9",                                  \
          "z10", "z11", "z14", "z15", "z18", "z19",                      \
          "z22", "z23",                                                  \
          "cc", "memory");

#define ASM_BLOCK_C_RES_ACCUMULATE                                       \
    asm volatile(                                                        \
        "mov     x9,    %[c_fp32_ptr]                \n"                 \
        "ld1w    z2.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "ld1w    z3.s,  p2/z, [x9, #1, MUL VL]       \n"                 \
        "fadd    z10.s, z10.s,  z2.s                 \n"                 \
        "fadd    z12.s, z12.s,  z3.s                 \n"                 \
                                                                         \
        /* if (m + 1) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #1                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_ACCUMULATE "f          \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1w    z2.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "ld1w    z3.s,  p2/z, [x9, #1, MUL VL]       \n"                 \
        "fadd    z11.s, z11.s,  z2.s                 \n"                 \
        "fadd    z13.s, z13.s,  z3.s                 \n"                 \
                                                                         \
        /* if (m + 2) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #2                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_ACCUMULATE "f          \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1w    z2.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "ld1w    z3.s,  p2/z, [x9, #1, MUL VL]       \n"                 \
        "fadd    z14.s, z14.s,  z2.s                 \n"                 \
        "fadd    z16.s, z16.s,  z3.s                 \n"                 \
                                                                         \
        /* if (m + 3) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #3                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_ACCUMULATE "f          \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1w    z2.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "ld1w    z3.s,  p2/z, [x9, #1, MUL VL]       \n"                 \
        "fadd    z15.s, z15.s,  z2.s                 \n"                 \
        "fadd    z17.s, z17.s,  z3.s                 \n"                 \
                                                                         \
        /* if (m + 4) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #4                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_ACCUMULATE "f          \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1w    z2.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "ld1w    z3.s,  p2/z, [x9, #1, MUL VL]       \n"                 \
        "fadd    z18.s, z18.s,  z2.s                 \n"                 \
        "fadd    z20.s, z20.s,  z3.s                 \n"                 \
                                                                         \
        /* if (m + 5) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #5                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_ACCUMULATE "f          \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1w    z2.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "ld1w    z3.s,  p2/z, [x9, #1, MUL VL]       \n"                 \
        "fadd    z19.s, z19.s,  z2.s                 \n"                 \
        "fadd    z21.s, z21.s,  z3.s                 \n"                 \
                                                                         \
        /* if (m + 6) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #6                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_ACCUMULATE "f          \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1w    z2.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "ld1w    z3.s,  p2/z, [x9, #1, MUL VL]       \n"                 \
        "fadd    z22.s, z22.s,  z2.s                 \n"                 \
        "fadd    z24.s, z24.s,  z3.s                 \n"                 \
                                                                         \
        /* if (m + 7) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #7                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_ACCUMULATE "f          \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1w    z2.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "ld1w    z3.s,  p2/z, [x9, #1, MUL VL]       \n"                 \
        "fadd    z23.s, z23.s,  z2.s                 \n"                 \
        "fadd    z25.s, z25.s,  z3.s                 \n"                 \
                                                                         \
        " " LABEL_SKIP_ACCUMULATE ":\n"                                  \
        : /* empty OutputOperands */                                     \
        : [c_fp32_ptr] "r"(c_fp32_ptr),                                  \
          [next_line_offset] "r"(next_line_offset),                      \
          [M] "r"(M)                                                     \
        : "p1", "p2", "x2", "x5", "x9",                                  \
          "z2", "z3",                                                    \
          "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", \
          "z20", "z21", "z22", "z23", "z24", "z25",                             \
          "cc", "memory");

#define ASM_BLOCK_C_RES_ACCUMULATE_FP16                                  \
    asm volatile(                                                        \
        "mov     x9,    %[c_fp16_ptr]                \n"                 \
        "ld1h    z2.h,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z10.h, z10.h,  z2.h                 \n"                 \
                                                                         \
        /* if (m + 1) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #1                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_ACCUMULATE "f          \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1h    z2.h,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z11.h, z11.h,  z2.h                 \n"                 \
                                                                         \
        /* if (m + 2) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #2                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_ACCUMULATE "f          \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1h    z2.h,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z14.h, z14.h,  z2.h                 \n"                 \
                                                                         \
        /* if (m + 3) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #3                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_ACCUMULATE "f          \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1h    z2.h,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z15.h, z15.h,  z2.h                 \n"                 \
                                                                         \
        /* if (m + 4) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #4                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_ACCUMULATE "f          \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1h    z2.h,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z18.h, z18.h,  z2.h                 \n"                 \
                                                                         \
        /* if (m + 5) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #5                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_ACCUMULATE "f          \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1h    z2.h,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z19.h, z19.h,  z2.h                 \n"                 \
                                                                         \
        /* if (m + 6) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #6                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_ACCUMULATE "f          \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1h    z2.h,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z22.h, z22.h,  z2.h                 \n"                 \
                                                                         \
        /* if (m + 7) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #7                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_ACCUMULATE "f          \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1h    z2.h,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z23.h, z23.h,  z2.h                 \n"                 \
                                                                         \
        " " LABEL_SKIP_ACCUMULATE ":\n"                                  \
        : /* empty OutputOperands */                                     \
        : [c_fp16_ptr] "r"(c_fp16_ptr),                                  \
          [next_line_offset] "r"(next_line_offset),                      \
          [M] "r"(M)                                                     \
        : "p1", "p2", "x2", "x5", "x9",                                  \
          "z2",                                                          \
          "z10", "z11", "z14", "z15", "z18", "z19",                      \
          "z22", "z23",                                                  \
          "cc", "memory");
/***********************/

#define ASM_BLOCK_ACTIVE_RELU                 \
    asm volatile(                             \
        "fmax    z10.s, p0/m, z10.s, #0.0 \n" \
        "fmax    z11.s, p0/m, z11.s, #0.0 \n" \
        "fmax    z12.s, p0/m, z12.s, #0.0 \n" \
        "fmax    z13.s, p0/m, z13.s, #0.0 \n" \
        "fmax    z14.s, p0/m, z14.s, #0.0 \n" \
        "fmax    z15.s, p0/m, z15.s, #0.0 \n" \
        "fmax    z16.s, p0/m, z16.s, #0.0 \n" \
        "fmax    z17.s, p0/m, z17.s, #0.0 \n" \
        "fmax    z18.s, p0/m, z18.s, #0.0 \n" \
        "fmax    z19.s, p0/m, z19.s, #0.0 \n" \
        "fmax    z20.s, p0/m, z20.s, #0.0 \n" \
        "fmax    z21.s, p0/m, z21.s, #0.0 \n" \
        "fmax    z22.s, p0/m, z22.s, #0.0 \n" \
        "fmax    z23.s, p0/m, z23.s, #0.0 \n" \
        "fmax    z24.s, p0/m, z24.s, #0.0 \n" \
        "fmax    z25.s, p0/m, z25.s, #0.0 \n" \
        : /* empty OutputOperands */          \
        : /* empty InputOperands */           \
        : "p0",                               \
        "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", \
        "z20", "z21", "z22", "z23", "z24", "z25",                             \
        "cc", "memory");


#define ASM_BLOCK_ACTIVE_SILU                                \
    asm volatile(                                            \
        "ld1w    z0.s,  p0/z, [%[exp_const], #0, MUL VL] \n" \
        "ld1w    z1.s,  p0/z, [%[exp_const], #1, MUL VL] \n" \
        "ld1w    z2.s,  p0/z, [%[exp_const], #2, MUL VL] \n" \
                                                             \
        ASM_BLOCK_SILU_MICRO(z10, z10, z0, z1, z2,           \
                            z3, z4, z5, z6, z7,              \
                            p0, p3)                          \
                                                             \
        ASM_BLOCK_SILU_MICRO(z11, z11, z0, z1, z2,           \
                            z3, z4, z5, z6, z7,              \
                            p0, p3)                          \
                                                             \
        ASM_BLOCK_SILU_MICRO(z12, z12, z0, z1, z2,           \
                            z3, z4, z5, z6, z7,              \
                            p0, p3)                          \
                                                             \
        ASM_BLOCK_SILU_MICRO(z13, z13, z0, z1, z2,           \
                            z3, z4, z5, z6, z7,              \
                            p0, p3)                          \
                                                             \
        ASM_BLOCK_SILU_MICRO(z14, z14, z0, z1, z2,           \
                            z3, z4, z5, z6, z7,              \
                            p0, p3)                          \
                                                             \
        ASM_BLOCK_SILU_MICRO(z15, z15, z0, z1, z2,           \
                            z3, z4, z5, z6, z7,              \
                            p0, p3)                          \
                                                             \
        ASM_BLOCK_SILU_MICRO(z16, z16, z0, z1, z2,           \
                            z3, z4, z5, z6, z7,              \
                            p0, p3)                          \
                                                             \
        ASM_BLOCK_SILU_MICRO(z17, z17, z0, z1, z2,           \
                            z3, z4, z5, z6, z7,              \
                            p0, p3)                          \
                                                             \
        ASM_BLOCK_SILU_MICRO(z18, z18, z0, z1, z2,           \
                            z3, z4, z5, z6, z7,              \
                            p0, p3)                          \
                                                             \
        ASM_BLOCK_SILU_MICRO(z19, z19, z0, z1, z2,           \
                            z3, z4, z5, z6, z7,              \
                            p0, p3)                          \
                                                             \
        ASM_BLOCK_SILU_MICRO(z20, z20, z0, z1, z2,           \
                            z3, z4, z5, z6, z7,              \
                            p0, p3)                          \
                                                             \
        ASM_BLOCK_SILU_MICRO(z21, z21, z0, z1, z2,           \
                            z3, z4, z5, z6, z7,              \
                            p0, p3)                          \
                                                             \
        ASM_BLOCK_SILU_MICRO(z22, z22, z0, z1, z2,           \
                            z3, z4, z5, z6, z7,              \
                            p0, p3)                          \
                                                             \
        ASM_BLOCK_SILU_MICRO(z23, z23, z0, z1, z2,           \
                            z3, z4, z5, z6, z7,              \
                            p0, p3)                          \
                                                             \
        ASM_BLOCK_SILU_MICRO(z24, z24, z0, z1, z2,           \
                            z3, z4, z5, z6, z7,              \
                            p0, p3)                          \
                                                             \
        ASM_BLOCK_SILU_MICRO(z25, z25, z0, z1, z2,           \
                            z3, z4, z5, z6, z7,              \
                            p0, p3)                          \
                                                             \
        : /* empty OutputOperands */                         \
        : [exp_const] "r"(constant.exp_const)                \
        : "p0", "p3",                                        \
        "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",      \
        "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", \
        "z20", "z21", "z22", "z23", "z24", "z25",                             \
        "cc", "memory");


#define ASM_BLOCK_ACTIVE_TANH                              \
    asm volatile(                                          \
      "ld1w    z0.s,  p0/z, [%[exp_const], #0, MUL VL] \n" \
      "ld1w    z1.s,  p0/z, [%[exp_const], #1, MUL VL] \n" \
      "ld1w    z2.s,  p0/z, [%[exp_const], #2, MUL VL] \n" \
                                                           \
      "mov     z3.s, p0/m, z10.s                       \n" \
      ASM_BLOCK_TANH(z3, z10, z0, z1, z2,                  \
                     z4, z5, z6, p0, p3)                   \
                                                           \
      "mov     z3.s, p0/m, z11.s                       \n" \
      ASM_BLOCK_TANH(z3, z11, z0, z1, z2,                  \
                     z4, z5, z6, p0, p3)                   \
                                                           \
      "mov     z3.s, p0/m, z12.s                       \n" \
      ASM_BLOCK_TANH(z3, z12, z0, z1, z2,                  \
                     z4, z5, z6, p0, p3)                   \
                                                           \
      "mov     z3.s, p0/m, z13.s                       \n" \
      ASM_BLOCK_TANH(z3, z13, z0, z1, z2,                  \
                     z4, z5, z6, p0, p3)                   \
                                                           \
      "mov     z3.s, p0/m, z14.s                       \n" \
      ASM_BLOCK_TANH(z3, z14, z0, z1, z2,                  \
                     z4, z5, z6, p0, p3)                   \
                                                           \
      "mov     z3.s, p0/m, z15.s                       \n" \
      ASM_BLOCK_TANH(z3, z15, z0, z1, z2,                  \
                     z4, z5, z6, p0, p3)                   \
                                                           \
      "mov     z3.s, p0/m, z16.s                       \n" \
      ASM_BLOCK_TANH(z3, z16, z0, z1, z2,                  \
                     z4, z5, z6, p0, p3)                   \
                                                           \
      "mov     z3.s, p0/m, z17.s                       \n" \
      ASM_BLOCK_TANH(z3, z17, z0, z1, z2,                  \
                     z4, z5, z6, p0, p3)                   \
                                                           \
      "mov     z3.s, p0/m, z18.s                       \n" \
      ASM_BLOCK_TANH(z3, z18, z0, z1, z2,                  \
                     z4, z5, z6, p0, p3)                   \
                                                           \
      "mov     z3.s, p0/m, z19.s                       \n" \
      ASM_BLOCK_TANH(z3, z19, z0, z1, z2,                  \
                     z4, z5, z6, p0, p3)                   \
                                                           \
      "mov     z3.s, p0/m, z20.s                       \n" \
      ASM_BLOCK_TANH(z3, z20, z0, z1, z2,                  \
                     z4, z5, z6, p0, p3)                   \
                                                           \
      "mov     z3.s, p0/m, z21.s                       \n" \
      ASM_BLOCK_TANH(z3, z21, z0, z1, z2,                  \
                     z4, z5, z6, p0, p3)                   \
                                                           \
      "mov     z3.s, p0/m, z22.s                       \n" \
      ASM_BLOCK_TANH(z3, z22, z0, z1, z2,                  \
                     z4, z5, z6, p0, p3)                   \
                                                           \
      "mov     z3.s, p0/m, z23.s                       \n" \
      ASM_BLOCK_TANH(z3, z23, z0, z1, z2,                  \
                     z4, z5, z6, p0, p3)                   \
                                                           \
      "mov     z3.s, p0/m, z24.s                       \n" \
      ASM_BLOCK_TANH(z3, z24, z0, z1, z2,                  \
                     z4, z5, z6, p0, p3)                   \
                                                           \
      "mov     z3.s, p0/m, z25.s                       \n" \
      ASM_BLOCK_TANH(z3, z25, z0, z1, z2,                  \
                     z4, z5, z6, p0, p3)                   \
                                                           \
      : /* empty OutputOperands */                         \
      : [exp_const] "r"(constant.exp_const)                \
      : "p0", "p3",                                        \
        "z0", "z1", "z2", "z3", "z4", "z5", "z6",          \
        "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", \
        "z20", "z21", "z22", "z23", "z24", "z25",                             \
        "cc", "memory");


#define ASM_BLOCK_ACTIVE_GELU_ERF                                  \
    asm volatile(                                                  \
        "ld1w    z0.s,  p0/z, [%[exp_const], #0, MUL VL] \n"       \
        "ld1w    z1.s,  p0/z, [%[exp_const], #1, MUL VL] \n"       \
        "ld1w    z2.s,  p0/z, [%[exp_const], #2, MUL VL] \n"       \
                                                                   \
        "ld1w    z3.s,  p0/z, [%[erf_const], #0, MUL VL] \n"       \
        "ld1w    z4.s,  p0/z, [%[erf_const], #1, MUL VL] \n"       \
                                                                   \
        ASM_BLOCK_GELU_ERF_MICRO(z10, z10,                         \
                                z0, z1, z2, z3, z4,                \
                                z5, z6, z7, z8, z9, z26, z27,      \
                                p0, p3)                            \
                                                                   \
        ASM_BLOCK_GELU_ERF_MICRO(z11, z11,                         \
                                z0, z1, z2, z3, z4,                \
                                z5, z6, z7, z8, z9, z26, z27,      \
                                p0, p3)                            \
                                                                   \
        ASM_BLOCK_GELU_ERF_MICRO(z12, z12,                         \
                                z0, z1, z2, z3, z4,                \
                                z5, z6, z7, z8, z9, z26, z27,      \
                                p0, p3)                            \
                                                                   \
        ASM_BLOCK_GELU_ERF_MICRO(z13, z13,                         \
                                z0, z1, z2, z3, z4,                \
                                z5, z6, z7, z8, z9, z26, z27,      \
                                p0, p3)                            \
                                                                   \
        ASM_BLOCK_GELU_ERF_MICRO(z14, z14,                         \
                                z0, z1, z2, z3, z4,                \
                                z5, z6, z7, z8, z9, z26, z27,      \
                                p0, p3)                            \
                                                                   \
        ASM_BLOCK_GELU_ERF_MICRO(z15, z15,                         \
                                z0, z1, z2, z3, z4,                \
                                z5, z6, z7, z8, z9, z26, z27,      \
                                p0, p3)                            \
                                                                   \
        ASM_BLOCK_GELU_ERF_MICRO(z16, z16,                         \
                                z0, z1, z2, z3, z4,                \
                                z5, z6, z7, z8, z9, z26, z27,      \
                                p0, p3)                            \
                                                                   \
        ASM_BLOCK_GELU_ERF_MICRO(z17, z17,                         \
                                z0, z1, z2, z3, z4,                \
                                z5, z6, z7, z8, z9, z26, z27,      \
                                p0, p3)                            \
                                                                   \
        ASM_BLOCK_GELU_ERF_MICRO(z18, z18,                         \
                                z0, z1, z2, z3, z4,                \
                                z5, z6, z7, z8, z9, z26, z27,      \
                                p0, p3)                            \
                                                                   \
        ASM_BLOCK_GELU_ERF_MICRO(z19, z19,                         \
                                z0, z1, z2, z3, z4,                \
                                z5, z6, z7, z8, z9, z26, z27,      \
                                p0, p3)                            \
                                                                   \
        ASM_BLOCK_GELU_ERF_MICRO(z20, z20,                         \
                                z0, z1, z2, z3, z4,                \
                                z5, z6, z7, z8, z9, z26, z27,      \
                                p0, p3)                            \
                                                                   \
        ASM_BLOCK_GELU_ERF_MICRO(z21, z21,                         \
                                z0, z1, z2, z3, z4,                \
                                z5, z6, z7, z8, z9, z26, z27,      \
                                p0, p3)                            \
                                                                   \
        ASM_BLOCK_GELU_ERF_MICRO(z22, z22,                         \
                                z0, z1, z2, z3, z4,                \
                                z5, z6, z7, z8, z9, z26, z27,      \
                                p0, p3)                            \
                                                                   \
        ASM_BLOCK_GELU_ERF_MICRO(z23, z23,                         \
                                z0, z1, z2, z3, z4,                \
                                z5, z6, z7, z8, z9, z26, z27,      \
                                p0, p3)                            \
                                                                   \
        ASM_BLOCK_GELU_ERF_MICRO(z24, z24,                         \
                                z0, z1, z2, z3, z4,                \
                                z5, z6, z7, z8, z9, z26, z27,      \
                                p0, p3)                            \
                                                                   \
        ASM_BLOCK_GELU_ERF_MICRO(z25, z25,                         \
                                z0, z1, z2, z3, z4,                \
                                z5, z6, z7, z8, z9, z26, z27,      \
                                p0, p3)                            \
                                                                   \
        : /* empty OutputOperands */                               \
        : [exp_const] "r"(constant.exp_const),                     \
        [erf_const] "r"(constant.erf_const),                       \
        [inv_sqrt] "r"(constant.inv_sqrt)                          \
        : "p0", "p3",                                              \
        "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9",           \
        "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", \
        "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27",               \
        "cc", "memory");


#define ASM_BLOCK_ACTIVE_GELU_TANH                           \
    asm volatile(                                            \
        "ld1w    z0.s,  p0/z, [%[exp_const], #0, MUL VL] \n" \
        "ld1w    z1.s,  p0/z, [%[exp_const], #1, MUL VL] \n" \
        "ld1w    z2.s,  p0/z, [%[exp_const], #2, MUL VL] \n" \
                                                             \
        ASM_BLOCK_GELU_TANH_MICRO(z10, z10,                  \
                                z0, z1, z2,                  \
                                z3, z4, z5, z6, z7,          \
                                p0, p3)                      \
                                                             \
        ASM_BLOCK_GELU_TANH_MICRO(z11, z11,                  \
                                z0, z1, z2,                  \
                                z3, z4, z5, z6, z7,          \
                                p0, p3)                      \
                                                             \
        ASM_BLOCK_GELU_TANH_MICRO(z12, z12,                  \
                                z0, z1, z2,                  \
                                z3, z4, z5, z6, z7,          \
                                p0, p3)                      \
                                                             \
        ASM_BLOCK_GELU_TANH_MICRO(z13, z13,                  \
                                z0, z1, z2,                  \
                                z3, z4, z5, z6, z7,          \
                                p0, p3)                      \
                                                             \
        ASM_BLOCK_GELU_TANH_MICRO(z14, z14,                  \
                                z0, z1, z2,                  \
                                z3, z4, z5, z6, z7,          \
                                p0, p3)                      \
                                                             \
        ASM_BLOCK_GELU_TANH_MICRO(z15, z15,                  \
                                z0, z1, z2,                  \
                                z3, z4, z5, z6, z7,          \
                                p0, p3)                      \
                                                             \
        ASM_BLOCK_GELU_TANH_MICRO(z16, z16,                  \
                                z0, z1, z2,                  \
                                z3, z4, z5, z6, z7,          \
                                p0, p3)                      \
                                                             \
        ASM_BLOCK_GELU_TANH_MICRO(z17, z17,                  \
                                z0, z1, z2,                  \
                                z3, z4, z5, z6, z7,          \
                                p0, p3)                      \
                                                             \
        ASM_BLOCK_GELU_TANH_MICRO(z18, z18,                  \
                                z0, z1, z2,                  \
                                z3, z4, z5, z6, z7,          \
                                p0, p3)                      \
                                                             \
        ASM_BLOCK_GELU_TANH_MICRO(z19, z19,                  \
                                z0, z1, z2,                  \
                                z3, z4, z5, z6, z7,          \
                                p0, p3)                      \
                                                             \
        ASM_BLOCK_GELU_TANH_MICRO(z20, z20,                  \
                                z0, z1, z2,                  \
                                z3, z4, z5, z6, z7,          \
                                p0, p3)                      \
                                                             \
        ASM_BLOCK_GELU_TANH_MICRO(z21, z21,                  \
                                z0, z1, z2,                  \
                                z3, z4, z5, z6, z7,          \
                                p0, p3)                      \
                                                             \
        ASM_BLOCK_GELU_TANH_MICRO(z22, z22,                  \
                                z0, z1, z2,                  \
                                z3, z4, z5, z6, z7,          \
                                p0, p3)                      \
                                                             \
        ASM_BLOCK_GELU_TANH_MICRO(z23, z23,                  \
                                z0, z1, z2,                  \
                                z3, z4, z5, z6, z7,          \
                                p0, p3)                      \
                                                             \
        ASM_BLOCK_GELU_TANH_MICRO(z24, z24,                  \
                                z0, z1, z2,                  \
                                z3, z4, z5, z6, z7,          \
                                p0, p3)                      \
                                                             \
        ASM_BLOCK_GELU_TANH_MICRO(z25, z25,                  \
                                z0, z1, z2,                  \
                                z3, z4, z5, z6, z7,          \
                                p0, p3)                      \
                                                             \
        : /* empty OutputOperands */                         \
        : [exp_const] "r"(constant.exp_const),               \
        [erf_const] "r"(constant.erf_const),                 \
        [const1] "r"(constant.gelu_tanh_const[0]),           \
        [const2] "r"(constant.gelu_tanh_const[1])            \
        : "p0", "p3",                                        \
        "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",      \
        "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", \
        "z20", "z21", "z22", "z23", "z24", "z25",                             \
        "cc", "memory");
// clang-format on

/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    activation_macro.h
 */
#pragma once
#include <arm_sve.h>

// clang-format off
#define ASM_BLOCK_EXP_ARMCL(SRC, DST,                           \
                            CONST1, CONST2, CONST3,             \
                            PRED1, PRED2, PRED3,                \
                            TMP1, TMP2, TMP3, TMP4, TMP5)       \
      "dup     "#TMP1".s,  "#CONST3".s[2]                   \n" /* min_input         */             \
      "dup     "#TMP2".s,  "#CONST3".s[1]                   \n" /* max_input         */             \
      "fcmlt   "#PRED2".s, "#PRED1"/z, "#SRC".s, "#TMP1".s  \n" /* cmp(x, min_input) */             \
      "fcmgt   "#PRED3".s, "#PRED1"/z, "#SRC".s, "#TMP2".s  \n" /* cmp(x, max_input) */             \
                                                                                                    \
      "dup     "#TMP1".s,  "#CONST2".s[1]                   \n" /* shift   */                       \
      "dup     "#TMP2".s,  "#CONST2".s[1]                   \n" /* shift   */                       \
      "dup     "#DST".s,   "#CONST2".s[2]                   \n" /* inv_ln2 */                       \
                                                                                                    \
      "fmla    "#TMP1".s,  "#PRED1"/m, "#SRC".s, "#DST".s   \n" /* z = x / ln(2) + 2^23 + 127    */ \
      "mov     "#DST".s,   "#PRED1"/m, "#TMP1".s            \n"                                     \
      "fsub    "#TMP1".s,  "#PRED1"/m, "#TMP1".s, "#TMP2".s \n" /* n = z - shift                 */ \
      "lsl     "#DST".s,   "#DST".s, #23                    \n" /* scale = (z << 23)             */ \
                                                                                                    \
      "dup     "#TMP2".s,  "#CONST2".s[3]                   \n" /* neg_ln2_hi                    */ \
      "dup     "#TMP3".s,  "#CONST3".s[0]                   \n" /* neg_ln2_lo                    */ \
      "fmla    "#SRC".s,   "#PRED1"/m, "#TMP1".s, "#TMP2".s \n" /* r_hi = n * neg_ln2_hi + x     */ \
      "fmla    "#SRC".s,   "#PRED1"/m, "#TMP1".s, "#TMP3".s \n" /* r = n * neg_ln2_hi + r_hi     */ \
                                                                                                    \
      "mov     "#TMP1".s,  "#PRED1"/m, "#SRC".s             \n"                                     \
      "fmul    "#TMP1".s,  "#PRED1"/m, "#TMP1".s, "#TMP1".s \n" /* r2 = r * r                    */ \
      "dup     "#TMP2".s,  "#CONST1".s[0]                   \n" /* exp_coeff[0], c1              */ \
      "fmul    "#TMP2".s,  "#PRED1"/m, "#TMP2".s, "#SRC".s  \n" /* p1 = c1 * r                   */ \
      "dup     "#TMP3".s,  "#CONST1".s[1]                   \n" /* exp_coeff[1], c2              */ \
      "dup     "#TMP4".s,  "#CONST1".s[2]                   \n" /* exp_coeff[2], c3              */ \
      "fmla    "#TMP3".s,  "#PRED1"/m, "#TMP4".s, "#SRC".s  \n" /* p23 = c2 + c3 * r             */ \
                                                                                                    \
      "dup     "#TMP4".s,  "#CONST1".s[3]                   \n" /* exp_coeff[3], c4              */ \
      "dup     "#TMP5".s,  "#CONST2".s[0]                   \n" /* exp_coeff[4], c5              */ \
      "fmla    "#TMP4".s,  "#PRED1"/m, "#TMP5".s, "#SRC".s  \n" /* p45 = c4 + c5 * r             */ \
      "fmla    "#TMP3".s,  "#PRED1"/m, "#TMP4".s, "#TMP1".s \n" /* p2345 = p23 + p45 * r2        */ \
      "fmla    "#TMP2".s,  "#PRED1"/m, "#TMP3".s, "#TMP1".s \n" /* p12345 = p1 + p2345 * r2      */ \
      "fmla    "#DST".s,   "#PRED1"/m, "#TMP2".s, "#DST".s  \n" /* poly = scale + p12345 * scale */ \
                                                                                                    \
      "dup     "#TMP1".s,  #0x0                             \n" /* zero */                          \
      "dup     "#TMP2".s,  "#CONST3".s[3]                   \n" /* inf  */                          \
                                                                                                    \
      "sel     "#DST".s,   "#PRED2",   "#TMP1".s, "#DST".s  \n" /* if (x < min_input), y = 0   */   \
      "sel     "#DST".s,   "#PRED3",   "#TMP2".s, "#DST".s  \n" /* if (x > max_input), y = inf */ 


#define ASM_BLOCK_EXP(SRC,  DST,  CONST1, CONST2, CONST3,       \
                      TMP1, TMP2, TMP3, PRED1, PRED2)           \
      "dup     "#TMP2".s,  "#CONST3".s[1]                   \n" /* max_input */              \
      "fmin    "#SRC".s,   "#PRED1"/m, "#SRC".s, "#TMP2".s  \n" /* cmp(x, max_input) */      \
                                                                                             \
      "fdup    "#TMP1".s,  #0.5                             \n"                              \
      "dup     "#TMP2".s,  "#CONST2".s[2]                   \n" /* inv_ln2 */                \
      "mov     "#DST".s,   "#PRED1"/m, "#SRC".s             \n" /* x */                      \
      "fmad    "#DST".s,   "#PRED1"/m, "#TMP2".s, "#TMP1".s \n" /* fx = x / ln(2) + 0.5 */   \
                                                                                             \
      "fcvtzs  "#TMP2".s,  "#PRED1"/m, "#DST".s             \n" /* round_to_zero(fx), s32 */ \
      "scvtf   "#TMP1".s,  "#PRED1"/m, "#TMP2".s            \n" /* round_to_zero(fx), f32 */ \
      "fcmgt   "#PRED2".s, "#PRED1"/z, "#TMP1".s, "#DST".s  \n"                              \
      "mov     "#DST".s,   "#PRED1"/m, "#TMP2".s            \n"                              \
      "sub     "#DST".s,   "#DST".s,   #1                   \n"                              \
      "sel     "#TMP3".s,  "#PRED2",   "#DST".s,  "#TMP2".s \n" /* n = floor(fx), s32 */     \
                                                                                             \
      "dup     "#TMP1".s,  "#CONST3".s[2]                   \n" /* min_input */              \
      "fcmlt   "#PRED2".s, "#PRED1"/z, "#SRC".s, "#TMP1".s  \n" /* cmp(x, min_input) */      \
                                                                                             \
      "mov     "#DST".s,   "#PRED1"/m, "#TMP3".s            \n" /* n */                      \
      "dup     "#TMP2".s,  "#CONST2".s[1]                   \n" /* exponent_bias - 1 */      \
      "add     "#DST".s,   "#PRED1"/m, "#DST".s,  "#TMP2".s \n" /* n + exponent_bias - 1 */  \
      "lsl     "#DST".s,   "#DST".s,   #23                  \n" /* scale = (n + exponent_bias - 1) << 23 */ \
      "scvtf   "#TMP1".s,  "#PRED1"/m, "#TMP3".s            \n" /* n = floor(fx), f32 */     \
                                                                                             \
      "dup     "#TMP2".s,  "#CONST2".s[3]                   \n" /* neg_ln2_hi */             \
      "dup     "#TMP3".s,  "#CONST3".s[0]                   \n" /* neg_ln2_lo */             \
      "fmla    "#SRC".s,   "#PRED1"/m, "#TMP1".s, "#TMP2".s \n" /* r_hi = n * neg_ln2_hi + x */ \
      "fmla    "#SRC".s,   "#PRED1"/m, "#TMP1".s, "#TMP3".s \n" /* r = n * neg_ln2_hi + r_hi */ \
                                                                                             \
      "dup     "#TMP1".s,  "#CONST2".s[0]                   \n" /* p5 */                     \
      "dup     "#TMP2".s,  "#CONST1".s[3]                   \n" /* p4 */                     \
      "fmad    "#TMP1".s,  "#PRED1"/m, "#SRC".s,  "#TMP2".s \n" /* p4+p5*r */                \
      "dup     "#TMP2".s,  "#CONST1".s[2]                   \n" /* p3 */                     \
      "fmad    "#TMP1".s,  "#PRED1"/m, "#SRC".s, "#TMP2".s  \n" /* p3+p4*r+p5*r^2 */         \
      "dup     "#TMP2".s,  "#CONST1".s[1]                   \n" /* p2 */                     \
      "fmad    "#TMP1".s,  "#PRED1"/m, "#SRC".s, "#TMP2".s  \n" /* p2+p3*r+p4*r^2+p5*r^3 */  \
      "dup     "#TMP2".s,  "#CONST1".s[0]                   \n" /* p1 */                     \
      "fmad    "#TMP1".s,  "#PRED1"/m, "#SRC".s, "#TMP2".s  \n" /* p1+p2*r+p3*r^2+p4*r^3+p5*r^4 */ \
      "fdup    "#TMP2".s,  #1.0                             \n" /* p0 */                     \
      "fmul    "#TMP1".s,  "#PRED1"/m, "#TMP1".s, "#SRC".s  \n" /* p1*r+p2*r^2+p3*r^3+p4*r^4+p5*r^5 */ \
                                                                                             \
      "fmla    "#DST".s,  "#PRED1"/m, "#TMP1".s, "#DST".s   \n" /* poly = scale + p12345 * scale */ \
      "fadd    "#DST".s,  "#PRED1"/m, "#DST".s,  "#DST".s   \n" /* poly *= 2 */              \
                                                                                             \
      "dup     "#TMP1".s, #0x0                              \n" /* zero */                   \
      "sel     "#DST".s,  "#PRED2",   "#TMP1".s, "#DST".s   \n" /* if (x < min_input), y = 0 */

#define ASM_BLOCK_SILU_MICRO(SRC,  DST,  CONST1, CONST2, CONST3, \
                             TMP1, TMP2, TMP3, TMP4, TMP5,       \
                             PRED1, PRED2)                       \
      "mov     "#TMP1".s,  "#PRED1"/m, "#SRC".s              \n" \
      "fneg    "#TMP1".s,  "#PRED1"/m, "#TMP1".s             \n" \
                                                                 \
      ASM_BLOCK_EXP(TMP1, TMP2, CONST1, CONST2, CONST3,          \
                    TMP3, TMP4, TMP5, PRED1, PRED2)              \
                                                                 \
      "fdup    "#TMP3".s,  #1.0                              \n" \
      "fadd    "#TMP2".s,  "#PRED1"/m, "#TMP2".s,  "#TMP3".s \n" \
      "fdiv    "#TMP3".s,  "#PRED1"/m, "#TMP3".s,  "#TMP2".s \n" \
      "fmul    "#DST".s,   "#PRED1"/m, "#SRC".s,   "#TMP3".s \n"

#define ASM_BLOCK_TANH(SRC,  DST,  CONST1, CONST2, CONST3,      \
                       TMP1, TMP2, TMP3, PRED1, PRED2)          \
      "fdup    "#TMP1".s,  #-10.0 \n" /* min */                 \
      "fdup    "#TMP2".s,  #10.0  \n" /* max */                 \
      "fmax    "#SRC".s,  "#PRED1"/m, "#SRC".s, "#TMP1".s \n"   \
      "fmin    "#SRC".s,  "#PRED1"/m, "#SRC".s, "#TMP2".s \n"   \
                                                                \
      ASM_BLOCK_EXP(SRC,  DST,  CONST1, CONST2, CONST3,         \
                    TMP1, TMP2, TMP3, PRED1, PRED2)             \
                                                                \
      "fmul    "#DST".s,   "#PRED1"/m, "#DST".s,  "#DST".s  \n" \
      "mov     "#TMP1".s,  "#PRED1"/m, "#DST".s             \n" \
      "fsub    "#DST".s,   "#PRED1"/m, "#DST".s,  #1.0      \n" \
      "fadd    "#TMP1".s,  "#PRED1"/m, "#TMP1".s, #1.0      \n" \
      "fdiv    "#DST".s,   "#PRED1"/m, "#DST".s,  "#TMP1".s \n"

#define ASM_BLOCK_ERF(SRC, DST,                                 \
                      CONST1, CONST2, CONST3, CONST4, CONST5,   \
                      TMP1, TMP2, TMP3, TMP4, TMP5, TMP6,       \
                      PRED1, PRED2)                             \
      "fabs    "#TMP1".s,  "#PRED1"/m, "#SRC".s                 \n" /* absx, tmp1 */     \
      "mov     "#TMP3".s,  "#PRED1"/m, "#TMP1".s                \n"                      \
      "fmul    "#TMP3".s,  "#PRED1"/m, "#TMP3".s, "#TMP3".s     \n" /* absx2 */          \
      "fneg    "#TMP3".s,  "#PRED1"/m, "#TMP3".s                \n" /* -absx2 */         \
                                                                                         \
      ASM_BLOCK_EXP(TMP3, TMP2, CONST1, CONST2, CONST3,                                  \
                    TMP4, TMP5, TMP6, PRED1, PRED2)                                      \
                                                                                         \
      "fneg    "#TMP2".s,  "#PRED1"/m, "#TMP2".s                \n" /* -exp(-x*x) */     \
                                                                                         \
      "fcmlt   "#PRED2".s, "#PRED1"/z, "#SRC".s, #0.0           \n"                      \
                                                                                         \
      "fdup    "#TMP5".s,  #1.0                                 \n"                      \
      "fmla    "#TMP5".s,  "#TMP1".s,  "#CONST4".s[0]           \n" /* (p*x+1), tmp2 */  \
                                                                                         \
      "fdup    "#TMP3".s,  #1.0                                 \n"                      \
      "fdiv    "#TMP3".s,  "#PRED1"/m, "#TMP3".s, "#TMP5".s     \n" /* t=1/tmp2, tmp3 */ \
      "fmul    "#TMP2".s,  "#PRED1"/m, "#TMP2".s, "#TMP3".s     \n" /* -exp(-x*x)*t */   \
                                                                                         \
      "dup     "#TMP5".s, "#CONST5".s[1]                        \n" /* p5 */             \
      "dup     "#TMP4".s, "#CONST5".s[0]                        \n" /* p4 */             \
      "fmad    "#TMP5".s, "#PRED1"/m, "#TMP3".s, "#TMP4".s      \n" /* p4+p5*r */        \
      "dup     "#TMP4".s, "#CONST4".s[3]                        \n" /* p3 */             \
      "fmad    "#TMP5".s, "#PRED1"/m, "#TMP3".s, "#TMP4".s      \n" /* p3+p4*r+p5*r^2 */ \
      "dup     "#TMP4".s, "#CONST4".s[2]                        \n" /* p2 */             \
      "fmad    "#TMP5".s, "#PRED1"/m, "#TMP3".s, "#TMP4".s      \n" /* p2+p3*r+p4*r^2+p5*r^3 */ \
      "dup     "#TMP4".s, "#CONST4".s[1]                        \n" /* p1 */             \
      "fmad    "#TMP5".s, "#PRED1"/m, "#TMP3".s, "#TMP4".s      \n" /* p1+p2*r+p3*r^2+p4*r^3+p5*r^4 */ \
                                                                                         \
      "fdup    "#TMP3".s, #1.0                                  \n"                      \
      "fmad    "#TMP2".s, "#PRED1"/m, "#TMP5".s, "#TMP3".s      \n" /* result */         \
      "mov     "#TMP5".s, "#PRED1"/m, "#TMP2".s                 \n"                      \
      "fneg    "#TMP5".s, "#PRED1"/m, "#TMP5".s                 \n" /* inverse */        \
      "sel     "#DST".s,  "#PRED2",   "#TMP5".s, "#TMP2".s      \n"

#define ASM_BLOCK_GELU_ERF_MICRO(SRC, DST,                                 \
                                 CONST1, CONST2, CONST3, CONST4, CONST5,   \
                                 TMP1, TMP2, TMP3, TMP4, TMP5, TMP6, TMP7, \
                                 PRED1, PRED2)                             \
      "mov     "#TMP1".s,  "#PRED1"/m, "#SRC".s                 \n"                            \
      "dup     "#TMP2".s,  %w[inv_sqrt]                         \n"                            \
      "fmul    "#TMP1".s,  "#PRED1"/m, "#TMP1".s, "#TMP2".s     \n" /* x * 0.707 */            \
      "fmul    "#SRC".s,  "#PRED1"/m, "#SRC".s, #0.5            \n" /* x * 0.5 */              \
                                                                                               \
      /* erf(x * 0.707) */                                                                     \
      ASM_BLOCK_ERF(TMP1, TMP1,                                                                \
                    CONST1, CONST2, CONST3, CONST4, CONST5,                                    \
                    TMP2, TMP3, TMP4, TMP5, TMP6, TMP7,                                        \
                    PRED1, PRED2)                                                              \
                                                                                               \
      "fadd    "#TMP1".s, "#PRED1"/m, "#TMP1".s, #1.0                 \n" /* 1.0+erf()  */     \
      "fmul    "#DST".s,  "#PRED1"/m, "#DST".s,  "#TMP1".s            \n" /* x*0.5*(1.0+erf()) */

#define ASM_BLOCK_ERF_PART_1(SRC,  DST1, DST2,              \
                             CONST1, CONST2, CONST3,        \
                             TMP1, TMP2, TMP3, TMP4,        \
                             PRED1, PRED2)                  \
      "fabs    "#DST1".s,  "#PRED1"/m, "#SRC".s             \n" /* absx, tmp1 */  \
      "mov     "#TMP1".s,  "#PRED1"/m, "#DST1".s            \n"                   \
      "fmul    "#TMP1".s,  "#PRED1"/m, "#TMP1".s, "#TMP1".s \n" /* absx2 */       \
      "fneg    "#TMP1".s,  "#PRED1"/m, "#TMP1".s            \n" /* -absx2 */      \
                                                                                  \
      ASM_BLOCK_EXP(TMP1, DST2, CONST1, CONST2, CONST3,                           \
                    TMP2, TMP3, TMP4,   PRED1,  PRED2)                            \
                                                                                  \
      "fneg    "#DST2".s,  "#PRED1"/m, "#DST2".s            \n" /* -exp(-x*x) */  \
                                                                                  \
      "fcmlt   "#PRED2".s,  "#PRED1"/z, "#SRC".s, #0.0      \n"


#define ASM_BLOCK_ERF_PART_2(SRC1, SRC2, DST,  CONST1, CONST2,       \
                             TMP1, TMP2, TMP3, PRED1, PRED2)         \
      "fdup    "#TMP1".s,  #1.0                             \n"                      \
      "fmla    "#TMP1".s,  "#SRC1".s,  "#CONST1".s[0]       \n" /* (p*x+1), tmp2 */  \
                                                                                     \
      "fdup    "#TMP2".s,  #1.0                             \n"                      \
      "fdiv    "#TMP2".s,  "#PRED1"/m, "#TMP2".s, "#TMP1".s \n" /* t=1/tmp2, tmp3 */ \
      "fmul    "#SRC2".s,  "#PRED1"/m, "#SRC2".s, "#TMP2".s \n" /* -exp(-x*x)*t   */ \
                                                                                     \
      "dup     "#TMP1".s,  "#CONST2".s[1]                   \n" /* p5 */             \
      "dup     "#TMP3".s,  "#CONST2".s[0]                   \n" /* p4 */             \
      "fmad    "#TMP1".s,  "#PRED1"/m, "#TMP2".s, "#TMP3".s \n" /* p4+p5*r */        \
      "dup     "#TMP3".s,  "#CONST1".s[3]                   \n" /* p3 */             \
      "fmad    "#TMP1".s,  "#PRED1"/m, "#TMP2".s, "#TMP3".s \n" /* p3+p4*r+p5*r^2 */ \
      "dup     "#TMP3".s,  "#CONST1".s[2]                   \n" /* p2 */             \
      "fmad    "#TMP1".s,  "#PRED1"/m, "#TMP2".s, "#TMP3".s \n" /* p2+p3*r+p4*r^2+p5*r^3 */ \
      "dup     "#TMP3".s,  "#CONST1".s[1]                   \n" /* p1 */             \
      "fmad    "#TMP1".s,  "#PRED1"/m, "#TMP2".s, "#TMP3".s \n" /* p1+p2*r+p3*r^2+p4*r^3+p5*r^4 */ \
                                                                                     \
      "fdup    "#TMP2".s,  #1.0                             \n"                      \
      "fmad    "#SRC2".s,  "#PRED1"/m, "#TMP1".s, "#TMP2".s \n" /* result */         \
      "mov     "#TMP1".s,  "#PRED1"/m, "#SRC2".s            \n"                      \
      "fneg    "#TMP1".s,  "#PRED1"/m, "#TMP1".s            \n" /* inverse */        \
      "sel     "#DST".s,   "#PRED2",   "#TMP1".s, "#SRC2".s \n"

#define ASM_BLOCK_GELU_TANH_MICRO(SRC, DST,                      \
                                  CONST1, CONST2, CONST3,        \
                                  TMP1, TMP2, TMP3, TMP4, TMP5,  \
                                  PRED1, PRED2)                  \
      "mov     "#TMP1".s,  "#PRED1"/m, "#SRC".s              \n"                  \
      "fmul    "#TMP1".s,  "#PRED1"/m, "#TMP1".s, "#TMP1".s  \n" /* x * x */      \
      "fmul    "#TMP1".s,  "#PRED1"/m, "#TMP1".s, "#SRC".s   \n" /* x * x * x */  \
                                                                                  \
      "dup     "#TMP2".s,  %w[const1]                        \n"                  \
      "fmad    "#TMP1".s,  "#PRED1"/m, "#TMP2".s, "#SRC".s   \n" /* (x + 0.044 * x^3) */ \
      "dup     "#TMP2".s,  %w[const2]                        \n"                  \
      "fmul    "#TMP1".s,  "#PRED1"/m, "#TMP1".s, "#TMP2".s  \n" /* 0.797 * (x + 0.044 * x^3) */ \
                                                                                  \
      /* tanh(0.797 * (x + 0.044 * x^3)) */                                       \
      ASM_BLOCK_TANH(TMP1, TMP2, CONST1, CONST2, CONST3,                          \
                     TMP3, TMP4, TMP5, PRED1, PRED2)                              \
                                                                                  \
      "fadd    "#TMP2".s,  "#PRED1"/m, "#TMP2".s, #1.0       \n" /* 1.0+tanh() */ \
      "fmul    "#SRC".s,   "#PRED1"/m, "#SRC".s, #0.5        \n" /* x*0.5 */      \
      "fmul    "#DST".s,   "#PRED1"/m, "#SRC".s, "#TMP2".s   \n" /* x*0.5*(1.0+tanh()) */
// clang-format on

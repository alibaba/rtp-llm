/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
  \brief Defines iterators used by warp-level matrix multiply operations targeting Tensor Cores.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/array.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/arch/arch.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor.h"

#include "cutlass/functional.h"
#include "cutlass/platform/platform.h"

// #include "cutlass_extensions/weight_only_quant_op.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/weight_only_quant_op.h"

#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

////////////////////////////////////////////////////////////////////////////////

namespace cutlass
{
namespace gemm
{
namespace warp
{

////////////////////////////////////////////////////////////////////////////////

template <
    /// Matrix multiply operator
    typename MmaOperator_,
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Operand identity
    Operand Operand,
    /// Data type of Scale elements
    typename Element_,
    /// Layout of operand
    typename Layout_,
    /// Number of threads participating in one matrix operation
    int Threads,
    ///
    WeightOnlyQuantOp QuantOp_,
    ///
    typename Enable = void>
class MmaTensorOpDequantizer;

////////////////////////////////////////////////////////////////////////////////
// Bfloat specialization for Ampere
template <
    /// Underlying matrix multiply operator (concept: MmaTensorOp)
    typename MmaOperator_,
    /// Shape of the warp level matrix multiply (concept: GemmShape)
    typename Shape_,
    ///
    WeightOnlyQuantOp QuantOp_>
class MmaTensorOpDequantizer<MmaOperator_, Shape_, Operand::kB, bfloat16_t, layout::RowMajor, 32, QuantOp_,
    typename platform::enable_if<MmaOperator_::ArchTag::kMinComputeCapability >= 80
        && platform::is_same<typename MmaOperator_::ArchMmaOperator::LayoutB, layout::ColumnMajor>::value>::type>
{

public:
    /// Mma Operator
    using MmaOperator = MmaOperator_;

    // The architecture specific mma ooperator being used
    using ArchMmaOperator = typename MmaOperator::ArchMmaOperator;

    // Mma Instruction Shape
    using InstructionShape = typename ArchMmaOperator::Shape;

    // This is the ratio of the load instruction vs the compute instruction.
    static constexpr int kExpansionFactor = MmaOperator::IteratorB::InstructionShape::kRow / InstructionShape::kK;

    /// Type of the scales
    using ElementScale = bfloat16_t;

    /// Fragment to hold B data before Mma
    using FragmentDequantizedOperand = Array<ElementScale, MmaOperator::FragmentB::kElements>;

    // Fragment to hold scale data to apply to B before mma
    // We need 1 fp16 per matrix iteration in the N dimension
    static constexpr int kColsPerMmaPerThread = 1;
    using FragmentScale = Array<ElementScale, kColsPerMmaPerThread * MmaOperator::MmaIterations::kColumn>;
    using FragmentZero = Array<ElementScale, kColsPerMmaPerThread * MmaOperator::MmaIterations::kColumn>;

    /// Warp mma shape
    using Shape = Shape_;

    /// Layout of the scales in shared memory
    using Layout = layout::RowMajor;

    /// TensorRef type for loading element from a tensor
    using TensorRef = TensorRef<ElementScale, Layout>;

    static constexpr WeightOnlyQuantOp QuantOp = QuantOp_;

    CUTLASS_DEVICE
    MmaTensorOpDequantizer(TensorRef smem_scales, TensorRef smem_zeros, const int warp_idx_n, const int lane_idx)
    {
        const int warp_offset = warp_idx_n * Shape::kN;
        const int quad = lane_idx / 4;
        const int thread_offset = warp_offset + quad;
        pointer_scale_ = smem_scales.data() + thread_offset;
        if constexpr (hasZero(QuantOp))
        {
            pointer_zero_ = smem_zeros.data() + thread_offset;
        }
    }

    CUTLASS_DEVICE
    MmaTensorOpDequantizer(TensorRef smem_scales, const int warp_idx_n, const int lane_idx)
        : MmaTensorOpDequantizer(smem_scales, TensorRef(), warp_idx_n, lane_idx)
    {
    }

    CUTLASS_DEVICE
    void load(FragmentScale& scale_frag)
    {
        CUTLASS_PRAGMA_UNROLL
        for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter)
        {
            scale_frag[mma_n_iter] = pointer_scale_[mma_n_iter * InstructionShape::kN];
        }
    }

    CUTLASS_DEVICE
    void dequantize(FragmentDequantizedOperand& operand_frag, const FragmentScale& scale_frag)
    {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && defined(ENABLE_BF16))
        using _MmaOperandB = typename ArchMmaOperator::FragmentB;
        using ExpandedMmaOperandB = Array<typename _MmaOperandB::Element, kExpansionFactor * _MmaOperandB::kElements>;
        static_assert(ExpandedMmaOperandB::kElements * MmaOperator::MmaIterations::kColumn
                == FragmentDequantizedOperand::kElements,
            "");

        const __nv_bfloat16* scale_ptr = reinterpret_cast<const __nv_bfloat16*>(&scale_frag);
        ExpandedMmaOperandB* operand_frag_ptr = reinterpret_cast<ExpandedMmaOperandB*>(&operand_frag);
        CUTLASS_PRAGMA_UNROLL
        for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter)
        {
            static_assert(ExpandedMmaOperandB::kElements % 2 == 0, "");

            __nv_bfloat162 scalex2 = __bfloat162bfloat162(scale_ptr[mma_n_iter]);
            __nv_bfloat162* operand_bf16x2_ptr = reinterpret_cast<__nv_bfloat162*>(&operand_frag_ptr[mma_n_iter]);

            CUTLASS_PRAGMA_UNROLL
            for (int ii = 0; ii < ExpandedMmaOperandB::kElements / 2; ++ii)
            {
                operand_bf16x2_ptr[ii] = __hmul2(operand_bf16x2_ptr[ii], scalex2);
            }
        }
#else
        // Slow path not implemented here on purpose. If we need to do HMMA on older arch, scale conversion should
        // happen before scales are stored to shared memory and we should use the fp16 dequantizer. This will avoid
        // numerous conversion instructions in GEMM main loop.
        arch::device_breakpoint();
#endif
    }

    CUTLASS_DEVICE
    void load(FragmentScale& scale_frag, FragmentScale& zero_frag)
    {
        if constexpr (hasZero(QuantOp))
        {
            CUTLASS_PRAGMA_UNROLL
            for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter)
            {
                scale_frag[mma_n_iter] = pointer_scale_[mma_n_iter * InstructionShape::kN];
                zero_frag[mma_n_iter] = pointer_zero_[mma_n_iter * InstructionShape::kN];
            }
        }
        else
        {
            CUTLASS_PRAGMA_UNROLL
            for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter)
            {
                scale_frag[mma_n_iter] = pointer_scale_[mma_n_iter * InstructionShape::kN];
            }
        }
    }

    CUTLASS_DEVICE
    void dequantize(
        FragmentDequantizedOperand& operand_frag, const FragmentScale& scale_frag, const FragmentScale& zero_frag)
    {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && defined(ENABLE_BF16))
        using _MmaOperandB = typename ArchMmaOperator::FragmentB;
        using ExpandedMmaOperandB = Array<typename _MmaOperandB::Element, kExpansionFactor * _MmaOperandB::kElements>;
        static_assert(ExpandedMmaOperandB::kElements * MmaOperator::MmaIterations::kColumn
                == FragmentDequantizedOperand::kElements,
            "");

        const __nv_bfloat16* scale_ptr = reinterpret_cast<const __nv_bfloat16*>(&scale_frag);
        const __nv_bfloat16* zero_ptr = reinterpret_cast<const __nv_bfloat16*>(&zero_frag);

        ExpandedMmaOperandB* operand_frag_ptr = reinterpret_cast<ExpandedMmaOperandB*>(&operand_frag);
        CUTLASS_PRAGMA_UNROLL
        for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter)
        {
            static_assert(ExpandedMmaOperandB::kElements % 2 == 0, "");

            __nv_bfloat162 scalex2 = __bfloat162bfloat162(scale_ptr[mma_n_iter]);
            __nv_bfloat162 zerox2 = __bfloat162bfloat162(zero_ptr[mma_n_iter]);
            __nv_bfloat162* operand_bf16x2_ptr = reinterpret_cast<__nv_bfloat162*>(&operand_frag_ptr[mma_n_iter]);

            if constexpr (hasZero(QuantOp))
            {
                CUTLASS_PRAGMA_UNROLL
                for (int ii = 0; ii < ExpandedMmaOperandB::kElements / 2; ++ii)
                {
                    operand_bf16x2_ptr[ii] = __hfma2(operand_bf16x2_ptr[ii], scalex2, zerox2);
                }
            }
            else
            {
                CUTLASS_PRAGMA_UNROLL
                for (int ii = 0; ii < ExpandedMmaOperandB::kElements / 2; ++ii)
                {
                    operand_bf16x2_ptr[ii] = __hmul2(operand_bf16x2_ptr[ii], scalex2);
                }
            }
        }
#else
        // Slow path not implemented here on purpose. If we need to do HMMA on older arch, scale conversion should
        // happen before scales are stored to shared memory and we should use the fp16 dequantizer. This will avoid
        // numerous conversion instructions in GEMM main loop.
        arch::device_breakpoint();
#endif
    }

    // Adds a pointer offset in units of elements.
    CUTLASS_DEVICE
    void add_pointer_offset(int64_t const& offset)
    {
        static_assert(sizeof(ElementScale) > 1, "");
        pointer_scale_ += offset;
        pointer_zero_ += offset;
    }

private:
    ElementScale const* pointer_scale_;
    ElementScale const* pointer_zero_;
};

////////////////////////////////////////////////////////////////////////////////

// Specialization for Turing & Ampere
template <
    /// Underlying matrix multiply operator (concept: MmaTensorOp)
    typename MmaOperator_,
    /// Shape of the warp level matrix multiply (concept: GemmShape)
    typename Shape_,
    ///
    WeightOnlyQuantOp QuantOp_>
class MmaTensorOpDequantizer<MmaOperator_, Shape_, Operand::kB, half_t, layout::RowMajor, 32, QuantOp_,
    typename platform::enable_if<MmaOperator_::ArchTag::kMinComputeCapability >= 75
        && platform::is_same<typename MmaOperator_::ArchMmaOperator::LayoutB, layout::ColumnMajor>::value>::type>
{

public:
    /// Mma Operator
    using MmaOperator = MmaOperator_;

    // The architecture specific mma ooperator being used
    using ArchMmaOperator = typename MmaOperator::ArchMmaOperator;

    // Mma Instruction Shape
    using InstructionShape = typename ArchMmaOperator::Shape;

    // This is the ratio of the load instruction vs the compute instruction.
    static constexpr int kExpansionFactor = MmaOperator::IteratorB::InstructionShape::kRow / InstructionShape::kK;

    /// Type of the scales
    using ElementScale = half_t;

    /// Fragment to hold B data before Mma
    using FragmentDequantizedOperand = Array<ElementScale, MmaOperator::FragmentB::kElements>;

    // Fragment to hold scale data to apply to B before mma
    // We need 1 fp16 per matrix iteration in the N dimension
    static constexpr int kColsPerMmaPerThread = 1;
    using FragmentScale = Array<ElementScale, kColsPerMmaPerThread * MmaOperator::MmaIterations::kColumn>;
    using FragmentZero = Array<ElementScale, kColsPerMmaPerThread * MmaOperator::MmaIterations::kColumn>;

    /// Warp mma shape
    using Shape = Shape_;

    /// Layout of the scales in shared memory
    using Layout = layout::RowMajor;

    /// TensorRef type for loading element from a tensor
    using TensorRef = TensorRef<ElementScale, Layout>;

    static constexpr WeightOnlyQuantOp QuantOp = QuantOp_;

    CUTLASS_DEVICE
    MmaTensorOpDequantizer(TensorRef smem_scales, TensorRef smem_zeros, const int warp_idx_n, const int lane_idx)
    {
        const int warp_offset = warp_idx_n * Shape::kN;
        const int quad = lane_idx / 4;
        const int thread_offset = warp_offset + quad;
        pointer_scale_ = smem_scales.data() + thread_offset;
        if constexpr (hasZero(QuantOp))
        {
            pointer_zero_ = smem_zeros.data() + thread_offset;
        }
    }

    CUTLASS_DEVICE
    MmaTensorOpDequantizer(TensorRef smem_scales, const int warp_idx_n, const int lane_idx)
        : MmaTensorOpDequantizer(smem_scales, TensorRef(), warp_idx_n, lane_idx)
    {
    }

    CUTLASS_DEVICE
    void load(FragmentScale& scale_frag)
    {
        CUTLASS_PRAGMA_UNROLL
        for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter)
        {
            scale_frag[mma_n_iter] = pointer_scale_[mma_n_iter * InstructionShape::kN];
        }
    }

    CUTLASS_DEVICE
    void dequantize(FragmentDequantizedOperand& operand_frag, const FragmentScale& scale_frag)
    {
        using _MmaOperandB = typename ArchMmaOperator::FragmentB;
        using ExpandedMmaOperandB = Array<typename _MmaOperandB::Element, kExpansionFactor * _MmaOperandB::kElements>;
        static_assert(ExpandedMmaOperandB::kElements * MmaOperator::MmaIterations::kColumn
                == FragmentDequantizedOperand::kElements,
            "");

        multiplies<ExpandedMmaOperandB> mul_op;

        ExpandedMmaOperandB* operand_frag_ptr = reinterpret_cast<ExpandedMmaOperandB*>(&operand_frag);
        CUTLASS_PRAGMA_UNROLL
        for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter)
        {
            operand_frag_ptr[mma_n_iter] = mul_op(operand_frag_ptr[mma_n_iter], scale_frag[mma_n_iter]);
        }
    }

    CUTLASS_DEVICE
    void load(FragmentScale& scale_frag, FragmentScale& zero_frag)
    {
        if constexpr (hasZero(QuantOp))
        {
            CUTLASS_PRAGMA_UNROLL
            for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter)
            {
                scale_frag[mma_n_iter] = pointer_scale_[mma_n_iter * InstructionShape::kN];
                zero_frag[mma_n_iter] = pointer_zero_[mma_n_iter * InstructionShape::kN];
            }
        }
        else
        {
            CUTLASS_PRAGMA_UNROLL
            for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter)
            {
                scale_frag[mma_n_iter] = pointer_scale_[mma_n_iter * InstructionShape::kN];
            }
        }
    }

    CUTLASS_DEVICE
    void dequantize(
        FragmentDequantizedOperand& operand_frag, const FragmentScale& scale_frag, const FragmentScale& zero_frag)
    {
        using _MmaOperandB = typename ArchMmaOperator::FragmentB;
        using ExpandedMmaOperandB = Array<typename _MmaOperandB::Element, kExpansionFactor * _MmaOperandB::kElements>;
        static_assert(ExpandedMmaOperandB::kElements * MmaOperator::MmaIterations::kColumn
                == FragmentDequantizedOperand::kElements,
            "");

        multiplies<ExpandedMmaOperandB> mul_op;
        ExpandedMmaOperandB* operand_frag_ptr = reinterpret_cast<ExpandedMmaOperandB*>(&operand_frag);

        if constexpr (hasZero(QuantOp))
        {
            plus<ExpandedMmaOperandB> plus_op;

            CUTLASS_PRAGMA_UNROLL
            for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter)
            {
                operand_frag_ptr[mma_n_iter]
                    = plus_op(mul_op(operand_frag_ptr[mma_n_iter], scale_frag[mma_n_iter]), zero_frag[mma_n_iter]);
            }
        }
        else
        {
            CUTLASS_PRAGMA_UNROLL
            for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter)
            {
                operand_frag_ptr[mma_n_iter] = mul_op(operand_frag_ptr[mma_n_iter], scale_frag[mma_n_iter]);
            }
        }
    }

    // Adds a pointer offset in units of elements.
    CUTLASS_DEVICE
    void add_pointer_offset(int64_t const& offset)
    {
        static_assert(sizeof(ElementScale) > 1, "");
        pointer_scale_ += offset;
        pointer_zero_ += offset;
    }

private:
    ElementScale const* pointer_scale_;
    ElementScale const* pointer_zero_;
};

////////////////////////////////////////////////////////////////////////////////

// Specialization for Volta A x RowMajor B tensorOp, for 32x32x4 interleaved gemm
template <
    /// Underlying matrix multiply operator (concept: MmaTensorOp)
    typename MmaOperator_,
    /// Shape of the warp level matrix multiply (concept: GemmShape)
    typename Shape_,
    ///
    WeightOnlyQuantOp QuantOp_>
class MmaTensorOpDequantizer<MmaOperator_, Shape_, Operand::kB, half_t, layout::RowMajor, 32, QuantOp_,
    typename platform::enable_if<platform::is_same<typename MmaOperator_::ArchTag, arch::Sm70>::value
        && platform::is_same<typename MmaOperator_::ArchMmaOperator::LayoutB, layout::RowMajor>::value>::type>
{

public:
    static_assert(platform::is_same<typename MmaOperator_::InterleavedTileShape, GemmShape<32, 32, 4>>::value, "");

    /// Mma Operator
    using MmaOperator = MmaOperator_;

    // The architecture specific mma ooperator being used
    using ArchMmaOperator = typename MmaOperator::ArchMmaOperator;

    // Mma Instruction Shape
    using InstructionShape = typename ArchMmaOperator::Shape;

    /// Type of the scales
    using ElementScale = half_t;

    /// Fragment to hold B data before Mma
    using FragmentDequantizedOperand = Array<ElementScale, MmaOperator::FragmentB::kElements>;

    /// Warp mma shape
    using Shape = Shape_;

    // Fragment to hold scale data to apply to B before mma
    // Each 32x32x4 matmul uses 8 elements from B.
    static constexpr int ColsPerMmaTile = 32;
    static constexpr int TileNIterations = Shape::kN / ColsPerMmaTile;
    using FragmentScale = Array<ElementScale, TileNIterations * 8>;
    using FragmentZero = Array<ElementScale, TileNIterations * 8>;
    using AccessType = Array<ElementScale, 8>;

    /// Layout of the scales in shared memory
    using Layout = layout::RowMajor;

    /// TensorRef type for loading element from a tensor
    using TensorRef = TensorRef<ElementScale, Layout>;

    static constexpr WeightOnlyQuantOp QuantOp = QuantOp_;
    //static_assert(QuantOp == WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, "");

    CUTLASS_DEVICE
    MmaTensorOpDequantizer(TensorRef smem_scales, TensorRef smem_zeros, const int warp_idx_n, const int lane_idx)
    {
        const int warp_offset = warp_idx_n * Shape::kN;
        const int base_col = lane_idx & 0xF8;
        const int thread_offset = warp_offset + base_col;
        pointer_scale_ = smem_scales.data() + thread_offset;
        if constexpr (hasZero(QuantOp))
        {
            pointer_zero_ = smem_zeros.data() + thread_offset;
        }

    }

    CUTLASS_DEVICE
    MmaTensorOpDequantizer(TensorRef smem_scales, const int warp_idx_n, const int lane_idx)
        : MmaTensorOpDequantizer(smem_scales, TensorRef(), warp_idx_n, lane_idx) {}

    CUTLASS_DEVICE
    void load(FragmentScale& scale_frag)
    {
        AccessType* scale_frag_ptr = reinterpret_cast<AccessType*>(&scale_frag);

        CUTLASS_PRAGMA_UNROLL
        for (int tile_iter = 0; tile_iter < TileNIterations; ++tile_iter)
        {
            // We jump by 32 here since volta does <32x32x4> super mmas inside a warp.
            scale_frag_ptr[tile_iter] = *reinterpret_cast<AccessType const*>(pointer_scale_ + ColsPerMmaTile * tile_iter);
        }
    }

    CUTLASS_DEVICE
    void dequantize(FragmentDequantizedOperand& operand_frag, const FragmentScale& scale_frag)
    {
        static_assert(FragmentScale::kElements == FragmentDequantizedOperand::kElements, "");

        multiplies<FragmentDequantizedOperand> mul_op;
        operand_frag = mul_op(operand_frag, scale_frag);
    }

    CUTLASS_DEVICE
    void load(FragmentScale& scale_frag, FragmentScale& zero_frag)
    {
        load(scale_frag);
        if constexpr (hasZero(QuantOp))
        {
            AccessType* zero_frag_ptr = reinterpret_cast<AccessType*>(&zero_frag);

            CUTLASS_PRAGMA_UNROLL
            for (int tile_iter = 0; tile_iter < TileNIterations; ++tile_iter)
            {
                // We jump by 32 here since volta does <32x32x4> super mmas inside a warp.
                zero_frag_ptr[tile_iter] = *reinterpret_cast<AccessType const*>(pointer_zero_ + ColsPerMmaTile * tile_iter);
            }
        }
    }

    CUTLASS_DEVICE
    void dequantize(
        FragmentDequantizedOperand& operand_frag, const FragmentScale& scale_frag, const FragmentScale& zero_frag)
    {
        if constexpr (hasZero(QuantOp))
        {
            static_assert(FragmentScale::kElements == FragmentDequantizedOperand::kElements, "");

            multiplies<FragmentDequantizedOperand> mul_op;
            plus<FragmentDequantizedOperand> plus_op;

            operand_frag = plus_op(mul_op(operand_frag, scale_frag), zero_frag);
        }
        else
        {
            dequantize(operand_frag, scale_frag);
        }
    }

    // Adds a pointer offset in units of elements.
    CUTLASS_DEVICE
    void add_pointer_offset(int64_t const& offset)
    {
        static_assert(sizeof(ElementScale) > 1, "");
        pointer_scale_ += offset;
        pointer_zero_ += offset;
    }

private:
    ElementScale const* pointer_scale_;
    ElementScale const* pointer_zero_;
};

} // namespace warp
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////

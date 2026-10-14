# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Legacy vector wrappers used by RTP-LLM's MI308X FlyDSL GDN kernels.

Adapted from AITER's former ``aiter.ops.flydsl.kernels.vector`` module. AITER
removed that module after migrating its own kernels to ``fx.Vector``; RTP's
megakernels still use these dialect wrappers. Keep the compatibility surface
local until those kernels are migrated as well.
"""

from flydsl._mlir import ir
from flydsl._mlir.dialects import vector as _vector
from flydsl._mlir.dialects.vector import *
from flydsl.expr.meta import dsl_loc_tracing
from flydsl.expr.typing import as_ir_value


def _as_index_ir_value(value):
    if isinstance(value, int):
        from flydsl.expr import arith as _arith_ext

        return _arith_ext.constant(value, index=True)
    result = as_ir_value(value)
    if isinstance(result.type, ir.IntegerType):
        from flydsl._mlir.dialects import arith as _arith

        result = _arith.IndexCastOp(ir.IndexType.get(), result).result
    return result


@dsl_loc_tracing
def from_elements(*args, **kwargs):
    if len(args) >= 2:
        args = list(args)
        if isinstance(args[1], (list, tuple)):
            args[1] = [as_ir_value(value) for value in args[1]]
    return _vector.from_elements(*args, **kwargs)


@dsl_loc_tracing
def store(value, memref, indices, **kwargs):
    return _vector.store(
        as_ir_value(value),
        as_ir_value(memref),
        [_as_index_ir_value(index) for index in indices],
        **kwargs,
    )


@dsl_loc_tracing
def extract(value, static_position=None, dynamic_position=None):
    static_position = [] if static_position is None else list(static_position)
    dynamic_position = [] if dynamic_position is None else dynamic_position
    dynamic_position = [_as_index_ir_value(index) for index in dynamic_position]
    if len(dynamic_position) > len(static_position):
        static_position.extend(
            [ir.ShapedType.get_dynamic_size()]
            * (len(dynamic_position) - len(static_position))
        )
    return _vector.ExtractOp(
        as_ir_value(value),
        static_position=static_position,
        dynamic_position=dynamic_position,
    ).result


@dsl_loc_tracing
def load_op(result_type, memref, indices):
    return _vector.LoadOp(
        result_type,
        as_ir_value(memref),
        [_as_index_ir_value(index) for index in indices],
    ).result


@dsl_loc_tracing
def bitcast(result_type, source):
    return _vector.BitCastOp(result_type, as_ir_value(source)).result

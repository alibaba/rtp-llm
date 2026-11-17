# SPDX-License-Identifier: Apache-2.0
"""Native NVVM barriers retain convergence in CUTLASS DSL 4.4.2."""

import cutlass
from cutlass._mlir.dialects import nvvm
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def named_barrier(*, barrier_id, number_of_threads, loc=None, ip=None):
    # The 4.4.2 inline-asm wrapper can be duplicated across divergent branches.
    nvvm.barrier(
        barrier_id=cutlass.Int32(barrier_id).ir_value(loc=loc, ip=ip),
        number_of_threads=cutlass.Int32(number_of_threads).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )

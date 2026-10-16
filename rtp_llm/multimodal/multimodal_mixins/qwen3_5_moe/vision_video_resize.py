# SPDX-License-Identifier: BSD-3-Clause
# Coefficient generation adapted from PyTorch v2.11.0:
# https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cpu/UpSampleKernel.cpp
# The upstream license is retained below.
# From PyTorch:
#
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
#
# From Caffe2:
#
# Copyright (c) 2016-present, Facebook Inc. All rights reserved.
#
# All contributions by Facebook:
# Copyright (c) 2016 Facebook Inc.
#
# All contributions by Google:
# Copyright (c) 2015 Google Inc.
# All rights reserved.
#
# All contributions by Yangqing Jia:
# Copyright (c) 2015 Yangqing Jia
# All rights reserved.
#
# All contributions by Kakao Brain:
# Copyright 2019-2020 Kakao Brain
#
# All contributions by Cruise LLC:
# Copyright (c) 2022 Cruise LLC.
# All rights reserved.
#
# All contributions by Tri Dao:
# Copyright (c) 2024 Tri Dao.
# All rights reserved.
#
# All contributions by Arm:
# Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates
#
# All contributions from Caffe:
# Copyright(c) 2013, 2014, 2015, the respective contributors
# All rights reserved.
#
# All other contributions:
# Copyright(c) 2015, 2016 the respective contributors
# All rights reserved.
#
# Caffe2 uses a copyright model similar to Caffe: each contributor holds
# copyright over their contributions to Caffe2. The project versioning records
# all such contribution and copyright details. If a contributor wants to further
# mark their specific copyright on a particular contribution, they should
# indicate their copyright solely in the commit message of the change when it is
# committed.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
#    and IDIAP Research Institute nor the names of its contributors may be
#    used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""GPU bicubic/antialias resize with CPU-compatible uint8 rounding.

Only dimension-derived coefficients are prepared on the host. Video pixels
stay on CUDA; horizontal and vertical passes each round and clamp to uint8,
as in the CPU processor. The bounded CPU cache never retains video data.
"""

import math
from functools import lru_cache

import torch
import triton
import triton.language as tl


def _cubic(x):
    x = abs(x)
    a = -0.5
    if x < 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    if x < 2.0:
        return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
    return 0.0


@lru_cache(maxsize=32)
def _coefficients(input_size, output_size):
    scale = input_size / output_size
    support = 2.0 * max(scale, 1.0)
    kmax = math.ceil(support) * 2 + 1
    invscale = 1.0 / scale if scale >= 1.0 else 1.0
    starts, lengths, weights = [], [], []
    wt_max = 0.0
    for i in range(output_size):
        center = scale * (i + 0.5)
        start = max(int(center - support + 0.5), 0)
        length = max(0, min(int(center + support + 0.5), input_size) - start)
        length = min(length, kmax)
        row = [_cubic((j + start - center + 0.5) * invscale) for j in range(length)]
        total = sum(row)
        if total != 0.0:
            row = [w / total for w in row]
            wt_max = max(wt_max, max(row, default=0.0))
        starts.append(start)
        lengths.append(length)
        weights.append(row + [0.0] * (kmax - length))
    precision = 0
    while precision < 22:
        if int(0.5 + wt_max * (1 << (precision + 1))) >= 1 << 15:
            break
        precision += 1
    integer = [
        [int(w * (1 << precision) + (-0.5 if w < 0.0 else 0.5)) for w in row]
        for row in weights
    ]
    return (
        torch.tensor(starts, dtype=torch.int64, device="cpu"),
        torch.tensor(lengths, dtype=torch.int64, device="cpu"),
        torch.tensor(integer, dtype=torch.int16, device="cpu"),
        precision,
    )


@triton.jit
def _resize_axis(
    SRC,
    DST,
    START,
    LENGTH,
    WEIGHTS,
    COUNT: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    VERTICAL: tl.constexpr,
    K: tl.constexpr,
    PRECISION: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = i < COUNT
    x = i % OW
    y = (i // OW) % OH
    plane = i // (OH * OW)
    pos = y if VERTICAL else x
    # Widen before row-stride multiplication, not after a possible overflow.
    start = tl.load(START + pos, valid, other=0).to(tl.int64)
    length = tl.load(LENGTH + pos, valid, other=0)
    acc = tl.full((BLOCK,), 1 << (PRECISION - 1), tl.int32)
    for j in range(K):
        mask = valid & (j < length)
        offset = plane * IH * IW
        if VERTICAL:
            offset += (start + j) * IW + x
        else:
            offset += y * IW + start + j
        value = tl.load(SRC + offset, mask, other=0).to(tl.int32)
        weight = tl.load(WEIGHTS + pos * K + j, mask, other=0).to(tl.int32)
        acc += value * weight
    result = tl.minimum(255, tl.maximum(0, acc >> PRECISION))
    tl.store(DST + i, result, valid)


def resize_bicubic_uint8(video, height, width):
    if (
        video.dtype != torch.uint8
        or video.ndim != 4
        or video.shape[1] != 3
        or not video.is_cuda
    ):
        raise ValueError("Expected CUDA uint8 NCHW RGB video")
    if min(height, width, *video.shape[-2:]) <= 0:
        raise ValueError("Resize dimensions must be positive")
    if tuple(video.shape[-2:]) == (height, width):
        return video
    result = video.contiguous()
    with torch.cuda.device(video.device):
        # CPU uint8 processing rounds and clamps after each separable pass.
        for vertical, size in ((False, width), (True, height)):
            old_size = result.shape[-2] if vertical else result.shape[-1]
            if old_size == size:
                continue
            starts, lengths, weights, precision = _coefficients(old_size, size)
            starts = starts.to(video.device)
            lengths = lengths.to(video.device)
            weights = weights.to(video.device)
            ih, iw = result.shape[-2:]
            oh, ow = (size, iw) if vertical else (ih, size)
            output = torch.empty(
                (*result.shape[:2], oh, ow), dtype=torch.uint8, device=video.device
            )
            _resize_axis[(triton.cdiv(output.numel(), 256),)](
                result,
                output,
                starts,
                lengths,
                weights,
                output.numel(),
                ih,
                iw,
                oh,
                ow,
                vertical,
                weights.shape[1],
                precision,
                BLOCK=256,
            )
            result = output
    return result

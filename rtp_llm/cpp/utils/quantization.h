/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "rtp_llm/cpp/utils/QuantInfo.h"
#include "stdlib.h"
#include <cstdint>
#include <string>
namespace tensorrt_llm {
namespace common {

class QuantAlgo{
public:
    QuantAlgo() = default;
    QuantAlgo(rtp_llm::QuantAlgo const& quant_algo)
        : weight_bits_(quant_algo.getWeightBits())
        , group_size_(quant_algo.getGroupSize())
        , weight_only_(quant_algo.isWeightOnlyPerCol() || quant_algo.isGptq() || quant_algo.isAwq())
        , sq_int8_(quant_algo.isSmoothQuant() || quant_algo.isOmniQuant())
        , fp8_(quant_algo.isFp8())
    {}
    QuantAlgo(int weight_bits, int64_t group_size, bool weight_only, bool sq_int8, bool fp8)
        : weight_bits_(weight_bits)
        , group_size_(group_size)
        , weight_only_(weight_only)
        , sq_int8_(sq_int8)
        , fp8_(fp8)
    {}

    int getWeightBits() const {
        return weight_bits_;
    }

    int getGroupSize() const {
        return group_size_;
    }

    bool weightOnly() const {
        return weight_only_;
    }

    bool smoothQuantInt8() const {
        return sq_int8_;
    }
    bool fp8() const {
        return fp8_;
    }

private:
    int weight_bits_ = 0;
    int group_size_ = 0;
    bool weight_only_ = false;
    bool sq_int8_    = false;
    bool fp8_ = false;
};

class QuantMode {
    // [WARNING] KEEP BELOW DEFINITION IN SYNC WITH tensorrt_llm/quantization/mode.py
public:
    using BaseType = std::uint32_t;

    explicit constexpr QuantMode(BaseType value) noexcept
        : mValue{value}
    {
    }

    QuantMode() noexcept = default;

    constexpr QuantMode(QuantMode const&) noexcept = default;

    constexpr QuantMode& operator=(const QuantMode& other) noexcept = default;

    static constexpr QuantMode none() noexcept
    {
        return QuantMode(BaseType(0));
    }

    static constexpr QuantMode int4Weights() noexcept
    {
        return QuantMode(BaseType(1u) << 0);
    }

    static constexpr QuantMode int8Weights() noexcept
    {
        return QuantMode(BaseType(1u) << 1);
    }

    static constexpr QuantMode activations() noexcept
    {
        return QuantMode(BaseType(1u) << 2);
    }

    static constexpr QuantMode perChannelScaling() noexcept
    {
        return QuantMode(BaseType(1u) << 3);
    }

    static constexpr QuantMode perTokenScaling() noexcept
    {
        return QuantMode(BaseType(1u) << 4);
    }

    static constexpr QuantMode perGroupScaling() noexcept
    {
        return QuantMode(BaseType(1u) << 5);
    }

    static constexpr QuantMode int8KvCache() noexcept
    {
        return QuantMode(BaseType(1u) << 6);
    }

    static constexpr QuantMode fp8KvCache() noexcept
    {
        return QuantMode(BaseType(1u) << 7);
    }

    static constexpr QuantMode fp8Qdq() noexcept
    {
        return QuantMode(BaseType(1u) << 8);
    }

    constexpr BaseType value() const noexcept
    {
        return mValue;
    }

    constexpr bool isSet(QuantMode const& mode) const noexcept
    {
        return (mValue & mode.value()) == mode.value();
    }

    constexpr bool hasInt4Weights() const noexcept
    {
        return isSet(int4Weights());
    }

    constexpr bool hasInt8Weights() const noexcept
    {
        return isSet(int8Weights());
    }

    constexpr bool hasActivations() const noexcept
    {
        return isSet(activations());
    }

    constexpr bool hasPerChannelScaling() const noexcept
    {
        return isSet(perChannelScaling());
    }

    constexpr bool hasPerTokenScaling() const noexcept
    {
        return isSet(perTokenScaling());
    }

    constexpr bool hasPerGroupScaling() const noexcept
    {
        return isSet(perGroupScaling());
    }

    constexpr bool hasStaticActivationScaling() const noexcept
    {
        return !hasPerTokenScaling();
    }

    constexpr bool hasInt8KvCache() const noexcept
    {
        return isSet(int8KvCache());
    }

    constexpr bool hasFp8KvCache() const noexcept
    {
        return isSet(fp8KvCache());
    }

    constexpr bool hasFp8Qdq() const noexcept
    {
        return isSet(fp8Qdq());
    }

    constexpr bool hasKvCacheQuant() const noexcept
    {
        return hasInt8KvCache() || hasFp8KvCache();
    }

    static constexpr QuantMode fromDescription(bool quantizeWeights = false, bool quantizeActivations = false,
        bool perToken = false, bool perChannel = false, bool useInt4Weights = false, bool useInt8KvCache = false,
        bool useFp8KvCache = false, bool useFp8Qdq = false)
    {
        QuantMode quantMode{};
        if (quantizeWeights)
        {
            if (useInt4Weights)
                quantMode += int4Weights();
            else
                quantMode += int8Weights();
        }

        if (quantizeActivations)
        {
            quantMode += activations();
        }

        if (perChannel)
        {
            quantMode += QuantMode::perChannelScaling();
        }
        if (perToken)
        {
            quantMode += QuantMode::perTokenScaling();
        }

        if (useInt8KvCache)
        {
            quantMode += int8KvCache();
        }

        if (useFp8KvCache)
        {
            quantMode += fp8KvCache();
        }

        if (useFp8Qdq)
        {
            quantMode += fp8Qdq();
        }

        return quantMode;
    }

    constexpr QuantMode operator+(const QuantMode& other) const noexcept
    {
        return QuantMode(mValue | other.mValue);
    }

    constexpr QuantMode& operator+=(const QuantMode& other) noexcept
    {
        return *this = *this + other;
    }

    constexpr QuantMode operator-(const QuantMode& other) const noexcept
    {
        return QuantMode(mValue & ~other.mValue);
    }

    constexpr QuantMode& operator-=(const QuantMode& other) noexcept
    {
        return *this = *this - other;
    }

    constexpr bool operator==(const QuantMode& other) const noexcept
    {
        return mValue == other.mValue;
    }

    constexpr bool operator!=(const QuantMode& other) const noexcept
    {
        return !(*this == other);
    }

private:
    BaseType mValue{0};
};

} // namespace common
}

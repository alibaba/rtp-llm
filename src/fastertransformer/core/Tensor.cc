/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "Tensor.h"

#include "src/fastertransformer/cuda/cuda_utils.h"
#include "src/fastertransformer/utils/string_utils.h"

#include "stdlib.h"
#include <dirent.h>
#include <numeric>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_map>
#include <vector>

namespace fastertransformer {

Tensor::Tensor():
    where_(MEMORY_CPU),
    type_(TYPE_INVALID),
    shape_({}),
    data_(nullptr)
    {}

Tensor::Tensor(const MemoryType where,
               const DataType type,
               const std::vector<size_t> shape,
               const void* data):
    where_(where),
    type_(type),
    shape_(shape),
    data_(const_cast<void*>(data))
    {}

Tensor::~Tensor() {
    if (isValid()) {
        type_ = TYPE_INVALID;
        shape_.clear();
        data_ = nullptr;
    }
}

bool Tensor::isValid() const {
    return (type_ != TYPE_INVALID) && (shape_.size() > 0) && (data_ != nullptr);
}

MemoryType Tensor::where() const {
    return where_;
}

DataType Tensor::type() const {
    return type_;
}

const std::vector<size_t>& Tensor::shape() const {
    return shape_;
}

void* Tensor::data() const {
    return data_;
}

void** Tensor::dataPtr() {
    return &data_;
}

// TODO(wangyin): move this implementation to DeviceOps.
template<typename T>
inline T Tensor::getVal(size_t index) const {
    FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    FT_CHECK(data_ != nullptr);
    FT_CHECK_WITH_INFO(index < size(), "index is larger than buffer size");

    if (getTensorType<T>() != type_) {
        FT_LOG_DEBUG("getVal with type %s, but data type is: %s",
                        getNumpyTypeDesc(getTensorType<T>()).c_str(),
                        getNumpyTypeDesc(type_).c_str());
    }
    if (where_ == MEMORY_CPU) {
        return ((T*)data())[index];
    } else {
        using ValueType = typename std::remove_const<T>::type;
        ValueType val;
        cudaMemcpy(&val, (ValueType*)data_ + index, sizeof(ValueType), cudaMemcpyDeviceToHost);
        return val;
    }
}

#define DECLARE_TEMPLATE_METHODS_WITH_TYPE(T) \
    template T Tensor::getVal<T>(size_t index) const; \
    template const T Tensor::getVal<const T>(size_t index) const; \

DECLARE_TEMPLATE_METHODS_WITH_TYPE(float)
DECLARE_TEMPLATE_METHODS_WITH_TYPE(half)
DECLARE_TEMPLATE_METHODS_WITH_TYPE(char)
DECLARE_TEMPLATE_METHODS_WITH_TYPE(int8_t)
DECLARE_TEMPLATE_METHODS_WITH_TYPE(int)
DECLARE_TEMPLATE_METHODS_WITH_TYPE(uint)
DECLARE_TEMPLATE_METHODS_WITH_TYPE(uint64_t)
DECLARE_TEMPLATE_METHODS_WITH_TYPE(long)
DECLARE_TEMPLATE_METHODS_WITH_TYPE(unsigned long long int)
DECLARE_TEMPLATE_METHODS_WITH_TYPE(bool)
// DECLARE_TEMPLATE_METHODS_WITH_TYPE(void)
#ifdef ENABLE_BF16
DECLARE_TEMPLATE_METHODS_WITH_TYPE(__nv_bfloat16)
#endif
#ifdef ENABLE_FP8
DECLARE_TEMPLATE_METHODS_WITH_TYPE(__nv_fp8_e4m3)
#endif

#undef DECLARE_TEMPLATE_METHODS_WITH_TYPE

size_t Tensor::size() const {
    if (data_ == nullptr || shape_.size() == 0) {
        return 0;
    }
    return std::accumulate(shape_.begin(), shape_.end(), (size_t)1, std::multiplies<size_t>());
}

size_t Tensor::sizeBytes() const {
    return size() * getTypeSize(type());
}

std::string Tensor::whereToString() const {
    static const std::unordered_map<MemoryType, std::string> mem_to_string{
        {MEMORY_CPU, "CPU"}, {MEMORY_CPU_PINNED, "CPU_PINNED"}, {MEMORY_GPU, "GPU"}};
    return mem_to_string.at(where());
}

std::string Tensor::toString() const {
    std::string memtype_str = whereToString();

    static const std::unordered_map<DataType, std::string> type_to_string{
        {TYPE_BOOL, "BOOL"},
        {TYPE_UINT8, "UINT8"},
        {TYPE_UINT16, "UINT16"},
        {TYPE_UINT32, "UINT32"},
        {TYPE_UINT64, "UINT64"},
        {TYPE_INT8, "INT8"},
        {TYPE_INT16, "INT16"},
        {TYPE_INT32, "INT32"},
        {TYPE_INT64, "INT64"},
        {TYPE_BF16, "BF16"},
        {TYPE_FP16, "FP16"},
        {TYPE_FP32, "FP32"},
        {TYPE_FP64, "FP64"},
        {TYPE_BYTES, "BYTES"},
        {TYPE_INVALID, "INVALID"},
        {TYPE_FP8_E4M3, "E4M3"},
        {TYPE_VOID, "VOID"},
    };
    return fmtstr("Tensor[where=%s, type=%s, shape=%s, data=%p]",
                  memtype_str.c_str(),
                  type_to_string.at(type()).c_str(),
                  vec2str(shape()).c_str(),
                  data());
}

std::string Tensor::getNumpyTypeDesc(DataType type) const {
    static const std::unordered_map<DataType, std::string> type_map{{TYPE_INVALID, "x"},
                                                                    {TYPE_BOOL, "?"},
                                                                    {TYPE_BYTES, "b"},
                                                                    {TYPE_UINT8, "u1"},
                                                                    {TYPE_UINT16, "u2"},
                                                                    {TYPE_UINT32, "u4"},
                                                                    {TYPE_UINT64, "u8"},
                                                                    {TYPE_INT8, "i1"},
                                                                    {TYPE_INT16, "i2"},
                                                                    {TYPE_INT32, "i4"},
                                                                    {TYPE_INT64, "i8"},
                                                                    {TYPE_FP16, "f2"},
                                                                    {TYPE_FP32, "f4"},
                                                                    {TYPE_FP64, "f8"}};

    if (type == TYPE_BF16) {
        FT_LOG_WARNING("getNumpyTypeDesc(TYPE_BF16) returns an invalid type 'x' since Numpy doesn't "
                       "support bfloat16 as of now, it will be properly extended if numpy supports. "
                       "Please refer for the discussions https://github.com/numpy/numpy/issues/19808.");
    }

    return type_map.count(type) > 0 ? type_map.at(type) : "x";
}


Tensor Tensor::slice(std::vector<size_t> shape, size_t offset) const {
    if (this->data() != nullptr) {
        size_t n_elts        = this->size();
        size_t n_sliced_elts = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        FT_CHECK_WITH_INFO(
            n_sliced_elts + offset <= n_elts,
            fmtstr("The number (%ld) of elements of sliced tensor exceeds that (%ld) of the original tensor",
                   n_sliced_elts + offset,
                   n_elts));
    }
    return Tensor(this->where(), this->type(), shape, this->getPtrWithOffset(offset));
}

TensorMap::TensorMap(const std::unordered_map<std::string, Tensor>& tensor_map) {
    for (auto& kv : tensor_map) {
        if (isValid(kv.second)) {
            insert(kv.first, kv.second);
        } else {
            FT_LOG_DEBUG(fmtstr("%s is not a valid tensor, skipping insert into TensorMap", kv.first.c_str()));
        }
    }
}

TensorMap::TensorMap(const std::vector<Tensor>& tensor_map) {
    for (size_t i = 0; i < tensor_map.size(); i++) {
        insert(std::to_string(i), tensor_map[i]);
    }
}

TensorMap::TensorMap(std::initializer_list<std::pair<std::string, Tensor>> tensor_map) {
    for (auto& pair : tensor_map) {
        if (isValid(pair.second)) {
            insert(pair.first, pair.second);
        } else {
            FT_LOG_DEBUG(fmtstr("%s is not a valid tensor, skipping insert into TensorMap", pair.first.c_str()));
        }
    }
}

TensorMap::~TensorMap() {
    tensor_map_.clear();
}

std::vector<std::string> TensorMap::keys() const {
    std::vector<std::string> key_names;
    for (auto& kv : tensor_map_) {
        key_names.push_back(kv.first);
    }
    return key_names;
}

std::string TensorMap::toString() const {
    std::stringstream ss;
    ss << "{";
    std::vector<std::string> key_names = keys();
    for (size_t i = 0; i < tensor_map_.size(); ++i) {
        ss << key_names[i] << ": " << at(key_names[i]).toString();
        if (i < tensor_map_.size() - 1) {
            ss << ", ";
        }
    }
    ss << "}";
    return ss.str();
}


}  // namespace fastertransformer

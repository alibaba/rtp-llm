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

#pragma once

#include "allocator.h"
#include "src/fastertransformer/utils/string_utils.h"
#include "src/fastertransformer/utils/assert_utils.h"

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

typedef enum datatype_enum {
    TYPE_INVALID,
    TYPE_BOOL,
    TYPE_UINT8,
    TYPE_UINT16,
    TYPE_UINT32,
    TYPE_UINT64,
    TYPE_INT8,
    TYPE_INT16,
    TYPE_INT32,
    TYPE_INT64,
    TYPE_FP16,
    TYPE_FP32,
    TYPE_FP64,
    TYPE_BYTES,
    TYPE_BF16,
    TYPE_FP8_E4M3,
    TYPE_STR,
    TYPE_VOID,
} DataType;

template<typename T>
DataType getTensorType();

typedef enum memorytype_enum {
    MEMORY_CPU,
    MEMORY_CPU_PINNED,
    MEMORY_GPU
} MemoryType;

struct Tensor {
public:
    Tensor();
    Tensor(const MemoryType          where,
           const DataType            type,
           const std::vector<size_t> shape,
           IAllocator*               allocator,
           const bool                is_set_zero);
    Tensor(const MemoryType where, const DataType type, const std::vector<size_t> shape, const void* data);

    ~Tensor();

    Tensor(const Tensor& tensor) = default;
    Tensor(Tensor&& tensor)      = default;
    Tensor& operator=(const Tensor& tensor) = default;
    Tensor& operator=(Tensor&& tensor) = default;

    MemoryType                 where() const;
    DataType                   type() const;
    const std::vector<size_t>& shape() const;
    void*                      data() const;
    IAllocator*                allocator() const;

    size_t size() const;
    size_t sizeBytes() const;

    std::string whereToString() const;
    std::string toString() const;
    std::string getNumpyTypeDesc(DataType type) const;

    void          saveNpy(const std::string& filename) const;
    static Tensor loadNpy(const std::string& npy_file, const MemoryType where);

    static DataType typeFromNumpyDesc(std::string type);
    static size_t   getTypeSize(DataType type);

    template<typename T>
    inline T getVal(size_t index) const;

    template<typename T>
    inline T getVal() const {
        FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
        if (getTensorType<T>() != type_) {
            FT_LOG_DEBUG("getVal with type %s, but data type is: %s",
                         getNumpyTypeDesc(getTensorType<T>()).c_str(),
                         getNumpyTypeDesc(type_).c_str());
        }
        return getVal<T>(0);
    }

    template<typename T>
    inline T* getPtr() const {
        FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
        if (getTensorType<T>() != type_) {
            FT_LOG_DEBUG("getPtr with type %s, but data type is: %s",
                         getNumpyTypeDesc(getTensorType<T>()).c_str(),
                         getNumpyTypeDesc(type_).c_str());
        }
        return (T*)data_;
    }

    inline void* getPtrWithOffset(size_t offset) const {
        FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
        if (data_ == nullptr) {
            return (void*)data_;
        } else {
            FT_CHECK_WITH_INFO(offset < size(),
                               "offset " + std::to_string(offset) + " is larger than buffer size"
                                   + std::to_string(size()));
            return (void*)((char*)data_ + offset * Tensor::getTypeSize(type_));
        }
    }

    template<typename T>
    inline T* getPtrWithOffset(size_t offset) const {
        FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
        if (getTensorType<T>() != type()) {
            FT_LOG_DEBUG("getVal with type %s, but data type is: %s",
                         getNumpyTypeDesc(getTensorType<T>()).c_str(),
                         getNumpyTypeDesc(type()).c_str());
        }
        if (data() == nullptr) {
            return (T*)data();
        } else {
            FT_CHECK_WITH_INFO(offset < size(),
                               fmtstr("offset (%lu) is larger than buffer size (%lu)", offset, size()));
            return ((T*)data()) + offset;
        }
    }

    template<typename T>
    T max() const {
        if (getTensorType<T>() != type()) {
            FT_LOG_DEBUG("getVal with type %s, but data type is: %s",
                         getNumpyTypeDesc(getTensorType<T>()).c_str(),
                         getNumpyTypeDesc(type()).c_str());
        }
        FT_CHECK_WITH_INFO(shape().size() > 0 && data() != nullptr, "Should be a non-empty tensor.");
        FT_CHECK_WITH_INFO(where_ == MEMORY_CPU || where_ == MEMORY_CPU_PINNED,
                           "max() supports MEMORY_CPU or MEMORY_CPU_PINNED tensor.");
        size_t max_idx = 0;
        T      max_val = getVal<T>(max_idx);
        for (size_t i = 1; i < size(); ++i) {
            T val = getVal<T>(i);
            if (val > max_val) {
                max_idx = i;
                max_val = val;
            }
        }
        return max_val;
    }

    template<typename T>
    T min() const {
        if (getTensorType<T>() != type()) {
            FT_LOG_DEBUG("getVal with type %s, but data type is: %s",
                         getNumpyTypeDesc(getTensorType<T>()).c_str(),
                         getNumpyTypeDesc(type()).c_str());
        }
        FT_CHECK_WITH_INFO(shape().size() > 0 && data() != nullptr, "Should be a non-empty tensor.");
        FT_CHECK_WITH_INFO(where_ == MEMORY_CPU || where_ == MEMORY_CPU_PINNED,
                           "min() supports MEMORY_CPU or MEMORY_CPU_PINNED tensor.");
        size_t min_idx = 0;
        T      min_val = getVal<T>(min_idx);
        for (size_t i = 1; i < size(); ++i) {
            T val = getVal<T>(i);
            if (val < min_val) {
                min_idx = i;
                min_val = val;
            }
        }
        return min_val;
    }

    template<typename T>
    T any(T val) const {
        if (getTensorType<T>() != type()) {
            FT_LOG_DEBUG("getVal with type %s, but data type is: %s",
                         getNumpyTypeDesc(getTensorType<T>()).c_str(),
                         getNumpyTypeDesc(type()).c_str());
        }
        FT_CHECK_WITH_INFO(shape().size() > 0 && data() != nullptr, "Should be a non-empty tensor.");
        FT_CHECK_WITH_INFO(where_ == MEMORY_CPU || where_ == MEMORY_CPU_PINNED,
                           "any() supports MEMORY_CPU or MEMORY_CPU_PINNED tensor.");
        for (size_t i = 0; i < size(); ++i) {
            if (getVal<T>(i) == val) {
                return true;
            }
        }
        return false;
    }

    template<typename T>
    T all(T val) const {
        if (getTensorType<T>() != type()) {
            FT_LOG_DEBUG("getVal with type %s, but data type is: %s",
                         getNumpyTypeDesc(getTensorType<T>()).c_str(),
                         getNumpyTypeDesc(type()).c_str());
        }
        FT_CHECK_WITH_INFO(shape().size() > 0 && data() != nullptr, "Should be a non-empty tensor.");
        FT_CHECK_WITH_INFO(where_ == MEMORY_CPU || where_ == MEMORY_CPU_PINNED,
                           "all() supports MEMORY_CPU or MEMORY_CPU_PINNED tensor.");
        for (size_t i = 0; i < size(); ++i) {
            if (getVal<T>(i) != val) {
                return false;
            }
        }
        return true;
    }

    Tensor slice(std::vector<size_t> shape, size_t offset = 0) const;

    template<typename T>
    std::string dataToString(size_t num_to_print = 0, size_t start = 0) const {
        std::string str = "";
        num_to_print    = num_to_print == 0 ? size() : std::min(num_to_print, size() - start);
        for (size_t i = start; i < num_to_print; ++i) {
            str += std::to_string(getVal<T>(i));
            if (i != size() - 1) {
                str += ", ";
            }
        }
        return str;
    }

private:
    static void parseNpyIntro(FILE*& f_ptr, uint32_t& header_len, uint32_t& start_data);
    static int  parseNpyHeader(FILE*& f_ptr, uint32_t header_len, DataType& type, std::vector<size_t>& shape);

private:
    MemoryType          where_;
    DataType            type_;
    std::vector<size_t> shape_;
    void*               data_      = nullptr;
    IAllocator*         allocator_ = nullptr;

    bool                 owned_;
    std::shared_ptr<int> ref_counter_;
};

class TensorMap {
private:
    std::unordered_map<std::string, Tensor> tensor_map_;

    inline bool isValid(const Tensor& tensor) {
        return tensor.size() > 0 && tensor.data() != nullptr;
    }

public:
    TensorMap() = default;
    TensorMap(const std::unordered_map<std::string, Tensor>& tensor_map);
    TensorMap(const std::vector<Tensor>& tensor_map);
    TensorMap(std::initializer_list<std::pair<std::string, Tensor>> tensor_map);
    ~TensorMap();

    inline size_t size() const {
        return tensor_map_.size();
    }

    inline bool isExist(const std::string& key) const {
        FT_LOG_DEBUG("%s for key: %s", __PRETTY_FUNCTION__, key.c_str());
        return tensor_map_.find(key) != tensor_map_.end();
    }

    std::vector<std::string> keys() const;

    inline void insert(const std::string& key, const Tensor& value) {
        FT_CHECK_WITH_INFO(!isExist(key), fmtstr("Duplicated key %s", key.c_str()));
        FT_CHECK_WITH_INFO(isValid(value), fmtstr("A none tensor or nullptr is not allowed (key is %s)", key.c_str()));
        tensor_map_.insert({key, value});
    }

    inline void insertIfValid(const std::string& key, const Tensor& value) {
        if (isValid(value)) {
            insert({key, value});
        }
    }

    inline void insert(std::pair<std::string, Tensor> p) {
        tensor_map_.insert(p);
    }

    // prevent converting int or size_t to string automatically
    Tensor at(int tmp)    = delete;
    Tensor at(size_t tmp) = delete;

    inline Tensor& at(const std::string& key) {
        FT_LOG_DEBUG("%s for key %s", __PRETTY_FUNCTION__, key.c_str());
        FT_CHECK_WITH_INFO(isExist(key),
                           fmtstr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                                  key.c_str(),
                                  vec2str(keys()).c_str()));
        return tensor_map_.at(key);
    }

    inline Tensor at(const std::string& key) const {
        FT_CHECK_WITH_INFO(isExist(key),
                           fmtstr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                                  key.c_str(),
                                  vec2str(keys()).c_str()));
        return std::move(tensor_map_.at(key));
    }

    inline Tensor& at(const std::string& key, Tensor& default_tensor) {
        FT_LOG_DEBUG("%s for key %s", __PRETTY_FUNCTION__, key.c_str());
        if (isExist(key)) {
            return tensor_map_.at(key);
        }
        return default_tensor;
    }

    inline Tensor at(const std::string& key, Tensor& default_tensor) const {
        FT_LOG_DEBUG("%s for key %s", __PRETTY_FUNCTION__, key.c_str());
        if (isExist(key)) {
            return tensor_map_.at(key);
        }
        return default_tensor;
    }

    inline Tensor& at(const std::string& key, Tensor&& default_tensor) {
        FT_LOG_DEBUG("%s for key %s", __PRETTY_FUNCTION__, key.c_str());
        if (isExist(key)) {
            return tensor_map_.at(key);
        }
        return default_tensor;
    }

    inline Tensor at(const std::string& key, Tensor&& default_tensor) const {
        if (isExist(key)) {
            return tensor_map_.at(key);
        }
        return default_tensor;
    }

    template<typename T>
    inline T getVal(const std::string& key) const {
        FT_CHECK_WITH_INFO(isExist(key),
                           fmtstr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                                  key.c_str(),
                                  vec2str(keys()).c_str()));
        return tensor_map_.at(key).getVal<T>();
    }

    template<typename T>
    inline T getVal(const std::string& key, T default_value) const {
        if (isExist(key)) {
            return tensor_map_.at(key).getVal<T>();
        }
        return default_value;
    }

    template<typename T>
    inline T getValWithOffset(const std::string& key, size_t index) const {
        FT_CHECK_WITH_INFO(isExist(key),
                           fmtstr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                                  key.c_str(),
                                  vec2str(keys()).c_str()));
        return tensor_map_.at(key).getVal<T>(index);
    }

    template<typename T>
    inline T getValWithOffset(const std::string& key, size_t index, T default_value) const {
        if (isExist(key)) {
            return tensor_map_.at(key).getVal<T>(index);
        }
        return default_value;
    }

    template<typename T>
    inline T* getPtr(const std::string& key) const {
        FT_CHECK_WITH_INFO(isExist(key),
                           fmtstr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                                  key.c_str(),
                                  vec2str(keys()).c_str()));
        return tensor_map_.at(key).getPtr<T>();
    }

    template<typename T>
    inline T* getPtr(const std::string& key, T* default_ptr) const {
        if (isExist(key)) {
            return tensor_map_.at(key).getPtr<T>();
        }
        return default_ptr;
    }

    template<typename T>
    inline T* getPtrWithOffset(const std::string& key, size_t index) const {
        FT_CHECK_WITH_INFO(isExist(key),
                           fmtstr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                                  key.c_str(),
                                  vec2str(keys()).c_str()));
        return tensor_map_.at(key).getPtrWithOffset<T>(index);
    }

    template<typename T>
    inline T* getPtrWithOffset(const std::string& key, size_t index, T* default_ptr) const {
        if (isExist(key)) {
            return tensor_map_.at(key).getPtrWithOffset<T>(index);
        }
        return default_ptr;
    }

    inline std::unordered_map<std::string, Tensor> getMap() const {
        return tensor_map_;
    }

    inline std::unordered_map<std::string, Tensor>::iterator begin() {
        return tensor_map_.begin();
    }

    inline std::unordered_map<std::string, Tensor>::iterator end() {
        return tensor_map_.end();
    }

    std::string      toString() const;
    static TensorMap fromNpyFolder(const std::string& base_folder);
    void             saveNpy(const std::string& base_folder);
};

}  // namespace fastertransformer
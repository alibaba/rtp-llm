#include "src/fastertransformer/core/Buffer.h"

using namespace std;

namespace fastertransformer {


Buffer::Buffer(const MemoryType where, const DataType type, const std::vector<size_t>& shape, const void* data)
    : where_(where)
    , type_(type)
    , shape_(shape)
    , data_(const_cast<void*>(data))
    {}

Buffer::~Buffer() {}

MemoryType Buffer::where() const {
    return where_;
}

DataType Buffer::type() const {
    return type_;
}

const std::vector<size_t>& Buffer::shape() const {
    return shape_;
}

void* Buffer::data() const {
    return data_;
}

size_t Buffer::size() const {
    size_t totalSize = 1;
    for (size_t dim : shape_) {
        totalSize *= dim;
    }
    return totalSize;
}

size_t Buffer::sizeBytes() const {
    return size() * getTypeSize(type_);
}

std::string Buffer::debugString() const {
    std::string debugStr = "Buffer( ";
    debugStr += "where=" + std::to_string(where_) + ", ";
    debugStr += "type=" + std::to_string(type_) + ", ";
    debugStr += "shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        debugStr += std::to_string(shape_[i]);
        if (i != shape_.size() - 1) {
            debugStr += ", ";
        }
    }
    debugStr += "], ";
    debugStr += "data=" + std::to_string(reinterpret_cast<uintptr_t>(data_));
    debugStr += " )";
    return debugStr;
}


} // namespace fastertransformer


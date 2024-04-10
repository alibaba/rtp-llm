#include "src/fastertransformer/core/Buffer.h"

#include <numeric>
#include <cassert>

using namespace std;

namespace fastertransformer {


Buffer::Buffer(const MemoryType where,
               const DataType type,
               const std::vector<size_t>& shape,
               const void* data,
               const std::function<void(Buffer *)> deleter)
    : where_(where)
    , type_(type)
    , shape_(shape)
    , data_(const_cast<void*>(data))
    , deleter_(deleter)
    {}

Buffer::~Buffer() {
    if (deleter_) {
        deleter_(this);
    }
}

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

void* Buffer::dataWithOffset(size_t offset) const {
    assert(type_ != DataType::TYPE_INVALID);
    return static_cast<char*>(data_) + offset * getTypeSize(type_);
}

size_t Buffer::size() const {
    if (shape_.empty()) {
        return 0;
    }
    size_t totalSize = 1;
    for (size_t dim : shape_) {
        totalSize *= dim;
    }
    return totalSize;
}

size_t Buffer::dim() const {
    return shape_.size();
}

size_t Buffer::sizeBytes() const {
    return size() * getTypeSize(type_);
}

void Buffer::reshape(std::vector<size_t>& shape) {
    int new_shape_size = std::accumulate(shape.begin(), shape.end(), 0);
    int old_shape_size = std::accumulate(shape_.begin(), shape_.end(), 0);
    assert(new_shape_size == old_shape_size);
    shape_ = shape;
}

// NOTE: view() always slices the buffer from 0-dim, no matter how many dimensions the buffer has.
// NOTE: new Buffer from view() has the same data pointer as the original buffer,
//       and also shorter life time than the original buffer.
//       user has the responsibility to keep the original buffer alive.
std::unique_ptr<Buffer> Buffer::view(size_t offset, size_t size) const {
    assert(offset + size <= this->shape_[0]);
    auto new_shape = shape_;
    new_shape[0] = size;
    const auto offset_size = this->size() / shape_[0] * offset;
    return std::make_unique<Buffer>(where_, type_, new_shape, dataWithOffset(offset_size));
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

bool Buffer::operator==(const Buffer& other) {

    return (other.data_ == data_) && (other.shape_ == shape_) &&
           (other.type_ == type_) && (other.where_ == where_);
}


} // namespace fastertransformer


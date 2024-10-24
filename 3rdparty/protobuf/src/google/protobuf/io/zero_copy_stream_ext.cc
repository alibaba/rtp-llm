#include <google/protobuf/io/zero_copy_stream_ext.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <google/protobuf/stubs/logging.h>


namespace google {
namespace protobuf {
namespace io {

MultiArrayInputStream::MultiArrayInputStream(const std::vector<void*>& bufs,
                                             const std::vector<int>& lens)
    : bufs_(bufs),
      lens_(lens),
      next_index_(0),
      input_(NULL),
      input_size_(0),
      bytes_count_(0)
{
    assert(bufs.size() == lens.size());
}

MultiArrayInputStream::~MultiArrayInputStream()
{
    if (input_ != NULL) {
        delete input_;
    }
}

bool MultiArrayInputStream::Next(const void** data, int* size)
{
    if (input_ == NULL || input_->ByteCount() >= input_size_ ) {
        if (!NextInput()) {
            return false;
        }
    }
    return input_->Next(data, size);
}

bool MultiArrayInputStream::NextInput()
{
    if (next_index_ >= (int)bufs_.size()) {
        return false;
    }
    if (input_ != NULL) {
        delete input_;
        input_ = NULL;
        bytes_count_ += input_size_;
    }
    input_ = new ArrayInputStream(bufs_[next_index_], lens_[next_index_]);
    input_size_ = lens_[next_index_];
    ++next_index_;
    return true;
}

void MultiArrayInputStream::BackUp(int count)
{
    GOOGLE_CHECK_NE(input_, NULL);
    if (input_ != NULL) {
        input_->BackUp(count);
    }
}

bool MultiArrayInputStream::Skip(int count)
{
    while (count > 0) {
        if (input_ == NULL && !NextInput()) {
            return false;
        }
        int left = input_size_ - input_->ByteCount();
        int toSkipCnt = std::min(left, count);
        bool ret = input_->Skip(toSkipCnt);
        if (!ret && input_->ByteCount() != input_size_) {
            return false;
        }
        count -= toSkipCnt;
    }
    return true;
}

int64 MultiArrayInputStream::ByteCount() const
{
    return bytes_count_ + (input_ == NULL ? 0 : input_->ByteCount());
}


}  // namespace io
}  // namespace protobuf
}  // namespace google

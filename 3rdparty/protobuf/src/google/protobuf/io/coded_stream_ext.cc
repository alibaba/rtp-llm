#include <google/protobuf/io/coded_stream_ext.h>
#include <google/protobuf/io/coded_stream_inl.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_ext.h>

namespace google {
namespace protobuf {
namespace io {

bool CodedStreamExtHelper::WriteRawString(CodedOutputStream* coded_out,
                                          std::RawString* rstr) {
  if (rstr == nullptr) {
    return false;
  }

  auto rawstr_output =
      dynamic_cast<ZeroCopyOutputStream4RawStr*>(coded_out->output_);
  if (!rawstr_output || !rawstr_output->SupportRawString()) {
    // The onput stream does not support zero copy of RawString, use original
    // WriteRaw() method.
    coded_out->WriteRaw(rstr->data(), rstr->size());
    return true;
  }

  coded_out->output_->BackUp(coded_out->buffer_size_);
  coded_out->buffer_ = NULL;
  coded_out->total_bytes_ -= coded_out->buffer_size_;
  coded_out->buffer_size_ = 0;
  int rstr_size = rstr->size();
  if (!rawstr_output->WriteRawString(rstr)) {
    return false;
  }

  coded_out->total_bytes_ += rstr_size;
  return true;
}

bool CodedStreamExtHelper::ReadRawString(CodedInputStream* coded_in,
                                         std::RawString* buffer, int size) {
  if (size < 0) {
    return false;
  }

  auto rawstr_input =
      dynamic_cast<ZeroCopyInputStream4RawStr*>(coded_in->input_);
  if (!rawstr_input) {
    // The input stream does not support zero copy of RawString. so we need to
    // create a new copy.
    char* raw = new char[size];
    if (!coded_in->InternalReadRawInline(raw, size)) {
      return false;
    }

    // Make RawString own this copy.
    std::shared_ptr<const char> shared_raw(raw, [raw](const char*){ delete[] raw; });
    buffer->assign(shared_raw, size);
    return true;
  }

  size_t filled_size = 0;
  while (size - filled_size > 0) {
    // Make sure 'coded_in' has buffer data.
    const size_t buf_size = coded_in->BufferSize();
    if (buf_size == 0) {
      if (!coded_in->Refresh()) {
        return false;
      }

      continue;
    }

    // Fill buffer data in 'coded_in' to RawString
    const size_t remaining_size = size - filled_size;
    const size_t data_size = std::min(buf_size, remaining_size);
    const char* data = reinterpret_cast<const char*>(coded_in->buffer_);

    if (!rawstr_input->FillRawString(buffer, size, data, data_size)) {
      return false;
    }
    coded_in->Advance(data_size);
    filled_size += data_size;
  }

  return true;
}

}  // namespace io
}  // namespace protobuf
}  // namespace google

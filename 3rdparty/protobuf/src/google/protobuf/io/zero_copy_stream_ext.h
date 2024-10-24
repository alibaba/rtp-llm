#ifndef GOOGLE_PROTOBUF_IO_ZERO_COPY_STREAM_EXT_H__
#define GOOGLE_PROTOBUF_IO_ZERO_COPY_STREAM_EXT_H__

#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <google/protobuf/rawstring.h>

#include <google/protobuf/port_def.inc>

namespace google {

namespace protobuf {
namespace io {

// Abstract interface to input stream for raw string.
// for compatible , have no better choice......
class PROTOBUF_EXPORT ZeroCopyInputStream4RawStr : public ZeroCopyInputStream {
 public:
  ZeroCopyInputStream4RawStr() {}
  virtual ~ZeroCopyInputStream4RawStr() {}

  virtual bool FillRawString(std::RawString* str, size_t str_size,
                             const char* data, size_t data_size) = 0;

 private:
  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(ZeroCopyInputStream4RawStr);
};

// A abstract output stream for raw string.
class PROTOBUF_EXPORT ZeroCopyOutputStream4RawStr
    : public ZeroCopyOutputStream {
 public:
  ZeroCopyOutputStream4RawStr() {}
  virtual ~ZeroCopyOutputStream4RawStr() {}
  // write RawString
  virtual bool WriteRawString(std::RawString* rstr) { return false; }
  virtual bool SupportRawString() const { return false; }

 private:
  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(ZeroCopyOutputStream4RawStr);
};

// A ZeroCopyInputStream which reads from a MultiArrayInputStream.  This is
// useful for implementing ZeroCopyInputStreams in the following scenario:
// Serialize a message to an non-continue memory space, this space split to
// some continue memory array, such as array of struct iovec or some pair of
// pointor or length.
class PROTOBUF_EXPORT MultiArrayInputStream
    : public ZeroCopyInputStream4RawStr {
 public:
  MultiArrayInputStream(const std::vector<void*>& bufs,
                        const std::vector<int>& lens);
  ~MultiArrayInputStream();
  /* override */ bool Next(const void** data, int* size);
  /* override */ void BackUp(int count);
  /* override */ bool Skip(int count);
  /* override */ int64 ByteCount() const;
  /* override */ bool SupportRawString() const { return true; }

 private:
  bool NextInput();

 private:
  std::vector<void*> bufs_;
  std::vector<int> lens_;
  size_t next_index_;
  ArrayInputStream* input_;
  size_t input_size_;
  int64 bytes_count_;
};

// wrapper for ArrayInputStream for support raw string.
class ResidentArrayInputStream : public io::ZeroCopyInputStream4RawStr {
 public:
  ResidentArrayInputStream(const void* data, int size, int block_size = -1)
      : input_(data, size, block_size) {}

  bool Next(const void** data, int* size) { return input_.Next(data, size); }
  void BackUp(int count) { return input_.BackUp(count); }
  bool Skip(int count) { return input_.Skip(count); }
  int64 ByteCount() const { return input_.ByteCount(); }
  // support raw string.
  bool SupportRawString() const { return true; }

 private:
  io::ArrayInputStream input_;
};

}  // namespace io
}  // namespace protobuf
}  // namespace google

#include <google/protobuf/port_undef.inc>

#endif

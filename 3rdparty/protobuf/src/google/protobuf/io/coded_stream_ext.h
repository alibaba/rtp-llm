#ifndef GOOGLE_PROTOBUF_IO_CODED_STREAM_EXT_H__
#define GOOGLE_PROTOBUF_IO_CODED_STREAM_EXT_H__

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/rawstring.h>

#include <google/protobuf/port_def.inc>

namespace google {
namespace protobuf {
namespace io {

class PROTOBUF_EXPORT CodedStreamExtHelper {
 public:
  static bool ReadRawString(CodedInputStream* coded_in,
                            std::RawString* buffer,
                            int size);
  // write raw string.
  static bool WriteRawString(CodedOutputStream* codec_out,
                             std::RawString* rstr);
};

}
}
}

#include <google/protobuf/port_undef.inc>

#endif


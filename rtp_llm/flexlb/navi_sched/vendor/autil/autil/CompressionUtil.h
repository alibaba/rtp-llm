#pragma once

#include <stddef.h>
#include <string>

#include "autil/mem_pool/Pool.h"

namespace autil {

enum class CompressType {
    NO_COMPRESS = 0,
    Z_SPEED_COMPRESS = 1,
    Z_DEFAULT_COMPRESS = 2,
    SNAPPY = 3,
    LZ4 = 4,
    Z_BEST_COMPRESS = 5,
    ZSTD = 6,
    MAX,
    INVALID_COMPRESS_TYPE = -1,
};

extern CompressType convertCompressType(const std::string &str);
extern const char *CompressType_Name(CompressType type);

class CompressionUtil {
public:
    template <typename StringType>
    static bool compress(const StringType &input,
                         CompressType type,
                         std::string &compressedResult,
                         autil::mem_pool::Pool *pool = nullptr) {
        return compressInternal(input.data(), input.size(), type, false, compressedResult, pool);
    }

    template <typename StringType>
    static bool decompress(const StringType &input,
                           CompressType type,
                           std::string &decompressedResult,
                           autil::mem_pool::Pool *pool = nullptr) {
        return decompressInternal(input.data(), input.size(), type, false, decompressedResult, pool);
    }

    static bool compressAppend(const char *data,
                               size_t len,
                               CompressType type,
                               std::string &compressedResult,
                               autil::mem_pool::Pool *pool = nullptr) {
        return compressInternal(data, len, type, true, compressedResult, pool);
    }

    static bool decompressAppend(const char *data,
                                 size_t len,
                                 CompressType type,
                                 std::string &decompressedResult,
                                 autil::mem_pool::Pool *pool = nullptr) {
        return decompressInternal(data, len, type, true, decompressedResult, pool);
    }

private:
    static bool compressInternal(const char *data,
                                 size_t len,
                                 CompressType type,
                                 bool append,
                                 std::string &compressedResult,
                                 autil::mem_pool::Pool *pool = nullptr);
    static bool decompressInternal(const char *data,
                                   size_t len,
                                   CompressType type,
                                   bool append,
                                   std::string &decompressedResult,
                                   autil::mem_pool::Pool *pool = nullptr);
};

} // namespace autil

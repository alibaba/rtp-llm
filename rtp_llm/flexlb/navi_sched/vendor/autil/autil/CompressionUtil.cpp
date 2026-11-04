#include "autil/CompressionUtil.h"

#include <cstring>

#include "autil/ZlibCompressor.h"
#include "autil/mem_pool/PoolBase.h"
#include "lz4.h"
#include "snappy.h"
#include "zlib.h"
#include "zstd.h"

namespace autil {

CompressType convertCompressType(const std::string &str) {
    if (str == "no_compress") {
        return CompressType::NO_COMPRESS;
    } else if (str == "z_speed_compress") {
        return CompressType::Z_SPEED_COMPRESS;
    } else if (str == "z_default_compress") {
        return CompressType::Z_DEFAULT_COMPRESS;
    } else if (str == "z_best_compress") {
        return CompressType::Z_BEST_COMPRESS;
    } else if (str == "snappy") {
        return CompressType::SNAPPY;
    } else if (str == "lz4") {
        return CompressType::LZ4;
    } else if (str == "zstd") {
        return CompressType::ZSTD;
    }
    return CompressType::INVALID_COMPRESS_TYPE;
}

const char *CompressType_Name(CompressType type) {
    switch (type) {
    case CompressType::Z_SPEED_COMPRESS:
        return "z_speed_compress";
    case CompressType::Z_DEFAULT_COMPRESS:
        return "z_default_compress";
    case CompressType::SNAPPY:
        return "snappy";
    case CompressType::LZ4:
        return "lz4";
    case CompressType::Z_BEST_COMPRESS:
        return "z_best_compress";
    case CompressType::ZSTD:
        return "zstd";
    default:
        return "no_compress";
    }
}

static int convertZlibCompressLevel(CompressType type) {
    switch (type) {
    case CompressType::Z_SPEED_COMPRESS:
        return Z_BEST_SPEED;
    case CompressType::Z_BEST_COMPRESS:
        return Z_BEST_COMPRESSION;
    default:
        return Z_DEFAULT_COMPRESSION;
    }
}

bool CompressionUtil::compressInternal(const char *input,
                                       size_t len,
                                       CompressType type,
                                       bool append,
                                       std::string &compressedResult,
                                       autil::mem_pool::Pool *pool) {
    switch (type) {
    case CompressType::Z_SPEED_COMPRESS:
    case CompressType::Z_DEFAULT_COMPRESS:
    case CompressType::Z_BEST_COMPRESS: {
        int compressType = convertZlibCompressLevel(type);
        autil::ZlibCompressor compressor(compressType);
        compressor.addDataToBufferIn(input, len);
        if (!compressor.compress()) {
            return false;
        }
        if (append) {
            compressedResult.append(compressor.getBufferOut(), compressor.getBufferOutLen());
        } else {
            compressedResult.assign(compressor.getBufferOut(), compressor.getBufferOutLen());
        }
        return true;
    }
    case CompressType::SNAPPY: {
        std::string tmpResult;
        auto result = snappy::Compress(input, len, &tmpResult);
        if (result) {
            if (append) {
                compressedResult += tmpResult;
            } else {
                compressedResult = std::move(tmpResult);
            }
        }
        return result;
    }
    case CompressType::LZ4: {
        int maxDestSize = LZ4_compressBound(len);
        if (0 == maxDestSize) {
            return false;
        }
        // header compress length
        char *dest = POOL_COMPATIBLE_NEW_VECTOR(pool, char, maxDestSize + sizeof(int));
        int compSize = LZ4_compress_fast(input, dest + sizeof(int), len, maxDestSize, 1);
        if (0 == compSize) {
            POOL_COMPATIBLE_DELETE_VECTOR(pool, dest);
            return false;
        }
        *(int *)dest = len;
        if (append) {
            compressedResult.append(dest, compSize + sizeof(int));
        } else {
            compressedResult.assign(dest, compSize + sizeof(int));
        }

        POOL_COMPATIBLE_DELETE_VECTOR(pool, dest);
        return true;
    }
    case CompressType::ZSTD: {
        int maxDestSize = ZSTD_compressBound(len);
        if (0 == maxDestSize) {
            return false;
        }
        char *dest = POOL_COMPATIBLE_NEW_VECTOR(pool, char, maxDestSize);
        auto compressedSize = ZSTD_compress(dest, maxDestSize, input, len, 1);
        if (ZSTD_isError(compressedSize)) {
            POOL_COMPATIBLE_DELETE_VECTOR(pool, dest);
            return false;
        }
        if (append) {
            compressedResult.append(dest, compressedSize);
        } else {
            compressedResult.assign(dest, compressedSize);
        }

        POOL_COMPATIBLE_DELETE_VECTOR(pool, dest);
        return true;
    }
    case CompressType::NO_COMPRESS: {
        if (append) {
            compressedResult.append(input, len);
        } else {
            compressedResult.assign(input, len);
        }
        return true;
    }
    default:
        return false;
    }
}

bool CompressionUtil::decompressInternal(const char *input,
                                         size_t len,
                                         CompressType type,
                                         bool append,
                                         std::string &decompressedResult,
                                         autil::mem_pool::Pool *pool) {
    switch (type) {
    case CompressType::SNAPPY: {
        std::string tmpResult;
        auto result = snappy::Uncompress(input, len, &tmpResult);
        if (result) {
            if (append) {
                decompressedResult += tmpResult;
            } else {
                decompressedResult = std::move(tmpResult);
            }
        }
        return result;
    }
    case CompressType::Z_SPEED_COMPRESS:
    case CompressType::Z_DEFAULT_COMPRESS:
    case CompressType::Z_BEST_COMPRESS: {
        int compressType = convertZlibCompressLevel(type);
        autil::ZlibCompressor compressor(compressType);
        compressor.addDataToBufferIn(input, len);
        if (!compressor.decompress()) {
            return false;
        }
        if (append) {
            decompressedResult.append(compressor.getBufferOut(), compressor.getBufferOutLen());
        } else {

            decompressedResult.assign(compressor.getBufferOut(), compressor.getBufferOutLen());
        }
        return true;
    }
    case CompressType::LZ4: {
        if (len < 4) {
            return false;
        }
        int originalSize;
        memcpy(&originalSize, input, sizeof(int));
        char *dest = POOL_COMPATIBLE_NEW_VECTOR(pool, char, originalSize + 1);
        int uncompRet =
            LZ4_decompress_safe_partial(input + sizeof(int), dest, len - sizeof(int), originalSize, originalSize + 1);
        if (uncompRet < 0 || uncompRet != originalSize) {
            POOL_COMPATIBLE_DELETE_VECTOR(pool, dest);
            return false;
        }
        if (append) {
            decompressedResult.append(dest, originalSize);
        } else {
            decompressedResult.assign(dest, originalSize);
        }
        POOL_COMPATIBLE_DELETE_VECTOR(pool, dest);
        return true;
    }
    case CompressType::ZSTD: {
        auto originalSize = ZSTD_getFrameContentSize(input, len);
        if (originalSize == ZSTD_CONTENTSIZE_ERROR || originalSize == ZSTD_CONTENTSIZE_UNKNOWN) {
            return false;
        }
        char *dest = POOL_COMPATIBLE_NEW_VECTOR(pool, char, originalSize + 1);
        auto uncompRet = ZSTD_decompress(dest, originalSize, input, len);
        if (ZSTD_isError(uncompRet)) {
            POOL_COMPATIBLE_DELETE_VECTOR(pool, dest);
            return false;
        }
        if (append) {
            decompressedResult.append(dest, originalSize);
        } else {
            decompressedResult.assign(dest, originalSize);
        }
        POOL_COMPATIBLE_DELETE_VECTOR(pool, dest);
        return true;
    }
    case CompressType::NO_COMPRESS: {
        if (append) {
            decompressedResult.append(input, len);
        } else {
            decompressedResult.assign(input, len);
        }
        return true;
    }
    default:
        return false;
    }
}
} // namespace autil

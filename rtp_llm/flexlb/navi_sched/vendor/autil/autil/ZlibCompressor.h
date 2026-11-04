#pragma once
#include <memory>
#include <stdint.h>
#include <string>
#include <zlib.h>

#include "autil/DynamicBuf.h"

namespace autil {
class ZlibCompressor {
public:
    static const int NO_COMPRESSION = Z_NO_COMPRESSION;
    static const int BEST_SPEED = Z_BEST_SPEED;
    static const int BEST_COMPRESSION = Z_BEST_COMPRESSION;
    static const int DEFAULT_COMPRESSION = Z_DEFAULT_COMPRESSION;

    static const uint32_t DEFAULT_BUFFER_SIZE = 64 * 1024; // 64k

public:
    ZlibCompressor(const int type = DEFAULT_COMPRESSION);
    ZlibCompressor(const ZlibCompressor &src);

    ~ZlibCompressor();

public:
    void setBufferInLen(const uint32_t bufferInLen) {
        _bufferIn.reset();
        _bufferIn.reserve(bufferInLen);
    }
    void setBufferOutLen(const uint32_t bufferOutLen) {
        _bufferOut.reset();
        _bufferOut.reserve(bufferOutLen);
    }

public:
    void addDataToBufferIn(const char *source, const uint32_t len);
    void addDataToBufferIn(const std::string &source);
    const char *getBufferOut();
    const char *getBufferIn();
    void copyBufferOut(char *out, const uint32_t len);
    bool compress();
    bool decompress();
    uint32_t getBufferOutLen();
    uint32_t getBufferInLen();
    void reset();

    // shortcut for addDataToBufferIn and compress()
    // and supports empty string
    bool compress(const std::string &input, std::string &output);
    bool decompress(const std::string &input, std::string &output);

private:
    int _compressType;
    DynamicBuf _bufferIn;
    DynamicBuf _bufferOut;
};
/////////////////////////////////////////////////////////////////////
//
inline void ZlibCompressor::addDataToBufferIn(const char *source, const uint32_t len) { _bufferIn.add(source, len); }

inline void ZlibCompressor::addDataToBufferIn(const std::string &source) {
    _bufferIn.add(source.c_str(), source.size());
}

inline const char *ZlibCompressor::getBufferOut() { return _bufferOut.getBuffer(); }

inline uint32_t ZlibCompressor::getBufferOutLen() { return _bufferOut.getDataLen(); }

inline uint32_t ZlibCompressor::getBufferInLen() { return _bufferIn.getDataLen(); }

inline const char *ZlibCompressor::getBufferIn() { return _bufferIn.getBuffer(); }

typedef std::shared_ptr<ZlibCompressor> ZlibCompressorPtr;

} // namespace autil

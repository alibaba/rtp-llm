#pragma once
#include <algorithm>
#include <assert.h>
#include <deque>
#include <memory>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/uio.h>

#include "rtp_llm/flexlb/navi_sched/vendor/autil/autil/Scope.h"

namespace anet {

class DataBuffer {
public:
    static constexpr size_t INITIAL_BUFFER_SIZE = 1024;

    static inline void int8ToBytes(int8_t value, char *dst) {
        //
        *dst = (char)value;
    }

    static inline void int16ToBytes(int16_t value, char *dst) {
        dst[1] = (char)value;
        value >>= 8;
        dst[0] = (char)value;
    }

    static inline void int32ToBytes(int32_t value, char *dst) {
        dst[3] = (char)value;
        value >>= 8;
        dst[2] = (char)value;
        value >>= 8;
        dst[1] = (char)value;
        value >>= 8;
        dst[0] = (char)value;
    }

    static inline void int64ToBytes(int64_t value, char *dst) {
        dst[7] = (unsigned char)value;
        value >>= 8;
        dst[6] = (unsigned char)value;
        value >>= 8;
        dst[5] = (unsigned char)value;
        value >>= 8;
        dst[4] = (unsigned char)value;
        value >>= 8;
        dst[3] = (unsigned char)value;
        value >>= 8;
        dst[2] = (unsigned char)value;
        value >>= 8;
        dst[1] = (unsigned char)value;
        value >>= 8;
        dst[0] = (unsigned char)value;
    }

    static inline int8_t bytesToInt8(const char *src) {
        assert(src);
        return (int8_t)*src;
    }

    static inline int16_t bytesToInt16(const char *src) {
        assert(src);
        auto unsignedSrc = reinterpret_cast<const unsigned char *>(src);

        // not portable!
        // need some macros to determine where big-endian or little endian
        int16_t value = unsignedSrc[0];
        value <<= 8;
        value |= static_cast<int16_t>(unsignedSrc[1]); // integral promotion
        return value;
    }

    static inline int32_t bytesToInt32(const char *src) {
        assert(src);
        auto unsignedSrc = reinterpret_cast<const unsigned char *>(src);

        int32_t value = unsignedSrc[0];
        value <<= 8;
        value |= unsignedSrc[1];
        value <<= 8;
        value |= unsignedSrc[2];
        value <<= 8;
        value |= unsignedSrc[3];
        return value;
    }

    static inline int64_t bytesToInt64(const char *src) {
        assert(src);
        auto unsignedSrc = reinterpret_cast<const unsigned char *>(src);

        int64_t value = unsignedSrc[0];
        value <<= 8;
        value |= unsignedSrc[1];
        value <<= 8;
        value |= unsignedSrc[2];
        value <<= 8;
        value |= unsignedSrc[3];
        value <<= 8;
        value |= unsignedSrc[4];
        value <<= 8;
        value |= unsignedSrc[5];
        value <<= 8;
        value |= unsignedSrc[6];
        value <<= 8;
        value |= unsignedSrc[7];
        return value;
    }

public:
    int64_t getSpaceUsed() { return _pend - _pstart.get(); }

    char *getData() {
        // getData() 方法要求返回一块连续内存，因此只能在 _savedChunks 为空的情况下被调用
        assert(_savedChunks.empty());
        return _pdata;
    }

    size_t getDataLen() { return (_pfree - _pdata) + _packedChunksTotalSize + _sharedChunksTotalSize; }

    bool hasSavedChunks() { return !_savedChunks.empty(); }

    std::vector<iovec> getDataIovs() {
        std::vector<iovec> iovs;
        int iovcnt = std::min(UIO_MAXIOV, (int)(_savedChunks.size() + 1));
        iovs.reserve(iovcnt);
        for (auto &chunk : _savedChunks) {
            if (--iovcnt < 0) {
                return iovs;
            }
            iovs.emplace_back(iovec{
                .iov_base = getChunkAddr(chunk),
                .iov_len = chunk.size,
            });
        }
        if (_pdata != nullptr && _pfree - _pdata > 0) {
            if (--iovcnt < 0) {
                return iovs;
            }
            iovs.emplace_back(iovec{
                .iov_base = _pdata,
                .iov_len = static_cast<size_t>(_pfree - _pdata),
            });
        }
        return iovs;
    }

    std::vector<iovec> getDataIovsByOffset(size_t offset) {
        if (offset >= getDataLen()) {
            return {};
        }

        if (_savedChunks.empty()) {
            return {iovec{
                .iov_base = _pdata + offset,
                .iov_len = static_cast<size_t>(_pfree - _pdata) - offset,
            }};
        }

        return doGetDataIovsByOffset(offset);
    }

    char *getFree() { return _pfree; }

    size_t getFreeLen() { return _pend - _pfree; }

    void drainData(size_t len) {
        if (len == 0) {
            return;
        }

        if (_savedChunks.empty()) {
            if (_pdata == nullptr) {
                return;
            }
            _pdata += len;
            if (_pdata > _pfree) { //~Bug #95
                _pdata = _pfree;
            }
        } else {
            doDrainData(len);
        }

        if (getDataLen() == 0 && _pstart.unique()) {
            clear();
        }
    }

    void pourData(size_t len) {
        if (len == 0) {
            return;
        }

        _pfree += len;
        if (_pfree > _pend) {
            _pfree = _pend;
        }
    }

    void stripData(size_t len) {
        if (len == 0 || getDataLen() == 0) {
            return;
        }

        if (_savedChunks.empty()) {
            _pfree -= len;
            if (_pfree < _pdata) {
                _pfree = _pdata;
            }
        } else {
            doStripData(len);
        }
    }

    void clear() {
        if (_pstart.unique()) {
            _pdata = _pstart.get();
            _pfree = _pstart.get();
        } else {
            _pstart.reset();
            _pdata = nullptr;
            _pfree = nullptr;
            _pend = nullptr;
        }

        _savedChunks.clear();
        _packedChunksTotalSize = 0;
        _sharedChunksTotalSize = 0;
    }

    void shrink() {
        if (_pstart == NULL) {
            return;
        }

        size_t dataLen = _pfree - _pdata + _packedChunksTotalSize;
        char *dataAddr = _pfree - dataLen;
        assert(dataLen >= 0);
        size_t savedChunkOffset = dataAddr - _pstart.get();

        if (dataLen > 0) {
            std::shared_ptr<char> newBuf((char *)malloc(dataLen), [](char *ptr) { free(ptr); });
            assert(newBuf != nullptr);
            memcpy(newBuf.get(), dataAddr, dataLen);
            _pstart = newBuf;
            _pdata = _pstart.get() + _packedChunksTotalSize;
            _pfree = _pstart.get() + dataLen;
            _pend = _pstart.get() + dataLen;
        } else {
            _pstart = nullptr;
            _pdata = nullptr;
            _pfree = nullptr;
            _pend = nullptr;
        }

        for (auto &chunk : _savedChunks) {
            if (chunk.packed) {
                chunk.offset -= savedChunkOffset;
            }
        }

        return;
    }

    void writeInt8(int8_t n) {
        expand(1);

        int8ToBytes(n, _pfree);
        _pfree += 1;
    }

    void writeInt16(int16_t n) {
        expand(2);

        int16ToBytes(n, _pfree);
        _pfree += 2;
    }

    void writeInt32(int32_t n) {
        expand(4);

        int32ToBytes(n, _pfree);
        _pfree += 4;
    }

    void writeInt64(int64_t n) {
        expand(8);

        int64ToBytes(n, _pfree);
        _pfree += 8;
    }

    void writeBytes(const void *src, size_t len) {
        if (src == nullptr || len == 0) {
            return;
        }

        expand(len);
        memcpy(_pfree, src, len);
        _pfree += len;
    }

    void writeBytes(const std::shared_ptr<const char> &src, size_t len) {
        if (src == nullptr || len <= 0) {
            return;
        }

        if (_pdata != nullptr && _pfree - _pdata > 0) {
            _savedChunks.emplace_back(LogicalChunk{
                .packed = true,
                .offset = static_cast<size_t>(_pdata - _pstart.get()),
                .size = static_cast<size_t>(_pfree - _pdata),
            });
            _packedChunksTotalSize += _pfree - _pdata;
            _pdata = _pfree;
        }

        _savedChunks.emplace_back(LogicalChunk{
            .packed = false,
            .sharedAddr = src,
            .size = len,
        });
        _sharedChunksTotalSize += len;
    }

    void fillInt32(unsigned char *dst, uint32_t n) {
        dst[3] = (unsigned char)n;
        n >>= 8;
        dst[2] = (unsigned char)n;
        n >>= 8;
        dst[1] = (unsigned char)n;
        n >>= 8;
        dst[0] = (unsigned char)n;
    }

    void fillInt32(size_t offset, int32_t value) {
        if (offset + 4 > getDataLen()) {
            return;
        }

        if (_savedChunks.empty()) {
            int32ToBytes(value, _pdata + offset);
            return;
        }

        std::array<char, 4> bytes;
        int32ToBytes(value, bytes.data());
        fillBytes(offset, bytes.data(), 4);
    }

    int8_t readInt8() {
        if (getDataLen() < 1) {
            return 0;
        }

        if (_savedChunks.empty()) {
            return bytesToInt8(_pdata++);
        } else {
            std::array<char, 1> bytes;
            doReadBytes(bytes.data(), bytes.size());
            return bytesToInt8(bytes.data());
        }
    }

    int16_t readInt16() {
        if (getDataLen() < 2) {
            return 0;
        }

        if (_savedChunks.empty()) {
            int16_t value = bytesToInt16(_pdata);
            _pdata += 2;
            return value;
        } else {
            std::array<char, 2> bytes;
            doReadBytes(bytes.data(), bytes.size());
            return bytesToInt16(bytes.data());
        }
    }

    int32_t readInt32() {
        if (getDataLen() < 4) {
            return 0;
        }

        if (_savedChunks.empty()) {
            int32_t value = bytesToInt32(_pdata);
            _pdata += 4;
            return value;
        } else {
            std::array<char, 4> bytes;
            doReadBytes(bytes.data(), bytes.size());
            return bytesToInt32(bytes.data());
        }
    }

    int64_t readInt64() {
        if (getDataLen() < 8) {
            return 0;
        }

        if (_savedChunks.empty()) {
            int64_t value = bytesToInt64(_pdata);
            _pdata += 8;
            return value;
        } else {
            std::array<char, 8> bytes;
            doReadBytes(bytes.data(), bytes.size());
            return bytesToInt64(bytes.data());
        }
    }

    void readBytes(void *dst, size_t len) {
        if (dst == nullptr || len == 0) {
            return;
        }

        if (getDataLen() < len) {
            return;
        }

        doReadBytes(dst, len);
    }

    void readBytes(std::shared_ptr<const char> &dst, size_t len) {
        if (len == 0 || getDataLen() < len) {
            return;
        }

        if (_savedChunks.empty()) {
            dst = std::shared_ptr<const char>(_pstart, _pdata);
            _pdata += len;
        } else {
            dst = std::shared_ptr<const char>((char *)malloc(len), [](char *ptr) { free(ptr); });
            doReadBytes((char *)dst.get(), len);
        }
    }

    void ensureFree(size_t len) {
        if (len == 0) {
            return;
        }

        expand(len);
    }

    ssize_t findBytes(const char *findstr, size_t len) {
        if (findstr == nullptr || len == 0) {
            return -1;
        }

        if (_savedChunks.empty()) {
            size_t dLen = _pfree - _pdata - len + 1;
            char *data = (char *)_pdata;
            for (size_t i = 0; i < dLen; i++) {
                if (data[i] == findstr[0] && memcmp(data + i, findstr, len) == 0) {
                    return i;
                }
            }
            return -1;
        } else {
            return findBytesInMultiChunks(findstr, len);
        }
    }

private:
    inline void expand(size_t need) {
        assert(need > 0);

        if (_pstart == nullptr) {
            size_t len = INITIAL_BUFFER_SIZE;
            while (len < need) {
                len <<= 1;
            }

            _pstart = std::shared_ptr<char>((char *)malloc(len), [](char *ptr) { free(ptr); });
            _pdata = _pstart.get();
            _pfree = _pstart.get();
            _pend = _pstart.get() + len;
        } else if (_pend - _pfree < need) {
            // no enough free space

            size_t dataLen = _pfree - _pdata + _packedChunksTotalSize;
            char *dataAddr = _pfree - dataLen;
            size_t savedChunkOffset = dataAddr - _pstart.get();

            if (_pstart.unique()) {
                // _pstart.unique() 为真，表示数据没有共享给其他人，这时候可以安全地执行 memmove 操作
                size_t freeLen = _pend - _pstart.get() - dataLen;

                if (freeLen < need || freeLen * 4 < dataLen) {
                    double bufsize = static_cast<double>((_pend - _pstart.get()) * 2);
                    while (static_cast<size_t>(bufsize) - dataLen < need) {
                        bufsize *= 1.5;
                    }

                    std::shared_ptr<char> newBuf((char *)malloc((size_t)bufsize), [](char *ptr) { free(ptr); });
                    if (dataLen > 0) {
                        memcpy(newBuf.get(), dataAddr, dataLen);
                    }

                    _pstart = newBuf;
                    _pdata = _pstart.get() + _packedChunksTotalSize;
                    _pfree = _pstart.get() + dataLen;
                    _pend = _pstart.get() + (size_t)bufsize;
                } else {
                    memmove(_pstart.get(), dataAddr, dataLen);
                    _pdata = _pstart.get() + _packedChunksTotalSize;
                    _pfree = _pstart.get() + dataLen;
                }
            } else {
                // _pstart.unique() 为假，表示 _pstart 已经共享给了其他人，这时候只能直接扩容，否则如果执行 memmove
                // 将覆盖其他人的数据
                size_t bufSize = INITIAL_BUFFER_SIZE;
                while (bufSize < dataLen + need) {
                    bufSize <<= 1;
                }

                std::shared_ptr<char> newBuf((char *)malloc((int)bufSize), [](char *ptr) { free(ptr); });
                if (dataLen > 0) {
                    memcpy(newBuf.get(), dataAddr, dataLen);
                }

                _pstart = newBuf;
                _pdata = _pstart.get() + _packedChunksTotalSize;
                _pfree = _pstart.get() + dataLen;
                _pend = _pstart.get() + (int)bufSize;
            }

            for (auto &chunk : _savedChunks) {
                if (chunk.packed) {
                    chunk.offset -= savedChunkOffset;
                }
            }
        }
    }

    void doDrainData(size_t toDrainSize) {
        // 排空已经写入的数据，可能涉及到多个 chunks, 因此从前往后逐个处理
        while (!_savedChunks.empty()) {
            auto &chunk = _savedChunks.front();

            size_t chunkToDrainSize = std::min(chunk.size, toDrainSize);
            chunk.offset += chunkToDrainSize;
            chunk.size -= chunkToDrainSize;

            if (chunk.packed) {
                _packedChunksTotalSize -= chunkToDrainSize;
            } else {
                _sharedChunksTotalSize -= chunkToDrainSize;
            }

            if (chunk.size == 0) {
                _savedChunks.pop_front();
            }

            toDrainSize -= chunkToDrainSize;
            if (toDrainSize == 0) {
                break;
            }
        }

        // 所有 saved chunks 都排空了，接着排空 pdata
        _pdata += toDrainSize;
        if (_pdata > _pfree) {
            _pdata = _pfree;
        }
    }

    void doStripData(size_t toStripSize) {
        if (toStripSize == 0) {
            return;
        }

        if (_pfree - _pdata > 0) {
            size_t len = std::min((size_t)(_pfree - _pdata), toStripSize);
            _pfree -= len;
            toStripSize -= len;
            if (toStripSize == 0) {
                return;
            }
        }

        // 裁切已写入的数据，可能涉及到多个 chunks, 因此从后往前逐个处理
        while (!_savedChunks.empty()) {
            auto &chunk = _savedChunks.back();

            size_t chunkToStripSize = std::min(chunk.size, toStripSize);
            chunk.size -= chunkToStripSize;

            if (chunk.packed) {
                _packedChunksTotalSize -= chunkToStripSize;
                _pdata = getChunkAddr(chunk);
                _pfree = getChunkAddr(chunk);
            } else {
                _sharedChunksTotalSize -= chunkToStripSize;
            }

            if (chunk.size == 0) {
                _savedChunks.pop_back();
            }

            toStripSize -= chunkToStripSize;
            if (toStripSize == 0) {
                break;
            }
        }
    }

    void doReadBytes(void *dst, size_t toReadSize) {
        assert(dst != nullptr);
        assert(toReadSize > 0);
        assert(getDataLen() >= toReadSize);

        // 读取已经写入的数据，可能涉及到多个 chunks, 因此从前往后逐个处理
        while (!_savedChunks.empty()) {
            auto &chunk = _savedChunks.front();

            size_t chunkToReadSize = std::min(chunk.size, toReadSize);

            memcpy(dst, getChunkAddr(chunk), chunkToReadSize);
            dst = (char *)dst + chunkToReadSize;

            chunk.offset += chunkToReadSize;
            chunk.size -= chunkToReadSize;

            if (chunk.packed) {
                _packedChunksTotalSize -= chunkToReadSize;
            } else {
                _sharedChunksTotalSize -= chunkToReadSize;
            }

            if (chunk.size == 0) {
                _savedChunks.pop_front();
            }

            toReadSize -= chunkToReadSize;
            if (toReadSize == 0) {
                break;
            }
        }

        // 所有 saved chunks 都读取完了，接着读取 pdata
        if (toReadSize > 0) {
            memcpy(dst, _pdata, toReadSize);
            _pdata += toReadSize;
        }
    }

    ssize_t findBytesInMultiChunks(const char *findstr, size_t len) {
        if (findstr == nullptr || len == 0 || getDataLen() < len) {
            return -1;
        }

        // 先把 _pdata 压入队列，方便在循环中处理查找逻辑，执行结束需要从队列中移除
        _savedChunks.emplace_back(LogicalChunk{
            .packed = true,
            .offset = static_cast<size_t>(_pdata - _pstart.get()),
            .size = static_cast<size_t>(_pfree - _pdata),
        });
        autil::ScopeGuard extraChunkGuard([this]() { _savedChunks.pop_back(); });

        auto toFindStr = findstr;
        auto toFindLen = len;

        // 在多块内存中查找指定字符串，采用暴力的方法逐个遍历字符
        ssize_t offset = -1;
        for (int i = 0; i < _savedChunks.size(); ++i) {
            auto &chunk = _savedChunks.at(i);
            auto chunkData = getChunkAddr(chunk);
            for (int j = 0; j < chunk.size; ++j) {
                offset += 1;
                toFindStr = findstr;
                toFindLen = len;

                if (chunkData[j] != toFindStr[0]) {
                    continue;
                }

                auto cmpSize = std::min(chunk.size - j, toFindLen);
                if (memcmp(chunkData + j, toFindStr, cmpSize) != 0) {
                    continue;
                }

                toFindStr += cmpSize;
                toFindLen -= cmpSize;
                if (toFindLen == 0) {
                    return offset;
                }

                for (int k = i + 1; k < _savedChunks.size(); ++k) {
                    auto &nextChunk = _savedChunks.at(k);
                    auto nextChunkData = getChunkAddr(nextChunk);
                    cmpSize = std::min(nextChunk.size, toFindLen);
                    if (memcmp(nextChunkData, toFindStr, cmpSize) != 0) {
                        break;
                    }

                    toFindStr += cmpSize;
                    toFindLen -= cmpSize;
                    if (toFindLen == 0) {
                        break;
                    }
                }

                if (toFindLen == 0) {
                    return offset;
                }
            }
        }

        return -1;
    }

    void fillBytes(size_t offset, const char *src, size_t toFillSize) {
        assert(offset >= 0);
        assert(src != nullptr);
        assert(toFillSize > 0);
        assert(offset + toFillSize <= getDataLen());

        // 先把 _pdata 压入队列，方便在循环中处理查找逻辑，执行结束需要从队列中移除
        _savedChunks.emplace_back(LogicalChunk{
            .packed = true,
            .offset = static_cast<size_t>(_pdata - _pstart.get()),
            .size = static_cast<size_t>(_pfree - _pdata),
        });
        autil::ScopeGuard extraChunkGuard([this]() { _savedChunks.pop_back(); });

        // 按照偏移填充数据，可能涉及到多个chunks, 因此从前往后逐个处理
        for (auto &chunk : _savedChunks) {
            size_t chunkToSkipSize = std::min(chunk.size, offset);
            offset -= chunkToSkipSize;
            if (offset > 0) {
                continue;
            }
            assert(offset == 0);

            auto dst = getChunkAddr(chunk) + chunkToSkipSize;
            size_t chunkToFillSize = std::min(toFillSize, chunk.size - chunkToSkipSize);
            if (chunkToFillSize == 0) {
                continue;
            }
            assert(chunkToFillSize > 0);

            memcpy(dst, src, chunkToFillSize);

            src += chunkToFillSize;
            toFillSize -= chunkToFillSize;

            if (toFillSize == 0) {
                break;
            }
        }

        assert(toFillSize == 0);
    }

    std::vector<iovec> doGetDataIovsByOffset(size_t offset) {
        assert(offset >= 0);
        assert(offset <= getDataLen());

        // 先把 _pdata 压入队列，方便在循环中处理查找逻辑，执行结束需要从队列中移除
        _savedChunks.emplace_back(LogicalChunk{
            .packed = true,
            .offset = static_cast<size_t>(_pdata - _pstart.get()),
            .size = static_cast<size_t>(_pfree - _pdata),
        });
        autil::ScopeGuard extraChunkGuard([this]() { _savedChunks.pop_back(); });

        std::vector<iovec> iovs;
        for (auto &chunk : _savedChunks) {
            size_t chunkToSkipSize = std::min(chunk.size, offset);
            offset -= chunkToSkipSize;
            if (offset > 0) {
                continue;
            }
            assert(offset == 0);

            size_t chunkRemainingSize = chunk.size - chunkToSkipSize;
            if (chunkRemainingSize == 0) {
                continue;
            }
            assert(chunkRemainingSize > 0);

            iovs.emplace_back(iovec{
                .iov_base = getChunkAddr(chunk) + chunkToSkipSize,
                .iov_len = chunkRemainingSize,
            });
        }

        return iovs;
    }

    struct LogicalChunk;

    char *getChunkAddr(LogicalChunk &chunk) {
        if (chunk.packed) {
            return _pstart.get() + chunk.offset;
        } else {
            return (char *)chunk.sharedAddr.get() + chunk.offset;
        }
    }

private:
    std::shared_ptr<char> _pstart; // buffer start address
    char *_pend{nullptr};          // buffer end address
    char *_pfree{nullptr};         // end of data
    char *_pdata{nullptr};         // begin of data

    struct LogicalChunk {
        bool packed{false};
        size_t offset{0};
        std::shared_ptr<const char> sharedAddr;
        size_t size{0};
    };

    std::deque<LogicalChunk> _savedChunks;

    size_t _packedChunksTotalSize{0};
    size_t _sharedChunksTotalSize{0};
};

} // namespace anet

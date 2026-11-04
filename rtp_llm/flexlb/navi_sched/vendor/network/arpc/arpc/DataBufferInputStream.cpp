#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/DataBufferInputStream.h"

ARPC_BEGIN_NAMESPACE(arpc)

bool DataBufferInputStream::Next(const void **data, int *size) {
    if (_messageData == nullptr) {
        return false;
    }
    if (_offset == _messageSize) {
        return false;
    }

    *data = _messageData.get() + _offset;
    *size = _messageSize - _offset;
    _offset = _messageSize;
    return true;
}

void DataBufferInputStream::BackUp(int count) {
    _offset -= count;
}

bool DataBufferInputStream::Skip(int count) {
    _offset += count;
    return  true;
}

int64_t DataBufferInputStream::ByteCount() const {
    return _offset;
}

bool DataBufferInputStream::FillRawString(std::RawString *str, size_t str_size, const char *data, size_t data_size) {
    if (str == nullptr || data == nullptr) {
        return false;
    }

    std::shared_ptr<const char> rawStringData(_messageData, data);
    str->assign(rawStringData, data_size);
    return true;
}


ARPC_END_NAMESPACE(arpc)

#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/DataBufferOutputStream.h"

#include <cassert>
#include <stddef.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/databuffer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/runnable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/util/Log.h"

ARPC_BEGIN_NAMESPACE(arpc);
using namespace anet;
const int DataBufferOutputStream::BLOCK_SIZE;

DataBufferOutputStream::DataBufferOutputStream(DataBuffer *dataBuffer, int blockSize) {
    assert(dataBuffer);
    _dataBuffer = dataBuffer;
    _blockSize = blockSize;
    _byteCount = 0;
}

DataBufferOutputStream::~DataBufferOutputStream() {
    _dataBuffer = NULL;
    _byteCount = 0;
}

bool DataBufferOutputStream::Next(void **data, int *size) {
    if (_dataBuffer->getFreeLen() == 0) {
        _dataBuffer->ensureFree(_blockSize);
    }

    *data = _dataBuffer->getFree();
    *size = _dataBuffer->getFreeLen();

    _dataBuffer->pourData(*size);
    _byteCount += *size;

    return true;
}

void DataBufferOutputStream::BackUp(int count) {
    _dataBuffer->stripData(count);
    _byteCount -= count;
}

RPCint64 DataBufferOutputStream::ByteCount() const { return _byteCount; }

bool DataBufferOutputStream::WriteRawString(std::RawString *rstr) {
    if (rstr == nullptr) {
        return false;
    }

    _dataBuffer->writeBytes(rstr->shared_data(), rstr->size());
    return true;
}


ARPC_END_NAMESPACE(arpc);

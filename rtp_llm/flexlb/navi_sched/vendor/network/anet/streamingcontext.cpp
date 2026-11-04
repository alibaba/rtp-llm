#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/streamingcontext.h"

#include <stddef.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/packet.h"

namespace anet {

StreamingContext::StreamingContext() {
    _packet = NULL;
    _completed = false;
    _broken = false;
    _eof = false;
    _errorNo = 0;
    _errorString = NULL;
}

StreamingContext::~StreamingContext() { reset(); }

bool StreamingContext::isCompleted() { return _completed; }

void StreamingContext::setCompleted(bool completed) { _completed = completed; }

bool StreamingContext::isBroken() { return _broken; }

void StreamingContext::setBroken(bool broken) { _broken = broken; }

bool StreamingContext::isEndOfFile() { return _eof; }

void StreamingContext::setEndOfFile(bool eof) { _eof = eof; }

Packet *StreamingContext::getPacket() { return _packet; }

Packet *StreamingContext::stealPacket() {
    Packet *tmpPacket = _packet;
    _packet = NULL;
    return tmpPacket;
}

void StreamingContext::setPacket(Packet *packet) {
    if (packet != _packet) {
        if (NULL != _packet) {
            _packet->free();
        }
        _packet = packet;
    }
}

void StreamingContext::setErrorNo(int errorNo) { _errorNo = errorNo; }

int StreamingContext::getErrorNo() { return _errorNo; }

void StreamingContext::setErrorString(const char *errorString) { _errorString = errorString; }

const char *StreamingContext::getErrorString() { return _errorString; }

void StreamingContext::reset() {
    setPacket(NULL);
    _completed = false;
    _broken = false;
    _eof = false;
    _errorNo = 0;
    _errorString = NULL;
}

} /*end namespace anet*/

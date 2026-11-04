/**
 * File name: delaydecodepacket.cpp
 * Author: lizhang
 * Create time: 2010-11-07 09:25:26
 * $Id$
 *
 * Description: ***add description here***
 *
 */

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/delaydecodepacket.h"

#include <assert.h>
#include <stddef.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/databuffer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/databufferserializable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/packet.h"

namespace anet {
DelayDecodePacket::DelayDecodePacket()
    : _ownContent(false), _content(NULL), _needDrainDataFlag(false), _inputDataBuffer(NULL) {}

DelayDecodePacket::~DelayDecodePacket() {
    clearContent();
    clearInputDataBuffer();
}

void DelayDecodePacket::setContent(DataBufferSerializable *content, bool ownContent) {
    assert(content);
    clearContent();
    _content = content;
    setContentOwnership(ownContent);
}

DataBufferSerializable *DelayDecodePacket::getContent() { return _content; }

DataBufferSerializable *DelayDecodePacket::stealContent() {
    DataBufferSerializable *ret = NULL;
    if (ownContent()) {
        ret = _content;
        _content = NULL;
        setContentOwnership(false);
    }
    return ret;
}

bool DelayDecodePacket::encode(DataBuffer *output) {
    assert(output);
    if (_content) {
        return _content->serialize(output);
    }
    return false;
}

// header is deprecated pamameter
bool DelayDecodePacket::decode(DataBuffer *input, PacketHeader *header) {
    assert(input);
    setInputDataBuffer(input);
    setNeedDrainDataFlag(true);
    return true;
}

bool DelayDecodePacket::decodeToContent() {
    assert(_inputDataBuffer);
    bool ret = false;
    if (_content) {
        ret = _content->deserialize(_inputDataBuffer, getEffecitveDataLength());
        setNeedDrainDataFlag(false);
    }
    return ret;
}

char *DelayDecodePacket::getData() {
    if (_inputDataBuffer) {
        return _inputDataBuffer->getData();
    } else {
        return NULL;
    }
}

int DelayDecodePacket::getEffecitveDataLength() { return _packetHeader._dataLen; }

void DelayDecodePacket::drainData() { drainData(getEffecitveDataLength()); }

void DelayDecodePacket::drainData(int len) {
    assert(_inputDataBuffer);
    _inputDataBuffer->drainData(len);

    // no matter how much data drained, clear _needDrainDataFlag
    setNeedDrainDataFlag(false);
}

void DelayDecodePacket::setInputDataBuffer(DataBuffer *inputDataBuffer) {
    clearInputDataBuffer();
    _inputDataBuffer = inputDataBuffer;
}

DataBuffer *DelayDecodePacket::getInputDataBuffer() { return _inputDataBuffer; }

void DelayDecodePacket::clearContent() {
    if (ownContent() && _content) {
        _content->free();
    }
    _content = NULL;
    setContentOwnership(false);
}

void DelayDecodePacket::clearInputDataBuffer() {
    if (needDrainData() && _inputDataBuffer) {
        drainData();
    }
    setNeedDrainDataFlag(false);
    _inputDataBuffer = NULL;
}

int64_t DelayDecodePacket::getSpaceUsed() {
    int64_t spaceUsed = 0;
    spaceUsed += sizeof(PacketHeader) + sizeof(DelayDecodePacket);
    if (ownContent() && _content) {
        spaceUsed += _content->getSpaceUsed();
    }
    return spaceUsed;
}

} /*end namespace anet*/

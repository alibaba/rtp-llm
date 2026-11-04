#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/directtcpconnection.h"

#include <assert.h>
#include <errno.h>
#include <stdint.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/channel.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/channelpool.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/connection.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/databuffer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/directpacket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/directpacketstreamer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/directstreamingcontext.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ilogger.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/iocomponent.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ipacketstreamer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/log.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/orderedpacketqueue.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/socket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/streamingcontext.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/tcpconnection.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/threadcond.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/timeutil.h"

namespace anet {
class IServerAdapter;

constexpr int32_t DIRECT_READ_WRITE_SIZE = READ_WRITE_SIZE;

static const DirectPayload kEmptyPayload{};

DirectTCPConnection::DirectTCPConnection(Socket *socket, IPacketStreamer *streamer, IServerAdapter *serverAdapter)
    : TCPConnection(socket, streamer, serverAdapter) {
    _directStreamer = dynamic_cast<DirectPacketStreamer *>(streamer);
    assert(nullptr != _directStreamer);
    _directContext = dynamic_cast<DirectStreamingContext *>(_context);
    assert(nullptr != _directContext);
    ANET_LOG(DEBUG, "DirectTCPConnection create");
}

DirectTCPConnection::~DirectTCPConnection() {
    ANET_LOG(DEBUG, "DirectTCPConnection destroy");
    clearWritingPacket(); // in case no error, but packet is still sending
}

bool DirectTCPConnection::writeData() {
    ANET_LOG(DEBUG, "DirectTCPConnection writeData");
    // to reduce the odds of blocking postPacket()
    _outputCond.lock();
    _outputQueue.moveTo(&_myQueue);
    if (_myQueue.size() == 0 && _output.getDataLen() == 0 && _payloadLeftToWrite == 0UL) {
        ANET_LOG(DEBUG, "IOC(%p)->enableWrite(false)", _iocomponent);
        _iocomponent->enableWrite(false);
        _outputCond.unlock();
        return true;
    }
    _outputCond.unlock();
    ssize_t ret = 0;
    int writeCnt = 0;
    size_t myQueueSize = _myQueue.size();
    int error = 0;
    _lasttime = TimeUtil::getTime();
    do {
        while (_output.getDataLen() < DIRECT_READ_WRITE_SIZE) {
            int64_t oldDataLen = _output.getDataLen();
            int64_t oldSpaceAllocated = _output.getSpaceUsed();
            if (_payloadLeftToWrite > 0UL) {
                assert(nullptr != _writingPacket);
                goto CONTINUE_WRITE_PAYLOAD;
            }
            if (myQueueSize == 0) {
                break;
            }
            // _writingPacket must be recycled on error
            _writingPacket = static_cast<anet::DirectPacket *>(_myQueue.pop());
            myQueueSize--;
            _directStreamer->encode(_writingPacket, &_output, _writingPayload);
            if (_writingPayload.getAddr() != nullptr && _writingPayload.getLen() > 0UL) {
                assert(_payloadLeftToWrite == 0UL);
                _payloadLeftToWrite = _writingPayload.getLen();
            }
            {
                int64_t newDataLen = _output.getDataLen();
                int64_t newSpaceAllocated = _output.getSpaceUsed();
                int64_t packetSizeInBuffer = newDataLen - oldDataLen;
                if (packetSizeInBuffer > _maxSendPacketSize) {
                    _maxSendPacketSize = packetSizeInBuffer;
                }
                addOutputBufferSpaceAllocated(newSpaceAllocated - oldSpaceAllocated);
                Channel *channel = _writingPacket->getChannel();
                if (channel) {
                    if (_defaultPacketHandler == NULL && channel->getHandler() == NULL) {
                        // free channel of packets not expecting reply
                        _channelPool.freeChannel(channel);
                    } else {
                        ANET_LOG(SPAM, "channel[%p] expire time[%ld]", channel, _writingPacket->getExpireTime());
                        channel->setExpireTime(_writingPacket->getExpireTime());
                        _channelPool.addToWaitingList(channel);
                    }
                }
            }
        CONTINUE_WRITE_PAYLOAD:
            if (_payloadLeftToWrite > 0) { // write direct payload:
                if (_output.getDataLen() > 0) {
                    ret = sendBuffer(writeCnt, error);
                    if (_output.getDataLen() > 0) {
                        break;
                    }
                }
                ret = sendPayload(writeCnt, error);
                if (0UL == _payloadLeftToWrite) { // write to socket success with payload
                    finishPacketWrite();
                }
                break;
            } else { // write to buffer success without payload
                finishPacketWrite();
            }
        }
        if (_output.getDataLen() > 0) {
            ret = sendBuffer(writeCnt, error);
        }
    } while (ret > 0 && _output.getDataLen() == 0 && _payloadLeftToWrite == 0 && myQueueSize > 0 && writeCnt < 10);
    _outputCond.lock();
    _outputQueue.moveBack(&_myQueue);
    if (error != 0 && error != EWOULDBLOCK && error != EAGAIN) {
        char spec[64];
        ANET_LOG(WARN, "Connection (%s) write error: %d", _socket->getAddr(spec, 64), error);
        _outputCond.unlock();
        clearOutputBuffer();
        clearWritingPacket(); // clear packet on error
        return false;
    }
    size_t queueSize = _outputQueue.size() + (_output.getDataLen() > 0 ? 1 : 0) + (_payloadLeftToWrite > 0 ? 1 : 0);
    if (queueSize > 0) {
        // when using level triggered mode, do NOT need to call enableWrite() any more.
    } else if (_writeFinishShutdown) {
        _outputCond.unlock();
        _iocomponent->enableWrite(false);
        _iocomponent->shutdownSocket();
        return true;
    } else if (_writeFinishClose) {
        ANET_LOG(DEBUG, "Initiative cut connect.");
        _outputCond.unlock();
        clearOutputBuffer();
        return false;
    } else {
        ANET_LOG(DEBUG, "IOC(%p)->enableWrite(false)", _iocomponent);
        _iocomponent->enableWrite(false);
    }
    if (!_isServer && _queueLimit > 0) {
        size_t queueTotalSize = _channelPool.getUseListCount();
        if (queueTotalSize < _queueLimit && _waitingThreads > 0) {
            _outputCond.broadcast();
        }
    }
    _outputCond.unlock();
    return true;
}

bool DirectTCPConnection::readData() {
    ANET_LOG(DEBUG, "DirectTCPConnection readData");
    ssize_t ret = 0;
    int readCnt = 0;
    bool broken = false;
    int error = 0;

    /** read loop
     * loop until we get a packet anet header, packet size header and message part,
     * then we process them and read payload directly into user given address.
     * broken packet should be handled.
     */
    do {
        ++readCnt;
        int64_t oldSpaceUsed = _input.getSpaceUsed();
        if (!_directContext->isPayloadCompleted()) {
            assert(nullptr != _directContext->getDirectPacket());
            goto CONTINUE_READ_PAYLOAD;
        }
        _input.ensureFree(DIRECT_READ_WRITE_SIZE);
        ret = _socket->read(_input.getFree(), DIRECT_READ_WRITE_SIZE);
        ANET_LOG(SPAM, "%ld bytes read", ret);
        if (ret < 0) {
            int64_t newSpaceUsed = _input.getSpaceUsed();
            addInputBufferSpaceAllocated(newSpaceUsed - oldSpaceUsed);
            error = _socket->getSoError();
            break;
        }
        if (0 == ret) {
            _directContext->setEndOfFile(true);
        }
        _input.pourData(ret);
        _stats.totalRxBytes += ret;
        while (_directStreamer->processData(&_input, _directContext)) {
            if (!_directContext->isCompleted()) {
                break;
            }
        CONTINUE_READ_PAYLOAD:
            bool success = _directStreamer->readPayload(&_input, _socket, &_channelPool, _directContext, ret);
            if (ret > 0) {
                _stats.totalRxBytes += ret;
            }
            if (!success) {
                _directContext->setPayloadCompleted(false);
                if (!_directContext->isBroken() && ret < 0) {
                    error = _socket->getSoError();
                }
                break;
            }

            // continue original logic
            auto packet = _directContext->stealDirectPacket();
            int64_t packetSizeInBuffer = ANET_DIRECT_PACKET_INFO_LEN + packet->getDirectHeader()._msgSize;
            if (packetSizeInBuffer > _maxRecvPacketSize) {
                _maxRecvPacketSize = packetSizeInBuffer;
            }
            assert((int32_t)_directContext->getPayloadReadOffset() == packet->getDirectHeader()._payloadSize);
            packet->cleanupFlags();
            handlePacket(packet);
            _directContext->reset();
        }
        broken = _directContext->isBroken();

        int64_t newSpaceUsed = _input.getSpaceUsed();
        addInputBufferSpaceAllocated(newSpaceUsed - oldSpaceUsed);
    } while (ret == DIRECT_READ_WRITE_SIZE && !broken && readCnt < 10);

    if (!broken) {
        if (ret == 0) { // maybe closing connection
            broken = true;
        } else if (ret < 0) {
            if (error != 0 && error != EAGAIN && error != EWOULDBLOCK) {
                char spec[64];
                ANET_LOG(DEBUG, "Connection (%s) read error: %d", _socket->getAddr(spec, 64), error);
                broken = true;
            }
        }
    }
    return !broken;
}

void DirectTCPConnection::clearWritingPacket() {
    assert(_payloadLeftToWrite <= 0UL || nullptr != _writingPacket);
    if (nullptr == _writingPacket) {
        return;
    }
    _writingPacket->free();
    _writingPacket = nullptr;
    _writingPayload = kEmptyPayload;
    _payloadLeftToWrite = 0UL;
}

void DirectTCPConnection::finishPacketWrite() {
    assert(0UL == _payloadLeftToWrite);
    _writingPacket->invokeDequeueCB();
    clearWritingPacket();
}

ssize_t DirectTCPConnection::sendBuffer(int &writeCnt, int &error) {
    ssize_t ret = _socket->write(_output.getData(), _output.getDataLen());
    ANET_LOG(SPAM, "%ld bytes written from buffer", ret);
    if (ret > 0) {
        _output.drainData(ret);
        _stats.totalTxBytes += ret;
    } else {
        error = _socket->getSoError();
    }
    writeCnt++;
    return ret;
}

ssize_t DirectTCPConnection::sendPayload(int &writeCnt, int &error) {
    assert(_payloadLeftToWrite > 0UL && nullptr != _writingPacket);
    auto addr = _writingPayload.getAddr();
    auto len = _writingPayload.getLen();
    ssize_t ret = _socket->write(addr + len - _payloadLeftToWrite, _payloadLeftToWrite);
    if (ret > 0) {
        ANET_LOG(SPAM, "%ld bytes written from payload:[%p]", ret, addr);
        _stats.totalTxBytes += ret;
        _payloadLeftToWrite -= ret;
    } else {
        error = _socket->getSoError();
    }
    writeCnt++;
    return ret;
}

} // namespace anet

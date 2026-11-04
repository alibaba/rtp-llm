#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/tcpconnection.h"

#include <errno.h>
#include <stdint.h>
#include <string.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/channel.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/channelpool.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/connection.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/databuffer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ilogger.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/iocomponent.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ipacketstreamer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/log.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/orderedpacketqueue.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/packet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/socket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/streamingcontext.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/threadcond.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/timeutil.h"

namespace anet {
class IServerAdapter;

TCPConnection::TCPConnection(Socket *socket, IPacketStreamer *streamer, IServerAdapter *serverAdapter)
    : Connection(socket, streamer, serverAdapter) {
    _gotHeader = false;
    _writeFinishClose = false;
    _writeFinishShutdown = false;
    memset(&_packetHeader, 0, sizeof(_packetHeader));
    _inputBufferSpaceAllocated = 0;
    _outputBufferSpaceAllocated = 0;
    _maxRecvPacketSize = 0;
    _maxSendPacketSize = 0;
}

TCPConnection::~TCPConnection() {
    addInputBufferSpaceAllocated(0 - _input.getSpaceUsed());
    addOutputBufferSpaceAllocated(0 - _output.getSpaceUsed());
}

void TCPConnection::clearOutputBuffer() {
    _output.clear();
}

bool TCPConnection::writeData() {
    // to reduce the odds of blocking postPacket()
    _outputCond.lock();
    _outputQueue.moveTo(&_myQueue);
    if (_myQueue.size() == 0 && _output.getDataLen() == 0) {
        ANET_LOG(DEBUG, "IOC(%p)->enableWrite(false)", _iocomponent);
        _iocomponent->enableWrite(false);
        _outputCond.unlock();
        return true;
    }
    _outputCond.unlock();

    Packet *packet;
    ssize_t ret = 0;
    size_t expect = 0;
    int writeCnt = 0;
    size_t myQueueSize = _myQueue.size();
    _stats.queueSize = myQueueSize;
    int error = 0;

    _lasttime = TimeUtil::getTime();
    do {
        while (_output.getDataLen() < _readWriteBufSize) {
            if (myQueueSize == 0) {
                break;
            }

            packet = _myQueue.pop();
            auto rt = TimeUtil::getTime() - packet->getPostTimeUs();
            if (rt > 3000000) {
                char spec[64];
                ANET_LOG(INFO, "Connection (%s) write packet post time(%ld) too long: post already cost time(%ld), packetId(%u) receive time(%ld).",
                        _socket->getAddr(spec, 64), packet->getPostTimeUs(), rt, packet->getPacketHeader()->_chid, packet->getReceiveTimeUs());
            }
            myQueueSize--;
            int64_t oldDataLen = _output.getDataLen();
            int64_t oldSpaceAllocated = _output.getSpaceUsed();
            _streamer->encode(packet, &_output);
            int64_t newDataLen = _output.getDataLen();
            int64_t newSpaceAllocated = _output.getSpaceUsed();
            int64_t packetSizeInBuffer = newDataLen - oldDataLen;
            if (packetSizeInBuffer > _maxSendPacketSize) {
                _maxSendPacketSize = packetSizeInBuffer;
            }
            addOutputBufferSpaceAllocated(newSpaceAllocated - oldSpaceAllocated);
            Channel *channel = packet->getChannel();
            if (channel) {
                if (_defaultPacketHandler == NULL && channel->getHandler() == NULL) {
                    // free channle of packets not expecting reply
                    _channelPool.freeChannel(channel);
                } else {
                    ANET_LOG(SPAM, "channel[%p] expire time[%ld]", channel, packet->getExpireTime());
                    channel->setExpireTime(packet->getExpireTime());
                    channel->setTimeoutMs(packet->getTimeoutMs());
                    _channelPool.addToWaitingList(channel);
                }
            }
            packet->invokeDequeueCB();
            packet->free();
        }

        if (_output.getDataLen() == 0) {
            break;
        }

        // write data
        if (_output.hasSavedChunks()) {
            expect = 0;
            std::vector<iovec> iovs = _output.getDataIovs();
            for (auto &iov : iovs) {
                expect += iov.iov_len;
            }
            ret = _socket->writev(iovs);
        } else {
            expect = _output.getDataLen();
            ret = _socket->write(_output.getData(), expect);
        }
        if (ret > 0) {
            _output.drainData(ret);
            _stats.totalTxBytes += ret;
            if (ret != expect) {
                char spec[64];
                ANET_LOG(DEBUG, "Connection (%s) write partial: expect(%ld), ret(%ld).",
                        _socket->getAddr(spec, 64), expect, ret);
            }
        } else {
            error = _socket->getSoError();
            char spec[64];
            ANET_LOG(DEBUG, "Connection (%s) write error: %d", _socket->getAddr(spec, 64), error);
        }

        writeCnt++;
    } while (ret > 0 // socket is healthy
             && ret == expect // socket buffer is not full
             && (_output.getDataLen() > 0 || myQueueSize > 0) // write is partially done
             /**@todo remove magic number 10*/
             && writeCnt < 10); // avoid occupying CPU for a long time
    _stats.callWriteCount += writeCnt;

    _outputCond.lock();
    _outputQueue.moveBack(&_myQueue);

    if (error != 0 && error != EWOULDBLOCK && error != EAGAIN) {
        char spec[64];
        ANET_LOG(WARN, "Connection (%s) write error: %d", _socket->getAddr(spec, 64), error);
        _outputCond.unlock();
        clearOutputBuffer();
        return false;
    }
    size_t queueSize = _outputQueue.size() + (_output.getDataLen() > 0 ? 1 : 0);
    if (queueSize > 0) {
        ANET_LOG(DEBUG, "Output queue is not drained: queueSize(%ld).", queueSize);
        // when using level triggered mode, do NOT need to call enableWrite() any more.
        //         ANET_LOG(DEBUG,"IOC(%p)->enableWrite(true)", _iocomponent);
        //         _iocomponent->enableWrite(true);
    } else if (_writeFinishShutdown) {
        ANET_LOG(DEBUG, "Write finish shutdown.");
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

    // 如果是client, 并且有queue长度的限制
    if (!_isServer && _queueLimit > 0) {
        size_t queueTotalSize = _channelPool.getUseListCount();
        if (queueTotalSize < _queueLimit && _waitingThreads > 0) {
            _outputCond.broadcast();
        }
    }
    _outputCond.unlock();

    return true;
}

bool TCPConnection::readData() {
    ssize_t ret = 0;
    int readCnt = 0;
    bool broken = false;
    int error = 0;

    do {
        readCnt++;
        int64_t oldSpaceUsed = _input.getSpaceUsed();
        _input.ensureFree(_readWriteBufSize);
        ret = _socket->read(_input.getFree(), _readWriteBufSize);
        ANET_LOG(SPAM, "%ld bytes read", ret);
        if (ret < 0) {
            int64_t newSpaceUsed = _input.getSpaceUsed();
            addInputBufferSpaceAllocated(newSpaceUsed - oldSpaceUsed);

            error = _socket->getSoError();
            break;
        }
        if (0 == ret) {
            _context->setEndOfFile(true);
        }
        _input.pourData(ret);
        _stats.totalRxBytes += ret;
        while (_streamer->processData(&_input, _context)) {
            if (!_context->isCompleted()) {
                break;
            }
            Packet *packet = _context->stealPacket();
            int64_t packetSizeInBuffer = PACKET_INFO_LEN + packet->getDataLen();
            if (packetSizeInBuffer > _maxRecvPacketSize) {
                _maxRecvPacketSize = packetSizeInBuffer;
            }

            /* Give packet a chance to do some cleanup before passing it to
             * higher level logic. */
            packet->cleanupFlags();
            packet->setReceiveTimeUs(TimeUtil::getTime());
            handlePacket(packet);
            _context->reset();
        }
        broken = _context->isBroken();

        // _streamer->processData() maybe invoke _input.ensureFree()
        int64_t newSpaceUsed = _input.getSpaceUsed();
        addInputBufferSpaceAllocated(newSpaceUsed - oldSpaceUsed);

    } while (ret == _readWriteBufSize && !broken && readCnt < 10);

    _stats.callReadCount += readCnt;

    if (!broken) {
        if (ret == 0) {
            broken = true;
        } else if (ret < 0) {
            if (error != 0 && error != EAGAIN && error != EWOULDBLOCK) {
                char spec[64];
                ANET_LOG(DEBUG, "Connection (%s) read error: %d", _socket->getAddr(spec, 64), error);
                broken = true;
            }
        }
    }

    // when using level triggered mode, do NOT need to call enableRead() any more.
    //      if (!broken) {
    //          _outputCond.lock();
    //          ANET_LOG(DEBUG,"IOC(%p)->enableRead(true)", _iocomponent);
    //          _iocomponent->enableRead(true);
    //          _outputCond.unlock();
    //      }

    return !broken;
}

void TCPConnection::addInputBufferSpaceAllocated(int64_t size) {
    if (size == 0) {
        return;
    }
    _inputBufferSpaceAllocated += size;
}

void TCPConnection::addOutputBufferSpaceAllocated(int64_t size) {
    if (size == 0) {
        return;
    }
    _outputBufferSpaceAllocated += size;
}

void TCPConnection::shrinkInputBuffer() {
    int64_t oldSpaceUsed = _input.getSpaceUsed();
    _input.shrink();
    int64_t newSpaceUsed = _input.getSpaceUsed();
    addInputBufferSpaceAllocated(newSpaceUsed - oldSpaceUsed);
}

void TCPConnection::shrinkOutputBuffer() {
    int64_t oldSpaceUsed = _output.getSpaceUsed();
    _output.shrink();
    int64_t newSpaceUsed = _output.getSpaceUsed();
    addOutputBufferSpaceAllocated(newSpaceUsed - oldSpaceUsed);
}

} // namespace anet

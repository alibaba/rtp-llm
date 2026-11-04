#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigStreamBase.h"

using namespace std;

namespace multi_call {

AUTIL_DECLARE_AND_SETUP_LOGGER(multi_call, GigStreamBase);

PartMessageQueue::PartMessageQueue() : _receiving(false), _count(0), _ec(MULTI_CALL_ERROR_NONE) {
}

PartMessageQueue::~PartMessageQueue() {
}

bool PartMessageQueue::beginReceive() {
    bool expect = false;
    return _receiving.compare_exchange_weak(expect, true);
}

bool PartMessageQueue::endReceive() {
    bool expect = true;
    return _receiving.compare_exchange_weak(expect, false);
}

bool PartMessageQueue::nextMessage(GigStreamMessage &message, MultiCallErrorCode &ec) {
    autil::ScopedLock lock(_lock);
    if (_messageList.empty()) {
        return false;
    }
    message = _messageList.front();
    _messageList.pop_front();
    if (_messageList.empty()) {
        ec = _ec;
    } else {
        ec = MULTI_CALL_ERROR_NONE;
    }
    _count--;
    return true;
}

bool PartMessageQueue::peekMessage(GigStreamMessage &message, MultiCallErrorCode &ec) {
    autil::ScopedLock lock(_lock);
    if (_messageList.empty()) {
        return false;
    }
    message = _messageList.front();
    ec = _ec;
    if (1u == _messageList.size()) {
        ec = _ec;
    } else {
        ec = MULTI_CALL_ERROR_NONE;
    }
    return true;
}

void PartMessageQueue::appendMessage(const GigStreamMessage &message, MultiCallErrorCode ec) {
    autil::ScopedLock lock(_lock);
    _messageList.push_back(message);
    _ec = ec;
    _count++;
}

GigStreamBase::GigStreamBase() {
}

GigStreamBase::~GigStreamBase() {
}

bool GigStreamBase::hasError() const {
    return false;
}

bool GigStreamBase::innerReceive(const GigStreamMessage &message, MultiCallErrorCode ec) {
    if (!_param.asyncMode) {
        if (MULTI_CALL_ERROR_NONE == ec) {
            return receive(message);
        } else {
            receiveCancel(message, ec);
            return true;
        }
    } else {
        return appendMessage(message, ec);
    }
}

bool GigStreamBase::appendMessage(const GigStreamMessage &message, MultiCallErrorCode ec) {
    auto partId = message.partId;
    auto &partQueue = getPartQueue(partId);
    partQueue.appendMessage(message, ec);
    return tryNotify(partId);
}

bool GigStreamBase::tryNotify(PartIdTy partId) {
    auto &partQueue = getPartQueue(partId);
    if (partQueue.empty()) {
        return true;
    }
    if (partQueue.beginReceive()) {
        return notifyReceive(partId);
    }
    return true;
}

bool GigStreamBase::asyncReceive(PartIdTy partId) {
    return asyncReceive(partId, 0);
}

bool GigStreamBase::asyncReceive(PartIdTy partId, size_t maxMessageCount) {
    auto &partQueue = getPartQueue(partId);
    size_t receivedCount = 0;
    while (true) {
        if (!doReceive(partQueue, receivedCount, maxMessageCount)) {
            return false;
        }
        if (maxMessageCount && receivedCount == maxMessageCount && !partQueue.empty()) {
            // Keep receive ownership while handing off, so arrivals cannot start a second consumer.
            return notifyReceive(partId);
        }
        partQueue.endReceive();
        MULTI_CALL_MEMORY_BARRIER();
        if (!partQueue.empty()) {
            if (!partQueue.beginReceive()) {
                break;
            }
        } else {
            break;
        }
    }
    return true;
}

bool GigStreamBase::doReceive(PartMessageQueue &partQueue, size_t &receivedCount,
                              size_t maxMessageCount) {
    while (!maxMessageCount || receivedCount < maxMessageCount) {
        if (partQueue.empty()) {
            break;
        }
        GigStreamMessage message;
        MultiCallErrorCode ec;
        if (partQueue.nextMessage(message, ec)) {
            ++receivedCount;
            if (MULTI_CALL_ERROR_NONE == ec) {
                if (!receive(message)) {
                    return false;
                }
            } else {
                receiveCancel(message, ec);
            }
        } else {
            continue;
        }
    }
    return true;
}

bool GigStreamBase::peekMessage(PartIdTy partId, GigStreamMessage &message,
                                MultiCallErrorCode &ec) {
    auto &partQueue = getPartQueue(partId);
    if (partQueue.empty()) {
        return false;
    }
    return partQueue.peekMessage(message, ec);
}

PartMessageQueue &GigStreamBase::getPartQueue(PartIdTy partId) {
    {
        autil::ScopedReadLock lock(_asyncQueueLock);
        auto it = _partAsyncQueues.find(partId);
        ;
        if (_partAsyncQueues.end() != it) {
            return it->second;
        }
    }
    {
        autil::ScopedWriteLock lock(_asyncQueueLock);
        auto it = _partAsyncQueues.find(partId);
        ;
        if (_partAsyncQueues.end() != it) {
            return it->second;
        }
        return _partAsyncQueues[partId];
    }
}

} // namespace multi_call

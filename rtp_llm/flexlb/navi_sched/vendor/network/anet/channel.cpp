#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/channel.h"

#include <stdint.h>
#include <stdlib.h>

namespace anet {
class IPacketHandler;

Channel::Channel() { init(0); }

void Channel::init(uint64_t id) {
    _prev = NULL;
    _next = NULL;
    _id = id;
    _expireTime = 0;
    _timeoutMs = 0;
    _handler = NULL;
    _args = NULL;
}

void Channel::setId(uint64_t id) { _id = id; }

uint64_t Channel::getId() { return _id; }

void Channel::setArgs(void *args) { _args = args; }

void *Channel::getArgs() { return _args; }

void Channel::setHandler(IPacketHandler *handler) { _handler = handler; }

IPacketHandler *Channel::getHandler() { return _handler; }

/*
 * set expirt time
 *
 * @param milliseconds, 0 never timeout
 */
void Channel::setExpireTime(int64_t expireTime) { _expireTime = expireTime; }

/*
 * set raw expirt time
 *
 */
void Channel::setTimeoutMs(int32_t timeoutMs) { _timeoutMs = timeoutMs; }

} // namespace anet

#pragma once
#include <stdint.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/directplaceholder.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ipackethandler.h"

BEGIN_ANET_NS();

class Channel {
    friend class ChannelPool;
    friend class ChannelPoolTest;
    friend class ChannelPoolTest_testAllocChannel_Test;
    friend class ChannelPoolTest_testFreeChannel_Test;

public:
    Channel();
    void init(uint64_t id);
    void setId(uint64_t id);
    uint64_t getId();

    void setArgs(void *args);
    void *getArgs();

    void setHandler(IPacketHandler *handler);
    IPacketHandler *getHandler();

    const DirectPlaceholder &getPlaceholder() const { return _placeholder; }
    void setPlaceholder(const DirectPlaceholder &placeholder) { _placeholder = placeholder; }

    void setExpireTime(int64_t expireTime);
    int64_t getExpireTime() { return _expireTime; }

    void setTimeoutMs(int32_t timeoutMs);
    int32_t getTimeoutMs() { return _timeoutMs; }

    Channel *getNext() { return _next; }
    Channel *getPrev() { return _prev; }

private:
    uint64_t _id; // channel id
    void *_args;  // call back argument
    IPacketHandler *_handler;
    DirectPlaceholder _placeholder;
    int64_t _expireTime;
    int32_t _timeoutMs;

private:
    Channel *_prev;
    Channel *_next;
};
END_ANET_NS();


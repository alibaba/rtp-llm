#pragma once

#include <google/protobuf/message.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"

namespace multi_call {

class GigStreamMessage
{
public:
    GigStreamMessage()
        : partId(INVALID_PART_ID)
        , message(nullptr)
        , eof(false)
        , handlerId(-1)
        , netLatencyUs(0) {
    }

public:
    void initArena() {
        arena.reset(new google::protobuf::Arena());
    }

public:
    PartIdTy partId;
    google::protobuf::Message *message;
    bool eof;
    std::shared_ptr<google::protobuf::Arena> arena;
    int64_t handlerId;
    int64_t netLatencyUs;
};

typedef std::map<PartIdTy, GigStreamMessage> GigStreamMessageMap;
} // namespace multi_call


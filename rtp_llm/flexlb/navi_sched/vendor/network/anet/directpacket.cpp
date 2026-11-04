#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/directpacket.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/databuffer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ilogger.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/log.h"

namespace anet {

bool DirectPacket::decode(DataBuffer *input, DirectPacketHeader *header) {
    if (input->getDataLen() < header->_msgSize) {
        ANET_LOG(ERROR,
                 "invalid input to decode with dataLen:[%ld] and header's msgSize:[%d]",
                 input->getDataLen(),
                 header->_msgSize);
        return false;
    }
    bool rc = setBody(input->getData(), header->_msgSize);
    input->drainData(header->_msgSize);
    return rc;
}

} // namespace anet

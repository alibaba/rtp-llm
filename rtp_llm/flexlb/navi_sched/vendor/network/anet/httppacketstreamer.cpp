#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/httppacketstreamer.h"

#include <string.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/databuffer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/defaultpacketstreamer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/packet.h"

namespace anet {
class IPacketFactory;
/*
 * 构造函数
 */

HttpPacketStreamer::HttpPacketStreamer(IPacketFactory *factory) : DefaultPacketStreamer(factory) {
    _existPacketHeader = false; // 不要输出头信息
}

/*
 * 数据包信息的设置
 */
bool HttpPacketStreamer::getPacketInfo(DataBuffer *input, PacketHeader *header, bool *broken) {
    if (input->getDataLen() == 0) {
        return false;
    }
    char *data = input->getData();
    size_t cmplen = input->getDataLen();
    if (cmplen > 4)
        cmplen = 4;
    if (memcmp(data, "GET ", cmplen)) {
        *broken = true;
        return false;
    }

    ssize_t nr = input->findBytes("\r\n\r\n", 4);
    if (nr < 0) {
        return false;
    }

    header->_pcode = 1;
    header->_chid = 0;
    header->_dataLen = static_cast<int32_t>(nr + 4);
    return true;
}

} // namespace anet

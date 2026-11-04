#include "navi/engine/ShufflePortQueue.h"
#include "navi/engine/Port.h"

namespace navi {

ShufflePortQueue::ShufflePortQueue() {
    _logger.addPrefix("shuffle");
}

ShufflePortQueue::~ShufflePortQueue() {
}

ErrorCode ShufflePortQueue::setEof(NaviPartId fromPartId) {
    auto retEc = EC_NONE;
    for (auto toPartId : _partInfo) {
        auto ec = doSetEof(toPartId, fromPartId);
        if (EC_NONE != ec && EC_NONE == retEc) {
            retEc = ec;
        }
    }
    return retEc;
}

ErrorCode ShufflePortQueue::push(NaviPartId toPartId, NaviPartId fromPartId,
                                 const DataPtr &data, bool eof, bool &retEof)
{
    assert(false);
    return EC_UNKNOWN;
}

ErrorCode ShufflePortQueue::pop(NaviPartId toPartId, DataPtr &data, bool &eof) {
    assert(false);
    return EC_UNKNOWN;
}

}

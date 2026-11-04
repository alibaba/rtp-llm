#include "navi/engine/RemoteInputPort.h"
#include "navi/engine/SubGraphBorder.h"

namespace navi {

RemoteInputPort::RemoteInputPort(const EdgeDef &def,
                                 SubGraphBorder *border)
    : Port(def.output(), def.input(), IOT_INPUT, PST_SERIALIZE_DATA, border)
{
    _logger.addPrefix("RemoteInputPort");
}

RemoteInputPort::~RemoteInputPort() {
}

void RemoteInputPort::setCallback(NaviPartId partId, bool eof, int64_t totalDataCount, int64_t queueSize) {
    _border->notifySend(partId);
}

}

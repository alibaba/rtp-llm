#include "navi/engine/RemoteOutputPort.h"

namespace navi {

RemoteOutputPort::RemoteOutputPort(const EdgeDef &def,
                                   SubGraphBorder *border)
    : Port(def.input(), def.output(), IOT_OUTPUT, PST_SERIALIZE_DATA, border)
{
    _logger.addPrefix("RemoteOutputPort");
}

RemoteOutputPort::~RemoteOutputPort() {
}

}

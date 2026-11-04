#include "navi/engine/PartInfo.h"
#include "navi/log/NaviLogger.h"
#include "navi/util/ReadyBitMap.h"

namespace navi {

void PartInfo::init(NaviPartId partCount) {
    _ids.init(partCount);
    _ids.set();
}

bool PartInfo::init(NaviPartId partCount, const PartInfoDef &def) {
    _ids.init(partCount);
    if (def.part_ids_size() == 0) {
        _ids.set();
    } else {
        for (const auto& id : def.part_ids()) {
            if (id >= partCount) {
                NAVI_KERNEL_LOG(ERROR, "partId [%d] overflow, partCount: %d", id, partCount);
                return false;
            }
            _ids.set(id, true);
        }
    }
    return true;
}

void PartInfo::fillDef(PartInfoDef &def) const {
    def.set_part_count(_ids.size());
    def.clear_part_ids();
    for (size_t i = 0; i < _ids.size(); ++i) {
        if (_ids[i]) {
            def.add_part_ids(i);
        }
    }
}

InlineReadyBitMap PartInfo::createReadyBitMap() const {
    auto eof = InlineReadyBitMap::createReadyBitMap(getFullPartCount());
    eof.setInValid();
    for (auto partId : *this) {
        eof.setValid(partId, true);
    }
    return eof;
}

NaviPartId PartInfo::getUsedPartId(NaviPartId index) const {
    for (size_t i = 0; i < _ids.size(); ++i) {
        if (_ids[i] && (index-- == 0)) {
            return i;
        }
    }
    assert(false && "index must less than size");
    return -1;
}

NaviPartId PartInfo::getUsedPartIndex(NaviPartId partId) const {
    if (partId >= getFullPartCount()) {
        return -1;
    }
    NaviPartId index = 0;
    for (NaviPartId i = 0; i < partId; ++i) {
        if (_ids[i]) {
            ++index;
        }
    }
    if (_ids[partId]) {
        return index;
    } else {
        return -1;
    }
}

bool PartInfo::isUsed(NaviPartId partId) const {
    if (partId < _ids.size()) {
        return _ids[partId];
    }
    return false;
}

}

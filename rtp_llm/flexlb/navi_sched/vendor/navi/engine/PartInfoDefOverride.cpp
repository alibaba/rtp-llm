#include "navi/engine/PartInfoDefOverride.h"
#include "navi/common.h"
#include <google/protobuf/wire_format.h>

namespace navi {

thread_local PartInfoDefGraphMap *part_info_def_graph_map = nullptr;

PartInfoDef *PartInfoDefGraphMap::setValue(const PartInfoDef *overrideInfo) {
    auto it = _map.find(overrideInfo);
    if (_map.end() != it) {
        return it->second;
    }
    auto partInfo = google::protobuf::Arena::CreateMessage<PartInfoDef>(&_arena);
    _map.emplace(overrideInfo, partInfo);
    return partInfo;
}

const PartInfoDef *PartInfoDefGraphMap::getValue(const PartInfoDef *overrideInfo) const {
    auto it = _map.find(overrideInfo);
    if (_map.end() != it) {
        return it->second;
    } else {
        return nullptr;
    }
}

PartInfoDefOverride::PartInfoDefOverride() {
}

PartInfoDefOverride::PartInfoDefOverride(::google::protobuf::Arena* arena)
    : PartInfoDef(arena)
{
}

PartInfoDefOverride::~PartInfoDefOverride() {
}

size_t PartInfoDefOverride::ByteSizeLong() const {
    auto tlsValue = getTlsValue();
    if (tlsValue) {
        return tlsValue->ByteSizeLong();
    } else {
        return PartInfoDef::ByteSizeLong();
    }
}

void PartInfoDefOverride::SerializeWithCachedSizes(::google::protobuf::io::CodedOutputStream *output) const {
    auto tlsValue = getTlsValue();
    if (tlsValue) {
        tlsValue->SerializeWithCachedSizes(output);
    } else {
        PartInfoDef::SerializeWithCachedSizes(output);
    }
}

::google::protobuf::uint8 *
PartInfoDefOverride::InternalSerializeWithCachedSizesToArray(::google::protobuf::uint8 *target) const {
    auto tlsValue = getTlsValue();
    if (tlsValue) {
        return tlsValue->InternalSerializeWithCachedSizesToArray(target);
    } else {
        return PartInfoDef::InternalSerializeWithCachedSizesToArray(target);
    }
}

int PartInfoDefOverride::GetCachedSize() const {
    auto tlsValue = getTlsValue();
    if (tlsValue) {
        return tlsValue->GetCachedSize();
    } else {
        return PartInfoDef::GetCachedSize();
    }
}

void PartInfoDefOverride::SetCachedSize(int size) const {
    auto tlsValue = getTlsValue();
    if (tlsValue) {
        tlsValue->SetCachedSize(size);
    } else {
        PartInfoDef::SetCachedSize(size);
    }
}

PartInfoDef *PartInfoDefOverride::setTlsValue() {
    if (!part_info_def_graph_map) {
        return nullptr;
    }
    return part_info_def_graph_map->setValue(this);
}

const PartInfoDef *PartInfoDefOverride::getTlsValue() const {
    if (part_info_def_graph_map) {
        return part_info_def_graph_map->getValue(this);
    }
    return nullptr;
}

PartInfoDefGraphMapScope::PartInfoDefGraphMapScope(PartInfoDefGraphMap *partInfoDefGraphMap)
    : _prev(part_info_def_graph_map)
{
    part_info_def_graph_map = partInfoDefGraphMap;
}

PartInfoDefGraphMapScope::~PartInfoDefGraphMapScope() {
    part_info_def_graph_map = _prev;
}

}

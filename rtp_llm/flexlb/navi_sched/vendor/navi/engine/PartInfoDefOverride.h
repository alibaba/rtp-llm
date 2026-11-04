#pragma once

#define private public
#define final
#include "navi/proto/PeerInfo.pb.h"
#undef final
#undef private
#include "navi/common.h"

namespace navi {

class PartInfoDefOverride : public PartInfoDef {
public:
    PartInfoDefOverride();
    explicit PartInfoDefOverride(::google::protobuf::Arena* arena);
    ~PartInfoDefOverride();
public:
    size_t ByteSizeLong() const final;
    void SerializeWithCachedSizes(::google::protobuf::io::CodedOutputStream *output) const final;
    ::google::protobuf::uint8 *InternalSerializeWithCachedSizesToArray(::google::protobuf::uint8 *target) const final;
    int GetCachedSize() const final;
    void SetCachedSize(int size) const final;
public:
    PartInfoDef *setTlsValue();
    void initCopy(const PartInfoDef &tlsValue) {
        if (!_init) {
            return;
        }
        _init = false;
        CopyFrom(tlsValue);
    }
public:
    template <typename T>
    static PartInfoDefOverride *replacePartInfo(T *objDef);
private:
    const PartInfoDef *getTlsValue() const;
private:
    bool _init = true;
};

typedef NaviUnorderedMap<const PartInfoDef *, PartInfoDef *> PartInfoDefMap;

class PartInfoDefGraphMap {
public:
    PartInfoDef *setValue(const PartInfoDef *overrideInfo);
    const PartInfoDef *getValue(const PartInfoDef *overrideInfo) const;
private:
    google::protobuf::Arena _arena;
    PartInfoDefMap _map;
};

class PartInfoDefGraphMapScope {
public:
    PartInfoDefGraphMapScope(PartInfoDefGraphMap *partInfoDefGraphMap);
    ~PartInfoDefGraphMapScope();
private:
    PartInfoDefGraphMap *_prev = nullptr;
};

template <typename T>
PartInfoDefOverride *PartInfoDefOverride::replacePartInfo(T *objDef) {
    if (!objDef) {
        return nullptr;
    }
    auto oldPartInfo = objDef->part_info_;
    if (oldPartInfo) {
        auto partInfoOverride = dynamic_cast<PartInfoDefOverride *>(oldPartInfo);
        if (partInfoOverride) {
            return partInfoOverride;
        }
    }
    auto arena = objDef->GetArenaNoVirtual();
    auto newPartInfo = google::protobuf::Arena::CreateMessage<PartInfoDefOverride>(arena);
    if (oldPartInfo) {
        newPartInfo->CopyFrom(*oldPartInfo);
    }
    if (arena == nullptr && oldPartInfo != nullptr) {
        delete oldPartInfo;
    }
    objDef->part_info_ = newPartInfo;
    return newPartInfo;
}

}


#pragma once

#include "navi/common.h"
#include "autil/legacy/jsonizable.h"

namespace navi {

class BorderDef;
class BorderIdDef;

struct BorderId {
    BorderId(IoType ioType_, GraphId id_, GraphId peer_);
    BorderId(GraphId id_, const BorderDef &borderDef);
    BorderId(const BorderIdDef &borderIdDef);
public:
    bool operator==(const BorderId &other) const;
    friend std::ostream &operator<<(std::ostream &os, const BorderId &borderId);
public:
    IoType ioType;
    GraphId id;
    GraphId peer;
};

struct BorderIdJsonize : public autil::legacy::Jsonizable {
public:
    BorderIdJsonize(const BorderId &borderId_);
    virtual ~BorderIdJsonize() {
    }
public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) override;
public:
    BorderId borderId;
};

class BorderIdHash {
public:
    size_t operator()(const BorderId &border) const;
};

}

#pragma once

#include "autil/legacy/jsonizable.h"

namespace navi {

class GraphConfig : public autil::legacy::Jsonizable
{
public:
    GraphConfig();
    ~GraphConfig();
public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) override;
public:
    std::string graphFile;
};

}

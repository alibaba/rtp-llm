#pragma once

#include "navi/engine/Resource.h"
#include <string>

namespace navi_sched {

class AgentInstanceR : public navi::Resource {
public:
    void def(navi::ResourceDefBuilder &builder) const override;
    navi::ErrorCode init(navi::ResourceInitContext &ctx) override;
    bool getMeta(std::string &meta) const override;

public:
    static const std::string RESOURCE_ID;

private:
    std::string _meta;
};

} // namespace navi_sched

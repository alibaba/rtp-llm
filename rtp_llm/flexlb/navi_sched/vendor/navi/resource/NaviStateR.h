#pragma once

#include "navi/engine/Resource.h"

namespace navi {

class Navi;

class NaviStateR : public ExternalResource
{
public:
    NaviStateR();
    ~NaviStateR();
    NaviStateR(const NaviStateR &) = delete;
    NaviStateR &operator=(const NaviStateR &) = delete;
public:
    static const std::string RESOURCE_ID;
private:
    friend class Navi;
    friend class NaviStateKernel;
private:
    RESOURCE_DEPEND_DECLARE();
private:
    Navi *_navi = nullptr;
};

NAVI_TYPEDEF_PTR(NaviStateR);

}


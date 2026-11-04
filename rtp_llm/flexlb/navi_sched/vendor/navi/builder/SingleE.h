#pragma once

#include "navi/common.h"

namespace navi {

class EdgeDef;
class EdgeParamDef;
class GraphBuilder;

class SingleE
{
public:
    SingleE(EdgeDef *edge, GraphId graphId, int32_t index);
    ~SingleE();
    SingleE(const SingleE &) = delete;
    SingleE &operator=(const SingleE &) = delete;
public:
    void require(bool require, GraphBuilder *builder);
    void timeoutMs(int64_t timeoutMs, GraphBuilder *builder);
    void addSelectIndex(const std::set<int32_t> &selectIndexSet);
    bool isRequire() const;
private:
    friend std::ostream &operator<<(std::ostream &os, const SingleE &e);
private:
    EdgeDef *_edge;
    EdgeParamDef *_param;
    GraphId _graphId;
    int32_t _index;
};

}

